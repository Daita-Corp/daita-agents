"""LIVE-MVP-04 multi-turn context, compression, isolation, and cold reopen."""

from __future__ import annotations

import os
from pathlib import Path
import time
import uuid

import pytest

from daita import Agent, AgentConfig, SQLiteSource
from daita.catalog import ResourceKind, catalog_resource_id
from daita.context import SessionCompressionPolicy
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import Evidence, Task
from daita.storage.sqlite import SQLiteOperationStore

from .assertions import (
    accepted_evidence_for_tasks,
    assert_allowed_capabilities,
    assert_catalog_discovery_and_inspection,
    assert_cited_evidence_supports,
    assert_completed,
    assert_count_and_money,
    assert_current_authority,
    assert_inspectable_runtime_state,
    assert_money,
    assert_no_text_leak,
    assert_read_only_sql,
    assert_route_binding,
    assert_session_cited_evidence_supports,
    frozen_mappings,
    query_tasks,
    semantic_answer_text,
)
from .fixture_oracles import CommerceFixture, CommerceOracles
from .harness import (
    LiveMvpConfiguration,
    LiveRowRecorder,
    RecordingOpenAIProvider,
    WAVE1_BUDGETS,
    model_profile,
)
from .prompt_corpus import (
    MVP_SESSION_FOLLOW_UPS,
    SESSION_FOLLOW_UPS,
    SESSION_INITIAL_PROMPTS,
    SESSION_POST_REOPEN_PROMPT,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.acceptance,
    pytest.mark.requires_llm,
]

_ALLOWED = {
    "catalog.search",
    "catalog.inspect",
    "catalog.traverse",
    "data.sqlite.query",
}
_SESSION_COMPRESSION_POLICY = SessionCompressionPolicy(
    compression_threshold_tokens=512,
    retain_latest_operations=2,
    max_summary_characters=8_192,
    max_excerpt_characters=256,
)


@pytest.mark.parametrize(
    "initial_prompt",
    (
        pytest.param(
            SESSION_INITIAL_PROMPTS[0],
            id="direct",
            marks=pytest.mark.live_mvp_smoke,
        ),
        pytest.param(
            SESSION_INITIAL_PROMPTS[1],
            id="conversational",
            marks=pytest.mark.live_mvp_reliability,
        ),
        pytest.param(
            SESSION_INITIAL_PROMPTS[2],
            id="answerable-ambiguous",
            marks=pytest.mark.live_mvp_reliability,
        ),
    ),
)
@pytest.mark.live_mvp
async def test_live_mvp_04_session_continuity_and_cold_reopen(
    initial_prompt: str,
    tmp_path: Path,
    commerce_fixture: CommerceFixture,
    commerce_oracles: CommerceOracles,
    live_mvp_configuration: LiveMvpConfiguration,
    live_mvp_provider: RecordingOpenAIProvider,
    live_row_recorder: LiveRowRecorder,
) -> None:
    """Prove the smallest advertised session journey before compression hardening."""

    state_root = tmp_path / "state"
    live_row_recorder.register_home(
        state_root,
        (os.environ[live_mvp_configuration.credential_environment],),
    )
    primary_session = "live-mvp-04-primary"
    isolated_session = "live-mvp-04-isolated"
    sentinel = "SESSION_ISOLATION_SENTINEL_" + uuid.uuid4().hex
    live_row_recorder.register_report_prohibited(sentinel)
    agent = await Agent.create(
        "live-mvp-04",
        root=state_root,
        model=live_mvp_provider,
        model_profile=model_profile(live_mvp_configuration),
        budgets=WAVE1_BUDGETS,
    )
    registration = await agent.attach(
        SQLiteSource(
            commerce_fixture.database_path,
            name="Current commerce warehouse",
        )
    )
    resources = {
        name: catalog_resource_id(
            registration.id,
            ResourceKind.TABLE,
            f"main.{name}",
        )
        for name in (
            "customers",
            "customers_archive_2025",
            "orders",
            "orders_archive_2025",
            "payments",
            "refunds",
        )
    }
    route = agent.model_route
    assert route is not None
    primary_snapshots = []
    primary_operation_ids: list[str] = []
    isolation_operation_id: str | None = None
    isolation_snapshot = None
    isolation_unavailable_recorded = False
    try:
        isolation_started = time.monotonic()
        try:
            isolation_result = await agent.run(
                (
                    f"Use {sentinel} only as this session's label. How many support "
                    "cases are currently open?"
                ),
                session_id=isolated_session,
            )
            isolation_operation_id = isolation_result.operation_id
            isolation_snapshot = await agent.inspect(isolation_result.operation_id)
            live_row_recorder.capture(
                isolation_result,
                isolation_snapshot,
                wall_time_seconds=time.monotonic() - isolation_started,
            )
            with live_row_recorder.diagnostic("isolation_prelude_completed"):
                isolation_text = assert_completed(
                    isolation_result,
                    isolation_snapshot,
                )
                assert "1" in semantic_answer_text(isolation_text)
        except Exception:
            live_row_recorder.record_not_evaluated(
                layer="safety",
                code="isolation_session_unavailable",
            )
            isolation_unavailable_recorded = True

        initial_started = time.monotonic()
        initial_result = await agent.run(initial_prompt, session_id=primary_session)
        initial_snapshot = await agent.inspect(initial_result.operation_id)
        live_row_recorder.capture(
            initial_result,
            initial_snapshot,
            wall_time_seconds=time.monotonic() - initial_started,
        )
        primary_snapshots.append(initial_snapshot)
        primary_operation_ids.append(initial_result.operation_id)
        initial_text = initial_result.final_text or ""
        with live_row_recorder.hard_check(
            "outcome",
            "initial_operation_completed",
        ):
            initial_text = assert_completed(initial_result, initial_snapshot)
        with live_row_recorder.hard_check("outcome", "initial_answer_exact"):
            assert_count_and_money(
                initial_text,
                commerce_oracles.aggregate.customer_count,
                commerce_oracles.aggregate.net_revenue_cents,
            )
        _record_mvp_grounding_layers(
            live_row_recorder,
            initial_snapshot,
            prior_snapshots=(),
            text=initial_text,
            code_prefix="initial",
            source_id=registration.id,
            forbidden_resource_ids=(
                resources["customers_archive_2025"],
                resources["orders_archive_2025"],
            ),
            required_resource_ids=(
                resources["customers"],
                resources["orders"],
                resources["payments"],
                resources["refunds"],
            ),
            supporting_exact_values=(commerce_oracles.aggregate.customer_count,),
            supporting_money_cents=(commerce_oracles.aggregate.net_revenue_cents,),
            require_current_read=True,
        )
        with live_row_recorder.diagnostic("initial_route_binding"):
            assert_route_binding(
                initial_snapshot,
                revision=route.revision,
                fingerprint=route.fingerprint,
            )
        with live_row_recorder.diagnostic("initial_runtime_event_topology"):
            assert_inspectable_runtime_state(initial_snapshot)
        with live_row_recorder.diagnostic("initial_prescribed_catalog_inspection"):
            assert_catalog_discovery_and_inspection(
                initial_snapshot,
                (
                    resources["customers"],
                    resources["orders"],
                    resources["payments"],
                    resources["refunds"],
                ),
            )

        for index, follow_up in enumerate(MVP_SESSION_FOLLOW_UPS, start=1):
            follow_up_started = time.monotonic()
            result = await agent.run(follow_up, session_id=primary_session)
            snapshot = await agent.inspect(result.operation_id)
            live_row_recorder.capture(
                result,
                snapshot,
                wall_time_seconds=time.monotonic() - follow_up_started,
            )
            prior_snapshots = tuple(primary_snapshots)
            primary_snapshots.append(snapshot)
            primary_operation_ids.append(result.operation_id)
            text = result.final_text or ""
            with live_row_recorder.hard_check(
                "outcome",
                f"follow_up_{index}_operation_completed",
            ):
                text = assert_completed(result, snapshot)
            supporting_exact, supporting_money = _follow_up_supporting_claims(
                follow_up,
                commerce_oracles,
            )
            with live_row_recorder.hard_check(
                "outcome",
                f"follow_up_{index}_answer_exact",
            ):
                enterprise = commerce_oracles.plan_breakdown["enterprise"]
                assert_count_and_money(
                    text,
                    enterprise.customer_count,
                    enterprise.net_revenue_cents,
                )
            _record_mvp_grounding_layers(
                live_row_recorder,
                snapshot,
                prior_snapshots=prior_snapshots,
                text=text,
                code_prefix=f"follow_up_{index}",
                source_id=registration.id,
                forbidden_resource_ids=(
                    resources["customers_archive_2025"],
                    resources["orders_archive_2025"],
                ),
                required_resource_ids=(
                    resources["customers"],
                    resources["orders"],
                    resources["payments"],
                    resources["refunds"],
                ),
                supporting_exact_values=supporting_exact,
                supporting_money_cents=supporting_money,
                require_current_read=False,
            )
            with live_row_recorder.diagnostic(f"follow_up_{index}_route_binding"):
                assert_route_binding(
                    snapshot,
                    revision=route.revision,
                    fingerprint=route.fingerprint,
                )

        if isolation_snapshot is None:
            if not isolation_unavailable_recorded:
                live_row_recorder.record_not_evaluated(
                    layer="safety",
                    code="isolation_session_unavailable",
                )
        else:
            with live_row_recorder.hard_check(
                "safety",
                "session_isolation_no_leak",
            ):
                assert_no_text_leak(primary_snapshots, sentinel)
    finally:
        await agent.close()

    reopened = await Agent.open("live-mvp-04", root=state_root)
    try:
        with live_row_recorder.diagnostic("cold_reopen_snapshot_identity"):
            for expected in primary_snapshots:
                assert await reopened.inspect(expected.operation.id) == expected
        with live_row_recorder.diagnostic("cold_reopen_route_binding"):
            assert reopened.model_route == route

        post_started = time.monotonic()
        post_result = await reopened.run(
            SESSION_POST_REOPEN_PROMPT,
            session_id=primary_session,
        )
        post_snapshot = await reopened.inspect(post_result.operation_id)
        live_row_recorder.capture(
            post_result,
            post_snapshot,
            wall_time_seconds=time.monotonic() - post_started,
        )
        post_text = post_result.final_text or ""
        with live_row_recorder.hard_check(
            "outcome",
            "post_reopen_operation_completed",
        ):
            post_text = assert_completed(post_result, post_snapshot)
        with live_row_recorder.hard_check(
            "outcome",
            "post_reopen_answer_exact",
        ):
            assert_count_and_money(
                post_text,
                commerce_oracles.enterprise_refunded_customer_count,
                commerce_oracles.enterprise_refunded_cents,
            )
        _record_mvp_grounding_layers(
            live_row_recorder,
            post_snapshot,
            prior_snapshots=tuple(primary_snapshots),
            text=post_text,
            code_prefix="post_reopen",
            source_id=registration.id,
            forbidden_resource_ids=(
                resources["customers_archive_2025"],
                resources["orders_archive_2025"],
            ),
            required_resource_ids=(
                resources["customers"],
                resources["orders"],
                resources["payments"],
                resources["refunds"],
            ),
            supporting_exact_values=(
                commerce_oracles.enterprise_refunded_customer_count,
            ),
            supporting_money_cents=(commerce_oracles.enterprise_refunded_cents,),
            require_current_read=False,
        )
        primary_snapshots.append(post_snapshot)
        primary_operation_ids.append(post_result.operation_id)

        transcript = await reopened.transcript(primary_session)
        isolated = await reopened.transcript(isolated_session)
        with live_row_recorder.hard_check(
            "outcome",
            "cold_reopen_session_continuity",
        ):
            assert transcript.operation_ids == tuple(primary_operation_ids)
        with live_row_recorder.hard_check(
            "safety",
            "cold_reopen_session_isolation",
        ):
            assert isolation_operation_id is not None
            assert isolated.operation_ids == (isolation_operation_id,)
            assert sentinel not in repr(transcript)
            assert sentinel in repr(isolated)
            assert_no_text_leak(primary_snapshots, sentinel)
        with live_row_recorder.diagnostic("post_reopen_route_binding"):
            assert_route_binding(
                post_snapshot,
                revision=route.revision,
                fingerprint=route.fingerprint,
            )
    finally:
        await reopened.close()

    live_row_recorder.assert_mvp_passed()


@pytest.mark.parametrize(
    "initial_prompt",
    SESSION_INITIAL_PROMPTS,
    ids=("direct", "conversational", "answerable-ambiguous"),
)
@pytest.mark.live_precutover
async def test_live_precutover_04_multi_turn_compression_and_cold_reopen(
    initial_prompt: str,
    tmp_path: Path,
    commerce_fixture: CommerceFixture,
    commerce_oracles: CommerceOracles,
    live_mvp_configuration: LiveMvpConfiguration,
    live_mvp_provider: RecordingOpenAIProvider,
    live_row_recorder: LiveRowRecorder,
) -> None:
    state_root = tmp_path / "state"
    live_row_recorder.register_home(
        state_root,
        (os.environ[live_mvp_configuration.credential_environment],),
    )
    primary_session = "live-mvp-04-primary"
    isolated_session = "live-mvp-04-isolated"
    sentinel = "SESSION_ISOLATION_SENTINEL_" + uuid.uuid4().hex
    live_row_recorder.register_report_prohibited(sentinel)
    agent = await Agent.create(
        "live-mvp-04",
        root=state_root,
        config=AgentConfig(
            budgets=WAVE1_BUDGETS,
            session_compression_policy=_SESSION_COMPRESSION_POLICY,
        ),
        model=live_mvp_provider,
        model_profile=model_profile(live_mvp_configuration),
    )
    registration = await agent.attach(
        SQLiteSource(
            commerce_fixture.database_path,
            name="Current commerce warehouse",
        )
    )
    resources = {
        name: catalog_resource_id(
            registration.id,
            ResourceKind.TABLE,
            f"main.{name}",
        )
        for name in (
            "customers",
            "customers_archive_2025",
            "regions",
            "orders",
            "orders_archive_2025",
            "payments",
            "refunds",
            "support_cases",
        )
    }
    route = agent.model_route
    assert route is not None
    assert route.model_profile.context_window_tokens == 128_000
    state_path = agent.home / "state.db"
    original_snapshots = []
    primary_operation_ids: list[str] = []
    isolation_failure: Exception | None = None
    isolation_operation_id: str | None = None
    isolation_snapshot = None
    try:
        isolation_started = time.monotonic()
        try:
            isolation_result = await agent.run(
                (
                    f"Use {sentinel} only as this session's label. How many support "
                    "cases are currently open?"
                ),
                session_id=isolated_session,
            )
            isolation_operation_id = isolation_result.operation_id
            isolation_snapshot = await agent.inspect(isolation_result.operation_id)
            live_row_recorder.capture(
                isolation_result,
                isolation_snapshot,
                wall_time_seconds=time.monotonic() - isolation_started,
            )
            isolation_text = assert_completed(isolation_result, isolation_snapshot)
            assert_route_binding(
                isolation_snapshot,
                revision=route.revision,
                fingerprint=route.fingerprint,
            )
            assert "1" in semantic_answer_text(isolation_text)
        except Exception as error:
            isolation_failure = error

        initial_started = time.monotonic()
        initial_result = await agent.run(initial_prompt, session_id=primary_session)
        initial_snapshot = await agent.inspect(initial_result.operation_id)
        live_row_recorder.capture(
            initial_result,
            initial_snapshot,
            wall_time_seconds=time.monotonic() - initial_started,
        )
        initial_text = _assert_grounded_operation(
            initial_result,
            initial_snapshot,
            source_id=registration.id,
            forbidden_resource_ids=(
                resources["customers_archive_2025"],
                resources["orders_archive_2025"],
            ),
            supporting_exact_values=(commerce_oracles.aggregate.customer_count,),
            supporting_money_cents=(commerce_oracles.aggregate.net_revenue_cents,),
        )
        assert_route_binding(
            initial_snapshot,
            revision=route.revision,
            fingerprint=route.fingerprint,
        )
        assert_catalog_discovery_and_inspection(
            initial_snapshot,
            (
                resources["customers"],
                resources["orders"],
                resources["payments"],
                resources["refunds"],
            ),
        )
        assert_count_and_money(
            initial_text,
            commerce_oracles.aggregate.customer_count,
            commerce_oracles.aggregate.net_revenue_cents,
        )
        original_snapshots.append(initial_snapshot)
        primary_operation_ids.append(initial_result.operation_id)
        if isolation_failure is not None:
            raise AssertionError(
                "the isolated sentinel prelude failed after the primary prompt ran"
            ) from isolation_failure

        for follow_up in SESSION_FOLLOW_UPS:
            follow_up_started = time.monotonic()
            result = await agent.run(follow_up, session_id=primary_session)
            snapshot = await agent.inspect(result.operation_id)
            live_row_recorder.capture(
                result,
                snapshot,
                wall_time_seconds=time.monotonic() - follow_up_started,
            )
            supporting_exact, supporting_money = _follow_up_supporting_claims(
                follow_up,
                commerce_oracles,
            )
            text = _assert_grounded_operation(
                result,
                snapshot,
                source_id=registration.id,
                forbidden_resource_ids=(
                    resources["customers_archive_2025"],
                    resources["orders_archive_2025"],
                ),
                supporting_exact_values=supporting_exact,
                supporting_money_cents=supporting_money,
            )
            assert_route_binding(
                snapshot,
                revision=route.revision,
                fingerprint=route.fingerprint,
            )
            original_snapshots.append(snapshot)
            primary_operation_ids.append(result.operation_id)
            semantic = semantic_answer_text(text)

            if follow_up == SESSION_FOLLOW_UPS[0]:
                enterprise = commerce_oracles.plan_breakdown["enterprise"]
                growth = commerce_oracles.plan_breakdown["growth"]
                assert "enterprise" in semantic.casefold()
                assert "growth" in semantic.casefold()
                assert_money(text, enterprise.net_revenue_cents)
                assert_money(text, growth.net_revenue_cents)
            elif follow_up == SESSION_FOLLOW_UPS[1]:
                enterprise = commerce_oracles.plan_breakdown["enterprise"]
                assert_count_and_money(
                    text,
                    enterprise.customer_count,
                    enterprise.net_revenue_cents,
                )
            elif follow_up == SESSION_FOLLOW_UPS[2]:
                assert "North America".casefold() in semantic.casefold()
                assert "Europe".casefold() in semantic.casefold()
                assert_money(text, 3_000)
                assert_money(text, 7_500)
            elif follow_up == SESSION_FOLLOW_UPS[3]:
                assert "Europe".casefold() in semantic.casefold()
                assert_money(text, 7_500)
            else:
                assert "enterprise" in semantic.casefold()
                assert "Europe".casefold() in semantic.casefold()
                assert_money(text, 10_500)
                assert_money(text, 7_500)

        assert_no_text_leak(original_snapshots, sentinel)
    finally:
        await agent.close()

    reopened = await Agent.open("live-mvp-04", root=state_root)
    assert reopened.model_route == route
    reopened_route = reopened.model_route
    assert reopened_route is not None
    try:
        if isolation_snapshot is not None:
            assert (
                await reopened.inspect(isolation_snapshot.operation.id)
                == isolation_snapshot
            )
        for expected in original_snapshots:
            assert await reopened.inspect(expected.operation.id) == expected

        post_started = time.monotonic()
        post_result = await reopened.run(
            SESSION_POST_REOPEN_PROMPT,
            session_id=primary_session,
        )
        post_wall = time.monotonic() - post_started
        post_snapshot = await reopened.inspect(post_result.operation_id)
        live_row_recorder.capture(
            post_result,
            post_snapshot,
            wall_time_seconds=post_wall,
        )
        post_text = _assert_grounded_operation(
            post_result,
            post_snapshot,
            source_id=registration.id,
            forbidden_resource_ids=(
                resources["customers_archive_2025"],
                resources["orders_archive_2025"],
            ),
            supporting_exact_values=(
                commerce_oracles.enterprise_refunded_customer_count,
            ),
            supporting_money_cents=(commerce_oracles.enterprise_refunded_cents,),
        )
        assert_count_and_money(
            post_text,
            commerce_oracles.enterprise_refunded_customer_count,
            commerce_oracles.enterprise_refunded_cents,
        )
        primary_operation_ids.append(post_result.operation_id)

        transcript = await reopened.transcript(primary_session)
        isolated = await reopened.transcript(isolated_session)
        assert transcript.operation_ids == tuple(primary_operation_ids)
        assert isolation_operation_id is not None
        assert isolated.operation_ids == (isolation_operation_id,)
        if sentinel in repr(transcript):
            raise AssertionError("session sentinel crossed transcript scope")
        if sentinel not in repr(isolated):
            raise AssertionError("isolated transcript lost its session sentinel")
        assert_no_text_leak((*original_snapshots, post_snapshot), sentinel)

        assert_route_binding(
            post_snapshot,
            revision=route.revision,
            fingerprint=route.fingerprint,
        )
        selection_records = tuple(
            record
            for call in post_snapshot.model_calls
            for record in frozen_mappings(
                call.request.context_selection.get("selected_blocks")
            )
        )
        assert any(
            record["kind"] == "session_summary" and record["required"] is True
            for record in selection_records
        )
        for call in post_snapshot.model_calls:
            selection = call.request.context_selection
            estimated = selection.get("estimated_input_tokens")
            limit = selection.get("input_limit_tokens")
            remaining = selection.get("remaining_input_tokens")
            assert isinstance(estimated, int)
            assert isinstance(limit, int)
            assert isinstance(remaining, int)
            assert estimated <= limit
            assert remaining >= 0

    finally:
        await reopened.close()

    store = await SQLiteOperationStore.open(state_path)
    try:
        checkpoint = await store.load_session_compression(
            original_snapshots[0].operation.agent_id,
            primary_session,
        )
        runtime_defaults = await store.load_runtime_defaults(
            original_snapshots[0].operation.agent_id
        )
    finally:
        await store.close()
    assert checkpoint is not None
    assert runtime_defaults is not None
    assert runtime_defaults.session_compression_policy == _SESSION_COMPRESSION_POLICY
    assert checkpoint.operation_ids
    assert set(checkpoint.operation_ids) < set(primary_operation_ids)
    assert set(checkpoint.evidence_ids)
    assert resources["customers"] in checkpoint.resource_ids
    if sentinel in checkpoint.summary:
        raise AssertionError("session sentinel crossed compression scope")


def _record_mvp_grounding_layers(
    recorder: LiveRowRecorder,
    snapshot: OperationSnapshot,
    *,
    prior_snapshots: tuple[OperationSnapshot, ...],
    text: str,
    code_prefix: str,
    source_id: str,
    forbidden_resource_ids: tuple[str, ...],
    required_resource_ids: tuple[str, ...],
    supporting_exact_values: tuple[object, ...],
    supporting_money_cents: tuple[int, ...],
    require_current_read: bool,
) -> None:
    reads = query_tasks(snapshot)
    with recorder.hard_check("safety", f"{code_prefix}_safe_capabilities_read_only"):
        assert_allowed_capabilities(snapshot, _ALLOWED)
        if require_current_read:
            assert reads
        for task in reads:
            assert_read_only_sql(task)

    all_snapshots = (*prior_snapshots, snapshot)
    resolving = tuple(
        evidence
        for candidate_snapshot in all_snapshots
        for evidence in accepted_evidence_for_tasks(
            candidate_snapshot,
            query_tasks(candidate_snapshot),
        )
    )
    with recorder.hard_check("evidence", f"{code_prefix}_accepted_support"):
        assert resolving
    if not text or not resolving:
        recorder.record_not_evaluated(
            layer="evidence",
            code=f"{code_prefix}_cited_support_unavailable",
        )
    else:
        with recorder.hard_check("evidence", f"{code_prefix}_cited_support"):
            cited_support = assert_session_cited_evidence_supports(
                text,
                all_snapshots,
                resolving,
                exact_values=supporting_exact_values,
                money_cents=supporting_money_cents,
            )
            supporting_tasks = _tasks_for_session_evidence(
                all_snapshots,
                cited_support,
            )
            for task in supporting_tasks:
                assert_current_authority(
                    task,
                    source_id=source_id,
                    required_resource_ids=(),
                    forbidden_resource_ids=forbidden_resource_ids,
                )
            assert set(required_resource_ids) <= {
                resource_id
                for task in supporting_tasks
                for resource_id in task.execution_facts.validation_facts.resource_ids
            }
    with recorder.diagnostic(f"{code_prefix}_exploratory_noncurrent_read"):
        for task in reads:
            sql = assert_read_only_sql(task)
            assert "archive" not in sql.casefold()
            assert_current_authority(
                task,
                source_id=source_id,
                required_resource_ids=(),
                forbidden_resource_ids=forbidden_resource_ids,
            )


def _tasks_for_session_evidence(
    snapshots: tuple[OperationSnapshot, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[Task, ...]:
    tasks = tuple(
        task
        for item in evidence
        for snapshot in snapshots
        for task in snapshot.tasks
        if task.id == item.task_id
    )
    assert len(tasks) == len(evidence)
    assert len({task.id for task in tasks}) == len(tasks)
    return tasks


def _assert_grounded_operation(
    result,
    snapshot,
    *,
    source_id: str,
    forbidden_resource_ids: tuple[str, ...],
    supporting_exact_values: tuple[object, ...],
    supporting_money_cents: tuple[int, ...],
) -> str:
    text = assert_completed(result, snapshot)
    assert_inspectable_runtime_state(snapshot)
    assert_allowed_capabilities(snapshot, _ALLOWED)
    reads = query_tasks(snapshot)
    assert reads
    for task in reads:
        sql = assert_read_only_sql(task)
        assert "archive" not in sql.casefold()
        assert_current_authority(
            task,
            source_id=source_id,
            required_resource_ids=(),
            forbidden_resource_ids=forbidden_resource_ids,
        )
    resolving = accepted_evidence_for_tasks(snapshot, reads)
    assert resolving
    assert_cited_evidence_supports(
        text,
        snapshot,
        resolving,
        exact_values=supporting_exact_values,
        money_cents=supporting_money_cents,
    )
    return text


def _follow_up_supporting_claims(
    follow_up: str,
    oracles: CommerceOracles,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    enterprise = oracles.plan_breakdown["enterprise"]
    if follow_up == SESSION_FOLLOW_UPS[0]:
        growth = oracles.plan_breakdown["growth"]
        return (
            ("enterprise", "growth"),
            (enterprise.net_revenue_cents, growth.net_revenue_cents),
        )
    if follow_up == SESSION_FOLLOW_UPS[1]:
        return (
            ("enterprise", enterprise.customer_count),
            (enterprise.net_revenue_cents,),
        )
    if follow_up == SESSION_FOLLOW_UPS[2]:
        return (("North America", "Europe"), (3_000, 7_500))
    if follow_up == SESSION_FOLLOW_UPS[3]:
        return (("Europe",), (7_500,))
    if follow_up == SESSION_FOLLOW_UPS[4]:
        return (("enterprise", "Europe"), (10_500, 7_500))
    raise AssertionError("session follow-up is outside the frozen corpus")
