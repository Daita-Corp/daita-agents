from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest
from _distribution_support import inbox_distribution_plan
from _toolbox_model_support import (
    ToolboxAwareMockModelProvider as MockModelProvider,
)
from _workspace_support import workspace_for

import daita.storage.sqlite as sqlite_module
from daita import (
    Agent,
    DeliveryState,
    DeliverySubjectKind,
    InboxView,
    JobStatus,
    OutcomeState,
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
    SQLiteSource,
)
from daita._json import FrozenJsonObject, canonical_json
from daita.autonomy import (
    FOLLOWUP_INSTRUCTION,
    MAX_FOLLOWUP_EVENT_BYTES,
    FollowupCompletionConflictError,
    FollowupDisposition,
    FollowupIdentityConflictError,
    create_terminal_job_followup,
    terminal_job_event_payload,
)
from daita.capabilities import AccessMode, OperationalEffect
from daita.distribution.models import MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES
from daita.domains.data.profile_jobs import DATA_PROFILE_EXECUTION_CAPABILITY_ID
from daita.jobs.models import MAX_JOB_RESOURCE_BINDINGS
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop.models import LoopLimits, RunOrigin
from daita.storage.sqlite import SQLiteStateStore

LIMITS = LoopLimits(
    max_estimated_cost_usd=Decimal("0.05"),
    max_total_tokens=10_000,
)
SEED_LIMITS = LoopLimits()
ALLOWED_CAPABILITIES = (
    "catalog.inspect",
    "catalog.schema",
    "data.postgresql.query",
    "data.sqlite.query",
    "jobs.inspect",
    "jobs.read_results",
)
USAGE = ModelUsage(
    input_tokens=10,
    output_tokens=5,
    cost_estimate=CostEstimate.complete(Decimal("0.001")),
)


class _IdFactory:
    def __init__(self, suffix: str) -> None:
        self._suffix = suffix
        self._counts: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        count = self._counts.get(prefix, 0) + 1
        self._counts[prefix] = count
        if prefix in {"artifact", "run"}:
            return f"{prefix}-{count:032x}"
        return f"{prefix}-{self._suffix}-{count}"


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers(id INTEGER PRIMARY KEY, region TEXT, spend REAL);
            INSERT INTO customers(region, spend) VALUES
                ('north', 12.5), ('south', NULL), ('north', 21.0);
            """)


def _start_profile(resource_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="start-profile",
                name="start_data_profile",
                arguments={"resource_ids": [resource_id], "sample_rows": 20},
            ),
        ),
        usage=USAGE,
    )


def _read_job_result(job_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="read-result",
                name="job_read_results",
                arguments={"job_id": job_id},
            ),
        ),
        usage=USAGE,
    )


def _inspect_job(job_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="inspect-job",
                name="job_inspect",
                arguments={"job_id": job_id},
            ),
        ),
        usage=USAGE,
    )


def _inspect_and_read_job(
    job_id: str,
    *,
    resource_id_for_disallowed_calls: str | None = None,
) -> ModelResponse:
    calls = [
        ToolCall(
            id="inspect-job",
            name="job_inspect",
            arguments={"job_id": job_id},
        ),
        ToolCall(
            id="read-result",
            name="job_read_results",
            arguments={"job_id": job_id},
        ),
    ]
    if resource_id_for_disallowed_calls is not None:
        calls.extend(
            (
                ToolCall(
                    id="disallowed-start",
                    name="start_data_profile",
                    arguments={"resource_ids": [resource_id_for_disallowed_calls]},
                ),
                ToolCall(
                    id="disallowed-cancel",
                    name="job_cancel",
                    arguments={"job_id": "job-outside-scope"},
                ),
            )
        )
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(calls),
        usage=USAGE,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=USAGE)


def test_followup_instruction_requires_both_exact_job_evidence_calls() -> None:
    assert "job_inspect" in FOLLOWUP_INSTRUCTION
    assert "job_read_results" in FOLLOWUP_INSTRUCTION
    assert "Both successful calls are required" in FOLLOWUP_INSTRUCTION
    assert "does not replace" in FOLLOWUP_INSTRUCTION


def _job_id_from_request(provider: MockModelProvider, request_index: int) -> str:
    results = tuple(
        block
        for message in provider.logical_requests[request_index].messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
        and block.output.get("kind")
        not in {"toolbox_load_receipt", "toolbox_search_results"}
    )
    assert len(results) == 1
    data = results[0].output["data"]
    assert isinstance(data, Mapping)
    job_id = data["job_id"]
    assert isinstance(job_id, str)
    return job_id


def _job_id(provider: MockModelProvider) -> str:
    return _job_id_from_request(provider, 1)


async def _terminal(agent: Agent, job_id: str):
    inspection = None
    for _ in range(400):
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.summary.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.NEEDS_ATTENTION,
        }:
            return inspection
        await asyncio.sleep(0.005)
    raise AssertionError(f"job did not become terminal: {inspection!r}")


async def _inbox(agent: Agent):
    for _ in range(500):
        items = await agent.inbox()
        if items:
            return items
        driver = agent._embedded._followup_driver
        if driver is not None and driver.done():
            raise AssertionError(f"follow-up driver failed: {driver.exception()!r}")
        await asyncio.sleep(0.005)
    followups = await agent._embedded._store.list_autonomous_followups(agent.id)
    raise AssertionError(f"follow-up did not produce an inbox result: {followups!r}")


async def _delivery_job_id(agent: Agent, delivery: InboxView) -> str:
    assert delivery.subject_kind is DeliverySubjectKind.AUTONOMOUS_FOLLOWUP
    followups = await agent._embedded._store.list_autonomous_followups(agent.id)
    return next(
        item.job_id for item in followups if item.followup_id == delivery.subject_id
    )


async def test_terminal_daita_job_runs_one_scoped_machine_followup_and_inbox(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    provider = MockModelProvider((), complete_pricing=True)
    ids = _IdFactory("stage-c")
    agent = await Agent.open(
        "stage-c",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    provider.replace_script(
        (
            _start_profile(resource.id),
            _stop("Job accepted."),
            _inspect_and_read_job(
                "job-stage-c-1",
                resource_id_for_disallowed_calls=resource.id,
            ),
            _stop("The durable profile completed and its result is available."),
        )
    )
    try:
        origin = await agent.run("Profile the customer data.")
        job_id = _job_id(provider)
        await _terminal(agent, job_id)
        items = await _inbox(agent)

        assert len(items) == 1
        delivery = items[0]
        assert delivery.state is DeliveryState.AVAILABLE
        assert await _delivery_job_id(agent, delivery) == job_id
        assert delivery.conclusion_preview == (
            "The durable profile completed and its result is available."
        )
        assert delivery.conclusion_preview_truncated is False
        followup_run_id = delivery.resulting_run_id
        transcript = await agent.transcript(followup_run_id)
        assert transcript.run.origin is RunOrigin.JOB_EVENT
        assert transcript.messages[0].role is MessageRole.SYSTEM
        assert all(
            message.role is not MessageRole.USER for message in transcript.messages
        )
        assert transcript.run.execution_scope is not None
        scope = transcript.run.execution_scope
        assert scope.allowed_access_modes == frozenset(
            {AccessMode.NONE, AccessMode.READ}
        )
        assert scope.allowed_operational_effects == frozenset({OperationalEffect.NONE})
        assert all(
            tool.name
            not in {
                "start_data_profile",
                "job_cancel",
                "memory_set",
                "skill_create",
                "postgresql_update",
            }
            for tool in provider.logical_requests[2].tools
        )
        assert "<untrusted_job_event_payload>" in str(
            provider.logical_requests[2].messages[-1].content[0]
        )
        rejected = tuple(
            block
            for message in provider.logical_requests[3].messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id.startswith("disallowed-")
        )
        assert len(rejected) == 2
        assert all(block.is_error for block in rejected)
        assert len(await agent.list_jobs()) == 1
        assert origin.run_id != followup_run_id

        acknowledged = await agent.acknowledge_inbox(delivery.delivery_id)
        assert acknowledged is not None
        assert acknowledged.state is DeliveryState.ACKNOWLEDGED
        assert await agent.inbox() == ()
        assert (await agent.inbox(include_acknowledged=True))[0] == acknowledged
    finally:
        await agent.close()


async def test_failed_terminal_job_delivers_grounded_no_result_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "failed-source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-failed-job", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted before failure")),
    )
    agent = await Agent.open(
        "stage-c-failed-job",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    _, executor = agent._embedded._capabilities.resolve_execution(
        DATA_PROFILE_EXECUTION_CAPABILITY_ID
    )

    async def fail_execution(request) -> None:
        del request
        raise RuntimeError("injected_stage_c_job_failure")

    monkeypatch.setattr(executor, "execute", fail_execution)
    try:
        await agent.run("profile and fail", source_id=source.id)
        job_id = _job_id(seed)
        terminal = await _terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.FAILED
        assert terminal.failure_code == "job_execution_failed"
        assert await agent.read_job_result(job_id) is None
    finally:
        await agent.close()

    provider = MockModelProvider(
        (
            _inspect_and_read_job(job_id),
            _stop("The profile job failed and has no successful result."),
        ),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-failed-job",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        items = await _inbox(agent)
        assert len(items) == 1
        assert items[0].state is DeliveryState.AVAILABLE
        assert items[0].conclusion_state is OutcomeState.SUCCEEDED
        assert items[0].conclusion_preview == (
            "The profile job failed and has no successful result."
        )
        transcript = await agent.transcript(items[0].resulting_run_id)
        result_reads = tuple(
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.capability_id == "jobs.read_results"
        )
        assert len(result_reads) == 1
        assert result_reads[0].is_error
        error = result_reads[0].output["error"]
        assert isinstance(error, Mapping)
        assert error["code"] == "job_result_not_ready"
        details = error["details"]
        assert isinstance(details, Mapping)
        assert details["status"] == JobStatus.FAILED.value
        followup = (await agent._embedded._store.list_autonomous_followups(agent.id))[0]
        assert followup.disposition is FollowupDisposition.COMPLETED
        assert followup.conclusion_evidence is not None
    finally:
        await agent.close()


async def test_injected_router_fallback_is_scoped_sticky_and_delivers_once(
    tmp_path: Path,
) -> None:
    database = tmp_path / "fallback-source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-router-fallback", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted before fallback")),
    )
    agent = await Agent.open(
        "stage-c-router-fallback",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile before fallback", source_id=source.id)
        job_id = _job_id(seed)
        terminal = await _terminal(agent, job_id)
        assert terminal.summary.status is JobStatus.SUCCEEDED
    finally:
        await agent.close()

    unavailable = MockModelProvider(
        (
            ModelProviderError(
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
                usage=ModelUsage(
                    cost_estimate=CostEstimate.complete(Decimal("0")),
                ),
            ),
        ),
        provider_id="mock:stage-c-unavailable",
        complete_pricing=True,
    )
    fallback = MockModelProvider(
        (
            _inspect_and_read_job(job_id),
            _stop("The fallback inspected and reported the completed job."),
        ),
        provider_id="mock:stage-c-fallback",
        complete_pricing=True,
    )
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=unavailable,
                profile=unavailable.model_profile,
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
            ModelProviderRegistration(
                provider=fallback,
                profile=fallback.model_profile,
                allowed_sensitivities=frozenset(ModelSensitivity),
            ),
        ),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )
    agent = await Agent.open(
        "stage-c-router-fallback",
        root=tmp_path,
        model=router,
        model_profile=router.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        items = await _inbox(agent)
        assert len(items) == 1
        assert await _delivery_job_id(agent, items[0]) == job_id
        assert items[0].conclusion_state is OutcomeState.SUCCEEDED, items[0]
        assert len(unavailable.requests) == 1
        assert len(fallback.requests) == 2
        assert len(await agent.inbox()) == 1
        followups = await agent._embedded._store.list_autonomous_followups(agent.id)
        assert len(followups) == 1
        assert followups[0].attempt_count == 1
        assert set(followups[0].execution_scope.eligible_model_routes) == {
            unavailable.provider_id,
            fallback.provider_id,
        }
    finally:
        await agent.close()


async def test_store_deduplicates_exact_event_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    provider = MockModelProvider(
        (_start_profile("placeholder"), _stop("accepted")),
    )
    agent = await Agent.create(
        "stage-c-store", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await agent.attach(SQLiteSource(database))
    resource = (await agent.list_catalog_resources(source_id=source.id))[0]
    await agent.close()

    provider = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted")),
        complete_pricing=False,
    )
    agent = await Agent.open(
        "stage-c-store",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(provider)
        await _terminal(agent, job_id)
        job = await agent._embedded._store.load_job(agent.id, job_id)
        assert job is not None
        home = agent._embedded.home
        identity = agent.id
    finally:
        await agent.close()

    store = await SQLiteStateStore.open(home / "state.db")
    try:
        followup = create_terminal_job_followup(
            job,
            followup_id="followup-one",
            grant_id="grant-one",
            scope_id="scope-one",
            received_at=datetime.now(UTC),
            allowed_capability_ids=("jobs.inspect", "jobs.read_results"),
            eligible_model_routes=("mock:scripted",),
            limits=LIMITS,
        )
        admitted = await store.admit_autonomous_followup(followup)
        assert await store.admit_autonomous_followup(followup) == admitted

        conflicting_payload = FrozenJsonObject.from_mapping(
            {**dict(followup.event_payload), "failure_code": "untrusted-change"}
        )
        conflicting_digest = (
            "sha256:"
            + sha256(canonical_json(conflicting_payload).encode("utf-8")).hexdigest()
        )
        with pytest.raises(FollowupIdentityConflictError):
            await store.admit_autonomous_followup(
                replace(
                    followup,
                    event_payload=conflicting_payload,
                    payload_digest=conflicting_digest,
                )
            )
        with pytest.raises(FollowupCompletionConflictError):
            await store.admit_autonomous_followup(
                replace(
                    followup,
                    followup_id="followup-two",
                    event_id="stage-c:other-event",
                )
            )
        bound_job = await store.load_job(identity, job_id)
        assert bound_job is not None
        assert bound_job.completion_binding is not None
        assert bound_job.completion_binding.owner_id == followup.followup_id

        claimed_at = datetime.now(UTC)
        first_claim = await store.claim_next_autonomous_followup(
            identity,
            claim_token="expired-before-bind",
            reserved_run_id="expired-before-bind-run",
            claimed_at=claimed_at,
            lease_seconds=1.0,
        )
        assert first_claim is not None
        assert (
            await store.bind_autonomous_followup_run(
                identity,
                followup.followup_id,
                claim_token="expired-before-bind",
                run_id="expired-before-bind-run",
                bound_at=claimed_at + timedelta(seconds=2),
                audit_context={"must_not_bind": True},
            )
            is None
        )
        after_expired_bind = await store.load_autonomous_followup(
            identity,
            followup.followup_id,
        )
        assert after_expired_bind is not None
        assert after_expired_bind.disposition is FollowupDisposition.AVAILABLE
        assert after_expired_bind.reserved_run_id is None
        claimed_at += timedelta(seconds=3)
        for attempt in range(2, 4):
            claimed = await store.claim_next_autonomous_followup(
                identity,
                claim_token=f"stale-claim-{attempt}",
                reserved_run_id=f"stale-run-{attempt}",
                claimed_at=claimed_at,
                lease_seconds=1.0,
            )
            assert claimed is not None
            recovered = await store.recover_stale_autonomous_followups(
                identity,
                recovered_at=claimed_at + timedelta(seconds=2),
            )
            assert len(recovered) == 1
            claimed_at += timedelta(seconds=3)
        exhausted = await store.load_autonomous_followup(
            identity,
            followup.followup_id,
        )
        assert exhausted is not None
        assert exhausted.disposition is FollowupDisposition.TERMINAL_FAILED
        assert exhausted.failure_code == "followup_budget_exhausted"
        assert exhausted.reserved_cost_usd == 0
        assert exhausted.reserved_tokens == 0
    finally:
        await store.close()


async def test_unavailable_pricing_fails_before_run_creation(tmp_path: Path) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-unpriced", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    provider = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted")),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-unpriced",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(provider)
        await _terminal(agent, job_id)
    finally:
        await agent.close()

    unpriced = MockModelProvider((), complete_pricing=False)
    agent = await Agent.open(
        "stage-c-unpriced",
        root=tmp_path,
        model=unpriced,
        model_profile=unpriced.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        for _ in range(300):
            followups = await agent._embedded._store.list_autonomous_followups(agent.id)
            if (
                followups
                and followups[0].disposition is FollowupDisposition.TERMINAL_FAILED
            ):
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("unpriced follow-up did not fail")
        followup = followups[0]
        assert followup.failure_code == "cost_limit_unpriced_route"
        assert followup.reserved_run_id is None
        assert unpriced.requests == ()
        runs = await agent.conversation_runs(followup.conversation_id)
        assert all(run.transcript.run.origin is RunOrigin.USER for run in runs)
        assert await agent.inbox() == ()
    finally:
        await agent.close()


async def test_revoked_resource_scope_blocks_followup_before_reasoning(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-revoked", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted")),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-revoked",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(seed)
        await _terminal(agent, job_id)
        await agent.detach(source.id)
    finally:
        await agent.close()

    provider = MockModelProvider((), complete_pricing=True)
    agent = await Agent.open(
        "stage-c-revoked",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        for _ in range(300):
            followups = await agent._embedded._store.list_autonomous_followups(agent.id)
            if (
                followups
                and followups[0].disposition is FollowupDisposition.TERMINAL_FAILED
            ):
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("revoked follow-up did not fail")
        assert provider.requests == ()
        assert followups[0].reserved_run_id is None
        assert followups[0].failure_code in {
            "resource_read_not_allowed",
            "data_profile_resource_unavailable",
            "followup_data_profile_resource_unavailable",
            "followup_capabilityinputerror",
        }
        assert await agent.inbox() == ()
    finally:
        await agent.close()


async def test_host_loss_before_run_creation_recovers_stale_claim(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-pre-run-loss", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted")),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-pre-run-loss",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(seed)
        await _terminal(agent, job_id)
        job = await agent._embedded._store.load_job(agent.id, job_id)
        assert job is not None
        home = agent._embedded.home
    finally:
        await agent.close()

    now = datetime.now(UTC)
    store = await SQLiteStateStore.open(home / "state.db")
    try:
        followup = create_terminal_job_followup(
            job,
            followup_id="followup-host-loss",
            grant_id="grant-host-loss",
            scope_id="scope-host-loss",
            received_at=now,
            allowed_capability_ids=("jobs.inspect", "jobs.read_results"),
            eligible_model_routes=("mock:scripted",),
            limits=LIMITS,
        )
        await store.admit_autonomous_followup(followup)
        claimed = await store.claim_next_autonomous_followup(
            job.agent_id,
            claim_token="claim-lost-host",
            reserved_run_id="run-lost-before-creation",
            claimed_at=now,
            lease_seconds=1.0,
        )
        assert claimed is not None
        assert claimed.disposition is FollowupDisposition.CLAIMED
        bound = await store.bind_autonomous_followup_run(
            job.agent_id,
            followup.followup_id,
            claim_token="claim-lost-host",
            run_id="run-lost-before-creation",
            bound_at=now,
            audit_context={"prepared": True},
        )
        assert bound is not None
        assert bound.disposition is FollowupDisposition.RUNNING
    finally:
        await store.close()

    recovered_time = now + timedelta(seconds=2)
    provider = MockModelProvider(
        (
            _inspect_and_read_job(job_id),
            _stop("Recovered after the host was reopened."),
        ),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-pre-run-loss",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        clock=lambda: recovered_time,
        workspace=workspace_for(tmp_path),
    )
    try:
        items = await _inbox(agent)
        assert len(items) == 1
        recovered = (await agent._embedded._store.list_autonomous_followups(agent.id))[
            0
        ]
        assert recovered.disposition is FollowupDisposition.COMPLETED
        assert recovered.attempt_count == 2
        assert recovered.reserved_run_id != "run-lost-before-creation"
        with pytest.raises(KeyError):
            await agent.transcript("run-lost-before-creation")
    finally:
        await agent.close()


async def test_host_loss_after_terminal_commit_retries_delivery_not_reasoning(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-post-run-loss", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    provider = MockModelProvider((), complete_pricing=True)
    ids = _IdFactory("post-run-loss")
    provider.replace_script(
        (
            _start_profile(resource.id),
            _stop("accepted"),
            _inspect_and_read_job("job-post-run-loss-1"),
            _stop("The terminal job result was inspected exactly once."),
        )
    )
    agent = await Agent.open(
        "stage-c-post-run-loss",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    entered_finalizer = asyncio.Event()
    never = asyncio.Event()

    async def blocked_finalizer(*args, **kwargs):
        del args, kwargs
        entered_finalizer.set()
        await never.wait()

    setattr(
        agent._embedded._store,
        "finalize_autonomous_followup",
        blocked_finalizer,
    )
    try:
        await agent.run("profile")
        job_id = _job_id(provider)
        await _terminal(agent, job_id)
        await asyncio.wait_for(entered_finalizer.wait(), timeout=3)
        followup = (await agent._embedded._store.list_autonomous_followups(agent.id))[0]
        assert (
            followup.disposition
            is FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
        )
        assert followup.reserved_run_id is not None
        assert await agent._embedded._store.result(followup.reserved_run_id) is not None
        original_request_count = len(provider.logical_requests)
        assert original_request_count == 4
    finally:
        await agent.close()

    recovery = MockModelProvider((), complete_pricing=True)
    agent = await Agent.open(
        "stage-c-post-run-loss",
        root=tmp_path,
        model=recovery,
        model_profile=recovery.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        items = await _inbox(agent)
        assert len(items) == 1
        assert await _delivery_job_id(agent, items[0]) == job_id
        assert recovery.requests == ()
        assert len(provider.logical_requests) == original_request_count
        finalized = await agent._embedded._store.finalize_autonomous_followup(
            agent.id,
            items[0].subject_id,
            delivery_id="delivery-retry-must-not-replace",
            finalized_at=datetime.now(UTC),
        )
        assert finalized is not None
        assert finalized[1].delivery_id == items[0].delivery_id
        assert finalized[1].outcome.conclusion_digest == items[0].conclusion_digest
        assert len(await agent.inbox()) == 1
        assert recovery.requests == ()
    finally:
        await agent.close()


async def test_acknowledgment_unblocks_pending_stage_c_delivery_at_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-capacity",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    monkeypatch.setattr(sqlite_module, "MAX_DELIVERIES_PER_AGENT", 1)
    provider = MockModelProvider((), complete_pricing=True)
    ids = _IdFactory("capacity")
    provider.replace_script(
        (
            _start_profile(resource.id),
            _stop("First job accepted."),
            _inspect_and_read_job("job-capacity-1"),
            _stop("First terminal result delivered."),
        )
    )
    agent = await Agent.open(
        "stage-c-capacity",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile first")
        first_job_id = _job_id(provider)
        assert first_job_id == "job-capacity-1"
        await _terminal(agent, first_job_id)
        (first_delivery,) = await _inbox(agent)

        provider.replace_script(
            (
                _start_profile(resource.id),
                _stop("Second job accepted."),
                _inspect_and_read_job("job-capacity-2"),
                _stop("Second terminal result delivered after capacity is freed."),
            )
        )
        await agent.run("profile second")
        second_job_id = _job_id_from_request(provider, 5)
        assert second_job_id == "job-capacity-2"
        await _terminal(agent, second_job_id)

        for _ in range(500):
            followups = await agent._embedded._store.list_autonomous_followups(agent.id)
            second_followup = next(
                (item for item in followups if item.job_id == second_job_id),
                None,
            )
            if (
                second_followup is not None
                and second_followup.disposition
                is FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
            ):
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("Stage C follow-up did not wait for delivery capacity")

        acknowledged = await agent.acknowledge_inbox(first_delivery.delivery_id)
        assert acknowledged is not None
        (second_delivery,) = await _inbox(agent)
        assert await _delivery_job_id(agent, second_delivery) == second_job_id
        assert second_delivery.delivery_id != first_delivery.delivery_id
        retained = await agent.inbox(include_acknowledged=True)
        assert retained == (second_delivery,)
    finally:
        await agent.close()


async def test_one_delivery_failure_does_not_block_a_sibling_followup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-sibling", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (
            _start_profile(resource.id),
            _stop("first accepted"),
            _start_profile(resource.id),
            _stop("second accepted"),
        ),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-sibling",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile first")
        first_job_id = _job_id_from_request(seed, 1)
        await agent.run("profile second")
        second_job_id = _job_id_from_request(seed, 3)
        await _terminal(agent, first_job_id)
        await _terminal(agent, second_job_id)
    finally:
        await agent.close()

    original_finalize = SQLiteStateStore.finalize_autonomous_followup

    async def fail_first_delivery(
        store: SQLiteStateStore,
        agent_id: str,
        followup_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
    ):
        followup = await store.load_autonomous_followup(agent_id, followup_id)
        assert followup is not None
        if followup.job_id == first_job_id:
            raise RuntimeError("injected_delivery_failure")
        return await original_finalize(
            store,
            agent_id,
            followup_id,
            delivery_id=delivery_id,
            finalized_at=finalized_at,
        )

    monkeypatch.setattr(
        SQLiteStateStore,
        "finalize_autonomous_followup",
        fail_first_delivery,
    )
    provider = MockModelProvider(
        (
            _inspect_and_read_job(first_job_id),
            _stop("First report committed before delivery failed."),
            _inspect_and_read_job(second_job_id),
            _stop("Second report delivered independently."),
        ),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-sibling",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        items = await _inbox(agent)
        assert len(items) == 1
        assert await _delivery_job_id(agent, items[0]) == second_job_id
        followups = await agent._embedded._store.list_autonomous_followups(agent.id)
        by_job = {item.job_id: item for item in followups}
        assert (
            by_job[first_job_id].disposition
            is FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
        )
        assert by_job[second_job_id].disposition is FollowupDisposition.COMPLETED
        assert len(provider.logical_requests) == 4
        driver = agent._embedded._followup_driver
        assert driver is not None and not driver.done()
    finally:
        await agent.close()


async def test_delivery_is_blocked_when_destination_sensitivity_is_too_low(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-sensitive", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted")),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-sensitive",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(seed)
        await _terminal(agent, job_id)
        job = await agent._embedded._store.load_job(agent.id, job_id)
        assert job is not None
        home = agent._embedded.home
    finally:
        await agent.close()

    store = await SQLiteStateStore.open(home / "state.db")
    try:
        followup = create_terminal_job_followup(
            job,
            followup_id="followup-sensitive",
            grant_id="grant-sensitive",
            scope_id="scope-sensitive",
            received_at=datetime.now(UTC),
            allowed_capability_ids=ALLOWED_CAPABILITIES,
            eligible_model_routes=("mock:scripted",),
            limits=LIMITS,
        )
        public_plan = inbox_distribution_plan(
            followup.conversation_id,
            ModelSensitivity.PUBLIC,
        )
        followup = replace(
            followup,
            grant=replace(
                followup.grant,
                distribution_plan=public_plan,
            ),
            execution_scope=replace(
                followup.execution_scope,
                distribution_plan_digest=public_plan.plan_digest,
            ),
        )
        await store.admit_autonomous_followup(followup)
    finally:
        await store.close()

    provider = MockModelProvider(
        (
            _inspect_and_read_job(job_id),
            _stop("This internal report must not be delivered as public."),
        ),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-sensitive",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        items = await _inbox(agent)
        assert len(items) == 1
        assert items[0].state is DeliveryState.BLOCKED
        assert items[0].conclusion_preview == ""
        assert items[0].blocked_reason_code == "sensitivity_exceeds_destination"
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("evidence_kind", "failure_code"),
    (
        ("none", "followup_inspection_and_result_evidence_missing"),
        ("inspect", "followup_result_evidence_missing"),
        ("read", "followup_inspection_evidence_missing"),
    ),
)
async def test_completed_model_text_without_all_job_evidence_fails_closed(
    tmp_path: Path,
    evidence_kind: str,
    failure_code: str,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-evidence", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    provider = MockModelProvider((), complete_pricing=True)
    ids = _IdFactory("evidence")
    followup_script = {
        "none": (_stop("I trusted the event payload without inspecting the job."),),
        "inspect": (
            _inspect_job("job-evidence-1"),
            _stop("I inspected the job without reading its validated result."),
        ),
        "read": (
            _read_job_result("job-evidence-1"),
            _stop("I read the result without inspecting the job lifecycle."),
        ),
    }[evidence_kind]
    provider.replace_script(
        (
            _start_profile(resource.id),
            _stop("accepted"),
            *followup_script,
        )
    )
    agent = await Agent.open(
        "stage-c-evidence",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(provider)
        await _terminal(agent, job_id)
        items = await _inbox(agent)

        assert len(items) == 1
        assert items[0].conclusion_state is OutcomeState.FAILED
        assert items[0].conclusion_preview == ""
        assert items[0].failure_code == failure_code
        followup = (await agent._embedded._store.list_autonomous_followups(agent.id))[0]
        assert followup.disposition is FollowupDisposition.TERMINAL_FAILED
        assert followup.grant_consumed_at is None
        assert followup.conclusion_evidence is None
    finally:
        await agent.close()


async def test_cleared_origin_conversation_does_not_revoke_followup(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-cleared", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted before clearing")),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-cleared",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("origin text must not be required after clearing")
        job_id = _job_id(seed)
        await _terminal(agent, job_id)
        assert await agent.clear_conversations() >= 1
        assert await agent.list_jobs()
    finally:
        await agent.close()

    provider = MockModelProvider(
        (
            _inspect_and_read_job(job_id),
            _stop("Recovered from durable job truth with empty prior history."),
        ),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-cleared",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        items = await _inbox(agent)
        assert items[0].conclusion_state is OutcomeState.SUCCEEDED
        assert items[0].conclusion_preview == (
            "Recovered from durable job truth with empty prior history."
        )
        assert "origin text must not be required after clearing" not in repr(
            provider.logical_requests[0].messages
        )
    finally:
        await agent.close()


async def test_followup_history_excludes_other_conversation_sources(
    tmp_path: Path,
) -> None:
    first_database = tmp_path / "first.sqlite"
    second_database = tmp_path / "second.sqlite"
    _database(first_database)
    _database(second_database)
    bootstrap = await Agent.create(
        "stage-c-cross-source", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    first = await bootstrap.attach(SQLiteSource(first_database))
    second = await bootstrap.attach(SQLiteSource(second_database))
    first_resource = (await bootstrap.list_catalog_resources(source_id=first.id))[0]
    second_resource = (await bootstrap.list_catalog_resources(source_id=second.id))[0]
    await bootstrap.close()

    provider = MockModelProvider((), complete_pricing=True)
    ids = _IdFactory("cross-source")
    provider.replace_script(
        (
            _stop("other-source-history-secret-sentinel"),
            _start_profile(first_resource.id),
            _stop("accepted"),
            _inspect_and_read_job("job-cross-source-1"),
            _stop("Scoped report."),
        )
    )
    agent = await Agent.open(
        "stage-c-cross-source",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    try:
        first_result = await agent.run(
            "remember the other source",
            source_id=second.id,
        )
        created_at = (await agent.transcript(first_result.run_id)).run.created_at
        for annotation_id, source_id, resource, statement in (
            (
                "first-source-definition",
                first.id,
                first_resource,
                "first-source-semantic-sentinel",
            ),
            (
                "second-source-definition",
                second.id,
                second_resource,
                "other-source-semantic-secret-sentinel",
            ),
        ):
            await agent._embedded._store.save_semantic_annotation(
                agent.id,
                SemanticAnnotation(
                    id=annotation_id,
                    agent_id=agent.id,
                    subject=SemanticSubject(
                        source_ids=(source_id,),
                        resource_ids=(resource.id,),
                        fields=(SemanticFieldReference(resource.id, "region"),),
                    ),
                    kind=SemanticKind.METRIC_DEFINITION,
                    statement=statement,
                    evidence=(
                        SemanticEvidence(
                            SemanticEvidenceKind.USER_ASSERTION,
                            first_result.run_id,
                            message_position=0,
                        ),
                    ),
                    catalog_revisions=(
                        ResourceRevisionBinding(
                            resource.id,
                            resource.current_revision,
                        ),
                    ),
                    created_at=created_at,
                    confirmed_at=created_at,
                    confirmed_by="local-user",
                ),
            )
        await agent.run(
            "profile the first source",
            conversation_id=first_result.conversation_id,
            source_id=first.id,
        )
        job_id = _job_id_from_request(provider, 2)
        await _terminal(agent, job_id)
        items = await _inbox(agent)

        assert items[0].conclusion_state is OutcomeState.SUCCEEDED
        followup_requests = provider.logical_requests[3:]
        assert len(followup_requests) == 2
        assert all(
            "other-source-history-secret-sentinel" not in repr(request.messages)
            for request in followup_requests
        )
        assert all(
            "other-source-semantic-secret-sentinel" not in repr(request.messages)
            for request in followup_requests
        )
        assert all(
            "first-source-semantic-sentinel" in repr(request.messages)
            for request in followup_requests
        )
    finally:
        await agent.close()


async def test_terminal_event_uses_bounded_result_preview_and_exact_digest(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-event-bound", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    seed = MockModelProvider(
        (_start_profile(resource.id), _stop("accepted")),
        complete_pricing=True,
    )
    agent = await Agent.open(
        "stage-c-event-bound",
        root=tmp_path,
        model=seed,
        model_profile=seed.model_profile,
        limits=SEED_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(seed)
        await _terminal(agent, job_id)
        job = await agent._embedded._store.load_job(agent.id, job_id)
        assert job is not None and job.result is not None
        large_summary = FrozenJsonObject.from_mapping({"blob": "x" * 60_000})
        large_job = replace(job, result=replace(job.result, summary=large_summary))

        payload = terminal_job_event_payload(large_job)
        encoded = canonical_json(payload).encode("utf-8")
        result_payload = payload["result"]
        assert isinstance(result_payload, Mapping)
        assert len(encoded) <= MAX_FOLLOWUP_EVENT_BYTES
        assert result_payload["summary_truncated"] is True
        assert len(str(result_payload["summary_preview"]).encode("utf-8")) <= 4_096
        assert result_payload["summary_digest"] == (
            "sha256:"
            + sha256(canonical_json(large_summary).encode("utf-8")).hexdigest()
        )
        now = datetime.now(UTC)
        template_binding = large_job.specification.resource_bindings[0]
        maximum_bindings = tuple(
            replace(
                template_binding,
                source_id=f"source-{index:02d}-" + ("s" * 1_000),
                resource_id=f"resource-{index:02d}-" + ("r" * 1_000),
            )
            for index in range(MAX_JOB_RESOURCE_BINDINGS)
        )
        maximum_specification = replace(
            large_job.specification,
            resource_bindings=maximum_bindings,
        )
        maximum_job = replace(
            large_job,
            specification=maximum_specification,
            specification_digest=maximum_specification.digest,
        )
        maximum_payload = terminal_job_event_payload(maximum_job)
        assert len(canonical_json(maximum_payload).encode("utf-8")) <= (
            MAX_FOLLOWUP_EVENT_BYTES
        )
        source_scope = maximum_payload["source_scope"]
        resource_scope = maximum_payload["resource_scope"]
        assert isinstance(source_scope, Mapping)
        assert isinstance(resource_scope, Mapping)
        assert source_scope["count"] == MAX_JOB_RESOURCE_BINDINGS
        assert resource_scope["count"] == MAX_JOB_RESOURCE_BINDINGS
        assert maximum_bindings[0].source_id not in canonical_json(maximum_payload)
        create_terminal_job_followup(
            maximum_job,
            followup_id="followup-maximum-event",
            grant_id="grant-maximum-event",
            scope_id="scope-maximum-event",
            received_at=now,
            allowed_capability_ids=("jobs.inspect", "jobs.read_results"),
            eligible_model_routes=("mock:scripted",),
            limits=LIMITS,
        )
        followup = create_terminal_job_followup(
            job,
            followup_id="followup-expiring-bind",
            grant_id="grant-expiring-bind",
            scope_id="scope-expiring-bind",
            received_at=now,
            allowed_capability_ids=("jobs.inspect", "jobs.read_results"),
            eligible_model_routes=("mock:scripted",),
            limits=LIMITS,
        )
        followup = replace(
            followup,
            grant=replace(followup.grant, expires_at=now + timedelta(seconds=10)),
        )
        await agent._embedded._store.admit_autonomous_followup(followup)
        claimed = await agent._embedded._store.claim_next_autonomous_followup(
            agent.id,
            claim_token="claim-before-grant-expiry",
            reserved_run_id="run-before-grant-expiry",
            claimed_at=now,
            lease_seconds=300.0,
        )
        assert claimed is not None
        assert (
            await agent._embedded._store.bind_autonomous_followup_run(
                agent.id,
                followup.followup_id,
                claim_token="claim-before-grant-expiry",
                run_id="run-before-grant-expiry",
                bound_at=now + timedelta(seconds=11),
                audit_context={"must_not_bind": True},
            )
            is None
        )
        expired = await agent._embedded._store.load_autonomous_followup(
            agent.id,
            followup.followup_id,
        )
        assert expired is not None
        assert expired.disposition is FollowupDisposition.EXPIRED
        assert expired.reserved_run_id is None
    finally:
        await agent.close()


async def test_inbox_uses_bounded_report_preview_and_run_reference(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite"
    _database(database)
    bootstrap = await Agent.create(
        "stage-c-report-bound", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await bootstrap.attach(SQLiteSource(database))
    resource = (await bootstrap.list_catalog_resources(source_id=source.id))[0]
    await bootstrap.close()

    report = "report-start-" + ("z" * 80_000) + "-report-end"
    provider = MockModelProvider((), complete_pricing=True)
    ids = _IdFactory("report-bound")
    provider.replace_script(
        (
            _start_profile(resource.id),
            _stop("accepted"),
            _inspect_and_read_job("job-report-bound-1"),
            _stop(report),
        )
    )
    agent = await Agent.open(
        "stage-c-report-bound",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        limits=LIMITS,
        id_factory=ids,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.run("profile")
        job_id = _job_id(provider)
        await _terminal(agent, job_id)
        item = (await _inbox(agent))[0]

        assert item.conclusion_preview_truncated is True
        preview = item.conclusion_preview
        assert isinstance(preview, str)
        assert len(preview.encode("utf-8")) <= MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES
        assert item.conclusion_digest == (
            "sha256:" + sha256(report.encode("utf-8")).hexdigest()
        )
        transcript = await agent.transcript(item.resulting_run_id)
        final_block = transcript.messages[-1].content[0]
        assert isinstance(final_block, TextBlock)
        assert final_block.text == report
    finally:
        await agent.close()
