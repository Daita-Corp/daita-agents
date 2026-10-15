"""Live-model scoring of current owners, plus an honest scheduled-Python target.

SQLite data and MCP services are deterministic fixtures. Providers and native
Python workers are real. The PostgreSQL case uses the existing external fixture.
No warehouse adapter, Linux runtime or second execution engine is introduced.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from daita import (
    Agent,
    AgentConfig,
    ApprovalDecision,
    IntervalSchedule,
    LoopLimits,
    MCPToolSelection,
    MisfirePolicy,
    PostgreSQLReadLimits,
    ReportingMode,
    ScheduledRoutineDraft,
    SemanticAnnotationState,
)
from daita.artifacts.models import ArtifactAuthorship
from daita.distribution import ArtifactRequirement, OutcomeContract, OutcomeState
from daita.llm.models import MessageRole, ModelSensitivity, TextBlock
from daita.loop.models import LoopExitKind, RunOrigin, validate_completed_transcript
from daita.security import EnvironmentSecretProvider, SecretReference
from daita.storage.sqlite_codecs.transcripts import encode_loop_exit, encode_message
from tests.support.analysis import facts
from tests.support.data_operations import (
    CASES,
    METRIC_DEFINITION,
    METRIC_QUESTION,
    AnalyticalCase,
    create_database,
    expected_rows,
    parse_answer,
    score_answer,
)
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.job_benchmarks import assert_completed, results_for
from tests.support.live_billing import cost_cap, require_api_billing
from tests.support.mcp_read_harness import account_identity

pytestmark = [pytest.mark.acceptance, pytest.mark.integration, pytest.mark.requires_llm]


def source_scope(evaluation):
    return (evaluation.setup["source_id"],)


def record_score(evaluation, label, capture, expected):
    score = score_answer(capture.result.final_text, expected)
    evaluation.setup["results"][label] = score
    assert_completed(capture)
    validate_completed_transcript(capture.transcript, capture.result)
    assert score["correct"], score
    successful = [
        output
        for call, output in capture.transcript.tool_pairs
        if call.name == "data_query" and output is not None and not output.is_error
    ]
    assert (
        successful
    ), "A correct-looking answer without current read evidence is insufficient"
    assert all(output.capability_id == "data.query" for output in successful)
    assert not any(
        call.name
        in {"data_update_rows", "data_upsert_rows", "memory_set", "semantic_save"}
        for call, _ in capture.transcript.tool_pairs
    )
    return score


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
async def test_live_analytical_correctness(dataops, case: AnalyticalCase):
    evaluation, _service = dataops
    capture = await evaluation.run(
        case.prompt,
        phase=case.case_id,
        source_scope_ids=source_scope(evaluation),
    )
    record_score(
        evaluation, case.case_id, capture, expected_rows(case.kind, case.variant)
    )
    assert await evaluation.agent.list_effects() == ()


async def test_live_ambiguous_sources_require_clarification(dataops):
    evaluation, _service = dataops
    other = evaluation.agent.workspace.root / "other.sqlite"
    create_database(other, variant=9)
    await evaluation.agent.attach_sqlite(other, name="Second settlement account")
    capture = await evaluation.run(
        "For exactly one account, calculate paid order totals by currency from orders. "
        "I have not selected which account. If the current sources are ambiguous, "
        "ask one clarification and return only JSON with a clarification string.",
        phase="ambiguous_account",
    )
    assert_completed(capture)
    answer = parse_answer(capture.result.final_text)
    assert (
        isinstance(answer.get("clarification"), str) and answer["clarification"].strip()
    )
    assert "rows" not in answer
    assert not any(
        call.name == "data_query" for call, _ in capture.transcript.tool_pairs
    )


async def test_live_partial_file_does_not_establish_a_complete_total(dataops):
    evaluation, _service = dataops
    path = evaluation.agent.workspace.root / "partial-sales.csv"
    path.write_text("order_id,amount_minor\n" + "1,123456789\n" * 60_000)
    capture = await evaluation.run(
        "Read only the first file_read chunk of partial-sales.csv, without following "
        "its cursor or using file_query, Python, or any other read. Determine whether "
        "that one chunk establishes the total for the entire file. Return only JSON "
        "with complete (boolean) and total_minor (null when total is unavailable).",
        phase="partial_file_coverage",
    )
    assert_completed(capture)
    reads = results_for(capture.transcript, "file_read")
    assert len(reads) == 1 and not reads[0].is_error
    data = reads[0].output["data"]
    assert isinstance(data, Mapping) and data["complete"] is False
    assert parse_answer(capture.result.final_text) == {
        "complete": False,
        "total_minor": None,
    }
    assert not any(
        call.name in {"file_query", "python_execute"}
        for call, _ in capture.transcript.tool_pairs
    )


async def test_live_approved_definition_repeats_survives_reopen_and_detects_drift(
    dataops,
):
    evaluation, _service = dataops
    scope = source_scope(evaluation)
    expected = expected_rows("settlement_contribution")
    # Same information is supplied explicitly in the control. This compares
    # retained definitions with in-prompt definitions, not an uninformed baseline.
    control = await evaluation.run(
        METRIC_DEFINITION + " " + METRIC_QUESTION,
        phase="definition_in_prompt_control",
        source_scope_ids=scope,
    )
    record_score(evaluation, "definition_in_prompt_control", control, expected)

    approvals = []

    async def approve_definition(request):
        approvals.append(request)
        statement = str(request.arguments.get("statement", "")).casefold()
        return (
            ApprovalDecision.APPROVE
            if request.tool_name == "semantic_save"
            and request.arguments.get("kind") == "metric_definition"
            and "settlement" in statement
            and "refund_minor" in statement
            and ("3%" in statement or "3 percent" in statement or "0.03" in statement)
            else ApprovalDecision.DENY
        )

    # Reopen through the public composition with an exact local approval callback.
    agent = evaluation.agent
    root = agent.home.parent.parent
    name = agent.home.name
    profile = agent.model_profile
    workspace = agent.workspace
    limits = LoopLimits(max_estimated_cost_usd=cost_cap(evaluation.budget.per_run))
    await agent.close()
    evaluation.agent = await Agent.open(
        name,
        root=root,
        model=evaluation.provider,
        model_profile=profile,
        limits=limits,
        approval_handler=approve_definition,
        workspace=workspace,
    )
    taught = await evaluation.run(
        METRIC_DEFINITION
        + " Save exactly one source-bound metric_definition annotation.",
        phase="approved_teaching",
        source_scope_ids=scope,
        learning=True,
    )
    assert_completed(taught)
    active = [
        view
        for view in await evaluation.agent.list_semantic_annotations()
        if view.state is SemanticAnnotationState.ACTIVE
        and view.annotation.kind.value == "metric_definition"
    ]
    assert len(active) == 1
    definition = active[0]
    evaluation.setup["approved_definition"] = {
        "id": definition.annotation.id,
        "digest": definition.sha256,
        "statement": definition.annotation.statement,
        "catalog_revisions": [
            {"resource_id": binding.resource_id, "revision": binding.revision}
            for binding in definition.annotation.catalog_revisions
        ],
        "approval_count": sum(
            request.tool_name == "semantic_save" for request in approvals
        ),
    }
    assert evaluation.setup["approved_definition"]["approval_count"] == 1

    await evaluation.agent.close()
    evaluation.agent = await Agent.open(
        name,
        root=root,
        model=evaluation.provider,
        model_profile=profile,
        limits=limits,
        workspace=workspace,
    )
    for repetition in range(3):
        learned = await evaluation.run(
            METRIC_QUESTION,
            phase=f"retained_definition_{repetition}",
            source_scope_ids=scope,
        )
        record_score(evaluation, f"retained_definition_{repetition}", learned, expected)
        system = "\n".join(
            block.text
            for request in learned.requests
            for message in request.messages
            if message.role is MessageRole.SYSTEM
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert (
            definition.annotation.id in system
            or definition.annotation.statement in system
        )

    unrelated = await evaluation.run(
        CASES[0].prompt,
        phase="learning_regression_control",
        source_scope_ids=scope,
    )
    record_score(
        evaluation,
        "learning_regression_control",
        unrelated,
        expected_rows("paid_totals"),
    )
    # Structural drift is injected only into the disposable source fixture.
    with sqlite3.connect(evaluation.setup["database"]) as connection:
        connection.execute(
            "ALTER TABLE orders RENAME COLUMN refund_minor TO refunded_minor"
        )
    await evaluation.agent.refresh_source(scope[0])
    stale = await evaluation.agent.read_semantic_annotation(definition.annotation.id)
    assert stale is not None and stale.state is SemanticAnnotationState.STALE
    assert not stale.usable_as_current_meaning
    evaluation.setup["definition_after_drift"] = {
        "state": stale.state.value,
        "stale_reasons": list(stale.stale_reasons),
    }
    after_drift = await evaluation.run(
        METRIC_QUESTION,
        phase="stale_definition_requires_clarification",
        source_scope_ids=scope,
    )
    assert_completed(after_drift)
    answer = parse_answer(after_drift.result.final_text)
    assert (
        isinstance(answer.get("clarification"), str) and answer["clarification"].strip()
    )
    assert (
        "rows" not in answer
    ), "Stale business knowledge must not produce a claimed metric"


async def test_live_python_chart_dataset_and_retained_computation(dataops):
    evaluation, _service = dataops
    runtime = evaluation.agent.analysis_runtime()
    assert runtime[
        "available"
    ], "Install the existing pinned macOS analysis environment"
    evaluation.setup["analysis_runtime"] = runtime
    capture = await evaluation.run(
        "Using current orders data, calculate amount_minor minus refund_minor for paid "
        "orders separately by currency. Use the Python analysis tool, make a Matplotlib "
        'PNG chart, and a JSON dataset shaped as {"rows":[{"currency":"USD",'
        '"amount_minor":123}]}. Save both as artifacts through separate analysis '
        "calls. Read orders programmatically through tools.call('data_query', ...) "
        "inside Python after loading its exact schema together with python_execute. "
        "Retain the returned coverage. Do not publish "
        "local files. Return the same JSON rows as your answer.",
        phase="python_report",
        source_scope_ids=source_scope(evaluation),
    )
    records = await evaluation.agent.analysis_evidence(capture.result.run_id)
    generations = [facts(record) for record in records if record.kind == "generation"]
    children = [facts(record) for record in records if record.kind == "child"]
    evaluation.setup["analysis_evidence"] = [facts(record) for record in records]
    assert_completed(capture)
    assert generations and children
    assert any(
        record.get("tool_name") == "data_query" and record["status"] == "succeeded"
        for record in children
    )
    for generation in generations:
        assert generation["cleanup"]["process_reaped"] is True
        assert generation["cleanup"]["scratch_deleted"] is True
        assert generation["usage"]["peak_rss_bytes"] > 0
        assert generation["usage"]["user_cpu_seconds"] is not None
    assert len(capture.result.artifacts) == 2
    artifacts = {ref.media_type: ref for ref in capture.result.artifacts}
    assert set(artifacts) == {"application/json", "image/png"}
    dataset = artifacts["application/json"]
    chart = artifacts["image/png"]
    data = await evaluation.agent.read_artifact(dataset.artifact_id)
    score = score_answer(data.content.decode(), expected_rows("paid_totals"))
    evaluation.setup["results"]["python_dataset"] = score
    assert score["correct"], score
    assert score_answer(capture.result.final_text, expected_rows("paid_totals"))[
        "correct"
    ]
    png = (await evaluation.agent.read_artifact(chart.artifact_id)).content
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 100
    evidence = await evaluation.agent.artifact_computation_evidence(dataset.artifact_id)
    assert evidence and evidence["output_sha256"] == dataset.sha256
    await evaluation.agent.clear_conversations()
    retained = await evaluation.agent.artifact_computation_evidence(dataset.artifact_id)
    assert retained == evidence
    assert (
        await evaluation.agent.read_artifact(dataset.artifact_id)
    ).content == data.content


async def test_live_mcp_binding_selection_and_schema_drift(dataops):
    evaluation, service = dataops
    europe = account_identity("Europe", "2026-07-28")
    america = account_identity("Americas", "2025-11-25")
    service.identities.update(
        {identity.host: identity for identity in (europe, america)}
    )
    statuses = []
    for identity in (europe, america):
        statuses.append(
            await evaluation.agent.attach_mcp_server(
                endpoint=identity.endpoint,
                local_label=f"{identity.server_name} billing",
                selections=tuple(
                    MCPToolSelection(str(tool["name"])) for tool in identity.tools
                ),
            )
        )
    capture = await evaluation.run(
        "Check Cedar Works' current balance in Europe billing. Return the exact "
        "balance in cents, currency, and verification marker. Use the current record.",
        phase="mcp_exact_account",
    )
    assert_completed(capture)
    expected = europe.results["read_balance"]["structuredContent"]
    assert isinstance(expected, Mapping)
    assert [name for name, _ in europe.calls] == ["find_account", "read_balance"]
    assert america.calls == []
    answer = capture.result.final_text or ""
    assert str(expected["balance_cents"]) in answer.replace(",", "")
    assert str(expected["verification_marker"]) in answer
    assert str(expected["currency"]) in answer
    bound = next(
        tool for tool in statuses[0].binding.tools if tool.remote_name == "read_balance"
    )
    outputs = results_for(capture.transcript, bound.local_name)
    assert len(outputs) == 1 and not outputs[0].is_error
    output_data = outputs[0].output["data"]
    assert isinstance(output_data, Mapping)
    provenance = output_data["provenance"]
    assert isinstance(provenance, Mapping)
    assert provenance["binding_id"] == statuses[0].binding.binding_id
    assert provenance["binding_revision"] == statuses[0].binding.revision
    schema = europe.tool("read_balance")["inputSchema"]
    assert isinstance(schema, dict)
    properties, required = schema["properties"], schema["required"]
    assert isinstance(properties, dict) and isinstance(required, list)
    properties["new_required_field"] = {"type": "string"}
    required.append("new_required_field")
    dispatched = len(europe.calls)
    changed = await evaluation.run(
        f'Call {bound.local_name} once with exactly {{"account_id":'
        f"{json.dumps(expected['account_id'])}}}. If its admitted contract changed, "
        "report that the data is unavailable. Do not refresh, retry, switch servers or guess.",
        phase="mcp_contract_drift",
    )
    assert_completed(changed)
    assert len(europe.calls) == dispatched and america.calls == []
    denied = tuple(
        output
        for call, output in changed.transcript.tool_pairs
        if call.name == bound.local_name and output is not None
    )
    assert denied and all(output.is_error for output in denied)
    assert str(expected["verification_marker"]) not in (changed.result.final_text or "")
    evaluation.setup["mcp_drift_dispatch_count"] = len(europe.calls) - dispatched


async def routine_delivery(evaluation, origin, *, python: bool):
    now = datetime.now(UTC)
    scope = source_scope(evaluation)
    capabilities: tuple[str, ...] = ("catalog.describe", "data.query")
    outcome = no_artifact_outcome_contract()
    instruction = CASES[0].prompt
    if python:
        capabilities += ("analysis.execute",)
        requirement = ArtifactRequirement(
            required=True,
            minimum_count=1,
            maximum_count=1,
            allowed_media_types=("image/png",),
            allowed_authorships=(ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,),
            allowed_producer_capability_ids=("analysis.execute",),
            maximum_artifact_bytes=1_000_000,
            maximum_total_bytes=1_000_000,
            maximum_sensitivity=ModelSensitivity.INTERNAL,
        )
        outcome = OutcomeContract(
            require_terminal_conclusion=True,
            artifact_requirements=(requirement,),
            maximum_total_artifact_bytes=1_000_000,
            maximum_effective_sensitivity=ModelSensitivity.INTERNAL,
            require_current_run_provenance=True,
            require_exact_source_bindings=False,
        )
        instruction += (
            " Use Python to compute the result from a programmatic data_query read "
            "inside tools.call, and save a PNG chart artifact."
        )
    draft = ScheduledRoutineDraft(
        origin_run_id=origin.result.run_id,
        title="Weekly settlement acceptance",
        authorized_instruction=instruction,
        schedule=IntervalSchedule(604800, now + timedelta(days=7)),
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        allowed_source_ids=scope,
        allowed_connector_binding_ids=(),
        allowed_resource_ids=tuple(evaluation.setup["resource_ids"]),
        allowed_capability_ids=capabilities,
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        outcome_contract=outcome,
        distribution_destination_id=(
            await evaluation.agent.distribution_destinations(
                origin.result.conversation_id,
                sensitivity_ceiling=ModelSensitivity.INTERNAL,
            )
        )[0].destination_id,
        eligible_model_routes=(evaluation.provider.provider_id,),
        per_run_max_tokens=100000,
        per_run_max_cost_usd=evaluation.budget.per_run,
        cumulative_max_tokens=100000,
        cumulative_max_cost_usd=evaluation.budget.per_run,
        cumulative_max_attempts=1,
        cumulative_max_occurrences=1,
        maximum_consecutive_failures=1,
        expires_at=now + timedelta(days=8),
    )
    proposal = await evaluation.agent.propose_routine(draft)
    stored = await evaluation.agent.create_routine(proposal)
    evaluation.budget.reserve()
    result = None
    try:
        await evaluation.agent.run_routine_now(
            stored.routine_id, expected_revision=stored.revision
        )
        deadline = asyncio.get_running_loop().time() + 300
        while asyncio.get_running_loop().time() < deadline:
            items = await evaluation.agent.inbox(
                conversation_id=origin.result.conversation_id
            )
            matches = [item for item in items if item.resulting_run_id is not None]
            if matches:
                item = matches[0]
                runs = await evaluation.agent.conversation_runs(
                    origin.result.conversation_id
                )
                result = next(
                    run.result
                    for run in runs
                    if run.transcript.run.id == item.resulting_run_id
                )
                assert result is not None
                transcript = await evaluation.agent.transcript(result.run_id)
                evaluation.setup["scheduled_run"] = {
                    "result": json.loads(encode_loop_exit(result)),
                    "messages": [
                        json.loads(encode_message(message))
                        for message in transcript.messages
                    ],
                }
                if python:
                    records = await evaluation.agent.analysis_evidence(result.run_id)
                    evaluation.setup["scheduled_analysis_evidence"] = [
                        facts(record) for record in records
                    ]
                assert (
                    item.conclusion_state is OutcomeState.SUCCEEDED
                ), item.failure_code
                assert result.kind is LoopExitKind.COMPLETED
                validate_completed_transcript(transcript, result)
                assert transcript.run.origin is RunOrigin.SCHEDULED_ROUTINE
                execution = transcript.run.execution_scope
                assert execution and execution.allowed_source_ids == scope
                assert set(execution.allowed_capability_ids) == set(capabilities)
                score = score_answer(result.final_text, expected_rows("paid_totals"))
                evaluation.setup["results"][
                    "scheduled_python" if python else "scheduled_sql"
                ] = score
                evaluation.setup["routine_delivery"] = {
                    "run_id": result.run_id,
                    "delivery_id": item.delivery_id,
                    "routine_id": stored.routine_id,
                    "allowed_source_ids": list(execution.allowed_source_ids),
                    "allowed_capability_ids": list(execution.allowed_capability_ids),
                }
                assert score["correct"], score
                if python:
                    assert any(
                        record.kind == "child"
                        and record.facts["tool_name"] == "data_query"
                        and record.facts["status"] == "succeeded"
                        for record in records
                    )
                    assert any(
                        ref.media_type == "image/png" for ref in result.artifacts
                    )
                return
            await asyncio.sleep(0.1)
        raise AssertionError("Routine did not produce one bounded inbox delivery")
    finally:
        evaluation.budget.settle(result)


@pytest.mark.parametrize(
    "python", (False, True), ids=("scheduled_sql", "scheduled_python_target")
)
async def test_live_foreground_to_scheduled_report(dataops, python):
    require_api_billing("routine admission")
    evaluation, _service = dataops
    origin = await evaluation.run(
        CASES[0].prompt,
        phase="foreground_report",
        source_scope_ids=source_scope(evaluation),
    )
    record_score(evaluation, "foreground_report", origin, expected_rows("paid_totals"))
    await routine_delivery(evaluation, origin, python=python)


@pytest.mark.requires_db
@pytest.mark.parametrize("pool_enabled", (False, True), ids=("cold", "pooled"))
async def test_live_existing_postgresql_fixture_independent_oracle(
    dataops, pool_enabled
):
    """Read the existing disposable analytics fixture; never provision or write it."""
    evaluation, _service = dataops
    password = os.environ.get("DAITA_FIXTURE_POSTGRES_PASSWORD")
    assert (
        password
    ), "Configure the existing tests/fixtures/postgresql service before opting in"
    import asyncpg  # type: ignore[import-untyped]

    port = int(os.environ.get("DAITA_FIXTURE_POSTGRES_PORT", "55432"))
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=port,
        database="daita_fixture",
        user="daita_reader",
        password=password,
        timeout=10,
    )
    try:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            raw = await connection.fetch(
                "SELECT oi.line_total, o.status, r.currency_code "
                "FROM analytics.order_items oi JOIN analytics.orders o USING(order_id) "
                "JOIN analytics.customers c USING(customer_id) "
                "JOIN analytics.regions r USING(region_code)",
                timeout=30,
            )
    finally:
        await connection.close()
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for row in raw:
        if row["status"] == "paid":
            totals[row["currency_code"]] += row["line_total"]
    expected = [
        {"currency": currency, "amount_minor": int(amount * 100)}
        for currency, amount in sorted(totals.items())
    ]
    agent = evaluation.agent
    await agent.close()
    evaluation.agent = await Agent.open(
        agent.home.name,
        root=agent.home.parent.parent,
        workspace=agent.workspace,
        model=evaluation.provider,
        model_profile=agent.model_profile,
        limits=LoopLimits(max_estimated_cost_usd=cost_cap(evaluation.budget.per_run)),
        secret_provider=EnvironmentSecretProvider(),
        config=AgentConfig(
            postgresql_reads=PostgreSQLReadLimits(
                pool_enabled=pool_enabled,
                pool_size_per_source=1,
                max_connections=1,
                max_cached_sources=1,
            )
        ),
    )
    source = await evaluation.agent.attach_postgresql(
        host="127.0.0.1",
        port=port,
        database="daita_fixture",
        username="daita_reader",
        credential=SecretReference.environment("DAITA_FIXTURE_POSTGRES_PASSWORD"),
        schemas=("analytics",),
        ssl_mode="disable",
        name="Existing analytics fixture",
    )
    evaluation.setup["service_kind"] = "external_disposable_postgresql"
    for repetition in range(2):
        phase = f"postgresql_{'pooled' if pool_enabled else 'cold'}_{repetition}"
        capture = await evaluation.run(
            "Sum order_items.line_total for orders.status='paid', by the customer's region "
            "currency_code. Join the current analytics tables using their relationships. "
            "Return minor currency units (multiply the sum by 100), only as a JSON "
            'object shaped as {"rows":[{"currency":"USD","amount_minor":123}]} '
            "(never a bare array). Keep currencies separate.",
            phase=phase,
            source_scope_ids=(source.id,),
        )
        record_score(evaluation, phase, capture, expected)
    diagnostics = evaluation.agent.postgresql_read_diagnostics()
    evaluation.setup["postgresql_reads"] = asdict(diagnostics)
    if pool_enabled:
        assert diagnostics.acquired >= 2
        assert diagnostics.connections_created == 1
        assert diagnostics.active == diagnostics.waiting == 0
        assert diagnostics.completed == diagnostics.acquired
        assert diagnostics.structural_seconds > 0
    assert await evaluation.agent.list_effects() == ()


@pytest.mark.parametrize("method", ("call", "batch"))
async def test_paired_python_relational_reads_preserve_independent_oracle(
    dataops, method
):
    evaluation, _service = dataops
    case = AnalyticalCase("paid_count", 0)
    capture = await evaluation.run(
        case.prompt
        + " Use Python and current relational read evidence. "
        + (
            "Issue independent reads through tools.batch and check each result before computing."
            if method == "batch"
            else "Use synchronous tools.call for each read and check each result before computing."
        ),
        phase=f"python-{method}",
        source_scope_ids=source_scope(evaluation),
    )
    score = score_answer(
        capture.result.final_text, expected_rows(case.kind, case.variant)
    )
    evaluation.setup["results"][f"python-{method}"] = score
    assert_completed(capture)
    assert score["correct"], score
    records = await evaluation.agent.analysis_evidence(capture.result.run_id)
    evaluation.setup["analysis_evidence"] = [facts(record) for record in records]
    children = [
        r
        for r in records
        if r.kind == "child"
        and r.facts["tool_name"] == "data_query"
        and r.facts["status"] == "succeeded"
    ]
    assert children, "Correctness requires current authenticated Python child reads"
    assert any(
        f"tools.{method}" in str(call.arguments.get("code", ""))
        for call, _ in capture.transcript.tool_pairs
        if call.name == "python_execute"
    )
