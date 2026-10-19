"""Repeated refusals warn the model in every run type without changing the run."""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    ArtifactRequirement,
    IntervalSchedule,
    MisfirePolicy,
    OutcomeContract,
    ReportingMode,
    ScheduledRoutineDraft,
    SQLiteSource,
)
from daita._json import canonical_json
from daita.artifacts.models import ArtifactAuthorship
from daita.context import AgentContextBuilder
from daita.domains.data.export_capabilities import DATA_EXPORT_TABULAR_CAPABILITY_ID
from daita.jobs.graph.models import TaskRole
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop.session import RunSessionEvidence
from tests.support.conversations import CatalogSpy
from tests.support.model_graph_integration import READ_TOOL_NAME, ModelGraphIntegration
from tests.support.toolbox_model import ToolboxAwareMockModelProvider, result_ref
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.acceptance

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
WARNING = "Repeated failure:"
# Agent, routine, attempt and catalog identities differ between agent homes.
_IDENTITY = re.compile(r"[0-9a-f]{32,64}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ][0-9:.]+(?:Z|[+-]\d{2}:\d{2})?")
USAGE = ModelUsage(
    input_tokens=120,
    output_tokens=10,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)


def _tool(call_id: str, name: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        usage=USAGE,
    )


def _refused(call_id: str) -> ModelResponse:
    # Not a tool in any projection: refused before dispatch with a
    # correct_arguments recovery, identically each time.
    return _tool(call_id, "warehouse_row_count", table="orders")


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=USAGE)


def _profile(
    provider: MockModelProvider | ToolboxAwareMockModelProvider,
) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=False,
    )


def _status(request: ModelRequest) -> str:
    # Per-step status, including progress warnings, follows the transcript.
    status = request.messages[-1]
    assert status.role is MessageRole.SYSTEM
    text = status.content[0]
    assert isinstance(text, TextBlock)
    return text.text


def _warned(request: ModelRequest) -> bool:
    return WARNING in _status(request)


def _shape(messages: tuple[CanonicalMessage, ...]) -> tuple[str, ...]:
    """Transcript content with per-home identities and times masked."""

    rendered = []
    for message in messages:
        blocks = tuple(
            (
                {
                    "call_id": block.call_id,
                    "is_error": block.is_error,
                    "output": block.output,
                }
                if isinstance(block, ToolResultBlock)
                else (
                    {"text": block.text}
                    if isinstance(block, TextBlock)
                    else repr(block)
                )
            )
            for block in message.content
        )
        calls = tuple(
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        )
        text = canonical_json(
            {"role": message.role.value, "content": blocks, "calls": calls}
        )
        rendered.append(_TIMESTAMP.sub("<time>", _IDENTITY.sub("<id>", text)))
    return tuple(rendered)


def _disable_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RunSessionEvidence, "observe_progress", lambda self, call, result: None
    )


async def _foreground(tmp_path: Path) -> tuple[MockModelProvider, tuple[object, ...]]:
    provider = MockModelProvider(
        (
            _refused("first"),
            _refused("second"),
            _tool("exact-1", "toolbox_search", query="data_update_rows"),
            _tool("exact-2", "toolbox_search", query="data_update_rows"),
            _stop("No row-count tool is callable here."),
            _refused("next-run"),
            _stop("Still no row-count tool."),
        ),
        provider_id="mock:repeated-failure-foreground",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "repeated-failure-foreground",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
    )
    try:
        outcomes = []
        for prompt in ("Count the orders.", "Count the orders again."):
            result = await agent.run(prompt)
            transcript = await agent.transcript(result.run_id)
            outcomes.append(
                (
                    result.kind,
                    result.reason,
                    result.steps,
                    result.usage,
                    result.final_text,
                    _shape(transcript.messages),
                )
            )
        provider.assert_consumed()
        return provider, tuple(outcomes)
    finally:
        await agent.close()


async def test_foreground_run_warns_after_second_identical_refusal(
    tmp_path: Path,
) -> None:
    provider, _ = await _foreground(tmp_path)
    warned = tuple(_warned(request) for request in provider.requests)
    # Requests 0-4 are the first run; 5-6 are the next run, which starts fresh.
    assert warned == (False, False, True, True, True, False, False)
    status = _status(provider.requests[4])
    assert "`warehouse_row_count`" in status
    assert "exact-name searches found no callable tool named" in status
    assert "`data_update_rows`" in status


async def test_foreground_transcript_results_and_budgets_match_without_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, warned = await _foreground(tmp_path / "warned")
    _disable_warnings(monkeypatch)
    provider, silent = await _foreground(tmp_path / "silent")
    assert not any(_warned(request) for request in provider.requests)
    assert warned == silent


def _artifact_contract() -> OutcomeContract:
    requirement = ArtifactRequirement(
        required=True,
        minimum_count=1,
        maximum_count=1,
        allowed_media_types=("text/csv",),
        allowed_authorships=(ArtifactAuthorship.EXACT_SOURCE_DATA,),
        allowed_producer_capability_ids=(DATA_EXPORT_TABULAR_CAPABILITY_ID,),
        maximum_artifact_bytes=1024 * 1024,
        maximum_total_bytes=1024 * 1024,
        maximum_sensitivity=ModelSensitivity.INTERNAL,
    )
    return OutcomeContract(
        require_terminal_conclusion=True,
        artifact_requirements=(requirement,),
        maximum_total_artifact_bytes=1024 * 1024,
        maximum_effective_sensitivity=ModelSensitivity.INTERNAL,
        require_current_run_provenance=True,
        require_exact_source_bindings=True,
    )


async def _wait_for_deliveries(agent: Agent, conversation_id: str, count: int):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 15.0
    while loop.time() < deadline:
        inbox = await agent.list_deliveries(conversation_id=conversation_id)
        if len(inbox) >= count:
            return inbox
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} routine deliveries")


async def _routine(tmp_path: Path) -> tuple[MockModelProvider, tuple[object, ...]]:
    """Two consecutive occurrences; each misses its artifact and so fails."""

    tmp_path.mkdir()
    database = tmp_path / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO orders VALUES (7)")
    now = [NOW]
    clock: Callable[[], datetime] = lambda: now[0]  # noqa: E731
    provider = MockModelProvider(
        (
            _stop("Foreground authorization complete."),
            _refused("occurrence-1-a"),
            _refused("occurrence-1-b"),
            _stop("No row-count tool was callable."),
            _refused("occurrence-2-a"),
            _stop("No row-count tool was callable."),
        ),
        provider_id="mock:repeated-failure-routine",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "repeated-failure-routine",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=clock,
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        origin = await agent.run("Authorize a scheduled export of orders.")
        assert origin.conversation_id is not None
        (destination,) = await agent.distribution_destinations(
            origin.conversation_id,
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
        )
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Repeated refusal routine",
            authorized_instruction="Export the orders table as CSV.",
            schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(source.id,),
            allowed_connector_binding_ids=(),
            allowed_resource_ids=(resource.id,),
            allowed_capability_ids=(DATA_EXPORT_TABULAR_CAPABILITY_ID,),
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
            outcome_contract=_artifact_contract(),
            distribution_destination_id=destination.destination_id,
            eligible_model_routes=(provider.provider_id,),
            per_run_max_tokens=2_000,
            per_run_max_cost_usd=None,
            cumulative_max_tokens=20_000,
            cumulative_max_cost_usd=None,
            cumulative_max_attempts=10,
            cumulative_max_occurrences=10,
            maximum_consecutive_failures=3,
            expires_at=NOW + timedelta(days=30),
        )
        created = await agent.create_routine(await agent.propose_routine(draft))
        await agent.run_routine_now(
            created.routine_id, expected_revision=created.revision
        )
        await _wait_for_deliveries(agent, origin.conversation_id, 1)
        now[0] = NOW + timedelta(hours=1, seconds=1)
        agent._embedded._routine_supervisor.wake()
        inbox = await _wait_for_deliveries(agent, origin.conversation_id, 2)
        inspection = await agent.inspect_routine(created.routine_id)
        assert inspection is not None
        transcripts = []
        for occurrence in sorted(
            inspection.recent_occurrences, key=lambda item: item.scheduled_for
        ):
            assert occurrence.terminal_run_id is not None
            transcript = await agent.transcript(occurrence.terminal_run_id)
            transcripts.append(_shape(transcript.messages))
        provider.assert_consumed()
        return provider, (
            inspection.routine.consecutive_failures,
            tuple(sorted(item.failure_code or "" for item in inbox)),
            tuple(transcripts),
        )
    finally:
        await agent.close()


async def test_routine_occurrences_warn_separately_and_keep_their_failure_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, warned = await _routine(tmp_path / "warned")
    # Request 0 is the foreground authorization run, 1-3 the first occurrence
    # and 4-5 the second, whose single refusal does not inherit the first's.
    assert tuple(_warned(request) for request in provider.requests) == (
        False,
        False,
        False,
        True,
        False,
        False,
    )
    consecutive_failures, failure_codes, _ = warned
    assert consecutive_failures == 2
    assert failure_codes == (
        "outcome_artifact_contract_failed",
        "outcome_artifact_contract_failed",
    )

    _disable_warnings(monkeypatch)
    silent_provider, silent = await _routine(tmp_path / "silent")
    assert not any(_warned(request) for request in silent_provider.requests)
    assert warned == silent


def _graph_script() -> tuple[ModelResponse, ...]:
    return (
        _refused("attempt-1-a"),
        _refused("attempt-1-b"),
        # Plain text is a protocol violation; the task gets a second attempt.
        _stop("Done without a lifecycle result."),
        _refused("attempt-2-a"),
        _tool(
            "read-call",
            READ_TOOL_NAME,
            resource_ids=tuple(f"resource-{index}" for index in range(5)),
        ),
        _tool(
            "complete-call",
            "task_complete",
            result_kind="test.graph.read",
            summary="Read after correcting the call.",
            payload={"value": 42},
            evidence_refs=(result_ref("read-call"),),
            artifact_ids=(),
            residual_risk=None,
            downstream_constraints={},
        ),
    )


async def _graph(tmp_path: Path) -> tuple[MockModelProvider, tuple[object, ...]]:
    integration = await ModelGraphIntegration.open(tmp_path, script=_graph_script())
    loop = integration.supervisor._graph_model_loop
    assert loop is not None
    loop._context_builder = AgentContextBuilder(
        CatalogSpy(), profile=integration.provider.model_profile
    )
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        work = next(item for item in terminal.tasks if item.role is TaskRole.WORKER)
        attempts = tuple(
            item for item in terminal.attempts if item.task_id == work.task_id
        )
        transcripts = []
        for attempt in attempts:
            transcript = await integration.store.load(attempt.run_id)
            transcripts.append(_shape(transcript.messages))
        integration.provider.assert_consumed()
        return integration.provider, (
            terminal.job.state,
            work.attempt_count,
            tuple(item.state for item in attempts),
            tuple(
                sorted(
                    (
                        item.task_id is None,
                        item.dimension,
                        item.ceiling,
                        item.settled,
                        item.reserved,
                    )
                    for item in terminal.budget_ledgers
                )
            ),
            tuple(transcripts),
        )
    finally:
        await integration.close()


async def test_graph_task_attempts_warn_separately_without_changing_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, warned = await _graph(tmp_path / "warned")
    # Requests 0-2 are the first attempt; 3-5 the second, which starts fresh.
    assert tuple(_warned(request) for request in provider.requests) == (
        False,
        False,
        True,
        False,
        False,
        False,
    )
    assert "`warehouse_row_count`" in _status(provider.requests[2])

    _disable_warnings(monkeypatch)
    silent_provider, silent = await _graph(tmp_path / "silent")
    assert not any(_warned(request) for request in silent_provider.requests)
    assert warned == silent


async def test_repeated_invalid_sql_warns_on_the_second_identical_query(
    tmp_path: Path,
) -> None:
    database = tmp_path / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (value INTEGER NOT NULL)")
    provider = ToolboxAwareMockModelProvider(
        (),
        provider_id="mock:repeated-invalid-sql",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "repeated-invalid-sql",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
    )
    try:
        source = await agent.attach(SQLiteSource(database))
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        query = {
            "source_id": source.id,
            "resource_ids": (resource.id,),
            "sql": "SELECT missing_total FROM orders",
            "parameters": (),
        }
        provider.replace_script(
            (
                _tool("sql-1", "data_query", **query),
                _tool("sql-2", "data_query", **query),
                _stop("The orders table has no missing_total column."),
            )
        )
        result = await agent.run("Total the orders.", source_scope_ids=(source.id,))
        transcript = await agent.transcript(result.run_id)
        codes = {
            block.call_id: block.output["error"]["code"]  # type: ignore[index]
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        assert codes == {
            "sql-1": "sql_validation_failed",
            "sql-2": "sql_validation_failed",
        }
        assert tuple(_warned(request) for request in provider.requests) == (
            False,
            False,
            True,
        )
        assert "`data_query`" in _status(provider.requests[2])
    finally:
        await agent.close()
