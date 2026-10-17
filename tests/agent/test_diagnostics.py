"""The host diagnostic sink: every run origin, payload-free, never steering."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import stat
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from daita import (
    Agent,
    ArtifactRequirement,
    GraphState,
    IntervalSchedule,
    MisfirePolicy,
    OutcomeContract,
    ReportingMode,
    ScheduledRoutineDraft,
    SQLiteSource,
)
from daita._json import canonical_json
from daita.artifacts.models import ArtifactAuthorship
from daita.domains.data.export_capabilities import DATA_EXPORT_TABULAR_CAPABILITY_ID
from daita.evaluation import measure_observer_events
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
    tool_result_refs,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop import LoopExitKind
from daita.observation import (
    AgentEvent,
    AgentEventKind,
    PrivateDiagnosticSink,
    summarize_diagnostics,
)
from tests.jobs.graph.test_public_composition import (
    _ProductionGraphProvider,
    _wait_terminal,
)
from tests.support.job_benchmarks import create_probe_home
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.acceptance

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
SENTINEL = "zq-diagnostic-sentinel-7f3a"
REPOSITORY = Path(__file__).resolve().parents[2]
USAGE = ModelUsage(
    input_tokens=120,
    output_tokens=10,
    cache_read_tokens=64,
    cost_estimate=CostEstimate.complete(Decimal("0.002")),
)
_IDENTITY = re.compile(r"[0-9a-f]{32,64}")


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )


def _foreground_provider() -> MockModelProvider:
    return MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="search-1",
                        name="toolbox_search",
                        arguments={"query": f"find {SENTINEL} tools"},
                    ),
                ),
                usage=USAGE,
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=f"The answer mentions {SENTINEL}.",
                usage=USAGE,
            ),
        ),
        provider_id="mock:diagnostics",
        complete_pricing=True,
    )


async def _foreground(
    tmp_path: Path,
    *,
    diagnostics_path: Path | None,
    observer: Callable[[AgentEvent], None] | None = None,
):
    provider = _foreground_provider()
    agent = await Agent.create(
        "diagnostics-foreground",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
        observer=observer,
        diagnostics_path=diagnostics_path,
    )
    try:
        result = await agent.run(f"Please look up {SENTINEL} for me.")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()
    provider.assert_consumed()
    return result, transcript


def _masked(result, transcript) -> tuple[object, ...]:
    return (
        result.kind,
        result.reason,
        result.steps,
        result.usage,
        result.final_text,
        tuple(
            _IDENTITY.sub("<id>", canonical_json(_message(message)))
            for message in transcript.messages
        ),
    )


def _message(message) -> dict[str, object]:
    return {
        "role": message.role.value,
        "content": [
            (
                {"call_id": block.call_id, "output": block.output}
                if isinstance(block, ToolResultBlock)
                else repr(block)
            )
            for block in message.content
        ],
        "calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
    }


async def test_foreground_records_are_private_payload_free_and_sized(tmp_path):
    path = tmp_path / "private" / "diagnostics.jsonl"
    observed: list[AgentEvent] = []
    result, transcript = await _foreground(
        tmp_path, diagnostics_path=path, observer=observed.append
    )
    assert result.kind is LoopExitKind.COMPLETED

    text = path.read_text()
    assert SENTINEL not in text
    assert "Please look up" not in text
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700

    records = _records(path)
    assert [record["kind"] for record in records] == [
        "run.started",
        "model.completed",
        "tool.completed",
        "model.completed",
        "run.completed",
    ]
    # The per-call observer keeps working and receives the same events.
    assert [
        event.kind.value
        for event in observed
        if event.kind.value in {record["kind"] for record in records}
    ] == [record["kind"] for record in records]
    assert {record["run_origin"] for record in records} == {"user"}

    first, second = (r["data"] for r in records if r["kind"] == "model.completed")
    assert first["cache_read_tokens"] == 64
    assert first["cache_read_tokens"] <= first["input_tokens"] == 120
    assert first["remaining_tokens_before"] > second["remaining_tokens_before"]
    # The host routes the injected provider, so each step records its attempt.
    assert first["attempt_count"] == 1
    assert [item["outcome"] for item in first["attempts"]] == ["completed"]
    assert first["cost_amount_usd"] == "0.002"

    (tool,) = (r["data"] for r in records if r["kind"] == "tool.completed")
    (block,) = (
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )
    ref = tool_result_refs(transcript.messages).by_call_id[block.call_id]
    assert tool["result_bytes"] == {
        "stored": len(canonical_json(block.output).encode()),
        "model": len(canonical_json(model_tool_result(block, ref)).encode()),
    }
    assert set(tool) <= {
        "call_id",
        "tool_name",
        "capability_id",
        "toolbox_id",
        "duration_ms",
        "success",
        "error_code",
        "result_bytes",
    }


async def test_a_failing_sink_changes_nothing(tmp_path, monkeypatch):
    plain = await _foreground(tmp_path / "plain", diagnostics_path=None)

    def broken(self, event):
        raise OSError("disk full")

    monkeypatch.setattr(PrivateDiagnosticSink, "__call__", broken)
    observed: list[AgentEvent] = []
    failing = await _foreground(
        tmp_path / "failing",
        diagnostics_path=tmp_path / "failing-diagnostics.jsonl",
        observer=observed.append,
    )
    assert _masked(*failing) == _masked(*plain)
    assert measure_observer_events(observed).model_calls == 2
    assert AgentEventKind.RUN_COMPLETED in {event.kind for event in observed}


def test_sink_rejects_repository_paths_relative_paths_and_symlinks(tmp_path):
    with pytest.raises(ValueError, match="outside a source repository"):
        PrivateDiagnosticSink(REPOSITORY / "diagnostics.jsonl")
    with pytest.raises(ValueError, match="outside a source repository"):
        PrivateDiagnosticSink(REPOSITORY / "src" / "nested" / "diagnostics.jsonl")
    assert not (REPOSITORY / "diagnostics.jsonl").exists()
    with pytest.raises(ValueError, match="absolute"):
        PrivateDiagnosticSink(Path("diagnostics.jsonl"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\n")
    with pytest.raises(ValueError, match="outside a source repository"):
        PrivateDiagnosticSink(project / "out" / "diagnostics.jsonl")

    target = tmp_path / "target.jsonl"
    target.write_text("")
    target.chmod(0o600)
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(OSError):
        PrivateDiagnosticSink(link)
    shared = tmp_path / "shared.jsonl"
    shared.write_text("")
    shared.chmod(0o644)
    with pytest.raises(ValueError, match="readable by other users"):
        PrivateDiagnosticSink(shared)


async def test_host_rejects_a_repository_diagnostics_path_before_opening(tmp_path):
    with pytest.raises(ValueError, match="outside a source repository"):
        await Agent.create(
            "diagnostics-rejected",
            root=tmp_path / "state",
            workspace=workspace_for(tmp_path),
            diagnostics_path=REPOSITORY / "diagnostics.jsonl",
        )
    assert not (tmp_path / "state").exists()


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


async def test_routine_run_reaches_the_host_sink_without_an_observer(tmp_path):
    database = tmp_path / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO orders VALUES (7)")
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Authorized.", usage=USAGE
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text=f"Routine answer {SENTINEL}.",
                usage=USAGE,
            ),
        ),
        provider_id="mock:diagnostics-routine",
        complete_pricing=True,
    )
    path = tmp_path / "diagnostics" / "routine.jsonl"
    agent = await Agent.create(
        "diagnostics-routine",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
        diagnostics_path=path,
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
            title="Diagnostics routine",
            authorized_instruction=f"Export the orders table as CSV {SENTINEL}.",
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
        deadline = asyncio.get_running_loop().time() + 15
        while not await agent.inbox(conversation_id=origin.conversation_id):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
    finally:
        await agent.close()
    provider.assert_consumed()

    records = _records(path)
    routine = [r for r in records if r["run_origin"] == "scheduled_routine"]
    assert [r["kind"] for r in routine] == [
        "run.started",
        "model.completed",
        "run.completed",
    ]
    assert routine[1]["data"]["input_tokens"] == 120
    assert SENTINEL not in path.read_text()
    summary = {item["run_id"]: item for item in summarize_diagnostics(path)}
    assert summary[routine[0]["run_id"]]["run_origin"] == "scheduled_routine"
    assert summary[routine[0]["run_id"]]["completed_model_steps"] == 1


async def test_graph_task_run_reaches_the_host_sink_without_an_observer(tmp_path):
    home = await create_probe_home(tmp_path, "diagnostics-graph", distractor_tables=1)
    provider = _ProductionGraphProvider(
        source_id=home.source_id,
        resources=tuple(sorted(home.resource_ids.items())),
    )
    path = tmp_path / "diagnostics" / "graph.jsonl"
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(home.root),
        diagnostics_path=path,
    )
    try:
        foreground = await agent.run(
            "Start a durable graph query over all exact resources.",
            source_scope_ids=(home.source_id,),
        )
        assert foreground.kind is LoopExitKind.COMPLETED
        (job,) = await agent.list_jobs()
        inspection = await _wait_terminal(agent, job.job_id)
        assert inspection.job.state is GraphState.SUCCEEDED
    finally:
        await agent.close()

    records = _records(path)
    task = [r for r in records if r["run_origin"] == "job_task"]
    assert {r["kind"] for r in task} >= {
        "run.started",
        "model.completed",
        "tool.completed",
        "run.completed",
    }
    query = next(
        r["data"]
        for r in task
        if r["kind"] == "tool.completed" and r["data"]["tool_name"] == "data_query"
    )
    assert query["result_bytes"]["stored"] > 0
    assert query["result_bytes"]["model"] > query["result_bytes"]["stored"]
    assert "SELECT" not in path.read_text()
    origins = {item["run_origin"] for item in summarize_diagnostics(path)}
    assert origins >= {"user", "job_task"}


def test_summary_groups_runs_and_never_sums_unknown_as_zero(tmp_path):
    def record(kind: str, run_id: str, data: dict[str, object]) -> str:
        return canonical_json(
            {
                "kind": kind,
                "occurred_at": NOW.isoformat(),
                "run_id": run_id,
                "conversation_id": "conversation",
                "run_origin": "user",
                "data": data,
            }
        )

    def attempt(outcome: str, retried: bool, tokens: int | None) -> dict:
        return {
            "outcome": outcome,
            "retried": retried,
            "usage_status": "reported" if tokens is not None else "unavailable",
            "input_tokens": tokens,
        }

    path = tmp_path / "summary.jsonl"
    path.write_text(
        "\n".join(
            (
                record(
                    "model.completed",
                    "run-a",
                    {
                        "model_call_index": 1,
                        "usage_status": "partial",
                        "input_tokens": 100,
                        "cache_read_tokens": 40,
                        "output_tokens": 5,
                        "attempts": [
                            attempt("stalled", True, None),
                            attempt("completed", False, 100),
                        ],
                    },
                ),
                record(
                    "tool.completed",
                    "run-a",
                    {
                        "tool_name": "small",
                        "call_id": "c1",
                        "success": True,
                        "result_bytes": {"stored": 10, "model": 40},
                    },
                ),
                record(
                    "tool.completed",
                    "run-a",
                    {
                        "tool_name": "large",
                        "call_id": "c2",
                        "success": True,
                        "result_bytes": {"stored": 74_000, "model": 74_100},
                    },
                ),
                record(
                    "model.failed",
                    "run-b",
                    {
                        "model_call_index": 1,
                        "error_code": "cancelled",
                        "usage_status": "unavailable",
                        "input_tokens": None,
                        "cache_read_tokens": None,
                        "output_tokens": None,
                        "attempts": None,
                    },
                ),
                record("run.completed", "run-b", {"exit_kind": "interrupted"}),
                "not json",
            )
        )
    )
    first, second = cast(list[dict[str, Any]], summarize_diagnostics(path))
    assert first["run_id"] == "run-a"
    assert first["completed_model_steps"] == 1
    assert first["model_attempts"] == 2
    assert first["retries"] == 1
    assert first["stalls"] == 1
    assert first["usage_complete"] is False
    assert first["input_tokens"] == 100 and first["cache_read_tokens"] == 40
    assert [item["tool_name"] for item in first["largest_tool_results"]] == [
        "large",
        "small",
    ]
    assert second["failed_model_steps"] == 1
    assert second["exit_kind"] == "interrupted"
    assert second["input_tokens"] is None
    assert second["model_attempts"] is None
