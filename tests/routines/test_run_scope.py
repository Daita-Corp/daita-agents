"""A scheduled run's scope is fixed when it starts, minus the exclusions."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    ScheduledRoutineDraft,
    SQLiteSource,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.loop.models import Transcript
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)
USAGE = ModelUsage(
    input_tokens=100,
    output_tokens=10,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)
Step = ModelResponse | Callable[[ModelRequest], Awaitable[ModelResponse]]


class _Model:
    """Scripted model whose steps may await, so a test can act mid-run."""

    provider_id = "mock:run-scope"
    model_profile = ModelProfile(
        id=provider_id,
        context_window_tokens=64_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        assert self.steps, "script exhausted"
        step = self.steps.pop(0)
        return step if isinstance(step, ModelResponse) else await step(request)

    async def close(self, *, deadline: float | None = None) -> None:
        return None


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(call_id, name, arguments),),
        usage=USAGE,
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=USAGE)


def _table(path: Path, name: str, value: int) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE {name} (value INTEGER NOT NULL)")
        connection.execute(f"INSERT INTO {name} VALUES ({value})")
    return path


def _tool_result(transcript: Transcript, call_id: str) -> ToolResultBlock:
    return next(
        block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )


async def _agent(tmp_path: Path, model: _Model) -> Agent:
    return await Agent.create(
        "run-scope",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=model,
        model_profile=model.model_profile,
        clock=lambda: NOW,
    )


async def _routine(agent: Agent, model: _Model) -> tuple[str, str, str]:
    """A routine naming no sources, so each run takes every current source."""

    model.steps.append(_stop("Noted."))
    origin = await agent.run("Report the current values every hour.")
    assert origin.conversation_id is not None
    (destination,) = await agent.distribution_destinations(origin.conversation_id)
    routine = await agent.create_routine(
        await agent.propose_routine(
            ScheduledRoutineDraft(
                origin_run_id=origin.run_id,
                title="Current values",
                authorized_instruction="Report the current values.",
                schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
                misfire_policy=MisfirePolicy.LATEST_ONLY,
                reporting_mode=ReportingMode.ALWAYS,
                precheck=None,
                allowed_source_ids=(),
                allowed_connector_binding_ids=(),
                allowed_resource_ids=(),
                allowed_capability_ids=("artifact.create_document",),
                sensitivity_ceiling=ModelSensitivity.INTERNAL,
                outcome_contract=no_artifact_outcome_contract(),
                distribution_destination_id=destination.destination_id,
                eligible_model_routes=(model.provider_id,),
                per_run_max_tokens=2_000,
                per_run_max_cost_usd=None,
                cumulative_max_tokens=20_000,
                cumulative_max_cost_usd=None,
                cumulative_max_attempts=10,
                cumulative_max_occurrences=10,
                maximum_consecutive_failures=3,
                expires_at=NOW + timedelta(days=30),
            )
        )
    )
    return routine.routine_id, origin.conversation_id, origin.run_id


async def _run_once(
    agent: Agent, routine_id: str, conversation_id: str, seen: set[str]
) -> Transcript:
    """Run the routine now and return the transcript of its new delivery."""

    inspection = await agent.inspect_routine(routine_id)
    assert inspection is not None
    await agent.run_routine_now(
        routine_id, expected_revision=inspection.routine.revision
    )
    async with asyncio.timeout(30):
        while True:
            # The fixed clock gives every delivery the same time; match by id.
            new = [
                item
                for item in await agent.list_deliveries(
                    conversation_id=conversation_id, include_acknowledged=True
                )
                if item.delivery_id not in seen
            ]
            if new:
                break
            await asyncio.sleep(0.01)
    (delivery,) = new
    seen.add(delivery.delivery_id)
    run_id = delivery.resulting_run_id
    assert run_id is not None
    return await agent.transcript(run_id)


async def test_a_run_started_before_a_source_was_added_never_sees_it(
    tmp_path: Path,
) -> None:
    first = _table(tmp_path / "first.sqlite", "first_value", 1)
    second = _table(tmp_path / "second.sqlite", "second_value", 2)
    model = _Model()
    agent = await _agent(tmp_path, model)
    try:
        source_a = await agent.attach(SQLiteSource(first))
        routine_id, conversation_id, _origin = await _routine(agent, model)
        seen: set[str] = set()
        started, release = asyncio.Event(), asyncio.Event()
        added: dict[str, str] = {}

        async def query_after_attach(request: ModelRequest) -> ModelResponse:
            started.set()
            await release.wait()
            return _call(
                "query-second",
                "data_query",
                {
                    "source_id": added["source"],
                    "resource_ids": [added["resource"]],
                    "sql": "SELECT value FROM second_value",
                },
            )

        model.steps += [query_after_attach, _stop("Reported.")]
        running = asyncio.create_task(
            _run_once(agent, routine_id, conversation_id, seen)
        )
        await asyncio.wait_for(started.wait(), 30)
        # The run already holds its scope; a source attached now is not in it.
        source_b = await agent.attach(SQLiteSource(second))
        (resource_b,) = await agent.list_catalog_resources(source_id=source_b.id)
        added.update(source=source_b.id, resource=resource_b.id)
        release.set()
        before = await running
        assert before.run.execution_scope is not None
        assert before.run.execution_scope.allowed_source_ids == (source_a.id,)
        assert _tool_result(before, "query-second").is_error

        model.steps += [
            _call(
                "query-second-again",
                "data_query",
                {
                    "source_id": source_b.id,
                    "resource_ids": [resource_b.id],
                    "sql": "SELECT value FROM second_value",
                },
            ),
            _stop("Reported."),
        ]
        after = await _run_once(agent, routine_id, conversation_id, seen)
        assert after.run.execution_scope is not None
        assert set(after.run.execution_scope.allowed_source_ids) == {
            source_a.id,
            source_b.id,
        }
        result = _tool_result(after, "query-second-again")
        assert not result.is_error, result.output
        assert not model.steps
    finally:
        await agent.close()


async def test_scheduled_runs_cannot_use_local_files_or_routine_and_job_tools(
    tmp_path: Path,
) -> None:
    model = _Model()
    agent = await _agent(tmp_path, model)
    try:
        (agent.workspace.root / "notes.txt").write_text("private notes\n")
        routine_id, conversation_id, origin_run_id = await _routine(agent, model)
        foreground = await agent.transcript(origin_run_id)
        assert foreground.run.resolved_source_scope is not None
        # The person's own run reaches the session folder; scheduled runs never do.
        (session_source,) = foreground.run.resolved_source_scope.source_ids
        model.steps += [
            _call(
                "load-management",
                "toolbox_load",
                {"tool_names": ["routine_save", "routine_control", "job_cancel"]},
            ),
            _call(
                "read-local",
                "file_read",
                {"source_id": session_source, "path": "notes.txt"},
            ),
            _stop("Done."),
        ]
        scheduled = await _run_once(agent, routine_id, conversation_id, set())
        scope = scheduled.run.execution_scope
        assert scope is not None
        assert session_source not in scope.allowed_source_ids
        assert not any(
            capability_id.startswith(("routines.", "jobs."))
            for capability_id in scope.allowed_capability_ids
        )
        assert _tool_result(scheduled, "load-management").is_error
        assert _tool_result(scheduled, "read-local").is_error
        assert not model.steps
    finally:
        await agent.close()
