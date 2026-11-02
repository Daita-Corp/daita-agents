from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    IntervalSchedule,
    RoutineIntent,
    RoutineState,
    SQLiteSource,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from tests.support.jobs import JobModel, open_job_agent, reply, tool_result
from tests.support.model_surface import identity_leaks
from tests.support.workspace import workspace_for


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=64_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


@pytest.mark.parametrize("update", [False, True])
async def test_routine_input_error_precedes_approval(tmp_path, update):
    from daita.capabilities import ApprovalDecision
    from tests.support.capability_runtime import execute_projected

    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        tuple(
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Authorize this report."
            )
            for _ in range(2)
        )
    )
    agent = await Agent.create(
        "routine-input",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        origin = await agent.run("Create a scheduled report.")
        now = datetime.now(UTC)
        arguments: dict[str, object] = {
            "title": "Report",
            "prompt": "Report completion.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
        }
        runtime = agent._embedded._capability_runtime
        store = agent._embedded._store
        run = (await store.load(origin.run_id)).run
        if update:
            created = await execute_projected(
                runtime, run, (ToolCall("valid-create", "routine_save", arguments),)
            )
            assert not created.ordered_results[0].is_error, created.ordered_results[
                0
            ].output
            current = (await agent.list_routines())[0]
            arguments = {"routine_id": current.routine_id}
        approvals.clear()
        arguments["schedule"] = {
            "kind": "once",
            "exact_at": (now - timedelta(hours=1)).isoformat(),
        }
        result = await execute_projected(
            runtime, run, (ToolCall("invalid-schedule", "routine_save", arguments),)
        )
        error = result.ordered_results[0].output["error"]
        assert error["code"] == "routine_schedule_passed", error
        assert not approvals
        assert len(await agent.list_routines()) == int(update)
        if update:
            assert (await agent.list_routines())[0].revision == current.revision
        assert not await agent.list_effects()
    finally:
        await agent.close()


async def test_public_routine_surface_walks_create_and_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "current.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
    connection.execute("INSERT INTO current_value VALUES (7)")
    connection.commit()
    connection.close()

    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
        ),
        provider_id="mock:routines-public",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "routine-public",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Current"))
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        origin = await agent.run("Read the current value for a scheduled report.")
        assert origin.reason == "completed"
        assert origin.conversation_id is not None
        now = datetime.now(UTC)
        intent = RoutineIntent(
            title="Current value report",
            prompt=(
                "Inspect the exact current_value resource and report its current value."
            ),
            schedule=IntervalSchedule(3_600, now + timedelta(hours=1)),
            sources=("Current",),
        )
        created = await agent.create_routine(
            intent, conversation_id=origin.conversation_id
        )
        assert created.state is RoutineState.ACTIVE
        assert created.source_ids == (source.id,)
        assert created.conversation_id == origin.conversation_id
        assert (await agent.list_routines())[0].routine_id == created.routine_id
        assert await agent.inspect_routine(created.routine_id) is not None

        revised = await agent.update_routine(
            created.routine_id,
            RoutineIntent(title="Revised current value report"),
            expected_revision=created.revision,
        )
        assert revised.revision == created.revision + 1
        assert revised.title == "Revised current value report"
        assert (revised.authorized_instruction, revised.source_ids) == (
            created.authorized_instruction,
            created.source_ids,
        )

        paused = await agent.pause_routine(
            revised.routine_id,
            expected_revision=revised.revision,
        )
        resumed = await agent.resume_routine(
            paused.routine_id,
            expected_revision=paused.revision,
        )
        await agent.set_memory("Unrelated private conversation note.")
        await agent.save_skill(
            "later-procedure",
            "Private later procedure",
            "Never part of this assignment.",
        )
        from daita import (
            ResourceRevisionBinding,
            SemanticAnnotation,
            SemanticEvidence,
            SemanticEvidenceKind,
            SemanticKind,
            SemanticSubject,
        )

        await agent.save_semantic_annotation(
            SemanticAnnotation(
                id="later-private-meaning",
                agent_id=agent.id,
                subject=SemanticSubject(
                    source_ids=(source.id,), resource_ids=(resource.id,), fields=()
                ),
                kind=SemanticKind.METRIC_DEFINITION,
                statement="Private later meaning of current_value.",
                evidence=(
                    SemanticEvidence(
                        SemanticEvidenceKind.USER_ASSERTION,
                        origin.run_id,
                        message_position=0,
                    ),
                ),
                catalog_revisions=(
                    ResourceRevisionBinding(resource.id, resource.current_revision),
                ),
                created_at=now,
                confirmed_at=now,
                sensitivity=ModelSensitivity.RESTRICTED,
            )
        )
        running = await agent.run_routine_now(
            resumed.routine_id,
            expected_revision=resumed.revision,
        )
        for _ in range(1_000):
            deliveries = await agent.list_deliveries(
                conversation_id=origin.conversation_id
            )
            if deliveries:
                break
            await asyncio.sleep(0.005)
        inspection = await agent.inspect_routine(running.routine_id)
        assert inspection is not None
        assert len(deliveries) == 1, (
            tuple(
                (
                    item.disposition,
                    item.failure_code,
                    item.reserved_run_id,
                    item.terminal_run_id,
                )
                for item in inspection.recent_occurrences
            ),
            len(provider.requests),
        )
        delivery = await agent.inspect_delivery(deliveries[0].delivery_id)
        assert delivery is not None
        scheduled_request = provider.requests[-1]
        assert scheduled_request.sensitivity is ModelSensitivity.INTERNAL
        scheduled_text = repr(scheduled_request.messages)
        assert intent.prompt is not None and intent.prompt in scheduled_text
        assert "Authorize the revised scheduled report definition" not in scheduled_text
        assert "Unrelated private conversation note" not in scheduled_text
        assert "later-procedure" not in scheduled_text
        assert "Private later meaning" not in scheduled_text
        assert delivery.delivery.outcome.conclusion_digest == (
            deliveries[0].conclusion_digest
        )
        disabled = await agent.disable_routine(
            running.routine_id,
            expected_revision=inspection.routine.revision,
        )
        assert disabled.state is RoutineState.DISABLED
    finally:
        await agent.close()


async def test_public_source_free_once_routine_commits_its_document(tmp_path):
    from daita import OnceSchedule
    from daita.distribution.models import OutcomeState
    from tests.support.toolbox_model import ToolboxAwareMockModelProvider

    usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
    provider = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The assignment is ready for approval.",
                usage=usage,
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                usage=usage,
                tool_calls=(
                    ToolCall(
                        "write-brief",
                        "artifact_create",
                        {
                            "from": "document",
                            "format": "markdown",
                            "content": "# Local briefing\nNo outside research was requested.",
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The requested document is available.",
                usage=usage,
            ),
        ),
        complete_pricing=True,
    )
    agent = await Agent.create(
        "source-free-routine",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=replace(provider.model_profile, context_window_tokens=64000),
    )
    try:
        origin = await agent.run("Prepare a one-time local briefing document.")
        assert origin.reason == "completed" and origin.conversation_id is not None
        created = await agent.create_routine(
            RoutineIntent(
                title="One-time local briefing",
                prompt=(
                    "Create a Markdown briefing documenting that no outside "
                    "research was requested."
                ),
                schedule=OnceSchedule(datetime.now(UTC) + timedelta(seconds=1)),
            ),
            conversation_id=origin.conversation_id,
        )
        for _ in range(1000):
            deliveries = await agent.list_deliveries(
                conversation_id=origin.conversation_id
            )
            if deliveries:
                break
            await asyncio.sleep(0.005)
        assert len(deliveries) == 1
        assert deliveries[0].conclusion_state is OutcomeState.SUCCEEDED, tuple(
            item.result
            for item in await agent.conversation_runs(origin.conversation_id)
        )
        assert len(deliveries[0].artifact_references) == 1
        # A routine's prompt counts as internal; it has no ceiling of its own.
        assert provider.requests[-1].sensitivity is ModelSensitivity.INTERNAL
        inspection = await agent.inspect_routine(created.routine_id)
        assert (
            inspection is not None
            and inspection.routine.state is RoutineState.COMPLETED
        )
        # The occurrence's run records the scope it started with.
        terminal_run_id = inspection.recent_occurrences[0].terminal_run_id
        scope = next(
            item.transcript.run.execution_scope
            for item in await agent.conversation_runs(origin.conversation_id)
            if item.transcript.run.id == terminal_run_id
        )
        assert (
            scope is not None
            and scope.allowed_source_ids
            == scope.allowed_resource_ids
            == scope.allowed_connector_binding_ids
            == ()
        )
        assert inspection.routine.next_due_at is None
    finally:
        await agent.close()


async def test_routine_save_creates_with_code_defaults_and_updates_one_revision(
    tmp_path,
):
    from daita.capabilities import ApprovalDecision
    from tests.support.capability_runtime import execute_projected
    from tests.support.toolbox_model import ToolboxAwareMockModelProvider

    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = ToolboxAwareMockModelProvider(
        [ModelResponse(finish_reason=FinishReason.STOP, text="Ready.")]
    )
    agent = await Agent.create(
        "routine-save",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=provider.model_profile,
        approval_handler=approve,
    )
    try:
        origin = await agent.run("Report hourly.")
        now = datetime.now(UTC)
        common = {
            "title": "Report",
            "prompt": "Report completion.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
        }
        runtime = agent._embedded._capability_runtime
        run = (await agent._embedded._store.load(origin.run_id)).run
        created = await execute_projected(
            runtime, run, (ToolCall("create", "routine_save", common),)
        )
        assert not created.ordered_results[0].is_error
        assert created.ordered_results[0].capability_id == "routines.create"
        assert [item.tool_name for item in approvals] == ["routine_save"]
        assert approvals[0].capability_id == "routines.create"
        (current,) = await agent.list_routines()
        inspection = await agent.inspect_routine(current.routine_id)
        assert inspection is not None
        routine = inspection.routine
        # The code defaults: this conversation, every current source, active.
        assert routine.conversation_id == origin.conversation_id
        assert routine.source_ids == ()
        assert routine.state is RoutineState.ACTIVE

        orphan = await execute_projected(
            runtime,
            run,
            (ToolCall("orphan", "routine_save", {**common, "expected_revision": 1}),),
        )
        assert (
            orphan.ordered_results[0].output["error"]["code"]
            == "invalid_argument_for_mode"
        )
        replacement = {"title": "Revised report", "routine_id": current.routine_id}
        stale = await execute_projected(
            runtime,
            run,
            (
                ToolCall(
                    "stale",
                    "routine_save",
                    {**replacement, "expected_revision": current.revision + 1},
                ),
            ),
        )
        assert (
            stale.ordered_results[0].output["error"]["code"]
            == "routine_revision_changed"
        )
        immediate = await execute_projected(
            runtime,
            run,
            (
                ToolCall(
                    "immediate",
                    "routine_save",
                    {**replacement, "run_immediately": True},
                ),
            ),
        )
        assert immediate.ordered_results[0].is_error
        revised = await execute_projected(
            runtime, run, (ToolCall("revise", "routine_save", replacement),)
        )
        assert not revised.ordered_results[0].is_error
        assert revised.ordered_results[0].capability_id == "routines.update"
        assert approvals[-1].capability_id == "routines.update"
        (after,) = await agent.list_routines()
        assert after.revision == current.revision + 1
        assert after.title == "Revised report"
        saved = await agent.inspect_routine(after.routine_id)
        assert saved is not None
        assert saved.routine.schedule == routine.schedule
        assert saved.routine.authorized_instruction == routine.authorized_instruction
    finally:
        await agent.close()


async def test_a_routine_run_shows_only_names_and_cannot_name_its_way_out(
    tmp_path: Path,
) -> None:
    shop, staff = tmp_path / "shop.db", tmp_path / "staff.db"
    with sqlite3.connect(shop) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    with sqlite3.connect(staff) as connection:
        connection.execute("CREATE TABLE people (id INTEGER PRIMARY KEY)")
    model = JobModel()
    reads = (
        ("in-scope", "data_query", {"source": "shop", "sql": "SELECT id FROM orders"}),
        ("source", "data_query", {"source": "staff", "sql": "SELECT id FROM people"}),
        ("sql", "data_query", {"source": "shop", "sql": "SELECT id FROM people"}),
        ("table", "catalog_describe", {"tables": ["main.people"]}),
    )
    model.script(
        user=[
            *(reply(ToolCall(call, name, args)) for call, name, args in reads),
            reply(text="Reported."),
        ]
    )
    agent = await open_job_agent(tmp_path, model)
    try:
        await agent.attach(SQLiteSource(shop, name="shop"))
        await agent.attach(SQLiteSource(staff, name="staff"))
        routine = await agent.create_routine(
            RoutineIntent(
                title="Orders",
                prompt="Report the orders table.",
                schedule=IntervalSchedule(
                    3_600, datetime.now(UTC) + timedelta(hours=1)
                ),
                sources=("shop",),
            )
        )
        await agent.run_routine_now(
            routine.routine_id, expected_revision=routine.revision
        )
        for _ in range(1_000):
            if await agent.list_deliveries(conversation_id=routine.conversation_id):
                break
            await asyncio.sleep(0.005)
    finally:
        await agent.close()
    last = model.requests[-1][1]
    codes = {
        call: tool_result(last, call).output.get("error", {}).get("code")  # type: ignore[attr-defined]
        for call, _, _ in reads
    }
    assert codes == {
        "in-scope": None,
        "source": "source_unknown",
        "sql": "resource_read_not_allowed",
        "table": "table_unknown",
    }
    assert identity_leaks(request for _, request in model.requests) == []
