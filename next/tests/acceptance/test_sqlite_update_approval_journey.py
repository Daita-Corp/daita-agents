from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from daita import Agent, SQLiteSource
from daita.catalog import ResourceKind, catalog_resource_id
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from daita.loop.models import LoopExitKind
from daita.operations.governance import ApprovalStatus
from daita.operations.models import TaskStatus

NOW = datetime(2026, 7, 19, 15, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:sqlite-update-journey",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


class JourneyProvider:
    provider_id = PROFILE.id

    def __init__(self) -> None:
        self.script: list[ModelResponse] = []
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("unexpected model call")
        return self.script.pop(0)


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE orders (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL
            );
            INSERT INTO orders (id, status) VALUES ('order-1', 'pending');
            """)


def _status(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status FROM orders WHERE id = 'order-1'"
        ).fetchone()
    assert row is not None
    return str(row[0])


async def test_public_sqlite_update_waits_reopens_and_resumes_exactly_once(
    tmp_path: Path,
) -> None:
    database = tmp_path / "orders.db"
    _database(database)
    root = tmp_path / "state"
    provider = JourneyProvider()
    id_factory = _ids()
    agent = await Agent.create(
        "atlas",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
        id_factory=id_factory,
    )
    registration = await agent.attach(
        SQLiteSource(database, name="Orders", allow_writes=True)
    )
    resource_id = catalog_resource_id(
        registration.id,
        ResourceKind.TABLE,
        "main.orders",
    )
    recipe: dict[str, object] = {
        "source_id": registration.id,
        "resource_id": resource_id,
        "key_column": "id",
        "key_value": "order-1",
        "target_column": "status",
        "expected_value": "pending",
        "new_value": "complete",
    }
    provider.script.extend(
        (
            _tool(
                "call-preview",
                "data_preview_sqlite_update",
                recipe,
            ),
            _tool(
                "call-update",
                "data_update_sqlite",
                {**recipe, "impact_evidence_id": "evidence-1"},
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Order order-1 is complete. [evidence:evidence-2]",
            ),
        )
    )

    waiting_exit = await agent.run(
        "Change order-1 from pending to complete after preview and approval.",
        session_id="session-atlas",
    )
    waiting = await agent.inspect(waiting_exit.operation_id)

    assert waiting_exit.kind is LoopExitKind.WAITING
    assert _status(database) == "pending"
    assert len(provider.requests) == 2
    assert [task.capability_id for task in waiting.tasks] == [
        "data.sqlite.update_impact",
        "data.sqlite.update",
    ]
    assert [task.status for task in waiting.tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.WAITING_FOR_APPROVAL,
    ]
    preview_task = waiting.tasks[0]
    impact_evidence = waiting.evidence[0]
    assert impact_evidence.id == "evidence-1"
    assert impact_evidence.task_id == preview_task.id
    assert preview_task.evidence_ids == (impact_evidence.id,)
    assert impact_evidence.payload["eligible_rows"] == 1
    assert len(waiting.approvals) == 1
    assert waiting.approvals[0].status is ApprovalStatus.PENDING
    write_task = waiting.tasks[1]
    assert write_task.execution_facts.side_effecting is True
    assert write_task.execution_facts.idempotent is False
    assert write_task.execution_facts.replay_safe is False
    validation = write_task.execution_facts.validation_facts
    assert validation.schema_version == 1
    assert validation.validation_passed is True
    assert validation.in_scope is True
    assert validation.destructive is False
    assert validation.source_id == registration.id
    assert validation.resource_ids == (resource_id,)
    assert validation.evidence_ids == ("evidence-1",)
    assert validation.impact["eligible_rows"] == 1
    assert validation.impact["maximum_rows"] == 1
    assert waiting.task_dependencies[0].task_id == write_task.id
    assert waiting.task_dependencies[0].prerequisite_task_id == waiting.tasks[0].id
    prior_model_call_ids = tuple(item.id for item in waiting.model_calls)
    prior_turn_ids = tuple(item.id for item in waiting.turns)
    operation_id = waiting.operation.id
    approval_id = waiting.approvals[0].id
    await agent.close()

    reopened = await Agent.open(
        "atlas",
        root=root,
        model=provider,
        model_profile=PROFILE,
        clock=lambda: NOW,
        id_factory=id_factory,
    )
    persisted_wait = await reopened.inspect(operation_id)
    assert persisted_wait == waiting

    decision = await reopened.approve(
        approval_id,
        decided_by="reviewer-7",
        reason="The one-row impact and expected current value were reviewed.",
    )
    after_decision = await reopened.inspect(operation_id)
    assert decision.status is ApprovalStatus.APPROVED
    assert after_decision.tasks == waiting.tasks
    assert _status(database) == "pending"
    assert len(provider.requests) == 2

    completed_exit = await reopened.resume(operation_id)
    completed = await reopened.inspect(operation_id)

    assert completed_exit.kind is LoopExitKind.COMPLETED
    assert completed_exit.operation_id == operation_id
    assert completed_exit.final_text == (
        "Order order-1 is complete. [evidence:evidence-2]"
    )
    assert _status(database) == "complete"
    assert tuple(item.id for item in completed.model_calls[:2]) == prior_model_call_ids
    assert tuple(item.id for item in completed.turns[:2]) == prior_turn_ids
    assert len(provider.requests) == 3
    assert [task.status for task in completed.tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]
    completed_write_task = completed.tasks[1]
    assert completed_write_task.id == write_task.id
    result_evidence = completed.evidence[-1]
    assert result_evidence.id == "evidence-2"
    assert result_evidence.task_id == completed_write_task.id
    assert completed_write_task.evidence_ids == (result_evidence.id,)
    assert result_evidence.kind == "data.sqlite.update_result"
    assert result_evidence.payload["affected_rows"] == 1
    assert result_evidence.payload["maximum_rows"] == 1
    assert result_evidence.payload["impact_evidence_id"] == "evidence-1"

    approval = completed.approvals[0]
    assert approval.task_id == write_task.id
    assert approval.status is ApprovalStatus.APPROVED
    assert approval.decided_by == "reviewer-7"
    assert approval.decision_reason == (
        "The one-row impact and expected current value were reviewed."
    )
    requested = next(
        event for event in completed.events if event.type == "approval.requested"
    )
    assert requested.approval_id == approval.id
    assert requested.task_id == write_task.id
    assert requested.payload["task_fingerprint"] == approval.task_fingerprint
    requested_facts = requested.payload["facts"]
    assert isinstance(requested_facts, Mapping)
    requested_validation = requested_facts["validation"]
    assert isinstance(requested_validation, Mapping)
    assert requested_validation["source_id"] == registration.id
    assert requested_validation["evidence_ids"] == ("evidence-1",)
    assert {
        "approval.approved",
        "approval.applied",
        "executor.started",
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
    } <= {event.type for event in completed.events}
    update_executor_completed = next(
        event
        for event in completed.events
        if event.type == "executor.completed" and event.task_id == write_task.id
    )
    final_model_started = next(
        event
        for event in completed.events
        if event.type == "model_call.started"
        and event.model_call_id == completed.model_calls[-1].id
    )
    assert completed.events.index(update_executor_completed) < completed.events.index(
        final_model_started
    )
    accepted_result = next(
        event
        for event in completed.events
        if event.type == "evidence.accepted" and event.task_id == write_task.id
    )
    assert accepted_result.evidence_id == result_evidence.id
    succeeded_write = next(
        event
        for event in completed.events
        if event.type == "task.succeeded" and event.task_id == write_task.id
    )
    assert succeeded_write.evidence_id == result_evidence.id

    replayed_exit = await reopened.resume(operation_id)
    replayed = await reopened.inspect(operation_id)
    await reopened.close()

    assert replayed_exit == completed_exit
    assert replayed == completed
    assert len(provider.requests) == 3
    assert _status(database) == "complete"
    assert (
        sum(evidence.id == impact_evidence.id for evidence in completed.evidence) == 1
    )
    for event_type in (
        "executor.started",
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
    ):
        assert (
            sum(
                event.type == event_type and event.task_id == preview_task.id
                for event in completed.events
            )
            == 1
        )
    assert (
        sum(
            event.type == "executor.started" and event.task_id == write_task.id
            for event in completed.events
        )
        == 1
    )
    for event_type in (
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
    ):
        assert (
            sum(
                event.type == event_type and event.task_id == write_task.id
                for event in completed.events
            )
            == 1
        )
    assert provider.script == []
