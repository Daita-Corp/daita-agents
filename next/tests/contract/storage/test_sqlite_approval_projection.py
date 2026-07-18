from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from daita.events.models import RuntimeEvent
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Turn
from daita.operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from daita.operations.governance import ApprovalRequest, ApprovalStatus
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    Task,
    TaskStatus,
    TriggerKind,
)
from daita.operations.store import (
    InvalidOperationCheckpointError,
    OperationNotFoundError,
)
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 18, 12, 30, tzinfo=timezone.utc)
DECIDED_AT = NOW + timedelta(seconds=5)
TASK_FINGERPRINT = "sha256:" + ("a" * 64)
POLICY_FINGERPRINT = "sha256:" + ("b" * 64)


def _event(
    snapshot: OperationSnapshot | None,
    *,
    operation_id: str,
    event_id: str,
    event_type: str,
    created_at: datetime,
    turn_id: str | None = None,
    model_call_id: str | None = None,
    call_id: str | None = None,
    task_id: str | None = None,
    approval_id: str | None = None,
) -> RuntimeEvent:
    agent_id = "agent-approval" if snapshot is None else snapshot.operation.agent_id
    session_id = (
        "session-approval" if snapshot is None else snapshot.operation.session_id
    )
    return RuntimeEvent(
        id=event_id,
        type=event_type,
        agent_id=agent_id,
        operation_id=operation_id,
        session_id=session_id,
        turn_id=turn_id,
        model_call_id=model_call_id,
        call_id=call_id,
        task_id=task_id,
        approval_id=approval_id,
        capability_id="fake.write" if task_id is not None else None,
        executor_id="fake.write.executor" if task_id is not None else None,
        payload={"event_id": event_id},
        created_at=created_at,
    )


def _waiting_snapshot(
    *,
    operation_id: str = "operation-approval",
    trigger_id: str = "trigger-approval",
    approval_id: str = "approval-global",
) -> OperationSnapshot:
    turn_id = f"{operation_id}:turn"
    model_call_id = f"{operation_id}:model"
    call_id = f"{operation_id}:call"
    task_id = f"{operation_id}:task"
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn_id,
        messages=(
            CanonicalMessage(
                agent_id="agent-approval",
                operation_id=operation_id,
                session_id="session-approval",
                turn_id=turn_id,
                role=MessageRole.USER,
                content=(TextBlock("Change the durable marker."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="fake_write",
                description="Change one test-owned durable marker.",
                input_schema={"type": "object"},
            ),
        ),
    )
    response = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id=call_id,
                name="fake_write",
                arguments={"value": "approved"},
            ),
        ),
    )
    trigger = AgentTrigger(
        id=trigger_id,
        agent_id="agent-approval",
        kind=TriggerKind.USER,
        source_id="user-approval",
        session_id="session-approval",
        payload={"message": "change marker"},
        created_at=NOW,
    )
    operation = Operation(
        id=operation_id,
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        session_id=trigger.session_id,
        status=OperationStatus.WAITING_FOR_APPROVAL,
        created_at=NOW,
        updated_at=NOW,
    )
    turn = Turn(
        id=turn_id,
        operation_id=operation_id,
        number=1,
        model_request_id=model_call_id,
        model_response_id=model_call_id,
        created_at=NOW,
    )
    model_call = ModelCall(
        id=model_call_id,
        operation_id=operation_id,
        turn_id=turn_id,
        provider_id="mock:approval",
        request=request,
        response=response,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    task = Task(
        id=task_id,
        operation_id=operation_id,
        turn_id=turn_id,
        call_id=call_id,
        capability_id="fake.write",
        executor_id="fake.write.executor",
        status=TaskStatus.WAITING_FOR_APPROVAL,
        attempt=1,
        arguments={"value": "approved"},
        created_at=NOW,
        updated_at=NOW,
    )
    approval = ApprovalRequest(
        id=approval_id,
        operation_id=operation_id,
        task_id=task_id,
        task_fingerprint=TASK_FINGERPRINT,
        policy_fingerprint=POLICY_FINGERPRINT,
        requested_at=NOW,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.AWAITING_APPROVAL,
            turn_count=1,
            action_count=1,
            waiting_approval_id=approval_id,
        ),
        budgets=LoopBudgets(),
        turns=(turn,),
        model_calls=(model_call,),
        readiness=(),
        tasks=(task,),
        approvals=(approval,),
        evidence=(),
        observations=(),
        events=(
            _event(
                None,
                operation_id=operation_id,
                event_id=f"{operation_id}:created",
                event_type="operation.created",
                created_at=NOW,
            ),
            _event(
                None,
                operation_id=operation_id,
                event_id=f"{operation_id}:approval-requested",
                event_type="approval.requested",
                created_at=NOW,
                turn_id=turn_id,
                model_call_id=model_call_id,
                call_id=call_id,
                task_id=task_id,
                approval_id=approval_id,
            ),
        ),
    )


def _decided_snapshot(
    snapshot: OperationSnapshot,
    status: ApprovalStatus = ApprovalStatus.APPROVED,
) -> OperationSnapshot:
    approval = replace(
        snapshot.approvals[0],
        status=status,
        decided_at=DECIDED_AT,
        decided_by="user:test-approver",
        decision_reason="The exact durable marker change was reviewed.",
    )
    task = snapshot.tasks[0]
    model_call = snapshot.model_calls[0]
    return replace(
        snapshot,
        approvals=(approval,),
        events=(
            *snapshot.events,
            _event(
                snapshot,
                operation_id=snapshot.operation.id,
                event_id=f"{snapshot.operation.id}:approval-{status.value}",
                event_type=f"approval.{status.value}",
                created_at=DECIDED_AT,
                turn_id=task.turn_id,
                model_call_id=model_call.id,
                call_id=task.call_id,
                task_id=task.id,
                approval_id=approval.id,
            ),
        ),
    )


async def test_migration_six_normalizes_approvals_and_event_correlation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteOperationStore.open(path)
    await store.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (6,)
        approval_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(approvals)")
        )
        assert approval_columns == (
            "operation_id",
            "position",
            "id",
            "task_id",
            "task_fingerprint",
            "policy_fingerprint",
            "requested_at",
            "status",
            "decided_at",
            "decided_by",
            "decision_reason",
        )
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(runtime_events)")
        }
        assert "approval_id" in event_columns
        foreign_keys = {
            (row[2], row[3], row[4])
            for row in connection.execute("PRAGMA foreign_key_list(approvals)")
        }
        assert ("operations", "operation_id", "id") in foreign_keys
        assert ("tasks", "operation_id", "operation_id") in foreign_keys
        assert ("tasks", "task_id", "id") in foreign_keys
    finally:
        connection.close()


async def test_migration_six_fails_legacy_waiting_task_closed_without_approval(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    waiting = _waiting_snapshot()
    legacy_seed = replace(
        waiting,
        operation=replace(waiting.operation, status=OperationStatus.RUNNING),
        loop_state=replace(
            waiting.loop_state,
            phase=LoopPhase.AWAITING_EXECUTION,
            waiting_approval_id=None,
        ),
        tasks=(replace(waiting.tasks[0], status=TaskStatus.PENDING),),
        approvals=(),
        events=(),
    )
    version_five = await sqlite_owner._open_with_migrations(
        path,
        migrations=sqlite_owner._MIGRATIONS[:5],
        verify_owned_schema=True,
    )
    try:
        connection = version_five._connection
        connection.execute("BEGIN IMMEDIATE")
        sqlite_owner._insert_snapshot(connection, legacy_seed, revision=1)
        connection.execute(
            "UPDATE operations SET status = 'waiting_for_approval' WHERE id = ?",
            (waiting.operation.id,),
        )
        connection.execute(
            "UPDATE loop_state SET phase = 'awaiting_approval', "
            "waiting_approval_id = ? WHERE operation_id = ?",
            (waiting.approvals[0].id, waiting.operation.id),
        )
        connection.execute(
            "UPDATE tasks SET status = 'waiting_for_approval' WHERE id = ?",
            (waiting.tasks[0].id,),
        )
        connection.execute("COMMIT")
    finally:
        await version_five.close()

    upgraded = await SQLiteOperationStore.open(path)
    try:
        loaded = await upgraded.load(waiting.operation.id)
        snapshot = loaded.snapshot
        assert loaded.revision == 2
        assert snapshot.operation.status is OperationStatus.RUNNING
        assert snapshot.loop_state.phase is LoopPhase.AWAITING_EXECUTION
        assert snapshot.loop_state.waiting_approval_id is None
        assert snapshot.approvals == ()
        assert snapshot.tasks[0].status is TaskStatus.MANUAL_RECOVERY_REQUIRED
        assert (
            snapshot.tasks[0].manual_recovery_reason
            == "legacy_waiting_task_missing_approval"
        )
        assert tuple(event.type for event in snapshot.events) == (
            "task.manual_recovery_required",
        )
        assert snapshot.events[0].approval_id is None
        assert dict(snapshot.events[0].payload) == {
            "from_status": "waiting_for_approval",
            "reason": "legacy_waiting_task_missing_approval",
            "to_status": "manual_recovery_required",
        }
        assert await upgraded.load_by_approval(waiting.approvals[0].id) is None
    finally:
        await upgraded.close()


async def test_pending_approval_and_event_round_trip_and_load_by_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _waiting_snapshot()
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        assert created.operation.snapshot == snapshot
        assert await store.load(snapshot.operation.id) == created.operation
        assert (
            await store.load_by_approval(snapshot.approvals[0].id) == created.operation
        )
        assert await store.load_by_approval("approval-missing") is None

        committed_events = await store.read_after(
            snapshot.operation.agent_id,
            None,
            limit=10,
        )
        requested = next(
            item.event
            for item in committed_events
            if item.event.type == "approval.requested"
        )
        assert requested.approval_id == snapshot.approvals[0].id
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert (
            await reopened.load_by_approval(snapshot.approvals[0].id)
            == created.operation
        )
    finally:
        await reopened.close()


async def test_decision_update_is_exact_and_atomic_with_its_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _waiting_snapshot()
    decided = _decided_snapshot(snapshot)
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        committed = await store.commit(
            decided, expected_revision=created.operation.revision
        )
        assert committed.operation.revision == 2
        assert committed.operation.snapshot == decided
        assert committed.committed_events == (decided.events[-1],)
        assert (
            await store.load_by_approval(snapshot.approvals[0].id)
            == committed.operation
        )
    finally:
        await store.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT status, decided_at, decided_by, decision_reason "
            "FROM approvals WHERE id = ?",
            (snapshot.approvals[0].id,),
        ).fetchone() == (
            "approved",
            "2026-07-18T12:30:05.000000Z",
            "user:test-approver",
            "The exact durable marker change was reviewed.",
        )
        assert connection.execute(
            "SELECT approval_id FROM runtime_events WHERE id = ?",
            (decided.events[-1].id,),
        ).fetchone() == (snapshot.approvals[0].id,)
    finally:
        connection.close()


async def test_duplicate_approval_identity_across_operations_is_typed_and_atomic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    first = _waiting_snapshot()
    second = _waiting_snapshot(
        operation_id="operation-other",
        trigger_id="trigger-other",
    )
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(first)
        with pytest.raises(
            InvalidOperationCheckpointError,
            match="approval identity is already claimed",
        ):
            await store.create(second)
        with pytest.raises(OperationNotFoundError):
            await store.load(second.operation.id)
        assert await store.load_by_approval(first.approvals[0].id) == await store.load(
            first.operation.id
        )
    finally:
        await store.close()


async def test_failed_event_insert_rolls_back_approval_decision(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    snapshot = _waiting_snapshot()
    decided = _decided_snapshot(snapshot)
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        store._connection.execute("""
            CREATE TRIGGER abort_approval_decision_event
            BEFORE INSERT ON runtime_events
            WHEN NEW.type = 'approval.approved'
            BEGIN
                SELECT RAISE(ABORT, 'injected approval event failure');
            END
            """)
        with pytest.raises(sqlite3.IntegrityError, match="injected approval"):
            await store.commit(decided, expected_revision=created.operation.revision)

        loaded = await store.load(snapshot.operation.id)
        assert loaded == created.operation
        assert loaded.snapshot.approvals[0].status is ApprovalStatus.PENDING
    finally:
        await store.close()
