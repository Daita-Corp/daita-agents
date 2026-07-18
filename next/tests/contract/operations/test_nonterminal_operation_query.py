from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone, tzinfo
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
from daita.operations.governance import ApprovalRequest
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    Task,
    TaskStatus,
    TriggerKind,
)
from daita.operations.store import InMemoryOperationStore, VersionedOperation
from daita.storage.sqlite import SQLiteCorruptionError, SQLiteOperationStore

NOW = datetime(2026, 7, 17, 16, 0, tzinfo=timezone.utc)


class FoldOffset(tzinfo):
    """Fold-sensitive zone without relying on an external IANA database."""

    def utcoffset(self, value: datetime | None) -> timedelta:
        return timedelta(hours=-5 - (0 if value is None else value.fold))

    def dst(self, value: datetime | None) -> timedelta:
        del value
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        del value
        return "fold-offset"


def _snapshot(
    operation_id: str,
    *,
    agent_id: str,
    status: OperationStatus,
    updated_at: datetime,
) -> OperationSnapshot:
    trigger = AgentTrigger(
        id=f"trigger-{operation_id}",
        agent_id=agent_id,
        kind=TriggerKind.USER,
        source_id=f"source-{operation_id}",
        payload={"operation_id": operation_id},
        created_at=NOW,
    )
    terminal = status in {
        OperationStatus.SUCCEEDED,
        OperationStatus.FAILED,
        OperationStatus.CANCELLED,
        OperationStatus.INTERRUPTED,
    }
    operation = Operation(
        id=operation_id,
        agent_id=agent_id,
        trigger_id=trigger.id,
        status=status,
        created_at=NOW,
        updated_at=updated_at,
        final_text="done" if status is OperationStatus.SUCCEEDED else None,
        terminal_reason=f"terminal.{status.value}" if terminal else None,
    )
    event = RuntimeEvent(
        id=f"event-{operation_id}-created",
        type="operation.created",
        agent_id=agent_id,
        operation_id=operation_id,
        payload={"status": status.value},
        created_at=NOW,
    )
    approval_wait = status is OperationStatus.WAITING_FOR_APPROVAL
    turns: tuple[Turn, ...] = ()
    model_calls: tuple[ModelCall, ...] = ()
    tasks: tuple[Task, ...] = ()
    approvals: tuple[ApprovalRequest, ...] = ()
    waiting_approval_id: str | None = None
    if approval_wait:
        turn_id = f"turn-{operation_id}"
        model_call_id = f"model-{operation_id}"
        call_id = f"call-{operation_id}"
        task_id = f"task-{operation_id}"
        waiting_approval_id = f"approval-{operation_id}"
        request = ModelRequest(
            operation_id=operation_id,
            turn_id=turn_id,
            messages=(
                CanonicalMessage(
                    agent_id=agent_id,
                    operation_id=operation_id,
                    turn_id=turn_id,
                    role=MessageRole.USER,
                    content=(TextBlock("Change the test marker."),),
                ),
            ),
            tools=(
                ToolDefinition(
                    name="fake_write",
                    description="Change one test marker.",
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
        model_calls = (
            ModelCall(
                id=model_call_id,
                operation_id=operation_id,
                turn_id=turn_id,
                provider_id="mock:approval-query",
                request=request,
                response=response,
                status=ModelCallStatus.COMPLETED,
                created_at=NOW,
                updated_at=NOW,
            ),
        )
        turns = (
            Turn(
                id=turn_id,
                operation_id=operation_id,
                number=1,
                model_request_id=model_call_id,
                model_response_id=model_call_id,
                created_at=NOW,
            ),
        )
        tasks = (
            Task(
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
            ),
        )
        approvals = (
            ApprovalRequest(
                id=waiting_approval_id,
                operation_id=operation_id,
                task_id=task_id,
                task_fingerprint="sha256:" + ("a" * 64),
                policy_fingerprint="sha256:" + ("b" * 64),
                requested_at=NOW,
            ),
        )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=(
                LoopPhase.TERMINAL
                if terminal
                else (
                    LoopPhase.AWAITING_APPROVAL
                    if approval_wait
                    else LoopPhase.PREPARING_CONTEXT
                )
            ),
            turn_count=1 if approval_wait else 0,
            action_count=1 if approval_wait else 0,
            waiting_approval_id=waiting_approval_id,
        ),
        budgets=LoopBudgets(),
        turns=turns,
        model_calls=model_calls,
        readiness=(),
        tasks=tasks,
        approvals=approvals,
        evidence=(),
        observations=(),
        events=(event,),
    )


def _advance(
    snapshot: OperationSnapshot,
    *,
    updated_at: datetime,
    status: OperationStatus | None = None,
) -> OperationSnapshot:
    next_status = snapshot.operation.status if status is None else status
    terminal = next_status in {
        OperationStatus.SUCCEEDED,
        OperationStatus.FAILED,
        OperationStatus.CANCELLED,
        OperationStatus.INTERRUPTED,
    }
    event = RuntimeEvent(
        id=f"event-{snapshot.operation.id}-{len(snapshot.events) + 1}",
        type="operation.updated",
        agent_id=snapshot.operation.agent_id,
        operation_id=snapshot.operation.id,
        payload={"status": next_status.value},
        created_at=updated_at,
    )
    return replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            status=next_status,
            updated_at=updated_at,
            final_text="done" if next_status is OperationStatus.SUCCEEDED else None,
            terminal_reason=f"terminal.{next_status.value}" if terminal else None,
        ),
        loop_state=replace(
            snapshot.loop_state,
            phase=LoopPhase.TERMINAL if terminal else snapshot.loop_state.phase,
        ),
        events=(*snapshot.events, event),
    )


async def _open_store(
    kind: str,
    path: Path,
) -> InMemoryOperationStore | SQLiteOperationStore:
    if kind == "memory":
        return InMemoryOperationStore()
    return await SQLiteOperationStore.open(path)


@pytest.mark.parametrize("kind", ("memory", "sqlite"))
async def test_nonterminal_query_is_complete_agent_scoped_and_deterministic(
    tmp_path: Path,
    kind: str,
) -> None:
    store = await _open_store(kind, tmp_path / f"{kind}.db")
    fold_offset = FoldOffset()
    earlier_fold = datetime(2026, 11, 1, 1, 30, tzinfo=fold_offset, fold=0)
    later_fold = datetime(2026, 11, 1, 1, 30, tzinfo=fold_offset, fold=1)
    snapshots = (
        _snapshot(
            "operation-d-pending",
            agent_id="agent-target",
            status=OperationStatus.PENDING,
            updated_at=NOW + timedelta(seconds=4),
        ),
        _snapshot(
            "operation-b-running",
            agent_id="agent-target",
            status=OperationStatus.RUNNING,
            updated_at=NOW + timedelta(seconds=1),
        ),
        _snapshot(
            "operation-a-approval",
            agent_id="agent-target",
            status=OperationStatus.WAITING_FOR_APPROVAL,
            updated_at=NOW + timedelta(seconds=1),
        ),
        _snapshot(
            "operation-c-input",
            agent_id="agent-target",
            status=OperationStatus.WAITING_FOR_INPUT,
            updated_at=NOW + timedelta(seconds=3),
        ),
        _snapshot(
            "operation-other-agent",
            agent_id="agent-other",
            status=OperationStatus.RUNNING,
            updated_at=NOW,
        ),
        _snapshot(
            "operation-a-fold-later",
            agent_id="agent-target",
            status=OperationStatus.RUNNING,
            updated_at=later_fold,
        ),
        _snapshot(
            "operation-z-fold-earlier",
            agent_id="agent-target",
            status=OperationStatus.RUNNING,
            updated_at=earlier_fold,
        ),
        *(
            _snapshot(
                f"operation-terminal-{status.value}",
                agent_id="agent-target",
                status=status,
                updated_at=NOW + timedelta(seconds=2),
            )
            for status in (
                OperationStatus.SUCCEEDED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
                OperationStatus.INTERRUPTED,
            )
        ),
    )
    try:
        committed: dict[str, VersionedOperation] = {}
        for snapshot in snapshots:
            result = await store.create(snapshot)
            committed[snapshot.operation.id] = result.operation

        running = snapshots[1]
        advanced = _advance(running, updated_at=NOW + timedelta(seconds=1))
        revised = await store.commit(advanced, expected_revision=1)
        committed[running.operation.id] = revised.operation

        authoritative = {
            snapshot.operation.id: await store.load(snapshot.operation.id)
            for snapshot in snapshots
        }

        found = await store.load_nonterminal("agent-target")

        expected_ids = (
            "operation-a-approval",
            "operation-b-running",
            "operation-c-input",
            "operation-d-pending",
            "operation-z-fold-earlier",
            "operation-a-fold-later",
        )
        assert found == tuple(
            authoritative[operation_id] for operation_id in expected_ids
        )
        assert tuple(item.revision for item in found) == (1, 2, 1, 1, 1, 1)
        assert {item.snapshot.operation.status for item in found} == {
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
            OperationStatus.WAITING_FOR_APPROVAL,
            OperationStatus.WAITING_FOR_INPUT,
        }
        assert await store.load_nonterminal("agent-other") == (
            authoritative["operation-other-agent"],
        )
        assert await store.load_nonterminal("agent-missing") == ()
        with pytest.raises(ValueError, match="agent_id"):
            await store.load_nonterminal("   ")
    finally:
        if isinstance(store, SQLiteOperationStore):
            await store.close()


async def test_sqlite_query_is_one_snapshot_across_instances_and_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recovery-query.db"
    writer = await SQLiteOperationStore.open(path)
    reader = await SQLiteOperationStore.open(path)
    first = _snapshot(
        "operation-first",
        agent_id="agent-target",
        status=OperationStatus.RUNNING,
        updated_at=NOW,
    )
    second = _snapshot(
        "operation-second",
        agent_id="agent-target",
        status=OperationStatus.WAITING_FOR_INPUT,
        updated_at=NOW + timedelta(seconds=1),
    )
    traced: list[str] = []
    try:
        first_created = await writer.create(first)
        assert await reader.load_nonterminal("agent-target") == (
            first_created.operation,
        )

        first_terminal = _advance(
            first,
            updated_at=NOW + timedelta(seconds=2),
            status=OperationStatus.SUCCEEDED,
        )
        await writer.commit(first_terminal, expected_revision=1)
        second_created = await writer.create(second)

        reader._connection.set_trace_callback(traced.append)
        found = await reader.load_nonterminal("agent-target")
        reader._connection.set_trace_callback(None)

        assert found == (second_created.operation,)
        normalized = tuple(statement.strip().upper() for statement in traced)
        assert normalized[0] == "BEGIN"
        assert normalized[-1] == "COMMIT"
    finally:
        reader._connection.set_trace_callback(None)
        await reader.close()
        await writer.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert await reopened.load_nonterminal("agent-target") == (
            second_created.operation,
        )
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    "corrupt_status",
    ("future_paused", sqlite3.Binary(b"future_paused")),
    ids=("unknown-text", "non-text"),
)
async def test_sqlite_query_fails_closed_on_unknown_agent_operation_status(
    tmp_path: Path,
    corrupt_status: object,
) -> None:
    path = tmp_path / "unknown-status.db"
    snapshot = _snapshot(
        "operation-unknown-status",
        agent_id="agent-target",
        status=OperationStatus.RUNNING,
        updated_at=NOW,
    )
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(snapshot)
    finally:
        await store.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE operations SET status = ? WHERE id = ?",
            (corrupt_status, snapshot.operation.id),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        with pytest.raises(
            SQLiteCorruptionError,
            match="cannot classify agent-scoped operation status",
        ):
            await reopened.load_nonterminal("agent-target")
        assert not reopened._connection.in_transaction
    finally:
        await reopened.close()


async def test_sqlite_query_holds_one_snapshot_during_mid_decode_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "consistent-query.db"
    writer = await SQLiteOperationStore.open(path)
    reader = await SQLiteOperationStore.open(path)
    snapshot = _snapshot(
        "operation-consistent-read",
        agent_id="agent-target",
        status=OperationStatus.RUNNING,
        updated_at=NOW,
    )
    fired = False
    callback_errors: list[BaseException] = []
    try:
        created = await writer.create(snapshot)

        def commit_during_decode(statement: str) -> None:
            nonlocal fired
            if fired or not statement.startswith("SELECT * FROM operations WHERE id ="):
                return
            fired = True
            try:
                writer._connection.execute("BEGIN IMMEDIATE")
                writer._connection.execute(
                    "UPDATE operations SET revision = 2 WHERE id = ?",
                    (snapshot.operation.id,),
                )
                writer._connection.execute("COMMIT")
            except BaseException as error:
                callback_errors.append(error)
                if writer._connection.in_transaction:
                    writer._connection.execute("ROLLBACK")

        reader._connection.set_trace_callback(commit_during_decode)
        batch = await reader.load_nonterminal("agent-target")
        reader._connection.set_trace_callback(None)
        latest = await reader.load(snapshot.operation.id)

        assert fired
        assert callback_errors == []
        assert batch == (created.operation,)
        assert batch[0].revision == 1
        assert latest.revision == 2
        assert latest.snapshot == batch[0].snapshot
    finally:
        reader._connection.set_trace_callback(None)
        if writer._connection.in_transaction:
            writer._connection.execute("ROLLBACK")
        await reader.close()
        await writer.close()
