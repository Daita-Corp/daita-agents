from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.events.models import RuntimeEvent
from daita.loop.models import LoopBudgets, LoopPhase, LoopState
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.store import InMemoryOperationStore, VersionedOperation
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 17, 16, 0, tzinfo=timezone.utc)


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
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.TERMINAL if terminal else LoopPhase.PREPARING_CONTEXT
        ),
        budgets=LoopBudgets(),
        turns=(),
        model_calls=(),
        readiness=(),
        tasks=(),
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


async def _open_store(kind: str, path: Path):  # type: ignore[no-untyped-def]
    if kind == "memory":
        return InMemoryOperationStore()
    return await SQLiteOperationStore.open(path)


@pytest.mark.parametrize("kind", ("memory", "sqlite"))
async def test_nonterminal_query_is_complete_agent_scoped_and_deterministic(
    tmp_path: Path,
    kind: str,
) -> None:
    store = await _open_store(kind, tmp_path / f"{kind}.db")
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
        advanced = _advance(running, updated_at=NOW + timedelta(seconds=2))
        revised = await store.commit(advanced, expected_revision=1)
        committed[running.operation.id] = revised.operation

        found = await store.load_nonterminal("agent-target")

        expected_ids = (
            "operation-a-approval",
            "operation-b-running",
            "operation-c-input",
            "operation-d-pending",
        )
        assert found == tuple(committed[operation_id] for operation_id in expected_ids)
        assert tuple(item.revision for item in found) == (1, 2, 1, 1)
        assert {item.snapshot.operation.status for item in found} == {
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
            OperationStatus.WAITING_FOR_APPROVAL,
            OperationStatus.WAITING_FOR_INPUT,
        }
        assert await store.load_nonterminal("agent-other") == (
            committed["operation-other-agent"],
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
