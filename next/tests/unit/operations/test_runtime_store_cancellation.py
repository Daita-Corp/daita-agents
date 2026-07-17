from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from daita.events.models import RuntimeEvent
from daita.loop.models import LoopExitKind, LoopPhase
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    OperationStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime
from daita.operations.store import (
    CommitResult,
    InMemoryOperationStore,
)

NOW = datetime(2026, 7, 17, 17, 0, tzinfo=timezone.utc)


def _trigger(trigger_id: str) -> AgentTrigger:
    return AgentTrigger(
        id=trigger_id,
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        session_id="session-1",
        payload={"message": "hello"},
        created_at=NOW,
    )


class CreatedThenBlockedStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__()
        self.created = asyncio.Event()
        self.release = asyncio.Event()
        self.interruption_committed = asyncio.Event()
        self.release_interruption = asyncio.Event()

    async def create(self, snapshot: OperationSnapshot) -> CommitResult:
        result = await super().create(snapshot)
        self.created.set()
        await self.release.wait()
        return result

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        result = await super().commit(
            snapshot,
            expected_revision=expected_revision,
        )
        if snapshot.operation.status is OperationStatus.INTERRUPTED:
            self.interruption_committed.set()
            await self.release_interruption.wait()
        return result


async def test_cancellation_after_durable_create_interrupts_the_new_operation() -> None:
    store = CreatedThenBlockedStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)
    running = asyncio.create_task(runtime.begin(_trigger("cancel-create")))

    await store.created.wait()
    running.cancel()
    running.cancel()
    store.release.set()
    try:
        await store.interruption_committed.wait()
        running.cancel()
        store.release_interruption.set()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        store.release_interruption.set()

    committed = await store.load_by_trigger("cancel-create")
    assert committed is not None
    snapshot = committed.snapshot
    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.operation.terminal_reason == "run_cancelled"
    assert snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert snapshot.loop_state.interruption_reason == "run_cancelled"
    assert [event.type for event in snapshot.events][-1] == "operation.interrupted"


class CommittedTurnThenBlockedStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__()
        self.turn_committed = asyncio.Event()
        self.release_turn = asyncio.Event()

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        result = await super().commit(
            snapshot,
            expected_revision=expected_revision,
        )
        if snapshot.events[-1].type == "turn.created":
            self.turn_committed.set()
            await self.release_turn.wait()
        return result


async def test_cancelled_transition_waits_for_its_durable_commit_to_finish() -> None:
    store = CommittedTurnThenBlockedStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)
    started = await runtime.begin(_trigger("cancel-transition"))
    transition = asyncio.create_task(runtime.begin_turn(started.operation.id))

    await store.turn_committed.wait()
    transition.cancel()
    transition.cancel()
    store.release_turn.set()

    with pytest.raises(asyncio.CancelledError):
        await transition

    committed = await store.load(started.operation.id)
    assert committed.revision == 2
    assert len(committed.snapshot.turns) == 1
    assert committed.snapshot.events[-1].type == "turn.created"

    await runtime.interrupt(started.operation.id)
    terminal = await store.load(started.operation.id)
    assert terminal.snapshot.operation.status is OperationStatus.INTERRUPTED
    assert terminal.snapshot.events[-1].type == "operation.interrupted"


class BlockedInterruptCommitStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__()
        self.block_next_commit = False
        self.commit_blocked = asyncio.Event()
        self.release_commit = asyncio.Event()

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        if self.block_next_commit:
            self.block_next_commit = False
            self.commit_blocked.set()
            await self.release_commit.wait()
        return await super().commit(
            snapshot,
            expected_revision=expected_revision,
        )

    async def commit_external(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        return await InMemoryOperationStore.commit(
            self,
            snapshot,
            expected_revision=expected_revision,
        )


async def test_interruption_reloads_and_converges_after_one_lost_revision() -> None:
    store = BlockedInterruptCommitStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)
    started = await runtime.begin(_trigger("interrupt-cas"))
    external_event = RuntimeEvent(
        id="event-external",
        type="checkpoint.external",
        agent_id=started.operation.agent_id,
        operation_id=started.operation.id,
        session_id=started.operation.session_id,
        payload={},
        created_at=NOW,
    )
    external = replace(started, events=(*started.events, external_event))

    store.block_next_commit = True
    interrupting = asyncio.create_task(runtime.interrupt(started.operation.id))
    await store.commit_blocked.wait()
    external_result = await store.commit_external(
        external,
        expected_revision=1,
    )
    store.release_commit.set()

    result = await interrupting

    assert result.kind is LoopExitKind.INTERRUPTED
    committed = await store.load(started.operation.id)
    assert committed.revision == external_result.operation.revision + 1
    assert committed.snapshot.events[: len(external.events)] == external.events
    assert committed.snapshot.operation.status is OperationStatus.INTERRUPTED
    assert committed.snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert committed.snapshot.events[-1].type == "operation.interrupted"
