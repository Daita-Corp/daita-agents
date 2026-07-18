from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from daita.loop.models import LoopPhase
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import AgentTrigger, TriggerKind
from daita.operations.runtime import OperationRuntime, OperationStateError
from daita.operations.store import (
    CommitResult,
    InMemoryOperationStore,
    OperationRevisionConflict,
)

NOW = datetime(2026, 7, 17, 16, 0, tzinfo=timezone.utc)


def _trigger(trigger_id: str = "trigger-1") -> AgentTrigger:
    return AgentTrigger(
        id=trigger_id,
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        session_id="session-1",
        payload={"message": "hello"},
        created_at=NOW,
    )


class RecordingStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__()
        self.created: list[OperationSnapshot] = []
        self.commits: list[tuple[OperationSnapshot, int]] = []
        self.commit_error: Exception | None = None

    async def create(self, snapshot: OperationSnapshot) -> CommitResult:
        self.created.append(snapshot)
        return await super().create(snapshot)

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        self.commits.append((snapshot, expected_revision))
        if self.commit_error is not None:
            raise self.commit_error
        return await super().commit(
            snapshot,
            expected_revision=expected_revision,
        )


async def test_runtime_begin_commits_one_complete_checkpoint_through_store() -> None:
    store = RecordingStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)

    snapshot = await runtime.begin(_trigger())

    assert store.created == [snapshot]
    assert snapshot.loop_state.phase is LoopPhase.PREPARING_CONTEXT
    assert snapshot.turns == ()
    assert snapshot.model_calls == ()
    assert snapshot.tasks == ()
    assert [event.type for event in snapshot.events] == [
        "trigger.received",
        "operation.created",
    ]
    assert all(event.session_id == "session-1" for event in snapshot.events)
    committed = await store.load(snapshot.operation.id)
    assert committed.revision == 1
    assert committed.snapshot == snapshot


async def test_runtime_transition_uses_loaded_revision_for_one_cas_commit() -> None:
    store = RecordingStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)
    started = await runtime.begin(_trigger())

    turn = await runtime.begin_turn(started.operation.id)

    assert len(store.commits) == 1
    candidate, expected_revision = store.commits[0]
    assert expected_revision == 1
    assert candidate.turns == (turn,)
    assert candidate.events[-1].type == "turn.created"
    assert (await store.load(started.operation.id)).revision == 2


async def test_failed_store_commit_leaves_runtime_inspection_unchanged() -> None:
    store = RecordingStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)
    started = await runtime.begin(_trigger())
    before = await runtime.inspect(started.operation.id)
    store.commit_error = RuntimeError("injected operation commit failure")

    with pytest.raises(RuntimeError, match="injected operation commit failure"):
        await runtime.begin_turn(started.operation.id)

    after = await runtime.inspect(started.operation.id)
    assert after == before
    assert all(event.type != "turn.created" for event in after.events)


async def test_runtime_surfaces_optimistic_conflict_without_retrying() -> None:
    store = RecordingStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)
    started = await runtime.begin(_trigger())
    store.commit_error = OperationRevisionConflict(
        started.operation.id,
        expected_revision=1,
        actual_revision=2,
    )

    with pytest.raises(OperationStateError) as error:
        await runtime.begin_turn(started.operation.id)

    assert isinstance(error.value.__cause__, OperationRevisionConflict)
    assert len(store.commits) == 1
    assert (await store.load(started.operation.id)).revision == 1


async def test_shared_store_is_authoritative_across_runtime_instances() -> None:
    store = InMemoryOperationStore()
    writer = OperationRuntime(store=store, clock=lambda: NOW)
    reader = OperationRuntime(store=store, clock=lambda: NOW)
    started = await writer.begin(_trigger())

    turn = await writer.begin_turn(started.operation.id)
    observed = await reader.inspect(started.operation.id)

    assert observed.turns == (turn,)
    assert observed.events[-1].type == "turn.created"


async def test_two_runtimes_redeliver_the_same_exact_trigger_operation() -> None:
    store = InMemoryOperationStore()
    runtimes = (
        OperationRuntime(store=store, clock=lambda: NOW),
        OperationRuntime(store=store, clock=lambda: NOW),
    )

    outcomes = await asyncio.gather(
        *(runtime.begin(_trigger()) for runtime in runtimes),
        return_exceptions=True,
    )

    successes = [item for item in outcomes if isinstance(item, OperationSnapshot)]
    assert len(successes) == 2
    assert successes[0].operation.id == successes[1].operation.id
    assert await store.load_by_trigger("trigger-1") is not None


async def test_two_runtimes_cannot_claim_conflicting_trigger_inputs() -> None:
    store = InMemoryOperationStore()
    runtimes = (
        OperationRuntime(store=store, clock=lambda: NOW),
        OperationRuntime(store=store, clock=lambda: NOW),
    )
    conflicting = AgentTrigger(
        id="trigger-1",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        session_id="session-1",
        payload={"message": "different"},
        created_at=NOW,
    )

    outcomes = await asyncio.gather(
        runtimes[0].begin(_trigger()),
        runtimes[1].begin(conflicting),
        return_exceptions=True,
    )

    successes = [item for item in outcomes if isinstance(item, OperationSnapshot)]
    conflicts = [item for item in outcomes if isinstance(item, OperationStateError)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert await store.load_by_trigger("trigger-1") is not None
