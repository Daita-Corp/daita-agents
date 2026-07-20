from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Readiness
from daita.events.models import RuntimeEvent
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.store import (
    CommitResult,
    InMemoryOperationStore,
    InvalidOperationCheckpointError,
    OperationAlreadyExistsError,
    OperationNotFoundError,
    OperationRevisionConflict,
    OperationStoreError,
    TriggerAlreadyClaimedError,
    VersionedOperation,
)

NOW = datetime(2026, 7, 17, 14, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(seconds=1)


def _event(
    event_id: str,
    event_type: str,
    *,
    operation_id: str = "operation-1",
    agent_id: str = "agent-1",
    created_at: datetime = NOW,
) -> RuntimeEvent:
    return RuntimeEvent(
        id=event_id,
        type=event_type,
        agent_id=agent_id,
        operation_id=operation_id,
        payload={"event_id": event_id},
        created_at=created_at,
    )


def _initial_snapshot(
    *,
    operation_id: str = "operation-1",
    trigger_id: str = "trigger-1",
    agent_id: str = "agent-1",
) -> OperationSnapshot:
    trigger = AgentTrigger(
        id=trigger_id,
        agent_id=agent_id,
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"message": "hello"},
        created_at=NOW,
    )
    operation = Operation(
        id=operation_id,
        agent_id=agent_id,
        trigger_id=trigger_id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
    )
    events = (
        _event(
            f"{operation_id}-event-1",
            "trigger.received",
            operation_id=operation_id,
            agent_id=agent_id,
        ),
        _event(
            f"{operation_id}-event-2",
            "operation.created",
            operation_id=operation_id,
            agent_id=agent_id,
        ),
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(phase=LoopPhase.PREPARING_CONTEXT),
        budgets=LoopBudgets(),
        turns=(),
        model_calls=(),
        readiness=(),
        tasks=(),
        evidence=(),
        observations=(),
        events=events,
    )


def _advanced_snapshot(
    snapshot: OperationSnapshot,
    *events: RuntimeEvent,
) -> OperationSnapshot:
    return replace(
        snapshot,
        operation=replace(snapshot.operation, updated_at=LATER),
        events=(*snapshot.events, *events),
    )


async def test_create_commits_the_initial_checkpoint_and_both_event_records() -> None:
    store = InMemoryOperationStore()
    snapshot = _initial_snapshot()

    result = await store.create(snapshot)

    assert isinstance(result, CommitResult)
    assert isinstance(result.operation, VersionedOperation)
    assert result.operation.revision == 1
    assert result.operation.snapshot == snapshot
    assert result.committed_events == snapshot.events
    assert await store.load(snapshot.operation.id) == result.operation
    assert await store.load_by_trigger(snapshot.trigger.id) == result.operation


async def test_list_operations_is_agent_scoped_filtered_bounded_and_newest_first() -> (
    None
):
    store = InMemoryOperationStore()
    first = _initial_snapshot(operation_id="operation-1", trigger_id="trigger-1")
    second = _initial_snapshot(operation_id="operation-2", trigger_id="trigger-2")
    other = _initial_snapshot(
        operation_id="operation-other",
        trigger_id="trigger-other",
        agent_id="agent-other",
    )
    await store.create(first)
    await store.create(second)
    await store.create(other)
    succeeded = replace(
        first,
        operation=replace(
            first.operation,
            status=OperationStatus.SUCCEEDED,
            updated_at=LATER,
            final_text="done",
            terminal_reason="completed",
        ),
        loop_state=replace(first.loop_state, phase=LoopPhase.TERMINAL),
        events=(
            *first.events,
            _event(
                "operation-1-event-3",
                "operation.succeeded",
                operation_id="operation-1",
                created_at=LATER,
            ),
        ),
    )
    await store.commit(succeeded, expected_revision=1)

    assert tuple(
        value.snapshot.operation.id for value in await store.list_operations("agent-1")
    ) == ("operation-1", "operation-2")
    assert tuple(
        value.snapshot.operation.id
        for value in await store.list_operations(
            "agent-1",
            statuses=(OperationStatus.RUNNING,),
            limit=1,
        )
    ) == ("operation-2",)
    assert await store.list_operations("agent-missing") == ()

    with pytest.raises(ValueError, match="statuses"):
        await store.list_operations("agent-1", statuses=())
    with pytest.raises(ValueError, match="limit"):
        await store.list_operations("agent-1", limit=1_001)


async def test_missing_operation_is_typed_and_trigger_lookup_is_optional() -> None:
    store = InMemoryOperationStore()

    with pytest.raises(OperationNotFoundError) as operation_error:
        await store.load("operation-missing")

    assert operation_error.value.operation_id == "operation-missing"
    assert issubclass(type(operation_error.value), OperationStoreError)
    assert await store.load_by_trigger("trigger-missing") is None


async def test_operation_and_trigger_identity_are_claimed_atomically() -> None:
    store = InMemoryOperationStore()
    original = _initial_snapshot()
    created = await store.create(original)

    same_operation = _initial_snapshot(
        operation_id=original.operation.id,
        trigger_id="trigger-other",
    )
    with pytest.raises(OperationAlreadyExistsError) as operation_error:
        await store.create(same_operation)

    same_trigger = _initial_snapshot(
        operation_id="operation-other",
        trigger_id=original.trigger.id,
    )
    with pytest.raises(TriggerAlreadyClaimedError) as trigger_error:
        await store.create(same_trigger)

    assert operation_error.value.operation_id == original.operation.id
    assert trigger_error.value.trigger_id == original.trigger.id
    assert trigger_error.value.operation_id == original.operation.id
    assert issubclass(type(operation_error.value), OperationStoreError)
    assert issubclass(type(trigger_error.value), OperationStoreError)
    assert await store.load(original.operation.id) == created.operation
    assert await store.load_by_trigger(original.trigger.id) == created.operation
    with pytest.raises(OperationNotFoundError):
        await store.load("operation-other")
    assert await store.load_by_trigger("trigger-other") is None


async def test_commit_increments_revision_and_returns_only_the_event_suffix() -> None:
    store = InMemoryOperationStore()
    initial = _initial_snapshot()
    created = await store.create(initial)
    suffix = (
        _event("operation-1-event-3", "turn.created", created_at=LATER),
        _event("operation-1-event-4", "context.built", created_at=LATER),
    )
    advanced = _advanced_snapshot(initial, *suffix)

    result = await store.commit(advanced, expected_revision=1)

    assert result.operation.revision == 2
    assert result.operation.snapshot == advanced
    assert result.committed_events == suffix
    assert await store.load(initial.operation.id) == result.operation
    assert await store.load_by_trigger(initial.trigger.id) == result.operation
    assert created.operation.revision == 1
    assert created.operation.snapshot == initial


async def test_revision_conflict_commits_no_state_or_events() -> None:
    store = InMemoryOperationStore()
    initial = _initial_snapshot()
    created = await store.create(initial)
    advanced = _advanced_snapshot(
        initial,
        _event("operation-1-event-3", "turn.created", created_at=LATER),
    )

    with pytest.raises(OperationRevisionConflict) as conflict:
        await store.commit(advanced, expected_revision=0)

    assert conflict.value.operation_id == initial.operation.id
    assert conflict.value.expected_revision == 0
    assert conflict.value.actual_revision == 1
    assert issubclass(type(conflict.value), OperationStoreError)
    assert await store.load(initial.operation.id) == created.operation
    assert await store.load_by_trigger(initial.trigger.id) == created.operation


async def test_two_concurrent_cas_writers_have_exactly_one_winner() -> None:
    store = InMemoryOperationStore()
    initial = _initial_snapshot()
    await store.create(initial)
    candidates = (
        _advanced_snapshot(
            initial,
            _event("operation-1-event-a", "candidate.a", created_at=LATER),
        ),
        _advanced_snapshot(
            initial,
            _event("operation-1-event-b", "candidate.b", created_at=LATER),
        ),
    )

    outcomes = await asyncio.gather(
        *(store.commit(candidate, expected_revision=1) for candidate in candidates),
        return_exceptions=True,
    )

    winners = [outcome for outcome in outcomes if isinstance(outcome, CommitResult)]
    losers = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, OperationRevisionConflict)
    ]
    assert len(winners) == 1
    assert len(losers) == 1
    winner = winners[0]
    loaded = await store.load(initial.operation.id)
    assert loaded == winner.operation
    assert loaded.revision == 2
    assert winner.committed_events == (loaded.snapshot.events[-1],)
    assert loaded.snapshot in candidates


async def test_event_history_must_be_an_exact_append_only_prefix() -> None:
    store = InMemoryOperationStore()
    initial = _initial_snapshot()
    created = await store.create(initial)
    altered_first_event = replace(
        initial.events[0],
        payload={"event_id": "silently-rewritten"},
    )
    invalid_checkpoints = (
        replace(initial, events=initial.events[:-1]),
        replace(initial, events=(altered_first_event, *initial.events[1:])),
    )

    for invalid in invalid_checkpoints:
        with pytest.raises(InvalidOperationCheckpointError) as error:
            await store.commit(invalid, expected_revision=1)
        assert error.value.operation_id == initial.operation.id
        assert error.value.reason
        assert issubclass(type(error.value), OperationStoreError)
        assert await store.load(initial.operation.id) == created.operation


async def test_checkpoint_identity_mismatch_is_rejected_atomically() -> None:
    store = InMemoryOperationStore()
    initial = _initial_snapshot()
    created = await store.create(initial)
    forged_trigger = replace(initial.trigger, id="trigger-forged")
    mismatched = replace(
        initial,
        trigger=forged_trigger,
        operation=replace(
            initial.operation,
            trigger_id=forged_trigger.id,
            updated_at=LATER,
        ),
        events=(
            *initial.events,
            _event("operation-1-event-3", "turn.created", created_at=LATER),
        ),
    )

    with pytest.raises(InvalidOperationCheckpointError) as error:
        await store.commit(mismatched, expected_revision=1)

    assert error.value.operation_id == initial.operation.id
    assert error.value.reason
    assert issubclass(type(error.value), OperationStoreError)
    assert await store.load(initial.operation.id) == created.operation
    assert await store.load_by_trigger(initial.trigger.id) == created.operation


async def test_commit_of_unknown_operation_does_not_claim_its_trigger() -> None:
    store = InMemoryOperationStore()
    unknown = _initial_snapshot()

    with pytest.raises(OperationNotFoundError) as error:
        await store.commit(unknown, expected_revision=1)

    assert issubclass(type(error.value), OperationStoreError)
    with pytest.raises(OperationNotFoundError):
        await store.load(unknown.operation.id)
    assert await store.load_by_trigger(unknown.trigger.id) is None


async def test_committed_lifecycle_history_cannot_be_rewritten_or_removed() -> None:
    readiness = Readiness(
        allowed=False,
        code="evidence_incomplete",
        message="More evidence is required.",
        missing_facts=("customer count",),
        evaluated_at=LATER,
    )

    for candidate_history in (
        (replace(readiness, message="silently rewritten"),),
        (),
    ):
        store = InMemoryOperationStore()
        initial = _initial_snapshot()
        await store.create(initial)
        with_history = _advanced_snapshot(
            replace(initial, readiness=(readiness,)),
            _event(
                "operation-1-event-readiness",
                "readiness.recorded",
                created_at=LATER,
            ),
        )
        committed = await store.commit(with_history, expected_revision=1)
        before = await store.load(initial.operation.id)
        next_time = LATER + timedelta(seconds=1)
        candidate = replace(
            with_history,
            operation=replace(with_history.operation, updated_at=next_time),
            readiness=candidate_history,
            events=(
                *with_history.events,
                _event(
                    f"operation-1-event-attempt-{len(candidate_history)}",
                    "checkpoint.updated",
                    created_at=next_time,
                ),
            ),
        )

        with pytest.raises(InvalidOperationCheckpointError):
            await store.commit(
                candidate, expected_revision=committed.operation.revision
            )

        after = await store.load(initial.operation.id)
        assert after == before == committed.operation
        assert after.snapshot.events == with_history.events
