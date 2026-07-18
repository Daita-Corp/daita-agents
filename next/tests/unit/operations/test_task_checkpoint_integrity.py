from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import cast

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
from daita.operations.leases import TaskLease
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    Task,
    TaskDependency,
    TaskStatus,
    TriggerKind,
)
from daita.operations.store import (
    InMemoryOperationStore,
    InvalidOperationCheckpointError,
    TaskClaimResult,
)

NOW = datetime(2026, 7, 17, 17, 0, tzinfo=timezone.utc)


def _task(task_id: str, call_id: str) -> Task:
    return Task(
        id=task_id,
        operation_id="operation-1",
        turn_id="turn-1",
        call_id=call_id,
        capability_id="fake.read",
        executor_id="fake.executor",
        status=TaskStatus.PENDING,
        attempt=1,
        arguments={"key": task_id},
        created_at=NOW,
        updated_at=NOW,
    )


def _snapshot(
    *,
    task_dependencies: tuple[TaskDependency, ...] = (),
    task_leases: tuple[TaskLease, ...] = (),
) -> OperationSnapshot:
    trigger = AgentTrigger(
        id="trigger-1",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        session_id="session-1",
        payload={"message": "Read three values."},
        created_at=NOW,
    )
    operation = Operation(
        id="operation-1",
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        session_id=trigger.session_id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
    )
    request = ModelRequest(
        operation_id=operation.id,
        turn_id="turn-1",
        messages=(
            CanonicalMessage(
                agent_id=operation.agent_id,
                operation_id=operation.id,
                turn_id="turn-1",
                role=MessageRole.USER,
                content=(TextBlock("Read three values."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="fake.read",
                description="Read a deterministic test value.",
                input_schema={"type": "object"},
            ),
        ),
    )
    response = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(
            ToolCall(id=call_id, name="fake.read", arguments={"key": task_id})
            for task_id, call_id in (
                ("task-a", "call-a"),
                ("task-b", "call-b"),
                ("task-c", "call-c"),
            )
        ),
    )
    model_call = ModelCall(
        id="model-call-1",
        operation_id=operation.id,
        turn_id="turn-1",
        provider_id="mock:scripted",
        request=request,
        response=response,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    turn = Turn(
        id="turn-1",
        operation_id=operation.id,
        number=1,
        model_request_id=model_call.id,
        model_response_id=model_call.id,
        created_at=NOW,
    )
    events = (
        RuntimeEvent(
            id="event-1",
            type="trigger.received",
            agent_id=operation.agent_id,
            operation_id=operation.id,
            session_id=operation.session_id,
            payload={},
            created_at=NOW,
        ),
        RuntimeEvent(
            id="event-2",
            type="operation.created",
            agent_id=operation.agent_id,
            operation_id=operation.id,
            session_id=operation.session_id,
            payload={},
            created_at=NOW,
        ),
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.OBSERVING,
            turn_count=1,
            action_count=3,
        ),
        budgets=LoopBudgets(),
        turns=(turn,),
        model_calls=(model_call,),
        readiness=(),
        tasks=(
            _task("task-a", "call-a"),
            _task("task-b", "call-b"),
            _task("task-c", "call-c"),
        ),
        task_dependencies=task_dependencies,
        task_leases=task_leases,
        evidence=(),
        observations=(),
        events=events,
    )


def _dependency(task_id: str, prerequisite_task_id: str) -> TaskDependency:
    return TaskDependency(
        operation_id="operation-1",
        task_id=task_id,
        prerequisite_task_id=prerequisite_task_id,
    )


def _lease(
    *,
    task_id: str = "task-a",
    operation_id: str = "operation-1",
    attempt: int = 1,
    fencing_token: int = 1,
    holder_id: str = "worker-1",
    acquired_at: datetime = NOW,
    expires_at: datetime = NOW + timedelta(seconds=30),
    started_at: datetime | None = None,
    renewed_at: datetime | None = None,
    released_at: datetime | None = None,
    release_reason: str | None = None,
) -> TaskLease:
    return TaskLease(
        operation_id=operation_id,
        task_id=task_id,
        attempt=attempt,
        fencing_token=fencing_token,
        holder_id=holder_id,
        acquired_at=acquired_at,
        expires_at=expires_at,
        started_at=started_at,
        renewed_at=renewed_at,
        released_at=released_at,
        release_reason=release_reason,
    )


def _released_lease(
    *,
    task_id: str = "task-a",
    attempt: int,
    fencing_token: int,
    offset_seconds: int,
) -> TaskLease:
    acquired_at = NOW + timedelta(seconds=offset_seconds)
    return _lease(
        task_id=task_id,
        attempt=attempt,
        fencing_token=fencing_token,
        acquired_at=acquired_at,
        expires_at=acquired_at + timedelta(seconds=30),
        started_at=acquired_at + timedelta(seconds=1),
        released_at=acquired_at + timedelta(seconds=2),
        release_reason="completed",
    )


def _with_event(snapshot: OperationSnapshot, event_id: str) -> OperationSnapshot:
    event = RuntimeEvent(
        id=event_id,
        type="checkpoint.updated",
        agent_id=snapshot.operation.agent_id,
        operation_id=snapshot.operation.id,
        session_id=snapshot.operation.session_id,
        payload={},
        created_at=NOW + timedelta(minutes=1),
    )
    return replace(snapshot, events=(*snapshot.events, event))


def test_snapshot_normalizes_dependency_and_lease_collections_to_tuples() -> None:
    dependencies = [_dependency("task-a", "task-b")]
    leases = [_lease()]
    base = _snapshot()

    snapshot = replace(
        base,
        task_dependencies=cast(tuple[TaskDependency, ...], dependencies),
        task_leases=cast(tuple[TaskLease, ...], leases),
    )
    dependencies.clear()
    leases.clear()

    assert snapshot.task_dependencies == (_dependency("task-a", "task-b"),)
    assert snapshot.task_leases == (_lease(),)
    assert isinstance(snapshot.task_dependencies, tuple)
    assert isinstance(snapshot.task_leases, tuple)


def test_snapshot_requires_strict_dependency_record_types() -> None:
    base = _snapshot()

    with pytest.raises(TypeError, match="task_dependencies|TaskDependency"):
        replace(
            base,
            task_dependencies=cast(tuple[TaskDependency, ...], (object(),)),
        )


def test_snapshot_requires_strict_lease_record_types() -> None:
    base = _snapshot()

    with pytest.raises(TypeError, match="task_leases|TaskLease"):
        replace(
            base,
            task_leases=cast(tuple[TaskLease, ...], (object(),)),
        )


def test_dependencies_must_bind_tasks_in_the_same_operation() -> None:
    base = _snapshot()

    with pytest.raises(ValueError, match="dependency.*operation|operation.*dependency"):
        replace(
            base,
            task_dependencies=(
                TaskDependency(
                    operation_id="operation-other",
                    task_id="task-a",
                    prerequisite_task_id="task-b",
                ),
            ),
        )
    with pytest.raises(ValueError, match="dependency task|task.*exist"):
        replace(base, task_dependencies=(_dependency("task-missing", "task-b"),))
    with pytest.raises(ValueError, match="prerequisite|task.*exist"):
        replace(base, task_dependencies=(_dependency("task-a", "task-missing"),))


def test_dependencies_reject_duplicate_and_self_edges() -> None:
    base = _snapshot()
    dependency = _dependency("task-a", "task-b")

    with pytest.raises(ValueError, match="duplicate.*dependency|dependency.*duplicate"):
        replace(base, task_dependencies=(dependency, dependency))
    with pytest.raises(ValueError, match="self|itself|cycle"):
        replace(base, task_dependencies=(_dependency("task-a", "task-a"),))


def test_dependencies_reject_cycles_across_the_operation_graph() -> None:
    base = _snapshot()

    with pytest.raises(ValueError, match="cycle|cyclic"):
        replace(
            base,
            task_dependencies=(
                _dependency("task-a", "task-b"),
                _dependency("task-b", "task-c"),
                _dependency("task-c", "task-a"),
            ),
        )


def test_leases_must_bind_existing_tasks_in_the_same_operation() -> None:
    base = _snapshot()

    with pytest.raises(ValueError, match="lease.*operation|operation.*lease"):
        replace(base, task_leases=(_lease(operation_id="operation-other"),))
    with pytest.raises(ValueError, match="lease task|task.*exist"):
        replace(base, task_leases=(_lease(task_id="task-missing"),))


def test_lease_attempt_and_fencing_token_pairs_are_unique_per_task() -> None:
    base = _snapshot()
    first = _released_lease(attempt=1, fencing_token=1, offset_seconds=0)

    with pytest.raises(ValueError, match="fenc|token|duplicate|unique"):
        replace(
            base,
            task_leases=(
                first,
                _released_lease(
                    attempt=2,
                    fencing_token=1,
                    offset_seconds=3,
                ),
            ),
        )
    with pytest.raises(ValueError, match="attempt|duplicate|unique"):
        replace(
            base,
            task_leases=(
                first,
                _released_lease(
                    attempt=1,
                    fencing_token=2,
                    offset_seconds=3,
                ),
            ),
        )


def test_lease_attempts_and_fencing_tokens_strictly_increase_in_history_order() -> None:
    base = _snapshot()

    with pytest.raises(ValueError, match="attempt.*increase|increasing.*attempt"):
        replace(
            base,
            task_leases=(
                _released_lease(
                    attempt=2,
                    fencing_token=1,
                    offset_seconds=0,
                ),
                _released_lease(
                    attempt=1,
                    fencing_token=2,
                    offset_seconds=3,
                ),
            ),
        )
    with pytest.raises(ValueError, match="fenc.*increase|increasing.*fenc"):
        replace(
            base,
            task_leases=(
                _released_lease(
                    attempt=1,
                    fencing_token=2,
                    offset_seconds=0,
                ),
                _released_lease(
                    attempt=2,
                    fencing_token=1,
                    offset_seconds=3,
                ),
            ),
        )


def test_each_task_has_at_most_one_unreleased_lease() -> None:
    base = _snapshot()
    first = _lease(expires_at=NOW + timedelta(seconds=2))
    second = _lease(
        attempt=2,
        fencing_token=2,
        acquired_at=NOW + timedelta(seconds=3),
        expires_at=NOW + timedelta(seconds=33),
    )

    with pytest.raises(ValueError, match="unreleased|active lease|release"):
        replace(base, task_leases=(first, second))


def test_later_lease_attempt_cannot_overlap_the_prior_attempt() -> None:
    base = _snapshot()
    first = _released_lease(attempt=1, fencing_token=1, offset_seconds=0)
    overlapping = _released_lease(
        attempt=2,
        fencing_token=2,
        offset_seconds=1,
    )

    with pytest.raises(ValueError, match="overlap|prior.*release"):
        replace(base, task_leases=(first, overlapping))


def test_task_lease_requires_forward_internal_chronology() -> None:
    with pytest.raises(ValueError, match="expires|acquired|precede|after"):
        _lease(expires_at=NOW)
    with pytest.raises(ValueError, match="started|acquired|precede"):
        _lease(started_at=NOW - timedelta(microseconds=1))
    with pytest.raises(ValueError, match="renewed|acquired|precede"):
        _lease(renewed_at=NOW - timedelta(microseconds=1))
    with pytest.raises(ValueError, match="released|started|renewed|precede"):
        _lease(
            started_at=NOW + timedelta(seconds=2),
            released_at=NOW + timedelta(seconds=1),
            release_reason="completed",
        )
    with pytest.raises(ValueError, match="release_reason|reason|released"):
        _lease(release_reason="completed")
    with pytest.raises(ValueError, match="release_reason|reason|released"):
        _lease(released_at=NOW + timedelta(seconds=1))


def test_snapshot_defers_task_status_to_lease_coupling_to_the_runtime() -> None:
    snapshot = _snapshot(task_leases=(_lease(),))

    task = next(task for task in snapshot.tasks if task.id == "task-a")
    assert task.status is TaskStatus.PENDING
    assert snapshot.task_leases[0].released_at is None


async def test_ordinary_commit_allows_dependency_suffix_but_not_prefix_rewrite() -> (
    None
):
    first = _dependency("task-a", "task-b")
    second = _dependency("task-a", "task-c")
    initial = _snapshot(task_dependencies=(first,))
    store = InMemoryOperationStore()
    created = await store.create(initial)

    appended = _with_event(
        replace(initial, task_dependencies=(first, second)),
        "event-dependency-appended",
    )
    committed = await store.commit(
        appended,
        expected_revision=created.operation.revision,
    )

    assert committed.operation.snapshot.task_dependencies == (first, second)

    invalid_candidates = (
        (second,),
        (second, first),
        (_dependency("task-b", "task-c"), second),
    )
    for index, dependencies in enumerate(invalid_candidates, start=1):
        candidate = _with_event(
            replace(appended, task_dependencies=dependencies),
            f"event-dependency-rewrite-{index}",
        )
        with pytest.raises(InvalidOperationCheckpointError, match="dependency"):
            await store.commit(
                candidate,
                expected_revision=committed.operation.revision,
            )


async def test_dependency_suffix_is_rejected_after_task_readiness() -> None:
    initial = _snapshot()
    ready_task = replace(initial.tasks[0], status=TaskStatus.READY)
    initial = replace(initial, tasks=(ready_task, *initial.tasks[1:]))
    store = InMemoryOperationStore()
    created = await store.create(initial)
    candidate = _with_event(
        replace(
            initial,
            task_dependencies=(_dependency("task-a", "task-b"),),
        ),
        "event-late-dependency",
    )

    with pytest.raises(InvalidOperationCheckpointError, match="after readiness"):
        await store.commit(
            candidate,
            expected_revision=created.operation.revision,
        )


@pytest.mark.parametrize(
    "advance",
    (
        lambda lease: replace(lease, holder_id="worker-other"),
        lambda lease: replace(
            lease,
            started_at=lease.acquired_at + timedelta(seconds=1),
        ),
        lambda lease: replace(
            lease,
            renewed_at=lease.acquired_at + timedelta(seconds=5),
            expires_at=lease.expires_at + timedelta(seconds=30),
        ),
        lambda lease: replace(
            lease,
            released_at=lease.acquired_at + timedelta(seconds=5),
            release_reason="yielded",
        ),
    ),
    ids=("identity", "start", "renew", "release"),
)
async def test_ordinary_commit_cannot_mutate_lease_history(
    advance: Callable[[TaskLease], TaskLease],
) -> None:
    lease = _lease()
    initial = _snapshot(task_leases=(lease,))
    store = InMemoryOperationStore()
    created = await store.create(initial)
    candidate = _with_event(
        replace(initial, task_leases=(advance(lease),)),
        "event-lease-mutated",
    )

    with pytest.raises(InvalidOperationCheckpointError, match="lease"):
        await store.commit(
            candidate,
            expected_revision=created.operation.revision,
        )


@pytest.mark.parametrize("change", ("append", "delete"))
async def test_ordinary_commit_cannot_append_or_delete_lease_history(
    change: str,
) -> None:
    first = _released_lease(attempt=1, fencing_token=1, offset_seconds=0)
    initial = _snapshot(task_leases=(first,))
    store = InMemoryOperationStore()
    created = await store.create(initial)
    leases: tuple[TaskLease, ...]
    if change == "append":
        leases = (
            first,
            _lease(
                attempt=2,
                fencing_token=2,
                acquired_at=NOW + timedelta(seconds=3),
                expires_at=NOW + timedelta(seconds=33),
            ),
        )
    else:
        leases = ()
    candidate = _with_event(
        replace(initial, task_leases=leases),
        f"event-lease-{change}",
    )

    with pytest.raises(InvalidOperationCheckpointError, match="lease"):
        await store.commit(
            candidate,
            expected_revision=created.operation.revision,
        )


async def test_ordinary_commit_cannot_advance_an_actively_leased_task() -> None:
    lease = _lease()
    initial = _snapshot(task_leases=(lease,))
    store = InMemoryOperationStore()
    created = await store.create(initial)
    leased_task = initial.tasks[0]
    candidate = _with_event(
        replace(
            initial,
            tasks=(
                replace(
                    leased_task,
                    status=TaskStatus.RUNNING,
                    updated_at=NOW + timedelta(seconds=1),
                ),
                *initial.tasks[1:],
            ),
        ),
        "event-unfenced-task-start",
    )

    with pytest.raises(InvalidOperationCheckpointError, match="lease|fenc"):
        await store.commit(
            candidate,
            expected_revision=created.operation.revision,
        )


async def test_ordinary_commit_cannot_start_an_unleased_pending_task() -> None:
    initial = _snapshot()
    store = InMemoryOperationStore()
    created = await store.create(initial)
    pending = initial.tasks[0]
    candidate = _with_event(
        replace(
            initial,
            tasks=(
                replace(
                    pending,
                    status=TaskStatus.RUNNING,
                    updated_at=NOW + timedelta(seconds=1),
                ),
                *initial.tasks[1:],
            ),
        ),
        "event-unleased-task-start",
    )

    with pytest.raises(InvalidOperationCheckpointError, match="claim|fenc|lease"):
        await store.commit(
            candidate,
            expected_revision=created.operation.revision,
        )


async def test_ordinary_commit_cannot_terminalize_an_unleased_running_task() -> None:
    pending = _snapshot()
    running_task = replace(
        pending.tasks[0],
        status=TaskStatus.RUNNING,
        updated_at=NOW + timedelta(seconds=1),
    )
    initial = replace(
        pending,
        tasks=(running_task, *pending.tasks[1:]),
    )
    store = InMemoryOperationStore()
    created = await store.create(initial)
    candidate = _with_event(
        replace(
            initial,
            tasks=(
                replace(
                    running_task,
                    status=TaskStatus.FAILED,
                    error_code="executor_failed",
                    updated_at=NOW + timedelta(seconds=2),
                ),
                *initial.tasks[1:],
            ),
        ),
        "event-unleased-task-failure",
    )

    with pytest.raises(InvalidOperationCheckpointError, match="fenc|lease"):
        await store.commit(
            candidate,
            expected_revision=created.operation.revision,
        )


async def test_task_claim_result_binds_task_and_lease_execution_identity() -> None:
    lease = _lease()
    snapshot = _snapshot(task_leases=(lease,))
    commit_result = await InMemoryOperationStore().create(snapshot)
    task = snapshot.tasks[0]

    result = TaskClaimResult(commit_result=commit_result, task=task, lease=lease)
    assert result.task.id == result.lease.task_id

    with pytest.raises(ValueError, match="execution identity"):
        TaskClaimResult(
            commit_result=commit_result,
            task=task,
            lease=replace(lease, task_id="task-b"),
        )
