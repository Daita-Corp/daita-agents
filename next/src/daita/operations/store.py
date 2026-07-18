"""Portable optimistic persistence contract for operation checkpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
from typing import Literal, Protocol

from ..events.models import RuntimeEvent
from .checkpoints import OperationSnapshot
from .governance import ApprovalStatus
from .leases import TaskClaimRequest, TaskLease, TaskLeaseGuard
from .models import OperationStatus, Task, TaskStatus


class OperationStoreError(RuntimeError):
    """Base class for typed operation-repository failures."""


class OperationNotFoundError(OperationStoreError):
    """Raised when an operation or trigger has no committed checkpoint."""

    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        super().__init__(f"unknown operation: {operation_id}")


class OperationAlreadyExistsError(OperationStoreError):
    """Raised when an operation identity is already committed."""

    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        super().__init__(f"operation already exists: {operation_id}")


class TriggerAlreadyClaimedError(OperationStoreError):
    """Raised when a trigger already belongs to a committed operation."""

    def __init__(self, trigger_id: str, operation_id: str) -> None:
        self.trigger_id = trigger_id
        self.operation_id = operation_id
        super().__init__(f"trigger {trigger_id} already owns operation: {operation_id}")


class OperationRevisionConflict(OperationStoreError):
    """Raised when an optimistic commit is based on a stale revision."""

    def __init__(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        actual_revision: int,
    ) -> None:
        self.operation_id = operation_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"operation {operation_id} revision conflict: expected "
            f"{expected_revision}, found {actual_revision}"
        )


class InvalidOperationCheckpointError(OperationStoreError):
    """Raised when a checkpoint rewrites identity or committed event history."""

    def __init__(self, operation_id: str, reason: str) -> None:
        self.operation_id = operation_id
        self.reason = reason
        super().__init__(f"invalid operation checkpoint {operation_id}: {reason}")


class TaskNotFoundError(OperationStoreError):
    """Raised when a task is absent from its operation checkpoint."""

    def __init__(self, operation_id: str, task_id: str) -> None:
        self.operation_id = operation_id
        self.task_id = task_id
        super().__init__(f"operation {operation_id} has no task {task_id}")


class TaskDependenciesNotReadyError(OperationStoreError):
    """Raised when a task has prerequisites that have not succeeded."""

    def __init__(
        self,
        operation_id: str,
        task_id: str,
        dependency_ids: tuple[str, ...],
    ) -> None:
        self.operation_id = operation_id
        self.task_id = task_id
        self.dependency_ids = tuple(dependency_ids)
        super().__init__(
            f"operation {operation_id} task {task_id} has dependencies not ready: "
            f"{', '.join(self.dependency_ids)}"
        )


class TaskClaimConflictError(OperationStoreError):
    """Raised when another holder already owns the task's live lease."""

    def __init__(
        self,
        operation_id: str,
        task_id: str,
        holder_id: str,
        fencing_token: int,
        expires_at: datetime,
    ) -> None:
        self.operation_id = operation_id
        self.task_id = task_id
        self.holder_id = holder_id
        self.fencing_token = fencing_token
        self.expires_at = expires_at
        super().__init__(
            f"operation {operation_id} task {task_id} is claimed by {holder_id} "
            f"with fence {fencing_token} until {expires_at.isoformat()}"
        )


class TaskNotClaimableError(OperationStoreError):
    """Raised when durable task state does not permit a claim."""

    def __init__(
        self,
        operation_id: str,
        task_id: str,
        status: TaskStatus,
    ) -> None:
        self.operation_id = operation_id
        self.task_id = task_id
        self.status = status
        super().__init__(
            f"operation {operation_id} task {task_id} is not claimable from "
            f"status {status.value}"
        )


class StaleTaskFenceError(OperationStoreError):
    """Raised when a task mutation presents an obsolete fencing token."""

    def __init__(
        self,
        operation_id: str,
        task_id: str,
        expected_fencing_token: int,
        actual_fencing_token: int,
    ) -> None:
        self.operation_id = operation_id
        self.task_id = task_id
        self.expected_fencing_token = expected_fencing_token
        self.actual_fencing_token = actual_fencing_token
        super().__init__(
            f"operation {operation_id} task {task_id} fence is stale: expected "
            f"{expected_fencing_token}, found {actual_fencing_token}"
        )


class ExpiredTaskLeaseError(OperationStoreError):
    """Raised when a holder checks or commits at or after lease expiry."""

    def __init__(
        self,
        operation_id: str,
        task_id: str,
        fencing_token: int,
        expires_at: datetime,
        checked_at: datetime,
    ) -> None:
        self.operation_id = operation_id
        self.task_id = task_id
        self.fencing_token = fencing_token
        self.expires_at = expires_at
        self.checked_at = checked_at
        super().__init__(
            f"operation {operation_id} task {task_id} lease fence "
            f"{fencing_token} expired at {expires_at.isoformat()} before check "
            f"at {checked_at.isoformat()}"
        )


@dataclass(frozen=True, slots=True)
class VersionedOperation:
    """One immutable operation checkpoint paired with its optimistic revision."""

    snapshot: OperationSnapshot
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OperationSnapshot):
            raise TypeError("snapshot must be an OperationSnapshot record")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("operation revision must be a positive integer")


@dataclass(frozen=True, slots=True)
class CommitResult:
    """A committed checkpoint and only the events added by that commit."""

    operation: VersionedOperation
    committed_events: tuple[RuntimeEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation, VersionedOperation):
            raise TypeError("operation must be a VersionedOperation record")
        events = tuple(self.committed_events)
        if any(not isinstance(event, RuntimeEvent) for event in events):
            raise TypeError("committed_events must contain RuntimeEvent records")
        object.__setattr__(self, "committed_events", events)


@dataclass(frozen=True, slots=True)
class TaskClaimResult:
    """One atomic claim commit and its authoritative task/lease records."""

    commit_result: CommitResult
    task: Task
    lease: TaskLease

    def __post_init__(self) -> None:
        if not isinstance(self.commit_result, CommitResult):
            raise TypeError("commit_result must be a CommitResult record")
        if not isinstance(self.task, Task):
            raise TypeError("task must be a Task record")
        if not isinstance(self.lease, TaskLease):
            raise TypeError("lease must be a TaskLease record")
        if (
            self.task.operation_id != self.lease.operation_id
            or self.task.id != self.lease.task_id
            or self.task.attempt != self.lease.attempt
        ):
            raise ValueError("claim task and lease execution identity must match")
        snapshot = self.commit_result.operation.snapshot
        if self.task not in snapshot.tasks or self.lease not in snapshot.task_leases:
            raise ValueError(
                "claim task and lease must be exact records from the committed snapshot"
            )


class OperationStore(Protocol):
    """Persist operation-owned lifecycle records with optimistic concurrency."""

    async def create(self, snapshot: OperationSnapshot) -> CommitResult: ...

    async def load(self, operation_id: str) -> VersionedOperation: ...

    async def load_nonterminal(
        self,
        agent_id: str,
    ) -> tuple[VersionedOperation, ...]: ...

    async def load_by_trigger(
        self,
        trigger_id: str,
    ) -> VersionedOperation | None: ...

    async def load_by_approval(
        self,
        approval_id: str,
    ) -> VersionedOperation | None: ...

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult: ...


class TaskExecutionStore(OperationStore, Protocol):
    """Atomic task-execution operations layered on operation persistence."""

    async def claim_task(
        self,
        request: TaskClaimRequest,
        *,
        expected_revision: int,
    ) -> TaskClaimResult: ...

    async def renew_task_lease(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
        lease_duration_seconds: float,
    ) -> CommitResult: ...

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult: ...

    async def recover_expired_task(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult: ...


_NONTERMINAL_OPERATION_STATUSES = (
    OperationStatus.PENDING,
    OperationStatus.RUNNING,
    OperationStatus.WAITING_FOR_APPROVAL,
    OperationStatus.WAITING_FOR_INPUT,
)
_TERMINAL_OPERATION_STATUSES = (
    OperationStatus.SUCCEEDED,
    OperationStatus.FAILED,
    OperationStatus.CANCELLED,
    OperationStatus.INTERRUPTED,
)


class InMemoryOperationStore:
    """Lock-protected reference adapter for the portable operation contract."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        max_lease_duration_seconds: float = 300.0,
    ) -> None:
        self._clock = clock if clock is not None else _utc_now
        if not callable(self._clock):
            raise TypeError("clock must be callable")
        self._max_lease_duration_seconds = _require_lease_duration(
            max_lease_duration_seconds,
            "max_lease_duration_seconds",
        )
        self._lock = asyncio.Lock()
        self._operations: dict[str, VersionedOperation] = {}
        self._operation_by_trigger: dict[str, str] = {}
        self._operation_by_approval: dict[str, str] = {}

    async def create(self, snapshot: OperationSnapshot) -> CommitResult:
        _validate_new_checkpoint(snapshot)
        operation_id = snapshot.operation.id
        trigger_id = snapshot.trigger.id
        async with self._lock:
            if operation_id in self._operations:
                raise OperationAlreadyExistsError(operation_id)
            claimed_operation_id = self._operation_by_trigger.get(trigger_id)
            if claimed_operation_id is not None:
                raise TriggerAlreadyClaimedError(trigger_id, claimed_operation_id)
            self._claim_approval_ids(snapshot)

            committed = VersionedOperation(snapshot=snapshot, revision=1)
            self._operations[operation_id] = committed
            self._operation_by_trigger[trigger_id] = operation_id
            return CommitResult(
                operation=committed,
                committed_events=snapshot.events,
            )

    async def load(self, operation_id: str) -> VersionedOperation:
        _require_identity(operation_id, "operation_id")
        async with self._lock:
            try:
                return self._operations[operation_id]
            except KeyError as error:
                raise OperationNotFoundError(operation_id) from error

    async def load_nonterminal(
        self,
        agent_id: str,
    ) -> tuple[VersionedOperation, ...]:
        """Load one agent's exact resumable checkpoints in stable order."""

        _require_identity(agent_id, "agent_id")
        async with self._lock:
            selected: list[VersionedOperation] = []
            for item in self._operations.values():
                operation = item.snapshot.operation
                if operation.agent_id != agent_id:
                    continue
                if operation.status in _NONTERMINAL_OPERATION_STATUSES:
                    selected.append(item)
                    continue
                if operation.status not in _TERMINAL_OPERATION_STATUSES:
                    raise InvalidOperationCheckpointError(
                        operation.id,
                        f"unclassified operation status: {operation.status!r}",
                    )
            return tuple(
                sorted(
                    selected,
                    key=lambda item: (
                        item.snapshot.operation.updated_at.astimezone(timezone.utc),
                        item.snapshot.operation.id,
                    ),
                )
            )

    async def load_by_trigger(self, trigger_id: str) -> VersionedOperation | None:
        _require_identity(trigger_id, "trigger_id")
        async with self._lock:
            operation_id = self._operation_by_trigger.get(trigger_id)
            if operation_id is None:
                return None
            return self._operations[operation_id]

    async def load_by_approval(
        self,
        approval_id: str,
    ) -> VersionedOperation | None:
        _require_identity(approval_id, "approval_id")
        async with self._lock:
            operation_id = self._operation_by_approval.get(approval_id)
            if operation_id is None:
                return None
            return self._operations[operation_id]

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        _validate_new_checkpoint(snapshot)
        _require_revision(expected_revision)
        operation_id = snapshot.operation.id
        async with self._lock:
            try:
                current = self._operations[operation_id]
            except KeyError as error:
                raise OperationNotFoundError(operation_id) from error
            if current.revision != expected_revision:
                raise OperationRevisionConflict(
                    operation_id,
                    expected_revision=expected_revision,
                    actual_revision=current.revision,
                )

            committed_events = _validate_commit_candidate(
                current.snapshot,
                snapshot,
            )
            self._claim_approval_ids(snapshot)
            committed = VersionedOperation(
                snapshot=snapshot,
                revision=current.revision + 1,
            )
            self._operations[operation_id] = committed
            return CommitResult(
                operation=committed,
                committed_events=committed_events,
            )

    async def claim_task(
        self,
        request: TaskClaimRequest,
        *,
        expected_revision: int,
    ) -> TaskClaimResult:
        if not isinstance(request, TaskClaimRequest):
            raise TypeError("request must be a TaskClaimRequest record")
        _require_revision(expected_revision)
        _require_bounded_lease_duration(
            request.lease_duration_seconds,
            maximum=self._max_lease_duration_seconds,
        )
        async with self._lock:
            current = self._current_operation(
                request.operation_id,
                expected_revision=expected_revision,
            )
            now = _authoritative_time(self._clock)
            candidate, task, lease = _prepare_task_claim(
                current.snapshot,
                request,
                now=now,
                max_lease_duration_seconds=self._max_lease_duration_seconds,
            )
            commit_result = self._commit_prepared(current, candidate)
            return TaskClaimResult(
                commit_result=commit_result,
                task=task,
                lease=lease,
            )

    async def renew_task_lease(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
        lease_duration_seconds: float,
    ) -> CommitResult:
        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("snapshot must be an OperationSnapshot record")
        if not isinstance(guard, TaskLeaseGuard):
            raise TypeError("guard must be a TaskLeaseGuard record")
        _require_revision(expected_revision)
        _require_bounded_lease_duration(
            lease_duration_seconds,
            maximum=self._max_lease_duration_seconds,
        )
        async with self._lock:
            current = self._current_operation(
                guard.operation_id,
                expected_revision=expected_revision,
            )
            now = _authoritative_time(self._clock)
            candidate = _prepare_task_lease_renewal(
                current.snapshot,
                snapshot,
                guard,
                now=now,
                lease_duration_seconds=lease_duration_seconds,
                max_lease_duration_seconds=self._max_lease_duration_seconds,
            )
            return self._commit_prepared(current, candidate)

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("snapshot must be an OperationSnapshot record")
        if not isinstance(guard, TaskLeaseGuard):
            raise TypeError("guard must be a TaskLeaseGuard record")
        _require_revision(expected_revision)
        async with self._lock:
            current = self._current_operation(
                guard.operation_id,
                expected_revision=expected_revision,
            )
            now = _authoritative_time(self._clock)
            candidate = _prepare_fenced_task_commit(
                current.snapshot,
                snapshot,
                guard,
                now=now,
            )
            return self._commit_prepared(current, candidate)

    async def recover_expired_task(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("snapshot must be an OperationSnapshot record")
        if not isinstance(guard, TaskLeaseGuard):
            raise TypeError("guard must be a TaskLeaseGuard record")
        _require_revision(expected_revision)
        async with self._lock:
            current = self._current_operation(
                guard.operation_id,
                expected_revision=expected_revision,
            )
            now = _authoritative_time(self._clock)
            candidate = _prepare_expired_task_recovery(
                current.snapshot,
                snapshot,
                guard,
                now=now,
            )
            return self._commit_prepared(current, candidate)

    def _current_operation(
        self,
        operation_id: str,
        *,
        expected_revision: int,
    ) -> VersionedOperation:
        try:
            current = self._operations[operation_id]
        except KeyError as error:
            raise OperationNotFoundError(operation_id) from error
        if current.revision != expected_revision:
            raise OperationRevisionConflict(
                operation_id,
                expected_revision=expected_revision,
                actual_revision=current.revision,
            )
        return current

    def _commit_prepared(
        self,
        current: VersionedOperation,
        candidate: OperationSnapshot,
    ) -> CommitResult:
        committed_events = _committed_event_suffix(current.snapshot, candidate)
        self._claim_approval_ids(candidate)
        committed = VersionedOperation(
            snapshot=candidate,
            revision=current.revision + 1,
        )
        self._operations[candidate.operation.id] = committed
        return CommitResult(
            operation=committed,
            committed_events=committed_events,
        )

    def _claim_approval_ids(self, snapshot: OperationSnapshot) -> None:
        operation_id = snapshot.operation.id
        for approval in snapshot.approvals:
            claimed_operation_id = self._operation_by_approval.get(approval.id)
            if (
                claimed_operation_id is not None
                and claimed_operation_id != operation_id
            ):
                raise InvalidOperationCheckpointError(
                    operation_id,
                    f"approval identity is already claimed: {approval.id}",
                )
        for approval in snapshot.approvals:
            self._operation_by_approval[approval.id] = operation_id


_TaskExecutionMutation = Literal["claim", "renew", "fenced", "recover"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _authoritative_time(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime):
        raise TypeError("store clock must return a timezone-aware datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("store clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _require_lease_duration(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite lease duration")
    return float(value)


def _require_bounded_lease_duration(
    value: object,
    *,
    maximum: float,
) -> float:
    duration = _require_lease_duration(value, "lease_duration_seconds")
    if duration > maximum:
        raise ValueError(
            "lease duration exceeds the adapter maximum " f"of {maximum:g} seconds"
        )
    return duration


def _task_by_id(snapshot: OperationSnapshot, task_id: str) -> Task:
    try:
        return next(task for task in snapshot.tasks if task.id == task_id)
    except StopIteration as error:
        raise TaskNotFoundError(snapshot.operation.id, task_id) from error


def _replace_task(
    snapshot: OperationSnapshot,
    replacement: Task,
) -> tuple[Task, ...]:
    return tuple(
        replacement if task.id == replacement.id else task for task in snapshot.tasks
    )


def _replace_lease(
    snapshot: OperationSnapshot,
    current: TaskLease,
    replacement: TaskLease,
) -> tuple[TaskLease, ...]:
    return tuple(
        replacement if lease == current else lease for lease in snapshot.task_leases
    )


def _active_task_lease(
    snapshot: OperationSnapshot,
    task_id: str,
) -> TaskLease | None:
    return next(
        (
            lease
            for lease in reversed(snapshot.task_leases)
            if lease.task_id == task_id and lease.released_at is None
        ),
        None,
    )


def _require_exact_task_lease(
    snapshot: OperationSnapshot,
    guard: TaskLeaseGuard,
) -> TaskLease:
    if not isinstance(guard, TaskLeaseGuard):
        raise TypeError("guard must be a TaskLeaseGuard record")
    if guard.operation_id != snapshot.operation.id:
        raise TaskNotFoundError(guard.operation_id, guard.task_id)
    _task_by_id(snapshot, guard.task_id)
    active = _active_task_lease(snapshot, guard.task_id)
    if active is None:
        previous_fences = [
            lease.fencing_token
            for lease in snapshot.task_leases
            if lease.task_id == guard.task_id
        ]
        raise StaleTaskFenceError(
            guard.operation_id,
            guard.task_id,
            guard.fencing_token,
            max(previous_fences, default=0),
        )
    if (
        active.holder_id != guard.holder_id
        or active.attempt != guard.attempt
        or active.fencing_token != guard.fencing_token
    ):
        raise StaleTaskFenceError(
            guard.operation_id,
            guard.task_id,
            guard.fencing_token,
            active.fencing_token,
        )
    task = _task_by_id(snapshot, guard.task_id)
    if task.attempt != active.attempt:
        raise InvalidOperationCheckpointError(
            snapshot.operation.id,
            f"task {task.id} attempt does not match its active lease",
        )
    return active


def _require_live_task_lease(
    snapshot: OperationSnapshot,
    guard: TaskLeaseGuard,
    *,
    now: datetime,
) -> TaskLease:
    active = _require_exact_task_lease(snapshot, guard)
    if now >= active.expires_at:
        raise ExpiredTaskLeaseError(
            guard.operation_id,
            guard.task_id,
            guard.fencing_token,
            active.expires_at,
            now,
        )
    return active


def _dependency_ids_not_ready(
    snapshot: OperationSnapshot,
    task_id: str,
) -> tuple[str, ...]:
    task_by_id = {task.id: task for task in snapshot.tasks}
    return tuple(
        dependency.prerequisite_task_id
        for dependency in snapshot.task_dependencies
        if dependency.task_id == task_id
        and task_by_id[dependency.prerequisite_task_id].status
        is not TaskStatus.SUCCEEDED
    )


def _model_call_id_for_task(snapshot: OperationSnapshot, task: Task) -> str:
    for model_call in snapshot.model_calls:
        if model_call.turn_id != task.turn_id or model_call.response is None:
            continue
        if any(call.id == task.call_id for call in model_call.response.tool_calls):
            return model_call.id
    raise InvalidOperationCheckpointError(
        snapshot.operation.id,
        f"task {task.id} has no committed model call",
    )


def _normalized_task_event(
    snapshot: OperationSnapshot,
    task: Task,
    template: RuntimeEvent,
    *,
    now: datetime,
    payload: Mapping[str, object] | None = None,
) -> RuntimeEvent:
    if not isinstance(template, RuntimeEvent):
        raise TypeError("task lifecycle event must be a RuntimeEvent")
    event_payload = template.payload if payload is None else payload
    return RuntimeEvent(
        id=template.id,
        type=template.type,
        agent_id=snapshot.operation.agent_id,
        operation_id=snapshot.operation.id,
        session_id=snapshot.operation.session_id,
        turn_id=task.turn_id,
        model_call_id=_model_call_id_for_task(snapshot, task),
        call_id=task.call_id,
        task_id=task.id,
        evidence_id=(
            template.evidence_id
            if template.type in {"evidence.accepted", "task.succeeded"}
            else None
        ),
        capability_id=task.capability_id,
        executor_id=task.executor_id,
        payload=event_payload,
        created_at=now,
    )


def _single_task_event_template(
    current: OperationSnapshot,
    candidate: OperationSnapshot,
    *,
    task_id: str,
    event_type: str,
) -> RuntimeEvent:
    suffix = _committed_event_suffix(current, candidate)
    if len(suffix) != 1:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            f"{event_type} transition must append exactly one runtime event",
        )
    event = suffix[0]
    if event.type != event_type or event.task_id != task_id:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            f"{event_type} transition requires an event for task {task_id}",
        )
    return event


def _validate_task_execution_candidate(
    current: OperationSnapshot,
    candidate: OperationSnapshot,
    *,
    task_id: str,
    mutation: _TaskExecutionMutation,
) -> tuple[RuntimeEvent, ...]:
    """Validate a narrow lifecycle delta without granting a generic bypass."""

    _validate_checkpoint_structure(candidate)
    _validate_stable_identity(current, candidate)
    suffix = _committed_event_suffix(current, candidate)

    if (
        replace(
            candidate.operation,
            updated_at=current.operation.updated_at,
        )
        != current.operation
    ):
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "task lifecycle commit cannot mutate operation state",
        )
    for field_name in (
        "turns",
        "model_calls",
        "readiness",
        "observations",
        "task_dependencies",
    ):
        if getattr(candidate, field_name) != getattr(current, field_name):
            raise InvalidOperationCheckpointError(
                current.operation.id,
                f"task lifecycle commit cannot mutate {field_name}",
            )
    if candidate.trigger != current.trigger or candidate.budgets != current.budgets:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "task lifecycle commit cannot mutate root identity or budgets",
        )
    if candidate.loop_state != current.loop_state:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "task lifecycle commit cannot mutate loop state or progression",
        )

    if len(candidate.tasks) != len(current.tasks):
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "task lifecycle commit cannot append or remove tasks",
        )
    current_task_ids = tuple(task.id for task in current.tasks)
    candidate_task_ids = tuple(task.id for task in candidate.tasks)
    if current_task_ids != candidate_task_ids:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "task lifecycle commit cannot rewrite or reorder task identities",
        )
    current_tasks = {task.id: task for task in current.tasks}
    candidate_tasks = {task.id: task for task in candidate.tasks}
    for current_task_id, before in current_tasks.items():
        after = candidate_tasks[current_task_id]
        if current_task_id != task_id:
            if after != before:
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    f"task lifecycle commit changed unrelated task: {before.id}",
                )
            continue
        if mutation == "claim":
            normalized = replace(
                after,
                status=before.status,
                attempt=before.attempt,
                updated_at=before.updated_at,
            )
        elif mutation == "fenced":
            normalized = replace(
                after,
                status=before.status,
                updated_at=before.updated_at,
                evidence_ids=before.evidence_ids,
                error_code=before.error_code,
                cancellation_requested=before.cancellation_requested,
                manual_recovery_reason=before.manual_recovery_reason,
            )
        elif mutation == "recover":
            normalized = replace(
                after,
                status=before.status,
                updated_at=before.updated_at,
                manual_recovery_reason=before.manual_recovery_reason,
            )
        else:
            normalized = replace(
                after,
                status=before.status,
                updated_at=before.updated_at,
            )
        if before.cancellation_requested and not after.cancellation_requested:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                f"task cancellation intent is monotonic: {before.id}",
            )
        if normalized != before:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                f"task lifecycle commit rewrote immutable task facts: {before.id}",
            )

    if mutation == "fenced":
        if candidate.evidence[: len(current.evidence)] != current.evidence:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced evidence history must preserve the committed prefix",
            )
        target = candidate_tasks[task_id]
        if any(
            evidence.task_id != task_id
            or evidence.attempt != target.attempt
            or evidence.capability_id != target.capability_id
            or evidence.executor_id != target.executor_id
            for evidence in candidate.evidence[len(current.evidence) :]
        ):
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced evidence must match the guarded task execution identity",
            )
    elif candidate.evidence != current.evidence:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "only a fenced task outcome may append evidence",
        )

    if mutation == "claim":
        if (
            candidate.task_leases[: len(current.task_leases)] != current.task_leases
            or len(candidate.task_leases) != len(current.task_leases) + 1
            or candidate.task_leases[-1].task_id != task_id
        ):
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "task claim must append exactly one lease",
            )
    else:
        if len(candidate.task_leases) != len(current.task_leases):
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "task lifecycle commit cannot append or remove lease attempts",
            )
        changed = tuple(
            (before, after)
            for before, after in zip(
                current.task_leases,
                candidate.task_leases,
                strict=True,
            )
            if before != after
        )
        if mutation == "fenced":
            if len(changed) > 1 or (changed and changed[0][0].task_id != task_id):
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "fenced task lifecycle may mutate only the guarded lease",
                )
        elif len(changed) != 1 or changed[0][0].task_id != task_id:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "task lifecycle commit must mutate exactly the guarded lease",
            )
    return suffix


def _prepare_task_claim(
    current: OperationSnapshot,
    request: TaskClaimRequest,
    *,
    now: datetime,
    max_lease_duration_seconds: float,
) -> tuple[OperationSnapshot, Task, TaskLease]:
    if request.operation_id != current.operation.id:
        raise OperationNotFoundError(request.operation_id)
    duration = _require_bounded_lease_duration(
        request.lease_duration_seconds,
        maximum=max_lease_duration_seconds,
    )
    task = _task_by_id(current, request.task_id)
    if task.status is not TaskStatus.READY or task.cancellation_requested:
        raise TaskNotClaimableError(current.operation.id, task.id, task.status)
    not_ready = _dependency_ids_not_ready(current, task.id)
    if not_ready:
        raise TaskDependenciesNotReadyError(current.operation.id, task.id, not_ready)

    active = _active_task_lease(current, task.id)
    if active is not None and now < active.expires_at:
        raise TaskClaimConflictError(
            current.operation.id,
            task.id,
            active.holder_id,
            active.fencing_token,
            active.expires_at,
        )
    if active is not None:
        raise TaskNotClaimableError(current.operation.id, task.id, task.status)

    prior_leases = tuple(
        lease for lease in current.task_leases if lease.task_id == task.id
    )
    if not prior_leases:
        if task.attempt != 1:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                f"first claim requires task attempt 1: {task.id}",
            )
        attempt = 1
    else:
        latest_attempt = max(lease.attempt for lease in prior_leases)
        if task.attempt != latest_attempt:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                f"task attempt does not match durable lease history: {task.id}",
            )
        attempt = latest_attempt + 1
    fencing_token = (
        1
        if not prior_leases
        else max(lease.fencing_token for lease in prior_leases) + 1
    )
    expires_at = now + timedelta(seconds=duration)
    claimed_task = replace(
        task,
        status=TaskStatus.CLAIMED,
        attempt=attempt,
        updated_at=now,
    )
    lease = TaskLease(
        operation_id=current.operation.id,
        task_id=task.id,
        attempt=attempt,
        fencing_token=fencing_token,
        holder_id=request.holder_id,
        acquired_at=now,
        expires_at=expires_at,
    )
    event = _normalized_task_event(
        current,
        claimed_task,
        request.event,
        now=now,
        payload={
            "holder_id": lease.holder_id,
            "attempt": lease.attempt,
            "fencing_token": lease.fencing_token,
            "acquired_at": lease.acquired_at.isoformat(),
            "expires_at": lease.expires_at.isoformat(),
        },
    )
    candidate = replace(
        current,
        operation=replace(current.operation, updated_at=now),
        tasks=_replace_task(current, claimed_task),
        task_leases=(*current.task_leases, lease),
        events=(*current.events, event),
    )
    _validate_task_execution_candidate(
        current,
        candidate,
        task_id=task.id,
        mutation="claim",
    )
    return candidate, claimed_task, lease


def _prepare_task_lease_renewal(
    current: OperationSnapshot,
    proposed: OperationSnapshot,
    guard: TaskLeaseGuard,
    *,
    now: datetime,
    lease_duration_seconds: float,
    max_lease_duration_seconds: float,
) -> OperationSnapshot:
    duration = _require_bounded_lease_duration(
        lease_duration_seconds,
        maximum=max_lease_duration_seconds,
    )
    active = _require_live_task_lease(current, guard, now=now)
    task = _task_by_id(current, guard.task_id)
    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskNotClaimableError(current.operation.id, task.id, task.status)
    if (task.status is TaskStatus.CLAIMED) != (active.started_at is None):
        raise InvalidOperationCheckpointError(
            current.operation.id,
            f"task {task.id} status does not match its lease start checkpoint",
        )
    _validate_task_execution_candidate(
        current,
        proposed,
        task_id=guard.task_id,
        mutation="renew",
    )
    proposed_lease = proposed.task_leases[current.task_leases.index(active)]
    if (
        proposed_lease.renewed_at is None
        or replace(
            proposed_lease,
            renewed_at=active.renewed_at,
            expires_at=active.expires_at,
        )
        != active
    ):
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "lease renewal may change only renewed_at and expires_at",
        )
    expires_at = now + timedelta(seconds=duration)
    if expires_at <= active.expires_at:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "lease renewal must strictly extend expiry",
        )
    renewed = replace(active, renewed_at=now, expires_at=expires_at)
    event_template = _single_task_event_template(
        current,
        proposed,
        task_id=guard.task_id,
        event_type="task.lease_renewed",
    )
    event = _normalized_task_event(
        current,
        task,
        event_template,
        now=now,
        payload={
            "holder_id": active.holder_id,
            "attempt": active.attempt,
            "fencing_token": active.fencing_token,
            "renewed_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        },
    )
    candidate = replace(
        current,
        operation=replace(current.operation, updated_at=now),
        task_leases=_replace_lease(current, active, renewed),
        events=(*current.events, event),
    )
    _validate_task_execution_candidate(
        current,
        candidate,
        task_id=guard.task_id,
        mutation="renew",
    )
    return candidate


def _prepare_fenced_task_commit(
    current: OperationSnapshot,
    proposed: OperationSnapshot,
    guard: TaskLeaseGuard,
    *,
    now: datetime,
) -> OperationSnapshot:
    active = _require_live_task_lease(current, guard, now=now)
    suffix = _validate_task_execution_candidate(
        current,
        proposed,
        task_id=guard.task_id,
        mutation="fenced",
    )
    before_task = _task_by_id(current, guard.task_id)
    after_task = _task_by_id(proposed, guard.task_id)
    proposed_lease = proposed.task_leases[current.task_leases.index(active)]
    new_evidence = proposed.evidence[len(current.evidence) :]

    if before_task.status is TaskStatus.CLAIMED:
        if before_task.cancellation_requested or after_task.cancellation_requested:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced start cannot erase or execute after cancellation intent",
            )
        if after_task.status is not TaskStatus.RUNNING:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "claimed task may only enter running through a fenced commit",
            )
        if (
            active.started_at is not None
            or proposed_lease.started_at is None
            or replace(proposed_lease, started_at=active.started_at) != active
        ):
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced start may change only the lease started_at checkpoint",
            )
        if len(suffix) != 1 or suffix[0].type != "executor.started":
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced start must append exactly one executor.started event",
            )
        if new_evidence:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced start cannot append evidence",
            )
        committed_lease = replace(active, started_at=now)
    elif (
        before_task.status is TaskStatus.RUNNING
        and after_task.status is TaskStatus.RUNNING
    ):
        if active.started_at is None:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "running task requires a durable lease start checkpoint",
            )
        if proposed_lease != active:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced outcome annotation cannot mutate its live lease",
            )
        if after_task.cancellation_requested != before_task.cancellation_requested:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced outcome annotation cannot manufacture cancellation intent",
            )
        if new_evidence:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced outcome annotation cannot append evidence",
            )
        if len(suffix) != 1 or suffix[0].type != "task.outcome_unknown":
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "running task annotation requires one task.outcome_unknown event",
            )
        reason = suffix[0].payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "task.outcome_unknown requires a non-empty reason",
            )
        committed_lease = active
    elif before_task.status is TaskStatus.RUNNING:
        if active.started_at is None:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "running task requires a durable lease start checkpoint",
            )
        if after_task.status not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "running task requires a fenced terminal outcome",
            )
        if (
            proposed_lease.released_at is None
            or proposed_lease.release_reason is None
            or replace(
                proposed_lease,
                released_at=active.released_at,
                release_reason=active.release_reason,
            )
            != active
        ):
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced outcome may change only the lease release checkpoint",
            )
        expected_event_types: tuple[str, ...]
        release_reason: str
        if after_task.status is TaskStatus.SUCCEEDED:
            if after_task.cancellation_requested != before_task.cancellation_requested:
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "task.succeeded cannot manufacture cancellation intent",
                )
            if not new_evidence or any(
                not evidence.accepted for evidence in new_evidence
            ):
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "task.succeeded requires newly accepted evidence",
                )
            expected_event_types = (
                "executor.completed",
                *("evidence.accepted" for _ in new_evidence),
                "task.succeeded",
            )
            evidence_event_ids = tuple(
                event.evidence_id
                for event in suffix
                if event.type == "evidence.accepted"
            )
            new_evidence_ids = tuple(evidence.id for evidence in new_evidence)
            if evidence_event_ids != new_evidence_ids:
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "evidence.accepted events must map exactly to new evidence",
                )
            if suffix[-1].evidence_id != new_evidence_ids[-1]:
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "task.succeeded must reference the committed evidence",
                )
            release_reason = "completed"
        elif after_task.status is TaskStatus.FAILED:
            if after_task.cancellation_requested != before_task.cancellation_requested:
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "task.failed cannot manufacture cancellation intent",
                )
            if new_evidence:
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "task.failed cannot append evidence",
                )
            expected_event_types = ("task.failed", "executor.failed")
            assert after_task.error_code is not None
            release_reason = after_task.error_code
        else:
            if new_evidence:
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "task.cancelled cannot append evidence",
                )
            if (
                not before_task.cancellation_requested
                or not after_task.cancellation_requested
            ):
                raise InvalidOperationCheckpointError(
                    current.operation.id,
                    "task.cancelled requires previously durable cancellation intent",
                )
            expected_event_types = ("task.cancelled",)
            release_reason = "cancelled"
        if tuple(event.type for event in suffix) != expected_event_types:
            expected_events = ", ".join(expected_event_types)
            raise InvalidOperationCheckpointError(
                current.operation.id,
                f"{after_task.status.value} task requires events: {expected_events}",
            )
        if any(event.task_id != guard.task_id for event in suffix):
            raise InvalidOperationCheckpointError(
                current.operation.id,
                "fenced outcome events must match the guarded task",
            )
        committed_lease = replace(
            active,
            released_at=now,
            release_reason=release_reason,
        )
    else:
        raise TaskNotClaimableError(
            current.operation.id,
            before_task.id,
            before_task.status,
        )

    committed_task = replace(after_task, updated_at=now)
    committed_evidence = (
        *current.evidence,
        *(replace(evidence, created_at=now) for evidence in new_evidence),
    )

    def normalized_payload(event: RuntimeEvent) -> Mapping[str, object]:
        if event.type == "executor.started":
            return {
                "task_id": committed_task.id,
                "executor_id": committed_task.executor_id,
                "holder_id": committed_lease.holder_id,
                "attempt": committed_lease.attempt,
                "fencing_token": committed_lease.fencing_token,
            }
        if event.type == "executor.completed":
            return {
                "task_id": committed_task.id,
                "executor_id": committed_task.executor_id,
            }
        if event.type == "evidence.accepted":
            return {
                "task_id": committed_task.id,
                "evidence_id": event.evidence_id,
            }
        if event.type == "task.succeeded":
            return {"task_id": committed_task.id}
        if event.type == "task.outcome_unknown":
            return {
                "task_id": committed_task.id,
                "executor_id": committed_task.executor_id,
                "reason": event.payload["reason"],
                "status": committed_task.status.value,
                "attempt": committed_lease.attempt,
                "fencing_token": committed_lease.fencing_token,
            }
        if event.type in {"task.failed", "executor.failed"}:
            payload: dict[str, object] = {
                "task_id": committed_task.id,
                "error_code": committed_task.error_code,
            }
            if event.type == "executor.failed":
                payload["executor_id"] = committed_task.executor_id
            return payload
        return {"task_id": committed_task.id}

    committed_events = tuple(
        _normalized_task_event(
            proposed,
            committed_task,
            event,
            now=now,
            payload=normalized_payload(event),
        )
        for event in suffix
    )
    candidate = replace(
        proposed,
        operation=replace(proposed.operation, updated_at=now),
        tasks=_replace_task(proposed, committed_task),
        task_leases=_replace_lease(proposed, proposed_lease, committed_lease),
        evidence=committed_evidence,
        events=(*current.events, *committed_events),
    )
    _validate_task_execution_candidate(
        current,
        candidate,
        task_id=guard.task_id,
        mutation="fenced",
    )
    return candidate


def _task_execution_is_replay_safe(task: Task) -> bool:
    facts = task.execution_facts
    if not facts.replay_safe or not facts.idempotent:
        return False
    if facts.access_mode.value == "read" and not facts.side_effecting:
        return True
    return facts.side_effecting and facts.idempotency_key is not None


def _prepare_expired_task_recovery(
    current: OperationSnapshot,
    proposed: OperationSnapshot,
    guard: TaskLeaseGuard,
    *,
    now: datetime,
) -> OperationSnapshot:
    active = _require_exact_task_lease(current, guard)
    if now < active.expires_at:
        raise InvalidOperationCheckpointError(
            current.operation.id,
            "task lease has not expired and cannot be recovered",
        )
    _validate_task_execution_candidate(
        current,
        proposed,
        task_id=guard.task_id,
        mutation="recover",
    )
    event_template = _single_task_event_template(
        current,
        proposed,
        task_id=guard.task_id,
        event_type="task.lease_lost",
    )
    task = _task_by_id(current, guard.task_id)
    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskNotClaimableError(current.operation.id, task.id, task.status)

    never_started = task.status is TaskStatus.CLAIMED and active.started_at is None
    consistently_started = (
        task.status is TaskStatus.RUNNING and active.started_at is not None
    )
    if never_started:
        next_status = (
            TaskStatus.CANCELLED if task.cancellation_requested else TaskStatus.READY
        )
        release_reason = (
            "cancelled_before_start"
            if task.cancellation_requested
            else "expired_before_start"
        )
        manual_recovery_reason = None
    elif consistently_started and _task_execution_is_replay_safe(task):
        next_status = (
            TaskStatus.CANCELLED if task.cancellation_requested else TaskStatus.READY
        )
        release_reason = (
            "cancelled_replay_safe"
            if task.cancellation_requested
            else "expired_replay_safe"
        )
        manual_recovery_reason = None
    else:
        next_status = TaskStatus.MANUAL_RECOVERY_REQUIRED
        release_reason = "expired_unknown_outcome"
        manual_recovery_reason = "unknown_side_effect_outcome"

    recovered_task = replace(
        task,
        status=next_status,
        updated_at=now,
        manual_recovery_reason=manual_recovery_reason,
    )
    released_lease = replace(
        active,
        released_at=now,
        release_reason=release_reason,
    )
    event = _normalized_task_event(
        current,
        recovered_task,
        event_template,
        now=now,
        payload={
            "holder_id": active.holder_id,
            "attempt": active.attempt,
            "fencing_token": active.fencing_token,
            "from_status": task.status.value,
            "to_status": next_status.value,
            "reason": release_reason,
        },
    )
    candidate = replace(
        current,
        operation=replace(current.operation, updated_at=now),
        tasks=_replace_task(current, recovered_task),
        task_leases=_replace_lease(current, active, released_lease),
        events=(*current.events, event),
    )
    _validate_task_execution_candidate(
        current,
        candidate,
        task_id=guard.task_id,
        mutation="recover",
    )
    return candidate


def _require_identity(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_revision(revision: int) -> None:
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise ValueError("expected_revision must be a non-negative integer")


def _validate_new_checkpoint(snapshot: OperationSnapshot) -> None:
    """Validate portable aggregate facts before any concrete adapter mutates."""

    _validate_checkpoint_structure(snapshot)


def _validate_commit_candidate(
    current: OperationSnapshot,
    candidate: OperationSnapshot,
) -> tuple[RuntimeEvent, ...]:
    """Validate a replacement checkpoint and return its exact event suffix."""

    _validate_checkpoint_structure(candidate)
    _validate_stable_identity(current, candidate)
    _validate_stable_history(current, candidate)
    return _committed_event_suffix(current, candidate)


def _validate_checkpoint_structure(snapshot: OperationSnapshot) -> None:
    if not isinstance(snapshot, OperationSnapshot):
        raise TypeError("snapshot must be an OperationSnapshot record")
    trigger = snapshot.trigger
    operation = snapshot.operation
    if trigger.id != operation.trigger_id:
        raise InvalidOperationCheckpointError(
            operation.id,
            "checkpoint trigger does not match operation trigger identity",
        )
    if trigger.agent_id != operation.agent_id:
        raise InvalidOperationCheckpointError(
            operation.id,
            "checkpoint trigger does not match operation agent identity",
        )
    if trigger.session_id != operation.session_id:
        raise InvalidOperationCheckpointError(
            operation.id,
            "checkpoint trigger does not match operation session identity",
        )
    if not snapshot.events:
        raise InvalidOperationCheckpointError(
            operation.id,
            "operation checkpoint must contain at least one committed event",
        )
    event_ids: set[str] = set()
    for event in snapshot.events:
        if not isinstance(event, RuntimeEvent):
            raise InvalidOperationCheckpointError(
                operation.id,
                "operation checkpoint events must be RuntimeEvent records",
            )
        if event.id in event_ids:
            raise InvalidOperationCheckpointError(
                operation.id,
                f"operation checkpoint contains duplicate event id: {event.id}",
            )
        event_ids.add(event.id)
        if (
            event.agent_id != operation.agent_id
            or event.operation_id != operation.id
            or event.session_id != operation.session_id
        ):
            raise InvalidOperationCheckpointError(
                operation.id,
                f"event {event.id} does not match operation identity",
            )


def _validate_stable_identity(
    current: OperationSnapshot,
    candidate: OperationSnapshot,
) -> None:
    if candidate.trigger != current.trigger:
        raise InvalidOperationCheckpointError(
            candidate.operation.id,
            "committed operation trigger identity is immutable",
        )
    current_operation = current.operation
    candidate_operation = candidate.operation
    if (
        candidate_operation.id != current_operation.id
        or candidate_operation.agent_id != current_operation.agent_id
        or candidate_operation.trigger_id != current_operation.trigger_id
        or candidate_operation.session_id != current_operation.session_id
        or candidate_operation.created_at != current_operation.created_at
    ):
        raise InvalidOperationCheckpointError(
            candidate_operation.id,
            "committed operation identity is immutable",
        )
    if candidate_operation.updated_at < current_operation.updated_at:
        raise InvalidOperationCheckpointError(
            candidate_operation.id,
            "operation updated_at cannot precede its committed value",
        )
    if candidate.budgets != current.budgets:
        raise InvalidOperationCheckpointError(
            candidate_operation.id,
            "committed operation budgets are immutable",
        )


def _committed_event_suffix(
    current: OperationSnapshot,
    candidate: OperationSnapshot,
) -> tuple[RuntimeEvent, ...]:
    committed_count = len(current.events)
    if candidate.events[:committed_count] != current.events:
        raise InvalidOperationCheckpointError(
            candidate.operation.id,
            "operation event history must preserve the exact committed prefix",
        )
    suffix = candidate.events[committed_count:]
    if not suffix:
        raise InvalidOperationCheckpointError(
            candidate.operation.id,
            "operation commit must append at least one runtime event",
        )
    return suffix


def _validate_stable_history(
    current: OperationSnapshot,
    candidate: OperationSnapshot,
) -> None:
    operation_id = candidate.operation.id
    actively_leased_task_ids = {
        lease.task_id for lease in current.task_leases if lease.released_at is None
    }
    for appended_approval in candidate.approvals[len(current.approvals) :]:
        if appended_approval.status is not ApprovalStatus.PENDING:
            raise InvalidOperationCheckpointError(
                operation_id,
                f"new approval must be pending: {appended_approval.id}",
            )
    _validate_record_id_prefix(
        current.approvals,
        candidate.approvals,
        operation_id,
        "approval",
    )
    for before_approval, after_approval in zip(
        current.approvals,
        candidate.approvals,
        strict=False,
    ):
        if (
            after_approval.operation_id != before_approval.operation_id
            or after_approval.task_id != before_approval.task_id
            or after_approval.task_fingerprint != before_approval.task_fingerprint
            or after_approval.policy_fingerprint != before_approval.policy_fingerprint
            or after_approval.requested_at != before_approval.requested_at
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"committed approval identity is immutable: {before_approval.id}",
            )
        if before_approval.status is not ApprovalStatus.PENDING:
            if after_approval != before_approval:
                raise InvalidOperationCheckpointError(
                    operation_id,
                    f"terminal approval history is immutable: {before_approval.id}",
                )
        elif after_approval.status not in {
            ApprovalStatus.PENDING,
            ApprovalStatus.APPROVED,
            ApprovalStatus.DENIED,
            ApprovalStatus.CANCELLED,
        }:
            raise InvalidOperationCheckpointError(
                operation_id,
                f"invalid approval transition: {before_approval.id}",
            )
    for appended_task in candidate.tasks[len(current.tasks) :]:
        if (
            appended_task.status is not TaskStatus.PENDING
            or appended_task.attempt != 1
            or appended_task.cancellation_requested
            or appended_task.evidence_ids
            or appended_task.error_code is not None
            or appended_task.manual_recovery_reason is not None
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"new task must materialize as a clean pending attempt: "
                f"{appended_task.id}",
            )
    _validate_record_id_prefix(current.turns, candidate.turns, operation_id, "turn")
    for before_turn, after_turn in zip(
        current.turns,
        candidate.turns,
        strict=False,
    ):
        if (
            after_turn.operation_id != before_turn.operation_id
            or after_turn.number != before_turn.number
            or after_turn.created_at != before_turn.created_at
            or (
                before_turn.model_request_id is not None
                and after_turn.model_request_id != before_turn.model_request_id
            )
            or (
                before_turn.model_response_id is not None
                and after_turn.model_response_id != before_turn.model_response_id
            )
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"committed turn history is immutable: {before_turn.id}",
            )

    _validate_record_id_prefix(
        current.model_calls,
        candidate.model_calls,
        operation_id,
        "model call",
    )
    for before_call, after_call in zip(
        current.model_calls,
        candidate.model_calls,
        strict=False,
    ):
        if (
            after_call.operation_id != before_call.operation_id
            or after_call.turn_id != before_call.turn_id
            or after_call.provider_id != before_call.provider_id
            or after_call.request != before_call.request
            or after_call.created_at != before_call.created_at
            or after_call.updated_at < before_call.updated_at
            or (
                before_call.response is not None
                and after_call.response != before_call.response
            )
            or (
                before_call.error_code is not None
                and after_call.error_code != before_call.error_code
            )
            or (
                before_call.cancellation_requested
                and not after_call.cancellation_requested
            )
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"committed model-call history is immutable: {before_call.id}",
            )

    _validate_record_id_prefix(current.tasks, candidate.tasks, operation_id, "task")
    for before_task, after_task in zip(
        current.tasks,
        candidate.tasks,
        strict=False,
    ):
        if (
            before_task.status
            in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.MANUAL_RECOVERY_REQUIRED,
            }
            and after_task != before_task
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"terminal task history is immutable: {before_task.id}",
            )
        allowed_statuses = {
            TaskStatus.PENDING: {
                TaskStatus.PENDING,
                TaskStatus.READY,
                TaskStatus.WAITING_FOR_APPROVAL,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.MANUAL_RECOVERY_REQUIRED,
            },
            TaskStatus.READY: {TaskStatus.READY},
            TaskStatus.CLAIMED: {TaskStatus.CLAIMED},
            TaskStatus.RUNNING: {TaskStatus.RUNNING},
            TaskStatus.WAITING_FOR_APPROVAL: {
                TaskStatus.WAITING_FOR_APPROVAL,
                TaskStatus.READY,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            },
            TaskStatus.SUCCEEDED: {TaskStatus.SUCCEEDED},
            TaskStatus.FAILED: {TaskStatus.FAILED},
            TaskStatus.CANCELLED: {TaskStatus.CANCELLED},
            TaskStatus.MANUAL_RECOVERY_REQUIRED: {
                TaskStatus.MANUAL_RECOVERY_REQUIRED,
            },
        }
        if after_task.status not in allowed_statuses[before_task.status]:
            raise InvalidOperationCheckpointError(
                operation_id,
                f"ordinary commit cannot transition task {before_task.id} from "
                f"{before_task.status.value} to {after_task.status.value}; "
                "claim and fenced transitions require a lease",
            )
        if (
            before_task.status is TaskStatus.WAITING_FOR_APPROVAL
            and after_task.status is not TaskStatus.WAITING_FOR_APPROVAL
        ):
            approval = next(
                (item for item in candidate.approvals if item.task_id == after_task.id),
                None,
            )
            expected_approval_statuses = {
                TaskStatus.READY: {ApprovalStatus.APPROVED},
                TaskStatus.FAILED: {ApprovalStatus.DENIED},
                TaskStatus.CANCELLED: {
                    ApprovalStatus.APPROVED,
                    ApprovalStatus.CANCELLED,
                },
            }[after_task.status]
            if approval is None or approval.status not in expected_approval_statuses:
                raise InvalidOperationCheckpointError(
                    operation_id,
                    f"approval does not authorize task transition: {after_task.id}",
                )
        if (
            before_task.status is TaskStatus.PENDING
            and after_task.status is TaskStatus.READY
        ):
            not_ready = _dependency_ids_not_ready(candidate, after_task.id)
            if not_ready:
                raise InvalidOperationCheckpointError(
                    operation_id,
                    f"task {after_task.id} cannot become ready before dependencies "
                    f"succeed: {', '.join(not_ready)}",
                )
        if (
            before_task.status is TaskStatus.PENDING
            and after_task.status is TaskStatus.WAITING_FOR_APPROVAL
        ):
            approval = next(
                (
                    item
                    for item in candidate.approvals
                    if item.task_id == after_task.id
                    and item.status is ApprovalStatus.PENDING
                ),
                None,
            )
            if (
                approval is None
                or candidate.loop_state.waiting_approval_id != approval.id
                or candidate.operation.status
                is not OperationStatus.WAITING_FOR_APPROVAL
            ):
                raise InvalidOperationCheckpointError(
                    operation_id,
                    f"waiting task requires its pending approval: {after_task.id}",
                )
        if (
            before_task.id in actively_leased_task_ids
            and after_task.status is not before_task.status
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"actively leased task requires a fenced commit: {before_task.id}",
            )
        if (
            after_task.operation_id != before_task.operation_id
            or after_task.turn_id != before_task.turn_id
            or after_task.call_id != before_task.call_id
            or after_task.capability_id != before_task.capability_id
            or after_task.executor_id != before_task.executor_id
            or after_task.attempt != before_task.attempt
            or after_task.arguments != before_task.arguments
            or after_task.execution_facts != before_task.execution_facts
            or after_task.created_at != before_task.created_at
            or after_task.updated_at < before_task.updated_at
            or after_task.evidence_ids[: len(before_task.evidence_ids)]
            != before_task.evidence_ids
            or (
                before_task.error_code is not None
                and after_task.error_code != before_task.error_code
            )
            or (
                before_task.cancellation_requested
                and not after_task.cancellation_requested
            )
            or (
                before_task.manual_recovery_reason is not None
                and after_task.manual_recovery_reason
                != before_task.manual_recovery_reason
            )
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"committed task history is immutable: {before_task.id}",
            )

    _validate_exact_history_prefix(
        current.evidence,
        candidate.evidence,
        operation_id,
        "evidence",
    )
    if any(
        evidence.task_id in actively_leased_task_ids
        for evidence in candidate.evidence[len(current.evidence) :]
    ):
        raise InvalidOperationCheckpointError(
            operation_id,
            "actively leased task evidence requires a fenced commit",
        )
    _validate_exact_history_prefix(
        current.readiness,
        candidate.readiness,
        operation_id,
        "readiness",
    )
    _validate_exact_history_prefix(
        current.observations,
        candidate.observations,
        operation_id,
        "observation",
    )
    _validate_exact_history_prefix(
        current.task_dependencies,
        candidate.task_dependencies,
        operation_id,
        "task dependency",
    )
    current_task_by_id = {task.id: task for task in current.tasks}
    leased_task_ids = {lease.task_id for lease in current.task_leases}
    for dependency in candidate.task_dependencies[len(current.task_dependencies) :]:
        current_task = current_task_by_id.get(dependency.task_id)
        candidate_task = next(
            task for task in candidate.tasks if task.id == dependency.task_id
        )
        if (
            candidate_task.status is not TaskStatus.PENDING
            or dependency.task_id in leased_task_ids
            or (
                current_task is not None
                and current_task.status is not TaskStatus.PENDING
            )
        ):
            raise InvalidOperationCheckpointError(
                operation_id,
                f"task dependencies are immutable after readiness: "
                f"{dependency.task_id}",
            )
    if candidate.task_leases != current.task_leases:
        raise InvalidOperationCheckpointError(
            operation_id,
            "ordinary operation commit cannot mutate task lease history",
        )


def _validate_exact_history_prefix(
    current: tuple[object, ...],
    candidate: tuple[object, ...],
    operation_id: str,
    label: str,
) -> None:
    if candidate[: len(current)] != current:
        raise InvalidOperationCheckpointError(
            operation_id,
            f"committed {label} history must remain an exact prefix",
        )


def _validate_record_id_prefix(
    current: tuple[object, ...],
    candidate: tuple[object, ...],
    operation_id: str,
    label: str,
) -> None:
    current_ids = tuple(getattr(record, "id") for record in current)
    candidate_ids = tuple(getattr(record, "id") for record in candidate)
    if candidate_ids[: len(current_ids)] != current_ids:
        raise InvalidOperationCheckpointError(
            operation_id,
            f"committed {label} identities must remain an exact prefix",
        )
