"""Portable optimistic persistence contract for operation checkpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..events.models import RuntimeEvent
from .checkpoints import OperationSnapshot
from .leases import TaskClaimRequest, TaskLease, TaskLeaseGuard
from .models import Task, TaskStatus


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

    async def load_by_trigger(
        self,
        trigger_id: str,
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


class InMemoryOperationStore:
    """Lock-protected reference adapter for the portable operation contract."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._operations: dict[str, VersionedOperation] = {}
        self._operation_by_trigger: dict[str, str] = {}

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

    async def load_by_trigger(self, trigger_id: str) -> VersionedOperation | None:
        _require_identity(trigger_id, "trigger_id")
        async with self._lock:
            operation_id = self._operation_by_trigger.get(trigger_id)
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
            committed = VersionedOperation(
                snapshot=snapshot,
                revision=current.revision + 1,
            )
            self._operations[operation_id] = committed
            return CommitResult(
                operation=committed,
                committed_events=committed_events,
            )


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
