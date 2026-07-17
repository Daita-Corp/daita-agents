"""Portable optimistic persistence contract for operation checkpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from ..events.models import RuntimeEvent
from .checkpoints import OperationSnapshot


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
            after_task.operation_id != before_task.operation_id
            or after_task.turn_id != before_task.turn_id
            or after_task.call_id != before_task.call_id
            or after_task.capability_id != before_task.capability_id
            or after_task.executor_id != before_task.executor_id
            or after_task.attempt != before_task.attempt
            or after_task.arguments != before_task.arguments
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
