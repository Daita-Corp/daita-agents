"""Pure graph scheduling and attempt-settlement reductions.

This module deliberately owns no store, lock, permit, provider, registry, or
background lifecycle.  The one :class:`JobSupervisor` and the SQLite operation
boundary apply the plans produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import AttemptState, GraphTask, TaskState

FIRST_RETRY_BACKOFF_SECONDS = 1
SECOND_RETRY_BACKOFF_SECONDS = 5
MAX_COMPATIBLE_FAILURES = 3
MAX_PROTOCOL_VIOLATIONS = 2


@dataclass(frozen=True, slots=True)
class AttemptFailureReduction:
    """Deterministic task-local outcome for one failed attempt."""

    task_state: TaskState
    not_before: datetime | None
    failure_streak: int
    circuit_open: bool
    retry: bool


def retry_backoff_seconds(failure_streak: int) -> int:
    """Return the persisted V1 retry delay for a compatible failure count."""

    if failure_streak < 1:
        raise ValueError("failure streak must be positive")
    return (
        FIRST_RETRY_BACKOFF_SECONDS
        if failure_streak == 1
        else SECOND_RETRY_BACKOFF_SECONDS
    )


def reduce_attempt_failure(
    task: GraphTask,
    *,
    failed_at: datetime,
    retryable: bool,
    attempt_state: AttemptState,
    maximum_attempts: int,
    deadline_at: datetime,
) -> AttemptFailureReduction:
    """Reduce one effect-free failure without changing any durable state."""

    if task.state is not TaskState.RUNNING:
        raise ValueError("attempt failure reduction requires a running task")
    if attempt_state not in {
        AttemptState.FAILED,
        AttemptState.PROTOCOL_VIOLATION,
        AttemptState.TIMED_OUT,
        AttemptState.FENCED,
    }:
        raise ValueError("attempt failure reduction received an invalid state")
    next_streak = task.failure_streak + 1
    protocol_retry_allowed = not (
        attempt_state is AttemptState.PROTOCOL_VIOLATION
        and task.attempt_count >= MAX_PROTOCOL_VIOLATIONS
    )
    circuit_open = retryable and next_streak >= MAX_COMPATIBLE_FAILURES
    can_retry = (
        retryable
        and protocol_retry_allowed
        and not circuit_open
        and task.attempt_count < maximum_attempts
        and deadline_at > failed_at
    )
    return AttemptFailureReduction(
        task_state=(
            TaskState.READY
            if can_retry
            else TaskState.BLOCKED if circuit_open else TaskState.FAILED
        ),
        not_before=(
            failed_at + timedelta(seconds=retry_backoff_seconds(next_streak))
            if can_retry
            else task.not_before
        ),
        failure_streak=next_streak,
        circuit_open=circuit_open,
        retry=can_retry,
    )


def fair_graph_dispatch_order(
    ready: tuple[GraphTask, ...],
    *,
    last_job_id: str | None,
    consecutive: int,
) -> tuple[GraphTask, ...]:
    """Preserve task ordering while bounding one ready graph to two dispatches."""

    remaining = list(ready)
    ordered: list[GraphTask] = []
    current_job_id = last_job_id
    current_consecutive = consecutive
    while remaining:
        index = 0
        if current_job_id is not None:
            if current_consecutive < 2:
                index = next(
                    (
                        candidate_index
                        for candidate_index, candidate in enumerate(remaining)
                        if candidate.job_id == current_job_id
                    ),
                    0,
                )
            else:
                index = next(
                    (
                        candidate_index
                        for candidate_index, candidate in enumerate(remaining)
                        if candidate.job_id != current_job_id
                    ),
                    0,
                )
        selected = remaining.pop(index)
        ordered.append(selected)
        if selected.job_id == current_job_id:
            current_consecutive += 1
        else:
            current_job_id = selected.job_id
            current_consecutive = 1
    return tuple(ordered)


__all__ = [
    "MAX_COMPATIBLE_FAILURES",
    "MAX_PROTOCOL_VIOLATIONS",
    "AttemptFailureReduction",
    "fair_graph_dispatch_order",
    "reduce_attempt_failure",
    "retry_backoff_seconds",
]
