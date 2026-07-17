"""Portable value objects for durable, fenced task execution leases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..events.models import RuntimeEvent


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a non-empty string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a timezone-aware datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class TaskLease:
    """One immutable checkpoint in a task's fenced execution attempt.

    A claim may be renewed while it is still queued, so ``renewed_at`` may
    precede ``started_at``. Both remain inside the same live lease interval.
    """

    operation_id: str
    task_id: str
    attempt: int
    fencing_token: int
    holder_id: str
    acquired_at: datetime
    expires_at: datetime
    started_at: datetime | None = None
    renewed_at: datetime | None = None
    released_at: datetime | None = None
    release_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "operation_id")
        _required_text(self.task_id, "task_id")
        _required_text(self.holder_id, "holder_id")
        _positive_integer(self.attempt, "attempt")
        _positive_integer(self.fencing_token, "fencing_token")

        acquired_at = _aware_datetime(self.acquired_at, "acquired_at")
        expires_at = _aware_datetime(self.expires_at, "expires_at")
        if expires_at <= acquired_at:
            raise ValueError("expires_at must be after acquired_at")

        checkpoints: tuple[tuple[str, datetime | None], ...] = (
            ("started_at", self.started_at),
            ("renewed_at", self.renewed_at),
        )
        for field_name, checkpoint in checkpoints:
            if checkpoint is None:
                continue
            checkpoint = _aware_datetime(checkpoint, field_name)
            if checkpoint < acquired_at or checkpoint >= expires_at:
                raise ValueError(
                    f"{field_name} must fall within [acquired_at, expires_at)"
                )

        has_released_at = self.released_at is not None
        has_release_reason = self.release_reason is not None
        if has_released_at != has_release_reason:
            raise ValueError("released_at and release_reason must be provided together")
        if not has_released_at:
            return

        released_at = _aware_datetime(self.released_at, "released_at")
        _required_text(self.release_reason, "release_reason")
        lower_bound = max(
            timestamp
            for timestamp in (acquired_at, self.started_at, self.renewed_at)
            if timestamp is not None
        )
        if released_at < lower_bound:
            raise ValueError(
                "released_at cannot precede acquired_at, started_at, or renewed_at"
            )


@dataclass(frozen=True, slots=True)
class TaskClaimRequest:
    """Portable candidate for an atomic task claim and claim event commit."""

    operation_id: str
    task_id: str
    holder_id: str
    acquired_at: datetime
    expires_at: datetime
    event: RuntimeEvent

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "operation_id")
        _required_text(self.task_id, "task_id")
        _required_text(self.holder_id, "holder_id")
        acquired_at = _aware_datetime(self.acquired_at, "acquired_at")
        expires_at = _aware_datetime(self.expires_at, "expires_at")
        if expires_at <= acquired_at:
            raise ValueError("expires_at must be after acquired_at")

        if not isinstance(self.event, RuntimeEvent):
            raise TypeError("event must be a RuntimeEvent")
        if self.event.operation_id != self.operation_id:
            raise ValueError("event operation_id must match claim operation_id")
        if self.event.task_id != self.task_id:
            raise ValueError("event task_id must match claim task_id")
        if self.event.type != "task.claimed":
            raise ValueError("event type must be task.claimed")


@dataclass(frozen=True, slots=True)
class TaskLeaseGuard:
    """Strict proof required to mutate state owned by one live lease fence."""

    operation_id: str
    task_id: str
    holder_id: str
    attempt: int
    fencing_token: int
    checked_at: datetime

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "operation_id")
        _required_text(self.task_id, "task_id")
        _required_text(self.holder_id, "holder_id")
        _positive_integer(self.attempt, "attempt")
        _positive_integer(self.fencing_token, "fencing_token")
        _aware_datetime(self.checked_at, "checked_at")
