"""SQLite-owned durable records shared with the database-write executor."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256

from .._json import canonical_json

_DATABASE_WRITE_RECEIPT_ID = re.compile(r"database-write-receipt:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")


class DatabaseWriteOutcome(str, Enum):
    STARTED = "started"
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class DatabaseWriteReceiptConflictError(RuntimeError):
    """The durable receipt identity or immutable terminal state conflicts."""


def database_write_text(value: str, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty text without surrounding space")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def database_write_aware(value: datetime, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")
    return value


def database_write_receipt_id(
    *,
    agent_id: str,
    run_id: str,
    call_id: str,
    capability_id: str,
    intent_sha256: str,
) -> str:
    identity = {
        "agent_id": database_write_text(agent_id, "receipt agent_id"),
        "call_id": database_write_text(call_id, "receipt call_id"),
        "capability_id": database_write_text(
            capability_id, "receipt capability_id", maximum=128
        ),
        "intent_sha256": intent_sha256,
        "run_id": database_write_text(run_id, "receipt run_id"),
    }
    if (
        not isinstance(intent_sha256, str)
        or _DATABASE_WRITE_HASH.fullmatch(intent_sha256) is None
    ):
        raise ValueError("receipt intent_sha256 must be a sha256 hash")
    digest = sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return f"database-write-receipt:sha256:{digest}"


def validate_database_write_receipt_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or _DATABASE_WRITE_RECEIPT_ID.fullmatch(value) is None
    ):
        raise ValueError("receipt_id must be a canonical database-write receipt id")
    return value


@dataclass(frozen=True, slots=True)
class DatabaseWriteReceipt:
    """Bounded durable metadata for one exact external database-write attempt."""

    receipt_id: str
    agent_id: str
    run_id: str
    call_id: str
    capability_id: str
    source_id: str
    resource_id: str
    intent_sha256: str
    preview_fingerprint: str
    outcome: DatabaseWriteOutcome
    affected_rows: int | None
    normalized_error_code: str | None
    started_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        validate_database_write_receipt_id(self.receipt_id)
        database_write_text(self.agent_id, "receipt agent_id")
        database_write_text(self.run_id, "receipt run_id")
        database_write_text(self.call_id, "receipt call_id")
        database_write_text(self.capability_id, "receipt capability_id", maximum=128)
        if (
            not isinstance(self.source_id, str)
            or _DATABASE_WRITE_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("receipt source_id must be a canonical source id")
        if (
            not isinstance(self.resource_id, str)
            or _DATABASE_WRITE_RESOURCE_ID.fullmatch(self.resource_id) is None
        ):
            raise ValueError("receipt resource_id must be a canonical resource id")
        for value, name in (
            (self.intent_sha256, "intent_sha256"),
            (self.preview_fingerprint, "preview_fingerprint"),
        ):
            if (
                not isinstance(value, str)
                or _DATABASE_WRITE_HASH.fullmatch(value) is None
            ):
                raise ValueError(f"receipt {name} must be a sha256 hash")
        if not isinstance(self.outcome, DatabaseWriteOutcome):
            raise TypeError("receipt outcome must be a DatabaseWriteOutcome")
        database_write_aware(self.started_at, "receipt started_at")
        if self.completed_at is not None:
            database_write_aware(self.completed_at, "receipt completed_at")
            if self.completed_at < self.started_at:
                raise ValueError("receipt cannot complete before it starts")
        expected_id = database_write_receipt_id(
            agent_id=self.agent_id,
            run_id=self.run_id,
            call_id=self.call_id,
            capability_id=self.capability_id,
            intent_sha256=self.intent_sha256,
        )
        if self.receipt_id != expected_id:
            raise ValueError("receipt_id does not match its execution identity")
        if self.normalized_error_code is not None and (
            not isinstance(self.normalized_error_code, str)
            or _DATABASE_WRITE_ERROR_CODE.fullmatch(self.normalized_error_code) is None
        ):
            raise ValueError("receipt normalized_error_code is invalid")
        if self.affected_rows is not None and (
            not isinstance(self.affected_rows, int)
            or isinstance(self.affected_rows, bool)
        ):
            raise TypeError("receipt affected_rows must be an integer or None")
        if self.outcome is DatabaseWriteOutcome.STARTED:
            if any(
                value is not None
                for value in (
                    self.affected_rows,
                    self.normalized_error_code,
                    self.completed_at,
                )
            ):
                raise ValueError("started receipt cannot contain terminal fields")
        elif self.outcome is DatabaseWriteOutcome.COMMITTED:
            if (
                self.affected_rows != 1
                or self.normalized_error_code is not None
                or self.completed_at is None
            ):
                raise ValueError("committed receipt must record one affected row")
        elif self.outcome is DatabaseWriteOutcome.NOT_COMMITTED:
            if (
                self.affected_rows != 0
                or self.normalized_error_code is None
                or self.completed_at is None
            ):
                raise ValueError(
                    "not_committed receipt must record zero rows and an error code"
                )
        elif (
            self.affected_rows is not None
            or self.normalized_error_code != "write_outcome_unknown"
            or self.completed_at is None
        ):
            raise ValueError(
                "outcome_unknown receipt must omit affected rows and use its stable code"
            )

    @classmethod
    def start(
        cls,
        *,
        agent_id: str,
        run_id: str,
        call_id: str,
        capability_id: str,
        source_id: str,
        resource_id: str,
        intent_sha256: str,
        preview_fingerprint: str,
        started_at: datetime,
    ) -> DatabaseWriteReceipt:
        return cls(
            receipt_id=database_write_receipt_id(
                agent_id=agent_id,
                run_id=run_id,
                call_id=call_id,
                capability_id=capability_id,
                intent_sha256=intent_sha256,
            ),
            agent_id=agent_id,
            run_id=run_id,
            call_id=call_id,
            capability_id=capability_id,
            source_id=source_id,
            resource_id=resource_id,
            intent_sha256=intent_sha256,
            preview_fingerprint=preview_fingerprint,
            outcome=DatabaseWriteOutcome.STARTED,
            affected_rows=None,
            normalized_error_code=None,
            started_at=started_at,
            completed_at=None,
        )

    def finish(
        self,
        outcome: DatabaseWriteOutcome,
        *,
        completed_at: datetime,
        affected_rows: int | None,
        normalized_error_code: str | None,
    ) -> DatabaseWriteReceipt:
        if self.outcome is not DatabaseWriteOutcome.STARTED:
            raise ValueError("only a started receipt can reach a terminal outcome")
        if outcome is DatabaseWriteOutcome.STARTED:
            raise ValueError("receipt terminal outcome cannot be started")
        return replace(
            self,
            outcome=outcome,
            affected_rows=affected_rows,
            normalized_error_code=normalized_error_code,
            completed_at=completed_at,
        )

    def as_started(self) -> DatabaseWriteReceipt:
        return replace(
            self,
            outcome=DatabaseWriteOutcome.STARTED,
            affected_rows=None,
            normalized_error_code=None,
            completed_at=None,
        )


__all__ = [
    "DatabaseWriteOutcome",
    "DatabaseWriteReceipt",
    "DatabaseWriteReceiptConflictError",
    "database_write_aware",
    "database_write_receipt_id",
    "database_write_text",
    "validate_database_write_receipt_id",
]
