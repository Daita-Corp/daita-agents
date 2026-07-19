"""Durable foreground-host inbox records and narrow store contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
import re
from typing import Protocol

from .._json import FrozenJsonObject, canonical_json

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_METHOD = re.compile(r"[a-z][a-z0-9_.-]*\Z")


def _identity(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a bounded stable identifier")
    return value


def _bounded_text(value: str, field_name: str, *, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be bounded non-empty text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8 text") from error
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{field_name} exceeds {maximum_bytes} UTF-8 bytes")
    return value


def _aware(value: datetime, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class HostInboxKind(str, Enum):
    TRIGGER = "trigger"
    APPROVAL_WAKE = "approval_wake"


class HostInboxStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"


def host_inbox_request_hash(
    *,
    kind: HostInboxKind,
    payload: Mapping[str, object],
    trigger_id: str | None = None,
) -> str:
    """Hash the exact immutable request identity stored by the host."""

    if not isinstance(kind, HostInboxKind):
        raise TypeError("host inbox kind must be HostInboxKind")
    if trigger_id is not None:
        _identity(trigger_id, "host inbox trigger_id")
    frozen = FrozenJsonObject.from_mapping(payload)
    material = canonical_json(
        {
            "kind": kind.value,
            "payload": frozen,
            "trigger_id": trigger_id,
        }
    )
    return "sha256:" + sha256(material.encode("utf-8")).hexdigest()


def host_mutation_request_hash(
    *,
    method: str,
    params: Mapping[str, object],
) -> str:
    """Hash one canonical host mutation before any mutable work begins."""

    _bounded_text(method, "host mutation method", maximum_bytes=128)
    if _METHOD.fullmatch(method) is None:
        raise ValueError("host mutation method has an invalid format")
    material = canonical_json(
        {
            "method": method,
            "params": FrozenJsonObject.from_mapping(params),
        }
    )
    return "sha256:" + sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HostMutationAdmission:
    """Immutable binding from one client key to one exact mutation request."""

    agent_id: str
    idempotency_key: str
    method: str
    request_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        _identity(self.agent_id, "host mutation agent_id")
        _bounded_text(
            self.idempotency_key,
            "host mutation idempotency_key",
            maximum_bytes=256,
        )
        _bounded_text(self.method, "host mutation method", maximum_bytes=128)
        if _METHOD.fullmatch(self.method) is None:
            raise ValueError("host mutation method has an invalid format")
        if (
            not isinstance(self.request_hash, str)
            or _HASH.fullmatch(self.request_hash) is None
        ):
            raise ValueError("host mutation request_hash must use canonical sha256")
        object.__setattr__(
            self,
            "created_at",
            _aware(self.created_at, "host mutation created_at"),
        )


@dataclass(frozen=True, slots=True)
class HostInboxItem:
    """One immutable request identity with a one-way completion projection."""

    id: str
    agent_id: str
    kind: HostInboxKind
    idempotency_key: str
    request_hash: str
    payload: Mapping[str, object]
    revision: int
    status: HostInboxStatus
    created_at: datetime
    updated_at: datetime
    trigger_id: str | None = None
    operation_id: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("host inbox id", self.id),
            ("host inbox agent_id", self.agent_id),
            ("host inbox idempotency_key", self.idempotency_key),
        ):
            _identity(value, field_name)
        if not isinstance(self.kind, HostInboxKind):
            raise TypeError("host inbox kind must be HostInboxKind")
        if not isinstance(self.status, HostInboxStatus):
            raise TypeError("host inbox status must be HostInboxStatus")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision not in {1, 2}
        ):
            raise ValueError("host inbox revision must be one or two")
        payload = FrozenJsonObject.from_mapping(self.payload)
        object.__setattr__(self, "payload", payload)
        expected_hash = host_inbox_request_hash(
            kind=self.kind,
            payload=payload,
            trigger_id=self.trigger_id,
        )
        if (
            not isinstance(self.request_hash, str)
            or _HASH.fullmatch(self.request_hash) is None
            or self.request_hash != expected_hash
        ):
            raise ValueError("host inbox request_hash does not match its request")
        created_at = _aware(self.created_at, "host inbox created_at")
        updated_at = _aware(self.updated_at, "host inbox updated_at")
        if updated_at < created_at:
            raise ValueError("host inbox updated_at cannot precede created_at")
        if self.trigger_id is not None:
            _identity(self.trigger_id, "host inbox trigger_id")
        if self.operation_id is not None:
            _identity(self.operation_id, "host inbox operation_id")
        if self.kind is HostInboxKind.TRIGGER and self.trigger_id is None:
            raise ValueError("trigger inbox item requires trigger_id")
        if self.kind is HostInboxKind.APPROVAL_WAKE and self.trigger_id is not None:
            raise ValueError("approval wake cannot carry trigger_id")
        if self.error is not None:
            if (
                not isinstance(self.error, str)
                or not self.error.strip()
                or len(self.error) > 2_000
            ):
                raise ValueError("host inbox error must be bounded non-empty text")
            object.__setattr__(self, "error", self.error.strip())
        if self.status is HostInboxStatus.PENDING:
            if (
                self.revision != 1
                or updated_at != created_at
                or self.operation_id is not None
                or self.error is not None
            ):
                raise ValueError("pending host inbox item has completion state")
        elif self.revision != 2:
            raise ValueError("completed host inbox item must have revision two")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)


class HostInboxStoreError(RuntimeError):
    """Base class for portable host-inbox persistence failures."""


class HostInboxNotFoundError(HostInboxStoreError):
    def __init__(self, agent_id: str, item_id: str) -> None:
        self.agent_id = agent_id
        self.item_id = item_id
        super().__init__(f"unknown host inbox item for {agent_id}: {item_id}")


class HostInboxEnqueueConflictError(HostInboxStoreError):
    def __init__(self, agent_id: str, idempotency_key: str) -> None:
        self.agent_id = agent_id
        self.idempotency_key = idempotency_key
        super().__init__(
            f"host inbox idempotency conflict for {agent_id}: {idempotency_key}"
        )


class HostInboxRevisionConflict(HostInboxStoreError):
    def __init__(
        self,
        item_id: str,
        *,
        expected_revision: int,
        actual_revision: int,
    ) -> None:
        self.item_id = item_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"host inbox {item_id} revision conflict: expected "
            f"{expected_revision}, found {actual_revision}"
        )


class HostMutationConflictError(HostInboxStoreError):
    def __init__(self, agent_id: str, idempotency_key: str) -> None:
        self.agent_id = agent_id
        self.idempotency_key = idempotency_key
        super().__init__(
            f"host mutation idempotency conflict for {agent_id}: {idempotency_key}"
        )


class HostInboxStore(Protocol):
    async def admit_host_mutation(
        self,
        request: HostMutationAdmission,
    ) -> HostMutationAdmission: ...

    async def enqueue_host_inbox(self, item: HostInboxItem) -> HostInboxItem: ...

    async def list_pending_host_inbox(
        self,
        agent_id: str,
        *,
        limit: int,
    ) -> tuple[HostInboxItem, ...]: ...

    async def complete_host_inbox(
        self,
        item: HostInboxItem,
        *,
        expected_revision: int,
    ) -> HostInboxItem: ...


__all__ = [
    "HostInboxEnqueueConflictError",
    "HostInboxItem",
    "HostInboxKind",
    "HostInboxNotFoundError",
    "HostInboxRevisionConflict",
    "HostInboxStatus",
    "HostInboxStore",
    "HostInboxStoreError",
    "HostMutationAdmission",
    "HostMutationConflictError",
    "host_inbox_request_hash",
    "host_mutation_request_hash",
]
