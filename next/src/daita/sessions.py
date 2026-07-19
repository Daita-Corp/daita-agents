"""Canonical durable session records and their narrow repository contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Protocol

from .llm.models import CanonicalMessage

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_CHECKPOINT_REFERENCES = 1_024


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    agent_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _required_text(self.id, "session id")
        _required_text(self.agent_id, "session agent_id")
        _required_text(self.title, "session title")
        if len(self.title) > 256:
            raise ValueError("session title must contain at most 256 characters")
        for value, name in (
            (self.created_at, "session created_at"),
            (self.updated_at, "session updated_at"),
        ):
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ValueError(f"{name} must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("session updated_at cannot precede created_at")


@dataclass(frozen=True, slots=True)
class SessionTranscript:
    session: Session
    operation_ids: tuple[str, ...]
    messages: tuple[CanonicalMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session, Session):
            raise TypeError("transcript session must be a Session record")
        operation_ids = tuple(self.operation_ids)
        messages = tuple(self.messages)
        if any(
            not isinstance(value, str) or not value.strip() for value in operation_ids
        ):
            raise ValueError("transcript operation IDs must be non-empty strings")
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("transcript operation IDs must be unique")
        if any(not isinstance(message, CanonicalMessage) for message in messages):
            raise TypeError("transcript messages must be canonical messages")
        if any(message.agent_id != self.session.agent_id for message in messages):
            raise ValueError("transcript messages must belong to its agent")
        if any(message.session_id != self.session.id for message in messages):
            raise ValueError("transcript messages must belong to its session")
        if any(message.operation_id not in operation_ids for message in messages):
            raise ValueError("transcript messages must belong to a listed operation")
        object.__setattr__(self, "operation_ids", operation_ids)
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class SessionCompressionCheckpoint:
    """Versioned extractive summary of an immutable session-history prefix."""

    id: str
    agent_id: str
    session_id: str
    version: int
    through_position: int
    through_operation_id: str
    source_fingerprint: str
    summary: str
    operation_ids: tuple[str, ...]
    created_at: datetime
    evidence_ids: tuple[str, ...] = ()
    approval_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.id, "compression checkpoint id"),
            (self.agent_id, "compression checkpoint agent_id"),
            (self.session_id, "compression checkpoint session_id"),
            (self.through_operation_id, "compression checkpoint operation id"),
            (self.summary, "compression checkpoint summary"),
        ):
            _required_text(value, name)
        if len(self.summary) > 32_768:
            raise ValueError("compression checkpoint summary exceeds 32768 characters")
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise ValueError("compression checkpoint version must be positive")
        if (
            not isinstance(self.through_position, int)
            or isinstance(self.through_position, bool)
            or self.through_position < 0
        ):
            raise ValueError("compression checkpoint position cannot be negative")
        if (
            not isinstance(self.source_fingerprint, str)
            or _SHA256.fullmatch(self.source_fingerprint) is None
        ):
            raise ValueError("compression checkpoint fingerprint must use sha256")
        normalized: dict[str, tuple[str, ...]] = {}
        for name, values in (
            ("operation_ids", self.operation_ids),
            ("evidence_ids", self.evidence_ids),
            ("approval_ids", self.approval_ids),
            ("resource_ids", self.resource_ids),
        ):
            items = tuple(values)
            if len(items) > _MAX_CHECKPOINT_REFERENCES:
                raise ValueError(f"compression checkpoint {name} exceeds its bound")
            if any(not isinstance(item, str) or not item.strip() for item in items):
                raise ValueError(
                    f"compression checkpoint {name} must contain non-empty strings"
                )
            if len(items) != len(set(items)):
                raise ValueError(f"compression checkpoint {name} contains duplicates")
            normalized[name] = items
        if not normalized["operation_ids"] or (
            self.through_operation_id not in normalized["operation_ids"]
        ):
            raise ValueError(
                "compression checkpoint frontier must be a summarized operation"
            )
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("compression checkpoint created_at must be timezone-aware")
        for name, items in normalized.items():
            object.__setattr__(self, name, items)


class SessionAlreadyExistsError(RuntimeError):
    """Raised when a session identity is already claimed."""


class SessionNotFoundError(RuntimeError):
    """Raised when a session is unknown to its agent."""


class SessionStore(Protocol):
    async def create_session(self, session: Session) -> Session: ...

    async def load_session(
        self, agent_id: str, session_id: str
    ) -> SessionTranscript | None: ...


class SessionCompressionStore(Protocol):
    async def load_session_compression(
        self,
        agent_id: str,
        session_id: str,
    ) -> SessionCompressionCheckpoint | None: ...

    async def commit_session_compression(
        self,
        checkpoint: SessionCompressionCheckpoint,
        *,
        expected_version: int,
    ) -> SessionCompressionCheckpoint: ...
