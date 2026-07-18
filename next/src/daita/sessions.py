"""Canonical durable session records and their narrow repository contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .llm.models import CanonicalMessage


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


class SessionAlreadyExistsError(RuntimeError):
    """Raised when a session identity is already claimed."""


class SessionNotFoundError(RuntimeError):
    """Raised when a session is unknown to its agent."""


class SessionStore(Protocol):
    async def create_session(self, session: Session) -> Session: ...

    async def load_session(
        self, agent_id: str, session_id: str
    ) -> SessionTranscript | None: ...
