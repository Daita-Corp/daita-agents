"""Portable read-only access to durably committed runtime events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .models import CommittedEvent, EventCursor


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class EventReadError(RuntimeError):
    """Base class for portable committed-event read failures."""


class EventCursorMismatchError(EventReadError):
    """Raised when an agent attempts to consume another agent's cursor."""

    def __init__(
        self,
        *,
        requested_agent_id: str,
        cursor: EventCursor,
    ) -> None:
        _required_text(requested_agent_id, "requested event agent_id")
        if not isinstance(cursor, EventCursor):
            raise TypeError("event cursor mismatch requires an EventCursor")
        self.requested_agent_id = requested_agent_id
        self.cursor = cursor
        super().__init__(
            f"event cursor for {cursor.agent_id} cannot read events for "
            f"{requested_agent_id}"
        )


class EventCursorNotFoundError(EventReadError):
    """Raised when a supplied cursor is not in committed history."""

    def __init__(self, cursor: EventCursor) -> None:
        if not isinstance(cursor, EventCursor):
            raise TypeError("missing event cursor must be an EventCursor")
        self.cursor = cursor
        super().__init__(
            f"event cursor does not exist for {cursor.agent_id}: {cursor.sequence}"
        )


class CommittedEventReader(Protocol):
    """Replay and follow committed events without exposing an append API."""

    async def latest_cursor(self, agent_id: str) -> EventCursor | None: ...

    async def read_after(
        self,
        agent_id: str,
        cursor: EventCursor | None,
        *,
        limit: int,
    ) -> tuple[CommittedEvent, ...]: ...

    def subscribe(
        self,
        agent_id: str,
        cursor: EventCursor | None,
    ) -> AsyncIterator[CommittedEvent]: ...
