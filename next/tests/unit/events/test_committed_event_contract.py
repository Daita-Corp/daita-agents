from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from typing import cast

import pytest

from daita.events.models import CommittedEvent, EventCursor, RuntimeEvent
from daita.events.protocols import (
    CommittedEventReader,
    EventCursorMismatchError,
    EventCursorNotFoundError,
    EventReadError,
)

NOW = datetime(2026, 7, 17, 16, 0, tzinfo=timezone.utc)


def _event(*, agent_id: str = "agent-1") -> RuntimeEvent:
    return RuntimeEvent(
        id="event-1",
        type="operation.created",
        agent_id=agent_id,
        operation_id="operation-1",
        created_at=NOW,
        payload={"source": "operation-commit"},
    )


def test_event_cursor_is_an_immutable_positive_agent_bound_position() -> None:
    cursor = EventCursor(agent_id="agent-1", sequence=1)

    assert cursor.agent_id == "agent-1"
    assert cursor.sequence == 1
    with pytest.raises(FrozenInstanceError):
        setattr(cursor, "sequence", 2)

    for agent_id in ("", " "):
        with pytest.raises(ValueError, match="agent_id.*non-empty"):
            EventCursor(agent_id=agent_id, sequence=1)
    with pytest.raises((TypeError, ValueError), match="agent_id"):
        EventCursor(agent_id=cast(str, object()), sequence=1)

    for sequence in (0, -1, True):
        with pytest.raises(ValueError, match="sequence.*positive integer"):
            EventCursor(agent_id="agent-1", sequence=sequence)
    with pytest.raises((TypeError, ValueError), match="sequence.*positive integer"):
        EventCursor(agent_id="agent-1", sequence=cast(int, object()))


def test_committed_event_binds_the_cursor_to_the_exact_event_agent() -> None:
    cursor = EventCursor(agent_id="agent-1", sequence=7)
    event = _event()

    committed = CommittedEvent(cursor=cursor, event=event)

    assert committed.cursor is cursor
    assert committed.event is event
    assert committed.event.created_at == NOW

    with pytest.raises(ValueError, match="cursor.*agent|agent.*cursor"):
        CommittedEvent(
            cursor=replace(cursor, agent_id="agent-2"),
            event=event,
        )


def test_committed_event_requires_canonical_record_types() -> None:
    cursor = EventCursor(agent_id="agent-1", sequence=1)
    event = _event()

    with pytest.raises(TypeError, match="cursor.*EventCursor"):
        CommittedEvent(cursor=cast(EventCursor, object()), event=event)
    with pytest.raises(TypeError, match="event.*RuntimeEvent"):
        CommittedEvent(cursor=cursor, event=cast(RuntimeEvent, object()))

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(event, created_at=NOW.replace(tzinfo=None))


def test_committed_event_reader_is_bounded_read_only_and_subscribable() -> None:
    public_names = {
        name for name in CommittedEventReader.__dict__ if not name.startswith("_")
    }

    assert public_names == {"read_after", "subscribe"}
    assert not hasattr(CommittedEventReader, "append")
    assert inspect.iscoroutinefunction(CommittedEventReader.read_after)
    assert not inspect.iscoroutinefunction(CommittedEventReader.subscribe)

    read_signature = inspect.signature(CommittedEventReader.read_after)
    assert tuple(read_signature.parameters) == (
        "self",
        "agent_id",
        "cursor",
        "limit",
    )
    assert read_signature.parameters["cursor"].default is inspect.Parameter.empty
    assert read_signature.parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert read_signature.parameters["limit"].default is inspect.Parameter.empty

    subscribe_signature = inspect.signature(CommittedEventReader.subscribe)
    assert tuple(subscribe_signature.parameters) == (
        "self",
        "agent_id",
        "cursor",
    )
    assert subscribe_signature.parameters["cursor"].default is inspect.Parameter.empty


def test_cross_agent_cursor_error_preserves_machine_readable_scope_facts() -> None:
    cursor = EventCursor(agent_id="agent-2", sequence=9)

    error = EventCursorMismatchError(
        requested_agent_id="agent-1",
        cursor=cursor,
    )

    assert isinstance(error, EventReadError)
    assert error.requested_agent_id == "agent-1"
    assert error.cursor is cursor
    assert error.cursor.agent_id == "agent-2"
    assert error.cursor.sequence == 9
    assert "agent-1" in str(error)
    assert "agent-2" in str(error)


def test_unknown_cursor_error_preserves_the_exact_replay_position() -> None:
    cursor = EventCursor(agent_id="agent-1", sequence=99)

    error = EventCursorNotFoundError(cursor)

    assert isinstance(error, EventReadError)
    assert error.cursor is cursor
    assert "agent-1" in str(error)
    assert "99" in str(error)
