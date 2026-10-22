"""Keep a bounded, sequenced in-memory event history per agent and fan it out."""

from __future__ import annotations

import asyncio
import bisect
import json
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field, replace

from .._json import FrozenJsonObject
from ..observation import AgentEvent, AgentEventKind
from .protocol import EventType, event_envelope

MAX_EVENTS = 20_000
MAX_AGE_SECONDS = 15 * 60
# AgentEvent bounds every data string to this many characters.
_MAX_DELTA_TEXT = 1_024


@dataclass(frozen=True, slots=True)
class _Record:
    seq: int
    event: str
    text: str
    at: float
    delta: AgentEvent | None = None


@dataclass(slots=True)
class _Stream:
    head: int
    # A cursor below the floor missed a dropped record.
    floor: int
    records: deque[_Record] = field(default_factory=deque)
    # Sorted [low, high) cursors that would see merged text twice.
    gaps: list[tuple[int, int]] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    subscribers: int = 0

    def lost(self, cursor: int) -> bool:
        if cursor < self.floor or cursor > self.head:
            return True
        index = bisect.bisect_right(self.gaps, (cursor, self.head + 1)) - 1
        return index >= 0 and self.gaps[index][0] <= cursor < self.gaps[index][1]


class EventBus:
    """Per-agent ring buffers; events are never durable (§7.5)."""

    def __init__(
        self,
        *,
        max_events: int = MAX_EVENTS,
        max_age_seconds: float = MAX_AGE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_events = max_events
        self._max_age = max_age_seconds
        self._clock = clock
        self._loop = asyncio.get_running_loop()
        # Sequences start at the wall clock so a restarted host never reuses a
        # sequence a client saw from the previous one.
        self._start = time.time_ns() // 1_000
        self._streams: dict[str, _Stream] = {}
        self._closed = False

    def publish(self, agent: str, event: EventType, data: object) -> None:
        """Record one event; safe to call from the observer of any thread."""

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not self._loop:
            self._loop.call_soon_threadsafe(self.publish, agent, event, data)
            return
        stream = self._stream(agent)
        stream.head += 1
        delta = (
            data
            if isinstance(data, AgentEvent)
            and data.kind is AgentEventKind.MODEL_TEXT_DELTA
            else None
        )
        now = self._clock()
        records = stream.records
        while records and now - records[0].at > self._max_age:
            stream.floor = records.popleft().seq
        if len(records) >= self._max_events:
            self._relieve(stream)
        records.append(_record(stream.head, EventType(event), data, now, delta))
        stream.changed.set()
        stream.changed = asyncio.Event()

    def head(self, agent: str) -> int:
        return self._stream(agent).head

    def attached(self, agent: str) -> int:
        stream = self._streams.get(agent)
        return 0 if stream is None else stream.subscribers

    def buffered_bytes(self, agent: str) -> int:
        """Return the encoded size of one agent's retained history."""

        return sum(len(record.text) for record in self._stream(agent).records)

    def subscribe(
        self, agent: str, after: int | None
    ) -> AsyncGenerator[tuple[int, str, str], None]:
        """Yield (seq, event, JSON record) after one sequence or a reset, then live.

        The starting point is fixed now, so nothing published before the first
        iteration is missed.
        """

        stream = self._stream(agent)
        cursor = stream.head if after is None else after
        if stream.lost(cursor):
            return self._follow(stream, stream.head, reset=True)
        return self._follow(stream, cursor, reset=False)

    async def _follow(
        self, stream: _Stream, cursor: int, *, reset: bool
    ) -> AsyncGenerator[tuple[int, str, str], None]:
        stream.subscribers += 1
        try:
            if reset:
                yield _reset(cursor)
            while not self._closed:
                if stream.lost(cursor):
                    cursor = stream.head
                    yield _reset(cursor)
                pending: list[_Record] = []
                for record in reversed(stream.records):
                    if record.seq <= cursor:
                        break
                    pending.append(record)
                waiter = stream.changed
                for record in reversed(pending):
                    if stream.lost(cursor):
                        break
                    cursor = record.seq
                    yield record.seq, record.event, record.text
                if not pending:
                    await waiter.wait()
        finally:
            stream.subscribers -= 1

    def close(self) -> None:
        """End every subscription; later publishes are still buffered."""

        self._closed = True
        for stream in self._streams.values():
            stream.changed.set()

    def _stream(self, agent: str) -> _Stream:
        stream = self._streams.get(agent)
        if stream is None:
            stream = self._streams[agent] = _Stream(self._start, self._start)
        return stream

    def _relieve(self, stream: _Stream) -> None:
        """Free one slot: merge or drop the oldest text delta, else the oldest event."""

        # shortcut: linear scan from the oldest record; deltas dominate, so the
        # first one is near the front. Index deltas separately if that changes.
        records = stream.records
        for index, record in enumerate(records):
            if record.delta is None:
                continue
            following = records[index + 1] if index + 1 < len(records) else None
            merged = _merged(record, following)
            del records[index]
            if merged is None or following is None:
                stream.floor = max(stream.floor, record.seq)
            else:
                records[index] = merged
                _add_gap(stream, record.seq, following.seq)
            break
        else:
            stream.floor = records.popleft().seq
        while stream.gaps and stream.gaps[0][1] <= stream.floor:
            stream.gaps.pop(0)


def _reset(cursor: int) -> tuple[int, str, str]:
    record = event_envelope(cursor, EventType.RESET, None)
    return cursor, EventType.RESET.value, json.dumps(record)


def _record(
    seq: int, event: EventType, data: object, at: float, delta: AgentEvent | None
) -> _Record:
    text = json.dumps(event_envelope(seq, event, data), separators=(",", ":"))
    return _Record(seq, event.value, text, at, delta)


def _merged(record: _Record, following: _Record | None) -> _Record | None:
    """Prepend one delta's text to the adjacent delta of the same model call."""

    first, second = record.delta, None if following is None else following.delta
    if first is None or second is None or following is None:
        return None
    if first.run_id != second.run_id or first.data.get(
        "model_call_index"
    ) != second.data.get("model_call_index"):
        return None
    text = f"{first.data['text']}{second.data['text']}"
    if len(text) > _MAX_DELTA_TEXT:
        return None
    data = FrozenJsonObject.from_mapping({**second.data.to_dict(), "text": text})
    combined = replace(second, data=data)
    return _record(
        following.seq, EventType.AGENT_EVENT, combined, following.at, combined
    )


def _add_gap(stream: _Stream, low: int, high: int) -> None:
    gaps = stream.gaps
    index = bisect.bisect_left(gaps, (low, low))
    # Successive merges into one record extend a single gap.
    if index > 0 and gaps[index - 1][1] == low:
        gaps[index - 1] = (gaps[index - 1][0], high)
    else:
        gaps.insert(index, (low, high))


__all__ = ["MAX_AGE_SECONDS", "MAX_EVENTS", "EventBus"]
