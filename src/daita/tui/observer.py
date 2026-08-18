"""Best-effort observer adapter that posts bounded Textual messages."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from textual.message import Message

from daita import AgentEvent, AgentEventKind

from .models import MAX_QUEUED_EVENTS
from .sanitization import sanitize_terminal_text

if TYPE_CHECKING:
    from textual.app import App


class ObserverEvent(Message):
    """One sanitized, process-local observation for the chat surface."""

    def __init__(self, event: AgentEvent) -> None:
        super().__init__()
        self.event = event


class RunObserver:
    """Non-directive observer that never directs Agent.run()."""

    def __init__(self, app: App[Any]) -> None:
        self._app = app
        self._queue: deque[AgentEvent] = deque()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self._queue.clear()

    def __call__(self, event: AgentEvent) -> None:
        if self.closed:
            return
        try:
            if len(self._queue) >= MAX_QUEUED_EVENTS:
                dropped = False
                for index, queued in enumerate(self._queue):
                    if queued.kind is AgentEventKind.MODEL_TEXT_DELTA:
                        del self._queue[index]
                        dropped = True
                        break
                if not dropped:
                    self._queue.popleft()
            self._queue.append(event)
            self._app.post_message(ObserverEvent(event))
        except Exception:
            return

    def drain(self) -> tuple[AgentEvent, ...]:
        events = tuple(self._queue)
        self._queue.clear()
        return events


def event_text(value: object, *, maximum: int, fallback: str) -> str:
    return sanitize_terminal_text(
        value,
        maximum=maximum,
        preserve_lines=False,
        fallback=fallback,
    )
