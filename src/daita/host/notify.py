"""Send desktop notifications when nobody watches an agent (§11).

Text is fixed per kind and names only the agent: never tool arguments, query
text, data values, routine titles or secrets. At most one notification per
agent per interval; later items wait and leave together ("3 items need you").
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..tui.sanitization import sanitize_terminal_text
from ..work_board import WorkBoard, WorkColumn
from .interactions import InteractionKind

_LOGGER = logging.getLogger("daita.host")
INTERVAL_SECONDS = 30.0
TITLE = "Daita"
Sender = Callable[[str, str], None]
_INTERACTION_TEXT = {
    InteractionKind.TOOL_APPROVAL: "{agent} needs your approval",
    InteractionKind.ROUTINE_CONFIRMATION: "{agent} needs you to confirm a routine",
    InteractionKind.SOURCE_EDIT_CONFIRMATION: (
        "{agent} needs you to confirm a source change"
    ),
}
_WORK_TEXT = {
    WorkColumn.NEEDS_YOU: "{agent}: an item needs you",
    WorkColumn.DONE: "{agent}: a result is ready",
}


@dataclass(slots=True)
class _Agent:
    next_at: float = 0.0
    pending: list[str] = field(default_factory=list)
    # Work cards already reported, and the time from which a card is new.
    seen: set[tuple[str, str, WorkColumn]] = field(default_factory=set)
    since: datetime | None = None


class Notifier:
    """Notify for new questions and new Needs you / Done cards, rate limited."""

    def __init__(
        self,
        *,
        attached: Callable[[str], int],
        enabled: Callable[[], bool],
        send: Sender | None,
        interval: float = INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._attached = attached
        self._enabled = enabled
        self._send = send
        self._interval = interval
        self._clock = clock
        self._now = now
        self._started = now()
        self._agents: dict[str, _Agent] = {}

    def interaction(self, agent: str, kind: InteractionKind) -> None:
        text = _INTERACTION_TEXT.get(kind)
        if text is not None:
            self._post(agent, (text.format(agent=agent),))

    def work(self, agent: str, board: WorkBoard) -> None:
        """Report cards that entered Needs you or Done since the last look."""

        state = self._state(agent)
        current = {
            (card.kind.value, card.id, column): card.updated_at
            for column in _WORK_TEXT
            for card in board.columns.get(column, ())
        }
        since = state.since or self._started
        fresh = [
            _WORK_TEXT[key[2]].format(agent=agent)
            for key, updated_at in current.items()
            if key not in state.seen and updated_at >= since
        ]
        state.seen = set(current)
        self._post(agent, fresh)

    def watched(self, agent: str) -> bool:
        """True while a client watches; what it sees then is never reported later."""

        if not self._attached(agent):
            return False
        state = self._state(agent)
        state.since = self._now()
        state.pending.clear()
        return True

    def flush(self) -> None:
        """Send what waited for an agent's interval, as one notification."""

        for agent, state in self._agents.items():
            if state.pending and self._clock() >= state.next_at:
                pending, state.pending = state.pending, []
                self._post(agent, pending)

    def _post(self, agent: str, texts: Iterable[str]) -> None:
        items = list(texts)
        if not items or self.watched(agent) or not self._is_enabled():
            return
        state = self._state(agent)
        now = self._clock()
        if now < state.next_at:
            state.pending.extend(items)
            return
        state.next_at = now + self._interval
        text = items[0] if len(items) == 1 else f"{agent}: {len(items)} items need you"
        self._deliver(text)

    def _deliver(self, text: str) -> None:
        safe = sanitize_terminal_text(
            text, maximum=200, preserve_lines=False, fallback="Daita needs you"
        )
        _LOGGER.info("notification: %s", safe)
        if self._send is not None:
            asyncio.get_running_loop().run_in_executor(None, self._safe_send, safe)

    def _safe_send(self, text: str) -> None:
        assert self._send is not None
        try:
            self._send(TITLE, text)
        except Exception as error:
            _LOGGER.warning("notification failed: %s", type(error).__name__)

    def _is_enabled(self) -> bool:
        try:
            return self._enabled()
        except (ValueError, OSError) as error:
            # A broken settings file must not override someone's off switch.
            _LOGGER.warning("notifications off: %s", error)
            return False

    def _state(self, agent: str) -> _Agent:
        return self._agents.setdefault(agent, _Agent())


def desktop_sender() -> Sender | None:
    """This platform's notifier, or None to log only."""

    if sys.platform == "darwin":
        return _osascript
    if shutil.which("notify-send") is not None:
        return _notify_send
    return None


def _osascript(title: str, text: str) -> None:
    # The text travels as an argument, never as part of the script.
    script = (
        "on run argv",
        "display notification (item 2 of argv) with title (item 1 of argv)",
        "end run",
    )
    command = ["osascript", *(part for line in script for part in ("-e", line))]
    subprocess.run([*command, title, text], capture_output=True, timeout=10)


def _notify_send(title: str, text: str) -> None:
    subprocess.run(["notify-send", title, text], capture_output=True, timeout=10)


__all__ = ["INTERVAL_SECONDS", "Notifier", "Sender", "desktop_sender"]
