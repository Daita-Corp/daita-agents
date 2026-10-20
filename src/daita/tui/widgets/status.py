"""Display compact identity, context-window, execution, and notification status."""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Tabs

from ..projection import format_token_count
from ..sanitization import sanitize_terminal_text


def context_window_text(used: int | None, total: int | None) -> Text:
    """Build a truthful compact view of the latest exact request usage."""

    if total is None or total <= 0 or used is None or used <= 0:
        # Nothing measured yet; an empty label hides the indicator.
        return Text()
    total_label = format_token_count(total)
    percentage = used / total * 100
    style = "red" if percentage >= 90 else "yellow" if percentage >= 75 else "dim"
    return Text(
        f"ctx {format_token_count(used)} / {total_label}",
        style=style,
    )


class StatusBar(Horizontal):
    """One header row with identity, caller-owned navigation, and context usage."""

    def __init__(self, navigation: Tabs | None = None) -> None:
        super().__init__(id="status-bar")
        self._navigation = navigation

    def compose(self) -> ComposeResult:
        yield Static(Text("DAITA", style="bold"), id="status-primary")
        if self._navigation is not None:
            yield self._navigation
        host = Static(Text("● host open", style="green"), id="host-status")
        host.tooltip = (
            "Routines and jobs run only while a Daita host is open: this window, "
            "or `daita host --agent <name>` after you close it. With no host open "
            "they wait."
        )
        host.display = False
        yield host
        context = Static(context_window_text(None, None), id="context-window")
        context.tooltip = (
            "Exact input tokens in the latest model request / model context window"
        )
        yield context

    def update_status(
        self,
        *,
        agent: str,
        model: str,
        source: str,
        state: str,
        context_used: int | None = None,
        context_total: int | None = None,
        scheduled: int = 0,
        running: int = 0,
        needs_you: int = 0,
        active_reports: int = 0,
        source_count: int = 0,
        too_small: bool = False,
    ) -> None:
        primary = self.query_one("#status-primary", Static)
        context = self.query_one("#context-window", Static)
        host = self.query_one("#host-status", Static)
        if too_small:
            host.display = False
            message = Text("DAITA", style="bold")
            message.append("  Terminal too small; 32x8 required", style="bold")
            primary.update(message)
            primary.tooltip = message.copy()
            context.display = False
            return
        label = context_window_text(context_used, context_total)
        context.update(label)
        context.display = bool(label) and self.app.size.width >= 100
        host.display = bool(scheduled or running) and self.app.size.width >= 80
        safe_source = sanitize_terminal_text(
            source, maximum=48, preserve_lines=False, fallback="none"
        )
        self.tooltip = (
            f"{source_count} sources · {scheduled} scheduled · {running} running · "
            f"{needs_you} need you. Sources: {safe_source}"
        )
        safe_agent = sanitize_terminal_text(
            agent, maximum=32, preserve_lines=False, fallback="agent"
        )
        safe_model = sanitize_terminal_text(
            model, maximum=48, preserve_lines=False, fallback="model"
        )
        safe_state = sanitize_terminal_text(
            state, maximum=32, preserve_lines=False, fallback="ready"
        )
        state_style = "bold yellow" if "approval" in safe_state else "dim"
        message = Text("DAITA", style="bold")
        message.append(f"  {safe_agent}")
        message.append(f" · {safe_model}", style="dim")
        # Foreground phases belong in the activity line; idle needs no badge.
        if safe_state in {
            "approval",
            "edit approval",
            "queued behind background",
            "background running",
        }:
            message.append(f"  [{safe_state}]", style=state_style)
        if active_reports:
            message.append(f"  reporting {active_reports}", style="dim")
        primary.update(message)
        primary.tooltip = message.copy()


class ActivityBar(Static):
    """Animated run activity with a short timeline of the run's tool steps.

    Private model chain-of-thought is never exposed; only executed steps are.
    """

    _FRAMES = ("◆", "◇")
    _VISIBLE_STEPS = 4

    def __init__(self) -> None:
        super().__init__("", id="activity-bar", markup=False)
        self._stage = "Thinking"
        self._started_at: float | None = None
        self._frame = 0
        self._timer: object | None = None
        # call_id -> (label, started_at, (mark, seconds) once finished).
        self._steps: dict[str, tuple[str, float, tuple[str, float] | None]] = {}
        self.display = False
        self.tooltip = (
            "Live execution activity. Private model chain-of-thought is not exposed."
        )

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.25, self._tick, pause=True)

    def start(self, stage: str = "Thinking") -> None:
        self._started_at = time.monotonic()
        self._frame = 0
        self._steps = {}
        self.display = True
        self.update_stage(stage)
        timer = self._timer
        if timer is not None:
            timer.resume()  # type: ignore[attr-defined]

    def update_stage(self, stage: str) -> None:
        self._stage = sanitize_terminal_text(
            stage, maximum=72, preserve_lines=False, fallback="Working"
        )
        self._render_activity()

    def step_started(self, call_id: str, label: str) -> None:
        self._steps[call_id] = (
            sanitize_terminal_text(
                label, maximum=72, preserve_lines=False, fallback="Tool"
            ),
            time.monotonic(),
            None,
        )
        self._render_activity()

    def step_finished(
        self, call_id: str, *, success: bool, duration_ms: object = None
    ) -> None:
        step = self._steps.get(call_id)
        if step is None:
            return
        seconds = (
            duration_ms / 1000
            if isinstance(duration_ms, (int, float))
            and not isinstance(duration_ms, bool)
            else time.monotonic() - step[1]
        )
        self._steps[call_id] = (
            step[0],
            step[1],
            ("✓" if success else "✗", max(0.0, seconds)),
        )
        self._render_activity()

    def stop(self) -> None:
        timer = self._timer
        if timer is not None:
            timer.pause()  # type: ignore[attr-defined]
        self._started_at = None
        self._steps = {}
        self.update("")
        self.display = False

    def _tick(self) -> None:
        if self._started_at is None:
            return
        self._frame = (self._frame + 1) % len(self._FRAMES)
        self._render_activity()

    def _render_activity(self) -> None:
        content = Text()
        finished = [step for step in self._steps.values() if step[2] is not None]
        hidden = len(finished) - self._VISIBLE_STEPS
        if hidden > 0:
            content.append(f"  … {hidden} earlier steps\n", style="dim")
        for label, _started, outcome in finished[-self._VISIBLE_STEPS :]:
            assert outcome is not None
            mark, seconds = outcome
            content.append(f"  {mark} ", style="green" if mark == "✓" else "red")
            content.append(f"{label}  {seconds:.1f}s\n", style="dim")
        elapsed = 0.0
        if self._started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._started_at)
        content.append(self._FRAMES[self._frame], style="bold")
        content.append(f"  {self._stage}", style="italic dim")
        content.append(f"  {elapsed:.1f}s", style="dim")
        self.update(content)


class NoticeBar(Static):
    def __init__(self) -> None:
        super().__init__("", id="notice-bar", markup=False)
        self.display = False

    def show(self, message: str) -> None:
        text = sanitize_terminal_text(
            message,
            maximum=2_048,
            preserve_lines=True,
            fallback="",
        )
        self.update(text)
        self.display = bool(text)

    def clear(self) -> None:
        self.update("")
        self.display = False


__all__ = [
    "ActivityBar",
    "NoticeBar",
    "StatusBar",
    "context_window_text",
    "format_token_count",
]
