"""Display compact identity, context-window, execution, and notification status."""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from ..sanitization import sanitize_terminal_text


def format_token_count(value: int) -> str:
    """Format a non-negative token count without obscuring its scale."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("token count must be a non-negative integer")
    if value < 1_000:
        return str(value)
    if value < 10_000:
        return f"{value / 1_000:.1f}K"
    if value < 1_000_000:
        return f"{value // 1_000}K"
    if value < 10_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value // 1_000_000}M"


def context_window_text(used: int | None, total: int | None) -> Text:
    """Build a truthful compact view of the latest exact request usage."""

    if total is None or total <= 0:
        return Text("ctx —", style="dim")
    total_label = format_token_count(total)
    if used is None or used <= 0:
        return Text(f"ctx — / {total_label}", style="dim")
    percentage = used / total * 100
    style = (
        "#DE3535" if percentage >= 90 else "#FBBF24" if percentage >= 75 else "#ACFD21"
    )
    return Text(
        f"ctx {format_token_count(used)} / {total_label}",
        style=style,
    )


class StatusBar(Horizontal):
    def __init__(self) -> None:
        super().__init__(id="status-bar")

    def compose(self) -> ComposeResult:
        yield Static(
            Text("DAITA", style="bold #ACFD21"),
            id="status-primary",
        )
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
        too_small: bool = False,
    ) -> None:
        primary = self.query_one("#status-primary", Static)
        context = self.query_one("#context-window", Static)
        if too_small:
            message = Text("DAITA", style="bold #ACFD21")
            message.append("  Terminal too small; 32x8 required", style="bold")
            primary.update(message)
            context.display = False
            return
        context.display = True
        safe_agent = sanitize_terminal_text(
            agent, maximum=32, preserve_lines=False, fallback="agent"
        )
        safe_model = sanitize_terminal_text(
            model, maximum=48, preserve_lines=False, fallback="model"
        )
        safe_source = sanitize_terminal_text(
            source, maximum=48, preserve_lines=False, fallback="source"
        )
        safe_state = sanitize_terminal_text(
            state, maximum=32, preserve_lines=False, fallback="ready"
        )
        state_style = "bold #FBBF24" if safe_state == "approval" else "bold #ACFD21"
        message = Text("DAITA", style="bold #ACFD21")
        message.append(f"  {safe_agent} / {safe_model} / {safe_source}  ", style="dim")
        message.append(f"[{safe_state}]", style=state_style)
        primary.update(message)
        context.update(context_window_text(context_used, context_total))


class ActivityBar(Static):
    """Animated, observable run activity without exposing hidden reasoning."""

    _FRAMES = ("◆", "◇")

    def __init__(self) -> None:
        super().__init__("", id="activity-bar", markup=False)
        self._stage = "Thinking"
        self._started_at: float | None = None
        self._frame = 0
        self._timer: object | None = None
        self.display = False
        self.tooltip = (
            "Live execution activity. Private model chain-of-thought is not exposed."
        )

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.25, self._tick, pause=True)

    def start(self, stage: str = "Thinking") -> None:
        self._started_at = time.monotonic()
        self._frame = 0
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

    def stop(self) -> None:
        timer = self._timer
        if timer is not None:
            timer.pause()  # type: ignore[attr-defined]
        self._started_at = None
        self.update("")
        self.display = False

    def _tick(self) -> None:
        if self._started_at is None:
            return
        self._frame = (self._frame + 1) % len(self._FRAMES)
        self._render_activity()

    def _render_activity(self) -> None:
        elapsed = 0.0
        if self._started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._started_at)
        content = Text(self._FRAMES[self._frame], style="bold #ACFD21")
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
