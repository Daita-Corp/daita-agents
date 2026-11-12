"""Show and dismiss one durable routine result in the terminal UI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, Static

from daita import DeliveryState, DeliveryView, OutcomeState
from daita.distribution import delivery_view

from ..sanitization import safe_display, sanitize_terminal_text


class ResultScreen(ModalScreen[None]):
    """Present one exact agent-owned result from the work board."""

    BINDINGS = [
        Binding("escape", "close", "Back", priority=True),
        Binding("d", "dismiss_result", "Dismiss", priority=True),
    ]

    def __init__(self, *, delivery_id: str, title: str = "") -> None:
        super().__init__()
        self._delivery_id = delivery_id
        self._title = title
        self._item: DeliveryView | None = None
        self._busy = True

    def compose(self) -> ComposeResult:
        with Vertical(id="result-manager"):
            yield Label(
                safe_display(self._title, fallback="Result", maximum=120),
                id="result-title",
                markup=False,
            )
            with VerticalScroll(id="result-detail-scroll"):
                yield Static("Loading…", id="result-detail", markup=False)
            yield Static(
                "Dismiss removes this from the work board. It never reruns the work.",
                id="result-help",
                markup=False,
            )
            with Horizontal(id="result-actions"):
                yield Button("Dismiss", id="result-dismiss", variant="primary")
                yield Button("Close", id="result-close")
            yield Static("", id="result-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self._update_actions()
        self.run_worker(
            self._load(),
            name="result-load",
            group="result-interaction",
            exclusive=True,
        )

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def action_dismiss_result(self) -> None:
        if self._busy or self._item is None:
            return
        self.run_worker(
            self._dismiss_result(),
            name="result-dismiss",
            group="result-interaction",
            exclusive=True,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "result-close":
            self.action_close()
        elif event.button.id == "result-dismiss":
            self.action_dismiss_result()

    async def _load(self) -> None:
        try:
            inspection = await self.app.controller.inspect_delivery(  # type: ignore[attr-defined]
                self._delivery_id
            )
            if inspection is None:
                raise ValueError("This result no longer exists for this agent.")
            self._item = delivery_view(inspection.delivery)
            self.query_one("#result-detail", Static).update(render_result(self._item))
        except (ValueError, RuntimeError, OSError) as error:
            self.query_one("#result-detail", Static).update("")
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._busy = False
                self._update_actions()
                self.query_one(
                    (
                        "#result-close"
                        if self.query_one("#result-dismiss", Button).disabled
                        else "#result-dismiss"
                    ),
                    Button,
                ).focus()

    async def _dismiss_result(self) -> None:
        self._busy = True
        self._update_actions()
        try:
            dismissed = await self.app.controller.acknowledge_delivery(  # type: ignore[attr-defined]
                self._delivery_id
            )
            if dismissed is None:
                raise ValueError("This result no longer exists for this agent.")
            await self.app.refresh_background_status(  # type: ignore[attr-defined]
                notify_new=False
            )
            self.dismiss(None)
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
            if self.is_mounted:
                self._busy = False
                self._update_actions()

    def _update_actions(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#result-dismiss", Button).disabled = (
            self._busy
            or self._item is None
            or self._item.state is DeliveryState.ACKNOWLEDGED
        )
        self.query_one("#result-close", Button).disabled = self._busy

    def _show_error(self, error: Exception) -> None:
        self.query_one("#result-error", Static).update(
            sanitize_terminal_text(
                str(error),
                maximum=512,
                preserve_lines=False,
                fallback="Result action failed.",
            )
        )


_STATUS = {
    OutcomeState.SUCCEEDED: "✓ Succeeded",
    OutcomeState.FAILED: "✗ Failed",
    OutcomeState.SKIPPED_NO_CHANGE: "– Nothing changed",
}


def _size(byte_size: int) -> str:
    for unit, size in (("MB", 1_048_576), ("KB", 1_024)):
        if byte_size >= size:
            return f"{byte_size / size:.1f} {unit}"
    return f"{byte_size} bytes"


def render_result(item: DeliveryView) -> str:
    """Render one result for a person: status, time, report and files.

    Run ids, checksums and routing stay in the stored record, not on screen.
    """

    status = _STATUS.get(item.conclusion_state, item.conclusion_state.value)
    if item.state is DeliveryState.BLOCKED:
        status = "Withheld"
    elif item.state is DeliveryState.ACKNOWLEDGED:
        status += " · Dismissed"
    finished = item.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [f"{status} · {finished}", ""]
    report = item.conclusion_preview
    if item.state is DeliveryState.BLOCKED:
        lines.append(
            "This result is more sensitive than this conversation may show, "
            "so it is withheld here."
        )
    elif report:
        lines.append(
            sanitize_terminal_text(
                report,
                maximum=49_152,
                preserve_lines=True,
                fallback="The report is unavailable.",
            )
        )
        if item.conclusion_preview_truncated:
            lines.append("\nPreview truncated; the full report is in the run.")
    elif item.resulting_run_id is None:
        lines.append("It failed before it could start.")
    else:
        lines.append("No report was written.")
    reason = item.blocked_reason_code or item.failure_code
    if reason is not None:
        lines.extend(
            ("", "Reason: " + safe_display(reason, fallback="unknown", maximum=256))
        )
    if item.artifact_references:
        lines.extend(("", "Files (open them in /artifacts)"))
        lines.extend(
            f"  • {artifact.media_type}, {_size(artifact.byte_size)}"
            for artifact in item.artifact_references
        )
    return "\n".join(lines)


__all__ = ["ResultScreen", "render_result"]
