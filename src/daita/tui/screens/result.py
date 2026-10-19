"""Show and dismiss one durable routine result in the terminal UI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, Static

from daita import DeliveryState, DeliverySubjectKind, DeliveryView
from daita.distribution import delivery_view

from ..sanitization import safe_display, sanitize_terminal_text


class ResultScreen(ModalScreen[None]):
    """Present one exact agent-owned result from the work board."""

    BINDINGS = [
        Binding("escape", "close", "Back", priority=True),
        Binding("d", "dismiss_result", "Dismiss", priority=True),
    ]

    def __init__(self, *, delivery_id: str) -> None:
        super().__init__()
        self._delivery_id = delivery_id
        self._item: DeliveryView | None = None
        self._busy = True

    def compose(self) -> ComposeResult:
        with Vertical(id="result-manager"):
            yield Label("Result", id="result-title", markup=False)
            with VerticalScroll(id="result-detail-scroll"):
                yield Static("Loading…", id="result-detail", markup=False)
            yield Static(
                "Dismissing a result removes it from the work board. "
                "It never reruns the work.",
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


def render_result(item: DeliveryView) -> str:
    """Render one bounded delivery without treating report text as markup."""

    outcome = item.conclusion_state.value
    reason = item.failure_code or item.blocked_reason_code or outcome
    is_routine = item.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE
    subject_label = "Routine" if is_routine else "Job"
    subject_identity = item.subject_id
    run_label = (
        "No model run started"
        if item.resulting_run_id is None
        else safe_display(item.resulting_run_id, fallback="unknown", maximum=256)
    )
    detail_heading = "Escalation" if item.resulting_run_id is None else "Report"
    lines = [
        item.subject_kind.value.replace("_", " ").title(),
        f"State: {item.state.value} · Outcome: {outcome}",
        f"Created: {item.created_at.isoformat()}",
        "Origin conversation: "
        + safe_display(item.conversation_id, fallback="unknown", maximum=256),
        subject_label
        + ": "
        + safe_display(subject_identity, fallback="unknown", maximum=256),
        "Result run: " + run_label,
        "Sensitivity: " + item.effective_sensitivity.value,
        "Conclusion checksum: " + item.conclusion_digest,
        "Provenance checksum: " + item.provenance_digest,
        "Destination: "
        + safe_display(item.destination_id, fallback="destination", maximum=256),
        "Reason: " + reason,
        "",
        detail_heading,
    ]
    report = item.conclusion_preview
    if report:
        lines.append(
            sanitize_terminal_text(
                report,
                maximum=49_152,
                preserve_lines=True,
                fallback="Report preview is unavailable.",
            )
        )
        if item.conclusion_preview_truncated:
            lines.append(
                "\nPreview truncated. The result run remains the authoritative reference."
            )
    elif item.state is DeliveryState.BLOCKED:
        lines.append(
            "Preview withheld because the result exceeds this destination's "
            "sensitivity eligibility."
        )
    elif item.resulting_run_id is None:
        lines.append("The occurrence failed before a model run could start.")
    else:
        lines.append("No report preview is available for this result.")
    detail = item.blocked_reason_code or item.failure_code
    if detail is not None:
        lines.extend(
            (
                "",
                "Delivery detail: "
                + safe_display(
                    detail,
                    fallback="unavailable",
                    maximum=256,
                ),
            )
        )
    if item.artifact_references:
        lines.extend(("", "Outcome artifacts"))
        for artifact in item.artifact_references:
            lines.extend(
                (
                    "Artifact: "
                    + safe_display(
                        artifact.artifact_id,
                        fallback="artifact",
                        maximum=256,
                    ),
                    "  Media: " + artifact.media_type,
                    f"  Bytes: {artifact.byte_size}",
                    "  Checksum: " + artifact.sha256,
                    "  Provenance: " + artifact.provenance_digest,
                    "  Authorship: " + artifact.authorship.value,
                    "  Producer: "
                    + safe_display(
                        artifact.producer_capability_id,
                        fallback="unknown",
                        maximum=256,
                    ),
                    "  Sensitivity: " + artifact.sensitivity.value,
                )
            )
    return "\n".join(lines)


__all__ = ["ResultScreen", "render_result"]
