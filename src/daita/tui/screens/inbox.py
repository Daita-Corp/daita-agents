"""Inspect and acknowledge durable autonomous results in the terminal UI."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, OptionList, Static
from textual.widgets.option_list import Option

from daita import DeliveryState, DeliverySubjectKind, InboxItem

from ..sanitization import safe_display, sanitize_terminal_text


class InboxScreen(ModalScreen[None]):
    """Present the current agent's bounded durable conversation inbox."""

    BINDINGS = [
        Binding("escape", "close", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("a", "acknowledge", "Acknowledge", priority=True),
    ]

    def __init__(self, *, delivery_id: str | None = None) -> None:
        super().__init__()
        self._items: tuple[InboxItem, ...] = ()
        self._target_delivery_id = delivery_id
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="inbox-manager"):
            yield Label("Inbox", id="inbox-title", markup=False)
            yield Static("Loading…", id="inbox-summary", markup=False)
            yield Static("", id="inbox-notice", markup=False)
            yield OptionList(id="inbox-list")
            with VerticalScroll(id="inbox-detail-scroll"):
                yield Static("", id="inbox-detail", markup=False)
            yield Static(
                "Reports are delivered once to this durable inbox. "
                "Acknowledgment changes attention state and never reruns reasoning.",
                id="inbox-help",
                markup=False,
            )
            with Horizontal(id="inbox-actions"):
                yield Button("Refresh", id="inbox-refresh")
                yield Button("Acknowledge", id="inbox-acknowledge", variant="primary")
                yield Button("Close", id="inbox-close")
            yield Static("", id="inbox-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self.run_worker(
            self._load_initial(),
            name="inbox-initial-load",
            group="inbox-interaction",
            exclusive=True,
        )

    def action_close(self) -> None:
        if not self._busy:
            self.dismiss(None)

    def action_refresh(self) -> None:
        self._schedule("refresh")

    def action_acknowledge(self) -> None:
        self._schedule("acknowledge")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "inbox-close":
            self.action_close()
        elif button_id == "inbox-refresh":
            self.action_refresh()
        elif button_id == "inbox-acknowledge":
            self.action_acknowledge()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        del event
        item = self._selected_item()
        if item is not None:
            self._render_item(item)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        del event
        if not self._busy:
            item = self._selected_item()
            if item is not None:
                self._render_item(item)
        self._update_actions()

    def _schedule(self, action: str) -> None:
        if self._busy:
            return
        self.run_worker(
            self._handle_action(action),
            name=f"inbox-{action}",
            group="inbox-interaction",
            exclusive=True,
        )

    async def _load_initial(self) -> None:
        self._set_busy(True)
        try:
            await self._load_items()
            self.query_one("#inbox-list", OptionList).focus()
            item = self._selected_item()
            if item is not None:
                self._render_item(item)
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._set_busy(False)

    async def _handle_action(self, action: str) -> None:
        self._set_busy(True)
        self.query_one("#inbox-error", Static).update("")
        try:
            if action == "refresh":
                await self._load_items()
                self.query_one("#inbox-notice", Static).update("Inbox refreshed.")
                item = self._selected_item()
                if item is not None:
                    self._render_item(item)
                await self.app.refresh_background_status(  # type: ignore[attr-defined]
                    notify_new=False
                )
                return
            item = self._selected_item()
            if item is None:
                raise ValueError("Select an inbox result first.")
            acknowledged = await self.app.controller.acknowledge_inbox(  # type: ignore[attr-defined]
                item.delivery_id
            )
            if acknowledged is None:
                raise ValueError(
                    "The inbox result no longer exists within this agent boundary."
                )
            self.query_one("#inbox-notice", Static).update(
                "Result acknowledged · "
                + safe_display(item.delivery_id, fallback="delivery", maximum=256)
            )
            await self._load_items(selected_delivery_id=None)
            next_item = self._selected_item()
            if next_item is not None:
                self._render_item(next_item)
            await self.app.refresh_background_status(  # type: ignore[attr-defined]
                notify_new=False
            )
        except (ValueError, RuntimeError, OSError) as error:
            self._show_error(error)
        finally:
            if self.is_mounted:
                self._set_busy(False)

    async def _load_items(self, *, selected_delivery_id: str | None = "") -> None:
        selected = (
            self._selected_delivery_id()
            if selected_delivery_id == ""
            else selected_delivery_id
        )
        if selected is None:
            selected = self._target_delivery_id
        self._items = await self.app.controller.list_inbox()  # type: ignore[attr-defined]
        listing = self.query_one("#inbox-list", OptionList)
        listing.clear_options()
        for item in self._items:
            listing.add_option(
                Option(Text(self._list_label(item)), id=item.delivery_id)
            )
        if self._items:
            listing.highlighted = next(
                (
                    index
                    for index, item in enumerate(self._items)
                    if item.delivery_id == selected
                ),
                0,
            )
        else:
            self.query_one("#inbox-detail", Static).update(
                "No unacknowledged results. Completed background reports will appear here."
            )
        noun = "result" if len(self._items) == 1 else "results"
        self.query_one("#inbox-summary", Static).update(
            f"{len(self._items)} unacknowledged {noun}"
        )
        self._update_actions()

    def _selected_delivery_id(self) -> str | None:
        listing = self.query_one("#inbox-list", OptionList)
        if listing.highlighted is None:
            return None
        option = listing.get_option_at_index(listing.highlighted)
        return str(option.id) if option.id is not None else None

    def _selected_item(self) -> InboxItem | None:
        selected = self._selected_delivery_id()
        if selected is None:
            return None
        return next(
            (item for item in self._items if item.delivery_id == selected), None
        )

    def _render_item(self, item: InboxItem) -> None:
        self.query_one("#inbox-detail", Static).update(render_inbox_item(item))

    @staticmethod
    def _list_label(item: InboxItem) -> str:
        outcome = safe_display(
            item.payload.get("outcome"), fallback=item.state.value, maximum=32
        ).upper()
        subject = item.subject.kind.value.replace("_", " ")
        short_id = (
            item.delivery_id
            if len(item.delivery_id) <= 20
            else "…" + item.delivery_id[-19:]
        )
        return sanitize_terminal_text(
            f"{outcome:<10} {subject} · {short_id} · "
            f"{item.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            maximum=512,
            preserve_lines=False,
            fallback="inbox result",
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_actions()

    def _update_actions(self) -> None:
        if not self.is_mounted:
            return
        item = self._selected_item()
        self.query_one("#inbox-refresh", Button).disabled = self._busy
        self.query_one("#inbox-acknowledge", Button).disabled = (
            self._busy or item is None or item.state is DeliveryState.ACKNOWLEDGED
        )
        self.query_one("#inbox-close", Button).disabled = self._busy

    def _show_error(self, error: Exception) -> None:
        self.query_one("#inbox-error", Static).update(
            sanitize_terminal_text(
                str(error),
                maximum=512,
                preserve_lines=False,
                fallback="Inbox action failed.",
            )
        )


def render_inbox_item(item: InboxItem) -> str:
    """Render one bounded delivery without treating report text as markup."""

    payload = item.payload
    outcome = safe_display(payload.get("outcome"), fallback="unknown", maximum=32)
    reason = safe_display(payload.get("reason"), fallback="unknown", maximum=256)
    is_routine = item.subject.kind is DeliverySubjectKind.ROUTINE_OCCURRENCE
    subject_label = "Routine" if is_routine else "Job"
    subject_identity = (
        payload.get("routine_id") if is_routine else payload.get("job_id")
    )
    run_label = (
        "No model run started"
        if item.resulting_run_id is None
        else safe_display(item.resulting_run_id, fallback="unknown", maximum=256)
    )
    detail_heading = (
        "Escalation"
        if item.resulting_run_id is None and payload.get("escalation") is True
        else "Report"
    )
    lines = [
        item.subject.kind.value.replace("_", " ").title(),
        f"State: {item.state.value} · Outcome: {outcome}",
        f"Created: {item.created_at.isoformat()}",
        "Origin conversation: "
        + safe_display(item.conversation_id, fallback="unknown", maximum=256),
        subject_label
        + ": "
        + safe_display(subject_identity, fallback="unknown", maximum=256),
        "Result run: " + run_label,
        "Sensitivity: " + item.sensitivity.value,
        "Destination: " + safe_display(item.destination, fallback="inbox", maximum=256),
        "Reason: " + reason,
        "",
        detail_heading,
    ]
    report = payload.get("report_preview")
    if isinstance(report, str) and report:
        lines.append(
            sanitize_terminal_text(
                report,
                maximum=49_152,
                preserve_lines=True,
                fallback="Report preview is unavailable.",
            )
        )
        if payload.get("report_truncated") is True:
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
    if item.terminal_error is not None:
        lines.extend(
            (
                "",
                "Delivery detail: "
                + safe_display(
                    item.terminal_error,
                    fallback="unavailable",
                    maximum=256,
                ),
            )
        )
    return "\n".join(lines)


__all__ = ["InboxScreen", "render_inbox_item"]
