"""Exact once-only side-effect approval as a typed modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from daita import ApprovalDecision, ApprovalRequest

from ..models import MIN_APPROVAL_ROWS
from ..projection import approval_review_document
from ..sanitization import sanitize_terminal_text


class ApprovalCancelled(Exception):
    """Approval was interrupted without fabricating DENY."""


class ApprovalScreen(ModalScreen[ApprovalDecision | None]):
    """None means cancelled; explicit Yes/No return APPROVE/DENY once."""

    BINDINGS = [
        Binding("y", "approve", "Yes", show=True),
        Binding("n", "deny", "No", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self._request = request
        self._decided = False
        rendered = request.render_arguments_for_review()
        self._document, self._reviewable = approval_review_document(
            tool_name=request.tool_name,
            capability_id=request.capability_id,
            arguments_text=rendered,
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Label("Approval required", id="approval-title", markup=False)
            with VerticalScroll(id="approval-document"):
                yield Static(self._document_text(), markup=False, id="approval-text")
            if self._reviewable:
                yield Button("Yes, approve once", id="approval-yes", variant="success")
                yield Button("No", id="approval-no", variant="error")
            else:
                yield Label(
                    "This change cannot be reviewed here. Approval is unavailable.",
                    id="approval-unreviewable",
                    markup=False,
                )
                yield Button("Close", id="approval-close")

    def on_mount(self) -> None:
        if self.size.height < MIN_APPROVAL_ROWS:
            self._decide(None)

    def on_resize(self) -> None:
        if self.size.height < MIN_APPROVAL_ROWS:
            self._decide(None)

    def _document_text(self) -> str:
        if self._document is None:
            return "Approval denied: exact arguments exceed the terminal review bound."
        return sanitize_terminal_text(
            self._document,
            maximum=len(self._document) + 1,
            preserve_lines=True,
            fallback="unreviewable",
        )

    def _decide(self, decision: ApprovalDecision | None) -> None:
        if self._decided:
            return
        if decision is ApprovalDecision.APPROVE and not self._reviewable:
            return
        self._decided = True
        for button in self.query(Button):
            button.disabled = True
        self.dismiss(decision)

    def action_approve(self) -> None:
        if self._reviewable:
            self._decide(ApprovalDecision.APPROVE)

    def action_deny(self) -> None:
        if self._reviewable:
            self._decide(ApprovalDecision.DENY)

    def action_cancel(self) -> None:
        self._decide(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-yes":
            self._decide(ApprovalDecision.APPROVE)
        elif event.button.id == "approval-no":
            self._decide(ApprovalDecision.DENY)
        else:
            self._decide(None)
