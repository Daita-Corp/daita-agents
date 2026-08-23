"""Display once-only side-effect approvals and return the user's exact decision."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, Static

from daita import ApprovalDecision, ApprovalRequest

from ..models import MIN_APPROVAL_ROWS
from ..projection import approval_review_document
from ..sanitization import sanitize_terminal_text


class ApprovalPanel(Vertical):
    """One transient inline review; None means interrupted, never denial."""

    BINDINGS = [
        Binding("y", "approve", "Approve once", show=True),
        Binding("n", "deny", "Deny", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self) -> None:
        super().__init__(id="approval-inline")
        self._future: asyncio.Future[ApprovalDecision | None] | None = None
        self._reviewable = False

    def compose(self) -> ComposeResult:
        yield Label("Approval required", id="approval-inline-title", markup=False)
        with VerticalScroll(id="approval-inline-document"):
            yield Static("", id="approval-inline-text", markup=False)
        yield Label(
            "This change cannot be reviewed here. Approval is unavailable.",
            id="approval-inline-unreviewable",
            markup=False,
        )
        with Horizontal(id="approval-inline-actions"):
            yield Button("Y  Approve once", id="approval-inline-yes")
            yield Button("N  Deny", id="approval-inline-no")
            yield Button("Close", id="approval-inline-close")

    def on_mount(self) -> None:
        self.display = False

    @property
    def active(self) -> bool:
        return self._future is not None and not self._future.done()

    async def request(self, request: ApprovalRequest) -> ApprovalDecision | None:
        if self.active:
            raise RuntimeError("another approval is already awaiting review")
        if self.app.size.height < MIN_APPROVAL_ROWS:
            return None

        rendered = request.render_arguments_for_review()
        document, self._reviewable = approval_review_document(
            tool_name=request.tool_name,
            capability_id=request.capability_id,
            arguments_text=rendered,
        )
        self.query_one("#approval-inline-text", Static).update(
            self._document_text(document)
        )
        self.query_one("#approval-inline-unreviewable", Label).display = (
            not self._reviewable
        )
        self.query_one("#approval-inline-yes", Button).display = self._reviewable
        self.query_one("#approval-inline-no", Button).display = self._reviewable
        self.query_one("#approval-inline-close", Button).display = not self._reviewable
        for button in self.query(Button):
            button.disabled = False

        future: asyncio.Future[ApprovalDecision | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._future = future
        self.display = True
        target = self.query_one(
            "#approval-inline-no" if self._reviewable else "#approval-inline-close",
            Button,
        )
        self.call_after_refresh(target.focus)
        try:
            return await future
        finally:
            if self._future is future:
                self._future = None
                self.display = False

    def on_resize(self) -> None:
        if self.active and self.app.size.height < MIN_APPROVAL_ROWS:
            self._decide(None)

    def on_unmount(self) -> None:
        self._decide(None)

    def action_approve(self) -> None:
        if self._reviewable:
            self._decide(ApprovalDecision.APPROVE)

    def action_deny(self) -> None:
        if self._reviewable:
            self._decide(ApprovalDecision.DENY)

    def action_cancel(self) -> None:
        self._decide(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-inline-yes":
            self.action_approve()
        elif event.button.id == "approval-inline-no":
            self.action_deny()
        else:
            self.action_cancel()

    def _decide(self, decision: ApprovalDecision | None) -> None:
        future = self._future
        if future is None or future.done():
            return
        if decision is ApprovalDecision.APPROVE and not self._reviewable:
            return
        for button in self.query(Button):
            button.disabled = True
        future.set_result(decision)

    def _document_text(self, document: str | None) -> str:
        if document is None:
            return "Approval unavailable: exact arguments exceed the review bound."
        return sanitize_terminal_text(
            document,
            maximum=len(document) + 1,
            preserve_lines=True,
            fallback="unreviewable",
        )
