"""Destructive local confirmations."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from ..sanitization import sanitize_terminal_text


class ConfirmScreen(ModalScreen[bool]):
    """Yes/No confirmation. False means cancelled or declined."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("y", "accept", "Yes"),
        Binding("n", "decline", "No"),
    ]

    def __init__(self, message: str, *, expected_text: str | None = None) -> None:
        super().__init__()
        self._message = message
        self._expected = expected_text
        self._decided = False

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Label(
                sanitize_terminal_text(
                    self._message,
                    maximum=1_024,
                    preserve_lines=True,
                    fallback="Confirm?",
                ),
                id="confirm-message",
                markup=False,
            )
            if self._expected is not None:
                yield Input(
                    placeholder=f"Type {self._expected} to confirm", id="confirm-input"
                )
            with Horizontal(id="confirm-actions"):
                yield Button("Yes", id="confirm-yes", variant="error")
                yield Button("No", id="confirm-no")

    def _decide(self, accepted: bool) -> None:
        if self._decided:
            return
        if accepted and self._expected is not None:
            typed = self.query_one("#confirm-input", Input).value
            if typed != self._expected:
                self.query_one("#confirm-message", Label).update(
                    "Confirmation text did not match."
                )
                return
        self._decided = True
        for button in self.query(Button):
            button.disabled = True
        self.dismiss(accepted)

    def action_accept(self) -> None:
        self._decide(True)

    def action_decline(self) -> None:
        self._decide(False)

    def action_cancel(self) -> None:
        self._decide(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._decide(event.button.id == "confirm-yes")
