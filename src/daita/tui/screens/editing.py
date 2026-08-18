"""Small Textual forms that precede external document editing or review."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..models import (
    DEFAULT_CANDIDATE_REVIEW_COST_LIMIT_USD,
    parse_candidate_review_cost_limit,
)


class SkillNameScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, *, initial_name: str = "") -> None:
        super().__init__()
        self._initial_name = initial_name

    def compose(self) -> ComposeResult:
        with Vertical(id="editor-prompt"):
            yield Label("Create a skill", id="onboard-title", markup=False)
            yield Static(
                "Choose a lowercase name. The description and procedure will open "
                "in $EDITOR.",
                markup=False,
            )
            yield Input(
                value=self._initial_name,
                placeholder="skill-name",
                id="skill-name",
            )
            yield Label("", id="editor-prompt-error", markup=False)
            yield Button("Continue", id="skill-name-continue", variant="primary")
            yield Button("Cancel", id="editor-prompt-cancel")

    def on_mount(self) -> None:
        self.query_one("#skill-name", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "editor-prompt-cancel":
            self.dismiss(None)
            return
        if event.button.id == "skill-name-continue":
            self._submit()

    def _submit(self) -> None:
        name = self.query_one("#skill-name", Input).value.strip()
        if not name:
            self.query_one("#editor-prompt-error", Label).update("Enter a skill name.")
            return
        self.dismiss(name)


class ReviewCostScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="review-cost"):
            yield Label(
                "Authorize one learning review",
                id="onboard-title",
                markup=False,
            )
            yield Static(
                "This can make one model call and only adds suggestions to the "
                "review inbox. Memory and skills change only after acceptance. "
                "Provider charges can still apply.",
                markup=False,
            )
            yield Input(
                value=str(DEFAULT_CANDIDATE_REVIEW_COST_LIMIT_USD),
                placeholder="Maximum estimated cost in USD",
                id="review-cost-value",
            )
            yield Label("", id="review-cost-error", markup=False)
            yield Button("Run review", id="review-cost-accept", variant="primary")
            yield Button("Cancel", id="review-cost-cancel")

    def on_mount(self) -> None:
        self.query_one("#review-cost-value", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "review-cost-cancel":
            self.dismiss(None)
            return
        if event.button.id == "review-cost-accept":
            self._submit()

    def _submit(self) -> None:
        value = self.query_one("#review-cost-value", Input).value.strip()
        try:
            parse_candidate_review_cost_limit(value)
        except ValueError:
            self.query_one("#review-cost-error", Label).update(
                "Enter a finite non-negative USD amount."
            )
            return
        self.dismiss(value)
