"""Collect skill names and review cost limits before editing or evaluation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..sanitization import sanitize_terminal_text


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


class ScheduleScreen(ModalScreen[str | None]):
    """Ask when to repeat the latest answer; the agent proposes the routine."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, *, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="editor-prompt"):
            yield Label("Run this on a schedule", id="onboard-title", markup=False)
            yield Static(
                sanitize_terminal_text(
                    self._question, maximum=512, preserve_lines=False, fallback=""
                ),
                id="schedule-question",
                markup=False,
            )
            yield Static(
                "Daita proposes a routine from this answer for your approval.",
                markup=False,
            )
            yield Input(
                placeholder="every weekday at 9am",
                id="schedule-when",
            )
            yield Label("", id="editor-prompt-error", markup=False)
            yield Button("Continue", id="schedule-continue", variant="primary")
            yield Button("Cancel", id="editor-prompt-cancel")

    def on_mount(self) -> None:
        self.query_one("#schedule-when", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "editor-prompt-cancel":
            self.dismiss(None)
            return
        if event.button.id == "schedule-continue":
            self._submit()

    def _submit(self) -> None:
        when = self.query_one("#schedule-when", Input).value.strip()
        if not when:
            self.query_one("#editor-prompt-error", Label).update(
                "Say when it should run."
            )
            return
        self.dismiss(when)
