"""Multiline composer with the existing character bound and key contracts."""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from ..models import MAX_COMPOSER_CHARACTERS
from ..sanitization import sanitize_terminal_text


class ComposerSubmitted(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class Composer(TextArea):
    """Enter submits, Ctrl-J inserts a newline, Escape-Escape clears."""

    BINDINGS = [
        ("enter", "submit", "Submit"),
        ("ctrl+j", "newline", "Newline"),
        ("tab", "tab", "Complete"),
        ("escape", "escape", "Clear"),
        ("ctrl+d", "eof", "Exit"),
    ]

    can_focus = True
    compact = True
    show_line_numbers = False

    def __init__(self) -> None:
        super().__init__("", id="composer", language=None)
        self._escape_armed = False
        self.completion_active = False
        self.disabled_reason = ""

    def on_mount(self) -> None:
        self.tab_behavior = "indent"

    async def _on_key(self, event: events.Key) -> None:
        action = {
            "enter": self.action_submit,
            "ctrl+j": self.action_newline,
            "tab": self.action_tab,
            "escape": self.action_escape,
            "ctrl+d": self.action_eof,
        }.get(event.key)
        if action is not None:
            event.stop()
            event.prevent_default()
            action()
            return
        await super()._on_key(event)

    def watch_text(self, value: str) -> None:
        if len(value) > MAX_COMPOSER_CHARACTERS:
            self.load_text(value[:MAX_COMPOSER_CHARACTERS])
            self.post_message(ComposerLimitReached())

    def action_newline(self) -> None:
        if len(self.text) >= MAX_COMPOSER_CHARACTERS:
            self.post_message(ComposerLimitReached())
            return
        self.insert("\n")

    def action_submit(self) -> None:
        if self.disabled:
            return
        if self.completion_active:
            self.post_message(ComposerCompletionAccepted())
            return
        self.post_message(ComposerSubmitted(self.text))

    def action_tab(self) -> None:
        if self.disabled:
            return
        if self.completion_active:
            self.post_message(ComposerCompletionAccepted())
            return
        if len(self.text) >= MAX_COMPOSER_CHARACTERS:
            self.post_message(ComposerLimitReached())
            return
        self.insert("\t")

    def action_cursor_up(self, select: bool = False) -> None:
        if self.completion_active and not select:
            self.post_message(ComposerCompletionMoved(-1))
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if self.completion_active and not select:
            self.post_message(ComposerCompletionMoved(1))
            return
        super().action_cursor_down(select)

    def action_escape(self) -> None:
        if self.completion_active:
            self._escape_armed = False
            self.post_message(ComposerCompletionDismissed())
            return
        if self._escape_armed:
            self.clear()
            self._escape_armed = False
            return
        self._escape_armed = True
        self.set_timer(0.8, self._clear_escape)

    def _clear_escape(self) -> None:
        self._escape_armed = False

    def action_eof(self) -> None:
        if self.text.strip():
            return
        self.post_message(ComposerExitRequested())

    def set_submitting(self, submitting: bool) -> None:
        self.disabled = submitting
        if submitting:
            self.completion_active = False

    def consume(self) -> str:
        text = self.text
        self.clear()
        return text


class ComposerLimitReached(Message):
    pass


class ComposerExitRequested(Message):
    pass


class ComposerCompletionAccepted(Message):
    pass


class ComposerCompletionDismissed(Message):
    pass


class ComposerCompletionMoved(Message):
    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = -1 if delta < 0 else 1


class CompletionPopup(Static):
    """Contextual slash, skill, and source completions."""

    matches: reactive[tuple[tuple[str, str, str], ...]] = reactive(())

    def compose(self):
        yield Static(
            "COMMANDS  Up/Down navigate  Tab/Enter insert",
            id="completion-title",
            markup=False,
        )
        yield OptionList(id="completion-list")

    def update_matches(self, matches: tuple[tuple[str, str, str], ...]) -> None:
        listing = self.query_one(OptionList)
        listing.clear_options()
        self.matches = matches[:12]
        for index, (_insertion, shown, description) in enumerate(self.matches):
            label = Text(
                sanitize_terminal_text(
                    shown,
                    maximum=160,
                    preserve_lines=False,
                    fallback="command",
                ),
                style="bold",
            )
            label.append("  ")
            label.append(
                sanitize_terminal_text(
                    description,
                    maximum=320,
                    preserve_lines=False,
                    fallback="",
                ),
                style="dim",
            )
            listing.add_option(Option(label, id=f"completion-{index}"))
        listing.highlighted = 0 if self.matches else None
        self.display = bool(self.matches)

    def dismiss(self) -> None:
        self.update_matches(())

    def move_highlight(self, delta: int) -> None:
        listing = self.query_one(OptionList)
        if not self.matches:
            return
        current = listing.highlighted
        if current is None:
            listing.highlighted = 0
            return
        listing.highlighted = (current + delta) % len(self.matches)
        listing.scroll_to_highlight()

    def insertion_at(self, index: int) -> str | None:
        if not 0 <= index < len(self.matches):
            return None
        return self.matches[index][0]

    def selected_insertion(self) -> str | None:
        listing = self.query_one(OptionList)
        if listing.highlighted is None:
            return None
        return self.insertion_at(listing.highlighted)
