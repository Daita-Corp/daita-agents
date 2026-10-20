"""Show keys and grouped commands in one dismissible reference."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from ..commands import COMMAND_GROUPS

_GETTING_STARTED = (
    ("Ask", "Type a question in plain words; Daita picks the sources"),
    ("@source", 'Narrow one question to a source: @"sales db" <question>'),
    ("/files", "Ask using only local files"),
    ("After", "Save, schedule or teach from the row under each answer"),
)
_KEYS = (
    ("Enter", "Send"),
    ("Ctrl-J", "New line"),
    ("Esc Esc", "Clear input"),
    ("Ctrl-O", "Show or hide the steps behind answers"),
    ("Ctrl-C", "Copy selection, or cancel the run"),
    ("Ctrl-1…4", "Chat, Sources, Work, Artifacts"),
    ("F6", "Move focus to the tabs"),
    ("PgUp/PgDn", "Scroll; Ctrl-Home start, Ctrl-End latest"),
    ("Work tab", "Arrows move, Enter opens, d dismisses a result"),
    ("Ctrl-D", "Quit"),
)


def help_text() -> Text:
    """Render the reference: getting started, keys, then commands by group."""

    text = Text()

    def section(title: str, rows: tuple[tuple[str, str], ...], width: int) -> None:
        if text:
            text.append("\n")
        text.append(title.upper() + "\n", style="bold")
        for key, description in rows:
            text.append(f"  {key.ljust(width)}", style="bold")
            text.append(f"  {description}\n", style="dim")

    section("Getting started", _GETTING_STARTED, 9)
    section("Keys", _KEYS, 9)
    for group, commands in COMMAND_GROUPS:
        rows = tuple((display, description) for _i, display, description in commands)
        section(f"Commands · {group}", rows, max(len(row[0]) for row in rows))
    return text


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("f1", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-panel", classes="modal-panel"):
            yield Static(help_text(), id="help-text", markup=False)

    def action_close(self) -> None:
        self.dismiss(None)
