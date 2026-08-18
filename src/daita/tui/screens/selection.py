"""Shared picker used by agent, model, source, and permission flows."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label, OptionList
from textual.widgets.option_list import Option

from ..models import PickerOption
from ..sanitization import sanitize_terminal_text


class SelectionScreen(ModalScreen[tuple[str, ...] | None]):
    """Filterable single or multi-select picker. None means cancelled."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "confirm", "Confirm", show=True),
        Binding("space", "toggle_selected", "Toggle", show=False),
    ]

    def __init__(
        self,
        *,
        title: str,
        options: tuple[PickerOption, ...],
        multi: bool = False,
        allow_select_all: bool = False,
        allow_empty: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self._options = options
        self._multi = multi
        self._allow_select_all = allow_select_all
        self._allow_empty = allow_empty
        self._selected: set[str] = set()
        self._visible: tuple[PickerOption, ...] = options

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label(
                sanitize_terminal_text(
                    self._title,
                    maximum=240,
                    preserve_lines=False,
                    fallback="Select",
                ),
                id="picker-title",
                markup=False,
            )
            yield Input(placeholder="Filter", id="picker-filter")
            yield OptionList(id="picker-options")
            yield Label("", id="picker-error", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self._render_options(self._options)
        self.query_one("#picker-filter", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "picker-filter":
            return
        needle = event.value.casefold()
        visible = tuple(
            option
            for option in self._options
            if needle in option.label.casefold()
            or needle in option.description.casefold()
            or needle in option.identity.casefold()
        )
        self._render_options(visible)

    def _render_options(self, options: tuple[PickerOption, ...]) -> None:
        listing = self.query_one("#picker-options", OptionList)
        listing.clear_options()
        self._visible = options
        for option in options:
            mark = (
                "[x] "
                if option.identity in self._selected
                else "[ ] " if self._multi else ""
            )
            description = f" — {option.description}" if option.description else ""
            listing.add_option(
                Option(
                    sanitize_terminal_text(
                        f"{mark}{option.label}{description}",
                        maximum=240,
                        preserve_lines=False,
                        fallback="option",
                    ),
                    id=option.identity,
                )
            )

    def action_toggle_selected(self) -> None:
        if not self._multi:
            return
        identity = self._highlighted()
        if identity is None:
            return
        if identity in self._selected:
            self._selected.discard(identity)
        else:
            self._selected.add(identity)
        self._render_options(self._visible)

    def action_confirm(self) -> None:
        if self._multi:
            if not self._selected and not self._allow_empty:
                self.query_one("#picker-error", Label).update(
                    "Select at least one option."
                )
                return
            self.dismiss(
                tuple(
                    option.identity
                    for option in self._options
                    if option.identity in self._selected
                )
            )
            return
        identity = self._highlighted()
        if identity is None:
            self.query_one("#picker-error", Label).update("Select an option.")
            return
        self.dismiss((identity,))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._multi:
            self.action_toggle_selected()
            return
        if event.option.id is not None:
            self.dismiss((str(event.option.id),))

    def _highlighted(self) -> str | None:
        listing = self.query_one("#picker-options", OptionList)
        if listing.highlighted is None:
            return None
        option = listing.get_option_at_index(listing.highlighted)
        return str(option.id) if option.id is not None else None
