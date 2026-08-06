"""Lazy, bounded terminal selection presentation."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any, Generic, TextIO, TypeVar
import unicodedata

from ._installation import repair_guidance

_Value = TypeVar("_Value")
_MAX_OPTIONS = 128
_MAX_LABEL_CHARACTERS = 128
_MAX_DESCRIPTION_CHARACTERS = 256
_MAX_SEARCH_TERMS = 8
_MAX_FILTER_CHARACTERS = 64
_MAX_VALIDATION_CHARACTERS = 256
_MAX_MULTI_SELECTIONS = 32
_NO_SELECTION = object()


class SelectionCancelled(EOFError):
    """The user cancelled an enhanced selector with Escape."""


@dataclass(frozen=True)
class SelectionOption(Generic[_Value]):
    """One bounded display option carrying an independent stable value."""

    value: _Value
    label: str
    description: str = ""
    search_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DisplayedOption(Generic[_Value]):
    identity: int
    value: _Value
    label: str
    description: str
    searchable: str


class _SelectionState(Generic[_Value]):
    def __init__(self, options: tuple[_DisplayedOption[_Value], ...]) -> None:
        self.options = options
        self.filter_text = ""
        self.position = 0

    @property
    def visible(self) -> tuple[_DisplayedOption[_Value], ...]:
        query = self.filter_text.casefold()
        if not query:
            return self.options
        return tuple(option for option in self.options if query in option.searchable)

    def move(self, amount: int) -> None:
        visible = self.visible
        if visible:
            self.position = (self.position + amount) % len(visible)

    def add_filter_text(self, text: str) -> None:
        if len(self.filter_text) >= _MAX_FILTER_CHARACTERS:
            return
        projected = _safe_text(text, maximum=1, fallback="")
        if projected:
            self.filter_text += projected
            self.position = 0

    def backspace(self) -> None:
        if self.filter_text:
            self.filter_text = self.filter_text[:-1]
            self.position = 0

    def selected_value(self) -> object:
        selected = self.current_option()
        if selected is None:
            return _NO_SELECTION
        return selected.value

    def current_option(self) -> _DisplayedOption[_Value] | None:
        visible = self.visible
        if not visible:
            return None
        self.position %= len(visible)
        return visible[self.position]


class _MultiSelectionState(_SelectionState[_Value]):
    def __init__(
        self,
        options: tuple[_DisplayedOption[_Value], ...],
        *,
        maximum: int,
        empty_message: str,
        maximum_message: str,
    ) -> None:
        super().__init__(options)
        self.declared_options = options
        self.maximum = maximum
        self.empty_message = empty_message
        self.maximum_message = maximum_message
        self.selected_identities: set[int] = set()
        self.validation_message = ""

    def toggle(self) -> None:
        option = self.current_option()
        if option is None:
            return
        if option.identity in self.selected_identities:
            self.selected_identities.remove(option.identity)
            self.validation_message = ""
            return
        if len(self.selected_identities) >= self.maximum:
            self.validation_message = self.maximum_message
            return
        self.selected_identities.add(option.identity)
        self.validation_message = ""

    def selected_values(self) -> object:
        if not self.selected_identities:
            self.validation_message = self.empty_message
            return _NO_SELECTION
        return tuple(
            option.value
            for option in self.declared_options
            if option.identity in self.selected_identities
        )


async def select_one(
    title: str,
    options: tuple[SelectionOption[_Value], ...],
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    enhanced_input: Any = None,
    enhanced_output: Any = None,
    invalid_message: str | None = None,
    show_title_in_fallback: bool = True,
) -> _Value:
    """Select one stable value using enhanced navigation or numbered fallback."""

    displayed = _normalize_options(options)
    if (enhanced_input is None) != (enhanced_output is None):
        raise ValueError("enhanced input and output must be supplied together")

    if enhanced_input is None and not _streams_support_enhanced(
        input_stream,
        output_stream,
    ):
        return _select_numbered(
            title,
            displayed,
            input_stream=input_stream,
            output_stream=output_stream,
            invalid_message=invalid_message,
            show_title=show_title_in_fallback,
        )

    toolkit = _load_prompt_toolkit()

    if enhanced_input is None:
        try:
            enhanced_input = toolkit["create_input"](stdin=input_stream)
            enhanced_output = toolkit["create_output"](stdout=output_stream)
        except (AttributeError, OSError, RuntimeError, ValueError):
            return _select_numbered(
                title,
                displayed,
                input_stream=input_stream,
                output_stream=output_stream,
                invalid_message=invalid_message,
                show_title=show_title_in_fallback,
            )

    state = _SelectionState(displayed)
    try:
        application = _create_application(
            toolkit,
            _safe_text(title, maximum=_MAX_LABEL_CHARACTERS, fallback="Select"),
            state,
            enhanced_input,
            enhanced_output,
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        _restore_output(enhanced_output)
        return _select_numbered(
            title,
            displayed,
            input_stream=input_stream,
            output_stream=output_stream,
            invalid_message=invalid_message,
            show_title=show_title_in_fallback,
        )
    try:
        selected = await _run_application(application)
    finally:
        _restore_output(enhanced_output)
    return selected


async def select_many(
    title: str,
    options: tuple[SelectionOption[_Value], ...],
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    enhanced_input: Any = None,
    enhanced_output: Any = None,
    maximum: int = _MAX_MULTI_SELECTIONS,
    empty_message: str | None = None,
    maximum_message: str | None = None,
    invalid_message: str | None = None,
    fallback_prompt: str = "Choices (comma-separated numbers): ",
    show_title_in_fallback: bool = True,
) -> tuple[_Value, ...]:
    """Select stable values using enhanced navigation or numbered fallback."""

    displayed = _normalize_options(options)
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= _MAX_MULTI_SELECTIONS
    ):
        raise ValueError(
            f"multi-selection maximum must be from 1 through {_MAX_MULTI_SELECTIONS}"
        )
    if (enhanced_input is None) != (enhanced_output is None):
        raise ValueError("enhanced input and output must be supplied together")

    safe_title = _safe_text(
        title,
        maximum=_MAX_LABEL_CHARACTERS,
        fallback="Select",
    )
    limit = min(maximum, len(displayed))
    safe_empty_message = _safe_text(
        empty_message or "Select at least one option.",
        maximum=_MAX_VALIDATION_CHARACTERS,
        fallback="Select at least one option.",
    )
    safe_maximum_message = _safe_text(
        maximum_message or f"Select at most {limit} options.",
        maximum=_MAX_VALIDATION_CHARACTERS,
        fallback=f"Select at most {limit} options.",
    )
    safe_invalid_message = _safe_text(
        invalid_message or f"Choose 1 to {limit} distinct option numbers.",
        maximum=_MAX_VALIDATION_CHARACTERS,
        fallback=f"Choose 1 to {limit} distinct option numbers.",
    )
    safe_fallback_prompt = _safe_text(
        fallback_prompt,
        maximum=_MAX_LABEL_CHARACTERS,
        fallback="Choices (comma-separated numbers): ",
    )

    def numbered_fallback() -> tuple[_Value, ...]:
        return _select_numbered_many(
            safe_title,
            displayed,
            input_stream=input_stream,
            output_stream=output_stream,
            maximum=maximum,
            invalid_message=safe_invalid_message,
            prompt=safe_fallback_prompt,
            show_title=show_title_in_fallback,
        )

    if enhanced_input is None and not _streams_support_enhanced(
        input_stream,
        output_stream,
    ):
        return numbered_fallback()

    toolkit = _load_prompt_toolkit()

    if enhanced_input is None:
        try:
            enhanced_input = toolkit["create_input"](stdin=input_stream)
            enhanced_output = toolkit["create_output"](stdout=output_stream)
        except (AttributeError, OSError, RuntimeError, ValueError):
            return numbered_fallback()

    state = _MultiSelectionState(
        displayed,
        maximum=maximum,
        empty_message=safe_empty_message,
        maximum_message=safe_maximum_message,
    )
    try:
        application = _create_multi_application(
            toolkit,
            safe_title,
            state,
            enhanced_input,
            enhanced_output,
        )
    except Exception:
        _restore_output(enhanced_output)
        return numbered_fallback()
    except BaseException:
        _restore_output(enhanced_output)
        raise
    try:
        selected = await _run_application(application)
    finally:
        _restore_output(enhanced_output)
    return selected


def _load_prompt_toolkit() -> dict[str, Any]:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.input import create_input
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.output import create_output
        from prompt_toolkit.styles import Style
    except ImportError as error:
        raise ImportError(
            "Daita's terminal runtime dependency is unavailable. "
            f"{repair_guidance()}"
        ) from error

    return {
        "Application": Application,
        "FormattedTextControl": FormattedTextControl,
        "KeyBindings": KeyBindings,
        "Layout": Layout,
        "Style": Style,
        "Window": Window,
        "create_input": create_input,
        "create_output": create_output,
    }


def _create_application(
    toolkit: dict[str, Any],
    title: str,
    state: _SelectionState[Any],
    enhanced_input: Any,
    enhanced_output: Any,
) -> Any:
    from . import terminal_tui

    capabilities = terminal_tui._terminal_capabilities(enhanced_output)
    glyphs = terminal_tui._terminal_glyphs(capabilities)
    keys = toolkit["KeyBindings"]()

    def terminal_usable() -> bool:
        columns, rows = terminal_tui._terminal_size(enhanced_output)
        return columns >= 32 and rows >= 6

    def invalidate(event: Any) -> None:
        event.app.invalidate()

    @keys.add("up")
    def move_up(event: Any) -> None:
        state.move(-1)
        invalidate(event)

    @keys.add("down")
    def move_down(event: Any) -> None:
        state.move(1)
        invalidate(event)

    @keys.add("enter")
    def confirm(event: Any) -> None:
        if not terminal_usable():
            invalidate(event)
            return
        selected = state.selected_value()
        if selected is not _NO_SELECTION:
            event.app.exit(result=selected)

    @keys.add("backspace")
    @keys.add("c-h")
    def backspace(event: Any) -> None:
        state.backspace()
        invalidate(event)

    @keys.add("escape", eager=True)
    def cancel(event: Any) -> None:
        event.app.exit(exception=SelectionCancelled())

    @keys.add("c-c")
    def interrupt(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt())

    @keys.add("c-d")
    def end_of_file(event: Any) -> None:
        event.app.exit(exception=EOFError())

    @keys.add("c-l")
    def redraw(event: Any) -> None:
        event.app.renderer.clear()
        invalidate(event)

    @keys.add("<any>")
    def filter_character(event: Any) -> None:
        state.add_filter_text(event.data)
        invalidate(event)

    control = toolkit["FormattedTextControl"](
        lambda: _render_fragments(
            title,
            state,
            glyphs=glyphs,
            size=terminal_tui._terminal_size(enhanced_output),
        ),
        focusable=True,
        show_cursor=False,
    )
    window = toolkit["Window"](
        content=control,
        dont_extend_height=True,
        wrap_lines=False,
    )
    application = toolkit["Application"](
        layout=toolkit["Layout"](window),
        key_bindings=keys,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
        input=enhanced_input,
        output=enhanced_output,
        style=toolkit["Style"].from_dict(
            terminal_tui._semantic_style_rules(capabilities)
        ),
        terminal_size_polling_interval=terminal_tui._terminal_size_polling_interval(),
    )
    application.ttimeoutlen = 0.01
    return application


def _create_multi_application(
    toolkit: dict[str, Any],
    title: str,
    state: _MultiSelectionState[Any],
    enhanced_input: Any,
    enhanced_output: Any,
) -> Any:
    from . import terminal_tui

    capabilities = terminal_tui._terminal_capabilities(enhanced_output)
    glyphs = terminal_tui._terminal_glyphs(capabilities)
    keys = toolkit["KeyBindings"]()

    def terminal_usable() -> bool:
        columns, rows = terminal_tui._terminal_size(enhanced_output)
        return columns >= 32 and rows >= 6

    def invalidate(event: Any) -> None:
        event.app.invalidate()

    @keys.add("up")
    def move_up(event: Any) -> None:
        state.move(-1)
        invalidate(event)

    @keys.add("down")
    def move_down(event: Any) -> None:
        state.move(1)
        invalidate(event)

    @keys.add(" ")
    def toggle(event: Any) -> None:
        state.toggle()
        invalidate(event)

    @keys.add("enter")
    def confirm(event: Any) -> None:
        if not terminal_usable():
            invalidate(event)
            return
        selected = state.selected_values()
        if selected is not _NO_SELECTION:
            event.app.exit(result=selected)
        else:
            invalidate(event)

    @keys.add("backspace")
    @keys.add("c-h")
    def backspace(event: Any) -> None:
        state.backspace()
        invalidate(event)

    @keys.add("escape", eager=True)
    def cancel(event: Any) -> None:
        event.app.exit(exception=SelectionCancelled())

    @keys.add("c-c")
    def interrupt(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt())

    @keys.add("c-d")
    def end_of_file(event: Any) -> None:
        event.app.exit(exception=EOFError())

    @keys.add("c-l")
    def redraw(event: Any) -> None:
        event.app.renderer.clear()
        invalidate(event)

    @keys.add("<any>")
    def filter_character(event: Any) -> None:
        state.add_filter_text(event.data)
        invalidate(event)

    control = toolkit["FormattedTextControl"](
        lambda: _render_multi_fragments(
            title,
            state,
            glyphs=glyphs,
            size=terminal_tui._terminal_size(enhanced_output),
        ),
        focusable=True,
        show_cursor=False,
    )
    window = toolkit["Window"](
        content=control,
        dont_extend_height=True,
        wrap_lines=False,
    )
    application = toolkit["Application"](
        layout=toolkit["Layout"](window),
        key_bindings=keys,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
        input=enhanced_input,
        output=enhanced_output,
        style=toolkit["Style"].from_dict(
            terminal_tui._semantic_style_rules(capabilities)
        ),
        terminal_size_polling_interval=terminal_tui._terminal_size_polling_interval(),
    )
    application.ttimeoutlen = 0.01
    return application


async def _run_application(application: Any) -> Any:
    return await application.run_async()


def _render_fragments(
    title: str,
    state: _SelectionState[Any],
    *,
    glyphs: Any = None,
    size: tuple[int, int] | None = None,
) -> list[tuple[str, str]]:
    from . import terminal_tui

    glyphs = glyphs or terminal_tui._terminal_glyphs(
        terminal_tui._terminal_capabilities()
    )
    if size is not None and (size[0] < 32 or size[1] < 6):
        return _small_terminal_fragments(size, glyphs)
    narrow = size is not None and size[0] < 70
    help_text = (
        "  ↑/↓ move · Enter select · type to filter · Esc back\n"
        if glyphs.prompt == "›"
        else "  Up/Down move | Enter select | type to filter | Esc back\n"
    )
    fragments = [
        ("class:selection.identity", "DAITA SETUP\n"),
        ("class:selection.title", f"{title}\n\n"),
        ("class:selection.help", help_text),
    ]
    if state.filter_text:
        fragments.append(
            (
                "class:selection.filter",
                f"  Filter: {state.filter_text}\n",
            )
        )
    fragments.append(("", "\n"))
    visible = state.visible
    if not visible:
        fragments.append(("class:selection.empty", "  No matches\n"))
        return fragments
    state.position %= len(visible)
    for index, option in enumerate(visible):
        prefix = f"{glyphs.prompt} " if index == state.position else "  "
        description = (
            "" if narrow or not option.description else f"   {option.description}"
        )
        style = "class:selection.current" if index == state.position else ""
        fragments.append((style, f"{prefix}{option.label}{description}\n"))
        if narrow and option.description:
            fragments.append(("class:selection.help", f"    {option.description}\n"))
    return fragments


def _render_multi_fragments(
    title: str,
    state: _MultiSelectionState[Any],
    *,
    glyphs: Any = None,
    size: tuple[int, int] | None = None,
) -> list[tuple[str, str]]:
    from . import terminal_tui

    glyphs = glyphs or terminal_tui._terminal_glyphs(
        terminal_tui._terminal_capabilities()
    )
    if size is not None and (size[0] < 32 or size[1] < 6):
        return _small_terminal_fragments(size, glyphs)
    narrow = size is not None and size[0] < 70
    help_text = (
        "  ↑/↓ move · Space toggle · Enter continue · type to filter · Esc back\n"
        if glyphs.prompt == "›"
        else (
            "  Up/Down move | Space toggle | Enter continue | "
            "type to filter | Esc back\n"
        )
    )
    fragments = [
        ("class:selection.identity", "DAITA SETUP\n"),
        ("class:selection.title", f"{title}\n\n"),
        ("class:selection.help", help_text),
    ]
    if state.filter_text:
        fragments.append(
            (
                "class:selection.filter",
                f"  Filter: {state.filter_text}\n",
            )
        )
    if state.validation_message:
        fragments.append(
            (
                "class:selection.validation",
                f"  {state.validation_message}\n",
            )
        )
    fragments.append(("", "\n"))
    visible = state.visible
    if not visible:
        fragments.append(("class:selection.empty", "  No matches\n"))
        return fragments
    state.position %= len(visible)
    for index, option in enumerate(visible):
        prefix = f"{glyphs.prompt} " if index == state.position else "  "
        checked = "x" if option.identity in state.selected_identities else " "
        description = (
            "" if narrow or not option.description else f"   {option.description}"
        )
        style = "class:selection.current" if index == state.position else ""
        fragments.append((style, f"{prefix}[{checked}] {option.label}{description}\n"))
        if narrow and option.description:
            fragments.append(("class:selection.help", f"      {option.description}\n"))
    return fragments


def _small_terminal_fragments(
    size: tuple[int, int],
    glyphs: Any,
) -> list[tuple[str, str]]:
    columns, rows = size
    message = (
        f"{glyphs.warning} Terminal too small ({columns}x{rows}). "
        "Resize to at least 32x6."
    )
    safe = _safe_text(
        message,
        maximum=max(32, min(_MAX_VALIDATION_CHARACTERS, columns * 3)),
        fallback="Resize the terminal.",
    )
    return [
        ("class:selection.identity", "DAITA SETUP\n\n"),
        ("class:selection.validation", f"  {safe}\n"),
    ]


def _select_numbered(
    title: str,
    options: tuple[_DisplayedOption[_Value], ...],
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    invalid_message: str | None,
    show_title: bool,
) -> _Value:
    if show_title:
        print(title, file=output_stream)
        print(file=output_stream)
    for index, option in enumerate(options, start=1):
        suffix = f" · {option.description}" if option.description else ""
        print(f"{index}. {option.label}{suffix}", file=output_stream)
    message = invalid_message or f"Enter a number from 1 to {len(options)}."
    while True:
        print("Choice: ", end="", flush=True, file=output_stream)
        value = input_stream.readline()
        if value == "":
            raise EOFError
        try:
            selected = int(value.rstrip("\r\n").strip())
        except ValueError:
            selected = 0
        if 1 <= selected <= len(options):
            return options[selected - 1].value
        print(message, file=output_stream)


def _select_numbered_many(
    title: str,
    options: tuple[_DisplayedOption[_Value], ...],
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    maximum: int,
    invalid_message: str,
    prompt: str,
    show_title: bool,
) -> tuple[_Value, ...]:
    if show_title:
        print(title, file=output_stream)
        print(file=output_stream)
    for index, option in enumerate(options, start=1):
        suffix = f" · {option.description}" if option.description else ""
        print(f"{index}. {option.label}{suffix}", file=output_stream)
    while True:
        print(prompt, end="", flush=True, file=output_stream)
        value = input_stream.readline()
        if value == "":
            raise EOFError
        raw = value.rstrip("\r\n").strip()
        pieces = tuple(piece.strip() for piece in raw.split(","))
        try:
            indexes = (
                tuple(int(piece) for piece in pieces) if raw and all(pieces) else ()
            )
        except ValueError:
            indexes = ()
        if (
            indexes
            and len(indexes) <= maximum
            and len(indexes) == len(set(indexes))
            and all(1 <= index <= len(options) for index in indexes)
        ):
            return tuple(options[index - 1].value for index in indexes)
        print(invalid_message, file=output_stream)


def _normalize_options(
    options: tuple[SelectionOption[_Value], ...],
) -> tuple[_DisplayedOption[_Value], ...]:
    if not options:
        raise ValueError("selection requires at least one option")
    if len(options) > _MAX_OPTIONS:
        raise ValueError(f"selection supports at most {_MAX_OPTIONS} options")
    displayed = []
    for identity, option in enumerate(options):
        label = _safe_text(
            option.label,
            maximum=_MAX_LABEL_CHARACTERS,
            fallback="option",
        )
        description = _safe_text(
            option.description,
            maximum=_MAX_DESCRIPTION_CHARACTERS,
            fallback="",
        )
        terms = tuple(
            _safe_text(term, maximum=_MAX_LABEL_CHARACTERS, fallback="")
            for term in option.search_terms[:_MAX_SEARCH_TERMS]
        )
        searchable = " ".join((label, description, *terms)).casefold()
        displayed.append(
            _DisplayedOption(
                identity=identity,
                value=option.value,
                label=label,
                description=description,
                searchable=searchable,
            )
        )
    return tuple(displayed)


def _safe_text(value: object, *, maximum: int, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    projected = "".join(
        (
            character
            if character.isprintable()
            and unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
            else "?"
        )
        for character in value
    )
    if len(projected) > maximum:
        projected = projected[: max(0, maximum - 3)] + "..."
    return projected or fallback


def _streams_support_enhanced(input_stream: TextIO, output_stream: TextIO) -> bool:
    if input_stream is not sys.stdin or output_stream is not sys.stdout:
        return False
    if os.environ.get("TERM", "").strip().casefold() in {"dumb", "unknown"}:
        return False
    try:
        return (
            input_stream.isatty()
            and output_stream.isatty()
            and os.isatty(input_stream.fileno())
            and os.isatty(output_stream.fileno())
        )
    except (AttributeError, OSError, ValueError):
        return False


def _restore_output(output: Any) -> None:
    for method_name in (
        "reset_attributes",
        "reset_cursor_key_mode",
        "reset_cursor_shape",
        "enable_autowrap",
        "quit_alternate_screen",
        "show_cursor",
        "flush",
    ):
        try:
            getattr(output, method_name)()
        except Exception:
            continue


__all__ = ["SelectionCancelled", "SelectionOption", "select_many", "select_one"]
