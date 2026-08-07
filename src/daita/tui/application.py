"""Lazy prompt-toolkit composition and lifecycle for Daita's terminal UI."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import os
import re
import sys
from typing import Any, TextIO

from .._installation import repair_guidance
from ..capabilities import ApprovalDecision, ApprovalRequest
from ..loop.models import Transcript
from ..terminal_transcript import (
    TranscriptFollowState,
    TranscriptProjection,
    bounded_scroll_rows,
    bounded_selection_auto_scroll,
)
from .capabilities import (
    RUNNING_GLYPHS as _RUNNING_GLYPHS,
    ResponsiveProjection,
    TerminalCapabilities,
    TerminalGlyphs,
    mouse_reporting_available as _mouse_reporting_available,
    terminal_capabilities as _terminal_capabilities,
    terminal_glyphs as _terminal_glyphs,
    terminal_size as _terminal_size,
    terminal_size_polling_interval as _terminal_size_polling_interval,
)
from .clipboard import (
    FORCE_SELECTION_GUIDANCE as _FORCE_SELECTION_GUIDANCE,
    MAX_CLIPBOARD_UTF8_BYTES,
    ClipboardResult,
    deliver_clipboard as _deliver_clipboard,
)
from .rendering import (
    render_markdown_text as _render_markdown_text_with_runtime,
    semantic_style_rules as _semantic_style_rules,
)
from .state import (
    TerminalApprovalBridge,
    TerminalObserverBridge,
    TerminalStartupInfo,
    TerminalViewState,
    _model_text_event_fields,
)
from .shell import (
    StatusProjection,
    _approval_panel_for_request,
    _render_approval_panel_fragments,
    _resize_message_fragments,
    _responsive_for_output,
    _slash_command_menu_fragments,
    _slash_completion_maps,
    _status_left_fragments,
    _status_projection,
    _status_right_fragments,
    _status_single_line_fragments,
    _write_setup_prompt as _write_setup_prompt_impl,
    _write_setup_status as _write_setup_status_impl,
)
from .text import (
    display_width as _display_width,
    sanitize_terminal_text as _sanitize_terminal_text,
)
from .tool_view import (
    ToolCardDetails,
    ToolCardState,
    ToolTablePreview,
    _render_tool_card_fragments as _render_tool_card_fragments_impl,
)
from .transcript_view import (
    _EMPTY_RENDERED_TRANSCRIPT_MAP,
    _RenderedTranscriptMap,
    _fragment_line_metrics,
    _highlight_transcript_line,
    _render_startup_fragments,
    _render_transcript_fragments,
)

MAX_COMPOSER_CHARACTERS = 16_384
_MAX_COMPOSER_ROWS = 6
_MOUSE_SCROLL_LINES = 3
_SELECTION_AUTOSCROLL_LINES = 1
_SELECTION_COMPLETE_HINT = "Selected · Ctrl+C copy · Esc clear"
_MOUSE_FAILURE_NOTICE = (
    "Mouse interaction unavailable; keyboard controls remain active."
)
_ANIMATION_INTERVAL_SECONDS = 0.12
_STREAM_REPAINT_INTERVAL_SECONDS = 1 / 30
_PASTED_TEXT_PLACEHOLDER_PATTERN = re.compile(r"\[Pasted Text #[1-9][0-9]*\]")


class TerminalTUIUnavailable(RuntimeError):
    """The enhanced application could not be admitted for this terminal."""


class TerminalUserInputError(ValueError):
    """One recoverable composer input error that should not close the shell."""


@dataclass(frozen=True, slots=True)
class _ComposerDraft:
    """One process-local composer history entry with hidden pasted text."""

    text: str
    pasted_texts: tuple[tuple[str, str], ...] = ()
    next_paste_number: int = 1


@dataclass(frozen=True, slots=True)
class TerminalCommandResult:
    """The controller result of one suspended local slash command."""

    conversation_id: str | None
    action: str | None = None
    output: str = ""
    presentation: str = "local"
    source_summary: str | None = None
    model_message: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalApplicationResult:
    """The reason the focused shell yielded to its controller."""

    conversation_id: str | None
    action: str


class TerminalSuspendBridge:
    """Suspend the active TUI while an existing terminal prompt takes over."""

    def __init__(self) -> None:
        self._runner: (
            Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]] | None
        ) = None
        self.enhanced_input: Any = None
        self.enhanced_output: Any = None

    async def run(self, action: Callable[[], Awaitable[Any]]) -> Any:
        runner = self._runner
        if runner is None:
            return await action()
        return await runner(action)

    def install(
        self,
        runner: Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]],
        *,
        enhanced_input: Any,
        enhanced_output: Any,
    ) -> tuple[
        Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]] | None,
        Any,
        Any,
    ]:
        previous = (self._runner, self.enhanced_input, self.enhanced_output)
        self._runner = runner
        self.enhanced_input = enhanced_input
        self.enhanced_output = enhanced_output
        return previous

    def restore(
        self,
        previous: tuple[
            Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]] | None,
            Any,
            Any,
        ],
    ) -> None:
        self._runner, self.enhanced_input, self.enhanced_output = previous


def supports_terminal_tui(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    enhanced_input: Any = None,
    enhanced_output: Any = None,
) -> bool:
    """Return whether the full-screen enhanced shell can own these streams."""

    if (enhanced_input is None) != (enhanced_output is None):
        raise ValueError("TUI input and output must be supplied together")
    if enhanced_input is not None:
        return True
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


async def run_terminal_tui(
    state: TerminalViewState,
    *,
    run_message: Callable[[str, str | None], Awaitable[Any]],
    load_transcript: Callable[[str], Awaitable[Transcript]] | None = None,
    handle_command: Callable[[str, str | None], Awaitable[TerminalCommandResult]],
    command_requires_suspension: Callable[[str], bool] | None = None,
    skill_completions: Sequence[tuple[str, str]] = (),
    load_skill_completions: (
        Callable[[], Awaitable[Sequence[tuple[str, str]]]] | None
    ) = None,
    source_completions: Sequence[tuple[str, str, str]] = (),
    load_source_completions: (
        Callable[[], Awaitable[Sequence[tuple[str, str, str]]]] | None
    ) = None,
    input_stream: TextIO,
    output_stream: TextIO,
    suspend_bridge: TerminalSuspendBridge,
    observer_bridge: TerminalObserverBridge | None = None,
    approval_bridge: TerminalApprovalBridge | None = None,
    enhanced_input: Any = None,
    enhanced_output: Any = None,
) -> TerminalApplicationResult:
    """Run the ready-agent shell until exit or a controller-level transition."""

    if not isinstance(state, TerminalViewState):
        raise TypeError("state must be TerminalViewState")
    if (enhanced_input is None) != (enhanced_output is None):
        raise ValueError("TUI input and output must be supplied together")
    if command_requires_suspension is not None and not callable(
        command_requires_suspension
    ):
        raise TypeError("command_requires_suspension must be callable")
    observer_bridge = observer_bridge or TerminalObserverBridge()

    runtime = _load_terminal_runtime()
    owns_input = enhanced_input is None
    if enhanced_input is None:
        try:
            enhanced_input = runtime["create_input"](stdin=input_stream)
            enhanced_output = runtime["create_output"](stdout=output_stream)
        except Exception as error:
            _restore_terminal(enhanced_output)
            if enhanced_input is not None:
                try:
                    enhanced_input.close()
                except Exception:
                    pass
            raise TerminalTUIUnavailable(
                "enhanced terminal admission failed"
            ) from error
        except BaseException:
            _restore_terminal(enhanced_output)
            if enhanced_input is not None:
                try:
                    enhanced_input.close()
                except Exception:
                    pass
            raise

    try:
        application, approval_previous, cancel_pending_approval = _create_application(
            runtime,
            state,
            run_message=run_message,
            load_transcript=load_transcript,
            handle_command=handle_command,
            command_requires_suspension=command_requires_suspension,
            skill_completions=skill_completions,
            load_skill_completions=load_skill_completions,
            source_completions=source_completions,
            load_source_completions=load_source_completions,
            observer_bridge=observer_bridge,
            approval_bridge=approval_bridge,
            enhanced_input=enhanced_input,
            enhanced_output=enhanced_output,
        )
    except Exception as error:
        _restore_terminal(enhanced_output)
        if owns_input:
            try:
                enhanced_input.close()
            except Exception:
                pass
        raise TerminalTUIUnavailable("enhanced terminal admission failed") from error
    except BaseException:
        _restore_terminal(enhanced_output)
        if owns_input:
            try:
                enhanced_input.close()
            except Exception:
                pass
        raise

    async def suspend(action: Callable[[], Awaitable[Any]]) -> Any:
        async with runtime["in_terminal"]():
            return await action()

    previous = suspend_bridge.install(
        suspend,
        enhanced_input=enhanced_input,
        enhanced_output=enhanced_output,
    )
    event_task = asyncio.create_task(
        _consume_observer_events(
            observer_bridge,
            state,
            application,
        )
    )
    application_failure: BaseException | None = None
    try:
        try:
            result = await _run_application(application)
        except BaseException as error:
            application_failure = error
            raise
        if not isinstance(result, TerminalApplicationResult):
            raise RuntimeError("terminal application returned an invalid result")
        return result
    finally:
        try:
            if application_failure is not None and not isinstance(
                application_failure,
                (asyncio.CancelledError, KeyboardInterrupt, SystemExit),
            ):
                # Restore the user's terminal before waiting for authoritative
                # execution to settle. Presentation failure must not cancel it.
                _restore_application(application, enhanced_output)
            cancel_pending_approval()
            await asyncio.sleep(0)
            active = state.active_task
            if active is not None and not active.done():
                if application_failure is not None and not isinstance(
                    application_failure,
                    (asyncio.CancelledError, KeyboardInterrupt, SystemExit),
                ):
                    try:
                        await asyncio.shield(active)
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
                    active.cancel()
                    try:
                        await active
                    except (asyncio.CancelledError, Exception):
                        pass
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
            _project_pending_events(observer_bridge, state)
            state.settle_cancelled_run()
            state.active_task = None
        finally:
            suspend_bridge.restore(previous)
            if approval_bridge is not None:
                approval_bridge.restore(approval_previous)
            _restore_application(application, enhanced_output)
            if owns_input:
                try:
                    enhanced_input.close()
                except Exception:
                    pass


def _load_terminal_runtime() -> dict[str, Any]:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.application.run_in_terminal import in_terminal
        from prompt_toolkit.completion import CompleteEvent, Completion
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.formatted_text import ANSI, FormattedText
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.input import create_input
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import (
            ConditionalContainer,
            HSplit,
            VSplit,
            Window,
        )
        from prompt_toolkit.layout.controls import FormattedTextControl, UIContent
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.mouse_events import MouseEventType
        from prompt_toolkit.output import create_output
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, TextArea
        from prompt_toolkit.keys import Keys
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.text import Text
        from rich.theme import Theme
    except (AttributeError, ImportError) as error:
        raise ImportError(
            "Daita's terminal runtime dependency is unavailable. "
            f"{repair_guidance()}"
        ) from error

    return {
        "ANSI": ANSI,
        "Application": Application,
        "Condition": Condition,
        "ConditionalContainer": ConditionalContainer,
        "Console": Console,
        "Dimension": Dimension,
        "FormattedText": FormattedText,
        "FormattedTextControl": FormattedTextControl,
        "Frame": Frame,
        "HSplit": HSplit,
        "InMemoryHistory": InMemoryHistory,
        "KeyBindings": KeyBindings,
        "Keys": Keys,
        "Layout": Layout,
        "Markdown": Markdown,
        "MouseEventType": MouseEventType,
        "Point": Point,
        "CompleteEvent": CompleteEvent,
        "Completion": Completion,
        "Style": Style,
        "Syntax": Syntax,
        "Table": Table,
        "TextArea": TextArea,
        "Text": Text,
        "Theme": Theme,
        "UIContent": UIContent,
        "VSplit": VSplit,
        "Window": Window,
        "create_input": create_input,
        "create_output": create_output,
        "in_terminal": in_terminal,
    }


def _write_setup_prompt(output_stream: TextIO, prompt: object) -> None:
    _write_setup_prompt_impl(
        _load_terminal_runtime(),
        output_stream,
        prompt,
    )


def _write_setup_status(
    output_stream: TextIO,
    value: object,
    *,
    role: str,
) -> None:
    _write_setup_status_impl(
        _load_terminal_runtime(),
        output_stream,
        value,
        role=role,
    )


def _create_application(
    runtime: dict[str, Any],
    state: TerminalViewState,
    *,
    run_message: Callable[[str, str | None], Awaitable[Any]],
    load_transcript: Callable[[str], Awaitable[Transcript]] | None,
    handle_command: Callable[[str, str | None], Awaitable[TerminalCommandResult]],
    command_requires_suspension: Callable[[str], bool] | None = None,
    skill_completions: Sequence[tuple[str, str]] = (),
    load_skill_completions: (
        Callable[[], Awaitable[Sequence[tuple[str, str]]]] | None
    ) = None,
    source_completions: Sequence[tuple[str, str, str]] = (),
    load_source_completions: (
        Callable[[], Awaitable[Sequence[tuple[str, str, str]]]] | None
    ) = None,
    observer_bridge: TerminalObserverBridge,
    approval_bridge: TerminalApprovalBridge | None,
    enhanced_input: Any,
    enhanced_output: Any,
) -> tuple[
    Any,
    Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None,
    Callable[[], None],
]:
    capabilities = _terminal_capabilities(enhanced_output)
    glyphs = _terminal_glyphs(capabilities)
    mouse_reporting = _mouse_reporting_available(enhanced_output)
    if not mouse_reporting:
        state.notice = _MOUSE_FAILURE_NOTICE
    keys = runtime["KeyBindings"]()
    current_skill_completions = tuple(skill_completions)
    current_source_completions = tuple(source_completions)
    composer_buffer: Any = None

    def completion_is_active() -> bool:
        if composer_buffer is None:
            return False
        text = composer_buffer.document.text_before_cursor
        return text.startswith(("/", "@")) and "\n" not in text

    completion_filter = runtime["Condition"](completion_is_active)

    def composer_completer() -> Any:
        slash_display, slash_descriptions = _slash_completion_maps(
            current_skill_completions
        )
        source_display = {
            insertion: display
            for insertion, display, _description in current_source_completions
        }
        source_descriptions = {
            insertion: description
            for insertion, _display, description in current_source_completions
        }

        class ComposerCompleter:
            def get_completions(
                self,
                document: Any,
                complete_event: Any,
            ) -> Any:
                del complete_event
                text = document.text_before_cursor
                if "\n" in text:
                    return
                if text.startswith("/"):
                    display = slash_display
                    descriptions = slash_descriptions
                    query = text.casefold()
                elif text.startswith("@"):
                    display = source_display
                    descriptions = source_descriptions
                    query = text.casefold()
                    if query.startswith('@"'):
                        query = "@" + query[2:]
                else:
                    return
                for insertion, label in display.items():
                    searchable_insertion = insertion.casefold()
                    searchable_label = label.casefold()
                    if not (
                        searchable_insertion.startswith(text.casefold())
                        or searchable_label.startswith(query)
                    ):
                        continue
                    yield runtime["Completion"](
                        insertion,
                        start_position=-len(text),
                        display=label,
                        display_meta=descriptions[insertion],
                    )

            async def get_completions_async(
                self,
                document: Any,
                complete_event: Any,
            ) -> Any:
                for completion in self.get_completions(document, complete_event):
                    yield completion

        return ComposerCompleter()

    composer = runtime["TextArea"](
        multiline=True,
        wrap_lines=True,
        height=runtime["Dimension"](min=1, max=_MAX_COMPOSER_ROWS),
        dont_extend_height=True,
        prompt=runtime["FormattedText"]([("class:tui.prompt", f"{glyphs.prompt} ")]),
        style="class:tui.composer",
        name="composer",
        completer=composer_completer(),
        complete_while_typing=completion_filter,
        history=runtime["InMemoryHistory"](),
    )
    composer_buffer = composer.buffer

    async def refresh_completions(*, include_sources: bool) -> None:
        nonlocal current_skill_completions, current_source_completions
        changed = False
        if load_skill_completions is not None:
            try:
                current_skill_completions = tuple(await load_skill_completions())
            except (asyncio.CancelledError, Exception):
                pass
            else:
                changed = True
        if include_sources and load_source_completions is not None:
            try:
                current_source_completions = tuple(await load_source_completions())
            except (asyncio.CancelledError, Exception):
                pass
            else:
                changed = True
        if changed:
            composer.buffer.completer = composer_completer()

    enforcing_bound = False
    last_valid_composer_document = composer.buffer.document
    pasted_texts: dict[str, str] = {}
    next_paste_number = 1
    input_history: list[_ComposerDraft] = []
    history_position = 0
    history_draft = _ComposerDraft("")
    content_line_count = 1
    content_last_line_width = 0
    rendered_transcript_map = _EMPTY_RENDERED_TRANSCRIPT_MAP
    transcript_cache_key: tuple[int, int, int, ResponsiveProjection] | None = None
    transcript_cache_fragments: list[tuple[str, str]] | None = None
    transcript_base_content_key: (
        tuple[int, int, int, ResponsiveProjection, int] | None
    ) = None
    transcript_base_content: Any = None
    responsive_projection = _responsive_for_output(enhanced_output, state)
    mouse_press_owner: str | None = None

    def complete_captured_transcript_drag(*, show_hint: bool) -> None:
        selected = state.transcript_selection.end_drag()
        state.transient_selection_hint = (
            _SELECTION_COMPLETE_HINT
            if show_hint and selected is not None and bool(selected.text)
            else ""
        )

    def mouse_action_owned(owner: str, event_type: Any) -> bool:
        """Capture one press sequence for exactly one visible presentation owner."""

        nonlocal mouse_press_owner
        if event_type == runtime["MouseEventType"].MOUSE_DOWN:
            if mouse_press_owner == "transcript" and owner != "transcript":
                complete_captured_transcript_drag(show_hint=False)
            mouse_press_owner = owner
            state.transient_selection_hint = ""
            return True
        if event_type == runtime["MouseEventType"].MOUSE_MOVE:
            return mouse_press_owner == owner
        if event_type == runtime["MouseEventType"].MOUSE_UP:
            previous_owner = mouse_press_owner
            mouse_press_owner = None
            if previous_owner == "transcript" and owner != "transcript":
                complete_captured_transcript_drag(show_hint=True)
            return previous_owner == owner
        return True

    def mouse_failed() -> None:
        """Contain pointer failures inside disposable presentation state."""

        nonlocal mouse_press_owner
        mouse_press_owner = None
        state.transcript_selection.end_drag()
        state.transient_selection_hint = ""
        state.notice = _MOUSE_FAILURE_NOTICE
        try:
            application.invalidate()
        except Exception:
            pass

    def approval_owns_mouse(mouse_event: Any) -> bool:
        panel = state.approval_panel
        if panel is None:
            return False
        event_type = mouse_event.event_type
        state.transient_selection_hint = ""
        if event_type == runtime["MouseEventType"].SCROLL_UP:
            panel.move(-_MOUSE_SCROLL_LINES)
        elif event_type == runtime["MouseEventType"].SCROLL_DOWN:
            panel.move(_MOUSE_SCROLL_LINES)
        elif getattr(mouse_event.button, "name", "") == "LEFT":
            owned = mouse_action_owned("approval", event_type)
            if event_type == runtime["MouseEventType"].MOUSE_UP and owned:
                remind_approval(application)
                return True
        else:
            return True
        application.invalidate()
        return True

    def responsive() -> ResponsiveProjection:
        return responsive_projection

    def refresh_responsive_projection(_application: Any) -> None:
        nonlocal responsive_projection
        responsive_projection = _responsive_for_output(enhanced_output, state)

    def terminal_is_usable() -> bool:
        return _responsive_for_output(enhanced_output, state).usable

    def rendered_terminal_is_usable() -> bool:
        return responsive().usable

    def current_composer_draft(*, text: str | None = None) -> _ComposerDraft:
        return _ComposerDraft(
            composer.buffer.text if text is None else text,
            tuple(pasted_texts.items()),
            next_paste_number,
        )

    def restore_composer_draft(draft: _ComposerDraft) -> None:
        nonlocal pasted_texts, next_paste_number
        pasted_texts = dict(draft.pasted_texts)
        next_paste_number = draft.next_paste_number
        _replace_composer_text(composer.buffer, draft.text)

    def composer_inline_capacity() -> int:
        columns, _rows = _terminal_size(enhanced_output)
        prompt_width = _display_width(f"{glyphs.prompt} ")
        return max(1, columns - prompt_width)

    def prune_unreferenced_pasted_texts(buffer: Any) -> None:
        if not pasted_texts:
            return
        active_placeholders = set(_PASTED_TEXT_PLACEHOLDER_PATTERN.findall(buffer.text))
        removed = tuple(
            placeholder
            for placeholder in pasted_texts
            if placeholder not in active_placeholders
        )
        if not removed:
            return
        for placeholder in removed:
            del pasted_texts[placeholder]

    def enforce_bound(buffer: Any) -> None:
        nonlocal enforcing_bound, last_valid_composer_document
        if enforcing_bound:
            return
        prune_unreferenced_pasted_texts(buffer)
        if len(buffer.text) <= MAX_COMPOSER_CHARACTERS and (
            not pasted_texts
            or len(_materialize_pasted_texts(buffer.text, pasted_texts))
            <= MAX_COMPOSER_CHARACTERS
        ):
            if history_position < len(input_history):
                input_history[history_position] = current_composer_draft(
                    text=buffer.text
                )
            last_valid_composer_document = buffer.document
            return
        enforcing_bound = True
        try:
            buffer.set_document(
                last_valid_composer_document,
                bypass_readonly=True,
            )
            state.notice = f"Input is limited to {MAX_COMPOSER_CHARACTERS} characters."
        finally:
            enforcing_bound = False

    composer.buffer.on_text_changed += enforce_bound

    composer_mouse_handler_base = composer.control.mouse_handler

    def composer_mouse_handler(mouse_event: Any) -> Any:
        try:
            if approval_owns_mouse(mouse_event):
                return None
            event_type = mouse_event.event_type
            if getattr(mouse_event.button, "name", "") == "LEFT" and event_type in {
                runtime["MouseEventType"].MOUSE_DOWN,
                runtime["MouseEventType"].MOUSE_MOVE,
                runtime["MouseEventType"].MOUSE_UP,
            }:
                if not mouse_action_owned("composer", event_type):
                    return None
            elif event_type in {
                runtime["MouseEventType"].SCROLL_UP,
                runtime["MouseEventType"].SCROLL_DOWN,
            }:
                state.transient_selection_hint = ""
            return composer_mouse_handler_base(mouse_event)
        except Exception:
            mouse_failed()
            return None

    composer.control.mouse_handler = composer_mouse_handler

    def transcript_fragments() -> list[tuple[str, str]]:
        nonlocal content_last_line_width, content_line_count
        nonlocal rendered_transcript_map
        nonlocal transcript_cache_fragments, transcript_cache_key
        projection = responsive()
        key = (
            state.transcript_render_generation,
            state.transcript_document.generation,
            len(state.blocks),
            projection,
        )
        if transcript_cache_key != key or transcript_cache_fragments is None:
            row_maps: list[_RenderedTranscriptMap] = []
            try:
                fragments = _render_transcript_fragments(
                    runtime,
                    state,
                    width=projection.content_width,
                    responsive=projection,
                    capabilities=capabilities,
                    glyphs=glyphs,
                    rendered_transcript_maps=row_maps,
                    highlight_selection=False,
                )
            except Exception:
                state.notice = "Some terminal content could not be rendered."
                rendered_transcript_map = _EMPTY_RENDERED_TRANSCRIPT_MAP
                fragments = [
                    (
                        "class:tui.status.failure",
                        f"\n {glyphs.failure} Content unavailable\n",
                    )
                ]
            else:
                rendered_transcript_map = (
                    row_maps[0] if row_maps else _EMPTY_RENDERED_TRANSCRIPT_MAP
                )
            content_line_count, content_last_line_width = _fragment_line_metrics(
                fragments
            )
            transcript_cache_key = (
                state.transcript_render_generation,
                state.transcript_document.generation,
                len(state.blocks),
                projection,
            )
            transcript_cache_fragments = fragments

        assert transcript_cache_key is not None
        assert transcript_cache_fragments is not None
        return transcript_cache_fragments

    semantic_style = runtime["Style"].from_dict(_semantic_style_rules(capabilities))

    approval_waiter: asyncio.Future[ApprovalDecision] | None = None
    approval_lock = asyncio.Lock()
    clipboard_task: asyncio.Task[Any] | None = None

    def clear_approval_view() -> None:
        state.approval_panel = None
        state.run_status = "working" if state.running else "ready"
        try:
            application.layout.focus(composer)
            application.invalidate()
        except Exception:
            pass

    def resolve_approval(decision: ApprovalDecision) -> None:
        nonlocal approval_waiter
        waiter = approval_waiter
        if waiter is None or waiter.done():
            return
        waiter.set_result(decision)
        clear_approval_view()

    def fail_approval(error: Exception) -> None:
        nonlocal approval_waiter
        waiter = approval_waiter
        if waiter is None or waiter.done():
            return
        waiter.set_exception(error)
        clear_approval_view()

    def cancel_approval() -> None:
        nonlocal approval_waiter
        waiter = approval_waiter
        if waiter is None or waiter.done():
            return
        waiter.cancel()
        clear_approval_view()

    def approval_fragments() -> list[tuple[str, str]]:
        panel = state.approval_panel
        if panel is None:
            return []
        try:
            fragments = _render_approval_panel_fragments(
                panel,
                glyphs=glyphs,
            )
        except Exception as error:
            fail_approval(error)
            return [
                (
                    "class:tui.approval.failure",
                    " Approval unavailable: review rendering failed.\n",
                )
            ]
        panel.rendered_line_count = max(
            1,
            sum(text.count("\n") for _style, text in fragments) + 1,
        )
        panel.cursor_line = min(
            panel.cursor_line,
            max(0, panel.rendered_line_count - 1),
        )
        return fragments

    approval_control = runtime["FormattedTextControl"](
        approval_fragments,
        focusable=True,
        show_cursor=False,
        get_cursor_position=lambda: runtime["Point"](
            x=0,
            y=(
                state.approval_panel.cursor_line
                if state.approval_panel is not None
                else 0
            ),
        ),
    )

    def approval_mouse_handler(mouse_event: Any) -> Any:
        try:
            return None if approval_owns_mouse(mouse_event) else NotImplemented
        except Exception:
            mouse_failed()
            return None

    approval_control.mouse_handler = approval_mouse_handler
    approval_window = runtime["Window"](
        content=approval_control,
        wrap_lines=True,
        always_hide_cursor=True,
        height=runtime["Dimension"](min=5, max=12, preferred=8),
        style="class:tui.approval",
    )
    approval_filter = runtime["Condition"](lambda: state.approval_panel is not None)

    async def present_approval(request: ApprovalRequest) -> ApprovalDecision:
        nonlocal approval_waiter, mouse_press_owner
        async with approval_lock:
            panel = _approval_panel_for_request(request)
            if panel is None:
                state.notice = (
                    "Approval unavailable: exact arguments cannot be reviewed safely."
                )
                try:
                    application.invalidate()
                except Exception:
                    pass
                raise RuntimeError("approval arguments cannot be reviewed safely")
            loop = asyncio.get_running_loop()
            waiter = loop.create_future()
            approval_waiter = waiter
            mouse_press_owner = None
            state.transcript_selection.end_drag()
            state.transient_selection_hint = ""
            state.approval_panel = panel
            state.run_status = "approval"
            try:
                approval_fragments()
                if state.approval_panel is not None:
                    application.layout.focus(approval_window)
                    application.invalidate()
                decision = await waiter
            finally:
                if state.approval_panel is panel:
                    clear_approval_view()
                if approval_waiter is waiter:
                    approval_waiter = None
            if not isinstance(decision, ApprovalDecision):
                raise TypeError("approval presenter must return ApprovalDecision")
            return decision

    def cancel_pending_approval() -> None:
        cancel_approval()

    def invalidate(application: Any) -> None:
        application.invalidate()

    async def execute_message(
        application: Any,
        message: str,
        *,
        settle_task: bool = True,
    ) -> None:
        try:
            result = await run_message(message, state.conversation_id)
            _project_pending_events(observer_bridge, state)
            if result is None:
                partial_removed = state.settle_cancelled_run()
                state.notice = (
                    "Partial assistant output was interrupted and was not recorded."
                    if partial_removed
                    else "Run interrupted; returning to the composer."
                )
            else:
                hydration_notice = ""
                run_id = getattr(result, "run_id", None)
                if load_transcript is not None and isinstance(run_id, str) and run_id:
                    try:
                        transcript = await load_transcript(run_id)
                    except asyncio.CancelledError:
                        _clear_current_task_cancellation()
                        hydration_notice = (
                            "Run completed; recorded tool details are unavailable."
                        )
                    except Exception:
                        hydration_notice = (
                            "Run completed; recorded tool details are unavailable."
                        )
                    else:
                        try:
                            state.hydrate_transcript(transcript, run_id=run_id)
                        except Exception:
                            hydration_notice = (
                                "Run completed; recorded tool details are unavailable."
                            )
                state.apply_result(result)
                if hydration_notice:
                    state.notice = hydration_notice
        except asyncio.CancelledError:
            _project_pending_events(observer_bridge, state)
            partial_removed = state.settle_cancelled_run()
            state.notice = (
                "Partial assistant output was interrupted and was not recorded."
                if partial_removed
                else "Run interrupted; returning to the composer."
            )
        except TerminalUserInputError as error:
            state.append_plain("local", str(error))
            state.run_status = "ready"
        except BaseException as error:
            application.exit(exception=error)
            return
        finally:
            if settle_task:
                await refresh_completions(include_sources=False)
                _project_pending_events(observer_bridge, state)
                state.running = False
                state.active_task = None
                invalidate(application)

    async def execute_command(application: Any, command: str) -> None:
        try:
            if command_requires_suspension is not None and command_requires_suspension(
                command
            ):
                async with runtime["in_terminal"]():
                    result = await handle_command(command, state.conversation_id)
            else:
                result = await handle_command(command, state.conversation_id)
            state.select_conversation(result.conversation_id)
            if result.source_summary is not None:
                state.source_summary = _sanitize_terminal_text(
                    result.source_summary,
                    maximum=128,
                    preserve_lines=False,
                    fallback="source",
                )
            if result.model_message is not None:
                await execute_message(
                    application,
                    result.model_message,
                    settle_task=False,
                )
                return
            state.append_local(result.presentation, result.output)
            if result.action is not None:
                application.exit(
                    result=TerminalApplicationResult(
                        conversation_id=result.conversation_id,
                        action=result.action,
                    )
                )
                return
            state.notice = ""
        except BaseException as error:
            application.exit(exception=error)
            return
        finally:
            await refresh_completions(include_sources=True)
            _project_pending_events(observer_bridge, state)
            state.running = False
            state.active_task = None
            invalidate(application)

    def start_task(application: Any, coroutine: Awaitable[None]) -> None:
        state.running = True
        state.run_status = "working"
        state.notice = ""
        state.transient_selection_hint = ""
        state.active_task = application.create_background_task(coroutine)
        invalidate(application)

    def remind_approval(application: Any) -> None:
        state.notice = "Press Y to approve once or N to deny."
        invalidate(application)

    def composer_selection_text() -> str:
        if composer.buffer.selection_state is None:
            return ""
        _document, clipboard_data = composer.buffer.document.cut_selection()
        return clipboard_data.text

    async def copy_text(application: Any, text: str) -> None:
        nonlocal clipboard_task
        try:
            result = await _deliver_clipboard(text, output=enhanced_output)
        except asyncio.CancelledError:
            raise
        except Exception:
            result = ClipboardResult(
                "failure",
                "none",
                f"Copy failed. {_FORCE_SELECTION_GUIDANCE}",
            )
        finally:
            clipboard_task = None
        state.notice = result.message
        invalidate(application)

    def request_copy(application: Any, text: str) -> None:
        nonlocal clipboard_task
        state.transient_selection_hint = ""
        if clipboard_task is not None and not clipboard_task.done():
            state.notice = "A clipboard copy is already in progress."
            invalidate(application)
            return
        clipboard_task = application.create_background_task(
            copy_text(application, text)
        )

    @keys.add("c-m", eager=True)
    def submit(event: Any) -> None:
        nonlocal history_position, history_draft
        nonlocal pasted_texts, next_paste_number
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        if not terminal_is_usable():
            state.notice = "Resize the terminal before submitting input."
            invalidate(event.app)
            return
        active = state.active_task
        if state.running or (active is not None and not active.done()):
            state.notice = "A run is already active; Ctrl-C cancels it."
            invalidate(event.app)
            return
        display_message = composer.buffer.text.strip()
        if not display_message:
            state.notice = "Enter a message before submitting."
            invalidate(event.app)
            return
        message = _materialize_pasted_texts(
            display_message,
            pasted_texts,
        ).strip()
        if not message:
            state.notice = "Enter a message before submitting."
            invalidate(event.app)
            return
        if len(message) > MAX_COMPOSER_CHARACTERS:
            state.notice = f"Input is limited to {MAX_COMPOSER_CHARACTERS} characters."
            invalidate(event.app)
            return
        input_history.append(current_composer_draft(text=display_message))
        history_position = len(input_history)
        history_draft = _ComposerDraft("")
        composer.buffer.reset(append_to_history=True)
        pasted_texts = {}
        next_paste_number = 1
        state.append_user(message)
        if display_message.startswith("/"):
            start_task(event.app, execute_command(event.app, message))
            return
        start_task(event.app, execute_message(event.app, message))

    @keys.add("c-j", eager=True)
    def insert_newline(event: Any) -> None:
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        if len(composer.buffer.text) < MAX_COMPOSER_CHARACTERS:
            composer.buffer.insert_text("\n")
        else:
            state.notice = f"Input is limited to {MAX_COMPOSER_CHARACTERS} characters."
        invalidate(event.app)

    @keys.add(runtime["Keys"].BracketedPaste, eager=True)
    def paste(event: Any) -> None:
        nonlocal next_paste_number
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        if not data:
            return
        buffer = composer.buffer
        document = buffer.document
        current_line = (
            document.current_line_before_cursor
            + data
            + document.current_line_after_cursor
        )
        if (
            "\n" not in data
            and _display_width(current_line) <= composer_inline_capacity()
        ):
            buffer.insert_text(data)
            invalidate(event.app)
            return

        placeholder = f"[Pasted Text #{next_paste_number}]"
        while placeholder in buffer.text or placeholder in pasted_texts:
            next_paste_number += 1
            placeholder = f"[Pasted Text #{next_paste_number}]"
        if len(buffer.text) + len(placeholder) > MAX_COMPOSER_CHARACTERS:
            state.notice = f"Input is limited to {MAX_COMPOSER_CHARACTERS} characters."
            invalidate(event.app)
            return

        cursor = document.cursor_position
        candidate_display = (
            document.text[:cursor] + placeholder + document.text[cursor:]
        )
        candidate_pastes = dict(pasted_texts)
        candidate_pastes[placeholder] = ""
        base_characters = len(
            _materialize_pasted_texts(candidate_display, candidate_pastes)
        )
        available_characters = MAX_COMPOSER_CHARACTERS - base_characters
        if available_characters <= 0:
            state.notice = f"Input is limited to {MAX_COMPOSER_CHARACTERS} characters."
            invalidate(event.app)
            return

        stored = data[:available_characters]
        pasted_texts[placeholder] = stored
        next_paste_number += 1
        buffer.insert_text(placeholder)
        if len(stored) < len(data):
            state.notice = (
                f"{placeholder} was limited to "
                f"{MAX_COMPOSER_CHARACTERS} message characters."
            )
        else:
            state.notice = f"Stored as {placeholder}."
        invalidate(event.app)

    @keys.add("c-c", eager=True)
    def interrupt(event: Any) -> None:
        if state.approval_panel is not None:
            cancel_approval()
            return
        if state.transcript_selection.active:
            request_copy(event.app, state.transcript_selection.text)
            return
        composer_text = composer_selection_text()
        if composer_text:
            request_copy(event.app, composer_text)
            return
        active = state.active_task
        if active is not None and not active.done():
            state.notice = "Cancelling the active run…"
            active.cancel()
        else:
            state.notice = "Input interrupted; composer remains active."
        invalidate(event.app)

    @keys.add("c-d", eager=True)
    def end_of_file(event: Any) -> None:
        if state.approval_panel is not None:
            cancel_approval()
            return
        if composer.buffer.text:
            composer.buffer.delete()
            return
        active = state.active_task
        if state.running or (active is not None and not active.done()):
            state.notice = "A run is active; Ctrl-C cancels it."
            invalidate(event.app)
            return
        event.app.exit(
            result=TerminalApplicationResult(
                conversation_id=state.conversation_id,
                action="exit",
            )
        )

    def current_transcript_projection() -> TranscriptProjection | None:
        if state.blocks:
            transcript_fragments()
        return state.transcript_projection

    def scroll_transcript(lines: int) -> None:
        projection = current_transcript_projection()
        if projection is None:
            return
        height = _viewport_height(content_window)
        direction = -1 if lines > 0 else 1 if lines < 0 else 0
        movement = bounded_scroll_rows(direction * abs(lines), viewport_rows=height)
        current = viewport_rendered_top(content_window)
        latest = max(0, content_line_count - height)
        target = min(latest, max(0, current + movement))
        if direction > 0 and target >= latest:
            state.transcript_viewport.follow_latest()
        elif target != current:
            position = rendered_transcript_map.position_for_row(target)
            if position is not None:
                state.transcript_viewport.review_position(
                    state.transcript_document,
                    position,
                )

    @keys.add("pageup", eager=True)
    def page_up(event: Any) -> None:
        panel = state.approval_panel
        if panel is not None:
            panel.move(-max(1, _viewport_height(approval_window)))
        else:
            scroll_transcript(_viewport_height(content_window))
        invalidate(event.app)

    @keys.add("pagedown", eager=True)
    def page_down(event: Any) -> None:
        panel = state.approval_panel
        if panel is not None:
            panel.move(max(1, _viewport_height(approval_window)))
        else:
            scroll_transcript(-_viewport_height(content_window))
        invalidate(event.app)

    @keys.add("c-home", eager=True)
    def transcript_home(event: Any) -> None:
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        projection = current_transcript_projection()
        if projection is not None:
            state.transcript_viewport.review_start(projection)
        invalidate(event.app)

    @keys.add("c-end", eager=True)
    def transcript_end(event: Any) -> None:
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        state.transcript_viewport.follow_latest()
        invalidate(event.app)

    @keys.add("c-o", eager=True)
    def toggle_tool_detail(event: Any) -> None:
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        if state.running:
            state.notice = "Tool results are available after the run completes."
        elif state.toggle_tool_history():
            state.notice = (
                "Tool results shown; Ctrl-O hides them."
                if state.tool_history_run_id is not None
                else "Tool results hidden."
            )
        else:
            state.notice = "No completed tool details are available."
        invalidate(event.app)

    @keys.add("c-l", eager=True)
    def redraw(event: Any) -> None:
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        event.app.renderer.clear()
        invalidate(event.app)

    @keys.add("tab", eager=True)
    def complete_command(event: Any) -> None:
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        buffer = composer.buffer
        if buffer.complete_state is None:
            completions = tuple(
                buffer.completer.get_completions(
                    buffer.document,
                    runtime["CompleteEvent"](completion_requested=True),
                )
            )
            if completions:
                buffer.apply_completion(completions[0])
        else:
            buffer.complete_next()
        invalidate(event.app)

    escape_filter = runtime["Condition"](
        lambda: state.approval_panel is not None
        or state.transcript_selection.has_state
        or composer.buffer.complete_state is not None
    )

    @keys.add("escape", filter=escape_filter, eager=True)
    def escape(event: Any) -> None:
        if state.approval_panel is not None:
            remind_approval(event.app)
            return
        if state.transcript_selection.clear():
            state.transient_selection_hint = ""
            state.notice = "Transcript selection cleared."
            invalidate(event.app)
            return
        if composer.buffer.complete_state is not None:
            composer.buffer.cancel_completion()
            invalidate(event.app)

    clear_composer_filter = runtime["Condition"](
        lambda: state.approval_panel is None
        and not state.transcript_selection.has_state
        and composer.buffer.complete_state is None
        and bool(composer.buffer.text)
    )

    @keys.add("escape", "escape", filter=clear_composer_filter, eager=True)
    def clear_composer(event: Any) -> None:
        nonlocal history_position, history_draft
        nonlocal pasted_texts, next_paste_number
        composer.buffer.reset()
        pasted_texts = {}
        next_paste_number = 1
        history_position = len(input_history)
        history_draft = _ComposerDraft("")
        state.notice = "Input cleared."
        invalidate(event.app)

    @keys.add("up", eager=True)
    def move_up(event: Any) -> None:
        nonlocal history_position, history_draft
        panel = state.approval_panel
        if panel is not None:
            panel.move(-1)
        elif composer.buffer.complete_state is not None:
            composer.buffer.complete_previous()
        elif composer.buffer.document.cursor_position_row > 0:
            composer.buffer.cursor_up()
        elif input_history:
            if history_position >= len(input_history):
                history_draft = current_composer_draft()
            history_position = max(0, history_position - 1)
            restore_composer_draft(input_history[history_position])
        invalidate(event.app)

    @keys.add("down", eager=True)
    def move_down(event: Any) -> None:
        nonlocal history_position
        panel = state.approval_panel
        if panel is not None:
            panel.move(1)
        elif composer.buffer.complete_state is not None:
            composer.buffer.complete_next()
        elif (
            composer.buffer.document.cursor_position_row
            < composer.buffer.document.line_count - 1
        ):
            composer.buffer.cursor_down()
        elif history_position < len(input_history) - 1:
            history_position += 1
            restore_composer_draft(input_history[history_position])
        elif history_position == len(input_history) - 1:
            history_position = len(input_history)
            restore_composer_draft(history_draft)
        invalidate(event.app)

    @keys.add("home", filter=approval_filter, eager=True)
    def approval_home(event: Any) -> None:
        panel = state.approval_panel
        if panel is not None:
            panel.cursor_line = 0
        invalidate(event.app)

    @keys.add("end", filter=approval_filter, eager=True)
    def approval_end(event: Any) -> None:
        panel = state.approval_panel
        if panel is not None:
            panel.cursor_line = max(0, panel.rendered_line_count - 1)
        invalidate(event.app)

    @keys.add("y", filter=approval_filter, eager=True)
    @keys.add("Y", filter=approval_filter, eager=True)
    def approve_once(event: Any) -> None:
        if not terminal_is_usable():
            state.notice = "Resize the terminal to review this approval."
            invalidate(event.app)
            return
        resolve_approval(ApprovalDecision.APPROVE)

    @keys.add("n", filter=approval_filter, eager=True)
    @keys.add("N", filter=approval_filter, eager=True)
    def deny_once(event: Any) -> None:
        resolve_approval(ApprovalDecision.DENY)

    @keys.add(runtime["Keys"].Any, filter=approval_filter, eager=True)
    def ignore_unknown_approval_input(event: Any) -> None:
        remind_approval(event.app)

    def projected_status() -> StatusProjection:
        projection = responsive()
        return _status_projection(
            state,
            width=projection.columns,
            mode=projection.mode,
            glyphs=glyphs,
        )

    def border_fragments(
        *,
        top: bool,
        title: str = "",
        corners: bool = True,
    ) -> list[tuple[str, str]]:
        projection = responsive()
        width = max(2, projection.columns)
        if not corners and not title:
            return [("", glyphs.horizontal * width)]
        left = (glyphs.top_left if top else glyphs.bottom_left) if corners else ""
        right = (glyphs.top_right if top else glyphs.bottom_right) if corners else ""
        safe_title = _sanitize_terminal_text(
            title,
            maximum=max(0, width - 6),
            preserve_lines=False,
            fallback="",
        )
        if safe_title:
            middle = f"{glyphs.horizontal} {safe_title} "
            fill = glyphs.horizontal * max(
                0,
                width - _display_width(middle) - _display_width(left + right),
            )
            line = left + middle + fill + right
        else:
            line = (
                left
                + (glyphs.horizontal * max(0, width - _display_width(left + right)))
                + right
            )
        return [("", line)]

    def bordered(
        body: Any,
        *,
        title: str = "",
        bottom_title: str = "",
        style: str = "class:tui.frame",
        sides: bool = True,
    ) -> Any:
        framed_body = (
            runtime["VSplit"](
                [
                    runtime["Window"](
                        width=1,
                        char=glyphs.vertical,
                        style=style,
                    ),
                    body,
                    runtime["Window"](
                        width=1,
                        char=glyphs.vertical,
                        style=style,
                    ),
                ]
            )
            if sides
            else body
        )
        return runtime["HSplit"](
            [
                runtime["Window"](
                    runtime["FormattedTextControl"](
                        lambda: border_fragments(
                            top=True,
                            title=title,
                            corners=sides,
                        )
                    ),
                    height=1,
                    dont_extend_height=True,
                    style=style,
                ),
                framed_body,
                runtime["Window"](
                    runtime["FormattedTextControl"](
                        lambda: border_fragments(
                            top=False,
                            title=bottom_title,
                            corners=sides,
                        )
                    ),
                    height=1,
                    dont_extend_height=True,
                    style=style,
                ),
            ]
        )

    composer_frame = bordered(composer, sides=False)
    approval_container = runtime["ConditionalContainer"](
        bordered(
            approval_window,
            title="APPROVAL REQUIRED",
            style="class:tui.approval.frame",
        ),
        filter=approval_filter,
    )
    wide_status = runtime["VSplit"](
        [
            runtime["Window"](
                runtime["FormattedTextControl"](
                    lambda: _status_left_fragments(
                        state,
                        projection=projected_status(),
                    )
                ),
                height=1,
                dont_extend_height=True,
            ),
            runtime["Window"](
                runtime["FormattedTextControl"](
                    lambda: _status_right_fragments(
                        state,
                        projection=projected_status(),
                    )
                ),
                height=1,
                dont_extend_height=True,
                align="RIGHT",
            ),
        ],
        height=1,
    )
    compact_status = runtime["Window"](
        runtime["FormattedTextControl"](
            lambda: _status_single_line_fragments(
                state,
                projection=projected_status(),
            )
        ),
        height=1,
        dont_extend_height=True,
    )
    wide_status_filter = runtime["Condition"](lambda: responsive().two_sided_status)
    compact_status_filter = runtime["Condition"](
        lambda: not responsive().two_sided_status
    )
    status = runtime["HSplit"](
        [
            runtime["ConditionalContainer"](
                wide_status,
                filter=wide_status_filter,
            ),
            runtime["ConditionalContainer"](
                compact_status,
                filter=compact_status_filter,
            ),
        ]
    )

    def command_menu_state() -> Any:
        complete_state = composer.buffer.complete_state
        if complete_state is None or not complete_state.completions:
            return None
        return complete_state

    def command_menu_visible() -> bool:
        return (
            rendered_terminal_is_usable()
            and state.approval_panel is None
            and command_menu_state() is not None
        )

    def command_menu_fragments() -> list[tuple[str, str]]:
        complete_state = command_menu_state()
        if complete_state is None:
            return []
        items = tuple(
            (completion.display_text, completion.display_meta_text)
            for completion in complete_state.completions
        )
        return _slash_command_menu_fragments(
            items,
            selected_index=(
                complete_state.complete_index
                if complete_state.complete_index is not None
                else 0
            ),
            width=responsive().columns,
            glyphs=glyphs,
        )

    def command_menu_cursor() -> Any:
        complete_state = command_menu_state()
        selected_index = (
            complete_state.complete_index
            if complete_state is not None and complete_state.complete_index is not None
            else 0
        )
        return runtime["Point"](x=0, y=max(0, selected_index))

    command_menu_rows = runtime["Window"](
        runtime["FormattedTextControl"](
            command_menu_fragments,
            focusable=False,
            show_cursor=False,
            get_cursor_position=command_menu_cursor,
        ),
        wrap_lines=False,
        always_hide_cursor=True,
        dont_extend_height=True,
        style="class:tui.command-menu",
    )

    def command_menu_mouse_handler(mouse_event: Any) -> Any:
        try:
            if approval_owns_mouse(mouse_event):
                return None
            complete_state = command_menu_state()
            if complete_state is None:
                return NotImplemented
            event_type = mouse_event.event_type
            if event_type == runtime["MouseEventType"].SCROLL_UP:
                state.transient_selection_hint = ""
                composer.buffer.complete_previous()
            elif event_type == runtime["MouseEventType"].SCROLL_DOWN:
                state.transient_selection_hint = ""
                composer.buffer.complete_next()
            elif getattr(mouse_event.button, "name", "") == "LEFT" and event_type in {
                runtime["MouseEventType"].MOUSE_DOWN,
                runtime["MouseEventType"].MOUSE_MOVE,
                runtime["MouseEventType"].MOUSE_UP,
            }:
                if not mouse_action_owned("command_menu", event_type):
                    return None
                if event_type != runtime["MouseEventType"].MOUSE_UP:
                    return None
                index = mouse_event.position.y
                if 0 <= index < len(complete_state.completions):
                    composer.buffer.go_to_completion(index)
                    application.layout.focus(composer)
            else:
                return NotImplemented
            invalidate(application)
            return None
        except Exception:
            mouse_failed()
            return None

    command_menu_rows.content.mouse_handler = command_menu_mouse_handler

    def command_menu_rule() -> Any:
        return runtime["Window"](
            height=1,
            char=glyphs.horizontal,
            style="class:tui.command-menu.rule",
            dont_extend_height=True,
        )

    command_menu = runtime["ConditionalContainer"](
        runtime["HSplit"](
            [
                command_menu_rule(),
                command_menu_rows,
                command_menu_rule(),
            ]
        ),
        filter=runtime["Condition"](command_menu_visible),
    )

    def empty_shell_fragments() -> list[tuple[str, str]]:
        agent = _sanitize_terminal_text(
            state.agent_label,
            maximum=128,
            preserve_lines=False,
            fallback="agent",
        )
        source = _sanitize_terminal_text(
            state.source_summary,
            maximum=128,
            preserve_lines=False,
            fallback="",
        )
        return [
            ("class:tui.identity", f"\n DAITA  {agent}\n"),
            ("class:tui.header.meta", f" {source}\n" if source else ""),
            (
                "class:tui.rule",
                glyphs.horizontal * responsive().content_width + "\n",
            ),
            (
                "class:tui.empty",
                "\n Ask a question about your data, or type /help for commands.\n",
            ),
        ]

    def shell_content_fragments() -> list[tuple[str, str]]:
        nonlocal content_last_line_width, content_line_count
        if state.startup is not None and not state.blocks:
            fragments = _render_startup_fragments(
                state,
                width=responsive().content_width,
                capabilities=capabilities,
                glyphs=glyphs,
            )
        elif state.blocks:
            return transcript_fragments()
        else:
            fragments = empty_shell_fragments()
        content_line_count, content_last_line_width = _fragment_line_metrics(fragments)
        return fragments

    def viewport_rendered_top(window: Any) -> int:
        height = _viewport_height(window)
        if state.transcript_viewport.state is TranscriptFollowState.FOLLOWING:
            return max(0, content_line_count - height)
        projection = state.transcript_projection
        if projection is None:
            return 0
        state.transcript_viewport.top_row(
            projection,
            viewport_rows=height,
        )
        rendered_row = rendered_transcript_map.row_for_anchor(
            state.transcript_document,
            state.transcript_viewport.anchor,
        )
        return min(
            max(0, content_line_count - height),
            max(0, 0 if rendered_row is None else rendered_row),
        )

    def shell_content_cursor() -> Any:
        following = state.transcript_viewport.state is TranscriptFollowState.FOLLOWING
        return runtime["Point"](
            x=content_last_line_width if following else 0,
            y=(
                max(0, content_line_count - 1)
                if following
                else viewport_rendered_top(content_window)
            ),
        )

    def transcript_mouse_handler(mouse_event: Any) -> Any:
        try:
            if approval_owns_mouse(mouse_event):
                return None
            event_type = mouse_event.event_type
            left_button = getattr(mouse_event.button, "name", "") == "LEFT"
            if event_type == runtime["MouseEventType"].SCROLL_UP:
                state.transient_selection_hint = ""
                scroll_transcript(_MOUSE_SCROLL_LINES)
            elif event_type == runtime["MouseEventType"].SCROLL_DOWN:
                state.transient_selection_hint = ""
                scroll_transcript(-_MOUSE_SCROLL_LINES)
            elif event_type == runtime["MouseEventType"].MOUSE_DOWN and left_button:
                mouse_action_owned("transcript", event_type)
                transcript_fragments()
                position = rendered_transcript_map.position_for_cell(
                    mouse_event.position.y,
                    mouse_event.position.x,
                )
                if position is None:
                    state.transcript_selection.clear()
                else:
                    state.transcript_selection.begin(
                        state.transcript_document,
                        position,
                    )
                    state.notice = ""
            elif event_type == runtime["MouseEventType"].MOUSE_MOVE and left_button:
                if not mouse_action_owned("transcript", event_type):
                    return None
                if not state.transcript_selection.dragging:
                    return None
                height = _viewport_height(content_window)
                top = viewport_rendered_top(content_window)
                movement = bounded_selection_auto_scroll(
                    mouse_event.position.y,
                    viewport_top=top,
                    viewport_rows=height,
                )
                if movement < 0:
                    scroll_transcript(_SELECTION_AUTOSCROLL_LINES)
                elif movement > 0:
                    scroll_transcript(-_SELECTION_AUTOSCROLL_LINES)
                target_row = max(0, mouse_event.position.y + movement)
                position = rendered_transcript_map.position_for_cell(
                    target_row,
                    mouse_event.position.x,
                )
                if position is not None:
                    try:
                        state.transcript_selection.extend(
                            state.transcript_document,
                            position,
                        )
                    except (RuntimeError, ValueError):
                        state.transcript_selection.clear()
                state.notice = ""
            elif event_type == runtime["MouseEventType"].MOUSE_UP and left_button:
                if not mouse_action_owned("transcript", event_type):
                    return None
                if not state.transcript_selection.dragging:
                    return None
                transcript_fragments()
                position = rendered_transcript_map.position_for_cell(
                    mouse_event.position.y,
                    mouse_event.position.x,
                )
                if position is None:
                    state.transcript_selection.clear()
                    state.transient_selection_hint = ""
                else:
                    try:
                        selected = state.transcript_selection.finish(
                            state.transcript_document,
                            position,
                        )
                    except (RuntimeError, ValueError):
                        state.transcript_selection.clear()
                        state.transient_selection_hint = ""
                    else:
                        state.transient_selection_hint = (
                            _SELECTION_COMPLETE_HINT
                            if selected is not None and bool(selected.text)
                            else ""
                        )
            else:
                return NotImplemented
            invalidate(application)
            return None
        except Exception:
            mouse_failed()
            return None

    def create_transcript_content(
        control: Any,
        width: int,
        height: int | None,
    ) -> Any:
        nonlocal transcript_base_content, transcript_base_content_key

        # FormattedTextControl splits every logical row before consulting its
        # own cache. Keep that complete work tied to transcript or width
        # changes; navigation and selection then request only viewport rows.
        shell_content_fragments()
        key = (
            state.transcript_render_generation,
            state.transcript_document.generation,
            len(state.blocks),
            responsive(),
            width,
        )
        if transcript_base_content is None or transcript_base_content_key != key:
            transcript_base_content = runtime["FormattedTextControl"].create_content(
                control,
                width,
                height,
            )
            transcript_base_content_key = key
        base = transcript_base_content
        selected = state.transcript_selection.range

        def get_line(row: int) -> list[tuple[str, str]]:
            selected_cells = (
                None
                if selected is None
                else rendered_transcript_map.selected_cells_for_row(
                    state.transcript_document,
                    selected,
                    row,
                )
            )
            return _highlight_transcript_line(base.get_line(row), selected_cells)

        return runtime["UIContent"](
            get_line=get_line,
            line_count=base.line_count,
            cursor_position=shell_content_cursor(),
            menu_position=base.menu_position,
            show_cursor=base.show_cursor,
        )

    transcript_control_type = type(
        "TranscriptFormattedTextControl",
        (runtime["FormattedTextControl"],),
        {"create_content": create_transcript_content},
    )
    content_control = transcript_control_type(
        shell_content_fragments,
        focusable=False,
        show_cursor=False,
        get_cursor_position=shell_content_cursor,
    )
    content_control.mouse_handler = transcript_mouse_handler
    content_window = runtime["Window"](
        content_control,
        wrap_lines=False,
        always_hide_cursor=True,
        height=runtime["Dimension"](weight=1),
        get_vertical_scroll=viewport_rendered_top,
        style="class:tui.transcript",
    )

    def new_output_visible() -> bool:
        return (
            state.approval_panel is None
            and state.transcript_viewport.state is TranscriptFollowState.REVIEWING
            and state.transcript_viewport.unseen_items > 0
        )

    def activate_new_output(mouse_event: Any) -> Any:
        try:
            if approval_owns_mouse(mouse_event):
                return None
            event_type = mouse_event.event_type
            if getattr(mouse_event.button, "name", "") != "LEFT" or event_type not in {
                runtime["MouseEventType"].MOUSE_DOWN,
                runtime["MouseEventType"].MOUSE_MOVE,
                runtime["MouseEventType"].MOUSE_UP,
            }:
                return NotImplemented
            if not mouse_action_owned("jump_to_latest", event_type):
                return None
            if event_type == runtime["MouseEventType"].MOUSE_UP:
                state.transcript_viewport.follow_latest()
                invalidate(application)
            return None
        except Exception:
            mouse_failed()
            return None

    def new_output_fragments() -> list[Any]:
        count = state.transcript_viewport.unseen_items
        marker = "↓" if capabilities.unicode else "v"
        return [
            (
                "class:tui.new-output",
                f"{marker} {count} new {'item' if count == 1 else 'items'} ",
                activate_new_output,
            )
        ]

    new_output = runtime["ConditionalContainer"](
        runtime["Window"](
            runtime["FormattedTextControl"](new_output_fragments),
            height=1,
            dont_extend_height=True,
            align="RIGHT",
            style="class:tui.new-output",
        ),
        filter=runtime["Condition"](new_output_visible),
    )
    main_shell = runtime["HSplit"](
        [
            content_window,
            new_output,
            approval_container,
            composer_frame,
            command_menu,
            status,
        ]
    )
    usable_filter = runtime["Condition"](rendered_terminal_is_usable)
    resize_filter = runtime["Condition"](lambda: not rendered_terminal_is_usable())
    resize_window = runtime["Window"](
        runtime["FormattedTextControl"](
            lambda: _resize_message_fragments(
                responsive(),
                glyphs=glyphs,
            )
        ),
        wrap_lines=True,
        always_hide_cursor=True,
        style="class:tui.resize",
    )
    root = runtime["HSplit"](
        [
            runtime["ConditionalContainer"](
                main_shell,
                filter=usable_filter,
            ),
            runtime["ConditionalContainer"](
                resize_window,
                filter=resize_filter,
            ),
        ]
    )
    application = runtime["Application"](
        layout=runtime["Layout"](root, focused_element=composer),
        key_bindings=keys,
        full_screen=True,
        erase_when_done=True,
        mouse_support=mouse_reporting,
        input=enhanced_input,
        output=enhanced_output,
        style=semantic_style,
        terminal_size_polling_interval=_terminal_size_polling_interval(),
        before_render=refresh_responsive_projection,
    )
    application.ttimeoutlen = 0.01
    approval_previous = (
        approval_bridge.install(present_approval)
        if approval_bridge is not None
        else None
    )
    return application, approval_previous, cancel_pending_approval


async def _run_application(application: Any) -> Any:
    return await application.run_async()


def _replace_composer_text(buffer: Any, value: str) -> None:
    buffer.set_document(
        buffer.document.__class__(value, cursor_position=len(value)),
        bypass_readonly=True,
    )


def _materialize_pasted_texts(
    text: str,
    pasted_texts: Mapping[str, str],
) -> str:
    return _PASTED_TEXT_PLACEHOLDER_PATTERN.sub(
        lambda match: pasted_texts.get(match.group(0), match.group(0)),
        text,
    )


async def _consume_observer_events(
    bridge: TerminalObserverBridge,
    state: TerminalViewState,
    application: Any,
) -> None:
    loop = asyncio.get_running_loop()
    last_animation = loop.time()
    while True:
        projected = _project_pending_events(bridge, state)
        now = loop.time()
        animate = state.running and now - last_animation >= _ANIMATION_INTERVAL_SECONDS
        if animate:
            state.animation_frame = (state.animation_frame + 1) % len(_RUNNING_GLYPHS)
            last_animation = now
        if projected or animate:
            try:
                application.invalidate()
            except Exception:
                pass
        await asyncio.sleep(_STREAM_REPAINT_INTERVAL_SECONDS)


def _project_pending_events(
    bridge: TerminalObserverBridge,
    state: TerminalViewState,
) -> int:
    try:
        pending = bridge.drain()
    except Exception:
        return 0
    projected = 0
    fragment_key: tuple[str, int] | None = None
    fragment_text: list[str] = []

    def flush_fragments() -> None:
        nonlocal fragment_key, fragment_text
        if fragment_key is not None and fragment_text:
            try:
                state.apply_model_text_delta(
                    fragment_key[0],
                    fragment_key[1],
                    "".join(fragment_text),
                )
            except Exception:
                pass
        fragment_key = None
        fragment_text = []

    for event in pending:
        fields = _model_text_event_fields(event)
        if fields is not None:
            key = (event.run_id, fields[0])
            if fragment_key is not None and fragment_key != key:
                flush_fragments()
            fragment_key = key
            fragment_text.append(fields[1])
            projected += 1
            continue
        flush_fragments()
        try:
            state.apply_event(event)
        except Exception:
            continue
        projected += 1
    flush_fragments()
    return projected


def _render_tool_card_fragments(
    card: ToolCardState | None,
    *,
    width: int,
    runtime: dict[str, Any] | None = None,
    responsive: ResponsiveProjection | None = None,
    capabilities: TerminalCapabilities | None = None,
    glyphs: TerminalGlyphs | None = None,
) -> list[tuple[str, str]]:
    return _render_tool_card_fragments_impl(
        card,
        width=width,
        runtime=_load_terminal_runtime() if runtime is None else runtime,
        responsive=responsive,
        capabilities=capabilities,
        glyphs=glyphs,
    )


def _render_markdown_text(value: object, *, width: int = 80) -> str:
    """Render sanitized Markdown without terminal control sequences for tests."""

    return _render_markdown_text_with_runtime(
        _load_terminal_runtime(),
        value,
        width=width,
    )


def _clear_current_task_cancellation() -> None:
    current = asyncio.current_task()
    if current is None:
        return
    while current.cancelling():
        current.uncancel()


def _viewport_height(window: Any) -> int:
    render_info = getattr(window, "render_info", None)
    height = getattr(render_info, "window_height", 0)
    return max(1, height or 8)


def _restore_application(application: Any, output: Any) -> None:
    """Idempotently restore framework-owned modes, then basic output state."""

    try:
        application.renderer.reset()
    except Exception:
        for method_name in (
            "disable_mouse_support",
            "disable_bracketed_paste",
            "quit_alternate_screen",
        ):
            try:
                getattr(output, method_name)()
            except Exception:
                continue
    _restore_terminal(output)


def _restore_terminal(output: Any) -> None:
    for method_name in (
        "reset_attributes",
        "reset_cursor_key_mode",
        "reset_cursor_shape",
        "enable_autowrap",
        "show_cursor",
        "flush",
    ):
        try:
            getattr(output, method_name)()
        except Exception:
            continue


__all__ = [
    "ClipboardResult",
    "MAX_CLIPBOARD_UTF8_BYTES",
    "MAX_COMPOSER_CHARACTERS",
    "TerminalApplicationResult",
    "TerminalCommandResult",
    "TerminalObserverBridge",
    "TerminalStartupInfo",
    "TerminalSuspendBridge",
    "TerminalTUIUnavailable",
    "TerminalUserInputError",
    "TerminalViewState",
    "ToolCardDetails",
    "ToolCardState",
    "ToolTablePreview",
    "run_terminal_tui",
    "supports_terminal_tui",
]
