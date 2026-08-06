"""Shell controls, onboarding, approval, and status projection for the TUI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from typing import Any, TextIO

from ..capabilities import ApprovalRequest
from .capabilities import (
    MAX_RENDER_WIDTH as _MAX_RENDER_WIDTH,
    MIN_USABLE_COLUMNS as _MIN_USABLE_COLUMNS,
    ResponsiveProjection,
    TerminalCapabilities,
    TerminalGlyphs,
    responsive_projection as _responsive_projection,
    terminal_capabilities as _terminal_capabilities,
    terminal_glyphs as _terminal_glyphs,
    terminal_size as _terminal_size,
)
from .rendering import rich_theme_rules as _rich_theme_rules
from .state import ApprovalPanelState, TerminalViewState
from .text import (
    MAX_RENDER_CHARACTERS as _MAX_RENDER_CHARACTERS,
    display_width as _display_width,
    sanitize_terminal_text as _sanitize_terminal_text,
    truncate_display_text as _truncate_display_text,
)
from .tool_view import _SENSITIVE_KEY_PARTS

MAX_APPROVAL_DOCUMENT_CHARACTERS = 64 * 1_024


_SLASH_COMMAND_COMPLETIONS = (
    ("/model", "/model", "Choose or validate the active model"),
    ("/sources", "/sources", "List registered data sources"),
    ("/source", "/source", "Choose the active query source"),
    ("/source use ", "/source use <name>", "Use a source for new conversations"),
    ("/source add", "/source add", "Add a data source"),
    ("/source refresh ", "/source refresh <id>", "Refresh a source catalog"),
    (
        "/source detach ",
        "/source detach <source>",
        "Detach a source and delete its Daita-owned credential",
    ),
    ("/catalog", "/catalog", "Show the current catalog summary"),
    ("/settings", "/settings", "Show agent and model settings"),
    ("/new", "/new", "Start a new conversation"),
    ("/resume ", "/resume <id>", "Resume a previous conversation"),
    (
        "/conversation clear",
        "/conversation clear",
        "Delete all conversation history",
    ),
    ("/learn ", "/learn <material>", "Teach durable knowledge or a procedure"),
    (
        "/review",
        "/review [cost-usd]",
        "Review recent runs for memory or skill suggestions",
    ),
    (
        "/memory",
        "/memory",
        "Inspect global memory, semantics, learning candidates, and "
        "duplicate, stale, conflicting, and superseded states",
    ),
    ("/user", "/user", "View or edit the user profile"),
    ("/skills", "/skills", "List available skills"),
    ("/skills create", "/skills create", "Start guided skill creation"),
    (
        "/skills use ",
        "/skills use <name> [request]",
        "Invoke a skill by name",
    ),
    ("/status", "/status", "Show current agent status"),
    ("/conversation", "/conversation", "Show the current conversation ID"),
    ("/agent delete", "/agent delete", "Permanently delete this agent"),
    ("/help", "/help", "Show command help"),
    ("/exit", "/exit", "Exit Daita"),
)
_SLASH_COMMAND_SURFACE = tuple(
    display for _insertion, display, _description in _SLASH_COMMAND_COMPLETIONS
)
_SLASH_COMMAND_INSERTIONS = tuple(
    (insertion, display)
    for insertion, display, _description in _SLASH_COMMAND_COMPLETIONS
)
_SLASH_COMMAND_DESCRIPTIONS = tuple(
    (insertion, description)
    for insertion, _display, description in _SLASH_COMMAND_COMPLETIONS
)
_BUILTIN_SLASH_COMMAND_ROOTS = frozenset(
    display.split(maxsplit=1)[0]
    for _insertion, display, _description in _SLASH_COMMAND_COMPLETIONS
)


def _slash_completion_maps(
    skill_completions: Sequence[tuple[str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    display = dict(_SLASH_COMMAND_INSERTIONS)
    descriptions = dict(_SLASH_COMMAND_DESCRIPTIONS)
    for name, description in skill_completions:
        command = f"/{name}"
        if command in _BUILTIN_SLASH_COMMAND_ROOTS:
            continue
        insertion = f"{command} "
        display[insertion] = command
        descriptions[insertion] = description
    return display, descriptions


@dataclass(frozen=True, slots=True)
class StatusProjection:
    """One deterministic responsive projection of status metadata."""

    left: str
    right: str
    source_summary: str
    collapsed: tuple[str, ...]


def _responsive_for_output(
    output: Any,
    state: TerminalViewState,
) -> ResponsiveProjection:
    columns, rows = _terminal_size(output)
    return _responsive_projection(
        columns,
        rows,
        approving=state.approval_panel is not None,
    )


def _status_projection(
    state: TerminalViewState,
    *,
    width: int,
    mode: str,
    glyphs: TerminalGlyphs,
) -> StatusProjection:
    """Collapse status metadata in the documented deterministic order."""

    agent = _sanitize_terminal_text(
        state.agent_label,
        maximum=64,
        preserve_lines=False,
        fallback="agent",
    )
    model = _sanitize_terminal_text(
        state.model_label,
        maximum=96,
        preserve_lines=False,
        fallback="model",
    )
    source = _sanitize_terminal_text(
        state.source_summary,
        maximum=96,
        preserve_lines=False,
        fallback="",
    )
    if state.running:
        activity = state.active_tool_activity()
        if activity is None:
            activity_text = state.run_status
        else:
            card, active_count = activity
            verb = "approval for" if card.state == "approval" else "calling"
            suffix = f" (+{active_count - 1})" if active_count > 1 else ""
            activity_text = f"{verb} {card.label}{suffix}"
        state_word = _sanitize_terminal_text(
            activity_text,
            maximum=32,
            preserve_lines=False,
            fallback="working",
        )
        state_glyph = glyphs.running[
            state.animation_frame % max(1, len(glyphs.running))
        ]
    elif state.run_status in {"failed", "interrupted"}:
        state_word = _sanitize_terminal_text(
            state.run_status,
            maximum=32,
            preserve_lines=False,
            fallback="failed",
        )
        state_glyph = glyphs.failure
    else:
        state_word = "ready"
        state_glyph = glyphs.ready

    show_cost = True
    show_tokens = True
    shortened_model = False
    source_limit = 96
    show_source = bool(source)
    show_steps = True
    show_model = True
    collapsed: list[str] = []
    budget = max(1, width - 1)

    def current_text() -> tuple[str, str, str]:
        projected_model = model
        if shortened_model:
            projected_model, _truncated = _truncate_display_text(
                model,
                18,
                marker="..." if not glyphs.top_left.startswith("╭") else "…",
            )
        projected_source, _source_truncated = _truncate_display_text(
            source,
            source_limit,
            marker="..." if not glyphs.top_left.startswith("╭") else "…",
        )
        left_parts = [agent]
        if show_source:
            left_parts.append(f"source: {projected_source}")
        if show_model:
            left_parts.append(projected_model)
        left_parts.append(f"{state_glyph} {state_word}")
        right_parts: list[str] = []
        if show_steps:
            right_parts.append(f"{state.steps} steps")
        if show_tokens:
            context_progress = _context_progress_text(state, glyphs=glyphs)
            right_parts.append(
                context_progress or f"{_format_token_count(state.total_tokens)} tokens"
            )
        if show_cost:
            right_parts.append(state.estimated_cost)
        left = glyphs.separator.join(left_parts)
        right = glyphs.separator.join(right_parts)
        header_source = projected_source if show_source else ""
        return left, right, header_source

    def fits() -> bool:
        left, right, header_source = current_text()
        status_width = _display_width(left) + (
            2 + _display_width(right) if right else 0
        )
        header_width = (
            _display_width(" DAITA ")
            + _display_width(agent)
            + (2 + _display_width(header_source) if header_source else 0)
        )
        return max(status_width, header_width) <= budget

    forced = 0
    if mode == "compact":
        forced = 1
    elif mode == "narrow":
        forced = 5
    for index, field_name in enumerate(
        ("cost", "tokens", "shorten_model", "model", "steps", "shorten_source"),
        start=1,
    ):
        if fits() and index > forced:
            break
        if field_name == "cost":
            show_cost = False
        elif field_name == "tokens":
            show_tokens = False
        elif field_name == "shorten_model":
            shortened_model = True
        elif field_name == "model":
            show_model = False
        elif field_name == "steps":
            show_steps = False
        else:
            source_limit = min(source_limit, 24)
        collapsed.append(field_name)

    left, right, header_source = current_text()
    if _display_width(left) > budget and show_source:
        fixed = (
            _display_width(agent)
            + _display_width(f"{glyphs.separator}source: ")
            + _display_width(f"{glyphs.separator}{state_glyph} {state_word}")
            + (_display_width(f"{glyphs.separator}{model}") if show_model else 0)
        )
        source_limit = max(1, budget - fixed)
        if "shorten_source" not in collapsed:
            collapsed.append("shorten_source")
        left, right, header_source = current_text()
    if _display_width(left) > budget:
        available = max(1, budget - (_display_width(left) - _display_width(agent)))
        agent, _truncated = _truncate_display_text(
            agent,
            available,
            marker="..." if glyphs.top_left == "+" else "…",
        )
        left, right, header_source = current_text()
    return StatusProjection(
        left=left,
        right=right,
        source_summary=header_source,
        collapsed=tuple(collapsed),
    )


def _context_progress_text(
    state: TerminalViewState,
    *,
    glyphs: TerminalGlyphs,
) -> str:
    capacity = state.context_capacity_tokens
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        return ""
    bar_width = 10
    tokens = state.conversation_context_tokens
    if tokens is None:
        filled_cells = 0
        percentage = "--"
    else:
        bounded_tokens = max(0, min(tokens, capacity))
        filled_cells = min(
            bar_width,
            (bounded_tokens * bar_width + capacity - 1) // capacity,
        )
        percentage = f"{min(100, round(tokens * 100 / capacity))}%"
    if glyphs.top_left == "+":
        filled, empty = "#", "-"
    else:
        filled, empty = "█", "░"
    bar = filled * filled_cells + empty * (bar_width - filled_cells)
    return f"ctx [{bar}] {percentage}"


def _format_token_count(value: int) -> str:
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1_000:.1f}k"
    return f"{value / 1_000_000:.1f}m"


def _stream_is_interactive(output_stream: TextIO) -> bool:
    if os.environ.get("TERM", "").strip().casefold() in {"dumb", "unknown"}:
        return False
    try:
        return bool(output_stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _text_stream_width(output_stream: TextIO) -> int:
    try:
        columns = os.get_terminal_size(output_stream.fileno()).columns
    except (AttributeError, OSError, TypeError, ValueError):
        try:
            columns = int(os.environ.get("COLUMNS", "80"))
        except ValueError:
            columns = 80
    return max(1, min(_MAX_RENDER_WIDTH, columns))


def _setup_prompt_text(
    prompt: object,
    output_stream: TextIO,
) -> str:
    safe = _sanitize_terminal_text(
        prompt,
        maximum=256,
        preserve_lines=False,
        fallback="Value: ",
    )
    if not _stream_is_interactive(output_stream):
        return safe
    capabilities = _terminal_capabilities(text_stream=output_stream)
    glyphs = _terminal_glyphs(capabilities)
    return f"{glyphs.prompt} {safe}"


def _write_setup_prompt(
    runtime: dict[str, Any],
    output_stream: TextIO,
    prompt: object,
) -> None:
    safe = _setup_prompt_text(prompt, output_stream)
    if not _stream_is_interactive(output_stream):
        print(safe, end="", flush=True, file=output_stream)
        return
    capabilities = _terminal_capabilities(text_stream=output_stream)
    console = runtime["Console"](
        file=output_stream,
        force_terminal=not capabilities.no_color,
        color_system=capabilities.rich_color_system,
        no_color=capabilities.no_color,
        markup=False,
        highlight=False,
        soft_wrap=True,
        theme=runtime["Theme"](_rich_theme_rules(capabilities)),
    )
    console.print(runtime["Text"](safe, style="brand"), end="")


def _write_setup_status(
    runtime: dict[str, Any],
    output_stream: TextIO,
    value: object,
    *,
    role: str,
) -> None:
    safe = _sanitize_terminal_text(
        value,
        maximum=512,
        preserve_lines=False,
        fallback="Setup status unavailable.",
    )
    capabilities = _terminal_capabilities(text_stream=output_stream)
    glyphs = _terminal_glyphs(capabilities)
    if not capabilities.unicode:
        for marker, replacement in (
            ("✓ ", f"{glyphs.success} "),
            ("… ", f"{glyphs.running[0]} "),
            ("◐ ", f"{glyphs.running[0]} "),
        ):
            if safe.startswith(marker):
                safe = replacement + safe[len(marker) :]
                break
    if not _stream_is_interactive(output_stream):
        print(safe, file=output_stream)
        return
    style = {
        "progress": "brand",
        "success": "brand",
        "warning": "warning",
        "failure": "error",
        "muted": "muted",
    }.get(role, "")
    console = runtime["Console"](
        file=output_stream,
        force_terminal=not capabilities.no_color,
        color_system=capabilities.rich_color_system,
        no_color=capabilities.no_color,
        markup=False,
        highlight=False,
        soft_wrap=True,
        theme=runtime["Theme"](_rich_theme_rules(capabilities)),
    )
    console.print(runtime["Text"](safe, style=style))


def _slash_command_menu_fragments(
    items: Sequence[tuple[str, str]],
    *,
    selected_index: int | None,
    width: int,
    glyphs: TerminalGlyphs,
) -> list[tuple[str, str]]:
    """Render a bounded, full-width command-and-description palette."""

    safe_width = max(1, int(width))
    marker_width = 3
    command_width = min(24, max(16, safe_width // 4))
    description_width = max(0, safe_width - marker_width - command_width - 2)
    truncation_marker = "..." if glyphs.top_left == "+" else "…"

    def fitted(value: object, cell_width: int, *, maximum: int) -> str:
        safe = _sanitize_terminal_text(
            value,
            maximum=maximum,
            preserve_lines=False,
            fallback="",
        )
        projected, _truncated = _truncate_display_text(
            safe,
            max(1, cell_width),
            marker=truncation_marker,
        )
        return projected + (" " * max(0, cell_width - _display_width(projected)))

    fragments: list[tuple[str, str]] = []
    for index, (command, description) in enumerate(items):
        selected = index == selected_index
        marker = glyphs.prompt if selected else " "
        command_style = (
            "class:tui.command-menu.command.current"
            if selected
            else "class:tui.command-menu.command"
        )
        description_style = (
            "class:tui.command-menu.description.current"
            if selected
            else "class:tui.command-menu.description"
        )
        fragments.append(
            (
                (
                    "class:tui.command-menu.marker.current"
                    if selected
                    else "class:tui.command-menu.marker"
                ),
                f" {marker} ",
            )
        )
        fragments.append(
            (
                command_style,
                fitted(command, command_width, maximum=128),
            )
        )
        if description_width:
            fragments.append(("", "  "))
            fragments.append(
                (
                    description_style,
                    fitted(description, description_width, maximum=256),
                )
            )
        fragments.append(("", "\n"))
    return fragments


def _slash_command_completion_surface() -> tuple[str, ...]:
    """Return the documented terminal-local completion choices."""

    return _SLASH_COMMAND_SURFACE


def _approval_panel_for_request(
    request: ApprovalRequest,
) -> ApprovalPanelState | None:
    if not isinstance(request, ApprovalRequest):
        raise TypeError("approval presentation requires ApprovalRequest")
    arguments = request.arguments.to_dict()
    if _contains_sensitive_key(arguments):
        return None
    rendered = json.dumps(
        arguments,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if len(rendered) > MAX_APPROVAL_DOCUMENT_CHARACTERS:
        return None
    return ApprovalPanelState(
        tool_name=_sanitize_terminal_text(
            request.tool_name,
            maximum=256,
            preserve_lines=False,
            fallback="tool",
        ),
        capability_id=_sanitize_terminal_text(
            request.capability_id,
            maximum=256,
            preserve_lines=False,
            fallback="capability",
        ),
        arguments_text=rendered,
    )


def _contains_sensitive_key(value: object, *, key: str = "") -> bool:
    normalized_key = key.casefold().replace("-", "_")
    if key and any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_sensitive_key(item, key=str(item_key))
            for item_key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _render_approval_panel_fragments(
    panel: ApprovalPanelState,
    *,
    glyphs: TerminalGlyphs | None = None,
) -> list[tuple[str, str]]:
    if not isinstance(panel, ApprovalPanelState):
        raise TypeError("approval panel must be ApprovalPanelState")
    glyphs = glyphs or _terminal_glyphs(_terminal_capabilities())
    return [
        ("class:tui.approval.label", " Tool          "),
        ("class:tui.approval.identity", f"{panel.tool_name}\n"),
        ("class:tui.approval.label", " Capability    "),
        ("class:tui.approval.identity", f"{panel.capability_id}\n\n"),
        ("class:tui.approval.label", " Exact arguments\n"),
        ("class:tui.approval.arguments", f"{panel.arguments_text}\n\n"),
        (
            "class:tui.approval.action",
            f" {glyphs.approval} [Y] Approve once"
            "                                      [N] Deny\n",
        ),
    ]


def _status_state_style(state: TerminalViewState) -> str:
    if state.running and state.run_status == "approval":
        return "class:tui.status.approval"
    if state.running:
        return "class:tui.status.running"
    if state.run_status in {"failed", "interrupted"}:
        return "class:tui.status.failure"
    return "class:tui.status.ready"


def _status_left_fragments(
    state: TerminalViewState,
    *,
    projection: StatusProjection | None = None,
) -> list[tuple[str, str]]:
    if projection is None:
        glyphs = _terminal_glyphs(_terminal_capabilities())
        projection = _status_projection(
            state,
            width=100,
            mode="full",
            glyphs=glyphs,
        )
    return [(_status_state_style(state), f" {projection.left}")]


def _status_right_fragments(
    state: TerminalViewState,
    *,
    projection: StatusProjection | None = None,
) -> list[tuple[str, str]]:
    notice = state.notice or (
        state.transient_selection_hint if state.approval_panel is None else ""
    )
    if notice:
        return [
            (
                "class:tui.status.notice",
                _sanitize_terminal_text(
                    notice,
                    maximum=256,
                    preserve_lines=False,
                    fallback="",
                )
                + " ",
            )
        ]
    if projection is None:
        glyphs = _terminal_glyphs(_terminal_capabilities())
        projection = _status_projection(
            state,
            width=100,
            mode="full",
            glyphs=glyphs,
        )
    return [
        (
            "class:tui.status.meta",
            f"{projection.right} " if projection.right else "",
        )
    ]


def _status_single_line_fragments(
    state: TerminalViewState,
    *,
    projection: StatusProjection,
) -> list[tuple[str, str]]:
    fragments = [(_status_state_style(state), f" {projection.left}")]
    suffix = ""
    suffix_style = "class:tui.status.meta"
    notice = state.notice or (
        state.transient_selection_hint if state.approval_panel is None else ""
    )
    if notice:
        suffix = _sanitize_terminal_text(
            notice,
            maximum=128,
            preserve_lines=False,
            fallback="",
        )
        suffix_style = "class:tui.status.notice"
    elif projection.right:
        suffix = projection.right
    if suffix:
        fragments.append((suffix_style, f"  {suffix}"))
    return fragments


def _resize_message_fragments(
    projection: ResponsiveProjection,
    *,
    glyphs: TerminalGlyphs,
) -> list[tuple[str, str]]:
    message = (
        f"{glyphs.warning} Terminal too small "
        f"({projection.columns}x{projection.rows}). "
        f"Resize to at least {_MIN_USABLE_COLUMNS}x{projection.minimum_rows}."
    )
    maximum = max(1, min(_MAX_RENDER_CHARACTERS, projection.columns * 3))
    safe = _sanitize_terminal_text(
        message,
        maximum=maximum,
        preserve_lines=False,
        fallback="Resize the terminal.",
    )
    return [
        ("class:tui.resize", "\n"),
        ("class:tui.resize", f" {safe}\n"),
    ]
