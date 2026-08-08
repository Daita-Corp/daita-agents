"""Bounded tool-result projection and card rendering for the terminal UI."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._json import freeze_json, thaw_json
from ..llm.models import ToolCall, ToolResultBlock
from .capabilities import (
    EXPANDED_TABLE_COLUMNS as _EXPANDED_TABLE_COLUMNS,
    MAX_RENDER_WIDTH as _MAX_RENDER_WIDTH,
    MIN_RENDER_WIDTH as _MIN_RENDER_WIDTH,
    ResponsiveProjection,
    TerminalCapabilities,
    TerminalGlyphs,
    responsive_projection as _responsive_projection,
    terminal_capabilities as _terminal_capabilities,
    terminal_glyphs as _terminal_glyphs,
)
from .rendering import render_rich_fragment_lines as _render_rich_fragment_lines
from .text import (
    MAX_RENDER_CHARACTERS as _MAX_RENDER_CHARACTERS,
    display_width as _display_width,
    one_logical_line as _one_logical_line,
    sanitize_terminal_text as _sanitize_terminal_text,
    truncate_display_text as _truncate_display_text,
)

_MAX_DETAIL_UTF8_BYTES = 16 * 1_024
_MAX_CODE_VISIBLE_LINES = 80
_COLLAPSED_TABLE_ROWS = 10
_EXPANDED_TABLE_ROWS = 50
_MAX_CELL_DISPLAY_CHARACTERS = 240
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_CAPABILITY_LABELS = {
    "catalog.search": "Search catalog",
    "catalog.inspect": "Inspect schema",
    "catalog.traverse": "Follow relationships",
    "data.sqlite.query": "Query SQLite",
    "data.postgresql.query": "Query PostgreSQL",
    "data.file.read": "Read data file",
    "artifact.create_document": "Create document",
    "artifact.save_local": "Save artifact",
    "artifact.set_export_location": "Set export location",
    "memory.set": "Update memory",
    "skill.view": "Read skill",
    "skill.save": "Save skill",
    "skill.delete": "Delete skill",
}
_TOOL_ERROR_HEADINGS = {
    "postgresql_connect_failed": "Connection unavailable",
    "postgresql_credential_unavailable": "Credentials unavailable",
    "postgresql_credential_invalid": "Credentials invalid",
}
_TOOL_ERROR_GUIDANCE = {
    "postgresql_connect_failed": (
        "Check that the database is running and reachable, then retry."
    ),
    "postgresql_credential_unavailable": (
        "Check keychain access or replace the saved database password, then retry."
    ),
    "postgresql_credential_invalid": (
        "Replace the saved database password, then retry."
    ),
}


@dataclass(frozen=True, slots=True)
class ToolTablePreview:
    """A bounded row/column projection from one recorded tool result."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    recorded_rows: int
    recorded_columns: int
    total_rows: int | None = None
    cells_truncated: bool = False


@dataclass(frozen=True, slots=True)
class ToolCardDetails:
    """Safe, bounded, process-local detail hydrated from one transcript pair."""

    summary: str
    code: str | None = None
    code_language: str | None = None
    arguments_text: str | None = None
    result_text: str | None = None
    error_message: str | None = None
    table: ToolTablePreview | None = None


@dataclass(slots=True)
class ToolCardState:
    """One bounded live projection of a model-requested tool call."""

    run_id: str
    call_id: str
    capability_id: str | None
    label: str
    state: str = "queued"
    duration_ms: int | None = None
    error_code: str | None = None
    approval_outcome: str | None = None
    details: ToolCardDetails | None = None
    expanded: bool = False


def _project_tool_details(
    call: ToolCall,
    result: ToolResultBlock,
) -> ToolCardDetails:
    arguments = thaw_json(freeze_json(call.arguments))
    output = thaw_json(freeze_json(result.output))
    assert isinstance(arguments, dict)
    assert isinstance(output, dict)

    code_value = arguments.get("sql")
    code_language = "sql"
    if not isinstance(code_value, str):
        code_value = arguments.get("code")
        code_language = "text"

    presented_arguments = _redact_presentation_value(arguments)
    assert isinstance(presented_arguments, dict)
    presented_arguments.pop("sql", None)
    presented_arguments.pop("code", None)
    arguments_text = (
        _bounded_json_text(presented_arguments) if presented_arguments else None
    )

    if result.is_error:
        error = output.get("error")
        error_mapping = error if isinstance(error, dict) else {}
        error_code = _sanitize_terminal_text(
            error_mapping.get("code"),
            maximum=128,
            preserve_lines=False,
            fallback="tool_failed",
        )
        error_message = _bounded_plain_text(
            error_mapping.get("message"),
            fallback="Tool execution failed.",
        )
        error_details = error_mapping.get("details")
        result_text = (
            _bounded_json_text(_redact_presentation_value(error_details))
            if error_details not in (None, {}, [])
            else None
        )
        error_heading = _TOOL_ERROR_HEADINGS.get(error_code, error_code)
        summary = _one_logical_line(f"{error_heading} · {error_message}")
        arguments_text, fitted_error_message, result_text = _fit_detail_text_budget(
            arguments_text,
            error_message,
            result_text,
        )
        assert fitted_error_message is not None
        return ToolCardDetails(
            summary=summary,
            code=_bounded_code_text(code_value),
            code_language=code_language if isinstance(code_value, str) else None,
            arguments_text=arguments_text,
            result_text=result_text,
            error_message=fitted_error_message,
        )

    data = output.get("data")
    data_mapping = data if isinstance(data, dict) else None
    if not isinstance(code_value, str) and data_mapping is not None:
        canonical_sql = data_mapping.get("canonical_sql")
        if isinstance(canonical_sql, str):
            code_value = canonical_sql
            code_language = "sql"

    table = _project_table_preview(data_mapping)
    result_kind = _sanitize_terminal_text(
        output.get("kind"),
        maximum=256,
        preserve_lines=False,
        fallback="Tool result",
    )
    if table is not None:
        assert data_mapping is not None
        result_projection = {
            key: value
            for key, value in data_mapping.items()
            if key not in {"rows", "columns", "canonical_sql"}
        }
        result_text = (
            _bounded_json_text(_redact_presentation_value(result_projection))
            if result_projection
            else None
        )
        summary = (
            f"{table.recorded_rows} recorded rows · "
            f"{table.recorded_columns} columns"
        )
    else:
        result_text = (
            _bounded_json_text(_redact_presentation_value(data))
            if data is not None
            else None
        )
        summary = result_kind
    if isinstance(code_value, str):
        summary = _one_logical_line(code_value)
    elif result_text is not None and summary == "Tool result":
        summary = _one_logical_line(result_text)
    arguments_text, _unused_error, result_text = _fit_detail_text_budget(
        arguments_text,
        None,
        result_text,
    )
    return ToolCardDetails(
        summary=_sanitize_terminal_text(
            summary,
            maximum=_MAX_RENDER_CHARACTERS,
            preserve_lines=False,
            fallback="Completed.",
        ),
        code=_bounded_code_text(code_value),
        code_language=code_language if isinstance(code_value, str) else None,
        arguments_text=arguments_text,
        result_text=result_text,
        table=table,
    )


def _project_table_preview(
    data: dict[str, object] | None,
) -> ToolTablePreview | None:
    if data is None:
        return None
    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list):
        return None
    raw_columns = data.get("columns")
    if isinstance(raw_columns, list) and all(
        isinstance(column, str) for column in raw_columns
    ):
        columns = list(raw_columns)
    elif raw_rows and isinstance(raw_rows[0], dict):
        columns = list(raw_rows[0])
    else:
        return None

    projected_columns: list[str] = []
    for column in columns[:_EXPANDED_TABLE_COLUMNS]:
        projected, _truncated = _truncate_display_text(
            _sanitize_terminal_text(
                column,
                maximum=_MAX_CELL_DISPLAY_CHARACTERS * 2,
                preserve_lines=False,
                fallback="column",
            ),
            _MAX_CELL_DISPLAY_CHARACTERS,
        )
        projected_columns.append(projected)

    rows: list[tuple[str, ...]] = []
    cells_truncated = False
    for raw_row in raw_rows[:_EXPANDED_TABLE_ROWS]:
        if not isinstance(raw_row, dict):
            continue
        row: list[str] = []
        for column in columns[:_EXPANDED_TABLE_COLUMNS]:
            cell, truncated = _cell_text(raw_row.get(column))
            row.append(cell)
            cells_truncated = cells_truncated or truncated
        rows.append(tuple(row))

    total_rows = data.get("total_rows")
    if (
        not isinstance(total_rows, int)
        or isinstance(total_rows, bool)
        or total_rows < len(raw_rows)
    ):
        total_rows = None
    return ToolTablePreview(
        columns=tuple(projected_columns),
        rows=tuple(rows),
        recorded_rows=len(raw_rows),
        recorded_columns=len(columns),
        total_rows=total_rows,
        cells_truncated=cells_truncated,
    )


def _redact_presentation_value(value: object, *, key: str = "") -> object:
    normalized_key = key.casefold().replace("-", "_")
    if key and any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_presentation_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_presentation_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_presentation_value(item) for item in value]
    return value


def _bounded_json_text(value: object) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError):
        rendered = json.dumps(
            _sanitize_terminal_text(
                str(value),
                maximum=_MAX_DETAIL_UTF8_BYTES,
                preserve_lines=True,
                fallback="",
            ),
            ensure_ascii=False,
        )
    safe = _sanitize_terminal_text(
        rendered,
        maximum=max(1, len(rendered) + 1),
        preserve_lines=True,
        fallback="{}",
    )
    return _bound_utf8_detail(safe)


def _bounded_plain_text(value: object, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    safe = _sanitize_terminal_text(
        value,
        maximum=max(1, len(value) + 1),
        preserve_lines=True,
        fallback=fallback,
    )
    return _bound_utf8_detail(safe)


def _bounded_code_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _sanitize_terminal_text(
        value,
        maximum=_MAX_CODE_VISIBLE_LINES * _MAX_RENDER_CHARACTERS,
        preserve_lines=True,
        fallback="",
    )


def _bound_utf8_detail(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_DETAIL_UTF8_BYTES:
        return value
    indicator = f"\n… detail truncated at {_MAX_DETAIL_UTF8_BYTES // 1_024} KiB"
    indicator_bytes = indicator.encode("utf-8")
    prefix = encoded[: _MAX_DETAIL_UTF8_BYTES - len(indicator_bytes)].decode(
        "utf-8",
        errors="ignore",
    )
    return prefix + indicator


def _fit_detail_text_budget(
    arguments_text: str | None,
    error_message: str | None,
    result_text: str | None,
) -> tuple[str | None, str | None, str | None]:
    values = [arguments_text, error_message, result_text]
    combined_bytes = sum(
        len(value.encode("utf-8")) for value in values if value is not None
    )
    if combined_bytes <= _MAX_DETAIL_UTF8_BYTES:
        return arguments_text, error_message, result_text

    indicator = "\n… remaining text/JSON detail omitted at 16 KiB"
    indicator_bytes = indicator.encode("utf-8")
    content_budget = _MAX_DETAIL_UTF8_BYTES - len(indicator_bytes)
    projected: list[str | None] = [None, None, None]
    used = 0
    last_index: int | None = None
    for index, value in enumerate(values):
        if value is None:
            continue
        separator_bytes = 1 if last_index is not None else 0
        available = content_budget - used - separator_bytes
        if available <= 0:
            break
        encoded = value.encode("utf-8")
        if len(encoded) <= available:
            projected[index] = value
            used += separator_bytes + len(encoded)
            last_index = index
            continue
        projected[index] = encoded[:available].decode("utf-8", errors="ignore")
        last_index = index
        break
    if last_index is None:
        projected[0] = indicator.lstrip("\n")
    else:
        projected[last_index] = (projected[last_index] or "") + indicator
    return projected[0], projected[1], projected[2]


def _cell_text(value: object) -> tuple[str, bool]:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            rendered = str(value)
    safe = _sanitize_terminal_text(
        rendered,
        maximum=max(1, min(len(rendered) + 1, 4 * _MAX_CELL_DISPLAY_CHARACTERS)),
        preserve_lines=False,
        fallback="",
    )
    truncated_before_display = len(safe) < len(rendered)
    projected, display_truncated = _truncate_display_text(
        safe,
        _MAX_CELL_DISPLAY_CHARACTERS,
    )
    return projected, truncated_before_display or display_truncated


def _tool_result_error_code(result: ToolResultBlock) -> str:
    error = result.output.get("error")
    if isinstance(error, Mapping):
        return _sanitize_terminal_text(
            error.get("code"),
            maximum=128,
            preserve_lines=False,
            fallback="tool_failed",
        )
    return "tool_failed"


def _tool_run_summary_text(
    cards: Sequence[ToolCardState],
    *,
    glyphs: TerminalGlyphs,
) -> str:
    count = len(cards)
    failed = sum(card.state == "failed" for card in cards)
    call_label = "tool call" if count == 1 else "tool calls"
    parts = [f"{count} {call_label}"]
    if failed:
        parts.append(f"{failed} failed")
    parts.append("Ctrl-O view results")
    return glyphs.separator.join(parts)


def _render_tool_run_summary_fragments(
    cards: Sequence[ToolCardState],
    *,
    width: int,
    capabilities: TerminalCapabilities,
    glyphs: TerminalGlyphs,
) -> list[tuple[str, str]]:
    failed = any(card.state == "failed" for card in cards)
    glyph = glyphs.failure if failed else glyphs.success
    style = "class:tui.tool.failure" if failed else "class:tui.tool.success"
    summary = _tool_run_summary_text(cards, glyphs=glyphs)
    marker = "…" if capabilities.unicode else "..."
    visible = _truncate_display_text(
        summary,
        max(1, width - _display_width(glyph) - 2),
        marker=marker,
    )[0]
    return [(style, f" {glyph} {visible}\n")]


def _render_tool_card_fragments(
    card: ToolCardState | None,
    *,
    width: int,
    runtime: dict[str, Any] | None = None,
    responsive: ResponsiveProjection | None = None,
    capabilities: TerminalCapabilities | None = None,
    glyphs: TerminalGlyphs | None = None,
) -> list[tuple[str, str]]:
    if not isinstance(card, ToolCardState):
        return [
            ("class:tui.tool.failure", " ! Tool status unavailable\n"),
            ("", "\n"),
        ]
    safe_width = max(_MIN_RENDER_WIDTH, min(width, _MAX_RENDER_WIDTH))
    if runtime is None:
        raise ValueError("tool card rendering requires the lazy terminal runtime")
    capabilities = capabilities or _terminal_capabilities()
    glyphs = glyphs or _terminal_glyphs(capabilities)
    responsive = responsive or _responsive_projection(width, 24)
    label = _sanitize_terminal_text(
        card.label,
        maximum=max(8, safe_width - 16),
        preserve_lines=False,
        fallback="Tool call",
    )
    if card.state == "running":
        glyph = glyphs.running[0]
        style = "class:tui.tool.running"
        fallback_body = "Running…" if capabilities.unicode else "Running..."
    elif card.state == "approval":
        glyph = glyphs.approval
        style = "class:tui.tool.approval"
        fallback_body = (
            "Approval required…" if capabilities.unicode else "Approval required..."
        )
    elif card.state == "succeeded":
        glyph = glyphs.success
        style = "class:tui.tool.success"
        fallback_body = "Completed."
    else:
        glyph = glyphs.failure
        style = "class:tui.tool.failure"
        fallback_body = _sanitize_terminal_text(
            card.error_code,
            maximum=max(8, safe_width - 8),
            preserve_lines=False,
            fallback="Tool failed.",
        )
    duration = (
        f"{glyphs.separator}{_format_duration(card.duration_ms)}"
        if card.duration_ms is not None
        else ""
    )
    title = _sanitize_terminal_text(
        f"{glyph} {label}{duration}",
        maximum=max(8, safe_width - 7),
        preserve_lines=False,
        fallback=f"{glyph} Tool call",
    )
    fragments: list[tuple[str, str]]
    if responsive.bordered_cards:
        top_fill = glyphs.horizontal * max(
            1,
            safe_width - _display_width(title) - 6,
        )
        fragments = [
            (
                style,
                " "
                f"{glyphs.top_left}{glyphs.horizontal} {title} "
                f"{top_fill}{glyphs.top_right}\n",
            ),
        ]
    else:
        fragments = [(style, f" {glyph} {label}\n")]
        if card.duration_ms is not None:
            fragments.append(
                (
                    "class:tui.metadata",
                    f"   {_format_duration(card.duration_ms)}\n",
                )
            )
    if card.details is None or card.state not in {"succeeded", "failed"}:
        fragments.extend(
            _card_plain_lines(
                (
                    _sanitize_terminal_text(
                        fallback_body,
                        maximum=max(8, safe_width - 7),
                        preserve_lines=False,
                        fallback="Status unavailable.",
                    ),
                ),
                style=style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
    else:
        fragments.extend(
            _render_tool_details(
                runtime,
                card,
                width=max(_MIN_RENDER_WIDTH, safe_width - 6),
                border_style=style,
                responsive=responsive,
                capabilities=capabilities,
                glyphs=glyphs,
            )
        )
    if responsive.bordered_cards:
        bottom_fill = glyphs.horizontal * max(4, safe_width - 3)
        fragments.append(
            (
                style,
                f" {glyphs.bottom_left}{bottom_fill}{glyphs.bottom_right}\n",
            )
        )
    fragments.append(("", "\n"))
    return fragments


def _canonical_tool_card_text(
    card: ToolCardState | None,
    *,
    runtime: dict[str, Any],
    responsive: ResponsiveProjection,
    capabilities: TerminalCapabilities,
    glyphs: TerminalGlyphs,
) -> str:
    """Use the existing card renderer for a wrap-stable selectable projection."""

    canonical_responsive = ResponsiveProjection(
        columns=_MAX_RENDER_WIDTH + 2,
        rows=responsive.rows,
        mode=responsive.mode,
        content_width=_MAX_RENDER_WIDTH,
        collapsed_preview_columns=responsive.collapsed_preview_columns,
        expanded_preview_columns=responsive.expanded_preview_columns,
        bordered_cards=False,
        stacked_metadata=responsive.stacked_metadata,
        two_sided_status=responsive.two_sided_status,
        usable=responsive.usable,
        minimum_rows=responsive.minimum_rows,
        transcript_rows=responsive.transcript_rows,
    )
    rendered = "".join(
        text
        for _style, text in _render_tool_card_fragments(
            card,
            width=_MAX_RENDER_WIDTH,
            runtime=runtime,
            responsive=canonical_responsive,
            capabilities=capabilities,
            glyphs=glyphs,
        )
    ).rstrip("\n")
    lines = rendered.splitlines()
    if card is not None and lines:
        label = _sanitize_terminal_text(
            card.label,
            maximum=_MAX_RENDER_CHARACTERS,
            preserve_lines=False,
            fallback="Tool call",
        )
        label_index = lines[0].find(label)
        if label_index >= 0:
            lines[0] = lines[0][label_index:]
    return "\n".join(line[3:] if line.startswith("   ") else line for line in lines)


def _render_tool_details(
    runtime: dict[str, Any],
    card: ToolCardState,
    *,
    width: int,
    border_style: str,
    responsive: ResponsiveProjection,
    capabilities: TerminalCapabilities,
    glyphs: TerminalGlyphs,
) -> list[tuple[str, str]]:
    details = card.details
    if details is None:
        return _card_plain_lines(
            ("Status unavailable.",),
            style=border_style,
            glyphs=glyphs,
            bordered=responsive.bordered_cards,
        )

    fragments: list[tuple[str, str]] = []
    if not card.expanded:
        summary, _truncated = _truncate_display_text(
            _one_logical_line(details.summary),
            max(8, width),
        )
        fragments.extend(
            _card_plain_lines(
                (summary,),
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
        if details.table is not None:
            fragments.extend(
                _card_rich_lines(
                    runtime,
                    _table_renderable(
                        runtime,
                        details.table,
                        row_limit=_COLLAPSED_TABLE_ROWS,
                        column_limit=responsive.collapsed_preview_columns,
                        width=width,
                    ),
                    width=width,
                    border_style=border_style,
                    capabilities=capabilities,
                    glyphs=glyphs,
                    bordered=responsive.bordered_cards,
                )
            )
            fragments.extend(
                _card_plain_lines(
                    _table_truncation_lines(
                        details.table,
                        shown_rows=min(
                            _COLLAPSED_TABLE_ROWS,
                            len(details.table.rows),
                        ),
                        shown_columns=min(
                            responsive.collapsed_preview_columns,
                            len(details.table.columns),
                        ),
                    ),
                    style=border_style,
                    glyphs=glyphs,
                    bordered=responsive.bordered_cards,
                )
            )
        fragments.extend(
            _card_plain_lines(
                ("Ctrl-O view result",),
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
        return fragments

    if details.code is not None:
        label = "SQL" if details.code_language == "sql" else "Code"
        fragments.extend(
            _card_plain_lines(
                (label,),
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
        syntax = runtime["Syntax"](
            details.code,
            details.code_language or "text",
            theme="bw",
            background_color="default",
            line_numbers=False,
            word_wrap=True,
        )
        fragments.extend(
            _card_rich_lines(
                runtime,
                syntax,
                width=width,
                border_style=border_style,
                maximum_lines=_MAX_CODE_VISIBLE_LINES,
                capabilities=capabilities,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
                truncation_line=(
                    ("…" if capabilities.unicode else "...") + " code truncated at "
                    f"{_MAX_CODE_VISIBLE_LINES} visible lines"
                ),
            )
        )
    if details.arguments_text is not None:
        fragments.extend(
            _card_plain_lines(
                ("Arguments",),
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
        fragments.extend(
            _card_rich_lines(
                runtime,
                runtime["Syntax"](
                    details.arguments_text,
                    "json",
                    theme="bw",
                    background_color="default",
                    line_numbers=False,
                    word_wrap=True,
                ),
                width=width,
                border_style=border_style,
                capabilities=capabilities,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
    if details.error_message is not None:
        error_heading = _TOOL_ERROR_HEADINGS.get(card.error_code or "", "Error")
        error_lines = [error_heading, details.error_message]
        error_guidance = _TOOL_ERROR_GUIDANCE.get(card.error_code or "")
        if error_guidance is not None:
            error_lines.append(error_guidance)
        fragments.extend(
            _card_plain_lines(
                error_lines,
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
    if details.table is not None:
        fragments.extend(
            _card_plain_lines(
                ("Recorded result",),
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
        fragments.extend(
            _card_rich_lines(
                runtime,
                _table_renderable(
                    runtime,
                    details.table,
                    row_limit=_EXPANDED_TABLE_ROWS,
                    column_limit=responsive.expanded_preview_columns,
                    width=width,
                ),
                width=width,
                border_style=border_style,
                capabilities=capabilities,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
        fragments.extend(
            _card_plain_lines(
                _table_truncation_lines(
                    details.table,
                    shown_rows=min(_EXPANDED_TABLE_ROWS, len(details.table.rows)),
                    shown_columns=min(
                        responsive.expanded_preview_columns,
                        len(details.table.columns),
                    ),
                ),
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
    if details.result_text is not None:
        fragments.extend(
            _card_plain_lines(
                ("Result details",),
                style=border_style,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
        fragments.extend(
            _card_rich_lines(
                runtime,
                runtime["Syntax"](
                    details.result_text,
                    "json",
                    theme="bw",
                    background_color="default",
                    line_numbers=False,
                    word_wrap=True,
                ),
                width=width,
                border_style=border_style,
                capabilities=capabilities,
                glyphs=glyphs,
                bordered=responsive.bordered_cards,
            )
        )
    fragments.extend(
        _card_plain_lines(
            ("Ctrl-O hide tool results",),
            style=border_style,
            glyphs=glyphs,
            bordered=responsive.bordered_cards,
        )
    )
    return fragments


def _table_renderable(
    runtime: dict[str, Any],
    preview: ToolTablePreview,
    *,
    row_limit: int,
    column_limit: int,
    width: int,
) -> Any:
    table = runtime["Table"](
        box=None,
        show_header=True,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
        highlight=False,
    )
    columns = preview.columns[:column_limit]
    column_width = max(
        3, (max(_MIN_RENDER_WIDTH, width) - len(columns)) // max(1, len(columns))
    )
    for column in columns:
        table.add_column(
            runtime["Text"](column, style="data"),
            overflow="ellipsis",
            no_wrap=True,
            max_width=column_width,
        )
    for row in preview.rows[:row_limit]:
        table.add_row(
            *(runtime["Text"](cell) for cell in row[: len(columns)]),
        )
    return table


def _table_truncation_lines(
    preview: ToolTablePreview,
    *,
    shown_rows: int,
    shown_columns: int,
) -> tuple[str, ...]:
    notices: list[str] = []
    omitted_recorded_rows = max(0, preview.recorded_rows - shown_rows)
    if omitted_recorded_rows:
        notices.append(
            f"… {omitted_recorded_rows} more rows in the recorded tool result"
        )
    omitted_recorded_columns = max(0, preview.recorded_columns - shown_columns)
    if omitted_recorded_columns:
        notices.append(
            f"… {omitted_recorded_columns} more columns in the recorded tool result"
        )
    if preview.total_rows is not None and preview.total_rows > preview.recorded_rows:
        notices.append(
            "… "
            f"{preview.total_rows - preview.recorded_rows} additional rows were not "
            "recorded by the bounded tool result"
        )
    if preview.cells_truncated:
        notices.append(
            f"… cells truncated to {_MAX_CELL_DISPLAY_CHARACTERS} display characters"
        )
    return tuple(notices)


def _card_plain_lines(
    lines: Sequence[str],
    *,
    style: str,
    glyphs: TerminalGlyphs | None = None,
    bordered: bool = True,
) -> list[tuple[str, str]]:
    glyphs = glyphs or _terminal_glyphs(_terminal_capabilities())
    fragments: list[tuple[str, str]] = []
    for line in lines:
        safe = _sanitize_terminal_text(
            line,
            maximum=_MAX_RENDER_CHARACTERS,
            preserve_lines=False,
            fallback="",
        )
        if safe:
            prefix = f" {glyphs.vertical} " if bordered else "   "
            fragments.extend(
                [
                    (style, prefix),
                    ("class:tui.tool.text", safe),
                    (style, "\n"),
                ]
            )
    return fragments


def _tool_detail_text_style(style: str) -> str:
    """Keep Rich emphasis while inheriting the application's body foreground."""

    emphasis = tuple(
        token
        for token in style.split()
        if token in {"bold", "italic", "underline", "strike"}
    )
    return " ".join(("class:tui.tool.text", *emphasis))


def _card_rich_lines(
    runtime: dict[str, Any],
    renderable: Any,
    *,
    width: int,
    border_style: str,
    maximum_lines: int | None = None,
    truncation_line: str = "… content truncated",
    capabilities: TerminalCapabilities | None = None,
    glyphs: TerminalGlyphs | None = None,
    bordered: bool = True,
) -> list[tuple[str, str]]:
    capabilities = capabilities or _terminal_capabilities()
    glyphs = glyphs or _terminal_glyphs(capabilities)
    lines = _render_rich_fragment_lines(
        runtime,
        renderable,
        width=width,
        capabilities=capabilities,
    )
    if maximum_lines is not None and len(lines) > maximum_lines:
        lines = [
            *lines[: max(0, maximum_lines - 1)],
            [("", truncation_line)],
        ]
    fragments: list[tuple[str, str]] = []
    for line in lines:
        prefix = f" {glyphs.vertical} " if bordered else "   "
        fragments.append((border_style, prefix))
        fragments.extend((_tool_detail_text_style(style), text) for style, text in line)
        fragments.append((border_style, "\n"))
    return fragments


def _format_duration(duration_ms: int) -> str:
    if duration_ms < 1_000:
        return f"{duration_ms}ms"
    seconds = duration_ms / 1_000
    if seconds < 100:
        return f"{seconds:.1f}s"
    return f"{int(seconds)}s"
