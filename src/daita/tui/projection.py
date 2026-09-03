"""Redact, bound, and project tools, transcripts, conversations, and artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from daita import ConversationRun, Transcript
from daita.llm.models import MessageRole, ToolCall, ToolResultBlock

from .models import ToolCardDetails, ToolCardState, ToolTablePreview, TranscriptBlock
from .sanitization import (
    MAX_RENDER_CHARACTERS,
    one_logical_line,
    sanitize_terminal_text,
    truncate_display_text,
)

_MAX_DETAIL_UTF8_BYTES = 16 * 1_024
_MAX_CODE_VISIBLE_LINES = 80
_EXPANDED_TABLE_COLUMNS = 20
_EXPANDED_TABLE_ROWS = 50
_MAX_CELL_DISPLAY_CHARACTERS = 240
SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
CAPABILITY_LABELS = {
    "toolbox_search": "Search toolboxes",
    "toolbox_load": "Load selected tools",
    "catalog_search": "Search catalog",
    "catalog_schema": "Read catalog schema",
    "catalog_inspect": "Inspect catalog resource",
    "catalog_traverse": "Follow relationships",
    "data_query": "Query data",
    "file_search": "Search workspace files",
    "file_read": "Read workspace file",
    "file_query": "Query workspace data",
    "artifact_create_document": "Create document",
    "artifact_edit_text": "Prepare workspace edit",
    "artifact_save_local": "Save artifact locally",
    "artifact_set_export_location": "Set export location",
    "memory_set": "Update memory",
    "skill_view": "Read skill",
    "skill_save": "Save skill",
    "skill_delete": "Delete skill",
}
_TOOL_ERROR_HEADINGS = {
    "postgresql_connect_failed": "Connection unavailable",
    "postgresql_credential_unavailable": "Credentials unavailable",
    "postgresql_credential_invalid": "Credentials invalid",
}


def redact_presentation_value(value: object, *, key: str = "") -> object:
    normalized_key = key.casefold().replace("-", "_")
    if key and any(part in normalized_key for part in SENSITIVE_KEY_PARTS):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_presentation_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_presentation_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_presentation_value(item) for item in value]
    return value


def bound_utf8_detail(value: str) -> str:
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


def bounded_json_text(value: object) -> str:
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
            sanitize_terminal_text(
                str(value),
                maximum=_MAX_DETAIL_UTF8_BYTES,
                preserve_lines=True,
                fallback="",
            ),
            ensure_ascii=False,
        )
    safe = sanitize_terminal_text(
        rendered,
        maximum=max(1, len(rendered) + 1),
        preserve_lines=True,
        fallback="{}",
    )
    return bound_utf8_detail(safe)


def project_tool_details(call: ToolCall, result: ToolResultBlock) -> ToolCardDetails:
    arguments = dict(call.arguments)
    output = dict(result.output)

    code_value = arguments.get("sql")
    code_language = "sql"
    if not isinstance(code_value, str):
        code_value = arguments.get("code")
        code_language = "text"

    presented_arguments = redact_presentation_value(arguments)
    assert isinstance(presented_arguments, dict)
    presented_arguments.pop("sql", None)
    presented_arguments.pop("code", None)
    arguments_text = (
        bounded_json_text(presented_arguments) if presented_arguments else None
    )

    if result.is_error:
        error = output.get("error")
        error_mapping = error if isinstance(error, dict) else {}
        error_code = sanitize_terminal_text(
            error_mapping.get("code"),
            maximum=128,
            preserve_lines=False,
            fallback="tool_failed",
        )
        error_message = bound_utf8_detail(
            sanitize_terminal_text(
                error_mapping.get("message"),
                maximum=max(1, _MAX_DETAIL_UTF8_BYTES),
                preserve_lines=True,
                fallback="Tool execution failed.",
            )
        )
        error_details = error_mapping.get("details")
        result_text = (
            bounded_json_text(redact_presentation_value(error_details))
            if error_details not in (None, {}, [])
            else None
        )
        error_heading = _TOOL_ERROR_HEADINGS.get(error_code, error_code)
        summary = one_logical_line(f"{error_heading} · {error_message}")
        return ToolCardDetails(
            summary=sanitize_terminal_text(
                summary,
                maximum=MAX_RENDER_CHARACTERS,
                preserve_lines=False,
                fallback="Failed.",
            ),
            code=_bounded_code_text(code_value),
            code_language=code_language if isinstance(code_value, str) else None,
            arguments_text=arguments_text,
            result_text=result_text,
            error_message=error_message,
        )

    data = output.get("data")
    data_mapping = dict(data) if isinstance(data, Mapping) else None
    if not isinstance(code_value, str) and data_mapping is not None:
        canonical_sql = data_mapping.get("canonical_sql")
        if isinstance(canonical_sql, str):
            code_value = canonical_sql
            code_language = "sql"

    table = project_table_preview(data_mapping)
    result_kind = sanitize_terminal_text(
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
            bounded_json_text(redact_presentation_value(result_projection))
            if result_projection
            else None
        )
        summary = (
            f"{table.recorded_rows} recorded rows · {table.recorded_columns} columns"
        )
    else:
        result_text = (
            bounded_json_text(redact_presentation_value(data))
            if data is not None
            else None
        )
        summary = result_kind
    if isinstance(code_value, str):
        summary = one_logical_line(code_value)
    elif result_text is not None and summary == "Tool result":
        summary = one_logical_line(result_text)
    return ToolCardDetails(
        summary=sanitize_terminal_text(
            summary,
            maximum=MAX_RENDER_CHARACTERS,
            preserve_lines=False,
            fallback="Completed.",
        ),
        code=_bounded_code_text(code_value),
        code_language=code_language if isinstance(code_value, str) else None,
        arguments_text=arguments_text,
        result_text=result_text,
        table=table,
    )


def project_table_preview(data: dict[str, object] | None) -> ToolTablePreview | None:
    if data is None:
        return None
    raw_rows = data.get("rows")
    if not isinstance(raw_rows, (list, tuple)):
        return None
    raw_columns = data.get("columns")
    if isinstance(raw_columns, (list, tuple)) and all(
        isinstance(column, str) for column in raw_columns
    ):
        columns = list(raw_columns)
    elif raw_rows and isinstance(raw_rows[0], dict):
        columns = list(raw_rows[0])
    else:
        return None

    projected_columns: list[str] = []
    for column in columns[:_EXPANDED_TABLE_COLUMNS]:
        projected, _truncated = truncate_display_text(
            sanitize_terminal_text(
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
    safe = sanitize_terminal_text(
        rendered,
        maximum=max(1, min(len(rendered) + 1, 4 * _MAX_CELL_DISPLAY_CHARACTERS)),
        preserve_lines=False,
        fallback="",
    )
    truncated_before_display = len(safe) < len(rendered)
    projected, display_truncated = truncate_display_text(
        safe,
        _MAX_CELL_DISPLAY_CHARACTERS,
    )
    return projected, truncated_before_display or display_truncated


def _bounded_code_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return sanitize_terminal_text(
        value,
        maximum=_MAX_CODE_VISIBLE_LINES * MAX_RENDER_CHARACTERS,
        preserve_lines=True,
        fallback="",
    )


def tool_result_error_code(result: ToolResultBlock) -> str:
    error = result.output.get("error")
    if isinstance(error, Mapping):
        return sanitize_terminal_text(
            error.get("code"),
            maximum=128,
            preserve_lines=False,
            fallback="tool_failed",
        )
    return "tool_failed"


def completed_tool_pairs(
    transcript: Transcript,
) -> tuple[tuple[ToolCall, ToolResultBlock | None], ...]:
    calls: list[ToolCall] = []
    call_ids: set[str] = set()
    results: dict[str, ToolResultBlock] = {}
    for message in transcript.messages:
        if message.role is MessageRole.ASSISTANT:
            for call in message.tool_calls:
                if call.id in call_ids:
                    raise ValueError("completed transcript repeats a tool call ID")
                call_ids.add(call.id)
                calls.append(call)
        elif message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise TypeError("tool transcript message contains non-tool content")
                if block.call_id in results:
                    raise ValueError("completed transcript repeats a tool result ID")
                results[block.call_id] = block
    if not set(results).issubset(call_ids):
        raise ValueError("completed transcript contains an unmatched tool result")
    return tuple((call, results.get(call.id)) for call in calls)


def artifact_delivery_messages(
    pairs: tuple[tuple[ToolCall, ToolResultBlock | None], ...],
) -> tuple[str, ...]:
    messages: list[str] = []
    created_artifact_ids: list[str] = []
    local_save_artifact_ids: set[str] = set()
    for call, result in pairs:
        if result is not None and not result.is_error:
            artifact = result.output.get("artifact")
            if isinstance(artifact, Mapping):
                artifact_id = artifact.get("artifact_id")
                if (
                    isinstance(artifact_id, str)
                    and artifact_id not in created_artifact_ids
                ):
                    created_artifact_ids.append(artifact_id)
        if call.name != "artifact_save_local" or result is None:
            continue
        artifact_id = call.arguments.get("artifact_id")
        if isinstance(artifact_id, str):
            local_save_artifact_ids.add(artifact_id)
        if not result.is_error:
            data = result.output.get("data")
            if not isinstance(data, Mapping):
                continue
            if data.get("mode") != "replace_bound_file":
                continue
            outcome = data.get("outcome")
            relative_path = sanitize_terminal_text(
                data.get("relative_path"),
                maximum=512,
                preserve_lines=False,
                fallback="the bound workspace file",
            )
            if outcome == "failed":
                messages.append(
                    f"Workspace file {relative_path} was not updated; the committed edit artifact remains available."
                )
            elif outcome == "uncertain":
                messages.append(
                    f"The update outcome for workspace file {relative_path} is uncertain; re-read the target before any further edit."
                )
            continue
        error = result.output.get("error")
        if not isinstance(artifact_id, str) or not isinstance(error, Mapping):
            continue
        code = sanitize_terminal_text(
            error.get("code"),
            maximum=128,
            preserve_lines=False,
            fallback="artifact_delivery_failed",
        )
        detail = sanitize_terminal_text(
            error.get("message"),
            maximum=512,
            preserve_lines=False,
            fallback="The artifact was not saved locally.",
        )
        safe_id = sanitize_terminal_text(
            artifact_id,
            maximum=64,
            preserve_lines=False,
            fallback="the internal artifact",
        )
        messages.append(
            f"Artifact {safe_id} remains available; local delivery failed: "
            f"{code}: {detail}"
        )
    for artifact_id in created_artifact_ids:
        if artifact_id in local_save_artifact_ids:
            continue
        safe_id = sanitize_terminal_text(
            artifact_id,
            maximum=64,
            preserve_lines=False,
            fallback="The internal artifact",
        )
        messages.append(
            f"Artifact {safe_id} was created internally but was not saved locally; "
            "no delivery completed."
        )
    return tuple(messages)


def project_transcript(
    transcript: Transcript,
    *,
    run_id: str,
    tools_expanded: bool = False,
) -> tuple[TranscriptBlock, ...]:
    blocks: list[TranscriptBlock] = []
    pairs = completed_tool_pairs(transcript)
    tool_index = 0
    for message in transcript.messages:
        if message.role is MessageRole.USER:
            text_parts = [
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "text", None)
            ]
            text = sanitize_terminal_text(
                "\n".join(part for part in text_parts if isinstance(part, str)),
                maximum=MAX_RENDER_CHARACTERS,
                preserve_lines=True,
                fallback="",
            )
            if text:
                blocks.append(
                    TranscriptBlock("user", f"{run_id}:user:{len(blocks)}", text)
                )
        elif message.role is MessageRole.ASSISTANT:
            text_parts = [
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "text", None)
            ]
            text = sanitize_terminal_text(
                "\n".join(part for part in text_parts if isinstance(part, str)),
                maximum=MAX_RENDER_CHARACTERS,
                preserve_lines=True,
                fallback="",
            )
            if text:
                blocks.append(
                    TranscriptBlock(
                        "assistant", f"{run_id}:assistant:{len(blocks)}", text
                    )
                )
            for call in message.tool_calls:
                result = pairs[tool_index][1] if tool_index < len(pairs) else None
                tool_index += 1
                label = CAPABILITY_LABELS.get(
                    getattr(call, "name", ""),
                    sanitize_terminal_text(
                        getattr(call, "name", "tool"),
                        maximum=64,
                        preserve_lines=False,
                        fallback="tool",
                    ),
                )
                card = ToolCardState(
                    run_id=run_id,
                    call_id=call.id,
                    capability_id=getattr(call, "name", None),
                    label=label,
                    state=(
                        "failed" if result is not None and result.is_error else "done"
                    ),
                    details=(
                        None if result is None else project_tool_details(call, result)
                    ),
                    expanded=tools_expanded,
                )
                blocks.append(
                    TranscriptBlock("tool", f"{run_id}:tool:{call.id}", "", card)
                )
    return tuple(blocks)


def project_conversation(
    runs: Sequence[ConversationRun],
    *,
    tools_expanded: bool = False,
) -> tuple[TranscriptBlock, ...]:
    """Project every persisted run in one conversation in turn order."""

    blocks: list[TranscriptBlock] = []
    for run in runs:
        blocks.extend(
            project_transcript(
                run.transcript,
                run_id=run.transcript.run.id,
                tools_expanded=tools_expanded,
            )
        )
    return tuple(blocks)


def format_status_label(
    *,
    agent: str,
    model: str,
    source: str,
    running: bool,
    run_status: str,
    activity: str | None,
) -> str:
    state = activity or ("working" if running else run_status or "ready")
    parts = [safe_part(agent, "agent"), safe_part(model, "model")]
    if source:
        parts.append(safe_part(source, "source"))
    parts.append(safe_part(state, "ready"))
    return " · ".join(parts)


def safe_part(value: object, fallback: str) -> str:
    return sanitize_terminal_text(
        value,
        maximum=64,
        preserve_lines=False,
        fallback=fallback,
    )


def approval_review_document(
    *,
    tool_name: str,
    capability_id: str,
    arguments_text: str | None,
    reason: str | None = None,
) -> tuple[str | None, bool]:
    """Return the review document and whether it is reviewable."""

    if arguments_text is None:
        return None, False
    header = (
        f"Tool: {sanitize_terminal_text(tool_name, maximum=128, preserve_lines=False, fallback='tool')}\n"
        f"Capability: {sanitize_terminal_text(capability_id, maximum=128, preserve_lines=False, fallback='capability')}\n"
        + (
            "Change: "
            + sanitize_terminal_text(
                reason,
                maximum=768,
                preserve_lines=False,
                fallback="Review this exact change once.",
            )
            + "\n"
            if reason is not None
            else ""
        )
        + "Arguments:\n"
    )
    document = header + arguments_text
    if looks_secret_shaped(arguments_text):
        return document, False
    return document, True


def looks_secret_shaped(value: str) -> bool:
    folded = value.casefold()
    return any(part in folded for part in SENSITIVE_KEY_PARTS)
