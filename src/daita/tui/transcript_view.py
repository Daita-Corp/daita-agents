"""Startup and transcript rendering for Daita's terminal UI."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any

from ..terminal_transcript import (
    PresentationBlockId,
    SemanticPosition,
    SemanticRange,
    SemanticViewportAnchor,
    TranscriptDocument,
)
from .capabilities import (
    MAX_RENDER_WIDTH as _MAX_RENDER_WIDTH,
    ResponsiveProjection,
    TerminalCapabilities,
    TerminalGlyphs,
    responsive_projection as _responsive_projection,
    terminal_capabilities as _terminal_capabilities,
    terminal_glyphs as _terminal_glyphs,
)
from .rendering import (
    render_markdown_fragments as _render_markdown_fragments,
    render_user_message_fragments as _render_user_message_fragments,
)
from .state import TerminalBlock, TerminalViewState
from .text import (
    display_clusters as _display_clusters,
    display_width as _display_width,
    pad_display_text as _pad_display_text,
    sanitize_terminal_text as _sanitize_terminal_text,
    truncate_display_text as _truncate_display_text,
    truncate_middle_display_text as _truncate_middle_display_text,
    wrap_display_text as _wrap_display_text,
)
from .tool_view import (
    ToolCardState,
    _canonical_tool_card_text,
    _render_tool_card_fragments,
    _render_tool_run_summary_fragments,
    _tool_run_summary_text,
)

_STARTUP_WORDMARK = (
    "████▄    █████   █████  ████████  █████ ",
    "██  ██  ██   ██    ██      ██    ██   ██",
    "██  ██  ███████    ██      ██    ███████",
    "████▀   ██   ██  █████     ██    ██   ██",
)
_STARTUP_QUICK_ACTIONS = ("/sources", "/catalog", "/help")
_STARTUP_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|credential|password|"
    r"private[_-]?key|secret|token)\b(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"
)


@dataclass(frozen=True, slots=True)
class _RenderedSelectionRow:
    """Semantic offsets for each terminal-cell boundary in one visible row."""

    block_id: PresentationBlockId
    revision: int
    cell_offsets: tuple[int, ...]
    first_selectable_cell: int
    last_selectable_cell: int

    def position_for_cell(self, cell: int) -> SemanticPosition:
        bounded = min(len(self.cell_offsets) - 1, max(0, cell))
        return SemanticPosition(
            self.block_id,
            self.cell_offsets[bounded],
            self.revision,
        )

    def selected_cells(self, start: int, end: int) -> tuple[int, int] | None:
        if start >= end or len(self.cell_offsets) < 2:
            return None
        row_start = self.cell_offsets[0]
        row_end = self.cell_offsets[-1]
        if end <= row_start or start >= row_end:
            return None
        first = (
            self.first_selectable_cell
            if start <= row_start
            else bisect_left(self.cell_offsets, start)
        )
        last = (
            self.last_selectable_cell
            if end >= row_end
            else bisect_left(self.cell_offsets, end)
        )
        first = max(self.first_selectable_cell, first)
        last = min(self.last_selectable_cell, last)
        return None if last <= first else (first, last)


@dataclass(frozen=True, slots=True)
class _RenderedTranscriptMap:
    """Exact transient navigation positions for the currently rendered rows."""

    row_positions: tuple[SemanticPosition, ...]
    block_offsets: Mapping[PresentationBlockId, tuple[int, ...]]
    block_rows: Mapping[PresentationBlockId, tuple[int, ...]]
    selection_rows: tuple[_RenderedSelectionRow | None, ...]
    block_indexes: Mapping[PresentationBlockId, int]

    def position_for_row(self, row: int) -> SemanticPosition | None:
        if not self.row_positions:
            return None
        bounded = min(len(self.row_positions) - 1, max(0, row))
        return self.row_positions[bounded]

    def row_for_anchor(
        self,
        document: TranscriptDocument,
        anchor: SemanticViewportAnchor | None,
    ) -> int | None:
        if anchor is None:
            return None
        current = document.reconcile_anchor(anchor)
        if current is None:
            return None
        offsets = self.block_offsets.get(current.position.block_id)
        rows = self.block_rows.get(current.position.block_id)
        if not offsets or not rows:
            return None
        index = max(0, bisect_right(offsets, current.position.offset) - 1)
        return rows[index]

    def position_for_cell(self, row: int, cell: int) -> SemanticPosition | None:
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError("rendered row must be an integer")
        if not isinstance(cell, int) or isinstance(cell, bool):
            raise TypeError("rendered cell must be an integer")
        if row < 0 or row >= len(self.selection_rows) or cell < 0:
            return None
        selected_row = self.selection_rows[row]
        return None if selected_row is None else selected_row.position_for_cell(cell)

    def selected_cells(
        self,
        document: TranscriptDocument,
        selected: SemanticRange,
    ) -> dict[int, tuple[int, int]]:
        current = document.reconcile_range(selected)
        if current is None:
            return {}
        start_index = self.block_indexes[current.start.block_id]
        end_index = self.block_indexes[current.end.block_id]
        ranges: dict[int, tuple[int, int]] = {}
        for row, row_map in enumerate(self.selection_rows):
            if row_map is None:
                continue
            block_index = self.block_indexes.get(row_map.block_id)
            if block_index is None or not start_index <= block_index <= end_index:
                continue
            local_start = current.start.offset if block_index == start_index else 0
            local_end = (
                current.end.offset
                if block_index == end_index
                else len(document.text(row_map.block_id))
            )
            cells = row_map.selected_cells(local_start, local_end)
            if cells is not None:
                ranges[row] = cells
        return ranges

    def selected_cells_for_row(
        self,
        document: TranscriptDocument,
        selected: SemanticRange,
        row: int,
    ) -> tuple[int, int] | None:
        """Resolve one rendered row without scanning the transcript."""

        if row < 0 or row >= len(self.selection_rows):
            return None
        row_map = self.selection_rows[row]
        current = document.reconcile_range(selected)
        if row_map is None or current is None:
            return None
        block_index = self.block_indexes.get(row_map.block_id)
        start_index = self.block_indexes.get(current.start.block_id)
        end_index = self.block_indexes.get(current.end.block_id)
        if (
            block_index is None
            or start_index is None
            or end_index is None
            or not start_index <= block_index <= end_index
        ):
            return None
        local_start = current.start.offset if block_index == start_index else 0
        local_end = (
            current.end.offset
            if block_index == end_index
            else len(document.text(row_map.block_id))
        )
        return row_map.selected_cells(local_start, local_end)


_EMPTY_RENDERED_TRANSCRIPT_MAP = _RenderedTranscriptMap((), {}, {}, (), {})


@dataclass(frozen=True, slots=True)
class _PendingRenderedBlockMap:
    """One rendered block awaiting its stable document identity."""

    block: TerminalBlock
    rendered_start: int
    row_offsets: tuple[int, ...]
    rendered_lines: tuple[str, ...]


def _render_startup_fragments(
    state: TerminalViewState,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
    glyphs: TerminalGlyphs | None = None,
) -> list[tuple[str, str]]:
    """Render one compact, width-bounded startup projection."""

    startup = state.startup
    if startup is None:
        return [
            (
                "class:tui.empty",
                "\n  Ask a question about your data, or type /help for commands.\n",
            )
        ]
    capabilities = capabilities or _terminal_capabilities()
    glyphs = glyphs or _terminal_glyphs(capabilities)
    safe_width = max(1, min(_MAX_RENDER_WIDTH, int(width)))
    marker = "…" if capabilities.unicode else "..."
    agent = _startup_safe_text(state.agent_label, fallback="agent")
    provider = _startup_safe_text(startup.provider_label, fallback="provider")
    model = _startup_safe_text(state.model_label, fallback="model")
    model_status = _startup_safe_text(startup.model_status, fallback="configured")
    version = _startup_safe_text(startup.version, fallback="unknown")
    home = _startup_safe_text(startup.agent_home, fallback="unavailable")
    conversation = _startup_safe_text(
        state.conversation_id,
        fallback="new",
    )
    connection_count = str(startup.source_count)
    catalog = (
        f"{startup.resource_count} "
        f"{'resource' if startup.resource_count == 1 else 'resources'}"
        f"{glyphs.separator}{startup.relationship_count} "
        f"{'relationship' if startup.relationship_count == 1 else 'relationships'}"
    )
    read_only = (
        glyphs.separator.join(
            _startup_safe_text(value, fallback="read")
            for value in startup.read_capabilities
        )
        if startup.read_capabilities
        else "None until a source is added"
    )
    warnings = tuple(
        _startup_safe_text(value, fallback="Configuration needs attention")
        for value in startup.warnings[:2]
    )
    model_text = f"{provider}{glyphs.separator}{model}{glyphs.separator}{model_status}"
    fragments: list[tuple[str, str]] = [("", "\n")]

    if safe_width >= 80 and capabilities.unicode:
        for line in _STARTUP_WORDMARK:
            fragments.append(
                (
                    "class:tui.identity",
                    _truncate_display_text(line, safe_width, marker=marker)[0] + "\n",
                )
            )
        version_line = f"DAITA {version}"
        fragments.append(("class:tui.metadata", f"{version_line}\n\n"))
    else:
        heading = _truncate_display_text(
            f"DAITA  {version}",
            safe_width,
            marker=marker,
        )[0]
        fragments.append(("class:tui.identity", f"{heading}\n\n"))

    if safe_width < 60:
        essential = (
            ("class:tui.status.ready", f"{glyphs.ready} Ready"),
            ("", f"{agent}{glyphs.separator}{model_text}"),
            ("class:tui.metadata", f"Connections: {connection_count}"),
            ("class:tui.metadata", catalog),
            ("class:tui.metadata", f"Read-only: {read_only}"),
        )
        for style, text in essential:
            bounded = _truncate_display_text(text, safe_width, marker=marker)[0]
            fragments.append((style, f"{bounded}\n"))
        if warnings:
            warning = f"{glyphs.warning} {warnings[0]}"
            bounded = _truncate_display_text(
                warning,
                safe_width,
                marker=marker,
            )[0]
            fragments.append(("class:tui.status.notice", f"{bounded}\n"))
    else:
        card_width = safe_width
        inner_width = max(1, card_width - 4)
        fragments.append(
            (
                "class:tui.rule",
                glyphs.top_left
                + glyphs.horizontal * max(0, card_width - 2)
                + glyphs.top_right
                + "\n",
            )
        )
        if safe_width >= 120:
            left_rows = (
                ("Status", f"{glyphs.ready} Ready"),
                ("Agent", agent),
                ("Model", model_text),
                ("Home", _truncate_middle_display_text(home, 40, marker=marker)),
            )
            right_rows = (
                ("Connections", connection_count),
                ("Catalog", catalog),
                ("Version", version),
                ("Conversation", conversation),
            )
            gap = 3
            left_width = max(1, (inner_width - gap) // 2)
            right_width = max(1, inner_width - gap - left_width)
            for index, (
                (left_label, left_value),
                (right_label, right_value),
            ) in enumerate(zip(left_rows, right_rows, strict=True)):
                left = _startup_cell(
                    left_label,
                    left_value,
                    left_width,
                    marker=marker,
                )
                right = _startup_cell(
                    right_label,
                    right_value,
                    right_width,
                    marker=marker,
                )
                style = "class:tui.status.ready" if index == 0 else "class:tui.startup"
                fragments.append(
                    (
                        style,
                        f"{glyphs.vertical} {left}{' ' * gap}{right} "
                        f"{glyphs.vertical}\n",
                    )
                )
            read_cell = _startup_cell(
                "Read-only",
                read_only,
                inner_width,
                marker=marker,
            )
            fragments.append(
                (
                    "class:tui.startup",
                    f"{glyphs.vertical} {read_cell} {glyphs.vertical}\n",
                )
            )
        else:
            rows = (
                ("Status", f"{glyphs.ready} Ready"),
                ("Agent", agent),
                ("Model", model_text),
                ("Home", _truncate_middle_display_text(home, 120, marker=marker)),
                ("Connections", connection_count),
                ("Catalog", catalog),
                ("Read-only", read_only),
                ("Conversation", conversation),
            )
            for index, (label, value) in enumerate(rows):
                cell = _startup_cell(label, value, inner_width, marker=marker)
                style = (
                    "class:tui.status.ready"
                    if index == 0
                    else (
                        "class:tui.metadata"
                        if label in {"Home", "Conversation"}
                        else "class:tui.startup"
                    )
                )
                fragments.append(
                    (
                        style,
                        f"{glyphs.vertical} {cell} {glyphs.vertical}\n",
                    )
                )
        for warning in warnings:
            warning_lines = _wrap_display_text(
                f"{glyphs.warning} Warning: {warning}",
                inner_width,
                maximum_lines=2,
                marker=marker,
            )
            for warning_line in warning_lines:
                fragments.append(
                    (
                        "class:tui.status.notice",
                        f"{glyphs.vertical} "
                        f"{_pad_display_text(warning_line, inner_width)} "
                        f"{glyphs.vertical}\n",
                    )
                )
        fragments.append(
            (
                "class:tui.rule",
                glyphs.bottom_left
                + glyphs.horizontal * max(0, card_width - 2)
                + glyphs.bottom_right
                + "\n",
            )
        )

    welcome = "Ask a question about your data, or type /help for commands."
    welcome = _truncate_display_text(welcome, safe_width, marker=marker)[0]
    fragments.extend(
        [
            ("class:tui.startup", f"\n{welcome}\n"),
            (
                "class:tui.prompt",
                _truncate_display_text(
                    "Quick actions: " + "  ".join(_STARTUP_QUICK_ACTIONS),
                    safe_width,
                    marker=marker,
                )[0]
                + "\n",
            ),
        ]
    )
    return fragments


def _render_startup_text(
    state: TerminalViewState,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
) -> str:
    capabilities = capabilities or _terminal_capabilities()
    return "".join(
        text
        for _style, text in _render_startup_fragments(
            state,
            width=width,
            capabilities=capabilities,
            glyphs=_terminal_glyphs(capabilities),
        )
    )


def _startup_cell(
    label: str,
    value: str,
    width: int,
    *,
    marker: str,
) -> str:
    label_width = min(width, max(10, len(label) + 2))
    safe_label = _truncate_display_text(label, label_width, marker=marker)[0]
    prefix = _pad_display_text(safe_label, label_width)
    value_width = max(0, width - label_width)
    safe_value = _truncate_display_text(value, value_width, marker=marker)[0]
    return _pad_display_text(prefix + safe_value, width)


def _startup_safe_text(value: object, *, fallback: str) -> str:
    safe = _sanitize_terminal_text(
        value,
        maximum=2_048,
        preserve_lines=False,
        fallback=fallback,
    )
    return _STARTUP_SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[redacted]",
        safe,
    )


def _fragment_line_metrics(fragments: Sequence[tuple[str, str]]) -> tuple[int, int]:
    """Count rendered rows without joining an unchanged transcript again."""

    line_count = 1
    last_line_width = 0
    for _style, text in fragments:
        newline_count = text.count("\n")
        line_count += newline_count
        if newline_count:
            last_line_width = len(text.rpartition("\n")[2])
        else:
            last_line_width += len(text)
    return line_count, last_line_width


def _semantic_offsets_for_rendered_lines(
    lines: Sequence[str],
    canonical_text: str,
) -> tuple[int, ...]:
    """Align visible content rows to stable canonical word offsets in order."""

    canonical_tokens = tuple(re.finditer(r"\w+", canonical_text, flags=re.UNICODE))
    occurrences: dict[str, list[int]] = {}
    for index, match in enumerate(canonical_tokens):
        occurrences.setdefault(match.group(0), []).append(index)

    cursor = 0
    matched: list[int | None] = []
    for line in lines:
        semantic_offset: int | None = None
        for visible in re.finditer(r"\w+", line, flags=re.UNICODE):
            indexes = occurrences.get(visible.group(0))
            if not indexes:
                continue
            occurrence = bisect_left(indexes, cursor)
            if occurrence >= len(indexes):
                continue
            token_index = indexes[occurrence]
            semantic_offset = canonical_tokens[token_index].start()
            cursor = token_index + 1
            break
        matched.append(semantic_offset)

    if not matched:
        return ()
    known = [(row, offset) for row, offset in enumerate(matched) if offset is not None]
    if not known:
        if len(matched) == 1:
            return (0,)
        return tuple(
            len(canonical_text) * row // (len(matched) - 1)
            for row in range(len(matched))
        )

    resolved = [0] * len(matched)
    first_row, first_offset = known[0]
    if first_row:
        for row in range(first_row):
            resolved[row] = first_offset * row // first_row
    for row, offset in known:
        resolved[row] = offset
    for (left_row, left_offset), (right_row, right_offset) in zip(
        known,
        known[1:],
    ):
        distance = right_row - left_row
        for row in range(left_row + 1, right_row):
            resolved[row] = left_offset + (
                (right_offset - left_offset) * (row - left_row) // distance
            )
    last_row, last_offset = known[-1]
    final_row = len(matched) - 1
    if last_row < final_row:
        distance = final_row - last_row
        for row in range(last_row + 1, len(matched)):
            resolved[row] = last_offset + (
                (len(canonical_text) - last_offset) * (row - last_row) // distance
            )
    return tuple(resolved)


def _rendered_row_cell_offsets(
    line: str,
    canonical_text: str,
    *,
    start: int,
    end: int,
) -> tuple[tuple[int, ...], int, int]:
    """Align one bounded rendered row to monotonic canonical text boundaries."""

    start = min(len(canonical_text), max(0, start))
    end = min(len(canonical_text), max(start, end))
    visible = _display_clusters(line)
    logical = _display_clusters(canonical_text[start:end])
    cell_count = sum(cluster[3] for cluster in visible)
    if cell_count < 1 or not logical or start == end:
        return (start,), 0, 0

    offsets: list[int | None] = [None] * (cell_count + 1)
    offsets[0] = start
    visible_cells: list[int] = []
    cell = 0
    for _text, _cluster_start, _cluster_end, width in visible:
        visible_cells.append(cell)
        cell += width

    matcher = SequenceMatcher(
        None,
        tuple(cluster[0] for cluster in visible),
        tuple(cluster[0] for cluster in logical),
        autojunk=False,
    )
    matched_cells: list[tuple[int, int]] = []
    for visible_index, logical_index, size in matcher.get_matching_blocks():
        for matched in range(size):
            actual = visible[visible_index + matched]
            semantic = logical[logical_index + matched]
            first_cell = visible_cells[visible_index + matched]
            width = actual[3]
            semantic_start = start + semantic[1]
            semantic_end = start + semantic[2]
            if width:
                matched_cells.append((first_cell, first_cell + width))
            offsets[first_cell] = semantic_start
            for inner_cell in range(1, width):
                offsets[first_cell + inner_cell] = semantic_start
            offsets[first_cell + width] = semantic_end

    offsets[-1] = end
    prior = start
    resolved: list[int] = []
    for value in offsets:
        if value is not None:
            prior = min(end, max(prior, value))
        resolved.append(prior)
    resolved[-1] = end
    first_selectable = min((cells[0] for cells in matched_cells), default=0)
    last_selectable = max((cells[1] for cells in matched_cells), default=0)
    return tuple(resolved), first_selectable, last_selectable


def _build_rendered_transcript_map(
    document: TranscriptDocument,
    pending: Sequence[_PendingRenderedBlockMap],
    *,
    line_count: int,
) -> _RenderedTranscriptMap:
    """Resolve stable semantic positions for every current rendered row."""

    row_positions: list[SemanticPosition | None] = [None] * max(1, line_count)
    selection_rows: list[_RenderedSelectionRow | None] = [None] * max(1, line_count)
    block_offsets: dict[PresentationBlockId, tuple[int, ...]] = {}
    block_rows: dict[PresentationBlockId, tuple[int, ...]] = {}
    for item in pending:
        block_id = item.block.presentation_id
        if block_id is None or not document.contains(block_id):
            continue
        text_length = len(document.text(block_id))
        for local_row, offset in enumerate(item.row_offsets):
            rendered_row = item.rendered_start + local_row
            if rendered_row >= len(row_positions):
                break
            row_positions[rendered_row] = document.position(
                block_id,
                min(text_length, max(0, offset)),
            )

        canonical_text = document.text(block_id)
        revision = document.position(block_id, 0).revision
        for local_row, line in enumerate(item.rendered_lines):
            rendered_row = item.rendered_start + local_row
            if rendered_row >= len(selection_rows) or not line:
                continue
            start = min(text_length, max(0, item.row_offsets[local_row]))
            end = text_length
            for later_offset in item.row_offsets[local_row + 1 :]:
                candidate = min(text_length, max(0, later_offset))
                if candidate > start:
                    end = candidate
                    break
            cell_offsets, first_selectable, last_selectable = (
                _rendered_row_cell_offsets(
                    line,
                    canonical_text,
                    start=start,
                    end=end,
                )
            )
            if len(cell_offsets) > 1 and cell_offsets[-1] > cell_offsets[0]:
                selection_rows[rendered_row] = _RenderedSelectionRow(
                    block_id,
                    revision,
                    cell_offsets,
                    first_selectable,
                    last_selectable,
                )

        inverse: dict[int, int] = {0: item.rendered_start}
        for local_row, offset in enumerate(item.row_offsets):
            inverse[min(text_length, max(0, offset))] = item.rendered_start + local_row
        ordered = tuple(sorted(inverse.items()))
        block_offsets[block_id] = tuple(offset for offset, _row in ordered)
        block_rows[block_id] = tuple(row for _offset, row in ordered)

    first = next((position for position in row_positions if position is not None), None)
    if first is None:
        return _EMPTY_RENDERED_TRANSCRIPT_MAP
    current = first
    resolved_positions: list[SemanticPosition] = []
    for position in row_positions:
        if position is not None:
            current = position
        resolved_positions.append(current)
    return _RenderedTranscriptMap(
        tuple(resolved_positions),
        block_offsets,
        block_rows,
        tuple(selection_rows),
        {block_id: index for index, block_id in enumerate(document.presentation_ids)},
    )


def _highlight_transcript_fragments(
    fragments: Sequence[tuple[str, str]],
    rendered_map: _RenderedTranscriptMap,
    document: TranscriptDocument,
    selected: SemanticRange | None,
) -> list[tuple[str, str]]:
    """Overlay selection style without changing or copying transcript text."""

    if selected is None:
        return list(fragments)
    selected_rows = rendered_map.selected_cells(document, selected)
    if not selected_rows:
        return list(fragments)

    highlighted: list[tuple[str, str]] = []
    row = 0
    cell = 0
    previous_selected = False

    def append(style: str, text: str) -> None:
        if not text:
            return
        if highlighted and highlighted[-1][0] == style:
            prior_style, prior_text = highlighted[-1]
            highlighted[-1] = (prior_style, prior_text + text)
        else:
            highlighted.append((style, text))

    for style, text in fragments:
        part_start = 0
        for index, character in enumerate(text):
            if character != "\n":
                continue
            part = text[part_start:index]
            for cluster, _start, _end, width in _display_clusters(part):
                selected_cells = selected_rows.get(row)
                is_selected = (
                    previous_selected
                    if width == 0
                    else (
                        selected_cells is not None
                        and cell < selected_cells[1]
                        and cell + width > selected_cells[0]
                    )
                )
                append(
                    (
                        f"{style} class:tui.transcript.selection".strip()
                        if is_selected
                        else style
                    ),
                    cluster,
                )
                cell += width
                if width:
                    previous_selected = is_selected
            append(style, "\n")
            row += 1
            cell = 0
            previous_selected = False
            part_start = index + 1
        for cluster, _start, _end, width in _display_clusters(text[part_start:]):
            selected_cells = selected_rows.get(row)
            is_selected = (
                previous_selected
                if width == 0
                else (
                    selected_cells is not None
                    and cell < selected_cells[1]
                    and cell + width > selected_cells[0]
                )
            )
            append(
                (
                    f"{style} class:tui.transcript.selection".strip()
                    if is_selected
                    else style
                ),
                cluster,
            )
            cell += width
            if width:
                previous_selected = is_selected
    return highlighted


def _highlight_transcript_line(
    fragments: Sequence[tuple[str, str]],
    selected_cells: tuple[int, int] | None,
) -> list[tuple[str, str]]:
    """Style one requested viewport row in work proportional to that row."""

    if selected_cells is None:
        return list(fragments)
    highlighted: list[tuple[str, str]] = []
    cell = 0
    previous_selected = False

    def append(style: str, text: str) -> None:
        if not text:
            return
        if highlighted and highlighted[-1][0] == style:
            prior_style, prior_text = highlighted[-1]
            highlighted[-1] = (prior_style, prior_text + text)
        else:
            highlighted.append((style, text))

    for style, text in fragments:
        for cluster, _start, _end, width in _display_clusters(text):
            is_selected = (
                previous_selected
                if width == 0
                else (cell < selected_cells[1] and cell + width > selected_cells[0])
            )
            append(
                (
                    f"{style} class:tui.transcript.selection".strip()
                    if is_selected
                    else style
                ),
                cluster,
            )
            cell += width
            if width:
                previous_selected = is_selected
    return highlighted


def _render_transcript_fragments(
    runtime: dict[str, Any],
    state: TerminalViewState,
    *,
    width: int,
    responsive: ResponsiveProjection | None = None,
    capabilities: TerminalCapabilities | None = None,
    glyphs: TerminalGlyphs | None = None,
    rendered_transcript_maps: list[_RenderedTranscriptMap] | None = None,
    highlight_selection: bool = True,
) -> list[tuple[str, str]]:
    capabilities = capabilities or _terminal_capabilities()
    glyphs = glyphs or _terminal_glyphs(capabilities)
    responsive = responsive or _responsive_projection(width, 24)
    if not state.blocks:
        return _render_startup_fragments(
            state,
            width=width,
            capabilities=capabilities,
            glyphs=glyphs,
        )
    tool_runs: dict[str, list[ToolCardState]] = {}
    for candidate in state.blocks:
        if candidate.kind != "tool":
            continue
        candidate_card = candidate.tool_card or state.tool_cards.get(candidate.text)
        if candidate_card is not None:
            tool_runs.setdefault(candidate_card.run_id, []).append(candidate_card)
    first_tool_call_by_run = {
        run_id: cards[0].call_id for run_id, cards in tool_runs.items() if cards
    }
    fragments: list[tuple[str, str]] = [("", "\n")]
    selectable_texts: list[str] = []
    rendered_row = 1
    pending_maps: list[_PendingRenderedBlockMap] = []
    for block in state.blocks:
        fragment_index = len(fragments)
        rendered_start = rendered_row
        selectable_text = block.text
        presentation_rows = 0
        if block.kind == "user":
            fragments.append(("class:tui.user.label", " You\n"))
            presentation_rows = 1
            fragments.extend(
                _render_user_message_fragments(
                    runtime,
                    block.text,
                    width=width,
                    capabilities=capabilities,
                )
            )
            fragments.append(("", "\n"))
        elif block.kind in {"assistant", "assistant.partial"}:
            fragments.append(("class:tui.assistant.label", " Daita\n"))
            presentation_rows = 1
            fragments.extend(
                _render_markdown_fragments(
                    runtime,
                    block.text,
                    width=width,
                    capabilities=capabilities,
                )
            )
            fragments.append(("", "\n"))
            selectable_text = _canonical_assistant_text(
                runtime,
                block.text,
                capabilities=capabilities,
            )
        elif block.kind == "metadata":
            fragments.append(("class:tui.metadata", f" {block.text}\n\n"))
        elif block.kind in {
            "local.status",
            "local.sources",
            "local.catalog",
            "local.settings",
        }:
            presentation = block.kind.removeprefix("local.")
            label = {
                "status": "Status",
                "sources": "Sources",
                "catalog": "Catalog",
                "settings": "Settings",
            }[presentation]
            fragments.extend(
                [
                    (
                        f"class:tui.local.{presentation}.label",
                        f" {label}\n",
                    ),
                    (
                        f"class:tui.local.{presentation}",
                        f" {block.text}\n\n",
                    ),
                ]
            )
            presentation_rows = 1
        elif block.kind == "tool":
            try:
                card = block.tool_card or state.tool_cards.get(block.text)
                if not isinstance(card, ToolCardState):
                    raise TypeError("tool card is unavailable")
                run_cards = tuple(tool_runs.get(card.run_id, (card,)))
                active = card.run_id == state.active_run_id or card.state in {
                    "queued",
                    "running",
                    "approval",
                }
                if active:
                    selectable_text = ""
                elif state.tool_history_run_id == card.run_id:
                    fragments.extend(
                        _render_tool_card_fragments(
                            card,
                            width=width,
                            runtime=runtime,
                            responsive=responsive,
                            capabilities=capabilities,
                            glyphs=glyphs,
                        )
                    )
                    selectable_text = _canonical_tool_card_text(
                        card,
                        runtime=runtime,
                        responsive=responsive,
                        capabilities=capabilities,
                        glyphs=glyphs,
                    )
                elif first_tool_call_by_run.get(card.run_id) == card.call_id:
                    fragments.extend(
                        _render_tool_run_summary_fragments(
                            run_cards,
                            width=width,
                            capabilities=capabilities,
                            glyphs=glyphs,
                        )
                    )
                    selectable_text = _tool_run_summary_text(
                        run_cards,
                        glyphs=glyphs,
                    )
                else:
                    selectable_text = ""
            except Exception:
                fragments.extend(
                    [
                        ("class:tui.tool.failure", " ! Tool status unavailable\n"),
                        ("", "\n"),
                    ]
                )
                selectable_text = "Tool status unavailable"
        else:
            fragments.extend(
                [
                    ("class:tui.local.label", " Local\n"),
                    ("class:tui.local", f" {block.text}\n\n"),
                ]
            )
            presentation_rows = 1
        selectable_texts.append(selectable_text)
        rendered_text = "".join(text for _style, text in fragments[fragment_index:])
        rendered_rows = rendered_text.count("\n")
        rendered_lines = rendered_text.split("\n")
        if rendered_text.endswith("\n"):
            rendered_lines.pop()
        for index in range(min(presentation_rows, len(rendered_lines))):
            rendered_lines[index] = ""
        row_offsets = _semantic_offsets_for_rendered_lines(
            rendered_lines,
            selectable_text,
        )
        pending_maps.append(
            _PendingRenderedBlockMap(
                block,
                rendered_start,
                row_offsets,
                tuple(rendered_lines),
            )
        )
        rendered_row += rendered_rows
    state._sync_transcript_document(tuple(selectable_texts), width=width)
    rendered_map = _build_rendered_transcript_map(
        state.transcript_document,
        pending_maps,
        line_count=rendered_row + 1,
    )
    if rendered_transcript_maps is not None:
        rendered_transcript_maps.clear()
        rendered_transcript_maps.append(rendered_map)
    if highlight_selection and state.transcript_selection.range is not None:
        return _highlight_transcript_fragments(
            fragments,
            rendered_map,
            state.transcript_document,
            state.transcript_selection.range,
        )
    return fragments


def _canonical_assistant_text(
    runtime: dict[str, Any],
    value: object,
    *,
    capabilities: TerminalCapabilities,
) -> str:
    """Project visible Markdown text once at the supported maximum width."""

    return "".join(
        text
        for _style, text in _render_markdown_fragments(
            runtime,
            value,
            width=_MAX_RENDER_WIDTH,
            capabilities=capabilities,
        )
    ).rstrip("\n")
