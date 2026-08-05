"""Lazy, full-screen presentation for the ready-agent terminal shell."""

from __future__ import annotations

import asyncio
import base64
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import io
import json
import os
import re
import sys
import threading
from typing import Any, TextIO
import unicodedata

from ._installation import PIPX_REPAIR_GUIDANCE
from ._json import freeze_json, thaw_json
from .capabilities import ApprovalDecision, ApprovalRequest
from .llm.models import MessageRole, ToolCall, ToolResultBlock
from .llm.pricing import CostEstimate, format_cost_estimate
from .loop.models import Transcript
from .observation import AgentEvent, AgentEventKind
from .terminal_transcript import (
    _cluster_cell_width,
    _next_grapheme_end,
    PresentationBlockId,
    SemanticPosition,
    SemanticRange,
    SemanticViewportAnchor,
    TranscriptFollowState,
    TranscriptDocument,
    TranscriptProjection,
    TranscriptSelection,
    TranscriptViewport,
    bounded_scroll_rows,
    bounded_selection_auto_scroll,
)

MAX_COMPOSER_CHARACTERS = 16_384
MAX_APPROVAL_DOCUMENT_CHARACTERS = 64 * 1_024
MAX_CLIPBOARD_UTF8_BYTES = 64 * 1_024
_MAX_COMPOSER_ROWS = 6
_MOUSE_SCROLL_LINES = 3
_SELECTION_AUTOSCROLL_LINES = 1
_CLIPBOARD_TIMEOUT_SECONDS = 1.0
_SELECTION_COMPLETE_HINT = "Selected · Ctrl+C copy · Esc clear"
_MOUSE_FAILURE_NOTICE = (
    "Mouse interaction unavailable; keyboard controls remain active."
)
_MAX_RENDER_CHARACTERS = 16_384
_MAX_DETAIL_UTF8_BYTES = 16 * 1_024
_MAX_CODE_VISIBLE_LINES = 80
_COLLAPSED_TABLE_ROWS = 10
_COLLAPSED_TABLE_COLUMNS = 12
_EXPANDED_TABLE_ROWS = 50
_EXPANDED_TABLE_COLUMNS = 20
_MAX_CELL_DISPLAY_CHARACTERS = 240
_MIN_RENDER_WIDTH = 20
_MAX_RENDER_WIDTH = 240
_MIN_USABLE_COLUMNS = 32
_MIN_READY_ROWS = 8
_MIN_APPROVAL_ROWS = 15
_MAX_QUEUED_EVENTS = 4_096
_MAX_EVENT_COUNTER = 999_999_999_999
_MAX_TRACKED_CONTEXT_CONVERSATIONS = 64
_ANIMATION_INTERVAL_SECONDS = 0.12
_STREAM_REPAINT_INTERVAL_SECONDS = 1 / 30
_RUNNING_GLYPHS = ("◐", "◓", "◑", "◒")
_ASCII_RUNNING_GLYPHS = ("~", "-", "~", "+")
_STARTUP_WORDMARK = (
    "████▄    █████   █████  ████████  █████ ",
    "██  ██  ██   ██    ██      ██    ██   ██",
    "██  ██  ███████    ██      ██    ███████",
    "████▀   ██   ██  █████     ██    ██   ██",
)
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
_STARTUP_QUICK_ACTIONS = tuple(
    command
    for command in ("/sources", "/catalog", "/help")
    if command in _SLASH_COMMAND_SURFACE
)[:3]
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_STARTUP_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|credential|password|"
    r"private[_-]?key|secret|token)\b(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"
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


_PASTED_TEXT_PLACEHOLDER_PATTERN = re.compile(r"\[Pasted Text #[1-9][0-9]*\]")
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


class TerminalTUIUnavailable(RuntimeError):
    """The enhanced application could not be admitted for this terminal."""


class TerminalUserInputError(ValueError):
    """One recoverable composer input error that should not close the shell."""


@dataclass(frozen=True, slots=True)
class TerminalCapabilities:
    """One process-local projection of output color and character support."""

    color_depth: str
    unicode: bool

    @property
    def no_color(self) -> bool:
        return self.color_depth == "none"

    @property
    def rich_color_system(self) -> str | None:
        return {
            "truecolor": "truecolor",
            "256": "256",
            "16": "standard",
            "none": None,
        }[self.color_depth]


@dataclass(frozen=True, slots=True)
class TerminalGlyphs:
    """Structural glyphs with a complete readable ASCII projection."""

    top_left: str
    top_right: str
    bottom_left: str
    bottom_right: str
    horizontal: str
    vertical: str
    prompt: str
    running: tuple[str, ...]
    ready: str
    success: str
    failure: str
    warning: str
    approval: str
    separator: str


@dataclass(frozen=True, slots=True)
class ResponsiveProjection:
    """Pure layout facts derived from the latest terminal size."""

    columns: int
    rows: int
    mode: str
    content_width: int
    collapsed_preview_columns: int
    expanded_preview_columns: int
    bordered_cards: bool
    stacked_metadata: bool
    two_sided_status: bool
    usable: bool
    minimum_rows: int
    transcript_rows: int


@dataclass(frozen=True, slots=True)
class StatusProjection:
    """One deterministic responsive projection of status metadata."""

    left: str
    right: str
    source_summary: str
    collapsed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TerminalStartupInfo:
    """Safe runtime facts rendered once when the focused shell opens."""

    version: str
    provider_label: str
    model_status: str
    agent_home: str
    source_count: int
    source_types: tuple[str, ...]
    source_names: tuple[str, ...]
    resource_count: int
    relationship_count: int
    read_capabilities: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_count", "resource_count", "relationship_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"startup {name} must be non-negative")
        for name in (
            "source_types",
            "source_names",
            "read_capabilities",
            "warnings",
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) for value in values):
                raise TypeError(f"startup {name} must contain strings")
            object.__setattr__(self, name, values)


@dataclass(slots=True)
class TerminalBlock:
    """One disposable transcript block shown in the current process."""

    kind: str
    text: str
    tool_card: ToolCardState | None = None
    presentation_id: PresentationBlockId | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class ClipboardResult:
    """Truthful outcome of one bounded terminal clipboard delivery attempt."""

    status: str
    mechanism: str
    message: str


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


@dataclass(frozen=True, slots=True)
class _ComposerDraft:
    """One process-local composer history entry with hidden pasted text."""

    text: str
    pasted_texts: tuple[tuple[str, str], ...] = ()
    next_paste_number: int = 1


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


@dataclass(slots=True)
class ApprovalPanelState:
    """One focused, bounded, process-local exact-approval projection."""

    tool_name: str
    capability_id: str
    arguments_text: str
    cursor_line: int = 0
    rendered_line_count: int = 1

    def move(self, amount: int) -> None:
        last_line = max(0, self.rendered_line_count - 1)
        self.cursor_line = min(last_line, max(0, self.cursor_line + amount))


class TerminalApprovalBridge:
    """Route the one existing approval callback to the active terminal surface."""

    def __init__(
        self,
        fallback: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]],
    ) -> None:
        if not callable(fallback):
            raise TypeError("approval fallback must be callable")
        self._fallback = fallback
        self._presenter: (
            Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None
        ) = None

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        presenter = self._presenter
        decision = await (
            self._fallback(request) if presenter is None else presenter(request)
        )
        if not isinstance(decision, ApprovalDecision):
            raise TypeError("approval presenter must return ApprovalDecision")
        return decision

    def install(
        self,
        presenter: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]],
    ) -> Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None:
        if not callable(presenter):
            raise TypeError("approval presenter must be callable")
        previous = self._presenter
        self._presenter = presenter
        return previous

    def restore(
        self,
        previous: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]] | None,
    ) -> None:
        self._presenter = previous


class TerminalObserverBridge:
    """Best-effort enqueue-only bridge from execution into terminal state."""

    def __init__(self) -> None:
        self._events: deque[AgentEvent] = deque(maxlen=_MAX_QUEUED_EVENTS)

    def __call__(self, event: AgentEvent, /) -> None:
        try:
            self._events.append(event)
        except Exception:
            pass

    def drain(self) -> tuple[AgentEvent, ...]:
        pending: list[AgentEvent] = []
        while True:
            try:
                pending.append(self._events.popleft())
            except IndexError:
                return tuple(pending)


@dataclass(slots=True)
class TerminalViewState:
    """Bounded, process-local state for the focused terminal shell."""

    agent_label: str
    model_label: str
    source_summary: str
    conversation_id: str | None = None
    context_capacity_tokens: int | None = None
    conversation_context_tokens: int | None = None
    startup: TerminalStartupInfo | None = None
    blocks: list[TerminalBlock] = field(default_factory=list)
    running: bool = False
    notice: str = ""
    transient_selection_hint: str = ""
    steps: int = 0
    total_tokens: int = 0
    estimated_cost: str = "$0"
    active_task: asyncio.Task[Any] | None = None
    active_run_id: str | None = None
    run_status: str = "ready"
    run_duration_ms: int | None = None
    model_duration_ms: int | None = None
    run_input_tokens: int = 0
    run_output_tokens: int = 0
    animation_frame: int = 0
    tool_cards: dict[str, ToolCardState] = field(default_factory=dict)
    approval_panel: ApprovalPanelState | None = None
    transcript_document: TranscriptDocument = field(
        default_factory=TranscriptDocument,
        repr=False,
        compare=False,
    )
    transcript_viewport: TranscriptViewport = field(
        default_factory=TranscriptViewport,
        repr=False,
        compare=False,
    )
    transcript_selection: TranscriptSelection = field(
        default_factory=TranscriptSelection,
        repr=False,
        compare=False,
    )
    _transcript_render_generation: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )
    _context_by_conversation: dict[str, int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _partial_assistant_block_id: PresentationBlockId | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _partial_assistant_run_id: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _partial_model_call_index: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _partial_counted_unseen: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _unrecorded_partial_run_id: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.context_capacity_tokens, "context_capacity_tokens"),
            (self.conversation_context_tokens, "conversation_context_tokens"),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.context_capacity_tokens == 0:
            raise ValueError("context_capacity_tokens must be positive")
        if (
            self.conversation_id is not None
            and self.conversation_context_tokens is not None
        ):
            self._remember_conversation_context(
                self.conversation_id,
                self.conversation_context_tokens,
            )

    def select_conversation(self, conversation_id: str | None) -> None:
        """Select one conversation and project its process-local context usage."""

        if conversation_id == self.conversation_id:
            return
        self.conversation_id = conversation_id
        self.conversation_context_tokens = (
            None
            if conversation_id is None
            else self._context_by_conversation.get(conversation_id)
        )

    @property
    def transcript_projection(self) -> TranscriptProjection | None:
        """Expose the viewport-owned cached projection to the renderer."""

        return self.transcript_viewport.projection

    @property
    def transcript_render_generation(self) -> int:
        """Return the disposable content generation used by the TUI render cache."""

        return self._transcript_render_generation

    def _mark_transcript_dirty(self) -> None:
        self._transcript_render_generation += 1

    def reconcile_transcript_selection(self) -> bool:
        """Clear a selection rather than copying text no longer projected."""

        had_state = self.transcript_selection.has_state
        survived = self.transcript_selection.reconcile(self.transcript_document)
        if had_state and not survived:
            self.transient_selection_hint = ""
            self.notice = (
                "Transcript selection cleared because visible content changed."
            )
        return survived

    def _remember_conversation_context(
        self,
        conversation_id: str,
        tokens: int,
    ) -> None:
        if conversation_id in self._context_by_conversation:
            del self._context_by_conversation[conversation_id]
        self._context_by_conversation[conversation_id] = tokens
        while len(self._context_by_conversation) > _MAX_TRACKED_CONTEXT_CONVERSATIONS:
            oldest = next(iter(self._context_by_conversation))
            del self._context_by_conversation[oldest]

    def append_plain(
        self,
        kind: str,
        value: object,
        *,
        maximum: int | None = _MAX_RENDER_CHARACTERS,
    ) -> None:
        safe = _sanitize_terminal_text(
            value,
            maximum=maximum,
            preserve_lines=True,
            fallback="",
        )
        if safe:
            self._append_block(TerminalBlock(kind, safe))

    def append_user(self, message: str) -> None:
        self.append_plain("user", message, maximum=None)

    def apply_model_text_delta(
        self,
        run_id: str,
        model_call_index: int,
        text: str,
    ) -> None:
        """Append one coalesced fragment batch to the sole disposable partial."""

        if not isinstance(run_id, str) or not run_id:
            return
        if (
            not isinstance(model_call_index, int)
            or isinstance(model_call_index, bool)
            or model_call_index < 1
        ):
            return
        safe = _sanitize_terminal_text(
            text,
            maximum=None,
            preserve_lines=True,
            fallback="",
        )
        if not safe:
            return
        block = self._partial_assistant_block()
        if block is None or (
            self._partial_assistant_run_id,
            self._partial_model_call_index,
        ) != (run_id, model_call_index):
            self._remove_partial_assistant()
            counted_unseen = (
                self.transcript_viewport.state is TranscriptFollowState.REVIEWING
            )
            block = TerminalBlock("assistant.partial", safe)
            self._append_block(block)
            self._partial_assistant_block_id = block.presentation_id
            self._partial_assistant_run_id = run_id
            self._partial_model_call_index = model_call_index
            self._partial_counted_unseen = counted_unseen
            self._unrecorded_partial_run_id = None
            return
        prior_text = block.text
        block.text += safe
        assert block.presentation_id is not None
        if self.transcript_document.text(block.presentation_id) == prior_text:
            self.transcript_document.replace(block.presentation_id, block.text)
            self.reconcile_transcript_selection()
        self._mark_transcript_dirty()

    def append_local(self, presentation: str, value: object) -> None:
        kind = (
            f"local.{presentation}"
            if presentation in {"status", "sources", "catalog", "settings"}
            else "local"
        )
        self.append_plain(kind, value)

    def apply_result(self, result: Any) -> None:
        previous_conversation = self.conversation_id
        candidate_conversation = getattr(result, "conversation_id", None)
        if isinstance(candidate_conversation, str) and candidate_conversation:
            self.select_conversation(candidate_conversation)
            if previous_conversation is None:
                self.append_plain(
                    "metadata",
                    f"Conversation  {candidate_conversation}",
                )

        result_run_id = getattr(result, "run_id", None)
        final_text = getattr(result, "final_text", None)
        if final_text is not None:
            safe_answer = _sanitize_terminal_text(
                final_text,
                maximum=None,
                preserve_lines=True,
                fallback="(empty response)",
            )
        else:
            kind = getattr(getattr(result, "kind", None), "value", None)
            reason = _sanitize_terminal_text(
                getattr(result, "reason", None),
                maximum=256,
                preserve_lines=False,
                fallback="failed",
            )
            safe_answer = f"{kind or 'failed'}: {reason}"
        if final_text is not None and self._reconcile_partial_assistant(
            result_run_id,
            safe_answer,
        ):
            pass
        else:
            if final_text is None and isinstance(result_run_id, str):
                self._remove_partial_assistant(
                    run_id=result_run_id,
                    unrecorded=True,
                )
            self._append_block(TerminalBlock("assistant", safe_answer))
        for receipt in tuple(getattr(result, "artifact_deliveries", ())):
            filename = getattr(receipt, "filename", None)
            saved_path = getattr(receipt, "saved_path", None)
            if not isinstance(filename, str) or not isinstance(saved_path, str):
                continue
            self.append_plain(
                "artifact.delivery",
                f"Saved {filename} to {saved_path}",
                maximum=None,
            )

        steps = getattr(result, "steps", 0)
        usage = getattr(result, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0)
        cost_estimate = getattr(usage, "cost_estimate", None)
        self.steps = steps if isinstance(steps, int) and steps >= 0 else 0
        self.total_tokens = (
            total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else 0
        )
        self.estimated_cost = _sanitize_terminal_text(
            (
                format_cost_estimate(cost_estimate)
                if isinstance(cost_estimate, CostEstimate)
                else "cost unavailable"
            ),
            maximum=96,
            preserve_lines=False,
            fallback="cost unavailable",
        )
        kind = getattr(getattr(result, "kind", None), "value", None)
        self.run_status = (
            "ready"
            if final_text is not None or kind == "completed"
            else _sanitize_terminal_text(
                kind,
                maximum=32,
                preserve_lines=False,
                fallback="failed",
            )
        )
        if (
            isinstance(result_run_id, str)
            and result_run_id
            and self.active_run_id == result_run_id
        ):
            result_kind = (
                kind
                if isinstance(kind, str) and kind
                else ("completed" if final_text is not None else "failed")
            )
            result_reason = _sanitize_terminal_text(
                getattr(result, "reason", None),
                maximum=128,
                preserve_lines=False,
                fallback=result_kind,
            )
            self._settle_run_cards(result_run_id, result_kind, result_reason)
            self.active_run_id = None
        if (
            isinstance(result_run_id, str)
            and self._unrecorded_partial_run_id == result_run_id
        ):
            self.notice = (
                "Partial assistant output was interrupted and was not recorded."
            )
            self._unrecorded_partial_run_id = None
        else:
            self.notice = ""

    def hydrate_transcript(
        self,
        transcript: Transcript,
        *,
        run_id: str,
        initial: bool = False,
    ) -> None:
        """Hydrate and canonically reorder one completed run's tool cards."""

        if not isinstance(transcript, Transcript):
            raise TypeError("completed transcript must be a Transcript")
        if transcript.run.id != run_id:
            raise ValueError("completed transcript belongs to a different run")

        pairs = _completed_tool_pairs(transcript)
        canonical_ids = {call.id for call, _result in pairs}
        prior_cards = {
            call_id: card
            for call_id, card in self.tool_cards.items()
            if card.run_id == run_id
        }
        canonical_cards: list[ToolCardState] = []
        for call, result in pairs:
            card = prior_cards.get(call.id)
            capability_id = None if card is None else card.capability_id
            label = (
                _CAPABILITY_LABELS.get(capability_id or "")
                if capability_id is not None
                else None
            )
            if label is None:
                label = _sanitize_terminal_text(
                    call.name,
                    maximum=128,
                    preserve_lines=False,
                    fallback="Tool call",
                )
            if card is None:
                card = ToolCardState(
                    run_id=run_id,
                    call_id=call.id,
                    capability_id=capability_id,
                    label=label,
                )
            else:
                card.label = label

            if result is not None:
                already_hydrated = card.details is not None
                card.details = _project_tool_details(call, result)
                if result.is_error:
                    card.state = "failed"
                    card.error_code = _tool_result_error_code(result)
                    if not already_hydrated:
                        card.expanded = True
                else:
                    card.state = "succeeded"
                    card.error_code = None
                    if not already_hydrated:
                        card.expanded = False
            canonical_cards.append(card)

        target_indexes = [
            index
            for index, block in enumerate(self.blocks)
            if block.kind == "tool"
            and block.tool_card is not None
            and block.tool_card.run_id == run_id
        ]
        insertion_index = min(target_indexes, default=len(self.blocks))
        retained: list[TerminalBlock] = []
        retained_before_insertion = 0
        for index, block in enumerate(self.blocks):
            is_target = (
                block.kind == "tool"
                and block.tool_card is not None
                and block.tool_card.run_id == run_id
            )
            if is_target:
                continue
            if index < insertion_index:
                retained_before_insertion += 1
            retained.append(block)
        prior_presentation_ids = {
            block.tool_card.call_id: block.presentation_id
            for block in self.blocks
            if block.kind == "tool"
            and block.tool_card is not None
            and block.tool_card.run_id == run_id
        }
        canonical_blocks = [
            TerminalBlock(
                "tool",
                card.call_id,
                tool_card=card,
                presentation_id=prior_presentation_ids.get(card.call_id),
            )
            for card in canonical_cards
        ]
        delivery_failures = _artifact_delivery_messages(pairs)
        canonical_blocks.extend(
            TerminalBlock("artifact.delivery", message) for message in delivery_failures
        )
        self.blocks = [
            *retained[:retained_before_insertion],
            *canonical_blocks,
            *retained[retained_before_insertion:],
        ]

        for call_id, card in tuple(self.tool_cards.items()):
            if card.run_id == run_id and call_id not in canonical_ids:
                del self.tool_cards[call_id]
        for card in canonical_cards:
            self.tool_cards[card.call_id] = card
        self._sync_transcript_document(
            tuple(
                (
                    self.transcript_document.text(block.presentation_id)
                    if block.presentation_id is not None
                    and self.transcript_document.contains(block.presentation_id)
                    else block.text
                )
                for block in self.blocks
            ),
            count_as_unseen=not initial,
        )
        self._mark_transcript_dirty()

    def toggle_expanded_detail(self) -> bool:
        """Toggle the most recent completed hydrated card in this process."""

        for block in reversed(self.blocks):
            card = block.tool_card
            if (
                block.kind == "tool"
                and card is not None
                and card.state in {"succeeded", "failed"}
                and card.details is not None
            ):
                card.expanded = not card.expanded
                self._mark_transcript_dirty()
                return True
        return False

    def apply_event(self, event: AgentEvent) -> None:
        """Project one bounded observation event into disposable view state."""

        if not isinstance(event, AgentEvent):
            raise TypeError("terminal event must be AgentEvent")
        if event.kind is AgentEventKind.RUN_STARTED:
            self._remove_partial_assistant()
            self._unrecorded_partial_run_id = None
            self.active_run_id = event.run_id
            self.running = True
            self.run_status = "working"
            self.run_duration_ms = None
            self.model_duration_ms = None
            self.run_input_tokens = 0
            self.run_output_tokens = 0
            self.steps = 0
            self.total_tokens = 0
            self.estimated_cost = "cost unavailable"
            self.animation_frame = 0
            return
        if event.kind is AgentEventKind.MODEL_TEXT_DELTA:
            fields = _model_text_event_fields(event)
            if fields is not None:
                model_call_index, text = fields
                self.apply_model_text_delta(event.run_id, model_call_index, text)
            return
        if event.kind is AgentEventKind.MODEL_COMPLETED:
            completed_model_call_index = _event_counter(
                event.data.get("model_call_index")
            )
            if (
                completed_model_call_index is not None
                and self._partial_assistant_run_id == event.run_id
                and (
                    self._partial_model_call_index != completed_model_call_index
                    or event.data.get("has_tool_calls") is True
                    or event.data.get("has_text") is not True
                )
            ):
                self._remove_partial_assistant(run_id=event.run_id)
            self.model_duration_ms = _event_counter(event.data.get("duration_ms"))
            context_input_tokens = _event_counter(
                event.data.get("context_input_tokens")
            )
            if context_input_tokens is None:
                context_input_tokens = _event_counter(event.data.get("input_tokens"))
            if context_input_tokens is not None:
                self._remember_conversation_context(
                    event.conversation_id,
                    context_input_tokens,
                )
                if self.conversation_id in {None, event.conversation_id}:
                    self.conversation_context_tokens = context_input_tokens
            self.run_input_tokens = min(
                _MAX_EVENT_COUNTER,
                self.run_input_tokens
                + (_event_counter(event.data.get("input_tokens")) or 0),
            )
            self.run_output_tokens = min(
                _MAX_EVENT_COUNTER,
                self.run_output_tokens
                + (_event_counter(event.data.get("output_tokens")) or 0),
            )
            self.total_tokens = min(
                _MAX_EVENT_COUNTER,
                self.run_input_tokens + self.run_output_tokens,
            )
            if self.running:
                self.run_status = "working"
            return
        if event.kind is AgentEventKind.TOOL_STARTED:
            self._remove_partial_assistant(run_id=event.run_id)
            card = self._card_for_event(event)
            if card is None:
                return
            card.state = "running"
            card.duration_ms = None
            card.error_code = None
            self.run_status = "querying"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.APPROVAL_REQUESTED:
            card = self._card_for_event(event)
            if card is None:
                return
            card.state = "approval"
            card.approval_outcome = None
            card.expanded = True
            self.run_status = "approval"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.APPROVAL_DECIDED:
            card = self._card_for_event(event)
            if card is None:
                return
            outcome = _event_text(
                event.data.get("outcome"),
                maximum=32,
                fallback="failed",
            )
            card.approval_outcome = outcome
            if outcome == "approved":
                card.state = "running"
                card.expanded = False
                self.run_status = "querying"
            else:
                card.state = "failed"
                card.expanded = True
                card.error_code = (
                    "approval_denied" if outcome == "denied" else "approval_failed"
                )
                self.run_status = "working"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.TOOL_COMPLETED:
            card = self._card_for_event(event)
            if card is None:
                return
            card.duration_ms = _event_counter(event.data.get("duration_ms"))
            success = event.data.get("success")
            if success is True:
                card.state = "succeeded"
                card.error_code = None
                card.expanded = False
            else:
                card.state = "failed"
                card.expanded = True
                card.error_code = _event_text(
                    event.data.get("error_code"),
                    maximum=128,
                    fallback="tool_failed",
                )
            self.run_status = "working"
            self._mark_transcript_dirty()
            return
        if event.kind is AgentEventKind.RUN_COMPLETED:
            self.run_duration_ms = _event_counter(event.data.get("duration_ms"))
            self.steps = _event_counter(event.data.get("steps")) or 0
            self.run_input_tokens = _event_counter(event.data.get("input_tokens")) or 0
            self.run_output_tokens = (
                _event_counter(event.data.get("output_tokens")) or 0
            )
            self.total_tokens = _event_counter(event.data.get("total_tokens")) or 0
            self.estimated_cost = _event_text(
                event.data.get("cost_display"),
                maximum=96,
                fallback="cost unavailable",
            )
            exit_kind = _event_text(
                event.data.get("exit_kind"),
                maximum=32,
                fallback="failed",
            )
            reason = _event_text(
                event.data.get("reason"),
                maximum=128,
                fallback=exit_kind,
            )
            if exit_kind != "completed":
                self._remove_partial_assistant(
                    run_id=event.run_id,
                    unrecorded=True,
                )
            self._settle_run_cards(event.run_id, exit_kind, reason)
            if self.active_run_id == event.run_id:
                self.active_run_id = None
            self.running = False
            self.run_status = "ready" if exit_kind == "completed" else exit_kind

    def settle_cancelled_run(self) -> bool:
        removed_partial = self._remove_partial_assistant(
            run_id=self.active_run_id,
            unrecorded=True,
        )
        had_unrecorded_partial = (
            removed_partial or self._unrecorded_partial_run_id is not None
        )
        run_id = self.active_run_id
        if run_id is not None:
            self._settle_run_cards(run_id, "interrupted", "cancelled")
            self.active_run_id = None
            self.run_status = "interrupted"
        self.running = False
        return had_unrecorded_partial

    def _partial_assistant_block(self) -> TerminalBlock | None:
        block_id = self._partial_assistant_block_id
        if block_id is None:
            return None
        for block in self.blocks:
            if block.presentation_id == block_id and block.kind == "assistant.partial":
                return block
        self._clear_partial_assistant_identity()
        return None

    def _remove_partial_assistant(
        self,
        *,
        run_id: str | None = None,
        unrecorded: bool = False,
    ) -> bool:
        if run_id is not None and self._partial_assistant_run_id != run_id:
            return False
        block = self._partial_assistant_block()
        partial_run_id = self._partial_assistant_run_id
        if block is None:
            return False
        self.blocks.remove(block)
        if block.presentation_id is not None and self.transcript_document.contains(
            block.presentation_id
        ):
            self.transcript_document.remove(block.presentation_id)
            self.reconcile_transcript_selection()
        if self._partial_counted_unseen:
            self.transcript_viewport.record_removed()
        self._clear_partial_assistant_identity()
        if unrecorded:
            self._unrecorded_partial_run_id = partial_run_id
        self._mark_transcript_dirty()
        return True

    def _reconcile_partial_assistant(
        self,
        run_id: object,
        finalized_text: str,
    ) -> bool:
        if not isinstance(run_id, str) or self._partial_assistant_run_id != run_id:
            return False
        block = self._partial_assistant_block()
        if block is None:
            return False
        prior_text = block.text
        block.kind = "assistant"
        block.text = finalized_text
        assert block.presentation_id is not None
        if self.transcript_document.text(block.presentation_id) == prior_text:
            self.transcript_document.replace(block.presentation_id, finalized_text)
            self.reconcile_transcript_selection()
        self._clear_partial_assistant_identity()
        self._unrecorded_partial_run_id = None
        self._mark_transcript_dirty()
        return True

    def _clear_partial_assistant_identity(self) -> None:
        self._partial_assistant_block_id = None
        self._partial_assistant_run_id = None
        self._partial_model_call_index = None
        self._partial_counted_unseen = False

    def _card_for_event(self, event: AgentEvent) -> ToolCardState | None:
        call_id = _event_text(
            event.data.get("call_id"),
            maximum=256,
            fallback="",
        )
        if not call_id:
            return None
        capability_id = _optional_event_text(
            event.data.get("capability_id"),
            maximum=256,
        )
        tool_name = _event_text(
            event.data.get("tool_name"),
            maximum=128,
            fallback="Tool call",
        )
        label = _CAPABILITY_LABELS.get(capability_id or "", tool_name)
        label = _sanitize_terminal_text(
            label,
            maximum=128,
            preserve_lines=False,
            fallback="Tool call",
        )
        card = self.tool_cards.get(call_id)
        if card is None or card.run_id != event.run_id:
            card = ToolCardState(
                run_id=event.run_id,
                call_id=call_id,
                capability_id=capability_id,
                label=label,
            )
            self.tool_cards[call_id] = card
            self._append_block(TerminalBlock("tool", call_id, tool_card=card))
        else:
            if capability_id is not None:
                card.capability_id = capability_id
            if capability_id is not None or card.label == "Tool call":
                card.label = label
        return card

    def _settle_run_cards(
        self,
        run_id: str,
        exit_kind: str,
        reason: str,
    ) -> None:
        changed = False
        for card in self.tool_cards.values():
            if card.run_id != run_id or card.state not in {
                "queued",
                "running",
                "approval",
            }:
                continue
            card.state = "failed"
            card.expanded = True
            card.error_code = (
                "cancelled"
                if exit_kind == "interrupted" or reason == "cancelled"
                else "observation_incomplete"
            )
            changed = True
        if changed:
            self._mark_transcript_dirty()

    def _append_block(self, block: TerminalBlock) -> None:
        """Admit one block to both disposable views under one stable identity."""

        snapshot = self.transcript_document.append(block.text)
        block.presentation_id = snapshot.id
        self.blocks.append(block)
        self.transcript_viewport.record_appended()
        self._mark_transcript_dirty()

    def _sync_transcript_document(
        self,
        selectable_texts: Sequence[str],
        *,
        width: int | None = None,
        count_as_unseen: bool = True,
    ) -> None:
        """Reconcile renderer-owned selectable projections without persisting them."""

        if len(selectable_texts) != len(self.blocks):
            raise ValueError("each terminal block requires one selectable projection")
        prior_ids = set(self.transcript_document.presentation_ids)
        current_ids: list[PresentationBlockId] = []
        seen: set[PresentationBlockId] = set()
        for block, selectable_text in zip(
            self.blocks,
            selectable_texts,
            strict=True,
        ):
            block_id = block.presentation_id
            if (
                block_id is None
                or block_id in seen
                or not self.transcript_document.contains(block_id)
            ):
                block_id = self.transcript_document.append(selectable_text).id
                block.presentation_id = block_id
            else:
                self.transcript_document.replace(block_id, selectable_text)
            current_ids.append(block_id)
            seen.add(block_id)
        for block_id in tuple(self.transcript_document.presentation_ids):
            if block_id not in seen:
                self.transcript_document.remove(block_id)
        self.transcript_document.reorder(tuple(current_ids))
        self.reconcile_transcript_selection()
        if count_as_unseen:
            self.transcript_viewport.record_appended(
                sum(block_id not in prior_ids for block_id in current_ids)
            )
        if width is not None:
            self.transcript_viewport.projection_for(
                self.transcript_document,
                width=width,
            )


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


def _terminal_capabilities(
    output: Any = None,
    *,
    text_stream: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> TerminalCapabilities:
    """Detect bounded semantic color and Unicode support without terminal I/O."""

    environment = os.environ if environ is None else environ
    if "NO_COLOR" in environment:
        color_depth = "none"
    else:
        color_depth = _detected_color_depth(output, environment)

    ascii_override = environment.get("DAITA_ASCII", "").strip().casefold()
    unicode_supported = ascii_override not in {"1", "true", "yes", "on"}
    encoding = getattr(text_stream, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        encoding = getattr(output, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        encoding = environment.get("PYTHONIOENCODING", "").partition(":")[0]
    if isinstance(encoding, str) and encoding:
        try:
            "╭✓◐›●".encode(encoding)
        except (LookupError, UnicodeEncodeError):
            unicode_supported = False
    locale_name = (
        environment.get("LC_ALL")
        or environment.get("LC_CTYPE")
        or environment.get("LANG")
        or ""
    ).strip()
    if (
        locale_name.casefold() in {"c", "posix"}
        and environment.get("PYTHONUTF8", "").strip() != "1"
    ):
        unicode_supported = False
    return TerminalCapabilities(
        color_depth=color_depth,
        unicode=unicode_supported,
    )


def _detected_color_depth(
    output: Any,
    environment: Mapping[str, str],
) -> str:
    color_term = environment.get("COLORTERM", "").strip().casefold()
    if color_term in {"truecolor", "24bit"}:
        return "truecolor"
    term = environment.get("TERM", "").strip().casefold()
    if "direct" in term or "truecolor" in term:
        return "truecolor"
    if "256color" in term:
        return "256"
    try:
        depth = str(output.get_default_color_depth()).casefold()
    except (AttributeError, OSError, TypeError, ValueError):
        depth = ""
    if "true" in depth or "24" in depth:
        return "truecolor"
    if "256" in depth or "8_bit" in depth:
        return "256"
    if "4_bit" in depth or "16" in depth or "standard" in depth:
        return "16"
    if "1_bit" in depth or "monochrome" in depth:
        return "none"
    if term and term not in {"dumb", "unknown"}:
        return "16"
    return "truecolor"


def _terminal_glyphs(capabilities: TerminalCapabilities) -> TerminalGlyphs:
    if capabilities.unicode:
        return TerminalGlyphs(
            top_left="╭",
            top_right="╮",
            bottom_left="╰",
            bottom_right="╯",
            horizontal="─",
            vertical="│",
            prompt="›",
            running=_RUNNING_GLYPHS,
            ready="●",
            success="✓",
            failure="!",
            warning="!",
            approval="!",
            separator=" · ",
        )
    return TerminalGlyphs(
        top_left="+",
        top_right="+",
        bottom_left="+",
        bottom_right="+",
        horizontal="-",
        vertical="|",
        prompt=">",
        running=_ASCII_RUNNING_GLYPHS,
        ready="OK",
        success="OK",
        failure="!",
        warning="!",
        approval="!",
        separator=" | ",
    )


def _terminal_size(output: Any) -> tuple[int, int]:
    try:
        size = output.get_size()
        columns = int(size.columns)
        rows = int(size.rows)
    except (AttributeError, OSError, TypeError, ValueError):
        return 80, 24
    return max(1, columns), max(1, rows)


def _terminal_size_polling_interval(
    *,
    platform: str | None = None,
    main_thread: bool | None = None,
) -> float | None:
    """Use polling only where prompt-toolkit cannot rely on SIGWINCH."""

    current_platform = sys.platform if platform is None else platform
    running_on_main_thread = (
        threading.current_thread() is threading.main_thread()
        if main_thread is None
        else main_thread
    )
    if current_platform == "win32" or not running_on_main_thread:
        return 0.5
    return None


def _responsive_projection(
    columns: int,
    rows: int,
    *,
    approving: bool = False,
) -> ResponsiveProjection:
    safe_columns = max(1, int(columns))
    safe_rows = max(1, int(rows))
    if safe_columns >= 100:
        mode = "full"
        collapsed_columns = _COLLAPSED_TABLE_COLUMNS
        expanded_columns = _EXPANDED_TABLE_COLUMNS
    elif safe_columns >= 70:
        mode = "compact"
        collapsed_columns = 8
        expanded_columns = 12
    else:
        mode = "narrow"
        collapsed_columns = 4
        expanded_columns = 6
    minimum_rows = _MIN_APPROVAL_ROWS if approving else _MIN_READY_ROWS
    transcript_rows = max(0, safe_rows - minimum_rows + 1)
    return ResponsiveProjection(
        columns=safe_columns,
        rows=safe_rows,
        mode=mode,
        content_width=max(
            _MIN_RENDER_WIDTH,
            min(_MAX_RENDER_WIDTH, safe_columns - 2),
        ),
        collapsed_preview_columns=collapsed_columns,
        expanded_preview_columns=expanded_columns,
        bordered_cards=mode != "narrow",
        stacked_metadata=mode == "narrow",
        two_sided_status=mode == "full",
        usable=(
            safe_columns >= _MIN_USABLE_COLUMNS
            and safe_rows >= minimum_rows
            and transcript_rows >= 1
        ),
        minimum_rows=minimum_rows,
        transcript_rows=transcript_rows,
    )


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
        state_word = _sanitize_terminal_text(
            state.run_status,
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
    output_stream: TextIO,
    prompt: object,
) -> None:
    safe = _setup_prompt_text(prompt, output_stream)
    if not _stream_is_interactive(output_stream):
        print(safe, end="", flush=True, file=output_stream)
        return
    capabilities = _terminal_capabilities(text_stream=output_stream)
    runtime = _load_terminal_runtime()
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
    runtime = _load_terminal_runtime()
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
        suspend_bridge.restore(previous)
        if approval_bridge is not None:
            approval_bridge.restore(approval_previous)
        _restore_terminal(enhanced_output)
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
            f"{PIPX_REPAIR_GUIDANCE}"
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


_FORCE_SELECTION_GUIDANCE = (
    "Hold your terminal's selection bypass modifier (often Shift) "
    "to use terminal-owned selection."
)


def _clipboard_mechanism(
    *,
    platform: str,
    environ: Mapping[str, str],
) -> str:
    """Choose one reviewed clipboard path without probing arbitrary commands."""

    if environ.get("SSH_TTY") or environ.get("SSH_CONNECTION"):
        return "osc52"
    return "pbcopy" if platform == "darwin" else "osc52"


def _osc52_sequence(payload: bytes, *, tmux: bool) -> str:
    """Encode text so selected control-shaped data cannot become terminal syntax."""

    if len(payload) > MAX_CLIPBOARD_UTF8_BYTES:
        raise ValueError("clipboard payload exceeds the 64 KiB UTF-8 limit")
    encoded = base64.b64encode(payload).decode("ascii")
    request = f"\x1b]52;c;{encoded}\x07"
    return f"\x1bPtmux;\x1b{request}\x1b\\" if tmux else request


def _copy_with_pbcopy(payload: bytes) -> ClipboardResult:
    """Invoke the exact local macOS clipboard utility with a fixed timeout."""

    try:
        import subprocess

        completed = subprocess.run(
            ("/usr/bin/pbcopy",),
            input=payload,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_CLIPBOARD_TIMEOUT_SECONDS,
        )
    except Exception:
        return ClipboardResult(
            "failure",
            "pbcopy",
            f"Copy failed. {_FORCE_SELECTION_GUIDANCE}",
        )
    if completed.returncode != 0:
        return ClipboardResult(
            "failure",
            "pbcopy",
            f"Copy failed. {_FORCE_SELECTION_GUIDANCE}",
        )
    return ClipboardResult("copied", "pbcopy", "Copied")


def _send_osc52_request(
    output: Any,
    payload: bytes,
    *,
    tmux: bool,
) -> ClipboardResult:
    """Send one bounded, unacknowledged terminal clipboard request."""

    try:
        output.write_raw(_osc52_sequence(payload, tmux=tmux))
        output.flush()
    except Exception:
        return ClipboardResult(
            "failure",
            "osc52-tmux" if tmux else "osc52",
            f"Copy failed. {_FORCE_SELECTION_GUIDANCE}",
        )
    return ClipboardResult(
        "requested",
        "osc52-tmux" if tmux else "osc52",
        "Copy request sent to terminal",
    )


async def _deliver_clipboard(
    text: str,
    *,
    output: Any,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ClipboardResult:
    """Deliver only one bounded visible selection through the chosen path."""

    if not isinstance(text, str) or not text:
        return ClipboardResult("failure", "none", "Copy failed: selection is empty.")
    payload = text.encode("utf-8")
    if len(payload) > MAX_CLIPBOARD_UTF8_BYTES:
        return ClipboardResult(
            "failure",
            "none",
            "Copy failed: selection exceeds the 64 KiB UTF-8 limit.",
        )
    environment = os.environ if environ is None else environ
    mechanism = _clipboard_mechanism(
        platform=sys.platform if platform is None else platform,
        environ=environment,
    )
    if mechanism == "pbcopy":
        try:
            return await asyncio.to_thread(_copy_with_pbcopy, payload)
        except Exception:
            return ClipboardResult(
                "failure",
                "pbcopy",
                f"Copy failed. {_FORCE_SELECTION_GUIDANCE}",
            )
    return _send_osc52_request(
        output,
        payload,
        tmux=bool(environment.get("TMUX")),
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
        if state.toggle_expanded_detail():
            state.notice = ""
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
        base = runtime["FormattedTextControl"].create_content(control, width, height)
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
            cursor_position=base.cursor_position,
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
        mouse_support=True,
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


def _completed_tool_pairs(
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


def _artifact_delivery_messages(
    pairs: tuple[tuple[ToolCall, ToolResultBlock | None], ...],
) -> tuple[str, ...]:
    messages: list[str] = []
    created_artifact_ids: list[str] = []
    delivery_attempts: set[str] = set()
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
            delivery_attempts.add(artifact_id)
        if not result.is_error:
            continue
        error = result.output.get("error")
        if not isinstance(artifact_id, str) or not isinstance(error, Mapping):
            continue
        code = _sanitize_terminal_text(
            error.get("code"),
            maximum=128,
            preserve_lines=False,
            fallback="artifact_delivery_failed",
        )
        detail = _sanitize_terminal_text(
            error.get("message"),
            maximum=512,
            preserve_lines=False,
            fallback="The artifact was not saved locally.",
        )
        safe_id = _sanitize_terminal_text(
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
        if artifact_id in delivery_attempts:
            continue
        safe_id = _sanitize_terminal_text(
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
    sources = _startup_sources_text(startup, separator=glyphs.separator)
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
            ("class:tui.metadata", sources),
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
                ("Sources", sources),
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
            source_label = "Source" if startup.source_count == 1 else "Sources"
            rows = (
                ("Status", f"{glyphs.ready} Ready"),
                ("Agent", agent),
                ("Model", model_text),
                ("Home", _truncate_middle_display_text(home, 120, marker=marker)),
                (source_label, sources),
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


def _startup_sources_text(
    startup: TerminalStartupInfo,
    *,
    separator: str,
) -> str:
    count = startup.source_count
    if count == 0:
        return f"0 cataloged{separator}none attached"
    source_types = ", ".join(
        _startup_safe_text(value, fallback="source") for value in startup.source_types
    )
    names = [
        _startup_safe_text(value, fallback="source")
        for value in startup.source_names[:3]
    ]
    if count > len(names):
        names.append(f"+{count - len(names)} more")
    if count == 1:
        primary = names[0] if names else "1 source"
        details = tuple(value for value in ("cataloged", source_types) if value)
        return primary + separator + separator.join(details)
    details = tuple(value for value in (source_types, ", ".join(names)) if value)
    suffix = separator + separator.join(details) if details else ""
    return f"{count} cataloged{suffix}"


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


def _wrap_display_text(
    value: str,
    width: int,
    *,
    maximum_lines: int,
    marker: str,
) -> tuple[str, ...]:
    if width <= 0 or maximum_lines <= 0:
        return ()
    remaining = _one_logical_line(value).strip()
    lines: list[str] = []
    while remaining and len(lines) < maximum_lines:
        if _display_width(remaining) <= width:
            lines.append(remaining)
            remaining = ""
            break
        prefix, _truncated = _truncate_display_text(remaining, width, marker="")
        split_at = prefix.rfind(" ")
        if split_at >= max(1, len(prefix) // 3):
            prefix = prefix[:split_at]
        consumed = len(prefix)
        lines.append(prefix.rstrip())
        remaining = remaining[consumed:].lstrip()
    if remaining and lines:
        combined = f"{lines[-1]} {remaining}".strip()
        lines[-1] = _truncate_display_text(
            combined,
            width,
            marker=marker,
        )[0]
    return tuple(lines or ("",))


def _pad_display_text(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))


def _truncate_middle_display_text(
    value: str,
    width: int,
    *,
    marker: str,
) -> str:
    if width <= 0:
        return ""
    if _display_width(value) <= width:
        return value
    marker_width = _display_width(marker)
    if marker_width >= width:
        return _truncate_display_text(marker, width, marker="")[0]
    left_width = (width - marker_width + 1) // 2
    right_width = width - marker_width - left_width
    left = _truncate_display_text(value, left_width, marker="")[0]
    reversed_right = _truncate_display_text(
        value[::-1],
        right_width,
        marker="",
    )[0]
    return left + marker + reversed_right[::-1]


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


def _display_clusters(value: str) -> tuple[tuple[str, int, int, int], ...]:
    """Return grapheme text, code-point bounds, and terminal-cell width."""

    clusters: list[tuple[str, int, int, int]] = []
    offset = 0
    while offset < len(value):
        end = _next_grapheme_end(value, offset)
        cluster = value[offset:end]
        clusters.append((cluster, offset, end, max(0, _cluster_cell_width(cluster))))
        offset = end
    return tuple(clusters)


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
    runtime = _load_terminal_runtime() if runtime is None else runtime
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
                ("Ctrl+O expand",),
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
            ("Ctrl+O collapse",),
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
            fragments.extend([(style, prefix), ("", safe), (style, "\n")])
    return fragments


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
        fragments.extend(line)
        fragments.append((border_style, "\n"))
    return fragments


def _render_rich_fragment_lines(
    runtime: dict[str, Any],
    renderable: Any,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
) -> list[list[tuple[str, str]]]:
    target = io.StringIO()
    capabilities = capabilities or _terminal_capabilities()
    console = runtime["Console"](
        file=target,
        width=max(_MIN_RENDER_WIDTH, min(width, _MAX_RENDER_WIDTH)),
        force_terminal=not capabilities.no_color,
        color_system=capabilities.rich_color_system,
        no_color=capabilities.no_color,
        markup=False,
        highlight=False,
        soft_wrap=False,
        theme=runtime["Theme"](_rich_theme_rules(capabilities)),
    )
    console.print(renderable, end="")
    formatted = runtime["ANSI"](target.getvalue()).__pt_formatted_text__()
    lines: list[list[tuple[str, str]]] = [[]]
    for style, text in formatted:
        parts = text.split("\n")
        for index, part in enumerate(parts):
            if part:
                lines[-1].append((style, part))
            if index < len(parts) - 1:
                lines.append([])
    while len(lines) > 1 and not lines[-1]:
        lines.pop()
    return lines


def _render_user_message_fragments(
    runtime: dict[str, Any],
    value: object,
    *,
    width: int,
    capabilities: TerminalCapabilities,
) -> list[tuple[str, str]]:
    safe = _sanitize_terminal_text(
        value,
        maximum=None,
        preserve_lines=True,
        fallback="(empty message)",
    )
    lines = _render_rich_fragment_lines(
        runtime,
        runtime["Text"](safe),
        width=max(1, width - 1),
        capabilities=capabilities,
    )
    fragments: list[tuple[str, str]] = []
    for line in lines:
        fragments.append(("", " "))
        fragments.extend(line)
        fragments.append(("", "\n"))
    return fragments


def _render_markdown_fragments(
    runtime: dict[str, Any],
    value: object,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
) -> list[tuple[str, str]]:
    rendered = _render_markdown_ansi(
        runtime,
        value,
        width=width,
        capabilities=capabilities,
    )
    fragments = runtime["ANSI"](rendered).__pt_formatted_text__()
    return [
        (style, f" {text}" if index == 0 else text)
        for index, (style, text) in enumerate(fragments)
    ]


def _render_markdown_ansi(
    runtime: dict[str, Any],
    value: object,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
) -> str:
    safe = _sanitize_terminal_text(
        value,
        maximum=None,
        preserve_lines=True,
        fallback="(empty response)",
    )
    target = io.StringIO()
    capabilities = capabilities or _terminal_capabilities()
    theme = runtime["Theme"](_rich_theme_rules(capabilities))
    console = runtime["Console"](
        file=target,
        width=max(_MIN_RENDER_WIDTH, min(width, _MAX_RENDER_WIDTH)),
        force_terminal=not capabilities.no_color,
        color_system=capabilities.rich_color_system,
        no_color=capabilities.no_color,
        markup=False,
        highlight=False,
        soft_wrap=False,
        theme=theme,
    )
    console.print(
        runtime["Markdown"](
            safe,
            code_theme="bw",
            hyperlinks=False,
        ),
        end="",
    )
    return target.getvalue()


def _render_markdown_text(value: object, *, width: int = 80) -> str:
    """Render sanitized Markdown without terminal control sequences for tests."""

    runtime = _load_terminal_runtime()
    target = io.StringIO()
    safe = _sanitize_terminal_text(
        value,
        maximum=None,
        preserve_lines=True,
        fallback="(empty response)",
    )
    console = runtime["Console"](
        file=target,
        width=max(_MIN_RENDER_WIDTH, min(width, _MAX_RENDER_WIDTH)),
        force_terminal=False,
        color_system=None,
        no_color=True,
        markup=False,
        highlight=False,
        soft_wrap=False,
    )
    console.print(
        runtime["Markdown"](safe, code_theme="bw", hyperlinks=False),
        end="",
    )
    return target.getvalue()


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


def _semantic_style_rules(
    capabilities: TerminalCapabilities | None = None,
) -> dict[str, str]:
    capabilities = capabilities or _terminal_capabilities()
    if capabilities.no_color:
        return {
            "tui.identity": "bold",
            "tui.header.agent": "bold",
            "tui.header.meta": "",
            "tui.rule": "",
            "tui.user.label": "bold",
            "tui.assistant.label": "bold",
            "tui.local.label": "bold",
            "tui.local": "",
            "tui.local.status.label": "bold",
            "tui.local.status": "",
            "tui.local.sources.label": "bold",
            "tui.local.sources": "",
            "tui.local.catalog.label": "bold",
            "tui.local.catalog": "",
            "tui.local.settings.label": "bold",
            "tui.local.settings": "",
            "tui.metadata": "",
            "tui.empty": "",
            "tui.prompt": "bold",
            "tui.composer": "",
            "tui.composer.frame": "",
            "tui.transcript.selection": "reverse",
            "tui.frame": "",
            "tui.resize": "bold",
            "tui.new-output": "bold underline",
            "tui.command-menu": "",
            "tui.command-menu.rule": "",
            "tui.command-menu.marker": "",
            "tui.command-menu.marker.current": "bold",
            "tui.command-menu.command": "",
            "tui.command-menu.command.current": "bold underline",
            "tui.command-menu.description": "",
            "tui.command-menu.description.current": "bold",
            "tui.approval": "",
            "tui.approval.frame": "",
            "tui.approval.label": "bold",
            "tui.approval.identity": "",
            "tui.approval.arguments": "",
            "tui.approval.action": "bold",
            "tui.approval.failure": "bold",
            "frame.border": "",
            "tui.status": "",
            "tui.status.ready": "bold",
            "tui.status.running": "bold",
            "tui.status.approval": "bold",
            "tui.status.failure": "bold",
            "tui.status.notice": "",
            "tui.status.meta": "",
            "tui.tool.running": "",
            "tui.tool.approval": "bold",
            "tui.tool.success": "",
            "tui.tool.failure": "bold",
            "selection.identity": "bold",
            "selection.title": "bold",
            "selection.help": "",
            "selection.filter": "bold",
            "selection.validation": "bold",
            "selection.empty": "",
            "selection.current": "bold underline",
        }
    colors = _semantic_colors(capabilities)
    return {
        "tui.identity": f"bold {colors['brand']}",
        "tui.header.agent": "bold",
        "tui.header.meta": colors["muted"],
        "tui.rule": colors["muted_green"],
        "tui.user.label": "bold",
        "tui.assistant.label": f"bold {colors['brand']}",
        "tui.local.label": f"bold {colors['muted']}",
        "tui.local": "",
        "tui.local.status.label": f"bold {colors['brand']}",
        "tui.local.status": "",
        "tui.local.sources.label": f"bold {colors['data']}",
        "tui.local.sources": "",
        "tui.local.catalog.label": f"bold {colors['data']}",
        "tui.local.catalog": "",
        "tui.local.settings.label": f"bold {colors['brand']}",
        "tui.local.settings": "",
        "tui.metadata": colors["muted"],
        "tui.empty": colors["muted"],
        "tui.prompt": f"bold {colors['focus']}",
        "tui.composer": "",
        "tui.composer.frame": "",
        "tui.transcript.selection": "reverse",
        "tui.frame": colors["focus"],
        "tui.resize": f"bold {colors['warning']}",
        "tui.new-output": f"bold underline {colors['focus']}",
        "tui.command-menu": "",
        "tui.command-menu.rule": colors["muted"],
        "tui.command-menu.marker": colors["muted"],
        "tui.command-menu.marker.current": f"bold {colors['focus']}",
        "tui.command-menu.command": colors["muted"],
        "tui.command-menu.command.current": f"bold {colors['focus']}",
        "tui.command-menu.description": colors["muted"],
        "tui.command-menu.description.current": colors["focus"],
        "tui.approval": "",
        "tui.approval.frame": colors["warning"],
        "tui.approval.label": f"bold {colors['warning']}",
        "tui.approval.identity": "bold",
        "tui.approval.arguments": "",
        "tui.approval.action": f"bold {colors['warning']}",
        "tui.approval.failure": f"bold {colors['error']}",
        "frame.border": colors["focus"],
        "tui.status": colors["muted"],
        "tui.status.ready": f"bold {colors['brand']}",
        "tui.status.running": f"bold {colors['muted_green']}",
        "tui.status.approval": f"bold {colors['warning']}",
        "tui.status.failure": f"bold {colors['error']}",
        "tui.status.notice": colors["warning"],
        "tui.status.meta": colors["muted"],
        "tui.tool.running": colors["muted_green"],
        "tui.tool.approval": colors["warning"],
        "tui.tool.success": colors["brand"],
        "tui.tool.failure": colors["error"],
        "selection.identity": f"bold {colors['brand']}",
        "selection.title": "bold",
        "selection.help": colors["muted"],
        "selection.filter": f"bold {colors['data']}",
        "selection.validation": f"bold {colors['warning']}",
        "selection.empty": colors["muted"],
        "selection.current": f"bold underline {colors['focus']}",
    }


def _semantic_colors(capabilities: TerminalCapabilities) -> dict[str, str]:
    if capabilities.color_depth == "truecolor":
        return {
            "brand": "#22c55e",
            "focus": "#4ade80",
            "muted_green": "#15803d",
            "data": "#38bdf8",
            "warning": "#f59e0b",
            "error": "#f87171",
            "muted": "#71717a",
        }
    if capabilities.color_depth == "256":
        return {
            "brand": "ansibrightgreen",
            "focus": "ansibrightgreen",
            "muted_green": "ansigreen",
            "data": "ansibrightcyan",
            "warning": "ansiyellow",
            "error": "ansibrightred",
            "muted": "ansibrightblack",
        }
    return {
        "brand": "ansigreen",
        "focus": "ansibrightgreen",
        "muted_green": "ansigreen",
        "data": "ansicyan",
        "warning": "ansiyellow",
        "error": "ansired",
        "muted": "ansibrightblack",
    }


def _rich_theme_rules(
    capabilities: TerminalCapabilities,
) -> dict[str, str]:
    if capabilities.no_color:
        return {
            "brand": "bold",
            "data": "",
            "warning": "bold",
            "error": "bold",
            "muted": "",
            "markdown.h1": "bold",
            "markdown.h2": "bold",
            "markdown.h3": "bold",
            "markdown.item.bullet": "bold",
            "markdown.code": "",
            "markdown.code_block": "",
        }
    if capabilities.color_depth == "truecolor":
        brand = "#22c55e"
        data = "#38bdf8"
        warning = "#f59e0b"
        error = "#f87171"
        muted = "#71717a"
    elif capabilities.color_depth == "256":
        brand = "bright_green"
        data = "bright_cyan"
        warning = "yellow"
        error = "bright_red"
        muted = "bright_black"
    else:
        brand = "green"
        data = "cyan"
        warning = "yellow"
        error = "red"
        muted = "bright_black"
    return {
        "brand": f"bold {brand}",
        "data": data,
        "warning": warning,
        "error": error,
        "muted": muted,
        "markdown.h1": f"bold {brand}",
        "markdown.h2": f"bold {brand}",
        "markdown.h3": f"bold {brand}",
        "markdown.item.bullet": brand,
        "markdown.code": data,
        "markdown.code_block": data,
    }


def _event_counter(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return min(value, _MAX_EVENT_COUNTER)


def _model_text_event_fields(event: AgentEvent) -> tuple[int, str] | None:
    if event.kind is not AgentEventKind.MODEL_TEXT_DELTA:
        return None
    model_call_index = _event_counter(event.data.get("model_call_index"))
    text = event.data.get("text")
    if (
        model_call_index is None
        or model_call_index < 1
        or not isinstance(text, str)
        or not text
    ):
        return None
    return model_call_index, text


def _event_text(value: object, *, maximum: int, fallback: str) -> str:
    return _sanitize_terminal_text(
        value,
        maximum=maximum,
        preserve_lines=False,
        fallback=fallback,
    )


def _optional_event_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return _event_text(value, maximum=maximum, fallback="") or None


def _format_duration(duration_ms: int) -> str:
    if duration_ms < 1_000:
        return f"{duration_ms}ms"
    seconds = duration_ms / 1_000
    if seconds < 100:
        return f"{seconds:.1f}s"
    return f"{int(seconds)}s"


def _one_logical_line(value: str) -> str:
    return " ".join(value.split())


def _display_width(value: str) -> int:
    width = 0
    for character in value:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def _truncate_display_text(
    value: str,
    maximum: int,
    *,
    marker: str = "…",
) -> tuple[str, bool]:
    if maximum < 1:
        return "", bool(value)
    if _display_width(value) <= maximum:
        return value, False
    safe_marker = marker
    if _display_width(safe_marker) > maximum:
        safe_marker = safe_marker[:maximum]
    available = max(0, maximum - _display_width(safe_marker))
    projected: list[str] = []
    width = 0
    for character in value:
        character_width = (
            0
            if unicodedata.combining(character)
            else (2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1)
        )
        if width + character_width > available:
            break
        projected.append(character)
        width += character_width
    return "".join(projected) + safe_marker, True


def _clear_current_task_cancellation() -> None:
    current = asyncio.current_task()
    if current is None:
        return
    while current.cancelling():
        current.uncancel()


def _sanitize_terminal_text(
    value: object,
    *,
    maximum: int | None,
    preserve_lines: bool,
    fallback: str,
) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = value.replace("\r\n", "\n")
    projected: list[str] = []
    for character in normalized:
        if character == "\n" and preserve_lines:
            projected.append(character)
            continue
        if character == "\t" and preserve_lines:
            projected.append(character)
            continue
        category = unicodedata.category(character)
        if (
            character.isprintable()
            and category not in {"Cc", "Cf", "Cs"}
            and character != "\r"
        ):
            projected.append(character)
        else:
            projected.append("?")
    rendered = "".join(projected)
    if maximum is not None and len(rendered) > maximum:
        rendered = rendered[: max(0, maximum - 3)] + "..."
    return rendered or fallback


def _render_width(output: Any) -> int:
    columns, _rows = _terminal_size(output)
    return max(_MIN_RENDER_WIDTH, min(columns - 2, _MAX_RENDER_WIDTH))


def _viewport_height(window: Any) -> int:
    render_info = getattr(window, "render_info", None)
    height = getattr(render_info, "window_height", 0)
    return max(1, height or 8)


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
