"""Compatibility facade for Daita's single full-screen terminal UI."""

from __future__ import annotations

# This module intentionally re-exports the legacy terminal surface while the
# implementation lives under ``daita.tui``. Most compatibility names are
# accessed by importers rather than by this module itself.
# pyright: reportUnusedImport=false
from .terminal_transcript import (
    TranscriptFollowState,
    bounded_selection_auto_scroll,
)
from .tui.application import (
    _MAX_COMPOSER_ROWS,
    _MOUSE_SCROLL_LINES,
    _STREAM_REPAINT_INTERVAL_SECONDS,
    MAX_COMPOSER_CHARACTERS,
    TerminalApplicationResult,
    TerminalCommandResult,
    TerminalSuspendBridge,
    TerminalTUIUnavailable,
    TerminalUserInputError,
    _create_application,
    _load_terminal_runtime,
    _project_pending_events,
    _render_markdown_text,
    _render_tool_card_fragments,
    _restore_application,
    _run_application,
    _write_setup_prompt,
    _write_setup_status,
    run_terminal_tui,
    supports_terminal_tui,
)
from .tui.capabilities import (
    ResponsiveProjection,
    TerminalCapabilities,
    TerminalGlyphs,
    responsive_projection as _responsive_projection,
    terminal_capabilities as _terminal_capabilities,
    terminal_glyphs as _terminal_glyphs,
    terminal_size as _terminal_size,
    terminal_size_polling_interval as _terminal_size_polling_interval,
)
from .tui.clipboard import (
    MAX_CLIPBOARD_UTF8_BYTES,
    ClipboardResult,
    clipboard_mechanism as _clipboard_mechanism,
    deliver_clipboard as _deliver_clipboard,
    osc52_sequence as _osc52_sequence,
    send_osc52_request as _send_osc52_request,
)
from .tui.rendering import (
    render_user_message_fragments as _render_user_message_fragments,
    semantic_style_rules as _semantic_style_rules,
)
from .tui.shell import (
    _SLASH_COMMAND_COMPLETIONS,
    MAX_APPROVAL_DOCUMENT_CHARACTERS,
    StatusProjection,
    _approval_panel_for_request,
    _context_progress_text,
    _render_approval_panel_fragments,
    _resize_message_fragments,
    _setup_prompt_text,
    _slash_command_completion_surface,
    _slash_command_menu_fragments,
    _slash_completion_maps,
    _status_projection,
    _status_right_fragments,
    _stream_is_interactive,
    _text_stream_width,
)
from .tui.state import (
    ApprovalPanelState,
    TerminalApprovalBridge,
    TerminalBlock,
    TerminalObserverBridge,
    TerminalStartupInfo,
    TerminalViewState,
    _artifact_delivery_messages,
    _completed_tool_pairs,
)
from .tui.text import (
    display_width as _display_width,
    sanitize_terminal_text as _sanitize_terminal_text,
)
from .tui.tool_view import (
    ToolCardDetails,
    ToolCardState,
    ToolTablePreview,
    _card_rich_lines,
    _project_tool_details,
)
from .tui.transcript_view import (
    _highlight_transcript_line,
    _render_startup_text,
    _render_transcript_fragments,
)

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
