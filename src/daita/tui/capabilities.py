"""Terminal capability detection and responsive layout projection."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TextIO

COLLAPSED_TABLE_COLUMNS = 12
EXPANDED_TABLE_COLUMNS = 20
MIN_RENDER_WIDTH = 20
MAX_RENDER_WIDTH = 240
MIN_USABLE_COLUMNS = 32
MIN_READY_ROWS = 8
MIN_APPROVAL_ROWS = 15
RUNNING_GLYPHS = ("◐", "◓", "◑", "◒")
ASCII_RUNNING_GLYPHS = ("~", "-", "~", "+")


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


def terminal_capabilities(
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
    if not encoding and not locale_name:
        unicode_supported = False
    if (
        locale_name.casefold() in {"c", "posix"}
        and environment.get("PYTHONUTF8", "").strip() != "1"
    ):
        unicode_supported = False
    return TerminalCapabilities(
        color_depth=color_depth,
        unicode=unicode_supported,
    )


def _detected_color_depth(output: Any, environment: Mapping[str, str]) -> str:
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
    return "none"


def terminal_glyphs(capabilities: TerminalCapabilities) -> TerminalGlyphs:
    if capabilities.unicode:
        return TerminalGlyphs(
            top_left="╭",
            top_right="╮",
            bottom_left="╰",
            bottom_right="╯",
            horizontal="─",
            vertical="│",
            prompt="›",
            running=RUNNING_GLYPHS,
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
        running=ASCII_RUNNING_GLYPHS,
        ready="OK",
        success="OK",
        failure="!",
        warning="!",
        approval="!",
        separator=" | ",
    )


def terminal_size(output: Any) -> tuple[int, int]:
    try:
        size = output.get_size()
        columns = int(size.columns)
        rows = int(size.rows)
    except (AttributeError, OSError, TypeError, ValueError):
        # Keep enough width for a visible resize notice, but never invent the
        # rows needed to admit input before the terminal reports a real size.
        return MIN_USABLE_COLUMNS, 1
    return max(1, columns), max(1, rows)


def terminal_size_polling_interval(
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


def mouse_reporting_available(output: Any) -> bool:
    """Admit pointer handling only when output exposes paired mode controls."""

    return all(
        callable(getattr(output, method_name, None))
        for method_name in ("enable_mouse_support", "disable_mouse_support")
    )


def responsive_projection(
    columns: int,
    rows: int,
    *,
    approving: bool = False,
) -> ResponsiveProjection:
    safe_columns = max(1, int(columns))
    safe_rows = max(1, int(rows))
    if safe_columns >= 100:
        mode = "full"
        collapsed_columns = COLLAPSED_TABLE_COLUMNS
        expanded_columns = EXPANDED_TABLE_COLUMNS
    elif safe_columns >= 70:
        mode = "compact"
        collapsed_columns = 8
        expanded_columns = 12
    else:
        mode = "narrow"
        collapsed_columns = 4
        expanded_columns = 6
    minimum_rows = MIN_APPROVAL_ROWS if approving else MIN_READY_ROWS
    transcript_rows = max(0, safe_rows - minimum_rows + 1)
    return ResponsiveProjection(
        columns=safe_columns,
        rows=safe_rows,
        mode=mode,
        content_width=max(
            MIN_RENDER_WIDTH,
            min(MAX_RENDER_WIDTH, safe_columns - 2),
        ),
        collapsed_preview_columns=collapsed_columns,
        expanded_preview_columns=expanded_columns,
        bordered_cards=mode != "narrow",
        stacked_metadata=mode == "narrow",
        two_sided_status=mode == "full",
        usable=(
            safe_columns >= MIN_USABLE_COLUMNS
            and safe_rows >= minimum_rows
            and transcript_rows >= 1
        ),
        minimum_rows=minimum_rows,
        transcript_rows=transcript_rows,
    )
