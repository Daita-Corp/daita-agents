"""Safe terminal text projection shared by the TUI presentation owners."""

from __future__ import annotations

import unicodedata

from ..terminal_transcript import _cluster_cell_width, _next_grapheme_end

MAX_RENDER_CHARACTERS = 16_384


def one_logical_line(value: str) -> str:
    return " ".join(value.split())


def display_width(value: str) -> int:
    width = 0
    for character in value:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def truncate_display_text(
    value: str,
    maximum: int,
    *,
    marker: str = "…",
) -> tuple[str, bool]:
    if maximum < 1:
        return "", bool(value)
    if display_width(value) <= maximum:
        return value, False
    safe_marker = marker
    if display_width(safe_marker) > maximum:
        safe_marker = safe_marker[:maximum]
    available = max(0, maximum - display_width(safe_marker))
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


def wrap_display_text(
    value: str,
    width: int,
    *,
    maximum_lines: int,
    marker: str,
) -> tuple[str, ...]:
    if width <= 0 or maximum_lines <= 0:
        return ()
    remaining = one_logical_line(value).strip()
    lines: list[str] = []
    while remaining and len(lines) < maximum_lines:
        if display_width(remaining) <= width:
            lines.append(remaining)
            remaining = ""
            break
        prefix, _truncated = truncate_display_text(remaining, width, marker="")
        split_at = prefix.rfind(" ")
        if split_at >= max(1, len(prefix) // 3):
            prefix = prefix[:split_at]
        consumed = len(prefix)
        lines.append(prefix.rstrip())
        remaining = remaining[consumed:].lstrip()
    if remaining and lines:
        combined = f"{lines[-1]} {remaining}".strip()
        lines[-1] = truncate_display_text(combined, width, marker=marker)[0]
    return tuple(lines or ("",))


def pad_display_text(value: str, width: int) -> str:
    return value + " " * max(0, width - display_width(value))


def truncate_middle_display_text(
    value: str,
    width: int,
    *,
    marker: str,
) -> str:
    if width <= 0:
        return ""
    if display_width(value) <= width:
        return value
    marker_width = display_width(marker)
    if marker_width >= width:
        return truncate_display_text(marker, width, marker="")[0]
    left_width = (width - marker_width + 1) // 2
    right_width = width - marker_width - left_width
    left = truncate_display_text(value, left_width, marker="")[0]
    reversed_right = truncate_display_text(value[::-1], right_width, marker="")[0]
    return left + marker + reversed_right[::-1]


def display_clusters(value: str) -> tuple[tuple[str, int, int, int], ...]:
    """Return grapheme text, code-point bounds, and terminal-cell width."""

    clusters: list[tuple[str, int, int, int]] = []
    offset = 0
    while offset < len(value):
        end = _next_grapheme_end(value, offset)
        cluster = value[offset:end]
        clusters.append((cluster, offset, end, max(0, _cluster_cell_width(cluster))))
        offset = end
    return tuple(clusters)


def sanitize_terminal_text(
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
