"""Untrusted-text sanitization and display bounds for the terminal UI."""

from __future__ import annotations

import unicodedata

MAX_RENDER_CHARACTERS = 16_384
MAX_DISPLAY_CHARACTERS = 16_384
MAX_LABEL_CHARACTERS = 128


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


def safe_display(
    value: object,
    *,
    fallback: str = "source",
    maximum: int = MAX_LABEL_CHARACTERS,
) -> str:
    return sanitize_terminal_text(
        value,
        maximum=maximum,
        preserve_lines=False,
        fallback=fallback,
    )


def render_model_answer(
    value: object,
    *,
    fallback: str = "(empty response)",
    maximum: int | None = MAX_DISPLAY_CHARACTERS,
) -> str:
    return sanitize_terminal_text(
        value,
        maximum=maximum,
        preserve_lines=True,
        fallback=fallback,
    )


def sanitize_markdown(
    value: object, *, maximum: int | None = MAX_RENDER_CHARACTERS
) -> str:
    """Bound model-authored Markdown and neutralize terminal controls."""

    return sanitize_terminal_text(
        value,
        maximum=maximum,
        preserve_lines=True,
        fallback="",
    )
