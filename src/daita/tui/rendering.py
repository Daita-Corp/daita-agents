"""Shared Rich projection and semantic terminal styles for the TUI."""

from __future__ import annotations

import io
from typing import Any

from .capabilities import (
    MAX_RENDER_WIDTH,
    MIN_RENDER_WIDTH,
    TerminalCapabilities,
    terminal_capabilities,
)
from .text import sanitize_terminal_text


def render_rich_fragment_lines(
    runtime: dict[str, Any],
    renderable: Any,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
) -> list[list[tuple[str, str]]]:
    target = io.StringIO()
    capabilities = capabilities or terminal_capabilities()
    console = runtime["Console"](
        file=target,
        width=max(MIN_RENDER_WIDTH, min(width, MAX_RENDER_WIDTH)),
        force_terminal=not capabilities.no_color,
        color_system=capabilities.rich_color_system,
        no_color=capabilities.no_color,
        markup=False,
        highlight=False,
        soft_wrap=False,
        theme=runtime["Theme"](rich_theme_rules(capabilities)),
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


def render_user_message_fragments(
    runtime: dict[str, Any],
    value: object,
    *,
    width: int,
    capabilities: TerminalCapabilities,
) -> list[tuple[str, str]]:
    safe = sanitize_terminal_text(
        value,
        maximum=None,
        preserve_lines=True,
        fallback="(empty message)",
    )
    lines = render_rich_fragment_lines(
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


def render_markdown_fragments(
    runtime: dict[str, Any],
    value: object,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
) -> list[tuple[str, str]]:
    rendered = render_markdown_ansi(
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


def render_markdown_ansi(
    runtime: dict[str, Any],
    value: object,
    *,
    width: int,
    capabilities: TerminalCapabilities | None = None,
) -> str:
    safe = sanitize_terminal_text(
        value,
        maximum=None,
        preserve_lines=True,
        fallback="(empty response)",
    )
    target = io.StringIO()
    capabilities = capabilities or terminal_capabilities()
    theme = runtime["Theme"](rich_theme_rules(capabilities))
    console = runtime["Console"](
        file=target,
        width=max(MIN_RENDER_WIDTH, min(width, MAX_RENDER_WIDTH)),
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


def render_markdown_text(
    runtime: dict[str, Any],
    value: object,
    *,
    width: int = 80,
) -> str:
    """Render sanitized Markdown without terminal control sequences for tests."""

    target = io.StringIO()
    safe = sanitize_terminal_text(
        value,
        maximum=None,
        preserve_lines=True,
        fallback="(empty response)",
    )
    console = runtime["Console"](
        file=target,
        width=max(MIN_RENDER_WIDTH, min(width, MAX_RENDER_WIDTH)),
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


def semantic_style_rules(
    capabilities: TerminalCapabilities | None = None,
) -> dict[str, str]:
    capabilities = capabilities or terminal_capabilities()
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
            "tui.tool.text": "",
            "selection.identity": "bold",
            "selection.title": "bold",
            "selection.help": "",
            "selection.filter": "bold",
            "selection.validation": "bold",
            "selection.empty": "",
            "selection.current": "bold underline",
        }
    colors = semantic_colors(capabilities)
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
        "tui.tool.text": "",
        "selection.identity": f"bold {colors['brand']}",
        "selection.title": "bold",
        "selection.help": colors["muted"],
        "selection.filter": f"bold {colors['data']}",
        "selection.validation": f"bold {colors['warning']}",
        "selection.empty": colors["muted"],
        "selection.current": f"bold underline {colors['focus']}",
    }


def semantic_colors(capabilities: TerminalCapabilities) -> dict[str, str]:
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


def rich_theme_rules(capabilities: TerminalCapabilities) -> dict[str, str]:
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
