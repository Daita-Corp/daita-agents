from __future__ import annotations

import subprocess
import sys

import pytest

from daita import terminal_tui


def _startup_state(
    *,
    model: str = "gpt-5.6-sol",
    home: str = "/Users/demo/.daita/agents/atlas",
    source_count: int = 2,
    source_types: tuple[str, ...] = ("PostgreSQL", "SQLite"),
    source_names: tuple[str, ...] = ("Warehouse", "Reporting"),
    resource_count: int = 24,
    relationship_count: int = 3,
    read_capabilities: tuple[str, ...] = (
        "Catalog search & inspection",
        "PostgreSQL queries",
        "SQLite queries",
    ),
    warnings: tuple[str, ...] = (),
) -> terminal_tui.TerminalViewState:
    return terminal_tui.TerminalViewState(
        agent_label="atlas",
        model_label=model,
        source_summary="2 sources",
        startup=terminal_tui.TerminalStartupInfo(
            version="2.0.0a0",
            provider_label="OpenAI",
            model_status="configured",
            agent_home=home,
            source_count=source_count,
            source_types=source_types,
            source_names=source_names,
            resource_count=resource_count,
            relationship_count=relationship_count,
            read_capabilities=read_capabilities,
            warnings=warnings,
        ),
    )


@pytest.mark.parametrize("width", (120, 80, 70, 59, 42))
def test_startup_layouts_are_responsive_and_never_exceed_width(width: int):
    state = _startup_state()
    capabilities = terminal_tui.TerminalCapabilities("truecolor", True)

    rendered = terminal_tui._render_startup_text(
        state,
        width=width,
        capabilities=capabilities,
    )

    assert "DAITA" in rendered
    assert "Ready" in rendered
    assert "gpt-5.6-sol" in rendered
    assert "/help" in rendered
    assert (
        max(terminal_tui._display_width(line) for line in rendered.splitlines())
        <= width
    )
    if width >= 80:
        assert "████▄" in rendered
    else:
        assert "████▄" not in rendered
    if width < 60:
        assert "╭" not in rendered
        assert "Read-only:" in rendered
    else:
        assert "╭" in rendered
        assert "Catalog" in rendered


def test_startup_ascii_and_no_color_modes_preserve_semantic_information():
    state = _startup_state()
    ascii_rendered = terminal_tui._render_startup_text(
        state,
        width=80,
        capabilities=terminal_tui.TerminalCapabilities("none", False),
    )
    no_color_rendered = terminal_tui._render_startup_text(
        state,
        width=80,
        capabilities=terminal_tui.TerminalCapabilities("none", True),
    )

    assert "DAITA  2.0.0a0" in ascii_rendered
    assert "+-" in ascii_rendered
    assert "| Status" in ascii_rendered
    assert "OK Ready" in ascii_rendered
    assert all(symbol not in ascii_rendered for symbol in "╭╮╰╯─│✓●›█")
    assert "████▄" in no_color_rendered
    assert "● Ready" in no_color_rendered
    assert "\x1b" not in ascii_rendered + no_color_rendered


def test_startup_long_values_are_bounded_and_credential_shapes_are_redacted():
    state = _startup_state(
        model="model-" + "x" * 180,
        home="/Users/demo/" + "/very-long-directory" * 20 + "/atlas",
        source_count=1,
        source_types=("PostgreSQL",),
        source_names=("warehouse password=hunter2 " + "x" * 180,),
        warnings=("API_KEY=super-secret must be replaced before launch",),
    )

    rendered = terminal_tui._render_startup_text(
        state,
        width=80,
        capabilities=terminal_tui.TerminalCapabilities("truecolor", True),
    )

    assert "Warning:" in rendered
    assert "[redacted]" in rendered
    assert "hunter2" not in rendered
    assert "super-secret" not in rendered
    assert "…" in rendered
    assert (
        max(terminal_tui._display_width(line) for line in rendered.splitlines()) <= 80
    )


def test_startup_empty_catalog_has_an_actionable_bounded_warning():
    state = _startup_state(
        source_count=0,
        source_types=(),
        source_names=(),
        resource_count=0,
        relationship_count=0,
        read_capabilities=(),
        warnings=("No data sources. Use /source add to attach one.",),
    )

    rendered = terminal_tui._render_startup_text(
        state,
        width=60,
        capabilities=terminal_tui.TerminalCapabilities("none", False),
    )

    assert "0 cataloged | none attached" in rendered
    assert "0 resources | 0 relationships" in rendered
    assert "None until a source is added" in rendered
    assert "/source add" in rendered
    assert (
        max(terminal_tui._display_width(line) for line in rendered.splitlines()) <= 60
    )


def test_startup_rendering_keeps_terminal_and_integration_packages_lazy():
    script = """
import builtins

blocked = {
    "anthropic",
    "asyncpg",
    "google",
    "keyring",
    "openai",
    "prompt_toolkit",
    "rich",
    "sqlglot",
}
original = builtins.__import__

def guarded(name, *args, **kwargs):
    level = kwargs.get("level", args[3] if len(args) >= 4 else 0)
    if level == 0 and name.split(".")[0] in blocked:
        raise AssertionError(f"eager integration import: {name}")
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
from daita import terminal_tui

state = terminal_tui.TerminalViewState(
    "atlas",
    "model",
    "source",
    startup=terminal_tui.TerminalStartupInfo(
        "2.0.0a0",
        "provider",
        "configured",
        "/tmp/atlas",
        0,
        (),
        (),
        0,
        0,
        (),
    ),
)
rendered = terminal_tui._render_startup_text(
    state,
    width=60,
    capabilities=terminal_tui.TerminalCapabilities("none", False),
)
assert "DAITA" in rendered
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
