from __future__ import annotations

import asyncio
import builtins
import io
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from daita import Agent, terminal, terminal_selection, terminal_tui
from daita.terminal_selection import (
    SelectionCancelled,
    SelectionOption,
    select_many,
    select_one,
)


class _RecordingOutput(DummyOutput):
    def __init__(self) -> None:
        self.fragments: list[str] = []
        self.hide_count = 0
        self.show_count = 0
        self.size = Size(rows=24, columns=80)
        self.size_checks = 0

    def write(self, data: str) -> None:
        self.fragments.append(data)

    def write_raw(self, data: str) -> None:
        self.fragments.append(data)

    def hide_cursor(self) -> None:
        self.hide_count += 1

    def show_cursor(self) -> None:
        self.show_count += 1

    def get_size(self) -> Size:
        self.size_checks += 1
        return self.size

    @property
    def text(self) -> str:
        return "".join(self.fragments)


_OPTIONS = (
    SelectionOption("alpha-id", "Alpha", "First"),
    SelectionOption("beta-id", "Beta", "Second"),
    SelectionOption("gamma-id", "Gamma", "Third"),
)


async def _wait_for_output(output: _RecordingOutput, text: str) -> None:
    async with asyncio.timeout(1):
        while text not in output.text:
            await asyncio.sleep(0)


async def _enhanced_choice(
    keys: str,
    *,
    options: tuple[SelectionOption[str], ...] = _OPTIONS,
    output: _RecordingOutput | None = None,
) -> tuple[str, _RecordingOutput]:
    recorded = _RecordingOutput() if output is None else output
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        selected = await select_one(
            "Choose",
            options,
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            enhanced_input=pipe,
            enhanced_output=recorded,
        )
    return selected, recorded


async def _enhanced_choices(
    keys: str,
    *,
    options: tuple[SelectionOption[str], ...] = _OPTIONS,
    output: _RecordingOutput | None = None,
    maximum: int = 32,
    empty_message: str | None = None,
    maximum_message: str | None = None,
) -> tuple[tuple[str, ...], _RecordingOutput]:
    recorded = _RecordingOutput() if output is None else output
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        selected = await select_many(
            "Choose many",
            options,
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            enhanced_input=pipe,
            enhanced_output=recorded,
            maximum=maximum,
            empty_message=empty_message,
            maximum_message=maximum_message,
        )
    return selected, recorded


async def test_initial_highlight_and_enter_return_first_stable_value():
    selected, output = await _enhanced_choice("\r")

    assert selected == "alpha-id"
    assert "› Alpha" in output.text
    assert output.hide_count >= 1
    assert output.show_count >= 1


@pytest.mark.parametrize(
    ("keys", "expected"),
    (
        ("\x1b[B\r", "beta-id"),
        ("\x1b[A\r", "gamma-id"),
        ("\x1b[B\x1b[B\x1b[B\r", "alpha-id"),
    ),
)
async def test_up_down_and_wraparound(keys: str, expected: str):
    selected, _ = await _enhanced_choice(keys)

    assert selected == expected


async def test_filtering_backspace_and_clearing_restore_declared_order():
    selected, _ = await _enhanced_choice("beta\x7f\x7f\x7f\x7f\r")

    assert selected == "alpha-id"


async def test_filtering_returns_stable_identity_not_label_or_position():
    selected, _ = await _enhanced_choice("GA\r")

    assert selected == "gamma-id"


async def test_no_match_cannot_be_confirmed():
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            select_one(
                "Choose",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        pipe.send_text("zz\r")
        await _wait_for_output(output, "No matches")
        assert not task.done()
        assert "No matches" in output.text
        pipe.send_text("\x7f\x7f\r")
        selected = await task

    assert selected == "alpha-id"


@pytest.mark.parametrize(
    ("keys", "error"),
    (
        ("\x1b", SelectionCancelled),
        ("\x03", KeyboardInterrupt),
        ("\x04", EOFError),
    ),
)
async def test_escape_ctrl_c_and_eof_have_distinct_cancellation_outcomes(
    keys: str,
    error: type[BaseException],
):
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        with pytest.raises(error):
            await select_one(
                "Choose",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )

    assert output.show_count >= 1


async def test_terminal_size_change_triggers_redraw_before_selection(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        terminal_tui,
        "_terminal_size_polling_interval",
        lambda: 0.01,
    )
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            select_one(
                "Choose",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        await asyncio.sleep(0.02)
        checks_before_resize = output.size_checks
        output.size = Size(rows=12, columns=40)
        await asyncio.sleep(0.08)
        pipe.send_text("\r")
        assert await task == "alpha-id"

    assert output.size_checks > checks_before_resize


async def test_very_small_selector_waits_for_resize_and_keeps_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        terminal_tui,
        "_terminal_size_polling_interval",
        lambda: 0.01,
    )
    output = _RecordingOutput()
    output.size = Size(rows=5, columns=31)
    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            select_one(
                "Choose",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        await _wait_for_output(output, "Terminal too small (31x5)")
        pipe.send_text("\r")
        await asyncio.sleep(0.05)
        assert not task.done()

        output.size = Size(rows=24, columns=80)
        await asyncio.sleep(0.08)
        pipe.send_text("\x1b[B\r")
        selected = await task

    assert selected == "beta-id"


def test_selector_visuals_share_ascii_theme_without_changing_values():
    state = terminal_selection._SelectionState(
        terminal_selection._normalize_options(_OPTIONS)
    )
    glyphs = terminal_tui._terminal_glyphs(
        terminal_tui.TerminalCapabilities("16", False)
    )

    rendered = "".join(
        text
        for _style, text in terminal_selection._render_fragments(
            "Choose",
            state,
            glyphs=glyphs,
            size=(80, 24),
        )
    )

    assert "DAITA SETUP" in rendered
    assert "> Alpha" in rendered
    assert "Up/Down move" in rendered
    assert "›" not in rendered
    assert state.selected_value() == "alpha-id"


def test_multi_selector_visuals_render_checked_state_deterministically():
    state = terminal_selection._MultiSelectionState(
        terminal_selection._normalize_options(_OPTIONS),
        maximum=3,
        empty_message="Select at least one option.",
        maximum_message="Select at most three options.",
    )
    state.toggle()

    rendered = "".join(
        text
        for _style, text in terminal_selection._render_multi_fragments(
            "Choose many",
            state,
            size=(80, 24),
        )
    )

    assert "› [x] Alpha" in rendered


async def test_labels_and_descriptions_are_sanitized_and_bounded():
    unsafe = "unsafe\x1b[31m\u202e" + ("x" * 400)
    selected, output = await _enhanced_choice(
        "\r",
        options=(SelectionOption("stable", unsafe, unsafe),),
    )

    assert selected == "stable"
    assert "\x1b[31m" not in output.text
    assert "\u202e" not in output.text
    assert "x" * 129 not in output.text
    assert len(output.text) < 4_096


async def test_cursor_is_restored_when_application_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fail(application: Any) -> Any:
        del application
        raise RuntimeError("render failed")

    monkeypatch.setattr(terminal_selection, "_run_application", fail)
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        with pytest.raises(RuntimeError, match="render failed"):
            await select_one(
                "Choose",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )

    assert output.show_count >= 1


async def test_multi_select_initial_highlight_starts_empty_and_space_selects_it():
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            select_many(
                "Choose many",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        await _wait_for_output(output, "› [ ] Alpha")
        assert "› [ ] Alpha" in output.text
        pipe.send_text(" \r")
        selected = await task

    assert selected == ("alpha-id",)
    assert output.show_count >= 1


@pytest.mark.parametrize(
    ("keys", "expected"),
    (
        ("\x1b[B \r", ("beta-id",)),
        ("\x1b[A \r", ("gamma-id",)),
        ("\x1b[B\x1b[B\x1b[B \r", ("alpha-id",)),
    ),
)
async def test_multi_select_up_down_and_wraparound(
    keys: str,
    expected: tuple[str, ...],
):
    selected, _ = await _enhanced_choices(keys)

    assert selected == expected


async def test_multi_select_space_toggles_and_untoggles_without_committing_empty():
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            select_many(
                "Choose many",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        pipe.send_text("  \r")
        await _wait_for_output(output, "Select at least one option.")
        assert not task.done()
        assert "Select at least one option." in output.text
        pipe.send_text("\x1b[B \r")
        selected = await task

    assert selected == ("beta-id",)


async def test_multi_select_returns_multiple_values_in_declared_option_order():
    selected, _ = await _enhanced_choices("\x1b[B \x1b[A \r")

    assert selected == ("alpha-id", "beta-id")


async def test_multi_select_stable_identities_survive_filtering_and_navigation():
    selected, _ = await _enhanced_choices(" ga \r")

    assert selected == ("alpha-id", "gamma-id")


async def test_multi_select_stable_values_are_independent_of_display_positions():
    reordered = (
        SelectionOption("gamma-id", "Alpha display"),
        SelectionOption("alpha-id", "Zulu display"),
        SelectionOption("beta-id", "Middle display"),
    )

    selected, _ = await _enhanced_choices(
        "\x1b[B \x1b[A \r",
        options=reordered,
    )

    assert selected == ("gamma-id", "alpha-id")


async def test_multi_select_enter_rejects_empty_then_confirms_nonempty_selection():
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            select_many(
                "Choose many",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        pipe.send_text("\r")
        for _ in range(100):
            if "Select at least one option." in output.text:
                break
            await asyncio.sleep(0.01)
        assert not task.done()
        assert "Select at least one option." in output.text
        pipe.send_text(" \r")
        selected = await task

    assert selected == ("alpha-id",)
    assert output.show_count >= 1


async def test_multi_select_enforces_maximum_without_replacing_existing_values():
    options = tuple(
        SelectionOption(f"schema-{index}", f"Schema {index}") for index in range(1, 34)
    )
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        task = asyncio.create_task(
            select_many(
                "Choose many",
                options,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        pipe.send_text(" " + ("\x1b[B " * 31))
        await asyncio.sleep(0.02)
        pipe.send_text("\x1b[B ")
        await asyncio.sleep(0.02)
        assert "Select at most 32 options." in output.text
        pipe.send_text("\r")
        selected = await task

    assert selected == tuple(f"schema-{index}" for index in range(1, 33))
    assert "schema-33" not in selected


async def test_multi_select_validation_messages_are_sanitized_and_bounded():
    unsafe = "unsafe\x1b[31m\u202e" + ("x" * 400)

    selected, output = await _enhanced_choices(
        "\r \r",
        empty_message=unsafe,
    )

    assert selected == ("alpha-id",)
    assert "\x1b[31m" not in output.text
    assert "\u202e" not in output.text
    assert "x" * 257 not in output.text
    assert len(output.text) < 8_192


@pytest.mark.parametrize(
    ("keys", "error"),
    (
        ("\x1b", SelectionCancelled),
        ("\x03", KeyboardInterrupt),
        ("\x04", EOFError),
    ),
)
async def test_multi_select_cancellation_restores_cursor(
    keys: str,
    error: type[BaseException],
):
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        with pytest.raises(error):
            await select_many(
                "Choose many",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )

    assert output.show_count >= 1


async def test_multi_select_restores_cursor_when_application_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fail(application: Any) -> Any:
        del application
        raise RuntimeError("multi render failed")

    monkeypatch.setattr(terminal_selection, "_run_application", fail)
    output = _RecordingOutput()
    with create_pipe_input() as pipe:
        with pytest.raises(RuntimeError, match="multi render failed"):
            await select_many(
                "Choose many",
                _OPTIONS,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                enhanced_input=pipe,
                enhanced_output=output,
            )

    assert output.show_count >= 1


async def test_multi_select_numbered_fallback_retries_and_preserves_requested_order():
    output = io.StringIO()

    selected = await select_many(
        "Choose many",
        _OPTIONS,
        input_stream=io.StringIO("\ninvalid\n1,1\n0\n1,,2\n1,2,3,1\n1,4\n3,1\n"),
        output_stream=output,
        maximum=3,
    )

    assert selected == ("gamma-id", "alpha-id")
    assert "Choices (comma-separated numbers): " in output.getvalue()
    assert output.getvalue().count("Choose 1 to 3 distinct option numbers.") == 7


async def test_multi_select_numbered_fallback_rejects_more_than_32_choices():
    options = tuple(
        SelectionOption(f"schema-{index}", f"Schema {index}") for index in range(1, 34)
    )
    excessive = ",".join(str(index) for index in range(1, 34))
    accepted = ",".join(str(index) for index in range(32, 0, -1))

    selected = await select_many(
        "Choose many",
        options,
        input_stream=io.StringIO(f"{excessive}\n{accepted}\n"),
        output_stream=io.StringIO(),
    )

    assert selected == tuple(f"schema-{index}" for index in range(32, 0, -1))


async def test_multi_select_import_failure_propagates_installation_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing() -> Any:
        raise ImportError("Repair it with: pipx reinstall daita-agents")

    monkeypatch.setattr(terminal_selection, "_load_prompt_toolkit", missing)

    with pytest.raises(ImportError, match="pipx reinstall daita-agents"):
        await select_many(
            "Choose many",
            _OPTIONS,
            input_stream=io.StringIO("2,1\n"),
            output_stream=io.StringIO(),
            enhanced_input=object(),
            enhanced_output=object(),
        )


async def test_multi_select_initialization_failure_restores_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(terminal_selection, "_create_multi_application", fail)
    output = _RecordingOutput()

    selected = await select_many(
        "Choose many",
        _OPTIONS,
        input_stream=io.StringIO("2\n"),
        output_stream=io.StringIO(),
        enhanced_input=object(),
        enhanced_output=output,
    )

    assert selected == ("beta-id",)
    assert output.show_count >= 1


async def test_single_select_import_failure_propagates_installation_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing() -> Any:
        raise ImportError("Repair it with: pipx reinstall daita-agents")

    monkeypatch.setattr(terminal_selection, "_load_prompt_toolkit", missing)

    with pytest.raises(ImportError, match="pipx reinstall daita-agents"):
        await select_one(
            "Choose",
            _OPTIONS,
            input_stream=io.StringIO("2\n"),
            output_stream=io.StringIO(),
            enhanced_input=object(),
            enhanced_output=object(),
        )


def test_missing_prompt_toolkit_uses_application_repair_guidance(
    monkeypatch: pytest.MonkeyPatch,
):
    real_import = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any):
        if name.startswith("prompt_toolkit"):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="pipx reinstall daita-agents"):
        terminal_selection._load_prompt_toolkit()


async def test_injected_text_streams_use_existing_numbered_fallback():
    output = io.StringIO()

    selected = await select_one(
        "Choose",
        _OPTIONS,
        input_stream=io.StringIO("3\n"),
        output_stream=output,
    )

    assert selected == "gamma-id"
    assert "Choice: " in output.getvalue()


async def test_agent_picker_uses_keyboard_selector_values(tmp_path: Path):
    for name in ("alpha", "zulu"):
        agent = await Agent.create(name, root=tmp_path)
        await agent.close()

    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[B\r")
        selected = await terminal._select_agent(
            root=tmp_path,
            requested_name=None,
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            keychain=None,
            model_validator=None,
            approval_handler=None,
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    try:
        assert selected.name == "zulu"
    finally:
        await selected.close()


@pytest.mark.parametrize(
    ("keys", "expected_code", "message"),
    (
        ("\x1b", 0, "Setup cancelled."),
        ("\x04", 0, "Setup cancelled."),
        ("\x03", 130, "Setup interrupted."),
    ),
)
async def test_enhanced_agent_cancellation_preserves_agents_and_releases_locks(
    tmp_path: Path,
    keys: str,
    expected_code: int,
    message: str,
):
    for name in ("alpha", "zulu"):
        agent = await Agent.create(name, root=tmp_path)
        await agent.close()
    output = io.StringIO()

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        result = await terminal.run_terminal_application(
            root=tmp_path,
            input_stream=io.StringIO(),
            output_stream=output,
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    assert result == expected_code
    assert message in output.getvalue()
    assert await Agent.list(root=tmp_path) == ("alpha", "zulu")
    for name in ("alpha", "zulu"):
        reopened = await Agent.open(name, root=tmp_path)
        await reopened.close()


async def test_provider_source_and_repair_pickers_use_shared_keyboard_selector():
    async def choose(call: Any, keys: str) -> Any:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            return await call(
                io.StringIO(),
                io.StringIO(),
                selection_input=pipe,
                selection_output=DummyOutput(),
            )

    assert (await choose(terminal._select_provider, "\x1b[B\r"))[0] == "anthropic"
    assert await choose(terminal._select_source_type, "\x1b[B\r") == "directory"
    assert await choose(terminal._select_catalog_repair, "\x1b[B\r") == "exit"


def test_provider_labels_distinguish_api_subscription_and_local_routes():
    assert terminal._PROVIDERS == (
        ("openai", "OpenAI API"),
        ("anthropic", "Anthropic API"),
        ("gemini", "Gemini API"),
        ("grok", "xAI (Grok) API"),
        ("ollama", "Ollama local"),
        ("codex", "Codex subscription"),
        ("claude-code", "Claude Code subscription"),
        ("grok-build", "Grok Build subscription"),
        ("custom", "Custom API (OpenAI-compatible)"),
    )


_EXPECTED_MODEL_SUGGESTIONS = {
    "openai": ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
    "anthropic": (
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
    ),
    "codex": ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
    "claude-code": (
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-haiku-4-5-20251001",
    ),
    "grok-build": ("grok-4.5",),
    "gemini": (
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ),
    "grok": ("grok-4.5",),
    "ollama": ("qwen3", "llama3.1", "mistral-small3.2"),
}


def test_model_suggestions_are_deterministic_bounded_presentation_metadata():
    assert tuple(terminal._MODEL_SUGGESTIONS) == tuple(_EXPECTED_MODEL_SUGGESTIONS)
    for provider, expected_ids in _EXPECTED_MODEL_SUGGESTIONS.items():
        suggestions = terminal._MODEL_SUGGESTIONS[provider]
        assert tuple(item.model_id for item in suggestions) == expected_ids
        assert all(item.provider_id == provider for item in suggestions)
        assert all(1 <= len(item.label) <= 64 for item in suggestions)
        assert all(1 <= len(item.description) <= 128 for item in suggestions)
        assert all("\n" not in item.description for item in suggestions)
        assert all(
            item.recommendation is None or len(item.recommendation) <= 32
            for item in suggestions
        )
        assert suggestions[0].recommendation is not None


async def test_model_menu_filters_case_insensitively_and_returns_stable_model_id():
    with create_pipe_input() as pipe:
        pipe.send_text("LuNa\r")
        selected = await terminal._select_model(
            "openai",
            "OpenAI",
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    assert selected == "gpt-5.6-luna"


@pytest.mark.parametrize("provider", tuple(_EXPECTED_MODEL_SUGGESTIONS))
async def test_each_model_menu_returns_its_exact_first_stable_id(provider: str):
    with create_pipe_input() as pipe:
        pipe.send_text("\r")
        selected = await terminal._select_model(
            provider,
            provider.title(),
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    assert selected == _EXPECTED_MODEL_SUGGESTIONS[provider][0]


async def test_model_menu_sanitizes_and_bounds_suggestion_display(
    monkeypatch: pytest.MonkeyPatch,
):
    unsafe = "unsafe\x1b[31m\u202e" + ("x" * 400)
    monkeypatch.setitem(
        terminal._MODEL_SUGGESTIONS,
        "openai",
        (
            terminal._ModelSuggestion(
                "openai",
                "stable-model-id",
                unsafe,
                unsafe,
                unsafe,
            ),
        ),
    )
    output = _RecordingOutput()

    with create_pipe_input() as pipe:
        pipe.send_text("\r")
        selected = await terminal._select_model(
            "openai",
            "OpenAI",
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            selection_input=pipe,
            selection_output=output,
        )

    assert selected == "stable-model-id"
    assert "\x1b[31m" not in output.text
    assert "\u202e" not in output.text
    assert "x" * 129 not in output.text
    assert len(output.text) < 4_096


@pytest.mark.parametrize("provider", tuple(_EXPECTED_MODEL_SUGGESTIONS))
async def test_every_builtin_model_menu_has_numbered_manual_entry(provider: str):
    manual_number = len(_EXPECTED_MODEL_SUGGESTIONS[provider]) + 1
    output = io.StringIO()

    selected = await terminal._select_model(
        provider,
        provider.title(),
        input_stream=io.StringIO(f"{manual_number}\nunlisted-model\n"),
        output_stream=output,
    )

    assert selected == "unlisted-model"
    assert "Enter a model ID manually…" in output.getvalue()
    assert "Model identifier: " in output.getvalue()


@pytest.mark.parametrize("provider", tuple(_EXPECTED_MODEL_SUGGESTIONS))
async def test_every_builtin_model_menu_can_return_to_provider_selection(provider: str):
    back_number = len(_EXPECTED_MODEL_SUGGESTIONS[provider]) + 2

    selected = await terminal._select_model(
        provider,
        provider.title(),
        input_stream=io.StringIO(f"{back_number}\n"),
        output_stream=io.StringIO(),
    )

    assert selected is None


async def test_escape_from_model_menu_returns_to_provider_selection():
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b")
        selected = await terminal._select_model(
            "openai",
            "OpenAI",
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    assert selected is None


@pytest.mark.parametrize(
    ("keys", "expected_code", "message"),
    (
        ("\r\x1b\x04", 0, "Setup cancelled."),
        ("\r\x03", 130, "Setup interrupted."),
        ("\r\x04", 0, "Setup cancelled."),
    ),
)
async def test_model_menu_cancellation_commits_no_configuration_and_releases_lock(
    tmp_path: Path,
    keys: str,
    expected_code: int,
    message: str,
):
    output = io.StringIO()
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        result = await terminal.run_terminal_application(
            root=tmp_path,
            input_stream=io.StringIO("atlas\n"),
            output_stream=output,
            selection_input=pipe,
            selection_output=DummyOutput(),
        )

    assert result == expected_code
    assert message in output.getvalue()
    assert not (tmp_path / "agents" / "atlas" / "config.json").exists()
    reopened = await Agent.open("atlas", root=tmp_path)
    await reopened.close()
