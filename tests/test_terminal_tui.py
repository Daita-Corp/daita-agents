from __future__ import annotations

import ast
import asyncio
import base64
from datetime import datetime, timezone
from decimal import Decimal
import io
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, TextIO

import pytest
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding.bindings.mouse import xterm_sgr_mouse_events
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import (
    MouseButton,
    MouseEvent,
    MouseEventType,
    MouseModifier,
)
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style

from daita import ApprovalDecision, ApprovalRequest, terminal, terminal_tui
from daita._json import FrozenJsonObject
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostBasis, CostEstimate
from daita.loop.models import RunInput, Transcript
from daita.observation import AgentEvent, AgentEventKind
from daita.tui import clipboard as tui_clipboard
from daita.tui import application as tui_application
from daita.tui import transcript_view as tui_transcript_view
from daita.terminal_tui import (
    MAX_COMPOSER_CHARACTERS,
    TerminalApplicationResult,
    TerminalCommandResult,
    TerminalObserverBridge,
    TerminalSuspendBridge,
    TerminalViewState,
    run_terminal_tui,
)


class _RecordingOutput(DummyOutput):
    def __init__(self) -> None:
        self.fragments: list[str] = []
        self.show_count = 0
        self.alternate_enter_count = 0
        self.alternate_exit_count = 0
        self.attribute_reset_count = 0
        self.autowrap_count = 0
        self.mouse_enable_count = 0
        self.mouse_disable_count = 0
        self.bracketed_paste_disable_count = 0
        self.cursor_key_reset_count = 0
        self.cursor_shape_reset_count = 0
        self.flush_count = 0
        self.size = Size(rows=30, columns=100)
        self.size_checks = 0

    def write(self, data: str) -> None:
        self.fragments.append(data)

    def write_raw(self, data: str) -> None:
        self.fragments.append(data)

    def get_size(self) -> Size:
        self.size_checks += 1
        return self.size

    def show_cursor(self) -> None:
        self.show_count += 1

    def enter_alternate_screen(self) -> None:
        self.alternate_enter_count += 1

    def quit_alternate_screen(self) -> None:
        self.alternate_exit_count += 1

    def reset_attributes(self) -> None:
        self.attribute_reset_count += 1

    def enable_autowrap(self) -> None:
        self.autowrap_count += 1

    def enable_mouse_support(self) -> None:
        self.mouse_enable_count += 1

    def disable_mouse_support(self) -> None:
        self.mouse_disable_count += 1

    def disable_bracketed_paste(self) -> None:
        self.bracketed_paste_disable_count += 1

    def reset_cursor_key_mode(self) -> None:
        self.cursor_key_reset_count += 1

    def reset_cursor_shape(self) -> None:
        self.cursor_shape_reset_count += 1

    def flush(self) -> None:
        self.flush_count += 1

    @property
    def text(self) -> str:
        return "".join(self.fragments)


def _result(
    text: str,
    *,
    run_id: str = "run-one",
    conversation_id: str = "conversation-one",
    steps: int = 1,
    tokens: int = 24,
    cost_estimate: CostEstimate | None = None,
) -> Any:
    return SimpleNamespace(
        run_id=run_id,
        conversation_id=conversation_id,
        final_text=text,
        kind=SimpleNamespace(value="completed"),
        reason="completed",
        steps=steps,
        usage=SimpleNamespace(
            total_tokens=tokens,
            cost_estimate=(cost_estimate or CostEstimate.complete(Decimal("0.01"))),
        ),
    )


async def _wait_until(predicate: Any) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


async def _run_shell(
    pipe: Any,
    output: _RecordingOutput,
    state: TerminalViewState,
    *,
    run_message: Any,
    load_transcript: Any = None,
    handle_command: Any = None,
    suspend_bridge: TerminalSuspendBridge | None = None,
    observer_bridge: TerminalObserverBridge | None = None,
    approval_bridge: terminal_tui.TerminalApprovalBridge | None = None,
    command_requires_suspension: Any = None,
    skill_completions: tuple[tuple[str, str], ...] = (),
    load_skill_completions: Any = None,
    source_completions: tuple[tuple[str, str, str], ...] = (),
    load_source_completions: Any = None,
) -> asyncio.Task[TerminalApplicationResult]:
    async def no_commands(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    return asyncio.create_task(
        run_terminal_tui(
            state,
            run_message=run_message,
            load_transcript=load_transcript,
            handle_command=handle_command or no_commands,
            command_requires_suspension=command_requires_suspension,
            skill_completions=skill_completions,
            load_skill_completions=load_skill_completions,
            source_completions=source_completions,
            load_source_completions=load_source_completions,
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            suspend_bridge=suspend_bridge or TerminalSuspendBridge(),
            observer_bridge=observer_bridge,
            approval_bridge=approval_bridge,
            enhanced_input=pipe,
            enhanced_output=output,
        )
    )


def _event(
    kind: AgentEventKind,
    data: dict[str, object],
    *,
    run_id: str = "run-live",
    conversation_id: str = "conversation-live",
) -> AgentEvent:
    return AgentEvent(
        kind=kind,
        occurred_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        run_id=run_id,
        conversation_id=conversation_id,
        data=FrozenJsonObject.from_mapping(data),
    )


def _approval_request(
    arguments: dict[str, object],
    *,
    call_id: str = "call-approval",
) -> ApprovalRequest:
    return ApprovalRequest(
        run_id="run-approval",
        call_id=call_id,
        tool_name="skill_save",
        capability_id="skills.write",
        arguments=FrozenJsonObject.from_mapping(arguments),
        reason="Allow this exact side effect once?",
    )


def _tool_transcript(
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResultBlock, ...],
    *,
    run_id: str = "run-one",
) -> Transcript:
    return Transcript(
        run=RunInput(
            id=run_id,
            agent_id="agent-one",
            message="question",
            created_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
            conversation_id="conversation-one",
        ),
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("question"),),
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=calls,
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=results,
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock("answer"),),
            ),
        ),
    )


async def test_ready_agent_enters_the_focused_tui(
    monkeypatch: pytest.MonkeyPatch,
):
    entered: list[TerminalViewState] = []

    async def fake_tui(
        state: TerminalViewState,
        **kwargs: Any,
    ) -> TerminalApplicationResult:
        entered.append(state)
        assert kwargs["run_message"]
        assert kwargs["load_transcript"]
        assert kwargs["handle_command"]
        assert kwargs["command_requires_suspension"]("/model") is True
        assert kwargs["command_requires_suspension"]("/help") is False
        assert isinstance(kwargs["observer_bridge"], TerminalObserverBridge)
        return TerminalApplicationResult(None, "exit")

    monkeypatch.setattr(terminal_tui, "run_terminal_tui", fake_tui)

    class _FakeAgent:
        name = "atlas"
        home = Path("/tmp/daita/agents/atlas")
        model_route = SimpleNamespace(
            candidates=(SimpleNamespace(provider_id="openai:gpt-5.6-sol"),)
        )

        async def list_sources(self) -> tuple[Any, ...]:
            return (SimpleNamespace(active=True, display_name="Warehouse"),)

        async def catalog_summary(self) -> Any:
            return SimpleNamespace(resource_count=1, relationship_count=0)

    marker_input = object()
    marker_output = object()
    agent = cast(Any, _FakeAgent())
    selected, conversation_id, action = await terminal._chat(
        agent,
        root=None,
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=None,
        validated=False,
        tui_input=marker_input,
        tui_output=marker_output,
        suspend_bridge=TerminalSuspendBridge(),
    )

    assert selected is agent
    assert conversation_id is None
    assert action == "exit"
    assert len(entered) == 1
    assert entered[0].agent_label == "atlas"
    assert entered[0].model_label == "gpt-5.6-sol"
    assert entered[0].source_summary == "Warehouse"
    assert entered[0].startup is not None
    assert entered[0].startup.agent_home == "/tmp/daita/agents/atlas"


async def test_ready_tui_emits_startup_before_any_transcript_blocks():
    output = _RecordingOutput()
    state = TerminalViewState(
        "atlas",
        "gpt-5.6-sol",
        "Warehouse",
        startup=terminal_tui.TerminalStartupInfo(
            version="1.0.0",
            provider_label="OpenAI",
            model_status="configured",
            agent_home="/tmp/daita/atlas",
            source_count=1,
            resource_count=12,
            relationship_count=2,
            read_capabilities=(
                "Catalog search & inspection",
                "SQLite queries",
            ),
        ),
    )

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=no_message,
        )
        await _wait_until(lambda: "Ask a question about your data" in output.text)
        pipe.send_text("\x04")
        result = await task

    assert result == TerminalApplicationResult(None, "exit")
    assert state.blocks == []
    assert "████▄" in output.text
    assert "OpenAI · gpt-5.6-sol · configured" in output.text
    assert "12 resources · 2 relationships" in output.text
    assert "Quick actions: /sources  /catalog  /help" in output.text


def test_ready_tui_reflows_live_startup_when_terminal_width_changes():
    output = _RecordingOutput()
    state = TerminalViewState(
        "atlas",
        "gpt-5.6-sol",
        "Warehouse",
        startup=terminal_tui.TerminalStartupInfo(
            version="1.0.0",
            provider_label="OpenAI",
            model_status="configured",
            agent_home="/tmp/daita/agents/atlas",
            source_count=1,
            resource_count=12,
            relationship_count=2,
            read_capabilities=(
                "Catalog search & inspection",
                "SQLite queries",
            ),
        ),
    )

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        root = application.layout.container
        main_shell = root.children[0].content
        startup_window = main_shell.children[0]

        def rendered_startup() -> str:
            return "".join(text for _style, text in startup_window.content.text())

        initial = rendered_startup()
        assert "████▄" in initial
        assert (
            max(terminal_tui._display_width(line) for line in initial.splitlines())
            <= 100
        )

        output.size = Size(rows=30, columns=60)
        application.before_render.fire()
        narrow = rendered_startup()
        assert "DAITA  1.0.0" in narrow
        assert "████▄" not in narrow
        assert (
            max(terminal_tui._display_width(line) for line in narrow.splitlines()) <= 60
        )

        output.size = Size(rows=30, columns=120)
        application.before_render.fire()
        wide = rendered_startup()
        assert "████▄" in wide
        assert (
            max(terminal_tui._display_width(line) for line in wide.splitlines()) <= 120
        )
        assert wide != narrow


async def test_full_screen_resize_repaints_without_leaving_alternate_screen():
    output = _RecordingOutput()
    state = TerminalViewState(
        "atlas",
        "gpt-5.6-sol",
        "Warehouse",
        startup=terminal_tui.TerminalStartupInfo(
            version="1.0.0",
            provider_label="OpenAI",
            model_status="configured",
            agent_home="/tmp/daita/agents/atlas",
            source_count=1,
            resource_count=12,
            relationship_count=2,
            read_capabilities=("Catalog search & inspection", "SQLite queries"),
        ),
    )

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)

        output.size = Size(rows=24, columns=60)
        application._on_resize()
        await _wait_until(lambda: application.renderer._last_size == output.size)

        assert output.alternate_enter_count == 1
        assert output.alternate_exit_count == 0

        pipe.send_text("\x04")
        result = await task

    assert result == TerminalApplicationResult(None, "exit")
    assert output.alternate_enter_count == 1
    assert output.alternate_exit_count == 1


async def test_full_screen_transcript_uses_page_keys_and_mouse_wheel_for_scrolling(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    for index in range(50):
        state.blocks.append(
            terminal_tui.TerminalBlock(
                "assistant",
                f"Transcript row {index:02d}",
            )
        )

    canonical_renders = 0
    original_canonical_assistant_text = tui_transcript_view._canonical_assistant_text

    def counted_canonical_assistant_text(*args: Any, **kwargs: Any) -> str:
        nonlocal canonical_renders
        canonical_renders += 1
        return original_canonical_assistant_text(*args, **kwargs)

    monkeypatch.setattr(
        tui_transcript_view,
        "_canonical_assistant_text",
        counted_canonical_assistant_text,
    )

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        content_window = application.layout.container.children[0].content.children[0]
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        assert application.mouse_support() is True
        assert output.mouse_enable_count == 1

        initial_content = content_window.content.create_content(98, None)
        initial_cursor_row = initial_content.cursor_position.y
        initial_builds = state.transcript_viewport.projection_build_count
        initial_canonical_renders = canonical_renders
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.FOLLOWING
        )
        render_counter = application.render_counter
        pipe.send_text("\x1b[5~")
        await _wait_until(lambda: application.render_counter > render_counter)
        page_up_content = content_window.content.create_content(98, None)
        assert page_up_content.cursor_position.y < initial_cursor_row
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )

        first_review_cursor_row = page_up_content.cursor_position.y
        render_counter = application.render_counter
        pipe.send_text("\x1b[5~")
        await _wait_until(lambda: application.render_counter > render_counter)
        second_page_up_content = content_window.content.create_content(98, None)
        assert second_page_up_content.cursor_position.y < first_review_cursor_row

        render_counter = application.render_counter
        pipe.send_text("\x1b[6~")
        await _wait_until(lambda: application.render_counter > render_counter)
        page_down_content = content_window.content.create_content(98, None)
        assert page_down_content.cursor_position.y == first_review_cursor_row
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )

        render_counter = application.render_counter
        pipe.send_text("\x1b[6~")
        await _wait_until(lambda: application.render_counter > render_counter)
        final_page_down_content = content_window.content.create_content(98, None)
        assert final_page_down_content.cursor_position.y == initial_cursor_row
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.FOLLOWING
        )
        latest_scroll = content_window.vertical_scroll

        scroll_up = MouseEvent(
            position=Point(x=1, y=1),
            event_type=MouseEventType.SCROLL_UP,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
        render_counter = application.render_counter
        assert content_window.content.mouse_handler(scroll_up) is None
        await _wait_until(lambda: application.render_counter > render_counter)
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        mouse_anchor = state.transcript_viewport.anchor
        assert mouse_anchor is not None
        assert (
            1
            <= latest_scroll - content_window.vertical_scroll
            <= (terminal_tui._MOUSE_SCROLL_LINES)
        )

        scroll_down = MouseEvent(
            position=Point(x=1, y=1),
            event_type=MouseEventType.SCROLL_DOWN,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
        render_counter = application.render_counter
        assert content_window.content.mouse_handler(scroll_down) is None
        await _wait_until(lambda: application.render_counter > render_counter)
        mouse_down_content = content_window.content.create_content(98, None)
        assert mouse_down_content.cursor_position.y == initial_cursor_row
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.FOLLOWING
        )
        assert state.transcript_viewport.projection_build_count == initial_builds
        assert canonical_renders == initial_canonical_renders

        pipe.send_text("\x04")
        await task
        assert output.mouse_disable_count == 1


async def test_ctrl_home_new_output_affordance_click_and_ctrl_end_follow_latest():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    for index in range(40):
        state.append_plain("assistant", f"Transcript row {index:02d}")
    original_texts = tuple(block.text for block in state.blocks)

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        main_shell = application.layout.container.children[0].content
        indicator_window = main_shell.children[1].content
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)

        pipe.send_text("\x1b[1;5H")
        await _wait_until(
            lambda: state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        anchor = state.transcript_viewport.anchor
        assert anchor is not None
        projection = cast(Any, state.transcript_projection)
        assert state.transcript_viewport.top_row(projection, viewport_rows=8) == 0
        assert tuple(block.text for block in state.blocks) == original_texts

        for index in range(3):
            state.append_plain("assistant", f"new row {index}")
        application.invalidate()
        await _wait_until(lambda: "3 new items" in output.text)
        assert state.transcript_viewport.anchor == anchor
        assert state.transcript_viewport.unseen_items == 3

        indicator_window.content.create_content(output.size.columns, 1)
        transcript_press = MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        cross_control_release = MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        assert main_shell.children[0].content.mouse_handler(transcript_press) is None
        assert indicator_window.content.mouse_handler(cross_control_release) is None
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        assert state.transcript_viewport.unseen_items == 3

        press = MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        release = MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        assert indicator_window.content.mouse_handler(press) is None
        assert indicator_window.content.mouse_handler(release) is None
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.FOLLOWING
        )
        assert state.transcript_viewport.unseen_items == 0

        pipe.send_text("\x1b[1;5H")
        await _wait_until(
            lambda: state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        pipe.send_text("\x1b[1;5F")
        await _wait_until(
            lambda: state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.FOLLOWING
        )
        pipe.send_text("\x04")
        await task


async def test_submit_while_reviewing_preserves_anchor_and_counts_new_blocks():
    output = _RecordingOutput()
    state = TerminalViewState(
        "atlas",
        "model",
        "source",
        conversation_id="conversation-one",
    )
    for index in range(40):
        state.append_plain("assistant", f"earlier row {index}")
    messages: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        assert conversation_id == "conversation-one"
        messages.append(message)
        return _result("new answer")

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        await _wait_until(lambda: output.alternate_enter_count == 1)
        pipe.send_text("\x1b[1;5H")
        await _wait_until(
            lambda: state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        anchor = state.transcript_viewport.anchor

        pipe.send_text("new question\r")
        await _wait_until(lambda: messages == ["new question"] and not state.running)

        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        assert state.transcript_viewport.anchor == anchor
        assert state.transcript_viewport.unseen_items == 2
        pipe.send_text("\x04")
        await task


def test_review_counter_counts_new_blocks_not_tool_status_or_initial_hydration():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    state.append_plain("assistant", "earlier output\n" * 10)
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=40,
        capabilities=capabilities,
    )
    projection = cast(Any, state.transcript_projection)
    state.transcript_viewport.review_start(projection)

    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {"call_id": "call-new", "tool_name": "catalog_search"},
        )
    )
    assert state.transcript_viewport.unseen_items == 0
    state.apply_event(
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": "call-new",
                "tool_name": "catalog_search",
                "duration_ms": 5,
                "success": True,
            },
        )
    )
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=40,
        capabilities=capabilities,
    )
    assert state.transcript_viewport.unseen_items == 0

    call = ToolCall(id="call-history", name="catalog_search", arguments={"query": "x"})
    history = _tool_transcript(
        (call,),
        (
            ToolResultBlock(
                call_id=call.id,
                output={"kind": "catalog.search_result", "data": {"value": 1}},
            ),
        ),
        run_id="run-history",
    )
    state.hydrate_transcript(history, run_id="run-history", initial=True)
    assert state.transcript_viewport.unseen_items == 0


def test_installed_prompt_toolkit_sgr_mouse_protocol_spike_is_deterministic():
    packets = (
        "\x1b[<0;10;5M",  # left down
        "\x1b[<32;11;5M",  # left drag/move
        "\x1b[<0;11;5m",  # left up
        "\x1b[<64;12;5M",  # wheel up
        "\x1b[<81;12;5M",  # control + wheel down
    )
    parsed: list[Any] = []
    parser = Vt100Parser(parsed.append)

    parser.feed_and_flush("".join(packets))

    assert [(press.key, press.data) for press in parsed] == [
        (Keys.Vt100MouseEvent, packet) for packet in packets
    ]
    assert xterm_sgr_mouse_events[(0, "M")] == (
        MouseButton.LEFT,
        MouseEventType.MOUSE_DOWN,
        frozenset(),
    )
    assert xterm_sgr_mouse_events[(32, "M")] == (
        MouseButton.LEFT,
        MouseEventType.MOUSE_MOVE,
        frozenset(),
    )
    assert xterm_sgr_mouse_events[(0, "m")] == (
        MouseButton.LEFT,
        MouseEventType.MOUSE_UP,
        frozenset(),
    )
    assert xterm_sgr_mouse_events[(64, "M")] == (
        MouseButton.NONE,
        MouseEventType.SCROLL_UP,
        frozenset(),
    )
    assert xterm_sgr_mouse_events[(81, "M")] == (
        MouseButton.NONE,
        MouseEventType.SCROLL_DOWN,
        frozenset({MouseModifier.CONTROL}),
    )
    mouse_event = MouseEvent(
        position=Point(0, 0),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    )
    assert not hasattr(mouse_event, "click_count")


def test_rendered_selection_maps_wide_combining_and_emoji_cells_exactly():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("e\u0301界🙂x")
    maps: list[Any] = []

    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=20,
        capabilities=capabilities,
        rendered_transcript_maps=maps,
    )
    rendered_map = maps[0]
    block = state.transcript_document.blocks[0]

    assert rendered_map.position_for_cell(2, 1) == state.transcript_document.position(
        block.id, 0
    )
    assert rendered_map.position_for_cell(2, 2) == state.transcript_document.position(
        block.id, 2
    )
    assert rendered_map.position_for_cell(2, 3) == state.transcript_document.position(
        block.id, 2
    )
    assert rendered_map.position_for_cell(2, 4) == state.transcript_document.position(
        block.id, 3
    )
    assert rendered_map.position_for_cell(2, 5) == state.transcript_document.position(
        block.id, 3
    )
    assert rendered_map.position_for_cell(2, 7) == state.transcript_document.position(
        block.id, len(block.text)
    )

    state.transcript_selection.begin(
        state.transcript_document,
        rendered_map.position_for_cell(2, 1),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        rendered_map.position_for_cell(2, 7),
    )
    highlighted = terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=20,
        capabilities=capabilities,
    )

    assert state.transcript_selection.text == "e\u0301界🙂x"
    assert (
        "".join(
            text
            for style, text in highlighted
            if "class:tui.transcript.selection" in style
        )
        == "e\u0301界🙂x"
    )


def test_drag_mapping_across_wrapped_blocks_copies_logical_text_once():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("alpha beta gamma delta epsilon zeta eta theta")
    state.append_user("path /tmp/data.csv\n| east | 42 |")
    maps: list[Any] = []
    fragments = terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=20,
        capabilities=capabilities,
        rendered_transcript_maps=maps,
    )
    lines = "".join(text for _style, text in fragments).splitlines()
    start_row = next(row for row, line in enumerate(lines) if "beta" in line)
    end_row = next(row for row, line in enumerate(lines) if "42" in line)
    start_cell = lines[start_row].index("beta")
    end_cell = lines[end_row].index("42") + len("42")
    rendered_map = maps[0]
    start = rendered_map.position_for_cell(start_row, start_cell)
    end = rendered_map.position_for_cell(end_row, end_cell)
    assert start is not None and end is not None

    state.transcript_selection.begin(state.transcript_document, start)
    selected = state.transcript_selection.finish(state.transcript_document, end)
    assert selected is not None

    first, second = state.transcript_document.blocks
    expected = state.transcript_document.normalize_range(
        state.transcript_document.position(first.id, first.text.index("beta")),
        state.transcript_document.position(
            second.id,
            second.text.index("42") + len("42"),
        ),
    )
    assert selected.text == expected.text
    assert selected.text.count("beta") == 1
    assert selected.text.count("42") == 1


def test_selection_copy_text_is_wrap_stable_and_visible_projection_changes_clear_it():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    logical = "SELECT café, 界, 🙂 FROM /tmp/very-long-path/data.csv\n| east | 42 |"
    state.append_user(logical)
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=24,
        capabilities=capabilities,
    )
    block = state.transcript_document.blocks[0]
    state.transcript_selection.begin(
        state.transcript_document,
        state.transcript_document.position(block.id, 0),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        state.transcript_document.position(block.id, len(logical)),
    )

    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=100,
        capabilities=capabilities,
    )
    assert state.transcript_selection.text == logical

    state.blocks[0].text = "SELECT café, 界, 🙂"
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=40,
        capabilities=capabilities,
    )
    assert state.transcript_selection.active is False
    assert state.notice == (
        "Transcript selection cleared because visible content changed."
    )


def test_streaming_append_preserves_selection_and_cancellation_removal_clears_it():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    state.apply_model_text_delta("run-stream", 1, "draft answer")
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=40,
        capabilities=capabilities,
    )
    block = state.transcript_document.blocks[0]
    start = state.transcript_document.text(block.id).index("draft")
    state.transcript_selection.begin(
        state.transcript_document,
        state.transcript_document.position(block.id, start),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        state.transcript_document.position(block.id, start + len("draft")),
    )

    state.apply_model_text_delta("run-stream", 1, " continues")
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=20,
        capabilities=capabilities,
    )

    assert state.transcript_selection.text == "draft"
    state.settle_cancelled_run()
    assert state.transcript_selection.has_state is False


def test_collapsed_tool_payload_and_approval_arguments_are_not_selectable():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    secret = "hidden-approval-token"
    card = terminal_tui.ToolCardState(
        run_id="run-safe",
        call_id="call-safe",
        capability_id="data.sqlite.query",
        label="Query SQLite",
        state="succeeded",
        details=terminal_tui.ToolCardDetails(
            summary="3 rows returned",
            arguments_text=f'{{"token":"{secret}"}}',
            result_text=secret,
        ),
        expanded=False,
    )
    state.blocks.append(terminal_tui.TerminalBlock("tool", card.call_id, card))
    state.approval_panel = terminal_tui.ApprovalPanelState(
        "skill_save",
        "skills.write",
        f'{{"token":"{secret}"}}',
    )

    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=80,
        capabilities=capabilities,
    )

    selectable = "\n".join(
        state.transcript_document.text(block.id)
        for block in state.transcript_document.blocks
    )
    assert "1 tool call" in selectable
    assert "Ctrl-O view results" in selectable
    assert "3 rows returned" not in selectable
    assert secret not in selectable


def test_clipboard_priority_and_osc52_tmux_encoding_are_bounded_and_safe():
    assert terminal_tui._clipboard_mechanism(platform="darwin", environ={}) == (
        "pbcopy"
    )
    assert (
        terminal_tui._clipboard_mechanism(
            platform="darwin", environ={"SSH_TTY": "/dev/pts/1"}
        )
        == "osc52"
    )
    assert (
        terminal_tui._clipboard_mechanism(
            platform="darwin", environ={"TMUX": "/tmp/tmux"}
        )
        == "pbcopy"
    )
    assert (
        terminal_tui._clipboard_mechanism(
            platform="linux", environ={"TMUX": "/tmp/tmux"}
        )
        == "osc52"
    )

    payload = b"visible\x1b]52;c;inert\x07"
    plain = terminal_tui._osc52_sequence(payload, tmux=False)
    wrapped = terminal_tui._osc52_sequence(payload, tmux=True)
    encoded = plain.removeprefix("\x1b]52;c;").removesuffix("\x07")

    assert base64.b64decode(encoded) == payload
    assert plain.count("\x1b") == 1
    assert wrapped == f"\x1bPtmux;\x1b{plain}\x1b\\"
    assert (
        len(
            terminal_tui._osc52_sequence(
                b"x" * terminal_tui.MAX_CLIPBOARD_UTF8_BYTES,
                tmux=True,
            ).encode("ascii")
        )
        < 88_000
    )


async def test_clipboard_acknowledgement_failure_and_utf8_bound_are_truthful(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    seen: list[bytes] = []

    def acknowledge(payload: bytes) -> terminal_tui.ClipboardResult:
        seen.append(payload)
        return terminal_tui.ClipboardResult("copied", "pbcopy", "Copied")

    monkeypatch.setattr(tui_clipboard, "copy_with_pbcopy", acknowledge)
    copied = await terminal_tui._deliver_clipboard(
        "café🙂",
        output=output,
        platform="darwin",
        environ={},
    )
    requested = await terminal_tui._deliver_clipboard(
        "remote",
        output=output,
        platform="linux",
        environ={"SSH_TTY": "/dev/pts/1"},
    )
    oversized = await terminal_tui._deliver_clipboard(
        "é" * (terminal_tui.MAX_CLIPBOARD_UTF8_BYTES // 2 + 1),
        output=output,
        platform="darwin",
        environ={},
    )

    assert seen == ["café🙂".encode("utf-8")]
    assert copied == terminal_tui.ClipboardResult("copied", "pbcopy", "Copied")
    assert requested.status == "requested"
    assert requested.message == "Copy request sent to terminal"
    assert oversized.status == "failure"
    assert "64 KiB UTF-8 limit" in oversized.message
    assert seen == ["café🙂".encode("utf-8")]


def test_failed_osc52_delivery_does_not_mutate_selection():
    class BrokenOutput:
        def write_raw(self, data: str) -> None:
            del data
            raise OSError("blocked")

    state = TerminalViewState("atlas", "model", "source")
    state.append_user("keep selected")
    block = state.transcript_document.blocks[0]
    state.transcript_selection.begin(
        state.transcript_document,
        state.transcript_document.position(block.id, 0),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        state.transcript_document.position(block.id, len(block.text)),
    )

    result = terminal_tui._send_osc52_request(
        BrokenOutput(),
        state.transcript_selection.text.encode("utf-8"),
        tmux=False,
    )

    assert result.status == "failure"
    assert "often Shift" in result.message
    assert state.transcript_selection.text == "keep selected"


async def test_mouse_drag_highlights_and_ctrl_c_copies_without_cancelling_run(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("select this text")
    copied: list[str] = []

    async def deliver(
        text: str,
        **kwargs: Any,
    ) -> terminal_tui.ClipboardResult:
        del kwargs
        copied.append(text)
        return terminal_tui.ClipboardResult("failure", "test", "Copy failed in test.")

    monkeypatch.setattr(tui_application, "_deliver_clipboard", deliver)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        await asyncio.Event().wait()

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        content_window = application.layout.container.children[0].content.children[0]
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        content_window.content.create_content(98, None)

        down = MouseEvent(
            position=Point(x=1, y=2),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        move = MouseEvent(
            position=Point(x=7, y=2),
            event_type=MouseEventType.MOUSE_MOVE,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        up = MouseEvent(
            position=Point(x=7, y=2),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        render_counter = application.render_counter
        assert content_window.content.mouse_handler(down) is None
        assert content_window.content.mouse_handler(move) is None
        assert content_window.content.mouse_handler(up) is None
        assert state.transcript_selection.text == "select"
        assert state.transient_selection_hint == "Selected · Ctrl+C copy · Esc clear"
        await _wait_until(lambda: application.render_counter > render_counter)
        highlighted = content_window.content.create_content(98, None)
        assert any(
            "class:tui.transcript.selection" in style
            for row in range(highlighted.line_count)
            for style, _text in highlighted.get_line(row)
        )

        pipe.send_text("keep running\r")
        await _wait_until(lambda: state.running and state.active_task is not None)
        active = state.active_task
        pipe.send_text("\x03")
        await _wait_until(lambda: copied == ["select"])

        assert active is state.active_task
        assert active is not None and not active.done()
        assert state.running is True
        assert state.transcript_selection.text == "select"
        assert state.notice == "Copy failed in test."

        pipe.send_text("\x1b")
        await _wait_until(lambda: not state.transcript_selection.has_state)
        assert active is not None and not active.done()
        pipe.send_text("\x03")
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert copied == ["select"]


async def test_mouse_failure_is_presentation_only_during_an_active_run(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("select this text")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        await asyncio.Event().wait()

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        content = application.layout.container.children[0].content.children[0].content
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        content.create_content(98, None)
        pipe.send_text("keep running\r")
        await _wait_until(lambda: state.running and state.active_task is not None)
        active = state.active_task

        down = MouseEvent(
            position=Point(x=1, y=2),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        move = MouseEvent(
            position=Point(x=7, y=2),
            event_type=MouseEventType.MOUSE_MOVE,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        assert content.mouse_handler(down) is None

        def fail_mouse(*args: Any, **kwargs: Any) -> int:
            del args, kwargs
            raise RuntimeError("mouse failed")

        monkeypatch.setattr(
            tui_application,
            "bounded_selection_auto_scroll",
            fail_mouse,
        )
        assert content.mouse_handler(move) is None

        assert state.notice == (
            "Mouse interaction unavailable; keyboard controls remain active."
        )
        assert state.active_task is active
        assert active is not None and not active.done()
        assert state.running is True
        pipe.send_text("\x03")
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task


def test_selection_guidance_is_transient_status_not_permanent_footer_or_approval_chrome():
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("selected")
    block = state.transcript_document.blocks[0]
    state.transcript_selection.begin(
        state.transcript_document,
        state.transcript_document.position(block.id, 0),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        state.transcript_document.position(block.id, len(block.text)),
    )

    permanent = "".join(
        text for _style, text in terminal_tui._status_right_fragments(state)
    )
    assert "Ctrl+C copy" not in permanent
    assert "Esc clear" not in permanent

    state.transient_selection_hint = "Selected · Ctrl+C copy · Esc clear"
    transient = "".join(
        text for _style, text in terminal_tui._status_right_fragments(state)
    )
    assert "Selected · Ctrl+C copy · Esc clear" in transient

    state.approval_panel = cast(
        Any,
        terminal_tui._approval_panel_for_request(_approval_request({"name": "safe"})),
    )
    approval_status = "".join(
        text for _style, text in terminal_tui._status_right_fragments(state)
    )
    assert "Ctrl+C copy" not in approval_status


def test_transcript_press_replaces_selection_and_blank_transcript_press_clears_it():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("alpha beta gamma")

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        content = application.layout.container.children[0].content.children[0].content
        content.create_content(98, None)

        def left(event_type: MouseEventType, *, x: int, y: int) -> MouseEvent:
            return MouseEvent(
                position=Point(x=x, y=y),
                event_type=event_type,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            )

        assert content.mouse_handler(left(MouseEventType.MOUSE_DOWN, x=1, y=2)) is None
        assert content.mouse_handler(left(MouseEventType.MOUSE_UP, x=6, y=2)) is None
        assert state.transcript_selection.text == "alpha"

        assert content.mouse_handler(left(MouseEventType.MOUSE_DOWN, x=7, y=2)) is None
        assert state.transcript_selection.text == ""
        assert state.transcript_selection.dragging is True
        assert state.transient_selection_hint == ""
        assert content.mouse_handler(left(MouseEventType.MOUSE_UP, x=11, y=2)) is None
        assert state.transcript_selection.text == "beta"

        assert content.mouse_handler(left(MouseEventType.MOUSE_DOWN, x=0, y=0)) is None
        assert content.mouse_handler(left(MouseEventType.MOUSE_UP, x=0, y=0)) is None
        assert state.transcript_selection.has_state is False
        assert state.transient_selection_hint == ""


async def test_command_menu_and_composer_clicks_exclusively_own_their_press_sequence():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("transcript selection")
    block = state.transcript_document.blocks[0]

    def restore_selection() -> None:
        state.transcript_selection.begin(
            state.transcript_document,
            state.transcript_document.position(block.id, 0),
        )
        state.transcript_selection.finish(
            state.transcript_document,
            state.transcript_document.position(block.id, len("transcript")),
        )

    restore_selection()

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        main_shell = application.layout.container.children[0].content
        transcript_control = main_shell.children[0].content
        menu_control = main_shell.children[4].content.children[1].content
        composer_control = application.layout.current_control
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        transcript_control.create_content(98, None)
        pipe.send_text("/")
        await _wait_until(lambda: application.current_buffer.complete_state is not None)

        transcript_press = MouseEvent(
            position=Point(x=1, y=2),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        menu_release = MouseEvent(
            position=Point(x=1, y=1),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        assert transcript_control.mouse_handler(transcript_press) is None
        assert menu_control.mouse_handler(menu_release) is None
        assert application.current_buffer.text == "/"
        assert state.transcript_selection.dragging is False

        restore_selection()
        menu_press = MouseEvent(
            position=Point(x=1, y=1),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        assert menu_control.mouse_handler(menu_press) is None
        assert menu_control.mouse_handler(menu_release) is None
        assert application.current_buffer.text != "/"
        assert state.transcript_selection.text == "transcript"

        state.transient_selection_hint = "Selected · Ctrl+C copy · Esc clear"
        composer_press = MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        composer_release = MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.MOUSE_UP,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        assert composer_control.mouse_handler(composer_press) is None
        assert composer_control.mouse_handler(composer_release) is None
        assert state.transcript_selection.text == "transcript"
        assert state.transient_selection_hint == ""

        application.current_buffer.reset()
        pipe.send_text("\x04")
        await task


async def test_transcript_then_composer_copy_precedence_never_submits_draft(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("transcript")
    block = state.transcript_document.blocks[0]
    state.transcript_selection.begin(
        state.transcript_document,
        state.transcript_document.position(block.id, 0),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        state.transcript_document.position(block.id, len(block.text)),
    )
    copied: list[str] = []
    submitted: list[str] = []

    async def deliver(
        text: str,
        **kwargs: Any,
    ) -> terminal_tui.ClipboardResult:
        del kwargs
        copied.append(text)
        if text == "draft":
            return terminal_tui.ClipboardResult(
                "failure", "test", "Copy failed in test."
            )
        return terminal_tui.ClipboardResult("copied", "test", "Copied")

    monkeypatch.setattr(tui_application, "_deliver_clipboard", deliver)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result("unexpected")

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        pipe.send_text("draft remains")
        await _wait_until(lambda: application.current_buffer.text == "draft remains")
        application.current_buffer.cursor_position = 0
        application.current_buffer.start_selection()
        application.current_buffer.cursor_position = len("draft")

        pipe.send_text("\x03")
        await _wait_until(lambda: copied == ["transcript"])
        assert application.current_buffer.selection_state is not None

        pipe.send_text("\x1b")
        await _wait_until(lambda: not state.transcript_selection.active)
        assert application.current_buffer.selection_state is not None
        pipe.send_text("\x03")
        await _wait_until(lambda: copied == ["transcript", "draft"])

        assert submitted == []
        assert application.current_buffer.text == "draft remains"
        assert application.current_buffer.selection_state is not None
        assert [item.text for item in state.blocks] == ["transcript"]
        application.current_buffer.reset()
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert submitted == []


async def test_selection_drag_auto_scrolls_only_one_row_per_mouse_event():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("\n".join(f"row-{index:03d}" for index in range(100)))

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        content_window = application.layout.container.children[0].content.children[0]
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(
            lambda: output.alternate_enter_count == 1
            and content_window.render_info is not None
        )
        initial_top = content_window.vertical_scroll
        down = MouseEvent(
            position=Point(x=1, y=initial_top + 2),
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )
        move = MouseEvent(
            position=Point(x=1, y=initial_top),
            event_type=MouseEventType.MOUSE_MOVE,
            button=MouseButton.LEFT,
            modifiers=frozenset(),
        )

        assert content_window.content.mouse_handler(down) is None
        assert content_window.content.mouse_handler(move) is None
        await _wait_until(lambda: content_window.vertical_scroll != initial_top)

        assert initial_top - content_window.vertical_scroll == 1
        assert state.transcript_selection.active is True
        application.current_buffer.reset()
        pipe.send_text("\x04")
        await task


async def test_long_transcript_navigation_work_is_bounded_to_viewport_overscan(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    text = "\n".join(f"/tmp/daita/data/file-{index:05d}.csv" for index in range(20_000))
    state.append_user(text)
    block = state.transcript_document.blocks[0]
    state.transcript_selection.begin(
        state.transcript_document,
        state.transcript_document.position(block.id, 0),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        state.transcript_document.position(block.id, len("row-00000")),
    )
    highlighted_rows = 0
    original = terminal_tui._highlight_transcript_line
    runtime = terminal_tui._load_terminal_runtime()
    base_content_builds = 0
    original_base_create = runtime["FormattedTextControl"].create_content

    def counted_highlight(*args: Any, **kwargs: Any) -> list[tuple[str, str]]:
        nonlocal highlighted_rows
        highlighted_rows += 1
        return original(*args, **kwargs)

    def counted_base_create(control: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal base_content_builds
        if type(control).__name__ == "TranscriptFormattedTextControl":
            base_content_builds += 1
        return original_base_create(control, *args, **kwargs)

    monkeypatch.setattr(
        tui_application,
        "_highlight_transcript_line",
        counted_highlight,
    )
    monkeypatch.setattr(
        runtime["FormattedTextControl"],
        "create_content",
        counted_base_create,
    )

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                runtime,
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        content_control = (
            application.layout.container.children[0].content.children[0].content
        )
        content_window = application.layout.container.children[0].content.children[0]
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(
            lambda: output.alternate_enter_count == 1
            and content_window.render_info is not None
        )
        assert len(text.encode("utf-8")) < 2 * 1_024 * 1_024
        viewport_rows = content_window.render_info.window_height
        initial_builds = state.transcript_viewport.projection_build_count
        initial_base_content_builds = base_content_builds
        highlighted_rows = 0
        render_counter = application.render_counter

        scroll_up = MouseEvent(
            position=Point(x=1, y=1),
            event_type=MouseEventType.SCROLL_UP,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
        assert content_control.mouse_handler(scroll_up) is None
        await _wait_until(lambda: application.render_counter > render_counter)

        assert 0 < highlighted_rows <= viewport_rows * 3
        assert state.transcript_viewport.projection_build_count == initial_builds == 1
        assert base_content_builds == initial_base_content_builds == 1
        pipe.send_text("\x04")
        await task


async def test_terminal_controller_projects_themed_local_command_results(
    monkeypatch: pytest.MonkeyPatch,
):
    projected: list[tuple[str, str]] = []

    async def fake_tui(
        state: TerminalViewState,
        **kwargs: Any,
    ) -> TerminalApplicationResult:
        del state
        for command, expected in (
            ("/status", "status"),
            ("/sources", "sources"),
            ("/catalog", "catalog"),
            ("/settings", "settings"),
        ):
            result = await kwargs["handle_command"](command, None)
            projected.append((result.presentation, result.output))
            assert result.presentation == expected
        return TerminalApplicationResult(None, "exit")

    monkeypatch.setattr(terminal_tui, "run_terminal_tui", fake_tui)

    class _FakeAgent:
        name = "atlas"
        model_route = SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    provider_id="openai:gpt-5.6-sol",
                    base_url=None,
                    secret_reference=None,
                ),
            )
        )

        async def list_sources(self) -> tuple[Any, ...]:
            return (
                SimpleNamespace(
                    active=True,
                    display_name="Warehouse",
                    adapter_id="sqlite",
                    id="source-one",
                ),
            )

        async def catalog_summary(self) -> Any:
            return SimpleNamespace(
                resource_count=1,
                relationship_count=0,
                latest_successful_sync_completed_at=None,
            )

        async def catalog_preview(self, *, limit: int) -> tuple[Any, ...]:
            assert limit == 12
            return (
                SimpleNamespace(
                    name="[orders]\x1b]0;unsafe\x07",
                    kind=SimpleNamespace(value="table"),
                ),
            )

    selected, conversation_id, action = await terminal._chat(
        cast(Any, _FakeAgent()),
        root=None,
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=None,
        validated=False,
        tui_input=object(),
        tui_output=object(),
        suspend_bridge=TerminalSuspendBridge(),
    )

    assert selected.name == "atlas"
    assert conversation_id is None
    assert action == "exit"
    assert [presentation for presentation, _output in projected] == [
        "status",
        "sources",
        "catalog",
        "settings",
    ]
    assert all(len(output) <= 16_384 for _presentation, output in projected)
    assert "\x1b" not in projected[2][1]
    assert "\x07" not in projected[2][1]


def test_terminal_controller_injects_one_named_bridge_at_every_agent_path():
    tree = ast.parse(inspect.getsource(terminal))
    construction_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "Agent"
        and node.func.attr in {"create", "open"}
    ]

    assert len(construction_calls) == 7
    for call in construction_calls:
        observer = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "observer"),
            None,
        )
        assert isinstance(observer, ast.Name)
        assert observer.id == "observer_bridge"


def test_observer_bridge_only_enqueues_until_the_tui_consumes_events():
    bridge = TerminalObserverBridge()
    state = TerminalViewState("atlas", "model", "source")
    started = _event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"})

    bridge(started)

    assert state.running is False
    assert state.active_run_id is None
    assert bridge.drain() == (started,)


def test_stream_fragments_coalesce_into_one_stable_partial_block():
    bridge = TerminalObserverBridge()
    state = TerminalViewState("atlas", "model", "source")
    state.apply_event(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
    before = state.transcript_render_generation
    for fragment in ("The ", "ordered ", "answer"):
        bridge(
            _event(
                AgentEventKind.MODEL_TEXT_DELTA,
                {"model_call_index": 1, "text": fragment},
            )
        )

    assert terminal_tui._project_pending_events(bridge, state) == 3

    assert state.transcript_render_generation == before + 1
    assert len(state.blocks) == 1
    partial = state.blocks[0]
    assert partial.kind == "assistant.partial"
    assert partial.text == "The ordered answer"
    partial_id = partial.presentation_id

    for _ in range(1_000):
        bridge(
            _event(
                AgentEventKind.MODEL_TEXT_DELTA,
                {"model_call_index": 1, "text": "."},
            )
        )
    before = state.transcript_render_generation
    assert terminal_tui._project_pending_events(bridge, state) == 1_000
    assert state.transcript_render_generation == before + 1
    assert len(state.blocks) == 1
    assert state.blocks[0].presentation_id == partial_id
    assert state.blocks[0].text.endswith("." * 1_000)
    assert terminal_tui._STREAM_REPAINT_INTERVAL_SECONDS >= 1 / 30


def test_streaming_follows_latest_and_preserves_reviewing_anchor():
    following = TerminalViewState("atlas", "model", "source")
    following.append_user("question")
    following.apply_model_text_delta("run-following", 1, "progressive text")

    assert (
        following.transcript_viewport.state
        is terminal_tui.TranscriptFollowState.FOLLOWING
    )
    assert following.blocks[-1].text == "progressive text"

    reviewing = TerminalViewState("atlas", "model", "source")
    reviewing.append_user("older question")
    reviewing.append_plain("assistant", "older answer")
    projection = reviewing.transcript_viewport.projection_for(
        reviewing.transcript_document,
        width=40,
    )
    reviewing.transcript_viewport.review_start(projection)
    anchor = reviewing.transcript_viewport.anchor
    assert anchor is not None
    reviewing.apply_model_text_delta("run-reviewing", 1, "first")
    reviewing.apply_model_text_delta("run-reviewing", 1, " second")

    assert (
        reviewing.transcript_viewport.state
        is terminal_tui.TranscriptFollowState.REVIEWING
    )
    assert reviewing.transcript_viewport.anchor == anchor
    assert reviewing.transcript_document.reconcile_anchor(anchor) == anchor
    assert reviewing.transcript_viewport.unseen_items == 1


def test_final_response_reconciles_partial_identity_without_duplication():
    state = TerminalViewState("atlas", "model", "source")
    state.apply_event(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
    state.apply_model_text_delta("run-live", 1, "draft projection")
    partial_id = state.blocks[0].presentation_id
    assert partial_id is not None
    state.apply_event(
        _event(
            AgentEventKind.MODEL_COMPLETED,
            {
                "model_call_index": 1,
                "has_text": True,
                "has_tool_calls": False,
                "provider_id": "mock:streaming",
                "duration_ms": 1,
                "input_tokens": 1,
                "output_tokens": 1,
            },
        )
    )
    state.apply_event(
        _event(
            AgentEventKind.RUN_COMPLETED,
            {"exit_kind": "completed", "reason": "completed"},
        )
    )

    state.apply_result(_result("Exact canonical final.", run_id="run-live"))

    assistant_blocks = [
        block for block in state.blocks if block.kind.startswith("assistant")
    ]
    assert len(assistant_blocks) == 1
    assert assistant_blocks[0].kind == "assistant"
    assert assistant_blocks[0].presentation_id == partial_id
    assert assistant_blocks[0].text == "Exact canonical final."
    assert state.transcript_document.text(partial_id) == "Exact canonical final."


def test_failed_stream_removes_partial_and_marks_it_unrecorded():
    state = TerminalViewState("atlas", "model", "source")
    state.apply_event(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
    state.apply_model_text_delta("run-live", 1, "unrecorded draft")
    state.apply_event(
        _event(
            AgentEventKind.RUN_COMPLETED,
            {"exit_kind": "failed", "reason": "provider_unavailable"},
        )
    )
    failed = SimpleNamespace(
        run_id="run-live",
        conversation_id="conversation-live",
        final_text=None,
        kind=SimpleNamespace(value="failed"),
        reason="provider_unavailable",
        steps=0,
        usage=SimpleNamespace(
            total_tokens=0,
            cost_estimate=CostEstimate.unavailable(),
        ),
        artifact_deliveries=(),
    )

    state.apply_result(failed)

    assert all(block.kind != "assistant.partial" for block in state.blocks)
    assert all("unrecorded draft" not in block.text for block in state.blocks)
    assert state.blocks[-1].text == "failed: provider_unavailable"
    assert state.notice == (
        "Partial assistant output was interrupted and was not recorded."
    )


def test_high_rate_fragments_do_not_delay_tool_and_approval_projection():
    bridge = TerminalObserverBridge()
    state = TerminalViewState("atlas", "model", "source")
    bridge(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
    for _ in range(2_000):
        bridge(
            _event(
                AgentEventKind.MODEL_TEXT_DELTA,
                {"model_call_index": 1, "text": "x"},
            )
        )
    bridge(
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "call-live",
                "tool_name": "skill_save",
                "capability_id": "skills.write",
            },
        )
    )
    bridge(
        _event(
            AgentEventKind.APPROVAL_REQUESTED,
            {
                "call_id": "call-live",
                "tool_name": "skill_save",
                "capability_id": "skills.write",
            },
        )
    )

    assert terminal_tui._project_pending_events(bridge, state) == 2_003

    assert state.tool_cards["call-live"].state == "approval"
    assert state.run_status == "approval"
    assert all(block.kind != "assistant.partial" for block in state.blocks)
    assert state.transcript_render_generation < 20


def test_initial_status_reports_zero_cost_before_the_first_run():
    state = TerminalViewState("atlas", "gpt-5.6-sol", "source")
    glyphs = terminal_tui._terminal_glyphs(
        terminal_tui.TerminalCapabilities("truecolor", True)
    )

    projected = terminal_tui._status_projection(
        state,
        width=120,
        mode="full",
        glyphs=glyphs,
    )

    assert state.steps == 0
    assert state.total_tokens == 0
    assert state.estimated_cost == "$0"
    assert projected.right == "0 steps · 0 tokens · $0"


def test_context_progress_uses_latest_conversation_request_and_input_capacity():
    state = TerminalViewState(
        "atlas",
        "gpt-5.6-sol",
        "source",
        conversation_id="conversation-live",
        context_capacity_tokens=1_000,
        conversation_context_tokens=340,
    )
    glyphs = terminal_tui._terminal_glyphs(
        terminal_tui.TerminalCapabilities("truecolor", True)
    )

    projected = terminal_tui._status_projection(
        state,
        width=120,
        mode="full",
        glyphs=glyphs,
    )

    assert "ctx [████░░░░░░] 34%" in projected.right
    assert "tokens" not in projected.right

    ascii_progress = terminal_tui._context_progress_text(
        state,
        glyphs=terminal_tui._terminal_glyphs(
            terminal_tui.TerminalCapabilities("16", False)
        ),
    )
    assert ascii_progress == "ctx [####------] 34%"


def test_context_progress_persists_per_conversation_and_not_per_run():
    state = TerminalViewState(
        "atlas",
        "model",
        "source",
        conversation_id="conversation-old",
        context_capacity_tokens=1_000,
        conversation_context_tokens=200,
    )

    state.apply_event(
        _event(
            AgentEventKind.RUN_STARTED,
            {"agent_id": "agent-live"},
            conversation_id="conversation-old",
        )
    )
    assert state.conversation_context_tokens == 200

    state.apply_event(
        _event(
            AgentEventKind.MODEL_COMPLETED,
            {
                "provider_id": "openai:model",
                "duration_ms": 19,
                "input_tokens": 500,
                "context_input_tokens": 250,
                "output_tokens": 3,
            },
            conversation_id="conversation-old",
        )
    )
    assert state.run_input_tokens == 500
    assert state.conversation_context_tokens == 250

    state.select_conversation(None)
    assert state.conversation_context_tokens is None
    state.select_conversation("conversation-old")
    assert state.conversation_context_tokens == 250
    state.select_conversation("conversation-unseen")
    assert state.conversation_context_tokens is None


def test_all_seven_observation_events_project_live_and_final_states():
    bridge = TerminalObserverBridge()
    state = TerminalViewState("atlas", "model", "source")

    bridge(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
    assert terminal_tui._project_pending_events(bridge, state) == 1
    assert state.running is True
    assert state.active_run_id == "run-live"
    assert state.run_status == "working"

    bridge(
        _event(
            AgentEventKind.MODEL_COMPLETED,
            {
                "provider_id": "openai:model",
                "duration_ms": 19,
                "input_tokens": 7,
                "output_tokens": 3,
            },
        )
    )
    terminal_tui._project_pending_events(bridge, state)
    assert state.model_duration_ms == 19
    assert state.total_tokens == 10

    bridge(
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "call-live",
                "tool_name": "data_query_sqlite",
                "capability_id": "data.sqlite.query",
            },
        )
    )
    terminal_tui._project_pending_events(bridge, state)
    card = state.tool_cards["call-live"]
    assert card.label == "Query SQLite"
    assert card.state == "running"

    bridge(
        _event(
            AgentEventKind.APPROVAL_REQUESTED,
            {
                "call_id": "call-live",
                "tool_name": "data_query_sqlite",
                "capability_id": "data.sqlite.query",
            },
        )
    )
    terminal_tui._project_pending_events(bridge, state)
    assert card.state == "approval"
    assert state.run_status == "approval"

    bridge(
        _event(
            AgentEventKind.APPROVAL_DECIDED,
            {"call_id": "call-live", "outcome": "approved"},
        )
    )
    terminal_tui._project_pending_events(bridge, state)
    assert card.state == "running"
    assert card.approval_outcome == "approved"

    bridge(
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": "call-live",
                "tool_name": "data_query_sqlite",
                "duration_ms": 42,
                "success": True,
                "error_code": None,
            },
        )
    )
    terminal_tui._project_pending_events(bridge, state)
    assert card.state == "succeeded"
    assert card.duration_ms == 42

    bridge(
        _event(
            AgentEventKind.RUN_COMPLETED,
            {
                "exit_kind": "completed",
                "reason": "completed",
                "steps": 2,
                "duration_ms": 75,
                "input_tokens": 11,
                "output_tokens": 5,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 16,
                "cost_status": "complete",
                "cost_amount_usd": "0.02",
                "cost_basis": None,
                "cost_rate_schedule_id": None,
                "cost_code": None,
                "cost_display": "$0.02 estimated",
            },
        )
    )
    terminal_tui._project_pending_events(bridge, state)

    assert state.running is False
    assert state.active_run_id is None
    assert state.run_status == "ready"
    assert state.run_duration_ms == 75
    assert state.steps == 2
    assert state.total_tokens == 16
    assert state.estimated_cost == "$0.02 estimated"


def test_concurrent_tool_completion_preserves_start_order_and_sibling_failure():
    bridge = TerminalObserverBridge()
    state = TerminalViewState("atlas", "model", "source")
    bridge(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
    for call_id, capability_id, tool_name in (
        ("call-first", "catalog.search", "catalog_search"),
        ("call-second", "data.sqlite.query", "data_query_sqlite"),
        ("call-third", "data.file.read", "data_read_file"),
    ):
        bridge(
            _event(
                AgentEventKind.TOOL_STARTED,
                {
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "capability_id": capability_id,
                },
            )
        )
    terminal_tui._project_pending_events(bridge, state)
    glyphs = terminal_tui._terminal_glyphs(
        terminal_tui.TerminalCapabilities("truecolor", True)
    )
    active_status = terminal_tui._status_projection(
        state,
        width=120,
        mode="full",
        glyphs=glyphs,
    )
    assert "calling Read data file (+2)" in active_status.left
    state.animation_frame += 1
    next_status = terminal_tui._status_projection(
        state,
        width=120,
        mode="full",
        glyphs=glyphs,
    )
    assert "calling Read data file (+2)" in next_status.left
    assert next_status.left != active_status.left
    assert state.toggle_tool_history() is False

    for call_id, success, error_code, duration_ms in (
        ("call-third", True, None, 8),
        ("call-second", False, "unknown_column", 13),
        ("call-first", True, None, 21),
    ):
        bridge(
            _event(
                AgentEventKind.TOOL_COMPLETED,
                {
                    "call_id": call_id,
                    "tool_name": "bounded-tool",
                    "success": success,
                    "error_code": error_code,
                    "duration_ms": duration_ms,
                },
            )
        )
    bridge(
        _event(
            AgentEventKind.RUN_COMPLETED,
            {"exit_kind": "completed", "reason": "completed"},
        )
    )
    terminal_tui._project_pending_events(bridge, state)

    assert [block.text for block in state.blocks if block.kind == "tool"] == [
        "call-first",
        "call-second",
        "call-third",
    ]
    assert state.tool_cards["call-first"].state == "succeeded"
    assert state.tool_cards["call-second"].state == "failed"
    assert state.tool_cards["call-second"].error_code == "unknown_column"
    assert state.tool_cards["call-third"].state == "succeeded"

    rendered = "".join(
        text
        for _, text in terminal_tui._render_transcript_fragments(
            terminal_tui._load_terminal_runtime(),
            state,
            width=96,
        )
    )
    assert "3 tool calls" in rendered
    assert "1 failed" in rendered
    assert "Search catalog" not in rendered
    assert "unknown_column" not in rendered

    assert state.toggle_tool_history() is True
    expanded = "".join(
        text
        for _, text in terminal_tui._render_transcript_fragments(
            terminal_tui._load_terminal_runtime(),
            state,
            width=96,
        )
    )
    assert expanded.index("Search catalog") < expanded.index("Query SQLite")
    assert expanded.index("Query SQLite") < expanded.index("Read data file")
    assert "unknown_column" in expanded
    assert "13ms" in expanded


def test_tool_event_fields_are_bounded_and_sanitized_before_rendering():
    state = TerminalViewState("atlas", "model", "source")
    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "call-unsafe",
                "tool_name": "unsafe\x1b]0;title\x07\u202e[bold]",
            },
        )
    )
    state.apply_event(
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": "call-unsafe",
                "tool_name": "unsafe\x1b[2J",
                "duration_ms": 5,
                "success": False,
                "error_code": "bad\x1b[2J\u202e[red]",
            },
        )
    )

    card = state.tool_cards["call-unsafe"]
    rendered = "".join(
        text for _, text in terminal_tui._render_tool_card_fragments(card, width=80)
    )
    assert "\x1b" not in card.label
    assert "\u202e" not in card.label
    assert "\x1b" not in rendered
    assert "\u202e" not in rendered
    assert "?]0;title?" in rendered
    assert "bad?[2J?[red]" in rendered


def test_projection_and_card_rendering_fail_closed_without_touching_run_result(
    monkeypatch: pytest.MonkeyPatch,
):
    bridge = TerminalObserverBridge()
    state = TerminalViewState("atlas", "model", "source")
    result = _result("answer remains authoritative")
    bridge(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))

    original_apply_event = TerminalViewState.apply_event

    def broken_projection(self: TerminalViewState, event: AgentEvent) -> None:
        del self, event
        raise RuntimeError("projection failed")

    monkeypatch.setattr(TerminalViewState, "apply_event", broken_projection)
    assert terminal_tui._project_pending_events(bridge, state) == 0
    assert result.final_text == "answer remains authoritative"
    state.apply_result(result)
    assert state.blocks[-1].text == "answer remains authoritative"

    monkeypatch.setattr(TerminalViewState, "apply_event", original_apply_event)
    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {"call_id": "call-live", "tool_name": "safe"},
        )
    )

    def broken_renderer(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("render failed")

    monkeypatch.setattr(
        tui_transcript_view,
        "_render_tool_card_fragments",
        broken_renderer,
    )
    card = state.tool_cards["call-live"]
    card.state = "succeeded"
    state.tool_history_run_id = card.run_id
    rendered = "".join(
        text
        for _, text in terminal_tui._render_transcript_fragments(
            terminal_tui._load_terminal_runtime(),
            state,
            width=80,
        )
    )
    assert "Tool status unavailable" in rendered
    assert result.final_text == "answer remains authoritative"


def test_loop_result_settles_live_state_when_a_completion_event_is_unavailable():
    state = TerminalViewState("atlas", "model", "source")
    state.apply_event(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {"call_id": "call-live", "tool_name": "bounded-tool"},
        )
    )
    result = SimpleNamespace(
        run_id="run-live",
        conversation_id="conversation-live",
        final_text="authoritative answer",
        kind=SimpleNamespace(value="completed"),
        reason="completed",
        steps=2,
        usage=SimpleNamespace(
            total_tokens=12,
            cost_estimate=CostEstimate.complete(Decimal("0.01")),
        ),
    )

    state.apply_result(result)

    assert state.active_run_id is None
    assert state.run_status == "ready"
    assert state.tool_cards["call-live"].state == "failed"
    assert state.tool_cards["call-live"].error_code == "observation_incomplete"
    assert state.blocks[-1].text == "authoritative answer"


@pytest.mark.parametrize(
    ("estimate", "rendered"),
    (
        (
            CostEstimate.complete(
                Decimal("0.02"),
                basis=CostBasis.PUBLIC_LIST,
                rate_schedule_id="public:test",
            ),
            "$0.02 estimated at public list rates",
        ),
        (
            CostEstimate.partial(
                Decimal("0.01"),
                code="unpriced_attempt",
            ),
            "≥$0.01 estimated; some attempts were unpriced",
        ),
        (
            CostEstimate.unavailable(),
            "cost unavailable",
        ),
        (
            CostEstimate.complete(Decimal("0")),
            "$0 explicit estimate",
        ),
    ),
)
def test_terminal_view_renders_every_cost_state(estimate, rendered):
    state = TerminalViewState("atlas", "model", "source")

    state.apply_result(_result("answer", cost_estimate=estimate))

    assert state.estimated_cost == rendered


async def test_controller_loads_only_the_exact_completed_transcript_for_hydration(
    monkeypatch: pytest.MonkeyPatch,
):
    call = ToolCall(
        id="call-exact",
        name="data_query_sqlite",
        arguments={"source_id": "source-one", "sql": "SELECT 1 AS value"},
    )
    recorded = _tool_transcript(
        (call,),
        (
            ToolResultBlock(
                call_id=call.id,
                output={
                    "kind": "data.sqlite.query_result",
                    "data": {
                        "columns": ["value"],
                        "rows": [{"value": 1}],
                        "total_rows": 1,
                    },
                },
            ),
        ),
    )
    calls = {"sources": 0, "runs": 0, "transcripts": 0}
    hydrated_states: list[TerminalViewState] = []

    class _FakeAgent:
        name = "atlas"
        model_route = SimpleNamespace(
            candidates=(SimpleNamespace(provider_id="openai:gpt-5.6-sol"),)
        )

        async def list_sources(self) -> tuple[Any, ...]:
            calls["sources"] += 1
            return (SimpleNamespace(active=True, display_name="Warehouse"),)

        async def catalog_summary(self) -> Any:
            return SimpleNamespace(resource_count=1, relationship_count=0)

        async def run(
            self,
            message: str,
            *,
            conversation_id: str | None,
        ) -> Any:
            calls["runs"] += 1
            assert message == "question"
            assert conversation_id is None
            return _result("answer from LoopExit")

        async def transcript(self, run_id: str) -> Transcript:
            calls["transcripts"] += 1
            assert run_id == "run-one"
            return recorded

        async def query_source(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError((args, kwargs))

    async def fake_tui(
        state: TerminalViewState,
        **kwargs: Any,
    ) -> TerminalApplicationResult:
        result = await kwargs["run_message"]("question", None)
        transcript = await kwargs["load_transcript"](result.run_id)
        state.hydrate_transcript(transcript, run_id=result.run_id)
        state.apply_result(result)
        hydrated_states.append(state)
        return TerminalApplicationResult(result.conversation_id, "exit")

    monkeypatch.setattr(terminal_tui, "run_terminal_tui", fake_tui)
    agent = cast(Any, _FakeAgent())
    selected, conversation_id, action = await terminal._chat(
        agent,
        root=None,
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
        hidden_input=lambda prompt: "",
        keychain=None,
        model_validator=None,
        approval_handler=None,
        conversation_id=None,
        validated=False,
        tui_input=object(),
        tui_output=object(),
        suspend_bridge=TerminalSuspendBridge(),
    )

    assert selected is agent
    assert conversation_id == "conversation-one"
    assert action == "exit"
    assert calls == {"sources": 1, "runs": 1, "transcripts": 1}
    assert hydrated_states[0].tool_cards["call-exact"].details is not None
    assert (
        hydrated_states[0].tool_cards["call-exact"].details.code == "SELECT 1 AS value"
    )


def test_hydration_matches_results_by_call_id_and_restores_transcript_order():
    first = ToolCall(
        id="call-first",
        name="catalog_search",
        arguments={"query": "orders"},
    )
    second = ToolCall(
        id="call-second",
        name="data_query_sqlite",
        arguments={"source_id": "source-one", "sql": "SELECT * FROM orders"},
    )
    transcript = _tool_transcript(
        (first, second),
        (
            ToolResultBlock(
                call_id=second.id,
                is_error=True,
                output={
                    "error": {
                        "code": "second_error",
                        "message": "second result",
                        "details": {"marker": "matched-second"},
                    }
                },
            ),
            ToolResultBlock(
                call_id=first.id,
                output={
                    "kind": "catalog.search_result",
                    "data": {"marker": "matched-first"},
                },
            ),
        ),
    )
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("question")
    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": second.id,
                "tool_name": second.name,
                "capability_id": "data.sqlite.query",
            },
            run_id="run-one",
        )
    )
    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": "call-phantom",
                "tool_name": "phantom",
            },
            run_id="run-one",
        )
    )
    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": first.id,
                "tool_name": first.name,
                "capability_id": "catalog.search",
            },
            run_id="run-one",
        )
    )
    prior_ids = {
        block.text: block.presentation_id
        for block in state.blocks
        if block.kind == "tool"
    }
    phantom_id = prior_ids["call-phantom"]
    assert phantom_id is not None
    phantom_anchor = state.transcript_document.make_anchor(
        state.transcript_document.position(phantom_id, 0)
    )

    state.hydrate_transcript(transcript, run_id="run-one")

    assert [block.text for block in state.blocks if block.kind == "tool"] == [
        first.id,
        second.id,
    ]
    assert [
        block.presentation_id for block in state.blocks if block.kind == "tool"
    ] == [prior_ids[first.id], prior_ids[second.id]]
    assert state.transcript_document.reconcile_anchor(phantom_anchor) is None
    assert "call-phantom" not in state.tool_cards
    first_details = state.tool_cards[first.id].details
    second_details = state.tool_cards[second.id].details
    assert first_details is not None
    assert second_details is not None
    assert "matched-first" in cast(str, first_details.result_text)
    assert "matched-second" in cast(str, second_details.result_text)
    assert state.tool_cards[first.id].state == "succeeded"
    assert state.tool_cards[first.id].expanded is False
    assert state.tool_cards[second.id].state == "failed"
    assert state.tool_cards[second.id].error_code == "second_error"
    assert state.tool_cards[second.id].expanded is True


def test_hydrated_sql_json_and_cells_cannot_escape_the_renderer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    call = ToolCall(
        id="call-unsafe-detail",
        name="data_query_sqlite",
        arguments={
            "source_id": "[bold]source[/bold]\x1b]0;source\x07",
            "sql": "SELECT '[cyan]data[/cyan]'\x1b[2J;\u202e",
        },
    )
    transcript = _tool_transcript(
        (call,),
        (
            ToolResultBlock(
                call_id=call.id,
                output={
                    "kind": "data.sqlite.query_result",
                    "data": {
                        "columns": ["na\x1b[2Jme"],
                        "rows": [
                            {
                                "na\x1b[2Jme": (
                                    "[red]value[/red]\x1b]52;c;clipboard\x07\u202e"
                                )
                            }
                        ],
                        "total_rows": 1,
                    },
                },
            ),
        ),
    )
    state = TerminalViewState("atlas", "model", "source")
    state.hydrate_transcript(transcript, run_id="run-one")
    card = state.tool_cards[call.id]
    card.expanded = True

    rendered = "".join(
        text
        for _style, text in terminal_tui._render_tool_card_fragments(
            card,
            width=120,
        )
    )

    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\u202e" not in rendered
    assert "[cyan]data[/cyan]" in rendered
    assert "[bold]source[/bold]" in rendered
    assert "[red]value[/red]" in rendered
    assert "?]52;c;clipboard?" in rendered


def test_result_previews_obey_every_stage_three_rendering_bound():
    columns = [f"c{index:02d}" for index in range(25)]
    rows = [
        {
            column: (
                f"row-{row_index:02d}"
                if column == "c00"
                else ("界" * 300 if row_index == 0 and column == "c01" else column)
            )
            for column in columns
        }
        for row_index in range(60)
    ]
    sql = "\n".join(f"SELECT {index} AS value;" for index in range(100))
    query = ToolCall(
        id="call-bounds",
        name="data_query_sqlite",
        arguments={"source_id": "source-one", "sql": sql},
    )
    text = ToolCall(
        id="call-text",
        name="catalog_search",
        arguments={"query": "large detail"},
    )
    transcript = _tool_transcript(
        (query, text),
        (
            ToolResultBlock(
                call_id=query.id,
                output={
                    "kind": "data.sqlite.query_result",
                    "data": {
                        "columns": columns,
                        "rows": rows,
                        "total_rows": 75,
                    },
                },
            ),
            ToolResultBlock(
                call_id=text.id,
                output={
                    "kind": "catalog.search_result",
                    "data": {"payload": "x" * 30_000},
                },
            ),
        ),
    )
    state = TerminalViewState("atlas", "model", "source")
    state.hydrate_transcript(transcript, run_id="run-one")
    query_card = state.tool_cards[query.id]
    query_details = query_card.details
    text_details = state.tool_cards[text.id].details
    assert query_details is not None
    assert text_details is not None
    assert "\n" not in query_details.summary
    assert query_details.table is not None
    assert len(query_details.table.rows) == 50
    assert len(query_details.table.columns) == 20
    assert all(
        terminal_tui._display_width(cell) <= 240
        for row in query_details.table.rows
        for cell in row
    )
    assert query_details.table.cells_truncated is True
    assert text_details.result_text is not None
    assert (
        sum(
            len(value.encode("utf-8"))
            for value in (
                text_details.arguments_text,
                text_details.error_message,
                text_details.result_text,
            )
            if value is not None
        )
        <= 16 * 1_024
    )
    assert "at 16 KiB" in text_details.result_text

    collapsed = "".join(
        text
        for _style, text in terminal_tui._render_tool_card_fragments(
            query_card,
            width=240,
        )
    )
    assert "row-09" in collapsed
    assert "row-10" not in collapsed
    assert "c11" in collapsed
    assert "c12" not in collapsed
    assert "50 more rows in the recorded tool result" in collapsed
    assert "13 more columns in the recorded tool result" in collapsed
    assert "15 additional rows were not recorded" in collapsed

    query_card.expanded = True
    expanded = "".join(
        text
        for _style, text in terminal_tui._render_tool_card_fragments(
            query_card,
            width=240,
        )
    )
    assert "row-49" in expanded
    assert "row-50" not in expanded
    assert "c19" in expanded
    assert "c20" not in expanded
    assert "10 more rows in the recorded tool result" in expanded
    assert "5 more columns in the recorded tool result" in expanded
    assert "code truncated at 80 visible lines" in expanded
    assert "cells truncated to 240 display characters" in expanded

    runtime = terminal_tui._load_terminal_runtime()
    syntax = runtime["Syntax"](
        cast(str, query_details.code),
        "sql",
        theme="ansi_dark",
        background_color="default",
        line_numbers=False,
        word_wrap=True,
    )
    bounded_code = terminal_tui._card_rich_lines(
        runtime,
        syntax,
        width=234,
        border_style="",
        maximum_lines=80,
        truncation_line="… code truncated at 80 visible lines",
    )
    assert sum(fragment.count("\n") for _style, fragment in bounded_code) == 80


def test_tool_history_is_hidden_by_default_and_toggles_per_process():
    failed = ToolCall(id="call-failed", name="failed_tool", arguments={"value": 1})
    succeeded = ToolCall(
        id="call-succeeded",
        name="successful_tool",
        arguments={"value": 2},
    )
    transcript = _tool_transcript(
        (failed, succeeded),
        (
            ToolResultBlock(
                call_id=failed.id,
                is_error=True,
                output={
                    "error": {
                        "code": "failed",
                        "message": "recorded failure",
                        "details": {},
                    }
                },
            ),
            ToolResultBlock(
                call_id=succeeded.id,
                output={"kind": "success", "data": {"value": 2}},
            ),
        ),
    )
    first_state = TerminalViewState("atlas", "model", "source")
    first_state.hydrate_transcript(transcript, run_id="run-one")
    assert first_state.tool_cards[failed.id].expanded is True
    assert first_state.tool_cards[succeeded.id].expanded is False

    collapsed = "".join(
        text
        for _style, text in terminal_tui._render_transcript_fragments(
            terminal_tui._load_terminal_runtime(),
            first_state,
            width=96,
        )
    )
    assert "2 tool calls" in collapsed
    assert "1 failed" in collapsed
    assert "recorded failure" not in collapsed

    assert first_state.toggle_tool_history() is True
    assert first_state.tool_history_run_id == "run-one"
    assert first_state.tool_cards[failed.id].expanded is True
    assert first_state.tool_cards[succeeded.id].expanded is True
    expanded = "".join(
        text
        for _style, text in terminal_tui._render_transcript_fragments(
            terminal_tui._load_terminal_runtime(),
            first_state,
            width=96,
        )
    )
    assert expanded.index("failed_tool") < expanded.index("successful_tool")
    assert "recorded failure" in expanded

    assert first_state.toggle_tool_history() is True
    assert first_state.tool_history_run_id is None
    assert (
        first_state.transcript_viewport.state
        is terminal_tui.TranscriptFollowState.FOLLOWING
    )

    second_state = TerminalViewState("atlas", "model", "source")
    second_state.hydrate_transcript(transcript, run_id="run-one")
    assert second_state.tool_cards[failed.id].expanded is True
    assert second_state.tool_cards[succeeded.id].expanded is False
    assert second_state.tool_history_run_id is None
    assert transcript.messages[1].tool_calls == (failed, succeeded)


async def test_ctrl_o_opens_and_hides_the_latest_completed_tool_run():
    output = _RecordingOutput()
    call = ToolCall(
        id="call-keyboard-history",
        name="data_query_sqlite",
        arguments={"source_id": "source-one", "sql": "SELECT 1 AS value"},
    )
    transcript = _tool_transcript(
        (call,),
        (
            ToolResultBlock(
                call_id=call.id,
                output={
                    "kind": "data.sqlite.query_result",
                    "data": {
                        "columns": ["value"],
                        "rows": [{"value": 1}],
                        "total_rows": 1,
                    },
                },
            ),
        ),
        run_id="run-keyboard-history",
    )
    state = TerminalViewState("atlas", "model", "source")
    state.hydrate_transcript(transcript, run_id="run-keyboard-history")

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=no_message)
        await _wait_until(
            lambda: any(
                "Ctrl-O view results" in block.text
                for block in state.transcript_document.blocks
            )
        )

        pipe.send_text("\x0f")
        await _wait_until(
            lambda: state.tool_history_run_id == "run-keyboard-history"
            and any(
                "SELECT 1 AS value" in block.text
                for block in state.transcript_document.blocks
            )
        )
        assert state.notice == "Tool results shown; Ctrl-O hides them."

        pipe.send_text("\x0f")
        await _wait_until(
            lambda: state.tool_history_run_id is None
            and any(
                "Ctrl-O view results" in block.text
                for block in state.transcript_document.blocks
            )
        )
        assert state.notice == "Tool results hidden."

        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"


async def test_live_tool_status_transitions_to_summary_before_results_open():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    observer = TerminalObserverBridge()
    finish_tool = asyncio.Event()
    call = ToolCall(
        id="call-live-status",
        name="data_query_sqlite",
        arguments={"source_id": "source-one", "sql": "SELECT 1 AS value"},
    )
    transcript = _tool_transcript(
        (call,),
        (
            ToolResultBlock(
                call_id=call.id,
                output={
                    "kind": "data.sqlite.query_result",
                    "data": {
                        "columns": ["value"],
                        "rows": [{"value": 1}],
                        "total_rows": 1,
                    },
                },
            ),
        ),
        run_id="run-live",
    )

    async def run_message(message: str, conversation_id: str | None) -> Any:
        assert message == "question"
        assert conversation_id is None
        observer(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
        observer(
            _event(
                AgentEventKind.TOOL_STARTED,
                {
                    "call_id": call.id,
                    "tool_name": call.name,
                    "capability_id": "data.sqlite.query",
                },
            )
        )
        await finish_tool.wait()
        observer(
            _event(
                AgentEventKind.TOOL_COMPLETED,
                {
                    "call_id": call.id,
                    "tool_name": call.name,
                    "success": True,
                    "duration_ms": 8,
                },
            )
        )
        observer(
            _event(
                AgentEventKind.RUN_COMPLETED,
                {"exit_kind": "completed", "reason": "completed"},
            )
        )
        return _result("Final answer.", run_id="run-live")

    async def load_transcript(run_id: str) -> Transcript:
        assert run_id == "run-live"
        return transcript

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            load_transcript=load_transcript,
            observer_bridge=observer,
        )
        pipe.send_text("question\r")
        await _wait_until(lambda: state.active_tool_activity() is not None)
        await asyncio.sleep(0.1)
        assert "call Query SQLite" in output.text
        tool_block = next(block for block in state.blocks if block.kind == "tool")
        assert tool_block.presentation_id is not None
        assert state.transcript_document.text(tool_block.presentation_id) == ""

        finish_tool.set()
        await _wait_until(
            lambda: not state.running
            and "Final answer." in output.text
            and "Ctrl-O view results" in output.text
        )
        assert "Recorded result" not in output.text

        pipe.send_text("\x0f")
        await _wait_until(
            lambda: state.tool_history_run_id == "run-live"
            and "Recorded result" in output.text
        )
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"


def test_postgresql_connection_failure_has_distinct_tool_card_heading():
    call = ToolCall(
        id="call-postgresql-offline",
        name="data_query_postgresql",
        arguments={
            "source_id": "source-postgresql",
            "sql": "SELECT amount FROM public.orders",
        },
    )
    result = ToolResultBlock(
        call_id=call.id,
        is_error=True,
        output={
            "error": {
                "code": "postgresql_connect_failed",
                "message": "PostgreSQL source could not be opened.",
            }
        },
    )
    details = terminal_tui._project_tool_details(call, result)
    card = terminal_tui.ToolCardState(
        run_id="run-postgresql-offline",
        call_id=call.id,
        capability_id="data.postgresql.query",
        label="Query PostgreSQL",
        state="failed",
        error_code="postgresql_connect_failed",
        details=details,
    )

    collapsed = "".join(
        text
        for _style, text in terminal_tui._render_tool_card_fragments(
            card,
            width=96,
        )
    )
    card.expanded = True
    expanded = "".join(
        text
        for _style, text in terminal_tui._render_tool_card_fragments(
            card,
            width=96,
        )
    )

    assert details.summary.startswith("Connection unavailable · ")
    assert "Connection unavailable" in collapsed
    assert "Connection unavailable" in expanded
    assert "PostgreSQL source could not be opened." in expanded
    assert "database is running and reachable" in expanded
    assert "PostgreSQLQueryError" not in collapsed + expanded


def test_expanded_tool_and_error_details_inherit_readable_application_text():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("truecolor", True)
    cards = (
        terminal_tui.ToolCardState(
            run_id="run-success",
            call_id="call-success",
            capability_id="data.sqlite.query",
            label="Query SQLite",
            state="succeeded",
            details=terminal_tui.ToolCardDetails(
                summary="One row returned",
                code="SELECT value FROM records",
                code_language="sql",
                arguments_text='{"sql":"SELECT value FROM records"}',
                result_text='{"value":"visible"}',
            ),
            expanded=True,
        ),
        terminal_tui.ToolCardState(
            run_id="run-failure",
            call_id="call-failure",
            capability_id="data.postgresql.query",
            label="Query PostgreSQL",
            state="failed",
            error_code="postgresql_connect_failed",
            details=terminal_tui.ToolCardDetails(
                summary="Connection unavailable",
                arguments_text='{"source_id":"source-postgresql"}',
                error_message="PostgreSQL source could not be opened.",
            ),
            expanded=True,
        ),
    )

    for card in cards:
        border_style = (
            "class:tui.tool.success"
            if card.state == "succeeded"
            else "class:tui.tool.failure"
        )
        fragments = terminal_tui._render_tool_card_fragments(
            card,
            width=96,
            runtime=runtime,
            capabilities=capabilities,
        )
        body_styles = [
            style for style, text in fragments if text.strip() and style != border_style
        ]

        assert body_styles
        assert all(style.split()[0] == "class:tui.tool.text" for style in body_styles)
        assert all("#000000" not in style for style in body_styles)
        assert all("bg:" not in style for style in body_styles)


def test_live_tool_status_then_history_toggle_reconcile_the_rendered_document():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    call = ToolCall(
        id="call-semantic",
        name="data_query_sqlite",
        arguments={"source_id": "source-one", "sql": "SELECT 1 AS value"},
    )
    transcript = _tool_transcript(
        (call,),
        (
            ToolResultBlock(
                call_id=call.id,
                output={
                    "kind": "data.sqlite.query_result",
                    "data": {
                        "columns": ["value"],
                        "rows": [{"value": 1}],
                        "total_rows": 1,
                    },
                },
            ),
        ),
    )
    state.apply_event(
        _event(
            AgentEventKind.RUN_STARTED,
            {"agent_id": "agent-live"},
            run_id="run-one",
        )
    )
    state.apply_event(
        _event(
            AgentEventKind.TOOL_STARTED,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "capability_id": "data.sqlite.query",
            },
            run_id="run-one",
        )
    )
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=96,
        capabilities=capabilities,
    )
    block = next(block for block in state.blocks if block.kind == "tool")
    block_id = block.presentation_id
    assert block_id is not None
    assert state.transcript_document.text(block_id) == ""
    status = terminal_tui._status_projection(
        state,
        width=120,
        mode="full",
        glyphs=terminal_tui._terminal_glyphs(capabilities),
    )
    assert "calling Query SQLite" in status.left

    state.apply_event(
        _event(
            AgentEventKind.TOOL_COMPLETED,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "duration_ms": 8,
                "success": True,
            },
            run_id="run-one",
        )
    )
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=96,
        capabilities=capabilities,
    )
    assert block.presentation_id == block_id
    assert state.transcript_document.text(block_id) == ""

    state.apply_event(
        _event(
            AgentEventKind.RUN_COMPLETED,
            {"exit_kind": "completed", "reason": "completed"},
            run_id="run-one",
        )
    )
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=96,
        capabilities=capabilities,
    )
    summary_text = state.transcript_document.text(block_id)
    assert summary_text.startswith("1 tool call")
    summary_start = summary_text.index("Ctrl-O view results")
    summary_range = state.transcript_document.normalize_range(
        state.transcript_document.position(block_id, summary_start),
        state.transcript_document.position(
            block_id,
            summary_start + len("Ctrl-O view results"),
        ),
    )

    state.hydrate_transcript(transcript, run_id="run-one")
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=96,
        capabilities=capabilities,
    )
    hydrated_block = next(block for block in state.blocks if block.kind == "tool")
    assert hydrated_block.presentation_id == block_id
    assert state.transcript_document.text(block_id).startswith("1 tool call")

    assert state.toggle_tool_history() is True
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=96,
        capabilities=capabilities,
    )

    expanded_text = state.transcript_document.text(block_id)
    assert expanded_text.startswith("Query SQLite\n")
    assert "SELECT 1 AS value" in expanded_text
    assert state.transcript_document.reconcile_range(summary_range) is None

    assert state.toggle_tool_history() is True
    terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=96,
        capabilities=capabilities,
    )
    assert state.transcript_document.text(block_id).startswith("1 tool call")
    assert (
        state.transcript_viewport.state is terminal_tui.TranscriptFollowState.FOLLOWING
    )


@pytest.mark.parametrize("failure", ("load", "projection"))
async def test_hydration_failures_do_not_change_the_authoritative_loop_exit(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    call = ToolCall(id="call-one", name="tool", arguments={"value": 1})
    transcript = _tool_transcript(
        (call,),
        (
            ToolResultBlock(
                call_id=call.id,
                output={"kind": "success", "data": {"value": 1}},
            ),
        ),
    )
    loaded: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        assert message == "question"
        assert conversation_id is None
        return _result("authoritative answer")

    async def load_transcript(run_id: str) -> Transcript:
        loaded.append(run_id)
        if failure == "load":
            raise RuntimeError("load failed")
        return transcript

    if failure == "projection":
        monkeypatch.setattr(
            TerminalViewState,
            "hydrate_transcript",
            lambda self, loaded_transcript, *, run_id: (_ for _ in ()).throw(
                RuntimeError("projection failed")
            ),
        )

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            load_transcript=load_transcript,
        )
        pipe.send_text("question\r")
        await _wait_until(
            lambda: any(block.kind == "assistant" for block in state.blocks)
        )
        pipe.send_text("\x04")
        result = await task

    assert result == TerminalApplicationResult("conversation-one", "exit")
    assert loaded == ["run-one"]
    assert state.blocks[-1].kind == "assistant"
    assert state.blocks[-1].text == "authoritative answer"
    assert state.run_status == "ready"
    assert state.notice == "Run completed; recorded tool details are unavailable."


async def test_text_turn_multiline_composer_and_green_shell_render():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "gpt-5.6-sol", "Warehouse")
    submitted: list[tuple[str, str | None]] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        submitted.append((message, conversation_id))
        return _result("EMEA leads.\n\nRevenue is higher.")

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pipe.send_text("first line\x0asecond line\r")
        await _wait_until(lambda: len(state.blocks) == 3)
        pipe.send_text("\x04")
        result = await task

    assert result == TerminalApplicationResult("conversation-one", "exit")
    assert submitted == [("first line\nsecond line", None)]
    assert [(block.kind, block.text) for block in state.blocks] == [
        ("user", "first line\nsecond line"),
        ("metadata", "Conversation  conversation-one"),
        ("assistant", "EMEA leads.\n\nRevenue is higher."),
    ]
    assert "DAITA" in output.text
    assert "atlas" in output.text
    rendered = "".join(
        text
        for _, text in terminal_tui._render_transcript_fragments(
            terminal_tui._load_terminal_runtime(),
            state,
            width=96,
        )
    )
    assert "first line" in rendered
    assert "second line" in rendered
    assert "EMEA" in rendered
    assert "Revenue is higher." in rendered
    assert "ready" in output.text
    assert output.show_count >= 1
    assert output.alternate_enter_count == 1
    assert output.alternate_exit_count == 1


def test_conversation_display_preserves_complete_messages_and_wraps_user_text():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    user_message = "USER_START_" + ("x" * 20_000) + "_USER_END"
    answer = "ASSISTANT_START\n" + ("y" * 20_000) + "\nASSISTANT_END"
    state = TerminalViewState("atlas", "model", "source")

    state.append_user(user_message)
    state.apply_result(_result(answer))

    assert state.blocks[0].text == user_message
    assert state.blocks[-1].text == answer

    user_rendered = "".join(
        text
        for _style, text in terminal_tui._render_user_message_fragments(
            runtime,
            state.blocks[0].text,
            width=40,
            capabilities=capabilities,
        )
    )
    user_lines = user_rendered.splitlines()
    assert user_lines
    assert all(terminal_tui._display_width(line) <= 40 for line in user_lines)
    assert "".join(line[1:] for line in user_lines) == user_message

    transcript = "".join(
        text
        for _style, text in terminal_tui._render_transcript_fragments(
            runtime,
            state,
            width=40,
            capabilities=capabilities,
        )
    )
    assert "USER_START_" in transcript
    assert "_USER_END" in transcript
    assert "ASSISTANT_START" in transcript
    assert "ASSISTANT_END" in transcript


def test_rich_markdown_rows_keep_exact_semantic_content_across_rewrap():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    state.append_plain(
        "assistant",
        """# Quarterly results

| Region | Revenue | Margin |
| --- | ---: | ---: |
| North America with a very long label | 123456 | 42% |
| Europe | 98765 | 38% |

- A deliberately long bullet that wraps differently at narrow widths.
""",
    )

    narrow_maps: list[Any] = []
    narrow_fragments = terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=40,
        capabilities=capabilities,
        rendered_transcript_maps=narrow_maps,
    )
    narrow_lines = "".join(text for _style, text in narrow_fragments).split("\n")
    europe_row = next(row for row, line in enumerate(narrow_lines) if "Europe" in line)
    position = narrow_maps[0].position_for_row(europe_row)
    assert position is not None
    assert state.transcript_document.text(position.block_id)[
        position.offset :
    ].startswith("Europe")
    anchor = state.transcript_document.make_anchor(position)

    wide_maps: list[Any] = []
    wide_fragments = terminal_tui._render_transcript_fragments(
        runtime,
        state,
        width=100,
        capabilities=capabilities,
        rendered_transcript_maps=wide_maps,
    )
    wide_lines = "".join(text for _style, text in wide_fragments).split("\n")
    wide_row = wide_maps[0].row_for_anchor(state.transcript_document, anchor)

    assert wide_row is not None
    assert "Europe" in wide_lines[wide_row]


def test_transcript_renderer_exercises_stable_semantic_projection_without_rewrap_churn():
    runtime = terminal_tui._load_terminal_runtime()
    capabilities = terminal_tui.TerminalCapabilities("none", True)
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("café 界\tpath/to/file\x1b[2J")
    block = state.blocks[0]
    block_id = block.presentation_id

    narrow = "".join(
        text
        for _style, text in terminal_tui._render_transcript_fragments(
            runtime,
            state,
            width=20,
            capabilities=capabilities,
        )
    )
    narrow_projection = state.transcript_projection
    narrow_revision = state.transcript_document.blocks[0].revision

    wide = "".join(
        text
        for _style, text in terminal_tui._render_transcript_fragments(
            runtime,
            state,
            width=80,
            capabilities=capabilities,
        )
    )

    assert block_id is not None
    assert block.presentation_id == block_id
    assert state.transcript_document.presentation_ids == (block_id,)
    assert state.transcript_document.text(block_id) == "café 界\tpath/to/file?[2J"
    assert state.transcript_document.blocks[0].revision == narrow_revision == 0
    assert narrow_projection is not None
    assert state.transcript_projection is not None
    assert narrow_projection.row_count > state.transcript_projection.row_count
    assert "\x1b" not in narrow + wide
    assert "?[2J" in narrow + wide


def test_green_identity_focus_theme_uses_semantic_styles():
    rules = terminal_tui._semantic_style_rules(
        terminal_tui.TerminalCapabilities("truecolor", True)
    )

    assert rules["tui.identity"] == "bold #22c55e"
    assert rules["tui.prompt"] == "bold #4ade80"
    assert rules["frame.border"] == "#4ade80"
    assert rules["tui.status.running"] == "bold #15803d"


@pytest.mark.parametrize(
    ("width", "mode", "collapsed_columns", "expanded_columns", "bordered"),
    (
        (69, "narrow", 4, 6, False),
        (70, "compact", 8, 12, True),
        (99, "compact", 8, 12, True),
        (100, "full", 12, 20, True),
        (140, "full", 12, 20, True),
    ),
)
def test_stage_five_width_modes_and_preview_bounds(
    width: int,
    mode: str,
    collapsed_columns: int,
    expanded_columns: int,
    bordered: bool,
):
    projection = terminal_tui._responsive_projection(width, 30)

    assert projection.mode == mode
    assert projection.collapsed_preview_columns == collapsed_columns
    assert projection.expanded_preview_columns == expanded_columns
    assert projection.bordered_cards is bordered
    assert projection.transcript_rows >= 1


def test_layout_reserves_one_transcript_row_or_switches_to_resize_message():
    idle = terminal_tui._responsive_projection(100, 8)
    approving = terminal_tui._responsive_projection(100, 15, approving=True)
    idle_too_short = terminal_tui._responsive_projection(100, 7)
    approval_too_short = terminal_tui._responsive_projection(
        100,
        14,
        approving=True,
    )

    assert idle.usable is True
    assert idle.transcript_rows == 1
    assert approving.usable is True
    assert approving.transcript_rows == 1
    assert idle_too_short.usable is False
    assert approval_too_short.usable is False


def test_terminal_size_polling_is_limited_to_platforms_without_sigwinch():
    assert (
        terminal_tui._terminal_size_polling_interval(
            platform="darwin",
            main_thread=True,
        )
        is None
    )
    assert (
        terminal_tui._terminal_size_polling_interval(
            platform="linux",
            main_thread=True,
        )
        is None
    )
    assert (
        terminal_tui._terminal_size_polling_interval(
            platform="win32",
            main_thread=True,
        )
        == 0.5
    )
    assert (
        terminal_tui._terminal_size_polling_interval(
            platform="darwin",
            main_thread=False,
        )
        == 0.5
    )


def test_full_screen_shell_keeps_header_inside_the_active_layout():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=handle_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        root = application.layout.container
        main_shell = root.children[0].content
        content_window = main_shell.children[0]

        def rendered_header() -> str:
            return "".join(text for _style, text in content_window.content.text())

        output.size = Size(rows=30, columns=69)
        application.before_render.fire()
        narrow = rendered_header()
        output.size = Size(rows=30, columns=100)
        application.before_render.fire()
        wide = rendered_header()

        assert application.full_screen is True
        assert len(main_shell.children) == 6
        assert narrow.count("DAITA") == 1
        assert wide.count("DAITA") == 1
        assert "atlas" in wide
        assert "source" in wide
        assert output.text == ""


def test_full_screen_has_no_embedded_scrollbars_and_composer_starts_one_line():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=handle_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        main_shell = application.layout.container.children[0].content
        transcript = main_shell.children[0]
        approval = main_shell.children[2].content.children[1].children[1]
        composer_frame = main_shell.children[3]
        top, composer, bottom = composer_frame.children
        glyphs = terminal_tui._terminal_glyphs(
            terminal_tui._terminal_capabilities(output)
        )

        top_line = "".join(text for _style, text in top.content.text())
        bottom_line = "".join(text for _style, text in bottom.content.text())

        assert transcript.right_margins == []
        assert approval.right_margins == []
        assert type(composer).__name__ == "Window"
        assert composer.wrap_lines() is True
        assert composer.dont_extend_height() is True
        assert composer.height.min == 1
        assert composer.height.max == terminal_tui._MAX_COMPOSER_ROWS
        assert top_line == glyphs.horizontal * output.size.columns
        assert bottom_line == glyphs.horizontal * output.size.columns
        assert glyphs.vertical not in top_line + bottom_line
        assert glyphs.top_left not in top_line
        assert glyphs.top_right not in top_line
        assert glyphs.bottom_left not in bottom_line
        assert glyphs.bottom_right not in bottom_line


async def test_composer_expands_and_shrinks_for_typed_wrapping_but_not_paste():
    output = _RecordingOutput()
    output.size = Size(rows=30, columns=40)
    state = TerminalViewState("atlas", "model", "source")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=handle_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        composer = (
            application.layout.container.children[0].content.children[3].children[1]
        )
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(
            lambda: composer.render_info is not None
            and composer.render_info.window_height == 1
        )

        pipe.send_text("x" * 39)
        await _wait_until(
            lambda: composer.render_info is not None
            and composer.render_info.window_height == 2
        )

        output.size = Size(rows=30, columns=60)
        application._on_resize()
        await _wait_until(
            lambda: composer.render_info is not None
            and composer.render_info.window_height == 1
        )
        output.size = Size(rows=30, columns=40)
        application._on_resize()
        await _wait_until(
            lambda: composer.render_info is not None
            and composer.render_info.window_height == 2
        )

        pipe.send_text("\x7f" * 2)
        await _wait_until(
            lambda: composer.render_info is not None
            and composer.render_info.window_height == 1
        )
        application.current_buffer.reset()

        pipe.send_text(f"\x1b[200~{'p' * 39}\x1b[201~")
        await _wait_until(lambda: application.current_buffer.text == "[Pasted Text #1]")
        assert composer.render_info is not None
        assert composer.render_info.window_height == 1

        application.current_buffer.reset()
        pipe.send_text("\x04")
        await task


async def test_double_escape_clears_the_full_composer_without_submitting():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result("handled")

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)

        pipe.send_text("\x1b[200~first line\nsecond line\x1b[201~")
        await _wait_until(lambda: application.current_buffer.text == "[Pasted Text #1]")
        pipe.send_text("\x1b\x1b")
        await _wait_until(lambda: application.current_buffer.text == "")

        assert state.notice == "Input cleared."
        assert submitted == []

        pipe.send_text("replacement\r")
        await _wait_until(lambda: submitted == ["replacement"] and not state.running)
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"


def test_responsive_metadata_and_status_collapse_order_are_deterministic():
    state = TerminalViewState(
        "atlas",
        "gpt-5.6-sol",
        "PostgreSQL · 3 sources",
    )
    state.steps = 2
    state.total_tokens = 1_800
    state.estimated_cost = "$0.02 estimated"
    glyphs = terminal_tui._terminal_glyphs(
        terminal_tui.TerminalCapabilities("truecolor", True)
    )

    full = terminal_tui._status_projection(
        state,
        width=100,
        mode="full",
        glyphs=glyphs,
    )
    compact = terminal_tui._status_projection(
        state,
        width=99,
        mode="compact",
        glyphs=glyphs,
    )
    narrow = terminal_tui._status_projection(
        state,
        width=69,
        mode="narrow",
        glyphs=glyphs,
    )

    assert full.collapsed == ("cost",)
    assert full.source_summary == "PostgreSQL · 3 sources"
    assert "source: PostgreSQL · 3 sources" in full.left
    assert full.right == "2 steps · 1.8k tokens"
    assert compact.collapsed == ("cost",)
    assert compact.right == "2 steps · 1.8k tokens"
    assert narrow.collapsed == (
        "cost",
        "tokens",
        "shorten_model",
        "model",
        "steps",
    )
    assert narrow.left == "atlas · source: PostgreSQL · 3 sources · ● ready"
    assert narrow.right == ""
    assert narrow.source_summary == "PostgreSQL · 3 sources"


def test_no_color_and_bounded_color_depth_projection_keep_semantic_text():
    assert (
        terminal_tui._terminal_capabilities(
            environ={"NO_COLOR": "1", "LANG": "en_US.UTF-8"}
        ).color_depth
        == "none"
    )
    assert (
        terminal_tui._terminal_capabilities(
            environ={"COLORTERM": "truecolor", "LANG": "en_US.UTF-8"}
        ).color_depth
        == "truecolor"
    )
    assert (
        terminal_tui._terminal_capabilities(
            environ={"TERM": "xterm-256color", "LANG": "en_US.UTF-8"}
        ).color_depth
        == "256"
    )
    assert (
        terminal_tui._terminal_capabilities(
            environ={"TERM": "xterm", "LANG": "en_US.UTF-8"}
        ).color_depth
        == "16"
    )
    assert (
        terminal_tui._terminal_capabilities(
            environ={
                "TERM": "xterm",
                "LANG": "en_US.UTF-8",
                "DAITA_ASCII": "1",
            }
        ).unicode
        is False
    )
    no_color = terminal_tui.TerminalCapabilities("none", True)
    no_color_rules = terminal_tui._semantic_style_rules(no_color)

    assert all(
        "#" not in value and "ansi" not in value for value in no_color_rules.values()
    )
    state = TerminalViewState("atlas", "model", "source")
    projection = terminal_tui._status_projection(
        state,
        width=100,
        mode="full",
        glyphs=terminal_tui._terminal_glyphs(no_color),
    )
    assert "atlas" in projection.left
    assert "ready" in projection.left

    for depth in ("truecolor", "256", "16"):
        capabilities = terminal_tui.TerminalCapabilities(depth, True)
        rules = terminal_tui._semantic_style_rules(capabilities)
        Style.from_dict(rules)
        assert all(
            "blink" not in value and "bg:" not in value for value in rules.values()
        )
        if depth == "truecolor":
            assert any("#22c55e" in value for value in rules.values())
        else:
            assert all("#" not in value for value in rules.values())
            assert any("ansi" in value for value in rules.values())


def test_unknown_terminal_capabilities_fail_down_to_visible_safe_fallbacks():
    class UnknownOutput:
        encoding = None

        def get_default_color_depth(self) -> Any:
            raise OSError("color unavailable")

        def get_size(self) -> Any:
            raise OSError("size unavailable")

    output = UnknownOutput()
    capabilities = terminal_tui._terminal_capabilities(output, environ={})
    size = terminal_tui._terminal_size(output)
    projection = terminal_tui._responsive_projection(*size)
    message = "".join(
        text
        for _style, text in terminal_tui._resize_message_fragments(
            projection,
            glyphs=terminal_tui._terminal_glyphs(capabilities),
        )
    )

    assert capabilities == terminal_tui.TerminalCapabilities("none", False)
    assert size == (32, 1)
    assert projection.usable is False
    assert "Terminal too small (32x1)" in message
    assert "Resize to at least 32x8" in message


async def test_unavailable_mouse_reporting_keeps_keyboard_navigation_active():
    class NoMouseOutput(_RecordingOutput):
        def __getattribute__(self, name: str) -> Any:
            if name in {"enable_mouse_support", "disable_mouse_support"}:
                raise AttributeError(name)
            return super().__getattribute__(name)

    output = NoMouseOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("\n".join(f"row-{index:03d}" for index in range(100)))

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=no_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        assert application.mouse_support() is False
        assert state.notice == (
            "Mouse interaction unavailable; keyboard controls remain active."
        )
        pipe.send_text("\x1b[5~")
        await _wait_until(
            lambda: state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert output.mouse_enable_count == 0


def test_ascii_projection_replaces_every_structural_border_and_state_glyph():
    capabilities = terminal_tui.TerminalCapabilities("16", False)
    glyphs = terminal_tui._terminal_glyphs(capabilities)
    state = TerminalViewState("atlas", "model", "source")
    state.running = True
    state.run_status = "querying"
    status = terminal_tui._status_projection(
        state,
        width=100,
        mode="full",
        glyphs=glyphs,
    )
    card = terminal_tui.ToolCardState(
        run_id="run-ascii",
        call_id="call-ascii",
        capability_id="data.sqlite.query",
        label="Query SQLite",
        state="succeeded",
    )
    rendered = "".join(
        text
        for _style, text in terminal_tui._render_tool_card_fragments(
            card,
            width=100,
            capabilities=capabilities,
            glyphs=glyphs,
        )
    )

    assert glyphs.top_left == "+"
    assert glyphs.horizontal == "-"
    assert glyphs.vertical == "|"
    assert glyphs.prompt == ">"
    assert glyphs.running == ("~", "-", "~", "+")
    assert glyphs.success == "OK"
    assert glyphs.failure == glyphs.warning == glyphs.approval == "!"
    assert all(symbol not in rendered for symbol in "╭╮╰╯─│✓◐●›")
    assert "+-" in rendered
    assert "OK Query SQLite" in rendered
    assert "querying" in status.left


def test_ascii_setup_status_and_plain_chat_prompt_are_readable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DAITA_ASCII", "1")
    output = io.StringIO()

    terminal_tui._write_setup_status(
        output,
        "✓ Connection validated",
        role="success",
    )
    capabilities = terminal_tui._terminal_capabilities(text_stream=output)

    assert output.getvalue() == "OK Connection validated\n"
    assert terminal_tui._terminal_glyphs(capabilities).prompt == ">"


async def test_composer_enforces_the_existing_input_bound(
    monkeypatch: pytest.MonkeyPatch,
):
    input_bound = 64
    monkeypatch.setattr(tui_application, "MAX_COMPOSER_CHARACTERS", input_bound)
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result("bounded")

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pipe.send_text(("x" * (input_bound + 100)) + "\r")
        await _wait_until(lambda: bool(submitted))
        pipe.send_text("\x04")
        await task

    assert len(submitted) == 1
    assert len(submitted[0]) == input_bound


async def test_bracketed_paste_uses_live_row_width_and_numbered_placeholders():
    output = _RecordingOutput()
    output.size = Size(rows=30, columns=40)
    state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result(f"answer-{len(submitted)}")

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    def bracketed(value: str) -> str:
        return f"\x1b[200~{value}\x1b[201~"

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)

        exact_row = "x" * 38
        pipe.send_text(bracketed(exact_row))
        await _wait_until(lambda: application.current_buffer.text == exact_row)
        assert "[Pasted Text #" not in application.current_buffer.text
        application.current_buffer.reset()

        output.size = Size(rows=30, columns=60)
        application._on_resize()
        await _wait_until(lambda: application.renderer._last_size == output.size)
        wider_row = "y" * 50
        pipe.send_text(bracketed(wider_row))
        await _wait_until(lambda: application.current_buffer.text == wider_row)
        application.current_buffer.reset()

        output.size = Size(rows=30, columns=40)
        application._on_resize()
        await _wait_until(lambda: application.renderer._last_size == output.size)
        first_paste = "a" * 50
        second_paste = "second\r\npaste"
        pipe.send_text(bracketed(first_paste))
        await _wait_until(lambda: application.current_buffer.text == "[Pasted Text #1]")
        pipe.send_text(" ")
        pipe.send_text(bracketed(second_paste))
        display_message = "[Pasted Text #1] [Pasted Text #2]"
        await _wait_until(lambda: application.current_buffer.text == display_message)

        pipe.send_text("\r")
        await _wait_until(lambda: len(submitted) == 1 and not state.running)
        assert submitted == [f"{first_paste} second\npaste"]
        assert state.blocks[0].text == submitted[0]

        pipe.send_text("\x1b[A")
        await _wait_until(lambda: application.current_buffer.text == display_message)
        pipe.send_text("\r")
        await _wait_until(lambda: len(submitted) == 2 and not state.running)
        pipe.send_text("\x04")
        await task

    assert submitted == [
        f"{first_paste} second\npaste",
        f"{first_paste} second\npaste",
    ]
    assert [block.text for block in state.blocks if block.kind == "user"] == submitted


async def test_hidden_paste_preserves_the_existing_message_bound():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result("bounded paste")

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pasted = "x" * (MAX_COMPOSER_CHARACTERS + 100)
        pipe.send_text(f"\x1b[200~{pasted}\x1b[201~")
        await _wait_until(lambda: "[Pasted Text #1]" in output.text)
        pipe.send_text("\r")
        await _wait_until(lambda: bool(submitted))
        pipe.send_text("\x04")
        await task

    assert len(submitted) == 1
    assert submitted[0] == "x" * MAX_COMPOSER_CHARACTERS


async def test_deleting_paste_placeholder_purges_draft_and_recalled_history():
    output = _RecordingOutput()
    output.size = Size(rows=30, columns=40)
    state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result(f"answer-{len(submitted)}")

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, _approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=None,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        pasted = "hidden" * 10
        placeholder = "[Pasted Text #1]"
        pipe.send_text("keep ")
        pipe.send_text(f"\x1b[200~{pasted}\x1b[201~")
        await _wait_until(
            lambda: application.current_buffer.text == f"keep {placeholder}"
        )
        pipe.send_text("\r")
        await _wait_until(lambda: len(submitted) == 1 and not state.running)
        assert submitted == [f"keep {pasted}"]

        pipe.send_text("\x1b[A")
        await _wait_until(
            lambda: application.current_buffer.text == f"keep {placeholder}"
        )
        pipe.send_text("\x7f" * len(placeholder))
        await _wait_until(lambda: application.current_buffer.text == "keep ")

        pipe.send_text("\x1b[B")
        await _wait_until(lambda: application.current_buffer.text == "")
        pipe.send_text("\x1b[A")
        await _wait_until(lambda: application.current_buffer.text == "keep ")
        pipe.send_text(placeholder)
        await _wait_until(
            lambda: application.current_buffer.text == f"keep {placeholder}"
        )
        pipe.send_text("\r")
        await _wait_until(lambda: len(submitted) == 2 and not state.running)
        pipe.send_text("\x04")
        await task

    assert submitted == [
        f"keep {pasted}",
        f"keep {placeholder}",
    ]


def test_model_output_preserves_unicode_lines_and_neutralizes_terminal_controls():
    unsafe = (
        "Summary:\r\n"
        "- café 東京\n"
        "ANSI \x1b[2J OSC \x1b]0;unsafe\x07 "
        "rewrite\rhidden bidi\u202e end"
    )

    safe = terminal_tui._sanitize_terminal_text(
        unsafe,
        maximum=16_384,
        preserve_lines=True,
        fallback="",
    )
    rendered = terminal_tui._render_markdown_text(unsafe, width=100)

    assert "Summary:\n- café 東京\n" in safe
    assert "café 東京" in rendered
    assert "\x1b" not in safe
    assert "\x07" not in safe
    assert "\r" not in safe
    assert "\u202e" not in safe
    assert "?[2J" in safe
    assert "?]0;unsafe?" in safe
    assert "rewrite?hidden bidi? end" in safe
    assert "\x1b" not in rendered


def test_rich_markdown_renders_fenced_code_after_sanitization():
    rendered = terminal_tui._render_markdown_text(
        "Query:\n\n```sql\nSELECT region FROM revenue;\n```",
        width=100,
    )

    assert "Query:" in rendered
    assert "SELECT region FROM revenue;" in rendered
    assert "\x1b" not in rendered


async def test_ctrl_c_cancels_active_run_and_returns_to_composer():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pipe.send_text("slow question\r")
        await started.wait()
        pipe.send_text("\x03")
        await _wait_until(lambda: cancelled.is_set() and not state.running)
        assert not task.done()
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert state.notice == "Run interrupted; returning to the composer."
    assert [block.kind for block in state.blocks] == ["user"]
    assert output.show_count >= 1
    assert output.alternate_enter_count == 1
    assert output.alternate_exit_count == 1


async def test_submit_shows_user_block_and_working_state_before_run_progress():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def run_message(message: str, conversation_id: str | None) -> Any:
        assert message == "show feedback"
        assert conversation_id is None
        entered.set()
        await release.wait()
        return _result("done")

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pipe.send_text("show feedback\r")
        await entered.wait()

        assert [(block.kind, block.text) for block in state.blocks] == [
            ("user", "show feedback")
        ]
        assert state.running is True
        assert state.run_status == "working"

        release.set()
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert [block.kind for block in state.blocks] == [
        "user",
        "metadata",
        "assistant",
    ]


async def test_ctrl_c_removes_streaming_partial_and_reports_unrecorded_notice():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    bridge = TerminalObserverBridge()
    started = asyncio.Event()

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        bridge(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
        bridge(
            _event(
                AgentEventKind.MODEL_TEXT_DELTA,
                {"model_call_index": 1, "text": "unrecorded live draft"},
            )
        )
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            bridge(
                _event(
                    AgentEventKind.RUN_COMPLETED,
                    {"exit_kind": "interrupted", "reason": "cancelled"},
                )
            )
            raise

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            observer_bridge=bridge,
        )
        pipe.send_text("cancel stream\r")
        await started.wait()
        await _wait_until(
            lambda: any(block.kind == "assistant.partial" for block in state.blocks)
        )
        pipe.send_text("\x03")
        await _wait_until(lambda: not state.running)
        assert all(block.kind != "assistant.partial" for block in state.blocks)
        assert all("unrecorded live draft" not in block.text for block in state.blocks)
        assert state.notice == (
            "Partial assistant output was interrupted and was not recorded."
        )
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"


async def test_cancellation_settles_observed_run_and_live_tool_card():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    bridge = TerminalObserverBridge()
    started = asyncio.Event()

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        bridge(_event(AgentEventKind.RUN_STARTED, {"agent_id": "agent-live"}))
        bridge(
            _event(
                AgentEventKind.TOOL_STARTED,
                {
                    "call_id": "call-cancelled",
                    "tool_name": "data_query_sqlite",
                    "capability_id": "data.sqlite.query",
                },
            )
        )
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            bridge(
                _event(
                    AgentEventKind.RUN_COMPLETED,
                    {
                        "exit_kind": "interrupted",
                        "reason": "cancelled",
                        "steps": 1,
                        "duration_ms": 20,
                        "input_tokens": 2,
                        "output_tokens": 0,
                        "reasoning_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "total_tokens": 2,
                        "cost_status": "unavailable",
                        "cost_amount_usd": None,
                        "cost_basis": None,
                        "cost_rate_schedule_id": None,
                        "cost_code": "pricing_schedule_unavailable",
                        "cost_display": "cost unavailable",
                    },
                )
            )
            raise

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            observer_bridge=bridge,
        )
        pipe.send_text("cancel this\r")
        await started.wait()
        await _wait_until(lambda: "call-cancelled" in state.tool_cards)
        pipe.send_text("\x03")
        await _wait_until(
            lambda: not state.running
            and state.tool_cards["call-cancelled"].state == "failed"
        )
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert state.active_run_id is None
    assert state.run_status == "interrupted"
    assert state.tool_cards["call-cancelled"].error_code == "cancelled"
    assert state.notice == "Run interrupted; returning to the composer."


async def test_ctrl_c_while_idle_does_not_exit_and_ctrl_d_empty_does():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pipe.send_text("\x03")
        await _wait_until(lambda: "composer remains active" in output.text)
        assert not task.done()
        pipe.send_text("\x04")
        result = await task

    assert result == TerminalApplicationResult(None, "exit")


async def test_very_small_terminal_blocks_submission_until_resize(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        tui_application,
        "_terminal_size_polling_interval",
        lambda: 0.01,
    )
    output = _RecordingOutput()
    output.size = Size(rows=5, columns=31)
    state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result("resized")

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        await _wait_until(lambda: "Terminal too small (31x5)" in output.text)
        pipe.send_text("blocked\r")
        await asyncio.sleep(0.05)
        assert submitted == []

        output.size = Size(rows=30, columns=100)
        await asyncio.sleep(0.08)
        pipe.send_text(("\x7f" * len("blocked")) + "accepted\r")
        await _wait_until(lambda: submitted == ["accepted"] and not state.running)
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert output.show_count >= 1
    assert output.autowrap_count >= 1


async def test_resize_idle_running_and_approving_preserves_focus_and_view_state(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        tui_application,
        "_terminal_size_polling_interval",
        lambda: 0.01,
    )
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    preserved_card = terminal_tui.ToolCardState(
        run_id="preserved-run",
        call_id="preserved-call",
        capability_id="catalog.search",
        label="Search catalog",
        state="failed",
        expanded=True,
    )
    state.tool_cards[preserved_card.call_id] = preserved_card
    state.blocks.append(
        terminal_tui.TerminalBlock(
            "tool",
            preserved_card.call_id,
            tool_card=preserved_card,
        )
    )
    running_started = asyncio.Event()
    release_running = asyncio.Event()
    approval_request = _approval_request({"name": "resize-safe"})
    approval_decisions: list[ApprovalDecision] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        if message == "running":
            running_started.set()
            await release_running.wait()
            return _result("run completed", run_id="run-resize")
        if message == "approval":
            approval_decisions.append(await approval_bridge(approval_request))
            return _result("approval completed", run_id="run-approval")
        raise AssertionError(message)

    applications: list[Any] = []
    original_create_application = tui_application._create_application

    def capture_application(*args: Any, **kwargs: Any) -> Any:
        created = original_create_application(*args, **kwargs)
        applications.append(created[0])
        return created

    monkeypatch.setattr(
        tui_application,
        "_create_application",
        capture_application,
    )

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        await _wait_until(lambda: len(applications) == 1)
        application = applications[0]
        idle_focus = application.layout.current_control
        original_blocks = tuple(state.blocks)

        output.size = Size(rows=30, columns=80)
        await asyncio.sleep(0.08)
        assert application.layout.current_control is idle_focus
        assert tuple(state.blocks) == original_blocks
        assert preserved_card.expanded is True

        pipe.send_text("running\r")
        await running_started.wait()
        output.size = Size(rows=6, columns=60)
        await asyncio.sleep(0.08)
        assert state.running is True
        assert preserved_card.expanded is True
        output.size = Size(rows=30, columns=60)
        await asyncio.sleep(0.08)
        assert application.layout.current_control is idle_focus
        release_running.set()
        await _wait_until(lambda: not state.running)

        output.size = Size(rows=30, columns=100)
        pipe.send_text("approval\r")
        await _wait_until(lambda: state.approval_panel is not None)
        approval_focus = application.layout.current_control
        panel = state.approval_panel
        for width in (80, 60, 100):
            output.size = Size(rows=30, columns=width)
            await asyncio.sleep(0.08)
            assert state.approval_panel is panel
            assert approval_decisions == []
            assert application.layout.current_control is approval_focus
            assert preserved_card.expanded is True
        output.size = Size(rows=10, columns=100)
        await asyncio.sleep(0.08)
        assert state.approval_panel is panel
        assert approval_decisions == []
        output.size = Size(rows=30, columns=100)
        await asyncio.sleep(0.08)
        assert application.layout.current_control is approval_focus
        pipe.send_text("n")
        await _wait_until(lambda: approval_decisions == [ApprovalDecision.DENY])
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert preserved_card.expanded is True


@pytest.mark.parametrize("error", (RuntimeError("render failed"), KeyboardInterrupt()))
async def test_terminal_state_is_restored_after_application_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")

    async def fail(application: Any) -> Any:
        application.renderer._in_alternate_screen = True
        application.renderer._mouse_support_enabled = True
        application.renderer._bracketed_paste_enabled = True
        raise error

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    monkeypatch.setattr(tui_application, "_run_application", fail)
    with create_pipe_input() as pipe:
        with pytest.raises(type(error)):
            await run_terminal_tui(
                state,
                run_message=run_message,
                handle_command=handle_command,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                suspend_bridge=TerminalSuspendBridge(),
                enhanced_input=pipe,
                enhanced_output=output,
            )

    assert output.show_count >= 1
    assert output.alternate_exit_count == 1
    assert output.mouse_disable_count == 1
    assert output.bracketed_paste_disable_count == 1
    assert output.attribute_reset_count >= 1
    assert output.autowrap_count >= 1
    assert output.cursor_key_reset_count >= 1
    assert output.cursor_shape_reset_count >= 1
    assert output.flush_count >= 1


async def test_rendering_failure_waits_for_active_execution_without_cancelling_it(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    completed = False
    cancelled = False
    restored_before_completion = False

    async def authoritative_execution() -> None:
        nonlocal completed, cancelled, restored_before_completion
        try:
            await asyncio.sleep(0.02)
            restored_before_completion = (
                output.alternate_exit_count == 1
                and output.mouse_disable_count == 1
                and output.bracketed_paste_disable_count == 1
            )
            completed = True
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def fail(application: Any) -> Any:
        application.renderer._in_alternate_screen = True
        application.renderer._mouse_support_enabled = True
        application.renderer._bracketed_paste_enabled = True
        state.running = True
        state.active_task = asyncio.create_task(authoritative_execution())
        raise RuntimeError("renderer failed")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    monkeypatch.setattr(tui_application, "_run_application", fail)
    with create_pipe_input() as pipe:
        with pytest.raises(RuntimeError, match="renderer failed"):
            await run_terminal_tui(
                state,
                run_message=run_message,
                handle_command=handle_command,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                suspend_bridge=TerminalSuspendBridge(),
                enhanced_input=pipe,
                enhanced_output=output,
            )

    assert completed is True
    assert cancelled is False
    assert restored_before_completion is True
    assert output.show_count >= 1
    assert output.alternate_exit_count == 1
    assert output.mouse_disable_count == 1
    assert output.bracketed_paste_disable_count == 1


def test_terminal_mode_restoration_falls_back_when_renderer_reset_fails():
    output = _RecordingOutput()

    def fail_reset() -> None:
        raise OSError("renderer reset failed")

    application = SimpleNamespace(renderer=SimpleNamespace(reset=fail_reset))

    terminal_tui._restore_application(application, output)

    assert output.alternate_exit_count == 1
    assert output.mouse_disable_count == 1
    assert output.bracketed_paste_disable_count == 1
    assert output.show_count == 1
    assert output.cursor_key_reset_count == 1
    assert output.cursor_shape_reset_count == 1
    assert output.attribute_reset_count == 1
    assert output.autowrap_count == 1
    assert output.flush_count == 1


async def test_pre_admission_application_failure_restores_output_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")

    def fail(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("layout failed")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    monkeypatch.setattr(tui_application, "_create_application", fail)
    with create_pipe_input() as pipe:
        with pytest.raises(terminal_tui.TerminalTUIUnavailable):
            await run_terminal_tui(
                state,
                run_message=run_message,
                handle_command=handle_command,
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
                suspend_bridge=TerminalSuspendBridge(),
                enhanced_input=pipe,
                enhanced_output=output,
            )

    assert output.show_count >= 1
    assert output.alternate_exit_count == 0
    assert output.attribute_reset_count >= 1
    assert output.autowrap_count >= 1
    assert output.flush_count >= 1


async def test_conversation_continuation_passes_the_selected_id_to_each_run():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    calls: list[tuple[str, str | None]] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        calls.append((message, conversation_id))
        return _result(
            f"answer {len(calls)}",
            conversation_id="conversation-stable",
        )

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pipe.send_text("first\r")
        await _wait_until(lambda: len(calls) == 1 and not state.running)
        pipe.send_text("follow-up\r")
        await _wait_until(lambda: len(calls) == 2 and not state.running)
        pipe.send_text("\x04")
        await task

    assert calls == [
        ("first", None),
        ("follow-up", "conversation-stable"),
    ]


async def test_captured_local_commands_render_without_suspending_the_shell():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    bridge = TerminalSuspendBridge()
    commands: list[tuple[str, str | None]] = []
    suspension_checks: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        commands.append((command, conversation_id))
        assert bridge.enhanced_input is not None
        assert bridge.enhanced_output is output
        return TerminalCommandResult(
            conversation_id,
            output="Commands\n  /help\n",
        )

    def command_stays_in_shell(command: str) -> bool:
        suspension_checks.append(command)
        return False

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
            suspend_bridge=bridge,
            command_requires_suspension=command_stays_in_shell,
        )
        pipe.send_text("/help\r")
        await _wait_until(lambda: "Commands" in output.text)
        pipe.send_text("\x04")
        await task

    assert commands == [("/help", None)]
    assert suspension_checks == ["/help"]
    assert state.blocks[-1].text == "Commands\n  /help\n"
    assert output.text.count("Commands") == 1
    assert output.alternate_enter_count == 1
    assert output.alternate_exit_count == 1
    assert bridge.enhanced_input is None
    assert bridge.enhanced_output is None


async def test_external_prompt_commands_temporarily_suspend_full_screen():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        return TerminalCommandResult(
            conversation_id,
            output="Prompt completed\n",
        )

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
            command_requires_suspension=lambda command: command == "/model",
        )
        pipe.send_text("/model\r")
        await _wait_until(lambda: "Prompt completed" in output.text)
        pipe.send_text("\x04")
        await task

    assert output.alternate_enter_count == 2
    assert output.alternate_exit_count == 2


def test_tui_command_output_is_captured_without_leaking_above_the_shell():
    terminal_stream = io.StringIO()
    captured = terminal._TerminalCommandOutput(
        terminal_stream,
        passthrough=False,
    )

    print("Sources", file=captured)

    assert captured.value == "Sources\n"
    assert terminal_stream.getvalue() == ""


@pytest.mark.parametrize(
    "command",
    (
        "/model",
        "/source",
        "/source use",
        "/source add",
        "/memory edit",
        "/review",
        "/user edit",
        "/skills edit forecast",
        "/skills delete forecast",
        "/skills create forecast",
        "/skills create",
    ),
)
def test_commands_with_external_prompts_keep_terminal_passthrough(command: str):
    assert terminal._command_uses_terminal_prompts(command) is True


async def test_exact_approval_panel_reviews_complete_frozen_arguments_and_approves_once():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    arguments: dict[str, object] = {
        "name": "bounded-skill",
        "content": "\n".join(f"line-{index:02d}" for index in range(40)),
        "metadata": {"enabled": True, "revision": 7},
    }
    request = _approval_request(arguments)
    decisions: list[ApprovalDecision] = []
    messages: list[tuple[str, str | None]] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        messages.append((message, conversation_id))
        if len(messages) == 1:
            decisions.append(await approval_bridge(request))
            return _result("saved")
        return _result("followed up")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("save this\r")
        await _wait_until(lambda: state.approval_panel is not None)
        panel = cast(terminal_tui.ApprovalPanelState, state.approval_panel)
        assert json.loads(panel.arguments_text) == arguments
        rendered = "".join(
            text
            for _style, text in terminal_tui._render_approval_panel_fragments(panel)
        )
        assert "skill_save" in rendered
        assert "skills.write" in rendered
        assert "line-00" in rendered
        assert "line-39" in rendered
        assert "[Y] Approve once" in rendered
        assert "[N] Deny" in rendered

        pipe.send_text("\x1b[6~")
        await _wait_until(lambda: panel.cursor_line > 0)
        pipe.send_text("y")
        await _wait_until(lambda: decisions == [ApprovalDecision.APPROVE])
        await _wait_until(lambda: not state.running)
        pipe.send_text("follow-up\r")
        await _wait_until(lambda: len(messages) == 2 and not state.running)
        pipe.send_text("\x04")
        result = await task

    assert result.action == "exit"
    assert state.approval_panel is None
    assert decisions == [ApprovalDecision.APPROVE]
    assert messages == [
        ("save this", None),
        ("follow-up", "conversation-one"),
    ]


async def test_approval_presentation_and_resolution_preserve_reviewed_anchor():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    for index in range(30):
        state.append_plain("assistant", f"earlier row {index}")
    request = _approval_request({"name": "anchor-safe", "content": "safe"})
    started = asyncio.Event()
    request_approval = asyncio.Event()

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        started.set()
        await request_approval.wait()
        assert await approval_bridge(request) is ApprovalDecision.DENY
        return _result("approval resolved")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("run with approval\r")
        await started.wait()
        pipe.send_text("\x1b[1;5H")
        await _wait_until(
            lambda: state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        anchor = state.transcript_viewport.anchor
        request_approval.set()
        await _wait_until(lambda: state.approval_panel is not None)
        assert state.transcript_viewport.anchor == anchor
        pipe.send_text("n")
        await _wait_until(lambda: state.approval_panel is None and not state.running)
        assert state.transcript_viewport.anchor == anchor
        assert (
            state.transcript_viewport.state
            is terminal_tui.TranscriptFollowState.REVIEWING
        )
        pipe.send_text("\x04")
        await task


@pytest.mark.parametrize(
    ("entered", "expected", "prompt_count"),
    (
        ("y\n", ApprovalDecision.APPROVE, 1),
        ("N\n", ApprovalDecision.DENY, 1),
        ("a\n\ny\n", ApprovalDecision.APPROVE, 3),
    ),
    ids=("approve", "deny", "invalid-then-approve"),
)
async def test_line_approval_uses_the_same_explicit_y_n_contract(
    entered: str,
    expected: ApprovalDecision,
    prompt_count: int,
) -> None:
    output = io.StringIO()
    decision = await terminal._prompt_for_exact_approval(
        _approval_request({"name": "bounded-skill", "content": "safe"}),
        input_stream=io.StringIO(entered),
        output_stream=output,
    )

    assert decision is expected
    assert (
        output.getvalue().count("Approve this exact change once? [y/n]") == prompt_count
    )
    assert output.getvalue().count("Enter y to approve or n to deny.") == max(
        0, prompt_count - 1
    )


def test_exact_approval_document_uses_the_existing_review_bound():
    reviewable = _approval_request(
        {
            "name": "bounded-skill",
            "content": "x" * 65_000,
        }
    )
    panel = terminal_tui._approval_panel_for_request(reviewable)

    assert panel is not None
    assert len(panel.arguments_text) <= terminal_tui.MAX_APPROVAL_DOCUMENT_CHARACTERS
    assert json.loads(panel.arguments_text) == reviewable.arguments.to_dict()

    oversized = _approval_request(
        {
            "name": "bounded-skill",
            "content": "x" * terminal_tui.MAX_APPROVAL_DOCUMENT_CHARACTERS,
        }
    )
    assert terminal_tui._approval_panel_for_request(oversized) is None


async def test_explicit_n_denies_approval_without_submitting_the_composer():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    request = _approval_request({"name": "bounded-skill", "content": "safe"})
    decisions: list[ApprovalDecision] = []
    messages: list[str] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        messages.append(message)
        decisions.append(await approval_bridge(request))
        return _result("denied")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("review\r")
        await _wait_until(lambda: state.approval_panel is not None)
        pipe.send_text("n")
        await _wait_until(lambda: decisions == [ApprovalDecision.DENY])
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task

    assert messages == ["review"]
    assert decisions == [ApprovalDecision.DENY]
    assert state.approval_panel is None


async def test_approval_owns_all_clicks_without_mouse_decision_or_run_effect():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    state.append_user("selected evidence")
    block = state.transcript_document.blocks[0]
    state.transcript_selection.begin(
        state.transcript_document,
        state.transcript_document.position(block.id, 0),
    )
    state.transcript_selection.finish(
        state.transcript_document,
        state.transcript_document.position(block.id, len("selected")),
    )
    state.transient_selection_hint = "Selected · Ctrl+C copy · Esc clear"
    request = _approval_request({"name": "bounded-skill", "content": "safe"})
    decisions: list[ApprovalDecision] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        decisions.append(await approval_bridge(request))
        return _result("handled")

    async def no_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    with create_pipe_input() as pipe:
        application, approval_previous, _deny_pending = (
            terminal_tui._create_application(
                terminal_tui._load_terminal_runtime(),
                state,
                run_message=run_message,
                load_transcript=None,
                handle_command=no_command,
                observer_bridge=TerminalObserverBridge(),
                approval_bridge=approval_bridge,
                enhanced_input=pipe,
                enhanced_output=output,
            )
        )
        main_shell = application.layout.container.children[0].content
        transcript_control = main_shell.children[0].content
        approval_control = (
            main_shell.children[2].content.children[1].children[1].content
        )
        composer_control = main_shell.children[3].children[1].content
        task = asyncio.create_task(terminal_tui._run_application(application))
        await _wait_until(lambda: output.alternate_enter_count == 1)
        pipe.send_text("review\r")
        await _wait_until(lambda: state.approval_panel is not None)
        active = state.active_task
        approval_focus = application.layout.current_control
        assert state.transient_selection_hint == ""

        def click(control: Any, *, x: int = 0, y: int = 0) -> None:
            for event_type in (MouseEventType.MOUSE_DOWN, MouseEventType.MOUSE_UP):
                event = MouseEvent(
                    position=Point(x=x, y=y),
                    event_type=event_type,
                    button=MouseButton.LEFT,
                    modifiers=frozenset(),
                )
                assert control.mouse_handler(event) is None

        click(approval_control)
        click(transcript_control, x=1, y=2)
        click(composer_control)
        pipe.send_text("\x1b[200~pasted\x1b[201~")
        await asyncio.sleep(0.05)

        assert decisions == []
        assert state.approval_panel is not None
        assert state.active_task is active
        assert active is not None and not active.done()
        assert state.transcript_selection.text == "selected"
        assert application.current_buffer.text == ""
        assert application.layout.current_control is approval_focus

        pipe.send_text("n")
        await _wait_until(lambda: decisions == [ApprovalDecision.DENY])
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task
        approval_bridge.restore(approval_previous)


@pytest.mark.parametrize(
    "key",
    (
        "a",
        "d",
        "x",
        "\r",
        " ",
        "\x1b",
        "\x1b\x1b",
        "\x1b[I",
        "\x1b[?1;2c",
    ),
    ids=(
        "old-approve",
        "old-deny",
        "invalid",
        "enter",
        "space",
        "escape",
        "double-escape",
        "terminal-focus-event",
        "terminal-device-response",
    ),
)
async def test_unrecognized_approval_keys_remain_pending(key: str):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    request = _approval_request({"name": "bounded-skill", "content": "safe"})
    decisions: list[ApprovalDecision] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        decisions.append(await approval_bridge(request))
        return _result("handled")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("review\r")
        await _wait_until(lambda: state.approval_panel is not None)
        panel = state.approval_panel
        pipe.send_text(key)
        await asyncio.sleep(0.05)
        remained_pending = state.approval_panel is panel and decisions == []
        if state.approval_panel is not None:
            pipe.send_text("n")
            await _wait_until(lambda: decisions == [ApprovalDecision.DENY])
            await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task

    assert remained_pending


async def test_approval_ctrl_c_interrupts_without_a_denial():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    request = _approval_request({"name": "bounded-skill", "content": "safe"})
    decisions: list[ApprovalDecision] = []
    cancelled = asyncio.Event()

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        try:
            decisions.append(await approval_bridge(request))
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _result("handled")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("review\r")
        await _wait_until(lambda: state.approval_panel is not None)
        pipe.send_text("\x03")
        await _wait_until(lambda: cancelled.is_set() and not state.running)
        pipe.send_text("\x04")
        await task

    assert decisions == []
    assert state.approval_panel is None


async def test_cancelled_and_invalid_approval_presenters_propagate_truthfully():
    request = _approval_request({"name": "bounded-skill", "content": "safe"})

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def cancelled(unexpected: ApprovalRequest) -> ApprovalDecision:
        assert unexpected is request
        raise asyncio.CancelledError

    previous = bridge.install(cancelled)
    with pytest.raises(asyncio.CancelledError):
        await bridge(request)
    bridge.restore(previous)

    async def invalid(unexpected: ApprovalRequest) -> Any:
        assert unexpected is request
        return "approve"

    previous = bridge.install(invalid)
    with pytest.raises(TypeError, match="ApprovalDecision"):
        await bridge(request)
    bridge.restore(previous)


async def test_approval_rendering_failure_propagates_without_a_false_denial(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    request = _approval_request({"name": "bounded-skill", "content": "safe"})
    failures: list[str] = []
    render_attempted = asyncio.Event()

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    def fail_render(
        panel: terminal_tui.ApprovalPanelState,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        render_attempted.set()
        raise RuntimeError(panel.tool_name)

    monkeypatch.setattr(
        tui_application,
        "_render_approval_panel_fragments",
        fail_render,
    )

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        try:
            await approval_bridge(request)
        except RuntimeError as error:
            failures.append(str(error))
        return _result("continued")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("review\r")
        await render_attempted.wait()
        await _wait_until(lambda: failures == ["skill_save"])
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task

    assert state.approval_panel is None
    assert state.blocks[-1].text == "continued"


async def test_application_shutdown_cancels_the_focused_approval():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    request = _approval_request({"name": "bounded-skill", "content": "safe"})
    decisions: list[ApprovalDecision] = []
    cancelled = asyncio.Event()

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        try:
            decisions.append(await approval_bridge(request))
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _result("shutdown")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("review\r")
        await _wait_until(lambda: state.approval_panel is not None)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cancelled.is_set()
    assert decisions == []
    assert state.approval_panel is None
    assert state.active_task is None


async def test_secret_shaped_approval_fails_without_a_false_user_denial():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    sentinel = "stage-four-credential-sentinel"
    request = _approval_request(
        {
            "name": "bounded-skill",
            "password": sentinel,
        }
    )
    failures: list[str] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        try:
            await approval_bridge(request)
        except RuntimeError as error:
            failures.append(str(error))
        return _result("denied safely")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            approval_bridge=approval_bridge,
        )
        pipe.send_text("review\r")
        await _wait_until(lambda: len(failures) == 1)
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task

    assert state.approval_panel is None
    assert "cannot be reviewed safely" in failures[0]
    assert sentinel not in output.text
    assert sentinel not in repr(state)


def test_themed_local_command_results_are_bounded_and_terminal_safe():
    state = TerminalViewState("atlas", "model", "source")
    unsafe = "[catalog]\x1b]52;c;clipboard\x07\u202e" + ("x" * 20_000)

    for presentation in ("status", "sources", "catalog", "settings"):
        state.append_local(presentation, unsafe)

    assert [block.kind for block in state.blocks] == [
        "local.status",
        "local.sources",
        "local.catalog",
        "local.settings",
    ]
    assert all(len(block.text) <= 16_384 for block in state.blocks)
    rendered_fragments = terminal_tui._render_transcript_fragments(
        terminal_tui._load_terminal_runtime(),
        state,
        width=100,
    )
    rendered = "".join(text for _style, text in rendered_fragments)
    styles = {style for style, _text in rendered_fragments}

    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\u202e" not in rendered
    assert "[catalog]" in rendered
    assert "class:tui.local.status.label" in styles
    assert "class:tui.local.sources.label" in styles
    assert "class:tui.local.catalog.label" in styles
    assert "class:tui.local.settings.label" in styles


async def test_slash_completion_covers_the_documented_surface_and_remains_local():
    assert terminal_tui._slash_command_completion_surface() == (
        "/model",
        "/sources",
        "/source",
        "/source use <name>",
        "/source add",
        "/source refresh <id>",
        "/source detach <source>",
        "/catalog",
        "/settings",
        "/new",
        "/resume <id>",
        "/conversation clear",
        "/learn <material>",
        "/review [cost-usd]",
        "/memory",
        "/user",
        "/skills",
        "/skills create",
        "/skills use <name> [request]",
        "/status",
        "/conversation",
        "/agent delete",
        "/help",
        "/exit",
    )
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    commands: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        commands.append(command)
        return TerminalCommandResult(
            conversation_id,
            output="Status\n  Ready\n",
            presentation="status",
        )

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
        )
        pipe.send_text("/sta\t")
        await _wait_until(lambda: "/status" in output.text)
        pipe.send_text("\r")
        await _wait_until(lambda: commands == ["/status"])
        pipe.send_text("\x04")
        await task

    assert state.blocks[-1].kind == "local.status"


def test_dynamic_skill_completion_adds_alias_and_preserves_builtin_priority():
    display, descriptions = terminal_tui._slash_completion_maps(
        (
            (
                "customer-health-investigation",
                "Investigate customer health.",
            ),
            ("status", "A colliding skill."),
        )
    )

    assert display["/customer-health-investigation "] == (
        "/customer-health-investigation"
    )
    assert descriptions["/customer-health-investigation "] == (
        "Investigate customer health."
    )
    assert display["/status"] == "/status"
    assert "/status " not in display


def test_source_completion_uses_friendly_quoted_one_run_selector():
    source = SimpleNamespace(
        id="source:sha256:" + ("a" * 64),
        display_name='Revenue "FY26"',
        adapter_id="postgresql",
        active=True,
    )

    assert terminal._source_override_completions((source,)) == (
        (
            '@"Revenue \\"FY26\\"" ',
            '@Revenue "FY26"',
            "Ask one question using PostgreSQL",
        ),
    )


async def test_skill_completion_refreshes_after_a_local_skill_mutation():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    commands: list[str] = []
    skills: list[tuple[str, str]] = []
    refreshes = 0

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def load_skill_completions() -> tuple[tuple[str, str], ...]:
        nonlocal refreshes
        refreshes += 1
        return tuple(skills)

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        commands.append(command)
        if command == "/skills":
            skills.append(
                (
                    "customer-health-investigation",
                    "Investigate customer health.",
                )
            )
            return TerminalCommandResult(conversation_id, output="Skill created.\n")
        assert command == "/customer-health-investigation"
        return TerminalCommandResult(conversation_id, action="exit")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
            load_skill_completions=load_skill_completions,
        )
        pipe.send_text("/skills\r")
        await _wait_until(lambda: commands == ["/skills"] and refreshes >= 1)
        pipe.send_text("/cus\t\r")
        result = await task

    assert commands == ["/skills", "/customer-health-investigation"]
    assert result.action == "exit"


async def test_at_dropdown_opens_filters_and_inserts_one_run_source_selector(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "First source")
    messages: list[str] = []
    applications: list[Any] = []
    original_create_application = tui_application._create_application

    def capture_application(*args: Any, **kwargs: Any) -> Any:
        created = original_create_application(*args, **kwargs)
        applications.append(created[0])
        return created

    monkeypatch.setattr(
        tui_application,
        "_create_application",
        capture_application,
    )

    async def run_message(message: str, conversation_id: str | None) -> Any:
        messages.append(message)
        return _result("Override complete.")

    def completion_state() -> Any:
        if not applications:
            return None
        return applications[0].current_buffer.complete_state

    source_completions = (
        (
            '@"First source" ',
            "@First source",
            "Ask one question using SQLite",
        ),
        (
            '@"Second source" ',
            "@Second source",
            "Ask one question using PostgreSQL",
        ),
    )
    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            source_completions=source_completions,
        )
        pipe.send_text("@")
        await _wait_until(
            lambda: completion_state() is not None
            and len(completion_state().completions) == 2
        )
        assert all(
            completion.display_meta_text
            for completion in completion_state().completions
        )

        pipe.send_text("sec")
        await _wait_until(
            lambda: completion_state() is not None
            and [completion.text for completion in completion_state().completions]
            == ['@"Second source" ']
        )
        await _wait_until(lambda: "Ask one question using PostgreSQL" in output.text)
        pipe.send_text("\t")
        await _wait_until(
            lambda: applications[0].current_buffer.text == '@"Second source" '
        )
        pipe.send_text("show recent orders\r")
        await _wait_until(lambda: messages == ['@"Second source" show recent orders'])
        pipe.send_text("\x04")
        await task

    assert state.source_summary == "First source"


async def test_source_completion_refreshes_after_a_local_source_mutation():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "First source")
    commands: list[str] = []
    messages: list[str] = []
    sources: list[tuple[str, str, str]] = []
    refreshes = 0

    async def load_source_completions() -> tuple[tuple[str, str, str], ...]:
        nonlocal refreshes
        refreshes += 1
        return tuple(sources)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        messages.append(message)
        return _result("Override complete.")

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        commands.append(command)
        sources.append(
            (
                '@"Second source" ',
                "@Second source",
                "Ask one question using SQLite",
            )
        )
        return TerminalCommandResult(conversation_id, output="Source added.\n")

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
            load_source_completions=load_source_completions,
        )
        pipe.send_text("/sources\r")
        await _wait_until(lambda: commands == ["/sources"] and refreshes >= 1)
        pipe.send_text("@sec\tquestion\r")
        await _wait_until(lambda: bool(messages))
        pipe.send_text("\x04")
        await task

    assert messages == ['@"Second source" question']


async def test_skill_command_result_delegates_exact_message_to_ordinary_model_run():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    invocation = "/customer-health-investigation inspect account 42"
    messages: list[tuple[str, str | None]] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        messages.append((message, conversation_id))
        return _result("Investigation complete.")

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        assert command == invocation
        return TerminalCommandResult(
            conversation_id,
            model_message=command,
        )

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
        )
        pipe.send_text(f"{invocation}\r")
        await _wait_until(lambda: messages == [(invocation, None)])
        await _wait_until(lambda: "Investigation complete." in output.text)
        pipe.send_text("\x04")
        await task

    assert state.conversation_id == "conversation-one"
    assert state.blocks[0].kind == "user"
    assert state.blocks[-1].kind == "assistant"


async def test_source_command_updates_the_pinned_status_source():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "First source")

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        assert command == "/source use Second source"
        return TerminalCommandResult(
            conversation_id,
            output="Source  Second source\n",
            source_summary="Second source",
        )

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
        )
        pipe.send_text("/source use Second source\r")
        await _wait_until(lambda: state.source_summary == "Second source")
        pipe.send_text("\x04")
        await task

    glyphs = terminal_tui._terminal_glyphs(
        terminal_tui.TerminalCapabilities("truecolor", True)
    )
    status = terminal_tui._status_projection(
        state,
        width=100,
        mode="full",
        glyphs=glyphs,
    )
    assert "atlas · source: Second source · model" in status.left


def test_slash_command_palette_uses_full_width_command_description_rows():
    glyphs = terminal_tui._terminal_glyphs(
        terminal_tui.TerminalCapabilities("truecolor", True)
    )
    fragments = terminal_tui._slash_command_menu_fragments(
        (
            ("/sources", "List registered data sources"),
            ("/status", "Show current agent status"),
        ),
        selected_index=1,
        width=80,
        glyphs=glyphs,
    )
    rendered = "".join(text for _style, text in fragments)
    lines = rendered.splitlines()
    styles = {style for style, _text in fragments}

    assert len(lines) == 2
    assert all(terminal_tui._display_width(line) == 80 for line in lines)
    assert "/sources" in lines[0]
    assert "List registered data sources" in lines[0]
    assert "› /status" in lines[1]
    assert "Show current agent status" in lines[1]
    assert "class:tui.command-menu.command.current" in styles
    assert "class:tui.command-menu.description.current" in styles


async def test_slash_dropdown_opens_filters_and_supports_arrow_navigation(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    commands: list[str] = []
    applications: list[Any] = []
    original_create_application = tui_application._create_application

    def capture_application(*args: Any, **kwargs: Any) -> Any:
        created = original_create_application(*args, **kwargs)
        applications.append(created[0])
        return created

    monkeypatch.setattr(
        tui_application,
        "_create_application",
        capture_application,
    )

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        commands.append(command)
        return TerminalCommandResult(
            conversation_id,
            output="Status\n  Ready\n",
            presentation="status",
        )

    def completion_state() -> Any:
        if not applications:
            return None
        return applications[0].current_buffer.complete_state

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
        )
        pipe.send_text("/")
        await _wait_until(
            lambda: completion_state() is not None
            and len(completion_state().completions)
            == len(terminal_tui._slash_command_completion_surface())
        )
        completions = completion_state().completions
        assert all(completion.display_meta_text for completion in completions)

        pipe.send_text("sta")
        await _wait_until(
            lambda: completion_state() is not None
            and [completion.text for completion in completion_state().completions]
            == ["/status"]
        )
        await _wait_until(lambda: "Show current agent status" in output.text)
        pipe.send_text("\x1b[B")
        await _wait_until(
            lambda: completion_state() is not None
            and completion_state().complete_index == 0
            and applications[0].current_buffer.text == "/status"
        )
        pipe.send_text("\r")
        await _wait_until(lambda: commands == ["/status"])
        pipe.send_text("\x04")
        await task

    assert state.blocks[-1].kind == "local.status"


async def test_composer_history_is_process_local_and_not_reused_by_a_new_shell():
    first_output = _RecordingOutput()
    first_state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result(
            f"answer-{len(submitted)}",
            conversation_id="conversation-history",
        )

    with create_pipe_input() as pipe:
        first_task = await _run_shell(
            pipe,
            first_output,
            first_state,
            run_message=run_message,
        )
        pipe.send_text("remember locally\r")
        await _wait_until(lambda: len(submitted) == 1 and not first_state.running)
        pipe.send_text("\x1b[A")
        await asyncio.sleep(0.05)
        pipe.send_text("\r")
        await _wait_until(lambda: len(submitted) == 2 and not first_state.running)
        pipe.send_text("\x04")
        await first_task

    assert submitted == ["remember locally", "remember locally"]

    second_output = _RecordingOutput()
    second_state = TerminalViewState("atlas", "model", "source")

    async def no_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    with create_pipe_input() as second_pipe:
        second_task = await _run_shell(
            second_pipe,
            second_output,
            second_state,
            run_message=no_message,
        )
        second_pipe.send_text("\x1b[A")
        await asyncio.sleep(0.05)
        second_pipe.send_text("\x04")
        result = await second_task

    assert result == TerminalApplicationResult(None, "exit")
    assert second_state.blocks == []


async def test_new_and_resume_commands_only_project_the_selected_conversation():
    output = _RecordingOutput()
    state = TerminalViewState(
        "atlas",
        "model",
        "source",
        conversation_id="conversation-old",
    )
    commands: list[tuple[str, str | None]] = []
    runs: list[tuple[str, str | None]] = []

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        commands.append((command, conversation_id))
        if command == "/new":
            return TerminalCommandResult(None, output="Conversation  new\n")
        if command == "/resume conversation-old":
            return TerminalCommandResult(
                "conversation-old",
                output="Conversation  conversation-old\n",
            )
        raise AssertionError(command)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        runs.append((message, conversation_id))
        return _result(
            f"answer-{len(runs)}",
            conversation_id=(
                "conversation-new" if conversation_id is None else conversation_id
            ),
        )

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
        )
        pipe.send_text("/new\r")
        await _wait_until(lambda: commands == [("/new", "conversation-old")])
        pipe.send_text("fresh question\r")
        await _wait_until(lambda: len(runs) == 1 and not state.running)
        pipe.send_text("/resume conversation-old\r")
        await _wait_until(lambda: len(commands) == 2 and not state.running)
        pipe.send_text("old follow-up\r")
        await _wait_until(lambda: len(runs) == 2 and not state.running)
        pipe.send_text("\x04")
        await task

    assert commands == [
        ("/new", "conversation-old"),
        ("/resume conversation-old", "conversation-new"),
    ]
    assert runs == [
        ("fresh question", None),
        ("old follow-up", "conversation-old"),
    ]


def test_non_tty_streams_keep_the_deterministic_plain_path():
    assert (
        terminal_tui.supports_terminal_tui(
            cast(TextIO, io.StringIO()),
            cast(TextIO, io.StringIO()),
        )
        is False
    )
