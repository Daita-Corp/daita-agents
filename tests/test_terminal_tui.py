from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timezone
import io
import inspect
import json
from types import SimpleNamespace
from typing import Any, cast, TextIO

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
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
from daita.loop.models import RunInput, Transcript
from daita.observation import AgentEvent, AgentEventKind
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
        self.alternate_exit_count = 0
        self.attribute_reset_count = 0
        self.autowrap_count = 0
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

    def quit_alternate_screen(self) -> None:
        self.alternate_exit_count += 1

    def reset_attributes(self) -> None:
        self.attribute_reset_count += 1

    def enable_autowrap(self) -> None:
        self.autowrap_count += 1

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
) -> Any:
    return SimpleNamespace(
        run_id=run_id,
        conversation_id=conversation_id,
        final_text=text,
        kind=SimpleNamespace(value="completed"),
        reason="completed",
        steps=steps,
        usage=SimpleNamespace(total_tokens=tokens, estimated_cost_usd="0.01"),
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
) -> AgentEvent:
    return AgentEvent(
        kind=kind,
        occurred_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        run_id=run_id,
        conversation_id="conversation-live",
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
        assert isinstance(kwargs["observer_bridge"], TerminalObserverBridge)
        return TerminalApplicationResult(None, "exit")

    monkeypatch.setattr(terminal_tui, "run_terminal_tui", fake_tui)

    class _FakeAgent:
        name = "atlas"
        model_route = SimpleNamespace(
            candidates=(SimpleNamespace(provider_id="openai:gpt-5.6-sol"),)
        )

        async def list_sources(self) -> tuple[Any, ...]:
            return (SimpleNamespace(active=True, display_name="Warehouse"),)

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

    assert len(construction_calls) == 5
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
                "estimated_cost_usd": "0.02",
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
    assert state.estimated_cost == "0.02"


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
    assert rendered.index("Search catalog") < rendered.index("Query SQLite")
    assert rendered.index("Query SQLite") < rendered.index("Read data file")
    assert "unknown_column" in rendered
    assert "13ms" in rendered


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

    monkeypatch.setattr(terminal_tui, "_render_tool_card_fragments", broken_renderer)
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
        usage=SimpleNamespace(total_tokens=12, estimated_cost_usd="0.01"),
    )

    state.apply_result(result)

    assert state.active_run_id is None
    assert state.run_status == "ready"
    assert state.tool_cards["call-live"].state == "failed"
    assert state.tool_cards["call-live"].error_code == "observation_incomplete"
    assert state.blocks[-1].text == "authoritative answer"


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

    state.hydrate_transcript(transcript, run_id="run-one")

    assert [block.text for block in state.blocks if block.kind == "tool"] == [
        first.id,
        second.id,
    ]
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


def test_success_collapse_failure_expansion_and_toggles_are_process_local():
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

    assert first_state.toggle_expanded_detail() is True
    assert first_state.tool_cards[succeeded.id].expanded is True

    second_state = TerminalViewState("atlas", "model", "source")
    second_state.hydrate_transcript(transcript, run_id="run-one")
    assert second_state.tool_cards[failed.id].expanded is True
    assert second_state.tool_cards[succeeded.id].expanded is False
    assert transcript.messages[1].tool_calls == (failed, succeeded)


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
    assert output.alternate_exit_count == 0


def test_green_identity_focus_theme_uses_semantic_styles(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NO_COLOR", raising=False)

    rules = terminal_tui._semantic_style_rules()

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


def test_inline_shell_emits_one_scrollable_header_outside_the_active_layout():
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
        rendered_header = output.text

        output.size = Size(rows=30, columns=69)
        application.before_render.fire()
        output.size = Size(rows=30, columns=100)
        application.before_render.fire()

        assert application.full_screen is False
        assert len(main_shell.children) == 2
        assert rendered_header.count("DAITA") == 1
        assert "atlas" in rendered_header
        assert "source" in rendered_header
        assert output.text == rendered_header


def test_inline_composer_uses_only_top_and_bottom_rules():
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
        ready_body = main_shell.children[0]._get_container()
        composer_frame = ready_body.children[-1]
        top, composer, bottom = composer_frame.children
        glyphs = terminal_tui._terminal_glyphs(
            terminal_tui._terminal_capabilities(output)
        )

        top_line = "".join(text for _style, text in top.content.text())
        bottom_line = "".join(text for _style, text in bottom.content.text())

        assert type(composer).__name__ == "Window"
        assert top_line == glyphs.horizontal * output.size.columns
        assert bottom_line == glyphs.horizontal * output.size.columns
        assert glyphs.vertical not in top_line + bottom_line
        assert glyphs.top_left not in top_line
        assert glyphs.top_right not in top_line
        assert glyphs.bottom_left not in bottom_line
        assert glyphs.bottom_right not in bottom_line


def test_responsive_metadata_and_status_collapse_order_are_deterministic():
    state = TerminalViewState(
        "atlas",
        "gpt-5.6-sol",
        "PostgreSQL · 3 sources",
    )
    state.steps = 2
    state.total_tokens = 1_800
    state.estimated_cost = "0.02"
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

    assert full.collapsed == ()
    assert full.source_summary == "PostgreSQL · 3 sources"
    assert full.right == "2 steps · 1.8k tokens · $0.02"
    assert compact.collapsed == ("cost",)
    assert compact.right == "2 steps · 1.8k tokens"
    assert narrow.collapsed[:4] == (
        "cost",
        "tokens",
        "shorten_model",
        "source_summary",
    )
    assert narrow.left == "atlas · ● ready"
    assert narrow.right == ""
    assert narrow.source_summary == ""


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


async def test_composer_enforces_the_existing_input_bound():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    submitted: list[str] = []

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del conversation_id
        submitted.append(message)
        return _result("bounded")

    with create_pipe_input() as pipe:
        task = await _run_shell(pipe, output, state, run_message=run_message)
        pipe.send_text(("x" * (MAX_COMPOSER_CHARACTERS + 100)) + "\r")
        await _wait_until(lambda: bool(submitted))
        pipe.send_text("\x04")
        await task

    assert len(submitted) == 1
    assert len(submitted[0]) == MAX_COMPOSER_CHARACTERS


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
    assert output.alternate_exit_count == 0


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
                        "estimated_cost_usd": "0",
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
        terminal_tui,
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
        terminal_tui,
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
    original_create_application = terminal_tui._create_application

    def capture_application(*args: Any, **kwargs: Any) -> Any:
        created = original_create_application(*args, **kwargs)
        applications.append(created[0])
        return created

    monkeypatch.setattr(
        terminal_tui,
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
        pipe.send_text("d")
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
        del application
        raise error

    async def run_message(message: str, conversation_id: str | None) -> Any:
        raise AssertionError((message, conversation_id))

    async def handle_command(
        command: str,
        conversation_id: str | None,
    ) -> TerminalCommandResult:
        raise AssertionError((command, conversation_id))

    monkeypatch.setattr(terminal_tui, "_run_application", fail)
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
    assert output.alternate_exit_count == 0
    assert output.attribute_reset_count >= 1
    assert output.autowrap_count >= 1
    assert output.flush_count >= 1


async def test_rendering_failure_waits_for_active_execution_without_cancelling_it(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    completed = False
    cancelled = False

    async def authoritative_execution() -> None:
        nonlocal completed, cancelled
        try:
            await asyncio.sleep(0.02)
            completed = True
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def fail(application: Any) -> Any:
        del application
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

    monkeypatch.setattr(terminal_tui, "_run_application", fail)
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
    assert output.show_count >= 1
    assert output.alternate_exit_count == 0


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

    monkeypatch.setattr(terminal_tui, "_create_application", fail)
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


async def test_local_commands_run_while_tui_is_suspended_and_restore_the_shell():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    bridge = TerminalSuspendBridge()
    commands: list[tuple[str, str | None]] = []

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

    with create_pipe_input() as pipe:
        task = await _run_shell(
            pipe,
            output,
            state,
            run_message=run_message,
            handle_command=handle_command,
            suspend_bridge=bridge,
        )
        pipe.send_text("/help\r")
        await _wait_until(lambda: any(block.kind == "local" for block in state.blocks))
        pipe.send_text("\x04")
        await task

    assert commands == [("/help", None)]
    assert state.blocks[-1].text == "Commands\n  /help\n"
    assert output.text.count("Commands") == 1
    assert bridge.enhanced_input is None
    assert bridge.enhanced_output is None


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
        "/source add",
        "/memory edit",
        "/user edit",
        "/skills edit forecast",
        "/skills delete forecast",
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
        assert "[A] Approve once" in rendered
        assert "[D] Deny" in rendered

        pipe.send_text("\x1b[6~")
        await _wait_until(lambda: panel.cursor_line > 0)
        pipe.send_text("a")
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


@pytest.mark.parametrize(
    "key",
    (
        "d",
        "\x1b",
        "\x03",
        "x",
        "\r",
    ),
    ids=("deny", "escape", "ctrl-c", "invalid", "composer-submit"),
)
async def test_approval_keys_fail_closed_and_never_submit_the_composer(key: str):
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
        pipe.send_text(key)
        await _wait_until(lambda: decisions == [ApprovalDecision.DENY])
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task

    assert messages == ["review"]
    assert decisions == [ApprovalDecision.DENY]
    assert state.approval_panel is None


async def test_cancelled_and_invalid_approval_presenters_return_existing_deny():
    request = _approval_request({"name": "bounded-skill", "content": "safe"})

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def cancelled(unexpected: ApprovalRequest) -> ApprovalDecision:
        assert unexpected is request
        raise asyncio.CancelledError

    previous = bridge.install(cancelled)
    assert await bridge(request) is ApprovalDecision.DENY
    bridge.restore(previous)

    async def invalid(unexpected: ApprovalRequest) -> Any:
        assert unexpected is request
        return "approve"

    previous = bridge.install(invalid)
    assert await bridge(request) is ApprovalDecision.DENY
    bridge.restore(previous)


async def test_approval_rendering_failure_denies_without_failing_the_run(
    monkeypatch: pytest.MonkeyPatch,
):
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    request = _approval_request({"name": "bounded-skill", "content": "safe"})
    decisions: list[ApprovalDecision] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    def fail_render(panel: terminal_tui.ApprovalPanelState) -> Any:
        raise RuntimeError(panel.tool_name)

    monkeypatch.setattr(terminal_tui, "_render_approval_panel_fragments", fail_render)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        decisions.append(await approval_bridge(request))
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
        await _wait_until(lambda: decisions == [ApprovalDecision.DENY])
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task

    assert state.approval_panel is None
    assert state.blocks[-1].text == "continued"


async def test_application_shutdown_denies_the_focused_approval():
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

    assert decisions == [ApprovalDecision.DENY]
    assert state.approval_panel is None
    assert state.active_task is None


async def test_secret_shaped_approval_is_denied_before_output_or_view_state():
    output = _RecordingOutput()
    state = TerminalViewState("atlas", "model", "source")
    sentinel = "stage-four-credential-sentinel"
    request = _approval_request(
        {
            "name": "bounded-skill",
            "password": sentinel,
        }
    )
    decisions: list[ApprovalDecision] = []

    async def fallback(unexpected: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError(unexpected)

    approval_bridge = terminal_tui.TerminalApprovalBridge(fallback)

    async def run_message(message: str, conversation_id: str | None) -> Any:
        del message, conversation_id
        decisions.append(await approval_bridge(request))
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
        await _wait_until(lambda: decisions == [ApprovalDecision.DENY])
        await _wait_until(lambda: not state.running)
        pipe.send_text("\x04")
        await task

    assert state.approval_panel is None
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
        "/source add",
        "/source refresh <id>",
        "/catalog",
        "/settings",
        "/new",
        "/resume <id>",
        "/memory",
        "/user",
        "/skills",
        "/status",
        "/conversation",
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
    original_create_application = terminal_tui._create_application

    def capture_application(*args: Any, **kwargs: Any) -> Any:
        created = original_create_application(*args, **kwargs)
        applications.append(created[0])
        return created

    monkeypatch.setattr(
        terminal_tui,
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
        await _wait_until(lambda: "Show current agent status" in output.text)

        pipe.send_text("sta")
        await _wait_until(
            lambda: completion_state() is not None
            and [completion.text for completion in completion_state().completions]
            == ["/status"]
        )
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
