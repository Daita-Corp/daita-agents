"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

import asyncio

from tests.support.host import StartHost, host_workspace, remote_agent
from tests.tui._support import (
    MAX_CLIPBOARD_UTF8_BYTES,
    MAX_COMPOSER_CHARACTERS,
    UTC,
    ActivityBar,
    AgentEvent,
    AgentEventKind,
    ChatScreen,
    ClipboardResult,
    CollapsibleTitle,
    Composer,
    ConfirmScreen,
    DaitaApp,
    FinishReason,
    FrozenJsonObject,
    LoopExit,
    MockModelProvider,
    ModelResponse,
    ObserverEvent,
    Offset,
    Selection,
    Static,
    ToolCard,
    ToolCardDetails,
    ToolCardState,
    TranscriptBlock,
    TranscriptView,
    _mock_profile,
    clipboard_mechanism,
    context_window_text,
    datetime,
    deliver_clipboard,
    format_token_count,
    osc52_sequence,
    pytest,
    sanitize_terminal_text,
    workspace_for,
)


def test_sanitize_strips_terminal_controls_and_bounds_text():
    assert (
        sanitize_terminal_text(
            "ok\x1b[31msecret\x07",
            maximum=32,
            preserve_lines=False,
            fallback="x",
        )
        == "ok?[31msecret?"
    )
    assert sanitize_terminal_text(
        "a" * 20,
        maximum=10,
        preserve_lines=False,
        fallback="x",
    ).endswith("...")


def test_clipboard_mechanism_and_osc52_bounds():
    assert clipboard_mechanism(platform="darwin", environ={}) == "pbcopy"
    assert clipboard_mechanism(platform="linux", environ={}) == "osc52"
    assert (
        clipboard_mechanism(
            platform="darwin",
            environ={"SSH_TTY": "/dev/pts/1"},
        )
        == "osc52"
    )
    payload = b"hello"
    assert "52;c;" in osc52_sequence(payload, tmux=False)
    with pytest.raises(ValueError):
        osc52_sequence(b"x" * (MAX_CLIPBOARD_UTF8_BYTES + 1), tmux=False)


async def test_clipboard_reports_empty_and_oversize_truthfully():
    assert await deliver_clipboard("") == ClipboardResult(
        "failure",
        "none",
        "Copy failed: selection is empty.",
    )
    huge = "é" * (MAX_CLIPBOARD_UTF8_BYTES // 2 + 1)
    result = await deliver_clipboard(huge)
    assert result.status == "failure"
    assert "64 KiB" in result.message


def test_context_window_copy_is_compact_exact_and_warns_near_capacity():
    assert format_token_count(999) == "999"
    assert format_token_count(8_500) == "8.5K"
    assert format_token_count(32_000) == "32K"
    assert format_token_count(1_050_000) == "1.1M"
    assert context_window_text(None, 32_000).plain == ""
    healthy = context_window_text(8_500, 32_000)
    warning = context_window_text(25_000, 32_000)
    critical = context_window_text(30_000, 32_000)
    assert healthy.plain == "ctx 8.5K / 32K"
    assert healthy.style == "dim"
    assert warning.style == "yellow"
    assert critical.style == "red"
    with pytest.raises(ValueError):
        format_token_count(-1)


def test_answer_summary_reports_queries_rows_steps_time_and_tokens():
    from datetime import timedelta

    from daita import LoopExitKind, Transcript
    from daita.llm.models import ModelUsage, ToolCall, ToolResultBlock
    from daita.loop.models import RunInput
    from daita.tui.projection import run_summary_text

    started = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    transcript = Transcript(
        run=RunInput(id="run-1", agent_id="agent", message="q", created_at=started)
    )
    result = LoopExit(
        run_id="run-1",
        conversation_id="conversation-1",
        kind=LoopExitKind.COMPLETED,
        reason="done",
        created_at=started + timedelta(seconds=4.1),
        final_text="answer",
        usage=ModelUsage(input_tokens=3_000, output_tokens=200),
    )
    pairs = (
        (
            ToolCall(id="c1", name="data_query"),
            ToolResultBlock(
                call_id="c1",
                output={
                    "data": {"columns": ["n"], "rows": [{"n": 1}], "total_rows": 1_204}
                },
            ),
        ),
        (
            ToolCall(id="c2", name="data_query"),
            ToolResultBlock(call_id="c2", output={"error": {}}, is_error=True),
        ),
        (ToolCall(id="c3", name="catalog_search"), None),
    )
    assert run_summary_text(result, transcript, pairs) == (
        "2 queries · 1,204 rows · 3 steps · 4.1s · 3.2K tokens"
    )
    assert run_summary_text(result, transcript, ()) == (
        "answered directly · 4.1s · 3.2K tokens"
    )


async def test_live_activity_and_exact_model_context_update_from_observation(
    start_host: StartHost,
):
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="unused",
            ),
        )
    )
    profile = _mock_profile(provider)
    harness = await start_host(model=provider, agent_options={"model_profile": profile})
    opened = await remote_agent(harness, "observed-agent")
    app = DaitaApp(
        root=harness.root, start_bootstrap=False, workspace=host_workspace(harness)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await app._show_chat()
            await pilot.pause()
            chat = app.chat()
            assert chat is not None
            activity = chat.query_one(ActivityBar)
            context = chat.query_one("#context-window", Static)
            assert str(context.content) == "" and not context.display

            pending_run = asyncio.create_task(asyncio.sleep(3600))
            app._run_task = pending_run
            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.RUN_STARTED,
                        occurred_at=datetime.now(UTC),
                        run_id="run-live",
                        conversation_id="conversation-live",
                        data=FrozenJsonObject.from_mapping({"agent_id": opened.id}),
                    )
                )
            )
            pending_run.cancel()
            await asyncio.gather(pending_run, return_exceptions=True)
            chat.set_activity("Thinking", restart=True)
            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.TOOL_COMPLETED,
                        occurred_at=datetime.now(UTC),
                        run_id="run-background",
                        conversation_id="conversation-live",
                        data=FrozenJsonObject.from_mapping(
                            {
                                "tool_name": "internal:profile",
                                "call_id": "call-background",
                            }
                        ),
                    )
                )
            )
            assert "Thinking" in str(activity.content)
            delta = AgentEvent(
                kind=AgentEventKind.MODEL_TEXT_DELTA,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping(
                    {"model_call_index": 1, "text": "Streaming answer"}
                ),
            )
            await app.on_observer_event(ObserverEvent(delta))
            assert activity.display is True
            assert "Writing answer" in str(activity.content)
            assert "Streaming answer" in chat.query_one(TranscriptView).copy_text()

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.MODEL_TEXT_DELTA,
                        occurred_at=datetime.now(UTC),
                        run_id="run-live",
                        conversation_id="conversation-live",
                        data=FrozenJsonObject.from_mapping(
                            {"model_call_index": 1, "text": " with **details**"}
                        ),
                    )
                )
            )
            await pilot.pause()
            transcript = chat.query_one(TranscriptView)
            assert "Streaming answer with **details**" in transcript.copy_text()
            assert (
                transcript.query_one("#block-assistant-partial", Static).content
                == "Streaming answer with **details**"
            )

            completed = AgentEvent(
                kind=AgentEventKind.MODEL_COMPLETED,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping(
                    {
                        "provider_id": provider.provider_id,
                        "model_call_index": 1,
                        "has_text": True,
                        "has_tool_calls": False,
                        "duration_ms": 12,
                        "input_tokens": 8_500,
                        "context_input_tokens": 8_500,
                        "output_tokens": 50,
                    }
                ),
            )
            await app.on_observer_event(ObserverEvent(completed))
            await pilot.pause()
            assert "ctx 8.5K / 32K" in str(context.content)

            for tool_name, expected in (
                ("toolbox_search", "Searching toolboxes"),
                ("toolbox_load", "Loading selected tools"),
                ("file_search", "Searching folder files"),
                ("file_read", "Reading folder file"),
                ("python_execute", "Running Python"),
                ("artifact_edit_text", "Preparing local file edit"),
                ("artifact_save_local", "Publishing local artifact"),
            ):
                started = AgentEvent(
                    kind=AgentEventKind.TOOL_STARTED,
                    occurred_at=datetime.now(UTC),
                    run_id="run-live",
                    conversation_id="conversation-live",
                    data=FrozenJsonObject.from_mapping(
                        {"tool_name": tool_name, "call_id": f"call-{tool_name}"}
                    ),
                )
                await app.on_observer_event(ObserverEvent(started))
                assert expected in str(activity.content)
                assert (
                    expected.casefold()
                    not in str(
                        chat.query_one("#status-primary", Static).content
                    ).casefold()
                )
                await app.on_observer_event(
                    ObserverEvent(
                        AgentEvent(
                            kind=AgentEventKind.TOOL_COMPLETED,
                            occurred_at=datetime.now(UTC),
                            run_id="run-live",
                            conversation_id="conversation-live",
                            data=FrozenJsonObject.from_mapping(
                                {"tool_name": tool_name, "call_id": f"call-{tool_name}"}
                            ),
                        )
                    )
                )
                assert "Waiting for model" in str(activity.content)
                assert "[working]" not in str(
                    chat.query_one("#status-primary", Static).content
                )

            for tool_name, call_id in (
                ("data_query", "call-live"),
                ("file_read", "call-file"),
            ):
                await app.on_observer_event(
                    ObserverEvent(
                        AgentEvent(
                            kind=AgentEventKind.TOOL_STARTED,
                            occurred_at=datetime.now(UTC),
                            run_id="run-live",
                            conversation_id="conversation-live",
                            data=FrozenJsonObject.from_mapping(
                                {"tool_name": tool_name, "call_id": call_id}
                            ),
                        )
                    )
                )

            tool_completed = AgentEvent(
                kind=AgentEventKind.TOOL_COMPLETED,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping(
                    {
                        "tool_name": "data_query",
                        "capability_id": "data.query",
                        "call_id": "call-live",
                        "duration_ms": 10,
                        "is_error": False,
                        "error_code": None,
                    }
                ),
            )
            await app.on_observer_event(ObserverEvent(tool_completed))
            assert "Reading folder file" in str(activity.content)
            # Finished steps stay visible as a timeline with their outcome.
            assert "✓ Query data  0.0s" in str(activity.content)
            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.TOOL_COMPLETED,
                        occurred_at=datetime.now(UTC),
                        run_id="run-live",
                        conversation_id="conversation-live",
                        data=FrozenJsonObject.from_mapping(
                            {
                                "tool_name": "file_read",
                                "call_id": "call-file",
                                "success": False,
                            }
                        ),
                    )
                )
            )
            assert "Waiting for model" in str(activity.content)
            assert "✗ Read folder file" in str(activity.content)

            run_completed = AgentEvent(
                kind=AgentEventKind.RUN_COMPLETED,
                occurred_at=datetime.now(UTC),
                run_id="run-live",
                conversation_id="conversation-live",
                data=FrozenJsonObject.from_mapping({"exit_kind": "completed"}),
            )
            await app.on_observer_event(ObserverEvent(run_completed))
            assert activity.display is False
            app.exit(0)
    finally:
        await opened.close()


async def test_composer_submit_runs_agent_once(start_host: StartHost):
    from datetime import UTC, datetime

    from daita import LoopExit, LoopExitKind
    from daita.tui.screens.chat import ChatScreen

    harness = await start_host()
    opened = await remote_agent(harness, "ready")
    calls: list[str] = []

    async def fake_run(message: str, **_kwargs: object) -> LoopExit:
        calls.append(message)
        return LoopExit(
            run_id="run-test",
            conversation_id="conv-test",
            kind=LoopExitKind.COMPLETED,
            reason="done",
            created_at=datetime.now(UTC),
            final_text="one answer",
        )

    opened.run = fake_run  # type: ignore[method-assign]
    app = DaitaApp(
        root=harness.root, start_bootstrap=False, workspace=host_workspace(harness)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app._show_chat()
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            await app.submit_composer("what is ready?")
            assert app._run_task is not None
            await app._run_task
            await pilot.pause()
            assert calls == ["what is ready?"]
            assert MAX_COMPOSER_CHARACTERS == 16_384
            app.exit(0)
    finally:
        await opened.close()


async def test_chat_accumulates_and_resumes_the_full_conversation_transcript(
    start_host: StartHost,
):
    provider = MockModelProvider(
        (
            ModelResponse(finish_reason=FinishReason.STOP, text="First answer"),
            ModelResponse(finish_reason=FinishReason.STOP, text="Second answer"),
        )
    )
    harness = await start_host(
        model=provider, agent_options={"model_profile": _mock_profile(provider)}
    )
    opened = await remote_agent(harness, "conversation-transcript")
    app = DaitaApp(
        root=harness.root, start_bootstrap=False, workspace=host_workspace(harness)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(80, 18)) as pilot:
            await app._show_chat()
            await app.submit_composer("First question")
            assert app._run_task is not None
            await app._run_task
            await pilot.pause()
            conversation_id = app.controller.conversation_id
            assert conversation_id is not None

            await app.submit_composer("Second question")
            assert app._run_task is not None
            await app._run_task
            await pilot.pause()

            chat = app.chat()
            assert chat is not None
            assert chat.query_one("#next-steps").display is True
            summaries = list(chat.query(".transcript-summary").results(Static))
            assert len(summaries) == 2
            assert "answered directly" in str(summaries[-1].content)
            last_run = app.controller.last_answer
            assert last_run is not None
            assert last_run[0] == conversation_id
            assert last_run[2] == "Second question"

            scheduled = await app.controller.dispatch_command("/schedule every monday")
            assert scheduled.kind == "run" and scheduled.run_message is not None
            assert last_run[1] in scheduled.run_message
            assert "Second question\nSchedule: every monday" in scheduled.run_message
            prompt = await app.controller.dispatch_command("/schedule")
            assert prompt.screen == "schedule_prompt"
            assert prompt.payload == {"question": "Second question"}
            taught = await app.controller.dispatch_command("/learn")
            assert taught.kind == "run" and taught.run_message is not None
            assert taught.run_message.startswith("/learn ")
            assert last_run[1] in taught.run_message
            saved = await app.controller.dispatch_command("/save")
            assert saved.kind == "run"

            transcript = app.screen.query_one(TranscriptView)
            rendered = transcript.copy_text()
            assert rendered.index("First question") < rendered.index("First answer")
            assert rendered.index("First answer") < rendered.index("Second question")
            assert rendered.index("Second question") < rendered.index("Second answer")

            await app._handle_command("/new")
            await pilot.pause()
            assert chat.query_one("#next-steps").display is False
            stale = await app.controller.dispatch_command("/save")
            assert stale.kind == "notice" and "Ask a question first" in stale.message
            assert "First question" not in transcript.copy_text()
            assert "Second answer" not in transcript.copy_text()

            await app._handle_command(f"/resume {conversation_id}")
            await pilot.pause()
            resumed = transcript.copy_text()
            assert resumed.index("First question") < resumed.index("First answer")
            assert resumed.index("First answer") < resumed.index("Second question")
            assert resumed.index("Second question") < resumed.index("Second answer")
            app.exit(0)
    finally:
        await opened.close()


async def test_transcript_history_supports_review_and_latest_navigation():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with app.run_test(size=(60, 14)) as pilot:
        await app.push_screen(ChatScreen())
        chat = app.chat()
        assert chat is not None
        chat.set_blocks(
            tuple(
                TranscriptBlock(
                    "assistant" if index % 2 else "user",
                    f"history-{index}",
                    f"Turn {index}\n" + "detail\n" * 3,
                )
                for index in range(12)
            )
        )
        transcript = chat.query_one(TranscriptView)
        await pilot.pause()
        transcript.follow_latest()
        await pilot.pause()
        bottom = transcript.scroll_y
        assert transcript.max_scroll_y > 0
        assert bottom == transcript.max_scroll_y

        await pilot.press("pageup")
        await pilot.pause()
        assert transcript.scroll_y < bottom
        assert transcript.following is False

        await pilot.press("ctrl+home")
        await pilot.pause()
        assert transcript.scroll_y == 0

        await pilot.press("ctrl+end")
        await pilot.pause()
        assert transcript.scroll_y == transcript.max_scroll_y
        assert transcript.following is True
        app.exit(0)


async def test_ctrl_o_toggles_tool_calls_without_exiting_chat():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            await app.push_screen(ChatScreen())
            chat = app.screen
            assert isinstance(chat, ChatScreen)
            chat.set_blocks(
                (
                    TranscriptBlock("user", "user-one", "Inspect the records"),
                    TranscriptBlock(
                        "tool",
                        "tool-one",
                        tool_card=ToolCardState(
                            run_id="run-one",
                            call_id="call-one",
                            capability_id="data.query",
                            label="Query records",
                            state="done",
                            details=ToolCardDetails(summary="Returned one record"),
                        ),
                    ),
                    TranscriptBlock("assistant", "assistant-one", "Done"),
                )
            )
            composer = chat.query_one(Composer)
            composer.focus()
            cards = list(chat.query(ToolCard))
            assert len(cards) == 1
            assert cards[0].display is False

            await pilot.press("ctrl+o")
            await pilot.pause()
            assert app.screen is chat
            assert cards[0].display is True
            assert composer.has_focus is True
            title = cards[0].query_one(CollapsibleTitle)
            assert title.styles.color.ansi == -1
            title.focus()
            await pilot.press("enter")
            await pilot.pause()
            detail = cards[0].query_one("#tool-detail-call-one", Static)
            assert "Returned one record" in str(detail.content)
            assert detail.styles.color.ansi == -1

            await pilot.press("ctrl+o")
            await pilot.pause()
            assert app.screen is chat
            assert cards[0].display is False
            app.exit(0)

        finally:
            app.exit(0)


async def test_copy_uses_native_wrap_independent_text_selection(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    copied: list[str] = []

    async def deliver(text: str) -> ClipboardResult:
        copied.append(text)
        return ClipboardResult("success", "test", "Copied selection.")

    monkeypatch.setattr("daita.tui.app.deliver_clipboard", deliver)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(ChatScreen())
        chat = app.chat()
        assert chat is not None
        chat.append_block(TranscriptBlock("user", "selection-user", "hello world"))
        await pilot.pause()
        widget = chat.query_one(".transcript-user", Static)
        chat.selections = {widget: Selection.from_offsets(Offset(0, 0), Offset(5, 0))}
        await app.copy_or_cancel()
        assert copied == ["hello"]
        app.exit(0)


async def test_popup_copy_uses_terminal_aware_clipboard(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    copied: list[str] = []

    async def deliver(text: str) -> ClipboardResult:
        copied.append(text)
        return ClipboardResult("success", "test", "Copied selection.")

    monkeypatch.setattr("daita.tui.app.deliver_clipboard", deliver)
    async with app.run_test(size=(80, 24)) as pilot:
        popup = ConfirmScreen("copy popup text")
        await app.push_screen(popup)
        await pilot.pause()
        message = popup.query_one("#confirm-message", Static)
        popup.selections = {
            message: Selection.from_offsets(Offset(0, 0), Offset(10, 0))
        }

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert copied == ["copy popup"]
        assert app.clipboard == "copy popup"
        assert app.screen is popup
        app.exit(0)
