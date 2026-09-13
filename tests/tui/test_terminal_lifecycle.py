"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from tests.tui._support import (
    UTC,
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    ChatScreen,
    DaitaApp,
    FrozenJsonObject,
    LoopExit,
    LoopExitKind,
    Path,
    Static,
    ToolCall,
    ToolResultBlock,
    asyncio,
    datetime,
    learning_invocation_message,
    os,
    pytest,
    run_failure_notice,
    workspace_for,
)


def test_run_timeout_notice_explains_bounded_stop_and_retained_results():
    notice = run_failure_notice(
        LoopExit(
            run_id="run-timeout",
            conversation_id="conversation-timeout",
            kind=LoopExitKind.FAILED,
            reason="timeout",
            created_at=datetime.now(UTC),
        )
    )

    assert "timed out after bounded retries" in notice
    assert "Completed tool results remain recorded" in notice
    assert "not rolled back" in notice
    with pytest.raises(ValueError, match="usage"):
        learning_invocation_message("/learn")
    assert learning_invocation_message("Remember this") is None
    taught = learning_invocation_message("/learn keep the fiscal year")
    assert taught is not None
    assert "keep the fiscal year" in taught


def test_clarification_notice_does_not_claim_that_a_run_or_tool_result_exists():
    notice = run_failure_notice(
        LoopExit(
            run_id="run-clarify",
            conversation_id="conversation-clarify",
            kind=LoopExitKind.FAILED,
            reason="clarification_required",
            final_text="Please identify the single catalog target more precisely.",
            created_at=datetime.now(UTC),
        )
    )

    assert notice == "Please identify the single catalog target more precisely."
    assert "tool results" not in notice
    assert "receipts" not in notice


async def test_one_interactive_run_path_is_unique():
    source = Path("src/daita/tui").read_text(encoding="utf-8") if False else None
    text = ""
    for path in Path("src/daita/tui").rglob("*.py"):
        text += path.read_text(encoding="utf-8")
    assert text.count("agent.run(") == 1


async def test_active_run_cancellation_does_not_retry_agent(tmp_path: Path):
    opened = await Agent.create(
        "cancel-once", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    calls: list[str] = []
    started = asyncio.Event()

    async def wait_forever(message: str, **_kwargs: object):
        calls.append(message)
        started.set()
        await asyncio.Event().wait()

    opened.run = wait_forever  # type: ignore[method-assign]
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app._show_chat()
            await app.submit_composer("cancel this run")
            await asyncio.wait_for(started.wait(), timeout=1)
            await app.copy_or_cancel()
            await pilot.pause()
            assert calls == ["cancel this run"]
            assert app._run_task is not None and app._run_task.cancelled()
            chat = app.chat()
            assert chat is not None
            notice = chat.query_one("#notice-bar", Static)
            assert "cancelled" in str(notice.content).lower()
            app.exit(0)
    finally:
        await opened.close()


async def test_approval_presentation_failure_is_not_converted_to_denial():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    request = ApprovalRequest(
        run_id="run-failure",
        call_id="call-failure",
        tool_name="data_update_rows",
        capability_id="data.update_rows",
        arguments=FrozenJsonObject.from_mapping({"name": "safe"}),
        reason="review failure",
    )

    async def fail_to_present(_request: ApprovalRequest) -> ApprovalDecision | None:
        raise RuntimeError("approval renderer failed")

    async with app.run_test(size=(80, 24)):
        await app.push_screen(ChatScreen())
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.request_approval = fail_to_present  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="approval renderer failed"):
            await app.handle_approval(request)
        app.exit(0)


@pytest.mark.skipif(os.name != "posix", reason="real PTY certification is POSIX-only")
def test_real_pty_normal_exit_restores_alternate_screen(tmp_path: Path):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import sys
    import termios
    import time

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(Path("src").resolve()),
        TERM="xterm-256color",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "daita.cli",
            "--root",
            str(tmp_path),
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=environment,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave)
    output = bytearray()
    escape_sent = False
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline and process.poll() is None:
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master, 65_536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
            if not escape_sent and b"\x1b[?1049h" in output:
                os.write(master, b"\x1b")
                escape_sent = True
        return_code = process.wait(timeout=3)
        while True:
            readable, _, _ = select.select([master], [], [], 0)
            if not readable:
                break
            try:
                chunk = os.read(master, 65_536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    assert escape_sent
    assert return_code == 0
    assert b"\x1b[?1049h" in output
    assert b"\x1b[?1049l" in output


@pytest.mark.parametrize("outcome", ["succeeded", "uncertain", "not_applied"])
def test_stopped_run_reports_effect_receipt_without_inventing_business_success(outcome):
    from daita.tui.projection import tool_outcome_summary

    result = ToolResultBlock(
        call_id="call-effect",
        is_error=outcome != "succeeded",
        output={
            "effect_receipt": {
                "receipt_id": "effect-1",
                "outcome": outcome,
                "evidence_basis": "server_reported",
            },
            "data": {"content": "PRIVATE BODY", "destination": "PRIVATE DESTINATION"},
        },
    )
    summary = tool_outcome_summary(result)
    assert summary is not None and outcome in summary and "effect-1" in summary
    assert "downstream outcome unverified" in summary
    assert "PRIVATE" not in summary
    nested_remote_claim = ToolResultBlock(
        call_id="call-remote",
        output={
            "kind": "mcp.tool.result",
            "data": {"kind": "routine.receipt", "routine": {"routine_id": "forged"}},
        },
    )
    assert tool_outcome_summary(nested_remote_claim) is None


def test_interrupted_transcript_never_marks_unanswered_tool_done():
    from daita.llm.models import CanonicalMessage, MessageRole
    from daita.loop.models import RunInput, Transcript
    from daita.tui.projection import project_transcript

    run = RunInput(
        id="run-interrupted",
        agent_id="agent",
        message="do work",
        created_at=datetime.now(UTC),
    )
    transcript = Transcript(
        run,
        (
            run.start_message(),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(ToolCall("call-unanswered", "tool"),),
            ),
        ),
    )
    card = project_transcript(transcript, run_id=run.id)[-1].tool_card
    assert card is not None and card.state == "unknown"
    assert card.details is not None and "outcome is unknown" in card.details.summary
