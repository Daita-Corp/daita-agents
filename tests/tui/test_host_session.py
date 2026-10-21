"""The terminal app as a client of the Daita host: rejoin, questions, connection."""

from __future__ import annotations

from typing import Any

import pytest
from textual.widgets import Static

from daita import Agent, connect
from daita.tui.app import DaitaApp
from daita.tui.widgets.approval import ApprovalPanel
from daita.tui.widgets.status import ActivityBar
from daita.tui.widgets.transcript import TranscriptView
from tests.support.approval_learning import (
    MockModelProvider,
    _call,
    _memory_call,
    _profile,
    _stop,
)
from tests.support.host import (
    StartHost,
    bounded,
    finished_run,
    host_workspace,
    pending_interaction,
    remote_agent,
    value,
)

pytestmark = pytest.mark.integration

NOTE = "Revenue means paid invoice subtotal."
SIZE = (110, 36)


def _remembering(answer: str) -> MockModelProvider:
    """A model that asks to write memory, which needs approval, then answers."""

    return MockModelProvider((_call(_memory_call(content=NOTE)), _stop(answer)))


def _answering(*answers: str) -> MockModelProvider:
    return MockModelProvider(tuple(_stop(answer) for answer in answers))


def _app(harness: Any, *, agent_name: str | None = None) -> DaitaApp:
    return DaitaApp(
        root=harness.root,
        workspace=host_workspace(harness),
        agent_name=agent_name,
        start_bootstrap=agent_name is not None,
    )


def _host_status(app: DaitaApp) -> str:
    chat = app.chat()
    assert chat is not None
    return str(chat.query_one("#host-status", Static).content)


def _transcript(app: DaitaApp) -> str:
    chat = app.chat()
    return "" if chat is None else chat.query_one(TranscriptView).copy_text()


def _approve(app: DaitaApp) -> None:
    # The "y" binding; pilot.press would wait on transcript blocks that the
    # settling run removes meanwhile.
    chat = app.chat()
    assert chat is not None
    chat.query_one(ApprovalPanel).action_approve()


def _card(app: DaitaApp) -> bool:
    chat = app.chat()
    return chat is not None and chat.is_mounted and chat.query_one(ApprovalPanel).active


async def test_a_reopened_app_rejoins_the_active_run_and_shows_its_answer(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    # A window that has since closed left this run waiting on a question.
    run = await harness.submit(message="Remember revenue.", cwd=str(harness.work()))
    question = await pending_interaction(harness.http, "sales")
    app = _app(harness, agent_name="sales")
    async with app.run_test(size=SIZE):
        try:
            await bounded(lambda: app._run_task is not None)
            chat = app.chat()
            assert chat is not None
            (active,) = value(
                await harness.http.get("/v1/agents/sales/runs?active=true")
            )
            assert app.controller.conversation_id == active["conversation_id"]
            assert chat.query_one(ActivityBar).display

            # Another client answers; the rejoined window settles on the answer.
            answered = await harness.http.post(
                f"/v1/agents/sales/interactions/{question['id']}",
                json={"decision": "approve"},
            )
            assert answered.status_code == 200, answered.text
            await finished_run(harness.http, "sales", run["run_ref"])
            await bounded(lambda: app._run_task is not None and app._run_task.done())
            await bounded(lambda: "Saved." in _transcript(app))
            assert not chat.query_one(ActivityBar).display
        finally:
            app.exit(0)


async def test_an_approval_raised_with_no_window_open_is_answered_on_reopen(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    run = await harness.submit(message="Remember revenue.", cwd=str(harness.work()))
    await pending_interaction(harness.http, "sales")

    app = _app(harness, agent_name="sales")
    async with app.run_test(size=SIZE):
        try:
            await bounded(lambda: _card(app))
            _approve(app)
            finished = await finished_run(harness.http, "sales", run["run_ref"])
            await bounded(lambda: not _card(app))
        finally:
            app.exit(0)

    assert finished["state"] == "completed"
    assert finished["exit"].final_text == "Saved."
    agent = await connect("sales", cwd=harness.work(), root=harness.root)
    try:
        assert NOTE in await agent.read_memory()
    finally:
        await agent.close()


async def test_answering_in_one_window_removes_the_card_from_the_other(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    run = await harness.submit(message="Remember revenue.", cwd=str(harness.work()))
    await pending_interaction(harness.http, "sales")

    first, second = _app(harness, agent_name="sales"), _app(harness, agent_name="sales")
    async with first.run_test(size=SIZE):
        try:
            async with second.run_test(size=SIZE):
                try:
                    await bounded(lambda: _card(first) and _card(second))
                    _approve(first)
                    finished = await finished_run(harness.http, "sales", run["run_ref"])
                    await bounded(lambda: not _card(second))
                finally:
                    second.exit(0)
        finally:
            first.exit(0)

    assert finished["state"] == "completed"


async def test_the_status_bar_shows_connected_reconnecting_and_stopped(
    start_host: StartHost,
) -> None:
    harness = await start_host()
    agent = await remote_agent(harness, "sales")
    app = _app(harness)
    app.controller.agent = agent
    async with app.run_test(size=SIZE):
        try:
            await app._show_chat()

            async def shows(text: str, *, drop: bool = False) -> bool:
                if drop:
                    harness.drop_connections()
                await app._refresh_status()
                return text in _host_status(app)

            assert await shows("● host")
            # A broken link reconnects by itself; drop it until the gap is seen.
            await bounded(lambda: shows("◌ reconnecting", drop=True))
            await bounded(lambda: shows("● host"))

            await harness.stop()
            await bounded(lambda: shows("○ host stopped"))
            # Background refreshes never restart a host someone stopped.
            await app.refresh_background_status(notify_new=True)
            assert not harness.paths.socket.exists()
            assert "○ host stopped" in _host_status(app)
        finally:
            app.exit(0)


async def test_a_host_reset_reloads_the_transcript_and_work_board(
    start_host: StartHost,
) -> None:
    provider = _answering("First answer")
    profile = {"model_profile": _profile(provider)}
    first_host = await start_host(model=provider, agent_options=profile)
    root = first_host.root
    agent = await remote_agent(first_host, "sales")
    app = _app(first_host)
    app.controller.agent = agent
    agent.on_reset = app.controller.on_reset
    boards = 0
    work_board = app.controller.work_board

    async def counted() -> Any:
        nonlocal boards
        boards += 1
        return await work_board()

    app.controller.work_board = counted  # type: ignore[method-assign]
    # Only the reset may refresh here, not the two-second poll.
    app._poll_background_status = _idle  # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        try:
            await app._show_chat()
            await app.submit_composer("First question")
            assert app._run_task is not None
            await app._run_task
            conversation_id = app.controller.conversation_id
            assert conversation_id is not None

            # The host restarts; meanwhile this conversation gains a run the
            # window never saw, so its event history cannot be replayed.
            await first_host.stop()
            await bounded(lambda: agent.connection == "stopped")
            later = _answering("Second answer")
            offline = await Agent.open(
                "sales",
                root=root,
                workspace=host_workspace(first_host),
                model=later,
                model_profile=_profile(later),
            )
            try:
                await offline.run("Second question", conversation_id=conversation_id)
            finally:
                await offline.close()
            seen = boards
            second_host = await start_host(
                root=root, model=later, agent_options=profile
            )
            await bounded(lambda: "Second answer" in _transcript(app), timeout=20)
            await bounded(lambda: boards > seen)
            await pilot.pause()
            assert agent.connection == "connected"
        finally:
            app.exit(0)
    # Stop it before the first host's teardown checks their shared socket.
    await second_host.stop()


async def test_a_busy_host_in_another_folder_keeps_the_draft(
    start_host: StartHost,
) -> None:
    # The other client's run still waits when the test ends; drain it quickly.
    harness = await start_host(model=_remembering("unused"), drain_seconds=1)
    agent = await remote_agent(harness, "sales")
    app = _app(harness)
    app.controller.agent = agent
    async with app.run_test(size=SIZE) as pilot:
        try:
            await app._show_chat()
            # Another client's run moves the agent to its folder and waits there.
            await harness.submit(
                message="Remember revenue.", cwd=str(harness.work("other"))
            )
            await pending_interaction(harness.http, "sales")

            await app.submit_composer("What changed this week?")
            assert app._run_task is not None
            await app._run_task
            await pilot.pause()
            chat = app.chat()
            assert chat is not None
            notice = str(chat.query_one("#notice-bar", Static).content)
            assert "Daita is busy in" in notice and str(harness.work("other")) in notice
            assert "ask without local files" in notice
            assert chat.composer().text == "What changed this week?"
            assert "What changed this week?" not in _transcript(app)
        finally:
            app.exit(0)


async def _idle() -> None:
    return None
