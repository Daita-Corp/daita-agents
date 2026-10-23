"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from daita.jobs import JobState
from daita.llm.models import ToolCall
from tests.support.host import StartHost, host_workspace, remote_agent
from tests.support.jobs import JobModel, in_state, open_job_agent, reply, wait_for_job
from tests.tui._support import (
    SLASH_COMMAND_COMPLETIONS,
    UTC,
    ActivityBar,
    AgentEvent,
    AgentEventKind,
    Button,
    DaitaApp,
    FrozenJsonObject,
    Input,
    JobsScreen,
    ObserverEvent,
    Static,
    TranscriptView,
    datetime,
    workspace_for,
)


async def test_jobs_command_opens_interactive_manager_without_model_calls():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    assert {
        insertion
        for insertion, _display, _description in SLASH_COMMAND_COMPLETIONS
        if insertion.startswith("/jobs")
    } == {"/jobs"}

    listed = await app.controller.dispatch_command("/jobs")
    assert listed.kind == "screen"
    assert listed.screen == "jobs"

    for removed_command in (
        "/jobs inspect job-running",
        "/jobs results job-succeeded",
        "/jobs cancel job-running",
    ):
        outcome = await app.controller.dispatch_command(removed_command)
        assert outcome.kind == "notice"
        assert outcome.message == "Usage: /jobs"


async def test_machine_origin_observations_do_not_project_into_foreground_chat(
    start_host: StartHost,
):
    harness = await start_host()
    opened = await remote_agent(harness, "origin-isolation-agent")
    app = DaitaApp(
        root=harness.root, start_bootstrap=False, workspace=host_workspace(harness)
    )
    app.controller.agent = opened
    observed = datetime.now(UTC)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await app._show_chat()
            await pilot.pause()
            chat = app.chat()
            assert chat is not None
            transcript = chat.query_one(TranscriptView)
            context = chat.query_one("#context-window", Static)

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.RUN_STARTED,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping({"agent_id": opened.id}),
                        run_origin="job_event",
                    )
                )
            )
            assert "reporting 1" in str(
                chat.query_one("#status-primary", Static).content
            )

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.MODEL_TEXT_DELTA,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping(
                            {"model_call_index": 1, "text": "Hidden report draft"}
                        ),
                        run_origin="job_event",
                    )
                )
            )
            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.MODEL_COMPLETED,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping(
                            {
                                "provider_id": "mock:scripted",
                                "model_call_index": 1,
                                "context_input_tokens": 9_999,
                            }
                        ),
                        run_origin="job_event",
                    )
                )
            )
            assert transcript.is_empty
            assert "Hidden report draft" not in transcript.copy_text()
            assert str(context.content) == ""
            assert chat.query_one(ActivityBar).display is False

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.TOOL_COMPLETED,
                        occurred_at=observed,
                        run_id="run-misattributed-background",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping(
                            {"tool_name": "internal:profile", "call_id": "call-job"}
                        ),
                        run_origin="user",
                    )
                )
            )
            assert chat.query_one(ActivityBar).display is False

            await app.on_observer_event(
                ObserverEvent(
                    AgentEvent(
                        kind=AgentEventKind.RUN_COMPLETED,
                        occurred_at=observed,
                        run_id="run-autonomous",
                        conversation_id="conversation-origin",
                        data=FrozenJsonObject.from_mapping({"exit_kind": "completed"}),
                        run_origin="job_event",
                    )
                )
            )
            assert "reporting" not in str(
                chat.query_one("#status-primary", Static).content
            )
            app.exit(0)
    finally:
        await opened.close()


async def test_jobs_screen_shows_the_checklist_and_answers_a_waiting_job(
    tmp_path, monkeypatch
):
    model = JobModel()
    model.script(
        main=[
            reply(
                ToolCall(
                    "plan",
                    "checklist_write",
                    {"items": [{"id": "a", "title": "Read \x1b[31mA"}]},
                )
            ),
            reply(ToolCall("q", "ask_user", {"question": "Which month?"})),
            reply(text="Used the answer."),
        ]
    )
    agent = await open_job_agent(tmp_path, model)
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    app.controller.agent = agent  # type: ignore[assignment]
    try:
        job = await agent.start_job("Compare the months.")
        await wait_for_job(agent, job.job_id, in_state(JobState.WAITING_FOR_YOU))
        async with app.run_test(size=(110, 36)) as pilot:
            await app.push_screen(JobsScreen())
            for _ in range(40):
                await pilot.pause(0.05)
                detail = str(app.screen.query_one("#jobs-detail", Static).content)
                if "Which month?" in detail:
                    break
            assert "Question for you:" in detail and "Which month?" in detail
            # Model-written checklist text is sanitized before it is shown.
            assert "[ ] Read ?[31mA" in detail and "\x1b" not in detail
            assert app.screen.query_one("#jobs-cancel", Button).disabled is False
            answer = app.screen.query_one("#jobs-answer", Input)
            assert answer.disabled is False
            answer.value = "September"
            answer.focus()
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if "Answered" in str(
                    app.screen.query_one("#jobs-notice", Static).content
                ):
                    break
            done = await wait_for_job(agent, job.job_id, in_state(JobState.DONE))
            assert [answer.text for answer in done.answers] == ["September"]
            app.exit(0)
    finally:
        await agent.close()
