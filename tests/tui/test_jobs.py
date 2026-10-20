"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from daita.jobs import ControlKind, ControlState
from daita.tui.screens.jobs import JobControlScreen
from tests.tui._support import (
    SLASH_COMMAND_COMPLETIONS,
    UTC,
    ActivityBar,
    Agent,
    AgentEvent,
    AgentEventKind,
    Button,
    ConfirmScreen,
    DaitaApp,
    FrozenJsonObject,
    GraphState,
    Input,
    JobsScreen,
    ObserverEvent,
    OptionList,
    Path,
    SimpleNamespace,
    Static,
    Text,
    TranscriptView,
    _tui_job_inspection,
    _tui_job_summary,
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
    tmp_path: Path,
):
    opened = await Agent.create(
        "origin-isolation-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
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


async def test_jobs_manager_lists_inspects_reads_cancels_and_refreshes(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    running = _tui_job_summary(
        "job-running-0123456789", GraphState.ACTIVE, result_available=False
    )
    succeeded = _tui_job_summary(
        "job-succeeded-0123456789", GraphState.SUCCEEDED, result_available=True
    )
    jobs = [running, succeeded]
    list_calls = 0
    cancel_calls: list[str] = []
    accepted_reviews: list[tuple[str, str, str, str]] = []
    review_control = SimpleNamespace(
        task_id="worker",
        control_id="review-control",
        kind=ControlKind.REVIEW_REQUESTED,
        state=ControlState.OPEN,
        created_at=datetime(2026, 8, 23, 14, 1, tzinfo=UTC),
        payload_digest="sha256:" + "9" * 64,
        payload=FrozenJsonObject.from_mapping(
            {
                "message": "Review the immutable candidate.",
                "candidate_digest": "sha256:" + "8" * 64,
            }
        ),
    )

    async def list_jobs() -> tuple[object, ...]:
        nonlocal list_calls
        list_calls += 1
        return tuple(jobs)

    async def inspect_job(job_id: str) -> object | None:
        inspection = next(
            (
                _tui_job_inspection(summary)
                for summary in jobs
                if summary.job_id == job_id
            ),
            None,
        )
        if inspection is not None and job_id == running.job_id:
            inspection.controls = (review_control,)
        return inspection

    async def read_job_result(job_id: str) -> object | None:
        if job_id != succeeded.job_id:
            return None
        observed = datetime(2026, 8, 23, 14, 1, tzinfo=UTC)
        return SimpleNamespace(
            job_id=job_id,
            result_id="result-profile",
            result_kind="data_profile.finalized",
            summary="Profiled one resource.",
            payload=FrozenJsonObject.from_mapping({"profiled_resources": 1}),
            sensitivity=SimpleNamespace(value="internal"),
            provenance=FrozenJsonObject.from_mapping(
                {"authority": "job_owner_agent_scope"}
            ),
            artifact_ids=(),
            completed_at=observed,
        )

    async def job_board(job_id: str) -> object | None:
        assert job_id in {running.job_id, succeeded.job_id}
        return SimpleNamespace(
            job_id=job_id,
            graph_state=next(item for item in jobs if item.job_id == job_id).state,
            graph_revision=3,
            columns=(
                SimpleNamespace(name="in_progress", task_ids=("worker",)),
                SimpleNamespace(name="blocked", task_ids=()),
            ),
            dependencies=(
                SimpleNamespace(
                    upstream_task_id="worker",
                    downstream_task_id="finalizer",
                    edge_kind="requires_accepted_success",
                    satisfied=False,
                ),
            ),
            diagnostics=SimpleNamespace(
                blockers=(),
                exhausted_budgets=(),
                deadlocked=False,
                deadlock_reason=None,
            ),
        )

    async def job_timeline(
        job_id: str, *, after_event_id: int, limit: int
    ) -> object | None:
        assert after_event_id == 0 and limit == 100
        observed = datetime(2026, 8, 23, 14, 1, tzinfo=UTC)
        return SimpleNamespace(
            job_id=job_id,
            graph_state=next(item for item in jobs if item.job_id == job_id).state,
            graph_revision=3,
            events=(
                SimpleNamespace(
                    event_id=11,
                    created_at=observed,
                    kind="task_claimed",
                    task_id="worker",
                ),
            ),
            next_cursor=None,
            diagnostics=SimpleNamespace(
                blockers=(),
                exhausted_budgets=(),
                deadlocked=False,
                deadlock_reason=None,
            ),
        )

    async def cancel_job(job_id: str) -> object | None:
        cancel_calls.append(job_id)
        if job_id != running.job_id:
            return None
        cancelled = _tui_job_summary(
            running.job_id, GraphState.CANCEL_REQUESTED, result_available=False
        )
        jobs[0] = cancelled
        return cancelled

    async def accept_job_review(
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        rationale: str,
        idempotency_key: str,
    ) -> object:
        assert idempotency_key.startswith("tui-")
        accepted_reviews.append((job_id, task_id, control_id, rationale))
        return SimpleNamespace(result_id="accepted-result")

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "inspect_job", inspect_job)
    monkeypatch.setattr(app.controller, "read_job_result", read_job_result)
    monkeypatch.setattr(app.controller, "job_board", job_board)
    monkeypatch.setattr(app.controller, "job_timeline", job_timeline)
    monkeypatch.setattr(app.controller, "cancel_job", cancel_job)
    monkeypatch.setattr(app.controller, "accept_job_review", accept_job_review)

    async with app.run_test(size=(110, 36)) as pilot:
        try:
            await app.push_screen(JobsScreen())
            for _ in range(200):
                await pilot.pause(0.05)
                if "2 jobs" in str(
                    app.screen.query_one("#jobs-summary", Static).content
                ):
                    break
            manager = app.screen
            assert isinstance(manager, JobsScreen)
            panel = manager.query_one("#jobs-manager")
            assert panel.styles.border_left[0] == "solid"
            assert panel.styles.background.ansi == -1
            listing = manager.query_one("#jobs-list", OptionList)
            assert listing.option_count == 2
            assert listing.has_focus is True
            first_prompt = listing.get_option_at_index(0).prompt
            assert isinstance(first_prompt, Text)
            assert "ACTIVE" in first_prompt.plain
            assert manager.query_one("#jobs-cancel", Button).disabled is False
            assert manager.query_one("#jobs-results", Button).disabled is True

            assert await pilot.click("#jobs-details") is True
            await pilot.pause()
            assert "Graph" in str(manager.query_one("#jobs-detail", Static).content)

            assert await pilot.click("#jobs-board") is True
            await pilot.pause()
            assert "worker -> finalizer" in str(
                manager.query_one("#jobs-detail", Static).content
            )

            assert await pilot.click("#jobs-timeline") is True
            await pilot.pause()
            timeline_text = str(manager.query_one("#jobs-detail", Static).content)
            assert "task_claimed" in timeline_text
            assert "Current state: active" in timeline_text

            assert await pilot.click("#jobs-controls") is True
            for _ in range(200):
                await pilot.pause(0.05)
                if isinstance(app.screen, JobControlScreen):
                    break
            assert isinstance(app.screen, JobControlScreen)
            app.screen.query_one("#job-control-value", Input).value = (
                "Authenticated candidate accepted."
            )
            assert await pilot.click("#job-control-submit") is True
            for _ in range(200):
                await pilot.pause(0.05)
                if app.screen is manager and accepted_reviews:
                    break
            assert accepted_reviews == [
                (
                    running.job_id,
                    "worker",
                    "review-control",
                    "Authenticated candidate accepted.",
                )
            ]
            assert app.screen is manager

            assert await pilot.click("#jobs-cancel") is True
            for _ in range(200):
                await pilot.pause(0.05)
                if isinstance(app.screen, ConfirmScreen):
                    break
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            for _ in range(200):
                await pilot.pause(0.05)
                if app.screen is manager and cancel_calls:
                    break
            assert app.screen is manager
            assert cancel_calls == [running.job_id]
            assert "Cancellation requested" in str(
                manager.query_one("#jobs-notice", Static).content
            )
            assert manager.query_one("#jobs-cancel", Button).disabled is True

            listing.highlighted = 1
            await pilot.pause()
            assert manager.query_one("#jobs-results", Button).disabled is False
            assert await pilot.click("#jobs-results") is True
            await pilot.pause()
            result_text = str(manager.query_one("#jobs-detail", Static).content)
            assert "profiled_resources" in result_text
            assert "Artifacts (0)" in result_text

            assert await pilot.click("#jobs-refresh") is True
            await pilot.pause()
            assert list_calls >= 3
            assert "Job statuses refreshed" in str(
                manager.query_one("#jobs-notice", Static).content
            )
            assert await pilot.click("#jobs-close") is True
            app.exit(0)
        finally:
            app.exit(0)
