"""Component-owned tests split from ``test_screens.py``."""

from __future__ import annotations

from tests.tui._support import (
    SLASH_COMMAND_COMPLETIONS,
    UTC,
    ActivityBar,
    Agent,
    AgentEvent,
    AgentEventKind,
    Button,
    ChatScreen,
    Composer,
    ConfirmScreen,
    DaitaApp,
    DeliveryState,
    DeliverySubjectKind,
    FrozenJsonObject,
    InboxScreen,
    InboxView,
    JobsScreen,
    JobStatus,
    ModelSensitivity,
    ObserverEvent,
    OptionList,
    OutcomeConclusionKind,
    OutcomeState,
    Path,
    SimpleNamespace,
    Static,
    Text,
    TranscriptView,
    UserInputError,
    _tui_inbox_item,
    _tui_job_inspection,
    _tui_job_summary,
    datetime,
    pytest,
    render_inbox_item,
    replace,
    workspace_for,
)


async def test_jobs_commands_route_without_model_calls(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    running = _tui_job_summary("job-running", JobStatus.RUNNING, result_available=False)
    succeeded = _tui_job_summary(
        "job-succeeded", JobStatus.SUCCEEDED, result_available=True
    )

    async def inspect_job(job_id: str) -> object | None:
        if job_id == running.job_id:
            return _tui_job_inspection(running)
        if job_id == succeeded.job_id:
            return _tui_job_inspection(succeeded)
        return None

    monkeypatch.setattr(app.controller, "inspect_job", inspect_job)

    assert {
        insertion
        for insertion, _display, _description in SLASH_COMMAND_COMPLETIONS
        if insertion.startswith("/jobs")
    } == {"/jobs", "/jobs inspect ", "/jobs results ", "/jobs cancel "}

    listed = await app.controller.dispatch_command("/jobs")
    assert listed.kind == "screen"
    assert listed.screen == "jobs"

    inspected = await app.controller.dispatch_command("/jobs inspect job-running")
    assert inspected.screen == "jobs"
    assert inspected.payload == {"job_id": "job-running", "view": "inspect"}

    results = await app.controller.dispatch_command("/jobs results job-succeeded")
    assert results.screen == "jobs"
    assert results.payload == {"job_id": "job-succeeded", "view": "results"}

    cancellation = await app.controller.dispatch_command("/jobs cancel job-running")
    assert cancellation.kind == "confirm"
    assert cancellation.screen == "confirm_cancel_job"
    assert cancellation.payload == {"job_id": "job-running"}
    assert "data_profile · running" in cancellation.message

    terminal = await app.controller.dispatch_command("/jobs cancel job-succeeded")
    assert terminal.kind == "notice"
    assert "succeeded and cannot be cancelled" in terminal.message

    malformed = await app.controller.dispatch_command("/jobs retry job-running")
    assert malformed.kind == "notice"
    assert malformed.message.startswith("Usage: /jobs")

    with pytest.raises(UserInputError, match="belongs to this agent"):
        await app.controller.dispatch_command("/jobs cancel job-missing")


async def test_inbox_command_routes_without_a_model_call():
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))

    assert (
        "/inbox",
        "/inbox",
        "Inspect and acknowledge completed background reports",
    ) in SLASH_COMMAND_COMPLETIONS
    outcome = await app.controller.dispatch_command("/inbox")
    assert outcome.kind == "screen"
    assert outcome.screen == "inbox"
    malformed = await app.controller.dispatch_command("/inbox extra")
    assert malformed.kind == "notice"
    assert malformed.message == "Usage: /inbox"


async def test_inbox_screen_inspects_sanitizes_and_acknowledges(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    item = _tui_inbox_item(report="Ready\x1b[31m @everyone")
    items = [item]
    acknowledgments: list[str] = []

    async def list_inbox() -> tuple[InboxView, ...]:
        return tuple(items)

    async def acknowledge_inbox(delivery_id: str) -> InboxView | None:
        acknowledgments.append(delivery_id)
        items.clear()
        return item

    async def list_distribution_destinations(
        conversation_id: str,
        *,
        sensitivity_ceiling: ModelSensitivity,
    ) -> tuple[SimpleNamespace, ...]:
        assert conversation_id == item.conversation_id
        assert sensitivity_ceiling is ModelSensitivity.INTERNAL
        return (
            SimpleNamespace(
                label="Current conversation Inbox",
                state=SimpleNamespace(value="available"),
                revision=1,
            ),
        )

    monkeypatch.setattr(app.controller, "list_inbox", list_inbox)
    monkeypatch.setattr(app.controller, "acknowledge_inbox", acknowledge_inbox)
    monkeypatch.setattr(
        app.controller,
        "list_distribution_destinations",
        list_distribution_destinations,
    )

    async with app.run_test(size=(110, 36)) as pilot:
        await app.push_screen(InboxScreen())
        for _ in range(20):
            await pilot.pause(0.05)
            if "1 unacknowledged result" in str(
                app.screen.query_one("#inbox-summary", Static).content
            ):
                break
        manager = app.screen
        assert isinstance(manager, InboxScreen)
        listing = manager.query_one("#inbox-list", OptionList)
        assert listing.option_count == 1
        assert listing.has_focus is True
        detail = str(manager.query_one("#inbox-detail", Static).content)
        assert "Ready?[31m @everyone" in detail
        assert "\x1b" not in detail
        assert "Result run: run-followup" in detail
        assert manager.query_one("#inbox-acknowledge", Button).disabled is False

        assert await pilot.click("#inbox-acknowledge") is True
        for _ in range(20):
            await pilot.pause(0.05)
            if "0 unacknowledged results" in str(
                manager.query_one("#inbox-summary", Static).content
            ):
                break
        assert acknowledgments == [item.delivery_id]
        assert listing.option_count == 0
        assert "never reruns reasoning" in str(
            manager.query_one("#inbox-help", Static).content
        )
        app.exit(0)


def test_inbox_rendering_withholds_blocked_reports_and_marks_bounded_previews():
    available = _tui_inbox_item(report="bounded preview")
    truncated = replace(
        available,
        conclusion_preview_truncated=True,
    )
    assert "Preview truncated" in render_inbox_item(truncated)

    blocked = replace(
        available,
        state=DeliveryState.BLOCKED,
        conclusion_preview="",
        blocked_reason_code="sensitivity_exceeds_destination",
    )
    rendered = render_inbox_item(blocked)
    assert "bounded preview" not in rendered
    assert "Preview withheld" in rendered
    assert "sensitivity_exceeds_destination" in rendered

    routine_escalation = replace(
        available,
        subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
        subject_id="occurrence-pre-run-failure",
        conclusion_kind=OutcomeConclusionKind.NO_MODEL_OCCURRENCE,
        conclusion_state=OutcomeState.FAILED,
        resulting_run_id=None,
        conclusion_preview="",
        failure_code="routine_precheck_unavailable",
    )
    escalation = render_inbox_item(routine_escalation)
    assert "Routine: occurrence-pre-run-failure" in escalation
    assert "Result run: No model run started" in escalation
    assert "Escalation" in escalation
    assert "failed before a model run could start" in escalation


async def test_background_status_notifies_once_and_remains_outside_transcript(
    tmp_path: Path,
    monkeypatch,
):
    opened = await Agent.create(
        "inbox-status-agent", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = opened
    running = _tui_job_summary(
        "job-running-status", JobStatus.RUNNING, result_available=False
    )
    current_inbox: list[InboxView] = []
    notifications: list[tuple[str, str | None]] = []

    async def list_jobs() -> tuple[object, ...]:
        return (running,)

    async def list_inbox() -> tuple[InboxView, ...]:
        return tuple(current_inbox)

    def notify(message: str, *, title: str | None = None, **_kwargs: object) -> None:
        notifications.append((message, title))

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "list_inbox", list_inbox)
    monkeypatch.setattr(app, "notify", notify)
    try:
        async with app.run_test(size=(110, 32)) as pilot:
            await app._show_chat()
            await pilot.pause()
            status = app.screen.query_one("#background-status", Static)
            assert "jobs 1" in str(status.content)
            assert status.display is True

            current_inbox.append(_tui_inbox_item())
            await app.refresh_background_status(notify_new=True)
            await pilot.pause()
            assert "jobs 1" in str(status.content)
            assert "inbox 1" in str(status.content)
            assert notifications == [
                ("1 background report is ready. Open /inbox to review.", "Inbox")
            ]
            await app.refresh_background_status(notify_new=True)
            assert len(notifications) == 1
            assert app.screen.query_one(TranscriptView).is_empty
            app.exit(0)
    finally:
        await opened.close()


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
                chat.query_one("#background-status", Static).content
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
            assert "ctx —" in str(context.content)
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
                chat.query_one("#background-status", Static).content
            )
            app.exit(0)
    finally:
        await opened.close()


async def test_jobs_manager_lists_inspects_reads_cancels_and_refreshes(monkeypatch):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    running = _tui_job_summary(
        "job-running-0123456789", JobStatus.RUNNING, result_available=False
    )
    succeeded = _tui_job_summary(
        "job-succeeded-0123456789", JobStatus.SUCCEEDED, result_available=True
    )
    jobs = [running, succeeded]
    list_calls = 0
    cancel_calls: list[str] = []

    async def list_jobs() -> tuple[object, ...]:
        nonlocal list_calls
        list_calls += 1
        return tuple(jobs)

    async def inspect_job(job_id: str) -> object | None:
        return next(
            (
                _tui_job_inspection(summary)
                for summary in jobs
                if summary.job_id == job_id
            ),
            None,
        )

    async def read_job_result(job_id: str) -> object | None:
        if job_id != succeeded.job_id:
            return None
        observed = datetime(2026, 8, 23, 14, 1, tzinfo=UTC)
        return SimpleNamespace(
            job_id=job_id,
            result_id="result-profile",
            summary=FrozenJsonObject.from_mapping({"profiled_resources": 1}),
            sensitivity=SimpleNamespace(value="internal"),
            provenance=FrozenJsonObject.from_mapping(
                {"authority": "job_owner_agent_scope"}
            ),
            artifact_refs=(),
            completed_at=observed,
        )

    async def cancel_job(job_id: str) -> object | None:
        cancel_calls.append(job_id)
        if job_id != running.job_id:
            return None
        cancelled = _tui_job_summary(
            running.job_id, JobStatus.CANCEL_REQUESTED, result_available=False
        )
        jobs[0] = cancelled
        return _tui_job_inspection(cancelled)

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "inspect_job", inspect_job)
    monkeypatch.setattr(app.controller, "read_job_result", read_job_result)
    monkeypatch.setattr(app.controller, "cancel_job", cancel_job)

    async with app.run_test(size=(110, 36)) as pilot:
        await app.push_screen(JobsScreen())
        for _ in range(20):
            await pilot.pause(0.05)
            if "2 jobs" in str(app.screen.query_one("#jobs-summary", Static).content):
                break
        manager = app.screen
        assert isinstance(manager, JobsScreen)
        panel = manager.query_one("#jobs-manager")
        assert panel.styles.border_left[0] == "solid"
        assert panel.styles.background.hex == "#111111"
        listing = manager.query_one("#jobs-list", OptionList)
        assert listing.option_count == 2
        assert listing.has_focus is True
        first_prompt = listing.get_option_at_index(0).prompt
        assert isinstance(first_prompt, Text)
        assert "RUNNING" in first_prompt.plain
        assert manager.query_one("#jobs-cancel", Button).disabled is False
        assert manager.query_one("#jobs-results", Button).disabled is True

        assert await pilot.click("#jobs-details") is True
        await pilot.pause()
        assert "Lifecycle" in str(manager.query_one("#jobs-detail", Static).content)

        assert await pilot.click("#jobs-cancel") is True
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen):
                break
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        for _ in range(20):
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


async def test_direct_jobs_cancel_command_confirms_and_opens_updated_manager(
    monkeypatch,
):
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(None))
    running = _tui_job_summary(
        "job-direct-cancel", JobStatus.RUNNING, result_available=False
    )
    current = running
    cancel_calls: list[str] = []

    async def list_jobs() -> tuple[object, ...]:
        return (current,)

    async def inspect_job(job_id: str) -> object | None:
        return _tui_job_inspection(current) if job_id == current.job_id else None

    async def cancel_job(job_id: str) -> object | None:
        nonlocal current
        cancel_calls.append(job_id)
        current = _tui_job_summary(
            job_id, JobStatus.CANCEL_REQUESTED, result_available=False
        )
        return _tui_job_inspection(current)

    async def skill_invocation_message(_message: str) -> None:
        return None

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "inspect_job", inspect_job)
    monkeypatch.setattr(app.controller, "cancel_job", cancel_job)
    monkeypatch.setattr(
        app.controller, "skill_invocation_message", skill_invocation_message
    )

    async with app.run_test(size=(110, 36)) as pilot:
        await app.push_screen(ChatScreen())
        composer = app.screen.query_one(Composer)
        composer.load_text(f"/jobs cancel {running.job_id}")
        composer.action_submit()
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, ConfirmScreen):
                break
        assert isinstance(app.screen, ConfirmScreen)
        assert "data_profile · running" in str(
            app.screen.query_one("#confirm-message").render()
        )
        await pilot.press("y")
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, JobsScreen):
                break
        manager = app.screen
        assert isinstance(manager, JobsScreen)
        assert cancel_calls == [running.job_id]
        assert "Cancellation requested" in str(
            manager.query_one("#jobs-notice", Static).content
        )
        app.exit(0)
