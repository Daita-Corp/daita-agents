"""The Work tab shows the work board, opens cards and dismisses results."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest
from textual.widgets import Button, OptionList, Static, Tab

from daita import (
    Agent,
    DeliveryInspection,
    DeliveryState,
    DeliverySubjectKind,
    GraphState,
    OutcomeConclusionKind,
    OutcomeState,
    RoutineState,
    ScheduledRoutineSummary,
    WorkBoard,
    WorkCard,
    WorkColumn,
    WorkKind,
)
from daita.distribution import delivery_view
from daita.distribution.owner import construct_logical_delivery
from daita.llm.models import ModelSensitivity
from daita.routines.models import ScheduleKind
from daita.tui.app import DaitaApp
from daita.tui.commands import SLASH_COMMAND_COMPLETIONS
from daita.tui.screens.jobs import JobsScreen
from daita.tui.screens.result import ResultScreen, render_result
from daita.tui.screens.routines import RoutinesScreen
from daita.tui.widgets.command_center import WorkBoardView
from daita.tui.widgets.transcript import TranscriptView
from tests.support.distribution import inbox_distribution_plan
from tests.support.workspace import workspace_for
from tests.tui._support import _tui_delivery_view, _tui_job_summary

NOW = datetime(2026, 10, 18, 12, tzinfo=UTC)


def _card(
    kind: WorkKind,
    card_id: str,
    column: WorkColumn,
    *,
    title: str = "Weekly revenue report",
    status: str = "active",
    summary: str = "",
    started_at: datetime | None = None,
    due_at: datetime | None = None,
    routine_id: str | None = None,
    delivery_id: str | None = None,
    updated_at: datetime = NOW - timedelta(hours=2),
) -> WorkCard:
    return WorkCard(
        kind=kind,
        id=card_id,
        column=column,
        title=title,
        status=status,
        summary=summary,
        updated_at=updated_at,
        started_at=started_at,
        due_at=due_at,
        routine_id=routine_id,
        delivery_id=delivery_id,
    )


def _board(
    *cards: WorkCard, truncated: dict[WorkColumn, int] | None = None
) -> WorkBoard:
    return WorkBoard(
        columns=MappingProxyType(
            {
                column: tuple(card for card in cards if card.column is column)
                for column in WorkColumn
            }
        ),
        truncated=MappingProxyType(
            {column: (truncated or {}).get(column, 0) for column in WorkColumn}
        ),
        generated_at=NOW,
    )


def _result_inspection(delivery_id: str = "delivery-1") -> DeliveryInspection:
    (target,) = inbox_distribution_plan("conversation-1").targets
    return DeliveryInspection(
        delivery=construct_logical_delivery(
            delivery_id=delivery_id,
            agent_id="agent-1",
            conversation_id="conversation-1",
            subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
            subject_id="occurrence-1",
            target=target,
            conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
            conclusion_state=OutcomeState.SUCCEEDED,
            conclusion_id="run-routine",
            conclusion_digest="sha256:" + "1" * 64,
            conclusion_preview="Ready\x1b[31m @everyone",
            conclusion_preview_truncated=False,
            resulting_run_id="run-routine",
            artifact_references=(),
            effective_sensitivity=ModelSensitivity.INTERNAL,
            provenance_digest="sha256:" + "2" * 64,
            failure_code=None,
            observed_at=NOW - timedelta(hours=1),
        )
    )


async def _app_with_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boards: list[WorkBoard]
) -> tuple[DaitaApp, Agent]:
    agent = await Agent.create(
        "work-board-tui", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = agent

    async def work_board() -> WorkBoard:
        return boards[-1]

    monkeypatch.setattr(app.controller, "work_board", work_board)
    return app, agent


def _option_ids(listing: OptionList) -> list[str | None]:
    return [listing.get_option_at_index(i).id for i in range(listing.option_count)]


async def test_work_command_routes_to_the_work_tab_without_a_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        "/work",
        "/work",
        "See scheduled, running, waiting and finished work",
    ) in SLASH_COMMAND_COMPLETIONS
    assert not any(item[0].startswith("/inbox") for item in SLASH_COMMAND_COMPLETIONS)
    app, agent = await _app_with_board(tmp_path, monkeypatch, [_board()])
    outcome = await app.controller.dispatch_command("/work")
    assert (outcome.kind, outcome.screen) == ("screen", "work")
    malformed = await app.controller.dispatch_command("/work extra")
    assert (malformed.kind, malformed.message) == ("notice", "Usage: /work")
    try:
        async with app.run_test(size=(140, 36)) as pilot:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            await app.submit_composer("/work")
            await pilot.pause()
            assert chat.section == "work"
            assert chat.query_one(WorkBoardView).display
            await pilot.press("ctrl+1")
            await pilot.pause()
            assert chat.section == "chat"
            await pilot.press("ctrl+3")
            await pilot.pause()
            assert chat.section == "work"
            assert isinstance(chat.focused, OptionList)
            assert chat.focused.id == "work-scheduled"
            assert app.screen.query_one(TranscriptView).is_empty
            app.exit(0)
    finally:
        await agent.close()


async def test_board_shows_four_columns_with_counts_truncation_and_empty_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = _board(
        _card(WorkKind.JOB, "job-running", WorkColumn.RUNNING, started_at=NOW),
        _card(
            WorkKind.JOB,
            "job-blocked",
            WorkColumn.NEEDS_YOU,
            status="blocked",
            summary="Waiting for you",
        ),
        _card(
            WorkKind.ROUTINE_RUN,
            "occurrence-failed",
            WorkColumn.NEEDS_YOU,
            status="failed",
            summary="routine_source_unavailable",
            delivery_id="delivery-failed",
        ),
        truncated={WorkColumn.NEEDS_YOU: 3},
    )
    app, agent = await _app_with_board(tmp_path, monkeypatch, [board])
    try:
        async with app.run_test(size=(140, 36)) as pilot:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.action_section("work")
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            headers = {
                column: str(
                    chat.query_one(f"#work-header-{column.value}", Static).content
                )
                for column in WorkColumn
            }
            assert headers == {
                WorkColumn.SCHEDULED: "SCHEDULED · 0",
                WorkColumn.RUNNING: "RUNNING · 1",
                WorkColumn.NEEDS_YOU: "NEEDS YOU · 5",
                WorkColumn.DONE: "DONE · 0",
            }
            assert chat.query_one("#work-header-needs_you").has_class("-attention")
            needs_you = chat.query_one("#work-needs_you", OptionList)
            assert _option_ids(needs_you) == [
                "job:job-blocked",
                "routine_run:occurrence-failed",
                None,
            ]
            assert str(needs_you.get_option_at_index(2).prompt) == "+3 more"
            scheduled = chat.query_one("#work-scheduled", OptionList)
            assert str(scheduled.get_option_at_index(0).prompt) == "Nothing here"
            assert scheduled.get_option_at_index(0).disabled
            blocked = str(needs_you.get_option_at_index(0).prompt)
            assert blocked.splitlines() == [
                "Weekly revenue report",
                "blocked · 2h ago · Waiting for you",
            ]
            running = str(
                chat.query_one("#work-running", OptionList)
                .get_option_at_index(0)
                .prompt
            )
            assert running.splitlines()[1] == "active · 0m 00s"
            assert str(chat.query_one("#tab-work", Tab).label) == "Work 5"
            assert chat.query_one("#tab-work").tooltip == (
                "1 running · 0 scheduled · 5 need you"
            )
            assert "5 need you" in str(chat.query_one("#status-bar").tooltip)
            app.exit(0)
    finally:
        await agent.close()


@pytest.mark.parametrize("width,narrow", [(140, False), (90, True)])
async def test_board_uses_a_two_by_two_grid_on_narrow_terminals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, narrow: bool
) -> None:
    app, agent = await _app_with_board(tmp_path, monkeypatch, [_board()])
    try:
        async with app.run_test(size=(width, 36)) as pilot:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.action_section("work")
            await pilot.pause()
            view = chat.query_one(WorkBoardView)
            assert view.has_class("-narrow") is narrow
            columns = [
                chat.query_one(f"#work-{column.value}", OptionList).region
                for column in WorkColumn
            ]
            if narrow:
                assert columns[0].y == columns[1].y < columns[2].y == columns[3].y
            else:
                assert len({region.y for region in columns}) == 1
            assert all(region.right <= width for region in columns)
            app.exit(0)
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "card,screen_type,selector,expected_id",
    [
        (
            _card(WorkKind.JOB, "job-target", WorkColumn.RUNNING, started_at=NOW),
            JobsScreen,
            "#jobs-list",
            "job-target",
        ),
        (
            _card(
                WorkKind.ROUTINE,
                "routine-target",
                WorkColumn.SCHEDULED,
                routine_id="routine-target",
                due_at=NOW + timedelta(hours=1),
            ),
            RoutinesScreen,
            "#routines-list",
            "routine-target",
        ),
        (
            _card(
                WorkKind.ROUTINE_RUN,
                "occurrence-running",
                WorkColumn.RUNNING,
                routine_id="routine-target",
                started_at=NOW,
            ),
            RoutinesScreen,
            "#routines-list",
            "routine-target",
        ),
        (
            _card(
                WorkKind.ROUTINE_RUN,
                "occurrence-1",
                WorkColumn.DONE,
                routine_id="routine-target",
                delivery_id="delivery-1",
                status="succeeded",
            ),
            ResultScreen,
            None,
            "delivery-1",
        ),
    ],
)
async def test_enter_opens_each_card_kind_in_its_owner_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    card: WorkCard,
    screen_type: type,
    selector: str | None,
    expected_id: str,
) -> None:
    app, agent = await _app_with_board(tmp_path, monkeypatch, [_board(card)])
    job = _tui_job_summary("job-target", GraphState.ACTIVE, result_available=False)
    routine = ScheduledRoutineSummary(
        routine_id="routine-target",
        title="Next report",
        state=RoutineState.ACTIVE,
        schedule_kind=ScheduleKind.ONCE,
        next_due_at=NOW + timedelta(hours=1),
        revision=1,
        occurrence_count=0,
        consecutive_failures=0,
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
    )

    async def list_jobs():
        return (job,)

    async def list_routines():
        return (routine,)

    async def inspect_delivery(delivery_id: str) -> DeliveryInspection | None:
        return _result_inspection(delivery_id)

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "list_routines", list_routines)
    monkeypatch.setattr(app.controller, "inspect_delivery", inspect_delivery)
    try:
        async with app.run_test(size=(140, 40)) as pilot:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.composer().load_text("keep this draft")
            chat.action_section("work")
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            chat.query_one(f"#work-{card.column.value}", OptionList).focus()
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.02)
                if isinstance(app.screen, screen_type):
                    break
            assert isinstance(app.screen, screen_type)
            if selector is None:
                for _ in range(100):
                    await pilot.pause(0.02)
                    if "Result run" in str(
                        app.screen.query_one("#result-detail", Static).content
                    ):
                        break
                screen = app.screen
                assert isinstance(screen, ResultScreen)
                assert screen._delivery_id == expected_id
            else:
                selected = app.screen.query_one(selector, OptionList)
                for _ in range(100):
                    await pilot.pause(0.02)
                    if selected.highlighted is not None:
                        break
                highlighted = selected.highlighted
                assert highlighted is not None
                assert selected.get_option_at_index(highlighted).id == expected_id
            await pilot.press("escape")
            await pilot.pause()
            assert app.chat() is chat
            assert chat.composer().text == "keep this draft"
            app.exit(0)
    finally:
        await agent.close()


async def test_d_dismisses_only_cards_that_carry_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _card(
        WorkKind.ROUTINE_RUN,
        "occurrence-done",
        WorkColumn.DONE,
        status="succeeded",
        routine_id="routine-1",
        delivery_id="delivery-done",
    )
    job = _card(WorkKind.JOB, "job-done", WorkColumn.DONE, status="succeeded")
    boards = [_board(result, job)]
    dismissed: list[str] = []

    async def acknowledge_delivery(delivery_id: str):
        dismissed.append(delivery_id)
        boards.append(_board(job))
        return _tui_delivery_view(delivery_id)

    app, agent = await _app_with_board(tmp_path, monkeypatch, boards)
    monkeypatch.setattr(app.controller, "acknowledge_delivery", acknowledge_delivery)
    try:
        async with app.run_test(size=(140, 36)) as pilot:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.action_section("work")
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            done = chat.query_one("#work-done", OptionList)
            done.focus()
            done.highlighted = _option_ids(done).index("job:job-done")
            await pilot.pause()
            view = chat.query_one(WorkBoardView)
            assert view.check_action("dismiss_card", ()) is False
            await pilot.press("d")
            await pilot.pause()
            assert dismissed == []

            done.highlighted = _option_ids(done).index("routine_run:occurrence-done")
            await pilot.pause()
            assert view.check_action("dismiss_card", ()) is True
            assert "d" in {
                key for key, value in chat.active_bindings.items() if value.binding.show
            }
            await pilot.press("d")
            for _ in range(100):
                await pilot.pause(0.02)
                if _option_ids(done) == ["job:job-done"]:
                    break
            assert dismissed == ["delivery-done"]
            assert _option_ids(done) == ["job:job-done"]
            app.exit(0)
    finally:
        await agent.close()


async def test_selection_survives_refresh_and_card_text_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _card(
        WorkKind.JOB,
        "job-one",
        WorkColumn.SCHEDULED,
        title="[bold]unsafe\x1b[2J",
        status="queued",
        summary="progress\x1b[31m [red]note[/red]",
    )
    second = _card(WorkKind.JOB, "job-two", WorkColumn.SCHEDULED, status="queued")
    boards = [_board(first, second)]
    app, agent = await _app_with_board(tmp_path, monkeypatch, boards)
    try:
        async with app.run_test(size=(140, 36)) as pilot:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.action_section("work")
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            scheduled = chat.query_one("#work-scheduled", OptionList)
            prompt = str(scheduled.get_option_at_index(0).prompt)
            assert "[bold]" in prompt and "[red]note[/red]" in prompt
            assert "\x1b" not in prompt
            assert prompt.count("\n") == 1
            scheduled.focus()
            scheduled.highlighted = 1
            boards.append(
                _board(
                    _card(WorkKind.JOB, "job-new", WorkColumn.SCHEDULED),
                    first,
                    second,
                )
            )
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            assert chat.focused is scheduled
            assert scheduled.get_option_at_index(scheduled.highlighted).id == (
                "job:job-two"
            )
            app.exit(0)
    finally:
        await agent.close()


async def test_new_needs_you_or_done_cards_notify_once_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waiting = _card(WorkKind.JOB, "job-waiting", WorkColumn.NEEDS_YOU, status="blocked")
    boards = [_board(waiting)]
    app, agent = await _app_with_board(tmp_path, monkeypatch, boards)
    notifications: list[tuple[str, str | None]] = []

    def notify(message: str, *, title: str | None = None, **_kwargs: object) -> None:
        notifications.append((message, title))

    monkeypatch.setattr(app, "notify", notify)
    try:
        async with app.run_test(size=(110, 32)) as pilot:
            await app._show_chat()
            await pilot.pause()
            await app.refresh_background_status(notify_new=True)
            assert notifications == []

            boards.append(
                _board(
                    waiting,
                    _card(WorkKind.JOB, "job-new-1", WorkColumn.NEEDS_YOU),
                    _card(WorkKind.JOB, "job-new-2", WorkColumn.NEEDS_YOU),
                    _card(WorkKind.JOB, "job-finished", WorkColumn.DONE),
                )
            )
            await app.refresh_background_status(notify_new=True)
            await app.refresh_background_status(notify_new=True)
            assert notifications == [("2 items need you. Open /work.", "Work")]

            boards.append(
                replace(
                    boards[-1],
                    columns=MappingProxyType(
                        {
                            **boards[-1].columns,
                            WorkColumn.DONE: (
                                *boards[-1].columns[WorkColumn.DONE],
                                _card(WorkKind.JOB, "job-later", WorkColumn.DONE),
                            ),
                        }
                    ),
                )
            )
            await app.refresh_background_status(notify_new=True)
            assert notifications[1:] == [("1 item finished. Open /work.", "Work")]
            assert app.screen.query_one(TranscriptView).is_empty
            app.exit(0)
    finally:
        await agent.close()


async def test_result_screen_shows_a_sanitized_result_and_dismisses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, agent = await _app_with_board(tmp_path, monkeypatch, [_board()])
    inspection = _result_inspection()
    dismissed: list[str] = []

    async def inspect_delivery(delivery_id: str) -> DeliveryInspection | None:
        return inspection if delivery_id == "delivery-1" else None

    async def acknowledge_delivery(delivery_id: str):
        dismissed.append(delivery_id)
        return delivery_view(inspection.delivery)

    monkeypatch.setattr(app.controller, "inspect_delivery", inspect_delivery)
    monkeypatch.setattr(app.controller, "acknowledge_delivery", acknowledge_delivery)
    try:
        async with app.run_test(size=(110, 36)) as pilot:
            await app.push_screen(ResultScreen(delivery_id="delivery-missing"))
            for _ in range(100):
                await pilot.pause(0.02)
                if str(app.screen.query_one("#result-error", Static).content):
                    break
            assert "no longer exists" in str(
                app.screen.query_one("#result-error", Static).content
            )
            assert app.screen.query_one("#result-dismiss", Button).disabled
            await pilot.press("escape")
            await pilot.pause()

            screen = ResultScreen(delivery_id="delivery-1")
            await app.push_screen(screen)
            for _ in range(100):
                await pilot.pause(0.02)
                if "Result run" in str(
                    screen.query_one("#result-detail", Static).content
                ):
                    break
            detail = str(screen.query_one("#result-detail", Static).content)
            assert "Ready?[31m @everyone" in detail
            assert "\x1b" not in detail
            assert "Result run: run-routine" in detail
            assert "never reruns" in str(
                screen.query_one("#result-help", Static).content
            )
            assert screen.query_one("#result-dismiss", Button).disabled is False
            assert await pilot.click("#result-dismiss") is True
            for _ in range(100):
                await pilot.pause(0.02)
                if app.screen is not screen:
                    break
            assert dismissed == ["delivery-1"]
            assert app.screen is not screen
            app.exit(0)
    finally:
        await agent.close()


def test_result_rendering_withholds_blocked_reports_and_marks_bounded_previews():
    available = _tui_delivery_view(report="bounded preview")
    truncated = replace(
        available,
        conclusion_preview_truncated=True,
    )
    assert "Preview truncated" in render_result(truncated)

    blocked = replace(
        available,
        state=DeliveryState.BLOCKED,
        conclusion_preview="",
        blocked_reason_code="sensitivity_exceeds_destination",
    )
    rendered = render_result(blocked)
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
    escalation = render_result(routine_escalation)
    assert "Routine: occurrence-pre-run-failure" in escalation
    assert "Result run: No model run started" in escalation
    assert "Escalation" in escalation
    assert "failed before a model run could start" in escalation
