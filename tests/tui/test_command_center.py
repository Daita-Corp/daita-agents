"""Command-center geometry and exact navigation through existing owner screens."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from textual.widgets import OptionList, Tab, Tabs

from daita import Agent, GraphState, RoutineState, ScheduledRoutineSummary
from daita.llm.models import ModelSensitivity
from daita.routines.models import ScheduleKind
from daita.tui.app import DaitaApp
from daita.tui.models import TranscriptBlock
from daita.tui.screens.artifacts import ArtifactsScreen
from daita.tui.screens.catalog import CatalogScreen
from daita.tui.screens.inbox import InboxScreen
from daita.tui.screens.jobs import JobsScreen
from daita.tui.screens.routines import RoutinesScreen
from daita.tui.widgets.command_center import (
    CommandSections,
    ProjectionList,
)
from daita.tui.widgets.composer import Composer
from daita.tui.widgets.transcript import TranscriptView
from tests.artifacts._public_surface_support import _create_artifact_agent
from tests.support.workspace import workspace_for
from tests.tui._support import (
    _create_sqlite_source,
    _tui_inbox_item,
    _tui_job_summary,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("size", [(80, 24), (100, 24), (140, 40)])
async def test_tabs_show_one_view_preserve_draft_and_fit_terminal(tmp_path, size):
    agent = await Agent.create(
        "geometry", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = agent
    async with app.run_test(size=size) as pilot:
        try:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.set_blocks(
                (TranscriptBlock("assistant", "answer", "Retained answer"),)
            )
            composer = chat.composer()
            composer.load_text("unsent question")
            await pilot.pause()
            transcript = chat.query_one(TranscriptView)
            sections = chat.query_one(CommandSections)
            assert transcript.region.width == size[0] - 2
            assert transcript.region.bottom <= composer.region.y
            assert not sections.display
            hints = list(chat.query("FooterKey"))
            assert [getattr(hint, "key") for hint in hints] == ["enter", "f6", "f1"]
            assert all(hint.region.right <= size[0] for hint in hints)

            for number, section in enumerate(
                ("sources", "work", "inbox", "artifacts"), 2
            ):
                await pilot.press(f"ctrl+{number}")
                await pilot.pause()
                assert chat.section == section
                assert sections.current == f"section-{section}"
                assert sections.display and not composer.display
                assert not chat.query_one("#conversation-pane").display
                assert sections.region.width == size[0] - 2
                assert chat.query_one(Tabs).region.bottom <= sections.region.y
                assert sections.region.bottom <= size[1] - 1
                assert composer.text == "unsent question"

            chat.append_block(TranscriptBlock("assistant", "new", "A live update"))
            assert chat.section == "artifacts"
            assert not chat.query_one("#conversation-pane").display
            await pilot.resize_terminal(40, 16)
            await pilot.press("ctrl+1")
            await pilot.pause()
            assert not composer.disabled
            assert composer.text == "unsent question"
            assert composer.region.right <= 40
            await pilot.press("f6", "right", "right", "right", "right")
            await pilot.pause()
            assert chat.section == "artifacts"
            await pilot.press("escape")
            await pilot.pause()
            assert chat.section == "chat" and chat.focused is composer
            assert not transcript.is_empty
            assert composer.text == "unsent question"
            await pilot.press("f1")
            await pilot.pause()
            assert composer.text == "unsent question"
            assert app._run_task is None
        finally:
            app.exit(0)


async def test_projection_refresh_preserves_focus_selection_and_untrusted_labels(
    tmp_path, monkeypatch
):
    agent = await Agent.create(
        "projection", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = agent
    jobs = [
        _tui_job_summary("job-one", GraphState.ACTIVE, result_available=False),
        _tui_job_summary("job-two", GraphState.QUEUED, result_available=False),
        _tui_job_summary("job-complete", GraphState.SUCCEEDED, result_available=True),
    ]
    jobs[0].specification.objective = "[bold]unsafe\x1b[2J"
    now = datetime.now(UTC)
    late = ScheduledRoutineSummary(
        routine_id="late",
        title="Later",
        state=RoutineState.ACTIVE,
        schedule_kind=ScheduleKind.ONCE,
        next_due_at=now + timedelta(hours=2),
        revision=1,
        occurrence_count=0,
        consecutive_failures=0,
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
    )
    early = replace(
        late, routine_id="early", title="Soon", next_due_at=now + timedelta(hours=1)
    )
    paused = replace(early, routine_id="paused", state=RoutineState.PAUSED)

    async def list_jobs():
        return tuple(jobs)

    async def list_routines():
        return (late, paused, early)

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "list_routines", list_routines)
    async with app.run_test(size=(140, 32)) as pilot:
        try:
            await app._show_chat()
            await pilot.pause()
            chat = app.chat()
            assert chat is not None
            await pilot.press("ctrl+3")
            await pilot.pause()
            listing = chat.query_one("#command-jobs", ProjectionList)
            assert listing.option_count == 2
            prompt = listing.get_option_at_index(0).prompt
            assert "[bold]" in str(prompt) and "\x1b" not in str(prompt)
            listing.highlighted = 1
            listing.focus()
            jobs.insert(
                0,
                _tui_job_summary("new-job", GraphState.ACTIVE, result_available=False),
            )
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            assert chat.focused is listing
            assert listing.get_option_at_index(listing.highlighted).id == "job-two"
            routines = chat.query_one("#command-routines", ProjectionList)
            assert [
                routines.get_option_at_index(i).id for i in range(routines.option_count)
            ] == ["early", "late"]
            assert str(chat.query_one("#tab-work", Tab).label) == "Work 3"
            await pilot.press("escape")
            assert chat.focused is chat.composer()

        finally:
            app.exit(0)


@pytest.mark.parametrize(
    "section", ["sources", "jobs", "routines", "inbox", "artifacts"]
)
async def test_projection_opens_exact_owner_item_and_restores_composer(
    tmp_path, monkeypatch, section
):
    # A real registered artifact keeps navigation tied to current inventory.
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    agent, artifact = await _create_artifact_agent(tmp_path, "navigation", downloads)
    source_path = tmp_path / "records.sqlite"
    _create_sqlite_source(source_path, "records")
    source = await agent.attach_sqlite(source_path, name="Records")
    app = DaitaApp(
        root=tmp_path, start_bootstrap=False, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = agent
    job = _tui_job_summary("job-target", GraphState.ACTIVE, result_available=False)
    routine = ScheduledRoutineSummary(
        routine_id="routine-target",
        title="Next report",
        state=RoutineState.ACTIVE,
        schedule_kind=ScheduleKind.ONCE,
        next_due_at=datetime.now(UTC) + timedelta(hours=1),
        revision=1,
        occurrence_count=0,
        consecutive_failures=0,
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
    )
    item = _tui_inbox_item("delivery-target")

    async def list_jobs():
        return (job,)

    async def list_routines():
        return (routine,)

    async def list_inbox():
        return (item,)

    monkeypatch.setattr(app.controller, "list_jobs", list_jobs)
    monkeypatch.setattr(app.controller, "list_routines", list_routines)
    monkeypatch.setattr(app.controller, "list_inbox", list_inbox)
    targets = {
        "sources": (CatalogScreen, "#catalog-tree", source.id),
        "jobs": (JobsScreen, "#jobs-list", job.job_id),
        "routines": (RoutinesScreen, "#routines-list", routine.routine_id),
        "inbox": (InboxScreen, "#inbox-list", item.delivery_id),
        "artifacts": (ArtifactsScreen, "#artifacts-list", artifact.artifact_id),
    }
    async with app.run_test(size=(140, 40)) as pilot:
        try:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.composer().load_text("keep this draft")
            chat.action_section("work" if section in {"jobs", "routines"} else section)
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            listing = chat.query_one(f"#command-{section}", ProjectionList)
            await pilot.click(listing, offset=(2, 0))
            await pilot.pause()
            screen_type, selector, expected_id = targets[section]
            assert isinstance(app.screen, screen_type)
            if section == "sources":
                assert app.screen._selected_source_id() == expected_id
            else:
                selected = app.screen.query_one(selector, OptionList)
                assert selected.highlighted is not None
                assert (
                    selected.get_option_at_index(selected.highlighted).id == expected_id
                )
            await pilot.press("escape")
            await pilot.pause()
            assert app.chat() is chat
            assert chat.focused is chat.composer()
            assert chat.composer().text == "keep this draft"
        finally:
            app.exit(0)
