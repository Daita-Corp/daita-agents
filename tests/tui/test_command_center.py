"""Command-center geometry and exact navigation through existing owner screens."""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.widgets import OptionList, Static, Tab, Tabs

from daita import Agent
from daita.tui.app import DaitaApp
from daita.tui.models import TranscriptBlock
from daita.tui.screens.artifacts import ArtifactsScreen
from daita.tui.screens.catalog import CatalogScreen
from daita.tui.screens.help import HelpScreen
from daita.tui.widgets.command_center import (
    CommandSections,
    ProjectionList,
)
from daita.tui.widgets.transcript import TranscriptView
from tests.artifacts._public_surface_support import _create_artifact_agent
from tests.support.workspace import workspace_for
from tests.tui._support import (
    _create_sqlite_source,
    settle_until,
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
            await settle_until(pilot, lambda: bool(chat.query("FooterKey")))
            navigation = chat.query_one(Tabs)
            identity = chat.query_one("#status-primary")
            context = chat.query_one("#context-window")
            assert navigation.region.y == identity.region.y
            assert navigation.region.height == identity.region.height == 1
            assert identity.region.right <= navigation.region.x
            # Context usage appears only once a model request has measured it.
            assert not context.display
            if context.display:
                assert context.region.y == identity.region.y
                assert navigation.region.right <= context.region.x
                assert context.region.right <= size[0]
            for tab in navigation.query(Tab):
                assert tab.region.y == identity.region.y
                assert navigation.region.x <= tab.region.x
                assert tab.region.right <= navigation.region.right
            transcript = chat.query_one(TranscriptView)
            sections = chat.query_one(CommandSections)
            assert transcript.region.width == size[0] - 2
            assert transcript.region.bottom <= composer.region.y
            assert not sections.display
            hints = list(chat.query("FooterKey"))
            assert [getattr(hint, "key") for hint in hints] == [
                "enter",
                "ctrl+o",
                "f6",
                "f1",
            ]
            assert all(hint.region.right <= size[0] for hint in hints)

            for number, section in enumerate(("sources", "work", "artifacts"), 2):
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
            assert navigation.region.y == identity.region.y
            assert not context.display
            assert navigation.region.right <= 40
            assert not composer.disabled
            assert composer.text == "unsent question"
            assert composer.region.right <= 40
            await pilot.press("f6", "right", "right", "right")
            await pilot.pause()
            assert chat.section == "artifacts"
            await pilot.press("escape")
            await pilot.pause()
            assert chat.section == "chat" and chat.focused is composer
            assert not transcript.is_empty
            assert composer.text == "unsent question"
            # Resize and the 2s background poll both refresh status from the
            # controller; give it the same model so a late refresh agrees.
            model = "provider/long-model-name-with-extra-identifiers"
            app.controller.model_label = lambda: model  # type: ignore[method-assign]
            await pilot.resize_terminal(*size)
            chat.set_status(
                agent="A long agent name that needs truncation",
                model=model,
                source="All readable sources",
                state="ready",
                source_count=50,
                running=50,
                needs_you=50,
            )
            await pilot.pause()
            assert navigation.region.y == identity.region.y
            assert isinstance(identity.tooltip, Text)
            assert "provider/long-model-name" in identity.tooltip.plain
            for tab in navigation.query(Tab):
                assert tab.region.right <= navigation.region.right
            await pilot.press("f1")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is chat
            assert composer.text == "unsent question"
            assert app._run_task is None
        finally:
            app.exit(0)


@pytest.mark.parametrize("section", ["sources", "artifacts"])
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
    targets = {
        "sources": (CatalogScreen, "#catalog-tree", source.id),
        "artifacts": (ArtifactsScreen, "#artifacts-list", artifact.artifact_id),
    }
    async with app.run_test(size=(140, 40)) as pilot:
        try:
            await app._show_chat()
            chat = app.chat()
            assert chat is not None
            chat.composer().load_text("keep this draft")
            chat.action_section(section)
            await app.refresh_background_status(notify_new=False)
            await pilot.pause()
            listing = chat.query_one(f"#command-{section}", ProjectionList)
            # Wide terminals show the highlighted item's facts beside the list.
            detail = chat.query_one(f"#detail-{section}", Static)
            assert detail.display and detail.region.x > listing.region.x
            expected_detail = "SQLite" if section == "sources" else "Size"
            assert expected_detail in str(detail.content)
            await pilot.resize_terminal(90, 40)
            await pilot.pause()
            assert detail.region.width == 0
            await pilot.resize_terminal(140, 40)
            await pilot.pause()
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
