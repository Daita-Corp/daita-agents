"""The sources screen says in plain words how current a catalog is."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from textual.widgets import Static

from daita import SQLiteSource
from daita.catalog.models import (
    CatalogChange,
    CatalogChangeKind,
    CatalogChangeSide,
    CatalogFreshness,
    CatalogStatus,
    ResourceKind,
)
from daita.tui.app import DaitaApp
from daita.tui.catalog_status import status_line, today_line, when_text
from daita.tui.screens.catalog import CatalogScreen
from daita.tui.widgets.composer import Composer
from tests.support.host import StartHost, host_workspace, remote_agent
from tests.tui._support import settle_until
from tests.tui.test_catalog import _create_sqlite_source

NOW = datetime.now(UTC).replace(hour=15, minute=0, second=0, microsecond=0)
HASH = "sha256:" + "a" * 64


def _status(state: CatalogFreshness, **fields: object) -> CatalogStatus:
    values: dict[str, object] = {
        "source_id": "s",
        "state": state,
        "keep_current": state is not CatalogFreshness.PAUSED,
        "checked_at": None,
        "changed_at": None,
        "full_check_at": None,
        "resources": 1_500,
        "described": 800,
        "inventory_truncated": False,
        "error_code": None,
    }
    return CatalogStatus(**{**values, **fields})  # type: ignore[arg-type]


def _change(
    kind: CatalogChangeKind,
    identity: str,
    *,
    before: tuple[tuple[str, str], ...] = (),
    after: tuple[tuple[str, str], ...] = (),
    at: datetime = NOW,
) -> CatalogChange:
    name = identity.rpartition(".")[2]

    def side(columns: tuple[tuple[str, str], ...]) -> CatalogChangeSide:
        return CatalogChangeSide(name, ResourceKind.TABLE, HASH, columns)

    return CatalogChange(
        agent_id="a",
        source_id="s",
        sequence=0,
        sync_id="x",
        changed_at=at,
        kind=kind,
        resource_id="r",
        native_identity=identity,
        before=None if kind is CatalogChangeKind.ADDED else side(before),
        after=None if kind is CatalogChangeKind.REMOVED else side(after),
    )


@pytest.mark.unit
def test_times_are_relative_when_recent_and_day_words_when_old() -> None:
    assert when_text(NOW - timedelta(minutes=2), NOW) == "2 min ago"
    assert when_text(NOW - timedelta(seconds=5), NOW) == "just now"
    assert when_text(NOW - timedelta(days=1), NOW) == "yesterday"


@pytest.mark.unit
def test_each_state_reads_as_one_plain_line() -> None:
    current = _status(CatalogFreshness.CURRENT, checked_at=NOW - timedelta(minutes=2))
    assert status_line(current, NOW) == "Up to date · checked 2 min ago · 1,500 objects"
    assert (
        status_line(_status(CatalogFreshness.UPDATING), NOW)
        == "Updating · 800 of 1,500 objects described"
    )
    assert status_line(_status(CatalogFreshness.NOT_CHECKED), NOW) == "Not checked yet"
    assert (
        status_line(_status(CatalogFreshness.UNREACHABLE), NOW) == "Can't reach source"
    )
    paused = _status(CatalogFreshness.PAUSED, changed_at=NOW - timedelta(days=3))
    assert status_line(paused, NOW).startswith("Keep current is off · last refreshed ")


@pytest.mark.unit
def test_today_groups_added_tables_and_skips_earlier_days() -> None:
    added = CatalogChangeKind.ADDED
    changes = [
        _change(added, "sales.a"),
        _change(added, "sales.b"),
        _change(added, "sales.c"),
        _change(
            CatalogChangeKind.CHANGED,
            "sales.orders",
            before=(("status", "int"),),
            after=(("status", "text"),),
        ),
        _change(added, "sales.old", at=NOW - timedelta(days=2)),
    ]
    assert (
        today_line(changes, NOW)
        == "Today: 3 objects added in sales · orders.status changed type"
    )
    assert today_line(changes[-1:], NOW) == ""
    renamed = _change(CatalogChangeKind.CHANGED, "sales.purchases")
    assert renamed.before is not None
    renamed = replace(renamed, before=replace(renamed.before, name="orders"))
    assert today_line([renamed], NOW) == "Today: orders renamed to purchases"


@pytest.mark.unit
def test_today_caps_the_line_and_cleans_hostile_names() -> None:
    changes = [_change(CatalogChangeKind.ADDED, f"s{index}.t") for index in range(5)]
    assert today_line(changes, NOW).endswith(" and 2 more")
    hostile = _change(CatalogChangeKind.ADDED, "evil\x1b[31m\x07.t")
    assert "\x1b" not in today_line([hostile], NOW)
    assert "\x07" not in today_line([hostile], NOW)


@pytest.mark.tui_driven
async def test_sources_screen_shows_status_and_toggles_keep_current(
    tmp_path: Path, start_host: StartHost
) -> None:
    path = tmp_path / "sales.sqlite"
    _create_sqlite_source(path, "orders")
    harness = await start_host()
    opened = await remote_agent(harness, "catalog-status")
    source = await opened.attach(SQLiteSource(path, name="Sales"))
    app = DaitaApp(
        root=harness.root, start_bootstrap=False, workspace=host_workspace(harness)
    )
    app.controller.agent = opened
    try:
        async with app.run_test(size=(100, 34)) as pilot:
            await app._show_chat()
            await pilot.pause()
            composer = app.screen.query_one(Composer)
            composer.load_text("/sources")
            composer.action_submit()
            await settle_until(
                pilot,
                lambda: isinstance(app.screen, CatalogScreen)
                and bool(
                    str(app.screen.query_one("#source-catalog-status", Static).content)
                ),
            )
            line = str(app.screen.query_one("#source-catalog-status", Static).content)
            assert line.split(" · ")[0] in {"Up to date", "Updating", "Not checked yet"}

            await pilot.click("#source-keep-current")
            await settle_until(
                pilot,
                lambda: isinstance(app.screen, CatalogScreen)
                and bool(app.screen.query("#catalog-notice")),
            )
            assert (await opened.catalog_status(source.id)).keep_current is False
            app.exit(0)
    finally:
        await opened.close()
