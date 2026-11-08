"""Catalog upkeep through a real agent: probes, status, keep current and fill."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from daita import Agent, SQLiteSource
from daita.adapters.sqlite import SQLiteResourceAdapter
from daita.catalog.models import CatalogFreshness
from daita.host.registry import _has_background_work
from daita.hosting import embedded as embedded_module
from tests.support.source_refresh import settled, stale_probes, sync_count
from tests.support.workspace import workspace_for


def _database(path: Path, tables: int = 1) -> None:
    with sqlite3.connect(path) as connection:
        for index in range(tables):
            connection.execute(f"CREATE TABLE t{index:02d} (id INTEGER)")


def _add_table(path: Path, name: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE {name} (id INTEGER)")


async def _names(agent: Agent, source_id: str) -> set[str]:
    return {
        item.name for item in await agent.list_catalog_resources(source_id=source_id)
    }


async def _agent(tmp_path: Path, name: str = "keeper") -> Agent:
    return await Agent.create(name, root=tmp_path, workspace=workspace_for(tmp_path))


async def test_a_table_added_to_the_file_reaches_the_catalog_without_a_refresh(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shop.sqlite"
    _database(database)
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        _add_table(database, "invoices")
        stale_probes(agent)
        agent._embedded.keeper.use((source.id,))
        await settled(agent)

        assert "invoices" in await _names(agent, source.id)
        status = await agent.catalog_status(source.id)
        assert status.state is CatalogFreshness.CURRENT
        assert status.resources == status.described == 2
        assert status.checked_at is not None and status.changed_at is not None
        kinds = {
            (c.kind.value, c.native_identity)
            for c in await agent.catalog_changes(source.id)
        }
        assert ("added", "main.invoices") in kinds
    finally:
        await agent.close()


async def test_data_writes_to_the_file_cause_no_catalog_read(tmp_path: Path) -> None:
    database = tmp_path / "shop.sqlite"
    _database(database)
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        reads = sync_count(agent)
        with sqlite3.connect(database) as connection:
            connection.execute("INSERT INTO t00 VALUES (1)")
        stale_probes(agent)
        await agent.catalog_status(source.id, check=True)
        await settled(agent)
        assert sync_count(agent) == reads
    finally:
        await agent.close()


async def test_keep_current_off_pauses_upkeep_and_on_resumes_it(tmp_path: Path) -> None:
    database = tmp_path / "shop.sqlite"
    _database(database)
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        paused = await agent.set_keep_current(source.id, False)
        assert paused.state is CatalogFreshness.PAUSED and not paused.keep_current

        _add_table(database, "invoices")
        stale_probes(agent)
        await agent.catalog_status(source.id, check=True)
        await settled(agent)
        assert "invoices" not in await _names(agent, source.id)

        resumed = await agent.set_keep_current(source.id, True)
        assert resumed.keep_current
        stale_probes(agent)
        await agent.catalog_status(source.id, check=True)
        await settled(agent)
        assert "invoices" in await _names(agent, source.id)
    finally:
        await agent.close()


async def test_status_of_an_unknown_source_is_a_value_error(tmp_path: Path) -> None:
    agent = await _agent(tmp_path)
    try:
        for call in (
            agent.catalog_status("source:sha256:" + "0" * 64),
            agent.catalog_changes("source:sha256:" + "0" * 64),
            agent.set_keep_current("source:sha256:" + "0" * 64, False),
        ):
            with pytest.raises(ValueError):
                await call
    finally:
        await agent.close()


async def test_an_unreachable_source_reports_a_code_and_keeps_its_catalog(
    tmp_path: Path,
) -> None:
    database = tmp_path / "private-name.sqlite"
    _database(database)
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        database.unlink()
        for _ in range(3):
            stale_probes(agent)
            agent._embedded.keeper.use((source.id,))
            keeper = agent._embedded.keeper
            keeper._mems[source.id].next_due = None
            await settled(agent)
            keeper._mems[source.id].next_due = None
        status = await agent.catalog_status(source.id)
        assert status.state is CatalogFreshness.UNREACHABLE
        assert status.error_code and "private-name" not in repr(status)
        assert "t00" in await _names(agent, source.id)
    finally:
        await agent.close()


async def test_a_persons_detach_cancels_a_background_read_and_does_not_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "shop.sqlite"
    _database(database)
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        await settled(agent)
        _add_table(database, "late")
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def stuck(self, request):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield  # pragma: no cover

        monkeypatch.setattr(SQLiteResourceAdapter, "reconcile", stuck)
        stale_probes(agent)
        agent._embedded.keeper.use((source.id,))
        await asyncio.wait_for(entered.wait(), timeout=5)

        await asyncio.wait_for(agent.detach(source.id), timeout=5)
        assert cancelled.is_set()
    finally:
        await agent.close()


async def test_an_unfinished_fill_resumes_after_the_agent_is_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "shop.sqlite"
    _database(database, tables=8)
    monkeypatch.setattr(embedded_module, "FILL_RESOURCES", 3)
    agent = await _agent(tmp_path)
    try:
        # Hold the keeper off so the attach is the only pass that has run.
        async with agent._embedded.keeper.person_work():
            source = await agent.attach(SQLiteSource(database, name="Shop"))
            before = await agent.catalog_status(source.id)
        assert (before.resources, before.described) == (8, 3)
        assert before.inventory_truncated
    finally:
        await agent.close()

    agent = await Agent.open("keeper", root=tmp_path, workspace=workspace_for(tmp_path))
    try:
        agent._embedded.keeper._fill_resources = 3
        await settled(agent)
        after = await agent.catalog_status(source.id)
        assert (after.resources, after.described) == (8, 8)
        assert not after.inventory_truncated
        assert after.state is CatalogFreshness.CURRENT
    finally:
        await agent.close()


async def test_an_idle_agent_with_keep_current_on_still_counts_as_idle(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shop.sqlite"
    _database(database)
    agent = await _agent(tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database, name="Shop"))
        assert (await agent.catalog_status(source.id)).keep_current
        assert await _has_background_work(agent) is False
        await asyncio.wait_for(agent.close(), timeout=5)
    finally:
        await agent.close()
