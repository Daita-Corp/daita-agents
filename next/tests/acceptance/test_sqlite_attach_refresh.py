from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from daita import Agent, SQLiteSource
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 18, 23, 0, tzinfo=timezone.utc)


class AdvancingClock:
    def __init__(self) -> None:
        self._value = NOW

    def __call__(self) -> datetime:
        current = self._value
        self._value += timedelta(seconds=1)
        return current


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY)")


async def test_public_reattach_validates_declarations_and_preserves_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    agent = await Agent.create(
        "refresh",
        root=tmp_path / "state",
        clock=AdvancingClock(),
    )

    first = await agent.attach(SQLiteSource(database))
    second = await agent.attach(SQLiteSource(database))
    home = agent.home
    await agent.close()

    assert second == first
    store = await SQLiteOperationStore.open(home / "state.db")
    try:
        resources = await store.list_resources(agent.id, first.id)
    finally:
        await store.close()
    assert len(resources) == 1
    assert resources[0].first_observed_at < resources[0].last_observed_at
