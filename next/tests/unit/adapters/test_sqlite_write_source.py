from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from daita.adapters import SQLiteSource
from daita.domains.data.controller import (
    SQLITE_QUERY_CAPABILITY_ID,
    SQLITE_UPDATE_CAPABILITY_ID,
    SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE records (id TEXT PRIMARY KEY, status TEXT NOT NULL)"
        )


async def test_sqlite_source_requires_explicit_write_admission(tmp_path: Path) -> None:
    path = tmp_path / "records.db"
    _database(path)

    read_only = await SQLiteSource(path).open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW,
    )
    writable = await SQLiteSource(path, allow_writes=True).open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW,
    )
    try:
        assert "write_access" not in read_only.registration.configuration
        assert tuple(item.id for item in read_only.declarations().capabilities) == (
            SQLITE_QUERY_CAPABILITY_ID,
        )

        assert writable.registration.configuration["write_access"] is True
        assert tuple(item.id for item in writable.declarations().capabilities) == (
            SQLITE_QUERY_CAPABILITY_ID,
            SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
            SQLITE_UPDATE_CAPABILITY_ID,
        )
        assert writable._connection.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        await read_only.close()
        await writable.close()
