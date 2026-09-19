"""Create a fresh state database directly at the current home revision."""

from __future__ import annotations

import sqlite3

from ..sqlite_schema import (
    CURRENT_SCHEMA,
    REVISION_2_DATABASE_SQL,
    require_healthy,
    require_schema,
)
from .registry import stamp_fresh_home


def create_current_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    if mode != "wal":
        raise ValueError("current state database requires WAL journal mode")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA wal_autocheckpoint = 1000")
    connection.executescript("BEGIN IMMEDIATE;\n" + REVISION_2_DATABASE_SQL)
    stamp_fresh_home(connection)
    require_schema(connection, CURRENT_SCHEMA)
    require_healthy(connection)
    connection.commit()


__all__ = ["create_current_database"]
