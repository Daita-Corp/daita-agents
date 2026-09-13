"""Create a fresh state database directly at the current home revision."""

from __future__ import annotations

import sqlite3

from ..sqlite_schema import (
    CURRENT_SCHEMA,
    REVISION_1_DATABASE_SQL,
    require_healthy,
    require_schema,
)
from .registry import stamp_fresh_home


def create_current_database(connection: sqlite3.Connection) -> None:
    connection.executescript("BEGIN IMMEDIATE;\n" + REVISION_1_DATABASE_SQL)
    stamp_fresh_home(connection)
    require_schema(connection, CURRENT_SCHEMA)
    require_healthy(connection)
    connection.commit()


__all__ = ["create_current_database"]
