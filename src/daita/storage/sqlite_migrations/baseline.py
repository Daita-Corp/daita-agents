"""Create one fresh SQLite database at the current journal baseline."""

from __future__ import annotations

import sqlite3

from ..sqlite_schema import (
    BASE_TABLE_SQL,
    JOURNAL_TABLE_SQL,
    POSTGRESQL_UPDATE_SCOPE_TABLE_SQL,
    RECEIPT_TABLE_SQL,
    SCOPED_PERMISSION_TABLES,
    SOURCE_READ_SCOPE_TABLE_SQL,
    require_healthy,
    require_schema,
)
from .runner import MIGRATIONS, insert_journal_row
from .scoped_source_permissions import validate_target


def create_current(connection: sqlite3.Connection) -> None:
    connection.executescript(
        "BEGIN IMMEDIATE;\n"
        + BASE_TABLE_SQL
        + RECEIPT_TABLE_SQL
        + ";\n"
        + JOURNAL_TABLE_SQL
        + ";\n"
        + SOURCE_READ_SCOPE_TABLE_SQL
        + ";\n"
        + POSTGRESQL_UPDATE_SCOPE_TABLE_SQL
        + ";\n"
    )
    for migration in MIGRATIONS:
        insert_journal_row(connection, migration)
    require_schema(connection, SCOPED_PERMISSION_TABLES)
    validate_target(connection)
    require_healthy(connection)
    connection.commit()
