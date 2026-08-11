"""Create one fresh SQLite database at the current journal baseline."""

from __future__ import annotations

import sqlite3

from ..sqlite_schema import (
    ADMISSION_TABLE_SQL,
    BASE_TABLE_SQL,
    CURRENT_TABLES,
    JOURNAL_TABLE_SQL,
    RECEIPT_TABLE_SQL,
    require_healthy,
    require_schema,
)
from .postgresql_write_admission import validate_target
from .runner import MIGRATIONS, insert_journal_row


def create_current(connection: sqlite3.Connection) -> None:
    connection.executescript(
        "BEGIN IMMEDIATE;\n"
        + BASE_TABLE_SQL
        + RECEIPT_TABLE_SQL
        + ";\n"
        + JOURNAL_TABLE_SQL
        + ";\n"
        + ADMISSION_TABLE_SQL
        + ";\n"
    )
    for migration in MIGRATIONS:
        insert_journal_row(connection, migration)
    require_schema(connection, CURRENT_TABLES)
    validate_target(connection)
    require_healthy(connection)
    connection.commit()
