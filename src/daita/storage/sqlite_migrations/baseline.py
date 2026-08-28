"""Create a fresh SQLite database at the current development baseline."""

from __future__ import annotations

import sqlite3

from ..sqlite_schema import (
    AUTONOMOUS_FOLLOWUP_TABLE_SQL,
    BASE_TABLE_SQL,
    CONVERSATION_INBOX_TABLE_SQL,
    CURRENT_TABLES,
    JOB_RUN_TABLE_SQL,
    JOURNAL_TABLE_SQL,
    MCP_SERVER_BINDING_TABLE_SQL,
    POSTGRESQL_UPDATE_SCOPE_TABLE_SQL,
    RECEIPT_TABLE_SQL,
    ROUTINE_OCCURRENCE_TABLE_SQL,
    SCHEDULED_ROUTINE_TABLE_SQL,
    SOURCE_READ_SCOPE_TABLE_SQL,
    require_healthy,
    require_schema,
)
from .runner import MIGRATIONS, insert_journal_row


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
        + MCP_SERVER_BINDING_TABLE_SQL
        + ";\n"
        + JOB_RUN_TABLE_SQL
        + ";\n"
        + AUTONOMOUS_FOLLOWUP_TABLE_SQL
        + ";\n"
        + CONVERSATION_INBOX_TABLE_SQL
        + ";\n"
        + SCHEDULED_ROUTINE_TABLE_SQL
        + ";\n"
        + ROUTINE_OCCURRENCE_TABLE_SQL
        + ";\n"
    )
    for migration in MIGRATIONS:
        insert_journal_row(connection, migration)
    require_schema(connection, CURRENT_TABLES)
    require_healthy(connection)
    connection.commit()
