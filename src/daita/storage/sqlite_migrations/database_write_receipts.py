"""Add the immutable receipt table required by database-write execution."""

from __future__ import annotations

import sqlite3

from ..sqlite_schema import (
    JOURNAL_INITIAL_TABLES,
    JOURNAL_RECEIPT_TABLES,
    RECEIPT_TABLE_SQL,
)
from .models import SQLiteMigration

MIGRATION_ID = "20260810_database_write_receipts"
DEFINITION = """20260810_database_write_receipts
create database_write_receipts with agent receipt identity primary key
and unique agent run call identity; no existing row transformation
"""


def apply(connection: sqlite3.Connection) -> None:
    connection.execute(RECEIPT_TABLE_SQL)


MIGRATION = SQLiteMigration(
    ordinal=1,
    migration_id=MIGRATION_ID,
    definition=DEFINITION,
    source_schema=JOURNAL_INITIAL_TABLES,
    target_schema=JOURNAL_RECEIPT_TABLES,
    apply=apply,
)
