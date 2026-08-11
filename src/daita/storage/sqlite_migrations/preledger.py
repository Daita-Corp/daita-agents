"""Bounded one-time bridge for every currently supported numeric-format home.

Removal gate: delete this unit only after the minimum supported Daita release
is guaranteed to have installed ``state_migrations`` and product support for
all pre-ledger homes has been deliberately ended. No normal journaled open
imports numeric format semantics from this module.
"""

from __future__ import annotations

import sqlite3
from enum import Enum

from ..sqlite_schema import (
    CURRENT_TABLES,
    INITIAL_TABLES,
    JOURNAL_RECEIPT_TABLES,
    JOURNAL_TABLE_SQL,
    RECEIPT_TABLES,
    require_healthy,
    require_schema,
    schema_matches,
    table_names,
)
from .database_write_receipts import MIGRATION as RECEIPT_MIGRATION
from .postgresql_write_admission import MIGRATION as ADMISSION_MIGRATION
from .runner import insert_journal_row

LEGACY_TABLE_MARKERS = frozenset({"evidence", "events", "operations", "tasks"})


class PreledgerShape(str, Enum):
    PRE_RECEIPT = "pre_receipt"
    RECEIPT_ERA = "receipt_era"


class PreledgerAdmissionError(ValueError):
    def __init__(self, reason: str, marker: int | None = None) -> None:
        self.reason = reason
        self.marker = marker
        super().__init__(reason)


class PreledgerNewerError(PreledgerAdmissionError):
    pass


class PreledgerLegacyError(PreledgerAdmissionError):
    pass


def identify(connection: sqlite3.Connection) -> PreledgerShape:
    marker = _read_legacy_marker(connection)
    tables = table_names(connection)
    is_pre_receipt = schema_matches(connection, INITIAL_TABLES)
    is_receipt_era = schema_matches(connection, RECEIPT_TABLES)
    if is_pre_receipt and marker in (0, 1):
        require_healthy(connection)
        return PreledgerShape.PRE_RECEIPT
    if is_receipt_era and marker in (0, 2, 3):
        require_healthy(connection)
        return PreledgerShape.RECEIPT_ERA
    if tables & LEGACY_TABLE_MARKERS:
        raise PreledgerLegacyError(
            "state belongs to the unsupported pre-1.0 framework", marker
        )
    if marker > 3 and (is_pre_receipt or is_receipt_era):
        raise PreledgerNewerError(
            "pre-ledger state was created by a newer release", marker
        )
    raise PreledgerAdmissionError(
        "state does not match a supported pre-ledger shape", marker
    )


def bridge(connection: sqlite3.Connection, expected: PreledgerShape) -> None:
    if identify(connection) is not expected:
        raise RuntimeError("pre-ledger state changed during admission")
    connection.execute(JOURNAL_TABLE_SQL)
    if expected is PreledgerShape.PRE_RECEIPT:
        RECEIPT_MIGRATION.apply(connection)
    insert_journal_row(connection, RECEIPT_MIGRATION)
    require_schema(connection, JOURNAL_RECEIPT_TABLES)
    ADMISSION_MIGRATION.apply(connection)
    insert_journal_row(connection, ADMISSION_MIGRATION)
    require_schema(connection, CURRENT_TABLES)
    if ADMISSION_MIGRATION.validate_target is not None:
        ADMISSION_MIGRATION.validate_target(connection)
    require_healthy(connection)


def _read_legacy_marker(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None or len(row) != 1 or not isinstance(row[0], int) or row[0] < 0:
        raise sqlite3.DatabaseError("pre-ledger state marker is invalid")
    return int(row[0])
