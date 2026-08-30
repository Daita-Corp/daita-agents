"""Validate and traverse the ordered checksummed SQLite migration journal."""

from __future__ import annotations

import re
import sqlite3

from ..sqlite_schema import CURRENT_TABLES, require_healthy, require_schema
from .models import SQLiteMigration

DEVELOPMENT_BASELINE_ID = "development_baseline"
DEVELOPMENT_BASELINE_DEFINITION = """development_baseline
current pre-production SQLite state shape;
MCPToolBinding codec-v1 uses exact toolbox presentation fields;
ArtifactProvenance codec-v1 includes exact local-file edit binding facts;
ArtifactDeliveryReceipt codec-v1 includes exact create/replace outcome facts;
ScheduledRoutine and RoutineOccurrence codec-v1 use the accepted D1 shape;
Delivery codec-v1 replaces the pre-production conversation inbox aggregate;
mutable until the first production state baseline is explicitly frozen
"""


def _baseline_noop(connection: sqlite3.Connection) -> None:
    connection.execute("SELECT 1")


DEVELOPMENT_BASELINE = SQLiteMigration(
    ordinal=1,
    migration_id=DEVELOPMENT_BASELINE_ID,
    definition=DEVELOPMENT_BASELINE_DEFINITION,
    source_schema=CURRENT_TABLES,
    target_schema=CURRENT_TABLES,
    apply=_baseline_noop,
)
MIGRATIONS: tuple[SQLiteMigration, ...] = (DEVELOPMENT_BASELINE,)
CURRENT_REVISION = MIGRATIONS[-1].migration_id


class MigrationJournalError(ValueError):
    """The stored journal is not an exact prefix of the declared ledger."""

    def __init__(self, reason: str, found_revision: str | None = None) -> None:
        self.reason = reason
        self.found_revision = found_revision
        super().__init__(reason)


class MigrationJournalNewerError(MigrationJournalError):
    """The stored ledger is a valid extension of this release's exact prefix."""


def insert_journal_row(
    connection: sqlite3.Connection,
    migration: SQLiteMigration,
) -> None:
    connection.execute(
        """INSERT INTO state_migrations(ordinal, migration_id, checksum)
           VALUES (?, ?, ?)""",
        (migration.ordinal, migration.migration_id, migration.checksum),
    )


def inspect_journal(connection: sqlite3.Connection) -> int:
    rows = tuple(connection.execute("""SELECT ordinal, migration_id, checksum
               FROM state_migrations ORDER BY ordinal"""))
    if not rows:
        raise MigrationJournalError("migration journal is empty")
    for position, (ordinal, migration_id, checksum) in enumerate(
        rows[: len(MIGRATIONS)], start=1
    ):
        expected = MIGRATIONS[position - 1]
        if ordinal != position:
            raise MigrationJournalError(
                "migration journal contains an ordinal gap", str(migration_id)
            )
        if migration_id != expected.migration_id:
            raise MigrationJournalError(
                "migration journal contains an unknown or reordered ID",
                str(migration_id),
            )
        if checksum != expected.checksum:
            raise MigrationJournalError(
                "migration journal checksum does not match the declared baseline",
                str(migration_id),
            )
    if len(rows) > len(MIGRATIONS):
        for position, (ordinal, migration_id, checksum) in enumerate(
            rows[len(MIGRATIONS) :], start=len(MIGRATIONS) + 1
        ):
            if (
                ordinal != position
                or not isinstance(migration_id, str)
                or not migration_id.strip()
                or not isinstance(checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            ):
                raise MigrationJournalError(
                    "migration journal contains an invalid later entry",
                    str(migration_id),
                )
        raise MigrationJournalNewerError(
            "migration journal extends beyond this release",
            str(rows[-1][1]),
        )
    applied = len(rows)
    require_schema(connection, MIGRATIONS[applied - 1].target_schema)
    validation = MIGRATIONS[applied - 1].validate_target
    if validation is not None:
        validation(connection)
    require_healthy(connection)
    return applied


def upgrade_journaled(connection: sqlite3.Connection, applied: int) -> None:
    if applied < 1 or applied >= len(MIGRATIONS):
        raise ValueError("journaled upgrade requires a non-current known prefix")
    for migration in MIGRATIONS[applied:]:
        require_schema(connection, migration.source_schema)
        require_healthy(connection)
        migration.apply(connection)
        insert_journal_row(connection, migration)
        require_schema(connection, migration.target_schema)
        if migration.validate_target is not None:
            migration.validate_target(connection)
        require_healthy(connection)


def migration_rows() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (migration.ordinal, migration.migration_id, migration.checksum)
        for migration in MIGRATIONS
    )
