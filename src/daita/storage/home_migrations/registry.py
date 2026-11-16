"""Own the sole ordered, checksummed agent-home revision registry."""

from __future__ import annotations

import re
import sqlite3

from ..graph_schema import REVISION_2_DATABASE_SQL
from ..schema_contract import SQLiteSchema, require_healthy, require_schema
from ..sqlite_schema import CURRENT_DATABASE_SQL, REVISION_3_DATABASE_SQL
from .models import HomeMigration
from .revision_0001 import REVISION_1
from .revision_0001_schema import REVISION_1_DATABASE_SQL
from .revision_0002 import REVISION_2
from .revision_0003 import REVISION_3
from .revision_0004 import REVISION_4

HOME_MIGRATIONS: tuple[HomeMigration, ...] = (
    REVISION_1,
    REVISION_2,
    REVISION_3,
    REVISION_4,
)
CURRENT_HOME_REVISION: int = HOME_MIGRATIONS[-1].revision
# Production revisions in this inclusive range upgrade automatically. The
# one preproduction bridge is admitted separately by revision 1.
MINIMUM_SUPPORTED_HOME_REVISION = 1
_RELEASED_CHECKSUMS = (
    "a08bdc56e3cb7c3dbe77dc0d8b8ed9aac1299a6a701902dba8e11a5ebe70f25e",
    "cb98e978fcb6db71cc1cb3ee5532d92ce99117db7359a008a98a4651df1c20e2",
)


class HomeMigrationJournalError(ValueError):
    def __init__(self, reason: str, found_revision: int | str | None = None) -> None:
        self.reason = reason
        self.found_revision = found_revision
        super().__init__(reason)


class HomeMigrationJournalNewerError(HomeMigrationJournalError):
    """The home was last committed by a newer Daita persistence implementation."""


def _require_registry() -> None:
    if tuple(item.revision for item in HOME_MIGRATIONS) != tuple(
        range(1, len(HOME_MIGRATIONS) + 1)
    ):
        raise RuntimeError("agent-home migration revisions must be contiguous")
    if len({item.migration_id for item in HOME_MIGRATIONS}) != len(HOME_MIGRATIONS):
        raise RuntimeError("agent-home migration IDs must be unique")
    if tuple(item.checksum for item in HOME_MIGRATIONS[:2]) != _RELEASED_CHECKSUMS:
        raise RuntimeError(
            "a released agent-home migration changed; append a new revision instead"
        )


_require_registry()


def migration_rows() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (migration.revision, migration.migration_id, migration.checksum)
        for migration in HOME_MIGRATIONS
    )


# SQL creating an empty database at each revision, where that revision defined it.
_DATABASE_SQL = {
    1: REVISION_1_DATABASE_SQL,
    2: REVISION_2_DATABASE_SQL,
    3: REVISION_3_DATABASE_SQL,
    4: CURRENT_DATABASE_SQL,
}


def revision_database(revision: int) -> tuple[str, SQLiteSchema]:
    """Return the SQL that creates an empty database at ``revision``, and its schema.

    A custom storage adapter exports its rows into this database, records
    revisions 1 through ``revision`` with ``insert_migration_row`` and upgrades it
    with ``daita.hosting.home_upgrade.upgrade_state_directory``. The SQL creates
    every table, including the empty ``agent_home_migrations`` ledger; the schema
    is the one that revision's migration checks.
    """

    if type(revision) is not int or revision not in _DATABASE_SQL:
        raise ValueError(f"unknown agent-home revision: {revision!r}")
    return _DATABASE_SQL[revision], HOME_MIGRATIONS[revision - 1].target_schema


def insert_migration_row(
    connection: sqlite3.Connection,
    migration: HomeMigration,
) -> None:
    connection.execute(
        "INSERT INTO agent_home_migrations(revision, migration_id, checksum) "
        "VALUES (?, ?, ?)",
        (migration.revision, migration.migration_id, migration.checksum),
    )


def stamp_fresh_home(connection: sqlite3.Connection) -> None:
    for migration in HOME_MIGRATIONS:
        insert_migration_row(connection, migration)


def inspect_home_revision(connection: sqlite3.Connection) -> int:
    rows = tuple(
        connection.execute(
            "SELECT revision, migration_id, checksum "
            "FROM agent_home_migrations ORDER BY revision"
        )
    )
    if not rows:
        raise HomeMigrationJournalError("agent-home migration journal is empty")
    expected_rows = migration_rows()
    for position, (revision, migration_id, checksum) in enumerate(
        rows[: len(expected_rows)], start=1
    ):
        expected = expected_rows[position - 1]
        if revision != position:
            raise HomeMigrationJournalError(
                "agent-home migration journal contains a revision gap", revision
            )
        if migration_id != expected[1]:
            raise HomeMigrationJournalError(
                "agent-home migration journal contains an unknown or reordered ID",
                str(migration_id),
            )
        if checksum != expected[2]:
            raise HomeMigrationJournalError(
                "agent-home migration checksum does not match its released definition",
                revision,
            )
    if len(rows) > len(expected_rows):
        for position, (revision, migration_id, checksum) in enumerate(
            rows[len(expected_rows) :], start=len(expected_rows) + 1
        ):
            if (
                revision != position
                or not isinstance(migration_id, str)
                or not migration_id.strip()
                or not isinstance(checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            ):
                raise HomeMigrationJournalError(
                    "agent-home migration journal has an invalid later entry", revision
                )
        raise HomeMigrationJournalNewerError(
            "agent-home revision is newer than this release", rows[-1][0]
        )
    revision = len(rows)
    require_schema(connection, HOME_MIGRATIONS[revision - 1].target_schema)
    require_healthy(connection)
    return revision


__all__ = [
    "CURRENT_HOME_REVISION",
    "HOME_MIGRATIONS",
    "MINIMUM_SUPPORTED_HOME_REVISION",
    "HomeMigrationJournalError",
    "HomeMigrationJournalNewerError",
    "insert_migration_row",
    "inspect_home_revision",
    "migration_rows",
    "revision_database",
    "stamp_fresh_home",
]
