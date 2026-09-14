"""Own the sole ordered, checksummed agent-home revision registry."""

from __future__ import annotations

import re
import sqlite3

from ..sqlite_schema import require_healthy, require_schema
from .models import HomeMigration
from .revision_0001 import REVISION_1
from .revision_0002 import REVISION_2

HOME_MIGRATIONS: tuple[HomeMigration, ...] = (REVISION_1, REVISION_2)
CURRENT_HOME_REVISION = HOME_MIGRATIONS[-1].revision
# Production revisions in this inclusive range upgrade automatically. The
# one preproduction bridge is admitted separately by revision 1.
MINIMUM_SUPPORTED_HOME_REVISION = 1
_RELEASED_CHECKSUMS = (
    "a08bdc56e3cb7c3dbe77dc0d8b8ed9aac1299a6a701902dba8e11a5ebe70f25e",
    "1090ba2cf009d09516f603832cba4e7224b2a33c841d21b159e597dff5e41b86",
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
    if tuple(item.checksum for item in HOME_MIGRATIONS) != _RELEASED_CHECKSUMS:
        raise RuntimeError(
            "a released agent-home migration changed; append a new revision instead"
        )


_require_registry()


def migration_rows() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (migration.revision, migration.migration_id, migration.checksum)
        for migration in HOME_MIGRATIONS
    )


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
    "stamp_fresh_home",
]
