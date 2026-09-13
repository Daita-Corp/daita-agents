"""Expose the authoritative whole-agent-home revision registry."""

from .baseline import create_current_database
from .models import HomeMigration
from .registry import (
    CURRENT_HOME_REVISION,
    HOME_MIGRATIONS,
    MINIMUM_SUPPORTED_HOME_REVISION,
    HomeMigrationJournalError,
    HomeMigrationJournalNewerError,
    insert_migration_row,
    inspect_home_revision,
    migration_rows,
    stamp_fresh_home,
)

__all__ = [
    "CURRENT_HOME_REVISION",
    "HOME_MIGRATIONS",
    "MINIMUM_SUPPORTED_HOME_REVISION",
    "HomeMigration",
    "HomeMigrationJournalError",
    "HomeMigrationJournalNewerError",
    "create_current_database",
    "insert_migration_row",
    "inspect_home_revision",
    "migration_rows",
    "stamp_fresh_home",
]
