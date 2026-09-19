"""Expose the authoritative whole-agent-home registry without import cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_REGISTRY_EXPORTS = frozenset(
    {
        "CURRENT_HOME_REVISION",
        "HOME_MIGRATIONS",
        "MINIMUM_SUPPORTED_HOME_REVISION",
        "HomeMigrationJournalError",
        "HomeMigrationJournalNewerError",
        "insert_migration_row",
        "inspect_home_revision",
        "migration_rows",
        "stamp_fresh_home",
    }
)


def __getattr__(name: str) -> Any:
    if name == "HomeMigration":
        from .models import HomeMigration

        return HomeMigration
    if name == "create_current_database":
        from .baseline import create_current_database

        return create_current_database
    if name in _REGISTRY_EXPORTS:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(name)


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
