"""Define and validate the sole current revision-2 physical SQLite schema."""

from __future__ import annotations

from .graph_schema import REVISION_2_DATABASE_SQL
from .home_migrations.revision_0001_schema import (
    MESSAGES_FOREIGN_KEYS as MESSAGES_FOREIGN_KEYS,
    NAMED_INDEXES as NAMED_INDEXES,
    REQUIRED_SQL_FRAGMENTS as REQUIRED_SQL_FRAGMENTS,
    REVISION_1_DATABASE_SQL as REVISION_1_DATABASE_SQL,
    ROUTINE_OCCURRENCE_FOREIGN_KEYS as ROUTINE_OCCURRENCE_FOREIGN_KEYS,
    SCHEMA_REVISION_1 as SCHEMA_REVISION_1,
    SOURCE_SCOPE_FOREIGN_KEYS as SOURCE_SCOPE_FOREIGN_KEYS,
    UNIQUE_CONSTRAINTS as UNIQUE_CONSTRAINTS,
)
from .schema_contract import (
    SQLiteSchema,
    require_healthy,
    require_schema,
    schema_from_sql,
    schema_matches,
    table_names,
)

SCHEMA_REVISION_2 = schema_from_sql(REVISION_2_DATABASE_SQL)
CURRENT_SCHEMA = SCHEMA_REVISION_2

__all__ = [
    "CURRENT_SCHEMA",
    "REVISION_2_DATABASE_SQL",
    "SCHEMA_REVISION_2",
    "SQLiteSchema",
    "require_healthy",
    "require_schema",
    "schema_matches",
    "table_names",
]
