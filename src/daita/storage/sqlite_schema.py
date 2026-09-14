"""Define, inspect, and validate the current physical SQLite schema."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

TableSchema = Mapping[str, tuple[tuple[object, ...], ...]]
ForeignKeySchema = Mapping[str, tuple[tuple[object, ...], ...]]
UniqueConstraintSchema = Mapping[str, frozenset[tuple[str, ...]]]
NamedIndexSchema = Mapping[str, tuple[str, bool, tuple[str, ...]]]
RequiredSQLSchema = Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SQLiteSchema:
    """Immutable complete contract for one agent-home SQLite shape."""

    tables: TableSchema
    foreign_keys: ForeignKeySchema
    unique_constraints: UniqueConstraintSchema
    named_indexes: NamedIndexSchema
    required_sql_fragments: RequiredSQLSchema

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tables",
            MappingProxyType(
                {table: tuple(columns) for table, columns in self.tables.items()}
            ),
        )
        object.__setattr__(
            self,
            "foreign_keys",
            MappingProxyType(
                {
                    table: tuple(foreign_keys)
                    for table, foreign_keys in self.foreign_keys.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "unique_constraints",
            MappingProxyType(
                {
                    table: frozenset(constraints)
                    for table, constraints in self.unique_constraints.items()
                }
            ),
        )
        object.__setattr__(
            self, "named_indexes", MappingProxyType(dict(self.named_indexes))
        )
        object.__setattr__(
            self,
            "required_sql_fragments",
            MappingProxyType(
                {
                    table: tuple(fragments)
                    for table, fragments in self.required_sql_fragments.items()
                }
            ),
        )


CORE_TABLES: dict[str, tuple[tuple[object, ...], ...]] = {
    "learning_candidates": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "messages": (
        ("run_id", "TEXT", 1, None, 1),
        ("position", "INTEGER", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "metadata": (
        ("key", "TEXT", 0, None, 1),
        ("data", "TEXT", 1, None, 0),
    ),
    "runs": (
        ("id", "TEXT", 0, None, 1),
        ("agent_id", "TEXT", 1, None, 0),
        ("conversation_id", "TEXT", 1, None, 0),
        ("turn_index", "INTEGER", 1, None, 0),
        ("input", "TEXT", 1, None, 0),
        ("result", "TEXT", 0, None, 0),
    ),
    "semantic_annotations": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "snapshots": (
        ("agent_id", "TEXT", 1, None, 1),
        ("source_id", "TEXT", 1, None, 2),
        ("sync_id", "TEXT", 1, None, 0),
        ("data", "TEXT", 1, None, 0),
    ),
    "sources": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "syncs": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("source_id", "TEXT", 1, None, 0),
        ("data", "TEXT", 1, None, 0),
    ),
}

RECEIPT_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("id", "TEXT", 1, None, 2),
    ("run_id", "TEXT", 1, None, 0),
    ("call_id", "TEXT", 1, None, 0),
    ("operation_key", "TEXT", 1, None, 0),
    ("routine_id", "TEXT", 0, None, 0),
    ("occurrence_id", "TEXT", 0, None, 0),
    ("grant_digest", "TEXT", 0, None, 0),
    ("unresolved", "INTEGER", 1, None, 0),
    ("data", "TEXT", 1, None, 0),
)

AGENT_HOME_MIGRATION_TABLE = (
    ("revision", "INTEGER", 1, None, 1),
    ("migration_id", "TEXT", 1, None, 0),
    ("checksum", "TEXT", 1, None, 0),
)
READ_SCOPE_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("source_id", "TEXT", 1, None, 2),
    ("data", "TEXT", 1, None, 0),
)
RELATIONAL_WRITE_SCOPE_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("source_id", "TEXT", 1, None, 2),
    ("resource_id", "TEXT", 1, None, 3),
    ("authorization_fingerprint", "TEXT", 1, None, 0),
    ("data", "TEXT", 1, None, 0),
)
MCP_BINDING_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("binding_id", "TEXT", 1, None, 2),
    ("data", "TEXT", 1, None, 0),
)
JOB_RUN_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("job_id", "TEXT", 1, None, 2),
    ("data", "TEXT", 1, None, 0),
)
AUTONOMOUS_FOLLOWUP_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("followup_id", "TEXT", 1, None, 2),
    ("job_id", "TEXT", 1, None, 0),
    ("event_id", "TEXT", 1, None, 0),
    ("data", "TEXT", 1, None, 0),
)
DELIVERY_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("delivery_id", "TEXT", 1, None, 2),
    ("conversation_id", "TEXT", 1, None, 0),
    ("subject_kind", "TEXT", 1, None, 0),
    ("subject_id", "TEXT", 1, None, 0),
    ("logical_key", "TEXT", 1, None, 0),
    ("target_kind", "TEXT", 1, None, 0),
    ("target_fingerprint", "TEXT", 1, None, 0),
    ("state", "TEXT", 1, None, 0),
    ("created_at_us", "INTEGER", 1, None, 0),
    ("data", "TEXT", 1, None, 0),
)
SCHEDULED_ROUTINE_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("routine_id", "TEXT", 1, None, 2),
    ("conversation_id", "TEXT", 1, None, 0),
    ("state", "TEXT", 1, None, 0),
    ("next_due_at_us", "INTEGER", 0, None, 0),
    ("data", "TEXT", 1, None, 0),
)
ROUTINE_OCCURRENCE_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("occurrence_id", "TEXT", 1, None, 2),
    ("routine_id", "TEXT", 1, None, 0),
    ("routine_revision", "INTEGER", 1, None, 0),
    ("slot_key", "TEXT", 1, None, 0),
    ("state", "TEXT", 1, None, 0),
    ("lease_expires_at_us", "INTEGER", 0, None, 0),
    ("reserved_run_id", "TEXT", 0, None, 0),
    ("data", "TEXT", 1, None, 0),
)

CURRENT_TABLES = {
    **CORE_TABLES,
    "effect_receipts": RECEIPT_TABLE,
    "agent_home_migrations": AGENT_HOME_MIGRATION_TABLE,
    "source_read_scopes": READ_SCOPE_TABLE,
    "relational_write_scopes": RELATIONAL_WRITE_SCOPE_TABLE,
    "mcp_server_bindings": MCP_BINDING_TABLE,
    "job_runs": JOB_RUN_TABLE,
    "autonomous_followups": AUTONOMOUS_FOLLOWUP_TABLE,
    "deliveries": DELIVERY_TABLE,
    "scheduled_routines": SCHEDULED_ROUTINE_TABLE,
    "routine_occurrences": ROUTINE_OCCURRENCE_TABLE,
}

MESSAGES_FOREIGN_KEYS = (("runs", "run_id", "id", "NO ACTION", "CASCADE", "NONE"),)
SOURCE_SCOPE_FOREIGN_KEYS = (
    ("sources", "agent_id", "agent_id", "NO ACTION", "CASCADE", "NONE"),
    ("sources", "source_id", "id", "NO ACTION", "CASCADE", "NONE"),
)
ROUTINE_OCCURRENCE_FOREIGN_KEYS = (
    (
        "scheduled_routines",
        "agent_id",
        "agent_id",
        "NO ACTION",
        "CASCADE",
        "NONE",
    ),
    (
        "scheduled_routines",
        "routine_id",
        "routine_id",
        "NO ACTION",
        "CASCADE",
        "NONE",
    ),
)
NAMED_INDEXES = {
    "effect_receipts_unresolved": (
        "effect_receipts",
        False,
        ("agent_id", "unresolved", "routine_id", "run_id"),
    ),
    "effect_receipts_grant_reservations": (
        "effect_receipts",
        False,
        ("agent_id", "occurrence_id", "grant_digest"),
    ),
    "runs_conversation_turn": (
        "runs",
        True,
        ("agent_id", "conversation_id", "turn_index"),
    ),
    "scheduled_routines_due": (
        "scheduled_routines",
        False,
        ("agent_id", "state", "next_due_at_us", "routine_id"),
    ),
    "routine_occurrences_stale": (
        "routine_occurrences",
        False,
        ("agent_id", "state", "lease_expires_at_us", "occurrence_id"),
    ),
    "deliveries_conversation_history": (
        "deliveries",
        False,
        ("agent_id", "conversation_id", "created_at_us", "delivery_id"),
    ),
}
UNIQUE_CONSTRAINTS = {
    "effect_receipts": frozenset(
        {("agent_id", "run_id", "call_id"), ("agent_id", "operation_key")}
    ),
    "agent_home_migrations": frozenset({("migration_id",)}),
    "autonomous_followups": frozenset(
        {("agent_id", "event_id"), ("agent_id", "job_id")}
    ),
    "deliveries": frozenset(
        {
            ("agent_id", "logical_key"),
            (
                "agent_id",
                "subject_kind",
                "subject_id",
                "target_fingerprint",
            ),
        }
    ),
    "routine_occurrences": frozenset(
        {
            ("agent_id", "routine_id", "routine_revision", "slot_key"),
            ("agent_id", "reserved_run_id"),
        }
    ),
}
REQUIRED_SQL_FRAGMENTS = {
    table: ("CHECK (json_valid(data))",)
    for table in (
        "deliveries",
        "mcp_server_bindings",
        "routine_occurrences",
        "scheduled_routines",
    )
}
SCHEMA_REVISION_1 = SQLiteSchema(
    tables=CURRENT_TABLES,
    foreign_keys={
        "messages": MESSAGES_FOREIGN_KEYS,
        "source_read_scopes": SOURCE_SCOPE_FOREIGN_KEYS,
        "relational_write_scopes": SOURCE_SCOPE_FOREIGN_KEYS,
        "routine_occurrences": ROUTINE_OCCURRENCE_FOREIGN_KEYS,
    },
    unique_constraints=UNIQUE_CONSTRAINTS,
    named_indexes=NAMED_INDEXES,
    required_sql_fragments=REQUIRED_SQL_FRAGMENTS,
)
# Revision 2 changes a retained record codec without changing the physical
# SQLite schema. Keep a revision-owned name so its migration never depends on
# a later release's moving current-schema alias.
SCHEMA_REVISION_2 = SCHEMA_REVISION_1
CURRENT_SCHEMA = SCHEMA_REVISION_2

BASE_TABLE_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE sources (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE TABLE syncs (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE TABLE snapshots (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    sync_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, source_id)
);
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    input TEXT NOT NULL,
    result TEXT
);
CREATE TABLE messages (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(run_id, position)
);
CREATE TABLE semantic_annotations (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE TABLE learning_candidates (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
CREATE UNIQUE INDEX runs_conversation_turn
    ON runs(agent_id, conversation_id, turn_index);
"""

RECEIPT_TABLE_SQL = """
CREATE TABLE effect_receipts (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    routine_id TEXT,
    occurrence_id TEXT,
    grant_digest TEXT,
    unresolved INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id),
    UNIQUE(agent_id, run_id, call_id),
    UNIQUE(agent_id, operation_key)
);
CREATE INDEX effect_receipts_unresolved ON effect_receipts(agent_id, unresolved, routine_id, run_id);
CREATE INDEX effect_receipts_grant_reservations ON effect_receipts(agent_id, occurrence_id, grant_digest)
"""

AGENT_HOME_MIGRATION_TABLE_SQL = """
CREATE TABLE agent_home_migrations (
    revision INTEGER NOT NULL PRIMARY KEY,
    migration_id TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL
)
"""

SOURCE_READ_SCOPE_TABLE_SQL = """
CREATE TABLE source_read_scopes (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, source_id),
    FOREIGN KEY (agent_id, source_id)
        REFERENCES sources(agent_id, id)
        ON DELETE CASCADE
)
"""

RELATIONAL_WRITE_SCOPE_TABLE_SQL = """
CREATE TABLE relational_write_scopes (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    authorization_fingerprint TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, source_id, resource_id),
    FOREIGN KEY (agent_id, source_id)
        REFERENCES sources(agent_id, id)
        ON DELETE CASCADE
)
"""

MCP_SERVER_BINDING_TABLE_SQL = """
CREATE TABLE mcp_server_bindings (
    agent_id TEXT NOT NULL,
    binding_id TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, binding_id)
)
"""

JOB_RUN_TABLE_SQL = """
CREATE TABLE job_runs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, job_id)
)
"""

AUTONOMOUS_FOLLOWUP_TABLE_SQL = """
CREATE TABLE autonomous_followups (
    agent_id TEXT NOT NULL,
    followup_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (agent_id, followup_id),
    UNIQUE (agent_id, event_id),
    UNIQUE (agent_id, job_id)
)
"""

DELIVERY_TABLE_SQL = """
CREATE TABLE deliveries (
    agent_id TEXT NOT NULL,
    delivery_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_fingerprint TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, delivery_id),
    UNIQUE (agent_id, logical_key),
    UNIQUE (agent_id, subject_kind, subject_id, target_fingerprint)
);
CREATE INDEX deliveries_conversation_history
    ON deliveries(agent_id, conversation_id, created_at_us, delivery_id)
"""

SCHEDULED_ROUTINE_TABLE_SQL = """
CREATE TABLE scheduled_routines (
    agent_id TEXT NOT NULL,
    routine_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    state TEXT NOT NULL,
    next_due_at_us INTEGER,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, routine_id)
);
CREATE INDEX scheduled_routines_due
    ON scheduled_routines(agent_id, state, next_due_at_us, routine_id)
"""

ROUTINE_OCCURRENCE_TABLE_SQL = """
CREATE TABLE routine_occurrences (
    agent_id TEXT NOT NULL,
    occurrence_id TEXT NOT NULL,
    routine_id TEXT NOT NULL,
    routine_revision INTEGER NOT NULL,
    slot_key TEXT NOT NULL,
    state TEXT NOT NULL,
    lease_expires_at_us INTEGER,
    reserved_run_id TEXT,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY (agent_id, occurrence_id),
    UNIQUE (agent_id, routine_id, routine_revision, slot_key),
    UNIQUE (agent_id, reserved_run_id),
    FOREIGN KEY (agent_id, routine_id)
        REFERENCES scheduled_routines(agent_id, routine_id)
        ON DELETE CASCADE
);
CREATE INDEX routine_occurrences_stale
    ON routine_occurrences(agent_id, state, lease_expires_at_us, occurrence_id)
"""

REVISION_1_DATABASE_SQL = (
    BASE_TABLE_SQL
    + RECEIPT_TABLE_SQL
    + ";\n"
    + AGENT_HOME_MIGRATION_TABLE_SQL
    + ";\n"
    + SOURCE_READ_SCOPE_TABLE_SQL
    + ";\n"
    + RELATIONAL_WRITE_SCOPE_TABLE_SQL
    + ";\n"
    + MCP_SERVER_BINDING_TABLE_SQL
    + ";\n"
    + JOB_RUN_TABLE_SQL
    + ";\n"
    + AUTONOMOUS_FOLLOWUP_TABLE_SQL
    + ";\n"
    + DELIVERY_TABLE_SQL
    + ";\n"
    + SCHEDULED_ROUTINE_TABLE_SQL
    + ";\n"
    + ROUTINE_OCCURRENCE_TABLE_SQL
    + ";\n"
)


def table_names(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )


def schema_matches(
    connection: sqlite3.Connection,
    definitions: TableSchema | SQLiteSchema,
) -> bool:
    try:
        require_schema(connection, definitions)
    except (sqlite3.Error, ValueError):
        return False
    return True


def require_schema(
    connection: sqlite3.Connection,
    definitions: TableSchema | SQLiteSchema,
) -> None:
    if isinstance(definitions, SQLiteSchema):
        tables = definitions.tables
        expected_foreign_keys = definitions.foreign_keys
        expected_unique_constraints = definitions.unique_constraints
        expected_named_indexes = definitions.named_indexes
        required_sql_fragments = definitions.required_sql_fragments
    else:
        tables = definitions
        expected_foreign_keys = {
            "messages": MESSAGES_FOREIGN_KEYS,
            **(
                {"source_read_scopes": SOURCE_SCOPE_FOREIGN_KEYS}
                if "source_read_scopes" in tables
                else {}
            ),
            **(
                {"relational_write_scopes": SOURCE_SCOPE_FOREIGN_KEYS}
                if "relational_write_scopes" in tables
                else {}
            ),
            **(
                {"routine_occurrences": ROUTINE_OCCURRENCE_FOREIGN_KEYS}
                if "routine_occurrences" in tables
                else {}
            ),
        }
        expected_unique_constraints = UNIQUE_CONSTRAINTS
        expected_named_indexes = NAMED_INDEXES
        required_sql_fragments = REQUIRED_SQL_FRAGMENTS
    if table_names(connection) != set(tables):
        raise ValueError("state tables do not match the declared revision")
    for table, expected in tables.items():
        actual = tuple(
            (row[1], str(row[2]).upper(), row[3], row[4], row[5], row[6])
            for row in connection.execute(f"PRAGMA table_xinfo({table})")
        )
        if actual != tuple((*column, 0) for column in expected):
            raise ValueError(f"state table does not match its revision: {table}")

    foreign_keys: dict[str, tuple[tuple[object, ...], ...]] = {
        table: tuple(
            (row[2], row[3], row[4], row[5], row[6], row[7])
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        for table in tables
    }
    for table, actual_foreign_keys in foreign_keys.items():
        if actual_foreign_keys != expected_foreign_keys.get(table, ()):
            raise ValueError(f"state foreign keys are invalid: {table}")

    for table in tables:
        index_rows = tuple(connection.execute(f"PRAGMA index_list({table})"))
        if any(row[4] != 0 for row in index_rows):
            raise ValueError(f"state partial indexes are invalid: {table}")
        actual_unique_constraints = frozenset(
            tuple(
                column[2]
                for column in connection.execute(f"PRAGMA index_info({index[1]})")
            )
            for index in index_rows
            if index[3] == "u"
        )
        if actual_unique_constraints != expected_unique_constraints.get(
            table, frozenset()
        ):
            raise ValueError(f"state unique constraints are invalid: {table}")

    named_indexes = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if named_indexes != {
        name: definition[0] for name, definition in expected_named_indexes.items()
    }:
        raise ValueError("state named indexes do not match the declared revision")
    for name, (
        table,
        expected_unique,
        expected_columns,
    ) in expected_named_indexes.items():
        indexes = {
            row[1]: bool(row[2])
            for row in connection.execute(f"PRAGMA index_list({table})")
            if not str(row[1]).startswith("sqlite_autoindex")
        }
        if indexes != {
            index_name: definition[1]
            for index_name, definition in expected_named_indexes.items()
            if definition[0] == table
        }:
            raise ValueError(f"state index is invalid: {name}")
        columns = tuple(
            row[2] for row in connection.execute(f"PRAGMA index_info({name})")
        )
        if columns != expected_columns:
            raise ValueError(f"state index columns are invalid: {name}")

    for table in tables:
        for index in connection.execute(f"PRAGMA index_list({table})"):
            key_columns = tuple(
                row
                for row in connection.execute(f'PRAGMA index_xinfo("{index[1]}")')
                if row[5] == 1
            )
            if any(row[3] != 0 or row[4] != "BINARY" for row in key_columns):
                raise ValueError(f"state index ordering is invalid: {index[1]}")

    for table in tables:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if row is None or not isinstance(row[0], str):
            raise ValueError(f"state table SQL is unavailable: {table}")
        normalized_sql = " ".join(row[0].split())
        fragments = required_sql_fragments.get(table, ())
        if any(fragment not in normalized_sql for fragment in fragments):
            raise ValueError(f"state table checks are invalid: {table}")
        if len(re.findall(r"\bCHECK\s*\(", normalized_sql, re.IGNORECASE)) != len(
            fragments
        ):
            raise ValueError(f"state table checks are invalid: {table}")
        if re.search(
            r"\b(COLLATE|DEFERRABLE|GENERATED|STRICT|WITHOUT\s+ROWID)\b",
            normalized_sql,
            re.IGNORECASE,
        ):
            raise ValueError(f"state table options are invalid: {table}")

    extra_objects = tuple(
        connection.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE type IN ('trigger', 'view') AND name NOT LIKE 'sqlite_%'"
        )
    )
    if extra_objects:
        raise ValueError("state database has unexpected triggers or views")


def require_healthy(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
        raise ValueError("state database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("state database foreign-key check failed")
