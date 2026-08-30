"""Define, inspect, and validate the current physical SQLite schema."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

TableSchema = Mapping[str, tuple[tuple[object, ...], ...]]

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
    ("data", "TEXT", 1, None, 0),
)
JOURNAL_TABLE = (
    ("ordinal", "INTEGER", 1, None, 0),
    ("migration_id", "TEXT", 1, None, 1),
    ("checksum", "TEXT", 1, None, 0),
)
READ_SCOPE_TABLE = (
    ("agent_id", "TEXT", 1, None, 1),
    ("source_id", "TEXT", 1, None, 2),
    ("data", "TEXT", 1, None, 0),
)
UPDATE_SCOPE_TABLE = (
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
    "database_write_receipts": RECEIPT_TABLE,
    "state_migrations": JOURNAL_TABLE,
    "source_read_scopes": READ_SCOPE_TABLE,
    "postgresql_update_scopes": UPDATE_SCOPE_TABLE,
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
    "database_write_receipts": frozenset({("agent_id", "run_id", "call_id")}),
    "state_migrations": frozenset({("ordinal",)}),
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
CREATE TABLE database_write_receipts (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id),
    UNIQUE(agent_id, run_id, call_id)
)
"""

JOURNAL_TABLE_SQL = """
CREATE TABLE state_migrations (
    ordinal INTEGER NOT NULL UNIQUE,
    migration_id TEXT NOT NULL PRIMARY KEY,
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

POSTGRESQL_UPDATE_SCOPE_TABLE_SQL = """
CREATE TABLE postgresql_update_scopes (
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


def table_names(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )


def schema_matches(connection: sqlite3.Connection, definitions: TableSchema) -> bool:
    try:
        require_schema(connection, definitions)
    except (sqlite3.Error, ValueError):
        return False
    return True


def require_schema(connection: sqlite3.Connection, definitions: TableSchema) -> None:
    if table_names(connection) != set(definitions):
        raise ValueError("state tables do not match the declared revision")
    for table, expected in definitions.items():
        actual = tuple(
            (row[1], str(row[2]).upper(), row[3], row[4], row[5])
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise ValueError(f"state table does not match its revision: {table}")

    foreign_keys: dict[str, tuple[tuple[object, ...], ...]] = {
        table: tuple(
            (row[2], row[3], row[4], row[5], row[6], row[7])
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        for table in definitions
    }
    expected_foreign_keys: dict[str, tuple[tuple[object, ...], ...]] = {
        "messages": MESSAGES_FOREIGN_KEYS,
        **(
            {"source_read_scopes": SOURCE_SCOPE_FOREIGN_KEYS}
            if "source_read_scopes" in definitions
            else {}
        ),
        **(
            {"postgresql_update_scopes": SOURCE_SCOPE_FOREIGN_KEYS}
            if "postgresql_update_scopes" in definitions
            else {}
        ),
        **(
            {"routine_occurrences": ROUTINE_OCCURRENCE_FOREIGN_KEYS}
            if "routine_occurrences" in definitions
            else {}
        ),
    }
    for table, actual_foreign_keys in foreign_keys.items():
        if actual_foreign_keys != expected_foreign_keys.get(table, ()):
            raise ValueError(f"state foreign keys are invalid: {table}")

    for table in definitions:
        actual_unique_constraints = frozenset(
            tuple(
                column[2]
                for column in connection.execute(f"PRAGMA index_info({index[1]})")
            )
            for index in connection.execute(f"PRAGMA index_list({table})")
            if index[3] == "u"
        )
        if actual_unique_constraints != UNIQUE_CONSTRAINTS.get(table, frozenset()):
            raise ValueError(f"state unique constraints are invalid: {table}")

    named_indexes = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if named_indexes != {
        name: definition[0] for name, definition in NAMED_INDEXES.items()
    }:
        raise ValueError("state named indexes do not match the declared revision")
    for name, (table, expected_unique, expected_columns) in NAMED_INDEXES.items():
        indexes = {
            row[1]: bool(row[2])
            for row in connection.execute(f"PRAGMA index_list({table})")
            if not str(row[1]).startswith("sqlite_autoindex")
        }
        if indexes != {name: expected_unique}:
            raise ValueError(f"state index is invalid: {name}")
        columns = tuple(
            row[2] for row in connection.execute(f"PRAGMA index_info({name})")
        )
        if columns != expected_columns:
            raise ValueError(f"state index columns are invalid: {name}")

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
