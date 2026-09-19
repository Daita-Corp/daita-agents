"""Immutable released revision-1 DDL fragments owned by home migration."""

from ..schema_contract import schema_from_sql

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

SCHEMA_REVISION_1 = schema_from_sql(REVISION_1_DATABASE_SQL)
MESSAGES_FOREIGN_KEYS = SCHEMA_REVISION_1.foreign_keys["messages"]
SOURCE_SCOPE_FOREIGN_KEYS = SCHEMA_REVISION_1.foreign_keys["source_read_scopes"]
ROUTINE_OCCURRENCE_FOREIGN_KEYS = SCHEMA_REVISION_1.foreign_keys["routine_occurrences"]
NAMED_INDEXES = SCHEMA_REVISION_1.named_indexes
UNIQUE_CONSTRAINTS = SCHEMA_REVISION_1.unique_constraints
REQUIRED_SQL_FRAGMENTS = SCHEMA_REVISION_1.required_sql_fragments

__all__ = [
    "AGENT_HOME_MIGRATION_TABLE_SQL",
    "AUTONOMOUS_FOLLOWUP_TABLE_SQL",
    "BASE_TABLE_SQL",
    "DELIVERY_TABLE_SQL",
    "JOB_RUN_TABLE_SQL",
    "MCP_SERVER_BINDING_TABLE_SQL",
    "RECEIPT_TABLE_SQL",
    "REQUIRED_SQL_FRAGMENTS",
    "RELATIONAL_WRITE_SCOPE_TABLE_SQL",
    "REVISION_1_DATABASE_SQL",
    "ROUTINE_OCCURRENCE_TABLE_SQL",
    "ROUTINE_OCCURRENCE_FOREIGN_KEYS",
    "SCHEMA_REVISION_1",
    "SCHEDULED_ROUTINE_TABLE_SQL",
    "SOURCE_READ_SCOPE_TABLE_SQL",
    "SOURCE_SCOPE_FOREIGN_KEYS",
    "MESSAGES_FOREIGN_KEYS",
    "NAMED_INDEXES",
    "UNIQUE_CONSTRAINTS",
]
