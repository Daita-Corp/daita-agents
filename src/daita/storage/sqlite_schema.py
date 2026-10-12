"""Define the current schema and immutable released schema contracts."""

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
ARTIFACT_REGISTRY_SQL = """
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY NOT NULL,
    agent_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    caller_principal_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('creating', 'ready', 'deleting')),
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX artifacts_by_conversation ON artifacts
    (agent_id, state, conversation_id, created_at_us, artifact_id);
CREATE INDEX artifacts_by_run ON artifacts (agent_id, run_id, state);
"""
REVISION_3_DATABASE_SQL = REVISION_2_DATABASE_SQL + ARTIFACT_REGISTRY_SQL
SCHEMA_REVISION_3 = schema_from_sql(REVISION_3_DATABASE_SQL)
ANALYSIS_EVIDENCE_SQL = """
CREATE TABLE analysis_evidence (
    agent_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('generation', 'cell', 'child', 'run')),
    data TEXT NOT NULL,
    reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes >= 0),
    PRIMARY KEY (run_id, evidence_id)
);
CREATE INDEX analysis_evidence_by_agent_run ON analysis_evidence (agent_id, run_id);
"""
CURRENT_DATABASE_SQL = REVISION_3_DATABASE_SQL + ANALYSIS_EVIDENCE_SQL
SCHEMA_REVISION_4 = schema_from_sql(CURRENT_DATABASE_SQL)
CURRENT_SCHEMA = SCHEMA_REVISION_4

__all__ = [
    "ARTIFACT_REGISTRY_SQL",
    "CURRENT_DATABASE_SQL",
    "CURRENT_SCHEMA",
    "REVISION_2_DATABASE_SQL",
    "SCHEMA_REVISION_2",
    "SCHEMA_REVISION_3",
    "SQLiteSchema",
    "require_healthy",
    "require_schema",
    "schema_matches",
    "table_names",
]
