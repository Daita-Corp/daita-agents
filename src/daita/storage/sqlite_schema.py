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
CATALOG_LINKS_SQL = """
CREATE TABLE catalog_links (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
);
"""
RETIRE_LEARNING_CANDIDATES_SQL = """
DROP TABLE learning_candidates;
DELETE FROM metadata WHERE key GLOB 'learning_review_stamps:*';
"""
# Children first: every graph table references the ones after it.
RETIRE_GRAPH_SQL = """
DROP TABLE effect_receipts;
DROP TABLE job_attempt_budget_reservations;
DROP TABLE job_task_budget_ledger;
DROP TABLE job_budget_ledger;
DROP TABLE job_graph_events;
DROP TABLE job_graph_mutations;
DROP TABLE job_task_controls;
DROP TABLE job_task_comments;
DROP TABLE job_task_checkpoints;
DROP TABLE job_task_results;
DROP TABLE job_task_attempts;
DROP TABLE job_task_dependencies;
DROP TABLE job_tasks;
DROP TABLE job_graphs;
DROP TABLE job_runs;
"""
JOBS_SQL = """
CREATE TABLE jobs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'running','waiting_for_you','done','failed','cancelled'
    )),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id)
);
CREATE INDEX jobs_by_state ON jobs (agent_id, state, updated_at_us);
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
    job_id TEXT,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id),
    UNIQUE(agent_id, run_id, call_id),
    UNIQUE(agent_id, operation_key),
    FOREIGN KEY(agent_id, job_id) REFERENCES jobs(agent_id, job_id)
        ON DELETE RESTRICT
);
CREATE INDEX effect_receipts_unresolved
    ON effect_receipts(agent_id, unresolved, routine_id, run_id);
CREATE INDEX effect_receipts_grant_reservations
    ON effect_receipts(agent_id, occurrence_id, grant_digest);
CREATE INDEX effect_receipts_by_job ON effect_receipts(agent_id, job_id, unresolved);
"""
CURRENT_DATABASE_SQL = (
    REVISION_3_DATABASE_SQL
    + ANALYSIS_EVIDENCE_SQL
    + CATALOG_LINKS_SQL
    + RETIRE_LEARNING_CANDIDATES_SQL
    + RETIRE_GRAPH_SQL
    + JOBS_SQL
)
SCHEMA_REVISION_4 = schema_from_sql(CURRENT_DATABASE_SQL)
CURRENT_SCHEMA = SCHEMA_REVISION_4

__all__ = [
    "ARTIFACT_REGISTRY_SQL",
    "CURRENT_DATABASE_SQL",
    "CURRENT_SCHEMA",
    "JOBS_SQL",
    "RETIRE_GRAPH_SQL",
    "RETIRE_LEARNING_CANDIDATES_SQL",
    "REVISION_2_DATABASE_SQL",
    "SCHEMA_REVISION_2",
    "SCHEMA_REVISION_3",
    "SQLiteSchema",
    "require_healthy",
    "require_schema",
    "schema_matches",
    "table_names",
]
