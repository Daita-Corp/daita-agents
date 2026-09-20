"""Frozen revision-2 SQLite DDL and connection policy."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Literal
from urllib.parse import quote

from .home_migrations.revision_0001_schema import (
    AGENT_HOME_MIGRATION_TABLE_SQL,
    BASE_TABLE_SQL,
    DELIVERY_TABLE_SQL,
    MCP_SERVER_BINDING_TABLE_SQL,
    RELATIONAL_WRITE_SCOPE_TABLE_SQL,
    ROUTINE_OCCURRENCE_TABLE_SQL,
    SCHEDULED_ROUTINE_TABLE_SQL,
    SOURCE_READ_SCOPE_TABLE_SQL,
)


class ClosingSQLiteConnection(sqlite3.Connection):
    """Close an owned SQLite connection when its transaction context exits."""

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            return super().__exit__(exception_type, exception, traceback)
        finally:
            self.close()


GRAPH_TABLE_NAMES = frozenset(
    {
        "metadata",
        "sources",
        "syncs",
        "snapshots",
        "runs",
        "messages",
        "semantic_annotations",
        "learning_candidates",
        "effect_receipts",
        "agent_home_migrations",
        "source_read_scopes",
        "relational_write_scopes",
        "mcp_server_bindings",
        "job_runs",
        "job_graphs",
        "job_tasks",
        "job_task_dependencies",
        "job_task_attempts",
        "job_task_results",
        "job_task_checkpoints",
        "job_task_comments",
        "job_task_controls",
        "job_graph_mutations",
        "job_graph_events",
        "job_budget_ledger",
        "job_task_budget_ledger",
        "job_attempt_budget_reservations",
        "deliveries",
        "scheduled_routines",
        "routine_occurrences",
    }
)

GRAPH_RECEIPT_SQL = """
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
    task_id TEXT,
    task_attempt_id TEXT,
    fencing_epoch INTEGER,
    task_spec_digest TEXT,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id),
    UNIQUE(agent_id, run_id, call_id),
    UNIQUE(agent_id, operation_key),
    CHECK (
        (job_id IS NULL AND task_id IS NULL AND task_attempt_id IS NULL
         AND fencing_epoch IS NULL AND task_spec_digest IS NULL)
        OR
        (job_id IS NOT NULL AND task_id IS NOT NULL AND task_attempt_id IS NOT NULL
         AND fencing_epoch IS NOT NULL AND task_spec_digest IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id, task_id, task_attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE INDEX effect_receipts_unresolved
    ON effect_receipts(agent_id, unresolved, routine_id, run_id);
CREATE INDEX effect_receipts_grant_reservations
    ON effect_receipts(agent_id, occurrence_id, grant_digest);
CREATE INDEX effect_receipts_graph_attempt
    ON effect_receipts(agent_id, job_id, task_id, task_attempt_id)
"""

GRAPH_TABLE_SQL = """
CREATE TABLE job_runs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    origin_run_id TEXT NOT NULL,
    origin_call_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'queued','active','blocked','needs_attention','cancel_requested',
        'succeeded','failed','cancelled'
    )),
    desired_state TEXT NOT NULL CHECK (desired_state IN ('run','cancel')),
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    deadline_at_us INTEGER NOT NULL,
    terminal_at_us INTEGER,
    finalizer_task_id TEXT,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id),
    UNIQUE(agent_id, origin_run_id, origin_call_id)
);
CREATE INDEX job_runs_state_deadline
    ON job_runs(agent_id, state, deadline_at_us);
CREATE INDEX job_runs_conversation_created
    ON job_runs(agent_id, conversation_id, created_at_us);
CREATE INDEX job_runs_updated
    ON job_runs(agent_id, updated_at_us);

CREATE TABLE job_graphs (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    task_count INTEGER NOT NULL CHECK (task_count >= 0),
    edge_count INTEGER NOT NULL CHECK (edge_count >= 0),
    mutation_count INTEGER NOT NULL CHECK (mutation_count >= 0),
    active_attempt_count INTEGER NOT NULL CHECK (active_attempt_count >= 0),
    next_ready_at_us INTEGER,
    finalization_attempt_id TEXT,
    finalization_started_revision INTEGER,
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id),
    CHECK (
        (finalization_attempt_id IS NULL AND finalization_started_revision IS NULL)
        OR
        (finalization_attempt_id IS NOT NULL
         AND finalization_started_revision IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_runs(agent_id, job_id) ON DELETE RESTRICT
);
CREATE INDEX job_graphs_ready
    ON job_graphs(agent_id, next_ready_at_us, updated_at_us);

CREATE TABLE job_tasks (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'pending','ready','running','blocked','review','succeeded','failed',
        'cancelled','skipped','superseded'
    )),
    role TEXT NOT NULL CHECK (role IN (
        'planner','worker','reviewer','finalizer','internal'
    )),
    task_kind TEXT NOT NULL CHECK (task_kind IN ('model','internal_capability')),
    priority INTEGER NOT NULL,
    not_before_us INTEGER,
    current_attempt_id TEXT,
    task_revision INTEGER NOT NULL CHECK (task_revision > 0),
    task_spec_digest TEXT NOT NULL,
    task_scope_digest TEXT NOT NULL,
    supersedes_task_id TEXT,
    superseded_by_task_id TEXT,
    latest_result_id TEXT,
    latest_control_id TEXT,
    latest_checkpoint_id TEXT,
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    terminal_at_us INTEGER,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id),
    CHECK (supersedes_task_id IS NULL OR supersedes_task_id <> task_id),
    CHECK (superseded_by_task_id IS NULL OR superseded_by_task_id <> task_id),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, supersedes_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, superseded_by_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE INDEX job_tasks_ready
    ON job_tasks(agent_id, state, not_before_us, priority, updated_at_us);
CREATE INDEX job_tasks_by_job
    ON job_tasks(agent_id, job_id, state, priority, created_at_us);
CREATE INDEX job_tasks_current_attempt
    ON job_tasks(agent_id, job_id, current_attempt_id);

CREATE TABLE job_task_dependencies (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    upstream_task_id TEXT NOT NULL,
    downstream_task_id TEXT NOT NULL,
    edge_kind TEXT NOT NULL CHECK (edge_kind = 'requires_accepted_success'),
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, upstream_task_id, downstream_task_id),
    CHECK (upstream_task_id <> downstream_task_id),
    FOREIGN KEY(agent_id, job_id, upstream_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, downstream_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE INDEX job_task_dependencies_reverse
    ON job_task_dependencies(
        agent_id, job_id, downstream_task_id, upstream_task_id
    );

CREATE TABLE job_task_attempts (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    fencing_epoch INTEGER NOT NULL CHECK (fencing_epoch > 0),
    state TEXT NOT NULL CHECK (state IN (
        'claimed','running','succeeded','failed','cancelled','blocked',
        'review_requested','timed_out','protocol_violation','fenced'
    )),
    claim_token TEXT NOT NULL,
    run_id TEXT NOT NULL,
    lease_expires_at_us INTEGER,
    absolute_deadline_at_us INTEGER NOT NULL,
    started_at_us INTEGER,
    heartbeat_at_us INTEGER,
    ended_at_us INTEGER,
    active_slot INTEGER,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, attempt_id),
    UNIQUE(agent_id, job_id, task_id, ordinal),
    UNIQUE(agent_id, job_id, task_id, fencing_epoch),
    UNIQUE(agent_id, run_id),
    UNIQUE(agent_id, job_id, task_id, active_slot),
    CHECK (
        (state IN ('claimed','running') AND active_slot = 1
         AND lease_expires_at_us IS NOT NULL AND ended_at_us IS NULL)
        OR
        (state NOT IN ('claimed','running') AND active_slot IS NULL
         AND lease_expires_at_us IS NULL AND ended_at_us IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);
CREATE INDEX job_task_attempts_stale
    ON job_task_attempts(agent_id, state, lease_expires_at_us);
CREATE INDEX job_task_attempts_by_task
    ON job_task_attempts(agent_id, job_id, task_id, ordinal);
CREATE INDEX job_task_attempts_deadline
    ON job_task_attempts(agent_id, job_id, state, absolute_deadline_at_us);

CREATE TABLE job_task_results (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    result_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    completed_at_us INTEGER NOT NULL,
    sensitivity TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id),
    UNIQUE(agent_id, job_id, result_id),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);

CREATE TABLE job_task_checkpoints (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    created_at_us INTEGER NOT NULL,
    payload_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, attempt_id, checkpoint_id),
    UNIQUE(agent_id, job_id, task_id, attempt_id, ordinal),
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);

CREATE TABLE job_task_comments (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    comment_id TEXT NOT NULL,
    author_kind TEXT NOT NULL,
    author_id TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    body_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, comment_id),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);

CREATE TABLE job_task_controls (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    control_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN (
        'needs_input','needs_authorization','needs_replan','review_requested',
        'changes_requested','effect_uncertain','capability_unavailable',
        'source_or_contract_drift','budget_exhausted','retry_circuit_open'
    )),
    state TEXT NOT NULL CHECK (state IN ('open','resolved','rejected','expired')),
    requesting_attempt_id TEXT,
    created_at_us INTEGER NOT NULL,
    resolved_at_us INTEGER,
    resolved_by_kind TEXT,
    resolved_by_id TEXT,
    payload_digest TEXT NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, task_id, control_id),
    CHECK (
        (state = 'open' AND resolved_at_us IS NULL
         AND resolved_by_kind IS NULL AND resolved_by_id IS NULL)
        OR
        (state <> 'open' AND resolved_at_us IS NOT NULL
         AND resolved_by_kind IS NOT NULL AND resolved_by_id IS NOT NULL)
    ),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id, requesting_attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE INDEX job_task_controls_open
    ON job_task_controls(agent_id, job_id, state, created_at_us);

CREATE TABLE job_graph_mutations (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    mutation_id TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    actor_key TEXT NOT NULL,
    creator_task_id TEXT,
    creator_attempt_id TEXT,
    idempotency_key TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    expected_revision INTEGER NOT NULL CHECK (expected_revision >= 0),
    committed_revision INTEGER NOT NULL CHECK (committed_revision >= 0),
    decision TEXT NOT NULL CHECK (decision IN ('committed','rejected')),
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    PRIMARY KEY(agent_id, job_id, mutation_id),
    UNIQUE(agent_id, job_id, actor_key, idempotency_key),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, creator_task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, creator_task_id, creator_attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);

CREATE TABLE job_graph_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT,
    attempt_id TEXT,
    kind TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    data TEXT NOT NULL CHECK (json_valid(data)),
    CHECK (attempt_id IS NULL OR task_id IS NOT NULL),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT,
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
);
CREATE INDEX job_graph_events_job
    ON job_graph_events(agent_id, job_id, event_id);
CREATE INDEX job_graph_events_task
    ON job_graph_events(agent_id, job_id, task_id, event_id);
CREATE INDEX job_graph_events_kind
    ON job_graph_events(agent_id, kind, event_id);

CREATE TABLE job_budget_ledger (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    ceiling INTEGER NOT NULL CHECK (ceiling >= 0),
    settled INTEGER NOT NULL CHECK (settled >= 0),
    reserved INTEGER NOT NULL CHECK (reserved >= 0),
    control_reserved INTEGER NOT NULL CHECK (control_reserved >= 0),
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY(agent_id, job_id, dimension),
    CHECK (settled + reserved <= ceiling),
    CHECK (control_reserved <= ceiling),
    FOREIGN KEY(agent_id, job_id)
        REFERENCES job_graphs(agent_id, job_id) ON DELETE RESTRICT
);

CREATE TABLE job_task_budget_ledger (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    ceiling INTEGER NOT NULL CHECK (ceiling >= 0),
    settled INTEGER NOT NULL CHECK (settled >= 0),
    reserved INTEGER NOT NULL CHECK (reserved >= 0),
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY(agent_id, job_id, task_id, dimension),
    CHECK (settled + reserved <= ceiling),
    FOREIGN KEY(agent_id, job_id, task_id)
        REFERENCES job_tasks(agent_id, job_id, task_id) ON DELETE RESTRICT
);

CREATE TABLE job_attempt_budget_reservations (
    agent_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    reserved INTEGER NOT NULL CHECK (reserved >= 0),
    settled INTEGER CHECK (settled IS NULL OR settled >= 0),
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY(agent_id, job_id, task_id, attempt_id, dimension),
    FOREIGN KEY(agent_id, job_id, task_id, attempt_id)
        REFERENCES job_task_attempts(agent_id, job_id, task_id, attempt_id)
        ON DELETE RESTRICT
)
"""

REVISION_2_DATABASE_SQL = (
    BASE_TABLE_SQL
    + GRAPH_RECEIPT_SQL
    + ";\n"
    + AGENT_HOME_MIGRATION_TABLE_SQL
    + ";\n"
    + SOURCE_READ_SCOPE_TABLE_SQL
    + ";\n"
    + RELATIONAL_WRITE_SCOPE_TABLE_SQL
    + ";\n"
    + MCP_SERVER_BINDING_TABLE_SQL
    + ";\n"
    + GRAPH_TABLE_SQL
    + ";\n"
    + DELIVERY_TABLE_SQL
    + ";\n"
    + SCHEDULED_ROUTINE_TABLE_SQL
    + ";\n"
    + ROUTINE_OCCURRENCE_TABLE_SQL
    + ";\n"
)

GRAPH_NAMED_INDEXES: dict[str, tuple[str, ...]] = {
    "runs_conversation_turn": ("agent_id", "conversation_id", "turn_index"),
    "effect_receipts_unresolved": (
        "agent_id",
        "unresolved",
        "routine_id",
        "run_id",
    ),
    "effect_receipts_grant_reservations": (
        "agent_id",
        "occurrence_id",
        "grant_digest",
    ),
    "effect_receipts_graph_attempt": (
        "agent_id",
        "job_id",
        "task_id",
        "task_attempt_id",
    ),
    "job_runs_state_deadline": ("agent_id", "state", "deadline_at_us"),
    "job_runs_conversation_created": (
        "agent_id",
        "conversation_id",
        "created_at_us",
    ),
    "job_runs_updated": ("agent_id", "updated_at_us"),
    "job_graphs_ready": ("agent_id", "next_ready_at_us", "updated_at_us"),
    "job_tasks_ready": (
        "agent_id",
        "state",
        "not_before_us",
        "priority",
        "updated_at_us",
    ),
    "job_tasks_by_job": (
        "agent_id",
        "job_id",
        "state",
        "priority",
        "created_at_us",
    ),
    "job_tasks_current_attempt": ("agent_id", "job_id", "current_attempt_id"),
    "job_task_dependencies_reverse": (
        "agent_id",
        "job_id",
        "downstream_task_id",
        "upstream_task_id",
    ),
    "job_task_attempts_stale": ("agent_id", "state", "lease_expires_at_us"),
    "job_task_attempts_by_task": (
        "agent_id",
        "job_id",
        "task_id",
        "ordinal",
    ),
    "job_task_attempts_deadline": (
        "agent_id",
        "job_id",
        "state",
        "absolute_deadline_at_us",
    ),
    "job_task_controls_open": ("agent_id", "job_id", "state", "created_at_us"),
    "job_graph_events_job": ("agent_id", "job_id", "event_id"),
    "job_graph_events_task": ("agent_id", "job_id", "task_id", "event_id"),
    "job_graph_events_kind": ("agent_id", "kind", "event_id"),
    "deliveries_conversation_history": (
        "agent_id",
        "conversation_id",
        "created_at_us",
        "delivery_id",
    ),
    "scheduled_routines_due": (
        "agent_id",
        "state",
        "next_due_at_us",
        "routine_id",
    ),
    "routine_occurrences_stale": (
        "agent_id",
        "state",
        "lease_expires_at_us",
        "occurrence_id",
    ),
}


def configure_graph_connection(connection: sqlite3.Connection) -> None:
    """Apply the frozen local revision-2 connection policy."""

    connection.execute("PRAGMA foreign_keys = ON")
    mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    if mode != "wal":
        raise ValueError("graph database requires WAL journal mode")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA wal_autocheckpoint = 1000")


def create_graph_database(connection: sqlite3.Connection) -> None:
    """Create one empty, unstamped revision-2 target database."""

    if (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
        ).fetchone()
        is not None
    ):
        raise ValueError("graph database builder requires an empty database")
    configure_graph_connection(connection)
    connection.executescript("BEGIN IMMEDIATE;\n" + REVISION_2_DATABASE_SQL)
    connection.commit()
    require_graph_schema(connection)


def initialize_graph_database(path: Path) -> None:
    """Create a revision-2 database without a home migration stamp."""

    if not isinstance(path, Path):
        raise TypeError("graph database path must be Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    connection = sqlite3.connect(path)
    try:
        create_graph_database(connection)
    finally:
        connection.close()
    os.chmod(path, 0o600)


def connect_graph(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(
            f"file:{quote(os.fspath(path))}?mode=ro",
            uri=True,
            timeout=5,
            factory=ClosingSQLiteConnection,
        )
        connection.execute("PRAGMA query_only = ON")
    else:
        connection = sqlite3.connect(
            path,
            timeout=5,
            factory=ClosingSQLiteConnection,
        )
    configure_graph_connection(connection)
    require_graph_schema(connection)
    return connection


def require_graph_wal(connection: sqlite3.Connection) -> None:
    expected = {
        "journal_mode": "wal",
        "synchronous": 2,
        "busy_timeout": 5_000,
        "wal_autocheckpoint": 1_000,
        "foreign_keys": 1,
    }
    for name, value in expected.items():
        observed = connection.execute(f"PRAGMA {name}").fetchone()
        if observed is None:
            raise ValueError(f"graph pragma is unavailable: {name}")
        actual = (
            str(observed[0]).lower() if name == "journal_mode" else int(observed[0])
        )
        if actual != value:
            raise ValueError(f"graph pragma is invalid: {name}")


def require_graph_schema(connection: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if tables != GRAPH_TABLE_NAMES:
        raise ValueError("graph tables do not match the frozen target")
    if "autonomous_followups" in tables:
        raise ValueError("graph target cannot contain autonomous follow-ups")
    indexes = {
        str(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if set(indexes) != set(GRAPH_NAMED_INDEXES):
        raise ValueError("graph indexes do not match the frozen target")
    for name, columns in GRAPH_NAMED_INDEXES.items():
        actual = tuple(
            str(row[2]) for row in connection.execute(f'PRAGMA index_info("{name}")')
        )
        if actual != columns:
            raise ValueError(f"graph index columns are invalid: {name}")
    for table in (
        "job_runs",
        "job_graphs",
        "job_tasks",
        "job_task_dependencies",
        "job_task_attempts",
        "job_task_results",
        "job_task_checkpoints",
        "job_task_comments",
        "job_task_controls",
        "job_graph_mutations",
        "job_graph_events",
    ):
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if row is None or "CHECK (json_valid(data))" not in " ".join(
            str(row[0]).split()
        ):
            raise ValueError(f"graph JSON check is absent: {table}")
    if connection.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
        raise ValueError("graph database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("graph database foreign keys are invalid")
    require_graph_wal(connection)


__all__ = [
    "GRAPH_NAMED_INDEXES",
    "GRAPH_TABLE_NAMES",
    "GRAPH_RECEIPT_SQL",
    "GRAPH_TABLE_SQL",
    "REVISION_2_DATABASE_SQL",
    "configure_graph_connection",
    "connect_graph",
    "create_graph_database",
    "initialize_graph_database",
    "require_graph_schema",
    "require_graph_wal",
]
