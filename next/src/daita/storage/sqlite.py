"""Standard-library SQLite foundation for the durable operation store.

The generic loop and operation runtime depend only on ``OperationStore``.
This module owns SQLite connection policy, file identity, migration history,
and backup-before-migrate behavior for the concrete adapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import TypeVar

from .._json import canonical_json
from ..capabilities import AccessMode, RiskLevel
from ..events.models import CommittedEvent, EventCursor, RuntimeEvent
from ..events.protocols import (
    EventCursorMismatchError,
    EventCursorNotFoundError,
)
from ..llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from ..loop.models import LoopBudgets, LoopPhase, LoopState, Readiness, Turn
from ..operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from ..operations.leases import TaskClaimRequest, TaskLease, TaskLeaseGuard
from ..operations.models import (
    AgentTrigger,
    Evidence,
    Observation,
    Operation,
    OperationStatus,
    Task,
    TaskDependency,
    TaskExecutionFacts,
    TaskStatus,
    TriggerKind,
)
from ..operations.store import (
    CommitResult,
    OperationAlreadyExistsError,
    OperationNotFoundError,
    OperationRevisionConflict,
    TaskClaimResult,
    TriggerAlreadyClaimedError,
    VersionedOperation,
    _authoritative_time,
    _committed_event_suffix,
    _prepare_expired_task_recovery,
    _prepare_fenced_task_commit,
    _prepare_task_claim,
    _prepare_task_lease_renewal,
    _require_bounded_lease_duration,
    _require_lease_duration,
    _require_revision,
    _validate_commit_candidate,
    _validate_new_checkpoint,
)

DAITA_V2_APPLICATION_ID = 0x44414932  # ASCII ``DAI2``.
_MAX_COMMITTED_EVENT_READ_LIMIT = 1_000
_COMMITTED_EVENT_SUBSCRIPTION_BATCH_SIZE = 100
_COMMITTED_EVENT_POLL_INTERVAL_SECONDS = 0.25


def _sqlite_utc_now() -> datetime:
    return datetime.now(timezone.utc)


_SCHEMA_HISTORY_SQL = """
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
)
""".strip()

_T = TypeVar("_T")
_STORE_CONSTRUCTION_TOKEN = object()
_MIGRATION_CONTROL_SQL = re.compile(
    r"\A(?:\s|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*"
    r"(?:BEGIN|COMMIT|END|ROLLBACK|SAVEPOINT|RELEASE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE | re.DOTALL,
)


class SQLiteStoreError(RuntimeError):
    """Base class for typed failures owned by the SQLite adapter."""


class SQLiteCompatibilityError(SQLiteStoreError):
    """Raised before mutation when a file is not a compatible v2 database."""


class SQLiteMigrationError(SQLiteStoreError):
    """Raised when backup or an ordered migration cannot complete atomically."""


class SQLiteCorruptionError(SQLiteStoreError):
    """Raised when durable rows cannot reconstruct their canonical records."""


class SQLiteCommitOutcomeUnknownError(SQLiteStoreError):
    """Raised when SQLite cannot prove whether one candidate was committed."""

    def __init__(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        candidate_revision: int,
    ) -> None:
        self.operation_id = operation_id
        self.expected_revision = expected_revision
        self.candidate_revision = candidate_revision
        super().__init__(
            f"SQLite commit outcome is unknown for {operation_id}: expected "
            f"revision {expected_revision}, candidate {candidate_revision}"
        )


@dataclass(frozen=True, slots=True)
class _SQLiteMigration:
    """One package-owned, checksummed SQLite migration."""

    version: int
    name: str
    statements: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise ValueError("migration version must be a positive integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("migration name must be a non-empty string")
        if isinstance(self.statements, (str, bytes)):
            raise TypeError("migration statements must be a sequence of SQL strings")
        statements = tuple(self.statements)
        if any(
            not isinstance(statement, str) or not statement.strip()
            for statement in statements
        ):
            raise ValueError("migration statements must contain non-empty SQL strings")
        if any(_MIGRATION_CONTROL_SQL.match(statement) for statement in statements):
            raise ValueError(
                "migration statements cannot own transaction or connection control"
            )
        object.__setattr__(self, "statements", statements)

    @property
    def checksum(self) -> str:
        """Return the deterministic digest of identity and ordered statements."""

        material = canonical_json(
            {
                "name": self.name,
                "statements": self.statements,
                "version": self.version,
            }
        )
        return sha256(material.encode("utf-8")).hexdigest()


_LIFECYCLE_SCHEMA_SQL = (
    """
    CREATE TABLE triggers (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        source_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        session_id TEXT
    )
    """.strip(),
    """
    CREATE TABLE operations (
        id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        agent_id TEXT NOT NULL,
        trigger_id TEXT NOT NULL UNIQUE REFERENCES triggers(id),
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        session_id TEXT,
        final_text TEXT,
        terminal_reason TEXT
    )
    """.strip(),
    """
    CREATE TABLE loop_state (
        operation_id TEXT PRIMARY KEY REFERENCES operations(id) ON DELETE CASCADE,
        phase TEXT NOT NULL,
        turn_count INTEGER NOT NULL,
        action_count INTEGER NOT NULL,
        repair_count INTEGER NOT NULL,
        identical_failure_count INTEGER NOT NULL,
        observation_characters INTEGER NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        estimated_cost_usd TEXT NOT NULL,
        waiting_approval_id TEXT,
        interruption_reason TEXT,
        final_answer_candidate TEXT,
        no_progress_fingerprints_json TEXT NOT NULL,
        budget_max_turns INTEGER NOT NULL,
        budget_max_actions INTEGER NOT NULL,
        budget_max_repairs INTEGER NOT NULL,
        budget_max_identical_failures INTEGER NOT NULL,
        budget_max_observation_characters INTEGER NOT NULL,
        budget_max_total_tokens INTEGER NOT NULL,
        budget_max_wall_time_seconds REAL NOT NULL,
        budget_task_timeout_seconds REAL NOT NULL,
        budget_max_estimated_cost_usd TEXT
    )
    """.strip(),
    """
    CREATE TABLE turns (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        id TEXT PRIMARY KEY,
        number INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        model_request_id TEXT,
        model_response_id TEXT,
        UNIQUE (operation_id, position),
        UNIQUE (operation_id, id),
        FOREIGN KEY (operation_id, model_request_id)
            REFERENCES model_calls(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, model_response_id)
            REFERENCES model_calls(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    CREATE TABLE model_calls (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        id TEXT PRIMARY KEY,
        turn_id TEXT NOT NULL,
        provider_id TEXT NOT NULL,
        request_json TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        response_json TEXT,
        error_code TEXT,
        cancellation_requested INTEGER NOT NULL CHECK (
            cancellation_requested IN (0, 1)
        ),
        UNIQUE (operation_id, position),
        UNIQUE (operation_id, id),
        FOREIGN KEY (operation_id, turn_id)
            REFERENCES turns(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    CREATE TABLE readiness (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        allowed INTEGER NOT NULL CHECK (allowed IN (0, 1)),
        code TEXT NOT NULL,
        message TEXT NOT NULL,
        evaluated_at TEXT NOT NULL,
        missing_facts_json TEXT NOT NULL,
        PRIMARY KEY (operation_id, position)
    )
    """.strip(),
    """
    CREATE TABLE tasks (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        id TEXT PRIMARY KEY,
        turn_id TEXT NOT NULL,
        call_id TEXT NOT NULL,
        capability_id TEXT NOT NULL,
        executor_id TEXT NOT NULL,
        status TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        arguments_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error_code TEXT,
        cancellation_requested INTEGER NOT NULL CHECK (
            cancellation_requested IN (0, 1)
        ),
        UNIQUE (operation_id, position),
        UNIQUE (operation_id, id),
        FOREIGN KEY (operation_id, turn_id)
            REFERENCES turns(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    CREATE TABLE evidence (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        capability_id TEXT NOT NULL,
        executor_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        attempt INTEGER NOT NULL,
        accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
        payload_json TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (operation_id, position),
        UNIQUE (operation_id, id),
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, turn_id)
            REFERENCES turns(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    CREATE TABLE task_evidence (
        operation_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        position INTEGER NOT NULL CHECK (position >= 0),
        evidence_id TEXT NOT NULL,
        PRIMARY KEY (operation_id, task_id, position),
        UNIQUE (operation_id, evidence_id),
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) ON DELETE CASCADE,
        FOREIGN KEY (operation_id, evidence_id)
            REFERENCES evidence(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    CREATE TABLE observations (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        turn_id TEXT NOT NULL,
        code TEXT NOT NULL,
        message TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        success INTEGER NOT NULL CHECK (success IN (0, 1)),
        created_at TEXT NOT NULL,
        call_id TEXT,
        task_id TEXT,
        evidence_id TEXT,
        truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
        PRIMARY KEY (operation_id, position),
        FOREIGN KEY (operation_id, turn_id)
            REFERENCES turns(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, evidence_id)
            REFERENCES evidence(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    CREATE TABLE runtime_events (
        id TEXT PRIMARY KEY,
        operation_id TEXT,
        position INTEGER,
        type TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        session_id TEXT,
        turn_id TEXT,
        model_call_id TEXT,
        call_id TEXT,
        task_id TEXT,
        evidence_id TEXT,
        capability_id TEXT,
        executor_id TEXT,
        payload_json TEXT NOT NULL,
        CHECK (
            (operation_id IS NULL AND position IS NULL)
            OR (operation_id IS NOT NULL AND position >= 0)
        ),
        UNIQUE (operation_id, position),
        FOREIGN KEY (operation_id) REFERENCES operations(id) ON DELETE CASCADE,
        FOREIGN KEY (operation_id, turn_id)
            REFERENCES turns(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, model_call_id)
            REFERENCES model_calls(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, evidence_id)
            REFERENCES evidence(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    CREATE INDEX operations_agent_status_idx
        ON operations(agent_id, status, updated_at)
    """.strip(),
    """
    CREATE INDEX tasks_operation_status_idx
        ON tasks(operation_id, status, updated_at)
    """.strip(),
)


_COMMITTED_EVENT_SCHEMA_SQL = (
    """
    CREATE TABLE runtime_events_v3 (
        id TEXT PRIMARY KEY,
        operation_id TEXT,
        position INTEGER,
        type TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        agent_sequence INTEGER NOT NULL CHECK (agent_sequence >= 1),
        created_at TEXT NOT NULL,
        session_id TEXT,
        turn_id TEXT,
        model_call_id TEXT,
        call_id TEXT,
        task_id TEXT,
        evidence_id TEXT,
        capability_id TEXT,
        executor_id TEXT,
        payload_json TEXT NOT NULL,
        CHECK (
            (operation_id IS NULL AND position IS NULL)
            OR (operation_id IS NOT NULL AND position >= 0)
        ),
        UNIQUE (operation_id, position),
        UNIQUE (agent_id, agent_sequence),
        FOREIGN KEY (operation_id) REFERENCES operations(id),
        FOREIGN KEY (operation_id, turn_id)
            REFERENCES turns(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, model_call_id)
            REFERENCES model_calls(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) DEFERRABLE INITIALLY DEFERRED,
        FOREIGN KEY (operation_id, evidence_id)
            REFERENCES evidence(operation_id, id) DEFERRABLE INITIALLY DEFERRED
    )
    """.strip(),
    """
    INSERT INTO runtime_events_v3(
        id, operation_id, position, type, agent_id, agent_sequence, created_at,
        session_id, turn_id, model_call_id, call_id, task_id, evidence_id,
        capability_id, executor_id, payload_json
    )
    SELECT
        id, operation_id, position, type, agent_id,
        ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY rowid),
        created_at, session_id, turn_id, model_call_id, call_id, task_id,
        evidence_id, capability_id, executor_id, payload_json
    FROM runtime_events
    ORDER BY rowid
    """.strip(),
    "DROP TABLE runtime_events",
    "ALTER TABLE runtime_events_v3 RENAME TO runtime_events",
    """
    CREATE TRIGGER runtime_events_reject_update
    BEFORE UPDATE ON runtime_events
    BEGIN
        SELECT RAISE(ABORT, 'runtime_events are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER runtime_events_reject_delete
    BEFORE DELETE ON runtime_events
    BEGIN
        SELECT RAISE(ABORT, 'runtime_events are append-only');
    END
    """.strip(),
)


_FENCED_TASK_EXECUTION_SCHEMA_SQL = (
    """
    ALTER TABLE tasks ADD COLUMN capability_fingerprint TEXT NOT NULL
        DEFAULT 'sha256:0000000000000000000000000000000000000000000000000000000000000000'
        CHECK (
            length(capability_fingerprint) = 71
            AND substr(capability_fingerprint, 1, 7) = 'sha256:'
            AND substr(capability_fingerprint, 8) NOT GLOB '*[^0-9a-f]*'
        )
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN arguments_hash TEXT NOT NULL
        DEFAULT 'sha256:0000000000000000000000000000000000000000000000000000000000000000'
        CHECK (
            length(arguments_hash) = 71
            AND substr(arguments_hash, 1, 7) = 'sha256:'
            AND substr(arguments_hash, 8) NOT GLOB '*[^0-9a-f]*'
        )
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN access_mode TEXT NOT NULL DEFAULT 'write'
        CHECK (access_mode IN ('read', 'write'))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN risk TEXT NOT NULL DEFAULT 'high'
        CHECK (risk IN ('low', 'medium', 'high'))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN side_effecting INTEGER NOT NULL DEFAULT 1
        CHECK (side_effecting IN (0, 1))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN idempotent INTEGER NOT NULL DEFAULT 0
        CHECK (idempotent IN (0, 1))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN replay_safe INTEGER NOT NULL DEFAULT 0
        CHECK (replay_safe IN (0, 1))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN idempotency_key TEXT
        CHECK (
            (idempotency_key IS NULL OR length(trim(idempotency_key)) > 0)
            AND (access_mode != 'read' OR side_effecting = 0)
            AND (replay_safe = 0 OR idempotent = 1)
            AND (
                idempotency_key IS NULL
                OR (side_effecting = 1 AND idempotent = 1)
            )
            AND (
                side_effecting = 0
                OR replay_safe = 0
                OR idempotency_key IS NOT NULL
            )
        )
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN manual_recovery_reason TEXT
        CHECK (
            (
                status = 'manual_recovery_required'
                AND manual_recovery_reason IS NOT NULL
                AND length(trim(manual_recovery_reason)) > 0
            )
            OR (
                status != 'manual_recovery_required'
                AND manual_recovery_reason IS NULL
            )
        )
    """.strip(),
    """
    CREATE TABLE task_dependencies (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        task_id TEXT NOT NULL,
        prerequisite_task_id TEXT NOT NULL,
        PRIMARY KEY (operation_id, position),
        UNIQUE (operation_id, task_id, prerequisite_task_id),
        CHECK (task_id != prerequisite_task_id),
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) ON DELETE CASCADE,
        FOREIGN KEY (operation_id, prerequisite_task_id)
            REFERENCES tasks(operation_id, id) ON DELETE CASCADE
    )
    """.strip(),
    """
    CREATE INDEX task_dependencies_task_idx
        ON task_dependencies(operation_id, task_id, position)
    """.strip(),
    """
    CREATE INDEX task_dependencies_prerequisite_idx
        ON task_dependencies(operation_id, prerequisite_task_id, position)
    """.strip(),
    """
    CREATE TRIGGER task_dependencies_reject_update
    BEFORE UPDATE ON task_dependencies
    BEGIN
        SELECT RAISE(ABORT, 'task dependencies are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER task_dependencies_reject_delete
    BEFORE DELETE ON task_dependencies
    BEGIN
        SELECT RAISE(ABORT, 'task dependencies are append-only');
    END
    """.strip(),
    """
    CREATE TABLE task_leases (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        task_id TEXT NOT NULL,
        attempt INTEGER NOT NULL CHECK (attempt >= 1),
        fencing_token INTEGER NOT NULL CHECK (fencing_token >= 1),
        holder_id TEXT NOT NULL CHECK (length(trim(holder_id)) > 0),
        acquired_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        started_at TEXT,
        renewed_at TEXT,
        released_at TEXT,
        release_reason TEXT,
        PRIMARY KEY (operation_id, position),
        UNIQUE (operation_id, task_id, attempt),
        CHECK (expires_at > acquired_at),
        CHECK (
            started_at IS NULL
            OR (started_at >= acquired_at AND started_at < expires_at)
        ),
        CHECK (
            renewed_at IS NULL
            OR (renewed_at >= acquired_at AND renewed_at < expires_at)
        ),
        CHECK (
            (released_at IS NULL AND release_reason IS NULL)
            OR (
                released_at IS NOT NULL
                AND release_reason IS NOT NULL
                AND length(trim(release_reason)) > 0
                AND released_at >= acquired_at
                AND (started_at IS NULL OR released_at >= started_at)
                AND (renewed_at IS NULL OR released_at >= renewed_at)
            )
        ),
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) ON DELETE CASCADE
    )
    """.strip(),
    """
    CREATE UNIQUE INDEX task_leases_task_fence_idx
        ON task_leases(operation_id, task_id, fencing_token)
    """.strip(),
    """
    CREATE UNIQUE INDEX task_leases_one_unreleased_idx
        ON task_leases(operation_id, task_id)
        WHERE released_at IS NULL
    """.strip(),
    """
    CREATE TRIGGER task_leases_reject_identity_update
    BEFORE UPDATE ON task_leases
    WHEN NEW.operation_id != OLD.operation_id
        OR NEW.position != OLD.position
        OR NEW.task_id != OLD.task_id
        OR NEW.attempt != OLD.attempt
        OR NEW.fencing_token != OLD.fencing_token
        OR NEW.holder_id != OLD.holder_id
        OR NEW.acquired_at != OLD.acquired_at
    BEGIN
        SELECT RAISE(ABORT, 'task lease identity is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER task_leases_reject_released_update
    BEFORE UPDATE ON task_leases
    WHEN OLD.released_at IS NOT NULL
    BEGIN
        SELECT RAISE(ABORT, 'released task lease is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER task_leases_reject_delete
    BEFORE DELETE ON task_leases
    BEGIN
        SELECT RAISE(ABORT, 'task lease history is append-only');
    END
    """.strip(),
    """
    INSERT INTO runtime_events(
        id, operation_id, position, type, agent_id, agent_sequence, created_at,
        session_id, turn_id, model_call_id, call_id, task_id, evidence_id,
        capability_id, executor_id, payload_json
    )
    WITH legacy_running AS (
        SELECT
            task.id AS task_id,
            task.operation_id,
            task.position AS task_position,
            task.turn_id,
            task.call_id,
            task.capability_id,
            task.executor_id,
            task.updated_at,
            operation.agent_id,
            operation.session_id,
            turn.model_response_id,
            COALESCE((
                SELECT MAX(existing.position) + 1
                FROM runtime_events AS existing
                WHERE existing.operation_id = task.operation_id
            ), 0) AS operation_event_base,
            COALESCE((
                SELECT MAX(existing.agent_sequence)
                FROM runtime_events AS existing
                WHERE existing.agent_id = operation.agent_id
            ), 0) AS agent_event_base,
            ROW_NUMBER() OVER (
                PARTITION BY task.operation_id ORDER BY task.position
            ) - 1 AS operation_event_offset,
            ROW_NUMBER() OVER (
                PARTITION BY operation.agent_id
                ORDER BY task.operation_id, task.position
            ) AS agent_event_offset
        FROM tasks AS task
        JOIN operations AS operation ON operation.id = task.operation_id
        JOIN turns AS turn
            ON turn.operation_id = task.operation_id
            AND turn.id = task.turn_id
        WHERE task.status = 'running'
    )
    SELECT
        'daita:v2:migration:4:manual-recovery:'
            || lower(hex(CAST(operation_id AS BLOB)))
            || ':'
            || lower(hex(CAST(task_id AS BLOB))),
        operation_id,
        operation_event_base + operation_event_offset,
        'task.manual_recovery_required',
        agent_id,
        agent_event_base + agent_event_offset,
        updated_at,
        session_id,
        turn_id,
        model_response_id,
        call_id,
        task_id,
        NULL,
        capability_id,
        executor_id,
        '{"from_status":"running","reason":"legacy_running_task_missing_lease","to_status":"manual_recovery_required"}'
    FROM legacy_running
    ORDER BY agent_id, agent_event_offset
    """.strip(),
    """
    UPDATE operations
    SET revision = revision + 1
    WHERE EXISTS (
        SELECT 1
        FROM tasks AS task
        WHERE task.operation_id = operations.id
            AND task.status = 'running'
    )
    """.strip(),
    """
    UPDATE tasks
    SET
        status = 'manual_recovery_required',
        manual_recovery_reason = 'legacy_running_task_missing_lease'
    WHERE status = 'running'
    """.strip(),
)

_BLOB_BACKED_EVIDENCE_SCHEMA_SQL = ("ALTER TABLE evidence ADD COLUMN blob_id TEXT",)


# Migration 1 records only the v2 file/migration foundation. Migration 2 adds
# the first normalized runtime lifecycle aggregate without an opaque snapshot.
# Migration 3 assigns one append-only committed-event sequence per agent.
# Migration 4 persists immutable execution-safety facts, dependency edges, and
# fenced lease history. Legacy in-flight work is failed closed during upgrade.
# Migration 5 adds the explicit nullable link from accepted evidence to the
# separately durable content-addressed blob manifest.
_MIGRATIONS = (
    _SQLiteMigration(
        version=1,
        name="initialize_sqlite_foundation",
        statements=(),
    ),
    _SQLiteMigration(
        version=2,
        name="normalize_operation_lifecycle",
        statements=_LIFECYCLE_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=3,
        name="project_committed_event_cursors",
        statements=_COMMITTED_EVENT_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=4,
        name="normalize_fenced_task_execution",
        statements=_FENCED_TASK_EXECUTION_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=5,
        name="link_blob_backed_evidence",
        statements=_BLOB_BACKED_EVIDENCE_SCHEMA_SQL,
    ),
)


@dataclass(frozen=True, slots=True)
class SQLiteFoundation:
    """Inspectable settings for the adapter's live SQLite connection."""

    application_id: int
    journal_mode: str
    foreign_keys: bool
    busy_timeout_ms: int
    synchronous: str


class SQLiteOperationStore:
    """Concrete operation-store adapter, beginning with its durable foundation."""

    def __init__(
        self,
        path: Path,
        connection: sqlite3.Connection,
        *,
        busy_timeout_ms: int,
        clock: Callable[[], datetime] | None = None,
        max_lease_duration_seconds: float = 300.0,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _STORE_CONSTRUCTION_TOKEN:
            raise TypeError(
                "construct SQLiteOperationStore with SQLiteOperationStore.open"
            )
        self._path = path
        self._connection = connection
        self._busy_timeout_ms = busy_timeout_ms
        resolved_clock = _sqlite_utc_now if clock is None else clock
        if not callable(resolved_clock):
            raise TypeError("clock must be callable")
        self._clock = resolved_clock
        self._max_lease_duration_seconds = _require_lease_duration(
            max_lease_duration_seconds,
            "max_lease_duration_seconds",
        )
        self._lock = asyncio.Lock()
        self._event_wake_hints: dict[str, set[asyncio.Event]] = {}
        self._closed = False

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        backup_path: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        max_lease_duration_seconds: float = 300.0,
    ) -> SQLiteOperationStore:
        """Open the fixed package-owned schema for one v2 SQLite database."""

        return await _open_with_migrations(
            path,
            migrations=_MIGRATIONS,
            busy_timeout_ms=busy_timeout_ms,
            backup_path=backup_path,
            verify_owned_schema=True,
            clock=clock,
            max_lease_duration_seconds=max_lease_duration_seconds,
        )

    @property
    def closed(self) -> bool:
        """Whether this adapter has closed its owned connection."""

        return self._closed

    async def inspect_foundation(self) -> SQLiteFoundation:
        """Read the marker and connection-local correctness settings."""

        async with self._lock:
            self._require_open()
            foundation, cancellation_requested = await _await_sync_completion(
                lambda: _inspect_foundation(self._connection)
            )
        if cancellation_requested:
            raise asyncio.CancelledError
        return foundation

    async def create(self, snapshot: OperationSnapshot) -> CommitResult:
        """Atomically claim an operation/trigger and persist every lifecycle row."""

        _validate_new_checkpoint(snapshot)
        result = await self._run_connection(
            lambda connection: _create_operation(connection, snapshot)
        )
        self._publish_committed_event_wake_hints(result.committed_events)
        return result

    async def load(self, operation_id: str) -> VersionedOperation:
        """Load one self-consistent normalized operation snapshot."""

        _require_identity(operation_id, "operation_id")
        return await self._run_connection(
            lambda connection: _load_versioned_operation(connection, operation_id)
        )

    async def load_by_trigger(
        self,
        trigger_id: str,
    ) -> VersionedOperation | None:
        """Load the operation claimed by a trigger, if one exists."""

        _require_identity(trigger_id, "trigger_id")
        return await self._run_connection(
            lambda connection: _load_versioned_by_trigger(connection, trigger_id)
        )

    async def read_after(
        self,
        agent_id: str,
        cursor: EventCursor | None,
        *,
        limit: int,
    ) -> tuple[CommittedEvent, ...]:
        """Read one bounded page from an agent's committed event history."""

        _validate_committed_event_scope(agent_id, cursor)
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or limit > _MAX_COMMITTED_EVENT_READ_LIMIT
        ):
            raise ValueError(
                "committed event read limit must be an integer from 1 through "
                f"{_MAX_COMMITTED_EVENT_READ_LIMIT}"
            )
        return await self._run_connection(
            lambda connection: _read_committed_events(
                connection,
                agent_id,
                cursor=cursor,
                limit=limit,
            )
        )

    def subscribe(
        self,
        agent_id: str,
        cursor: EventCursor | None,
    ) -> AsyncGenerator[CommittedEvent, None]:
        """Follow one agent's durable event sequence from an exact cursor."""

        _validate_committed_event_scope(agent_id, cursor)
        self._require_open()
        return self._subscribe_committed_events(agent_id, cursor)

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        """Atomically compare, update mutable rows, and append lifecycle suffixes."""

        _validate_new_checkpoint(snapshot)
        _require_revision(expected_revision)
        result = await self._run_connection(
            lambda connection: _commit_operation(
                connection,
                snapshot,
                expected_revision=expected_revision,
            )
        )
        self._publish_committed_event_wake_hints(result.committed_events)
        return result

    async def claim_task(
        self,
        request: TaskClaimRequest,
        *,
        expected_revision: int,
    ) -> TaskClaimResult:
        """Atomically claim one ready task with an authoritative fenced lease."""

        if not isinstance(request, TaskClaimRequest):
            raise TypeError("request must be a TaskClaimRequest record")
        _require_revision(expected_revision)
        _require_bounded_lease_duration(
            request.lease_duration_seconds,
            maximum=self._max_lease_duration_seconds,
        )
        result = await self._run_connection(
            lambda connection: _claim_task_execution(
                connection,
                request,
                expected_revision=expected_revision,
                clock=self._clock,
                max_lease_duration_seconds=self._max_lease_duration_seconds,
            )
        )
        self._publish_committed_event_wake_hints(result.commit_result.committed_events)
        return result

    async def renew_task_lease(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
        lease_duration_seconds: float,
    ) -> CommitResult:
        """Extend one exact live fence without allowing lease resurrection."""

        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("snapshot must be an OperationSnapshot record")
        if not isinstance(guard, TaskLeaseGuard):
            raise TypeError("guard must be a TaskLeaseGuard record")
        _require_revision(expected_revision)
        _require_bounded_lease_duration(
            lease_duration_seconds,
            maximum=self._max_lease_duration_seconds,
        )
        result = await self._run_connection(
            lambda connection: _renew_task_execution_lease(
                connection,
                snapshot,
                expected_revision=expected_revision,
                guard=guard,
                lease_duration_seconds=lease_duration_seconds,
                clock=self._clock,
                max_lease_duration_seconds=self._max_lease_duration_seconds,
            )
        )
        self._publish_committed_event_wake_hints(result.committed_events)
        return result

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        """Commit one task transition only while its exact fence remains live."""

        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("snapshot must be an OperationSnapshot record")
        if not isinstance(guard, TaskLeaseGuard):
            raise TypeError("guard must be a TaskLeaseGuard record")
        _require_revision(expected_revision)
        result = await self._run_connection(
            lambda connection: _commit_fenced_task_execution(
                connection,
                snapshot,
                expected_revision=expected_revision,
                guard=guard,
                clock=self._clock,
            )
        )
        self._publish_committed_event_wake_hints(result.committed_events)
        return result

    async def recover_expired_task(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        """Release an expired fence through the portable fail-closed matrix."""

        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("snapshot must be an OperationSnapshot record")
        if not isinstance(guard, TaskLeaseGuard):
            raise TypeError("guard must be a TaskLeaseGuard record")
        _require_revision(expected_revision)
        result = await self._run_connection(
            lambda connection: _recover_expired_task_execution(
                connection,
                snapshot,
                expected_revision=expected_revision,
                guard=guard,
                clock=self._clock,
            )
        )
        self._publish_committed_event_wake_hints(result.committed_events)
        return result

    async def close(self) -> None:
        """Close the SQLite connection before returning, even under cancellation."""

        cancellation_requested = await _acquire_lock_resistant(self._lock)
        try:
            if self._closed:
                if cancellation_requested:
                    raise asyncio.CancelledError
                return
            _, close_cancelled = await _await_sync_completion(self._connection.close)
            self._closed = True
        finally:
            self._lock.release()
        if cancellation_requested or close_cancelled:
            raise asyncio.CancelledError

    def _require_open(self) -> None:
        if self._closed:
            raise SQLiteStoreError(f"SQLite operation store is closed: {self._path}")

    def _publish_committed_event_wake_hints(
        self,
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        try:
            self._notify_committed_events(events)
        except Exception:
            # Durable replay is authoritative. A failed in-process wake hint
            # must never turn a successful transaction into a reported failure.
            return

    def _notify_committed_events(self, events: tuple[RuntimeEvent, ...]) -> None:
        notified_agents: set[str] = set()
        for event in events:
            if event.agent_id in notified_agents:
                continue
            notified_agents.add(event.agent_id)
            for wake_hint in tuple(self._event_wake_hints.get(event.agent_id, ())):
                wake_hint.set()

    async def _subscribe_committed_events(
        self,
        agent_id: str,
        cursor: EventCursor | None,
    ) -> AsyncGenerator[CommittedEvent, None]:
        wake_hint = asyncio.Event()
        subscribers = self._event_wake_hints.setdefault(agent_id, set())
        subscribers.add(wake_hint)
        current_cursor = cursor
        try:
            while True:
                page = await self.read_after(
                    agent_id,
                    current_cursor,
                    limit=_COMMITTED_EVENT_SUBSCRIPTION_BATCH_SIZE,
                )
                if not page:
                    wake_hint.clear()
                    page = await self.read_after(
                        agent_id,
                        current_cursor,
                        limit=_COMMITTED_EVENT_SUBSCRIPTION_BATCH_SIZE,
                    )
                    if not page:
                        try:
                            await asyncio.wait_for(
                                wake_hint.wait(),
                                timeout=_COMMITTED_EVENT_POLL_INTERVAL_SECONDS,
                            )
                        except TimeoutError:
                            pass
                        continue
                for committed_event in page:
                    current_cursor = committed_event.cursor
                    yield committed_event
        finally:
            subscribers.discard(wake_hint)
            if not subscribers:
                self._event_wake_hints.pop(agent_id, None)

    async def _run_connection(
        self,
        callback: Callable[[sqlite3.Connection], _T],
    ) -> _T:
        async with self._lock:
            self._require_open()
            result, cancellation_requested = await _await_sync_completion(
                lambda: callback(self._connection)
            )
        if cancellation_requested:
            raise asyncio.CancelledError
        return result


async def _open_with_migrations(
    path: str | Path,
    *,
    migrations: Sequence[_SQLiteMigration],
    busy_timeout_ms: int = 5_000,
    backup_path: str | Path | None = None,
    verify_owned_schema: bool = False,
    clock: Callable[[], datetime] | None = None,
    max_lease_duration_seconds: float = 300.0,
) -> SQLiteOperationStore:
    """Private migration harness used by the adapter and foundation tests."""

    database_path = _validate_database_path(path)
    migration_plan = _validate_migration_plan(migrations)
    timeout = _validate_busy_timeout(busy_timeout_ms)
    resolved_clock = _sqlite_utc_now if clock is None else clock
    if not callable(resolved_clock):
        raise TypeError("clock must be callable")
    lease_duration_limit = _require_lease_duration(
        max_lease_duration_seconds,
        "max_lease_duration_seconds",
    )
    resolved_backup_path = (
        None
        if backup_path is None
        else _validate_backup_path(database_path, backup_path)
    )
    existing_nonempty_file = database_path.exists() and database_path.stat().st_size > 0
    database_existed = database_path.exists()

    def open_sync() -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                database_path,
                timeout=timeout / 1_000,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            _prepare_database(
                connection,
                database_path,
                migrations=migration_plan,
                busy_timeout_ms=timeout,
                backup_path=resolved_backup_path,
                existing_nonempty_file=existing_nonempty_file,
                verify_owned_schema=verify_owned_schema,
            )
            return connection
        except BaseException as error:
            if connection is not None:
                try:
                    connection.close()
                finally:
                    if not existing_nonempty_file:
                        _restore_failed_initialization(
                            database_path,
                            database_existed=database_existed,
                        )
            elif not existing_nonempty_file and not database_existed:
                _remove_sqlite_files(database_path)
            if isinstance(error, SQLiteStoreError):
                raise
            raise

    connection, cancellation_requested = await _await_sync_completion(open_sync)
    if cancellation_requested:
        try:
            await _await_sync_completion(connection.close)
        except BaseException as close_error:
            raise asyncio.CancelledError from close_error
        raise asyncio.CancelledError
    return SQLiteOperationStore(
        database_path,
        connection,
        busy_timeout_ms=timeout,
        clock=resolved_clock,
        max_lease_duration_seconds=lease_duration_limit,
        _construction_token=_STORE_CONSTRUCTION_TOKEN,
    )


async def _await_sync_completion(
    callback: Callable[[], _T],
) -> tuple[_T, bool]:
    """Resolve an offloaded SQLite call before propagating caller cancellation."""

    worker = asyncio.create_task(asyncio.to_thread(callback))
    cancellation_requested = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancellation_requested = True
            continue
        except BaseException:
            break
    try:
        result = worker.result()
    except BaseException as error:
        if cancellation_requested and not isinstance(error, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return result, cancellation_requested


async def _acquire_lock_resistant(lock: asyncio.Lock) -> bool:
    """Acquire a required cleanup lock before honoring caller cancellation."""

    acquisition = asyncio.create_task(lock.acquire())
    cancellation_requested = False
    while not acquisition.done():
        try:
            await asyncio.shield(acquisition)
        except asyncio.CancelledError:
            cancellation_requested = True
            continue
        except BaseException:
            break
    try:
        acquisition.result()
    except BaseException as error:
        if cancellation_requested and not isinstance(error, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return cancellation_requested


def _validate_database_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("SQLite database path must be a string or Path")
    if str(path) == ":memory:":
        raise ValueError("SQLiteOperationStore requires a durable filesystem path")
    database_path = Path(path)
    if database_path.exists() and database_path.is_dir():
        raise ValueError(f"SQLite database path is a directory: {database_path}")
    return database_path


def _validate_backup_path(database_path: Path, backup_path: str | Path) -> Path:
    if not isinstance(backup_path, (str, Path)):
        raise TypeError("SQLite backup path must be a string or Path")
    if str(backup_path) == ":memory:":
        raise ValueError("SQLite migration backup requires a durable filesystem path")
    resolved = Path(backup_path)
    if resolved.resolve() == database_path.resolve():
        raise ValueError("SQLite backup path must differ from the source database")
    return resolved


def _validate_busy_timeout(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("busy_timeout_ms must be a positive integer")
    return value


def _validate_migration_plan(
    migrations: Sequence[_SQLiteMigration],
) -> tuple[_SQLiteMigration, ...]:
    if isinstance(migrations, (str, bytes)):
        raise TypeError("migrations must be a sequence of _SQLiteMigration records")
    plan = tuple(migrations)
    if not plan:
        raise ValueError("SQLite migration plan must not be empty")
    if any(not isinstance(migration, _SQLiteMigration) for migration in plan):
        raise TypeError("migrations must contain _SQLiteMigration records")
    expected_versions = tuple(range(1, len(plan) + 1))
    actual_versions = tuple(migration.version for migration in plan)
    if actual_versions != expected_versions:
        raise ValueError(
            "SQLite migrations must be contiguous and ordered from version 1"
        )
    names = tuple(migration.name for migration in plan)
    if len(names) != len(set(names)):
        raise ValueError("SQLite migration names must be unique")
    return plan


def _prepare_database(
    connection: sqlite3.Connection,
    database_path: Path,
    *,
    migrations: tuple[_SQLiteMigration, ...],
    busy_timeout_ms: int,
    backup_path: Path | None,
    existing_nonempty_file: bool,
    verify_owned_schema: bool,
) -> None:
    try:
        application_id = _pragma_int(connection, "application_id")
        user_tables = _user_tables(connection)
        user_version = _pragma_int(connection, "user_version")
    except sqlite3.DatabaseError as error:
        raise SQLiteCompatibilityError(
            f"not a readable SQLite v2 database: {database_path}"
        ) from error

    is_new = application_id == 0 and not existing_nonempty_file and not user_tables
    if not is_new and application_id != DAITA_V2_APPLICATION_ID:
        raise SQLiteCompatibilityError(
            f"database does not carry the Daita v2 marker: {database_path}"
        )

    applied = (
        ()
        if is_new
        else _validated_history(
            connection,
            database_path,
            migrations=migrations,
            user_tables=user_tables,
            user_version=user_version,
        )
    )
    pending = migrations[len(applied) :]

    if verify_owned_schema and not is_new:
        _validate_schema_manifest(
            connection,
            database_path,
            migrations=applied,
        )

    if pending and not is_new:
        destination = backup_path or database_path.with_name(
            f"{database_path.name}.before-v{len(applied)}.bak"
        )
        _backup_database(
            connection,
            database_path,
            destination,
            expected_history=applied,
        )

    _configure_connection(connection, busy_timeout_ms)
    if pending:
        _apply_migrations(
            connection,
            database_path,
            pending=pending,
            create_history_table="schema_migrations" not in user_tables,
        )

    if verify_owned_schema:
        _validate_schema_manifest(
            connection,
            database_path,
            migrations=migrations,
        )

    foundation = _inspect_foundation(connection)
    if foundation.application_id != DAITA_V2_APPLICATION_ID:
        raise SQLiteMigrationError(
            f"migration did not persist the Daita v2 marker: {database_path}"
        )


def _validated_history(
    connection: sqlite3.Connection,
    database_path: Path,
    *,
    migrations: tuple[_SQLiteMigration, ...],
    user_tables: set[str],
    user_version: int,
) -> tuple[_SQLiteMigration, ...]:
    if "schema_migrations" not in user_tables:
        raise SQLiteCompatibilityError(
            f"marked database has no recognized migration history: {database_path}"
        )

    try:
        rows = connection.execute(
            "SELECT version, name, checksum " "FROM schema_migrations ORDER BY version"
        ).fetchall()
    except sqlite3.DatabaseError as error:
        raise SQLiteCompatibilityError(
            f"cannot read SQLite migration history: {database_path}"
        ) from error

    if len(rows) > len(migrations):
        raise SQLiteCompatibilityError(
            f"database schema is newer than this Daita build: {database_path}"
        )
    applied: list[_SQLiteMigration] = []
    for index, row in enumerate(rows, start=1):
        try:
            version = _sqlite_int(row[0], "migration version")
            name = _sqlite_text(row[1], "migration name")
            checksum = _sqlite_text(row[2], "migration checksum")
        except (TypeError, ValueError, IndexError) as error:
            raise SQLiteCompatibilityError(
                f"invalid SQLite migration history row: {database_path}"
            ) from error
        if version != index:
            raise SQLiteCompatibilityError(
                f"SQLite migration history is not contiguous: {database_path}"
            )
        expected = migrations[index - 1]
        if name != expected.name or checksum != expected.checksum:
            raise SQLiteCompatibilityError(
                f"SQLite migration history has drifted at version {version}: "
                f"{database_path}"
            )
        applied.append(expected)

    if user_version != len(applied):
        raise SQLiteCompatibilityError(
            f"SQLite schema version disagrees with migration history: {database_path}"
        )
    return tuple(applied)


def _backup_database(
    source: sqlite3.Connection,
    database_path: Path,
    backup_path: Path,
    *,
    expected_history: tuple[_SQLiteMigration, ...],
) -> None:
    backup_existed = backup_path.exists()
    destination: sqlite3.Connection | None = None
    try:
        destination = sqlite3.connect(
            backup_path,
            isolation_level=None,
            check_same_thread=False,
        )
        if not backup_existed:
            source.backup(destination)
        if not _backup_matches_source(
            source,
            destination,
            expected_history=expected_history,
        ):
            raise SQLiteMigrationError(
                f"SQLite migration backup is absent, stale, or invalid: {backup_path}"
            )
    except BaseException as error:
        if destination is not None:
            destination.close()
            destination = None
        if not backup_existed and backup_path.exists():
            backup_path.unlink()
        if isinstance(error, SQLiteMigrationError):
            raise
        raise SQLiteMigrationError(
            f"could not back up {database_path} before migration"
        ) from error
    finally:
        if destination is not None:
            destination.close()


def _backup_matches_source(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    expected_history: tuple[_SQLiteMigration, ...],
) -> bool:
    try:
        expected_rows = [
            (migration.version, migration.name, migration.checksum)
            for migration in expected_history
        ]
        copied_rows = destination.execute(
            "SELECT version, name, checksum " "FROM schema_migrations ORDER BY version"
        ).fetchall()
        return (
            _pragma_int(destination, "application_id") == DAITA_V2_APPLICATION_ID
            and _pragma_int(destination, "user_version") == len(expected_history)
            and copied_rows == expected_rows
            and _database_dump_digest(destination) == _database_dump_digest(source)
        )
    except sqlite3.DatabaseError:
        return False


def _database_dump_digest(connection: sqlite3.Connection) -> str:
    digest = sha256()
    for statement in connection.iterdump():
        digest.update(statement.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_schema_manifest(
    connection: sqlite3.Connection,
    database_path: Path,
    *,
    migrations: tuple[_SQLiteMigration, ...],
) -> None:
    expected = _expected_schema_manifest(migrations)
    actual = _read_schema_manifest(connection)
    if actual != expected:
        raise SQLiteCompatibilityError(
            f"SQLite schema objects have drifted from migration history: "
            f"{database_path}"
        )


def _expected_schema_manifest(
    migrations: tuple[_SQLiteMigration, ...],
) -> tuple[tuple[str, str, str, str], ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.execute(_SCHEMA_HISTORY_SQL)
        for migration in migrations:
            for statement in migration.statements:
                connection.execute(statement)
        return _read_schema_manifest(connection)
    finally:
        connection.close()


def _read_schema_manifest(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str], ...]:
    rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema "
        "WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
        "ORDER BY type, name"
    ).fetchall()
    return tuple(
        (
            _sqlite_text(row[0], "schema object type"),
            _sqlite_text(row[1], "schema object name"),
            _sqlite_text(row[2], "schema table name"),
            _sqlite_text(row[3], "schema SQL"),
        )
        for row in rows
    )


def _restore_failed_initialization(
    database_path: Path,
    *,
    database_existed: bool,
) -> None:
    _remove_sqlite_files(database_path)
    if database_existed:
        database_path.touch()


def _remove_sqlite_files(database_path: Path) -> None:
    for candidate in (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    ):
        candidate.unlink(missing_ok=True)


def _configure_connection(
    connection: sqlite3.Connection,
    busy_timeout_ms: int,
) -> None:
    journal_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
    journal_mode = (
        ""
        if journal_row is None
        else _sqlite_text(journal_row[0], "journal mode").lower()
    )
    if journal_mode != "wal":
        raise SQLiteStoreError("SQLite database did not enter WAL journal mode")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    connection.execute("PRAGMA synchronous = FULL")

    foundation = _inspect_foundation(connection)
    if not foundation.foreign_keys:
        raise SQLiteStoreError("SQLite foreign-key enforcement is unavailable")
    if foundation.busy_timeout_ms != busy_timeout_ms:
        raise SQLiteStoreError("SQLite busy timeout did not apply")
    if foundation.synchronous != "FULL":
        raise SQLiteStoreError("SQLite synchronous mode is not FULL")


def _apply_migrations(
    connection: sqlite3.Connection,
    database_path: Path,
    *,
    pending: tuple[_SQLiteMigration, ...],
    create_history_table: bool,
) -> None:
    active_migration = pending[0]
    try:
        connection.execute("BEGIN IMMEDIATE")
        if create_history_table:
            connection.execute(_SCHEMA_HISTORY_SQL)
        if _pragma_int(connection, "application_id") == 0:
            connection.execute(f"PRAGMA application_id = {DAITA_V2_APPLICATION_ID}")
        for migration in pending:
            active_migration = migration
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name, checksum) "
                "VALUES (?, ?, ?)",
                (migration.version, migration.name, migration.checksum),
            )
            connection.execute(f"PRAGMA user_version = {migration.version}")
        connection.execute("COMMIT")
    except sqlite3.Error as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise SQLiteMigrationError(
            f"SQLite migration {active_migration.version} "
            f"({active_migration.name}) failed for {database_path}"
        ) from error


def _inspect_foundation(connection: sqlite3.Connection) -> SQLiteFoundation:
    synchronous_value = _pragma_int(connection, "synchronous")
    synchronous_names = {0: "OFF", 1: "NORMAL", 2: "FULL", 3: "EXTRA"}
    journal_row = connection.execute("PRAGMA journal_mode").fetchone()
    journal_mode = (
        "" if journal_row is None else _sqlite_text(journal_row[0], "journal mode")
    )
    return SQLiteFoundation(
        application_id=_pragma_int(connection, "application_id"),
        journal_mode=journal_mode,
        foreign_keys=bool(_pragma_int(connection, "foreign_keys")),
        busy_timeout_ms=_pragma_int(connection, "busy_timeout"),
        synchronous=synchronous_names.get(
            synchronous_value,
            f"UNKNOWN({synchronous_value})",
        ),
    )


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None:
        raise sqlite3.DatabaseError(f"PRAGMA {name} returned no value")
    return _sqlite_int(row[0], f"PRAGMA {name}")


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {_sqlite_text(row[0], "schema table name") for row in rows}


def _create_operation(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
) -> CommitResult:
    operation_id = snapshot.operation.id
    trigger_id = snapshot.trigger.id
    connection.execute("BEGIN IMMEDIATE")
    try:
        operation_row = connection.execute(
            "SELECT 1 FROM operations WHERE id = ?",
            (operation_id,),
        ).fetchone()
        if operation_row is not None:
            raise OperationAlreadyExistsError(operation_id)
        trigger_row = connection.execute(
            "SELECT id FROM operations WHERE trigger_id = ?",
            (trigger_id,),
        ).fetchone()
        if trigger_row is not None:
            raise TriggerAlreadyClaimedError(
                trigger_id,
                _sqlite_text(trigger_row[0], "claimed operation id"),
            )

        _insert_snapshot(connection, snapshot, revision=1)
        committed = VersionedOperation(snapshot=snapshot, revision=1)
        result = CommitResult(
            operation=committed,
            committed_events=snapshot.events,
        )
        _commit_with_reconciliation(connection, result)
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise

    return result


def _commit_operation(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
    *,
    expected_revision: int,
) -> CommitResult:
    operation_id = snapshot.operation.id
    connection.execute("BEGIN IMMEDIATE")
    try:
        revision_row = connection.execute(
            "SELECT revision FROM operations WHERE id = ?",
            (operation_id,),
        ).fetchone()
        if revision_row is None:
            raise OperationNotFoundError(operation_id)
        actual_revision = _sqlite_int(
            revision_row[0],
            "operation revision",
        )
        if actual_revision != expected_revision:
            raise OperationRevisionConflict(
                operation_id,
                expected_revision=expected_revision,
                actual_revision=actual_revision,
            )

        current = _load_versioned_operation_in_transaction(connection, operation_id)
        committed_events = _validate_commit_candidate(current.snapshot, snapshot)
        candidate_revision = actual_revision + 1
        _apply_commit_delta(
            connection,
            current.snapshot,
            snapshot,
            expected_revision=actual_revision,
            candidate_revision=candidate_revision,
        )
        committed = VersionedOperation(
            snapshot=snapshot,
            revision=candidate_revision,
        )
        result = CommitResult(
            operation=committed,
            committed_events=committed_events,
        )
        _commit_with_reconciliation(connection, result)
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise

    return result


def _commit_task_execution_mutation(
    connection: sqlite3.Connection,
    operation_id: str,
    *,
    expected_revision: int,
    clock: Callable[[], datetime],
    prepare: Callable[[OperationSnapshot, datetime], OperationSnapshot],
) -> CommitResult:
    """Apply one portable task transition inside the SQLite write boundary."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        revision_row = connection.execute(
            "SELECT revision FROM operations WHERE id = ?",
            (operation_id,),
        ).fetchone()
        if revision_row is None:
            raise OperationNotFoundError(operation_id)
        actual_revision = _sqlite_int(
            revision_row[0],
            "operation revision",
        )
        if actual_revision != expected_revision:
            raise OperationRevisionConflict(
                operation_id,
                expected_revision=expected_revision,
                actual_revision=actual_revision,
            )

        current = _load_versioned_operation_in_transaction(connection, operation_id)
        now = _authoritative_time(clock)
        candidate = prepare(current.snapshot, now)
        committed_events = _committed_event_suffix(current.snapshot, candidate)
        candidate_revision = actual_revision + 1
        _apply_commit_delta(
            connection,
            current.snapshot,
            candidate,
            expected_revision=actual_revision,
            candidate_revision=candidate_revision,
        )
        committed = VersionedOperation(
            snapshot=candidate,
            revision=candidate_revision,
        )
        observed = _load_versioned_operation_in_transaction(connection, operation_id)
        if observed != committed:
            raise SQLiteCorruptionError(
                f"task transition projection diverged for operation {operation_id}"
            )
        result = CommitResult(
            operation=committed,
            committed_events=committed_events,
        )
        _commit_with_reconciliation(connection, result)
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    return result


def _claim_task_execution(
    connection: sqlite3.Connection,
    request: TaskClaimRequest,
    *,
    expected_revision: int,
    clock: Callable[[], datetime],
    max_lease_duration_seconds: float,
) -> TaskClaimResult:
    commit_result = _commit_task_execution_mutation(
        connection,
        request.operation_id,
        expected_revision=expected_revision,
        clock=clock,
        prepare=lambda current, now: _prepare_task_claim(
            current,
            request,
            now=now,
            max_lease_duration_seconds=max_lease_duration_seconds,
        )[0],
    )
    snapshot = commit_result.operation.snapshot
    task = next(task for task in snapshot.tasks if task.id == request.task_id)
    lease = next(
        lease
        for lease in reversed(snapshot.task_leases)
        if lease.task_id == request.task_id and lease.released_at is None
    )
    return TaskClaimResult(
        commit_result=commit_result,
        task=task,
        lease=lease,
    )


def _renew_task_execution_lease(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
    *,
    expected_revision: int,
    guard: TaskLeaseGuard,
    lease_duration_seconds: float,
    clock: Callable[[], datetime],
    max_lease_duration_seconds: float,
) -> CommitResult:
    return _commit_task_execution_mutation(
        connection,
        guard.operation_id,
        expected_revision=expected_revision,
        clock=clock,
        prepare=lambda current, now: _prepare_task_lease_renewal(
            current,
            snapshot,
            guard,
            now=now,
            lease_duration_seconds=lease_duration_seconds,
            max_lease_duration_seconds=max_lease_duration_seconds,
        ),
    )


def _commit_fenced_task_execution(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
    *,
    expected_revision: int,
    guard: TaskLeaseGuard,
    clock: Callable[[], datetime],
) -> CommitResult:
    return _commit_task_execution_mutation(
        connection,
        guard.operation_id,
        expected_revision=expected_revision,
        clock=clock,
        prepare=lambda current, now: _prepare_fenced_task_commit(
            current,
            snapshot,
            guard,
            now=now,
        ),
    )


def _recover_expired_task_execution(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
    *,
    expected_revision: int,
    guard: TaskLeaseGuard,
    clock: Callable[[], datetime],
) -> CommitResult:
    return _commit_task_execution_mutation(
        connection,
        guard.operation_id,
        expected_revision=expected_revision,
        clock=clock,
        prepare=lambda current, now: _prepare_expired_task_recovery(
            current,
            snapshot,
            guard,
            now=now,
        ),
    )


def _commit_with_reconciliation(
    connection: sqlite3.Connection,
    result: CommitResult,
) -> None:
    try:
        connection.execute("COMMIT")
        return
    except sqlite3.Error as commit_error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
            raise SQLiteStoreError(
                "SQLite transaction failed before commit"
            ) from commit_error

        operation_id = result.operation.snapshot.operation.id
        try:
            observed = _load_after_commit(connection, operation_id)
            if _observed_commit_proves_result(observed, result.operation):
                return
        except BaseException:
            pass
        raise SQLiteCommitOutcomeUnknownError(
            operation_id,
            expected_revision=result.operation.revision - 1,
            candidate_revision=result.operation.revision,
        ) from commit_error


def _load_after_commit(
    connection: sqlite3.Connection,
    operation_id: str,
) -> VersionedOperation:
    connection.execute("BEGIN")
    try:
        return _load_versioned_operation_in_transaction(connection, operation_id)
    finally:
        if connection.in_transaction:
            connection.rollback()


def _observed_commit_proves_result(
    observed: VersionedOperation,
    candidate: VersionedOperation,
) -> bool:
    # A successor can reuse the same caller-supplied event record after this
    # writer failed, so an append-only event prefix alone cannot prove which
    # transaction committed. Only an exact read-back removes the ambiguity.
    return observed == candidate


def _apply_commit_delta(
    connection: sqlite3.Connection,
    current: OperationSnapshot,
    candidate: OperationSnapshot,
    *,
    expected_revision: int,
    candidate_revision: int,
) -> None:
    operation_id = candidate.operation.id
    state = candidate.loop_state
    loop_update = connection.execute(
        "UPDATE loop_state SET "
        "phase = ?, turn_count = ?, action_count = ?, repair_count = ?, "
        "identical_failure_count = ?, observation_characters = ?, "
        "input_tokens = ?, output_tokens = ?, estimated_cost_usd = ?, "
        "waiting_approval_id = ?, interruption_reason = ?, "
        "final_answer_candidate = ?, no_progress_fingerprints_json = ? "
        "WHERE operation_id = ?",
        (
            state.phase.value,
            state.turn_count,
            state.action_count,
            state.repair_count,
            state.identical_failure_count,
            state.observation_characters,
            state.input_tokens,
            state.output_tokens,
            _encode_decimal(state.estimated_cost_usd),
            state.waiting_approval_id,
            state.interruption_reason,
            state.final_answer_candidate,
            canonical_json(state.no_progress_fingerprints),
            operation_id,
        ),
    )
    _require_one_row(loop_update.rowcount, "loop checkpoint", operation_id)

    for position, (turn_before, turn_after) in enumerate(
        zip(current.turns, candidate.turns, strict=False)
    ):
        turn_update = connection.execute(
            "UPDATE turns SET model_request_id = ?, model_response_id = ? "
            "WHERE operation_id = ? AND position = ? AND id = ?",
            (
                turn_after.model_request_id,
                turn_after.model_response_id,
                operation_id,
                position,
                turn_before.id,
            ),
        )
        _require_one_row(turn_update.rowcount, "turn", turn_before.id)
    for position in range(len(current.turns), len(candidate.turns)):
        _insert_turn(connection, operation_id, position, candidate.turns[position])

    for position, (model_before, model_after) in enumerate(
        zip(
            current.model_calls,
            candidate.model_calls,
            strict=False,
        )
    ):
        model_update = connection.execute(
            "UPDATE model_calls SET status = ?, updated_at = ?, "
            "response_json = ?, error_code = ?, cancellation_requested = ? "
            "WHERE operation_id = ? AND position = ? AND id = ?",
            (
                model_after.status.value,
                _encode_datetime(model_after.updated_at),
                (
                    None
                    if model_after.response is None
                    else _encode_model_response(model_after.response)
                ),
                model_after.error_code,
                int(model_after.cancellation_requested),
                operation_id,
                position,
                model_before.id,
            ),
        )
        _require_one_row(model_update.rowcount, "model call", model_before.id)
    for position in range(len(current.model_calls), len(candidate.model_calls)):
        _insert_model_call(
            connection,
            operation_id,
            position,
            candidate.model_calls[position],
        )

    for position in range(len(current.readiness), len(candidate.readiness)):
        _insert_readiness(
            connection,
            operation_id,
            position,
            candidate.readiness[position],
        )

    for position, (task_before, task_after) in enumerate(
        zip(current.tasks, candidate.tasks, strict=False)
    ):
        task_update = connection.execute(
            "UPDATE tasks SET status = ?, attempt = ?, updated_at = ?, error_code = ?, "
            "cancellation_requested = ?, manual_recovery_reason = ? "
            "WHERE operation_id = ? AND position = ? AND id = ?",
            (
                task_after.status.value,
                task_after.attempt,
                _encode_datetime(task_after.updated_at),
                task_after.error_code,
                int(task_after.cancellation_requested),
                task_after.manual_recovery_reason,
                operation_id,
                position,
                task_before.id,
            ),
        )
        _require_one_row(task_update.rowcount, "task", task_before.id)
    for position in range(len(current.tasks), len(candidate.tasks)):
        _insert_task(connection, operation_id, position, candidate.tasks[position])

    for position in range(
        len(current.task_dependencies),
        len(candidate.task_dependencies),
    ):
        _insert_task_dependency(
            connection,
            operation_id,
            position,
            candidate.task_dependencies[position],
        )

    _apply_task_lease_delta(connection, current, candidate)

    for position in range(len(current.evidence), len(candidate.evidence)):
        _insert_evidence(
            connection,
            operation_id,
            position,
            candidate.evidence[position],
        )
    current_task_by_id = {task.id: task for task in current.tasks}
    for task in candidate.tasks:
        prior_task = current_task_by_id.get(task.id)
        prior_count = 0 if prior_task is None else len(prior_task.evidence_ids)
        for position in range(prior_count, len(task.evidence_ids)):
            _insert_task_evidence(
                connection,
                operation_id,
                task.id,
                position,
                task.evidence_ids[position],
            )

    for position in range(len(current.observations), len(candidate.observations)):
        _insert_observation(
            connection,
            operation_id,
            position,
            candidate.observations[position],
        )
    for position in range(len(current.events), len(candidate.events)):
        _insert_runtime_event(
            connection,
            operation_id,
            position,
            candidate.events[position],
        )

    operation = candidate.operation
    operation_update = connection.execute(
        "UPDATE operations SET revision = ?, status = ?, updated_at = ?, "
        "final_text = ?, terminal_reason = ? WHERE id = ? AND revision = ?",
        (
            candidate_revision,
            operation.status.value,
            _encode_datetime(operation.updated_at),
            operation.final_text,
            operation.terminal_reason,
            operation_id,
            expected_revision,
        ),
    )
    if operation_update.rowcount != 1:
        revision_row = connection.execute(
            "SELECT revision FROM operations WHERE id = ?",
            (operation_id,),
        ).fetchone()
        if revision_row is None:
            raise OperationNotFoundError(operation_id)
        raise OperationRevisionConflict(
            operation_id,
            expected_revision=expected_revision,
            actual_revision=_sqlite_int(
                revision_row[0],
                "operation revision",
            ),
        )


def _apply_task_lease_delta(
    connection: sqlite3.Connection,
    current: OperationSnapshot,
    candidate: OperationSnapshot,
) -> None:
    operation_id = candidate.operation.id
    for position, (before, after) in enumerate(
        zip(current.task_leases, candidate.task_leases, strict=False)
    ):
        if after == before:
            continue
        lease_update = connection.execute(
            "UPDATE task_leases SET expires_at = ?, started_at = ?, "
            "renewed_at = ?, released_at = ?, release_reason = ? "
            "WHERE operation_id = ? AND position = ? AND task_id = ? "
            "AND attempt = ? AND fencing_token = ? AND holder_id = ? "
            "AND acquired_at = ? AND expires_at = ? "
            "AND started_at IS ? AND renewed_at IS ? "
            "AND released_at IS ? AND release_reason IS ?",
            (
                _encode_datetime(after.expires_at),
                _encode_optional_datetime(after.started_at),
                _encode_optional_datetime(after.renewed_at),
                _encode_optional_datetime(after.released_at),
                after.release_reason,
                operation_id,
                position,
                before.task_id,
                before.attempt,
                before.fencing_token,
                before.holder_id,
                _encode_datetime(before.acquired_at),
                _encode_datetime(before.expires_at),
                _encode_optional_datetime(before.started_at),
                _encode_optional_datetime(before.renewed_at),
                _encode_optional_datetime(before.released_at),
                before.release_reason,
            ),
        )
        _require_one_row(lease_update.rowcount, "task lease", before.task_id)
    for position in range(len(current.task_leases), len(candidate.task_leases)):
        _insert_task_lease(
            connection,
            operation_id,
            position,
            candidate.task_leases[position],
        )


def _require_one_row(rowcount: int, label: str, identity: str) -> None:
    if rowcount != 1:
        raise SQLiteCorruptionError(
            f"normalized {label} row disappeared during commit: {identity}"
        )


def _insert_snapshot(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
    *,
    revision: int,
) -> None:
    trigger = snapshot.trigger
    operation = snapshot.operation
    state = snapshot.loop_state
    budgets = snapshot.budgets
    connection.execute(
        "INSERT INTO triggers("
        "id, agent_id, kind, source_id, payload_json, created_at, session_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            trigger.id,
            trigger.agent_id,
            trigger.kind.value,
            trigger.source_id,
            canonical_json(trigger.payload),
            _encode_datetime(trigger.created_at),
            trigger.session_id,
        ),
    )
    connection.execute(
        "INSERT INTO operations("
        "id, revision, agent_id, trigger_id, status, created_at, updated_at, "
        "session_id, final_text, terminal_reason"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            operation.id,
            revision,
            operation.agent_id,
            operation.trigger_id,
            operation.status.value,
            _encode_datetime(operation.created_at),
            _encode_datetime(operation.updated_at),
            operation.session_id,
            operation.final_text,
            operation.terminal_reason,
        ),
    )
    connection.execute(
        "INSERT INTO loop_state("
        "operation_id, phase, turn_count, action_count, repair_count, "
        "identical_failure_count, observation_characters, input_tokens, "
        "output_tokens, estimated_cost_usd, waiting_approval_id, "
        "interruption_reason, final_answer_candidate, "
        "no_progress_fingerprints_json, budget_max_turns, budget_max_actions, "
        "budget_max_repairs, budget_max_identical_failures, "
        "budget_max_observation_characters, budget_max_total_tokens, "
        "budget_max_wall_time_seconds, budget_task_timeout_seconds, "
        "budget_max_estimated_cost_usd"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?)",
        (
            operation.id,
            state.phase.value,
            state.turn_count,
            state.action_count,
            state.repair_count,
            state.identical_failure_count,
            state.observation_characters,
            state.input_tokens,
            state.output_tokens,
            _encode_decimal(state.estimated_cost_usd),
            state.waiting_approval_id,
            state.interruption_reason,
            state.final_answer_candidate,
            canonical_json(state.no_progress_fingerprints),
            budgets.max_turns,
            budgets.max_actions,
            budgets.max_repairs,
            budgets.max_identical_failures,
            budgets.max_observation_characters,
            budgets.max_total_tokens,
            budgets.max_wall_time_seconds,
            budgets.task_timeout_seconds,
            (
                None
                if budgets.max_estimated_cost_usd is None
                else _encode_decimal(budgets.max_estimated_cost_usd)
            ),
        ),
    )

    for position, turn in enumerate(snapshot.turns):
        _insert_turn(connection, operation.id, position, turn)
    for position, model_call in enumerate(snapshot.model_calls):
        _insert_model_call(connection, operation.id, position, model_call)
    for position, readiness in enumerate(snapshot.readiness):
        _insert_readiness(connection, operation.id, position, readiness)
    for position, task in enumerate(snapshot.tasks):
        _insert_task(connection, operation.id, position, task)
    for position, dependency in enumerate(snapshot.task_dependencies):
        _insert_task_dependency(
            connection,
            operation.id,
            position,
            dependency,
        )
    for position, lease in enumerate(snapshot.task_leases):
        _insert_task_lease(
            connection,
            operation.id,
            position,
            lease,
        )
    for position, evidence in enumerate(snapshot.evidence):
        _insert_evidence(connection, operation.id, position, evidence)
    for task in snapshot.tasks:
        for position, evidence_id in enumerate(task.evidence_ids):
            _insert_task_evidence(
                connection,
                operation.id,
                task.id,
                position,
                evidence_id,
            )
    for position, observation in enumerate(snapshot.observations):
        _insert_observation(connection, operation.id, position, observation)
    for position, event in enumerate(snapshot.events):
        _insert_runtime_event(connection, operation.id, position, event)


def _insert_turn(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    turn: Turn,
) -> None:
    connection.execute(
        "INSERT INTO turns("
        "operation_id, position, id, number, created_at, model_request_id, "
        "model_response_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            operation_id,
            position,
            turn.id,
            turn.number,
            _encode_datetime(turn.created_at),
            turn.model_request_id,
            turn.model_response_id,
        ),
    )


def _insert_model_call(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    model_call: ModelCall,
) -> None:
    connection.execute(
        "INSERT INTO model_calls("
        "operation_id, position, id, turn_id, provider_id, request_json, status, "
        "created_at, updated_at, response_json, error_code, "
        "cancellation_requested"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            operation_id,
            position,
            model_call.id,
            model_call.turn_id,
            model_call.provider_id,
            _encode_model_request(model_call.request),
            model_call.status.value,
            _encode_datetime(model_call.created_at),
            _encode_datetime(model_call.updated_at),
            (
                None
                if model_call.response is None
                else _encode_model_response(model_call.response)
            ),
            model_call.error_code,
            int(model_call.cancellation_requested),
        ),
    )


def _insert_readiness(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    readiness: Readiness,
) -> None:
    connection.execute(
        "INSERT INTO readiness("
        "operation_id, position, allowed, code, message, evaluated_at, "
        "missing_facts_json"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            operation_id,
            position,
            int(readiness.allowed),
            readiness.code,
            readiness.message,
            _encode_datetime(readiness.evaluated_at),
            canonical_json(readiness.missing_facts),
        ),
    )


def _insert_task(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    task: Task,
) -> None:
    connection.execute(
        "INSERT INTO tasks("
        "operation_id, position, id, turn_id, call_id, capability_id, "
        "executor_id, status, attempt, arguments_json, created_at, updated_at, "
        "error_code, cancellation_requested, capability_fingerprint, "
        "arguments_hash, access_mode, risk, side_effecting, idempotent, "
        "replay_safe, idempotency_key, manual_recovery_reason"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?)",
        (
            operation_id,
            position,
            task.id,
            task.turn_id,
            task.call_id,
            task.capability_id,
            task.executor_id,
            task.status.value,
            task.attempt,
            canonical_json(task.arguments),
            _encode_datetime(task.created_at),
            _encode_datetime(task.updated_at),
            task.error_code,
            int(task.cancellation_requested),
            task.execution_facts.capability_fingerprint,
            task.execution_facts.arguments_hash,
            task.execution_facts.access_mode.value,
            task.execution_facts.risk.value,
            int(task.execution_facts.side_effecting),
            int(task.execution_facts.idempotent),
            int(task.execution_facts.replay_safe),
            task.execution_facts.idempotency_key,
            task.manual_recovery_reason,
        ),
    )


def _insert_task_dependency(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    dependency: TaskDependency,
) -> None:
    connection.execute(
        "INSERT INTO task_dependencies("
        "operation_id, position, task_id, prerequisite_task_id"
        ") VALUES (?, ?, ?, ?)",
        (
            operation_id,
            position,
            dependency.task_id,
            dependency.prerequisite_task_id,
        ),
    )


def _insert_task_lease(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    lease: TaskLease,
) -> None:
    connection.execute(
        "INSERT INTO task_leases("
        "operation_id, position, task_id, attempt, fencing_token, holder_id, "
        "acquired_at, expires_at, started_at, renewed_at, released_at, "
        "release_reason"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            operation_id,
            position,
            lease.task_id,
            lease.attempt,
            lease.fencing_token,
            lease.holder_id,
            _encode_datetime(lease.acquired_at),
            _encode_datetime(lease.expires_at),
            _encode_optional_datetime(lease.started_at),
            _encode_optional_datetime(lease.renewed_at),
            _encode_optional_datetime(lease.released_at),
            lease.release_reason,
        ),
    )


def _insert_evidence(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    evidence: Evidence,
) -> None:
    connection.execute(
        "INSERT INTO evidence("
        "operation_id, position, id, task_id, turn_id, capability_id, "
        "executor_id, kind, schema_version, attempt, accepted, payload_json, "
        "content_hash, created_at, blob_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            operation_id,
            position,
            evidence.id,
            evidence.task_id,
            evidence.turn_id,
            evidence.capability_id,
            evidence.executor_id,
            evidence.kind,
            evidence.schema_version,
            evidence.attempt,
            int(evidence.accepted),
            canonical_json(evidence.payload),
            evidence.content_hash,
            _encode_datetime(evidence.created_at),
            evidence.blob_id,
        ),
    )


def _insert_task_evidence(
    connection: sqlite3.Connection,
    operation_id: str,
    task_id: str,
    position: int,
    evidence_id: str,
) -> None:
    connection.execute(
        "INSERT INTO task_evidence("
        "operation_id, task_id, position, evidence_id"
        ") VALUES (?, ?, ?, ?)",
        (operation_id, task_id, position, evidence_id),
    )


def _insert_observation(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    observation: Observation,
) -> None:
    connection.execute(
        "INSERT INTO observations("
        "operation_id, position, turn_id, code, message, payload_json, success, "
        "created_at, call_id, task_id, evidence_id, truncated"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            operation_id,
            position,
            observation.turn_id,
            observation.code,
            observation.message,
            canonical_json(observation.payload),
            int(observation.success),
            _encode_datetime(observation.created_at),
            observation.call_id,
            observation.task_id,
            observation.evidence_id,
            int(observation.truncated),
        ),
    )


def _insert_runtime_event(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    event: RuntimeEvent,
) -> None:
    connection.execute(
        "INSERT INTO runtime_events("
        "id, operation_id, position, type, agent_id, agent_sequence, created_at, "
        "session_id, turn_id, model_call_id, call_id, task_id, evidence_id, "
        "capability_id, executor_id, payload_json"
        ") VALUES (?, ?, ?, ?, ?, ("
        "SELECT COALESCE(MAX(agent_sequence), 0) + 1 FROM runtime_events "
        "WHERE agent_id = ?"
        "), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event.id,
            operation_id,
            position,
            event.type,
            event.agent_id,
            event.agent_id,
            _encode_datetime(event.created_at),
            event.session_id,
            event.turn_id,
            event.model_call_id,
            event.call_id,
            event.task_id,
            event.evidence_id,
            event.capability_id,
            event.executor_id,
            canonical_json(event.payload),
        ),
    )


def _read_committed_events(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    cursor: EventCursor | None,
    limit: int,
) -> tuple[CommittedEvent, ...]:
    after_sequence = 0 if cursor is None else cursor.sequence
    connection.execute("BEGIN")
    try:
        if cursor is not None:
            cursor_row = connection.execute(
                "SELECT 1 FROM runtime_events "
                "WHERE agent_id = ? AND agent_sequence = ?",
                (agent_id, cursor.sequence),
            ).fetchone()
            if cursor_row is None:
                raise EventCursorNotFoundError(cursor)
        rows = connection.execute(
            "SELECT * FROM runtime_events "
            "WHERE agent_id = ? AND agent_sequence > ? "
            "ORDER BY agent_sequence LIMIT ?",
            (agent_id, after_sequence, limit),
        ).fetchall()
        committed: list[CommittedEvent] = []
        expected_sequence = after_sequence + 1
        for row in rows:
            try:
                sequence = _sqlite_int(
                    row["agent_sequence"],
                    "event agent sequence",
                )
                if sequence != expected_sequence:
                    raise SQLiteCorruptionError(
                        f"committed event sequence for {agent_id} must be "
                        f"contiguous; expected {expected_sequence}, found {sequence}"
                    )
                event = _decode_runtime_event_row(row)
                committed.append(
                    CommittedEvent(
                        cursor=EventCursor(agent_id=agent_id, sequence=sequence),
                        event=event,
                    )
                )
            except SQLiteCorruptionError:
                raise
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise SQLiteCorruptionError(
                    f"cannot reconstruct committed event {agent_id}:"
                    f"{expected_sequence}"
                ) from error
            expected_sequence += 1
        connection.execute("COMMIT")
        return tuple(committed)
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _decode_runtime_event_row(row: sqlite3.Row) -> RuntimeEvent:
    return RuntimeEvent(
        id=_sqlite_text(row["id"], "event id"),
        type=_sqlite_text(row["type"], "event type"),
        agent_id=_sqlite_text(row["agent_id"], "event agent id"),
        operation_id=_optional_text(row["operation_id"]),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "event created_at")
        ),
        session_id=_optional_text(row["session_id"]),
        turn_id=_optional_text(row["turn_id"]),
        model_call_id=_optional_text(row["model_call_id"]),
        call_id=_optional_text(row["call_id"]),
        task_id=_optional_text(row["task_id"]),
        evidence_id=_optional_text(row["evidence_id"]),
        capability_id=_optional_text(row["capability_id"]),
        executor_id=_optional_text(row["executor_id"]),
        payload=_decode_json_object(_sqlite_text(row["payload_json"], "event payload")),
    )


def _load_versioned_operation(
    connection: sqlite3.Connection,
    operation_id: str,
) -> VersionedOperation:
    connection.execute("BEGIN")
    try:
        result = _load_versioned_operation_in_transaction(connection, operation_id)
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_versioned_by_trigger(
    connection: sqlite3.Connection,
    trigger_id: str,
) -> VersionedOperation | None:
    connection.execute("BEGIN")
    try:
        row = connection.execute(
            "SELECT id FROM operations WHERE trigger_id = ?",
            (trigger_id,),
        ).fetchone()
        result = (
            None
            if row is None
            else _load_versioned_operation_in_transaction(
                connection,
                _sqlite_text(row[0], "operation id"),
            )
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_versioned_operation_in_transaction(
    connection: sqlite3.Connection,
    operation_id: str,
) -> VersionedOperation:
    operation_row = connection.execute(
        "SELECT * FROM operations WHERE id = ?",
        (operation_id,),
    ).fetchone()
    if operation_row is None:
        raise OperationNotFoundError(operation_id)
    try:
        snapshot = _decode_snapshot(connection, operation_row)
        revision = _sqlite_int(operation_row["revision"], "operation revision")
        return VersionedOperation(snapshot=snapshot, revision=revision)
    except (OperationNotFoundError, SQLiteCorruptionError):
        raise
    except (KeyError, IndexError, TypeError, ValueError, InvalidOperation) as error:
        raise SQLiteCorruptionError(
            f"cannot reconstruct normalized operation {operation_id}"
        ) from error


def _decode_snapshot(
    connection: sqlite3.Connection,
    operation_row: sqlite3.Row,
) -> OperationSnapshot:
    operation_id = _sqlite_text(operation_row["id"], "operation id")
    trigger_row = connection.execute(
        "SELECT * FROM triggers WHERE id = ?",
        (_sqlite_text(operation_row["trigger_id"], "operation trigger id"),),
    ).fetchone()
    loop_row = connection.execute(
        "SELECT * FROM loop_state WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()
    if trigger_row is None or loop_row is None:
        raise SQLiteCorruptionError(
            f"operation {operation_id} is missing its trigger or loop checkpoint"
        )

    trigger = AgentTrigger(
        id=_sqlite_text(trigger_row["id"], "trigger id"),
        agent_id=_sqlite_text(trigger_row["agent_id"], "trigger agent id"),
        kind=TriggerKind(_sqlite_text(trigger_row["kind"], "trigger kind")),
        source_id=_sqlite_text(trigger_row["source_id"], "trigger source id"),
        payload=_decode_json_object(
            _sqlite_text(trigger_row["payload_json"], "trigger payload")
        ),
        created_at=_decode_datetime(
            _sqlite_text(trigger_row["created_at"], "trigger created_at")
        ),
        session_id=_optional_text(trigger_row["session_id"]),
    )
    operation = Operation(
        id=operation_id,
        agent_id=_sqlite_text(operation_row["agent_id"], "operation agent id"),
        trigger_id=_sqlite_text(operation_row["trigger_id"], "operation trigger id"),
        status=OperationStatus(
            _sqlite_text(operation_row["status"], "operation status")
        ),
        created_at=_decode_datetime(
            _sqlite_text(operation_row["created_at"], "operation created_at")
        ),
        updated_at=_decode_datetime(
            _sqlite_text(operation_row["updated_at"], "operation updated_at")
        ),
        session_id=_optional_text(operation_row["session_id"]),
        final_text=_optional_text(operation_row["final_text"]),
        terminal_reason=_optional_text(operation_row["terminal_reason"]),
    )
    loop_state = LoopState(
        phase=LoopPhase(_sqlite_text(loop_row["phase"], "loop phase")),
        turn_count=_sqlite_int(loop_row["turn_count"], "loop turn count"),
        action_count=_sqlite_int(loop_row["action_count"], "loop action count"),
        repair_count=_sqlite_int(loop_row["repair_count"], "loop repair count"),
        identical_failure_count=_sqlite_int(
            loop_row["identical_failure_count"],
            "loop identical failure count",
        ),
        observation_characters=_sqlite_int(
            loop_row["observation_characters"],
            "loop observation characters",
        ),
        input_tokens=_sqlite_int(loop_row["input_tokens"], "loop input tokens"),
        output_tokens=_sqlite_int(loop_row["output_tokens"], "loop output tokens"),
        estimated_cost_usd=_decode_decimal(loop_row["estimated_cost_usd"]),
        waiting_approval_id=_optional_text(loop_row["waiting_approval_id"]),
        interruption_reason=_optional_text(loop_row["interruption_reason"]),
        final_answer_candidate=_optional_text(loop_row["final_answer_candidate"]),
        no_progress_fingerprints=_decode_string_tuple(
            _sqlite_text(
                loop_row["no_progress_fingerprints_json"],
                "loop no-progress fingerprints",
            )
        ),
    )
    budget_cost = loop_row["budget_max_estimated_cost_usd"]
    budgets = LoopBudgets(
        max_turns=_sqlite_int(loop_row["budget_max_turns"], "budget max turns"),
        max_actions=_sqlite_int(loop_row["budget_max_actions"], "budget max actions"),
        max_repairs=_sqlite_int(loop_row["budget_max_repairs"], "budget max repairs"),
        max_identical_failures=_sqlite_int(
            loop_row["budget_max_identical_failures"],
            "budget max identical failures",
        ),
        max_observation_characters=_sqlite_int(
            loop_row["budget_max_observation_characters"],
            "budget max observation characters",
        ),
        max_total_tokens=_sqlite_int(
            loop_row["budget_max_total_tokens"], "budget max total tokens"
        ),
        max_wall_time_seconds=_sqlite_real(
            loop_row["budget_max_wall_time_seconds"],
            "budget max wall time",
        ),
        task_timeout_seconds=_sqlite_real(
            loop_row["budget_task_timeout_seconds"],
            "budget task timeout",
        ),
        max_estimated_cost_usd=(
            None if budget_cost is None else _decode_decimal(budget_cost)
        ),
    )

    turn_rows = _operation_rows(connection, "turns", operation_id)
    model_call_rows = _operation_rows(connection, "model_calls", operation_id)
    readiness_rows = _operation_rows(connection, "readiness", operation_id)
    task_rows = _operation_rows(connection, "tasks", operation_id)
    task_dependency_rows = _operation_rows(
        connection,
        "task_dependencies",
        operation_id,
    )
    task_lease_rows = _operation_rows(connection, "task_leases", operation_id)
    evidence_rows = _operation_rows(connection, "evidence", operation_id)
    observation_rows = _operation_rows(connection, "observations", operation_id)
    event_rows = _operation_rows(connection, "runtime_events", operation_id)

    turns = tuple(
        Turn(
            id=_sqlite_text(row["id"], "turn id"),
            operation_id=operation_id,
            number=_sqlite_int(row["number"], "turn number"),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "turn created_at")
            ),
            model_request_id=_optional_text(row["model_request_id"]),
            model_response_id=_optional_text(row["model_response_id"]),
        )
        for row in turn_rows
    )
    model_calls = tuple(
        ModelCall(
            id=_sqlite_text(row["id"], "model-call id"),
            operation_id=operation_id,
            turn_id=_sqlite_text(row["turn_id"], "model-call turn id"),
            provider_id=_sqlite_text(row["provider_id"], "model-call provider id"),
            request=_decode_model_request(
                _sqlite_text(row["request_json"], "model-call request")
            ),
            status=ModelCallStatus(_sqlite_text(row["status"], "model-call status")),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "model-call created_at")
            ),
            updated_at=_decode_datetime(
                _sqlite_text(row["updated_at"], "model-call updated_at")
            ),
            response=(
                None
                if row["response_json"] is None
                else _decode_model_response(
                    _sqlite_text(row["response_json"], "model-call response")
                )
            ),
            error_code=_optional_text(row["error_code"]),
            cancellation_requested=_decode_bool(row["cancellation_requested"]),
        )
        for row in model_call_rows
    )
    readiness = tuple(
        Readiness(
            allowed=_decode_bool(row["allowed"]),
            code=_sqlite_text(row["code"], "readiness code"),
            message=_sqlite_text(row["message"], "readiness message"),
            evaluated_at=_decode_datetime(
                _sqlite_text(row["evaluated_at"], "readiness evaluated_at")
            ),
            missing_facts=_decode_string_tuple(
                _sqlite_text(row["missing_facts_json"], "readiness missing facts")
            ),
        )
        for row in readiness_rows
    )
    tasks = tuple(
        Task(
            id=_sqlite_text(row["id"], "task id"),
            operation_id=operation_id,
            turn_id=_sqlite_text(row["turn_id"], "task turn id"),
            call_id=_sqlite_text(row["call_id"], "task call id"),
            capability_id=_sqlite_text(row["capability_id"], "task capability id"),
            executor_id=_sqlite_text(row["executor_id"], "task executor id"),
            status=TaskStatus(_sqlite_text(row["status"], "task status")),
            attempt=_sqlite_int(row["attempt"], "task attempt"),
            arguments=_decode_json_object(
                _sqlite_text(row["arguments_json"], "task arguments")
            ),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "task created_at")
            ),
            updated_at=_decode_datetime(
                _sqlite_text(row["updated_at"], "task updated_at")
            ),
            execution_facts=TaskExecutionFacts(
                capability_fingerprint=_sqlite_text(
                    row["capability_fingerprint"],
                    "task capability fingerprint",
                ),
                arguments_hash=_sqlite_text(
                    row["arguments_hash"],
                    "task arguments hash",
                ),
                access_mode=AccessMode(
                    _sqlite_text(row["access_mode"], "task access mode")
                ),
                risk=RiskLevel(_sqlite_text(row["risk"], "task risk")),
                side_effecting=_decode_bool(row["side_effecting"]),
                idempotent=_decode_bool(row["idempotent"]),
                replay_safe=_decode_bool(row["replay_safe"]),
                idempotency_key=_optional_text(row["idempotency_key"]),
            ),
            evidence_ids=_load_task_evidence_ids(
                connection,
                operation_id,
                _sqlite_text(row["id"], "task id"),
            ),
            error_code=_optional_text(row["error_code"]),
            cancellation_requested=_decode_bool(row["cancellation_requested"]),
            manual_recovery_reason=_optional_text(row["manual_recovery_reason"]),
        )
        for row in task_rows
    )
    task_dependencies = tuple(
        TaskDependency(
            operation_id=operation_id,
            task_id=_sqlite_text(row["task_id"], "task dependency task id"),
            prerequisite_task_id=_sqlite_text(
                row["prerequisite_task_id"],
                "task dependency prerequisite id",
            ),
        )
        for row in task_dependency_rows
    )
    task_leases = tuple(
        TaskLease(
            operation_id=operation_id,
            task_id=_sqlite_text(row["task_id"], "task lease task id"),
            attempt=_sqlite_int(row["attempt"], "task lease attempt"),
            fencing_token=_sqlite_int(
                row["fencing_token"],
                "task lease fencing token",
            ),
            holder_id=_sqlite_text(row["holder_id"], "task lease holder id"),
            acquired_at=_decode_datetime(
                _sqlite_text(row["acquired_at"], "task lease acquired_at")
            ),
            expires_at=_decode_datetime(
                _sqlite_text(row["expires_at"], "task lease expires_at")
            ),
            started_at=_decode_optional_datetime(
                row["started_at"],
                "task lease started_at",
            ),
            renewed_at=_decode_optional_datetime(
                row["renewed_at"],
                "task lease renewed_at",
            ),
            released_at=_decode_optional_datetime(
                row["released_at"],
                "task lease released_at",
            ),
            release_reason=_optional_text(row["release_reason"]),
        )
        for row in task_lease_rows
    )
    evidence = tuple(
        Evidence(
            id=_sqlite_text(row["id"], "evidence id"),
            operation_id=operation_id,
            task_id=_sqlite_text(row["task_id"], "evidence task id"),
            turn_id=_sqlite_text(row["turn_id"], "evidence turn id"),
            capability_id=_sqlite_text(row["capability_id"], "evidence capability id"),
            executor_id=_sqlite_text(row["executor_id"], "evidence executor id"),
            kind=_sqlite_text(row["kind"], "evidence kind"),
            schema_version=_sqlite_int(
                row["schema_version"], "evidence schema version"
            ),
            attempt=_sqlite_int(row["attempt"], "evidence attempt"),
            accepted=_decode_bool(row["accepted"]),
            payload=_decode_json_object(
                _sqlite_text(row["payload_json"], "evidence payload")
            ),
            content_hash=_sqlite_text(row["content_hash"], "evidence content hash"),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "evidence created_at")
            ),
            blob_id=_optional_text(row["blob_id"]),
        )
        for row in evidence_rows
    )
    observations = tuple(
        Observation(
            operation_id=operation_id,
            turn_id=_sqlite_text(row["turn_id"], "observation turn id"),
            code=_sqlite_text(row["code"], "observation code"),
            message=_sqlite_text(row["message"], "observation message"),
            payload=_decode_json_object(
                _sqlite_text(row["payload_json"], "observation payload")
            ),
            success=_decode_bool(row["success"]),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "observation created_at")
            ),
            call_id=_optional_text(row["call_id"]),
            task_id=_optional_text(row["task_id"]),
            evidence_id=_optional_text(row["evidence_id"]),
            truncated=_decode_bool(row["truncated"]),
        )
        for row in observation_rows
    )
    events = tuple(_decode_runtime_event_row(row) for row in event_rows)
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=loop_state,
        budgets=budgets,
        turns=turns,
        model_calls=model_calls,
        readiness=readiness,
        tasks=tasks,
        task_dependencies=task_dependencies,
        task_leases=task_leases,
        evidence=evidence,
        observations=observations,
        events=events,
    )


def _operation_rows(
    connection: sqlite3.Connection,
    table: str,
    operation_id: str,
) -> list[sqlite3.Row]:
    if table not in {
        "turns",
        "model_calls",
        "readiness",
        "tasks",
        "task_dependencies",
        "task_leases",
        "evidence",
        "observations",
        "runtime_events",
    }:
        raise ValueError(f"unsupported lifecycle table: {table}")
    rows = connection.execute(
        f"SELECT * FROM {table} WHERE operation_id = ? ORDER BY position",
        (operation_id,),
    ).fetchall()
    _validate_contiguous_positions(rows, label=table)
    return rows


def _load_task_evidence_ids(
    connection: sqlite3.Connection,
    operation_id: str,
    task_id: str,
) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT position, evidence_id FROM task_evidence "
        "WHERE operation_id = ? AND task_id = ? ORDER BY position",
        (operation_id, task_id),
    ).fetchall()
    _validate_contiguous_positions(rows, label=f"task evidence for {task_id}")
    return tuple(_sqlite_text(row[1], "task evidence id") for row in rows)


def _validate_contiguous_positions(
    rows: Sequence[sqlite3.Row],
    *,
    label: str,
) -> None:
    for expected, row in enumerate(rows):
        actual = _sqlite_int(row["position"], f"{label} position")
        if actual != expected:
            raise SQLiteCorruptionError(
                f"{label} positions must be contiguous from zero; "
                f"expected {expected}, found {actual}"
            )


def _require_identity(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_committed_event_scope(
    agent_id: str,
    cursor: EventCursor | None,
) -> None:
    _require_identity(agent_id, "event agent_id")
    if cursor is not None and not isinstance(cursor, EventCursor):
        raise TypeError("committed event cursor must be an EventCursor or None")
    if cursor is not None and cursor.agent_id != agent_id:
        raise EventCursorMismatchError(
            requested_agent_id=agent_id,
            cursor=cursor,
        )


def _sqlite_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"persisted {label} must use SQLite TEXT storage")
    return value


def _sqlite_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"persisted {label} must use SQLite INTEGER storage")
    return value


def _sqlite_real(value: object, label: str) -> float:
    if not isinstance(value, float):
        raise TypeError(f"persisted {label} must use SQLite REAL storage")
    return value


def _encode_datetime(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError("persisted datetime must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError("persisted datetime must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _encode_optional_datetime(value: datetime | None) -> str | None:
    return None if value is None else _encode_datetime(value)


def _decode_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("persisted datetime must be text")
    decoded = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if decoded.tzinfo is None or decoded.utcoffset() is None:
        raise ValueError("persisted datetime must be timezone-aware")
    if _encode_datetime(decoded) != value:
        raise ValueError("persisted datetime is not in canonical UTC form")
    return decoded


def _decode_optional_datetime(value: object, label: str) -> datetime | None:
    if value is None:
        return None
    return _decode_datetime(_sqlite_text(value, label))


def _encode_decimal(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("persisted Decimal must be finite")
    return str(value)


def _decode_decimal(value: object) -> Decimal:
    if not isinstance(value, str):
        raise TypeError("persisted Decimal must be text")
    decoded = Decimal(value)
    if not decoded.is_finite():
        raise ValueError("persisted Decimal must be finite")
    if _encode_decimal(decoded) != value:
        raise ValueError("persisted Decimal is not in canonical text form")
    return decoded


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("persisted optional text has the wrong type")
    return value


def _decode_bool(value: object) -> bool:
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
        raise ValueError("persisted boolean must be SQLite integer 0 or 1")
    return bool(value)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_json(value: str) -> object:
    if not isinstance(value, str):
        raise TypeError("persisted JSON must be text")
    decoded = json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if canonical_json(decoded) != value:
        raise ValueError("persisted JSON is not in canonical form")
    return decoded


def _decode_json_object(value: str) -> Mapping[str, object]:
    decoded = _decode_json(value)
    if not isinstance(decoded, dict):
        raise ValueError("persisted JSON value must be an object")
    return decoded


def _decode_string_tuple(value: str) -> tuple[str, ...]:
    decoded = _decode_json(value)
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise ValueError("persisted JSON value must be an array of strings")
    return tuple(decoded)


def _expect_object(
    value: object,
    *,
    keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(value) != keys:
        raise ValueError(f"{label} has unknown or missing fields")
    return value


def _expect_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _expect_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be JSON text")
    return value


def _expect_optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _expect_text(value, label)


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON integer")
    return value


def _tool_call_to_data(call: ToolCall) -> dict[str, object]:
    return {
        "arguments": call.arguments,
        "id": call.id,
        "name": call.name,
    }


def _tool_call_from_data(value: object) -> ToolCall:
    data = _expect_object(
        value,
        keys={"arguments", "id", "name"},
        label="tool call",
    )
    arguments = data["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("tool-call arguments must be a JSON object")
    return ToolCall(
        id=_expect_text(data["id"], "tool-call id"),
        name=_expect_text(data["name"], "tool-call name"),
        arguments=arguments,
    )


def _tool_definition_to_data(definition: ToolDefinition) -> dict[str, object]:
    return {
        "description": definition.description,
        "input_schema": definition.input_schema,
        "name": definition.name,
    }


def _tool_definition_from_data(value: object) -> ToolDefinition:
    data = _expect_object(
        value,
        keys={"description", "input_schema", "name"},
        label="tool definition",
    )
    input_schema = data["input_schema"]
    if not isinstance(input_schema, dict):
        raise ValueError("tool input schema must be a JSON object")
    return ToolDefinition(
        name=_expect_text(data["name"], "tool name"),
        description=_expect_text(data["description"], "tool description"),
        input_schema=input_schema,
    )


def _content_block_to_data(
    block: TextBlock | ToolResultBlock,
) -> dict[str, object]:
    if isinstance(block, TextBlock):
        return {"kind": "text", "text": block.text}
    if isinstance(block, ToolResultBlock):
        return {
            "call_id": block.call_id,
            "is_error": block.is_error,
            "kind": "tool_result",
            "output": block.output,
        }
    raise TypeError(f"unsupported canonical content block: {type(block).__name__}")


def _content_block_from_data(value: object) -> TextBlock | ToolResultBlock:
    if not isinstance(value, dict):
        raise ValueError("content block must be a JSON object")
    kind = value.get("kind")
    if kind == "text":
        data = _expect_object(
            value,
            keys={"kind", "text"},
            label="text content block",
        )
        return TextBlock(_expect_text(data["text"], "content text"))
    if kind == "tool_result":
        data = _expect_object(
            value,
            keys={"call_id", "is_error", "kind", "output"},
            label="tool-result content block",
        )
        output = data["output"]
        if not isinstance(output, dict):
            raise ValueError("tool-result output must be a JSON object")
        return ToolResultBlock(
            call_id=_expect_text(data["call_id"], "tool-result call id"),
            output=output,
            is_error=_expect_bool(data["is_error"], "tool-result is_error"),
        )
    raise ValueError(f"unknown canonical content-block kind: {kind!r}")


def _message_to_data(message: CanonicalMessage) -> dict[str, object]:
    return {
        "agent_id": message.agent_id,
        "content": [_content_block_to_data(block) for block in message.content],
        "operation_id": message.operation_id,
        "role": message.role.value,
        "session_id": message.session_id,
        "tool_calls": [_tool_call_to_data(call) for call in message.tool_calls],
        "turn_id": message.turn_id,
    }


def _message_from_data(value: object) -> CanonicalMessage:
    data = _expect_object(
        value,
        keys={
            "agent_id",
            "content",
            "operation_id",
            "role",
            "session_id",
            "tool_calls",
            "turn_id",
        },
        label="canonical message",
    )
    return CanonicalMessage(
        agent_id=_expect_text(data["agent_id"], "message agent_id"),
        operation_id=_expect_text(data["operation_id"], "message operation_id"),
        role=MessageRole(_expect_text(data["role"], "message role")),
        content=tuple(
            _content_block_from_data(item)
            for item in _expect_list(data["content"], "message content")
        ),
        turn_id=_expect_optional_text(data["turn_id"], "message turn_id"),
        session_id=_expect_optional_text(data["session_id"], "message session_id"),
        tool_calls=tuple(
            _tool_call_from_data(item)
            for item in _expect_list(data["tool_calls"], "message tool calls")
        ),
    )


def _encode_model_request(request: ModelRequest) -> str:
    return canonical_json(
        {
            "codec_version": 1,
            "messages": [_message_to_data(message) for message in request.messages],
            "operation_id": request.operation_id,
            "tools": [_tool_definition_to_data(tool) for tool in request.tools],
            "turn_id": request.turn_id,
        }
    )


def _decode_model_request(value: str) -> ModelRequest:
    data = _expect_object(
        _decode_json(value),
        keys={"codec_version", "messages", "operation_id", "tools", "turn_id"},
        label="model request",
    )
    if _expect_int(data["codec_version"], "model-request codec version") != 1:
        raise ValueError("unknown model-request codec version")
    return ModelRequest(
        operation_id=_expect_text(data["operation_id"], "request operation_id"),
        turn_id=_expect_text(data["turn_id"], "request turn_id"),
        messages=tuple(
            _message_from_data(item)
            for item in _expect_list(data["messages"], "request messages")
        ),
        tools=tuple(
            _tool_definition_from_data(item)
            for item in _expect_list(data["tools"], "request tools")
        ),
    )


def _usage_to_data(usage: ModelUsage) -> dict[str, object]:
    return {
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "estimated_cost_usd": _encode_decimal(usage.estimated_cost_usd),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _usage_from_data(value: object) -> ModelUsage:
    data = _expect_object(
        value,
        keys={
            "cache_read_tokens",
            "cache_write_tokens",
            "estimated_cost_usd",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
        },
        label="model usage",
    )
    return ModelUsage(
        input_tokens=_expect_int(data["input_tokens"], "usage input tokens"),
        output_tokens=_expect_int(data["output_tokens"], "usage output tokens"),
        reasoning_tokens=_expect_int(
            data["reasoning_tokens"], "usage reasoning tokens"
        ),
        cache_read_tokens=_expect_int(
            data["cache_read_tokens"], "usage cache-read tokens"
        ),
        cache_write_tokens=_expect_int(
            data["cache_write_tokens"], "usage cache-write tokens"
        ),
        estimated_cost_usd=_decode_decimal(
            _expect_text(data["estimated_cost_usd"], "usage estimated cost")
        ),
    )


def _encode_model_response(response: ModelResponse) -> str:
    return canonical_json(
        {
            "codec_version": 1,
            "finish_reason": response.finish_reason.value,
            "provider_metadata": response.provider_metadata,
            "provider_response_id": response.provider_response_id,
            "text": response.text,
            "tool_calls": [_tool_call_to_data(call) for call in response.tool_calls],
            "usage": _usage_to_data(response.usage),
        }
    )


def _decode_model_response(value: str) -> ModelResponse:
    data = _expect_object(
        _decode_json(value),
        keys={
            "codec_version",
            "finish_reason",
            "provider_metadata",
            "provider_response_id",
            "text",
            "tool_calls",
            "usage",
        },
        label="model response",
    )
    if _expect_int(data["codec_version"], "model-response codec version") != 1:
        raise ValueError("unknown model-response codec version")
    provider_metadata = data["provider_metadata"]
    if not isinstance(provider_metadata, dict):
        raise ValueError("provider metadata must be a JSON object")
    return ModelResponse(
        finish_reason=FinishReason(
            _expect_text(data["finish_reason"], "response finish reason")
        ),
        text=_expect_optional_text(data["text"], "response text"),
        tool_calls=tuple(
            _tool_call_from_data(item)
            for item in _expect_list(data["tool_calls"], "response tool calls")
        ),
        usage=_usage_from_data(data["usage"]),
        provider_response_id=_expect_optional_text(
            data["provider_response_id"],
            "provider response id",
        ),
        provider_metadata=provider_metadata,
    )
