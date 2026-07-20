"""Standard-library SQLite foundation for the durable operation store.

The generic loop and operation runtime depend only on ``OperationStore``.
This module owns SQLite connection policy, file identity, migration history,
and backup-before-migrate behavior for the concrete adapter.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import TypeVar

from .._json import canonical_json
from ..adapters.models import SourceRegistration
from ..capabilities import AccessMode, RiskLevel
from ..catalog.models import (
    CatalogFacet,
    CatalogPath,
    CatalogPathStep,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSearchHit,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    CatalogTraversalResult,
    FacetKind,
    RelationshipDirection,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
)
from ..catalog.protocols import (
    CatalogResourceNotFoundError,
    CatalogSyncConflictError,
)
from ..context.session import (
    SessionApprovalStateFact,
    SessionOperationFacts,
    SessionResourceScopeFact,
)
from ..config import (
    AgentRuntimeDefaults,
    AgentRuntimeDefaultsConflictError,
)
from ..events.models import CommittedEvent, EventCursor, RuntimeEvent
from ..events.protocols import (
    EventCursorMismatchError,
    EventCursorNotFoundError,
)
from ..identity import AgentIdentity, AgentIdentityConflictError
from ..hosting.inbox import (
    HostInboxEnqueueConflictError,
    HostInboxItem,
    HostInboxKind,
    HostInboxNotFoundError,
    HostInboxRevisionConflict,
    HostInboxStatus,
    HostMutationAdmission,
    HostMutationConflictError,
)
from ..learning import (
    LearningCandidateCategory,
    LearningDecision,
    LearningProposal,
    LearningProposalKind,
    LearningProposalState,
    LearningProvenance,
    LearningRejectionCategory,
    LearningSourceOutcome,
    LearningStoreConflictError,
    LearningTransitionError,
    resolve_learning_proposal,
)
from ..llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelRouteAttempt,
    ModelRouteAttemptOutcome,
    ModelRoutingTrace,
    ModelSensitivity,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from ..llm.protocols import ModelProfileConflictError
from ..loop.models import LoopBudgets, LoopPhase, LoopState, Readiness, Turn
from ..memory.models import (
    MemoryCreator,
    MemoryHistory,
    MemoryKind,
    MemoryProvenance,
    MemoryProvenanceKind,
    MemoryRecord,
    MemoryRestoreRequest,
    MemoryScope,
    MemorySensitivity,
    MemorySnapshot,
    MemoryState,
    MemorySupersessionRequest,
    MemoryVersion,
)
from ..memory.learning import (
    ExplicitCorrectionCommit,
    ExplicitCorrectionResult,
    ExplicitCorrectionStoreConflictError,
)
from ..memory.protocols import MemoryStoreConflictError
from ..monitors.models import (
    CatchUpPolicy,
    CronSchedule,
    IntervalSchedule,
    Monitor,
    MonitorBudgetOverrides,
    MonitorCheckpoint,
    MonitorCondition,
    MonitorConditionKind,
    MonitorConfirmation,
    MonitorConfirmationDecision,
    MonitorDefinition,
    MonitorFinding,
    MonitorFindingSeverity,
    MonitorInspection,
    MonitorLifecycleAction,
    MonitorLifecycleRecord,
    MonitorOccurrence,
    MonitorOccurrenceKind,
    MonitorProposal,
    MonitorRun,
    MonitorRunStatus,
    MonitorScheduleState,
    MonitorScope,
    MonitorStatus,
    MonitorTickLease,
    MonitorTimingPolicy,
    MonitorVersion,
)
from ..monitors.store import (
    ExpiredMonitorLeaseError,
    MonitorClaimResult,
    MonitorConfirmationCommit,
    MonitorConflictError,
    MonitorLifecycleCommit,
    MonitorNotFoundError,
    MonitorOccurrenceClaim,
    MonitorOutcomeCommit,
    MonitorOutcomeResult,
    MonitorProposalConflictError,
    MonitorProposalNotFoundError,
    StaleMonitorFenceError,
    MonitorTickClaimConflictError,
)
from ..operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from ..operations.governance import (
    ApprovalRequest,
    ApprovalStatus,
    DefaultPolicyProfile,
)
from ..operations.leases import TaskClaimRequest, TaskLease, TaskLeaseGuard
from ..operations.models import (
    ActionValidationFacts,
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
    InvalidOperationCheckpointError,
    OperationAlreadyExistsError,
    OperationNotFoundError,
    OperationRevisionConflict,
    TaskClaimResult,
    TriggerAlreadyClaimedError,
    VersionedOperation,
    _NONTERMINAL_OPERATION_STATUSES,
    _TERMINAL_OPERATION_STATUSES,
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
from ..sessions import (
    Session,
    SessionAlreadyExistsError,
    SessionCompressionCheckpoint,
    SessionTranscript,
)
from ..skills.models import (
    Skill,
    SkillActivation,
    SkillActivationMode,
    SkillIndex,
    SkillInspection,
    SkillSource,
    SkillVersion,
)
from ..skills.learning import (
    SkillChangeAcceptanceResult,
    SkillChangeCommit,
    SkillChangeConflictError,
)
from ..skills.service import (
    SkillActivationConflictError,
    SkillDiscoveryError,
)

DAITA_V2_APPLICATION_ID = 0x44414932  # ASCII ``DAI2``.
_MAX_COMMITTED_EVENT_READ_LIMIT = 1_000
_COMMITTED_EVENT_SUBSCRIPTION_BATCH_SIZE = 100
_COMMITTED_EVENT_POLL_INTERVAL_SECONDS = 0.25
_CATALOG_SEARCH_MAX_TERMS = 32
_CATALOG_SEARCH_TERM = re.compile(r"[A-Za-z0-9]+")


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


_APPROVAL_SCHEMA_SQL = (
    """
    CREATE TABLE approvals (
        operation_id TEXT NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        task_fingerprint TEXT NOT NULL CHECK (
            length(task_fingerprint) = 71
            AND substr(task_fingerprint, 1, 7) = 'sha256:'
            AND substr(task_fingerprint, 8) NOT GLOB '*[^0-9a-f]*'
        ),
        policy_fingerprint TEXT NOT NULL CHECK (
            length(policy_fingerprint) = 71
            AND substr(policy_fingerprint, 1, 7) = 'sha256:'
            AND substr(policy_fingerprint, 8) NOT GLOB '*[^0-9a-f]*'
        ),
        requested_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('pending', 'approved', 'denied', 'cancelled')
        ),
        decided_at TEXT,
        decided_by TEXT,
        decision_reason TEXT,
        UNIQUE (operation_id, position),
        UNIQUE (operation_id, id),
        UNIQUE (operation_id, task_id),
        CHECK (
            (
                status = 'pending'
                AND decided_at IS NULL
                AND decided_by IS NULL
                AND decision_reason IS NULL
            )
            OR (
                status != 'pending'
                AND decided_at IS NOT NULL
                AND decided_at >= requested_at
                AND decided_by IS NOT NULL
                AND length(trim(decided_by)) > 0
                AND decision_reason IS NOT NULL
                AND length(trim(decision_reason)) > 0
            )
        ),
        FOREIGN KEY (operation_id, task_id)
            REFERENCES tasks(operation_id, id) ON DELETE CASCADE
    )
    """.strip(),
    "ALTER TABLE runtime_events ADD COLUMN approval_id TEXT",
    """
    INSERT INTO runtime_events(
        id, operation_id, position, type, agent_id, agent_sequence, created_at,
        session_id, turn_id, model_call_id, call_id, task_id, evidence_id,
        capability_id, executor_id, payload_json, approval_id
    )
    WITH legacy_waiting AS (
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
        WHERE task.status = 'waiting_for_approval'
    )
    SELECT
        'daita:v2:migration:6:manual-recovery:'
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
        '{"from_status":"waiting_for_approval","reason":"legacy_waiting_task_missing_approval","to_status":"manual_recovery_required"}',
        NULL
    FROM legacy_waiting
    ORDER BY agent_id, agent_event_offset
    """.strip(),
    """
    UPDATE operations
    SET
        revision = revision + 1,
        status = CASE
            WHEN status = 'waiting_for_approval' THEN 'running'
            ELSE status
        END
    WHERE EXISTS (
        SELECT 1
        FROM tasks AS task
        WHERE task.operation_id = operations.id
            AND task.status = 'waiting_for_approval'
    )
    """.strip(),
    """
    UPDATE loop_state
    SET
        phase = 'awaiting_execution',
        waiting_approval_id = NULL
    WHERE EXISTS (
        SELECT 1
        FROM tasks AS task
        WHERE task.operation_id = loop_state.operation_id
            AND task.status = 'waiting_for_approval'
    )
    """.strip(),
    """
    UPDATE tasks
    SET
        status = 'manual_recovery_required',
        manual_recovery_reason = 'legacy_waiting_task_missing_approval'
    WHERE status = 'waiting_for_approval'
    """.strip(),
    """
    CREATE INDEX approvals_operation_status_idx
        ON approvals(operation_id, status, requested_at)
    """.strip(),
    """
    CREATE TRIGGER approvals_reject_identity_update
    BEFORE UPDATE ON approvals
    WHEN NEW.operation_id != OLD.operation_id
        OR NEW.position != OLD.position
        OR NEW.id != OLD.id
        OR NEW.task_id != OLD.task_id
        OR NEW.task_fingerprint != OLD.task_fingerprint
        OR NEW.policy_fingerprint != OLD.policy_fingerprint
        OR NEW.requested_at != OLD.requested_at
    BEGIN
        SELECT RAISE(ABORT, 'approval identity is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER approvals_reject_terminal_update
    BEFORE UPDATE ON approvals
    WHEN OLD.status != 'pending'
    BEGIN
        SELECT RAISE(ABORT, 'terminal approval is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER approvals_reject_delete
    BEFORE DELETE ON approvals
    BEGIN
        SELECT RAISE(ABORT, 'approval history is append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER runtime_events_validate_approval_insert
    BEFORE INSERT ON runtime_events
    WHEN NEW.approval_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1
            FROM approvals AS approval
            WHERE approval.id = NEW.approval_id
                AND approval.operation_id = NEW.operation_id
                AND approval.task_id = NEW.task_id
        )
    BEGIN
        SELECT RAISE(ABORT, 'event approval correlation is invalid');
    END
    """.strip(),
)

_AGENT_SESSION_SCHEMA_SQL = (
    """
    CREATE TABLE agents (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        id TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        state_schema_generation INTEGER NOT NULL CHECK (
            state_schema_generation = 2
        )
    )
    """.strip(),
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (agent_id) REFERENCES agents(id)
    )
    """.strip(),
    """
    CREATE TABLE session_operations (
        session_id TEXT NOT NULL REFERENCES sessions(id),
        position INTEGER NOT NULL CHECK (position >= 0),
        operation_id TEXT NOT NULL UNIQUE REFERENCES operations(id),
        PRIMARY KEY (session_id, position)
    )
    """.strip(),
    """
    CREATE INDEX operations_agent_session_created_idx
        ON operations(agent_id, session_id, created_at, id)
    """.strip(),
)

_CATALOG_SCHEMA_SQL = (
    """
    CREATE TABLE attached_sources (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        adapter_id TEXT NOT NULL,
        native_identity TEXT NOT NULL,
        display_name TEXT NOT NULL,
        configuration_json TEXT NOT NULL,
        attached_at TEXT NOT NULL,
        detached_at TEXT,
        UNIQUE (agent_id, adapter_id, native_identity)
    )
    """.strip(),
    """
    CREATE INDEX attached_sources_agent_active_idx
        ON attached_sources(agent_id, detached_at, display_name, id)
    """.strip(),
    """
    CREATE TABLE catalog_syncs (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        source_id TEXT NOT NULL,
        adapter_id TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        source_revision TEXT,
        resource_count INTEGER NOT NULL CHECK (resource_count >= 0),
        relationship_count INTEGER NOT NULL CHECK (relationship_count >= 0),
        error_code TEXT
    )
    """.strip(),
    """
    CREATE INDEX catalog_syncs_agent_source_started_idx
        ON catalog_syncs(agent_id, source_id, started_at, id)
    """.strip(),
    """
    CREATE TABLE catalog_resource_revisions (
        agent_id TEXT NOT NULL REFERENCES agents(id),
        resource_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        sync_id TEXT NOT NULL REFERENCES catalog_syncs(id),
        observed_at TEXT NOT NULL,
        facet_revisions_json TEXT NOT NULL,
        relationship_revisions_json TEXT NOT NULL,
        source_revision TEXT,
        PRIMARY KEY (agent_id, resource_id, revision, sync_id)
    )
    """.strip(),
    """
    CREATE INDEX catalog_resource_revisions_lookup_idx
        ON catalog_resource_revisions(
            agent_id, resource_id, revision, observed_at, sync_id
        )
    """.strip(),
    """
    CREATE TABLE catalog_facets (
        agent_id TEXT NOT NULL REFERENCES agents(id),
        resource_id TEXT NOT NULL,
        sync_id TEXT NOT NULL REFERENCES catalog_syncs(id),
        kind TEXT NOT NULL,
        schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
        revision TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        PRIMARY KEY (agent_id, resource_id, revision, sync_id, kind)
    )
    """.strip(),
    """
    CREATE TABLE catalog_relationships (
        agent_id TEXT NOT NULL REFERENCES agents(id),
        id TEXT NOT NULL,
        revision TEXT NOT NULL,
        source_id TEXT NOT NULL,
        from_resource_id TEXT NOT NULL,
        to_resource_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        provenance TEXT NOT NULL,
        confidence REAL NOT NULL,
        sync_id TEXT NOT NULL REFERENCES catalog_syncs(id),
        observed_at TEXT NOT NULL,
        field_pairs_json TEXT NOT NULL,
        attributes_json TEXT NOT NULL,
        PRIMARY KEY (agent_id, id, sync_id)
    )
    """.strip(),
    """
    CREATE INDEX catalog_relationships_traversal_idx
        ON catalog_relationships(
            agent_id, sync_id, from_resource_id, to_resource_id, kind, id
        )
    """.strip(),
    """
    CREATE TABLE catalog_resources (
        agent_id TEXT NOT NULL REFERENCES agents(id),
        id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        native_identity TEXT NOT NULL,
        external_uri TEXT NOT NULL,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        sensitivity TEXT NOT NULL,
        current_revision TEXT NOT NULL,
        current_sync_id TEXT NOT NULL REFERENCES catalog_syncs(id),
        first_observed_at TEXT NOT NULL,
        last_observed_at TEXT NOT NULL,
        PRIMARY KEY (agent_id, id),
        UNIQUE (agent_id, source_id, kind, native_identity)
    )
    """.strip(),
    """
    CREATE INDEX catalog_resources_source_name_idx
        ON catalog_resources(agent_id, source_id, kind, name, id)
    """.strip(),
    """
    CREATE VIRTUAL TABLE catalog_resource_search USING fts5(
        agent_id UNINDEXED,
        resource_id UNINDEXED,
        source_id UNINDEXED,
        kind UNINDEXED,
        name,
        native_identity,
        external_uri,
        structural_text,
        tokenize = 'unicode61'
    )
    """.strip(),
)

_CONTEXT_MEMORY_LEARNING_SKILL_SCHEMA_SQL = (
    """
    CREATE TABLE agent_model_profiles (
        agent_id TEXT PRIMARY KEY REFERENCES agents(id),
        profile_id TEXT NOT NULL,
        context_window_tokens INTEGER NOT NULL CHECK (context_window_tokens > 0),
        max_output_tokens INTEGER NOT NULL CHECK (
            max_output_tokens > 0 AND max_output_tokens < context_window_tokens
        ),
        supports_tools INTEGER NOT NULL CHECK (supports_tools IN (0, 1)),
        supports_parallel_tools INTEGER NOT NULL CHECK (
            supports_parallel_tools IN (0, 1)
        ),
        supports_structured_output INTEGER NOT NULL CHECK (
            supports_structured_output IN (0, 1)
        ),
        supports_streaming INTEGER NOT NULL CHECK (supports_streaming IN (0, 1)),
        supports_reasoning INTEGER NOT NULL CHECK (supports_reasoning IN (0, 1)),
        supports_vision INTEGER NOT NULL CHECK (supports_vision IN (0, 1)),
        supports_documents INTEGER NOT NULL CHECK (supports_documents IN (0, 1)),
        supports_prompt_caching INTEGER NOT NULL CHECK (
            supports_prompt_caching IN (0, 1)
        ),
        supports_native_continuation INTEGER NOT NULL CHECK (
            supports_native_continuation IN (0, 1)
        ),
        input_cost_per_million_usd TEXT,
        output_cost_per_million_usd TEXT,
        data_routing_classification TEXT NOT NULL,
        available INTEGER NOT NULL CHECK (available IN (0, 1)),
        healthy INTEGER NOT NULL CHECK (healthy IN (0, 1)),
        CHECK (supports_parallel_tools = 0 OR supports_tools = 1)
    )
    """.strip(),
    """
    CREATE TRIGGER agent_model_profiles_reject_update
    BEFORE UPDATE ON agent_model_profiles
    BEGIN
        SELECT RAISE(ABORT, 'agent model-profile binding is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER agent_model_profiles_reject_delete
    BEFORE DELETE ON agent_model_profiles
    BEGIN
        SELECT RAISE(ABORT, 'agent model-profile binding is immutable');
    END
    """.strip(),
    """
    CREATE TABLE session_compression_checkpoints (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        session_id TEXT NOT NULL REFERENCES sessions(id),
        version INTEGER NOT NULL CHECK (version >= 1),
        through_position INTEGER NOT NULL CHECK (through_position >= 0),
        through_operation_id TEXT NOT NULL REFERENCES operations(id),
        source_fingerprint TEXT NOT NULL,
        summary TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (agent_id, session_id, version)
    )
    """.strip(),
    """
    CREATE INDEX session_compression_current_idx
        ON session_compression_checkpoints(agent_id, session_id, version DESC)
    """.strip(),
    """
    CREATE TABLE session_compression_operations (
        checkpoint_id TEXT NOT NULL REFERENCES session_compression_checkpoints(id),
        position INTEGER NOT NULL CHECK (position >= 0),
        operation_id TEXT NOT NULL REFERENCES operations(id),
        PRIMARY KEY (checkpoint_id, position),
        UNIQUE (checkpoint_id, operation_id)
    )
    """.strip(),
    """
    CREATE TABLE session_compression_evidence (
        checkpoint_id TEXT NOT NULL REFERENCES session_compression_checkpoints(id),
        position INTEGER NOT NULL CHECK (position >= 0),
        evidence_id TEXT NOT NULL REFERENCES evidence(id),
        PRIMARY KEY (checkpoint_id, position),
        UNIQUE (checkpoint_id, evidence_id)
    )
    """.strip(),
    """
    CREATE TABLE session_compression_approvals (
        checkpoint_id TEXT NOT NULL REFERENCES session_compression_checkpoints(id),
        position INTEGER NOT NULL CHECK (position >= 0),
        approval_id TEXT NOT NULL REFERENCES approvals(id),
        PRIMARY KEY (checkpoint_id, position),
        UNIQUE (checkpoint_id, approval_id)
    )
    """.strip(),
    """
    CREATE TABLE session_compression_resources (
        checkpoint_id TEXT NOT NULL REFERENCES session_compression_checkpoints(id),
        position INTEGER NOT NULL CHECK (position >= 0),
        resource_id TEXT NOT NULL,
        PRIMARY KEY (checkpoint_id, position),
        UNIQUE (checkpoint_id, resource_id)
    )
    """.strip(),
    """
    CREATE TRIGGER session_compression_checkpoints_reject_update
    BEFORE UPDATE ON session_compression_checkpoints
    BEGIN
        SELECT RAISE(ABORT, 'session compression checkpoints are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_checkpoints_reject_delete
    BEFORE DELETE ON session_compression_checkpoints
    BEGIN
        SELECT RAISE(ABORT, 'session compression checkpoints are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_operations_reject_update
    BEFORE UPDATE ON session_compression_operations
    BEGIN
        SELECT RAISE(ABORT, 'session compression operation links are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_operations_reject_delete
    BEFORE DELETE ON session_compression_operations
    BEGIN
        SELECT RAISE(ABORT, 'session compression operation links are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_evidence_reject_update
    BEFORE UPDATE ON session_compression_evidence
    BEGIN
        SELECT RAISE(ABORT, 'session compression evidence links are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_evidence_reject_delete
    BEFORE DELETE ON session_compression_evidence
    BEGIN
        SELECT RAISE(ABORT, 'session compression evidence links are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_approvals_reject_update
    BEFORE UPDATE ON session_compression_approvals
    BEGIN
        SELECT RAISE(ABORT, 'session compression approval links are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_approvals_reject_delete
    BEFORE DELETE ON session_compression_approvals
    BEGIN
        SELECT RAISE(ABORT, 'session compression approval links are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_resources_reject_update
    BEFORE UPDATE ON session_compression_resources
    BEGIN
        SELECT RAISE(ABORT, 'session compression resource links are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER session_compression_resources_reject_delete
    BEFORE DELETE ON session_compression_resources
    BEGIN
        SELECT RAISE(ABORT, 'session compression resource links are append-only');
    END
    """.strip(),
    """
    CREATE TABLE memory_records (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        user_id TEXT,
        session_id TEXT REFERENCES sessions(id),
        source_id TEXT,
        resource_id TEXT,
        scope_fingerprint TEXT NOT NULL,
        kind TEXT NOT NULL,
        logical_key TEXT NOT NULL,
        current_version INTEGER NOT NULL CHECK (current_version >= 1),
        state TEXT NOT NULL CHECK (state IN ('active', 'superseded', 'rejected')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        superseded_by_id TEXT REFERENCES memory_records(id),
        UNIQUE (agent_id, scope_fingerprint, kind, logical_key),
        CHECK (resource_id IS NULL OR source_id IS NOT NULL),
        CHECK (
            (state = 'superseded' AND superseded_by_id IS NOT NULL)
            OR (state != 'superseded' AND superseded_by_id IS NULL)
        )
    )
    """.strip(),
    """
    CREATE INDEX memory_records_scope_idx
        ON memory_records(
            agent_id, state, user_id, session_id, source_id, resource_id,
            updated_at, id
        )
    """.strip(),
    """
    CREATE TABLE memory_versions (
        memory_id TEXT NOT NULL REFERENCES memory_records(id),
        version INTEGER NOT NULL CHECK (version >= 1),
        content TEXT NOT NULL,
        creator TEXT NOT NULL,
        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        sensitivity TEXT NOT NULL CHECK (
            sensitivity IN ('public', 'internal', 'confidential', 'restricted')
        ),
        provenance_kind TEXT NOT NULL,
        provenance_content_hash TEXT NOT NULL,
        provenance_operation_id TEXT REFERENCES operations(id),
        provenance_trigger_id TEXT REFERENCES triggers(id),
        provenance_evidence_id TEXT REFERENCES evidence(id),
        provenance_session_id TEXT REFERENCES sessions(id),
        provenance_external_ref TEXT,
        attributes_json TEXT NOT NULL,
        expires_at TEXT,
        resource_revision TEXT,
        supersedes_version INTEGER,
        created_at TEXT NOT NULL,
        PRIMARY KEY (memory_id, version),
        CHECK (
            (version = 1 AND supersedes_version IS NULL)
            OR (version > 1 AND supersedes_version IS NOT NULL
                AND supersedes_version < version)
        )
    )
    """.strip(),
    """
    CREATE TRIGGER memory_versions_reject_update
    BEFORE UPDATE ON memory_versions
    BEGIN
        SELECT RAISE(ABORT, 'memory versions are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER memory_versions_reject_delete
    BEFORE DELETE ON memory_versions
    BEGIN
        SELECT RAISE(ABORT, 'memory versions are append-only');
    END
    """.strip(),
    """
    CREATE VIRTUAL TABLE memory_search USING fts5(
        memory_id UNINDEXED,
        logical_key,
        content,
        attributes,
        tokenize = 'unicode61'
    )
    """.strip(),
    """
    CREATE TABLE learning_proposals (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        kind TEXT NOT NULL CHECK (kind IN ('memory', 'skill')),
        category TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('proposed', 'committed', 'rejected')),
        operation_id TEXT NOT NULL REFERENCES operations(id),
        trigger_id TEXT NOT NULL REFERENCES triggers(id),
        source_outcome TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        evidence_id TEXT REFERENCES evidence(id),
        evidence_accepted INTEGER NOT NULL CHECK (evidence_accepted IN (0, 1)),
        candidate_hash TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        candidate_payload_json TEXT,
        created_at TEXT NOT NULL,
        resolved_at TEXT,
        decision_hash TEXT,
        result_memory_id TEXT,
        result_memory_version INTEGER,
        result_skill_id TEXT,
        result_skill_version INTEGER,
        rejection_category TEXT,
        rejection_reason TEXT,
        UNIQUE (agent_id, idempotency_key)
    )
    """.strip(),
    """
    CREATE INDEX learning_proposals_list_idx
        ON learning_proposals(agent_id, operation_id, state, created_at, id)
    """.strip(),
    """
    CREATE TABLE learning_decisions (
        proposal_id TEXT NOT NULL REFERENCES learning_proposals(id),
        decision_hash TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        candidate_hash TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('committed', 'rejected')),
        decided_at TEXT NOT NULL,
        result_memory_id TEXT,
        result_memory_version INTEGER,
        result_skill_id TEXT,
        result_skill_version INTEGER,
        rejection_category TEXT,
        rejection_reason TEXT,
        PRIMARY KEY (proposal_id, decision_hash)
    )
    """.strip(),
    """
    CREATE TRIGGER learning_proposals_reject_identity_update
    BEFORE UPDATE ON learning_proposals
    WHEN OLD.state != 'proposed'
        OR NEW.id != OLD.id
        OR NEW.agent_id != OLD.agent_id
        OR NEW.kind != OLD.kind
        OR NEW.category != OLD.category
        OR NEW.operation_id != OLD.operation_id
        OR NEW.trigger_id != OLD.trigger_id
        OR NEW.source_outcome != OLD.source_outcome
        OR NEW.source_hash != OLD.source_hash
        OR NEW.evidence_id IS NOT OLD.evidence_id
        OR NEW.evidence_accepted != OLD.evidence_accepted
        OR NEW.candidate_hash != OLD.candidate_hash
        OR NEW.idempotency_key != OLD.idempotency_key
        OR NEW.created_at != OLD.created_at
    BEGIN
        SELECT RAISE(ABORT, 'learning proposal identity is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER learning_proposals_reject_delete
    BEFORE DELETE ON learning_proposals
    BEGIN
        SELECT RAISE(ABORT, 'learning proposal history is append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER learning_decisions_reject_update
    BEFORE UPDATE ON learning_decisions
    BEGIN
        SELECT RAISE(ABORT, 'learning decisions are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER learning_decisions_reject_delete
    BEFORE DELETE ON learning_decisions
    BEGIN
        SELECT RAISE(ABORT, 'learning decisions are append-only');
    END
    """.strip(),
    """
    CREATE TABLE skills (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        stable_name TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (agent_id, stable_name)
    )
    """.strip(),
    """
    CREATE TABLE skill_versions (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        skill_id TEXT NOT NULL REFERENCES skills(id),
        stable_name TEXT NOT NULL,
        version TEXT NOT NULL,
        description TEXT NOT NULL,
        domains_json TEXT NOT NULL,
        resource_kinds_json TEXT NOT NULL,
        required_capability_ids_json TEXT NOT NULL,
        activation_mode TEXT NOT NULL,
        sensitivity_notes TEXT,
        policy_notes TEXT,
        source TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        instructions TEXT NOT NULL,
        source_path TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (agent_id, skill_id, version)
    )
    """.strip(),
    """
    CREATE TABLE skill_indexes (
        agent_id TEXT NOT NULL REFERENCES agents(id),
        skill_id TEXT NOT NULL REFERENCES skills(id),
        version_id TEXT NOT NULL REFERENCES skill_versions(id),
        stable_name TEXT NOT NULL,
        version TEXT NOT NULL,
        description TEXT NOT NULL,
        domains_json TEXT NOT NULL,
        resource_kinds_json TEXT NOT NULL,
        required_capability_ids_json TEXT NOT NULL,
        activation_mode TEXT NOT NULL,
        source TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        active_version_id TEXT REFERENCES skill_versions(id),
        updated_at TEXT NOT NULL,
        PRIMARY KEY (agent_id, skill_id)
    )
    """.strip(),
    """
    CREATE INDEX skill_indexes_name_idx
        ON skill_indexes(agent_id, stable_name, skill_id)
    """.strip(),
    """
    CREATE TABLE skill_activations (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        skill_id TEXT NOT NULL REFERENCES skills(id),
        version_id TEXT NOT NULL REFERENCES skill_versions(id),
        previous_version_id TEXT REFERENCES skill_versions(id),
        actor_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        activated_at TEXT NOT NULL
    )
    """.strip(),
    """
    CREATE INDEX skill_activations_history_idx
        ON skill_activations(agent_id, skill_id, activated_at, id)
    """.strip(),
    """
    CREATE TRIGGER skill_versions_reject_update
    BEFORE UPDATE ON skill_versions
    BEGIN
        SELECT RAISE(ABORT, 'skill versions are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER skill_versions_reject_delete
    BEFORE DELETE ON skill_versions
    BEGIN
        SELECT RAISE(ABORT, 'skill versions are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER skill_activations_reject_update
    BEFORE UPDATE ON skill_activations
    BEGIN
        SELECT RAISE(ABORT, 'skill activations are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER skill_activations_reject_delete
    BEFORE DELETE ON skill_activations
    BEGIN
        SELECT RAISE(ABORT, 'skill activations are append-only');
    END
    """.strip(),
)


_MONITOR_SCHEMA_SQL = (
    """
    CREATE TABLE monitor_proposals (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        intended_monitor_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        candidate_hash TEXT NOT NULL,
        candidate_json TEXT NOT NULL,
        source_operation_id TEXT REFERENCES operations(id),
        created_at TEXT NOT NULL,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, idempotency_key)
    )
    """.strip(),
    """
    CREATE INDEX monitor_proposals_history_idx
        ON monitor_proposals(agent_id, created_at, id)
    """.strip(),
    """
    CREATE TRIGGER monitor_proposals_validate_source_insert
    BEFORE INSERT ON monitor_proposals
    WHEN NEW.source_operation_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM operations AS operation
            WHERE operation.id = NEW.source_operation_id
                AND operation.agent_id = NEW.agent_id
        )
    BEGIN
        SELECT RAISE(ABORT, 'monitor proposal source scope is invalid');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_proposals_reject_update
    BEFORE UPDATE ON monitor_proposals
    BEGIN
        SELECT RAISE(ABORT, 'monitor proposals are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_proposals_reject_delete
    BEFORE DELETE ON monitor_proposals
    BEGIN
        SELECT RAISE(ABORT, 'monitor proposals are append-only');
    END
    """.strip(),
    """
    CREATE TABLE monitors (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        status TEXT NOT NULL CHECK (status IN ('enabled', 'paused', 'deleted')),
        current_version INTEGER NOT NULL CHECK (current_version >= 1),
        revision INTEGER NOT NULL CHECK (revision >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        paused_at TEXT,
        deleted_at TEXT,
        UNIQUE (agent_id, id),
        CHECK (updated_at >= created_at),
        CHECK (
            (status = 'enabled' AND paused_at IS NULL AND deleted_at IS NULL)
            OR (status = 'paused' AND paused_at IS NOT NULL AND deleted_at IS NULL)
            OR (status = 'deleted' AND deleted_at IS NOT NULL)
        )
    )
    """.strip(),
    """
    CREATE INDEX monitors_status_idx
        ON monitors(agent_id, status, updated_at, id)
    """.strip(),
    """
    CREATE TRIGGER monitors_reject_identity_update
    BEFORE UPDATE ON monitors
    WHEN NEW.id != OLD.id
        OR NEW.agent_id != OLD.agent_id
        OR NEW.created_at != OLD.created_at
        OR NEW.revision != OLD.revision + 1
        OR NEW.current_version < OLD.current_version
        OR NEW.current_version > OLD.current_version + 1
    BEGIN
        SELECT RAISE(ABORT, 'monitor identity or revision is invalid');
    END
    """.strip(),
    """
    CREATE TRIGGER monitors_reject_terminal_update
    BEFORE UPDATE ON monitors
    WHEN OLD.status = 'deleted'
    BEGIN
        SELECT RAISE(ABORT, 'deleted monitor is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER monitors_reject_delete
    BEFORE DELETE ON monitors
    BEGIN
        SELECT RAISE(ABORT, 'monitor history is retained');
    END
    """.strip(),
    """
    CREATE TABLE monitor_versions (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 1),
        definition_json TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        proposal_id TEXT NOT NULL,
        source_operation_id TEXT REFERENCES operations(id),
        created_at TEXT NOT NULL,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, monitor_id, version),
        FOREIGN KEY (agent_id, monitor_id)
            REFERENCES monitors(agent_id, id),
        FOREIGN KEY (agent_id, proposal_id)
            REFERENCES monitor_proposals(agent_id, id)
    )
    """.strip(),
    """
    CREATE INDEX monitor_versions_history_idx
        ON monitor_versions(agent_id, monitor_id, version, id)
    """.strip(),
    """
    CREATE TRIGGER monitor_versions_validate_source_insert
    BEFORE INSERT ON monitor_versions
    WHEN NEW.source_operation_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM operations AS operation
            WHERE operation.id = NEW.source_operation_id
                AND operation.agent_id = NEW.agent_id
        )
    BEGIN
        SELECT RAISE(ABORT, 'monitor version source scope is invalid');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_versions_reject_update
    BEFORE UPDATE ON monitor_versions
    BEGIN
        SELECT RAISE(ABORT, 'monitor versions are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_versions_reject_delete
    BEFORE DELETE ON monitor_versions
    BEGIN
        SELECT RAISE(ABORT, 'monitor versions are append-only');
    END
    """.strip(),
    """
    CREATE TABLE monitor_confirmations (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        proposal_id TEXT NOT NULL,
        decision TEXT NOT NULL CHECK (decision IN ('confirmed', 'rejected')),
        candidate_hash TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        decided_at TEXT NOT NULL,
        resulting_monitor_id TEXT,
        resulting_version_id TEXT,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, proposal_id),
        FOREIGN KEY (agent_id, proposal_id)
            REFERENCES monitor_proposals(agent_id, id),
        CHECK (
            (decision = 'confirmed'
                AND resulting_monitor_id IS NOT NULL
                AND resulting_version_id IS NOT NULL)
            OR (decision = 'rejected'
                AND resulting_monitor_id IS NULL
                AND resulting_version_id IS NULL)
        )
    )
    """.strip(),
    """
    CREATE TRIGGER monitor_confirmations_reject_update
    BEFORE UPDATE ON monitor_confirmations
    BEGIN
        SELECT RAISE(ABORT, 'monitor confirmations are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_confirmations_reject_delete
    BEFORE DELETE ON monitor_confirmations
    BEGIN
        SELECT RAISE(ABORT, 'monitor confirmations are append-only');
    END
    """.strip(),
    """
    CREATE TABLE monitor_lifecycle (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        action TEXT NOT NULL CHECK (
            action IN ('activate', 'update', 'pause', 'resume', 'delete', 'run_now')
        ),
        from_status TEXT CHECK (from_status IN ('enabled', 'paused', 'deleted')),
        to_status TEXT NOT NULL CHECK (to_status IN ('enabled', 'paused', 'deleted')),
        from_revision INTEGER NOT NULL CHECK (from_revision >= 0),
        to_revision INTEGER NOT NULL CHECK (to_revision = from_revision + 1),
        monitor_version INTEGER NOT NULL CHECK (monitor_version >= 1),
        actor_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        operation_id TEXT REFERENCES operations(id),
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, idempotency_key),
        FOREIGN KEY (agent_id, monitor_id)
            REFERENCES monitors(agent_id, id)
    )
    """.strip(),
    """
    CREATE INDEX monitor_lifecycle_history_idx
        ON monitor_lifecycle(agent_id, monitor_id, to_revision, id)
    """.strip(),
    """
    CREATE TRIGGER monitor_lifecycle_validate_operation_insert
    BEFORE INSERT ON monitor_lifecycle
    WHEN NEW.operation_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM operations AS operation
            WHERE operation.id = NEW.operation_id
                AND operation.agent_id = NEW.agent_id
        )
    BEGIN
        SELECT RAISE(ABORT, 'monitor lifecycle operation scope is invalid');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_lifecycle_reject_update
    BEFORE UPDATE ON monitor_lifecycle
    BEGIN
        SELECT RAISE(ABORT, 'monitor lifecycle is append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_lifecycle_reject_delete
    BEFORE DELETE ON monitor_lifecycle
    BEGIN
        SELECT RAISE(ABORT, 'monitor lifecycle is append-only');
    END
    """.strip(),
    """
    CREATE TABLE monitor_schedule_state (
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        next_scheduled_at TEXT,
        updated_at TEXT NOT NULL,
        last_scheduled_at TEXT,
        cooldown_until TEXT,
        backoff_until TEXT,
        consecutive_failures INTEGER NOT NULL CHECK (consecutive_failures >= 0),
        consecutive_matches INTEGER NOT NULL CHECK (consecutive_matches >= 0),
        checkpoint_version INTEGER NOT NULL CHECK (checkpoint_version >= 0),
        last_occurrence_id TEXT,
        last_run_id TEXT,
        last_operation_id TEXT REFERENCES operations(id),
        PRIMARY KEY (agent_id, monitor_id),
        FOREIGN KEY (agent_id, monitor_id)
            REFERENCES monitors(agent_id, id)
    )
    """.strip(),
    """
    CREATE INDEX monitor_schedule_due_idx
        ON monitor_schedule_state(agent_id, next_scheduled_at, monitor_id)
    """.strip(),
    """
    CREATE TABLE monitor_occurrences (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        monitor_version INTEGER NOT NULL CHECK (monitor_version >= 1),
        kind TEXT NOT NULL CHECK (kind IN ('scheduled', 'run_now')),
        scheduled_for TEXT NOT NULL,
        occurrence_key TEXT NOT NULL,
        trigger_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        manual_key TEXT,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, occurrence_key),
        UNIQUE (trigger_id),
        UNIQUE (run_id),
        FOREIGN KEY (agent_id, monitor_id, monitor_version)
            REFERENCES monitor_versions(agent_id, monitor_id, version),
        CHECK (
            (kind = 'scheduled' AND manual_key IS NULL)
            OR (kind = 'run_now' AND manual_key IS NOT NULL)
        )
    )
    """.strip(),
    """
    CREATE INDEX monitor_occurrences_history_idx
        ON monitor_occurrences(agent_id, monitor_id, scheduled_for, id)
    """.strip(),
    """
    CREATE UNIQUE INDEX monitor_occurrences_manual_key_idx
        ON monitor_occurrences(agent_id, monitor_id, manual_key)
        WHERE manual_key IS NOT NULL
    """.strip(),
    """
    CREATE TRIGGER monitor_occurrences_reject_update
    BEFORE UPDATE ON monitor_occurrences
    BEGIN
        SELECT RAISE(ABORT, 'monitor occurrences are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_occurrences_reject_delete
    BEFORE DELETE ON monitor_occurrences
    BEGIN
        SELECT RAISE(ABORT, 'monitor occurrences are append-only');
    END
    """.strip(),
    """
    CREATE TABLE monitor_tick_leases (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        occurrence_id TEXT NOT NULL,
        holder_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL CHECK (fencing_token >= 1),
        claimed_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        released_at TEXT,
        release_reason TEXT,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, occurrence_id, fencing_token),
        FOREIGN KEY (agent_id, occurrence_id)
            REFERENCES monitor_occurrences(agent_id, id),
        FOREIGN KEY (agent_id, monitor_id)
            REFERENCES monitors(agent_id, id),
        CHECK (expires_at > claimed_at),
        CHECK ((released_at IS NULL) = (release_reason IS NULL))
    )
    """.strip(),
    """
    CREATE UNIQUE INDEX monitor_tick_leases_live_idx
        ON monitor_tick_leases(agent_id, occurrence_id)
        WHERE released_at IS NULL
    """.strip(),
    """
    CREATE TRIGGER monitor_tick_leases_reject_identity_update
    BEFORE UPDATE ON monitor_tick_leases
    WHEN NEW.id != OLD.id
        OR NEW.agent_id != OLD.agent_id
        OR NEW.monitor_id != OLD.monitor_id
        OR NEW.occurrence_id != OLD.occurrence_id
        OR NEW.holder_id != OLD.holder_id
        OR NEW.fencing_token != OLD.fencing_token
        OR NEW.claimed_at != OLD.claimed_at
        OR NEW.expires_at != OLD.expires_at
    BEGIN
        SELECT RAISE(ABORT, 'monitor tick-lease identity is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_tick_leases_reject_terminal_update
    BEFORE UPDATE ON monitor_tick_leases
    WHEN OLD.released_at IS NOT NULL
    BEGIN
        SELECT RAISE(ABORT, 'released monitor tick lease is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_tick_leases_reject_delete
    BEFORE DELETE ON monitor_tick_leases
    BEGIN
        SELECT RAISE(ABORT, 'monitor tick-lease history is append-only');
    END
    """.strip(),
    """
    CREATE TABLE monitor_runs (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        occurrence_id TEXT NOT NULL,
        trigger_id TEXT NOT NULL,
        attempt INTEGER NOT NULL CHECK (attempt >= 1),
        fencing_token INTEGER NOT NULL CHECK (fencing_token >= 1),
        status TEXT NOT NULL CHECK (
            status IN ('pending', 'running', 'waiting', 'succeeded', 'failed',
                'cancelled', 'skipped')
        ),
        started_at TEXT NOT NULL,
        operation_id TEXT UNIQUE REFERENCES operations(id),
        completed_at TEXT,
        failure_reason TEXT,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, occurrence_id),
        UNIQUE (trigger_id),
        FOREIGN KEY (agent_id, occurrence_id)
            REFERENCES monitor_occurrences(agent_id, id),
        FOREIGN KEY (agent_id, monitor_id)
            REFERENCES monitors(agent_id, id),
        CHECK (
            (status IN ('succeeded', 'failed', 'cancelled', 'skipped')
                AND completed_at IS NOT NULL)
            OR (status IN ('pending', 'running', 'waiting')
                AND completed_at IS NULL)
        )
    )
    """.strip(),
    """
    CREATE INDEX monitor_runs_history_idx
        ON monitor_runs(agent_id, monitor_id, started_at, id)
    """.strip(),
    """
    CREATE TRIGGER monitor_runs_reject_identity_update
    BEFORE UPDATE ON monitor_runs
    WHEN NEW.id != OLD.id
        OR NEW.agent_id != OLD.agent_id
        OR NEW.monitor_id != OLD.monitor_id
        OR NEW.occurrence_id != OLD.occurrence_id
        OR NEW.trigger_id != OLD.trigger_id
        OR NEW.attempt < OLD.attempt
        OR NEW.fencing_token < OLD.fencing_token
    BEGIN
        SELECT RAISE(ABORT, 'monitor run identity or fence is invalid');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_runs_reject_terminal_update
    BEFORE UPDATE ON monitor_runs
    WHEN OLD.completed_at IS NOT NULL
    BEGIN
        SELECT RAISE(ABORT, 'terminal monitor run is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_runs_reject_delete
    BEFORE DELETE ON monitor_runs
    BEGIN
        SELECT RAISE(ABORT, 'monitor run history is retained');
    END
    """.strip(),
    """
    CREATE TABLE monitor_checkpoints (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 1),
        run_id TEXT NOT NULL,
        cursor_json TEXT NOT NULL,
        cursor_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        previous_version INTEGER,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, monitor_id, version),
        UNIQUE (agent_id, run_id),
        FOREIGN KEY (agent_id, run_id)
            REFERENCES monitor_runs(agent_id, id),
        CHECK (
            (version = 1 AND previous_version IS NULL)
            OR (version > 1 AND previous_version = version - 1)
        )
    )
    """.strip(),
    """
    CREATE TRIGGER monitor_checkpoints_reject_update
    BEFORE UPDATE ON monitor_checkpoints
    BEGIN
        SELECT RAISE(ABORT, 'monitor checkpoints are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_checkpoints_reject_delete
    BEFORE DELETE ON monitor_checkpoints
    BEGIN
        SELECT RAISE(ABORT, 'monitor checkpoints are append-only');
    END
    """.strip(),
    """
    CREATE TABLE monitor_findings (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        monitor_id TEXT NOT NULL,
        occurrence_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        operation_id TEXT NOT NULL REFERENCES operations(id),
        evidence_id TEXT NOT NULL REFERENCES evidence(id),
        severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
        summary TEXT NOT NULL,
        details_json TEXT NOT NULL,
        dedupe_key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, run_id, dedupe_key),
        FOREIGN KEY (agent_id, run_id)
            REFERENCES monitor_runs(agent_id, id),
        FOREIGN KEY (agent_id, occurrence_id)
            REFERENCES monitor_occurrences(agent_id, id),
        FOREIGN KEY (agent_id, monitor_id)
            REFERENCES monitors(agent_id, id)
    )
    """.strip(),
    """
    CREATE INDEX monitor_findings_history_idx
        ON monitor_findings(agent_id, monitor_id, created_at, id)
    """.strip(),
    """
    CREATE TRIGGER monitor_findings_validate_evidence_insert
    BEFORE INSERT ON monitor_findings
    WHEN NOT EXISTS (
        SELECT 1
        FROM evidence
        JOIN operations ON operations.id = evidence.operation_id
        WHERE evidence.id = NEW.evidence_id
            AND evidence.operation_id = NEW.operation_id
            AND evidence.accepted = 1
            AND operations.agent_id = NEW.agent_id
    )
    BEGIN
        SELECT RAISE(ABORT, 'monitor finding evidence is invalid');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_findings_reject_update
    BEFORE UPDATE ON monitor_findings
    BEGIN
        SELECT RAISE(ABORT, 'monitor findings are append-only');
    END
    """.strip(),
    """
    CREATE TRIGGER monitor_findings_reject_delete
    BEFORE DELETE ON monitor_findings
    BEGIN
        SELECT RAISE(ABORT, 'monitor findings are append-only');
    END
    """.strip(),
    "ALTER TABLE runtime_events ADD COLUMN monitor_id TEXT",
    """
    CREATE INDEX runtime_events_monitor_idx
        ON runtime_events(agent_id, monitor_id, agent_sequence)
    """.strip(),
    """
    CREATE TRIGGER runtime_events_validate_monitor_insert
    BEFORE INSERT ON runtime_events
    WHEN NEW.monitor_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM monitors
            WHERE monitors.id = NEW.monitor_id
                AND monitors.agent_id = NEW.agent_id
        )
        AND NOT EXISTS (
            SELECT 1 FROM monitor_proposals
            WHERE monitor_proposals.intended_monitor_id = NEW.monitor_id
                AND monitor_proposals.agent_id = NEW.agent_id
        )
    BEGIN
        SELECT RAISE(ABORT, 'event monitor correlation is invalid');
    END
    """.strip(),
)


_HOST_INBOX_SCHEMA_SQL = (
    """
    CREATE TABLE host_inbox (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        kind TEXT NOT NULL CHECK (kind IN ('trigger', 'approval_wake')),
        idempotency_key TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision IN (1, 2)),
        status TEXT NOT NULL CHECK (status IN ('pending', 'completed')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        trigger_id TEXT,
        operation_id TEXT REFERENCES operations(id),
        error TEXT,
        UNIQUE (agent_id, id),
        UNIQUE (agent_id, idempotency_key),
        CHECK (updated_at >= created_at),
        CHECK (
            (kind = 'trigger' AND trigger_id IS NOT NULL)
            OR (kind = 'approval_wake' AND trigger_id IS NULL)
        ),
        CHECK (
            (status = 'pending' AND revision = 1
                AND updated_at = created_at
                AND operation_id IS NULL AND error IS NULL)
            OR (status = 'completed' AND revision = 2)
        )
    )
    """.strip(),
    """
    CREATE INDEX host_inbox_pending_idx
        ON host_inbox(agent_id, status, created_at, id)
    """.strip(),
    """
    CREATE TRIGGER host_inbox_reject_identity_update
    BEFORE UPDATE ON host_inbox
    WHEN NEW.id != OLD.id
        OR NEW.agent_id != OLD.agent_id
        OR NEW.kind != OLD.kind
        OR NEW.idempotency_key != OLD.idempotency_key
        OR NEW.request_hash != OLD.request_hash
        OR NEW.payload_json != OLD.payload_json
        OR NEW.created_at != OLD.created_at
        OR NEW.trigger_id IS NOT OLD.trigger_id
        OR NEW.revision != OLD.revision + 1
    BEGIN
        SELECT RAISE(ABORT, 'host inbox request identity is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER host_inbox_reject_terminal_update
    BEFORE UPDATE ON host_inbox
    WHEN OLD.status = 'completed'
    BEGIN
        SELECT RAISE(ABORT, 'completed host inbox item is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER host_inbox_validate_operation_update
    BEFORE UPDATE ON host_inbox
    WHEN NEW.operation_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM operations AS operation
            WHERE operation.id = NEW.operation_id
                AND operation.agent_id = NEW.agent_id
                AND (
                    NEW.kind != 'trigger'
                    OR operation.trigger_id = NEW.trigger_id
                )
        )
    BEGIN
        SELECT RAISE(ABORT, 'host inbox operation correlation is invalid');
    END
    """.strip(),
    """
    CREATE TRIGGER host_inbox_reject_delete
    BEFORE DELETE ON host_inbox
    BEGIN
        SELECT RAISE(ABORT, 'host inbox history is retained');
    END
    """.strip(),
)


_HOST_MUTATION_ADMISSION_SCHEMA_SQL = (
    """
    CREATE TABLE host_mutation_admissions (
        agent_id TEXT NOT NULL REFERENCES agents(id),
        idempotency_key TEXT NOT NULL,
        method TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (agent_id, idempotency_key)
    )
    """.strip(),
    """
    CREATE TRIGGER host_mutation_admissions_reject_update
    BEFORE UPDATE ON host_mutation_admissions
    BEGIN
        SELECT RAISE(ABORT, 'host mutation admission is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER host_mutation_admissions_reject_delete
    BEFORE DELETE ON host_mutation_admissions
    BEGIN
        SELECT RAISE(ABORT, 'host mutation admission is retained');
    END
    """.strip(),
)


_TASK_VALIDATION_SCHEMA_SQL = (
    """
    ALTER TABLE tasks ADD COLUMN validation_schema_version INTEGER NOT NULL
        DEFAULT 0 CHECK (validation_schema_version IN (0, 1))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_passed INTEGER NOT NULL DEFAULT 1
        CHECK (validation_passed IN (0, 1))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_in_scope INTEGER NOT NULL DEFAULT 1
        CHECK (validation_in_scope IN (0, 1))
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_destructive INTEGER NOT NULL DEFAULT 0
        CHECK (
            validation_destructive IN (0, 1)
            AND (
                validation_destructive = 0
                OR (access_mode = 'write' AND side_effecting = 1)
            )
        )
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_sensitivity_class TEXT NOT NULL
        DEFAULT 'internal' CHECK (
            length(trim(validation_sensitivity_class)) BETWEEN 1 AND 128
            AND validation_sensitivity_class = trim(validation_sensitivity_class)
        )
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_source_id TEXT CHECK (
        (
            validation_source_id IS NULL
            AND validation_schema_version = 0
        )
        OR (
            validation_source_id IS NOT NULL
            AND length(trim(validation_source_id)) BETWEEN 1 AND 512
            AND validation_source_id = trim(validation_source_id)
            AND validation_schema_version >= 1
        )
    )
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_resource_ids_json TEXT NOT NULL
        DEFAULT '[]'
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_resource_revisions_json TEXT NOT NULL
        DEFAULT '[]'
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_source_revision TEXT CHECK (
        validation_source_revision IS NULL
        OR (
            validation_schema_version >= 1
            AND length(trim(validation_source_revision)) BETWEEN 1 AND 1024
            AND validation_source_revision = trim(validation_source_revision)
        )
    )
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_impact_json TEXT NOT NULL DEFAULT '{}'
        CHECK (length(validation_impact_json) <= 16384)
    """.strip(),
    """
    ALTER TABLE tasks ADD COLUMN validation_evidence_ids_json TEXT NOT NULL
        DEFAULT '[]' CHECK (
            validation_schema_version >= 1
            OR (
                validation_passed = 1
                AND validation_in_scope = 1
                AND validation_destructive = 0
                AND validation_sensitivity_class = 'internal'
                AND validation_source_id IS NULL
                AND validation_resource_ids_json = '[]'
                AND validation_resource_revisions_json = '[]'
                AND validation_source_revision IS NULL
                AND validation_impact_json = '{}'
                AND validation_evidence_ids_json = '[]'
            )
        )
    """.strip(),
)


_AGENT_RUNTIME_DEFAULTS_SCHEMA_SQL = (
    """
    CREATE TABLE agent_runtime_defaults (
        agent_id TEXT PRIMARY KEY REFERENCES agents(id),
        schema_version INTEGER NOT NULL CHECK (schema_version = 1),
        revision INTEGER NOT NULL CHECK (revision = 1),
        fingerprint TEXT NOT NULL CHECK (
            length(fingerprint) = 64
            AND fingerprint NOT GLOB '*[^0-9a-f]*'
        ),
        budget_max_turns INTEGER NOT NULL CHECK (budget_max_turns > 0),
        budget_max_actions INTEGER NOT NULL CHECK (budget_max_actions > 0),
        budget_max_repairs INTEGER NOT NULL CHECK (budget_max_repairs > 0),
        budget_max_identical_failures INTEGER NOT NULL CHECK (
            budget_max_identical_failures > 0
        ),
        budget_max_observation_characters INTEGER NOT NULL CHECK (
            budget_max_observation_characters > 0
        ),
        budget_max_total_tokens INTEGER NOT NULL CHECK (
            budget_max_total_tokens > 0
        ),
        budget_max_wall_time_seconds REAL NOT NULL CHECK (
            budget_max_wall_time_seconds > 0
        ),
        budget_task_timeout_seconds REAL NOT NULL CHECK (
            budget_task_timeout_seconds > 0
        ),
        budget_max_estimated_cost_usd TEXT,
        policy_id TEXT NOT NULL CHECK (length(trim(policy_id)) > 0),
        policy_version TEXT NOT NULL CHECK (length(trim(policy_version)) > 0),
        policy_allow_destructive INTEGER NOT NULL CHECK (
            policy_allow_destructive IN (0, 1)
        ),
        bound_at TEXT NOT NULL
    )
    """.strip(),
    """
    CREATE TRIGGER agent_runtime_defaults_reject_update
    BEFORE UPDATE ON agent_runtime_defaults
    BEGIN
        SELECT RAISE(ABORT, 'agent runtime-default binding is immutable');
    END
    """.strip(),
    """
    CREATE TRIGGER agent_runtime_defaults_reject_delete
    BEFORE DELETE ON agent_runtime_defaults
    BEGIN
        SELECT RAISE(ABORT, 'agent runtime-default binding is immutable');
    END
    """.strip(),
)


# Migration 1 records only the v2 file/migration foundation. Migration 2 adds
# the first normalized runtime lifecycle aggregate without an opaque snapshot.
# Migration 3 assigns one append-only committed-event sequence per agent.
# Migration 4 persists immutable execution-safety facts, dependency edges, and
# fenced lease history. Legacy in-flight work is failed closed during upgrade.
# Migration 5 adds the explicit nullable link from accepted evidence to the
# separately durable content-addressed blob manifest.
# Migration 6 normalizes exact approval decisions and correlates their events.
# Legacy approval-waiting tasks have no exact fingerprints, so upgrade fails
# them closed into manual recovery rather than synthesizing authority.
# Migration 7 adds the authoritative per-home agent identity and durable session
# directory. Canonical transcript content remains operation-owned and is read
# from the already-normalized trigger/model/observation rows.
# Migration 8 adds attached-source ownership, catalog sync history, complete
# current-source projections, revision components, bounded FTS5 search, and
# relationship traversal facts.
# Migration 9 adds append-only session compression, scoped versioned memory,
# redaction-safe learning proposals, and immutable skill version history.
# Migration 10 adds the durable monitor control plane, stable scheduled
# occurrences, fenced tick claims, atomic outcomes, and monitor event
# correlation without introducing an alternate operation-execution path.
# Migration 11 adds the foreground host's durable, idempotent trigger and
# approval-wake inbox. The single writer owns processing, so no queue lease or
# parallel scheduling authority is introduced here.
# Migration 12 binds each local-control idempotency key to one canonical method
# and parameter hash before dispatch. The admission ledger owns no execution or
# result state; route owners remain responsible for replay-safe recovery.
# Migration 13 persists validator-owned scope, sensitivity, impact, and accepted
# prerequisite evidence on the immutable task record. Schema-zero defaults
# preserve Phase-2 approval fingerprints for already-pending v12 operations.
# Migration 14 binds future-operation budgets and default policy once per agent.
# Model and retry-route configuration remains in the immutable model-profile
# binding, whose router identity includes the complete retry-policy fingerprint.
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
    _SQLiteMigration(
        version=6,
        name="normalize_approval_lifecycle",
        statements=_APPROVAL_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=7,
        name="add_agent_identity_and_sessions",
        statements=_AGENT_SESSION_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=8,
        name="add_source_and_catalog_store",
        statements=_CATALOG_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=9,
        name="add_context_memory_learning_and_skills",
        statements=_CONTEXT_MEMORY_LEARNING_SKILL_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=10,
        name="add_monitor_lifecycle_and_scheduling",
        statements=_MONITOR_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=11,
        name="add_host_trigger_inbox",
        statements=_HOST_INBOX_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=12,
        name="bind_host_mutation_idempotency",
        statements=_HOST_MUTATION_ADMISSION_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=13,
        name="persist_task_validation_facts",
        statements=_TASK_VALIDATION_SCHEMA_SQL,
    ),
    _SQLiteMigration(
        version=14,
        name="bind_agent_runtime_defaults",
        statements=_AGENT_RUNTIME_DEFAULTS_SCHEMA_SQL,
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

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity:
        """Create or verify the one authoritative identity for this database."""

        if not isinstance(identity, AgentIdentity):
            raise TypeError("identity must be an AgentIdentity record")
        return await self._run_connection(
            lambda connection: _initialize_agent_identity(connection, identity)
        )

    async def load_identity(self) -> AgentIdentity | None:
        """Load the authoritative database identity, if initialized."""

        return await self._run_connection(_load_agent_identity)

    async def bind_runtime_defaults(
        self,
        agent_id: str,
        defaults: AgentRuntimeDefaults,
    ) -> AgentRuntimeDefaults:
        """Bind or verify the agent's immutable future-operation defaults."""

        _require_identity(agent_id, "agent_id")
        if not isinstance(defaults, AgentRuntimeDefaults):
            raise TypeError("defaults must be AgentRuntimeDefaults")
        bound_at = self._clock()
        return await self._run_connection(
            lambda connection: _bind_agent_runtime_defaults(
                connection,
                agent_id,
                defaults,
                bound_at=bound_at,
            )
        )

    async def load_runtime_defaults(
        self,
        agent_id: str,
    ) -> AgentRuntimeDefaults | None:
        """Load the exact immutable future-operation defaults, if bound."""

        _require_identity(agent_id, "agent_id")
        return await self._run_connection(
            lambda connection: _load_agent_runtime_defaults(connection, agent_id)
        )

    async def admit_host_mutation(
        self,
        request: HostMutationAdmission,
    ) -> HostMutationAdmission:
        """Bind one client key to an exact local-control mutation."""

        if not isinstance(request, HostMutationAdmission):
            raise TypeError("request must be a HostMutationAdmission")
        return await self._run_connection(
            lambda connection: _admit_host_mutation(connection, request)
        )

    async def enqueue_host_inbox(self, item: HostInboxItem) -> HostInboxItem:
        """Enqueue one host request idempotently under its canonical hash."""

        if not isinstance(item, HostInboxItem):
            raise TypeError("item must be a HostInboxItem")
        if item.status is not HostInboxStatus.PENDING:
            raise ValueError("new host inbox item must be pending")
        return await self._run_connection(
            lambda connection: _enqueue_host_inbox(connection, item)
        )

    async def list_pending_host_inbox(
        self,
        agent_id: str,
        *,
        limit: int,
    ) -> tuple[HostInboxItem, ...]:
        """List bounded pending work in deterministic FIFO order."""

        _require_identity(agent_id, "agent_id")
        _require_bounded_limit(limit, maximum=1_000)
        return await self._run_connection(
            lambda connection: _list_pending_host_inbox(
                connection,
                agent_id,
                limit=limit,
            )
        )

    async def complete_host_inbox(
        self,
        item: HostInboxItem,
        *,
        expected_revision: int,
    ) -> HostInboxItem:
        """Complete one exact inbox request under an optimistic CAS guard."""

        if not isinstance(item, HostInboxItem):
            raise TypeError("item must be a HostInboxItem")
        if item.status is not HostInboxStatus.COMPLETED:
            raise ValueError("host inbox completion must be completed")
        _require_revision(expected_revision)
        return await self._run_connection(
            lambda connection: _complete_host_inbox(
                connection,
                item,
                expected_revision=expected_revision,
            )
        )

    async def bind_model_profile(
        self,
        agent_id: str,
        profile: ModelProfile,
    ) -> ModelProfile:
        """Bind or verify the agent's one immutable built-in model profile."""

        _require_identity(agent_id, "agent_id")
        if not isinstance(profile, ModelProfile):
            raise TypeError("profile must be a ModelProfile")
        return await self._run_connection(
            lambda connection: _bind_agent_model_profile(
                connection,
                agent_id,
                profile,
            )
        )

    async def load_model_profile(self, agent_id: str) -> ModelProfile | None:
        """Load the agent's exact configured built-in model profile."""

        _require_identity(agent_id, "agent_id")
        return await self._run_connection(
            lambda connection: _load_agent_model_profile(connection, agent_id)
        )

    async def create_session(self, session: Session) -> Session:
        """Persist one stable session identity before its first operation."""

        if not isinstance(session, Session):
            raise TypeError("session must be a Session record")
        return await self._run_connection(
            lambda connection: _create_session(connection, session)
        )

    async def load_session(
        self,
        agent_id: str,
        session_id: str,
    ) -> SessionTranscript | None:
        """Load one restart-safe provider-neutral session transcript."""

        _require_identity(agent_id, "agent_id")
        _require_identity(session_id, "session_id")
        return await self._run_connection(
            lambda connection: _load_session_transcript(
                connection,
                agent_id,
                session_id,
            )
        )

    async def load_session_compression(
        self,
        agent_id: str,
        session_id: str,
    ) -> SessionCompressionCheckpoint | None:
        """Load the latest immutable compression checkpoint for one session."""

        _require_identity(agent_id, "agent_id")
        _require_identity(session_id, "session_id")
        return await self._run_connection(
            lambda connection: _load_session_compression(
                connection,
                agent_id,
                session_id,
            )
        )

    async def commit_session_compression(
        self,
        checkpoint: SessionCompressionCheckpoint,
        *,
        expected_version: int,
    ) -> SessionCompressionCheckpoint:
        """Append one exact session-prefix checkpoint under a version guard."""

        if not isinstance(checkpoint, SessionCompressionCheckpoint):
            raise TypeError("checkpoint must be a SessionCompressionCheckpoint")
        if (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 0
        ):
            raise ValueError("expected_version must be a non-negative integer")
        return await self._run_connection(
            lambda connection: _commit_session_compression(
                connection,
                checkpoint,
                expected_version=expected_version,
            )
        )

    async def load_session_operation(
        self,
        operation_id: str,
    ) -> SessionOperationFacts | None:
        """Load the bounded durable facts used by session compression."""

        _require_identity(operation_id, "operation_id")
        return await self._run_connection(
            lambda connection: _load_session_operation_facts(
                connection,
                operation_id,
            )
        )

    async def create_memory(
        self,
        record: MemoryRecord,
        version: MemoryVersion,
    ) -> MemoryHistory:
        """Create one memory head and its immutable first version atomically."""

        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")
        if not isinstance(version, MemoryVersion):
            raise TypeError("version must be a MemoryVersion")
        return await self._run_connection(
            lambda connection: _create_memory(connection, record, version)
        )

    async def recall_candidates(
        self,
        *,
        query: str,
        scope: MemoryScope,
        states: tuple[MemoryState, ...],
        sensitivities: tuple[MemorySensitivity, ...],
        unexpired_at: datetime,
        limit: int,
    ) -> tuple[MemorySnapshot, ...]:
        """Return scoped current memory rows after structured pre-filtering."""

        _require_identity(query, "memory query")
        _validate_memory_query(scope, states, sensitivities, limit)
        if (
            not isinstance(unexpired_at, datetime)
            or unexpired_at.tzinfo is None
            or unexpired_at.utcoffset() is None
        ):
            raise ValueError("unexpired_at must be timezone-aware")
        return await self._run_connection(
            lambda connection: _recall_memory_candidates(
                connection,
                query=query,
                scope=scope,
                states=states,
                sensitivities=sensitivities,
                unexpired_at=unexpired_at,
                limit=limit,
            )
        )

    async def list_candidates(
        self,
        *,
        scope: MemoryScope,
        states: tuple[MemoryState, ...],
        sensitivities: tuple[MemorySensitivity, ...],
        limit: int,
    ) -> tuple[MemorySnapshot, ...]:
        """List current scoped memory rows without lexical ranking."""

        _validate_memory_query(scope, states, sensitivities, limit)
        return await self._run_connection(
            lambda connection: _list_memory_candidates(
                connection,
                scope=scope,
                states=states,
                sensitivities=sensitivities,
                limit=limit,
            )
        )

    async def load_history(
        self,
        agent_id: str,
        memory_id: str,
    ) -> MemoryHistory | None:
        """Load one agent-scoped memory and all immutable versions."""

        _require_identity(agent_id, "agent_id")
        _require_identity(memory_id, "memory_id")
        return await self._run_connection(
            lambda connection: _load_memory_history(
                connection,
                agent_id,
                memory_id,
            )
        )

    async def supersede(
        self,
        request: MemorySupersessionRequest,
    ) -> MemoryHistory:
        """Append a replacement memory version under an exact head guard."""

        if not isinstance(request, MemorySupersessionRequest):
            raise TypeError("request must be a MemorySupersessionRequest")
        return await self._run_connection(
            lambda connection: _replace_memory_version(
                connection,
                agent_id=request.agent_id,
                memory_id=request.memory_id,
                expected_version=request.expected_version,
                replacement=request.replacement,
                restore_version=None,
            )
        )

    async def restore(self, request: MemoryRestoreRequest) -> MemoryHistory:
        """Append a validated copy of one historical memory version."""

        if not isinstance(request, MemoryRestoreRequest):
            raise TypeError("request must be a MemoryRestoreRequest")
        return await self._run_connection(
            lambda connection: _replace_memory_version(
                connection,
                agent_id=request.agent_id,
                memory_id=request.memory_id,
                expected_version=request.expected_version,
                replacement=request.replacement,
                restore_version=request.restore_version,
            )
        )

    async def load_resource_alias(
        self,
        scope: MemoryScope,
        logical_key: str,
    ) -> MemoryHistory | None:
        """Load one exact resource-alias logical identity and its history."""

        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        _require_identity(logical_key, "logical_key")
        return await self._run_connection(
            lambda connection: _load_resource_alias(
                connection,
                scope,
                logical_key,
            )
        )

    async def commit_explicit_correction(
        self,
        request: ExplicitCorrectionCommit,
    ) -> ExplicitCorrectionResult:
        """Atomically commit one correction proposal and alias-memory change."""

        if not isinstance(request, ExplicitCorrectionCommit):
            raise TypeError("request must be an ExplicitCorrectionCommit")
        return await self._run_connection(
            lambda connection: _commit_explicit_correction(connection, request)
        )

    async def create_proposal(
        self,
        proposal: LearningProposal,
    ) -> LearningProposal:
        """Persist a proposal idempotently without retaining rejected payloads."""

        if not isinstance(proposal, LearningProposal):
            raise TypeError("proposal must be a LearningProposal")
        return await self._run_connection(
            lambda connection: _create_learning_proposal(connection, proposal)
        )

    async def load_proposal(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> LearningProposal | None:
        """Load one proposal within its authoritative agent scope."""

        _require_identity(agent_id, "agent_id")
        _require_identity(proposal_id, "proposal_id")
        return await self._run_connection(
            lambda connection: _load_learning_proposal(
                connection,
                agent_id,
                proposal_id,
            )
        )

    async def list_proposals(
        self,
        agent_id: str,
        *,
        operation_id: str | None,
        states: tuple[LearningProposalState, ...],
        limit: int,
    ) -> tuple[LearningProposal, ...]:
        """List bounded proposal history using explicit state filters."""

        _require_identity(agent_id, "agent_id")
        if operation_id is not None:
            _require_identity(operation_id, "operation_id")
        if not states or any(
            not isinstance(state, LearningProposalState) for state in states
        ):
            raise ValueError("states must contain LearningProposalState values")
        if len(states) != len(set(states)):
            raise ValueError("states cannot contain duplicates")
        _require_bounded_limit(limit, maximum=200)
        return await self._run_connection(
            lambda connection: _list_learning_proposals(
                connection,
                agent_id,
                operation_id=operation_id,
                states=states,
                limit=limit,
            )
        )

    async def resolve_proposal(
        self,
        decision: LearningDecision,
        *,
        expected_state: LearningProposalState,
    ) -> LearningProposal:
        """Commit one idempotent terminal learning decision under state CAS."""

        if not isinstance(decision, LearningDecision):
            raise TypeError("decision must be a LearningDecision")
        if not isinstance(expected_state, LearningProposalState):
            raise TypeError("expected_state must be a LearningProposalState")
        return await self._run_connection(
            lambda connection: _resolve_learning_proposal(
                connection,
                decision,
                expected_state=expected_state,
            )
        )

    async def record_discovery(
        self,
        skill: Skill,
        version: SkillVersion,
        index: SkillIndex,
    ) -> SkillIndex:
        """Record one immutable discovered version and refresh its projection."""

        if not isinstance(skill, Skill):
            raise TypeError("skill must be a Skill")
        if not isinstance(version, SkillVersion):
            raise TypeError("version must be a SkillVersion")
        if not isinstance(index, SkillIndex):
            raise TypeError("index must be a SkillIndex")
        return await self._run_connection(
            lambda connection: _record_skill_discovery(
                connection,
                skill,
                version,
                index,
            )
        )

    async def list_skill_index(self, agent_id: str) -> tuple[SkillIndex, ...]:
        """List compact skill projections in deterministic name order."""

        _require_identity(agent_id, "agent_id")
        return await self._run_connection(
            lambda connection: _list_skill_index(connection, agent_id)
        )

    async def load_skill_index(
        self,
        agent_id: str,
        skill_id: str,
    ) -> SkillIndex | None:
        """Load one compact skill projection by stable identity."""

        _require_identity(agent_id, "agent_id")
        _require_identity(skill_id, "skill_id")
        return await self._run_connection(
            lambda connection: _load_skill_index(connection, agent_id, skill_id)
        )

    async def load_skill_version(
        self,
        agent_id: str,
        version_id: str,
    ) -> SkillVersion | None:
        """Load one immutable skill version within its agent scope."""

        _require_identity(agent_id, "agent_id")
        _require_identity(version_id, "version_id")
        return await self._run_connection(
            lambda connection: _load_skill_version(connection, agent_id, version_id)
        )

    async def inspect_skill(
        self,
        agent_id: str,
        skill_id: str,
    ) -> SkillInspection | None:
        """Load one skill's complete version and activation audit history."""

        _require_identity(agent_id, "agent_id")
        _require_identity(skill_id, "skill_id")
        return await self._run_connection(
            lambda connection: _inspect_skill(connection, agent_id, skill_id)
        )

    async def activate_skill(
        self,
        activation: SkillActivation,
        *,
        expected_active_version_id: str | None,
    ) -> SkillInspection:
        """Append one activation and compare-and-swap the active projection."""

        if not isinstance(activation, SkillActivation):
            raise TypeError("activation must be a SkillActivation")
        if expected_active_version_id is not None:
            _require_identity(
                expected_active_version_id,
                "expected_active_version_id",
            )
        return await self._run_connection(
            lambda connection: _activate_skill(
                connection,
                activation,
                expected_active_version_id=expected_active_version_id,
            )
        )

    async def commit_skill_change(
        self,
        request: SkillChangeCommit,
    ) -> SkillChangeAcceptanceResult:
        """Atomically accept, version, activate, and resolve a skill proposal."""

        if not isinstance(request, SkillChangeCommit):
            raise TypeError("request must be a SkillChangeCommit")
        return await self._run_connection(
            lambda connection: _commit_skill_change(connection, request)
        )

    async def create_monitor_proposal(
        self,
        proposal: MonitorProposal,
        event: RuntimeEvent,
    ) -> MonitorProposal:
        """Persist one inert monitor proposal and its event atomically."""

        if not isinstance(proposal, MonitorProposal):
            raise TypeError("proposal must be a MonitorProposal")
        _validate_monitor_event(event, proposal.agent_id, proposal.intended_monitor_id)
        result, committed_events = await self._run_connection(
            lambda connection: _create_monitor_proposal(connection, proposal, event)
        )
        self._publish_committed_event_wake_hints(committed_events)
        return result

    async def load_monitor_proposal(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> MonitorProposal | None:
        """Load one inert proposal in its authoritative agent scope."""

        _require_identity(agent_id, "agent_id")
        _require_identity(proposal_id, "proposal_id")
        return await self._run_connection(
            lambda connection: _load_monitor_proposal(
                connection,
                agent_id,
                proposal_id,
            )
        )

    async def load_monitor_confirmation(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> MonitorConfirmation | None:
        """Load the terminal confirmation for one proposal, if present."""

        _require_identity(agent_id, "agent_id")
        _require_identity(proposal_id, "proposal_id")
        return await self._run_connection(
            lambda connection: _load_monitor_confirmation(
                connection,
                agent_id,
                proposal_id,
            )
        )

    async def list_monitor_proposals(
        self,
        agent_id: str,
        *,
        limit: int,
    ) -> tuple[MonitorProposal, ...]:
        """List bounded proposal history in deterministic creation order."""

        _require_identity(agent_id, "agent_id")
        _require_bounded_limit(limit, maximum=200)
        return await self._run_connection(
            lambda connection: _list_monitor_proposals(
                connection,
                agent_id,
                limit=limit,
            )
        )

    async def commit_monitor_confirmation(
        self,
        commit: MonitorConfirmationCommit,
    ) -> MonitorInspection | None:
        """Resolve a proposal and optionally activate version one atomically."""

        if not isinstance(commit, MonitorConfirmationCommit):
            raise TypeError("commit must be a MonitorConfirmationCommit")
        result, committed_events = await self._run_connection(
            lambda connection: _commit_monitor_confirmation(connection, commit)
        )
        self._publish_committed_event_wake_hints(committed_events)
        return result

    async def inspect_monitor(
        self,
        agent_id: str,
        monitor_id: str,
    ) -> MonitorInspection | None:
        """Load one monitor's complete durable lifecycle projection."""

        _require_identity(agent_id, "agent_id")
        _require_identity(monitor_id, "monitor_id")
        return await self._run_connection(
            lambda connection: _inspect_monitor(connection, agent_id, monitor_id)
        )

    async def list_monitors(
        self,
        agent_id: str,
        *,
        statuses: tuple[MonitorStatus, ...],
        limit: int,
    ) -> tuple[Monitor, ...]:
        """List current monitor projections under explicit status filters."""

        _require_identity(agent_id, "agent_id")
        if not statuses or any(
            not isinstance(status, MonitorStatus) for status in statuses
        ):
            raise ValueError("statuses must contain MonitorStatus values")
        if len(statuses) != len(set(statuses)):
            raise ValueError("statuses cannot contain duplicates")
        _require_bounded_limit(limit, maximum=200)
        return await self._run_connection(
            lambda connection: _list_monitors(
                connection,
                agent_id,
                statuses=statuses,
                limit=limit,
            )
        )

    async def commit_monitor_lifecycle(
        self,
        commit: MonitorLifecycleCommit,
        *,
        expected_revision: int,
    ) -> MonitorInspection:
        """Commit one versioned monitor transition under an exact CAS guard."""

        if not isinstance(commit, MonitorLifecycleCommit):
            raise TypeError("commit must be a MonitorLifecycleCommit")
        _require_revision(expected_revision)
        result, committed_events = await self._run_connection(
            lambda connection: _commit_monitor_lifecycle(
                connection,
                commit,
                expected_revision=expected_revision,
            )
        )
        self._publish_committed_event_wake_hints(committed_events)
        return result

    async def list_due_monitors(
        self,
        agent_id: str,
        *,
        now: datetime,
        limit: int,
    ) -> tuple[MonitorInspection, ...]:
        """Load enabled due monitors after durable cooldown/backoff gates."""

        _require_identity(agent_id, "agent_id")
        _require_monitor_datetime(now, "now")
        _require_bounded_limit(limit, maximum=200)
        return await self._run_connection(
            lambda connection: _list_due_monitors(
                connection,
                agent_id,
                now=now,
                limit=limit,
            )
        )

    async def claim_monitor_occurrence(
        self,
        claim: MonitorOccurrenceClaim,
        *,
        expected_monitor_revision: int,
        expected_schedule_revision: int,
        checked_at: datetime,
    ) -> MonitorClaimResult:
        """Create or reclaim one stable occurrence under a durable fence."""

        if not isinstance(claim, MonitorOccurrenceClaim):
            raise TypeError("claim must be a MonitorOccurrenceClaim")
        _require_revision(expected_monitor_revision)
        _require_revision(expected_schedule_revision)
        _require_monitor_datetime(checked_at, "checked_at")
        result, committed_events = await self._run_connection(
            lambda connection: _claim_monitor_occurrence(
                connection,
                claim,
                expected_monitor_revision=expected_monitor_revision,
                expected_schedule_revision=expected_schedule_revision,
                checked_at=checked_at,
            )
        )
        self._publish_committed_event_wake_hints(committed_events)
        return result

    async def load_monitor_claim_by_manual_key(
        self,
        agent_id: str,
        monitor_id: str,
        manual_key: str,
    ) -> MonitorClaimResult | None:
        """Load the authoritative run-now occurrence, run, and latest lease."""

        _require_identity(agent_id, "agent_id")
        _require_identity(monitor_id, "monitor_id")
        _require_identity(manual_key, "manual_key")
        return await self._run_connection(
            lambda connection: _load_monitor_claim_by_manual_key(
                connection,
                agent_id,
                monitor_id,
                manual_key,
            )
        )

    async def load_occurrence_by_trigger(
        self,
        agent_id: str,
        trigger_id: str,
    ) -> MonitorOccurrence | None:
        """Resolve a stable monitor occurrence from its ordinary trigger ID."""

        _require_identity(agent_id, "agent_id")
        _require_identity(trigger_id, "trigger_id")
        return await self._run_connection(
            lambda connection: _load_monitor_occurrence_by_trigger(
                connection,
                agent_id,
                trigger_id,
            )
        )

    async def commit_monitor_outcome(
        self,
        commit: MonitorOutcomeCommit,
        *,
        expected_monitor_revision: int,
        expected_schedule_revision: int,
        checked_at: datetime,
    ) -> MonitorOutcomeResult:
        """Commit a fenced run outcome, checkpoint, finding, and events."""

        if not isinstance(commit, MonitorOutcomeCommit):
            raise TypeError("commit must be a MonitorOutcomeCommit")
        _require_revision(expected_monitor_revision)
        _require_revision(expected_schedule_revision)
        _require_monitor_datetime(checked_at, "checked_at")
        result, committed_events = await self._run_connection(
            lambda connection: _commit_monitor_outcome(
                connection,
                commit,
                expected_monitor_revision=expected_monitor_revision,
                expected_schedule_revision=expected_schedule_revision,
                checked_at=checked_at,
            )
        )
        self._publish_committed_event_wake_hints(committed_events)
        return result

    async def register_source(
        self,
        registration: SourceRegistration,
    ) -> SourceRegistration:
        """Persist one immutable source attachment idempotently."""

        if not isinstance(registration, SourceRegistration):
            raise TypeError("registration must be a SourceRegistration record")
        return await self._run_connection(
            lambda connection: _register_source(connection, registration)
        )

    async def load_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration | None:
        """Load one agent-scoped source registration."""

        _require_identity(agent_id, "agent_id")
        _require_identity(source_id, "source_id")
        return await self._run_connection(
            lambda connection: _load_source(connection, agent_id, source_id)
        )

    async def list_sources(
        self,
        agent_id: str,
    ) -> tuple[SourceRegistration, ...]:
        """List source registrations in deterministic attachment order."""

        _require_identity(agent_id, "agent_id")
        return await self._run_connection(
            lambda connection: _list_sources(connection, agent_id)
        )

    async def detach_source(
        self,
        agent_id: str,
        source_id: str,
        detached_at: datetime,
    ) -> SourceRegistration:
        """Persist the one-way detach transition for a registered source."""

        _require_identity(agent_id, "agent_id")
        _require_identity(source_id, "source_id")
        if (
            not isinstance(detached_at, datetime)
            or detached_at.tzinfo is None
            or detached_at.utcoffset() is None
        ):
            raise ValueError("detached_at must be timezone-aware")
        return await self._run_connection(
            lambda connection: _detach_source(
                connection,
                agent_id,
                source_id,
                detached_at,
            )
        )

    async def record_sync(self, sync: CatalogSync) -> CatalogSync:
        """Persist one catalog sync lifecycle transition."""

        if not isinstance(sync, CatalogSync):
            raise TypeError("sync must be a CatalogSync record")
        return await self._run_connection(
            lambda connection: _record_catalog_sync(connection, sync)
        )

    async def commit_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
    ) -> SourceCatalogSnapshot:
        """Atomically replace one source projection and retain revision history."""

        if not isinstance(snapshot, SourceCatalogSnapshot):
            raise TypeError("snapshot must be a SourceCatalogSnapshot record")
        return await self._run_connection(
            lambda connection: _commit_catalog_snapshot(connection, snapshot)
        )

    async def load_sync(
        self,
        agent_id: str,
        sync_id: str,
    ) -> CatalogSync | None:
        """Load one agent-scoped catalog sync."""

        _require_identity(agent_id, "agent_id")
        _require_identity(sync_id, "sync_id")
        return await self._run_connection(
            lambda connection: _load_catalog_sync(connection, agent_id, sync_id)
        )

    async def load_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> CatalogResource | None:
        """Load one current agent-scoped catalog resource."""

        _require_identity(agent_id, "agent_id")
        _require_identity(resource_id, "resource_id")
        return await self._run_connection(
            lambda connection: _load_catalog_resource(
                connection,
                agent_id,
                resource_id,
            )
        )

    async def list_resources(
        self,
        agent_id: str,
        source_id: str | None = None,
    ) -> tuple[CatalogResource, ...]:
        """List the deterministic current resource projection for one agent."""

        _require_identity(agent_id, "agent_id")
        if source_id is not None:
            _require_identity(source_id, "source_id")
        return await self._run_connection(
            lambda connection: _list_catalog_resources(
                connection,
                agent_id,
                source_id,
            )
        )

    async def load_revision(
        self,
        agent_id: str,
        resource_id: str,
        revision: str,
    ) -> CatalogResourceRevision | None:
        """Load a deterministic observation of one structural revision."""

        _require_identity(agent_id, "agent_id")
        _require_identity(resource_id, "resource_id")
        _require_identity(revision, "revision")
        return await self._run_connection(
            lambda connection: _load_catalog_revision(
                connection,
                agent_id,
                resource_id,
                revision,
            )
        )

    async def load_facets(
        self,
        agent_id: str,
        resource_id: str,
        revision: str | None = None,
    ) -> tuple[CatalogFacet, ...]:
        """Load facets for one resource's current persisted projection."""

        _require_identity(agent_id, "agent_id")
        _require_identity(resource_id, "resource_id")
        if revision is not None:
            _require_identity(revision, "revision")
        return await self._run_connection(
            lambda connection: _load_current_catalog_facets(
                connection,
                agent_id,
                resource_id,
                revision,
            )
        )

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        """Run one bounded literal-token FTS search over current resources."""

        if not isinstance(request, CatalogSearchRequest):
            raise TypeError("request must be a CatalogSearchRequest record")
        return await self._run_connection(
            lambda connection: _search_catalog(connection, request)
        )

    async def traverse(
        self,
        request: CatalogTraversalRequest,
    ) -> CatalogTraversalResult:
        """Traverse current catalog relationships under explicit hard bounds."""

        if not isinstance(request, CatalogTraversalRequest):
            raise TypeError("request must be a CatalogTraversalRequest record")
        return await self._run_connection(
            lambda connection: _traverse_catalog(connection, request)
        )

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

    async def load_nonterminal(
        self,
        agent_id: str,
    ) -> tuple[VersionedOperation, ...]:
        """Load one agent's exact resumable checkpoints in stable order."""

        _require_identity(agent_id, "agent_id")
        return await self._run_connection(
            lambda connection: _load_nonterminal_operations(connection, agent_id)
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

    async def load_by_approval(
        self,
        approval_id: str,
    ) -> VersionedOperation | None:
        """Load the operation that owns one globally unique approval identity."""

        _require_identity(approval_id, "approval_id")
        return await self._run_connection(
            lambda connection: _load_versioned_by_approval(connection, approval_id)
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

    async def latest_cursor(self, agent_id: str) -> EventCursor | None:
        """Return the current durable tail without replaying event history."""

        _require_identity(agent_id, "agent_id")
        return await self._run_connection(
            lambda connection: _latest_committed_event_cursor(
                connection,
                agent_id,
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


def _initialize_agent_identity(
    connection: sqlite3.Connection,
    identity: AgentIdentity,
) -> AgentIdentity:
    connection.execute("BEGIN IMMEDIATE")
    try:
        current = _load_agent_identity(connection)
        if current is None:
            connection.execute(
                "INSERT INTO agents(singleton, id, display_name, created_at, "
                "state_schema_generation) VALUES (1, ?, ?, ?, ?)",
                (
                    identity.id,
                    identity.display_name,
                    _encode_datetime(identity.created_at),
                    identity.state_schema_generation,
                ),
            )
            current = identity
        elif current != identity:
            raise AgentIdentityConflictError(
                "SQLite database already belongs to a different agent identity"
            )
        connection.execute("COMMIT")
        return current
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_agent_identity(connection: sqlite3.Connection) -> AgentIdentity | None:
    rows = connection.execute(
        "SELECT id, display_name, created_at, state_schema_generation "
        "FROM agents ORDER BY singleton"
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise SQLiteCorruptionError("agent database must contain exactly one identity")
    row = rows[0]
    try:
        return AgentIdentity(
            id=_sqlite_text(row["id"], "agent id"),
            display_name=_sqlite_text(row["display_name"], "agent display name"),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "agent created_at")
            ),
            state_schema_generation=_sqlite_int(
                row["state_schema_generation"],
                "agent state schema generation",
            ),
        )
    except (TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            "cannot reconstruct authoritative agent identity"
        ) from error


def _bind_agent_runtime_defaults(
    connection: sqlite3.Connection,
    agent_id: str,
    defaults: AgentRuntimeDefaults,
    *,
    bound_at: datetime,
) -> AgentRuntimeDefaults:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != agent_id:
            raise AgentRuntimeDefaultsConflictError(
                "runtime defaults belong to another or uninitialized agent"
            )
        current = _load_agent_runtime_defaults(connection, agent_id)
        if current is None:
            budgets = defaults.budgets
            policy = defaults.policy_profile
            connection.execute(
                "INSERT INTO agent_runtime_defaults("
                "agent_id, schema_version, revision, fingerprint, "
                "budget_max_turns, budget_max_actions, budget_max_repairs, "
                "budget_max_identical_failures, "
                "budget_max_observation_characters, budget_max_total_tokens, "
                "budget_max_wall_time_seconds, budget_task_timeout_seconds, "
                "budget_max_estimated_cost_usd, policy_id, policy_version, "
                "policy_allow_destructive, bound_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    defaults.schema_version,
                    defaults.revision,
                    defaults.fingerprint,
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
                    policy.id,
                    policy.version,
                    int(policy.allow_destructive),
                    _encode_datetime(bound_at),
                ),
            )
            current = defaults
        elif current != defaults:
            raise AgentRuntimeDefaultsConflictError(
                "agent is already bound to different runtime defaults"
            )
        connection.execute("COMMIT")
        return current
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_agent_runtime_defaults(
    connection: sqlite3.Connection,
    agent_id: str,
) -> AgentRuntimeDefaults | None:
    rows = connection.execute(
        "SELECT schema_version, revision, fingerprint, budget_max_turns, "
        "budget_max_actions, budget_max_repairs, "
        "budget_max_identical_failures, budget_max_observation_characters, "
        "budget_max_total_tokens, budget_max_wall_time_seconds, "
        "budget_task_timeout_seconds, budget_max_estimated_cost_usd, "
        "policy_id, policy_version, policy_allow_destructive, bound_at "
        "FROM agent_runtime_defaults WHERE agent_id = ?",
        (agent_id,),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise SQLiteCorruptionError(
            "agent database must contain at most one runtime-default binding"
        )
    row = rows[0]
    try:
        cost = row["budget_max_estimated_cost_usd"]
        defaults = AgentRuntimeDefaults(
            schema_version=_sqlite_int(
                row["schema_version"],
                "runtime-default schema_version",
            ),
            revision=_sqlite_int(row["revision"], "runtime-default revision"),
            budgets=LoopBudgets(
                max_turns=_sqlite_int(
                    row["budget_max_turns"],
                    "runtime-default max_turns",
                ),
                max_actions=_sqlite_int(
                    row["budget_max_actions"],
                    "runtime-default max_actions",
                ),
                max_repairs=_sqlite_int(
                    row["budget_max_repairs"],
                    "runtime-default max_repairs",
                ),
                max_identical_failures=_sqlite_int(
                    row["budget_max_identical_failures"],
                    "runtime-default max_identical_failures",
                ),
                max_observation_characters=_sqlite_int(
                    row["budget_max_observation_characters"],
                    "runtime-default max_observation_characters",
                ),
                max_total_tokens=_sqlite_int(
                    row["budget_max_total_tokens"],
                    "runtime-default max_total_tokens",
                ),
                max_wall_time_seconds=_sqlite_real(
                    row["budget_max_wall_time_seconds"],
                    "runtime-default max_wall_time_seconds",
                ),
                task_timeout_seconds=_sqlite_real(
                    row["budget_task_timeout_seconds"],
                    "runtime-default task_timeout_seconds",
                ),
                max_estimated_cost_usd=(
                    None if cost is None else _decode_decimal(cost)
                ),
            ),
            policy_profile=DefaultPolicyProfile(
                id=_sqlite_text(row["policy_id"], "runtime-default policy_id"),
                version=_sqlite_text(
                    row["policy_version"],
                    "runtime-default policy_version",
                ),
                allow_destructive=bool(
                    _sqlite_int(
                        row["policy_allow_destructive"],
                        "runtime-default policy_allow_destructive",
                    )
                ),
            ),
        )
        fingerprint = _sqlite_text(
            row["fingerprint"],
            "runtime-default fingerprint",
        )
        _decode_datetime(_sqlite_text(row["bound_at"], "runtime-default bound_at"))
        if fingerprint != defaults.fingerprint:
            raise ValueError("runtime-default fingerprint does not match its fields")
        return defaults
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            "cannot reconstruct authoritative agent runtime defaults"
        ) from error


def _bind_agent_model_profile(
    connection: sqlite3.Connection,
    agent_id: str,
    profile: ModelProfile,
) -> ModelProfile:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != agent_id:
            raise ModelProfileConflictError(
                "model profile belongs to another or uninitialized agent"
            )
        current = _load_agent_model_profile(connection, agent_id)
        if current is None:
            connection.execute(
                "INSERT INTO agent_model_profiles("
                "agent_id, profile_id, context_window_tokens, max_output_tokens, "
                "supports_tools, supports_parallel_tools, "
                "supports_structured_output, supports_streaming, "
                "supports_reasoning, supports_vision, supports_documents, "
                "supports_prompt_caching, supports_native_continuation, "
                "input_cost_per_million_usd, output_cost_per_million_usd, "
                "data_routing_classification, available, healthy"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    profile.id,
                    profile.context_window_tokens,
                    profile.max_output_tokens,
                    int(profile.supports_tools),
                    int(profile.supports_parallel_tools),
                    int(profile.supports_structured_output),
                    int(profile.supports_streaming),
                    int(profile.supports_reasoning),
                    int(profile.supports_vision),
                    int(profile.supports_documents),
                    int(profile.supports_prompt_caching),
                    int(profile.supports_native_continuation),
                    (
                        None
                        if profile.input_cost_per_million_usd is None
                        else _encode_decimal(profile.input_cost_per_million_usd)
                    ),
                    (
                        None
                        if profile.output_cost_per_million_usd is None
                        else _encode_decimal(profile.output_cost_per_million_usd)
                    ),
                    profile.data_routing_classification,
                    int(profile.available),
                    int(profile.healthy),
                ),
            )
            current = profile
        elif current != profile:
            raise ModelProfileConflictError(
                "agent is already bound to a different model profile"
            )
        connection.execute("COMMIT")
        return current
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_agent_model_profile(
    connection: sqlite3.Connection,
    agent_id: str,
) -> ModelProfile | None:
    rows = connection.execute(
        "SELECT profile_id, context_window_tokens, max_output_tokens, "
        "supports_tools, supports_parallel_tools, supports_structured_output, "
        "supports_streaming, supports_reasoning, supports_vision, "
        "supports_documents, supports_prompt_caching, "
        "supports_native_continuation, input_cost_per_million_usd, "
        "output_cost_per_million_usd, data_routing_classification, available, "
        "healthy FROM agent_model_profiles WHERE agent_id = ?",
        (agent_id,),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise SQLiteCorruptionError(
            "agent database must contain at most one model-profile binding"
        )
    row = rows[0]
    try:
        input_cost = row["input_cost_per_million_usd"]
        output_cost = row["output_cost_per_million_usd"]
        return ModelProfile(
            id=_sqlite_text(row["profile_id"], "model-profile id"),
            context_window_tokens=_sqlite_int(
                row["context_window_tokens"],
                "model-profile context_window_tokens",
            ),
            max_output_tokens=_sqlite_int(
                row["max_output_tokens"],
                "model-profile max_output_tokens",
            ),
            supports_tools=_decode_bool(row["supports_tools"]),
            supports_parallel_tools=_decode_bool(row["supports_parallel_tools"]),
            supports_structured_output=_decode_bool(row["supports_structured_output"]),
            supports_streaming=_decode_bool(row["supports_streaming"]),
            supports_reasoning=_decode_bool(row["supports_reasoning"]),
            supports_vision=_decode_bool(row["supports_vision"]),
            supports_documents=_decode_bool(row["supports_documents"]),
            supports_prompt_caching=_decode_bool(row["supports_prompt_caching"]),
            supports_native_continuation=_decode_bool(
                row["supports_native_continuation"]
            ),
            input_cost_per_million_usd=(
                None if input_cost is None else _decode_decimal(input_cost)
            ),
            output_cost_per_million_usd=(
                None if output_cost is None else _decode_decimal(output_cost)
            ),
            data_routing_classification=_sqlite_text(
                row["data_routing_classification"],
                "model-profile data_routing_classification",
            ),
            available=_decode_bool(row["available"]),
            healthy=_decode_bool(row["healthy"]),
        )
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            "cannot reconstruct authoritative agent model profile"
        ) from error


def _decode_source_registration_row(row: sqlite3.Row) -> SourceRegistration:
    try:
        return SourceRegistration(
            id=_sqlite_text(row["id"], "source registration id"),
            agent_id=_sqlite_text(row["agent_id"], "source registration agent_id"),
            adapter_id=_sqlite_text(
                row["adapter_id"],
                "source registration adapter_id",
            ),
            native_identity=_sqlite_text(
                row["native_identity"],
                "source registration native_identity",
            ),
            display_name=_sqlite_text(
                row["display_name"],
                "source registration display_name",
            ),
            configuration=_decode_json_object(
                _sqlite_text(
                    row["configuration_json"],
                    "source registration configuration",
                )
            ),
            attached_at=_decode_datetime(
                _sqlite_text(row["attached_at"], "source registration attached_at")
            ),
            detached_at=_decode_optional_datetime(
                row["detached_at"],
                "source registration detached_at",
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct source registration") from error


_SOURCE_REGISTRATION_SELECT = (
    "SELECT id, agent_id, adapter_id, native_identity, display_name, "
    "configuration_json, attached_at, detached_at FROM attached_sources"
)


def _load_source_any_agent(
    connection: sqlite3.Connection,
    source_id: str,
) -> SourceRegistration | None:
    row = connection.execute(
        _SOURCE_REGISTRATION_SELECT + " WHERE id = ?",
        (source_id,),
    ).fetchone()
    return None if row is None else _decode_source_registration_row(row)


def _load_source(
    connection: sqlite3.Connection,
    agent_id: str,
    source_id: str,
) -> SourceRegistration | None:
    registration = _load_source_any_agent(connection, source_id)
    if registration is None or registration.agent_id != agent_id:
        return None
    return registration


def _list_sources(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[SourceRegistration, ...]:
    rows = connection.execute(
        _SOURCE_REGISTRATION_SELECT + " WHERE agent_id = ? ORDER BY attached_at, id",
        (agent_id,),
    ).fetchall()
    return tuple(_decode_source_registration_row(row) for row in rows)


def _register_source(
    connection: sqlite3.Connection,
    registration: SourceRegistration,
) -> SourceRegistration:
    if registration.detached_at is not None:
        raise ValueError("cannot register a source that is already detached")
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != registration.agent_id:
            raise AgentIdentityConflictError(
                "source agent does not match the authoritative database identity"
            )
        current = _load_source_any_agent(connection, registration.id)
        if current == registration:
            connection.execute("COMMIT")
            return registration
        if current is not None:
            raise SQLiteStoreError(f"source registration conflict: {registration.id}")
        try:
            connection.execute(
                "INSERT INTO attached_sources(id, agent_id, adapter_id, "
                "native_identity, display_name, configuration_json, attached_at, "
                "detached_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    registration.id,
                    registration.agent_id,
                    registration.adapter_id,
                    registration.native_identity,
                    registration.display_name,
                    canonical_json(registration.configuration),
                    _encode_datetime(registration.attached_at),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise SQLiteStoreError(
                f"source registration conflict: {registration.id}"
            ) from error
        connection.execute("COMMIT")
        return registration
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _detach_source(
    connection: sqlite3.Connection,
    agent_id: str,
    source_id: str,
    detached_at: datetime,
) -> SourceRegistration:
    connection.execute("BEGIN IMMEDIATE")
    try:
        current = _load_source(connection, agent_id, source_id)
        if current is None:
            raise SQLiteStoreError(f"unknown source registration: {source_id}")
        if current.detached_at is not None:
            raise SQLiteStoreError(f"source is already detached: {source_id}")
        detached = current.detach(detached_at)
        cursor = connection.execute(
            "UPDATE attached_sources SET detached_at = ? "
            "WHERE id = ? AND agent_id = ? AND detached_at IS NULL",
            (_encode_datetime(detached_at), source_id, agent_id),
        )
        if cursor.rowcount != 1:
            raise SQLiteStoreError(f"source detach conflict: {source_id}")
        connection.execute("COMMIT")
        return detached
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _require_catalog_agent(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    source_id: str,
    sync_id: str,
) -> None:
    identity = _load_agent_identity(connection)
    if identity is None or identity.id != agent_id:
        raise CatalogSyncConflictError(
            source_id,
            sync_id,
            "agent_identity_mismatch",
        )


def _catalog_sync_values(sync: CatalogSync) -> tuple[object, ...]:
    return (
        sync.id,
        sync.agent_id,
        sync.source_id,
        sync.adapter_id,
        sync.status.value,
        _encode_datetime(sync.started_at),
        _encode_optional_datetime(sync.completed_at),
        sync.source_revision,
        sync.resource_count,
        sync.relationship_count,
        sync.error_code,
    )


def _decode_catalog_sync_row(row: sqlite3.Row) -> CatalogSync:
    try:
        return CatalogSync(
            id=_sqlite_text(row["id"], "catalog sync id"),
            agent_id=_sqlite_text(row["agent_id"], "catalog sync agent_id"),
            source_id=_sqlite_text(row["source_id"], "catalog sync source_id"),
            adapter_id=_sqlite_text(row["adapter_id"], "catalog sync adapter_id"),
            status=CatalogSyncStatus(
                _sqlite_text(row["status"], "catalog sync status")
            ),
            started_at=_decode_datetime(
                _sqlite_text(row["started_at"], "catalog sync started_at")
            ),
            completed_at=_decode_optional_datetime(
                row["completed_at"],
                "catalog sync completed_at",
            ),
            source_revision=_optional_text(row["source_revision"]),
            resource_count=_sqlite_int(
                row["resource_count"],
                "catalog sync resource_count",
            ),
            relationship_count=_sqlite_int(
                row["relationship_count"],
                "catalog sync relationship_count",
            ),
            error_code=_optional_text(row["error_code"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct catalog sync") from error


def _load_catalog_sync_any_agent(
    connection: sqlite3.Connection,
    sync_id: str,
) -> CatalogSync | None:
    row = connection.execute(
        "SELECT id, agent_id, source_id, adapter_id, status, started_at, "
        "completed_at, source_revision, resource_count, relationship_count, "
        "error_code FROM catalog_syncs WHERE id = ?",
        (sync_id,),
    ).fetchone()
    return None if row is None else _decode_catalog_sync_row(row)


def _load_catalog_sync(
    connection: sqlite3.Connection,
    agent_id: str,
    sync_id: str,
) -> CatalogSync | None:
    sync = _load_catalog_sync_any_agent(connection, sync_id)
    if sync is None or sync.agent_id != agent_id:
        return None
    return sync


def _catalog_sync_identity_matches(left: CatalogSync, right: CatalogSync) -> bool:
    return (
        left.id == right.id
        and left.agent_id == right.agent_id
        and left.source_id == right.source_id
        and left.adapter_id == right.adapter_id
        and left.started_at == right.started_at
    )


def _insert_catalog_sync(connection: sqlite3.Connection, sync: CatalogSync) -> None:
    connection.execute(
        "INSERT INTO catalog_syncs(id, agent_id, source_id, adapter_id, status, "
        "started_at, completed_at, source_revision, resource_count, "
        "relationship_count, error_code) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _catalog_sync_values(sync),
    )


def _update_catalog_sync(connection: sqlite3.Connection, sync: CatalogSync) -> None:
    connection.execute(
        "UPDATE catalog_syncs SET status = ?, completed_at = ?, "
        "source_revision = ?, resource_count = ?, relationship_count = ?, "
        "error_code = ? WHERE id = ?",
        (
            sync.status.value,
            _encode_optional_datetime(sync.completed_at),
            sync.source_revision,
            sync.resource_count,
            sync.relationship_count,
            sync.error_code,
            sync.id,
        ),
    )


def _record_catalog_sync(
    connection: sqlite3.Connection,
    sync: CatalogSync,
) -> CatalogSync:
    connection.execute("BEGIN IMMEDIATE")
    try:
        _require_catalog_agent(
            connection,
            agent_id=sync.agent_id,
            source_id=sync.source_id,
            sync_id=sync.id,
        )
        current = _load_catalog_sync_any_agent(connection, sync.id)
        if current == sync:
            connection.execute("COMMIT")
            return sync
        if sync.status is CatalogSyncStatus.SUCCEEDED:
            raise CatalogSyncConflictError(
                sync.source_id,
                sync.id,
                "successful_sync_requires_snapshot",
            )
        if current is None:
            _insert_catalog_sync(connection, sync)
        else:
            if not _catalog_sync_identity_matches(current, sync):
                raise CatalogSyncConflictError(
                    sync.source_id,
                    sync.id,
                    "identity_changed",
                )
            if current.status is not CatalogSyncStatus.RUNNING:
                raise CatalogSyncConflictError(
                    sync.source_id,
                    sync.id,
                    "already_terminal",
                )
            if sync.status is CatalogSyncStatus.RUNNING:
                raise CatalogSyncConflictError(
                    sync.source_id,
                    sync.id,
                    "running_sync_changed",
                )
            _update_catalog_sync(connection, sync)
        connection.execute("COMMIT")
        return sync
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _decode_catalog_resource_row(row: sqlite3.Row) -> CatalogResource:
    try:
        return CatalogResource(
            id=_sqlite_text(row["id"], "catalog resource id"),
            agent_id=_sqlite_text(row["agent_id"], "catalog resource agent_id"),
            source_id=_sqlite_text(row["source_id"], "catalog resource source_id"),
            native_identity=_sqlite_text(
                row["native_identity"],
                "catalog resource native_identity",
            ),
            external_uri=_sqlite_text(
                row["external_uri"],
                "catalog resource external_uri",
            ),
            kind=ResourceKind(_sqlite_text(row["kind"], "catalog resource kind")),
            name=_sqlite_text(row["name"], "catalog resource name"),
            sensitivity=Sensitivity(
                _sqlite_text(row["sensitivity"], "catalog resource sensitivity")
            ),
            current_revision=_sqlite_text(
                row["current_revision"],
                "catalog resource current_revision",
            ),
            current_sync_id=_sqlite_text(
                row["current_sync_id"],
                "catalog resource current_sync_id",
            ),
            first_observed_at=_decode_datetime(
                _sqlite_text(
                    row["first_observed_at"],
                    "catalog resource first_observed_at",
                )
            ),
            last_observed_at=_decode_datetime(
                _sqlite_text(
                    row["last_observed_at"],
                    "catalog resource last_observed_at",
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct catalog resource") from error


_CATALOG_RESOURCE_SELECT = (
    "SELECT id, agent_id, source_id, native_identity, external_uri, kind, name, "
    "sensitivity, current_revision, current_sync_id, first_observed_at, "
    "last_observed_at FROM catalog_resources"
)


def _load_catalog_resource(
    connection: sqlite3.Connection,
    agent_id: str,
    resource_id: str,
) -> CatalogResource | None:
    row = connection.execute(
        _CATALOG_RESOURCE_SELECT + " WHERE agent_id = ? AND id = ?",
        (agent_id, resource_id),
    ).fetchone()
    return None if row is None else _decode_catalog_resource_row(row)


def _list_catalog_resources(
    connection: sqlite3.Connection,
    agent_id: str,
    source_id: str | None,
) -> tuple[CatalogResource, ...]:
    if source_id is None:
        rows = connection.execute(
            _CATALOG_RESOURCE_SELECT
            + " WHERE agent_id = ? ORDER BY source_id, kind, name, id",
            (agent_id,),
        ).fetchall()
    else:
        rows = connection.execute(
            _CATALOG_RESOURCE_SELECT
            + " WHERE agent_id = ? AND source_id = ? ORDER BY kind, name, id",
            (agent_id, source_id),
        ).fetchall()
    return tuple(_decode_catalog_resource_row(row) for row in rows)


def _decode_catalog_revision_row(row: sqlite3.Row) -> CatalogResourceRevision:
    try:
        return CatalogResourceRevision(
            resource_id=_sqlite_text(
                row["resource_id"],
                "catalog revision resource_id",
            ),
            revision=_sqlite_text(row["revision"], "catalog revision"),
            sync_id=_sqlite_text(row["sync_id"], "catalog revision sync_id"),
            observed_at=_decode_datetime(
                _sqlite_text(row["observed_at"], "catalog revision observed_at")
            ),
            facet_revisions=_decode_string_tuple(
                _sqlite_text(
                    row["facet_revisions_json"],
                    "catalog facet revisions",
                )
            ),
            relationship_revisions=_decode_string_tuple(
                _sqlite_text(
                    row["relationship_revisions_json"],
                    "catalog relationship revisions",
                )
            ),
            source_revision=_optional_text(row["source_revision"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct catalog revision") from error


def _load_catalog_revision(
    connection: sqlite3.Connection,
    agent_id: str,
    resource_id: str,
    revision: str,
) -> CatalogResourceRevision | None:
    row = connection.execute(
        "SELECT revision_row.resource_id, revision_row.revision, "
        "revision_row.sync_id, revision_row.observed_at, "
        "revision_row.facet_revisions_json, "
        "revision_row.relationship_revisions_json, revision_row.source_revision "
        "FROM catalog_resource_revisions AS revision_row "
        "LEFT JOIN catalog_resources AS resource "
        "ON resource.agent_id = revision_row.agent_id "
        "AND resource.id = revision_row.resource_id "
        "WHERE revision_row.agent_id = ? AND revision_row.resource_id = ? "
        "AND revision_row.revision = ? "
        "ORDER BY (revision_row.sync_id = resource.current_sync_id) DESC, "
        "revision_row.observed_at DESC, revision_row.sync_id DESC LIMIT 1",
        (agent_id, resource_id, revision),
    ).fetchone()
    return None if row is None else _decode_catalog_revision_row(row)


def _decode_catalog_facet_row(row: sqlite3.Row) -> CatalogFacet:
    try:
        return CatalogFacet(
            resource_id=_sqlite_text(row["resource_id"], "catalog facet resource_id"),
            sync_id=_sqlite_text(row["sync_id"], "catalog facet sync_id"),
            kind=FacetKind(_sqlite_text(row["kind"], "catalog facet kind")),
            schema_version=_sqlite_int(
                row["schema_version"],
                "catalog facet schema_version",
            ),
            revision=_sqlite_text(row["revision"], "catalog facet revision"),
            payload=_decode_json_object(
                _sqlite_text(row["payload_json"], "catalog facet payload")
            ),
            observed_at=_decode_datetime(
                _sqlite_text(row["observed_at"], "catalog facet observed_at")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct catalog facet") from error


def _load_current_catalog_facets(
    connection: sqlite3.Connection,
    agent_id: str,
    resource_id: str,
    revision: str | None,
) -> tuple[CatalogFacet, ...]:
    resource = _load_catalog_resource(connection, agent_id, resource_id)
    if resource is None:
        return ()
    if revision is not None and revision != resource.current_revision:
        return ()
    current_revision = _load_catalog_revision(
        connection,
        agent_id,
        resource_id,
        resource.current_revision,
    )
    if current_revision is None:
        raise SQLiteCorruptionError(
            "catalog resource references a missing current revision"
        )
    rows = connection.execute(
        "SELECT resource_id, sync_id, kind, schema_version, revision, "
        "payload_json, observed_at FROM catalog_facets WHERE agent_id = ? "
        "AND resource_id = ? AND sync_id = ? ORDER BY kind, revision",
        (agent_id, resource_id, resource.current_sync_id),
    ).fetchall()
    facets = tuple(_decode_catalog_facet_row(row) for row in rows)
    if {facet.revision for facet in facets} != set(current_revision.facet_revisions):
        raise SQLiteCorruptionError(
            "catalog current facets disagree with the resource revision"
        )
    return facets


def _decode_relationship_field_pairs(
    value: object,
) -> tuple[RelationshipFieldPair, ...]:
    if not isinstance(value, list):
        raise ValueError("catalog relationship field pairs must be an array")
    pairs: list[RelationshipFieldPair] = []
    for item in value:
        payload = _expect_object(
            item,
            keys={"ordinal", "source_field", "target_field"},
            label="catalog relationship field pair",
        )
        pairs.append(
            RelationshipFieldPair(
                source_field=_sqlite_text(
                    payload["source_field"],
                    "catalog relationship source_field",
                ),
                target_field=_sqlite_text(
                    payload["target_field"],
                    "catalog relationship target_field",
                ),
                ordinal=_sqlite_int(
                    payload["ordinal"],
                    "catalog relationship field-pair ordinal",
                ),
            )
        )
    return tuple(pairs)


def _decode_catalog_relationship_row(row: sqlite3.Row) -> CatalogRelationship:
    try:
        field_pairs = _decode_relationship_field_pairs(
            _decode_json(
                _sqlite_text(
                    row["field_pairs_json"],
                    "catalog relationship field pairs",
                )
            )
        )
        return CatalogRelationship(
            id=_sqlite_text(row["id"], "catalog relationship id"),
            revision=_sqlite_text(row["revision"], "catalog relationship revision"),
            source_id=_sqlite_text(
                row["source_id"],
                "catalog relationship source_id",
            ),
            from_resource_id=_sqlite_text(
                row["from_resource_id"],
                "catalog relationship from_resource_id",
            ),
            to_resource_id=_sqlite_text(
                row["to_resource_id"],
                "catalog relationship to_resource_id",
            ),
            kind=RelationshipKind(
                _sqlite_text(row["kind"], "catalog relationship kind")
            ),
            provenance=RelationshipProvenance(
                _sqlite_text(
                    row["provenance"],
                    "catalog relationship provenance",
                )
            ),
            confidence=_sqlite_real(
                row["confidence"],
                "catalog relationship confidence",
            ),
            sync_id=_sqlite_text(row["sync_id"], "catalog relationship sync_id"),
            observed_at=_decode_datetime(
                _sqlite_text(
                    row["observed_at"],
                    "catalog relationship observed_at",
                )
            ),
            field_pairs=field_pairs,
            attributes=_decode_json_object(
                _sqlite_text(
                    row["attributes_json"],
                    "catalog relationship attributes",
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            "cannot reconstruct catalog relationship"
        ) from error


_CATALOG_RELATIONSHIP_SELECT = (
    "SELECT id, revision, source_id, from_resource_id, to_resource_id, kind, "
    "provenance, confidence, sync_id, observed_at, field_pairs_json, "
    "attributes_json FROM catalog_relationships"
)


def _load_catalog_relationships_for_sync(
    connection: sqlite3.Connection,
    agent_id: str,
    sync_id: str,
) -> tuple[CatalogRelationship, ...]:
    rows = connection.execute(
        _CATALOG_RELATIONSHIP_SELECT
        + " WHERE agent_id = ? AND sync_id = ? ORDER BY id",
        (agent_id, sync_id),
    ).fetchall()
    return tuple(_decode_catalog_relationship_row(row) for row in rows)


def _catalog_snapshot_matches_current(
    connection: sqlite3.Connection,
    snapshot: SourceCatalogSnapshot,
) -> bool:
    resources = _list_catalog_resources(
        connection,
        snapshot.sync.agent_id,
        snapshot.sync.source_id,
    )
    if resources != tuple(
        sorted(
            snapshot.resources, key=lambda item: (item.kind.value, item.name, item.id)
        )
    ):
        return False
    revisions: list[CatalogResourceRevision] = []
    facets: list[CatalogFacet] = []
    for resource in resources:
        revision = _load_catalog_revision(
            connection,
            snapshot.sync.agent_id,
            resource.id,
            resource.current_revision,
        )
        if revision is None:
            return False
        revisions.append(revision)
        facets.extend(
            _load_current_catalog_facets(
                connection,
                snapshot.sync.agent_id,
                resource.id,
                resource.current_revision,
            )
        )
    relationships = _load_catalog_relationships_for_sync(
        connection,
        snapshot.sync.agent_id,
        snapshot.sync.id,
    )
    return (
        tuple(sorted(revisions, key=lambda item: item.resource_id))
        == tuple(sorted(snapshot.revisions, key=lambda item: item.resource_id))
        and tuple(
            sorted(
                facets,
                key=lambda item: (item.resource_id, item.kind.value, item.revision),
            )
        )
        == tuple(
            sorted(
                snapshot.facets,
                key=lambda item: (item.resource_id, item.kind.value, item.revision),
            )
        )
        and relationships
        == tuple(sorted(snapshot.relationships, key=lambda item: item.id))
    )


def _catalog_snapshot_is_stale(
    connection: sqlite3.Connection,
    sync: CatalogSync,
) -> bool:
    row = connection.execute(
        "SELECT started_at, completed_at, id FROM catalog_syncs "
        "WHERE agent_id = ? AND source_id = ? AND status = ? AND id != ? "
        "ORDER BY completed_at DESC, started_at DESC, id DESC LIMIT 1",
        (
            sync.agent_id,
            sync.source_id,
            CatalogSyncStatus.SUCCEEDED.value,
            sync.id,
        ),
    ).fetchone()
    if row is None:
        return False
    latest_completed = _decode_datetime(
        _sqlite_text(row["completed_at"], "latest catalog completed_at")
    )
    latest_started = _decode_datetime(
        _sqlite_text(row["started_at"], "latest catalog started_at")
    )
    assert sync.completed_at is not None
    return (latest_completed, latest_started, _sqlite_text(row["id"], "sync id")) > (
        sync.completed_at,
        sync.started_at,
        sync.id,
    )


def _insert_catalog_revision(
    connection: sqlite3.Connection,
    agent_id: str,
    revision: CatalogResourceRevision,
) -> None:
    connection.execute(
        "INSERT INTO catalog_resource_revisions(agent_id, resource_id, revision, "
        "sync_id, observed_at, facet_revisions_json, "
        "relationship_revisions_json, source_revision) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            agent_id,
            revision.resource_id,
            revision.revision,
            revision.sync_id,
            _encode_datetime(revision.observed_at),
            canonical_json(revision.facet_revisions),
            canonical_json(revision.relationship_revisions),
            revision.source_revision,
        ),
    )


def _insert_catalog_facet(
    connection: sqlite3.Connection,
    agent_id: str,
    facet: CatalogFacet,
) -> None:
    connection.execute(
        "INSERT INTO catalog_facets(agent_id, resource_id, sync_id, kind, "
        "schema_version, revision, payload_json, observed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            agent_id,
            facet.resource_id,
            facet.sync_id,
            facet.kind.value,
            facet.schema_version,
            facet.revision,
            canonical_json(facet.payload),
            _encode_datetime(facet.observed_at),
        ),
    )


def _insert_catalog_relationship(
    connection: sqlite3.Connection,
    agent_id: str,
    relationship: CatalogRelationship,
) -> None:
    connection.execute(
        "INSERT INTO catalog_relationships(agent_id, id, revision, source_id, "
        "from_resource_id, to_resource_id, kind, provenance, confidence, "
        "sync_id, observed_at, field_pairs_json, attributes_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            agent_id,
            relationship.id,
            relationship.revision,
            relationship.source_id,
            relationship.from_resource_id,
            relationship.to_resource_id,
            relationship.kind.value,
            relationship.provenance.value,
            relationship.confidence,
            relationship.sync_id,
            _encode_datetime(relationship.observed_at),
            canonical_json(
                tuple(pair.to_payload() for pair in relationship.field_pairs)
            ),
            canonical_json(relationship.attributes),
        ),
    )


def _insert_catalog_resource(
    connection: sqlite3.Connection,
    resource: CatalogResource,
) -> None:
    connection.execute(
        "INSERT INTO catalog_resources(agent_id, id, source_id, native_identity, "
        "external_uri, kind, name, sensitivity, current_revision, "
        "current_sync_id, first_observed_at, last_observed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            resource.agent_id,
            resource.id,
            resource.source_id,
            resource.native_identity,
            resource.external_uri,
            resource.kind.value,
            resource.name,
            resource.sensitivity.value,
            resource.current_revision,
            resource.current_sync_id,
            _encode_datetime(resource.first_observed_at),
            _encode_datetime(resource.last_observed_at),
        ),
    )


def _catalog_structural_text(
    resource_id: str,
    facets: tuple[CatalogFacet, ...],
    relationships: tuple[CatalogRelationship, ...],
) -> str:
    parts: list[str] = []
    for facet in facets:
        if facet.resource_id == resource_id:
            parts.extend((facet.kind.value, canonical_json(facet.payload)))
    for relationship in relationships:
        if resource_id not in {
            relationship.from_resource_id,
            relationship.to_resource_id,
        }:
            continue
        parts.extend(
            (
                relationship.kind.value,
                canonical_json(
                    tuple(pair.to_payload() for pair in relationship.field_pairs)
                ),
                canonical_json(relationship.attributes),
            )
        )
    return " ".join(parts)


def _commit_catalog_snapshot(
    connection: sqlite3.Connection,
    snapshot: SourceCatalogSnapshot,
) -> SourceCatalogSnapshot:
    sync = snapshot.sync
    connection.execute("BEGIN IMMEDIATE")
    try:
        _require_catalog_agent(
            connection,
            agent_id=sync.agent_id,
            source_id=sync.source_id,
            sync_id=sync.id,
        )
        current = _load_catalog_sync_any_agent(connection, sync.id)
        if current is not None:
            if not _catalog_sync_identity_matches(current, sync):
                raise CatalogSyncConflictError(
                    sync.source_id,
                    sync.id,
                    "identity_changed",
                )
        current_resources = {
            resource.id: resource
            for resource in _list_catalog_resources(
                connection,
                sync.agent_id,
                sync.source_id,
            )
        }
        normalized_resources: list[CatalogResource] = []
        for resource in snapshot.resources:
            previous = current_resources.get(resource.id)
            if previous is None:
                first_observed_at = resource.last_observed_at
            else:
                if (
                    resource.agent_id,
                    resource.source_id,
                    resource.native_identity,
                    resource.kind,
                ) != (
                    previous.agent_id,
                    previous.source_id,
                    previous.native_identity,
                    previous.kind,
                ):
                    raise CatalogSyncConflictError(
                        sync.source_id,
                        sync.id,
                        "resource_identity_changed",
                    )
                if resource.last_observed_at < previous.last_observed_at:
                    raise CatalogSyncConflictError(
                        sync.source_id,
                        sync.id,
                        "resource_freshness_regressed",
                    )
                first_observed_at = previous.first_observed_at
            normalized_resources.append(
                replace(resource, first_observed_at=first_observed_at)
            )
        normalized_snapshot = replace(
            snapshot,
            resources=tuple(normalized_resources),
        )
        if current is not None:
            if current.status is not CatalogSyncStatus.RUNNING:
                if current == sync and _catalog_snapshot_matches_current(
                    connection,
                    normalized_snapshot,
                ):
                    connection.execute("COMMIT")
                    return normalized_snapshot
                raise CatalogSyncConflictError(
                    sync.source_id,
                    sync.id,
                    "already_terminal",
                )
        if _catalog_snapshot_is_stale(connection, sync):
            raise CatalogSyncConflictError(
                sync.source_id,
                sync.id,
                "stale_snapshot",
            )
        snapshot = normalized_snapshot
        if current is None:
            _insert_catalog_sync(connection, sync)
        else:
            _update_catalog_sync(connection, sync)

        for revision in snapshot.revisions:
            _insert_catalog_revision(connection, sync.agent_id, revision)
        for facet in snapshot.facets:
            _insert_catalog_facet(connection, sync.agent_id, facet)
        for relationship in snapshot.relationships:
            _insert_catalog_relationship(connection, sync.agent_id, relationship)

        connection.execute(
            "DELETE FROM catalog_resource_search WHERE agent_id = ? AND source_id = ?",
            (sync.agent_id, sync.source_id),
        )
        connection.execute(
            "DELETE FROM catalog_resources WHERE agent_id = ? AND source_id = ?",
            (sync.agent_id, sync.source_id),
        )
        for resource in snapshot.resources:
            _insert_catalog_resource(connection, resource)
            connection.execute(
                "INSERT INTO catalog_resource_search(agent_id, resource_id, "
                "source_id, kind, name, native_identity, external_uri, "
                "structural_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    resource.agent_id,
                    resource.id,
                    resource.source_id,
                    resource.kind.value,
                    resource.name,
                    resource.native_identity,
                    resource.external_uri,
                    _catalog_structural_text(
                        resource.id,
                        snapshot.facets,
                        snapshot.relationships,
                    ),
                ),
            )
        connection.execute("COMMIT")
        return snapshot
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _catalog_search_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in _CATALOG_SEARCH_TERM.finditer(query.lower()):
        term = match.group(0)
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) == _CATALOG_SEARCH_MAX_TERMS:
            break
    return tuple(terms)


def _catalog_search_filters(
    request: CatalogSearchRequest,
    *,
    table_name: str,
) -> tuple[list[str], list[object]]:
    filters = [f"{table_name}.agent_id = ?"]
    parameters: list[object] = [request.agent_id]
    if request.source_ids:
        placeholders = ", ".join("?" for _ in request.source_ids)
        filters.append(f"{table_name}.source_id IN ({placeholders})")
        parameters.extend(request.source_ids)
    if request.resource_kinds:
        placeholders = ", ".join("?" for _ in request.resource_kinds)
        filters.append(f"{table_name}.kind IN ({placeholders})")
        parameters.extend(kind.value for kind in request.resource_kinds)
    return filters, parameters


def _catalog_hit_match_fields(
    row: sqlite3.Row,
    terms: tuple[str, ...],
) -> tuple[str, ...]:
    fields: list[str] = []
    for field_name in ("name", "native_identity", "external_uri", "structural_text"):
        value = _sqlite_text(row[field_name], f"catalog search {field_name}").lower()
        if any(term in value for term in terms):
            fields.append(
                "structure" if field_name == "structural_text" else field_name
            )
    return tuple(fields)


def _decode_catalog_search_hit(
    row: sqlite3.Row,
    terms: tuple[str, ...],
) -> CatalogSearchHit:
    try:
        rank_value = row["rank_value"]
        if not isinstance(rank_value, (int, float)) or isinstance(rank_value, bool):
            raise TypeError("catalog search rank must be numeric")
        score = min(1_000_000.0, max(0.0, -float(rank_value)))
        matched_fields = _catalog_hit_match_fields(row, terms) if terms else ()
        return CatalogSearchHit(
            resource_id=_sqlite_text(row["id"], "catalog search resource id"),
            source_id=_sqlite_text(row["source_id"], "catalog search source id"),
            kind=ResourceKind(_sqlite_text(row["kind"], "catalog search kind")),
            name=_sqlite_text(row["name"], "catalog search name"),
            revision=_sqlite_text(row["current_revision"], "catalog search revision"),
            sensitivity=Sensitivity(
                _sqlite_text(row["sensitivity"], "catalog search sensitivity")
            ),
            score=score,
            matched_fields=matched_fields,
            match_reasons=("lexical_fts",) if terms else (),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct catalog search hit") from error


def _search_catalog(
    connection: sqlite3.Connection,
    request: CatalogSearchRequest,
) -> CatalogSearchResult:
    terms = _catalog_search_terms(request.query)
    resource_filters, resource_parameters = _catalog_search_filters(
        request,
        table_name="resource",
    )
    if not terms:
        where = " AND ".join(resource_filters)
        total_row = connection.execute(
            f"SELECT COUNT(*) FROM catalog_resources AS resource WHERE {where}",
            tuple(resource_parameters),
        ).fetchone()
        assert total_row is not None
        total_matches = _sqlite_int(total_row[0], "catalog search total")
        rows = connection.execute(
            "SELECT resource.id, resource.source_id, resource.kind, resource.name, "
            "resource.native_identity, resource.external_uri, '' AS structural_text, "
            "resource.current_revision, resource.sensitivity, 0.0 AS rank_value "
            f"FROM catalog_resources AS resource WHERE {where} "
            "ORDER BY resource.name COLLATE BINARY, resource.id LIMIT ?",
            (*resource_parameters, request.limit),
        ).fetchall()
    else:
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        filters = ["catalog_resource_search MATCH ?", *resource_filters]
        parameters: list[object] = [fts_query, *resource_parameters]
        where = " AND ".join(filters)
        total_row = connection.execute(
            "SELECT COUNT(*) FROM catalog_resource_search "
            "JOIN catalog_resources AS resource "
            "ON resource.agent_id = catalog_resource_search.agent_id "
            "AND resource.id = catalog_resource_search.resource_id "
            f"WHERE {where}",
            tuple(parameters),
        ).fetchone()
        assert total_row is not None
        total_matches = _sqlite_int(total_row[0], "catalog search total")
        rows = connection.execute(
            "SELECT resource.id, resource.source_id, resource.kind, resource.name, "
            "catalog_resource_search.native_identity AS native_identity, "
            "catalog_resource_search.external_uri AS external_uri, "
            "catalog_resource_search.structural_text AS structural_text, "
            "resource.current_revision, resource.sensitivity, "
            "bm25(catalog_resource_search) AS rank_value "
            "FROM catalog_resource_search JOIN catalog_resources AS resource "
            "ON resource.agent_id = catalog_resource_search.agent_id "
            "AND resource.id = catalog_resource_search.resource_id "
            f"WHERE {where} ORDER BY rank_value, resource.name COLLATE BINARY, "
            "resource.id LIMIT ?",
            (*parameters, request.limit),
        ).fetchall()
    hits = tuple(_decode_catalog_search_hit(row, terms) for row in rows)
    return CatalogSearchResult(
        request=request,
        hits=hits,
        total_matches=total_matches,
        truncated=total_matches > len(hits),
    )


def _catalog_adjacency(
    connection: sqlite3.Connection,
    request: CatalogTraversalRequest,
    resource_id: str,
) -> tuple[CatalogRelationship, ...]:
    filters = [
        "relationship.agent_id = ?",
        "(relationship.from_resource_id = ? OR relationship.to_resource_id = ?)",
    ]
    parameters: list[object] = [request.agent_id, resource_id, resource_id]
    if request.relationship_kinds:
        placeholders = ", ".join("?" for _ in request.relationship_kinds)
        filters.append(f"relationship.kind IN ({placeholders})")
        parameters.extend(kind.value for kind in request.relationship_kinds)
    rows = connection.execute(
        "SELECT relationship.id, relationship.revision, relationship.source_id, "
        "relationship.from_resource_id, relationship.to_resource_id, "
        "relationship.kind, relationship.provenance, relationship.confidence, "
        "relationship.sync_id, relationship.observed_at, "
        "relationship.field_pairs_json, relationship.attributes_json "
        "FROM catalog_relationships AS relationship "
        "JOIN catalog_resources AS from_resource "
        "ON from_resource.agent_id = relationship.agent_id "
        "AND from_resource.id = relationship.from_resource_id "
        "AND from_resource.current_sync_id = relationship.sync_id "
        "JOIN catalog_resources AS to_resource "
        "ON to_resource.agent_id = relationship.agent_id "
        "AND to_resource.id = relationship.to_resource_id "
        "AND to_resource.current_sync_id = relationship.sync_id WHERE "
        + " AND ".join(filters)
        + " ORDER BY relationship.id LIMIT ?",
        (*parameters, request.max_edges + 1),
    ).fetchall()
    return tuple(_decode_catalog_relationship_row(row) for row in rows)


def _traverse_catalog(
    connection: sqlite3.Connection,
    request: CatalogTraversalRequest,
) -> CatalogTraversalResult:
    for resource_id in (*request.from_resource_ids, *request.to_resource_ids):
        if _load_catalog_resource(connection, request.agent_id, resource_id) is None:
            raise CatalogResourceNotFoundError(request.agent_id, resource_id)

    targets = set(request.to_resource_ids)
    roots = request.from_resource_ids[: request.max_nodes]
    truncated = len(roots) < len(request.from_resource_ids)
    visited_nodes = set(roots)
    visited_edges: set[str] = set()
    queue: deque[tuple[str, tuple[str, ...], tuple[CatalogPathStep, ...]]] = deque(
        (resource_id, (resource_id,), ()) for resource_id in roots
    )
    paths: list[CatalogPath] = []

    while queue:
        current_id, resource_ids, steps = queue.popleft()
        if current_id in targets:
            paths.append(CatalogPath(resource_ids=resource_ids, steps=steps))
            if len(paths) >= request.max_paths:
                truncated = truncated or bool(queue)
                break
            continue
        if len(steps) >= request.max_depth:
            truncated = True
            continue

        adjacency = _catalog_adjacency(connection, request, current_id)
        if len(adjacency) > request.max_edges:
            truncated = True
        for relationship in adjacency:
            if relationship.id in visited_edges:
                continue
            if len(visited_edges) >= request.max_edges:
                truncated = True
                break
            visited_edges.add(relationship.id)
            if relationship.from_resource_id == current_id:
                neighbor_id = relationship.to_resource_id
                direction = RelationshipDirection.FORWARD
            else:
                neighbor_id = relationship.from_resource_id
                direction = RelationshipDirection.REVERSE
            if neighbor_id in resource_ids:
                continue
            if neighbor_id not in visited_nodes:
                if len(visited_nodes) >= request.max_nodes:
                    truncated = True
                    continue
                visited_nodes.add(neighbor_id)
            step = CatalogPathStep(
                relationship_id=relationship.id,
                from_resource_id=current_id,
                to_resource_id=neighbor_id,
                direction=direction,
            )
            queue.append(
                (
                    neighbor_id,
                    (*resource_ids, neighbor_id),
                    (*steps, step),
                )
            )

    return CatalogTraversalResult(
        request=request,
        paths=tuple(paths),
        reachable=bool(paths),
        visited_nodes=len(visited_nodes),
        visited_edges=len(visited_edges),
        truncated=truncated,
    )


def _load_session_operation_facts(
    connection: sqlite3.Connection,
    operation_id: str,
) -> SessionOperationFacts | None:
    row = connection.execute(
        "SELECT operation.id, operation.agent_id, operation.session_id, "
        "operation.revision, operation.status, operation.final_text, "
        "operation.terminal_reason, trigger.payload_json AS trigger_payload_json "
        "FROM operations AS operation "
        "JOIN triggers AS trigger ON trigger.id = operation.trigger_id "
        "WHERE operation.id = ?",
        (operation_id,),
    ).fetchone()
    if row is None or row["session_id"] is None:
        return None
    try:
        evidence_rows = connection.execute(
            "SELECT id, payload_json FROM evidence "
            "WHERE operation_id = ? AND accepted = 1 ORDER BY position",
            (operation_id,),
        ).fetchall()
        approval_rows = connection.execute(
            "SELECT id, status FROM approvals "
            "WHERE operation_id = ? ORDER BY position",
            (operation_id,),
        ).fetchall()
        observation_rows = connection.execute(
            "SELECT payload_json FROM observations "
            "WHERE operation_id = ? AND success = 1 ORDER BY position",
            (operation_id,),
        ).fetchall()
        model_request_rows = connection.execute(
            "SELECT request_json FROM model_calls "
            "WHERE operation_id = ? ORDER BY position",
            (operation_id,),
        ).fetchall()
        evidence_payloads = tuple(
            _decode_json_object(
                _sqlite_text(
                    payload_row["payload_json"],
                    "session operation evidence payload",
                )
            )
            for payload_row in evidence_rows
        )
        observation_payloads = tuple(
            _decode_json_object(
                _sqlite_text(
                    payload_row["payload_json"],
                    "session operation observation payload",
                )
            )
            for payload_row in observation_rows
        )
        resource_ids: list[str] = []
        seen_resources: set[str] = set()
        for payload in (*evidence_payloads, *observation_payloads):
            for resource_id in _resource_ids_from_json(payload):
                if resource_id not in seen_resources:
                    seen_resources.add(resource_id)
                    resource_ids.append(resource_id)
        scope_by_resource: dict[str, SessionResourceScopeFact] = {}
        for payload in evidence_payloads:
            for scope_fact in _resource_scope_facts_from_json(payload):
                previous = scope_by_resource.get(scope_fact.resource_id)
                if previous is not None and previous != scope_fact:
                    raise ValueError(
                        "accepted evidence contains conflicting resource revision scope"
                    )
                scope_by_resource[scope_fact.resource_id] = scope_fact
        trigger_payload = _decode_json_object(
            _sqlite_text(
                row["trigger_payload_json"],
                "session operation trigger payload",
            )
        )
        sensitivity = max(
            (
                _decode_model_request(
                    _sqlite_text(
                        request_row["request_json"],
                        "session operation model request",
                    )
                ).sensitivity
                for request_row in model_request_rows
            ),
            default=ModelSensitivity.INTERNAL,
            key=lambda item: item.routing_rank,
        )
        return SessionOperationFacts(
            operation_id=_sqlite_text(row["id"], "session operation id"),
            agent_id=_sqlite_text(row["agent_id"], "session operation agent_id"),
            session_id=_sqlite_text(
                row["session_id"],
                "session operation session_id",
            ),
            revision=str(_sqlite_int(row["revision"], "session operation revision")),
            status=_sqlite_text(row["status"], "session operation status"),
            sensitivity=sensitivity,
            evidence_ids=tuple(
                _sqlite_text(item["id"], "session operation evidence id")
                for item in evidence_rows
            ),
            approval_ids=tuple(
                _sqlite_text(item["id"], "session operation approval id")
                for item in approval_rows
            ),
            resource_ids=tuple(resource_ids),
            final_text=_optional_text(row["final_text"]),
            objective=_bounded_session_fact(trigger_payload.get("message"), 2_048),
            terminal_reason=_bounded_session_fact(row["terminal_reason"], 512),
            approval_state_facts=tuple(
                SessionApprovalStateFact(
                    approval_id=_sqlite_text(
                        item["id"],
                        "session operation approval fact id",
                    ),
                    status=ApprovalStatus(
                        _sqlite_text(
                            item["status"],
                            "session operation approval fact status",
                        )
                    ),
                )
                for item in approval_rows
            ),
            resource_scope_facts=tuple(
                scope_by_resource[resource_id]
                for resource_id in resource_ids
                if resource_id in scope_by_resource
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            f"cannot reconstruct session operation facts {operation_id}"
        ) from error


def _resource_ids_from_json(value: object) -> tuple[str, ...]:
    values: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "resource_id" and isinstance(item, str) and item.strip():
                values.append(item)
            elif key == "resource_ids" and isinstance(item, (list, tuple)):
                values.extend(
                    child for child in item if isinstance(child, str) and child.strip()
                )
            else:
                values.extend(_resource_ids_from_json(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.extend(_resource_ids_from_json(item))
    return tuple(values)


def _resource_scope_facts_from_json(
    value: object,
) -> tuple[SessionResourceScopeFact, ...]:
    facts: list[SessionResourceScopeFact] = []
    if isinstance(value, Mapping):
        source_id = value.get("source_id")
        source_revision = value.get("source_revision")
        if (
            isinstance(source_id, str)
            and source_id.strip()
            and isinstance(source_revision, str)
            and source_revision.strip()
        ):
            resource_id = value.get("resource_id")
            resource_revision = value.get("resource_revision")
            if (
                isinstance(resource_id, str)
                and resource_id.strip()
                and isinstance(resource_revision, str)
                and resource_revision.strip()
            ):
                facts.append(
                    SessionResourceScopeFact(
                        source_id=source_id,
                        resource_id=resource_id,
                        source_revision=source_revision,
                        resource_revision=resource_revision,
                    )
                )
            resource_revisions = value.get("resource_revisions")
            if isinstance(resource_revisions, (list, tuple)):
                for item in resource_revisions:
                    if not isinstance(item, Mapping):
                        continue
                    item_resource_id = item.get("resource_id")
                    item_revision = item.get("revision")
                    if (
                        isinstance(item_resource_id, str)
                        and item_resource_id.strip()
                        and isinstance(item_revision, str)
                        and item_revision.strip()
                    ):
                        facts.append(
                            SessionResourceScopeFact(
                                source_id=source_id,
                                resource_id=item_resource_id,
                                source_revision=source_revision,
                                resource_revision=item_revision,
                            )
                        )
        for item in value.values():
            facts.extend(_resource_scope_facts_from_json(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            facts.extend(_resource_scope_facts_from_json(item))
    return tuple(facts)


def _bounded_session_fact(value: object, maximum: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if len(value) <= maximum:
        return value
    marker = "…[truncated]"
    return value[: maximum - len(marker)] + marker


def _load_session_compression(
    connection: sqlite3.Connection,
    agent_id: str,
    session_id: str,
) -> SessionCompressionCheckpoint | None:
    connection.execute("BEGIN")
    try:
        checkpoint = _load_session_compression_in_transaction(
            connection,
            agent_id,
            session_id,
        )
        connection.execute("COMMIT")
        return checkpoint
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_session_compression_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    session_id: str,
) -> SessionCompressionCheckpoint | None:
    rows = connection.execute(
        "SELECT * FROM session_compression_checkpoints "
        "WHERE agent_id = ? AND session_id = ? ORDER BY version",
        (agent_id, session_id),
    ).fetchall()
    if not rows:
        return None
    try:
        versions = tuple(
            _sqlite_int(row["version"], "session compression version") for row in rows
        )
        if versions != tuple(range(1, len(rows) + 1)):
            raise SQLiteCorruptionError(
                "session compression versions must be contiguous from one"
            )
        return _decode_session_compression(connection, rows[-1])
    except SQLiteCorruptionError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            f"cannot reconstruct session compression for {session_id}"
        ) from error


def _decode_session_compression(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> SessionCompressionCheckpoint:
    checkpoint_id = _sqlite_text(row["id"], "session compression id")

    def references(table: str, column: str) -> tuple[str, ...]:
        if table not in {
            "session_compression_operations",
            "session_compression_evidence",
            "session_compression_approvals",
            "session_compression_resources",
        }:
            raise ValueError("unsupported session compression reference table")
        values = connection.execute(
            f"SELECT position, {column} FROM {table} "
            "WHERE checkpoint_id = ? ORDER BY position",
            (checkpoint_id,),
        ).fetchall()
        _validate_contiguous_positions(
            values,
            label=f"session compression {column}",
        )
        return tuple(
            _sqlite_text(value[column], f"session compression {column}")
            for value in values
        )

    return SessionCompressionCheckpoint(
        id=checkpoint_id,
        agent_id=_sqlite_text(row["agent_id"], "session compression agent_id"),
        session_id=_sqlite_text(row["session_id"], "session compression session_id"),
        version=_sqlite_int(row["version"], "session compression version"),
        through_position=_sqlite_int(
            row["through_position"],
            "session compression through_position",
        ),
        through_operation_id=_sqlite_text(
            row["through_operation_id"],
            "session compression through operation id",
        ),
        source_fingerprint=_sqlite_text(
            row["source_fingerprint"],
            "session compression source fingerprint",
        ),
        summary=_sqlite_text(row["summary"], "session compression summary"),
        operation_ids=references(
            "session_compression_operations",
            "operation_id",
        ),
        evidence_ids=references("session_compression_evidence", "evidence_id"),
        approval_ids=references("session_compression_approvals", "approval_id"),
        resource_ids=references("session_compression_resources", "resource_id"),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "session compression created_at")
        ),
    )


def _commit_session_compression(
    connection: sqlite3.Connection,
    checkpoint: SessionCompressionCheckpoint,
    *,
    expected_version: int,
) -> SessionCompressionCheckpoint:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != checkpoint.agent_id:
            raise AgentIdentityConflictError(
                "compression checkpoint agent does not match database identity"
            )
        current = _load_session_compression_in_transaction(
            connection,
            checkpoint.agent_id,
            checkpoint.session_id,
        )
        current_version = 0 if current is None else current.version
        if current_version != expected_version:
            raise SQLiteStoreError(
                f"session compression version changed: expected {expected_version}, "
                f"found {current_version}"
            )
        if checkpoint.version != expected_version + 1:
            raise SQLiteStoreError(
                "session compression checkpoint version does not follow its guard"
            )
        session_row = connection.execute(
            "SELECT 1 FROM sessions WHERE id = ? AND agent_id = ?",
            (checkpoint.session_id, checkpoint.agent_id),
        ).fetchone()
        if session_row is None:
            raise SQLiteStoreError("compression checkpoint session does not exist")
        prefix_rows = connection.execute(
            "SELECT link.position, operation.id, operation.agent_id, "
            "operation.session_id FROM session_operations AS link "
            "JOIN operations AS operation ON operation.id = link.operation_id "
            "WHERE link.session_id = ? AND link.position <= ? ORDER BY link.position",
            (checkpoint.session_id, checkpoint.through_position),
        ).fetchall()
        _validate_contiguous_positions(prefix_rows, label="compression source prefix")
        prefix_ids = tuple(
            _sqlite_text(row["id"], "compression source operation id")
            for row in prefix_rows
        )
        if (
            checkpoint.through_position != len(checkpoint.operation_ids) - 1
            or prefix_ids != checkpoint.operation_ids
            or not prefix_ids
            or prefix_ids[-1] != checkpoint.through_operation_id
            or any(
                _sqlite_text(row["agent_id"], "compression source agent_id")
                != checkpoint.agent_id
                or _sqlite_text(row["session_id"], "compression source session_id")
                != checkpoint.session_id
                for row in prefix_rows
            )
        ):
            raise SQLiteStoreError(
                "compression checkpoint does not match the exact session prefix"
            )
        facts = tuple(
            _load_session_operation_facts(connection, operation_id)
            for operation_id in prefix_ids
        )
        if any(item is None for item in facts):
            raise SQLiteStoreError("compression checkpoint source facts are missing")
        typed_facts = tuple(item for item in facts if item is not None)
        expected_evidence = _ordered_unique_strings(
            evidence_id for item in typed_facts for evidence_id in item.evidence_ids
        )
        expected_approvals = _ordered_unique_strings(
            approval_id for item in typed_facts for approval_id in item.approval_ids
        )
        expected_resources = _ordered_unique_strings(
            resource_id for item in typed_facts for resource_id in item.resource_ids
        )
        if (
            checkpoint.evidence_ids != expected_evidence
            or checkpoint.approval_ids != expected_approvals
            or checkpoint.resource_ids != expected_resources
        ):
            raise SQLiteStoreError(
                "compression checkpoint references do not match its source prefix"
            )
        summary = _decode_json_object(checkpoint.summary)
        required_summary = {
            "operation_ids": list(checkpoint.operation_ids),
            "evidence_ids": list(checkpoint.evidence_ids),
            "approval_ids": list(checkpoint.approval_ids),
            "resource_ids": list(checkpoint.resource_ids),
        }
        if any(summary.get(key) != value for key, value in required_summary.items()):
            raise SQLiteStoreError(
                "compression checkpoint summary omits its durable references"
            )
        connection.execute(
            "INSERT INTO session_compression_checkpoints("
            "id, agent_id, session_id, version, through_position, "
            "through_operation_id, source_fingerprint, summary, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                checkpoint.id,
                checkpoint.agent_id,
                checkpoint.session_id,
                checkpoint.version,
                checkpoint.through_position,
                checkpoint.through_operation_id,
                checkpoint.source_fingerprint,
                checkpoint.summary,
                _encode_datetime(checkpoint.created_at),
            ),
        )
        for table, column, values in (
            (
                "session_compression_operations",
                "operation_id",
                checkpoint.operation_ids,
            ),
            (
                "session_compression_evidence",
                "evidence_id",
                checkpoint.evidence_ids,
            ),
            (
                "session_compression_approvals",
                "approval_id",
                checkpoint.approval_ids,
            ),
            (
                "session_compression_resources",
                "resource_id",
                checkpoint.resource_ids,
            ),
        ):
            connection.executemany(
                f"INSERT INTO {table}(checkpoint_id, position, {column}) "
                "VALUES (?, ?, ?)",
                (
                    (checkpoint.id, position, value)
                    for position, value in enumerate(values)
                ),
            )
        connection.execute("COMMIT")
        return checkpoint
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _ordered_unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


_MEMORY_SNAPSHOT_COLUMNS = """
    record.id AS record_id,
    record.agent_id AS record_agent_id,
    record.user_id AS record_user_id,
    record.session_id AS record_session_id,
    record.source_id AS record_source_id,
    record.resource_id AS record_resource_id,
    record.scope_fingerprint AS record_scope_fingerprint,
    record.kind AS record_kind,
    record.logical_key AS record_logical_key,
    record.current_version AS record_current_version,
    record.state AS record_state,
    record.created_at AS record_created_at,
    record.updated_at AS record_updated_at,
    record.superseded_by_id AS record_superseded_by_id,
    version.memory_id AS version_memory_id,
    version.version AS version_number,
    version.content AS version_content,
    version.creator AS version_creator,
    version.confidence AS version_confidence,
    version.sensitivity AS version_sensitivity,
    version.provenance_kind AS version_provenance_kind,
    version.provenance_content_hash AS version_provenance_content_hash,
    version.provenance_operation_id AS version_provenance_operation_id,
    version.provenance_trigger_id AS version_provenance_trigger_id,
    version.provenance_evidence_id AS version_provenance_evidence_id,
    version.provenance_session_id AS version_provenance_session_id,
    version.provenance_external_ref AS version_provenance_external_ref,
    version.attributes_json AS version_attributes_json,
    version.expires_at AS version_expires_at,
    version.resource_revision AS version_resource_revision,
    version.supersedes_version AS version_supersedes_version,
    version.created_at AS version_created_at
""".strip()


def _require_bounded_limit(limit: int, *, maximum: int) -> None:
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
        or limit > maximum
    ):
        raise ValueError(f"limit must be an integer from 1 through {maximum}")


def _validate_memory_query(
    scope: MemoryScope,
    states: tuple[MemoryState, ...],
    sensitivities: tuple[MemorySensitivity, ...],
    limit: int,
) -> None:
    if not isinstance(scope, MemoryScope):
        raise TypeError("scope must be a MemoryScope")
    if not states or any(not isinstance(state, MemoryState) for state in states):
        raise ValueError("states must contain MemoryState values")
    if len(states) != len(set(states)):
        raise ValueError("states cannot contain duplicates")
    if not sensitivities or any(
        not isinstance(value, MemorySensitivity) for value in sensitivities
    ):
        raise ValueError("sensitivities must contain MemorySensitivity values")
    if len(sensitivities) != len(set(sensitivities)):
        raise ValueError("sensitivities cannot contain duplicates")
    _require_bounded_limit(limit, maximum=256)


def _validate_memory_runtime_references(
    connection: sqlite3.Connection,
    record: MemoryRecord,
    version: MemoryVersion,
) -> None:
    identity = _load_agent_identity(connection)
    if identity is None or identity.id != record.scope.agent_id:
        raise AgentIdentityConflictError(
            "memory agent does not match the authoritative database identity"
        )
    if record.scope.session_id is not None:
        session = connection.execute(
            "SELECT agent_id FROM sessions WHERE id = ?",
            (record.scope.session_id,),
        ).fetchone()
        if (
            session is None
            or _sqlite_text(session[0], "memory session agent_id")
            != record.scope.agent_id
        ):
            raise MemoryStoreConflictError("memory session is outside its agent scope")
    if record.scope.source_id is not None:
        source = connection.execute(
            "SELECT agent_id FROM attached_sources WHERE id = ?",
            (record.scope.source_id,),
        ).fetchone()
        if (
            source is None
            or _sqlite_text(source[0], "memory source agent_id")
            != record.scope.agent_id
        ):
            raise MemoryStoreConflictError("memory source is outside its agent scope")
    provenance = version.provenance
    if provenance.operation_id is not None:
        operation = connection.execute(
            "SELECT agent_id, trigger_id, session_id FROM operations WHERE id = ?",
            (provenance.operation_id,),
        ).fetchone()
        if operation is None or (
            _sqlite_text(operation["agent_id"], "memory operation agent_id")
            != record.scope.agent_id
        ):
            raise MemoryStoreConflictError(
                "memory provenance operation is outside its agent scope"
            )
        if (
            provenance.trigger_id is not None
            and _sqlite_text(operation["trigger_id"], "memory operation trigger_id")
            != provenance.trigger_id
        ):
            raise MemoryStoreConflictError(
                "memory provenance trigger does not own its operation"
            )
    if provenance.evidence_id is not None:
        evidence = connection.execute(
            "SELECT operation_id, accepted FROM evidence WHERE id = ?",
            (provenance.evidence_id,),
        ).fetchone()
        if (
            evidence is None
            or _sqlite_int(evidence["accepted"], "memory evidence accepted") != 1
            or _sqlite_text(evidence["operation_id"], "memory evidence operation_id")
            != provenance.operation_id
        ):
            raise MemoryStoreConflictError(
                "memory provenance requires accepted evidence from its operation"
            )
    if provenance.session_id is not None:
        session = connection.execute(
            "SELECT agent_id FROM sessions WHERE id = ?",
            (provenance.session_id,),
        ).fetchone()
        if (
            session is None
            or _sqlite_text(session[0], "memory provenance session agent_id")
            != record.scope.agent_id
        ):
            raise MemoryStoreConflictError(
                "memory provenance session is outside its agent scope"
            )


def _create_memory(
    connection: sqlite3.Connection,
    record: MemoryRecord,
    version: MemoryVersion,
) -> MemoryHistory:
    try:
        MemorySnapshot(record=record, version=version)
    except (TypeError, ValueError) as error:
        raise MemoryStoreConflictError(
            "memory record and first version do not form one snapshot"
        ) from error
    if record.current_version != 1 or version.version != 1:
        raise MemoryStoreConflictError("new memory must begin at version one")
    connection.execute("BEGIN IMMEDIATE")
    try:
        _validate_memory_runtime_references(connection, record, version)
        existing = _load_memory_history_in_transaction(
            connection,
            record.scope.agent_id,
            record.id,
        )
        if existing is not None:
            if existing.record == record and existing.versions == (version,):
                connection.execute("COMMIT")
                return existing
            raise MemoryStoreConflictError(f"memory already exists: {record.id}")
        try:
            _insert_memory_record(connection, record)
            _insert_memory_version(connection, version)
            _replace_memory_search(connection, record, version)
        except sqlite3.IntegrityError as error:
            raise MemoryStoreConflictError(
                "memory logical identity or durable reference is already claimed"
            ) from error
        history = _load_memory_history_in_transaction(
            connection,
            record.scope.agent_id,
            record.id,
        )
        assert history is not None
        connection.execute("COMMIT")
        return history
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _insert_memory_record(
    connection: sqlite3.Connection,
    record: MemoryRecord,
) -> None:
    connection.execute(
        "INSERT INTO memory_records("
        "id, agent_id, user_id, session_id, source_id, resource_id, "
        "scope_fingerprint, kind, logical_key, current_version, state, "
        "created_at, updated_at, superseded_by_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.id,
            record.scope.agent_id,
            record.scope.user_id,
            record.scope.session_id,
            record.scope.source_id,
            record.scope.resource_id,
            record.scope.fingerprint,
            record.kind.value,
            record.logical_key,
            record.current_version,
            record.state.value,
            _encode_datetime(record.created_at),
            _encode_datetime(record.updated_at),
            record.superseded_by_id,
        ),
    )


def _insert_memory_version(
    connection: sqlite3.Connection,
    version: MemoryVersion,
) -> None:
    provenance = version.provenance
    connection.execute(
        "INSERT INTO memory_versions("
        "memory_id, version, content, creator, confidence, sensitivity, "
        "provenance_kind, provenance_content_hash, provenance_operation_id, "
        "provenance_trigger_id, provenance_evidence_id, provenance_session_id, "
        "provenance_external_ref, attributes_json, expires_at, resource_revision, "
        "supersedes_version, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            version.memory_id,
            version.version,
            version.content,
            version.creator.value,
            version.confidence,
            version.sensitivity.value,
            provenance.kind.value,
            provenance.content_hash,
            provenance.operation_id,
            provenance.trigger_id,
            provenance.evidence_id,
            provenance.session_id,
            provenance.external_ref,
            canonical_json(version.attributes),
            _encode_optional_datetime(version.expires_at),
            version.resource_revision,
            version.supersedes_version,
            _encode_datetime(version.created_at),
        ),
    )


def _replace_memory_search(
    connection: sqlite3.Connection,
    record: MemoryRecord,
    version: MemoryVersion,
) -> None:
    connection.execute("DELETE FROM memory_search WHERE memory_id = ?", (record.id,))
    connection.execute(
        "INSERT INTO memory_search(memory_id, logical_key, content, attributes) "
        "VALUES (?, ?, ?, ?)",
        (
            record.id,
            record.logical_key,
            version.content,
            canonical_json(version.attributes),
        ),
    )


def _decode_memory_record(row: sqlite3.Row) -> MemoryRecord:
    scope = MemoryScope(
        agent_id=_sqlite_text(row["record_agent_id"], "memory agent_id"),
        user_id=_optional_text(row["record_user_id"]),
        session_id=_optional_text(row["record_session_id"]),
        source_id=_optional_text(row["record_source_id"]),
        resource_id=_optional_text(row["record_resource_id"]),
    )
    stored_fingerprint = _sqlite_text(
        row["record_scope_fingerprint"],
        "memory scope fingerprint",
    )
    if scope.fingerprint != stored_fingerprint:
        raise SQLiteCorruptionError(
            "memory scope fingerprint does not match its fields"
        )
    return MemoryRecord(
        id=_sqlite_text(row["record_id"], "memory id"),
        scope=scope,
        kind=MemoryKind(_sqlite_text(row["record_kind"], "memory kind")),
        logical_key=_sqlite_text(row["record_logical_key"], "memory logical_key"),
        current_version=_sqlite_int(
            row["record_current_version"],
            "memory current_version",
        ),
        state=MemoryState(_sqlite_text(row["record_state"], "memory state")),
        created_at=_decode_datetime(
            _sqlite_text(row["record_created_at"], "memory created_at")
        ),
        updated_at=_decode_datetime(
            _sqlite_text(row["record_updated_at"], "memory updated_at")
        ),
        superseded_by_id=_optional_text(row["record_superseded_by_id"]),
    )


def _decode_memory_version(row: sqlite3.Row) -> MemoryVersion:
    return MemoryVersion(
        memory_id=_sqlite_text(row["version_memory_id"], "memory version memory_id"),
        version=_sqlite_int(row["version_number"], "memory version"),
        content=_sqlite_text(row["version_content"], "memory content"),
        creator=MemoryCreator(_sqlite_text(row["version_creator"], "memory creator")),
        confidence=_sqlite_real(row["version_confidence"], "memory confidence"),
        sensitivity=MemorySensitivity(
            _sqlite_text(row["version_sensitivity"], "memory sensitivity")
        ),
        provenance=MemoryProvenance(
            kind=MemoryProvenanceKind(
                _sqlite_text(
                    row["version_provenance_kind"],
                    "memory provenance kind",
                )
            ),
            content_hash=_sqlite_text(
                row["version_provenance_content_hash"],
                "memory provenance content_hash",
            ),
            operation_id=_optional_text(row["version_provenance_operation_id"]),
            trigger_id=_optional_text(row["version_provenance_trigger_id"]),
            evidence_id=_optional_text(row["version_provenance_evidence_id"]),
            session_id=_optional_text(row["version_provenance_session_id"]),
            external_ref=_optional_text(row["version_provenance_external_ref"]),
        ),
        created_at=_decode_datetime(
            _sqlite_text(row["version_created_at"], "memory version created_at")
        ),
        attributes=_decode_json_object(
            _sqlite_text(row["version_attributes_json"], "memory attributes")
        ),
        expires_at=_decode_optional_datetime(
            row["version_expires_at"],
            "memory expires_at",
        ),
        resource_revision=_optional_text(row["version_resource_revision"]),
        supersedes_version=(
            None
            if row["version_supersedes_version"] is None
            else _sqlite_int(
                row["version_supersedes_version"],
                "memory supersedes_version",
            )
        ),
    )


def _decode_memory_snapshot(row: sqlite3.Row) -> MemorySnapshot:
    try:
        return MemorySnapshot(
            record=_decode_memory_record(row),
            version=_decode_memory_version(row),
        )
    except SQLiteCorruptionError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct memory snapshot") from error


def _memory_structured_filters(
    *,
    scope: MemoryScope,
    states: tuple[MemoryState, ...],
    sensitivities: tuple[MemorySensitivity, ...],
    unexpired_at: datetime | None,
) -> tuple[list[str], list[object]]:
    state_slots = ", ".join("?" for _ in states)
    sensitivity_slots = ", ".join("?" for _ in sensitivities)
    filters = [
        "record.agent_id = ?",
        f"record.state IN ({state_slots})",
        f"version.sensitivity IN ({sensitivity_slots})",
        "(record.user_id IS NULL OR record.user_id = ?)",
        "(record.session_id IS NULL OR record.session_id = ?)",
        "(record.source_id IS NULL OR record.source_id = ?)",
        "(record.resource_id IS NULL OR record.resource_id = ?)",
    ]
    parameters: list[object] = [
        scope.agent_id,
        *(state.value for state in states),
        *(sensitivity.value for sensitivity in sensitivities),
        scope.user_id,
        scope.session_id,
        scope.source_id,
        scope.resource_id,
    ]
    if unexpired_at is not None:
        filters.append("(version.expires_at IS NULL OR version.expires_at > ?)")
        parameters.append(_encode_datetime(unexpired_at))
    return filters, parameters


def _memory_search_terms(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"\w+", query.casefold())))[:32]


def _list_memory_candidates(
    connection: sqlite3.Connection,
    *,
    scope: MemoryScope,
    states: tuple[MemoryState, ...],
    sensitivities: tuple[MemorySensitivity, ...],
    limit: int,
) -> tuple[MemorySnapshot, ...]:
    filters, parameters = _memory_structured_filters(
        scope=scope,
        states=states,
        sensitivities=sensitivities,
        unexpired_at=None,
    )
    rows = connection.execute(
        f"SELECT {_MEMORY_SNAPSHOT_COLUMNS} FROM memory_records AS record "
        "JOIN memory_versions AS version ON version.memory_id = record.id "
        "AND version.version = record.current_version "
        f"WHERE {' AND '.join(filters)} "
        "ORDER BY record.updated_at DESC, record.id LIMIT ?",
        (*parameters, limit),
    ).fetchall()
    return tuple(_decode_memory_snapshot(row) for row in rows)


def _recall_memory_candidates(
    connection: sqlite3.Connection,
    *,
    query: str,
    scope: MemoryScope,
    states: tuple[MemoryState, ...],
    sensitivities: tuple[MemorySensitivity, ...],
    unexpired_at: datetime,
    limit: int,
) -> tuple[MemorySnapshot, ...]:
    filters, parameters = _memory_structured_filters(
        scope=scope,
        states=states,
        sensitivities=sensitivities,
        unexpired_at=unexpired_at,
    )
    terms = _memory_search_terms(query)
    if not terms:
        return _list_memory_candidates(
            connection,
            scope=scope,
            states=states,
            sensitivities=sensitivities,
            limit=limit,
        )
    fts_query = " OR ".join(f'"{term}"' for term in terms)
    rows = connection.execute(
        "WITH eligible(memory_id) AS MATERIALIZED ("
        "SELECT record.id FROM memory_records AS record "
        "JOIN memory_versions AS version ON version.memory_id = record.id "
        "AND version.version = record.current_version "
        f"WHERE {' AND '.join(filters)}"
        ") "
        f"SELECT {_MEMORY_SNAPSHOT_COLUMNS}, bm25(memory_search) AS rank_value "
        "FROM eligible JOIN memory_search ON memory_search.memory_id = eligible.memory_id "
        "JOIN memory_records AS record ON record.id = eligible.memory_id "
        "JOIN memory_versions AS version ON version.memory_id = record.id "
        "AND version.version = record.current_version "
        "WHERE memory_search MATCH ? "
        "ORDER BY rank_value, record.updated_at DESC, record.id LIMIT ?",
        (*parameters, fts_query, limit),
    ).fetchall()
    return tuple(_decode_memory_snapshot(row) for row in rows)


def _load_memory_history(
    connection: sqlite3.Connection,
    agent_id: str,
    memory_id: str,
) -> MemoryHistory | None:
    connection.execute("BEGIN")
    try:
        history = _load_memory_history_in_transaction(connection, agent_id, memory_id)
        connection.execute("COMMIT")
        return history
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_memory_history_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    memory_id: str,
) -> MemoryHistory | None:
    record_row = connection.execute(
        f"SELECT {_MEMORY_SNAPSHOT_COLUMNS} FROM memory_records AS record "
        "JOIN memory_versions AS version ON version.memory_id = record.id "
        "AND version.version = record.current_version "
        "WHERE record.agent_id = ? AND record.id = ?",
        (agent_id, memory_id),
    ).fetchone()
    if record_row is None:
        return None
    try:
        record = _decode_memory_record(record_row)
        version_rows = connection.execute(
            "SELECT memory_id AS version_memory_id, version AS version_number, "
            "content AS version_content, creator AS version_creator, "
            "confidence AS version_confidence, sensitivity AS version_sensitivity, "
            "provenance_kind AS version_provenance_kind, "
            "provenance_content_hash AS version_provenance_content_hash, "
            "provenance_operation_id AS version_provenance_operation_id, "
            "provenance_trigger_id AS version_provenance_trigger_id, "
            "provenance_evidence_id AS version_provenance_evidence_id, "
            "provenance_session_id AS version_provenance_session_id, "
            "provenance_external_ref AS version_provenance_external_ref, "
            "attributes_json AS version_attributes_json, "
            "expires_at AS version_expires_at, "
            "resource_revision AS version_resource_revision, "
            "supersedes_version AS version_supersedes_version, "
            "created_at AS version_created_at "
            "FROM memory_versions WHERE memory_id = ? ORDER BY version",
            (memory_id,),
        ).fetchall()
        versions = tuple(_decode_memory_version(row) for row in version_rows)
        if tuple(item.version for item in versions) != tuple(
            range(1, len(versions) + 1)
        ):
            raise SQLiteCorruptionError(
                f"memory versions are not contiguous: {memory_id}"
            )
        return MemoryHistory(record=record, versions=versions)
    except SQLiteCorruptionError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            f"cannot reconstruct memory history {memory_id}"
        ) from error


def _replace_memory_version(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    memory_id: str,
    expected_version: int,
    replacement: MemoryVersion,
    restore_version: int | None,
) -> MemoryHistory:
    connection.execute("BEGIN IMMEDIATE")
    try:
        history = _load_memory_history_in_transaction(
            connection,
            agent_id,
            memory_id,
        )
        if history is None:
            raise MemoryStoreConflictError(f"memory does not exist: {memory_id}")
        if (
            history.record.state is not MemoryState.ACTIVE
            or history.record.current_version != expected_version
        ):
            raise MemoryStoreConflictError(
                f"memory {memory_id} head changed before replacement"
            )
        if (
            replacement.memory_id != memory_id
            or replacement.version != expected_version + 1
            or replacement.supersedes_version != expected_version
        ):
            raise MemoryStoreConflictError(
                "replacement does not follow the guarded memory version"
            )
        if replacement.resource_revision is not None and (
            history.record.scope.resource_id is None
        ):
            raise MemoryStoreConflictError(
                "revision-bound replacement requires resource-scoped memory"
            )
        if restore_version is not None:
            source = next(
                (item for item in history.versions if item.version == restore_version),
                None,
            )
            if source is None:
                raise MemoryStoreConflictError(
                    f"restore version does not exist: {restore_version}"
                )
            for field_name in (
                "content",
                "attributes",
                "confidence",
                "sensitivity",
                "expires_at",
                "resource_revision",
            ):
                if getattr(source, field_name) != getattr(replacement, field_name):
                    raise MemoryStoreConflictError(
                        f"restore replacement changed historical {field_name}"
                    )
        _validate_memory_runtime_references(connection, history.record, replacement)
        _insert_memory_version(connection, replacement)
        updated_record = MemoryRecord(
            id=history.record.id,
            scope=history.record.scope,
            kind=history.record.kind,
            logical_key=history.record.logical_key,
            current_version=replacement.version,
            state=history.record.state,
            created_at=history.record.created_at,
            updated_at=max(history.record.updated_at, replacement.created_at),
            superseded_by_id=history.record.superseded_by_id,
        )
        connection.execute(
            "UPDATE memory_records SET current_version = ?, updated_at = ? "
            "WHERE id = ? AND agent_id = ? AND current_version = ?",
            (
                updated_record.current_version,
                _encode_datetime(updated_record.updated_at),
                memory_id,
                agent_id,
                expected_version,
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            raise MemoryStoreConflictError(
                f"memory {memory_id} head changed before replacement"
            )
        _replace_memory_search(connection, updated_record, replacement)
        updated = _load_memory_history_in_transaction(connection, agent_id, memory_id)
        assert updated is not None
        connection.execute("COMMIT")
        return updated
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise MemoryStoreConflictError(
            f"memory {memory_id} replacement conflicts with durable history"
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_resource_alias(
    connection: sqlite3.Connection,
    scope: MemoryScope,
    logical_key: str,
) -> MemoryHistory | None:
    connection.execute("BEGIN")
    try:
        history = _load_resource_alias_in_transaction(
            connection,
            scope,
            logical_key,
        )
        connection.execute("COMMIT")
        return history
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_resource_alias_in_transaction(
    connection: sqlite3.Connection,
    scope: MemoryScope,
    logical_key: str,
) -> MemoryHistory | None:
    row = connection.execute(
        "SELECT id FROM memory_records WHERE agent_id = ? "
        "AND scope_fingerprint = ? AND kind = ? AND logical_key = ?",
        (
            scope.agent_id,
            scope.fingerprint,
            MemoryKind.RESOURCE_ALIAS.value,
            logical_key,
        ),
    ).fetchone()
    if row is None:
        return None
    return _load_memory_history_in_transaction(
        connection,
        scope.agent_id,
        _sqlite_text(row[0], "resource alias memory id"),
    )


def _explicit_proposal_identity_matches(
    stored: LearningProposal,
    request: LearningProposal,
) -> bool:
    return (
        stored.id,
        stored.kind,
        stored.category,
        stored.provenance,
        stored.candidate_hash,
        stored.idempotency_key,
        stored.candidate_payload,
    ) == (
        request.id,
        request.kind,
        request.category,
        request.provenance,
        request.candidate_hash,
        request.idempotency_key,
        request.candidate_payload,
    )


def _validate_explicit_replay_memory(
    stored: LearningProposal,
    memory: MemoryHistory,
    intended: MemorySnapshot,
) -> None:
    result_version = stored.result_memory_version
    if stored.result_memory_id != memory.record.id or result_version is None:
        raise SQLiteCorruptionError(
            "committed correction does not reference its memory history"
        )
    historical = next(
        (version for version in memory.versions if version.version == result_version),
        None,
    )
    if historical is None:
        raise SQLiteCorruptionError(
            "committed correction memory version is absent from history"
        )
    expected = intended.version
    if memory.record.logical_identity != intended.record.logical_identity or any(
        getattr(historical, field_name) != getattr(expected, field_name)
        for field_name in (
            "content",
            "creator",
            "confidence",
            "sensitivity",
            "provenance",
            "attributes",
            "expires_at",
            "resource_revision",
        )
    ):
        raise ExplicitCorrectionStoreConflictError(
            "idempotency key already committed another semantic correction"
        )


def _commit_explicit_correction(
    connection: sqlite3.Connection,
    request: ExplicitCorrectionCommit,
) -> ExplicitCorrectionResult:
    proposal = request.proposal
    agent_id = proposal.provenance.agent_id
    connection.execute("BEGIN IMMEDIATE")
    try:
        replay_row = connection.execute(
            "SELECT * FROM learning_proposals "
            "WHERE agent_id = ? AND idempotency_key = ?",
            (agent_id, proposal.idempotency_key),
        ).fetchone()
        if replay_row is not None:
            stored = _decode_learning_proposal(replay_row)
            _validate_learning_decision_history(connection, stored)
            if not _explicit_proposal_identity_matches(stored, proposal):
                raise ExplicitCorrectionStoreConflictError(
                    "idempotency key is already claimed by another proposal"
                )
            if stored.state is LearningProposalState.PROPOSED:
                raise SQLiteCorruptionError(
                    "atomic correction left a nonterminal proposal"
                )
            if stored.state is LearningProposalState.REJECTED:
                if proposal.state is not LearningProposalState.REJECTED:
                    raise ExplicitCorrectionStoreConflictError(
                        "idempotency key already resolved as rejected"
                    )
                result = ExplicitCorrectionResult(
                    proposal=stored,
                    memory=None,
                    replayed=True,
                )
                connection.execute("COMMIT")
                return result
            if request.intended_memory is None:
                raise ExplicitCorrectionStoreConflictError(
                    "idempotency key already resolved as committed"
                )
            assert stored.result_memory_id is not None
            history = _load_memory_history_in_transaction(
                connection,
                agent_id,
                stored.result_memory_id,
            )
            if history is None:
                raise SQLiteCorruptionError(
                    "committed correction memory history is missing"
                )
            _validate_explicit_replay_memory(
                stored,
                history,
                request.intended_memory,
            )
            result = ExplicitCorrectionResult(
                proposal=stored,
                memory=history,
                replayed=True,
            )
            connection.execute("COMMIT")
            return result

        duplicate_id = connection.execute(
            "SELECT 1 FROM learning_proposals WHERE id = ?",
            (proposal.id,),
        ).fetchone()
        if duplicate_id is not None:
            raise ExplicitCorrectionStoreConflictError(
                "proposal identity is already claimed by another idempotency key"
            )
        _validate_learning_runtime_references(connection, proposal)
        if proposal.state is LearningProposalState.REJECTED:
            _insert_learning_proposal_row(connection, proposal)
            _insert_learning_decision(
                connection,
                _learning_decision_from_proposal(proposal),
            )
            result = ExplicitCorrectionResult(
                proposal=proposal,
                memory=None,
                replayed=False,
            )
            connection.execute("COMMIT")
            return result

        intended = request.intended_memory
        decision = request.decision
        assert intended is not None and decision is not None
        try:
            resolved = resolve_learning_proposal(proposal, decision)
        except LearningTransitionError as error:
            raise ExplicitCorrectionStoreConflictError(
                "correction decision does not match its proposal"
            ) from error
        if (
            intended.record.scope.agent_id != agent_id
            or intended.record.kind is not MemoryKind.RESOURCE_ALIAS
            or intended.record.state is not MemoryState.ACTIVE
            or resolved.result_memory_id != intended.record.id
            or resolved.result_memory_version != intended.version.version
        ):
            raise ExplicitCorrectionStoreConflictError(
                "correction proposal and intended memory do not match"
            )
        _validate_memory_runtime_references(
            connection,
            intended.record,
            intended.version,
        )
        existing = _load_resource_alias_in_transaction(
            connection,
            intended.record.scope,
            intended.record.logical_key,
        )
        if request.expected_memory_version is None:
            if existing is not None:
                raise ExplicitCorrectionStoreConflictError(
                    "resource alias was created before correction commit"
                )
            if intended.record.current_version != 1 or intended.version.version != 1:
                raise ExplicitCorrectionStoreConflictError(
                    "new correction memory must begin at version one"
                )
            _insert_memory_record(connection, intended.record)
            _insert_memory_version(connection, intended.version)
            _replace_memory_search(connection, intended.record, intended.version)
        else:
            expected = request.expected_memory_version
            if (
                existing is None
                or existing.record.state is not MemoryState.ACTIVE
                or existing.record.current_version != expected
            ):
                raise ExplicitCorrectionStoreConflictError(
                    "resource alias head changed before correction commit"
                )
            if (
                intended.record.id != existing.record.id
                or intended.record.scope != existing.record.scope
                or intended.record.kind is not existing.record.kind
                or intended.record.logical_key != existing.record.logical_key
                or intended.record.created_at != existing.record.created_at
                or intended.record.superseded_by_id != existing.record.superseded_by_id
                or intended.record.current_version != expected + 1
                or intended.record.updated_at < existing.record.updated_at
            ):
                raise ExplicitCorrectionStoreConflictError(
                    "correction intended memory changed stable record identity"
                )
            _insert_memory_version(connection, intended.version)
            connection.execute(
                "UPDATE memory_records SET current_version = ?, updated_at = ? "
                "WHERE id = ? AND agent_id = ? AND current_version = ? "
                "AND state = 'active'",
                (
                    intended.record.current_version,
                    _encode_datetime(intended.record.updated_at),
                    intended.record.id,
                    agent_id,
                    expected,
                ),
            )
            changed_row = connection.execute("SELECT changes()").fetchone()
            assert changed_row is not None
            if _sqlite_int(changed_row[0], "correction memory changes") != 1:
                raise ExplicitCorrectionStoreConflictError(
                    "resource alias head changed during correction commit"
                )
            _replace_memory_search(connection, intended.record, intended.version)
        _insert_learning_proposal_row(connection, resolved)
        _insert_learning_decision(connection, decision)
        history = _load_memory_history_in_transaction(
            connection,
            agent_id,
            intended.record.id,
        )
        assert history is not None
        result = ExplicitCorrectionResult(
            proposal=resolved,
            memory=history,
            replayed=False,
        )
        connection.execute("COMMIT")
        return result
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise ExplicitCorrectionStoreConflictError(
            "correction conflicts with durable proposal or memory history"
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _validate_learning_runtime_references(
    connection: sqlite3.Connection,
    proposal: LearningProposal,
) -> None:
    identity = _load_agent_identity(connection)
    agent_id = proposal.provenance.agent_id
    if identity is None or identity.id != agent_id:
        raise AgentIdentityConflictError(
            "learning proposal agent does not match database identity"
        )
    operation = connection.execute(
        "SELECT agent_id, trigger_id, status FROM operations WHERE id = ?",
        (proposal.provenance.operation_id,),
    ).fetchone()
    if (
        operation is None
        or _sqlite_text(operation["agent_id"], "learning operation agent_id")
        != agent_id
        or _sqlite_text(operation["trigger_id"], "learning operation trigger_id")
        != proposal.provenance.trigger_id
    ):
        raise LearningStoreConflictError(
            "learning provenance does not match an operation in its agent scope"
        )
    operation_status = OperationStatus(
        _sqlite_text(operation["status"], "learning operation status")
    )
    actual_outcome = {
        OperationStatus.SUCCEEDED: LearningSourceOutcome.SUCCEEDED,
        OperationStatus.FAILED: LearningSourceOutcome.FAILED,
        OperationStatus.CANCELLED: LearningSourceOutcome.CANCELLED,
        OperationStatus.INTERRUPTED: LearningSourceOutcome.INTERRUPTED,
        OperationStatus.WAITING_FOR_APPROVAL: LearningSourceOutcome.BLOCKED,
        OperationStatus.WAITING_FOR_INPUT: LearningSourceOutcome.BLOCKED,
    }.get(operation_status)
    if (
        actual_outcome is None
        or actual_outcome is not proposal.provenance.source_outcome
    ):
        raise LearningStoreConflictError(
            "learning source outcome does not match durable operation status"
        )
    if proposal.provenance.evidence_id is not None:
        evidence = connection.execute(
            "SELECT operation_id, accepted FROM evidence WHERE id = ?",
            (proposal.provenance.evidence_id,),
        ).fetchone()
        if (
            evidence is None
            or _sqlite_text(
                evidence["operation_id"],
                "learning evidence operation_id",
            )
            != proposal.provenance.operation_id
            or _sqlite_int(evidence["accepted"], "learning evidence accepted") != 1
        ):
            raise LearningStoreConflictError(
                "learning provenance evidence is not accepted by its operation"
            )


def _learning_create_matches(
    existing: LearningProposal,
    candidate: LearningProposal,
) -> bool:
    return (
        existing.kind,
        existing.category,
        existing.provenance,
        existing.candidate_hash,
        existing.idempotency_key,
    ) == (
        candidate.kind,
        candidate.category,
        candidate.provenance,
        candidate.candidate_hash,
        candidate.idempotency_key,
    )


def _create_learning_proposal(
    connection: sqlite3.Connection,
    proposal: LearningProposal,
) -> LearningProposal:
    connection.execute("BEGIN IMMEDIATE")
    try:
        _validate_learning_runtime_references(connection, proposal)
        existing_row = connection.execute(
            "SELECT * FROM learning_proposals "
            "WHERE agent_id = ? AND (id = ? OR idempotency_key = ?) "
            "ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END LIMIT 1",
            (
                proposal.provenance.agent_id,
                proposal.id,
                proposal.idempotency_key,
                proposal.id,
            ),
        ).fetchone()
        if existing_row is not None:
            existing = _decode_learning_proposal(existing_row)
            _validate_learning_decision_history(connection, existing)
            if _learning_create_matches(existing, proposal):
                connection.execute("COMMIT")
                return existing
            raise LearningStoreConflictError(
                "learning proposal identity or idempotency key is already claimed"
            )
        _insert_learning_proposal_row(connection, proposal)
        if proposal.state is not LearningProposalState.PROPOSED:
            _insert_learning_decision(
                connection,
                _learning_decision_from_proposal(proposal),
            )
        stored = _load_learning_proposal_in_transaction(
            connection,
            proposal.provenance.agent_id,
            proposal.id,
        )
        assert stored is not None
        connection.execute("COMMIT")
        return stored
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise LearningStoreConflictError(
            "learning proposal conflicts with durable identity or provenance"
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _insert_learning_proposal_row(
    connection: sqlite3.Connection,
    proposal: LearningProposal,
) -> None:
    connection.execute(
        "INSERT INTO learning_proposals("
        "id, agent_id, kind, category, state, operation_id, trigger_id, "
        "source_outcome, source_hash, evidence_id, evidence_accepted, "
        "candidate_hash, idempotency_key, candidate_payload_json, created_at, "
        "resolved_at, decision_hash, result_memory_id, result_memory_version, "
        "result_skill_id, result_skill_version, rejection_category, "
        "rejection_reason"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?)",
        _learning_proposal_values(proposal),
    )


def _learning_proposal_values(proposal: LearningProposal) -> tuple[object, ...]:
    provenance = proposal.provenance
    return (
        proposal.id,
        provenance.agent_id,
        proposal.kind.value,
        proposal.category.value,
        proposal.state.value,
        provenance.operation_id,
        provenance.trigger_id,
        provenance.source_outcome.value,
        provenance.source_hash,
        provenance.evidence_id,
        int(provenance.evidence_accepted),
        proposal.candidate_hash,
        proposal.idempotency_key,
        (
            None
            if proposal.candidate_payload is None
            else canonical_json(proposal.candidate_payload)
        ),
        _encode_datetime(proposal.created_at),
        _encode_optional_datetime(proposal.resolved_at),
        proposal.decision_hash,
        proposal.result_memory_id,
        proposal.result_memory_version,
        proposal.result_skill_id,
        proposal.result_skill_version,
        (
            None
            if proposal.rejection_category is None
            else proposal.rejection_category.value
        ),
        proposal.rejection_reason,
    )


def _decode_learning_proposal(row: sqlite3.Row) -> LearningProposal:
    try:
        payload_value = row["candidate_payload_json"]
        return LearningProposal(
            id=_sqlite_text(row["id"], "learning proposal id"),
            kind=LearningProposalKind(
                _sqlite_text(row["kind"], "learning proposal kind")
            ),
            category=LearningCandidateCategory(
                _sqlite_text(row["category"], "learning proposal category")
            ),
            state=LearningProposalState(
                _sqlite_text(row["state"], "learning proposal state")
            ),
            provenance=LearningProvenance(
                agent_id=_sqlite_text(row["agent_id"], "learning proposal agent_id"),
                operation_id=_sqlite_text(
                    row["operation_id"],
                    "learning proposal operation_id",
                ),
                trigger_id=_sqlite_text(
                    row["trigger_id"],
                    "learning proposal trigger_id",
                ),
                source_outcome=LearningSourceOutcome(
                    _sqlite_text(
                        row["source_outcome"],
                        "learning proposal source_outcome",
                    )
                ),
                source_hash=_sqlite_text(
                    row["source_hash"],
                    "learning proposal source_hash",
                ),
                evidence_id=_optional_text(row["evidence_id"]),
                evidence_accepted=_decode_bool(row["evidence_accepted"]),
            ),
            candidate_hash=_sqlite_text(
                row["candidate_hash"],
                "learning proposal candidate_hash",
            ),
            idempotency_key=_sqlite_text(
                row["idempotency_key"],
                "learning proposal idempotency_key",
            ),
            candidate_payload=(
                None
                if payload_value is None
                else _decode_json_object(
                    _sqlite_text(
                        payload_value,
                        "learning proposal candidate payload",
                    )
                )
            ),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "learning proposal created_at")
            ),
            resolved_at=_decode_optional_datetime(
                row["resolved_at"],
                "learning proposal resolved_at",
            ),
            decision_hash=_optional_text(row["decision_hash"]),
            result_memory_id=_optional_text(row["result_memory_id"]),
            result_memory_version=(
                None
                if row["result_memory_version"] is None
                else _sqlite_int(
                    row["result_memory_version"],
                    "learning proposal result_memory_version",
                )
            ),
            result_skill_id=_optional_text(row["result_skill_id"]),
            result_skill_version=(
                None
                if row["result_skill_version"] is None
                else _sqlite_int(
                    row["result_skill_version"],
                    "learning proposal result_skill_version",
                )
            ),
            rejection_category=(
                None
                if row["rejection_category"] is None
                else LearningRejectionCategory(
                    _sqlite_text(
                        row["rejection_category"],
                        "learning proposal rejection_category",
                    )
                )
            ),
            rejection_reason=_optional_text(row["rejection_reason"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        proposal_id = _optional_text(row["id"]) or "unknown"
        raise SQLiteCorruptionError(
            f"cannot reconstruct learning proposal {proposal_id}"
        ) from error


def _learning_decision_from_proposal(proposal: LearningProposal) -> LearningDecision:
    if proposal.state is LearningProposalState.PROPOSED:
        raise ValueError("proposed learning has no terminal decision")
    assert proposal.resolved_at is not None
    return LearningDecision(
        proposal_id=proposal.id,
        idempotency_key=proposal.idempotency_key,
        candidate_hash=proposal.candidate_hash,
        state=proposal.state,
        decided_at=proposal.resolved_at,
        result_memory_id=proposal.result_memory_id,
        result_memory_version=proposal.result_memory_version,
        result_skill_id=proposal.result_skill_id,
        result_skill_version=proposal.result_skill_version,
        rejection_category=proposal.rejection_category,
        rejection_reason=proposal.rejection_reason,
    )


def _insert_learning_decision(
    connection: sqlite3.Connection,
    decision: LearningDecision,
) -> None:
    connection.execute(
        "INSERT INTO learning_decisions("
        "proposal_id, decision_hash, idempotency_key, candidate_hash, state, "
        "decided_at, result_memory_id, result_memory_version, result_skill_id, "
        "result_skill_version, rejection_category, rejection_reason"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            decision.proposal_id,
            decision.fingerprint,
            decision.idempotency_key,
            decision.candidate_hash,
            decision.state.value,
            _encode_datetime(decision.decided_at),
            decision.result_memory_id,
            decision.result_memory_version,
            decision.result_skill_id,
            decision.result_skill_version,
            (
                None
                if decision.rejection_category is None
                else decision.rejection_category.value
            ),
            decision.rejection_reason,
        ),
    )


def _validate_learning_decision_history(
    connection: sqlite3.Connection,
    proposal: LearningProposal,
) -> None:
    rows = connection.execute(
        "SELECT decision_hash FROM learning_decisions WHERE proposal_id = ?",
        (proposal.id,),
    ).fetchall()
    if proposal.state is LearningProposalState.PROPOSED:
        if rows:
            raise SQLiteCorruptionError(
                "proposed learning unexpectedly has a terminal decision"
            )
        return
    if len(rows) != 1 or (
        _sqlite_text(rows[0][0], "learning decision hash") != proposal.decision_hash
    ):
        raise SQLiteCorruptionError(
            "terminal learning proposal decision history does not match"
        )


def _load_learning_proposal(
    connection: sqlite3.Connection,
    agent_id: str,
    proposal_id: str,
) -> LearningProposal | None:
    connection.execute("BEGIN")
    try:
        proposal = _load_learning_proposal_in_transaction(
            connection,
            agent_id,
            proposal_id,
        )
        connection.execute("COMMIT")
        return proposal
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_learning_proposal_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    proposal_id: str,
) -> LearningProposal | None:
    row = connection.execute(
        "SELECT * FROM learning_proposals WHERE agent_id = ? AND id = ?",
        (agent_id, proposal_id),
    ).fetchone()
    if row is None:
        return None
    proposal = _decode_learning_proposal(row)
    _validate_learning_decision_history(connection, proposal)
    return proposal


def _list_learning_proposals(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    operation_id: str | None,
    states: tuple[LearningProposalState, ...],
    limit: int,
) -> tuple[LearningProposal, ...]:
    state_slots = ", ".join("?" for _ in states)
    filters = ["agent_id = ?", f"state IN ({state_slots})"]
    parameters: list[object] = [agent_id, *(state.value for state in states)]
    if operation_id is not None:
        filters.append("operation_id = ?")
        parameters.append(operation_id)
    rows = connection.execute(
        "SELECT * FROM learning_proposals "
        f"WHERE {' AND '.join(filters)} "
        "ORDER BY created_at DESC, id LIMIT ?",
        (*parameters, limit),
    ).fetchall()
    proposals = tuple(_decode_learning_proposal(row) for row in rows)
    for proposal in proposals:
        _validate_learning_decision_history(connection, proposal)
    return proposals


def _resolve_learning_proposal(
    connection: sqlite3.Connection,
    decision: LearningDecision,
    *,
    expected_state: LearningProposalState,
) -> LearningProposal:
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT * FROM learning_proposals WHERE id = ?",
            (decision.proposal_id,),
        ).fetchone()
        if row is None:
            raise LearningStoreConflictError(
                f"learning proposal does not exist: {decision.proposal_id}"
            )
        proposal = _decode_learning_proposal(row)
        _validate_learning_decision_history(connection, proposal)
        if proposal.state is not LearningProposalState.PROPOSED:
            try:
                replayed = resolve_learning_proposal(proposal, decision)
            except LearningTransitionError as error:
                raise LearningStoreConflictError(
                    "learning proposal already has another decision"
                ) from error
            connection.execute("COMMIT")
            return replayed
        if proposal.state is not expected_state:
            raise LearningStoreConflictError(
                f"learning proposal state changed: expected {expected_state.value}, "
                f"found {proposal.state.value}"
            )
        try:
            resolved = resolve_learning_proposal(proposal, decision)
        except LearningTransitionError as error:
            raise LearningStoreConflictError(
                "learning decision does not match the durable proposal"
            ) from error
        _insert_learning_decision(connection, decision)
        values = _learning_proposal_values(resolved)
        connection.execute(
            "UPDATE learning_proposals SET state = ?, candidate_payload_json = ?, "
            "resolved_at = ?, decision_hash = ?, result_memory_id = ?, "
            "result_memory_version = ?, result_skill_id = ?, "
            "result_skill_version = ?, rejection_category = ?, rejection_reason = ? "
            "WHERE id = ? AND state = ?",
            (
                values[4],
                values[13],
                values[15],
                values[16],
                values[17],
                values[18],
                values[19],
                values[20],
                values[21],
                values[22],
                proposal.id,
                expected_state.value,
            ),
        )
        changed_row = connection.execute("SELECT changes()").fetchone()
        assert changed_row is not None
        if _sqlite_int(changed_row[0], "learning resolution changes") != 1:
            raise LearningStoreConflictError(
                "learning proposal state changed during resolution"
            )
        stored = _load_learning_proposal_in_transaction(
            connection,
            proposal.provenance.agent_id,
            proposal.id,
        )
        assert stored is not None
        connection.execute("COMMIT")
        return stored
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise LearningStoreConflictError(
            "learning decision conflicts with durable history"
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _record_skill_discovery(
    connection: sqlite3.Connection,
    skill: Skill,
    version: SkillVersion,
    index: SkillIndex,
) -> SkillIndex:
    if (
        skill.agent_id != version.agent_id
        or skill.agent_id != index.agent_id
        or skill.id != version.skill_id
        or skill.id != index.skill_id
        or skill.stable_name != version.stable_name
        or skill.stable_name != index.stable_name
        or skill.source is not version.source
        or skill.source is not index.source
        or not index.matches(version)
    ):
        raise SkillDiscoveryError(
            "skill, version, and index do not describe one discovery"
        )
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != skill.agent_id:
            raise AgentIdentityConflictError(
                "skill agent does not match the authoritative database identity"
            )
        skill_row = connection.execute(
            "SELECT * FROM skills WHERE agent_id = ? AND (id = ? OR stable_name = ?) "
            "ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END LIMIT 1",
            (skill.agent_id, skill.id, skill.stable_name, skill.id),
        ).fetchone()
        if skill_row is None:
            connection.execute(
                "INSERT INTO skills(id, agent_id, stable_name, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    skill.id,
                    skill.agent_id,
                    skill.stable_name,
                    skill.source.value,
                    _encode_datetime(skill.created_at),
                ),
            )
        elif _decode_skill(skill_row) != skill:
            raise SkillDiscoveryError(
                "skill identity or stable name is already claimed"
            )
        version_row = connection.execute(
            "SELECT * FROM skill_versions WHERE agent_id = ? AND "
            "(id = ? OR (skill_id = ? AND version = ?)) "
            "ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END LIMIT 1",
            (
                skill.agent_id,
                version.id,
                skill.id,
                version.version,
                version.id,
            ),
        ).fetchone()
        if version_row is None:
            _insert_skill_version(connection, version)
        elif _decode_skill_version(version_row) != version:
            raise SkillDiscoveryError(
                "skill version identity or semantic version is already claimed"
            )
        current_row = connection.execute(
            "SELECT * FROM skill_indexes WHERE agent_id = ? AND skill_id = ?",
            (skill.agent_id, skill.id),
        ).fetchone()
        current = None if current_row is None else _decode_skill_index(current_row)
        if (
            current is not None
            and index.active_version_id is not None
            and index.active_version_id != current.active_version_id
        ):
            raise SkillDiscoveryError(
                "skill discovery cannot change the active version"
            )
        stored_index = (
            current
            if current is not None and current.active_version_id is not None
            else SkillIndex.from_version(
                version,
                active_version_id=None,
                updated_at=max(
                    index.updated_at,
                    current.updated_at if current is not None else index.updated_at,
                ),
            )
        )
        if current is None:
            connection.execute(
                "INSERT INTO skill_indexes("
                "agent_id, skill_id, version_id, stable_name, version, description, "
                "domains_json, resource_kinds_json, required_capability_ids_json, "
                "activation_mode, source, content_hash, active_version_id, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _skill_index_values(stored_index),
            )
        elif current != stored_index:
            connection.execute(
                "UPDATE skill_indexes SET version_id = ?, stable_name = ?, "
                "version = ?, description = ?, domains_json = ?, "
                "resource_kinds_json = ?, required_capability_ids_json = ?, "
                "activation_mode = ?, source = ?, content_hash = ?, "
                "active_version_id = ?, updated_at = ? "
                "WHERE agent_id = ? AND skill_id = ?",
                (
                    *_skill_index_values(stored_index)[2:],
                    stored_index.agent_id,
                    stored_index.skill_id,
                ),
            )
        connection.execute("COMMIT")
        return stored_index
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise SkillDiscoveryError(
            "skill discovery conflicts with durable identity or history"
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _decode_skill(row: sqlite3.Row) -> Skill:
    try:
        return Skill(
            id=_sqlite_text(row["id"], "skill id"),
            agent_id=_sqlite_text(row["agent_id"], "skill agent_id"),
            stable_name=_sqlite_text(row["stable_name"], "skill stable_name"),
            source=SkillSource(_sqlite_text(row["source"], "skill source")),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "skill created_at")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct skill identity") from error


def _insert_skill_version(
    connection: sqlite3.Connection,
    version: SkillVersion,
) -> None:
    connection.execute(
        "INSERT INTO skill_versions("
        "id, agent_id, skill_id, stable_name, version, description, domains_json, "
        "resource_kinds_json, required_capability_ids_json, activation_mode, "
        "sensitivity_notes, policy_notes, source, content_hash, instructions, "
        "source_path, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            version.id,
            version.agent_id,
            version.skill_id,
            version.stable_name,
            version.version,
            version.description,
            canonical_json(version.domains),
            canonical_json(version.resource_kinds),
            canonical_json(version.required_capability_ids),
            version.activation_mode.value,
            version.sensitivity_notes,
            version.policy_notes,
            version.source.value,
            version.content_hash,
            version.instructions,
            version.source_path,
            _encode_datetime(version.created_at),
        ),
    )


def _decode_skill_version(row: sqlite3.Row) -> SkillVersion:
    try:
        return SkillVersion(
            id=_sqlite_text(row["id"], "skill version id"),
            agent_id=_sqlite_text(row["agent_id"], "skill version agent_id"),
            skill_id=_sqlite_text(row["skill_id"], "skill version skill_id"),
            stable_name=_sqlite_text(
                row["stable_name"],
                "skill version stable_name",
            ),
            version=_sqlite_text(row["version"], "skill version"),
            description=_sqlite_text(
                row["description"],
                "skill version description",
            ),
            domains=_decode_string_tuple(
                _sqlite_text(row["domains_json"], "skill version domains")
            ),
            resource_kinds=_decode_string_tuple(
                _sqlite_text(
                    row["resource_kinds_json"],
                    "skill version resource kinds",
                )
            ),
            required_capability_ids=_decode_string_tuple(
                _sqlite_text(
                    row["required_capability_ids_json"],
                    "skill version capability ids",
                )
            ),
            activation_mode=SkillActivationMode(
                _sqlite_text(
                    row["activation_mode"],
                    "skill version activation mode",
                )
            ),
            sensitivity_notes=_optional_text(row["sensitivity_notes"]),
            policy_notes=_optional_text(row["policy_notes"]),
            source=SkillSource(_sqlite_text(row["source"], "skill version source")),
            content_hash=_sqlite_text(
                row["content_hash"],
                "skill version content_hash",
            ),
            instructions=_sqlite_text(
                row["instructions"],
                "skill version instructions",
            ),
            source_path=_optional_text(row["source_path"]),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "skill version created_at")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct skill version") from error


def _skill_index_values(index: SkillIndex) -> tuple[object, ...]:
    return (
        index.agent_id,
        index.skill_id,
        index.version_id,
        index.stable_name,
        index.version,
        index.description,
        canonical_json(index.domains),
        canonical_json(index.resource_kinds),
        canonical_json(index.required_capability_ids),
        index.activation_mode.value,
        index.source.value,
        index.content_hash,
        index.active_version_id,
        _encode_datetime(index.updated_at),
    )


def _decode_skill_index(row: sqlite3.Row) -> SkillIndex:
    try:
        return SkillIndex(
            agent_id=_sqlite_text(row["agent_id"], "skill index agent_id"),
            skill_id=_sqlite_text(row["skill_id"], "skill index skill_id"),
            version_id=_sqlite_text(row["version_id"], "skill index version_id"),
            stable_name=_sqlite_text(
                row["stable_name"],
                "skill index stable_name",
            ),
            version=_sqlite_text(row["version"], "skill index version"),
            description=_sqlite_text(
                row["description"],
                "skill index description",
            ),
            domains=_decode_string_tuple(
                _sqlite_text(row["domains_json"], "skill index domains")
            ),
            resource_kinds=_decode_string_tuple(
                _sqlite_text(
                    row["resource_kinds_json"],
                    "skill index resource kinds",
                )
            ),
            required_capability_ids=_decode_string_tuple(
                _sqlite_text(
                    row["required_capability_ids_json"],
                    "skill index capability ids",
                )
            ),
            activation_mode=SkillActivationMode(
                _sqlite_text(
                    row["activation_mode"],
                    "skill index activation mode",
                )
            ),
            source=SkillSource(_sqlite_text(row["source"], "skill index source")),
            content_hash=_sqlite_text(
                row["content_hash"],
                "skill index content_hash",
            ),
            active_version_id=_optional_text(row["active_version_id"]),
            updated_at=_decode_datetime(
                _sqlite_text(row["updated_at"], "skill index updated_at")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct skill index") from error


def _list_skill_index(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[SkillIndex, ...]:
    rows = connection.execute(
        "SELECT * FROM skill_indexes WHERE agent_id = ? "
        "ORDER BY stable_name, skill_id",
        (agent_id,),
    ).fetchall()
    return tuple(_decode_skill_index(row) for row in rows)


def _load_skill_index(
    connection: sqlite3.Connection,
    agent_id: str,
    skill_id: str,
) -> SkillIndex | None:
    row = connection.execute(
        "SELECT * FROM skill_indexes WHERE agent_id = ? AND skill_id = ?",
        (agent_id, skill_id),
    ).fetchone()
    return None if row is None else _decode_skill_index(row)


def _load_skill_version(
    connection: sqlite3.Connection,
    agent_id: str,
    version_id: str,
) -> SkillVersion | None:
    row = connection.execute(
        "SELECT * FROM skill_versions WHERE agent_id = ? AND id = ?",
        (agent_id, version_id),
    ).fetchone()
    return None if row is None else _decode_skill_version(row)


def _decode_skill_activation(row: sqlite3.Row) -> SkillActivation:
    try:
        return SkillActivation(
            id=_sqlite_text(row["id"], "skill activation id"),
            agent_id=_sqlite_text(row["agent_id"], "skill activation agent_id"),
            skill_id=_sqlite_text(row["skill_id"], "skill activation skill_id"),
            version_id=_sqlite_text(
                row["version_id"],
                "skill activation version_id",
            ),
            previous_version_id=_optional_text(row["previous_version_id"]),
            actor_id=_sqlite_text(row["actor_id"], "skill activation actor_id"),
            reason=_sqlite_text(row["reason"], "skill activation reason"),
            activated_at=_decode_datetime(
                _sqlite_text(row["activated_at"], "skill activation activated_at")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SQLiteCorruptionError("cannot reconstruct skill activation") from error


def _inspect_skill(
    connection: sqlite3.Connection,
    agent_id: str,
    skill_id: str,
) -> SkillInspection | None:
    skill_row = connection.execute(
        "SELECT * FROM skills WHERE agent_id = ? AND id = ?",
        (agent_id, skill_id),
    ).fetchone()
    if skill_row is None:
        return None
    index_row = connection.execute(
        "SELECT * FROM skill_indexes WHERE agent_id = ? AND skill_id = ?",
        (agent_id, skill_id),
    ).fetchone()
    if index_row is None:
        raise SQLiteCorruptionError("discovered skill is missing its index")
    version_rows = connection.execute(
        "SELECT * FROM skill_versions WHERE agent_id = ? AND skill_id = ? "
        "ORDER BY created_at, id",
        (agent_id, skill_id),
    ).fetchall()
    activation_rows = connection.execute(
        "SELECT * FROM skill_activations WHERE agent_id = ? AND skill_id = ? "
        "ORDER BY activated_at, id",
        (agent_id, skill_id),
    ).fetchall()
    try:
        return SkillInspection(
            skill=_decode_skill(skill_row),
            index=_decode_skill_index(index_row),
            versions=tuple(_decode_skill_version(row) for row in version_rows),
            activations=tuple(_decode_skill_activation(row) for row in activation_rows),
        )
    except SQLiteCorruptionError:
        raise
    except (TypeError, ValueError) as error:
        raise SQLiteCorruptionError(
            f"cannot reconstruct skill inspection {skill_id}"
        ) from error


def _activate_skill(
    connection: sqlite3.Connection,
    activation: SkillActivation,
    *,
    expected_active_version_id: str | None,
) -> SkillInspection:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != activation.agent_id:
            raise AgentIdentityConflictError(
                "skill activation agent does not match database identity"
            )
        existing_activation_row = connection.execute(
            "SELECT * FROM skill_activations WHERE id = ?",
            (activation.id,),
        ).fetchone()
        index = _load_skill_index(
            connection,
            activation.agent_id,
            activation.skill_id,
        )
        if index is None:
            raise SkillActivationConflictError(f"unknown skill: {activation.skill_id}")
        if existing_activation_row is not None:
            if (
                _decode_skill_activation(existing_activation_row) == activation
                and index.active_version_id == activation.version_id
            ):
                inspection = _inspect_skill(
                    connection,
                    activation.agent_id,
                    activation.skill_id,
                )
                assert inspection is not None
                connection.execute("COMMIT")
                return inspection
            raise SkillActivationConflictError(
                "skill activation identity is already claimed"
            )
        if (
            index.active_version_id != expected_active_version_id
            or activation.previous_version_id != expected_active_version_id
        ):
            raise SkillActivationConflictError(
                f"skill {activation.skill_id} active version changed"
            )
        version = _load_skill_version(
            connection,
            activation.agent_id,
            activation.version_id,
        )
        if version is None or version.skill_id != activation.skill_id:
            raise SkillActivationConflictError(
                f"unknown skill version: {activation.version_id}"
            )
        activated_index = SkillIndex.from_version(
            version,
            active_version_id=version.id,
            updated_at=max(index.updated_at, activation.activated_at),
        )
        connection.execute(
            "INSERT INTO skill_activations("
            "id, agent_id, skill_id, version_id, previous_version_id, actor_id, "
            "reason, activated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                activation.id,
                activation.agent_id,
                activation.skill_id,
                activation.version_id,
                activation.previous_version_id,
                activation.actor_id,
                activation.reason,
                _encode_datetime(activation.activated_at),
            ),
        )
        connection.execute(
            "UPDATE skill_indexes SET version_id = ?, stable_name = ?, "
            "version = ?, description = ?, domains_json = ?, "
            "resource_kinds_json = ?, required_capability_ids_json = ?, "
            "activation_mode = ?, source = ?, content_hash = ?, "
            "active_version_id = ?, updated_at = ? "
            "WHERE agent_id = ? AND skill_id = ? AND active_version_id IS ?",
            (
                *_skill_index_values(activated_index)[2:],
                activation.agent_id,
                activation.skill_id,
                expected_active_version_id,
            ),
        )
        changed_row = connection.execute("SELECT changes()").fetchone()
        assert changed_row is not None
        if _sqlite_int(changed_row[0], "skill activation changes") != 1:
            raise SkillActivationConflictError(
                f"skill {activation.skill_id} active version changed"
            )
        inspection = _inspect_skill(
            connection,
            activation.agent_id,
            activation.skill_id,
        )
        assert inspection is not None
        connection.execute("COMMIT")
        return inspection
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise SkillActivationConflictError(
            "skill activation conflicts with durable history"
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _commit_skill_change(
    connection: sqlite3.Connection,
    request: SkillChangeCommit,
) -> SkillChangeAcceptanceResult:
    agent_id = request.skill.agent_id
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != agent_id:
            raise SkillChangeConflictError(
                "skill change agent does not match database identity"
            )
        durable = _load_learning_proposal_in_transaction(
            connection,
            agent_id,
            request.proposal.id,
        )
        if durable is None:
            raise SkillChangeConflictError("skill-change proposal does not exist")
        if durable.state is LearningProposalState.COMMITTED:
            try:
                expected = resolve_learning_proposal(
                    request.proposal,
                    request.decision,
                )
            except LearningTransitionError as error:
                raise SkillChangeConflictError(
                    "skill-change replay decision does not match its proposal"
                ) from error
            if durable != expected:
                raise SkillChangeConflictError(
                    "skill-change proposal already has another decision"
                )
            inspection = _inspect_skill(connection, agent_id, request.skill.id)
            if inspection is None:
                raise SQLiteCorruptionError(
                    "committed skill change is missing its skill history"
                )
            stored_version = next(
                (item for item in inspection.versions if item.id == request.version.id),
                None,
            )
            stored_activation = next(
                (
                    item
                    for item in inspection.activations
                    if item.id == request.activation.id
                ),
                None,
            )
            if (
                stored_version != request.version
                or stored_activation != request.activation
            ):
                raise SkillChangeConflictError(
                    "skill-change replay does not match durable skill history"
                )
            result = SkillChangeAcceptanceResult(
                proposal=durable,
                inspection=inspection,
                replayed=True,
            )
            connection.execute("COMMIT")
            return result
        if durable.state is not LearningProposalState.PROPOSED or (
            durable != request.proposal
        ):
            raise SkillChangeConflictError(
                "skill-change proposal state or identity changed"
            )
        _validate_learning_runtime_references(connection, durable)

        expected_staged = SkillIndex.from_version(
            request.version,
            active_version_id=request.expected_active_version_id,
        )
        if request.staged_index != expected_staged:
            raise SkillChangeConflictError(
                "skill-change staged index is not the canonical proposal projection"
            )
        skill_row = connection.execute(
            "SELECT * FROM skills WHERE agent_id = ? AND (id = ? OR stable_name = ?) "
            "ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END LIMIT 1",
            (
                agent_id,
                request.skill.id,
                request.skill.stable_name,
                request.skill.id,
            ),
        ).fetchone()
        current_index: SkillIndex | None = None
        if skill_row is None:
            if request.expected_skill_version_count != 0 or (
                request.expected_active_version_id is not None
            ):
                raise SkillChangeConflictError(
                    "skill-change expected durable history for a new skill"
                )
        else:
            if _decode_skill(skill_row) != request.skill:
                raise SkillChangeConflictError(
                    "skill identity or stable name is already claimed"
                )
            inspection = _inspect_skill(connection, agent_id, request.skill.id)
            if inspection is None:
                raise SQLiteCorruptionError(
                    "durable skill identity is missing its history"
                )
            if len(inspection.versions) != request.expected_skill_version_count:
                raise SkillChangeConflictError(
                    "skill-change immutable version count changed"
                )
            current_index = inspection.index
            if current_index.active_version_id != request.expected_active_version_id:
                raise SkillChangeConflictError("skill-change active version changed")

        claimed_version = connection.execute(
            "SELECT * FROM skill_versions WHERE agent_id = ? AND "
            "(id = ? OR (skill_id = ? AND version = ?)) "
            "ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END LIMIT 1",
            (
                agent_id,
                request.version.id,
                request.skill.id,
                request.version.version,
                request.version.id,
            ),
        ).fetchone()
        if claimed_version is not None:
            raise SkillChangeConflictError(
                "skill-change version identity or semantic version is already claimed"
            )
        if skill_row is None:
            connection.execute(
                "INSERT INTO skills(id, agent_id, stable_name, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    request.skill.id,
                    agent_id,
                    request.skill.stable_name,
                    request.skill.source.value,
                    _encode_datetime(request.skill.created_at),
                ),
            )
        _insert_skill_version(connection, request.version)
        final_index = SkillIndex.from_version(
            request.version,
            active_version_id=request.version.id,
            updated_at=max(
                request.staged_index.updated_at,
                request.activation.activated_at,
            ),
        )
        if current_index is None:
            connection.execute(
                "INSERT INTO skill_indexes("
                "agent_id, skill_id, version_id, stable_name, version, description, "
                "domains_json, resource_kinds_json, required_capability_ids_json, "
                "activation_mode, source, content_hash, active_version_id, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _skill_index_values(final_index),
            )
        else:
            connection.execute(
                "UPDATE skill_indexes SET version_id = ?, stable_name = ?, "
                "version = ?, description = ?, domains_json = ?, "
                "resource_kinds_json = ?, required_capability_ids_json = ?, "
                "activation_mode = ?, source = ?, content_hash = ?, "
                "active_version_id = ?, updated_at = ? "
                "WHERE agent_id = ? AND skill_id = ? AND active_version_id IS ?",
                (
                    *_skill_index_values(final_index)[2:],
                    agent_id,
                    request.skill.id,
                    request.expected_active_version_id,
                ),
            )
            changed_row = connection.execute("SELECT changes()").fetchone()
            assert changed_row is not None
            if _sqlite_int(changed_row[0], "skill-change index changes") != 1:
                raise SkillChangeConflictError(
                    "skill-change active version changed during commit"
                )
        connection.execute(
            "INSERT INTO skill_activations("
            "id, agent_id, skill_id, version_id, previous_version_id, actor_id, "
            "reason, activated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request.activation.id,
                request.activation.agent_id,
                request.activation.skill_id,
                request.activation.version_id,
                request.activation.previous_version_id,
                request.activation.actor_id,
                request.activation.reason,
                _encode_datetime(request.activation.activated_at),
            ),
        )
        try:
            resolved = resolve_learning_proposal(durable, request.decision)
        except LearningTransitionError as error:
            raise SkillChangeConflictError(
                "skill-change decision does not match its proposal"
            ) from error
        _insert_learning_decision(connection, request.decision)
        values = _learning_proposal_values(resolved)
        connection.execute(
            "UPDATE learning_proposals SET state = ?, candidate_payload_json = ?, "
            "resolved_at = ?, decision_hash = ?, result_memory_id = ?, "
            "result_memory_version = ?, result_skill_id = ?, "
            "result_skill_version = ?, rejection_category = ?, rejection_reason = ? "
            "WHERE id = ? AND agent_id = ? AND state = ?",
            (
                values[4],
                values[13],
                values[15],
                values[16],
                values[17],
                values[18],
                values[19],
                values[20],
                values[21],
                values[22],
                durable.id,
                agent_id,
                LearningProposalState.PROPOSED.value,
            ),
        )
        changed_row = connection.execute("SELECT changes()").fetchone()
        assert changed_row is not None
        if _sqlite_int(changed_row[0], "skill-change proposal changes") != 1:
            raise SkillChangeConflictError(
                "skill-change proposal changed during commit"
            )
        inspection = _inspect_skill(connection, agent_id, request.skill.id)
        if inspection is None:
            raise SQLiteCorruptionError(
                "accepted skill change is missing its committed history"
            )
        stored = _load_learning_proposal_in_transaction(
            connection,
            agent_id,
            durable.id,
        )
        if stored is None or stored != resolved:
            raise SQLiteCorruptionError(
                "accepted skill change proposal did not round-trip"
            )
        result = SkillChangeAcceptanceResult(
            proposal=stored,
            inspection=inspection,
            replayed=False,
        )
        connection.execute("COMMIT")
        return result
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise SkillChangeConflictError(
            "skill change conflicts with durable proposal or skill history"
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _require_monitor_datetime(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


def _validate_monitor_event(
    event: RuntimeEvent,
    agent_id: str,
    monitor_id: str,
) -> None:
    if not isinstance(event, RuntimeEvent):
        raise TypeError("monitor event must be a RuntimeEvent")
    if event.agent_id != agent_id or event.monitor_id != monitor_id:
        raise ValueError("monitor event scope does not match its lifecycle record")


def _monitor_definition_data(definition: MonitorDefinition) -> dict[str, object]:
    schedule: dict[str, object]
    if isinstance(definition.schedule, IntervalSchedule):
        schedule = {
            "anchor_at": _encode_datetime(definition.schedule.anchor_at),
            "interval_seconds": definition.schedule.interval_seconds,
            "kind": "interval",
        }
    else:
        schedule = {
            "expression": definition.schedule.expression,
            "kind": "cron",
            "timezone_name": definition.schedule.timezone_name,
        }
    return {
        "budget_overrides": {
            "max_capability_calls": definition.budget_overrides.max_capability_calls,
            "max_turns": definition.budget_overrides.max_turns,
            "max_wall_time_seconds": definition.budget_overrides.max_wall_time_seconds,
        },
        "condition": {
            "configuration": definition.condition.configuration,
            "expression": definition.condition.expression,
            "kind": definition.condition.kind.value,
        },
        "name": definition.name,
        "objective": definition.objective,
        "operation_template": definition.operation_template,
        "policy_overrides": definition.policy_overrides,
        "schedule": schedule,
        "scope": {
            "resource_ids": definition.scope.resource_ids,
            "source_ids": definition.scope.source_ids,
        },
        "timing": {
            "backoff_multiplier": definition.timing.backoff_multiplier,
            "catch_up": definition.timing.catch_up.value,
            "cooldown_seconds": definition.timing.cooldown_seconds,
            "initial_backoff_seconds": definition.timing.initial_backoff_seconds,
            "max_backoff_seconds": definition.timing.max_backoff_seconds,
        },
    }


def _monitor_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SQLiteCorruptionError(f"{field_name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise SQLiteCorruptionError(f"{field_name} contains a non-string key")
    return value


def _monitor_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise SQLiteCorruptionError(f"{field_name} must be text")
    return value


def _monitor_optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _monitor_text(value, field_name)


def _monitor_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SQLiteCorruptionError(f"{field_name} must be an integer")
    return value


def _monitor_optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _monitor_int(value, field_name)


def _monitor_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SQLiteCorruptionError(f"{field_name} must be numeric")
    return float(value)


def _monitor_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SQLiteCorruptionError(f"{field_name} must be a JSON string array")
    return tuple(value)


def _decode_monitor_definition(value: str) -> MonitorDefinition:
    data = _monitor_mapping(_decode_json_object(value), "monitor definition")
    schedule_data = _monitor_mapping(data.get("schedule"), "monitor schedule")
    schedule_kind = _monitor_text(schedule_data.get("kind"), "schedule kind")
    schedule: IntervalSchedule | CronSchedule
    if schedule_kind == "interval":
        schedule = IntervalSchedule(
            interval_seconds=_monitor_int(
                schedule_data.get("interval_seconds"),
                "interval seconds",
            ),
            anchor_at=_decode_datetime(
                _monitor_text(schedule_data.get("anchor_at"), "schedule anchor")
            ),
        )
    elif schedule_kind == "cron":
        schedule = CronSchedule(
            expression=_monitor_text(
                schedule_data.get("expression"),
                "cron expression",
            ),
            timezone_name=_monitor_text(
                schedule_data.get("timezone_name"),
                "cron timezone",
            ),
        )
    else:
        raise SQLiteCorruptionError(f"unsupported monitor schedule: {schedule_kind}")
    scope_data = _monitor_mapping(data.get("scope"), "monitor scope")
    condition_data = _monitor_mapping(data.get("condition"), "monitor condition")
    budget_data = _monitor_mapping(
        data.get("budget_overrides"),
        "monitor budget overrides",
    )
    timing_data = _monitor_mapping(data.get("timing"), "monitor timing")
    return MonitorDefinition(
        name=_monitor_text(data.get("name"), "monitor name"),
        objective=_monitor_text(data.get("objective"), "monitor objective"),
        scope=MonitorScope(
            source_ids=_monitor_string_tuple(
                scope_data.get("source_ids"),
                "monitor source scope",
            ),
            resource_ids=_monitor_string_tuple(
                scope_data.get("resource_ids"),
                "monitor resource scope",
            ),
        ),
        schedule=schedule,
        condition=MonitorCondition(
            kind=MonitorConditionKind(
                _monitor_text(condition_data.get("kind"), "condition kind")
            ),
            expression=_monitor_optional_text(
                condition_data.get("expression"),
                "condition expression",
            ),
            configuration=_monitor_mapping(
                condition_data.get("configuration"),
                "condition configuration",
            ),
        ),
        budget_overrides=MonitorBudgetOverrides(
            max_turns=_monitor_optional_int(
                budget_data.get("max_turns"),
                "budget max_turns",
            ),
            max_capability_calls=_monitor_optional_int(
                budget_data.get("max_capability_calls"),
                "budget max_capability_calls",
            ),
            max_wall_time_seconds=_monitor_optional_int(
                budget_data.get("max_wall_time_seconds"),
                "budget max_wall_time_seconds",
            ),
        ),
        timing=MonitorTimingPolicy(
            catch_up=CatchUpPolicy(
                _monitor_text(timing_data.get("catch_up"), "catch-up policy")
            ),
            cooldown_seconds=_monitor_int(
                timing_data.get("cooldown_seconds"),
                "cooldown seconds",
            ),
            initial_backoff_seconds=_monitor_int(
                timing_data.get("initial_backoff_seconds"),
                "initial backoff seconds",
            ),
            max_backoff_seconds=_monitor_int(
                timing_data.get("max_backoff_seconds"),
                "max backoff seconds",
            ),
            backoff_multiplier=_monitor_float(
                timing_data.get("backoff_multiplier"),
                "backoff multiplier",
            ),
        ),
        policy_overrides=_monitor_mapping(
            data.get("policy_overrides"),
            "monitor policy overrides",
        ),
        operation_template=_monitor_mapping(
            data.get("operation_template"),
            "monitor operation template",
        ),
    )


def _monitor_proposal_values(proposal: MonitorProposal) -> tuple[object, ...]:
    return (
        proposal.id,
        proposal.agent_id,
        proposal.intended_monitor_id,
        proposal.idempotency_key,
        proposal.candidate_hash,
        canonical_json(_monitor_definition_data(proposal.candidate)),
        proposal.source_operation_id,
        _encode_datetime(proposal.created_at),
    )


def _decode_monitor_proposal_row(row: sqlite3.Row) -> MonitorProposal:
    return MonitorProposal(
        id=_sqlite_text(row["id"], "monitor proposal id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor proposal agent_id"),
        intended_monitor_id=_sqlite_text(
            row["intended_monitor_id"],
            "intended monitor id",
        ),
        idempotency_key=_sqlite_text(
            row["idempotency_key"],
            "monitor proposal idempotency key",
        ),
        candidate=_decode_monitor_definition(
            _sqlite_text(row["candidate_json"], "monitor proposal candidate")
        ),
        candidate_hash=_sqlite_text(
            row["candidate_hash"],
            "monitor proposal candidate hash",
        ),
        source_operation_id=_optional_text(row["source_operation_id"]),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "monitor proposal created_at")
        ),
    )


def _monitor_confirmation_values(
    confirmation: MonitorConfirmation,
) -> tuple[object, ...]:
    return (
        confirmation.id,
        confirmation.agent_id,
        confirmation.proposal_id,
        confirmation.decision.value,
        confirmation.candidate_hash,
        confirmation.actor_id,
        confirmation.reason,
        _encode_datetime(confirmation.decided_at),
        confirmation.resulting_monitor_id,
        confirmation.resulting_version_id,
    )


def _decode_monitor_confirmation_row(row: sqlite3.Row) -> MonitorConfirmation:
    return MonitorConfirmation(
        id=_sqlite_text(row["id"], "monitor confirmation id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor confirmation agent_id"),
        proposal_id=_sqlite_text(row["proposal_id"], "monitor proposal id"),
        decision=MonitorConfirmationDecision(
            _sqlite_text(row["decision"], "monitor confirmation decision")
        ),
        candidate_hash=_sqlite_text(
            row["candidate_hash"],
            "monitor confirmation candidate hash",
        ),
        actor_id=_sqlite_text(row["actor_id"], "monitor confirmation actor"),
        reason=_sqlite_text(row["reason"], "monitor confirmation reason"),
        decided_at=_decode_datetime(
            _sqlite_text(row["decided_at"], "monitor confirmation decided_at")
        ),
        resulting_monitor_id=_optional_text(row["resulting_monitor_id"]),
        resulting_version_id=_optional_text(row["resulting_version_id"]),
    )


def _monitor_values(monitor: Monitor) -> tuple[object, ...]:
    return (
        monitor.id,
        monitor.agent_id,
        monitor.status.value,
        monitor.current_version,
        monitor.revision,
        _encode_datetime(monitor.created_at),
        _encode_datetime(monitor.updated_at),
        None if monitor.paused_at is None else _encode_datetime(monitor.paused_at),
        None if monitor.deleted_at is None else _encode_datetime(monitor.deleted_at),
    )


def _decode_monitor_row(row: sqlite3.Row) -> Monitor:
    return Monitor(
        id=_sqlite_text(row["id"], "monitor id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor agent_id"),
        status=MonitorStatus(_sqlite_text(row["status"], "monitor status")),
        current_version=_sqlite_int(row["current_version"], "monitor current version"),
        revision=_sqlite_int(row["revision"], "monitor revision"),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "monitor created_at")
        ),
        updated_at=_decode_datetime(
            _sqlite_text(row["updated_at"], "monitor updated_at")
        ),
        paused_at=(
            None
            if row["paused_at"] is None
            else _decode_datetime(_sqlite_text(row["paused_at"], "monitor paused_at"))
        ),
        deleted_at=(
            None
            if row["deleted_at"] is None
            else _decode_datetime(_sqlite_text(row["deleted_at"], "monitor deleted_at"))
        ),
    )


def _monitor_version_values(version: MonitorVersion) -> tuple[object, ...]:
    return (
        version.id,
        version.agent_id,
        version.monitor_id,
        version.version,
        canonical_json(_monitor_definition_data(version.definition)),
        version.content_hash,
        version.proposal_id,
        version.source_operation_id,
        _encode_datetime(version.created_at),
    )


def _decode_monitor_version_row(row: sqlite3.Row) -> MonitorVersion:
    return MonitorVersion(
        id=_sqlite_text(row["id"], "monitor version id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor version agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor version monitor_id"),
        version=_sqlite_int(row["version"], "monitor version"),
        definition=_decode_monitor_definition(
            _sqlite_text(row["definition_json"], "monitor definition")
        ),
        content_hash=_sqlite_text(row["content_hash"], "monitor version hash"),
        proposal_id=_sqlite_text(row["proposal_id"], "monitor version proposal id"),
        source_operation_id=_optional_text(row["source_operation_id"]),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "monitor version created_at")
        ),
    )


def _monitor_lifecycle_values(
    lifecycle: MonitorLifecycleRecord,
) -> tuple[object, ...]:
    return (
        lifecycle.id,
        lifecycle.agent_id,
        lifecycle.monitor_id,
        lifecycle.action.value,
        None if lifecycle.from_status is None else lifecycle.from_status.value,
        lifecycle.to_status.value,
        lifecycle.from_revision,
        lifecycle.to_revision,
        lifecycle.monitor_version,
        lifecycle.actor_id,
        lifecycle.reason,
        lifecycle.idempotency_key,
        _encode_datetime(lifecycle.occurred_at),
        lifecycle.operation_id,
    )


def _decode_monitor_lifecycle_row(row: sqlite3.Row) -> MonitorLifecycleRecord:
    return MonitorLifecycleRecord(
        id=_sqlite_text(row["id"], "monitor lifecycle id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor lifecycle agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor lifecycle monitor_id"),
        action=MonitorLifecycleAction(
            _sqlite_text(row["action"], "monitor lifecycle action")
        ),
        from_status=(
            None
            if row["from_status"] is None
            else MonitorStatus(
                _sqlite_text(row["from_status"], "monitor lifecycle from status")
            )
        ),
        to_status=MonitorStatus(
            _sqlite_text(row["to_status"], "monitor lifecycle to status")
        ),
        from_revision=_sqlite_int(
            row["from_revision"],
            "monitor lifecycle from revision",
        ),
        to_revision=_sqlite_int(
            row["to_revision"],
            "monitor lifecycle to revision",
        ),
        monitor_version=_sqlite_int(
            row["monitor_version"],
            "monitor lifecycle version",
        ),
        actor_id=_sqlite_text(row["actor_id"], "monitor lifecycle actor"),
        reason=_sqlite_text(row["reason"], "monitor lifecycle reason"),
        idempotency_key=_sqlite_text(
            row["idempotency_key"],
            "monitor lifecycle idempotency key",
        ),
        occurred_at=_decode_datetime(
            _sqlite_text(row["occurred_at"], "monitor lifecycle occurred_at")
        ),
        operation_id=_optional_text(row["operation_id"]),
    )


def _monitor_schedule_state_values(
    state: MonitorScheduleState,
) -> tuple[object, ...]:
    return (
        state.agent_id,
        state.monitor_id,
        state.revision,
        (
            None
            if state.next_scheduled_at is None
            else _encode_datetime(state.next_scheduled_at)
        ),
        _encode_datetime(state.updated_at),
        (
            None
            if state.last_scheduled_at is None
            else _encode_datetime(state.last_scheduled_at)
        ),
        (
            None
            if state.cooldown_until is None
            else _encode_datetime(state.cooldown_until)
        ),
        None if state.backoff_until is None else _encode_datetime(state.backoff_until),
        state.consecutive_failures,
        state.consecutive_matches,
        state.checkpoint_version,
        state.last_occurrence_id,
        state.last_run_id,
        state.last_operation_id,
    )


def _decode_monitor_schedule_state_row(row: sqlite3.Row) -> MonitorScheduleState:
    return MonitorScheduleState(
        agent_id=_sqlite_text(row["agent_id"], "monitor state agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor state monitor_id"),
        revision=_sqlite_int(row["revision"], "monitor state revision"),
        next_scheduled_at=_decode_optional_datetime(
            row["next_scheduled_at"],
            "monitor next scheduled_at",
        ),
        updated_at=_decode_datetime(
            _sqlite_text(row["updated_at"], "monitor state updated_at")
        ),
        last_scheduled_at=_decode_optional_datetime(
            row["last_scheduled_at"],
            "monitor last scheduled_at",
        ),
        cooldown_until=_decode_optional_datetime(
            row["cooldown_until"],
            "monitor cooldown_until",
        ),
        backoff_until=_decode_optional_datetime(
            row["backoff_until"],
            "monitor backoff_until",
        ),
        consecutive_failures=_sqlite_int(
            row["consecutive_failures"],
            "monitor consecutive failures",
        ),
        consecutive_matches=_sqlite_int(
            row["consecutive_matches"],
            "monitor consecutive matches",
        ),
        checkpoint_version=_sqlite_int(
            row["checkpoint_version"],
            "monitor checkpoint version",
        ),
        last_occurrence_id=_optional_text(row["last_occurrence_id"]),
        last_run_id=_optional_text(row["last_run_id"]),
        last_operation_id=_optional_text(row["last_operation_id"]),
    )


def _monitor_occurrence_values(
    occurrence: MonitorOccurrence,
) -> tuple[object, ...]:
    return (
        occurrence.id,
        occurrence.agent_id,
        occurrence.monitor_id,
        occurrence.monitor_version,
        occurrence.kind.value,
        _encode_datetime(occurrence.scheduled_for),
        occurrence.occurrence_key,
        occurrence.trigger_id,
        occurrence.run_id,
        _encode_datetime(occurrence.created_at),
        occurrence.manual_key,
    )


def _decode_monitor_occurrence_row(row: sqlite3.Row) -> MonitorOccurrence:
    return MonitorOccurrence(
        id=_sqlite_text(row["id"], "monitor occurrence id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor occurrence agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor occurrence monitor_id"),
        monitor_version=_sqlite_int(
            row["monitor_version"],
            "monitor occurrence version",
        ),
        kind=MonitorOccurrenceKind(
            _sqlite_text(row["kind"], "monitor occurrence kind")
        ),
        scheduled_for=_decode_datetime(
            _sqlite_text(row["scheduled_for"], "monitor occurrence scheduled_for")
        ),
        occurrence_key=_sqlite_text(
            row["occurrence_key"],
            "monitor occurrence key",
        ),
        trigger_id=_sqlite_text(row["trigger_id"], "monitor trigger id"),
        run_id=_sqlite_text(row["run_id"], "monitor run id"),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "monitor occurrence created_at")
        ),
        manual_key=_optional_text(row["manual_key"]),
    )


def _monitor_lease_values(lease: MonitorTickLease) -> tuple[object, ...]:
    return (
        lease.id,
        lease.agent_id,
        lease.monitor_id,
        lease.occurrence_id,
        lease.holder_id,
        lease.fencing_token,
        _encode_datetime(lease.claimed_at),
        _encode_datetime(lease.expires_at),
        None if lease.released_at is None else _encode_datetime(lease.released_at),
        lease.release_reason,
    )


def _decode_monitor_lease_row(row: sqlite3.Row) -> MonitorTickLease:
    return MonitorTickLease(
        id=_sqlite_text(row["id"], "monitor lease id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor lease agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor lease monitor_id"),
        occurrence_id=_sqlite_text(
            row["occurrence_id"],
            "monitor lease occurrence_id",
        ),
        holder_id=_sqlite_text(row["holder_id"], "monitor lease holder_id"),
        fencing_token=_sqlite_int(row["fencing_token"], "monitor lease fence"),
        claimed_at=_decode_datetime(
            _sqlite_text(row["claimed_at"], "monitor lease claimed_at")
        ),
        expires_at=_decode_datetime(
            _sqlite_text(row["expires_at"], "monitor lease expires_at")
        ),
        released_at=_decode_optional_datetime(
            row["released_at"],
            "monitor lease released_at",
        ),
        release_reason=_optional_text(row["release_reason"]),
    )


def _monitor_run_values(run: MonitorRun) -> tuple[object, ...]:
    return (
        run.id,
        run.agent_id,
        run.monitor_id,
        run.occurrence_id,
        run.trigger_id,
        run.attempt,
        run.fencing_token,
        run.status.value,
        _encode_datetime(run.started_at),
        run.operation_id,
        None if run.completed_at is None else _encode_datetime(run.completed_at),
        run.failure_reason,
    )


def _decode_monitor_run_row(row: sqlite3.Row) -> MonitorRun:
    return MonitorRun(
        id=_sqlite_text(row["id"], "monitor run id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor run agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor run monitor_id"),
        occurrence_id=_sqlite_text(
            row["occurrence_id"],
            "monitor run occurrence_id",
        ),
        trigger_id=_sqlite_text(row["trigger_id"], "monitor run trigger_id"),
        attempt=_sqlite_int(row["attempt"], "monitor run attempt"),
        fencing_token=_sqlite_int(row["fencing_token"], "monitor run fence"),
        status=MonitorRunStatus(_sqlite_text(row["status"], "monitor run status")),
        started_at=_decode_datetime(
            _sqlite_text(row["started_at"], "monitor run started_at")
        ),
        operation_id=_optional_text(row["operation_id"]),
        completed_at=_decode_optional_datetime(
            row["completed_at"],
            "monitor run completed_at",
        ),
        failure_reason=_optional_text(row["failure_reason"]),
    )


def _monitor_checkpoint_values(
    checkpoint: MonitorCheckpoint,
) -> tuple[object, ...]:
    return (
        checkpoint.id,
        checkpoint.agent_id,
        checkpoint.monitor_id,
        checkpoint.version,
        checkpoint.run_id,
        canonical_json(checkpoint.cursor),
        checkpoint.cursor_hash,
        _encode_datetime(checkpoint.created_at),
        checkpoint.previous_version,
    )


def _decode_monitor_checkpoint_row(row: sqlite3.Row) -> MonitorCheckpoint:
    return MonitorCheckpoint(
        id=_sqlite_text(row["id"], "monitor checkpoint id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor checkpoint agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor checkpoint monitor_id"),
        version=_sqlite_int(row["version"], "monitor checkpoint version"),
        run_id=_sqlite_text(row["run_id"], "monitor checkpoint run_id"),
        cursor=_decode_json_object(
            _sqlite_text(row["cursor_json"], "monitor checkpoint cursor")
        ),
        cursor_hash=_sqlite_text(
            row["cursor_hash"],
            "monitor checkpoint cursor hash",
        ),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "monitor checkpoint created_at")
        ),
        previous_version=(
            None
            if row["previous_version"] is None
            else _sqlite_int(row["previous_version"], "previous checkpoint version")
        ),
    )


def _monitor_finding_values(finding: MonitorFinding) -> tuple[object, ...]:
    return (
        finding.id,
        finding.agent_id,
        finding.monitor_id,
        finding.occurrence_id,
        finding.run_id,
        finding.operation_id,
        finding.evidence_id,
        finding.severity.value,
        finding.summary,
        canonical_json(finding.details),
        finding.dedupe_key,
        _encode_datetime(finding.created_at),
    )


def _decode_monitor_finding_row(row: sqlite3.Row) -> MonitorFinding:
    return MonitorFinding(
        id=_sqlite_text(row["id"], "monitor finding id"),
        agent_id=_sqlite_text(row["agent_id"], "monitor finding agent_id"),
        monitor_id=_sqlite_text(row["monitor_id"], "monitor finding monitor_id"),
        occurrence_id=_sqlite_text(
            row["occurrence_id"],
            "monitor finding occurrence_id",
        ),
        run_id=_sqlite_text(row["run_id"], "monitor finding run_id"),
        operation_id=_sqlite_text(
            row["operation_id"],
            "monitor finding operation_id",
        ),
        evidence_id=_sqlite_text(row["evidence_id"], "monitor finding evidence_id"),
        severity=MonitorFindingSeverity(
            _sqlite_text(row["severity"], "monitor finding severity")
        ),
        summary=_sqlite_text(row["summary"], "monitor finding summary"),
        details=_decode_json_object(
            _sqlite_text(row["details_json"], "monitor finding details")
        ),
        dedupe_key=_sqlite_text(row["dedupe_key"], "monitor finding dedupe key"),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "monitor finding created_at")
        ),
    )


def _insert_monitor_version(
    connection: sqlite3.Connection,
    version: MonitorVersion,
) -> None:
    connection.execute(
        "INSERT INTO monitor_versions("
        "id, agent_id, monitor_id, version, definition_json, content_hash, "
        "proposal_id, source_operation_id, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _monitor_version_values(version),
    )


def _insert_monitor_lifecycle(
    connection: sqlite3.Connection,
    lifecycle: MonitorLifecycleRecord,
) -> None:
    connection.execute(
        "INSERT INTO monitor_lifecycle("
        "id, agent_id, monitor_id, action, from_status, to_status, "
        "from_revision, to_revision, monitor_version, actor_id, reason, "
        "idempotency_key, occurred_at, operation_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _monitor_lifecycle_values(lifecycle),
    )


def _insert_monitor_schedule_state(
    connection: sqlite3.Connection,
    state: MonitorScheduleState,
) -> None:
    connection.execute(
        "INSERT INTO monitor_schedule_state("
        "agent_id, monitor_id, revision, next_scheduled_at, updated_at, "
        "last_scheduled_at, cooldown_until, backoff_until, consecutive_failures, "
        "consecutive_matches, checkpoint_version, last_occurrence_id, last_run_id, "
        "last_operation_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _monitor_schedule_state_values(state),
    )


def _load_monitor_proposal_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    proposal_id: str,
) -> MonitorProposal | None:
    row = connection.execute(
        "SELECT * FROM monitor_proposals WHERE agent_id = ? AND id = ?",
        (agent_id, proposal_id),
    ).fetchone()
    return None if row is None else _decode_monitor_proposal_row(row)


def _load_monitor_confirmation_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    proposal_id: str,
) -> MonitorConfirmation | None:
    row = connection.execute(
        "SELECT * FROM monitor_confirmations " "WHERE agent_id = ? AND proposal_id = ?",
        (agent_id, proposal_id),
    ).fetchone()
    return None if row is None else _decode_monitor_confirmation_row(row)


def _load_monitor_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    monitor_id: str,
) -> Monitor | None:
    row = connection.execute(
        "SELECT * FROM monitors WHERE agent_id = ? AND id = ?",
        (agent_id, monitor_id),
    ).fetchone()
    return None if row is None else _decode_monitor_row(row)


def _load_monitor_state_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    monitor_id: str,
) -> MonitorScheduleState | None:
    row = connection.execute(
        "SELECT * FROM monitor_schedule_state WHERE agent_id = ? AND monitor_id = ?",
        (agent_id, monitor_id),
    ).fetchone()
    return None if row is None else _decode_monitor_schedule_state_row(row)


def _inspect_monitor_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    monitor_id: str,
) -> MonitorInspection | None:
    monitor = _load_monitor_in_transaction(connection, agent_id, monitor_id)
    if monitor is None:
        return None
    state = _load_monitor_state_in_transaction(connection, agent_id, monitor_id)
    if state is None:
        raise SQLiteCorruptionError(f"monitor {monitor_id} is missing schedule state")
    versions = tuple(
        _decode_monitor_version_row(row)
        for row in connection.execute(
            "SELECT * FROM monitor_versions WHERE agent_id = ? AND monitor_id = ? "
            "ORDER BY version, id",
            (agent_id, monitor_id),
        ).fetchall()
    )
    lifecycle = tuple(
        _decode_monitor_lifecycle_row(row)
        for row in connection.execute(
            "SELECT * FROM monitor_lifecycle WHERE agent_id = ? AND monitor_id = ? "
            "ORDER BY to_revision, id",
            (agent_id, monitor_id),
        ).fetchall()
    )
    proposal_rows = connection.execute(
        "SELECT proposal.* FROM monitor_proposals AS proposal "
        "JOIN monitor_versions AS version ON version.proposal_id = proposal.id "
        "AND version.agent_id = proposal.agent_id "
        "WHERE version.agent_id = ? AND version.monitor_id = ? "
        "ORDER BY proposal.created_at, proposal.id",
        (agent_id, monitor_id),
    ).fetchall()
    proposals = tuple(_decode_monitor_proposal_row(row) for row in proposal_rows)
    proposal_ids = tuple(proposal.id for proposal in proposals)
    confirmations: tuple[MonitorConfirmation, ...]
    if proposal_ids:
        placeholders = ",".join("?" for _ in proposal_ids)
        confirmations = tuple(
            _decode_monitor_confirmation_row(row)
            for row in connection.execute(
                "SELECT * FROM monitor_confirmations WHERE agent_id = ? "
                f"AND proposal_id IN ({placeholders}) ORDER BY decided_at, id",
                (agent_id, *proposal_ids),
            ).fetchall()
        )
    else:
        confirmations = ()
    occurrences = tuple(
        _decode_monitor_occurrence_row(row)
        for row in connection.execute(
            "SELECT * FROM monitor_occurrences WHERE agent_id = ? AND monitor_id = ? "
            "ORDER BY scheduled_for, id",
            (agent_id, monitor_id),
        ).fetchall()
    )
    leases = tuple(
        _decode_monitor_lease_row(row)
        for row in connection.execute(
            "SELECT * FROM monitor_tick_leases WHERE agent_id = ? AND monitor_id = ? "
            "ORDER BY claimed_at, fencing_token, id",
            (agent_id, monitor_id),
        ).fetchall()
    )
    runs = tuple(
        _decode_monitor_run_row(row)
        for row in connection.execute(
            "SELECT * FROM monitor_runs WHERE agent_id = ? AND monitor_id = ? "
            "ORDER BY started_at, id",
            (agent_id, monitor_id),
        ).fetchall()
    )
    findings = tuple(
        _decode_monitor_finding_row(row)
        for row in connection.execute(
            "SELECT * FROM monitor_findings WHERE agent_id = ? AND monitor_id = ? "
            "ORDER BY created_at, id",
            (agent_id, monitor_id),
        ).fetchall()
    )
    checkpoints = tuple(
        _decode_monitor_checkpoint_row(row)
        for row in connection.execute(
            "SELECT * FROM monitor_checkpoints WHERE agent_id = ? AND monitor_id = ? "
            "ORDER BY version, id",
            (agent_id, monitor_id),
        ).fetchall()
    )
    return MonitorInspection(
        monitor=monitor,
        versions=versions,
        lifecycle=lifecycle,
        schedule_state=state,
        proposals=proposals,
        confirmations=confirmations,
        occurrences=occurrences,
        leases=leases,
        runs=runs,
        findings=findings,
        checkpoints=checkpoints,
    )


def _insert_standalone_monitor_event(
    connection: sqlite3.Connection,
    event: RuntimeEvent,
) -> None:
    if event.operation_id is not None:
        raise ValueError("standalone monitor event cannot identify an operation")
    if any(
        value is not None
        for value in (
            event.session_id,
            event.turn_id,
            event.model_call_id,
            event.call_id,
            event.task_id,
            event.evidence_id,
            event.approval_id,
        )
    ):
        raise ValueError("standalone monitor event has operation-owned correlation")
    connection.execute(
        "INSERT INTO runtime_events("
        "id, operation_id, position, type, agent_id, agent_sequence, created_at, "
        "session_id, turn_id, model_call_id, call_id, task_id, evidence_id, "
        "capability_id, executor_id, payload_json, approval_id, monitor_id"
        ") VALUES (?, NULL, NULL, ?, ?, ("
        "SELECT COALESCE(MAX(agent_sequence), 0) + 1 FROM runtime_events "
        "WHERE agent_id = ?"
        "), ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, NULL, ?)",
        (
            event.id,
            event.type,
            event.agent_id,
            event.agent_id,
            _encode_datetime(event.created_at),
            event.capability_id,
            event.executor_id,
            canonical_json(event.payload),
            event.monitor_id,
        ),
    )


def _append_monitor_events_in_transaction(
    connection: sqlite3.Connection,
    events: tuple[RuntimeEvent, ...],
) -> None:
    for event in events:
        if event.operation_id is None:
            _insert_standalone_monitor_event(connection, event)
            continue
        current = _load_versioned_operation_in_transaction(
            connection,
            event.operation_id,
        )
        snapshot = current.snapshot
        candidate_operation = replace(
            snapshot.operation,
            updated_at=max(snapshot.operation.updated_at, event.created_at),
        )
        candidate = replace(
            snapshot,
            operation=candidate_operation,
            events=(*snapshot.events, event),
        )
        _validate_commit_candidate(snapshot, candidate)
        _apply_commit_delta(
            connection,
            snapshot,
            candidate,
            expected_revision=current.revision,
            candidate_revision=current.revision + 1,
        )


def _create_monitor_proposal(
    connection: sqlite3.Connection,
    proposal: MonitorProposal,
    event: RuntimeEvent,
) -> tuple[MonitorProposal, tuple[RuntimeEvent, ...]]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != proposal.agent_id:
            raise AgentIdentityConflictError(
                "monitor proposal does not match database identity"
            )
        existing_row = connection.execute(
            "SELECT * FROM monitor_proposals WHERE agent_id = ? "
            "AND (id = ? OR idempotency_key = ?) ORDER BY id LIMIT 1",
            (proposal.agent_id, proposal.id, proposal.idempotency_key),
        ).fetchone()
        if existing_row is not None:
            existing = _decode_monitor_proposal_row(existing_row)
            if (
                existing.agent_id != proposal.agent_id
                or existing.intended_monitor_id != proposal.intended_monitor_id
                or existing.idempotency_key != proposal.idempotency_key
                or existing.candidate_hash != proposal.candidate_hash
                or existing.source_operation_id != proposal.source_operation_id
            ):
                raise MonitorProposalConflictError(
                    proposal.id,
                    "identity or idempotency key already names another candidate",
                )
            connection.execute("COMMIT")
            return existing, ()
        connection.execute(
            "INSERT INTO monitor_proposals("
            "id, agent_id, intended_monitor_id, idempotency_key, candidate_hash, "
            "candidate_json, source_operation_id, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            _monitor_proposal_values(proposal),
        )
        _append_monitor_events_in_transaction(connection, (event,))
        connection.execute("COMMIT")
        return proposal, (event,)
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise MonitorProposalConflictError(
            proposal.id,
            "durable identity or event correlation conflicts",
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_monitor_proposal(
    connection: sqlite3.Connection,
    agent_id: str,
    proposal_id: str,
) -> MonitorProposal | None:
    connection.execute("BEGIN")
    try:
        result = _load_monitor_proposal_in_transaction(
            connection,
            agent_id,
            proposal_id,
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_monitor_confirmation(
    connection: sqlite3.Connection,
    agent_id: str,
    proposal_id: str,
) -> MonitorConfirmation | None:
    connection.execute("BEGIN")
    try:
        result = _load_monitor_confirmation_in_transaction(
            connection,
            agent_id,
            proposal_id,
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _list_monitor_proposals(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    limit: int,
) -> tuple[MonitorProposal, ...]:
    connection.execute("BEGIN")
    try:
        result = tuple(
            _decode_monitor_proposal_row(row)
            for row in connection.execute(
                "SELECT * FROM monitor_proposals WHERE agent_id = ? "
                "ORDER BY created_at, id LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _validate_monitor_activation(
    proposal: MonitorProposal,
    commit: MonitorConfirmationCommit,
) -> None:
    confirmation = commit.confirmation
    if confirmation.agent_id != proposal.agent_id:
        raise MonitorProposalConflictError(proposal.id, "agent scope changed")
    if (
        confirmation.candidate_hash != proposal.candidate_hash
        or confirmation.decided_at < proposal.created_at
    ):
        raise MonitorProposalConflictError(
            proposal.id,
            "confirmation does not match the proposed candidate",
        )
    if confirmation.decision is MonitorConfirmationDecision.REJECTED:
        return
    assert commit.monitor is not None
    assert commit.version is not None
    assert commit.lifecycle is not None
    assert commit.schedule_state is not None
    if (
        commit.monitor.id != proposal.intended_monitor_id
        or commit.version.definition != proposal.candidate
        or commit.version.content_hash != proposal.candidate_hash
        or commit.version.proposal_id != proposal.id
        or confirmation.resulting_version_id != commit.version.id
        or commit.monitor.status is not MonitorStatus.ENABLED
        or commit.monitor.revision != 1
        or commit.schedule_state.revision != 1
        or commit.lifecycle.action is not MonitorLifecycleAction.ACTIVATE
        or commit.lifecycle.from_status is not None
        or commit.lifecycle.to_status is not MonitorStatus.ENABLED
        or commit.lifecycle.monitor_version != 1
    ):
        raise MonitorProposalConflictError(
            proposal.id,
            "activation records do not reproduce the confirmed proposal",
        )


def _commit_monitor_confirmation(
    connection: sqlite3.Connection,
    commit: MonitorConfirmationCommit,
) -> tuple[MonitorInspection | None, tuple[RuntimeEvent, ...]]:
    confirmation = commit.confirmation
    connection.execute("BEGIN IMMEDIATE")
    try:
        proposal = _load_monitor_proposal_in_transaction(
            connection,
            confirmation.agent_id,
            commit.proposal_id,
        )
        if proposal is None:
            raise MonitorProposalNotFoundError(
                confirmation.agent_id,
                commit.proposal_id,
            )
        existing = _load_monitor_confirmation_in_transaction(
            connection,
            confirmation.agent_id,
            commit.proposal_id,
        )
        if existing is not None:
            if (
                existing.decision is not confirmation.decision
                or existing.candidate_hash != confirmation.candidate_hash
                or existing.actor_id != confirmation.actor_id
                or existing.reason != confirmation.reason
            ):
                raise MonitorProposalConflictError(
                    commit.proposal_id,
                    "proposal already has a different confirmation",
                )
            result = (
                None
                if existing.resulting_monitor_id is None
                else _inspect_monitor_in_transaction(
                    connection,
                    existing.agent_id,
                    existing.resulting_monitor_id,
                )
            )
            connection.execute("COMMIT")
            return result, ()
        _validate_monitor_activation(proposal, commit)
        if confirmation.decision is MonitorConfirmationDecision.CONFIRMED:
            assert commit.monitor is not None
            assert commit.version is not None
            assert commit.lifecycle is not None
            assert commit.schedule_state is not None
            if (
                _load_monitor_in_transaction(
                    connection,
                    commit.monitor.agent_id,
                    commit.monitor.id,
                )
                is not None
            ):
                raise MonitorProposalConflictError(
                    commit.proposal_id,
                    "intended monitor identity is already active",
                )
            connection.execute(
                "INSERT INTO monitors("
                "id, agent_id, status, current_version, revision, created_at, "
                "updated_at, paused_at, deleted_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _monitor_values(commit.monitor),
            )
            _insert_monitor_version(connection, commit.version)
            _insert_monitor_lifecycle(connection, commit.lifecycle)
            _insert_monitor_schedule_state(connection, commit.schedule_state)
        connection.execute(
            "INSERT INTO monitor_confirmations("
            "id, agent_id, proposal_id, decision, candidate_hash, actor_id, "
            "reason, decided_at, resulting_monitor_id, resulting_version_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _monitor_confirmation_values(confirmation),
        )
        _append_monitor_events_in_transaction(connection, (commit.event,))
        result = (
            None
            if confirmation.resulting_monitor_id is None
            else _inspect_monitor_in_transaction(
                connection,
                confirmation.agent_id,
                confirmation.resulting_monitor_id,
            )
        )
        connection.execute("COMMIT")
        return result, (commit.event,)
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise MonitorProposalConflictError(
            commit.proposal_id,
            "confirmation conflicts with durable monitor history",
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _inspect_monitor(
    connection: sqlite3.Connection,
    agent_id: str,
    monitor_id: str,
) -> MonitorInspection | None:
    connection.execute("BEGIN")
    try:
        result = _inspect_monitor_in_transaction(connection, agent_id, monitor_id)
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _list_monitors(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    statuses: tuple[MonitorStatus, ...],
    limit: int,
) -> tuple[Monitor, ...]:
    connection.execute("BEGIN")
    try:
        placeholders = ",".join("?" for _ in statuses)
        rows = connection.execute(
            "SELECT * FROM monitors WHERE agent_id = ? "
            f"AND status IN ({placeholders}) ORDER BY created_at, id LIMIT ?",
            (agent_id, *(status.value for status in statuses), limit),
        ).fetchall()
        result = tuple(_decode_monitor_row(row) for row in rows)
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_lifecycle_by_idempotency(
    connection: sqlite3.Connection,
    agent_id: str,
    idempotency_key: str,
) -> MonitorLifecycleRecord | None:
    row = connection.execute(
        "SELECT * FROM monitor_lifecycle WHERE agent_id = ? AND idempotency_key = ?",
        (agent_id, idempotency_key),
    ).fetchone()
    return None if row is None else _decode_monitor_lifecycle_row(row)


def _validate_monitor_lifecycle_commit(
    connection: sqlite3.Connection,
    current: Monitor,
    current_state: MonitorScheduleState,
    commit: MonitorLifecycleCommit,
    *,
    expected_revision: int,
) -> None:
    monitor = commit.monitor
    lifecycle = commit.lifecycle
    state = commit.schedule_state
    if current.revision != expected_revision:
        raise MonitorConflictError(
            current.id,
            expected_revision=expected_revision,
            actual_revision=current.revision,
        )
    if (
        lifecycle.from_revision != current.revision
        or lifecycle.to_revision != monitor.revision
        or lifecycle.from_status is not current.status
        or lifecycle.to_status is not monitor.status
        or state.revision != current_state.revision + 1
        or state.updated_at < current_state.updated_at
    ):
        raise MonitorConflictError(
            current.id,
            expected_revision=expected_revision,
            actual_revision=current.revision,
        )
    if lifecycle.action is MonitorLifecycleAction.UPDATE:
        if (
            commit.version is None
            or monitor.current_version != current.current_version + 1
            or lifecycle.monitor_version != monitor.current_version
        ):
            raise ValueError(
                "monitor update requires the exact next definition version"
            )
        proposal = _load_monitor_proposal_in_transaction(
            connection,
            monitor.agent_id,
            commit.version.proposal_id,
        )
        if proposal is None:
            raise MonitorProposalNotFoundError(
                monitor.agent_id,
                commit.version.proposal_id,
            )
    elif (
        commit.version is not None or monitor.current_version != current.current_version
    ):
        raise ValueError("only monitor update may activate a new definition version")
    expected_actions: dict[
        MonitorLifecycleAction, tuple[MonitorStatus, MonitorStatus]
    ] = {
        MonitorLifecycleAction.PAUSE: (MonitorStatus.ENABLED, MonitorStatus.PAUSED),
        MonitorLifecycleAction.RESUME: (MonitorStatus.PAUSED, MonitorStatus.ENABLED),
        MonitorLifecycleAction.DELETE: (current.status, MonitorStatus.DELETED),
        MonitorLifecycleAction.RUN_NOW: (current.status, current.status),
        MonitorLifecycleAction.UPDATE: (current.status, current.status),
    }
    expected_statuses = expected_actions.get(lifecycle.action)
    if expected_statuses is None or expected_statuses != (
        current.status,
        monitor.status,
    ):
        raise ValueError(
            "monitor lifecycle action does not match its status transition"
        )


def _commit_monitor_lifecycle(
    connection: sqlite3.Connection,
    commit: MonitorLifecycleCommit,
    *,
    expected_revision: int,
) -> tuple[MonitorInspection, tuple[RuntimeEvent, ...]]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        replay = _load_lifecycle_by_idempotency(
            connection,
            commit.monitor.agent_id,
            commit.lifecycle.idempotency_key,
        )
        if replay is not None:
            if (
                replay.monitor_id != commit.lifecycle.monitor_id
                or replay.action is not commit.lifecycle.action
                or replay.actor_id != commit.lifecycle.actor_id
                or replay.reason != commit.lifecycle.reason
                or replay.operation_id != commit.lifecycle.operation_id
            ):
                raise MonitorConflictError(
                    commit.monitor.id,
                    expected_revision=expected_revision,
                    actual_revision=replay.to_revision,
                )
            inspection = _inspect_monitor_in_transaction(
                connection,
                commit.monitor.agent_id,
                commit.monitor.id,
            )
            if (
                inspection is None
                or inspection.monitor.revision < replay.to_revision
                or replay not in inspection.lifecycle
            ):
                actual = 0 if inspection is None else inspection.monitor.revision
                raise MonitorConflictError(
                    commit.monitor.id,
                    expected_revision=commit.monitor.revision,
                    actual_revision=actual,
                )
            connection.execute("COMMIT")
            return inspection, ()
        current = _load_monitor_in_transaction(
            connection,
            commit.monitor.agent_id,
            commit.monitor.id,
        )
        if current is None:
            raise MonitorNotFoundError(commit.monitor.agent_id, commit.monitor.id)
        current_state = _load_monitor_state_in_transaction(
            connection,
            commit.monitor.agent_id,
            commit.monitor.id,
        )
        if current_state is None:
            raise SQLiteCorruptionError(
                f"monitor {commit.monitor.id} is missing schedule state"
            )
        _validate_monitor_lifecycle_commit(
            connection,
            current,
            current_state,
            commit,
            expected_revision=expected_revision,
        )
        if commit.version is not None:
            proposal = _load_monitor_proposal_in_transaction(
                connection,
                commit.monitor.agent_id,
                commit.version.proposal_id,
            )
            confirmation = _load_monitor_confirmation_in_transaction(
                connection,
                commit.monitor.agent_id,
                commit.version.proposal_id,
            )
            if (
                proposal is None
                or confirmation is None
                or confirmation.decision is not MonitorConfirmationDecision.CONFIRMED
                or confirmation.candidate_hash != commit.version.content_hash
                or proposal.candidate != commit.version.definition
            ):
                raise MonitorProposalConflictError(
                    commit.version.proposal_id,
                    "definition update is not an exactly confirmed proposal",
                )
            _insert_monitor_version(connection, commit.version)
        monitor_values = _monitor_values(commit.monitor)
        changed = connection.execute(
            "UPDATE monitors SET status = ?, current_version = ?, revision = ?, "
            "updated_at = ?, paused_at = ?, deleted_at = ? "
            "WHERE agent_id = ? AND id = ? AND revision = ?",
            (
                monitor_values[2],
                monitor_values[3],
                monitor_values[4],
                monitor_values[6],
                monitor_values[7],
                monitor_values[8],
                commit.monitor.agent_id,
                commit.monitor.id,
                expected_revision,
            ),
        )
        if changed.rowcount != 1:
            latest = _load_monitor_in_transaction(
                connection,
                commit.monitor.agent_id,
                commit.monitor.id,
            )
            raise MonitorConflictError(
                commit.monitor.id,
                expected_revision=expected_revision,
                actual_revision=0 if latest is None else latest.revision,
            )
        state_values = _monitor_schedule_state_values(commit.schedule_state)
        state_update = connection.execute(
            "UPDATE monitor_schedule_state SET revision = ?, next_scheduled_at = ?, "
            "updated_at = ?, last_scheduled_at = ?, cooldown_until = ?, "
            "backoff_until = ?, consecutive_failures = ?, consecutive_matches = ?, "
            "checkpoint_version = ?, last_occurrence_id = ?, last_run_id = ?, "
            "last_operation_id = ? WHERE agent_id = ? AND monitor_id = ? "
            "AND revision = ?",
            (
                *state_values[2:],
                commit.schedule_state.agent_id,
                commit.schedule_state.monitor_id,
                current_state.revision,
            ),
        )
        if state_update.rowcount != 1:
            raise MonitorConflictError(
                commit.monitor.id,
                expected_revision=current_state.revision,
                actual_revision=current_state.revision,
            )
        _insert_monitor_lifecycle(connection, commit.lifecycle)
        _append_monitor_events_in_transaction(connection, (commit.event,))
        result = _inspect_monitor_in_transaction(
            connection,
            commit.monitor.agent_id,
            commit.monitor.id,
        )
        assert result is not None
        connection.execute("COMMIT")
        return result, (commit.event,)
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _list_due_monitors(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    now: datetime,
    limit: int,
) -> tuple[MonitorInspection, ...]:
    connection.execute("BEGIN")
    try:
        now_text = _encode_datetime(now)
        rows = connection.execute(
            "SELECT monitor.id FROM monitors AS monitor "
            "JOIN monitor_schedule_state AS state "
            "ON state.agent_id = monitor.agent_id AND state.monitor_id = monitor.id "
            "WHERE monitor.agent_id = ? AND monitor.status = 'enabled' "
            "AND state.next_scheduled_at IS NOT NULL "
            "AND state.next_scheduled_at <= ? "
            "AND (state.cooldown_until IS NULL OR state.cooldown_until <= ?) "
            "AND (state.backoff_until IS NULL OR state.backoff_until <= ?) "
            "ORDER BY state.next_scheduled_at, monitor.id LIMIT ?",
            (agent_id, now_text, now_text, now_text, limit),
        ).fetchall()
        inspections: list[MonitorInspection] = []
        for row in rows:
            inspection = _inspect_monitor_in_transaction(
                connection,
                agent_id,
                _sqlite_text(row[0], "due monitor id"),
            )
            if inspection is None:
                raise SQLiteCorruptionError("due monitor projection disappeared")
            inspections.append(inspection)
        connection.execute("COMMIT")
        return tuple(inspections)
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_monitor_occurrence_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    occurrence_id: str,
) -> MonitorOccurrence | None:
    row = connection.execute(
        "SELECT * FROM monitor_occurrences WHERE agent_id = ? AND id = ?",
        (agent_id, occurrence_id),
    ).fetchone()
    return None if row is None else _decode_monitor_occurrence_row(row)


def _load_monitor_occurrence_by_trigger(
    connection: sqlite3.Connection,
    agent_id: str,
    trigger_id: str,
) -> MonitorOccurrence | None:
    connection.execute("BEGIN")
    try:
        row = connection.execute(
            "SELECT * FROM monitor_occurrences WHERE agent_id = ? AND trigger_id = ?",
            (agent_id, trigger_id),
        ).fetchone()
        result = None if row is None else _decode_monitor_occurrence_row(row)
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_monitor_claim_by_manual_key(
    connection: sqlite3.Connection,
    agent_id: str,
    monitor_id: str,
    manual_key: str,
) -> MonitorClaimResult | None:
    connection.execute("BEGIN")
    try:
        row = connection.execute(
            "SELECT * FROM monitor_occurrences WHERE agent_id = ? "
            "AND monitor_id = ? AND manual_key = ?",
            (agent_id, monitor_id, manual_key),
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        occurrence = _decode_monitor_occurrence_row(row)
        run = _load_monitor_run_in_transaction(
            connection,
            agent_id,
            occurrence.run_id,
        )
        lease = _load_latest_monitor_lease(
            connection,
            agent_id,
            occurrence.id,
        )
        if run is None or lease is None:
            raise SQLiteCorruptionError(
                "run-now occurrence is missing its run or lease"
            )
        result = MonitorClaimResult(
            occurrence=occurrence,
            lease=lease,
            run=run,
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_active_monitor_lease(
    connection: sqlite3.Connection,
    agent_id: str,
    occurrence_id: str,
) -> MonitorTickLease | None:
    row = connection.execute(
        "SELECT * FROM monitor_tick_leases WHERE agent_id = ? "
        "AND occurrence_id = ? AND released_at IS NULL",
        (agent_id, occurrence_id),
    ).fetchone()
    return None if row is None else _decode_monitor_lease_row(row)


def _load_latest_monitor_lease(
    connection: sqlite3.Connection,
    agent_id: str,
    occurrence_id: str,
) -> MonitorTickLease | None:
    row = connection.execute(
        "SELECT * FROM monitor_tick_leases WHERE agent_id = ? "
        "AND occurrence_id = ? ORDER BY fencing_token DESC LIMIT 1",
        (agent_id, occurrence_id),
    ).fetchone()
    return None if row is None else _decode_monitor_lease_row(row)


def _load_monitor_run_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    run_id: str,
) -> MonitorRun | None:
    row = connection.execute(
        "SELECT * FROM monitor_runs WHERE agent_id = ? AND id = ?",
        (agent_id, run_id),
    ).fetchone()
    return None if row is None else _decode_monitor_run_row(row)


def _validate_monitor_claim_frontier(
    monitor: Monitor,
    state: MonitorScheduleState,
    claim: MonitorOccurrenceClaim,
    *,
    expected_monitor_revision: int,
    expected_schedule_revision: int,
    checked_at: datetime,
) -> None:
    occurrence = claim.occurrence
    if monitor.revision != expected_monitor_revision:
        raise MonitorConflictError(
            monitor.id,
            expected_revision=expected_monitor_revision,
            actual_revision=monitor.revision,
        )
    if state.revision != expected_schedule_revision:
        raise MonitorConflictError(
            monitor.id,
            expected_revision=expected_schedule_revision,
            actual_revision=state.revision,
        )
    if occurrence.monitor_version != monitor.current_version:
        raise ValueError("occurrence does not use the active monitor version")
    if occurrence.kind is MonitorOccurrenceKind.SCHEDULED:
        if monitor.status is not MonitorStatus.ENABLED:
            raise ValueError("only an enabled monitor may claim a scheduled occurrence")
        if (
            state.next_scheduled_at is None
            or occurrence.scheduled_for != state.next_scheduled_at
            or occurrence.scheduled_for > checked_at
            or (state.cooldown_until is not None and state.cooldown_until > checked_at)
            or (state.backoff_until is not None and state.backoff_until > checked_at)
        ):
            raise ValueError("scheduled occurrence is not at the durable due frontier")
    elif monitor.status is MonitorStatus.DELETED:
        raise ValueError("deleted monitor cannot claim a run-now occurrence")
    if claim.lease.released_at is not None:
        raise ValueError("new monitor claim lease must be active")
    if claim.run.status is not MonitorRunStatus.PENDING:
        raise ValueError("new monitor claim run must be pending")


def _claim_monitor_occurrence(
    connection: sqlite3.Connection,
    claim: MonitorOccurrenceClaim,
    *,
    expected_monitor_revision: int,
    expected_schedule_revision: int,
    checked_at: datetime,
) -> tuple[MonitorClaimResult, tuple[RuntimeEvent, ...]]:
    occurrence = claim.occurrence
    connection.execute("BEGIN IMMEDIATE")
    try:
        monitor = _load_monitor_in_transaction(
            connection,
            occurrence.agent_id,
            occurrence.monitor_id,
        )
        if monitor is None:
            raise MonitorNotFoundError(occurrence.agent_id, occurrence.monitor_id)
        state = _load_monitor_state_in_transaction(
            connection,
            occurrence.agent_id,
            occurrence.monitor_id,
        )
        if state is None:
            raise SQLiteCorruptionError(
                f"monitor {occurrence.monitor_id} is missing schedule state"
            )
        _validate_monitor_claim_frontier(
            monitor,
            state,
            claim,
            expected_monitor_revision=expected_monitor_revision,
            expected_schedule_revision=expected_schedule_revision,
            checked_at=checked_at,
        )
        if occurrence.manual_key is not None:
            manual_row = connection.execute(
                "SELECT * FROM monitor_occurrences WHERE agent_id = ? "
                "AND monitor_id = ? AND manual_key = ?",
                (
                    occurrence.agent_id,
                    occurrence.monitor_id,
                    occurrence.manual_key,
                ),
            ).fetchone()
            if manual_row is not None:
                replayed_occurrence = _decode_monitor_occurrence_row(manual_row)
                replayed_run = _load_monitor_run_in_transaction(
                    connection,
                    occurrence.agent_id,
                    replayed_occurrence.run_id,
                )
                replayed_lease = _load_active_monitor_lease(
                    connection,
                    occurrence.agent_id,
                    replayed_occurrence.id,
                ) or _load_latest_monitor_lease(
                    connection,
                    occurrence.agent_id,
                    replayed_occurrence.id,
                )
                if replayed_run is None or replayed_lease is None:
                    raise SQLiteCorruptionError(
                        "run-now occurrence is missing its run or lease"
                    )
                if claim.lease.fencing_token == 1 and claim.run.attempt == 1:
                    connection.execute("COMMIT")
                    return (
                        MonitorClaimResult(
                            occurrence=replayed_occurrence,
                            lease=replayed_lease,
                            run=replayed_run,
                        ),
                        (),
                    )
                if replayed_occurrence != occurrence:
                    raise ValueError("stable run-now occurrence identity changed")
        stored_occurrence = _load_monitor_occurrence_in_transaction(
            connection,
            occurrence.agent_id,
            occurrence.id,
        )
        if stored_occurrence is None:
            if claim.lease.fencing_token != 1 or claim.run.attempt != 1:
                raise ValueError(
                    "first monitor occurrence claim must use fence/attempt one"
                )
            connection.execute(
                "INSERT INTO monitor_occurrences("
                "id, agent_id, monitor_id, monitor_version, kind, scheduled_for, "
                "occurrence_key, trigger_id, run_id, created_at, manual_key"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _monitor_occurrence_values(occurrence),
            )
            connection.execute(
                "INSERT INTO monitor_tick_leases("
                "id, agent_id, monitor_id, occurrence_id, holder_id, fencing_token, "
                "claimed_at, expires_at, released_at, release_reason"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _monitor_lease_values(claim.lease),
            )
            connection.execute(
                "INSERT INTO monitor_runs("
                "id, agent_id, monitor_id, occurrence_id, trigger_id, attempt, "
                "fencing_token, status, started_at, operation_id, completed_at, "
                "failure_reason"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _monitor_run_values(claim.run),
            )
        else:
            if stored_occurrence != occurrence:
                raise ValueError("stable monitor occurrence identity changed")
            active = _load_active_monitor_lease(
                connection,
                occurrence.agent_id,
                occurrence.id,
            )
            stored_run = _load_monitor_run_in_transaction(
                connection,
                occurrence.agent_id,
                occurrence.run_id,
            )
            if stored_run is None:
                raise SQLiteCorruptionError("monitor occurrence is missing its run")
            if active is not None and active.expires_at > checked_at:
                if active == claim.lease and stored_run == claim.run:
                    connection.execute("COMMIT")
                    return (
                        MonitorClaimResult(
                            occurrence=occurrence,
                            lease=active,
                            run=stored_run,
                        ),
                        (),
                    )
                raise MonitorTickClaimConflictError(
                    occurrence.id,
                    holder_id=active.holder_id,
                    fencing_token=active.fencing_token,
                    expires_at=active.expires_at,
                )
            if stored_run.completed_at is not None:
                latest = active or _load_latest_monitor_lease(
                    connection,
                    occurrence.agent_id,
                    occurrence.id,
                )
                assert latest is not None
                raise MonitorTickClaimConflictError(
                    occurrence.id,
                    holder_id=latest.holder_id,
                    fencing_token=latest.fencing_token,
                    expires_at=latest.expires_at,
                )
            latest = active or _load_latest_monitor_lease(
                connection,
                occurrence.agent_id,
                occurrence.id,
            )
            if latest is None:
                raise SQLiteCorruptionError("monitor occurrence has no lease history")
            if active is not None:
                connection.execute(
                    "UPDATE monitor_tick_leases SET released_at = ?, "
                    "release_reason = 'expired_reclaimed' "
                    "WHERE id = ? AND released_at IS NULL",
                    (_encode_datetime(checked_at), active.id),
                )
            if (
                claim.lease.fencing_token != latest.fencing_token + 1
                or claim.run.fencing_token != claim.lease.fencing_token
                or claim.run.attempt != stored_run.attempt + 1
            ):
                raise StaleMonitorFenceError(
                    occurrence.id,
                    expected_fencing_token=latest.fencing_token + 1,
                    actual_fencing_token=claim.lease.fencing_token,
                )
            connection.execute(
                "INSERT INTO monitor_tick_leases("
                "id, agent_id, monitor_id, occurrence_id, holder_id, fencing_token, "
                "claimed_at, expires_at, released_at, release_reason"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _monitor_lease_values(claim.lease),
            )
            run_values = _monitor_run_values(claim.run)
            connection.execute(
                "UPDATE monitor_runs SET attempt = ?, fencing_token = ?, status = ?, "
                "started_at = ?, operation_id = ?, completed_at = ?, "
                "failure_reason = ? WHERE agent_id = ? AND id = ?",
                (
                    *run_values[5:],
                    occurrence.agent_id,
                    occurrence.run_id,
                ),
            )
        _append_monitor_events_in_transaction(connection, (claim.event,))
        connection.execute("COMMIT")
        return (
            MonitorClaimResult(
                occurrence=occurrence,
                lease=claim.lease,
                run=claim.run,
            ),
            (claim.event,),
        )
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _validate_monitor_operation_link(
    connection: sqlite3.Connection,
    run: MonitorRun,
) -> None:
    if run.operation_id is None:
        return
    row = connection.execute(
        "SELECT operation.agent_id, operation.trigger_id "
        "FROM operations AS operation WHERE operation.id = ?",
        (run.operation_id,),
    ).fetchone()
    if row is None:
        raise OperationNotFoundError(run.operation_id)
    if (
        _sqlite_text(row["agent_id"], "monitor operation agent_id") != run.agent_id
        or _sqlite_text(row["trigger_id"], "monitor operation trigger_id")
        != run.trigger_id
    ):
        raise ValueError("monitor run operation does not use its stable trigger")


def _commit_monitor_outcome(
    connection: sqlite3.Connection,
    commit: MonitorOutcomeCommit,
    *,
    expected_monitor_revision: int,
    expected_schedule_revision: int,
    checked_at: datetime,
) -> tuple[MonitorOutcomeResult, tuple[RuntimeEvent, ...]]:
    guard = commit.guard
    connection.execute("BEGIN IMMEDIATE")
    try:
        monitor = _load_monitor_in_transaction(
            connection,
            guard.agent_id,
            guard.monitor_id,
        )
        if monitor is None:
            raise MonitorNotFoundError(guard.agent_id, guard.monitor_id)
        if monitor.revision != expected_monitor_revision:
            raise MonitorConflictError(
                monitor.id,
                expected_revision=expected_monitor_revision,
                actual_revision=monitor.revision,
            )
        state = _load_monitor_state_in_transaction(
            connection,
            guard.agent_id,
            guard.monitor_id,
        )
        if state is None:
            raise SQLiteCorruptionError(
                f"monitor {monitor.id} is missing schedule state"
            )
        if state.revision != expected_schedule_revision:
            raise MonitorConflictError(
                monitor.id,
                expected_revision=expected_schedule_revision,
                actual_revision=state.revision,
            )
        active = _load_active_monitor_lease(
            connection,
            guard.agent_id,
            guard.occurrence_id,
        )
        latest = active or _load_latest_monitor_lease(
            connection,
            guard.agent_id,
            guard.occurrence_id,
        )
        actual_fence = 0 if latest is None else latest.fencing_token
        if (
            active is None
            or active.holder_id != guard.holder_id
            or active.fencing_token != guard.fencing_token
        ):
            raise StaleMonitorFenceError(
                guard.occurrence_id,
                expected_fencing_token=guard.fencing_token,
                actual_fencing_token=actual_fence,
            )
        if active.expires_at <= checked_at:
            raise ExpiredMonitorLeaseError(
                guard.occurrence_id,
                fencing_token=guard.fencing_token,
                expires_at=active.expires_at,
                checked_at=checked_at,
            )
        if commit.released_lease.id != active.id or (
            commit.released_lease.agent_id,
            commit.released_lease.monitor_id,
            commit.released_lease.occurrence_id,
            commit.released_lease.holder_id,
            commit.released_lease.fencing_token,
            commit.released_lease.claimed_at,
            commit.released_lease.expires_at,
        ) != (
            active.agent_id,
            active.monitor_id,
            active.occurrence_id,
            active.holder_id,
            active.fencing_token,
            active.claimed_at,
            active.expires_at,
        ):
            raise StaleMonitorFenceError(
                guard.occurrence_id,
                expected_fencing_token=guard.fencing_token,
                actual_fencing_token=actual_fence,
            )
        stored_run = _load_monitor_run_in_transaction(
            connection,
            guard.agent_id,
            commit.run.id,
        )
        if stored_run is None:
            raise SQLiteCorruptionError("monitor outcome is missing its claimed run")
        if (
            stored_run.completed_at is not None
            or stored_run.occurrence_id != guard.occurrence_id
            or stored_run.fencing_token != guard.fencing_token
            or commit.run.attempt != stored_run.attempt
            or commit.run.fencing_token != stored_run.fencing_token
            or commit.run.started_at != stored_run.started_at
        ):
            raise StaleMonitorFenceError(
                guard.occurrence_id,
                expected_fencing_token=guard.fencing_token,
                actual_fencing_token=stored_run.fencing_token,
            )
        _validate_monitor_operation_link(connection, commit.run)
        if (
            commit.schedule_state.revision != state.revision + 1
            or commit.schedule_state.updated_at < state.updated_at
            or commit.schedule_state.last_occurrence_id != guard.occurrence_id
            or commit.schedule_state.last_run_id != commit.run.id
            or commit.schedule_state.last_operation_id != commit.run.operation_id
        ):
            raise MonitorConflictError(
                monitor.id,
                expected_revision=state.revision + 1,
                actual_revision=commit.schedule_state.revision,
            )
        if commit.checkpoint is None:
            if commit.schedule_state.checkpoint_version != state.checkpoint_version:
                raise ValueError(
                    "outcome without checkpoint cannot advance its version"
                )
        elif (
            commit.checkpoint.agent_id != guard.agent_id
            or commit.checkpoint.monitor_id != guard.monitor_id
            or commit.checkpoint.version != state.checkpoint_version + 1
            or commit.schedule_state.checkpoint_version != commit.checkpoint.version
        ):
            raise ValueError("monitor checkpoint does not advance the exact frontier")
        if commit.finding is not None:
            if (
                commit.finding.agent_id != guard.agent_id
                or commit.finding.monitor_id != guard.monitor_id
                or commit.finding.occurrence_id != guard.occurrence_id
                or commit.finding.operation_id != commit.run.operation_id
            ):
                raise ValueError("monitor finding does not match the outcome")
        run_values = _monitor_run_values(commit.run)
        run_update = connection.execute(
            "UPDATE monitor_runs SET attempt = ?, fencing_token = ?, status = ?, "
            "started_at = ?, operation_id = ?, completed_at = ?, failure_reason = ? "
            "WHERE agent_id = ? AND id = ? AND fencing_token = ? "
            "AND completed_at IS NULL",
            (
                *run_values[5:],
                guard.agent_id,
                commit.run.id,
                guard.fencing_token,
            ),
        )
        if run_update.rowcount != 1:
            raise StaleMonitorFenceError(
                guard.occurrence_id,
                expected_fencing_token=guard.fencing_token,
                actual_fencing_token=actual_fence,
            )
        released_at = commit.released_lease.released_at
        if released_at is None:
            raise ValueError("monitor outcome requires a released lease")
        lease_update = connection.execute(
            "UPDATE monitor_tick_leases SET released_at = ?, release_reason = ? "
            "WHERE id = ? AND released_at IS NULL AND holder_id = ? "
            "AND fencing_token = ?",
            (
                _encode_datetime(released_at),
                commit.released_lease.release_reason,
                active.id,
                guard.holder_id,
                guard.fencing_token,
            ),
        )
        if lease_update.rowcount != 1:
            raise StaleMonitorFenceError(
                guard.occurrence_id,
                expected_fencing_token=guard.fencing_token,
                actual_fencing_token=actual_fence,
            )
        state_values = _monitor_schedule_state_values(commit.schedule_state)
        state_update = connection.execute(
            "UPDATE monitor_schedule_state SET revision = ?, next_scheduled_at = ?, "
            "updated_at = ?, last_scheduled_at = ?, cooldown_until = ?, "
            "backoff_until = ?, consecutive_failures = ?, consecutive_matches = ?, "
            "checkpoint_version = ?, last_occurrence_id = ?, last_run_id = ?, "
            "last_operation_id = ? WHERE agent_id = ? AND monitor_id = ? "
            "AND revision = ?",
            (
                *state_values[2:],
                guard.agent_id,
                guard.monitor_id,
                expected_schedule_revision,
            ),
        )
        if state_update.rowcount != 1:
            raise MonitorConflictError(
                monitor.id,
                expected_revision=expected_schedule_revision,
                actual_revision=state.revision,
            )
        if commit.checkpoint is not None:
            connection.execute(
                "INSERT INTO monitor_checkpoints("
                "id, agent_id, monitor_id, version, run_id, cursor_json, "
                "cursor_hash, created_at, previous_version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _monitor_checkpoint_values(commit.checkpoint),
            )
        if commit.finding is not None:
            connection.execute(
                "INSERT INTO monitor_findings("
                "id, agent_id, monitor_id, occurrence_id, run_id, operation_id, "
                "evidence_id, severity, summary, details_json, dedupe_key, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _monitor_finding_values(commit.finding),
            )
        _append_monitor_events_in_transaction(connection, commit.events)
        inspection = _inspect_monitor_in_transaction(
            connection,
            guard.agent_id,
            guard.monitor_id,
        )
        if inspection is None:
            raise SQLiteCorruptionError("monitor disappeared during outcome commit")
        result = MonitorOutcomeResult(
            inspection=inspection,
            run=commit.run,
            finding=commit.finding,
        )
        connection.execute("COMMIT")
        return result, commit.events
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _host_inbox_values(item: HostInboxItem) -> tuple[object, ...]:
    return (
        item.id,
        item.agent_id,
        item.kind.value,
        item.idempotency_key,
        item.request_hash,
        canonical_json(item.payload),
        item.revision,
        item.status.value,
        _encode_datetime(item.created_at),
        _encode_datetime(item.updated_at),
        item.trigger_id,
        item.operation_id,
        item.error,
    )


def _decode_host_mutation_admission_row(
    row: sqlite3.Row,
) -> HostMutationAdmission:
    return HostMutationAdmission(
        agent_id=_sqlite_text(row["agent_id"], "host mutation agent_id"),
        idempotency_key=_sqlite_text(
            row["idempotency_key"],
            "host mutation idempotency key",
        ),
        method=_sqlite_text(row["method"], "host mutation method"),
        request_hash=_sqlite_text(
            row["request_hash"],
            "host mutation request hash",
        ),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "host mutation created_at")
        ),
    )


def _admit_host_mutation(
    connection: sqlite3.Connection,
    request: HostMutationAdmission,
) -> HostMutationAdmission:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != request.agent_id:
            raise AgentIdentityConflictError(
                "host mutation does not match database identity"
            )
        row = connection.execute(
            "SELECT * FROM host_mutation_admissions "
            "WHERE agent_id = ? AND idempotency_key = ?",
            (request.agent_id, request.idempotency_key),
        ).fetchone()
        if row is not None:
            existing = _decode_host_mutation_admission_row(row)
            if (
                existing.method != request.method
                or existing.request_hash != request.request_hash
            ):
                raise HostMutationConflictError(
                    request.agent_id,
                    request.idempotency_key,
                )
            connection.execute("COMMIT")
            return existing
        connection.execute(
            "INSERT INTO host_mutation_admissions("
            "agent_id, idempotency_key, method, request_hash, created_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                request.agent_id,
                request.idempotency_key,
                request.method,
                request.request_hash,
                _encode_datetime(request.created_at),
            ),
        )
        connection.execute("COMMIT")
        return request
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _decode_host_inbox_row(row: sqlite3.Row) -> HostInboxItem:
    return HostInboxItem(
        id=_sqlite_text(row["id"], "host inbox id"),
        agent_id=_sqlite_text(row["agent_id"], "host inbox agent_id"),
        kind=HostInboxKind(_sqlite_text(row["kind"], "host inbox kind")),
        idempotency_key=_sqlite_text(
            row["idempotency_key"],
            "host inbox idempotency key",
        ),
        request_hash=_sqlite_text(row["request_hash"], "host inbox request hash"),
        payload=_decode_json_object(
            _sqlite_text(row["payload_json"], "host inbox payload")
        ),
        revision=_sqlite_int(row["revision"], "host inbox revision"),
        status=HostInboxStatus(_sqlite_text(row["status"], "host inbox status")),
        created_at=_decode_datetime(
            _sqlite_text(row["created_at"], "host inbox created_at")
        ),
        updated_at=_decode_datetime(
            _sqlite_text(row["updated_at"], "host inbox updated_at")
        ),
        trigger_id=_optional_text(row["trigger_id"]),
        operation_id=_optional_text(row["operation_id"]),
        error=_optional_text(row["error"]),
    )


def _load_host_inbox_in_transaction(
    connection: sqlite3.Connection,
    agent_id: str,
    item_id: str,
) -> HostInboxItem | None:
    row = connection.execute(
        "SELECT * FROM host_inbox WHERE agent_id = ? AND id = ?",
        (agent_id, item_id),
    ).fetchone()
    return None if row is None else _decode_host_inbox_row(row)


def _enqueue_host_inbox(
    connection: sqlite3.Connection,
    item: HostInboxItem,
) -> HostInboxItem:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != item.agent_id:
            raise AgentIdentityConflictError(
                "host inbox item does not match database identity"
            )
        rows = tuple(
            _decode_host_inbox_row(row)
            for row in connection.execute(
                "SELECT * FROM host_inbox WHERE agent_id = ? "
                "AND (id = ? OR idempotency_key = ?)",
                (item.agent_id, item.id, item.idempotency_key),
            ).fetchall()
        )
        if rows:
            exact_key = tuple(
                existing
                for existing in rows
                if existing.idempotency_key == item.idempotency_key
            )
            id_collision = tuple(
                existing for existing in rows if existing.id == item.id
            )
            if (
                len(exact_key) != 1
                or exact_key[0].request_hash != item.request_hash
                or (id_collision and id_collision[0].id != exact_key[0].id)
            ):
                raise HostInboxEnqueueConflictError(
                    item.agent_id,
                    item.idempotency_key,
                )
            connection.execute("COMMIT")
            return exact_key[0]
        connection.execute(
            "INSERT INTO host_inbox("
            "id, agent_id, kind, idempotency_key, request_hash, payload_json, "
            "revision, status, created_at, updated_at, trigger_id, operation_id, error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _host_inbox_values(item),
        )
        connection.execute("COMMIT")
        return item
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise HostInboxEnqueueConflictError(
            item.agent_id,
            item.idempotency_key,
        ) from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _list_pending_host_inbox(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    limit: int,
) -> tuple[HostInboxItem, ...]:
    connection.execute("BEGIN")
    try:
        result = tuple(
            _decode_host_inbox_row(row)
            for row in connection.execute(
                "SELECT * FROM host_inbox WHERE agent_id = ? AND status = 'pending' "
                "ORDER BY created_at, id LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _validate_host_inbox_completion_link(
    connection: sqlite3.Connection,
    item: HostInboxItem,
) -> None:
    if item.operation_id is None:
        return
    row = connection.execute(
        "SELECT agent_id, trigger_id FROM operations WHERE id = ?",
        (item.operation_id,),
    ).fetchone()
    if row is None:
        raise OperationNotFoundError(item.operation_id)
    if _sqlite_text(row["agent_id"], "host inbox operation agent_id") != item.agent_id:
        raise ValueError("host inbox operation belongs to another agent")
    if item.kind is HostInboxKind.TRIGGER:
        if (
            _sqlite_text(row["trigger_id"], "host inbox operation trigger_id")
            != item.trigger_id
        ):
            raise ValueError("host inbox completion operation claimed another trigger")
        return
    approval_id = item.payload.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id.strip():
        raise ValueError("approval wake payload requires approval_id")
    approval_row = connection.execute(
        "SELECT 1 FROM approvals WHERE id = ? AND operation_id = ?",
        (approval_id, item.operation_id),
    ).fetchone()
    if approval_row is None:
        raise ValueError("approval wake completion operation owns another approval")


def _complete_host_inbox(
    connection: sqlite3.Connection,
    item: HostInboxItem,
    *,
    expected_revision: int,
) -> HostInboxItem:
    connection.execute("BEGIN IMMEDIATE")
    try:
        current = _load_host_inbox_in_transaction(
            connection,
            item.agent_id,
            item.id,
        )
        if current is None:
            raise HostInboxNotFoundError(item.agent_id, item.id)
        if current.revision != expected_revision:
            raise HostInboxRevisionConflict(
                item.id,
                expected_revision=expected_revision,
                actual_revision=current.revision,
            )
        if (
            item.revision != current.revision + 1
            or item.idempotency_key != current.idempotency_key
            or item.request_hash != current.request_hash
            or item.kind is not current.kind
            or item.payload != current.payload
            or item.trigger_id != current.trigger_id
            or item.created_at != current.created_at
            or item.updated_at < current.updated_at
        ):
            raise HostInboxRevisionConflict(
                item.id,
                expected_revision=current.revision + 1,
                actual_revision=item.revision,
            )
        _validate_host_inbox_completion_link(connection, item)
        values = _host_inbox_values(item)
        updated = connection.execute(
            "UPDATE host_inbox SET revision = ?, status = ?, updated_at = ?, "
            "operation_id = ?, error = ? WHERE agent_id = ? AND id = ? "
            "AND revision = ? AND status = 'pending'",
            (
                values[6],
                values[7],
                values[9],
                values[11],
                values[12],
                item.agent_id,
                item.id,
                expected_revision,
            ),
        )
        if updated.rowcount != 1:
            latest = _load_host_inbox_in_transaction(
                connection,
                item.agent_id,
                item.id,
            )
            raise HostInboxRevisionConflict(
                item.id,
                expected_revision=expected_revision,
                actual_revision=0 if latest is None else latest.revision,
            )
        connection.execute("COMMIT")
        return item
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _create_session(connection: sqlite3.Connection, session: Session) -> Session:
    connection.execute("BEGIN IMMEDIATE")
    try:
        identity = _load_agent_identity(connection)
        if identity is None or identity.id != session.agent_id:
            raise AgentIdentityConflictError(
                "session agent does not match the authoritative database identity"
            )
        try:
            connection.execute(
                "INSERT INTO sessions(id, agent_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session.id,
                    session.agent_id,
                    session.title,
                    _encode_datetime(session.created_at),
                    _encode_datetime(session.updated_at),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise SessionAlreadyExistsError(
                f"session already exists: {session.id}"
            ) from error
        connection.execute("COMMIT")
        return session
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_session_transcript(
    connection: sqlite3.Connection,
    agent_id: str,
    session_id: str,
) -> SessionTranscript | None:
    connection.execute("BEGIN")
    try:
        row = connection.execute(
            "SELECT id, agent_id, title, created_at, updated_at FROM sessions "
            "WHERE id = ? AND agent_id = ?",
            (session_id, agent_id),
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        session = Session(
            id=_sqlite_text(row["id"], "session id"),
            agent_id=_sqlite_text(row["agent_id"], "session agent id"),
            title=_sqlite_text(row["title"], "session title"),
            created_at=_decode_datetime(
                _sqlite_text(row["created_at"], "session created_at")
            ),
            updated_at=_decode_datetime(
                _sqlite_text(row["updated_at"], "session updated_at")
            ),
        )
        operation_rows = connection.execute(
            "SELECT operation.id FROM session_operations AS link "
            "JOIN operations AS operation ON operation.id = link.operation_id "
            "WHERE operation.agent_id = ? AND link.session_id = ? "
            "ORDER BY link.position",
            (agent_id, session_id),
        ).fetchall()
        snapshots = tuple(
            _load_versioned_operation_in_transaction(
                connection,
                _sqlite_text(operation_row[0], "session operation id"),
            ).snapshot
            for operation_row in operation_rows
        )
        operation_ids = tuple(snapshot.operation.id for snapshot in snapshots)
        messages: list[CanonicalMessage] = []
        updated_at = session.updated_at
        for snapshot in snapshots:
            updated_at = max(updated_at, snapshot.operation.updated_at)
            first_turn_id = snapshot.turns[0].id if snapshot.turns else None
            user_text = snapshot.trigger.payload.get("message")
            if isinstance(user_text, str) and user_text.strip():
                messages.append(
                    CanonicalMessage(
                        agent_id=agent_id,
                        operation_id=snapshot.operation.id,
                        session_id=session_id,
                        turn_id=first_turn_id,
                        role=MessageRole.USER,
                        content=(TextBlock(user_text),),
                    )
                )
            for model_call in snapshot.model_calls:
                response = model_call.response
                if response is None:
                    continue
                content = () if response.text is None else (TextBlock(response.text),)
                messages.append(
                    CanonicalMessage(
                        agent_id=agent_id,
                        operation_id=snapshot.operation.id,
                        session_id=session_id,
                        turn_id=model_call.turn_id,
                        role=MessageRole.ASSISTANT,
                        content=content,
                        tool_calls=response.tool_calls,
                        provider_id=response.provider_id,
                        provider_metadata=response.provider_metadata,
                    )
                )
                for observation in snapshot.observations:
                    if (
                        observation.turn_id != model_call.turn_id
                        or observation.call_id is None
                    ):
                        continue
                    messages.append(
                        CanonicalMessage(
                            agent_id=agent_id,
                            operation_id=snapshot.operation.id,
                            session_id=session_id,
                            turn_id=observation.turn_id,
                            role=MessageRole.TOOL,
                            content=(
                                ToolResultBlock(
                                    call_id=observation.call_id,
                                    output={
                                        "code": observation.code,
                                        "message": observation.message,
                                        "payload": observation.payload,
                                    },
                                    is_error=not observation.success,
                                ),
                            ),
                        )
                    )
        if updated_at != session.updated_at:
            session = Session(
                id=session.id,
                agent_id=session.agent_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=updated_at,
            )
        result = SessionTranscript(
            session=session,
            operation_ids=operation_ids,
            messages=tuple(messages),
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


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

        _validate_approval_id_claims(connection, snapshot)
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
        _validate_approval_id_claims(connection, snapshot)
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


def _validate_approval_id_claims(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
) -> None:
    operation_id = snapshot.operation.id
    for approval in snapshot.approvals:
        row = connection.execute(
            "SELECT operation_id FROM approvals WHERE id = ?",
            (approval.id,),
        ).fetchone()
        if row is None:
            continue
        claimed_operation_id = _sqlite_text(
            row[0],
            "approval operation id",
        )
        if claimed_operation_id != operation_id:
            raise InvalidOperationCheckpointError(
                operation_id,
                f"approval identity is already claimed: {approval.id}",
            )


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
    _apply_approval_delta(connection, current, candidate)

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
    _touch_session(connection, candidate)


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


def _apply_approval_delta(
    connection: sqlite3.Connection,
    current: OperationSnapshot,
    candidate: OperationSnapshot,
) -> None:
    operation_id = candidate.operation.id
    for position, (before, after) in enumerate(
        zip(current.approvals, candidate.approvals, strict=False)
    ):
        if after == before:
            continue
        approval_update = connection.execute(
            "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ?, "
            "decision_reason = ? WHERE operation_id = ? AND position = ? "
            "AND id = ? AND task_id = ? AND task_fingerprint = ? "
            "AND policy_fingerprint = ? AND requested_at = ? AND status = ? "
            "AND decided_at IS ? AND decided_by IS ? AND decision_reason IS ?",
            (
                after.status.value,
                _encode_optional_datetime(after.decided_at),
                after.decided_by,
                after.decision_reason,
                operation_id,
                position,
                before.id,
                before.task_id,
                before.task_fingerprint,
                before.policy_fingerprint,
                _encode_datetime(before.requested_at),
                before.status.value,
                _encode_optional_datetime(before.decided_at),
                before.decided_by,
                before.decision_reason,
            ),
        )
        _require_one_row(approval_update.rowcount, "approval", before.id)
    for position in range(len(current.approvals), len(candidate.approvals)):
        _insert_approval(
            connection,
            operation_id,
            position,
            candidate.approvals[position],
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
    for position, approval in enumerate(snapshot.approvals):
        _insert_approval(
            connection,
            operation.id,
            position,
            approval,
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
    _link_session_operation(connection, snapshot)
    _touch_session(connection, snapshot)


def _link_session_operation(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
) -> None:
    if _pragma_int(connection, "user_version") < 7:
        return
    identity = _load_agent_identity(connection)
    if identity is None:
        # A raw pre-Agent-Home operation store remains a supported migration
        # and contract-test state. Once identity is initialized, it is binding.
        return
    if identity.id != snapshot.operation.agent_id:
        raise InvalidOperationCheckpointError(
            snapshot.operation.id,
            "operation agent does not match authoritative database identity",
        )
    session_id = snapshot.operation.session_id
    if session_id is None:
        return
    session_row = connection.execute(
        "SELECT 1 FROM sessions WHERE id = ? AND agent_id = ?",
        (session_id, snapshot.operation.agent_id),
    ).fetchone()
    if session_row is None:
        raise InvalidOperationCheckpointError(
            snapshot.operation.id,
            "session-scoped operation requires a persisted session",
        )
    connection.execute(
        "INSERT INTO session_operations(session_id, position, operation_id) "
        "VALUES (?, (SELECT COALESCE(MAX(position), -1) + 1 "
        "FROM session_operations WHERE session_id = ?), ?)",
        (session_id, session_id, snapshot.operation.id),
    )


def _touch_session(
    connection: sqlite3.Connection,
    snapshot: OperationSnapshot,
) -> None:
    if _pragma_int(connection, "user_version") < 7:
        return
    identity = _load_agent_identity(connection)
    if identity is None:
        return
    session_id = snapshot.operation.session_id
    if session_id is None:
        return
    row = connection.execute(
        "SELECT updated_at FROM sessions WHERE id = ? AND agent_id = ?",
        (session_id, snapshot.operation.agent_id),
    ).fetchone()
    if row is None:
        raise InvalidOperationCheckpointError(
            snapshot.operation.id,
            "session-scoped operation requires a persisted session",
        )
    current_updated_at = _decode_datetime(_sqlite_text(row[0], "session updated_at"))
    updated_at = max(current_updated_at, snapshot.operation.updated_at)
    connection.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ? AND agent_id = ?",
        (
            _encode_datetime(updated_at),
            session_id,
            snapshot.operation.agent_id,
        ),
    )


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
    base_values = (
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
    )
    if _pragma_int(connection, "user_version") < 13:
        if task.execution_facts.validation_facts.schema_version != 0:
            raise ValueError(
                "historical task schema cannot store explicit validation facts"
            )
        connection.execute(
            "INSERT INTO tasks("
            "operation_id, position, id, turn_id, call_id, capability_id, "
            "executor_id, status, attempt, arguments_json, created_at, updated_at, "
            "error_code, cancellation_requested, capability_fingerprint, "
            "arguments_hash, access_mode, risk, side_effecting, idempotent, "
            "replay_safe, idempotency_key, manual_recovery_reason"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?)",
            base_values,
        )
        return

    validation = task.execution_facts.validation_facts
    connection.execute(
        "INSERT INTO tasks("
        "operation_id, position, id, turn_id, call_id, capability_id, "
        "executor_id, status, attempt, arguments_json, created_at, updated_at, "
        "error_code, cancellation_requested, capability_fingerprint, "
        "arguments_hash, access_mode, risk, side_effecting, idempotent, "
        "replay_safe, idempotency_key, manual_recovery_reason, "
        "validation_schema_version, validation_passed, validation_in_scope, "
        "validation_destructive, validation_sensitivity_class, "
        "validation_source_id, validation_resource_ids_json, "
        "validation_resource_revisions_json, validation_source_revision, "
        "validation_impact_json, validation_evidence_ids_json"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            *base_values,
            validation.schema_version,
            int(validation.validation_passed),
            int(validation.in_scope),
            int(validation.destructive),
            validation.sensitivity_class,
            validation.source_id,
            canonical_json(validation.resource_ids),
            canonical_json(validation.resource_revisions),
            validation.source_revision,
            canonical_json(validation.impact),
            canonical_json(validation.evidence_ids),
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


def _insert_approval(
    connection: sqlite3.Connection,
    operation_id: str,
    position: int,
    approval: ApprovalRequest,
) -> None:
    connection.execute(
        "INSERT INTO approvals("
        "operation_id, position, id, task_id, task_fingerprint, "
        "policy_fingerprint, requested_at, status, decided_at, decided_by, "
        "decision_reason"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            operation_id,
            position,
            approval.id,
            approval.task_id,
            approval.task_fingerprint,
            approval.policy_fingerprint,
            _encode_datetime(approval.requested_at),
            approval.status.value,
            _encode_optional_datetime(approval.decided_at),
            approval.decided_by,
            approval.decision_reason,
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
    has_approval_id = _pragma_int(connection, "user_version") >= 6
    has_monitor_id = _pragma_int(connection, "user_version") >= 10
    if not has_monitor_id and event.monitor_id is not None:
        raise SQLiteCorruptionError(
            "historical runtime event schema cannot store monitor identity"
        )
    if has_monitor_id:
        connection.execute(
            "INSERT INTO runtime_events("
            "id, operation_id, position, type, agent_id, agent_sequence, created_at, "
            "session_id, turn_id, model_call_id, call_id, task_id, evidence_id, "
            "capability_id, executor_id, payload_json, approval_id, monitor_id"
            ") VALUES (?, ?, ?, ?, ?, ("
            "SELECT COALESCE(MAX(agent_sequence), 0) + 1 FROM runtime_events "
            "WHERE agent_id = ?"
            "), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                event.approval_id,
                event.monitor_id,
            ),
        )
        return
    if not has_approval_id:
        if event.approval_id is not None:
            raise SQLiteCorruptionError(
                "historical runtime event schema cannot store approval identity"
            )
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
        return
    connection.execute(
        "INSERT INTO runtime_events("
        "id, operation_id, position, type, agent_id, agent_sequence, created_at, "
        "session_id, turn_id, model_call_id, call_id, task_id, evidence_id, "
        "capability_id, executor_id, payload_json, approval_id"
        ") VALUES (?, ?, ?, ?, ?, ("
        "SELECT COALESCE(MAX(agent_sequence), 0) + 1 FROM runtime_events "
        "WHERE agent_id = ?"
        "), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            event.approval_id,
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


def _latest_committed_event_cursor(
    connection: sqlite3.Connection,
    agent_id: str,
) -> EventCursor | None:
    row = connection.execute(
        "SELECT MAX(agent_sequence) AS sequence FROM runtime_events "
        "WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row is None or row["sequence"] is None:
        return None
    return EventCursor(
        agent_id=agent_id,
        sequence=_sqlite_int(row["sequence"], "latest event agent sequence"),
    )


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
        approval_id=_optional_text(row["approval_id"]),
        monitor_id=(
            _optional_text(row["monitor_id"]) if "monitor_id" in row.keys() else None
        ),
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


def _load_versioned_by_approval(
    connection: sqlite3.Connection,
    approval_id: str,
) -> VersionedOperation | None:
    connection.execute("BEGIN")
    try:
        row = connection.execute(
            "SELECT operation_id FROM approvals WHERE id = ?",
            (approval_id,),
        ).fetchone()
        result = (
            None
            if row is None
            else _load_versioned_operation_in_transaction(
                connection,
                _sqlite_text(row[0], "approval operation id"),
            )
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _load_nonterminal_operations(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[VersionedOperation, ...]:
    connection.execute("BEGIN")
    try:
        rows = connection.execute(
            "SELECT id, status FROM operations "
            "WHERE agent_id = ? "
            "ORDER BY updated_at ASC, id ASC",
            (agent_id,),
        ).fetchall()
        result: list[VersionedOperation] = []
        for row in rows:
            try:
                operation_id = _sqlite_text(row[0], "operation id")
                raw_status = _sqlite_text(row[1], "operation status")
                status = OperationStatus(raw_status)
            except (TypeError, ValueError) as error:
                raise SQLiteCorruptionError(
                    "cannot classify agent-scoped operation status"
                ) from error
            if status in _NONTERMINAL_OPERATION_STATUSES:
                result.append(
                    _load_versioned_operation_in_transaction(
                        connection,
                        operation_id,
                    )
                )
                continue
            if status not in _TERMINAL_OPERATION_STATUSES:
                raise SQLiteCorruptionError(
                    f"unclassified operation {operation_id} status: {raw_status!r}"
                )
        connection.execute("COMMIT")
        return tuple(result)
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
    approval_rows = _operation_rows(connection, "approvals", operation_id)
    evidence_rows = _operation_rows(connection, "evidence", operation_id)
    observation_rows = _operation_rows(connection, "observations", operation_id)
    event_rows = _operation_rows(connection, "runtime_events", operation_id)
    has_task_validation = _pragma_int(connection, "user_version") >= 13

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
                validation_facts=_decode_task_validation_facts(
                    row,
                    enabled=has_task_validation,
                ),
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
    approvals = tuple(
        ApprovalRequest(
            id=_sqlite_text(row["id"], "approval id"),
            operation_id=operation_id,
            task_id=_sqlite_text(row["task_id"], "approval task id"),
            task_fingerprint=_sqlite_text(
                row["task_fingerprint"],
                "approval task fingerprint",
            ),
            policy_fingerprint=_sqlite_text(
                row["policy_fingerprint"],
                "approval policy fingerprint",
            ),
            requested_at=_decode_datetime(
                _sqlite_text(row["requested_at"], "approval requested_at")
            ),
            status=ApprovalStatus(_sqlite_text(row["status"], "approval status")),
            decided_at=_decode_optional_datetime(
                row["decided_at"],
                "approval decided_at",
            ),
            decided_by=_optional_text(row["decided_by"]),
            decision_reason=_optional_text(row["decision_reason"]),
        )
        for row in approval_rows
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
        approvals=approvals,
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
        "approvals",
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


def _decode_revision_pairs(value: str) -> tuple[tuple[str, str], ...]:
    decoded = _decode_json(value)
    if not isinstance(decoded, list):
        raise ValueError("persisted revisions must be an array")
    revisions: list[tuple[str, str]] = []
    for item in decoded:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
        ):
            raise ValueError(
                "persisted revisions must contain identifier/revision pairs"
            )
        revisions.append((item[0], item[1]))
    return tuple(revisions)


def _decode_task_validation_facts(
    row: sqlite3.Row,
    *,
    enabled: bool,
) -> ActionValidationFacts:
    if not enabled:
        return ActionValidationFacts()
    return ActionValidationFacts(
        schema_version=_sqlite_int(
            row["validation_schema_version"],
            "task validation schema version",
        ),
        validation_passed=_decode_bool(row["validation_passed"]),
        in_scope=_decode_bool(row["validation_in_scope"]),
        destructive=_decode_bool(row["validation_destructive"]),
        sensitivity_class=_sqlite_text(
            row["validation_sensitivity_class"],
            "task validation sensitivity class",
        ),
        source_id=_optional_text(row["validation_source_id"]),
        resource_ids=_decode_string_tuple(
            _sqlite_text(
                row["validation_resource_ids_json"],
                "task validation resource ids",
            )
        ),
        resource_revisions=_decode_revision_pairs(
            _sqlite_text(
                row["validation_resource_revisions_json"],
                "task validation resource revisions",
            )
        ),
        source_revision=_optional_text(row["validation_source_revision"]),
        impact=_decode_json_object(
            _sqlite_text(
                row["validation_impact_json"],
                "task validation impact",
            )
        ),
        evidence_ids=_decode_string_tuple(
            _sqlite_text(
                row["validation_evidence_ids_json"],
                "task validation evidence ids",
            )
        ),
    )


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
    data: dict[str, object] = {
        "arguments": call.arguments,
        "id": call.id,
        "name": call.name,
    }
    if call.provider_call_id is not None:
        data["provider_call_id"] = call.provider_call_id
    return data


def _tool_call_from_data(value: object) -> ToolCall:
    if not isinstance(value, dict):
        raise ValueError("tool call must be a JSON object")
    keys = frozenset(value)
    legacy_keys = frozenset({"arguments", "id", "name"})
    current_keys = legacy_keys | {"provider_call_id"}
    if keys not in (legacy_keys, current_keys):
        raise ValueError("tool call has unknown or missing fields")
    data = value
    arguments = data["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("tool-call arguments must be a JSON object")
    return ToolCall(
        id=_expect_text(data["id"], "tool-call id"),
        name=_expect_text(data["name"], "tool-call name"),
        arguments=arguments,
        provider_call_id=_expect_optional_text(
            data.get("provider_call_id"),
            "tool-call provider_call_id",
        ),
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
    data: dict[str, object] = {
        "agent_id": message.agent_id,
        "content": [_content_block_to_data(block) for block in message.content],
        "operation_id": message.operation_id,
        "role": message.role.value,
        "session_id": message.session_id,
        "tool_calls": [_tool_call_to_data(call) for call in message.tool_calls],
        "turn_id": message.turn_id,
    }
    if message.provider_id is not None:
        data["provider_id"] = message.provider_id
    if message.provider_metadata:
        data["provider_metadata"] = message.provider_metadata
    return data


def _message_from_data(value: object) -> CanonicalMessage:
    if not isinstance(value, dict):
        raise ValueError("canonical message must be a JSON object")
    legacy_keys = frozenset(
        {
            "agent_id",
            "content",
            "operation_id",
            "role",
            "session_id",
            "tool_calls",
            "turn_id",
        }
    )
    optional_keys = frozenset({"provider_id", "provider_metadata"})
    message_keys = frozenset(value)
    if (
        not legacy_keys <= message_keys
        or not message_keys <= legacy_keys | optional_keys
    ):
        raise ValueError("canonical message has unknown or missing fields")
    data = value
    provider_metadata = data.get("provider_metadata", {})
    if not isinstance(provider_metadata, dict):
        raise ValueError("message provider_metadata must be a JSON object")
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
        provider_id=_expect_optional_text(
            data.get("provider_id"),
            "message provider_id",
        ),
        provider_metadata=provider_metadata,
    )


def _encode_model_request(request: ModelRequest) -> str:
    return canonical_json(
        {
            "codec_version": 3,
            "context_selection": request.context_selection,
            "messages": [_message_to_data(message) for message in request.messages],
            "operation_id": request.operation_id,
            "response_schema": request.response_schema,
            "sensitivity": request.sensitivity.value,
            "tools": [_tool_definition_to_data(tool) for tool in request.tools],
            "turn_id": request.turn_id,
        }
    )


def _decode_model_request(value: str) -> ModelRequest:
    decoded = _decode_json(value)
    if not isinstance(decoded, dict):
        raise ValueError("model request must be a JSON object")
    codec_version = _expect_int(
        decoded.get("codec_version"),
        "model-request codec version",
    )
    legacy_keys = {"codec_version", "messages", "operation_id", "tools", "turn_id"}
    if codec_version == 1:
        data = _expect_object(decoded, keys=legacy_keys, label="model request")
        context_selection: Mapping[str, object] = {}
        response_schema: Mapping[str, object] | None = None
        sensitivity = ModelSensitivity.INTERNAL
    elif codec_version == 2:
        data = _expect_object(
            decoded,
            keys=legacy_keys | {"context_selection"},
            label="model request",
        )
        selection_value = data["context_selection"]
        if not isinstance(selection_value, dict):
            raise ValueError("model-request context selection must be a JSON object")
        context_selection = selection_value
        response_schema = None
        sensitivity = ModelSensitivity.INTERNAL
    elif codec_version == 3:
        data = _expect_object(
            decoded,
            keys=legacy_keys | {"context_selection", "response_schema", "sensitivity"},
            label="model request",
        )
        selection_value = data["context_selection"]
        if not isinstance(selection_value, dict):
            raise ValueError("model-request context selection must be a JSON object")
        context_selection = selection_value
        schema_value = data["response_schema"]
        if schema_value is not None and not isinstance(schema_value, dict):
            raise ValueError("model-request response schema must be a JSON object")
        response_schema = schema_value
        sensitivity = ModelSensitivity(
            _expect_text(data["sensitivity"], "model-request sensitivity")
        )
    else:
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
        response_schema=response_schema,
        sensitivity=sensitivity,
        context_selection=context_selection,
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


def _routing_to_data(routing: ModelRoutingTrace) -> dict[str, object]:
    return {
        "attempts": [
            {
                "attempt": item.attempt,
                "error_code": item.error_code,
                "latency_ms": item.latency_ms,
                "outcome": item.outcome.value,
                "provider_id": item.provider_id,
            }
            for item in routing.attempts
        ],
        "primary_provider_id": routing.primary_provider_id,
        "route_id": routing.route_id,
        "selected_provider_id": routing.selected_provider_id,
        "terminal_error_code": routing.terminal_error_code,
    }


def _routing_from_data(value: object) -> ModelRoutingTrace:
    data = _expect_object(
        value,
        keys={
            "attempts",
            "primary_provider_id",
            "route_id",
            "selected_provider_id",
            "terminal_error_code",
        },
        label="model routing trace",
    )
    attempts: list[ModelRouteAttempt] = []
    for value_item in _expect_list(data["attempts"], "model routing attempts"):
        item = _expect_object(
            value_item,
            keys={
                "attempt",
                "error_code",
                "latency_ms",
                "outcome",
                "provider_id",
            },
            label="model routing attempt",
        )
        attempts.append(
            ModelRouteAttempt(
                provider_id=_expect_text(
                    item["provider_id"], "route-attempt provider id"
                ),
                attempt=_expect_int(item["attempt"], "route-attempt number"),
                outcome=ModelRouteAttemptOutcome(
                    _expect_text(item["outcome"], "route-attempt outcome")
                ),
                latency_ms=_expect_int(item["latency_ms"], "route-attempt latency"),
                error_code=_expect_optional_text(
                    item["error_code"], "route-attempt error code"
                ),
            )
        )
    return ModelRoutingTrace(
        route_id=_expect_text(data["route_id"], "routing route id"),
        primary_provider_id=_expect_text(
            data["primary_provider_id"], "routing primary provider id"
        ),
        attempts=tuple(attempts),
        selected_provider_id=_expect_optional_text(
            data["selected_provider_id"], "routing selected provider id"
        ),
        terminal_error_code=_expect_optional_text(
            data["terminal_error_code"], "routing terminal error code"
        ),
    )


def _encode_model_response(response: ModelResponse) -> str:
    return canonical_json(
        {
            "codec_version": 2,
            "finish_reason": response.finish_reason.value,
            "provider_id": response.provider_id,
            "provider_metadata": response.provider_metadata,
            "provider_response_id": response.provider_response_id,
            "routing": (
                None if response.routing is None else _routing_to_data(response.routing)
            ),
            "text": response.text,
            "tool_calls": [_tool_call_to_data(call) for call in response.tool_calls],
            "usage": _usage_to_data(response.usage),
        }
    )


def _decode_model_response(value: str) -> ModelResponse:
    decoded = _decode_json(value)
    if not isinstance(decoded, dict):
        raise ValueError("model response must be a JSON object")
    codec_version = _expect_int(
        decoded.get("codec_version"),
        "model-response codec version",
    )
    legacy_keys = {
        "codec_version",
        "finish_reason",
        "provider_metadata",
        "provider_response_id",
        "text",
        "tool_calls",
        "usage",
    }
    if codec_version == 1:
        data = _expect_object(decoded, keys=legacy_keys, label="model response")
        provider_id = None
        routing = None
    elif codec_version == 2:
        data = _expect_object(
            decoded,
            keys=legacy_keys | {"provider_id", "routing"},
            label="model response",
        )
        provider_id = _expect_optional_text(data["provider_id"], "response provider id")
        routing_value = data["routing"]
        routing = None if routing_value is None else _routing_from_data(routing_value)
    else:
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
        provider_id=provider_id,
        provider_response_id=_expect_optional_text(
            data["provider_response_id"],
            "provider response id",
        ),
        provider_metadata=provider_metadata,
        routing=routing,
    )
