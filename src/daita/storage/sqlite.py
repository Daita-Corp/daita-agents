"""Small SQLite state store for one embedded agent.

The database persists only product state the MVP actually uses: identity,
attached sources, current catalog snapshots, exact run transcripts, and the
minimal receipt needed to classify an external database-write attempt.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

from .._json import FrozenJsonObject, canonical_json
from ..adapters.models import SourceRegistration
from ..artifacts.models import (
    ArtifactAuthorship,
    ArtifactDeliveryReceipt,
    ArtifactProvenance,
    ArtifactRef,
    ArtifactResourceBinding,
    artifact_ref_from_mapping,
)
from ..catalog.models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSnapshotRef,
    CatalogSummary,
    CatalogSync,
    CatalogSyncStatus,
    FacetKind,
    RelationshipDirection,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
)
from ..catalog.protocols import CatalogStoreError
from ..errors import StateCompatibilityCode, StateCompatibilityError
from ..identity import AgentIdentity, AgentIdentityConflictError
from ..learning_candidates import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_STAMPS,
    DocumentCandidateContent,
    LearningCandidate,
    LearningCandidateAction,
    LearningCandidateError,
    LearningCandidateNotFoundError,
    LearningCandidateRejectionReason,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateStatus,
    LearningCandidateTarget,
    LearningReviewRunTail,
    SemanticCandidateContent,
    SkillCandidateContent,
)
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from ..llm.pricing import (
    CostBasis,
    CostComponent,
    CostEstimate,
    CostEstimateStatus,
    PricingModifier,
    PricingUsageRange,
)
from ..loop.models import ConversationRun, LoopExit, LoopExitKind, RunInput, Transcript
from ..semantics import (
    SEMANTIC_MAX_ANNOTATIONS,
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticDigestMismatchError,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticNotFoundError,
    SemanticSubject,
    SemanticValidationError,
    semantic_annotation_sha256,
)

_CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH = 64
_ACTIVE_SOURCE_KEY_PREFIX = "active_source:"
_LEARNING_REVIEW_STAMPS_KEY_PREFIX = "learning_review_stamps:"
_T = TypeVar("_T")
_DATABASE_WRITE_RECEIPT_ID = re.compile(r"database-write-receipt:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")


class DatabaseWriteOutcome(str, Enum):
    STARTED = "started"
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class DatabaseWriteReceiptConflictError(RuntimeError):
    """The durable receipt identity or immutable terminal state conflicts."""


def _database_write_text(value: str, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty text without surrounding space")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def _database_write_aware(value: datetime, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")
    return value


def database_write_receipt_id(
    *,
    agent_id: str,
    run_id: str,
    call_id: str,
    capability_id: str,
    intent_sha256: str,
) -> str:
    identity = {
        "agent_id": _database_write_text(agent_id, "receipt agent_id"),
        "call_id": _database_write_text(call_id, "receipt call_id"),
        "capability_id": _database_write_text(
            capability_id, "receipt capability_id", maximum=128
        ),
        "intent_sha256": intent_sha256,
        "run_id": _database_write_text(run_id, "receipt run_id"),
    }
    if (
        not isinstance(intent_sha256, str)
        or _DATABASE_WRITE_HASH.fullmatch(intent_sha256) is None
    ):
        raise ValueError("receipt intent_sha256 must be a sha256 hash")
    digest = sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return f"database-write-receipt:sha256:{digest}"


@dataclass(frozen=True, slots=True)
class DatabaseWriteReceipt:
    """Bounded durable metadata for one exact external database-write attempt."""

    receipt_id: str
    agent_id: str
    run_id: str
    call_id: str
    capability_id: str
    source_id: str
    resource_id: str
    intent_sha256: str
    preview_fingerprint: str
    outcome: DatabaseWriteOutcome
    affected_rows: int | None
    normalized_error_code: str | None
    started_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.receipt_id, str)
            or _DATABASE_WRITE_RECEIPT_ID.fullmatch(self.receipt_id) is None
        ):
            raise ValueError("receipt_id must be a canonical database-write receipt id")
        _database_write_text(self.agent_id, "receipt agent_id")
        _database_write_text(self.run_id, "receipt run_id")
        _database_write_text(self.call_id, "receipt call_id")
        _database_write_text(self.capability_id, "receipt capability_id", maximum=128)
        if (
            not isinstance(self.source_id, str)
            or _DATABASE_WRITE_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("receipt source_id must be a canonical source id")
        if (
            not isinstance(self.resource_id, str)
            or _DATABASE_WRITE_RESOURCE_ID.fullmatch(self.resource_id) is None
        ):
            raise ValueError("receipt resource_id must be a canonical resource id")
        for value, name in (
            (self.intent_sha256, "intent_sha256"),
            (self.preview_fingerprint, "preview_fingerprint"),
        ):
            if (
                not isinstance(value, str)
                or _DATABASE_WRITE_HASH.fullmatch(value) is None
            ):
                raise ValueError(f"receipt {name} must be a sha256 hash")
        if not isinstance(self.outcome, DatabaseWriteOutcome):
            raise TypeError("receipt outcome must be a DatabaseWriteOutcome")
        _database_write_aware(self.started_at, "receipt started_at")
        if self.completed_at is not None:
            _database_write_aware(self.completed_at, "receipt completed_at")
            if self.completed_at < self.started_at:
                raise ValueError("receipt cannot complete before it starts")
        expected_id = database_write_receipt_id(
            agent_id=self.agent_id,
            run_id=self.run_id,
            call_id=self.call_id,
            capability_id=self.capability_id,
            intent_sha256=self.intent_sha256,
        )
        if self.receipt_id != expected_id:
            raise ValueError("receipt_id does not match its execution identity")
        if self.normalized_error_code is not None and (
            not isinstance(self.normalized_error_code, str)
            or _DATABASE_WRITE_ERROR_CODE.fullmatch(self.normalized_error_code) is None
        ):
            raise ValueError("receipt normalized_error_code is invalid")
        if self.affected_rows is not None and (
            not isinstance(self.affected_rows, int)
            or isinstance(self.affected_rows, bool)
        ):
            raise TypeError("receipt affected_rows must be an integer or None")
        if self.outcome is DatabaseWriteOutcome.STARTED:
            if any(
                value is not None
                for value in (
                    self.affected_rows,
                    self.normalized_error_code,
                    self.completed_at,
                )
            ):
                raise ValueError("started receipt cannot contain terminal fields")
        elif self.outcome is DatabaseWriteOutcome.COMMITTED:
            if (
                self.affected_rows != 1
                or self.normalized_error_code is not None
                or self.completed_at is None
            ):
                raise ValueError("committed receipt must record one affected row")
        elif self.outcome is DatabaseWriteOutcome.NOT_COMMITTED:
            if (
                self.affected_rows != 0
                or self.normalized_error_code is None
                or self.completed_at is None
            ):
                raise ValueError(
                    "not_committed receipt must record zero rows and an error code"
                )
        elif (
            self.affected_rows is not None
            or self.normalized_error_code != "write_outcome_unknown"
            or self.completed_at is None
        ):
            raise ValueError(
                "outcome_unknown receipt must omit affected rows and use its stable code"
            )

    @classmethod
    def start(
        cls,
        *,
        agent_id: str,
        run_id: str,
        call_id: str,
        capability_id: str,
        source_id: str,
        resource_id: str,
        intent_sha256: str,
        preview_fingerprint: str,
        started_at: datetime,
    ) -> DatabaseWriteReceipt:
        return cls(
            receipt_id=database_write_receipt_id(
                agent_id=agent_id,
                run_id=run_id,
                call_id=call_id,
                capability_id=capability_id,
                intent_sha256=intent_sha256,
            ),
            agent_id=agent_id,
            run_id=run_id,
            call_id=call_id,
            capability_id=capability_id,
            source_id=source_id,
            resource_id=resource_id,
            intent_sha256=intent_sha256,
            preview_fingerprint=preview_fingerprint,
            outcome=DatabaseWriteOutcome.STARTED,
            affected_rows=None,
            normalized_error_code=None,
            started_at=started_at,
            completed_at=None,
        )

    def finish(
        self,
        outcome: DatabaseWriteOutcome,
        *,
        completed_at: datetime,
        affected_rows: int | None,
        normalized_error_code: str | None,
    ) -> DatabaseWriteReceipt:
        if self.outcome is not DatabaseWriteOutcome.STARTED:
            raise ValueError("only a started receipt can reach a terminal outcome")
        if outcome is DatabaseWriteOutcome.STARTED:
            raise ValueError("receipt terminal outcome cannot be started")
        return replace(
            self,
            outcome=outcome,
            affected_rows=affected_rows,
            normalized_error_code=normalized_error_code,
            completed_at=completed_at,
        )

    def as_started(self) -> DatabaseWriteReceipt:
        return replace(
            self,
            outcome=DatabaseWriteOutcome.STARTED,
            affected_rows=None,
            normalized_error_code=None,
            completed_at=None,
        )


STATE_FORMAT_VERSION = 3
_UNVERSIONED_STATE_FORMAT = 0
_LEGACY_TABLE_MARKERS = frozenset(
    {
        "evidence",
        "events",
        "operations",
        "tasks",
    }
)

_INITIAL_STATE_TABLE_DEFINITIONS = {
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
_RECEIPT_STATE_TABLE_DEFINITIONS = {
    "database_write_receipts": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("run_id", "TEXT", 1, None, 0),
        ("call_id", "TEXT", 1, None, 0),
        ("data", "TEXT", 1, None, 0),
    ),
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
_CURRENT_STATE_TABLE_DEFINITIONS = _RECEIPT_STATE_TABLE_DEFINITIONS
_MESSAGES_FOREIGN_KEYS = (("runs", "run_id", "id", "NO ACTION", "CASCADE", "NONE"),)
_NAMED_INDEXES = {
    "runs_conversation_turn": (
        "runs",
        True,
        ("agent_id", "conversation_id", "turn_index"),
    )
}

_StateSchema = Mapping[str, tuple[tuple[object, ...], ...]]


@dataclass(frozen=True, slots=True)
class _StateMigration:
    target_format: int
    source_schema: _StateSchema
    target_schema: _StateSchema
    apply: Callable[[sqlite3.Connection], None]
    validate_target: Callable[[sqlite3.Connection], None] | None = None

    def __post_init__(self) -> None:
        if self.target_format <= 0:
            raise ValueError("migration target format must be positive")


def _active_source_key(agent_id: str) -> str:
    return f"{_ACTIVE_SOURCE_KEY_PREFIX}{agent_id}"


def _learning_review_stamps_key(agent_id: str) -> str:
    return f"{_LEARNING_REVIEW_STAMPS_KEY_PREFIX}{agent_id}"


class _CatalogCommitGate:
    """Linearize task cancellation against one SQLite mutation transaction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._started = False

    def start(self, connection: sqlite3.Connection) -> bool:
        with self._lock:
            if self._cancelled:
                return False
        connection.execute("BEGIN IMMEDIATE")
        with self._lock:
            if self._cancelled:
                connection.rollback()
                return False
            self._started = True
            return True

    def cancel_before_start(self) -> bool:
        with self._lock:
            if self._started:
                return False
            self._cancelled = True
            return True


class SQLiteStateStore:
    """Versioned persistence boundary for one admitted local agent home."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._decoded_catalog_snapshots: dict[
            tuple[str, str, str], SourceCatalogSnapshot
        ] = {}
        self._decoded_catalog_snapshot_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        **_: object,
    ) -> SQLiteStateStore:
        resolved = Path(path).resolve()
        resolved_clock = clock or (lambda: datetime.now(timezone.utc))

        def admit() -> None:
            _initialize(resolved)
            _recover_started_database_write_receipts(resolved, resolved_clock)

        worker = asyncio.create_task(asyncio.to_thread(admit))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return cls(resolved)

    async def close(self) -> None:
        async with self._decoded_catalog_snapshot_lock:
            self._decoded_catalog_snapshots.clear()

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity:
        def write() -> AgentIdentity:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
                if row is not None:
                    current = _expect(_loads(row[0]), AgentIdentity)
                    if current != identity:
                        raise AgentIdentityConflictError(
                            "state database already belongs to another agent"
                        )
                    return current
                connection.execute(
                    "INSERT INTO metadata(key, data) VALUES ('identity', ?)",
                    (_dumps(identity),),
                )
                return identity

        return await asyncio.to_thread(write)

    async def load_identity(self) -> AgentIdentity | None:
        def read() -> AgentIdentity | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), AgentIdentity)

        return await asyncio.to_thread(read)

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration:
        def write() -> SourceRegistration:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (registration.agent_id, registration.id),
                ).fetchone()
                if row is not None:
                    current = _expect(_loads(row[0]), SourceRegistration)
                    if current != registration:
                        raise ValueError(
                            f"source registration already exists: {registration.id}"
                        )
                    return current
                connection.execute(
                    "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                    (registration.agent_id, registration.id, _dumps(registration)),
                )
                return registration

        return await asyncio.to_thread(write)

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None:
        def read() -> SourceRegistration | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (agent_id, source_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), SourceRegistration)

        return await asyncio.to_thread(read)

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        def read() -> tuple[SourceRegistration, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? ORDER BY id",
                    (agent_id,),
                ).fetchall()
            return tuple(_expect(_loads(row[0]), SourceRegistration) for row in rows)

        return await asyncio.to_thread(read)

    async def load_database_write_receipt(
        self,
        agent_id: str,
        receipt_id: str,
    ) -> DatabaseWriteReceipt | None:
        _database_write_text(agent_id, "receipt agent_id")
        if (
            not isinstance(receipt_id, str)
            or _DATABASE_WRITE_RECEIPT_ID.fullmatch(receipt_id) is None
        ):
            raise ValueError("receipt_id must be a canonical database-write receipt id")

        def read() -> DatabaseWriteReceipt | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, receipt_id),
                ).fetchone()
            return (
                None if row is None else _expect(_loads(row[0]), DatabaseWriteReceipt)
            )

        return await asyncio.to_thread(read)

    async def load_database_write_receipt_for_call(
        self,
        agent_id: str,
        run_id: str,
        call_id: str,
    ) -> DatabaseWriteReceipt | None:
        _database_write_text(agent_id, "receipt agent_id")
        _database_write_text(run_id, "receipt run_id")
        _database_write_text(call_id, "receipt call_id")

        def read() -> DatabaseWriteReceipt | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ? AND run_id = ? AND call_id = ?""",
                    (agent_id, run_id, call_id),
                ).fetchone()
            return (
                None if row is None else _expect(_loads(row[0]), DatabaseWriteReceipt)
            )

        return await asyncio.to_thread(read)

    async def start_database_write_receipt(
        self,
        receipt: DatabaseWriteReceipt,
    ) -> DatabaseWriteReceipt:
        if not isinstance(receipt, DatabaseWriteReceipt):
            raise TypeError("receipt must be a DatabaseWriteReceipt")
        if receipt.outcome is not DatabaseWriteOutcome.STARTED:
            raise ValueError(
                "database write execution must begin with a started receipt"
            )

        def write() -> DatabaseWriteReceipt:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ?
                         AND (id = ? OR (run_id = ? AND call_id = ?))""",
                    (
                        receipt.agent_id,
                        receipt.receipt_id,
                        receipt.run_id,
                        receipt.call_id,
                    ),
                ).fetchone()
                if current is not None:
                    raise DatabaseWriteReceiptConflictError(
                        "execution identity already has a receipt"
                    )
                connection.execute(
                    """INSERT INTO database_write_receipts(
                           agent_id, id, run_id, call_id, data
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        receipt.agent_id,
                        receipt.receipt_id,
                        receipt.run_id,
                        receipt.call_id,
                        _dumps(receipt),
                    ),
                )
                return receipt

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        result = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def finish_database_write_receipt(
        self,
        receipt: DatabaseWriteReceipt,
    ) -> DatabaseWriteReceipt:
        if not isinstance(receipt, DatabaseWriteReceipt):
            raise TypeError("receipt must be a DatabaseWriteReceipt")
        if receipt.outcome is DatabaseWriteOutcome.STARTED:
            raise ValueError(
                "finish requires a terminal receipt, not a started receipt"
            )

        def write() -> DatabaseWriteReceipt:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ?
                         AND (id = ? OR (run_id = ? AND call_id = ?))""",
                    (
                        receipt.agent_id,
                        receipt.receipt_id,
                        receipt.run_id,
                        receipt.call_id,
                    ),
                ).fetchone()
                if row is None:
                    raise DatabaseWriteReceiptConflictError(
                        "started receipt does not exist"
                    )
                current = _expect(_loads(row[0]), DatabaseWriteReceipt)
                if current.outcome is not DatabaseWriteOutcome.STARTED:
                    if current == receipt:
                        return current
                    raise DatabaseWriteReceiptConflictError(
                        "terminal receipt is immutable"
                    )
                if current != receipt.as_started():
                    raise DatabaseWriteReceiptConflictError(
                        "terminal receipt does not match its started identity"
                    )
                connection.execute(
                    """UPDATE database_write_receipts SET data = ?
                       WHERE agent_id = ? AND id = ?""",
                    (_dumps(receipt), receipt.agent_id, receipt.receipt_id),
                )
                return receipt

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        result = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def load_active_source_id(self, agent_id: str) -> str | None:
        def read() -> str | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = ?",
                    (_active_source_key(agent_id),),
                ).fetchone()
            if row is None:
                return None
            source_id = _loads(row[0])
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("stored active source id is invalid")
            return source_id

        return await asyncio.to_thread(read)

    async def set_active_source_id(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")

        def write() -> SourceRegistration:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (agent_id, source_id),
                ).fetchone()
                if row is None:
                    raise ValueError("unknown active source for this agent")
                registration = _expect(_loads(row[0]), SourceRegistration)
                if not registration.active:
                    raise ValueError("unknown active source for this agent")
                connection.execute(
                    """INSERT INTO metadata(key, data) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                    (_active_source_key(agent_id), _dumps(source_id)),
                )
                return registration

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        registration = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return registration

    async def set_source_write_access(
        self,
        agent_id: str,
        source_id: str,
        enabled: bool,
    ) -> SourceRegistration:
        """Atomically replace only one owned PostgreSQL admission flag."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        gate = _CatalogCommitGate()

        def write() -> SourceRegistration | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (agent_id, source_id),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        "source must be an active PostgreSQL source owned by this agent"
                    )
                current = _expect(_loads(row[0]), SourceRegistration)
                if (
                    current.agent_id != agent_id
                    or current.id != source_id
                    or current.adapter_id != "postgresql"
                    or not current.active
                ):
                    raise ValueError(
                        "source must be an active PostgreSQL source owned by this agent"
                    )
                current_flag = current.configuration.get("write_access", False)
                if not isinstance(current_flag, bool):
                    raise ValueError("PostgreSQL source write_access is invalid")
                if current_flag is enabled and "write_access" in current.configuration:
                    connection.commit()
                    return current
                configuration = dict(current.configuration)
                configuration["write_access"] = enabled
                updated = replace(current, configuration=configuration)
                connection.execute(
                    "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
                    (_dumps(updated), agent_id, source_id),
                )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        updated = worker.result()
        if cancelled_before_start:
            if updated is not None:
                raise AssertionError("cancelled source admission transaction committed")
            raise asyncio.CancelledError
        if updated is None:
            raise AssertionError(
                "source admission transaction stopped without cancellation"
            )
        if cancelled:
            raise asyncio.CancelledError
        return updated

    async def detach_source(
        self, agent_id: str, source_id: str, detached_at: datetime
    ) -> SourceRegistration:
        gate = _CatalogCommitGate()

        def write() -> SourceRegistration | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (agent_id, source_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown source: {source_id}")
                current = _expect(_loads(row[0]), SourceRegistration)
                detached = (
                    current if not current.active else current.detach(detached_at)
                )
                connection.execute(
                    "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
                    (_dumps(detached), agent_id, source_id),
                )
                selection = connection.execute(
                    "SELECT data FROM metadata WHERE key = ?",
                    (_active_source_key(agent_id),),
                ).fetchone()
                selected_id = None if selection is None else _loads(selection[0])
                if selected_id == source_id:
                    connection.execute(
                        "DELETE FROM metadata WHERE key = ?",
                        (_active_source_key(agent_id),),
                    )
                    remaining: list[SourceRegistration] = []
                    for (data,) in connection.execute(
                        "SELECT data FROM sources WHERE agent_id = ? ORDER BY id",
                        (agent_id,),
                    ).fetchall():
                        candidate = _expect(_loads(data), SourceRegistration)
                        if candidate.active:
                            remaining.append(candidate)
                    if len(remaining) == 1:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(agent_id),
                                _dumps(remaining[0].id),
                            ),
                        )
                connection.commit()
                return detached
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        detached = worker.result()
        if cancelled_before_start:
            if detached is not None:
                raise AssertionError("cancelled source detach transaction committed")
            raise asyncio.CancelledError
        if detached is None:
            raise AssertionError(
                "source detach transaction stopped without cancellation"
            )
        async with self._decoded_catalog_snapshot_lock:
            self._evict_decoded_catalog_source(agent_id, source_id)
        return detached

    async def record_sync(self, sync: CatalogSync) -> CatalogSync:
        def write() -> CatalogSync:
            with _connect(self.path) as connection:
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (sync.agent_id, sync.id, sync.source_id, _dumps(sync)),
                )
                return sync

        return await asyncio.to_thread(write)

    async def commit_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration | None = None,
    ) -> SourceCatalogSnapshot:
        sync = snapshot.sync
        if registration is not None and (
            registration.agent_id != sync.agent_id or registration.id != sync.source_id
        ):
            raise ValueError("catalog snapshot and source registration disagree")
        gate = _CatalogCommitGate()

        def write() -> SourceCatalogSnapshot | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                if registration is not None:
                    row = connection.execute(
                        "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                        (registration.agent_id, registration.id),
                    ).fetchone()
                    if row is None:
                        connection.execute(
                            "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                            (
                                registration.agent_id,
                                registration.id,
                                _dumps(registration),
                            ),
                        )
                    else:
                        current = _expect(_loads(row[0]), SourceRegistration)
                        if current != registration:
                            if (
                                current.active
                                or not registration.active
                                or current.adapter_id != registration.adapter_id
                                or current.native_identity
                                != registration.native_identity
                            ):
                                raise ValueError(
                                    "source registration already exists: "
                                    f"{registration.id}"
                                )
                            connection.execute(
                                """UPDATE sources SET data = ?
                                   WHERE agent_id = ? AND id = ?""",
                                (
                                    _dumps(registration),
                                    registration.agent_id,
                                    registration.id,
                                ),
                            )
                    selection = connection.execute(
                        "SELECT 1 FROM metadata WHERE key = ?",
                        (_active_source_key(registration.agent_id),),
                    ).fetchone()
                    if selection is None:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(registration.agent_id),
                                _dumps(registration.id),
                            ),
                        )
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (sync.agent_id, sync.id, sync.source_id, _dumps(sync)),
                )
                connection.execute(
                    """INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, source_id) DO UPDATE SET
                         sync_id = excluded.sync_id, data = excluded.data""",
                    (sync.agent_id, sync.source_id, sync.id, _dumps(snapshot)),
                )
                _commit_catalog_transaction(connection)
                return snapshot
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        committed = worker.result()
        if cancelled_before_start:
            if committed is not None:
                raise AssertionError("cancelled catalog transaction committed")
            raise asyncio.CancelledError
        if committed is None:
            raise AssertionError("catalog transaction stopped without cancellation")
        async with self._decoded_catalog_snapshot_lock:
            self._publish_decoded_catalog_snapshot(committed)
        return committed

    async def list_current_snapshot_refs(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
    ) -> tuple[CatalogSnapshotRef, ...]:
        _catalog_identifier(agent_id, "catalog snapshot agent_id")
        selected_source_ids = _catalog_snapshot_source_ids(source_ids)

        def read() -> tuple[CatalogSnapshotRef, ...]:
            with _connect(self.path) as connection:
                if not selected_source_ids:
                    rows = connection.execute(
                        "SELECT agent_id, source_id, sync_id FROM snapshots "
                        "WHERE agent_id = ? ORDER BY source_id, sync_id",
                        (agent_id,),
                    ).fetchall()
                else:
                    rows = []
                    for offset in range(
                        0,
                        len(selected_source_ids),
                        _CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH,
                    ):
                        batch = selected_source_ids[
                            offset : offset + _CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH
                        ]
                        placeholders = ", ".join("?" for _ in batch)
                        rows.extend(
                            connection.execute(
                                "SELECT agent_id, source_id, sync_id FROM snapshots "
                                f"WHERE agent_id = ? AND source_id IN ({placeholders})",
                                (agent_id, *batch),
                            ).fetchall()
                        )
            return tuple(
                CatalogSnapshotRef(
                    agent_id=row_agent_id,
                    source_id=source_id,
                    sync_id=sync_id,
                )
                for row_agent_id, source_id, sync_id in sorted(
                    rows,
                    key=lambda item: (item[1], item[2]),
                )
            )

        return await asyncio.to_thread(read)

    async def load_current_snapshot(
        self,
        ref: CatalogSnapshotRef,
    ) -> SourceCatalogSnapshot | None:
        if not isinstance(ref, CatalogSnapshotRef):
            raise TypeError("ref must be a CatalogSnapshotRef")
        key = (ref.agent_id, ref.source_id, ref.sync_id)

        async with self._decoded_catalog_snapshot_lock:
            cached = self._decoded_catalog_snapshots.get(key)
            if cached is not None:
                current_sync_id = await asyncio.to_thread(
                    _current_snapshot_sync_id,
                    self.path,
                    ref.agent_id,
                    ref.source_id,
                )
                if current_sync_id != ref.sync_id:
                    self._decoded_catalog_snapshots.pop(key, None)
                    return None
                return cached

            row = await asyncio.to_thread(
                _current_snapshot_row,
                self.path,
                ref.agent_id,
                ref.source_id,
            )
            if row is None or row[0] != ref.sync_id:
                return None
            snapshot = await asyncio.to_thread(_decode_catalog_snapshot, row[1])
            if (
                snapshot.sync.agent_id != ref.agent_id
                or snapshot.sync.source_id != ref.source_id
                or snapshot.sync.id != ref.sync_id
            ):
                raise CatalogStoreError(
                    "stored catalog snapshot does not match its exact reference"
                )
            current_sync_id = await asyncio.to_thread(
                _current_snapshot_sync_id,
                self.path,
                ref.agent_id,
                ref.source_id,
            )
            if current_sync_id != ref.sync_id:
                return None
            self._publish_decoded_catalog_snapshot(snapshot)
            return snapshot

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None:
        def read() -> CatalogSync | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM syncs WHERE agent_id = ? AND id = ?",
                    (agent_id, sync_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), CatalogSync)

        return await asyncio.to_thread(read)

    async def summarize_catalog(
        self,
        agent_id: str,
        active_source_ids: tuple[str, ...],
    ) -> CatalogSummary:
        """Count only current committed snapshots for the supplied active sources."""

        if not isinstance(active_source_ids, tuple) or any(
            not isinstance(source_id, str) or not source_id
            for source_id in active_source_ids
        ):
            raise TypeError("active_source_ids must be a tuple of non-empty strings")
        if len(active_source_ids) != len(set(active_source_ids)):
            raise ValueError("active_source_ids cannot contain duplicates")
        if not active_source_ids:
            return CatalogSummary(
                active_source_count=0,
                resource_count=0,
                relationship_count=0,
                latest_successful_sync_completed_at=None,
                is_empty=True,
            )
        snapshots = await self._snapshots(agent_id, active_source_ids)
        completion_times = tuple(
            snapshot.sync.completed_at
            for snapshot in snapshots
            if snapshot.sync.completed_at is not None
        )
        resource_count = sum(len(snapshot.resources) for snapshot in snapshots)
        return CatalogSummary(
            active_source_count=len(active_source_ids),
            resource_count=resource_count,
            relationship_count=sum(
                len(snapshot.relationships) for snapshot in snapshots
            ),
            latest_successful_sync_completed_at=(
                max(completion_times) if completion_times else None
            ),
            is_empty=resource_count == 0,
        )

    async def load_resource(
        self, agent_id: str, resource_id: str
    ) -> CatalogResource | None:
        for snapshot in await self._snapshots(agent_id):
            for resource in snapshot.resources:
                if resource.id == resource_id:
                    return resource
        return None

    async def load_revision(
        self, agent_id: str, resource_id: str, revision: str
    ) -> CatalogResourceRevision | None:
        for snapshot in await self._snapshots(agent_id):
            for item in snapshot.revisions:
                if item.resource_id == resource_id and item.revision == revision:
                    return item
        return None

    async def list_resources(
        self, agent_id: str, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        resources = [
            resource
            for snapshot in await self._snapshots(
                agent_id,
                () if source_id is None else (source_id,),
            )
            if source_id is None or snapshot.sync.source_id == source_id
            for resource in snapshot.resources
        ]
        return tuple(sorted(resources, key=lambda item: (item.name, item.id)))

    async def load_facets(
        self,
        agent_id: str,
        resource_id: str,
        revision: str | None = None,
    ) -> tuple[CatalogFacet, ...]:
        for snapshot in await self._snapshots(agent_id):
            resource = next(
                (item for item in snapshot.resources if item.id == resource_id), None
            )
            if resource is None or (
                revision is not None and resource.current_revision != revision
            ):
                continue
            return tuple(
                facet for facet in snapshot.facets if facet.resource_id == resource_id
            )
        return ()

    async def load_incident_relationships(
        self,
        agent_id: str,
        resource_id: str,
        *,
        relationship_kinds: tuple[RelationshipKind, ...] = (),
        limit: int = 50,
    ) -> tuple[CatalogRelationship, ...]:
        relationships = [
            item
            for item in await self._relationships(agent_id)
            if resource_id in (item.from_resource_id, item.to_resource_id)
            and (not relationship_kinds or item.kind in relationship_kinds)
        ]
        return tuple(sorted(relationships, key=lambda item: item.id)[:limit])

    async def list_semantic_annotations(
        self,
        agent_id: str,
    ) -> tuple[SemanticAnnotation, ...]:
        """Return one bounded deterministic agent-isolated semantic collection."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")

        def read() -> tuple[SemanticAnnotation, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT data FROM semantic_annotations
                       WHERE agent_id = ? ORDER BY id""",
                    (agent_id,),
                ).fetchall()
            if len(rows) > SEMANTIC_MAX_ANNOTATIONS:
                raise RuntimeError(
                    "stored semantic annotation collection exceeds bound"
                )
            return tuple(_expect(_loads(data), SemanticAnnotation) for (data,) in rows)

        return await asyncio.to_thread(read)

    async def load_semantic_annotation(
        self,
        agent_id: str,
        annotation_id: str,
    ) -> SemanticAnnotation | None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("annotation_id must be a non-empty string")

        def read() -> SemanticAnnotation | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM semantic_annotations
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, annotation_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), SemanticAnnotation)

        return await asyncio.to_thread(read)

    async def preflight_semantic_save(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        expected_sha256: str | None,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one exact semantic create or replacement."""

        _validate_semantic_owner(agent_id, annotation)

        def read() -> FrozenJsonObject:
            with _connect(self.path) as connection:
                return _semantic_save_fingerprint(
                    connection,
                    agent_id,
                    annotation,
                    expected_sha256,
                )

        return await asyncio.to_thread(read)

    async def save_semantic_annotation(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        *,
        expected_sha256: str | None = None,
    ) -> bool:
        """Atomically create, digest-replace, or digest-supersede one annotation."""

        _validate_semantic_owner(agent_id, annotation)
        gate = _CatalogCommitGate()

        def write() -> bool | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                _semantic_save_fingerprint(
                    connection,
                    agent_id,
                    annotation,
                    expected_sha256,
                )
                row = connection.execute(
                    """SELECT data FROM semantic_annotations
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, annotation.id),
                ).fetchone()
                changed = (
                    row is None
                    or _expect(_loads(row[0]), SemanticAnnotation) != annotation
                )
                connection.execute(
                    """INSERT INTO semantic_annotations(agent_id, id, data)
                       VALUES (?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (agent_id, annotation.id, _dumps(annotation)),
                )
                connection.commit()
                return changed
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        changed = worker.result()
        if cancelled_before_start:
            if changed is not None:
                raise AssertionError("cancelled semantic transaction committed")
            raise asyncio.CancelledError
        if changed is None:
            raise AssertionError("semantic transaction stopped without cancellation")
        return changed

    async def preflight_semantic_delete(
        self,
        agent_id: str,
        annotation_id: str,
        expected_sha256: str,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one digest-protected semantic deletion."""

        def read() -> FrozenJsonObject:
            with _connect(self.path) as connection:
                return _semantic_delete_fingerprint(
                    connection,
                    agent_id,
                    annotation_id,
                    expected_sha256,
                )

        return await asyncio.to_thread(read)

    async def delete_semantic_annotation(
        self,
        agent_id: str,
        annotation_id: str,
        *,
        expected_sha256: str,
    ) -> bool:
        """Atomically delete one annotation only when its rendered digest matches."""

        gate = _CatalogCommitGate()

        def write() -> bool | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                _semantic_delete_fingerprint(
                    connection,
                    agent_id,
                    annotation_id,
                    expected_sha256,
                )
                cursor = connection.execute(
                    """DELETE FROM semantic_annotations
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, annotation_id),
                )
                if cursor.rowcount != 1:
                    raise SemanticNotFoundError(annotation_id)
                connection.commit()
                return True
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        deleted = worker.result()
        if cancelled_before_start:
            if deleted is not None:
                raise AssertionError("cancelled semantic transaction committed")
            raise asyncio.CancelledError
        if deleted is None:
            raise AssertionError("semantic transaction stopped without cancellation")
        return deleted

    async def recent_completed_runs(
        self,
        agent_id: str,
        *,
        limit: int,
    ) -> LearningReviewRunTail:
        """Return a bounded chronological tail of terminal completed runs."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or limit > LEARNING_REVIEW_MAX_STAMPS
        ):
            raise ValueError("completed-run review limit is invalid")

        def read() -> LearningReviewRunTail:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, turn_index, input, result
                       FROM runs
                       WHERE agent_id = ? AND result IS NOT NULL
                       ORDER BY rowid DESC
                       LIMIT ?""",
                    (agent_id, limit),
                ).fetchall()
                records: list[ConversationRun] = []
                unreadable_run_count = 0
                for run_id, turn_index, input_data, result_data in rows:
                    try:
                        result = _expect(_loads(result_data), LoopExit)
                        if result.kind is not LoopExitKind.COMPLETED:
                            continue
                        message_rows = connection.execute(
                            """SELECT data FROM messages
                               WHERE run_id = ? ORDER BY position""",
                            (run_id,),
                        ).fetchall()
                        record = ConversationRun(
                            turn_index=int(turn_index),
                            transcript=Transcript(
                                run=_expect(_loads(input_data), RunInput),
                                messages=tuple(
                                    _expect(_loads(data), CanonicalMessage)
                                    for (data,) in message_rows
                                ),
                            ),
                            result=result,
                        )
                    except (KeyError, TypeError, ValueError):
                        unreadable_run_count += 1
                        continue
                    records.append(record)
            records.reverse()
            return LearningReviewRunTail(
                tuple(records),
                unreadable_run_count=unreadable_run_count,
            )

        return await asyncio.to_thread(read)

    async def list_learning_candidates(
        self,
        agent_id: str,
    ) -> tuple[LearningCandidate, ...]:
        """Return one bounded deterministic agent-isolated candidate inbox."""

        _validate_learning_agent_id(agent_id)

        def read() -> tuple[LearningCandidate, ...]:
            with _connect(self.path) as connection:
                return _learning_candidate_rows(connection, agent_id)

        return await asyncio.to_thread(read)

    async def load_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
    ) -> LearningCandidate | None:
        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_id(candidate_id)

        def read() -> LearningCandidate | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM learning_candidates
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, candidate_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), LearningCandidate)

        return await asyncio.to_thread(read)

    async def learning_candidate_review_stamps(
        self,
        agent_id: str,
    ) -> tuple[LearningCandidateReviewStamp, ...]:
        _validate_learning_agent_id(agent_id)

        def read() -> tuple[LearningCandidateReviewStamp, ...]:
            with _connect(self.path) as connection:
                return _learning_review_stamps(connection, agent_id)

        return await asyncio.to_thread(read)

    async def save_learning_candidate_review(
        self,
        agent_id: str,
        *,
        stamps: tuple[LearningCandidateReviewStamp, ...],
        candidates: tuple[LearningCandidate, ...],
    ) -> tuple[LearningCandidate, ...]:
        """Atomically record one completed review and its inactive candidates."""

        _validate_learning_agent_id(agent_id)
        stamps = tuple(stamps)
        candidates = tuple(candidates)
        if (
            not stamps
            or len(stamps) > LEARNING_REVIEW_MAX_STAMPS
            or any(
                not isinstance(item, LearningCandidateReviewStamp) for item in stamps
            )
            or len(set(stamps)) != len(stamps)
        ):
            raise LearningCandidateError("review stamps exceed their unique bound")
        if len(candidates) > LEARNING_REVIEW_MAX_PROPOSALS or any(
            not isinstance(item, LearningCandidate) for item in candidates
        ):
            raise LearningCandidateError("review candidates exceed their bound")
        for candidate in candidates:
            _validate_learning_candidate_owner(agent_id, candidate)
            if candidate.status is not LearningCandidateStatus.AWAITING_REVIEW:
                raise LearningCandidateError(
                    "new review candidates must be awaiting review"
                )

        def write(connection: sqlite3.Connection) -> tuple[LearningCandidate, ...]:
            current_stamps = _learning_review_stamps(connection, agent_id)
            current_stamp_set = set(current_stamps)
            if all(stamp in current_stamp_set for stamp in stamps):
                review_fingerprints = {item.review_fingerprint for item in candidates}
                return tuple(
                    item
                    for item in _learning_candidate_rows(connection, agent_id)
                    if item.review_fingerprint in review_fingerprints
                )
            merged_stamps = (
                *current_stamps,
                *(stamp for stamp in stamps if stamp not in current_stamp_set),
            )
            if len(merged_stamps) > LEARNING_REVIEW_MAX_STAMPS:
                raise LearningCandidateError(
                    "learning review stamp capacity is exhausted"
                )
            current = _learning_candidate_rows(connection, agent_id)
            by_id = {item.id: item for item in current}
            identities = {item.candidate_identity_sha256 for item in current}
            inserted: list[LearningCandidate] = []
            for candidate in candidates:
                existing = by_id.get(candidate.id)
                if existing is not None:
                    if (
                        existing.candidate_identity_sha256
                        == candidate.candidate_identity_sha256
                    ):
                        continue
                    raise LearningCandidateError(
                        "learning candidate record identity collision"
                    )
                if candidate.candidate_identity_sha256 in identities:
                    continue
                if len(current) + len(inserted) >= LEARNING_CANDIDATE_MAX_RECORDS:
                    raise LearningCandidateError(
                        "learning candidate capacity is exhausted"
                    )
                connection.execute(
                    """INSERT INTO learning_candidates(agent_id, id, data)
                       VALUES (?, ?, ?)""",
                    (agent_id, candidate.id, _dumps(candidate)),
                )
                by_id[candidate.id] = candidate
                identities.add(candidate.candidate_identity_sha256)
                inserted.append(candidate)
            connection.execute(
                """INSERT INTO metadata(key, data) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                (
                    _learning_review_stamps_key(agent_id),
                    _dumps(tuple(merged_stamps)),
                ),
            )
            return tuple(inserted)

        return await _run_candidate_transaction(self.path, write)

    async def edit_learning_candidate(
        self,
        agent_id: str,
        candidate: LearningCandidate,
        *,
        expected_fingerprint: str,
    ) -> LearningCandidate:
        """Atomically replace only one awaiting candidate's bounded content."""

        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_owner(agent_id, candidate)
        _validate_learning_digest(expected_fingerprint, "expected_fingerprint")

        def write(connection: sqlite3.Connection) -> LearningCandidate:
            current = _load_learning_candidate_required(
                connection,
                agent_id,
                candidate.id,
            )
            _require_candidate_transition(
                current,
                expected_fingerprint=expected_fingerprint,
            )
            if candidate.status is not LearningCandidateStatus.AWAITING_REVIEW:
                raise LearningCandidateError(
                    "edited candidate must remain awaiting review"
                )
            unchanged = (
                "id",
                "agent_id",
                "target",
                "source_ids",
                "reviewed_runs",
                "supporting_run_ids",
                "review_fingerprint",
                "artifact_state_sha256",
                "catalog_revisions",
                "status",
                "created_at",
                "rejection_reason",
            )
            if any(
                getattr(current, field_name) != getattr(candidate, field_name)
                for field_name in unchanged
            ):
                raise LearningCandidateError(
                    "candidate edit may change only bounded proposed content"
                )
            if candidate.updated_at < current.updated_at:
                raise LearningCandidateError(
                    "candidate edit timestamp cannot move backwards"
                )
            if any(
                item.id != candidate.id
                and item.candidate_identity_sha256
                == candidate.candidate_identity_sha256
                for item in _learning_candidate_rows(connection, agent_id)
            ):
                raise LearningCandidateError(
                    "candidate edit duplicates an existing normalized identity"
                )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (_dumps(candidate), agent_id, candidate.id),
            )
            return candidate

        return await _run_candidate_transaction(self.path, write)

    async def reject_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
        *,
        expected_fingerprint: str,
        reason: LearningCandidateRejectionReason,
        rejected_at: datetime,
    ) -> LearningCandidate:
        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_id(candidate_id)
        _validate_learning_digest(expected_fingerprint, "expected_fingerprint")
        if not isinstance(reason, LearningCandidateRejectionReason):
            raise TypeError("candidate rejection reason is invalid")

        def write(connection: sqlite3.Connection) -> LearningCandidate:
            current = _load_learning_candidate_required(
                connection,
                agent_id,
                candidate_id,
            )
            _require_candidate_transition(
                current,
                expected_fingerprint=expected_fingerprint,
            )
            rejected = LearningCandidate(
                **{
                    **{
                        field.name: getattr(current, field.name)
                        for field in fields(current)
                    },
                    "status": LearningCandidateStatus.REJECTED,
                    "updated_at": rejected_at,
                    "rejection_reason": reason,
                }
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (_dumps(rejected), agent_id, candidate_id),
            )
            return rejected

        return await _run_candidate_transaction(self.path, write)

    async def accept_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
        *,
        expected_fingerprint: str,
        accepted_at: datetime,
    ) -> LearningCandidate:
        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_id(candidate_id)
        _validate_learning_digest(expected_fingerprint, "expected_fingerprint")

        def write(connection: sqlite3.Connection) -> LearningCandidate:
            current = _load_learning_candidate_required(
                connection,
                agent_id,
                candidate_id,
            )
            _require_candidate_transition(
                current,
                expected_fingerprint=expected_fingerprint,
            )
            accepted = LearningCandidate(
                **{
                    **{
                        field.name: getattr(current, field.name)
                        for field in fields(current)
                    },
                    "status": LearningCandidateStatus.ACCEPTED,
                    "updated_at": accepted_at,
                }
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (_dumps(accepted), agent_id, candidate_id),
            )
            return accepted

        return await _run_candidate_transaction(self.path, write)

    async def clear_rejected_learning_candidates(self, agent_id: str) -> int:
        """Delete only explicit rejection tombstones and reset their review stamps."""

        _validate_learning_agent_id(agent_id)

        def write(connection: sqlite3.Connection) -> int:
            rejected_ids = tuple(
                item.id
                for item in _learning_candidate_rows(connection, agent_id)
                if item.status is LearningCandidateStatus.REJECTED
            )
            if rejected_ids:
                connection.executemany(
                    """DELETE FROM learning_candidates
                       WHERE agent_id = ? AND id = ?""",
                    ((agent_id, candidate_id) for candidate_id in rejected_ids),
                )
                connection.execute(
                    "DELETE FROM metadata WHERE key = ?",
                    (_learning_review_stamps_key(agent_id),),
                )
            return len(rejected_ids)

        return await _run_candidate_transaction(self.path, write)

    async def start(self, run: RunInput) -> Transcript:
        if run.conversation_id is None:
            raise ValueError("run conversation_id must be resolved before persistence")

        def write() -> Transcript:
            with _connect(self.path) as connection:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    row = connection.execute(
                        """SELECT COALESCE(MAX(turn_index), -1) + 1
                           FROM runs
                           WHERE agent_id = ? AND conversation_id = ?""",
                        (run.agent_id, run.conversation_id),
                    ).fetchone()
                    connection.execute(
                        """INSERT INTO runs(
                               id, agent_id, conversation_id, turn_index, input
                           ) VALUES (?, ?, ?, ?, ?)""",
                        (
                            run.id,
                            run.agent_id,
                            run.conversation_id,
                            int(row[0]),
                            _dumps(run),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"run already exists: {run.id}") from error
            return Transcript(run=run)

        return await asyncio.to_thread(write)

    async def append(self, run_id: str, message: CanonicalMessage) -> None:
        def write() -> None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if (
                    connection.execute(
                        "SELECT 1 FROM runs WHERE id = ?", (run_id,)
                    ).fetchone()
                    is None
                ):
                    raise KeyError(f"unknown run: {run_id}")
                connection.execute(
                    "INSERT INTO messages(run_id, position, data) VALUES (?, ?, ?)",
                    (run_id, int(row[0]), _dumps(message)),
                )

        await asyncio.to_thread(write)

    async def finish(self, result: LoopExit) -> None:
        def write() -> None:
            with _connect(self.path) as connection:
                cursor = connection.execute(
                    """UPDATE runs SET result = ?
                       WHERE id = ? AND conversation_id = ?""",
                    (_dumps(result), result.run_id, result.conversation_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown run: {result.run_id}")

        await asyncio.to_thread(write)

    async def load(self, run_id: str) -> Transcript:
        def read() -> Transcript:
            with _connect(self.path) as connection:
                run_row = connection.execute(
                    "SELECT input FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                if run_row is None:
                    raise KeyError(f"unknown run: {run_id}")
                rows = connection.execute(
                    "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                    (run_id,),
                ).fetchall()
            return Transcript(
                run=_expect(_loads(run_row[0]), RunInput),
                messages=tuple(
                    _expect(_loads(row[0]), CanonicalMessage) for row in rows
                ),
            )

        return await asyncio.to_thread(read)

    async def result(self, run_id: str) -> LoopExit | None:
        def read() -> LoopExit | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT result FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            return None if row[0] is None else _expect(_loads(row[0]), LoopExit)

        return await asyncio.to_thread(read)

    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Derive current reachable refs from persisted agent-scoped tool messages."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("run_id must be non-empty text or None")
        if conversation_id is not None and (
            not isinstance(conversation_id, str) or not conversation_id
        ):
            raise ValueError("conversation_id must be non-empty text or None")

        def read() -> tuple[ArtifactRef, ...]:
            clauses = ["r.agent_id = ?"]
            values: list[object] = [agent_id]
            if run_id is not None:
                clauses.append("r.id = ?")
                values.append(run_id)
            if conversation_id is not None:
                clauses.append("r.conversation_id = ?")
                values.append(conversation_id)
            where = " AND ".join(clauses)
            with _connect_read_only(self.path) as connection:
                rows = connection.execute(
                    f"""SELECT r.id, r.conversation_id, m.data
                        FROM runs AS r
                        JOIN messages AS m ON m.run_id = r.id
                        WHERE {where}
                        ORDER BY r.id, m.position""",
                    tuple(values),
                ).fetchall()
            refs: dict[str, ArtifactRef] = {}
            for stored_run_id, stored_conversation_id, data in rows:
                message = _expect(_loads(data), CanonicalMessage)
                if message.role is not MessageRole.TOOL:
                    continue
                for block in message.content:
                    if not isinstance(block, ToolResultBlock) or block.is_error:
                        continue
                    value = block.output.get("artifact")
                    if not isinstance(value, Mapping):
                        continue
                    try:
                        ref = artifact_ref_from_mapping(value)
                    except (TypeError, ValueError) as error:
                        raise RuntimeError(
                            "stored artifact reference is invalid"
                        ) from error
                    if (
                        ref.run_id != stored_run_id
                        or ref.conversation_id != stored_conversation_id
                        or ref.call_id != block.call_id
                    ):
                        raise RuntimeError(
                            "stored artifact reference identity does not match its run"
                        )
                    existing = refs.get(ref.artifact_id)
                    if existing is not None and existing != ref:
                        raise RuntimeError("stored artifact identity is ambiguous")
                    refs[ref.artifact_id] = ref
            return tuple(
                sorted(
                    refs.values(), key=lambda item: (item.created_at, item.artifact_id)
                )
            )

        return await asyncio.to_thread(read)

    async def conversation_runs(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        """Return every run in one agent-scoped conversation in turn order."""

        def read() -> tuple[ConversationRun, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, turn_index, input, result
                       FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       ORDER BY turn_index""",
                    (agent_id, conversation_id),
                ).fetchall()
                records: list[ConversationRun] = []
                for run_id, turn_index, input_data, result_data in rows:
                    message_rows = connection.execute(
                        """SELECT data FROM messages
                           WHERE run_id = ? ORDER BY position""",
                        (run_id,),
                    ).fetchall()
                    transcript = Transcript(
                        run=_expect(_loads(input_data), RunInput),
                        messages=tuple(
                            _expect(_loads(message[0]), CanonicalMessage)
                            for message in message_rows
                        ),
                    )
                    result = (
                        None
                        if result_data is None
                        else _expect(_loads(result_data), LoopExit)
                    )
                    records.append(
                        ConversationRun(
                            turn_index=int(turn_index),
                            transcript=transcript,
                            result=result,
                        )
                    )
            return tuple(records)

        return await asyncio.to_thread(read)

    async def conversation_exists(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> bool:
        """Return one agent-scoped existence fact without loading transcript data."""

        def read() -> bool:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT 1 FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       LIMIT 1""",
                    (agent_id, conversation_id),
                ).fetchone()
            return row is not None

        return await asyncio.to_thread(read)

    async def clear_conversations(self, agent_id: str) -> int:
        """Delete transcripts and candidate records derived from them."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        gate = _CatalogCommitGate()

        def write() -> int | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                row = connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE agent_id = ?",
                    (agent_id,),
                ).fetchone()
                run_count = int(row[0])
                connection.execute(
                    """DELETE FROM messages
                       WHERE run_id IN (
                           SELECT id FROM runs WHERE agent_id = ?
                       )""",
                    (agent_id,),
                )
                connection.execute(
                    "DELETE FROM runs WHERE agent_id = ?",
                    (agent_id,),
                )
                connection.execute(
                    "DELETE FROM learning_candidates WHERE agent_id = ?",
                    (agent_id,),
                )
                connection.execute(
                    "DELETE FROM metadata WHERE key = ?",
                    (_learning_review_stamps_key(agent_id),),
                )
                connection.commit()
                return run_count
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        cleared = worker.result()
        if cancelled_before_start:
            if cleared is not None:
                raise AssertionError(
                    "cancelled conversation-clear transaction committed"
                )
            raise asyncio.CancelledError
        if cleared is None:
            raise AssertionError(
                "conversation-clear transaction stopped without cancellation"
            )
        return cleared

    async def conversation_source_id(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> str | None:
        """Return the sticky source captured by the first run in a conversation."""

        def read() -> str | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT input FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       ORDER BY turn_index
                       LIMIT 1""",
                    (agent_id, conversation_id),
                ).fetchone()
            if row is None:
                return None
            run = _expect(_loads(row[0]), RunInput)
            return run.conversation_source_id or run.source_id

        return await asyncio.to_thread(read)

    async def completed_conversation_tail(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        limit: int = 8,
    ) -> tuple[bool, tuple[ConversationRun, ...], bool]:
        """Load one bounded newest-first candidate snapshot and existence fact."""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("conversation tail limit must be positive")

        def read() -> tuple[bool, tuple[ConversationRun, ...], bool]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, turn_index, input, result
                       FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       ORDER BY turn_index DESC""",
                    (agent_id, conversation_id),
                )
                exists = False
                older_completed_exists = False
                records: list[ConversationRun] = []
                for run_id, turn_index, input_data, result_data in rows:
                    exists = True
                    if result_data is None:
                        continue
                    result = _expect(_loads(result_data), LoopExit)
                    if result.kind is not LoopExitKind.COMPLETED:
                        continue
                    if len(records) >= limit:
                        older_completed_exists = True
                        break
                    message_rows = connection.execute(
                        """SELECT data FROM messages
                           WHERE run_id = ? ORDER BY position""",
                        (run_id,),
                    ).fetchall()
                    records.append(
                        ConversationRun(
                            turn_index=int(turn_index),
                            transcript=Transcript(
                                run=_expect(_loads(input_data), RunInput),
                                messages=tuple(
                                    _expect(_loads(message[0]), CanonicalMessage)
                                    for message in message_rows
                                ),
                            ),
                            result=result,
                        )
                    )
            records.reverse()
            return exists, tuple(records), older_completed_exists

        return await asyncio.to_thread(read)

    async def _snapshots(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[SourceCatalogSnapshot, ...]:
        for attempt in range(2):
            refs = await self.list_current_snapshot_refs(agent_id, source_ids)
            snapshots: list[SourceCatalogSnapshot] = []
            generation_changed = False
            for ref in refs:
                snapshot = await self.load_current_snapshot(ref)
                if snapshot is None:
                    generation_changed = True
                    break
                snapshots.append(snapshot)
            if (
                not generation_changed
                and refs
                == await self.list_current_snapshot_refs(
                    agent_id,
                    source_ids,
                )
            ):
                return tuple(snapshots)
            if attempt == 1:
                break
        raise CatalogStoreError("catalog snapshot generation changed repeatedly")

    async def _relationships(self, agent_id: str) -> tuple[CatalogRelationship, ...]:
        return tuple(
            relationship
            for snapshot in await self._snapshots(agent_id)
            for relationship in snapshot.relationships
        )

    def _publish_decoded_catalog_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
    ) -> None:
        sync = snapshot.sync
        key = (sync.agent_id, sync.source_id, sync.id)
        existing = self._decoded_catalog_snapshots.get(key)
        for cached_key in tuple(self._decoded_catalog_snapshots):
            if cached_key[:2] == key[:2] and cached_key != key:
                del self._decoded_catalog_snapshots[cached_key]
        if existing is None:
            self._decoded_catalog_snapshots[key] = snapshot

    def _evict_decoded_catalog_source(self, agent_id: str, source_id: str) -> None:
        for key in tuple(self._decoded_catalog_snapshots):
            if key[:2] == (agent_id, source_id):
                del self._decoded_catalog_snapshots[key]


def _catalog_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > 512:
        raise ValueError(f"{field_name} exceeds 512 characters")


def _catalog_snapshot_source_ids(source_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(source_ids, tuple):
        raise TypeError("catalog snapshot source_ids must be a tuple")
    for source_id in source_ids:
        _catalog_identifier(source_id, "catalog snapshot source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("catalog snapshot source_ids cannot contain duplicates")
    return source_ids


def _current_snapshot_sync_id(
    path: Path,
    agent_id: str,
    source_id: str,
) -> str | None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT sync_id FROM snapshots WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchone()
    return None if row is None else str(row[0])


def _current_snapshot_row(
    path: Path,
    agent_id: str,
    source_id: str,
) -> tuple[str, str] | None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT sync_id, data FROM snapshots "
            "WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1])


def _decode_catalog_snapshot(value: str) -> SourceCatalogSnapshot:
    return _expect(_loads(value), SourceCatalogSnapshot)


def _validate_semantic_owner(
    agent_id: str,
    annotation: SemanticAnnotation,
) -> None:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(annotation, SemanticAnnotation):
        raise TypeError("annotation must be SemanticAnnotation")
    if annotation.agent_id != agent_id:
        raise ValueError("semantic annotation belongs to another agent")


def _semantic_rows(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[tuple[str, SemanticAnnotation, str], ...]:
    rows = connection.execute(
        """SELECT id, data FROM semantic_annotations
           WHERE agent_id = ? ORDER BY id""",
        (agent_id,),
    ).fetchall()
    if len(rows) > SEMANTIC_MAX_ANNOTATIONS:
        raise RuntimeError("stored semantic annotation collection exceeds bound")
    return tuple(
        (annotation_id, _expect(_loads(data), SemanticAnnotation), data)
        for annotation_id, data in rows
    )


def _semantic_state_sha256(
    rows: tuple[tuple[str, SemanticAnnotation, str], ...],
) -> str:
    payload = "\n".join(f"{annotation_id}:{data}" for annotation_id, _, data in rows)
    return sha256(payload.encode("utf-8")).hexdigest()


def _semantic_save_fingerprint(
    connection: sqlite3.Connection,
    agent_id: str,
    annotation: SemanticAnnotation,
    expected_sha256: str | None,
) -> FrozenJsonObject:
    _validate_semantic_owner(agent_id, annotation)
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise SemanticDigestMismatchError(
            "semantic expected_sha256 must be lowercase SHA-256"
        )
    rows = _semantic_rows(connection, agent_id)
    by_id = {annotation_id: item for annotation_id, item, _ in rows}
    current = by_id.get(annotation.id)
    expected_target: SemanticAnnotation | None
    expected_target_id: str
    if current is not None:
        expected_target = current
        expected_target_id = current.id
        if annotation.created_at != current.created_at:
            raise SemanticValidationError(
                "semantic replacement must preserve created_at"
            )
    elif annotation.supersedes_id is not None:
        expected_target = by_id.get(annotation.supersedes_id)
        expected_target_id = annotation.supersedes_id
        if expected_target is None:
            raise SemanticNotFoundError(annotation.supersedes_id)
        if (
            expected_target.subject != annotation.subject
            or expected_target.kind is not annotation.kind
        ):
            raise SemanticValidationError(
                "a semantic annotation may supersede only the same subject and kind"
            )
        if annotation.created_at < expected_target.created_at:
            raise SemanticValidationError(
                "semantic supersession cannot predate its target"
            )
    else:
        expected_target = None
        expected_target_id = annotation.id

    if expected_target is None:
        if expected_sha256 is not None:
            raise SemanticDigestMismatchError(
                "new semantic annotations cannot include expected_sha256"
            )
        if len(rows) >= SEMANTIC_MAX_ANNOTATIONS:
            raise SemanticValidationError(
                f"semantic annotation collection is limited to "
                f"{SEMANTIC_MAX_ANNOTATIONS}"
            )
        current_sha256 = sha256(b"").hexdigest()
    else:
        current_sha256 = semantic_annotation_sha256(expected_target)
        if expected_sha256 is None:
            raise SemanticDigestMismatchError(
                "semantic replacement or supersession requires expected_sha256"
            )
        if expected_sha256 != current_sha256:
            raise SemanticDigestMismatchError(
                "semantic annotation changed; load it again with semantic_view"
            )

    return FrozenJsonObject.from_mapping(
        {
            "id": annotation.id,
            "exists": current is not None,
            "expected_target_id": expected_target_id,
            "current_sha256": current_sha256,
            "candidate_sha256": semantic_annotation_sha256(annotation),
            "state_sha256": _semantic_state_sha256(rows),
        }
    )


def _semantic_delete_fingerprint(
    connection: sqlite3.Connection,
    agent_id: str,
    annotation_id: str,
    expected_sha256: str,
) -> FrozenJsonObject:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(annotation_id, str) or not annotation_id:
        raise ValueError("annotation_id must be a non-empty string")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise SemanticDigestMismatchError(
            "semantic deletion requires lowercase expected_sha256"
        )
    rows = _semantic_rows(connection, agent_id)
    current = next(
        (item for item_id, item, _ in rows if item_id == annotation_id),
        None,
    )
    if current is None:
        raise SemanticNotFoundError(annotation_id)
    current_sha256 = semantic_annotation_sha256(current)
    if current_sha256 != expected_sha256:
        raise SemanticDigestMismatchError(
            "semantic annotation changed; load it again with semantic_view"
        )
    return FrozenJsonObject.from_mapping(
        {
            "id": annotation_id,
            "current_sha256": current_sha256,
            "state_sha256": _semantic_state_sha256(rows),
        }
    )


def _validate_learning_agent_id(agent_id: str) -> None:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("learning candidate agent_id must be non-empty text")


def _validate_learning_candidate_id(candidate_id: str) -> None:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("learning candidate id must be non-empty text")


def _validate_learning_digest(value: str, field_name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LearningCandidateError(f"{field_name} must be lowercase SHA-256")


def _validate_learning_candidate_owner(
    agent_id: str,
    candidate: LearningCandidate,
) -> None:
    if not isinstance(candidate, LearningCandidate):
        raise TypeError("candidate must be LearningCandidate")
    if candidate.agent_id != agent_id:
        raise LearningCandidateError("learning candidate belongs to another agent")


def _learning_candidate_rows(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[LearningCandidate, ...]:
    rows = connection.execute(
        """SELECT data FROM learning_candidates
           WHERE agent_id = ? ORDER BY id""",
        (agent_id,),
    ).fetchall()
    if len(rows) > LEARNING_CANDIDATE_MAX_RECORDS:
        raise RuntimeError("stored learning candidate collection exceeds bound")
    values = tuple(_expect(_loads(data), LearningCandidate) for (data,) in rows)
    if any(item.agent_id != agent_id for item in values):
        raise RuntimeError("stored learning candidate owner is invalid")
    return values


def _learning_review_stamps(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[LearningCandidateReviewStamp, ...]:
    row = connection.execute(
        "SELECT data FROM metadata WHERE key = ?",
        (_learning_review_stamps_key(agent_id),),
    ).fetchone()
    if row is None:
        return ()
    value = _loads(row[0])
    if not isinstance(value, tuple) or any(
        not isinstance(item, LearningCandidateReviewStamp) for item in value
    ):
        raise RuntimeError("stored learning review stamps are invalid")
    if len(value) > LEARNING_REVIEW_MAX_STAMPS or len(set(value)) != len(value):
        raise RuntimeError("stored learning review stamps exceed their bound")
    return value


def _load_learning_candidate_required(
    connection: sqlite3.Connection,
    agent_id: str,
    candidate_id: str,
) -> LearningCandidate:
    row = connection.execute(
        """SELECT data FROM learning_candidates
           WHERE agent_id = ? AND id = ?""",
        (agent_id, candidate_id),
    ).fetchone()
    if row is None:
        raise LearningCandidateNotFoundError(candidate_id)
    candidate = _expect(_loads(row[0]), LearningCandidate)
    _validate_learning_candidate_owner(agent_id, candidate)
    return candidate


def _require_candidate_transition(
    current: LearningCandidate,
    *,
    expected_fingerprint: str,
) -> None:
    if current.status is not LearningCandidateStatus.AWAITING_REVIEW:
        raise LearningCandidateError(
            f"candidate is not awaiting review: {current.status.value}"
        )
    if current.candidate_fingerprint != expected_fingerprint:
        raise LearningCandidateError(
            "learning candidate changed; load it again before mutation"
        )


async def _run_candidate_transaction(
    path: Path,
    callback: Callable[[sqlite3.Connection], _T],
) -> _T:
    gate = _CatalogCommitGate()

    def write() -> _T | None:
        connection = _connect(path)
        try:
            if not gate.start(connection):
                return None
            result = callback(connection)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    worker = asyncio.create_task(asyncio.to_thread(write))
    cancelled_before_start = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled_before_start = (
                gate.cancel_before_start() or cancelled_before_start
            )
    result = worker.result()
    if cancelled_before_start:
        if result is not None:
            raise AssertionError("cancelled learning candidate transaction committed")
        raise asyncio.CancelledError
    if result is None:
        raise AssertionError(
            "learning candidate transaction stopped without cancellation"
        )
    return result


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, 0o600)
        _admit_existing_state(path)
        return
    with _connect(path) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            CREATE TABLE IF NOT EXISTS database_write_receipts (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id),
                UNIQUE(agent_id, run_id, call_id)
            );
            CREATE TABLE IF NOT EXISTS syncs (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                agent_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                sync_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, source_id)
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                input TEXT NOT NULL,
                result TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(run_id, position)
            );
            CREATE TABLE IF NOT EXISTS semantic_annotations (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            CREATE TABLE IF NOT EXISTS learning_candidates (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            """)
        connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS runs_conversation_turn
               ON runs(agent_id, conversation_id, turn_index)""")
        connection.execute(f"PRAGMA user_version = {STATE_FORMAT_VERSION}")
        _require_schema(connection, _CURRENT_STATE_TABLE_DEFINITIONS)
        _require_current_source_records(connection)
        _require_healthy_database(connection)
    os.chmod(path, 0o600)


def _state_migration_path(source_format: int) -> tuple[_StateMigration, ...]:
    if not isinstance(source_format, int) or source_format <= 0:
        raise ValueError("migration source format must be positive")
    path: list[_StateMigration] = []
    visited: set[int] = set()
    current_format = source_format
    while current_format != STATE_FORMAT_VERSION:
        if current_format in visited or current_format > STATE_FORMAT_VERSION:
            raise ValueError("state migration ledger is invalid")
        visited.add(current_format)
        step = _STATE_MIGRATIONS.get(current_format)
        if (
            step is None
            or step.target_format <= current_format
            or step.target_format > STATE_FORMAT_VERSION
        ):
            raise ValueError("no complete state migration path is available")
        path.append(step)
        current_format = step.target_format
    return tuple(path)


def _unversioned_state_format(connection: sqlite3.Connection) -> int | None:
    matches = tuple(
        state_format
        for state_format, schema in _UNVERSIONED_STATE_SCHEMAS
        if _schema_matches(connection, schema)
    )
    if len(matches) != 1:
        return None
    return matches[0]


def _admit_existing_state(path: Path) -> None:
    found_format: int | None = None
    migration_source_format: int | None = None
    try:
        with _connect_read_only(path) as connection:
            found_format = _read_state_format(connection)
            tables = _table_names(connection)
            if found_format > STATE_FORMAT_VERSION:
                raise StateCompatibilityError(
                    StateCompatibilityCode.NEWER_FORMAT,
                    path,
                    (
                        f"This agent home uses state format {found_format}, but this "
                        f"Daita release supports through format {STATE_FORMAT_VERSION}. "
                        "Install the same or a newer Daita release. No state was changed."
                    ),
                    current_format=STATE_FORMAT_VERSION,
                    found_format=found_format,
                )
            if found_format == STATE_FORMAT_VERSION:
                _require_schema(connection, _CURRENT_STATE_TABLE_DEFINITIONS)
                _require_current_source_records(connection)
                _require_healthy_database(connection)
                return

            if found_format == _UNVERSIONED_STATE_FORMAT:
                migration_source_format = _unversioned_state_format(connection)
            elif found_format < STATE_FORMAT_VERSION:
                migration_source_format = found_format

            if migration_source_format is not None:
                try:
                    path_to_current = _state_migration_path(migration_source_format)
                except ValueError:
                    path_to_current = ()
                if path_to_current:
                    _require_schema(connection, path_to_current[0].source_schema)
                    _require_healthy_database(connection)
                else:
                    migration_source_format = None

            if migration_source_format is None:
                if tables & _LEGACY_TABLE_MARKERS:
                    raise StateCompatibilityError(
                        StateCompatibilityCode.LEGACY_FORMAT,
                        path,
                        (
                            "This agent home belongs to the unsupported pre-1.0 Daita "
                            "framework. Keep it intact and use a current-format agent "
                            "home; this release will not overwrite or partially import it."
                        ),
                        current_format=STATE_FORMAT_VERSION,
                        found_format=found_format,
                    )
                raise _damaged_state_error(path, found_format)
    except StateCompatibilityError:
        raise
    except (OSError, sqlite3.Error, ValueError):
        raise _damaged_state_error(path, found_format) from None

    assert found_format is not None
    assert migration_source_format is not None
    _migrate_existing_state(
        path,
        stored_format=found_format,
        source_format=migration_source_format,
    )


def _migrate_existing_state(
    path: Path,
    *,
    stored_format: int,
    source_format: int,
) -> None:
    connection: sqlite3.Connection | None = None
    try:
        path_to_current = _state_migration_path(source_format)
        connection = _connect(path)
        connection.execute("BEGIN IMMEDIATE")
        if _read_state_format(connection) != stored_format:
            raise RuntimeError("state format changed during migration admission")
        for step in path_to_current:
            _require_schema(connection, step.source_schema)
            _require_healthy_database(connection)
            step.apply(connection)
            connection.execute(f"PRAGMA user_version = {step.target_format}")
            _require_schema(connection, step.target_schema)
            if step.validate_target is not None:
                step.validate_target(connection)
            _require_healthy_database(connection)
        if _read_state_format(connection) != STATE_FORMAT_VERSION:
            raise RuntimeError("state migration did not reach the current format")
        connection.commit()
    except BaseException as error:
        if connection is not None:
            connection.rollback()
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise StateCompatibilityError(
            StateCompatibilityCode.MIGRATION_FAILED,
            path,
            (
                "Daita could not complete the automatic local-state upgrade. "
                "The migration transaction was rolled back and the existing state "
                "was preserved. Reinstall the prior working Daita package before "
                "continuing, then report this migration failure."
            ),
            current_format=STATE_FORMAT_VERSION,
            found_format=stored_format,
        ) from None
    finally:
        if connection is not None:
            connection.close()


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE database_write_receipts (
               agent_id TEXT NOT NULL,
               id TEXT NOT NULL,
               run_id TEXT NOT NULL,
               call_id TEXT NOT NULL,
               data TEXT NOT NULL,
               PRIMARY KEY(agent_id, id),
               UNIQUE(agent_id, run_id, call_id)
           )""")


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    rows = tuple(connection.execute("SELECT agent_id, id, data FROM sources"))
    for agent_id, source_id, data in rows:
        registration = _expect(_loads(data), SourceRegistration)
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
        if registration.adapter_id != "postgresql":
            continue
        write_access = registration.configuration.get("write_access", False)
        if not isinstance(write_access, bool):
            raise ValueError("stored PostgreSQL write_access is invalid")
        if "write_access" in registration.configuration:
            continue
        migrated = replace(
            registration,
            configuration={
                **dict(registration.configuration),
                "write_access": False,
            },
        )
        packed = json.loads(data)
        if not isinstance(packed, dict):
            raise ValueError("stored source payload is invalid")
        fields_payload = packed.get("fields")
        if not isinstance(fields_payload, dict):
            raise ValueError("stored source fields are invalid")
        configuration_payload = fields_payload.get("configuration")
        if not isinstance(configuration_payload, dict):
            raise ValueError("stored source configuration is invalid")
        configuration_payload["write_access"] = False
        migrated_data = json.dumps(packed, sort_keys=True, separators=(",", ":"))
        if _expect(_loads(migrated_data), SourceRegistration) != migrated:
            raise ValueError("migrated PostgreSQL source is invalid")
        connection.execute(
            "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
            (migrated_data, agent_id, source_id),
        )


def _require_current_source_records(connection: sqlite3.Connection) -> None:
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, id, data FROM sources"
    ):
        registration = _expect(_loads(data), SourceRegistration)
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
        if registration.adapter_id == "postgresql" and not isinstance(
            registration.configuration.get("write_access"), bool
        ):
            raise ValueError("stored PostgreSQL write_access is invalid")


_STATE_MIGRATIONS = {
    1: _StateMigration(
        target_format=2,
        source_schema=_INITIAL_STATE_TABLE_DEFINITIONS,
        target_schema=_RECEIPT_STATE_TABLE_DEFINITIONS,
        apply=_migrate_v1_to_v2,
    ),
    2: _StateMigration(
        target_format=3,
        source_schema=_RECEIPT_STATE_TABLE_DEFINITIONS,
        target_schema=_CURRENT_STATE_TABLE_DEFINITIONS,
        apply=_migrate_v2_to_v3,
        validate_target=_require_current_source_records,
    ),
}
_UNVERSIONED_STATE_SCHEMAS = (
    (1, _INITIAL_STATE_TABLE_DEFINITIONS),
    (2, _RECEIPT_STATE_TABLE_DEFINITIONS),
)


def _read_state_format(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None or len(row) != 1 or not isinstance(row[0], int) or row[0] < 0:
        raise sqlite3.DatabaseError("state format marker is invalid")
    return int(row[0])


def _table_names(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )


def _schema_matches(
    connection: sqlite3.Connection,
    definitions: Mapping[str, tuple[tuple[object, ...], ...]],
) -> bool:
    try:
        _require_schema(connection, definitions)
    except (sqlite3.Error, ValueError):
        return False
    return True


def _require_schema(
    connection: sqlite3.Connection,
    definitions: Mapping[str, tuple[tuple[object, ...], ...]],
) -> None:
    if _table_names(connection) != set(definitions):
        raise ValueError("state tables do not match the declared format")
    for table, expected in definitions.items():
        actual = tuple(
            (row[1], str(row[2]).upper(), row[3], row[4], row[5])
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise ValueError(f"state table does not match its format: {table}")

    messages_foreign_keys = tuple(
        (row[2], row[3], row[4], row[5], row[6], row[7])
        for row in connection.execute("PRAGMA foreign_key_list(messages)")
    )
    if messages_foreign_keys != _MESSAGES_FOREIGN_KEYS:
        raise ValueError("state message ownership constraint is invalid")
    for table in set(definitions) - {"messages"}:
        if tuple(connection.execute(f"PRAGMA foreign_key_list({table})")):
            raise ValueError(f"state table has unexpected foreign keys: {table}")

    named_indexes = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if named_indexes != {
        name: definition[0] for name, definition in _NAMED_INDEXES.items()
    }:
        raise ValueError("state named indexes do not match the declared format")
    for name, (table, expected_unique, expected_columns) in _NAMED_INDEXES.items():
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


def _require_healthy_database(connection: sqlite3.Connection) -> None:
    quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
    if quick_check != ("ok",):
        raise ValueError("state database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("state database foreign-key check failed")


def _damaged_state_error(
    path: Path,
    found_format: int | None,
) -> StateCompatibilityError:
    return StateCompatibilityError(
        StateCompatibilityCode.DAMAGED,
        path,
        (
            "This agent state database is damaged or does not match its declared "
            "Daita format. No state was changed. Run the matching Daita release's "
            "diagnostics or restore the database through your normal recovery process."
        ),
        current_format=STATE_FORMAT_VERSION,
        found_format=found_format,
    )


def _recover_started_database_write_receipts(
    path: Path,
    clock: Callable[[], datetime],
) -> None:
    try:
        with _connect_read_only(path) as connection:
            rows = tuple(
                connection.execute(
                    "SELECT agent_id, id, data FROM database_write_receipts"
                )
            )
        started = tuple(
            (agent_id, receipt_id, receipt)
            for agent_id, receipt_id, data in rows
            if (receipt := _expect(_loads(data), DatabaseWriteReceipt)).outcome
            is DatabaseWriteOutcome.STARTED
        )
        if not started:
            return
        completed_at = _database_write_aware(clock(), "receipt recovery completed_at")
        with _connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for agent_id, receipt_id, receipt in started:
                recovered = receipt.finish(
                    DatabaseWriteOutcome.OUTCOME_UNKNOWN,
                    completed_at=completed_at,
                    affected_rows=None,
                    normalized_error_code="write_outcome_unknown",
                )
                result = connection.execute(
                    """UPDATE database_write_receipts SET data = ?
                       WHERE agent_id = ? AND id = ? AND data = ?""",
                    (_dumps(recovered), agent_id, receipt_id, _dumps(receipt)),
                )
                if result.rowcount != 1:
                    raise RuntimeError(
                        "database write receipt changed during startup recovery"
                    )
    except RuntimeError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise _damaged_state_error(path, STATE_FORMAT_VERSION) from None


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro",
        timeout=30,
        uri=True,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA query_only = ON")
    return connection


def _commit_catalog_transaction(connection: sqlite3.Connection) -> None:
    connection.commit()


_RECORD_TYPES: dict[str, type[Any]] = {
    record.__name__: record
    for record in (
        AgentIdentity,
        SourceRegistration,
        DatabaseWriteReceipt,
        CatalogFacet,
        RelationshipFieldPair,
        CatalogRelationship,
        CatalogResourceRevision,
        CatalogResource,
        CatalogSync,
        SourceCatalogSnapshot,
        RunInput,
        LoopExit,
        ArtifactResourceBinding,
        ArtifactProvenance,
        ArtifactRef,
        ArtifactDeliveryReceipt,
        TextBlock,
        ToolCall,
        ToolResultBlock,
        CanonicalMessage,
        ModelUsage,
        PricingModifier,
        PricingUsageRange,
        CostComponent,
        CostEstimate,
        SemanticFieldReference,
        SemanticEvidence,
        ResourceRevisionBinding,
        SemanticSubject,
        SemanticAnnotation,
        DocumentCandidateContent,
        SemanticCandidateContent,
        SkillCandidateContent,
        LearningCandidateRunReference,
        LearningCandidateReviewStamp,
        LearningCandidate,
    )
}
_ENUM_TYPES: dict[str, type[Enum]] = {
    enum.__name__: enum
    for enum in (
        CatalogSyncStatus,
        FacetKind,
        RelationshipDirection,
        RelationshipKind,
        RelationshipProvenance,
        ResourceKind,
        Sensitivity,
        ArtifactAuthorship,
        LoopExitKind,
        MessageRole,
        CostBasis,
        CostEstimateStatus,
        SemanticEvidenceKind,
        SemanticKind,
        LearningCandidateStatus,
        LearningCandidateRejectionReason,
        LearningCandidateTarget,
        LearningCandidateAction,
        DatabaseWriteOutcome,
    )
}


def _pack(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _pack(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__record__": type(value).__name__,
            "fields": {
                field.name: _pack(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return {"__enum__": type(value).__name__, "value": value.value}
    if isinstance(value, datetime):
        timestamp = value.isoformat()
        offset = value.utcoffset()
        if offset is not None and offset.total_seconds() == 0:
            timestamp = (
                value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            )
        return {"__datetime__": timestamp}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, (tuple, list)):
        return [_pack(item) for item in value]
    return value


def _unpack(value: object) -> object:
    if isinstance(value, list):
        return tuple(_unpack(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "__datetime__" in value:
        return datetime.fromisoformat(str(value["__datetime__"]))
    if "__decimal__" in value:
        return Decimal(str(value["__decimal__"]))
    if "__enum__" in value:
        enum = _ENUM_TYPES[str(value["__enum__"])]
        return enum(value["value"])
    if "__record__" in value:
        record = _RECORD_TYPES[str(value["__record__"])]
        raw_fields = value["fields"]
        if not isinstance(raw_fields, dict):
            raise ValueError("invalid stored record fields")
        return record(**{name: _unpack(item) for name, item in raw_fields.items()})
    return {key: _unpack(item) for key, item in value.items()}


def _dumps(value: object) -> str:
    return json.dumps(_pack(value), sort_keys=True, separators=(",", ":"))


def _loads(value: str) -> object:
    return _unpack(json.loads(value))


def _expect(value: object, expected: type[Any]) -> Any:
    if not isinstance(value, expected):
        raise TypeError(f"stored value is not {expected.__name__}")
    return value


__all__ = [
    "STATE_FORMAT_VERSION",
    "SQLiteStateStore",
    "StateCompatibilityCode",
    "StateCompatibilityError",
]
