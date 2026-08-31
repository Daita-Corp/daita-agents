"""Persist all durable agent state through the sole SQLite storage boundary."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import TypeVar, cast

from .._json import FrozenJsonObject, canonical_json
from ..adapters.mcp import (
    MCP_MAX_ACTIVE_TOOLS_PER_AGENT,
    MCP_MAX_AGENT_CATALOG_BYTES,
    MCP_MAX_BINDING_CANONICAL_BYTES,
    MCP_MAX_BINDINGS_PER_AGENT,
    MCPAdmissionError,
    MCPBindingState,
    MCPServerBinding,
)
from ..adapters.models import SourceRegistration
from ..artifacts.models import (
    ArtifactRef,
    artifact_ref_from_mapping,
)
from ..autonomy import (
    MAX_AUTONOMOUS_FOLLOWUPS_PER_AGENT,
    AutonomousFollowup,
    FollowupCompletionConflictError,
    FollowupDisposition,
    FollowupIdentityConflictError,
    assess_followup_conclusion,
    terminal_job_event_payload,
)
from ..capabilities import ExecutionScope
from ..catalog.models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSnapshotRef,
    CatalogSummary,
    CatalogSync,
    FacetKind,
    RelationshipKind,
    SourceCatalogSnapshot,
)
from ..catalog.protocols import CatalogStoreError
from ..distribution.models import (
    MAX_DELIVERIES_PER_AGENT,
    MAX_DELIVERY_LIST_PAGE_SIZE,
    Delivery,
    DeliveryState,
    DeliverySubjectKind,
    OutcomeArtifactReference,
    OutcomeConclusionKind,
    OutcomeState,
    conclusion_preview_projection,
    outcome_artifact_reference,
    validate_outcome_artifact_references,
)
from ..distribution.owner import construct_logical_delivery
from ..errors import StateCompatibilityCode, StateCompatibilityError
from ..identity import AgentIdentity, AgentIdentityConflictError
from ..jobs.models import (
    MAX_ACTIVE_JOBS_PER_AGENT,
    MAX_JOB_ATTEMPTS,
    MAX_JOB_LIST_PAGE_SIZE,
    MAX_JOBS_PER_AGENT,
    MAX_QUEUED_JOBS_PER_AGENT,
    MAX_RUNNING_JOBS_PER_AGENT,
    MAX_RUNNING_JOBS_PER_SOURCE,
    ExternalIntent,
    ExternalIntentDisposition,
    ExternalIntentKind,
    ExternalObservation,
    JobAttempt,
    JobAttemptStatus,
    JobCompletionBinding,
    JobCompletionOwnerKind,
    JobDesiredState,
    JobExecutionMode,
    JobResult,
    JobRun,
    JobStatus,
)
from ..learning_candidates import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_STAMPS,
    LearningCandidate,
    LearningCandidateError,
    LearningCandidateNotFoundError,
    LearningCandidateRejectionReason,
    LearningCandidateReviewStamp,
    LearningCandidateStatus,
    LearningReviewRunTail,
)
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ToolResultBlock,
)
from ..llm.pricing import CostEstimateStatus
from ..loop.models import (
    ConversationRun,
    LoopExit,
    LoopExitKind,
    RunInput,
    RunOrigin,
    Transcript,
    validate_completed_transcript,
)
from ..routines.models import (
    MAX_ACTIVE_ROUTINES_PER_AGENT,
    MAX_ROUTINE_ATTEMPTS,
    MAX_ROUTINE_HISTORY_PAGE_SIZE,
    MAX_ROUTINE_LIST_PAGE_SIZE,
    MAX_SCHEDULED_ROUTINES_PER_AGENT,
    ROUTINE_CLAIM_LEASE_SECONDS,
    ResourceRevisionObservation,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineSlotKind,
    RoutineState,
    ScheduledRoutine,
)
from ..routines.schedule import (
    first_slot,
    manual_slot_key,
    next_slot,
    occurrence_id as routine_occurrence_id,
    scheduled_slot_key,
    select_due_slot,
    validate_schedule,
)
from ..semantics import (
    SEMANTIC_MAX_ANNOTATIONS,
    SemanticAnnotation,
    SemanticDigestMismatchError,
    SemanticNotFoundError,
    SemanticValidationError,
    semantic_annotation_sha256,
)
from .sqlite_codecs import (
    CurrentSourceAdapterError,
    decode_autonomous_followup,
    decode_catalog_snapshot,
    decode_catalog_sync,
    decode_delivery,
    decode_identifier,
    decode_identity,
    decode_job_run,
    decode_learning_candidate,
    decode_loop_exit,
    decode_mcp_binding,
    decode_message,
    decode_postgresql_update_scope,
    decode_receipt,
    decode_review_stamps,
    decode_routine_occurrence,
    decode_run_input,
    decode_scheduled_routine,
    decode_semantic_annotation,
    decode_source,
    decode_source_read_scope,
    encode_autonomous_followup,
    encode_catalog_snapshot,
    encode_catalog_sync,
    encode_delivery,
    encode_identifier,
    encode_identity,
    encode_job_run,
    encode_learning_candidate,
    encode_loop_exit,
    encode_mcp_binding,
    encode_message,
    encode_postgresql_update_scope,
    encode_receipt,
    encode_review_stamps,
    encode_routine_occurrence,
    encode_run_input,
    encode_scheduled_routine,
    encode_semantic_annotation,
    encode_source,
    encode_source_read_scope,
)
from .sqlite_migrations import (
    CURRENT_REVISION,
    MIGRATIONS,
    MigrationJournalError,
    MigrationJournalNewerError,
    create_current,
    inspect_journal,
    upgrade_journaled,
)
from .sqlite_records import (
    DatabaseWriteOutcome,
    DatabaseWriteReceipt,
    DatabaseWriteReceiptConflictError,
    PostgreSQLUpdateScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    database_write_aware as _database_write_aware,
    database_write_receipt_id,
    database_write_text as _database_write_text,
    postgresql_update_authorization_fingerprint,
    validate_database_write_receipt_id,
)
from .sqlite_schema import (
    CURRENT_TABLES,
    require_healthy,
    require_schema,
    table_names,
)

_CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH = 64
_ACTIVE_SOURCE_KEY_PREFIX = "active_source:"
_LEARNING_REVIEW_STAMPS_KEY_PREFIX = "learning_review_stamps:"
_T = TypeVar("_T")


def _active_mcp_tool_count(bindings: Iterable[MCPServerBinding]) -> int:
    return sum(
        len(binding.tools)
        for binding in bindings
        if binding.state is MCPBindingState.ACTIVE
    )


def _active_source_key(agent_id: str) -> str:
    return f"{_ACTIVE_SOURCE_KEY_PREFIX}{agent_id}"


def _learning_review_stamps_key(agent_id: str) -> str:
    return f"{_LEARNING_REVIEW_STAMPS_KEY_PREFIX}{agent_id}"


def _decode_job_rows(
    rows: Iterable[tuple[object, object]],
    *,
    agent_id: str,
) -> tuple[JobRun, ...]:
    material = tuple(rows)
    if len(material) > MAX_JOBS_PER_AGENT:
        raise RuntimeError("stored job count exceeds its fixed bound")
    jobs: list[JobRun] = []
    for job_id, data in material:
        if not isinstance(job_id, str) or not isinstance(data, str):
            raise RuntimeError("stored job identity is invalid")
        jobs.append(decode_job_run(data, agent_id=agent_id, job_id=job_id))
    return tuple(jobs)


def _load_job_row(
    connection: sqlite3.Connection,
    agent_id: str,
    job_id: str,
) -> tuple[JobRun, str] | None:
    row = connection.execute(
        "SELECT data FROM job_runs WHERE agent_id = ? AND job_id = ?",
        (agent_id, job_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored job payload is invalid")
    return decode_job_run(row[0], agent_id=agent_id, job_id=job_id), row[0]


def _replace_job_row(
    connection: sqlite3.Connection,
    current_data: str,
    job: JobRun,
) -> None:
    result = connection.execute(
        """UPDATE job_runs SET data = ?
           WHERE agent_id = ? AND job_id = ? AND data = ?""",
        (encode_job_run(job), job.agent_id, job.job_id, current_data),
    )
    if result.rowcount != 1:
        raise RuntimeError("job changed during its conditional transition")


def _load_followup_row(
    connection: sqlite3.Connection,
    agent_id: str,
    followup_id: str,
) -> tuple[AutonomousFollowup, str] | None:
    row = connection.execute(
        "SELECT data FROM autonomous_followups "
        "WHERE agent_id = ? AND followup_id = ?",
        (agent_id, followup_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored autonomous follow-up payload is invalid")
    return (
        decode_autonomous_followup(
            row[0],
            agent_id=agent_id,
            followup_id=followup_id,
        ),
        row[0],
    )


def _replace_followup_row(
    connection: sqlite3.Connection,
    current_data: str,
    followup: AutonomousFollowup,
) -> None:
    result = connection.execute(
        "UPDATE autonomous_followups SET data = ? "
        "WHERE agent_id = ? AND followup_id = ? AND data = ?",
        (
            encode_autonomous_followup(followup),
            followup.agent_id,
            followup.followup_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise RuntimeError("follow-up changed during its conditional transition")


def _load_delivery_row(
    connection: sqlite3.Connection,
    agent_id: str,
    delivery_id: str,
) -> tuple[Delivery, str] | None:
    row = connection.execute(
        "SELECT data FROM deliveries WHERE agent_id = ? AND delivery_id = ?",
        (agent_id, delivery_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored delivery payload is invalid")
    return (
        decode_delivery(row[0], agent_id=agent_id, delivery_id=delivery_id),
        row[0],
    )


def _insert_delivery(connection: sqlite3.Connection, delivery: Delivery) -> None:
    count = connection.execute(
        "SELECT COUNT(*) FROM deliveries WHERE agent_id = ?",
        (delivery.agent_id,),
    ).fetchone()
    if int(count[0]) >= MAX_DELIVERIES_PER_AGENT:
        acknowledged = connection.execute(
            "SELECT delivery_id FROM deliveries "
            "WHERE agent_id = ? AND state = ? "
            "ORDER BY created_at_us, delivery_id LIMIT 1",
            (delivery.agent_id, DeliveryState.ACKNOWLEDGED.value),
        ).fetchone()
        if acknowledged is None:
            raise ValueError("delivery_retention_limit_exceeded")
        deleted = connection.execute(
            "DELETE FROM deliveries "
            "WHERE agent_id = ? AND delivery_id = ? AND state = ?",
            (
                delivery.agent_id,
                acknowledged[0],
                DeliveryState.ACKNOWLEDGED.value,
            ),
        )
        if deleted.rowcount != 1:
            raise RuntimeError("acknowledged delivery changed during reclamation")
    connection.execute(
        """INSERT INTO deliveries(
               agent_id, delivery_id, conversation_id, subject_kind, subject_id,
               logical_key, target_kind, target_fingerprint, state,
               created_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            delivery.agent_id,
            delivery.delivery_id,
            delivery.conversation_id,
            delivery.subject_kind.value,
            delivery.subject_id,
            delivery.logical_key,
            "conversation_inbox",
            delivery.target.target_fingerprint,
            delivery.visibility_state.value,
            _datetime_us(delivery.created_at),
            encode_delivery(delivery),
        ),
    )


def _datetime_us(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("routine query instant must be timezone-aware UTC")
    delta = value - datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _load_routine_row(
    connection: sqlite3.Connection,
    agent_id: str,
    routine_id: str,
) -> tuple[ScheduledRoutine, str] | None:
    row = connection.execute(
        "SELECT data FROM scheduled_routines " "WHERE agent_id = ? AND routine_id = ?",
        (agent_id, routine_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored scheduled routine payload is invalid")
    return (
        decode_scheduled_routine(
            row[0],
            agent_id=agent_id,
            routine_id=routine_id,
        ),
        row[0],
    )


def _replace_routine_row(
    connection: sqlite3.Connection,
    current_data: str,
    routine: ScheduledRoutine,
) -> None:
    result = connection.execute(
        """UPDATE scheduled_routines
           SET conversation_id = ?, state = ?, next_due_at_us = ?, data = ?
           WHERE agent_id = ? AND routine_id = ? AND data = ?""",
        (
            routine.conversation_id,
            routine.state.value,
            _datetime_us(routine.next_due_at),
            encode_scheduled_routine(routine),
            routine.agent_id,
            routine.routine_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise RuntimeError("scheduled routine changed during its transition")


def _load_routine_occurrence_row(
    connection: sqlite3.Connection,
    agent_id: str,
    occurrence_id: str,
) -> tuple[RoutineOccurrence, str] | None:
    row = connection.execute(
        "SELECT data FROM routine_occurrences "
        "WHERE agent_id = ? AND occurrence_id = ?",
        (agent_id, occurrence_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored routine occurrence payload is invalid")
    return (
        decode_routine_occurrence(
            row[0],
            agent_id=agent_id,
            occurrence_id=occurrence_id,
        ),
        row[0],
    )


def _replace_routine_occurrence_row(
    connection: sqlite3.Connection,
    current_data: str,
    occurrence: RoutineOccurrence,
) -> None:
    result = connection.execute(
        """UPDATE routine_occurrences
           SET state = ?, lease_expires_at_us = ?, reserved_run_id = ?, data = ?
           WHERE agent_id = ? AND occurrence_id = ? AND data = ?""",
        (
            occurrence.disposition.value,
            _datetime_us(occurrence.lease_expires_at),
            occurrence.reserved_run_id,
            encode_routine_occurrence(occurrence),
            occurrence.agent_id,
            occurrence.occurrence_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise RuntimeError("routine occurrence changed during its transition")


def _insert_routine_occurrence(
    connection: sqlite3.Connection,
    occurrence: RoutineOccurrence,
) -> None:
    connection.execute(
        """INSERT INTO routine_occurrences(
               agent_id, occurrence_id, routine_id, routine_revision, slot_key,
               state, lease_expires_at_us, reserved_run_id, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            occurrence.agent_id,
            occurrence.occurrence_id,
            occurrence.routine_id,
            occurrence.routine_revision,
            occurrence.slot_key,
            occurrence.disposition.value,
            _datetime_us(occurrence.lease_expires_at),
            occurrence.reserved_run_id,
            encode_routine_occurrence(occurrence),
        ),
    )


def _model_sensitivity_rank(value) -> int:
    order = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3,
    }
    return order[value.value]


def _decode_owned_read_scope(
    connection: sqlite3.Connection,
    registration: SourceRegistration,
    data: object,
) -> SourceReadScope | None:
    if not registration.active:
        if data is not None:
            raise SourcePermissionStateError(
                "detached source unexpectedly retains a read scope"
            )
        return None
    if not isinstance(data, str):
        raise SourcePermissionStateError("active source is missing its read scope")
    try:
        scope = decode_source_read_scope(
            data,
            agent_id=registration.agent_id,
            source_id=registration.id,
        )
        if scope.mode is SourceReadMode.SELECTED:
            _require_nonforeign_read_resources(connection, scope)
        return scope
    except SourcePermissionStateError:
        raise
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "active source read scope is undecodable"
        ) from None


def _require_nonforeign_read_resources(
    connection: sqlite3.Connection,
    scope: SourceReadScope,
) -> None:
    requested = set(scope.resource_ids)
    if not requested:
        return
    for (data,) in connection.execute(
        "SELECT data FROM snapshots WHERE agent_id = ?",
        (scope.agent_id,),
    ):
        snapshot = decode_catalog_snapshot(data)
        for resource in snapshot.resources:
            if resource.id in requested and resource.source_id != scope.source_id:
                raise SourcePermissionStateError(
                    "active source read scope contains a foreign resource"
                )


def _decode_source_state(
    connection: sqlite3.Connection,
    *,
    row_agent_id: object,
    row_source_id: object,
    source_data: object,
    read_scope_data: object,
    update_scope_count: object,
) -> SourceRegistration:
    if (
        not isinstance(row_agent_id, str)
        or not isinstance(row_source_id, str)
        or not isinstance(source_data, str)
        or not isinstance(update_scope_count, int)
        or isinstance(update_scope_count, bool)
        or update_scope_count < 0
    ):
        raise SourcePermissionStateError("stored source permission state is invalid")
    try:
        registration = decode_source(source_data)
    except CurrentSourceAdapterError as error:
        raise SourcePermissionStateError(str(error)) from None
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "stored source registration is invalid"
        ) from None
    if registration.agent_id != row_agent_id or registration.id != row_source_id:
        raise SourcePermissionStateError("stored source ownership is invalid")
    _decode_owned_read_scope(connection, registration, read_scope_data)
    if update_scope_count and (
        not registration.active or registration.adapter_id != "postgresql"
    ):
        raise SourcePermissionStateError("stored PostgreSQL update scope is foreign")
    return registration


def _source_state_row(
    connection: sqlite3.Connection,
    agent_id: str,
    source_id: str,
) -> tuple[object, ...] | None:
    return connection.execute(
        """SELECT s.agent_id, s.id, s.data, r.data,
                  (SELECT COUNT(*) FROM postgresql_update_scopes AS u
                   WHERE u.agent_id = s.agent_id AND u.source_id = s.id)
           FROM sources AS s
           LEFT JOIN source_read_scopes AS r
             ON r.agent_id = s.agent_id AND r.source_id = s.id
           WHERE s.agent_id = ? AND s.id = ?""",
        (agent_id, source_id),
    ).fetchone()


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


class _UpgradeCommitGate:
    """Let task cancellation veto a not-yet-committed state migration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._committed = False

    def start(self, connection: sqlite3.Connection) -> bool:
        with self._lock:
            if self._cancelled:
                return False
        connection.execute("BEGIN IMMEDIATE")
        with self._lock:
            if self._cancelled:
                connection.rollback()
                return False
            return True

    def commit(self, connection: sqlite3.Connection) -> bool:
        with self._lock:
            if self._cancelled:
                connection.rollback()
                return False
            connection.commit()
            return True

    def activate(self, callback: Callable[[], None]) -> bool:
        with self._lock:
            if self._cancelled:
                return False
            callback()
            self._committed = True
            return True

    def cancel_before_commit(self) -> bool:
        with self._lock:
            if self._committed:
                return False
            self._cancelled = True
            return True


class SQLiteStateStore:
    """Sole persistence and upgrade boundary for one admitted agent home."""

    current_revision = CURRENT_REVISION

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
        resolved_clock = clock or (lambda: datetime.now(UTC))

        upgrade_gate = _UpgradeCommitGate()

        def admit() -> None:
            if _initialize(resolved, upgrade_gate=upgrade_gate):
                _recover_started_database_write_receipts(resolved, resolved_clock)

        worker = asyncio.create_task(asyncio.to_thread(admit))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
                upgrade_gate.cancel_before_commit()
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
                    current = decode_identity(row[0])
                    if current != identity:
                        raise AgentIdentityConflictError(
                            "state database already belongs to another agent"
                        )
                    return current
                connection.execute(
                    "INSERT INTO metadata(key, data) VALUES ('identity', ?)",
                    (encode_identity(identity),),
                )
                return identity

        return await asyncio.to_thread(write)

    async def load_identity(self) -> AgentIdentity | None:
        def read() -> AgentIdentity | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
            return None if row is None else decode_identity(row[0])

        return await asyncio.to_thread(read)

    async def load_mcp_binding(
        self,
        agent_id: str,
        binding_id: str,
    ) -> MCPServerBinding | None:
        def read() -> MCPServerBinding | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM mcp_server_bindings
                       WHERE agent_id = ? AND binding_id = ?""",
                    (agent_id, binding_id),
                ).fetchone()
            return (
                None
                if row is None
                else decode_mcp_binding(
                    row[0],
                    agent_id=agent_id,
                    binding_id=binding_id,
                )
            )

        return await asyncio.to_thread(read)

    async def list_mcp_bindings(
        self,
        agent_id: str,
    ) -> tuple[MCPServerBinding, ...]:
        def read() -> tuple[MCPServerBinding, ...]:
            with _connect(self.path) as connection:
                rows = tuple(
                    connection.execute(
                        """SELECT binding_id, data FROM mcp_server_bindings
                           WHERE agent_id = ? ORDER BY binding_id""",
                        (agent_id,),
                    )
                )
            if len(rows) > MCP_MAX_BINDINGS_PER_AGENT:
                raise RuntimeError("stored MCP binding count exceeds its fixed bound")
            encoded_sizes = tuple(len(data.encode("utf-8")) for _, data in rows)
            if any(size > MCP_MAX_BINDING_CANONICAL_BYTES for size in encoded_sizes):
                raise RuntimeError("stored MCP binding exceeds its byte bound")
            if sum(encoded_sizes) > MCP_MAX_AGENT_CATALOG_BYTES:
                raise RuntimeError("stored MCP agent catalog exceeds its byte bound")
            bindings = tuple(
                decode_mcp_binding(
                    data,
                    agent_id=agent_id,
                    binding_id=binding_id,
                )
                for binding_id, data in rows
            )
            if _active_mcp_tool_count(bindings) > MCP_MAX_ACTIVE_TOOLS_PER_AGENT:
                raise RuntimeError(
                    "stored active MCP tool catalog exceeds its fixed bound"
                )
            return bindings

        return await asyncio.to_thread(read)

    async def store_mcp_binding(
        self,
        binding: MCPServerBinding,
        *,
        expected_revision: int | None,
    ) -> MCPServerBinding:
        if not isinstance(binding, MCPServerBinding):
            raise TypeError("binding must be MCPServerBinding")
        if expected_revision is not None and (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 1
        ):
            raise ValueError("expected_revision must be positive or None")
        encoded = encode_mcp_binding(binding)
        if len(encoded.encode("utf-8")) > MCP_MAX_BINDING_CANONICAL_BYTES:
            raise ValueError("MCP binding exceeds its byte bound")

        def write(connection: sqlite3.Connection) -> MCPServerBinding:
            row = connection.execute(
                """SELECT data FROM mcp_server_bindings
                   WHERE agent_id = ? AND binding_id = ?""",
                (binding.agent_id, binding.binding_id),
            ).fetchone()
            if row is None:
                if expected_revision is not None or binding.revision != 1:
                    raise ValueError("MCP binding revision precondition failed")
            else:
                current = decode_mcp_binding(
                    row[0],
                    agent_id=binding.agent_id,
                    binding_id=binding.binding_id,
                )
                if (
                    expected_revision is None
                    or current.revision != expected_revision
                    or binding.revision != expected_revision + 1
                ):
                    raise ValueError("MCP binding revision precondition failed")
            other_rows = tuple(
                connection.execute(
                    """SELECT binding_id, data FROM mcp_server_bindings
                       WHERE agent_id = ? AND binding_id <> ?""",
                    (binding.agent_id, binding.binding_id),
                )
            )
            total_bytes = len(encoded.encode("utf-8")) + sum(
                len(data.encode("utf-8")) for _, data in other_rows
            )
            if total_bytes > MCP_MAX_AGENT_CATALOG_BYTES:
                raise ValueError("MCP agent catalog exceeds its byte bound")
            other_bindings = tuple(
                decode_mcp_binding(
                    data,
                    agent_id=binding.agent_id,
                    binding_id=other_binding_id,
                )
                for other_binding_id, data in other_rows
            )
            active_tool_count = _active_mcp_tool_count((*other_bindings, binding))
            if active_tool_count > MCP_MAX_ACTIVE_TOOLS_PER_AGENT:
                raise MCPAdmissionError(
                    "mcp_agent_tool_limit_exceeded",
                    "The active MCP tool catalog exceeds its fixed per-agent bound.",
                    {
                        "observed_tools": active_tool_count,
                        "maximum_tools": MCP_MAX_ACTIVE_TOOLS_PER_AGENT,
                    },
                )
            if row is None:
                count = connection.execute(
                    "SELECT COUNT(*) FROM mcp_server_bindings WHERE agent_id = ?",
                    (binding.agent_id,),
                ).fetchone()[0]
                if count >= MCP_MAX_BINDINGS_PER_AGENT:
                    raise ValueError("MCP binding count exceeds its fixed bound")
                connection.execute(
                    """INSERT INTO mcp_server_bindings(agent_id, binding_id, data)
                       VALUES (?, ?, ?)""",
                    (
                        binding.agent_id,
                        binding.binding_id,
                        encoded,
                    ),
                )
                return binding
            result = connection.execute(
                """UPDATE mcp_server_bindings SET data = ?
                   WHERE agent_id = ? AND binding_id = ? AND data = ?""",
                (
                    encoded,
                    binding.agent_id,
                    binding.binding_id,
                    row[0],
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("MCP binding changed during its transition")
            return binding

        return await _run_cancellation_safe_transaction(self.path, write)

    async def admit_scheduled_routine(
        self,
        routine: ScheduledRoutine,
    ) -> ScheduledRoutine:
        """Atomically admit one pristine bounded routine and its first due slot."""

        if not isinstance(routine, ScheduledRoutine):
            raise TypeError("routine must be ScheduledRoutine")
        if (
            routine.revision != 1
            or routine.active_occurrence_id is not None
            or routine.last_occurrence_id is not None
            or bool(routine.last_delivery_ids)
            or routine.reserved_tokens != 0
            or routine.reserved_cost_usd != Decimal("0")
            or routine.charged_tokens != 0
            or routine.charged_cost_usd != Decimal("0")
            or routine.attempt_count != 0
            or routine.occurrence_count != 0
            or routine.consecutive_failures != 0
            or routine.state not in {RoutineState.ACTIVE, RoutineState.PAUSED}
        ):
            raise ValueError(
                "new routine must be one pristine active or paused aggregate"
            )
        validate_schedule(routine.schedule)
        next_due = None
        if routine.state is RoutineState.ACTIVE:
            next_due = first_slot(
                routine.schedule,
                not_before=routine.created_at,
                expires_at=routine.expires_at,
            )
            if next_due is None:
                raise ValueError("routine_schedule_expired")
        normalized = replace(routine, next_due_at=next_due)
        encoded = encode_scheduled_routine(normalized)

        def write(connection: sqlite3.Connection) -> ScheduledRoutine:
            row = connection.execute(
                "SELECT 1 FROM scheduled_routines "
                "WHERE agent_id = ? AND routine_id = ?",
                (normalized.agent_id, normalized.routine_id),
            ).fetchone()
            if row is not None:
                raise ValueError("routine_identity_already_exists")
            count = connection.execute(
                "SELECT COUNT(*) FROM scheduled_routines WHERE agent_id = ?",
                (normalized.agent_id,),
            ).fetchone()
            if int(count[0]) >= MAX_SCHEDULED_ROUTINES_PER_AGENT:
                raise ValueError("routine_retention_limit_exceeded")
            active_count = connection.execute(
                "SELECT COUNT(*) FROM scheduled_routines "
                "WHERE agent_id = ? AND state = ?",
                (normalized.agent_id, RoutineState.ACTIVE.value),
            ).fetchone()
            if (
                normalized.state is RoutineState.ACTIVE
                and int(active_count[0]) >= MAX_ACTIVE_ROUTINES_PER_AGENT
            ):
                raise ValueError("routine_active_limit_exceeded")
            connection.execute(
                """INSERT INTO scheduled_routines(
                       agent_id, routine_id, conversation_id, state,
                       next_due_at_us, data
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    normalized.agent_id,
                    normalized.routine_id,
                    normalized.conversation_id,
                    normalized.state.value,
                    _datetime_us(normalized.next_due_at),
                    encoded,
                ),
            )
            return normalized

        return await _run_cancellation_safe_transaction(self.path, write)

    async def load_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
    ) -> ScheduledRoutine | None:
        def read() -> ScheduledRoutine | None:
            with _connect(self.path) as connection:
                loaded = _load_routine_row(connection, agent_id, routine_id)
            return None if loaded is None else loaded[0]

        return await asyncio.to_thread(read)

    async def list_scheduled_routines(
        self,
        agent_id: str,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = MAX_ROUTINE_LIST_PAGE_SIZE,
    ) -> tuple[ScheduledRoutine, ...]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_ROUTINE_LIST_PAGE_SIZE
        ):
            raise ValueError("routine list limit is outside its bound")
        states = frozenset(states)
        if any(not isinstance(item, RoutineState) for item in states):
            raise TypeError("routine states must contain RoutineState values")

        def read() -> tuple[ScheduledRoutine, ...]:
            with _connect(self.path) as connection:
                rows = tuple(
                    connection.execute(
                        "SELECT routine_id, data FROM scheduled_routines "
                        "WHERE agent_id = ?",
                        (agent_id,),
                    )
                )
            if len(rows) > MAX_SCHEDULED_ROUTINES_PER_AGENT:
                raise RuntimeError("stored routine count exceeds its fixed bound")
            decoded = tuple(
                decode_scheduled_routine(data, agent_id=agent_id, routine_id=routine_id)
                for routine_id, data in rows
            )
            selected = tuple(
                item for item in decoded if not states or item.state in states
            )
            return tuple(
                sorted(
                    selected,
                    key=lambda item: (item.updated_at, item.routine_id),
                    reverse=True,
                )[:limit]
            )

        return await asyncio.to_thread(read)

    async def list_routine_occurrences(
        self,
        agent_id: str,
        routine_id: str,
        *,
        limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE,
    ) -> tuple[RoutineOccurrence, ...]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_ROUTINE_HISTORY_PAGE_SIZE
        ):
            raise ValueError("routine history limit is outside its bound")

        def read() -> tuple[RoutineOccurrence, ...]:
            with _connect(self.path) as connection:
                rows = tuple(
                    connection.execute(
                        """SELECT occurrence_id, data FROM routine_occurrences
                           WHERE agent_id = ? AND routine_id = ?
                           ORDER BY occurrence_id DESC LIMIT ?""",
                        (agent_id, routine_id, limit),
                    )
                )
            return tuple(
                decode_routine_occurrence(
                    data,
                    agent_id=agent_id,
                    occurrence_id=occurrence_id,
                )
                for occurrence_id, data in rows
            )

        return await asyncio.to_thread(read)

    async def revise_scheduled_routine(
        self,
        routine: ScheduledRoutine,
        *,
        expected_revision: int,
    ) -> ScheduledRoutine | None:
        """Conditionally persist one complete material routine revision."""

        if not isinstance(routine, ScheduledRoutine):
            raise TypeError("routine must be ScheduledRoutine")
        validate_schedule(routine.schedule)

        def write(connection: sqlite3.Connection) -> ScheduledRoutine | None:
            loaded = _load_routine_row(connection, routine.agent_id, routine.routine_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if current.revision != expected_revision:
                return None
            if current.active_occurrence_id is not None:
                raise ValueError("routine_has_active_occurrence")
            if (
                routine.revision != current.revision + 1
                or routine.agent_id != current.agent_id
                or routine.routine_id != current.routine_id
                or routine.conversation_id != current.conversation_id
                or routine.owner_principal_id != current.owner_principal_id
                or routine.created_at != current.created_at
                or routine.reserved_tokens != current.reserved_tokens
                or routine.reserved_cost_usd != current.reserved_cost_usd
                or routine.charged_tokens != current.charged_tokens
                or routine.charged_cost_usd != current.charged_cost_usd
                or routine.attempt_count != current.attempt_count
                or routine.occurrence_count != current.occurrence_count
            ):
                raise ValueError("routine revision changed immutable lifecycle state")
            if routine.state is RoutineState.ACTIVE:
                due = first_slot(
                    routine.schedule,
                    not_before=routine.updated_at,
                    expires_at=routine.expires_at,
                )
                if due is None:
                    raise ValueError("routine_schedule_expired")
            else:
                due = None
            normalized = replace(
                routine,
                next_due_at=due,
                active_occurrence_id=None,
            )
            _replace_routine_row(connection, encoded, normalized)
            return normalized

        return await _run_cancellation_safe_transaction(self.path, write)

    async def transition_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        state: RoutineState,
        transitioned_at: datetime,
    ) -> ScheduledRoutine | None:
        if not isinstance(state, RoutineState):
            raise TypeError("routine transition state is invalid")
        _datetime_us(transitioned_at)

        def write(connection: sqlite3.Connection) -> ScheduledRoutine | None:
            loaded = _load_routine_row(connection, agent_id, routine_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if current.revision != expected_revision:
                return None
            if current.active_occurrence_id is not None:
                raise ValueError("routine_has_active_occurrence")
            if state is RoutineState.ACTIVE:
                due = next_slot(
                    current.schedule,
                    after=transitioned_at,
                    expires_at=current.expires_at,
                )
                if due is None:
                    state_value = RoutineState.EXPIRED
                    due = None
                else:
                    state_value = RoutineState.ACTIVE
            else:
                state_value = state
                due = None
            updated = replace(
                current,
                state=state_value,
                next_due_at=due,
                revision=current.revision + 1,
                updated_at=transitioned_at,
            )
            _replace_routine_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def next_routine_deadline(self, agent_id: str) -> datetime | None:
        def read() -> datetime | None:
            with _connect(self.path) as connection:
                rows = tuple(
                    connection.execute(
                        """SELECT routine_id, data FROM scheduled_routines
                           WHERE agent_id = ? AND state = ?
                             AND next_due_at_us IS NOT NULL
                           ORDER BY next_due_at_us, routine_id""",
                        (agent_id, RoutineState.ACTIVE.value),
                    )
                )
            for routine_id, data in rows:
                routine = decode_scheduled_routine(
                    data,
                    agent_id=agent_id,
                    routine_id=routine_id,
                )
                if routine.active_occurrence_id is None:
                    return routine.next_due_at
            return None

        return await asyncio.to_thread(read)

    async def load_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
    ) -> RoutineOccurrence | None:
        def read() -> RoutineOccurrence | None:
            with _connect(self.path) as connection:
                loaded = _load_routine_occurrence_row(
                    connection,
                    agent_id,
                    occurrence_id,
                )
            return None if loaded is None else loaded[0]

        return await asyncio.to_thread(read)

    async def claim_due_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        expected_due_at: datetime,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None:
        """Conditionally claim one due canonical slot and reserve its run budget."""

        due_us = _datetime_us(expected_due_at)
        _datetime_us(claimed_at)

        def write(connection: sqlite3.Connection) -> RoutineOccurrence | None:
            loaded = _load_routine_row(connection, agent_id, routine_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if (
                current.revision != expected_revision
                or current.state is not RoutineState.ACTIVE
                or current.next_due_at != expected_due_at
            ):
                return None
            if current.active_occurrence_id is not None:
                active = _load_routine_occurrence_row(
                    connection,
                    agent_id,
                    current.active_occurrence_id,
                )
                if (
                    active is not None
                    and active[0].slot_kind is RoutineSlotKind.SCHEDULED
                ):
                    return active[0]
                return None
            selection = select_due_slot(
                current.schedule,
                materialized_due_at=expected_due_at,
                now=claimed_at,
                expires_at=current.expires_at,
                misfire_policy=current.misfire_policy,
            )
            if selection.selected_at is None:
                skipped = replace(
                    current,
                    next_due_at=selection.next_due_at,
                    updated_at=claimed_at,
                    state=(
                        RoutineState.ACTIVE
                        if selection.next_due_at is not None
                        else RoutineState.EXPIRED
                    ),
                )
                _replace_routine_row(connection, encoded, skipped)
                return None
            return self._claim_routine_slot_in_transaction(
                connection,
                current,
                encoded,
                slot_kind=RoutineSlotKind.SCHEDULED,
                slot_key=scheduled_slot_key(
                    current.routine_id,
                    current.revision,
                    selection.selected_at,
                ),
                scheduled_for=selection.selected_at,
                claimed_at=claimed_at,
                claim_token=claim_token,
            )

        del due_us
        return await _run_cancellation_safe_transaction(self.path, write)

    async def claim_manual_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        authorized_control_call_id: str,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None:
        _datetime_us(claimed_at)
        slot = manual_slot_key(
            routine_id,
            expected_revision,
            authorized_control_call_id,
        )

        def write(connection: sqlite3.Connection) -> RoutineOccurrence | None:
            existing = connection.execute(
                """SELECT occurrence_id, data FROM routine_occurrences
                   WHERE agent_id = ? AND routine_id = ?
                     AND routine_revision = ? AND slot_key = ?""",
                (agent_id, routine_id, expected_revision, slot),
            ).fetchone()
            if existing is not None:
                return decode_routine_occurrence(
                    existing[1],
                    agent_id=agent_id,
                    occurrence_id=existing[0],
                )
            loaded = _load_routine_row(connection, agent_id, routine_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if (
                current.revision != expected_revision
                or current.state is not RoutineState.ACTIVE
                or current.active_occurrence_id is not None
                or claimed_at >= current.expires_at
            ):
                return None
            return self._claim_routine_slot_in_transaction(
                connection,
                current,
                encoded,
                slot_kind=RoutineSlotKind.MANUAL,
                slot_key=slot,
                scheduled_for=claimed_at,
                claimed_at=claimed_at,
                claim_token=claim_token,
            )

        return await _run_cancellation_safe_transaction(self.path, write)

    def _claim_routine_slot_in_transaction(
        self,
        connection: sqlite3.Connection,
        current: ScheduledRoutine,
        encoded: str,
        *,
        slot_kind: RoutineSlotKind,
        slot_key: str,
        scheduled_for: datetime,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence:
        if (
            current.attempt_count >= current.cumulative_max_attempts
            or current.occurrence_count >= current.cumulative_max_occurrences
        ):
            raise ValueError("routine_occurrence_budget_exhausted")
        if (
            current.reserved_tokens
            + current.charged_tokens
            + current.per_run_max_tokens
            > current.cumulative_max_tokens
            or current.reserved_cost_usd
            + current.charged_cost_usd
            + current.per_run_max_cost_usd
            > current.cumulative_max_cost_usd
        ):
            raise ValueError("routine_model_budget_exhausted")
        identity = routine_occurrence_id(current.routine_id, slot_key)
        occurrence = RoutineOccurrence(
            occurrence_id=identity,
            agent_id=current.agent_id,
            routine_id=current.routine_id,
            routine_revision=current.revision,
            slot_kind=slot_kind,
            slot_key=slot_key,
            scheduled_for=scheduled_for,
            claimed_at=claimed_at,
            claim_token=claim_token,
            lease_expires_at=claimed_at
            + timedelta(seconds=ROUTINE_CLAIM_LEASE_SECONDS),
            precheck_observation=None,
            execution_scope=None,
            execution_scope_digest=None,
            reserved_run_id=None,
            reserved_tokens=current.per_run_max_tokens,
            reserved_cost_usd=current.per_run_max_cost_usd,
            charged_tokens=0,
            charged_cost_usd=Decimal("0"),
            run_bound_at=None,
            run_terminal_at=None,
            conclusion_digest=None,
            terminal_run_id=None,
            delivery_ids=(),
            attempt_count=1,
            failure_code=None,
            retry_at=None,
            disposition=RoutineOccurrenceDisposition.CLAIMED,
            created_at=claimed_at,
            updated_at=claimed_at,
        )
        _insert_routine_occurrence(connection, occurrence)
        updated = replace(
            current,
            active_occurrence_id=identity,
            reserved_tokens=current.reserved_tokens + current.per_run_max_tokens,
            reserved_cost_usd=(
                current.reserved_cost_usd + current.per_run_max_cost_usd
            ),
            attempt_count=current.attempt_count + 1,
            occurrence_count=current.occurrence_count + 1,
            updated_at=claimed_at,
        )
        _replace_routine_row(connection, encoded, updated)
        return occurrence

    async def bind_routine_occurrence_run(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        claim_token: str,
        run_id: str,
        execution_scope: ExecutionScope,
        bound_at: datetime,
        precheck_observation: ResourceRevisionObservation | None = None,
    ) -> RoutineOccurrence | None:
        if not isinstance(execution_scope, ExecutionScope):
            raise TypeError("routine execution scope is invalid")
        _datetime_us(bound_at)

        def write(connection: sqlite3.Connection) -> RoutineOccurrence | None:
            loaded = _load_routine_occurrence_row(connection, agent_id, occurrence_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if current.reserved_run_id is not None:
                return current if current.reserved_run_id == run_id else None
            if (
                current.disposition is not RoutineOccurrenceDisposition.CLAIMED
                or current.claim_token != claim_token
                or current.lease_expires_at is None
                or current.lease_expires_at < bound_at
                or execution_scope.agent_id != current.agent_id
                or execution_scope.routine_id != current.routine_id
                or execution_scope.routine_revision != current.routine_revision
                or execution_scope.occurrence_id != current.occurrence_id
            ):
                return None
            routine = _load_routine_row(connection, agent_id, current.routine_id)
            if routine is None or routine[0].active_occurrence_id != occurrence_id:
                return None
            precheck = routine[0].precheck
            if (precheck is None) != (precheck_observation is None):
                return None
            if precheck is not None:
                assert precheck_observation is not None
                if (
                    precheck.source_id != precheck_observation.source_id
                    or precheck.resource_id != precheck_observation.resource_id
                ):
                    return None
            updated = replace(
                current,
                precheck_observation=precheck_observation,
                execution_scope=execution_scope,
                execution_scope_digest=execution_scope.digest,
                reserved_run_id=run_id,
                run_bound_at=bound_at,
                disposition=RoutineOccurrenceDisposition.RUNNING,
                updated_at=bound_at,
            )
            _replace_routine_occurrence_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def mark_routine_occurrence_run_terminal(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        run_id: str,
        terminal_at: datetime,
    ) -> RoutineOccurrence | None:
        _datetime_us(terminal_at)

        def write(connection: sqlite3.Connection) -> RoutineOccurrence | None:
            loaded = _load_routine_occurrence_row(connection, agent_id, occurrence_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if (
                current.disposition
                is RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
            ):
                return current
            if (
                current.disposition is not RoutineOccurrenceDisposition.RUNNING
                or current.reserved_run_id != run_id
            ):
                return None
            row = connection.execute(
                "SELECT result FROM runs WHERE id = ? AND agent_id = ?",
                (run_id, agent_id),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            result = decode_loop_exit(row[0])
            if result.run_id != run_id:
                raise ValueError("routine terminal run identity is invalid")
            updated = replace(
                current,
                run_terminal_at=terminal_at,
                terminal_run_id=run_id,
                lease_expires_at=None,
                disposition=(
                    RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                ),
                updated_at=terminal_at,
            )
            _replace_routine_occurrence_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def finalize_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
        skipped_no_change_observation: ResourceRevisionObservation | None = None,
        failure_code: str | None = None,
        artifact_references: tuple[OutcomeArtifactReference, ...] = (),
        outcome_contract_failure_code: str | None = None,
    ) -> tuple[RoutineOccurrence, Delivery | None] | None:
        """Converge one occurrence, routine budget, slot, and logical delivery."""

        _datetime_us(finalized_at)

        def write(
            connection: sqlite3.Connection,
        ) -> tuple[RoutineOccurrence, Delivery | None] | None:
            loaded = _load_routine_occurrence_row(connection, agent_id, occurrence_id)
            if loaded is None:
                return None
            current, occurrence_data = loaded
            if skipped_no_change_observation is not None and failure_code is not None:
                raise ValueError("routine finalization outcomes are mutually exclusive")
            if outcome_contract_failure_code is not None and not re.fullmatch(
                r"[a-z][a-z0-9_]{0,127}", outcome_contract_failure_code
            ):
                raise ValueError("routine outcome contract failure code is invalid")
            contract_failure_code = outcome_contract_failure_code
            routine_loaded = _load_routine_row(
                connection,
                agent_id,
                current.routine_id,
            )
            if routine_loaded is None:
                return None
            routine, routine_data = routine_loaded
            target = routine.distribution_plan.targets[0]
            existing_row = connection.execute(
                "SELECT delivery_id, data FROM deliveries "
                "WHERE agent_id = ? AND subject_kind = ? AND subject_id = ? "
                "AND target_fingerprint = ?",
                (
                    agent_id,
                    DeliverySubjectKind.ROUTINE_OCCURRENCE.value,
                    occurrence_id,
                    target.target_fingerprint,
                ),
            ).fetchone()
            if existing_row is not None:
                existing_delivery = decode_delivery(
                    existing_row[1],
                    agent_id=agent_id,
                    delivery_id=existing_row[0],
                )
                return current, existing_delivery
            if routine.active_occurrence_id != occurrence_id:
                return None
            pre_run_outcome = (
                skipped_no_change_observation is not None or failure_code is not None
            )
            if pre_run_outcome:
                expected_disposition = (
                    RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE
                    if skipped_no_change_observation is not None
                    else RoutineOccurrenceDisposition.TERMINAL_FAILED
                )
                if current.disposition is expected_disposition:
                    return current, None
                if (
                    current.disposition
                    not in {
                        RoutineOccurrenceDisposition.CLAIMED,
                        RoutineOccurrenceDisposition.PRECHECKING,
                    }
                    or current.reserved_run_id is not None
                ):
                    return None
            elif (
                current.disposition
                is not RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                or current.reserved_run_id is None
            ):
                return None

            occurrence_observation = current.precheck_observation
            resulting_run_id: str | None = None
            charged_tokens = 0
            charged_cost = Decimal("0")
            sensitivity = routine.sensitivity_ceiling
            report_digest: str | None = None
            report_preview: str | None = None
            report_truncated = False
            validated_artifact_references: tuple[OutcomeArtifactReference, ...] = ()
            if pre_run_outcome:
                occurrence_observation = skipped_no_change_observation
                successful = occurrence_observation is not None
                terminal_failure_code = failure_code
                disposition = (
                    RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE
                    if successful
                    else RoutineOccurrenceDisposition.TERMINAL_FAILED
                )
                outcome = "skipped_no_change" if successful else "failed"
                reason = (
                    "resource_revision_unchanged"
                    if successful
                    else terminal_failure_code
                )
                if occurrence_observation is not None:
                    precheck = routine.precheck
                    if (
                        precheck is None
                        or occurrence_observation.source_id != precheck.source_id
                        or occurrence_observation.resource_id != precheck.resource_id
                    ):
                        raise ValueError("routine precheck observation is out of scope")
                if successful and any(
                    requirement.minimum_count > 0
                    for requirement in routine.outcome_contract.artifact_requirements
                ):
                    successful = False
                    terminal_failure_code = "outcome_artifact_contract_failed"
                    disposition = RoutineOccurrenceDisposition.TERMINAL_FAILED
                    outcome = "failed"
                    reason = terminal_failure_code
            else:
                assert current.reserved_run_id is not None
                run_row = connection.execute(
                    "SELECT input, result FROM runs WHERE id = ? AND agent_id = ?",
                    (current.reserved_run_id, agent_id),
                ).fetchone()
                if run_row is None or run_row[1] is None:
                    return None
                run_input = decode_run_input(run_row[0])
                result = decode_loop_exit(run_row[1])
                if (
                    run_input.origin is not RunOrigin.SCHEDULED_ROUTINE
                    or run_input.execution_scope is None
                    or run_input.execution_scope.routine_id != current.routine_id
                    or run_input.execution_scope.routine_revision
                    != current.routine_revision
                    or run_input.execution_scope.occurrence_id != current.occurrence_id
                    or run_input.execution_scope.distribution_plan_digest
                    != routine.distribution_plan.plan_digest
                    or result.run_id != current.reserved_run_id
                ):
                    raise ValueError("routine terminal run scope is invalid")
                message_rows = connection.execute(
                    "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                    (current.reserved_run_id,),
                ).fetchall()
                transcript = Transcript(
                    run=run_input,
                    messages=tuple(
                        decode_message(message_data) for (message_data,) in message_rows
                    ),
                )
                successful = (
                    result.kind is LoopExitKind.COMPLETED
                    and isinstance(result.final_text, str)
                    and bool(result.final_text.strip())
                )
                for message in transcript.messages:
                    for block in message.content:
                        if (
                            isinstance(block, ToolResultBlock)
                            and block.sensitivity is not None
                            and block.sensitivity.routing_rank
                            > sensitivity.routing_rank
                        ):
                            sensitivity = block.sensitivity
                if (
                    sensitivity.routing_rank
                    > routine.outcome_contract.maximum_effective_sensitivity.routing_rank
                ):
                    successful = False
                    contract_failure_code = "outcome_sensitivity_contract_failed"
                estimate = result.usage.cost_estimate
                charged_cost = (
                    estimate.amount_usd
                    if estimate.status is CostEstimateStatus.COMPLETE
                    and estimate.amount_usd is not None
                    and estimate.amount_usd <= current.reserved_cost_usd
                    else current.reserved_cost_usd
                )
                charged_tokens = min(result.usage.total_tokens, current.reserved_tokens)
                if successful:
                    contract_failure = contract_failure_code
                    try:
                        expected_artifact_references = tuple(
                            sorted(
                                (
                                    outcome_artifact_reference(ref)
                                    for ref in result.artifacts
                                ),
                                key=lambda item: item.artifact_id,
                            )
                        )
                        validated_artifact_references = (
                            validate_outcome_artifact_references(
                                artifact_references,
                                contract=routine.outcome_contract,
                                resulting_run_id=current.reserved_run_id,
                            )
                        )
                        if (
                            validated_artifact_references
                            != expected_artifact_references
                        ):
                            raise ValueError("validated artifact references differ")
                    except (TypeError, ValueError):
                        contract_failure = "outcome_artifact_contract_failed"
                        validated_artifact_references = ()
                    if contract_failure is not None:
                        successful = False
                        contract_failure_code = contract_failure
                if result.final_text is not None:
                    report_digest, report_preview, report_truncated = (
                        conclusion_preview_projection(result.final_text)
                    )
                resulting_run_id = current.reserved_run_id
                terminal_failure_code = (
                    None
                    if successful
                    else (contract_failure_code or f"routine_run_{result.reason}")
                )
                disposition = (
                    RoutineOccurrenceDisposition.COMPLETED
                    if successful
                    else RoutineOccurrenceDisposition.TERMINAL_FAILED
                )
                outcome = "completed" if successful else "failed"
                reason = "completed" if successful else terminal_failure_code

            if current.slot_kind is RoutineSlotKind.MANUAL:
                following = routine.next_due_at
            else:
                following = next_slot(
                    routine.schedule,
                    after=current.scheduled_for,
                    expires_at=routine.expires_at,
                )
                while following is not None and following <= finalized_at:
                    following = next_slot(
                        routine.schedule,
                        after=following,
                        expires_at=routine.expires_at,
                    )
            failures = 0 if successful else routine.consecutive_failures + 1
            if successful:
                next_state = (
                    RoutineState.ACTIVE
                    if following is not None
                    else (
                        RoutineState.EXPIRED
                        if finalized_at >= routine.expires_at
                        else RoutineState.COMPLETED
                    )
                )
            elif failures >= routine.maximum_consecutive_failures or following is None:
                next_state = RoutineState.NEEDS_ATTENTION
                following = None
            else:
                next_state = RoutineState.ACTIVE
            escalation = not successful and next_state is RoutineState.NEEDS_ATTENTION

            payload = {
                "subject": {
                    "kind": DeliverySubjectKind.ROUTINE_OCCURRENCE.value,
                    "subject_id": occurrence_id,
                },
                "routine_id": current.routine_id,
                "routine_revision": current.routine_revision,
                "occurrence_id": current.occurrence_id,
                "scheduled_for": current.scheduled_for.isoformat(),
                "run_id": resulting_run_id,
                "outcome": outcome,
                "reason": reason,
                "escalation": escalation,
                "routine_state": next_state.value,
                "observation_digest": (
                    None
                    if occurrence_observation is None
                    else occurrence_observation.digest
                ),
                "report_digest": report_digest,
                "report_preview": report_preview,
                "report_truncated": report_truncated,
            }
            conclusion_digest = (
                "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            )
            conclusion_kind = (
                OutcomeConclusionKind.TERMINAL_RUN
                if resulting_run_id is not None
                else OutcomeConclusionKind.NO_MODEL_OCCURRENCE
            )
            delivery = construct_logical_delivery(
                delivery_id=delivery_id,
                agent_id=agent_id,
                conversation_id=routine.conversation_id,
                subject_kind=DeliverySubjectKind.ROUTINE_OCCURRENCE,
                subject_id=occurrence_id,
                target=target,
                conclusion_kind=conclusion_kind,
                conclusion_state=(
                    OutcomeState.SKIPPED_NO_CHANGE
                    if disposition is RoutineOccurrenceDisposition.SKIPPED_NO_CHANGE
                    else (OutcomeState.SUCCEEDED if successful else OutcomeState.FAILED)
                ),
                conclusion_id=resulting_run_id or occurrence_id,
                conclusion_digest=report_digest or conclusion_digest,
                conclusion_preview=report_preview or "",
                conclusion_preview_truncated=report_truncated,
                resulting_run_id=resulting_run_id,
                artifact_references=(
                    validated_artifact_references
                    if resulting_run_id is not None and successful
                    else ()
                ),
                effective_sensitivity=sensitivity,
                provenance_digest=(
                    current.execution_scope_digest
                    or (
                        None
                        if occurrence_observation is None
                        else occurrence_observation.digest
                    )
                    or conclusion_digest
                ),
                failure_code=terminal_failure_code,
                observed_at=finalized_at,
            )
            _insert_delivery(connection, delivery)

            completed_occurrence = replace(
                current,
                precheck_observation=occurrence_observation,
                reserved_tokens=0,
                reserved_cost_usd=Decimal("0"),
                charged_tokens=charged_tokens,
                charged_cost_usd=charged_cost,
                conclusion_digest=conclusion_digest,
                delivery_ids=(delivery.delivery_id,),
                failure_code=terminal_failure_code,
                lease_expires_at=None,
                disposition=disposition,
                updated_at=finalized_at,
            )
            acknowledged_observation = routine.last_acknowledged_precheck_observation
            if successful and occurrence_observation is not None:
                acknowledged_observation = occurrence_observation
            completed_routine = replace(
                routine,
                reserved_tokens=routine.reserved_tokens - current.reserved_tokens,
                reserved_cost_usd=(
                    routine.reserved_cost_usd - current.reserved_cost_usd
                ),
                charged_tokens=routine.charged_tokens + charged_tokens,
                charged_cost_usd=routine.charged_cost_usd + charged_cost,
                consecutive_failures=failures,
                last_acknowledged_precheck_observation=acknowledged_observation,
                active_occurrence_id=None,
                last_occurrence_id=current.occurrence_id,
                last_delivery_ids=(delivery.delivery_id,),
                next_due_at=following,
                state=next_state,
                updated_at=finalized_at,
            )
            _replace_routine_occurrence_row(
                connection,
                occurrence_data,
                completed_occurrence,
            )
            _replace_routine_row(connection, routine_data, completed_routine)
            return completed_occurrence, delivery

        return await _run_cancellation_safe_transaction(self.path, write)

    async def recover_stale_routine_occurrences(
        self,
        agent_id: str,
        *,
        recovered_at: datetime,
        claim_token_factory: Callable[[str], str],
    ) -> tuple[RoutineOccurrence, ...]:
        """Fence stale claims and converge already-terminal reserved runs."""

        recovered_us = _datetime_us(recovered_at)

        def write(connection: sqlite3.Connection) -> tuple[RoutineOccurrence, ...]:
            rows = tuple(
                connection.execute(
                    """SELECT occurrence_id, data FROM routine_occurrences
                       WHERE agent_id = ?
                         AND (state = ?
                              OR (state IN (?, ?, ?)
                                  AND (lease_expires_at_us IS NULL
                                       OR lease_expires_at_us <= ?)))
                       ORDER BY COALESCE(lease_expires_at_us, 0), occurrence_id""",
                    (
                        agent_id,
                        (
                            RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION.value
                        ),
                        RoutineOccurrenceDisposition.CLAIMED.value,
                        RoutineOccurrenceDisposition.PRECHECKING.value,
                        RoutineOccurrenceDisposition.RUNNING.value,
                        recovered_us,
                    ),
                )
            )
            recovered: list[RoutineOccurrence] = []
            for occurrence_id_value, occurrence_data in rows:
                current = decode_routine_occurrence(
                    occurrence_data,
                    agent_id=agent_id,
                    occurrence_id=occurrence_id_value,
                )
                if (
                    current.disposition
                    is RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                ):
                    recovered.append(current)
                    continue
                if current.disposition not in {
                    RoutineOccurrenceDisposition.CLAIMED,
                    RoutineOccurrenceDisposition.PRECHECKING,
                    RoutineOccurrenceDisposition.RUNNING,
                }:
                    continue
                if current.reserved_run_id is not None:
                    run_row = connection.execute(
                        "SELECT result FROM runs WHERE id = ? AND agent_id = ?",
                        (current.reserved_run_id, agent_id),
                    ).fetchone()
                    if run_row is not None and run_row[0] is not None:
                        updated = replace(
                            current,
                            run_terminal_at=recovered_at,
                            terminal_run_id=current.reserved_run_id,
                            lease_expires_at=None,
                            disposition=(
                                RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                            ),
                            updated_at=recovered_at,
                        )
                        _replace_routine_occurrence_row(
                            connection,
                            occurrence_data,
                            updated,
                        )
                        recovered.append(updated)
                        continue
                    if run_row is not None:
                        continue
                if current.attempt_count >= MAX_ROUTINE_ATTEMPTS:
                    continue
                routine_loaded = _load_routine_row(
                    connection,
                    agent_id,
                    current.routine_id,
                )
                if routine_loaded is None:
                    continue
                routine, routine_data = routine_loaded
                if routine.active_occurrence_id != current.occurrence_id:
                    continue
                token = claim_token_factory(current.occurrence_id)
                updated = replace(
                    current,
                    claimed_at=recovered_at,
                    claim_token=token,
                    lease_expires_at=recovered_at
                    + timedelta(seconds=ROUTINE_CLAIM_LEASE_SECONDS),
                    attempt_count=current.attempt_count + 1,
                    updated_at=recovered_at,
                )
                updated_routine = replace(
                    routine,
                    attempt_count=routine.attempt_count + 1,
                    updated_at=recovered_at,
                )
                _replace_routine_occurrence_row(
                    connection,
                    occurrence_data,
                    updated,
                )
                _replace_routine_row(connection, routine_data, updated_routine)
                recovered.append(updated)
            return tuple(recovered)

        return await _run_cancellation_safe_transaction(self.path, write)

    async def admit_job(self, job: JobRun) -> JobRun:
        if not isinstance(job, JobRun):
            raise TypeError("job must be JobRun")
        if (
            job.status is not JobStatus.QUEUED
            or job.desired_state is not JobDesiredState.RUN
            or job.revision != 1
            or job.attempts
        ):
            raise ValueError("new job must be one pristine queued aggregate")
        encoded = encode_job_run(job)

        def write(connection: sqlite3.Connection) -> JobRun:
            rows = tuple(
                connection.execute(
                    "SELECT job_id, data FROM job_runs WHERE agent_id = ?",
                    (job.agent_id,),
                )
            )
            jobs = _decode_job_rows(rows, agent_id=job.agent_id)
            if any(item.job_id == job.job_id for item in jobs):
                raise ValueError("job identity already exists")
            if len(jobs) >= MAX_JOBS_PER_AGENT:
                raise ValueError("job_retention_limit_exceeded")
            active = sum(not item.terminal for item in jobs)
            queued = sum(item.status is JobStatus.QUEUED for item in jobs)
            if active >= MAX_ACTIVE_JOBS_PER_AGENT:
                raise ValueError("job_active_limit_exceeded")
            if queued >= MAX_QUEUED_JOBS_PER_AGENT:
                raise ValueError("job_queue_limit_exceeded")
            connection.execute(
                "INSERT INTO job_runs(agent_id, job_id, data) VALUES (?, ?, ?)",
                (job.agent_id, job.job_id, encoded),
            )
            return job

        return await _run_cancellation_safe_transaction(self.path, write)

    async def load_job(self, agent_id: str, job_id: str) -> JobRun | None:
        def read() -> JobRun | None:
            with _connect(self.path) as connection:
                loaded = _load_job_row(connection, agent_id, job_id)
            return None if loaded is None else loaded[0]

        return await asyncio.to_thread(read)

    async def list_jobs(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = None,
        statuses: frozenset[JobStatus] = frozenset(),
        limit: int = MAX_JOB_LIST_PAGE_SIZE,
    ) -> tuple[JobRun, ...]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_JOB_LIST_PAGE_SIZE
        ):
            raise ValueError("job list limit is outside its bound")
        statuses = frozenset(statuses)
        if any(not isinstance(item, JobStatus) for item in statuses):
            raise TypeError("job statuses must contain JobStatus values")

        def read() -> tuple[JobRun, ...]:
            with _connect(self.path) as connection:
                jobs = _decode_job_rows(
                    tuple(
                        connection.execute(
                            "SELECT job_id, data FROM job_runs WHERE agent_id = ?",
                            (agent_id,),
                        )
                    ),
                    agent_id=agent_id,
                )
            selected = tuple(
                item
                for item in jobs
                if (conversation_id is None or item.conversation_id == conversation_id)
                and (not statuses or item.status in statuses)
            )
            return tuple(
                sorted(
                    selected,
                    key=lambda item: (item.created_at, item.job_id),
                    reverse=True,
                )[:limit]
            )

        return await asyncio.to_thread(read)

    async def list_unbound_terminal_daita_jobs(
        self,
        agent_id: str,
    ) -> tuple[JobRun, ...]:
        """Return every bounded terminal Daita job lacking a completion owner."""

        def read() -> tuple[JobRun, ...]:
            with _connect(self.path) as connection:
                jobs = _decode_job_rows(
                    connection.execute(
                        "SELECT job_id, data FROM job_runs WHERE agent_id = ?",
                        (agent_id,),
                    ),
                    agent_id=agent_id,
                )
            return tuple(
                sorted(
                    (
                        job
                        for job in jobs
                        if job.terminal
                        and job.specification.execution_mode is JobExecutionMode.DAITA
                        and job.completion_binding is None
                    ),
                    key=lambda item: (item.terminal_at, item.job_id),
                )
            )

        return await asyncio.to_thread(read)

    async def admit_autonomous_followup(
        self,
        followup: AutonomousFollowup,
    ) -> AutonomousFollowup:
        """Atomically bind one exact terminal Daita job to one follow-up."""

        if not isinstance(followup, AutonomousFollowup):
            raise TypeError("followup must be AutonomousFollowup")
        if (
            followup.disposition is not FollowupDisposition.AVAILABLE
            or followup.revision != 1
            or followup.attempt_count != 0
            or followup.reserved_cost_usd != 0
            or followup.reserved_tokens != 0
            or followup.charged_cost_usd != 0
            or followup.charged_tokens != 0
        ):
            raise ValueError("new follow-up must be one pristine available aggregate")

        def write(connection: sqlite3.Connection) -> AutonomousFollowup:
            event_row = connection.execute(
                "SELECT followup_id, data FROM autonomous_followups "
                "WHERE agent_id = ? AND event_id = ?",
                (followup.agent_id, followup.event_id),
            ).fetchone()
            if event_row is not None:
                existing = decode_autonomous_followup(
                    event_row[1],
                    agent_id=followup.agent_id,
                    followup_id=event_row[0],
                )
                if (
                    existing.job_id == followup.job_id
                    and existing.payload_digest == followup.payload_digest
                    and existing.event_type == followup.event_type
                    and existing.event_payload == followup.event_payload
                ):
                    return existing
                raise FollowupIdentityConflictError(
                    "terminal observation identity was reused with different content"
                )
            bound_row = connection.execute(
                "SELECT followup_id FROM autonomous_followups "
                "WHERE agent_id = ? AND job_id = ?",
                (followup.agent_id, followup.job_id),
            ).fetchone()
            if bound_row is not None:
                raise FollowupCompletionConflictError(
                    "terminal job already belongs to another follow-up"
                )
            count = connection.execute(
                "SELECT COUNT(*) FROM autonomous_followups WHERE agent_id = ?",
                (followup.agent_id,),
            ).fetchone()
            if int(count[0]) >= MAX_AUTONOMOUS_FOLLOWUPS_PER_AGENT:
                raise ValueError("autonomous_followup_retention_limit_exceeded")
            loaded = _load_job_row(
                connection,
                followup.agent_id,
                followup.job_id,
            )
            if loaded is None:
                raise FollowupCompletionConflictError("terminal job does not exist")
            job, job_data = loaded
            if (
                not job.terminal
                or job.specification.execution_mode is not JobExecutionMode.DAITA
                or job.revision != followup.job_terminal_revision
            ):
                raise FollowupCompletionConflictError(
                    "terminal job identity or revision is not eligible"
                )
            current_sensitivity = (
                job.specification.sensitivity
                if job.result is None
                else job.result.sensitivity
            )
            if (
                current_sensitivity.routing_rank
                > followup.execution_scope.sensitivity_ceiling.routing_rank
            ):
                raise FollowupCompletionConflictError(
                    "follow-up execution sensitivity is below its terminal job"
                )
            if job.completion_binding is not None:
                raise FollowupCompletionConflictError(
                    "terminal job already has a completion owner"
                )
            expected_payload = terminal_job_event_payload(job)
            if (
                followup.event_payload != expected_payload
                or followup.event_id != f"stage-c:{job.job_id}:{job.revision}"
                or followup.grant.allowed_terminal_job_observation != followup.event_id
                or followup.execution_scope.job_id != job.job_id
                or followup.execution_scope.job_revision != job.revision
            ):
                raise FollowupIdentityConflictError(
                    "terminal observation does not match authoritative job state"
                )
            connection.execute(
                "INSERT INTO autonomous_followups("
                "agent_id, followup_id, job_id, event_id, data"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    followup.agent_id,
                    followup.followup_id,
                    followup.job_id,
                    followup.event_id,
                    encode_autonomous_followup(followup),
                ),
            )
            binding = JobCompletionBinding(
                owner_kind=JobCompletionOwnerKind.STANDALONE_FOLLOWUP,
                owner_id=followup.followup_id,
                terminal_event_id=followup.event_id,
                bound_at=followup.received_at,
            )
            _replace_job_row(
                connection,
                job_data,
                replace(
                    job,
                    completion_binding=binding,
                    terminal_observed_at=followup.received_at,
                    updated_at=followup.received_at,
                    revision=job.revision + 1,
                ),
            )
            return followup

        return await _run_cancellation_safe_transaction(self.path, write)

    async def load_autonomous_followup(
        self,
        agent_id: str,
        followup_id: str,
    ) -> AutonomousFollowup | None:
        def read() -> AutonomousFollowup | None:
            with _connect(self.path) as connection:
                loaded = _load_followup_row(connection, agent_id, followup_id)
            return None if loaded is None else loaded[0]

        return await asyncio.to_thread(read)

    async def list_autonomous_followups(
        self,
        agent_id: str,
        *,
        dispositions: frozenset[FollowupDisposition] = frozenset(),
        limit: int = MAX_AUTONOMOUS_FOLLOWUPS_PER_AGENT,
    ) -> tuple[AutonomousFollowup, ...]:
        dispositions = frozenset(dispositions)
        if not 1 <= limit <= MAX_AUTONOMOUS_FOLLOWUPS_PER_AGENT:
            raise ValueError("follow-up list limit is outside its bound")

        def read() -> tuple[AutonomousFollowup, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    "SELECT followup_id, data FROM autonomous_followups "
                    "WHERE agent_id = ?",
                    (agent_id,),
                ).fetchall()
            items = tuple(
                decode_autonomous_followup(
                    data,
                    agent_id=agent_id,
                    followup_id=followup_id,
                )
                for followup_id, data in rows
            )
            return tuple(
                item
                for item in sorted(
                    items, key=lambda value: (value.created_at, value.followup_id)
                )
                if not dispositions or item.disposition in dispositions
            )[:limit]

        return await asyncio.to_thread(read)

    async def recover_stale_autonomous_followups(
        self,
        agent_id: str,
        *,
        recovered_at: datetime,
    ) -> tuple[AutonomousFollowup, ...]:
        """Recover expired claims without rerunning any bound terminal run."""

        def write(connection: sqlite3.Connection) -> tuple[AutonomousFollowup, ...]:
            rows = connection.execute(
                "SELECT followup_id, data FROM autonomous_followups "
                "WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
            recovered: list[AutonomousFollowup] = []
            for followup_id, encoded in rows:
                current = decode_autonomous_followup(
                    encoded,
                    agent_id=agent_id,
                    followup_id=followup_id,
                )
                if (
                    current.disposition
                    in {
                        FollowupDisposition.AVAILABLE,
                        FollowupDisposition.RETRYABLE_FAILED,
                    }
                    and current.grant.expires_at <= recovered_at
                ):
                    updated = replace(
                        current,
                        disposition=FollowupDisposition.EXPIRED,
                        updated_at=recovered_at,
                        revision=current.revision + 1,
                        failure_code="followup_grant_expired",
                    )
                elif (
                    current.disposition
                    in {
                        FollowupDisposition.CLAIMED,
                        FollowupDisposition.RUNNING,
                    }
                    and current.lease_expires_at is not None
                    and current.lease_expires_at <= recovered_at
                ):
                    result_row = (
                        None
                        if current.reserved_run_id is None
                        else connection.execute(
                            "SELECT result FROM runs WHERE id = ? AND agent_id = ?",
                            (current.reserved_run_id, agent_id),
                        ).fetchone()
                    )
                    if result_row is not None and result_row[0] is not None:
                        terminal = decode_loop_exit(result_row[0])
                        updated = replace(
                            current,
                            disposition=(
                                FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                            ),
                            updated_at=recovered_at,
                            revision=current.revision + 1,
                            run_bound_at=current.run_bound_at or current.updated_at,
                            run_terminal_at=terminal.created_at,
                            audit_context=(
                                current.audit_context
                                or {"recovered_run_id": current.reserved_run_id}
                            ),
                        )
                    else:
                        attempts_exhausted = (
                            current.attempt_count >= current.grant.max_attempts
                        )
                        updated = replace(
                            current,
                            disposition=(
                                FollowupDisposition.TERMINAL_FAILED
                                if attempts_exhausted
                                else FollowupDisposition.AVAILABLE
                            ),
                            updated_at=recovered_at,
                            revision=current.revision + 1,
                            claim_token=None,
                            lease_expires_at=None,
                            reserved_run_id=None,
                            reserved_cost_usd=Decimal("0"),
                            reserved_tokens=0,
                            run_bound_at=None,
                            run_terminal_at=None,
                            audit_context={},
                            failure_code=(
                                "followup_budget_exhausted"
                                if attempts_exhausted
                                else None
                            ),
                        )
                else:
                    continue
                _replace_followup_row(connection, encoded, updated)
                recovered.append(updated)
            return tuple(recovered)

        return await _run_cancellation_safe_transaction(self.path, write)

    async def next_autonomous_followup_deadline(
        self,
        agent_id: str,
    ) -> datetime | None:
        """Return the nearest feature-owned grant or live-claim deadline."""

        def read() -> datetime | None:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    "SELECT followup_id, data FROM autonomous_followups "
                    "WHERE agent_id = ?",
                    (agent_id,),
                ).fetchall()
            deadlines: list[datetime] = []
            for followup_id, data in rows:
                current = decode_autonomous_followup(
                    data,
                    agent_id=agent_id,
                    followup_id=followup_id,
                )
                if current.disposition in {
                    FollowupDisposition.AVAILABLE,
                    FollowupDisposition.RETRYABLE_FAILED,
                }:
                    deadlines.append(current.grant.expires_at)
                elif current.disposition in {
                    FollowupDisposition.CLAIMED,
                    FollowupDisposition.RUNNING,
                }:
                    deadlines.append(current.grant.expires_at)
                    if current.lease_expires_at is not None:
                        deadlines.append(current.lease_expires_at)
            return min(deadlines) if deadlines else None

        return await asyncio.to_thread(read)

    async def claim_next_autonomous_followup(
        self,
        agent_id: str,
        *,
        claim_token: str,
        reserved_run_id: str,
        claimed_at: datetime,
        lease_seconds: float,
    ) -> AutonomousFollowup | None:
        if not 0 < float(lease_seconds) <= 300:
            raise ValueError("follow-up lease_seconds is outside its bound")

        def write(connection: sqlite3.Connection) -> AutonomousFollowup | None:
            rows = connection.execute(
                "SELECT followup_id, data FROM autonomous_followups "
                "WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
            items = sorted(
                [
                    decode_autonomous_followup(
                        data,
                        agent_id=agent_id,
                        followup_id=followup_id,
                    )
                    for followup_id, data in rows
                ],
                key=lambda item: (item.created_at, item.followup_id),
            )
            row_data = {str(followup_id): str(data) for followup_id, data in rows}
            for current in items:
                encoded = row_data[current.followup_id]
                if current.disposition not in {
                    FollowupDisposition.AVAILABLE,
                    FollowupDisposition.RETRYABLE_FAILED,
                }:
                    continue
                if current.grant.expires_at <= claimed_at:
                    continue
                if (
                    current.attempt_count >= current.grant.max_attempts
                    or current.charged_cost_usd + current.grant.per_run_max_cost_usd
                    > current.grant.cumulative_max_cost_usd
                    or current.charged_tokens + current.grant.per_run_max_tokens
                    > current.grant.cumulative_max_tokens
                ):
                    updated = replace(
                        current,
                        disposition=FollowupDisposition.TERMINAL_FAILED,
                        updated_at=claimed_at,
                        revision=current.revision + 1,
                        failure_code="followup_budget_exhausted",
                    )
                    _replace_followup_row(connection, encoded, updated)
                    continue
                claimed = replace(
                    current,
                    disposition=FollowupDisposition.CLAIMED,
                    updated_at=claimed_at,
                    revision=current.revision + 1,
                    attempt_count=current.attempt_count + 1,
                    claim_token=claim_token,
                    lease_expires_at=claimed_at
                    + timedelta(seconds=float(lease_seconds)),
                    reserved_run_id=reserved_run_id,
                    reserved_cost_usd=current.grant.per_run_max_cost_usd,
                    reserved_tokens=current.grant.per_run_max_tokens,
                    failure_code=None,
                )
                _replace_followup_row(connection, encoded, claimed)
                return claimed
            return None

        return await _run_cancellation_safe_transaction(self.path, write)

    async def bind_autonomous_followup_run(
        self,
        agent_id: str,
        followup_id: str,
        *,
        claim_token: str,
        run_id: str,
        bound_at: datetime,
        audit_context: Mapping[str, object],
    ) -> AutonomousFollowup | None:
        def write(connection: sqlite3.Connection) -> AutonomousFollowup | None:
            loaded = _load_followup_row(connection, agent_id, followup_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if (
                current.disposition is not FollowupDisposition.CLAIMED
                or current.claim_token != claim_token
                or current.reserved_run_id != run_id
            ):
                return None
            if current.grant.expires_at <= bound_at:
                expired = replace(
                    current,
                    disposition=FollowupDisposition.EXPIRED,
                    updated_at=bound_at,
                    revision=current.revision + 1,
                    claim_token=None,
                    lease_expires_at=None,
                    reserved_run_id=None,
                    reserved_cost_usd=Decimal("0"),
                    reserved_tokens=0,
                    failure_code="followup_grant_expired",
                )
                _replace_followup_row(connection, encoded, expired)
                return None
            if current.lease_expires_at is None or current.lease_expires_at <= bound_at:
                attempts_exhausted = current.attempt_count >= current.grant.max_attempts
                stale = replace(
                    current,
                    disposition=(
                        FollowupDisposition.TERMINAL_FAILED
                        if attempts_exhausted
                        else FollowupDisposition.AVAILABLE
                    ),
                    updated_at=bound_at,
                    revision=current.revision + 1,
                    claim_token=None,
                    lease_expires_at=None,
                    reserved_run_id=None,
                    reserved_cost_usd=Decimal("0"),
                    reserved_tokens=0,
                    failure_code=(
                        "followup_attempts_exhausted"
                        if attempts_exhausted
                        else "followup_claim_expired"
                    ),
                )
                _replace_followup_row(connection, encoded, stale)
                return None
            updated = replace(
                current,
                disposition=FollowupDisposition.RUNNING,
                updated_at=bound_at,
                revision=current.revision + 1,
                run_bound_at=bound_at,
                audit_context=audit_context,
            )
            _replace_followup_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def fail_autonomous_followup_claim(
        self,
        agent_id: str,
        followup_id: str,
        *,
        claim_token: str,
        failed_at: datetime,
        failure_code: str,
        retryable: bool = False,
    ) -> AutonomousFollowup | None:
        def write(connection: sqlite3.Connection) -> AutonomousFollowup | None:
            loaded = _load_followup_row(connection, agent_id, followup_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if (
                current.disposition is not FollowupDisposition.CLAIMED
                or current.claim_token != claim_token
            ):
                return None
            updated = replace(
                current,
                disposition=(
                    FollowupDisposition.RETRYABLE_FAILED
                    if retryable and current.attempt_count < current.grant.max_attempts
                    else FollowupDisposition.TERMINAL_FAILED
                ),
                updated_at=failed_at,
                revision=current.revision + 1,
                claim_token=None,
                lease_expires_at=None,
                reserved_run_id=None,
                reserved_cost_usd=Decimal("0"),
                reserved_tokens=0,
                failure_code=failure_code,
            )
            _replace_followup_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def mark_autonomous_followup_run_terminal(
        self,
        agent_id: str,
        followup_id: str,
        *,
        run_id: str,
        terminal_at: datetime,
    ) -> AutonomousFollowup | None:
        def write(connection: sqlite3.Connection) -> AutonomousFollowup | None:
            loaded = _load_followup_row(connection, agent_id, followup_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if (
                current.disposition
                is FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
            ):
                return current
            if (
                current.disposition is not FollowupDisposition.RUNNING
                or current.reserved_run_id != run_id
            ):
                return None
            row = connection.execute(
                "SELECT result FROM runs WHERE id = ? AND agent_id = ?",
                (run_id, agent_id),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            updated = replace(
                current,
                disposition=FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
                updated_at=terminal_at,
                revision=current.revision + 1,
                run_terminal_at=terminal_at,
            )
            _replace_followup_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def finalize_autonomous_followup(
        self,
        agent_id: str,
        followup_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
    ) -> tuple[AutonomousFollowup, Delivery] | None:
        """Atomically charge, consume, and create exactly one logical delivery."""

        def write(
            connection: sqlite3.Connection,
        ) -> tuple[AutonomousFollowup, Delivery] | None:
            loaded = _load_followup_row(connection, agent_id, followup_id)
            if loaded is None:
                return None
            current, encoded = loaded
            target = current.grant.distribution_plan.targets[0]
            existing_row = connection.execute(
                "SELECT delivery_id, data FROM deliveries "
                "WHERE agent_id = ? AND subject_kind = ? AND subject_id = ? "
                "AND target_fingerprint = ?",
                (
                    agent_id,
                    DeliverySubjectKind.AUTONOMOUS_FOLLOWUP.value,
                    followup_id,
                    target.target_fingerprint,
                ),
            ).fetchone()
            if existing_row is not None:
                delivery = decode_delivery(
                    existing_row[1],
                    agent_id=agent_id,
                    delivery_id=existing_row[0],
                )
                return current, delivery
            if (
                current.disposition
                is not FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
            ):
                return None
            assert current.reserved_run_id is not None
            run_row = connection.execute(
                "SELECT input, result FROM runs WHERE id = ? AND agent_id = ?",
                (current.reserved_run_id, agent_id),
            ).fetchone()
            if run_row is None or run_row[1] is None:
                return None
            result = decode_loop_exit(run_row[1])
            message_rows = connection.execute(
                "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                (current.reserved_run_id,),
            ).fetchall()
            run_input = decode_run_input(run_row[0])
            if (
                run_input.origin is not RunOrigin.JOB_EVENT
                or run_input.execution_scope != current.execution_scope
                or run_input.execution_scope.distribution_plan_digest
                != current.grant.distribution_plan.plan_digest
                or result.run_id != current.reserved_run_id
            ):
                raise ValueError("follow-up terminal run scope is invalid")
            transcript = Transcript(
                run=run_input,
                messages=tuple(
                    decode_message(message_data) for (message_data,) in message_rows
                ),
            )
            loaded_job = _load_job_row(connection, agent_id, current.job_id)
            if loaded_job is None:
                return None
            job, _job_encoded = loaded_job
            evidence, conclusion_failure_code = assess_followup_conclusion(
                current,
                job,
                transcript,
                result,
            )
            successful = result.kind is LoopExitKind.COMPLETED and evidence is not None
            try:
                validate_outcome_artifact_references(
                    (),
                    contract=current.grant.outcome_contract,
                    resulting_run_id=current.reserved_run_id,
                )
                if result.artifacts:
                    raise ValueError("follow-up outcome contract permits no artifacts")
            except (TypeError, ValueError):
                successful = False
                conclusion_failure_code = "outcome_artifact_contract_failed"
            sensitivity = current.execution_scope.sensitivity_ceiling
            for message in transcript.messages:
                for block in message.content:
                    if (
                        isinstance(block, ToolResultBlock)
                        and block.sensitivity is not None
                        and block.sensitivity.routing_rank > sensitivity.routing_rank
                    ):
                        sensitivity = block.sensitivity
            if (
                sensitivity.routing_rank
                > current.grant.outcome_contract.maximum_effective_sensitivity.routing_rank
            ):
                successful = False
                conclusion_failure_code = "outcome_sensitivity_contract_failed"
            estimate = result.usage.cost_estimate
            charged_cost = (
                estimate.amount_usd
                if estimate.status is CostEstimateStatus.COMPLETE
                and estimate.amount_usd is not None
                and estimate.amount_usd <= current.reserved_cost_usd
                else current.reserved_cost_usd
            )
            charged_tokens = min(result.usage.total_tokens, current.reserved_tokens)
            report_digest: str | None = None
            report_preview: str | None = None
            report_truncated = False
            if result.final_text is not None:
                report_digest, bounded_report, report_truncated = (
                    conclusion_preview_projection(result.final_text)
                )
                report_preview = bounded_report
            if successful:
                failure_code = None
            elif result.kind is LoopExitKind.COMPLETED:
                if conclusion_failure_code is None:
                    raise ValueError(
                        "failed completed follow-up requires a conclusion failure code"
                    )
                failure_code = conclusion_failure_code
            else:
                failure_code = f"followup_run_{result.reason}"
            payload = {
                "subject": {
                    "kind": DeliverySubjectKind.AUTONOMOUS_FOLLOWUP.value,
                    "subject_id": followup_id,
                },
                "job_id": current.job_id,
                "run_id": current.reserved_run_id,
                "outcome": "completed" if successful else "failed",
                "reason": "completed" if successful else failure_code,
                "report_digest": report_digest,
                "report_preview": report_preview,
                "report_truncated": report_truncated,
                "evidence_digest": None if evidence is None else evidence.digest,
            }
            payload_digest = (
                "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            )
            conclusion_digest = report_digest or payload_digest
            delivery = construct_logical_delivery(
                delivery_id=delivery_id,
                agent_id=agent_id,
                conversation_id=current.conversation_id,
                subject_kind=DeliverySubjectKind.AUTONOMOUS_FOLLOWUP,
                subject_id=followup_id,
                target=target,
                conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
                conclusion_state=(
                    OutcomeState.SUCCEEDED if successful else OutcomeState.FAILED
                ),
                conclusion_id=current.reserved_run_id,
                conclusion_digest=conclusion_digest,
                conclusion_preview=report_preview or "",
                conclusion_preview_truncated=report_truncated,
                resulting_run_id=current.reserved_run_id,
                artifact_references=(),
                effective_sensitivity=sensitivity,
                provenance_digest=(
                    evidence.digest if evidence is not None else payload_digest
                ),
                failure_code=failure_code,
                observed_at=finalized_at,
            )
            _insert_delivery(connection, delivery)
            completed = replace(
                current,
                disposition=(
                    FollowupDisposition.COMPLETED
                    if successful
                    else FollowupDisposition.TERMINAL_FAILED
                ),
                updated_at=finalized_at,
                revision=current.revision + 1,
                reserved_cost_usd=Decimal("0"),
                reserved_tokens=0,
                charged_cost_usd=current.charged_cost_usd + charged_cost,
                charged_tokens=current.charged_tokens + charged_tokens,
                grant_consumed_at=(finalized_at if successful else None),
                conclusion_evidence=evidence,
                delivery_id=delivery_id,
                failure_code=failure_code,
            )
            _replace_followup_row(connection, encoded, completed)
            return completed, delivery

        return await _run_cancellation_safe_transaction(self.path, write)

    async def load_delivery(
        self,
        agent_id: str,
        delivery_id: str,
    ) -> Delivery | None:
        def read() -> Delivery | None:
            with _connect(self.path) as connection:
                loaded = _load_delivery_row(connection, agent_id, delivery_id)
                return None if loaded is None else loaded[0]

        return await asyncio.to_thread(read)

    async def list_deliveries(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = MAX_DELIVERY_LIST_PAGE_SIZE,
    ) -> tuple[Delivery, ...]:
        if not 1 <= limit <= MAX_DELIVERY_LIST_PAGE_SIZE:
            raise ValueError("delivery list limit is outside its bound")

        def read() -> tuple[Delivery, ...]:
            clauses = ["agent_id = ?"]
            parameters: list[object] = [agent_id]
            if conversation_id is not None:
                clauses.append("conversation_id = ?")
                parameters.append(conversation_id)
            if not include_acknowledged:
                clauses.append("state != ?")
                parameters.append(DeliveryState.ACKNOWLEDGED.value)
            parameters.append(limit)
            with _connect(self.path) as connection:
                rows = connection.execute(
                    "SELECT delivery_id, data FROM deliveries WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY created_at_us DESC, delivery_id DESC LIMIT ?",
                    tuple(parameters),
                ).fetchall()
            return tuple(
                decode_delivery(data, agent_id=agent_id, delivery_id=delivery_id)
                for delivery_id, data in rows
            )

        return await asyncio.to_thread(read)

    async def acknowledge_delivery(
        self,
        agent_id: str,
        delivery_id: str,
        *,
        acknowledged_at: datetime,
    ) -> Delivery | None:
        _datetime_us(acknowledged_at)

        def write(connection: sqlite3.Connection) -> Delivery | None:
            loaded = _load_delivery_row(connection, agent_id, delivery_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if current.visibility_state is DeliveryState.ACKNOWLEDGED:
                return current
            updated = replace(
                current,
                visibility_state=DeliveryState.ACKNOWLEDGED,
                updated_at=acknowledged_at,
                acknowledged_at=acknowledged_at,
            )
            result = connection.execute(
                "UPDATE deliveries SET state = ?, data = ? "
                "WHERE agent_id = ? AND delivery_id = ? AND data = ?",
                (
                    updated.visibility_state.value,
                    encode_delivery(updated),
                    agent_id,
                    delivery_id,
                    encoded,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("delivery changed during acknowledgment")
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def claim_next_job(
        self,
        agent_id: str,
        *,
        claim_token: str,
        execution_run_id: str,
        reserved_artifact_id: str,
        claimed_at: datetime,
        lease_seconds: float,
    ) -> JobRun | None:
        if (
            not isinstance(lease_seconds, (int, float))
            or isinstance(lease_seconds, bool)
            or not 0 < float(lease_seconds) <= 300
        ):
            raise ValueError("job lease_seconds is outside its bound")

        def write(connection: sqlite3.Connection) -> JobRun | None:
            rows = tuple(
                connection.execute(
                    "SELECT job_id, data FROM job_runs WHERE agent_id = ?",
                    (agent_id,),
                )
            )
            jobs = _decode_job_rows(rows, agent_id=agent_id)
            running = tuple(
                item
                for item in jobs
                if item.status in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}
            )
            if len(running) >= MAX_RUNNING_JOBS_PER_AGENT:
                return None
            per_source: dict[str, int] = {}
            for item in running:
                for source_id in item.source_ids:
                    per_source[source_id] = per_source.get(source_id, 0) + 1
            eligible = tuple(
                item
                for item in sorted(
                    jobs, key=lambda value: (value.created_at, value.job_id)
                )
                if item.status is JobStatus.QUEUED
                and item.desired_state is JobDesiredState.RUN
                and item.specification.deadline_at > claimed_at
                and len(item.attempts) < MAX_JOB_ATTEMPTS
                and all(
                    per_source.get(source_id, 0) < MAX_RUNNING_JOBS_PER_SOURCE
                    for source_id in item.source_ids
                )
            )
            if not eligible:
                return None
            current = eligible[0]
            loaded = _load_job_row(connection, agent_id, current.job_id)
            if loaded is None or loaded[0] != current:
                raise RuntimeError("job changed during claim selection")
            epoch = current.fencing_epoch + 1
            attempt = JobAttempt(
                number=len(current.attempts) + 1,
                fencing_epoch=epoch,
                claim_token=claim_token,
                execution_run_id=execution_run_id,
                reserved_artifact_id=reserved_artifact_id,
                status=JobAttemptStatus.CLAIMED,
                claimed_at=claimed_at,
                lease_expires_at=claimed_at + timedelta(seconds=float(lease_seconds)),
            )
            claimed = replace(
                current,
                status=JobStatus.RUNNING,
                updated_at=claimed_at,
                revision=current.revision + 1,
                fencing_epoch=epoch,
                attempts=(*current.attempts, attempt),
            )
            _replace_job_row(connection, loaded[1], claimed)
            return claimed

        return await _run_cancellation_safe_transaction(self.path, write)

    async def request_job_cancel(
        self,
        agent_id: str,
        job_id: str,
        *,
        requested_at: datetime,
    ) -> JobRun | None:
        def write(connection: sqlite3.Connection) -> JobRun | None:
            loaded = _load_job_row(connection, agent_id, job_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if current.terminal or current.desired_state is JobDesiredState.CANCEL:
                return current
            if current.status is JobStatus.QUEUED:
                updated = replace(
                    current,
                    desired_state=JobDesiredState.CANCEL,
                    cancel_requested_at=requested_at,
                    updated_at=requested_at,
                    revision=current.revision + 1,
                    status=JobStatus.CANCELLED,
                    terminal_at=requested_at,
                )
            else:
                updated = replace(
                    current,
                    desired_state=JobDesiredState.CANCEL,
                    cancel_requested_at=requested_at,
                    updated_at=requested_at,
                    revision=current.revision + 1,
                    status=JobStatus.CANCEL_REQUESTED,
                )
            _replace_job_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def finalize_job_attempt(
        self,
        agent_id: str,
        job_id: str,
        *,
        claim_token: str,
        fencing_epoch: int,
        attempt_status: JobAttemptStatus,
        completed_at: datetime,
        result: JobResult | None = None,
        failure_code: str | None = None,
    ) -> JobRun | None:
        if attempt_status is JobAttemptStatus.CLAIMED:
            raise ValueError("job finalization requires a settled attempt status")

        def write(connection: sqlite3.Connection) -> JobRun | None:
            loaded = _load_job_row(connection, agent_id, job_id)
            if loaded is None:
                return None
            current, encoded = loaded
            attempt = current.current_attempt
            if (
                current.status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}
                or attempt is None
                or attempt.status is not JobAttemptStatus.CLAIMED
                or attempt.claim_token != claim_token
                or attempt.fencing_epoch != fencing_epoch
                or current.fencing_epoch != fencing_epoch
            ):
                return None
            if attempt_status is JobAttemptStatus.SUCCEEDED:
                if not isinstance(result, JobResult):
                    raise ValueError("successful job finalization requires JobResult")
                if _model_sensitivity_rank(
                    result.sensitivity
                ) < _model_sensitivity_rank(current.specification.sensitivity):
                    raise ValueError("job result cannot lower sensitivity")
                if any(
                    ref.run_id != attempt.execution_run_id
                    or ref.conversation_id != current.conversation_id
                    or ref.capability_id
                    != current.specification.execution_capability_id
                    for ref in result.artifact_refs
                ):
                    raise ValueError("job artifact result identity is invalid")
                status = JobStatus.SUCCEEDED
            elif attempt_status is JobAttemptStatus.CANCELLED:
                if result is not None:
                    raise ValueError("cancelled job cannot retain a result")
                status = JobStatus.CANCELLED
            elif attempt_status is JobAttemptStatus.NEEDS_ATTENTION:
                if result is not None:
                    raise ValueError("needs-attention job cannot retain a result")
                status = JobStatus.NEEDS_ATTENTION
            else:
                if result is not None:
                    raise ValueError("failed job cannot retain a result")
                status = JobStatus.FAILED
            settled_attempt = replace(
                attempt,
                status=attempt_status,
                completed_at=completed_at,
                error_code=failure_code,
            )
            updated = replace(
                current,
                status=status,
                updated_at=completed_at,
                revision=current.revision + 1,
                attempts=(*current.attempts[:-1], settled_attempt),
                terminal_at=completed_at,
                result=result,
                failure_code=failure_code,
            )
            _replace_job_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def recover_stale_job(
        self,
        agent_id: str,
        job_id: str,
        *,
        recovered_at: datetime,
        restart_safe: bool,
    ) -> JobRun | None:
        def write(connection: sqlite3.Connection) -> JobRun | None:
            loaded = _load_job_row(connection, agent_id, job_id)
            if loaded is None:
                return None
            current, encoded = loaded
            attempt = current.current_attempt
            if (
                current.status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}
                or attempt is None
                or attempt.status is not JobAttemptStatus.CLAIMED
            ):
                return current
            if current.desired_state is JobDesiredState.CANCEL:
                attempt_status = JobAttemptStatus.CANCELLED
                status = JobStatus.CANCELLED
                terminal_at = recovered_at
                failure_code = None
            elif restart_safe and len(current.attempts) < MAX_JOB_ATTEMPTS:
                attempt_status = JobAttemptStatus.FENCED
                status = JobStatus.QUEUED
                terminal_at = None
                failure_code = None
            else:
                attempt_status = JobAttemptStatus.NEEDS_ATTENTION
                status = JobStatus.NEEDS_ATTENTION
                terminal_at = recovered_at
                failure_code = "job_recovery_unsafe"
            settled_attempt = replace(
                attempt,
                status=attempt_status,
                completed_at=recovered_at,
                error_code=failure_code,
            )
            updated = replace(
                current,
                status=status,
                updated_at=recovered_at,
                revision=current.revision + 1,
                fencing_epoch=current.fencing_epoch + 1,
                attempts=(*current.attempts[:-1], settled_attempt),
                terminal_at=terminal_at,
                failure_code=failure_code,
            )
            _replace_job_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def expire_due_jobs(
        self,
        agent_id: str,
        *,
        expired_at: datetime,
    ) -> tuple[JobRun, ...]:
        def write(connection: sqlite3.Connection) -> tuple[JobRun, ...]:
            rows = tuple(
                connection.execute(
                    "SELECT job_id, data FROM job_runs WHERE agent_id = ?",
                    (agent_id,),
                )
            )
            jobs = _decode_job_rows(rows, agent_id=agent_id)
            expired: list[JobRun] = []
            row_data = {str(job_id): str(data) for job_id, data in rows}
            for current in jobs:
                if (
                    current.status is not JobStatus.QUEUED
                    or current.specification.deadline_at > expired_at
                ):
                    continue
                updated = replace(
                    current,
                    status=JobStatus.FAILED,
                    updated_at=expired_at,
                    revision=current.revision + 1,
                    terminal_at=expired_at,
                    failure_code="job_deadline_exceeded",
                )
                _replace_job_row(connection, row_data[current.job_id], updated)
                expired.append(updated)
            return tuple(expired)

        return await _run_cancellation_safe_transaction(self.path, write)

    async def record_external_intent(
        self,
        agent_id: str,
        job_id: str,
        *,
        claim_token: str,
        fencing_epoch: int,
        intent: ExternalIntent,
    ) -> JobRun | None:
        if not isinstance(intent, ExternalIntent):
            raise TypeError("intent must be ExternalIntent")

        def write(connection: sqlite3.Connection) -> JobRun | None:
            loaded = _load_job_row(connection, agent_id, job_id)
            if loaded is None:
                return None
            current, encoded = loaded
            attempt = current.current_attempt
            if (
                attempt is None
                or attempt.status is not JobAttemptStatus.CLAIMED
                or attempt.claim_token != claim_token
                or attempt.fencing_epoch != fencing_epoch
                or current.fencing_epoch != fencing_epoch
            ):
                return None
            by_kind = {item.kind: item for item in attempt.external_intents}
            existing = by_kind.get(intent.kind)
            if existing is not None:
                return current if existing == intent else None
            updated_attempt = replace(
                attempt,
                external_intents=(*attempt.external_intents, intent),
            )
            updated = replace(
                current,
                updated_at=intent.requested_at,
                revision=current.revision + 1,
                attempts=(*current.attempts[:-1], updated_attempt),
            )
            _replace_job_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def settle_external_intent(
        self,
        agent_id: str,
        job_id: str,
        *,
        claim_token: str,
        fencing_epoch: int,
        kind: ExternalIntentKind,
        disposition: ExternalIntentDisposition,
        completed_at: datetime,
        external_job_id: str | None = None,
        reason_code: str | None = None,
    ) -> JobRun | None:
        if disposition is ExternalIntentDisposition.PENDING:
            raise ValueError("external intent settlement cannot remain pending")

        def write(connection: sqlite3.Connection) -> JobRun | None:
            loaded = _load_job_row(connection, agent_id, job_id)
            if loaded is None:
                return None
            current, encoded = loaded
            attempt = current.current_attempt
            if (
                attempt is None
                or attempt.status is not JobAttemptStatus.CLAIMED
                or attempt.claim_token != claim_token
                or attempt.fencing_epoch != fencing_epoch
                or current.fencing_epoch != fencing_epoch
            ):
                return None
            index = next(
                (
                    position
                    for position, item in enumerate(attempt.external_intents)
                    if item.kind is kind
                ),
                None,
            )
            if index is None:
                return None
            pending = attempt.external_intents[index]
            if pending.disposition is not ExternalIntentDisposition.PENDING:
                return current
            settled = replace(
                pending,
                disposition=disposition,
                completed_at=completed_at,
                external_job_id=external_job_id,
                reason_code=reason_code,
            )
            intents = list(attempt.external_intents)
            intents[index] = settled
            updated_attempt = replace(attempt, external_intents=tuple(intents))
            updated = replace(
                current,
                updated_at=completed_at,
                revision=current.revision + 1,
                attempts=(*current.attempts[:-1], updated_attempt),
            )
            _replace_job_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def record_external_observation(
        self,
        agent_id: str,
        job_id: str,
        *,
        claim_token: str,
        fencing_epoch: int,
        observation: ExternalObservation,
    ) -> JobRun | None:
        if not isinstance(observation, ExternalObservation):
            raise TypeError("observation must be ExternalObservation")

        def write(connection: sqlite3.Connection) -> JobRun | None:
            loaded = _load_job_row(connection, agent_id, job_id)
            if loaded is None:
                return None
            current, encoded = loaded
            attempt = current.current_attempt
            if (
                attempt is None
                or attempt.status is not JobAttemptStatus.CLAIMED
                or attempt.claim_token != claim_token
                or attempt.fencing_epoch != fencing_epoch
                or current.fencing_epoch != fencing_epoch
                or observation.sequence != len(attempt.external_observations) + 1
            ):
                return None
            updated_attempt = replace(
                attempt,
                external_observations=(
                    *attempt.external_observations,
                    observation,
                ),
            )
            updated = replace(
                current,
                updated_at=observation.observed_at,
                revision=current.revision + 1,
                attempts=(*current.attempts[:-1], updated_attempt),
            )
            _replace_job_row(connection, encoded, updated)
            return updated

        return await _run_cancellation_safe_transaction(self.path, write)

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration:
        def write() -> SourceRegistration:
            stored = registration
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = _source_state_row(
                    connection,
                    registration.agent_id,
                    registration.id,
                )
                if row is not None:
                    current = _decode_source_state(
                        connection,
                        row_agent_id=row[0],
                        row_source_id=row[1],
                        source_data=row[2],
                        read_scope_data=row[3],
                        update_scope_count=row[4],
                    )
                    if current != stored:
                        raise ValueError(
                            f"source registration already exists: {registration.id}"
                        )
                    return current
                connection.execute(
                    "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                    (stored.agent_id, stored.id, encode_source(stored)),
                )
                if stored.active:
                    scope = SourceReadScope.allow_all(
                        agent_id=stored.agent_id,
                        source_id=stored.id,
                    )
                    connection.execute(
                        """INSERT INTO source_read_scopes(agent_id, source_id, data)
                           VALUES (?, ?, ?)""",
                        (stored.agent_id, stored.id, encode_source_read_scope(scope)),
                    )
                return stored

        return await asyncio.to_thread(write)

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None:
        def read() -> SourceRegistration | None:
            with _connect(self.path) as connection:
                row = _source_state_row(connection, agent_id, source_id)
                return (
                    None
                    if row is None
                    else _decode_source_state(
                        connection,
                        row_agent_id=row[0],
                        row_source_id=row[1],
                        source_data=row[2],
                        read_scope_data=row[3],
                        update_scope_count=row[4],
                    )
                )

        return await asyncio.to_thread(read)

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        def read() -> tuple[SourceRegistration, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT s.agent_id, s.id, s.data, r.data,
                              (SELECT COUNT(*)
                               FROM postgresql_update_scopes AS u
                               WHERE u.agent_id = s.agent_id
                                 AND u.source_id = s.id)
                       FROM sources AS s
                       LEFT JOIN source_read_scopes AS r
                         ON r.agent_id = s.agent_id AND r.source_id = s.id
                       WHERE s.agent_id = ? ORDER BY s.id""",
                    (agent_id,),
                ).fetchall()
                return tuple(
                    _decode_source_state(
                        connection,
                        row_agent_id=row_agent_id,
                        row_source_id=row_source_id,
                        source_data=data,
                        read_scope_data=read_scope_data,
                        update_scope_count=update_scope_count,
                    )
                    for (
                        row_agent_id,
                        row_source_id,
                        data,
                        read_scope_data,
                        update_scope_count,
                    ) in rows
                )

        return await asyncio.to_thread(read)

    async def load_source_read_scope(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceReadScope | None:
        """Load one exact active-source scope; invalid state never becomes all."""

        def read() -> SourceReadScope | None:
            with _connect(self.path) as connection:
                row = _source_state_row(connection, agent_id, source_id)
                if row is None:
                    return None
                registration = _decode_source_state(
                    connection,
                    row_agent_id=row[0],
                    row_source_id=row[1],
                    source_data=row[2],
                    read_scope_data=row[3],
                    update_scope_count=row[4],
                )
                return _decode_owned_read_scope(connection, registration, row[3])

        return await asyncio.to_thread(read)

    async def list_postgresql_update_scopes(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[PostgreSQLUpdateScope, ...]:
        def read() -> tuple[PostgreSQLUpdateScope, ...]:
            with _connect(self.path) as connection:
                source_row = _source_state_row(connection, agent_id, source_id)
                if source_row is None:
                    return ()
                registration = _decode_source_state(
                    connection,
                    row_agent_id=source_row[0],
                    row_source_id=source_row[1],
                    source_data=source_row[2],
                    read_scope_data=source_row[3],
                    update_scope_count=source_row[4],
                )
                if not registration.active:
                    return ()
                if registration.adapter_id != "postgresql":
                    if source_row[4]:
                        raise SourcePermissionStateError(
                            "non-PostgreSQL source retains update scopes"
                        )
                    return ()
                rows = connection.execute(
                    """SELECT resource_id, authorization_fingerprint, data
                       FROM postgresql_update_scopes
                       WHERE agent_id = ? AND source_id = ?
                       ORDER BY resource_id""",
                    (agent_id, source_id),
                ).fetchall()
                try:
                    return tuple(
                        decode_postgresql_update_scope(
                            data,
                            agent_id=agent_id,
                            source_id=source_id,
                            resource_id=resource_id,
                            authorization_fingerprint=fingerprint,
                        )
                        for resource_id, fingerprint, data in rows
                    )
                except (TypeError, ValueError):
                    raise SourcePermissionStateError(
                        "stored PostgreSQL update scope is undecodable"
                    ) from None

        return await asyncio.to_thread(read)

    async def replace_source_permission_scopes(
        self,
        read_scope: SourceReadScope,
        update_scopes: tuple[PostgreSQLUpdateScope, ...],
    ) -> SourceRegistration:
        """Atomically replace only the two narrow scope families for one source."""

        if not isinstance(read_scope, SourceReadScope):
            raise TypeError("read_scope must be a SourceReadScope")
        if not isinstance(update_scopes, tuple) or any(
            not isinstance(scope, PostgreSQLUpdateScope) for scope in update_scopes
        ):
            raise TypeError("update_scopes must be a tuple of PostgreSQLUpdateScope")
        if len({scope.resource_id for scope in update_scopes}) != len(update_scopes):
            raise ValueError("update_scopes cannot contain duplicate resources")
        gate = _CatalogCommitGate()

        def write() -> SourceRegistration | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                row = _source_state_row(
                    connection,
                    read_scope.agent_id,
                    read_scope.source_id,
                )
                if row is None:
                    raise ValueError("permission scopes require an active owned source")
                registration = _decode_source_state(
                    connection,
                    row_agent_id=row[0],
                    row_source_id=row[1],
                    source_data=row[2],
                    read_scope_data=row[3],
                    update_scope_count=row[4],
                )
                if not registration.active:
                    raise ValueError("permission scopes require an active owned source")
                _require_nonforeign_read_resources(connection, read_scope)
                if any(
                    scope.agent_id != read_scope.agent_id
                    or scope.source_id != read_scope.source_id
                    for scope in update_scopes
                ):
                    raise ValueError("update scope belongs to another source")
                update_resource_ids = {scope.resource_id for scope in update_scopes}
                if read_scope.mode is SourceReadMode.NONE and update_resource_ids:
                    raise ValueError("PostgreSQL update scope requires read access")
                if (
                    read_scope.mode is SourceReadMode.SELECTED
                    and not update_resource_ids <= set(read_scope.resource_ids)
                ):
                    raise ValueError("PostgreSQL update scope must be a read subset")
                if update_scopes and registration.adapter_id != "postgresql":
                    raise ValueError(
                        "PostgreSQL update scopes require a PostgreSQL source"
                    )

                snapshot = _current_source_snapshot(
                    connection,
                    read_scope.agent_id,
                    read_scope.source_id,
                )
                resources = (
                    {}
                    if snapshot is None
                    else {item.id: item for item in snapshot.resources}
                )
                facets = (
                    {}
                    if snapshot is None
                    else {
                        item.resource_id: item
                        for item in snapshot.facets
                        if item.kind is FacetKind.TABULAR
                    }
                )
                for scope in update_scopes:
                    resource = resources.get(scope.resource_id)
                    facet = facets.get(scope.resource_id)
                    if resource is None or facet is None:
                        raise ValueError(
                            "PostgreSQL update scope requires a current table resource"
                        )
                    expected = postgresql_update_authorization_fingerprint(
                        source=registration,
                        resource=resource,
                        facet=facet,
                        allowed_assignment_columns=scope.allowed_assignment_columns,
                    )
                    if scope.authorization_fingerprint != expected:
                        raise ValueError(
                            "PostgreSQL update authorization fingerprint is stale"
                        )

                connection.execute(
                    """INSERT INTO source_read_scopes(agent_id, source_id, data)
                       VALUES (?, ?, ?)
                       ON CONFLICT(agent_id, source_id)
                       DO UPDATE SET data = excluded.data""",
                    (
                        read_scope.agent_id,
                        read_scope.source_id,
                        encode_source_read_scope(read_scope),
                    ),
                )
                connection.execute(
                    """DELETE FROM postgresql_update_scopes
                       WHERE agent_id = ? AND source_id = ?""",
                    (read_scope.agent_id, read_scope.source_id),
                )
                for scope in sorted(update_scopes, key=lambda item: item.resource_id):
                    connection.execute(
                        """INSERT INTO postgresql_update_scopes(
                               agent_id, source_id, resource_id,
                               authorization_fingerprint, data
                           ) VALUES (?, ?, ?, ?, ?)""",
                        (
                            scope.agent_id,
                            scope.source_id,
                            scope.resource_id,
                            scope.authorization_fingerprint,
                            encode_postgresql_update_scope(scope),
                        ),
                    )
                connection.commit()
                return registration
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
        updated = worker.result()
        if cancelled_before_start:
            if updated is not None:
                raise AssertionError(
                    "cancelled source permission transaction committed"
                )
            raise asyncio.CancelledError
        if updated is None:
            raise AssertionError(
                "source permission transaction stopped without cancellation"
            )
        return updated

    async def load_database_write_receipt(
        self,
        agent_id: str,
        receipt_id: str,
    ) -> DatabaseWriteReceipt | None:
        _database_write_text(agent_id, "receipt agent_id")
        validate_database_write_receipt_id(receipt_id)

        def read() -> DatabaseWriteReceipt | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, receipt_id),
                ).fetchone()
            return None if row is None else decode_receipt(row[0])

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
            return None if row is None else decode_receipt(row[0])

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
                        encode_receipt(receipt),
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
                current = decode_receipt(row[0])
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
                    (encode_receipt(receipt), receipt.agent_id, receipt.receipt_id),
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
            source_id = decode_identifier(row[0])
            if not source_id:
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
                row = _source_state_row(connection, agent_id, source_id)
                if row is None:
                    raise ValueError("unknown active source for this agent")
                registration = _decode_source_state(
                    connection,
                    row_agent_id=row[0],
                    row_source_id=row[1],
                    source_data=row[2],
                    read_scope_data=row[3],
                    update_scope_count=row[4],
                )
                if not registration.active:
                    raise ValueError("unknown active source for this agent")
                connection.execute(
                    """INSERT INTO metadata(key, data) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                    (_active_source_key(agent_id), encode_identifier(source_id)),
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
                current = decode_source(row[0])
                detached = (
                    current if not current.active else current.detach(detached_at)
                )
                connection.execute(
                    "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
                    (encode_source(detached), agent_id, source_id),
                )
                connection.execute(
                    """DELETE FROM source_read_scopes
                       WHERE agent_id = ? AND source_id = ?""",
                    (agent_id, source_id),
                )
                connection.execute(
                    """DELETE FROM postgresql_update_scopes
                       WHERE agent_id = ? AND source_id = ?""",
                    (agent_id, source_id),
                )
                selection = connection.execute(
                    "SELECT data FROM metadata WHERE key = ?",
                    (_active_source_key(agent_id),),
                ).fetchone()
                selected_id = (
                    None if selection is None else decode_identifier(selection[0])
                )
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
                        candidate = decode_source(data)
                        if candidate.active:
                            remaining.append(candidate)
                    if len(remaining) == 1:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(agent_id),
                                encode_identifier(remaining[0].id),
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
                    (
                        sync.agent_id,
                        sync.id,
                        sync.source_id,
                        encode_catalog_sync(sync),
                    ),
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
                    stored_registration = registration
                    row = _source_state_row(
                        connection,
                        stored_registration.agent_id,
                        stored_registration.id,
                    )
                    if row is None:
                        connection.execute(
                            "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                            (
                                stored_registration.agent_id,
                                stored_registration.id,
                                encode_source(stored_registration),
                            ),
                        )
                        if stored_registration.active:
                            attach_scope = SourceReadScope.allow_all(
                                agent_id=stored_registration.agent_id,
                                source_id=stored_registration.id,
                            )
                            connection.execute(
                                """INSERT INTO source_read_scopes(
                                       agent_id, source_id, data
                                   ) VALUES (?, ?, ?)""",
                                (
                                    stored_registration.agent_id,
                                    stored_registration.id,
                                    encode_source_read_scope(attach_scope),
                                ),
                            )
                    else:
                        current = _decode_source_state(
                            connection,
                            row_agent_id=row[0],
                            row_source_id=row[1],
                            source_data=row[2],
                            read_scope_data=row[3],
                            update_scope_count=row[4],
                        )
                        if current != stored_registration:
                            if (
                                current.active
                                or not stored_registration.active
                                or current.adapter_id != stored_registration.adapter_id
                                or current.native_identity
                                != stored_registration.native_identity
                            ):
                                raise ValueError(
                                    "source registration already exists: "
                                    f"{stored_registration.id}"
                                )
                            connection.execute(
                                """UPDATE sources SET data = ?
                                   WHERE agent_id = ? AND id = ?""",
                                (
                                    encode_source(stored_registration),
                                    stored_registration.agent_id,
                                    stored_registration.id,
                                ),
                            )
                            attach_scope = SourceReadScope.allow_all(
                                agent_id=stored_registration.agent_id,
                                source_id=stored_registration.id,
                            )
                            connection.execute(
                                """INSERT INTO source_read_scopes(
                                       agent_id, source_id, data
                                   ) VALUES (?, ?, ?)""",
                                (
                                    stored_registration.agent_id,
                                    stored_registration.id,
                                    encode_source_read_scope(attach_scope),
                                ),
                            )
                    selection = connection.execute(
                        "SELECT 1 FROM metadata WHERE key = ?",
                        (_active_source_key(stored_registration.agent_id),),
                    ).fetchone()
                    if selection is None:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(stored_registration.agent_id),
                                encode_identifier(stored_registration.id),
                            ),
                        )
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.id,
                        sync.source_id,
                        encode_catalog_sync(sync),
                    ),
                )
                connection.execute(
                    """INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, source_id) DO UPDATE SET
                         sync_id = excluded.sync_id, data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.source_id,
                        sync.id,
                        encode_catalog_snapshot(snapshot),
                    ),
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

    async def commit_source_edit(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration,
        replaced_source_id: str,
        replaced_at: datetime,
        read_scope: SourceReadScope,
    ) -> SourceCatalogSnapshot:
        """Atomically hand one active source connection to discovered truth."""

        if not isinstance(registration, SourceRegistration) or not registration.active:
            raise ValueError("source edit requires an active replacement registration")
        if not isinstance(replaced_source_id, str) or not replaced_source_id:
            raise ValueError("replaced_source_id must be a non-empty string")
        sync = snapshot.sync
        if registration.agent_id != sync.agent_id or registration.id != sync.source_id:
            raise ValueError("catalog snapshot and replacement registration disagree")
        if (
            read_scope.agent_id != registration.agent_id
            or read_scope.source_id != registration.id
        ):
            raise ValueError("replacement read scope belongs to another source")
        snapshot_resource_ids = {resource.id for resource in snapshot.resources}
        if (
            read_scope.mode is SourceReadMode.SELECTED
            and not set(read_scope.resource_ids) <= snapshot_resource_ids
        ):
            raise ValueError("replacement read scope is outside discovered catalog")
        gate = _CatalogCommitGate()

        def write() -> SourceCatalogSnapshot | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                replaced_row = _source_state_row(
                    connection,
                    registration.agent_id,
                    replaced_source_id,
                )
                if replaced_row is None:
                    raise ValueError("source edit requires an active owned source")
                replaced = _decode_source_state(
                    connection,
                    row_agent_id=replaced_row[0],
                    row_source_id=replaced_row[1],
                    source_data=replaced_row[2],
                    read_scope_data=replaced_row[3],
                    update_scope_count=replaced_row[4],
                )
                if not replaced.active:
                    raise ValueError("source edit requires an active owned source")
                if replaced.adapter_id != registration.adapter_id:
                    raise ValueError("source edit cannot change source type")

                if registration.id != replaced.id:
                    replacement_row = _source_state_row(
                        connection,
                        registration.agent_id,
                        registration.id,
                    )
                    if replacement_row is None:
                        connection.execute(
                            "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                            (
                                registration.agent_id,
                                registration.id,
                                encode_source(registration),
                            ),
                        )
                    else:
                        existing_replacement = _decode_source_state(
                            connection,
                            row_agent_id=replacement_row[0],
                            row_source_id=replacement_row[1],
                            source_data=replacement_row[2],
                            read_scope_data=replacement_row[3],
                            update_scope_count=replacement_row[4],
                        )
                        if existing_replacement.active:
                            raise ValueError(
                                "replacement connection is already attached"
                            )
                        if (
                            existing_replacement.adapter_id != registration.adapter_id
                            or existing_replacement.native_identity
                            != registration.native_identity
                        ):
                            raise ValueError(
                                "replacement source identity conflicts with stored state"
                            )
                        connection.execute(
                            """UPDATE sources SET data = ?
                               WHERE agent_id = ? AND id = ?""",
                            (
                                encode_source(registration),
                                registration.agent_id,
                                registration.id,
                            ),
                        )
                    detached = replaced.detach(replaced_at)
                    connection.execute(
                        """UPDATE sources SET data = ?
                           WHERE agent_id = ? AND id = ?""",
                        (
                            encode_source(detached),
                            replaced.agent_id,
                            replaced.id,
                        ),
                    )
                else:
                    connection.execute(
                        """UPDATE sources SET data = ?
                           WHERE agent_id = ? AND id = ?""",
                        (
                            encode_source(registration),
                            registration.agent_id,
                            registration.id,
                        ),
                    )

                for source_id in {replaced.id, registration.id}:
                    connection.execute(
                        """DELETE FROM source_read_scopes
                           WHERE agent_id = ? AND source_id = ?""",
                        (registration.agent_id, source_id),
                    )
                    connection.execute(
                        """DELETE FROM postgresql_update_scopes
                           WHERE agent_id = ? AND source_id = ?""",
                        (registration.agent_id, source_id),
                    )
                connection.execute(
                    """INSERT INTO source_read_scopes(agent_id, source_id, data)
                       VALUES (?, ?, ?)""",
                    (
                        read_scope.agent_id,
                        read_scope.source_id,
                        encode_source_read_scope(read_scope),
                    ),
                )
                connection.execute(
                    """INSERT INTO metadata(key, data) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                    (
                        _active_source_key(registration.agent_id),
                        encode_identifier(registration.id),
                    ),
                )
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.id,
                        sync.source_id,
                        encode_catalog_sync(sync),
                    ),
                )
                connection.execute(
                    """INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, source_id) DO UPDATE SET
                         sync_id = excluded.sync_id, data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.source_id,
                        sync.id,
                        encode_catalog_snapshot(snapshot),
                    ),
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
                raise AssertionError("cancelled source edit transaction committed")
            raise asyncio.CancelledError
        if committed is None:
            raise AssertionError("source edit transaction stopped without cancellation")
        async with self._decoded_catalog_snapshot_lock:
            self._evict_decoded_catalog_source(
                registration.agent_id,
                replaced_source_id,
            )
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
            return None if row is None else decode_catalog_sync(row[0])

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
            return tuple(decode_semantic_annotation(data) for (data,) in rows)

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
            return None if row is None else decode_semantic_annotation(row[0])

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
                    row is None or decode_semantic_annotation(row[0]) != annotation
                )
                connection.execute(
                    """INSERT INTO semantic_annotations(agent_id, id, data)
                       VALUES (?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (agent_id, annotation.id, encode_semantic_annotation(annotation)),
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
                        result = decode_loop_exit(result_data)
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
                                run=decode_run_input(input_data),
                                messages=tuple(
                                    decode_message(data) for (data,) in message_rows
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
            return None if row is None else decode_learning_candidate(row[0])

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
                    (agent_id, candidate.id, encode_learning_candidate(candidate)),
                )
                by_id[candidate.id] = candidate
                identities.add(candidate.candidate_identity_sha256)
                inserted.append(candidate)
            connection.execute(
                """INSERT INTO metadata(key, data) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                (
                    _learning_review_stamps_key(agent_id),
                    encode_review_stamps(tuple(merged_stamps)),
                ),
            )
            return tuple(inserted)

        return await _run_cancellation_safe_transaction(self.path, write)

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
                (encode_learning_candidate(candidate), agent_id, candidate.id),
            )
            return candidate

        return await _run_cancellation_safe_transaction(self.path, write)

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
            rejected = replace(
                current,
                status=LearningCandidateStatus.REJECTED,
                updated_at=rejected_at,
                rejection_reason=reason,
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (encode_learning_candidate(rejected), agent_id, candidate_id),
            )
            return rejected

        return await _run_cancellation_safe_transaction(self.path, write)

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
            accepted = replace(
                current,
                status=LearningCandidateStatus.ACCEPTED,
                updated_at=accepted_at,
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (encode_learning_candidate(accepted), agent_id, candidate_id),
            )
            return accepted

        return await _run_cancellation_safe_transaction(self.path, write)

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

        return await _run_cancellation_safe_transaction(self.path, write)

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
                            encode_run_input(run),
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
                    (run_id, int(row[0]), encode_message(message)),
                )

        await asyncio.to_thread(write)

    async def finish(self, result: LoopExit) -> None:
        if result.kind is LoopExitKind.COMPLETED:
            raise ValueError("completed runs require atomic transcript completion")

        def write() -> None:
            with _connect(self.path) as connection:
                cursor = connection.execute(
                    """UPDATE runs SET result = ?
                       WHERE id = ? AND conversation_id = ? AND result IS NULL""",
                    (
                        encode_loop_exit(result),
                        result.run_id,
                        result.conversation_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown run: {result.run_id}")

        await asyncio.to_thread(write)

    async def complete(
        self,
        result: LoopExit,
        final_message: CanonicalMessage,
    ) -> None:
        """Atomically append final assistant text and terminal run state."""

        if result.kind is not LoopExitKind.COMPLETED:
            raise ValueError("atomic transcript completion requires a completed exit")
        if final_message.role is not MessageRole.ASSISTANT:
            raise ValueError("atomic completion requires an assistant message")

        def write() -> None:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                run_row = connection.execute(
                    """SELECT input, result FROM runs
                       WHERE id = ? AND conversation_id = ?""",
                    (result.run_id, result.conversation_id),
                ).fetchone()
                if run_row is None:
                    raise KeyError(f"unknown run: {result.run_id}")
                if run_row[1] is not None:
                    raise ValueError(f"run is already terminal: {result.run_id}")
                rows = connection.execute(
                    """SELECT data FROM messages
                       WHERE run_id = ? ORDER BY position""",
                    (result.run_id,),
                ).fetchall()
                transcript = Transcript(
                    run=decode_run_input(run_row[0]),
                    messages=(
                        *(decode_message(row[0]) for row in rows),
                        final_message,
                    ),
                )
                validate_completed_transcript(transcript, result)
                connection.execute(
                    """INSERT INTO messages(run_id, position, data)
                       VALUES (?, ?, ?)""",
                    (result.run_id, len(rows), encode_message(final_message)),
                )
                cursor = connection.execute(
                    """UPDATE runs SET result = ?
                       WHERE id = ? AND conversation_id = ? AND result IS NULL""",
                    (
                        encode_loop_exit(result),
                        result.run_id,
                        result.conversation_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("run changed during atomic completion")

        await asyncio.to_thread(write)

    async def recover_unfinished_runs(
        self,
        agent_id: str,
        *,
        created_at: datetime,
    ) -> tuple[LoopExit, ...]:
        """Terminalize runs left unfinished by a previously admitted host."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")

        def write() -> tuple[LoopExit, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, conversation_id
                       FROM runs
                       WHERE agent_id = ? AND result IS NULL
                       ORDER BY conversation_id, turn_index""",
                    (agent_id,),
                ).fetchall()
                recovered: list[LoopExit] = []
                for run_id, conversation_id in rows:
                    messages = connection.execute(
                        "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                        (run_id,),
                    ).fetchall()
                    steps = sum(
                        decode_message(row[0]).role is MessageRole.ASSISTANT
                        for row in messages
                    )
                    result = LoopExit(
                        run_id=run_id,
                        conversation_id=conversation_id,
                        kind=LoopExitKind.INTERRUPTED,
                        reason="previous_process_terminated",
                        steps=steps,
                        created_at=created_at,
                    )
                    cursor = connection.execute(
                        "UPDATE runs SET result = ? WHERE id = ? AND result IS NULL",
                        (encode_loop_exit(result), run_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("unfinished run changed during recovery")
                    recovered.append(result)
                return tuple(recovered)

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        recovered = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return recovered

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
                run=decode_run_input(run_row[0]),
                messages=tuple(decode_message(row[0]) for row in rows),
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
            return None if row[0] is None else decode_loop_exit(row[0])

        return await asyncio.to_thread(read)

    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Derive reachable refs from persisted runs and successful jobs."""

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
                job_rows = tuple(
                    connection.execute(
                        "SELECT job_id, data FROM job_runs WHERE agent_id = ?",
                        (agent_id,),
                    )
                )
            refs: dict[str, ArtifactRef] = {}
            for stored_run_id, stored_conversation_id, data in rows:
                message = decode_message(data)
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
            jobs = _decode_job_rows(job_rows, agent_id=agent_id)
            for job in jobs:
                if job.result is None:
                    continue
                if (
                    conversation_id is not None
                    and job.conversation_id != conversation_id
                ):
                    continue
                for ref in job.result.artifact_refs:
                    if run_id is not None and ref.run_id != run_id:
                        continue
                    if ref.conversation_id != job.conversation_id:
                        raise RuntimeError(
                            "stored job artifact reference identity is invalid"
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

    async def list_delivery_artifact_references(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[OutcomeArtifactReference, ...]:
        """Return bounded artifact roots carried by retained logical deliveries."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("run_id must be non-empty text or None")
        if conversation_id is not None and (
            not isinstance(conversation_id, str) or not conversation_id
        ):
            raise ValueError("conversation_id must be non-empty text or None")

        def read() -> tuple[OutcomeArtifactReference, ...]:
            clauses = ["agent_id = ?"]
            parameters: list[object] = [agent_id]
            if conversation_id is not None:
                clauses.append("conversation_id = ?")
                parameters.append(conversation_id)
            with _connect_read_only(self.path) as connection:
                rows = connection.execute(
                    "SELECT delivery_id, data FROM deliveries WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY created_at_us, delivery_id",
                    tuple(parameters),
                ).fetchall()
            references: dict[str, OutcomeArtifactReference] = {}
            for delivery_id, data in rows:
                delivery = decode_delivery(
                    data,
                    agent_id=agent_id,
                    delivery_id=delivery_id,
                )
                for reference in delivery.outcome.artifact_references:
                    if run_id is not None and reference.producing_run_id != run_id:
                        continue
                    current = references.get(reference.artifact_id)
                    if current is not None and current != reference:
                        raise RuntimeError(
                            "stored delivery artifact identity is ambiguous"
                        )
                    references[reference.artifact_id] = reference
            return tuple(
                sorted(
                    references.values(),
                    key=lambda item: (item.producing_run_id, item.artifact_id),
                )
            )

        return await asyncio.to_thread(read)

    async def list_reserved_artifact_ids(
        self,
        agent_id: str,
    ) -> frozenset[tuple[str, str]]:
        """Return exact live job artifact reservations for admission recovery."""

        def read() -> frozenset[tuple[str, str]]:
            with _connect_read_only(self.path) as connection:
                jobs = _decode_job_rows(
                    tuple(
                        connection.execute(
                            "SELECT job_id, data FROM job_runs WHERE agent_id = ?",
                            (agent_id,),
                        )
                    ),
                    agent_id=agent_id,
                )
            return frozenset(
                (attempt.execution_run_id, attempt.reserved_artifact_id)
                for job in jobs
                if job.status in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}
                for attempt in (job.current_attempt,)
                if attempt is not None and attempt.status is JobAttemptStatus.CLAIMED
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
                        run=decode_run_input(input_data),
                        messages=tuple(
                            decode_message(message[0]) for message in message_rows
                        ),
                    )
                    result = (
                        None if result_data is None else decode_loop_exit(result_data)
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
                followup_rows = connection.execute(
                    "SELECT followup_id, data FROM autonomous_followups "
                    "WHERE agent_id = ?",
                    (agent_id,),
                ).fetchall()
                protected_run_ids = {
                    followup.reserved_run_id
                    for followup_id, data in followup_rows
                    for followup in (
                        decode_autonomous_followup(
                            data,
                            agent_id=agent_id,
                            followup_id=followup_id,
                        ),
                    )
                    if followup.reserved_run_id is not None
                    and followup.disposition
                    in {
                        FollowupDisposition.CLAIMED,
                        FollowupDisposition.RUNNING,
                        FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
                        FollowupDisposition.RETRYABLE_FAILED,
                    }
                }
                occurrence_rows = connection.execute(
                    "SELECT occurrence_id, data FROM routine_occurrences "
                    "WHERE agent_id = ?",
                    (agent_id,),
                ).fetchall()
                protected_run_ids.update(
                    occurrence.reserved_run_id
                    for occurrence_id, data in occurrence_rows
                    for occurrence in (
                        decode_routine_occurrence(
                            data,
                            agent_id=agent_id,
                            occurrence_id=occurrence_id,
                        ),
                    )
                    if occurrence.reserved_run_id is not None
                    and occurrence.disposition
                    in {
                        RoutineOccurrenceDisposition.CLAIMED,
                        RoutineOccurrenceDisposition.PRECHECKING,
                        RoutineOccurrenceDisposition.RUNNING,
                        (
                            RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                        ),
                        RoutineOccurrenceDisposition.RETRYABLE,
                    }
                )
                active_routine_conversations = {
                    routine.conversation_id
                    for routine_id, data in connection.execute(
                        "SELECT routine_id, data FROM scheduled_routines "
                        "WHERE agent_id = ?",
                        (agent_id,),
                    )
                    for routine in (
                        decode_scheduled_routine(
                            data,
                            agent_id=agent_id,
                            routine_id=routine_id,
                        ),
                    )
                    if routine.active_occurrence_id is not None
                }
                for conversation_id in active_routine_conversations:
                    anchor = connection.execute(
                        "SELECT id FROM runs "
                        "WHERE agent_id = ? AND conversation_id = ? "
                        "ORDER BY turn_index, id LIMIT 1",
                        (agent_id, conversation_id),
                    ).fetchone()
                    if anchor is not None:
                        protected_run_ids.add(str(anchor[0]))
                run_ids = tuple(
                    str(row[0])
                    for row in connection.execute(
                        "SELECT id FROM runs WHERE agent_id = ?",
                        (agent_id,),
                    )
                    if row[0] not in protected_run_ids
                )
                connection.executemany(
                    "DELETE FROM messages WHERE run_id = ?",
                    ((run_id,) for run_id in run_ids),
                )
                connection.executemany(
                    "DELETE FROM runs WHERE id = ? AND agent_id = ?",
                    ((run_id, agent_id) for run_id in run_ids),
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
                return len(run_ids)
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
            run = decode_run_input(row[0])
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
                    result = decode_loop_exit(result_data)
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
                                run=decode_run_input(input_data),
                                messages=tuple(
                                    decode_message(message[0])
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


def _current_source_snapshot(
    connection: sqlite3.Connection,
    agent_id: str,
    source_id: str,
) -> SourceCatalogSnapshot | None:
    row = connection.execute(
        "SELECT data FROM snapshots WHERE agent_id = ? AND source_id = ?",
        (agent_id, source_id),
    ).fetchone()
    if row is None:
        return None
    snapshot = decode_catalog_snapshot(row[0])
    if snapshot.sync.agent_id != agent_id or snapshot.sync.source_id != source_id:
        raise SourcePermissionStateError("stored catalog snapshot ownership is invalid")
    return snapshot


def _decode_catalog_snapshot(value: str) -> SourceCatalogSnapshot:
    return decode_catalog_snapshot(value)


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
        (annotation_id, decode_semantic_annotation(data), data)
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
    values = tuple(decode_learning_candidate(data) for (data,) in rows)
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
    value = decode_review_stamps(row[0])
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
    candidate = decode_learning_candidate(row[0])
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


async def _run_cancellation_safe_transaction(
    path: Path,
    callback: Callable[[sqlite3.Connection], _T],
) -> _T:
    gate = _CatalogCommitGate()
    cancelled_sentinel = object()

    def write() -> _T | object:
        connection = _connect(path)
        try:
            if not gate.start(connection):
                return cancelled_sentinel
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
        if result is not cancelled_sentinel:
            raise AssertionError("cancelled state transaction committed")
        raise asyncio.CancelledError
    if result is cancelled_sentinel:
        raise AssertionError("state transaction stopped without cancellation")
    return cast(_T, result)


def _validate_current_mcp_binding_bounds(connection: sqlite3.Connection) -> None:
    totals: dict[str, int] = {}
    binding_counts: dict[str, int] = {}
    active_tool_counts: dict[str, int] = {}
    for agent_id, binding_id, data in connection.execute(
        "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
    ):
        binding_counts[agent_id] = binding_counts.get(agent_id, 0) + 1
        if binding_counts[agent_id] > MCP_MAX_BINDINGS_PER_AGENT:
            raise ValueError("stored MCP binding count exceeds its fixed bound")
        encoded_bytes = len(data.encode("utf-8"))
        if encoded_bytes > MCP_MAX_BINDING_CANONICAL_BYTES:
            raise ValueError("stored MCP binding exceeds its byte bound")
        totals[agent_id] = totals.get(agent_id, 0) + encoded_bytes
        if totals[agent_id] > MCP_MAX_AGENT_CATALOG_BYTES:
            raise ValueError("stored MCP agent catalog exceeds its byte bound")
        binding = decode_mcp_binding(
            data,
            agent_id=agent_id,
            binding_id=binding_id,
        )
        if binding.state is MCPBindingState.ACTIVE:
            active_tool_counts[agent_id] = active_tool_counts.get(agent_id, 0) + len(
                binding.tools
            )
            if active_tool_counts[agent_id] > MCP_MAX_ACTIVE_TOOLS_PER_AGENT:
                raise ValueError(
                    "stored active MCP tool catalog exceeds its fixed bound"
                )


def _validate_current_records(connection: sqlite3.Connection) -> None:
    _validate_current_mcp_binding_bounds(connection)
    identity: AgentIdentity | None = None
    active_sources: dict[str, str] = {}
    for key, data in connection.execute("SELECT key, data FROM metadata"):
        if key == "identity":
            if identity is not None:
                raise ValueError("state contains duplicate agent identity")
            identity = decode_identity(data)
        elif isinstance(key, str) and key.startswith(_ACTIVE_SOURCE_KEY_PREFIX):
            agent_id = key.removeprefix(_ACTIVE_SOURCE_KEY_PREFIX)
            active_sources[agent_id] = decode_identifier(data)
        elif isinstance(key, str) and key.startswith(
            _LEARNING_REVIEW_STAMPS_KEY_PREFIX
        ):
            agent_id = key.removeprefix(_LEARNING_REVIEW_STAMPS_KEY_PREFIX)
            if not agent_id:
                raise ValueError("stored learning review owner is invalid")
            decode_review_stamps(data)
        else:
            raise ValueError("state metadata key is unsupported")

    mcp_binding_counts: dict[str, int] = {}
    for agent_id, binding_id, data in connection.execute(
        "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
    ):
        mcp_binding_counts[agent_id] = mcp_binding_counts.get(agent_id, 0) + 1
        if mcp_binding_counts[agent_id] > MCP_MAX_BINDINGS_PER_AGENT:
            raise ValueError("stored MCP binding count exceeds its fixed bound")
        binding = decode_mcp_binding(
            data,
            agent_id=agent_id,
            binding_id=binding_id,
        )
        if binding.agent_id != agent_id or binding.binding_id != binding_id:
            raise ValueError("stored MCP binding ownership is invalid")
        if identity is not None and binding.agent_id != identity.id:
            raise ValueError("stored MCP binding belongs to another agent")

    sources: dict[tuple[str, str], SourceRegistration] = {}
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, id, data FROM sources"
    ):
        registration = decode_source(data)
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
        sources[(agent_id, source_id)] = registration
    if identity is not None and any(agent_id != identity.id for agent_id, _ in sources):
        raise ValueError("stored source belongs to another agent")
    for agent_id, source_id in active_sources.items():
        active_registration = sources.get((agent_id, source_id))
        if active_registration is None or not active_registration.active:
            raise ValueError("stored active source selection is invalid")

    read_scopes: dict[tuple[str, str], SourceReadScope] = {}
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, source_id, data FROM source_read_scopes"
    ):
        key = (agent_id, source_id)
        if key in read_scopes:
            raise ValueError("stored source read scope identity is duplicated")
        read_scopes[key] = decode_source_read_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
        )
    for key, registration in sources.items():
        scope = read_scopes.pop(key, None)
        if registration.active and scope is None:
            raise ValueError("active source is missing its read scope")
        if not registration.active and scope is not None:
            raise ValueError("detached source retains a read scope")
    if read_scopes:
        raise ValueError("stored source read scope is foreign")

    for agent_id, source_id, resource_id, fingerprint, data in connection.execute(
        """SELECT agent_id, source_id, resource_id,
                  authorization_fingerprint, data
           FROM postgresql_update_scopes"""
    ):
        scope_registration = sources.get((agent_id, source_id))
        if (
            scope_registration is None
            or scope_registration.adapter_id != "postgresql"
            or not scope_registration.active
        ):
            raise ValueError("stored PostgreSQL update scope is foreign")
        decode_postgresql_update_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
            resource_id=resource_id,
            authorization_fingerprint=fingerprint,
        )

    syncs: dict[tuple[str, str], CatalogSync] = {}
    for agent_id, sync_id, source_id, data in connection.execute(
        "SELECT agent_id, id, source_id, data FROM syncs"
    ):
        sync = decode_catalog_sync(data)
        if (
            sync.agent_id != agent_id
            or sync.id != sync_id
            or sync.source_id != source_id
            or (agent_id, source_id) not in sources
        ):
            raise ValueError("stored catalog sync ownership is invalid")
        syncs[(agent_id, sync_id)] = sync
    for agent_id, source_id, sync_id, data in connection.execute(
        "SELECT agent_id, source_id, sync_id, data FROM snapshots"
    ):
        snapshot = decode_catalog_snapshot(data)
        if (
            snapshot.sync.agent_id != agent_id
            or snapshot.sync.source_id != source_id
            or snapshot.sync.id != sync_id
            or syncs.get((agent_id, sync_id)) != snapshot.sync
        ):
            raise ValueError("stored catalog snapshot ownership is invalid")

    run_ids: set[str] = set()
    message_positions: dict[str, list[int]] = {}
    for (
        run_id,
        agent_id,
        conversation_id,
        turn_index,
        input_data,
        result,
    ) in connection.execute(
        """SELECT id, agent_id, conversation_id, turn_index, input, result
           FROM runs"""
    ):
        run_input = decode_run_input(input_data)
        if (
            run_input.id != run_id
            or run_input.agent_id != agent_id
            or run_input.conversation_id != conversation_id
            or not isinstance(turn_index, int)
            or isinstance(turn_index, bool)
            or turn_index < 0
        ):
            raise ValueError("stored run ownership is invalid")
        if result is not None:
            try:
                exit_record = decode_loop_exit(result)
            except (TypeError, ValueError):
                pass
            else:
                if (
                    exit_record.run_id != run_id
                    or exit_record.conversation_id != conversation_id
                ):
                    raise ValueError("stored run result ownership is invalid")
        run_ids.add(run_id)
    for run_id, position, data in connection.execute(
        "SELECT run_id, position, data FROM messages ORDER BY run_id, position"
    ):
        if run_id not in run_ids:
            raise ValueError("stored message belongs to an unknown run")
        try:
            decode_message(data)
        except (TypeError, ValueError):
            pass
        message_positions.setdefault(run_id, []).append(position)
    if any(
        positions != list(range(len(positions)))
        for positions in message_positions.values()
    ):
        raise ValueError("stored transcript message positions are not contiguous")

    for agent_id, receipt_id, run_id, call_id, data in connection.execute(
        """SELECT agent_id, id, run_id, call_id, data
           FROM database_write_receipts"""
    ):
        receipt = decode_receipt(data)
        if (
            receipt.agent_id != agent_id
            or receipt.receipt_id != receipt_id
            or receipt.run_id != run_id
            or receipt.call_id != call_id
        ):
            raise ValueError("stored database write receipt ownership is invalid")
        if identity is not None and receipt.agent_id != identity.id:
            raise ValueError("stored database write receipt belongs to another agent")

    for agent_id, annotation_id, data in connection.execute(
        "SELECT agent_id, id, data FROM semantic_annotations"
    ):
        annotation = decode_semantic_annotation(data)
        if annotation.agent_id != agent_id or annotation.id != annotation_id:
            raise ValueError("stored semantic annotation ownership is invalid")
    for agent_id, candidate_id, data in connection.execute(
        "SELECT agent_id, id, data FROM learning_candidates"
    ):
        candidate = decode_learning_candidate(data)
        if candidate.agent_id != agent_id or candidate.id != candidate_id:
            raise ValueError("stored learning candidate ownership is invalid")

    routines: dict[tuple[str, str], ScheduledRoutine] = {}
    routine_counts: dict[str, int] = {}
    active_routine_counts: dict[str, int] = {}
    for (
        agent_id,
        routine_id,
        conversation_id,
        state,
        next_due_at_us,
        data,
    ) in connection.execute(
        """SELECT agent_id, routine_id, conversation_id, state,
                  next_due_at_us, data
           FROM scheduled_routines"""
    ):
        routine = decode_scheduled_routine(
            data,
            agent_id=agent_id,
            routine_id=routine_id,
        )
        if (
            routine.conversation_id != conversation_id
            or routine.state.value != state
            or _datetime_us(routine.next_due_at) != next_due_at_us
        ):
            raise ValueError("stored scheduled routine projection is invalid")
        validate_schedule(routine.schedule)
        if identity is not None and routine.agent_id != identity.id:
            raise ValueError("stored scheduled routine belongs to another agent")
        key = (agent_id, routine_id)
        routines[key] = routine
        routine_counts[agent_id] = routine_counts.get(agent_id, 0) + 1
        if routine.state is RoutineState.ACTIVE:
            active_routine_counts[agent_id] = active_routine_counts.get(agent_id, 0) + 1
    if any(
        value > MAX_SCHEDULED_ROUTINES_PER_AGENT for value in routine_counts.values()
    ):
        raise ValueError("stored routine count exceeds its fixed bound")
    if any(
        value > MAX_ACTIVE_ROUTINES_PER_AGENT
        for value in active_routine_counts.values()
    ):
        raise ValueError("stored active routine count exceeds its fixed bound")

    occurrences: dict[tuple[str, str], RoutineOccurrence] = {}
    occurrence_counts: dict[tuple[str, str], int] = {}
    for (
        agent_id,
        occurrence_id_value,
        routine_id,
        routine_revision,
        slot_key,
        state,
        lease_expires_at_us,
        reserved_run_id,
        data,
    ) in connection.execute(
        """SELECT agent_id, occurrence_id, routine_id, routine_revision,
                  slot_key, state, lease_expires_at_us, reserved_run_id, data
           FROM routine_occurrences"""
    ):
        occurrence = decode_routine_occurrence(
            data,
            agent_id=agent_id,
            occurrence_id=occurrence_id_value,
        )
        owning_routine = routines.get((agent_id, routine_id))
        if (
            owning_routine is None
            or occurrence.routine_id != routine_id
            or occurrence.routine_revision != routine_revision
            or occurrence.routine_revision > owning_routine.revision
            or occurrence.slot_key != slot_key
            or occurrence.disposition.value != state
            or _datetime_us(occurrence.lease_expires_at) != lease_expires_at_us
            or occurrence.reserved_run_id != reserved_run_id
        ):
            raise ValueError("stored routine occurrence projection is invalid")
        occurrences[(agent_id, occurrence_id_value)] = occurrence
        owner_key = (agent_id, routine_id)
        occurrence_counts[owner_key] = occurrence_counts.get(owner_key, 0) + 1
        if occurrence_counts[owner_key] > owning_routine.cumulative_max_occurrences:
            raise ValueError("stored routine occurrence count exceeds its ceiling")
    for routine in routines.values():
        if routine.active_occurrence_id is None:
            continue
        active = occurrences.get((routine.agent_id, routine.active_occurrence_id))
        if active is None or active.routine_id != routine.routine_id:
            raise ValueError("stored routine active occurrence is invalid")

    followups: dict[tuple[str, str], AutonomousFollowup] = {}
    for agent_id, followup_id, job_id, event_id, data in connection.execute(
        """SELECT agent_id, followup_id, job_id, event_id, data
           FROM autonomous_followups"""
    ):
        followup = decode_autonomous_followup(
            data,
            agent_id=agent_id,
            followup_id=followup_id,
        )
        if followup.job_id != job_id or followup.event_id != event_id:
            raise ValueError("stored autonomous follow-up projection is invalid")
        if identity is not None and followup.agent_id != identity.id:
            raise ValueError("stored autonomous follow-up belongs to another agent")
        followups[(agent_id, followup_id)] = followup

    delivery_counts: dict[str, int] = {}
    for (
        agent_id,
        delivery_id,
        conversation_id,
        subject_kind,
        subject_id,
        logical_key,
        target_kind,
        target_fingerprint_value,
        state,
        created_at_us,
        data,
    ) in connection.execute(
        """SELECT agent_id, delivery_id, conversation_id, subject_kind,
                  subject_id, logical_key, target_kind, target_fingerprint,
                  state, created_at_us, data
           FROM deliveries"""
    ):
        delivery = decode_delivery(
            data,
            agent_id=agent_id,
            delivery_id=delivery_id,
        )
        if (
            delivery.conversation_id != conversation_id
            or delivery.subject_kind.value != subject_kind
            or delivery.subject_id != subject_id
            or delivery.logical_key != logical_key
            or target_kind != "conversation_inbox"
            or delivery.target.target_fingerprint != target_fingerprint_value
            or delivery.visibility_state.value != state
            or _datetime_us(delivery.created_at) != created_at_us
        ):
            raise ValueError("stored delivery projection is invalid")
        if identity is not None and delivery.agent_id != identity.id:
            raise ValueError("stored delivery belongs to another agent")
        producer: AutonomousFollowup | RoutineOccurrence | None
        if delivery.subject_kind is DeliverySubjectKind.AUTONOMOUS_FOLLOWUP:
            producer = followups.get((agent_id, subject_id))
        else:
            producer = occurrences.get((agent_id, subject_id))
        producer_references_delivery = producer is not None and (
            producer.delivery_id == delivery_id
            if isinstance(producer, AutonomousFollowup)
            else delivery_id in producer.delivery_ids
        )
        if not producer_references_delivery:
            raise ValueError("stored delivery producer reference is invalid")
        delivery_counts[agent_id] = delivery_counts.get(agent_id, 0) + 1
        if delivery_counts[agent_id] > MAX_DELIVERIES_PER_AGENT:
            raise ValueError("stored delivery count exceeds its fixed bound")


def _logical_state_fingerprint(connection: sqlite3.Connection) -> str:
    objects = tuple(
        connection.execute("""SELECT type, name, tbl_name, sql FROM sqlite_master
               WHERE name NOT LIKE 'sqlite_%'
               ORDER BY type, name""")
    )
    rows = {
        table: tuple(connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
        for table in sorted(table_names(connection))
    }
    material = json.dumps(
        {"objects": objects, "rows": rows},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(material.encode("utf-8")).hexdigest()


def _temporary_state_path(path: Path, label: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.{label}-",
        suffix=".db",
        dir=path.parent,
    )
    os.close(descriptor)
    candidate = Path(raw_path)
    os.chmod(candidate, 0o600)
    return candidate


def _copy_state_database(
    source: Path,
    destination: Path,
    *,
    expected_fingerprint: str,
) -> None:
    with _connect_read_only(source) as source_connection:
        if _logical_state_fingerprint(source_connection) != expected_fingerprint:
            raise RuntimeError("state changed while its upgrade copy was prepared")
        destination_connection = _connect(destination)
        try:
            source_connection.backup(destination_connection)
            destination_connection.commit()
        finally:
            destination_connection.close()
    os.chmod(destination, 0o600)
    with _connect_read_only(destination) as copied_connection:
        if _logical_state_fingerprint(copied_connection) != expected_fingerprint:
            raise RuntimeError("state upgrade copy does not match its source")
        require_healthy(copied_connection)


def _rollback_state_path(
    path: Path,
    *,
    found_revision: str,
    fingerprint: str,
) -> Path:
    revision = re.sub(r"[^A-Za-z0-9._-]+", "-", found_revision).strip("-._")
    if not revision:
        revision = "unknown"
    return path.with_name(f"{path.name}.rollback-{revision[:80]}-{fingerprint[:12]}")


def _fsync_state_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prune_older_rollback_points(path: Path, retained: Path) -> None:
    for candidate in path.parent.glob(f"{path.name}.rollback-*"):
        if candidate == retained:
            continue
        try:
            candidate.unlink()
        except OSError:
            pass


def _upgrade_staged_state(
    path: Path,
    *,
    applied: int,
    source_fingerprint: str,
    found_revision: str,
    upgrade_gate: _UpgradeCommitGate | None,
) -> bool:
    rollback_candidate = _temporary_state_path(path, "rollback")
    staged = _temporary_state_path(path, "upgrade")
    retained_rollback = _rollback_state_path(
        path,
        found_revision=found_revision,
        fingerprint=source_fingerprint,
    )
    published_rollback = False
    activated = False
    try:
        _copy_state_database(
            path,
            rollback_candidate,
            expected_fingerprint=source_fingerprint,
        )
        _copy_state_database(
            rollback_candidate,
            staged,
            expected_fingerprint=source_fingerprint,
        )

        connection = _connect(staged)
        try:
            if upgrade_gate is None:
                connection.execute("BEGIN IMMEDIATE")
            elif not upgrade_gate.start(connection):
                return False
            if inspect_journal(connection) != applied:
                raise RuntimeError("migration journal changed during admission")
            upgrade_journaled(connection, applied)
            require_schema(connection, CURRENT_TABLES)
            require_healthy(connection)
            if upgrade_gate is None:
                connection.commit()
            elif not upgrade_gate.commit(connection):
                return False
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

        with _connect_read_only(staged) as staged_connection:
            if inspect_journal(staged_connection) != len(MIGRATIONS):
                raise RuntimeError("staged state did not reach the current revision")
            _validate_current_records(staged_connection)

        os.chmod(staged, 0o600)
        os.chmod(rollback_candidate, 0o600)
        _fsync_state_file(staged)
        _fsync_state_file(rollback_candidate)

        def activate() -> None:
            nonlocal activated, published_rollback
            if retained_rollback.exists():
                with _connect_read_only(retained_rollback) as existing_rollback:
                    if (
                        _logical_state_fingerprint(existing_rollback)
                        != source_fingerprint
                    ):
                        raise RuntimeError(
                            "existing rollback point has conflicting state"
                        )
                rollback_candidate.unlink()
            else:
                os.replace(rollback_candidate, retained_rollback)
                published_rollback = True
            try:
                os.replace(staged, path)
            except BaseException:
                if published_rollback:
                    retained_rollback.unlink(missing_ok=True)
                    published_rollback = False
                raise
            activated = True

        if upgrade_gate is None:
            activate()
        elif not upgrade_gate.activate(activate):
            return False
        _prune_older_rollback_points(path, retained_rollback)
        return True
    finally:
        rollback_candidate.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
        if published_rollback and not activated:
            retained_rollback.unlink(missing_ok=True)


def _initialize(
    path: Path,
    *,
    upgrade_gate: _UpgradeCommitGate | None = None,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        admitted = _admit_existing_state(path, upgrade_gate=upgrade_gate)
        if admitted:
            os.chmod(path, 0o600)
        return admitted
    with _connect(path) as connection:
        create_current(connection)
    os.chmod(path, 0o600)
    return True


def _admit_existing_state(
    path: Path,
    *,
    upgrade_gate: _UpgradeCommitGate | None = None,
) -> bool:
    try:
        with _connect_read_only(path) as connection:
            applied = inspect_journal(connection)
            if applied == len(MIGRATIONS):
                _validate_current_mcp_binding_bounds(connection)
                return True
            found_revision = MIGRATIONS[applied - 1].migration_id
            source_fingerprint = _logical_state_fingerprint(connection)
    except MigrationJournalNewerError as error:
        raise StateCompatibilityError(
            StateCompatibilityCode.NEWER_REVISION,
            path,
            (
                "This local state was created by a newer Daita release. Install "
                "the same or a newer package. No state was changed."
            ),
            current_revision=CURRENT_REVISION,
            found_revision=error.found_revision,
        ) from None
    except MigrationJournalError as error:
        raise StateCompatibilityError(
            StateCompatibilityCode.REVISION_UNSUPPORTED,
            path,
            (
                "This local state has an unknown, incomplete, reordered, or edited "
                "migration history. No state was changed. Install the matching "
                "Daita release before continuing."
            ),
            current_revision=CURRENT_REVISION,
            found_revision=error.found_revision or "invalid-journal",
        ) from None
    except StateCompatibilityError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise _damaged_state_error(path, None) from None

    try:
        return _upgrade_staged_state(
            path,
            applied=applied,
            source_fingerprint=source_fingerprint,
            found_revision=found_revision,
            upgrade_gate=upgrade_gate,
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise StateCompatibilityError(
            StateCompatibilityCode.UPGRADE_FAILED,
            path,
            (
                "Daita could not update the local state safely. The active "
                "database was not replaced. Reinstall the prior working Daita "
                "package before continuing, then report this upgrade failure."
            ),
            current_revision=CURRENT_REVISION,
            found_revision=found_revision,
        ) from None


def _damaged_state_error(
    path: Path,
    found_revision: str | None,
) -> StateCompatibilityError:
    return StateCompatibilityError(
        StateCompatibilityCode.DAMAGED,
        path,
        (
            "This agent state database is damaged or does not match its declared "
            "Daita revision. No state was changed. Reinstall the matching Daita "
            "release or restore the database through your normal recovery process."
        ),
        current_revision=CURRENT_REVISION,
        found_revision=found_revision,
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
            if (receipt := decode_receipt(data)).outcome is DatabaseWriteOutcome.STARTED
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
                    (
                        encode_receipt(recovered),
                        agent_id,
                        receipt_id,
                        encode_receipt(receipt),
                    ),
                )
                if result.rowcount != 1:
                    raise RuntimeError(
                        "database write receipt changed during startup recovery"
                    )
    except RuntimeError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise _damaged_state_error(path, CURRENT_REVISION) from None


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


__all__ = [
    "DatabaseWriteOutcome",
    "DatabaseWriteReceipt",
    "DatabaseWriteReceiptConflictError",
    "PostgreSQLUpdateScope",
    "SQLiteStateStore",
    "SourcePermissionStateError",
    "SourceReadMode",
    "SourceReadScope",
    "StateCompatibilityCode",
    "StateCompatibilityError",
    "database_write_receipt_id",
    "postgresql_update_authorization_fingerprint",
]
