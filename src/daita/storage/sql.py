"""Shared SQL state operations, validation and atomic domain transitions."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import TypeVar, cast

from .._json import FrozenJsonObject
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
    MAX_ARTIFACT_BYTES_PER_AGENT,
    MAX_ARTIFACT_BYTES_PER_RUN,
    MAX_ARTIFACTS_PER_AGENT,
    MAX_ARTIFACTS_PER_RUN,
    ArtifactError,
    ArtifactRecord,
    ArtifactRef,
    ArtifactState,
    artifact_ref_from_mapping,
)
from ..capabilities import (
    CapabilityGrant,
    EffectEvidenceBasis,
    EffectObservation,
    ExecutionScope,
    TaskAttemptGuard,
)
from ..catalog.models import (
    CATALOG_MAX_LINKS,
    CatalogFacet,
    CatalogLink,
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
    GraphJobDelivery,
    OutcomeArtifactReference,
    distribution_plan_digest,
)
from ..identity import AgentIdentity, AgentIdentityConflictError
from ..jobs.graph.guard import GraphAttemptAuthority
from ..jobs.graph.models import (
    ACTIVE_ATTEMPT_STATES,
    AttemptBudgetReservation,
    AttemptState,
    BudgetAmount,
    BudgetLedger,
    ControlKind,
    ControlState,
    GraphAdmission,
    GraphDesiredState,
    GraphEventPage,
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphMutationRequest,
    GraphState,
    GraphTask,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskResult,
    TaskState,
    canonical_digest,
    reserved_artifact_id,
)
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ToolResultBlock,
)
from ..loop.analysis import AnalysisEvidence, AnalysisTraceCapacityError
from ..loop.models import (
    ConversationRun,
    LoopExit,
    LoopExitKind,
    RunInput,
    Transcript,
    validate_completed_transcript,
)
from ..loop.transcripts import ConversationPredecessor
from ..routines.models import (
    MAX_ACTIVE_ROUTINES_PER_AGENT,
    MAX_ROUTINE_HISTORY_PAGE_SIZE,
    MAX_ROUTINE_LIST_PAGE_SIZE,
    MAX_SCHEDULED_ROUTINES_PER_AGENT,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineState,
    ScheduledRoutine,
)
from ..routines.schedule import (
    first_slot,
    manual_slot_key,
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
from . import sql_graph as _graph_store, sql_routines as _routine_store
from .sql_connection import (
    SQLConnection,
    SQLDatabase,
    StateIntegrityError,
    required_row,
)
from .sql_routines import (
    _datetime_us,
    _load_routine_occurrence_row,
    _load_routine_row,
    _replace_routine_row,
    _require_effects_unblocked,
)
from .sqlite_codecs import (
    decode_catalog_link,
    decode_catalog_snapshot,
    decode_catalog_sync,
    decode_delivery,
    decode_identity,
    decode_loop_exit,
    decode_mcp_binding,
    decode_message,
    decode_receipt,
    decode_relational_write_scope,
    decode_routine_occurrence,
    decode_run_input,
    decode_scheduled_routine,
    decode_semantic_annotation,
    decode_source,
    decode_source_configuration_for_deletion,
    decode_source_read_scope,
    encode_catalog_link,
    encode_catalog_snapshot,
    encode_catalog_sync,
    encode_delivery,
    encode_identity,
    encode_loop_exit,
    encode_mcp_binding,
    encode_message,
    encode_receipt,
    encode_relational_write_scope,
    encode_run_input,
    encode_scheduled_routine,
    encode_semantic_annotation,
    encode_source,
    encode_source_read_scope,
)
from .sqlite_codecs.analysis import decode_analysis_evidence, encode_analysis_evidence
from .sqlite_codecs.artifacts import (
    decode_artifact_record,
    encode_artifact_record,
)
from .sqlite_codecs.graph import (
    decode_graph_event,
    decode_graph_job,
    decode_graph_job_delivery,
)
from .sqlite_codecs.mcp_bindings import decode_mcp_credential_reference_for_deletion
from .sqlite_records import (
    EffectOutcome,
    EffectReceipt,
    EffectReceiptConflictError,
    EffectResolution,
    EffectResolutionDecision,
    RelationalWriteScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    effect_receipt_aware as _effect_receipt_aware,
    effect_receipt_text as _effect_receipt_text,
    relational_write_authorization_fingerprint,
    validate_effect_receipt_id,
)

_CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH = 64
_T = TypeVar("_T")


def _active_mcp_tool_count(bindings: Iterable[MCPServerBinding]) -> int:
    return sum(
        len(binding.tools)
        for binding in bindings
        if binding.state is MCPBindingState.ACTIVE
    )


def _load_delivery_row(
    connection: SQLConnection,
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


def _model_sensitivity_rank(value) -> int:
    order = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3,
    }
    return order[value.value]


def _decode_owned_read_scope(
    connection: SQLConnection,
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
    connection: SQLConnection,
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
    connection: SQLConnection,
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
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "stored source registration is invalid"
        ) from None
    if registration.agent_id != row_agent_id or registration.id != row_source_id:
        raise SourcePermissionStateError("stored source ownership is invalid")
    _decode_owned_read_scope(connection, registration, read_scope_data)
    if update_scope_count and not registration.active:
        raise SourcePermissionStateError("stored PostgreSQL update scope is foreign")
    return registration


def _source_state_row(
    connection: SQLConnection,
    agent_id: str,
    source_id: str,
) -> tuple[object, ...] | None:
    return connection.execute(
        """SELECT s.agent_id, s.id, s.data, r.data,
                  (SELECT COUNT(*) FROM relational_write_scopes AS u
                   WHERE u.agent_id = s.agent_id AND u.source_id = s.id)
           FROM sources AS s
           LEFT JOIN source_read_scopes AS r
             ON r.agent_id = s.agent_id AND r.source_id = s.id
           WHERE s.agent_id = ? AND s.id = ?""",
        (agent_id, source_id),
    ).fetchone()


class _CatalogCommitGate:
    """Linearize task cancellation against one state mutation transaction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._started = False

    def start(self, connection: SQLConnection) -> bool:
        with self._lock:
            if self._cancelled:
                return False
        connection.begin(write=True)
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


class SQLStateStore:
    """Shared domain operations over an explicitly supplied ``SQLDatabase``.

    The caller admits and owns the database. This class does not provision it,
    acquire execution ownership or close borrowed database resources. Its close
    method clears local caches after the caller has drained active operations.
    """

    def __init__(
        self, database: SQLDatabase, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._database = database
        self._clock = clock or (lambda: datetime.now(UTC))
        self._decoded_catalog_snapshots: dict[
            tuple[str, str, str], SourceCatalogSnapshot
        ] = {}
        self._decoded_catalog_snapshot_lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._decoded_catalog_snapshot_lock:
            self._decoded_catalog_snapshots.clear()

    async def recover_started_effect_receipts(
        self, agent_id: str, *, recovered_at: datetime
    ) -> None:
        """Recover abandoned effects only after acquiring exclusive agent ownership.

        Opening a connection is passive. The admitted host calls this once before
        starting any work, while retaining its writer lock through shutdown.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        recovered_at = _effect_receipt_aware(recovered_at, "receipt recovery time")
        await run_sql_transaction(
            self._database,
            lambda connection: _recover_started_effect_receipts(
                connection, agent_id, recovered_at
            ),
        )

    async def admit_graph(self, admission: GraphAdmission) -> GraphJob:
        if not isinstance(admission, GraphAdmission):
            raise TypeError("graph admission must be GraphAdmission")
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.admit_graph(connection, admission),
        )

    async def admit_replacement_graph(
        self,
        admission: GraphAdmission,
        *,
        replaced_job_id: str,
        replaced_task_id: str,
        control_id: str,
        principal_id: str,
        idempotency_key: str,
        resolved_at: datetime,
        expected_control_digest: str,
        expected_task_revision: int,
    ) -> GraphJob:
        if not isinstance(admission, GraphAdmission):
            raise TypeError("replacement graph admission must be GraphAdmission")
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.admit_replacement_graph(
                connection,
                admission,
                replaced_job_id=replaced_job_id,
                replaced_task_id=replaced_task_id,
                control_id=control_id,
                principal_id=principal_id,
                idempotency_key=idempotency_key,
                resolved_at=resolved_at,
                expected_control_digest=expected_control_digest,
                expected_task_revision=expected_task_revision,
            ),
        )

    async def read_graph_attempt_authority(
        self, agent_id: str, job_id: str, task_id: str, attempt_id: str
    ) -> GraphAttemptAuthority | None:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.read_graph_attempt_authority(
                connection, agent_id, job_id, task_id, attempt_id
            ),
        )

    async def inspect_graph(self, agent_id: str, job_id: str) -> GraphInspection | None:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.inspect_graph(connection, agent_id, job_id),
        )

    async def list_graph_jobs(
        self,
        agent_id: str,
        *,
        states: frozenset[GraphState] = frozenset(),
        limit: int = 50,
    ) -> tuple[GraphJob, ...]:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_graph_jobs(
                connection,
                agent_id,
                states=states,
                limit=limit,
            ),
        )

    async def apply_graph_mutation(
        self, request: GraphMutationRequest
    ) -> GraphMutation:
        if not isinstance(request, GraphMutationRequest):
            raise TypeError("graph mutation must be GraphMutationRequest")
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.apply_mutation(connection, request),
        )

    async def request_graph_cancel(
        self,
        agent_id: str,
        job_id: str,
        *,
        requested_at: datetime,
        requested_by_id: str,
    ) -> GraphJob | None:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.request_cancel(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                requested_at=requested_at,
                requested_by_id=requested_by_id,
            ),
        )

    async def list_ready_graph_tasks(
        self,
        agent_id: str,
        *,
        now: datetime,
        limit: int = 64,
    ) -> tuple[GraphTask, ...]:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_ready_tasks(
                connection, agent_id, now=now, limit=limit
            ),
        )

    async def expire_due_graphs(
        self,
        agent_id: str,
        *,
        expired_at: datetime,
        limit: int = 64,
    ) -> tuple[GraphJob, ...]:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.expire_due_graphs(
                connection,
                agent_id,
                expired_at=expired_at,
                limit=limit,
            ),
        )

    async def list_stale_graph_attempts(
        self,
        agent_id: str,
        *,
        now: datetime,
        limit: int = 64,
    ) -> tuple[TaskAttempt, ...]:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_stale_attempts(
                connection, agent_id, now=now, limit=limit
            ),
        )

    async def list_active_graph_attempts(
        self,
        agent_id: str,
        *,
        limit: int = 64,
    ) -> tuple[TaskAttempt, ...]:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_active_attempts(
                connection, agent_id, limit=limit
            ),
        )

    async def claim_graph_task(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        *,
        attempt_id: str,
        claim_token: str,
        run_id: str,
        executor_id: str,
        claimed_at: datetime,
        lease_seconds: int,
        absolute_deadline_at: datetime,
        budget_reservations: tuple[BudgetAmount, ...],
    ) -> TaskAttempt | None:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.claim_task(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                task_id=task_id,
                attempt_id=attempt_id,
                claim_token=claim_token,
                run_id=run_id,
                executor_id=executor_id,
                claimed_at=claimed_at,
                lease_seconds=lease_seconds,
                absolute_deadline_at=absolute_deadline_at,
                budget_reservations=budget_reservations,
            ),
        )

    async def start_graph_attempt(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        attempt_id: str,
        *,
        claim_token: str,
        fencing_epoch: int,
        started_at: datetime,
    ) -> TaskAttempt | None:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.start_attempt(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                task_id=task_id,
                attempt_id=attempt_id,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
                started_at=started_at,
            ),
        )

    async def heartbeat_graph_attempt(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        attempt_id: str,
        *,
        claim_token: str,
        fencing_epoch: int,
        heartbeat_at: datetime,
        lease_seconds: int = 30,
    ) -> TaskAttempt | None:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.heartbeat_attempt(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                task_id=task_id,
                attempt_id=attempt_id,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
                heartbeat_at=heartbeat_at,
                lease_seconds=lease_seconds,
            ),
        )

    async def checkpoint_graph_attempt(
        self, checkpoint: TaskCheckpoint, *, claim_token: str
    ) -> TaskCheckpoint:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.checkpoint_attempt(
                connection, checkpoint, claim_token=claim_token
            ),
        )

    async def add_graph_comment(
        self,
        comment: TaskComment,
        *,
        attempt_id: str | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskComment:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.add_comment(
                connection,
                comment,
                attempt_id=attempt_id,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
            ),
        )

    async def complete_graph_attempt(
        self,
        result: TaskResult,
        *,
        claim_token: str,
        fencing_epoch: int,
        usage: tuple[BudgetAmount, ...] | None,
    ) -> TaskResult:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.complete_attempt(
                connection,
                result,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
                usage=usage,
            ),
        )

    async def finalize_graph_attempt(
        self,
        result: TaskResult,
        delivery: GraphJobDelivery,
        *,
        claim_token: str,
        fencing_epoch: int,
        usage: tuple[BudgetAmount, ...] | None,
    ) -> TaskResult:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.complete_attempt(
                connection,
                result,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
                usage=usage,
                delivery=delivery,
            ),
        )

    async def list_graph_deliveries(
        self,
        agent_id: str,
        *,
        job_id: str | None = None,
        limit: int = 100,
    ) -> tuple[GraphJobDelivery, ...]:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_graph_deliveries(
                connection,
                agent_id,
                job_id=job_id,
                limit=limit,
            ),
        )

    async def fence_graph_attempt(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        attempt_id: str,
        *,
        fencing_epoch: int,
        fenced_at: datetime,
        requeue: bool,
        reason_code: str,
    ) -> TaskAttempt | None:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.fence_attempt(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                task_id=task_id,
                attempt_id=attempt_id,
                fencing_epoch=fencing_epoch,
                fenced_at=fenced_at,
                requeue=requeue,
                reason_code=reason_code,
            ),
        )

    async def fail_graph_attempt(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        attempt_id: str,
        *,
        claim_token: str,
        fencing_epoch: int,
        failed_at: datetime,
        retryable: bool,
        reason_code: str,
        attempt_state: AttemptState = AttemptState.FAILED,
    ) -> TaskAttempt | None:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.fail_attempt(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                task_id=task_id,
                attempt_id=attempt_id,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
                failed_at=failed_at,
                retryable=retryable,
                reason_code=reason_code,
                attempt_state=attempt_state,
            ),
        )

    async def open_graph_control(
        self,
        control: TaskControl,
        *,
        claim_token: str,
        fencing_epoch: int,
        replan_task: GraphTask | None = None,
        reviewer_task: GraphTask | None = None,
    ) -> TaskControl:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.open_control(
                connection,
                control,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
                replan_task=replan_task,
                reviewer_task=reviewer_task,
            ),
        )

    async def resolve_graph_control(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        state: ControlState,
        resolved_at: datetime,
        resolved_by_kind: str,
        resolved_by_id: str,
        resolution: dict[str, object],
        make_ready: bool,
        expected_control_digest: str | None = None,
        expected_task_revision: int | None = None,
    ) -> TaskControl | None:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.resolve_control(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                task_id=task_id,
                control_id=control_id,
                state=state,
                resolved_at=resolved_at,
                resolved_by_kind=resolved_by_kind,
                resolved_by_id=resolved_by_id,
                resolution=resolution,
                make_ready=make_ready,
                expected_control_digest=expected_control_digest,
                expected_task_revision=expected_task_revision,
            ),
        )

    async def accept_graph_review(
        self,
        *,
        agent_id: str,
        job_id: str,
        subject_task_id: str,
        control_id: str,
        reviewer_task_id: str,
        resolved_at: datetime,
        resolved_by_kind: str,
        resolved_by_id: str,
        rationale: str,
        idempotency_key: str,
        expected_control_digest: str,
        expected_subject_revision: int,
        reviewer_result: TaskResult | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskResult:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.accept_review(
                connection,
                agent_id=agent_id,
                job_id=job_id,
                subject_task_id=subject_task_id,
                control_id=control_id,
                reviewer_task_id=reviewer_task_id,
                resolved_at=resolved_at,
                resolved_by_kind=resolved_by_kind,
                resolved_by_id=resolved_by_id,
                rationale=rationale,
                idempotency_key=idempotency_key,
                expected_control_digest=expected_control_digest,
                expected_subject_revision=expected_subject_revision,
                reviewer_result=reviewer_result,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
            ),
        )

    async def request_graph_review_changes(
        self,
        *,
        changes_control: TaskControl,
        review_control_id: str,
        reviewer_task_id: str,
        resolved_at: datetime,
        resolved_by_kind: str,
        resolved_by_id: str,
        rationale: str,
        idempotency_key: str,
        expected_control_digest: str,
        expected_subject_revision: int,
        reviewer_result: TaskResult | None = None,
        claim_token: str | None = None,
        fencing_epoch: int | None = None,
    ) -> TaskControl:
        return await run_sql_transaction(
            self._database,
            lambda connection: _graph_store.request_review_changes(
                connection,
                changes_control=changes_control,
                review_control_id=review_control_id,
                reviewer_task_id=reviewer_task_id,
                resolved_at=resolved_at,
                resolved_by_kind=resolved_by_kind,
                resolved_by_id=resolved_by_id,
                rationale=rationale,
                idempotency_key=idempotency_key,
                expected_control_digest=expected_control_digest,
                expected_subject_revision=expected_subject_revision,
                reviewer_result=reviewer_result,
                claim_token=claim_token,
                fencing_epoch=fencing_epoch,
            ),
        )

    async def list_graph_events(
        self,
        agent_id: str,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 100,
        task_id: str | None = None,
    ) -> GraphEventPage:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_graph_events(
                connection,
                agent_id,
                job_id,
                after_event_id=after_event_id,
                limit=limit,
                task_id=task_id,
            ),
        )

    async def list_graph_budget_ledgers(
        self, agent_id: str, job_id: str
    ) -> tuple[BudgetLedger, ...]:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_budget_ledgers(
                connection, agent_id, job_id
            ),
        )

    async def list_graph_attempt_reservations(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        attempt_id: str,
    ) -> tuple[AttemptBudgetReservation, ...]:
        return await _run_graph_read(
            self._database,
            lambda connection: _graph_store.list_attempt_reservations(
                connection, agent_id, job_id, task_id, attempt_id
            ),
        )

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity:
        def write(connection: SQLConnection) -> AgentIdentity:
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

        return await run_sql_transaction(self._database, write)

    async def load_identity(self) -> AgentIdentity | None:
        def read() -> AgentIdentity | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
            return None if row is None else decode_identity(row[0])

        return await settled_thread(read)

    async def load_deletion_credential_inventory(
        self,
        agent_id: str,
    ) -> tuple[
        AgentIdentity | None,
        tuple[tuple[str, Mapping[str, object]], ...],
        tuple[str, ...],
    ]:
        """Read identity, source configurations and MCP credential candidates."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")

        def read() -> tuple[
            AgentIdentity | None,
            tuple[tuple[str, Mapping[str, object]], ...],
            tuple[str, ...],
        ]:
            with _connect_read_only(self._database) as connection:
                identity_row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
                rows = connection.execute(
                    """SELECT agent_id, id, data FROM sources
                       WHERE agent_id = ? ORDER BY id""",
                    (agent_id,),
                ).fetchall()
                mcp_rows = (
                    connection.execute(
                        "SELECT data FROM mcp_server_bindings WHERE agent_id = ?",
                        (agent_id,),
                    ).fetchall()
                    if connection.table_exists("mcp_server_bindings")
                    else ()
                )
            identity = (
                None if identity_row is None else decode_identity(identity_row[0])
            )
            sources: list[tuple[str, Mapping[str, object]]] = []
            for row_agent_id, source_id, data in rows:
                if not all(
                    isinstance(item, str) for item in (row_agent_id, source_id, data)
                ):
                    raise ValueError("stored source deletion inventory is invalid")
                sources.append(
                    decode_source_configuration_for_deletion(
                        data,
                        agent_id=agent_id,
                        source_id=source_id,
                    )
                )
            mcp_references: list[str] = []
            for (data,) in mcp_rows:
                reference = decode_mcp_credential_reference_for_deletion(data)
                if reference is not None:
                    mcp_references.append(reference)
            return identity, tuple(sources), tuple(mcp_references)

        return await settled_thread(read)

    async def load_mcp_binding(
        self,
        agent_id: str,
        binding_id: str,
    ) -> MCPServerBinding | None:
        def read() -> MCPServerBinding | None:
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

    async def list_mcp_bindings(
        self,
        agent_id: str,
    ) -> tuple[MCPServerBinding, ...]:
        def read() -> tuple[MCPServerBinding, ...]:
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

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

        def write(connection: SQLConnection) -> MCPServerBinding:
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
                count = required_row(
                    connection.execute(
                        "SELECT COUNT(*) FROM mcp_server_bindings WHERE agent_id = ?",
                        (binding.agent_id,),
                    )
                )[0]
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

        return await run_sql_transaction(self._database, write)

    async def update_mcp_discovery(
        self,
        agent_id: str,
        binding_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> MCPServerBinding:
        """Replace local hints without changing any execution admission or revision."""

        def write(connection: SQLConnection) -> MCPServerBinding:
            row = connection.execute(
                "SELECT data FROM mcp_server_bindings WHERE agent_id = ? AND binding_id = ?",
                (agent_id, binding_id),
            ).fetchone()
            if row is None:
                raise ValueError("MCP binding does not exist")
            current = decode_mcp_binding(
                row[0], agent_id=agent_id, binding_id=binding_id
            )
            updated = replace(
                current, summary=summary, when_to_use=when_to_use, keywords=keywords
            )
            encoded = encode_mcp_binding(updated)
            if len(encoded.encode("utf-8")) > MCP_MAX_BINDING_CANONICAL_BYTES:
                raise ValueError("MCP binding exceeds its byte bound")
            others = connection.execute(
                "SELECT data FROM mcp_server_bindings WHERE agent_id = ? AND binding_id <> ?",
                (agent_id, binding_id),
            )
            if (
                len(encoded.encode("utf-8"))
                + sum(len(item[0].encode("utf-8")) for item in others)
                > MCP_MAX_AGENT_CATALOG_BYTES
            ):
                raise ValueError("MCP agent catalog exceeds its byte bound")
            connection.execute(
                "UPDATE mcp_server_bindings SET data = ? WHERE agent_id = ? AND binding_id = ?",
                (encoded, agent_id, binding_id),
            )
            return updated

        return await run_sql_transaction(self._database, write)

    async def update_source_discovery(
        self,
        agent_id: str,
        source_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> SourceRegistration:
        """Replace local discovery hints while retaining source identity and scopes."""

        def write(connection: SQLConnection) -> SourceRegistration:
            row = _source_state_row(connection, agent_id, source_id)
            if row is None:
                raise ValueError("source does not exist")
            current = _decode_source_state(
                connection,
                row_agent_id=row[0],
                row_source_id=row[1],
                source_data=row[2],
                read_scope_data=row[3],
                update_scope_count=row[4],
            )
            if not current.active:
                raise ValueError("source is not active")
            updated = replace(
                current, summary=summary, when_to_use=when_to_use, keywords=keywords
            )
            connection.execute(
                "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
                (encode_source(updated), agent_id, source_id),
            )
            return updated

        return await run_sql_transaction(self._database, write)

    async def admit_scheduled_routine(
        self,
        routine: ScheduledRoutine,
        *,
        run_immediately: bool = False,
    ) -> ScheduledRoutine:
        """Atomically admit one pristine routine, its first due slot and, when
        asked, one immediate occurrence."""

        if not isinstance(routine, ScheduledRoutine):
            raise TypeError("routine must be ScheduledRoutine")
        if (
            routine.revision != 1
            or routine.active_occurrence_id is not None
            or routine.last_occurrence_id is not None
            or bool(routine.last_delivery_ids)
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
            next_due = first_slot(routine.schedule, not_before=routine.created_at)
            if next_due is None:
                raise ValueError("routine_schedule_passed")
        normalized = replace(routine, next_due_at=next_due)
        encoded = encode_scheduled_routine(normalized)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.admit_routine(
                connection,
                normalized,
                encoded,
                clock=self._clock,
                run_immediately=run_immediately,
            ),
        )

    async def load_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
    ) -> ScheduledRoutine | None:
        def read() -> ScheduledRoutine | None:
            with _connect_read_only(self._database) as connection:
                loaded = _load_routine_row(connection, agent_id, routine_id)
            return None if loaded is None else loaded[0]

        return await settled_thread(read)

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
            with _connect_read_only(self._database) as connection:
                return _routine_store.list_routines(
                    connection, agent_id, states=states, limit=limit
                )

        return await settled_thread(read)

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
            with _connect_read_only(self._database) as connection:
                return _routine_store.list_occurrences(
                    connection, agent_id, routine_id, limit=limit
                )

        return await settled_thread(read)

    async def list_active_routine_occurrences(
        self,
        agent_id: str,
        *,
        limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE,
    ) -> tuple[RoutineOccurrence, ...]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_ROUTINE_HISTORY_PAGE_SIZE
        ):
            raise ValueError("active occurrence limit is outside its bound")

        def read() -> tuple[RoutineOccurrence, ...]:
            with _connect_read_only(self._database) as connection:
                return _routine_store.list_active_occurrences(
                    connection, agent_id, limit=limit
                )

        return await settled_thread(read)

    async def load_routine_occurrences(
        self,
        agent_id: str,
        occurrence_ids: tuple[str, ...],
    ) -> tuple[RoutineOccurrence, ...]:
        if (
            not isinstance(occurrence_ids, tuple)
            or len(occurrence_ids) > MAX_DELIVERY_LIST_PAGE_SIZE
            or any(not isinstance(item, str) or not item for item in occurrence_ids)
        ):
            raise ValueError("occurrence identities are outside their bound")

        def read() -> tuple[RoutineOccurrence, ...]:
            with _connect_read_only(self._database) as connection:
                return _routine_store.load_occurrences(
                    connection, agent_id, occurrence_ids
                )

        return await settled_thread(read)

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
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.revise_routine(
                connection, routine, expected_revision=expected_revision
            ),
        )

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
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.transition_routine(
                connection,
                agent_id,
                routine_id,
                expected_revision=expected_revision,
                state=state,
                transitioned_at=transitioned_at,
            ),
        )

    async def next_routine_deadline(self, agent_id: str) -> datetime | None:
        def read() -> datetime | None:
            with _connect_read_only(self._database) as connection:
                return _routine_store.next_deadline(connection, agent_id)

        return await settled_thread(read)

    async def load_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
    ) -> RoutineOccurrence | None:
        def read() -> RoutineOccurrence | None:
            with _connect_read_only(self._database) as connection:
                loaded = _load_routine_occurrence_row(
                    connection,
                    agent_id,
                    occurrence_id,
                )
            return None if loaded is None else loaded[0]

        return await settled_thread(read)

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

        _datetime_us(expected_due_at)
        _datetime_us(claimed_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.claim_due_occurrence(
                connection,
                agent_id,
                routine_id,
                expected_revision=expected_revision,
                expected_due_at=expected_due_at,
                claimed_at=claimed_at,
                claim_token=claim_token,
            ),
        )

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
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.claim_manual_occurrence(
                connection,
                agent_id,
                routine_id,
                expected_revision=expected_revision,
                slot=slot,
                claimed_at=claimed_at,
                claim_token=claim_token,
            ),
        )

    async def bind_routine_occurrence_run(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        claim_token: str,
        run_id: str,
        execution_scope: ExecutionScope,
        bound_at: datetime,
    ) -> RoutineOccurrence | None:
        if not isinstance(execution_scope, ExecutionScope):
            raise TypeError("routine execution scope is invalid")
        _datetime_us(bound_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.bind_occurrence_run(
                connection,
                agent_id,
                occurrence_id,
                claim_token=claim_token,
                run_id=run_id,
                execution_scope=execution_scope,
                bound_at=bound_at,
            ),
        )

    async def mark_routine_occurrence_run_terminal(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        run_id: str,
        terminal_at: datetime,
    ) -> RoutineOccurrence | None:
        _datetime_us(terminal_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.mark_occurrence_run_terminal(
                connection,
                agent_id,
                occurrence_id,
                run_id=run_id,
                terminal_at=terminal_at,
            ),
        )

    async def finalize_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
        failure_code: str | None = None,
        artifact_references: tuple[OutcomeArtifactReference, ...] = (),
        outcome_contract_failure_code: str | None = None,
    ) -> tuple[RoutineOccurrence, Delivery | None] | None:
        """Converge one occurrence, routine counters, slot, and logical delivery."""

        _datetime_us(finalized_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.finalize_occurrence(
                connection,
                agent_id,
                occurrence_id,
                delivery_id=delivery_id,
                finalized_at=finalized_at,
                failure_code=failure_code,
                artifact_references=artifact_references,
                outcome_contract_failure_code=outcome_contract_failure_code,
            ),
        )

    async def recover_stale_routine_occurrences(
        self,
        agent_id: str,
        *,
        recovered_at: datetime,
        claim_token_factory: Callable[[str], str],
    ) -> tuple[RoutineOccurrence, ...]:
        """Fence stale claims and converge already-terminal reserved runs."""

        recovered_us = _datetime_us(recovered_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.recover_stale_occurrences(
                connection,
                agent_id,
                recovered_at=recovered_at,
                recovered_us=recovered_us,
                claim_token_factory=claim_token_factory,
            ),
        )

    async def load_delivery(
        self,
        agent_id: str,
        delivery_id: str,
    ) -> Delivery | None:
        def read() -> Delivery | None:
            with _connect_read_only(self._database) as connection:
                loaded = _load_delivery_row(connection, agent_id, delivery_id)
                return None if loaded is None else loaded[0]

        return await settled_thread(read)

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
            clauses = ["agent_id = ?", "subject_kind != 'graph_job'"]
            parameters: list[object] = [agent_id]
            if conversation_id is not None:
                clauses.append("conversation_id = ?")
                parameters.append(conversation_id)
            if not include_acknowledged:
                clauses.append("state != ?")
                parameters.append(DeliveryState.ACKNOWLEDGED.value)
            parameters.append(limit)
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

    async def acknowledge_delivery(
        self,
        agent_id: str,
        delivery_id: str,
        *,
        acknowledged_at: datetime,
    ) -> Delivery | None:
        _datetime_us(acknowledged_at)

        def write(connection: SQLConnection) -> Delivery | None:
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

        return await run_sql_transaction(self._database, write)

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration:
        def write(connection: SQLConnection) -> SourceRegistration:
            stored = registration
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

        return await run_sql_transaction(self._database, write)

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None:
        def read() -> SourceRegistration | None:
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        def read() -> tuple[SourceRegistration, ...]:
            with _connect_read_only(self._database) as connection:
                rows = connection.execute(
                    """SELECT s.agent_id, s.id, s.data, r.data,
                              (SELECT COUNT(*)
                               FROM relational_write_scopes AS u
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

        return await settled_thread(read)

    async def load_source_read_scope(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceReadScope | None:
        """Load one exact active-source scope; invalid state never becomes all."""

        def read() -> SourceReadScope | None:
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

    async def list_relational_write_scopes(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[RelationalWriteScope, ...]:
        def read() -> tuple[RelationalWriteScope, ...]:
            with _connect_read_only(self._database) as connection:
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
                rows = connection.execute(
                    """SELECT resource_id, authorization_fingerprint, data
                       FROM relational_write_scopes
                       WHERE agent_id = ? AND source_id = ?
                       ORDER BY resource_id""",
                    (agent_id, source_id),
                ).fetchall()
                try:
                    return tuple(
                        decode_relational_write_scope(
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

        return await settled_thread(read)

    async def replace_source_permission_scopes(
        self,
        read_scope: SourceReadScope,
        update_scopes: tuple[RelationalWriteScope, ...],
    ) -> SourceRegistration:
        """Atomically replace only the two narrow scope families for one source."""

        if not isinstance(read_scope, SourceReadScope):
            raise TypeError("read_scope must be a SourceReadScope")
        if not isinstance(update_scopes, tuple) or any(
            not isinstance(scope, RelationalWriteScope) for scope in update_scopes
        ):
            raise TypeError("update_scopes must be a tuple of RelationalWriteScope")
        if len({scope.resource_id for scope in update_scopes}) != len(update_scopes):
            raise ValueError("update_scopes cannot contain duplicate resources")

        def write(connection: SQLConnection) -> SourceRegistration:
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
                expected = relational_write_authorization_fingerprint(
                    source=registration,
                    resource=resource,
                    facet=facet,
                    scope=scope,
                )
                if scope.authorization_fingerprint != expected:
                    raise ValueError(
                        "PostgreSQL update authorization fingerprint is stale"
                    )
            connection.execute(
                f"""INSERT INTO source_read_scopes(agent_id, source_id, data)
                   VALUES (?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, source_id')})
                   DO UPDATE SET data = excluded.data""",
                (
                    read_scope.agent_id,
                    read_scope.source_id,
                    encode_source_read_scope(read_scope),
                ),
            )
            connection.execute(
                """DELETE FROM relational_write_scopes
                   WHERE agent_id = ? AND source_id = ?""",
                (read_scope.agent_id, read_scope.source_id),
            )
            for scope in sorted(update_scopes, key=lambda item: item.resource_id):
                connection.execute(
                    """INSERT INTO relational_write_scopes(
                           agent_id, source_id, resource_id,
                           authorization_fingerprint, data
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        scope.agent_id,
                        scope.source_id,
                        scope.resource_id,
                        scope.authorization_fingerprint,
                        encode_relational_write_scope(scope),
                    ),
                )
            return registration

        updated = await run_sql_transaction(self._database, write)
        return updated

    async def load_effect_receipt(
        self,
        agent_id: str,
        receipt_id: str,
    ) -> EffectReceipt | None:
        _effect_receipt_text(agent_id, "receipt agent_id")
        validate_effect_receipt_id(receipt_id)

        def read() -> EffectReceipt | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    """SELECT data FROM effect_receipts
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, receipt_id),
                ).fetchone()
            return None if row is None else decode_receipt(row[0])

        return await settled_thread(read)

    async def load_effect_receipt_for_call(
        self,
        agent_id: str,
        run_id: str,
        call_id: str,
    ) -> EffectReceipt | None:
        _effect_receipt_text(agent_id, "receipt agent_id")
        _effect_receipt_text(run_id, "receipt run_id")
        _effect_receipt_text(call_id, "receipt call_id")

        def read() -> EffectReceipt | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    """SELECT data FROM effect_receipts
                       WHERE agent_id = ? AND run_id = ? AND call_id = ?""",
                    (agent_id, run_id, call_id),
                ).fetchone()
            return None if row is None else decode_receipt(row[0])

        return await settled_thread(read)

    async def load_effect_receipt_for_operation(
        self, agent_id: str, operation_key: str
    ) -> EffectReceipt | None:
        _effect_receipt_text(agent_id, "receipt agent")
        if (
            not isinstance(operation_key, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", operation_key) is None
        ):
            raise ValueError("effect operation key is invalid")

        def read() -> EffectReceipt | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT data FROM effect_receipts WHERE agent_id = ? AND operation_key = ?",
                    (agent_id, operation_key),
                ).fetchone()
                return None if row is None else decode_receipt(row[0])

        return await settled_thread(read)

    async def list_effect_receipts(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        routine_id: str | None = None,
        unresolved_only: bool = False,
        limit: int = 20,
        offset: int = 0,
        caller_principal_id: str | None = None,
    ) -> tuple[EffectReceipt, ...]:
        _effect_receipt_text(agent_id, "receipt agent")
        if (
            type(limit) is not int
            or not 1 <= limit <= 50
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("receipt inspection page is outside its bound")
        for value in (run_id, routine_id):
            if value is not None:
                _effect_receipt_text(value, "receipt inspection identity")
        if caller_principal_id is not None:
            _effect_receipt_text(caller_principal_id, "receipt caller principal")
        if not isinstance(unresolved_only, bool):
            raise TypeError("unresolved_only must be boolean")

        def read() -> tuple[EffectReceipt, ...]:
            with _connect_read_only(self._database) as connection:
                if caller_principal_id is None:
                    rows = connection.execute(
                        "SELECT data FROM effect_receipts WHERE agent_id = ? AND (CAST(? AS TEXT) IS NULL OR run_id = ?) AND (CAST(? AS TEXT) IS NULL OR routine_id = ?) AND (? = 0 OR unresolved = 1) ORDER BY id LIMIT ? OFFSET ?",
                        (
                            agent_id,
                            run_id,
                            run_id,
                            routine_id,
                            routine_id,
                            int(unresolved_only),
                            limit,
                            offset,
                        ),
                    )
                else:
                    rows = connection.execute(
                        "SELECT effect_receipts.data FROM effect_receipts "
                        "JOIN runs ON runs.id = effect_receipts.run_id "
                        "AND runs.agent_id = effect_receipts.agent_id "
                        "WHERE effect_receipts.agent_id = ? "
                        f"AND {connection.caller_expression('runs.input')} = ? "
                        "AND (CAST(? AS TEXT) IS NULL OR effect_receipts.run_id = ?) "
                        "AND (CAST(? AS TEXT) IS NULL OR effect_receipts.routine_id = ?) "
                        "AND (? = 0 OR effect_receipts.unresolved = 1) "
                        "ORDER BY effect_receipts.id LIMIT ? OFFSET ?",
                        (
                            agent_id,
                            caller_principal_id,
                            run_id,
                            run_id,
                            routine_id,
                            routine_id,
                            int(unresolved_only),
                            limit,
                            offset,
                        ),
                    )
                return tuple(decode_receipt(row[0]) for row in rows)

        return await settled_thread(read)

    async def require_effects_unblocked(
        self, agent_id: str, *, run_id: str | None = None, routine_id: str | None = None
    ) -> None:
        def read() -> None:
            with _connect_read_only(self._database) as connection:
                _require_effects_unblocked(
                    connection, agent_id, run_id=run_id, routine_id=routine_id
                )

        await settled_thread(read)

    async def start_effect_receipt(
        self,
        receipt: EffectReceipt,
        *,
        grant: CapabilityGrant | None = None,
        task_attempt_guard: TaskAttemptGuard | None = None,
        max_receipts_per_run: int = 64,
    ) -> EffectReceipt:
        if (
            not isinstance(receipt, EffectReceipt)
            or receipt.outcome is not EffectOutcome.STARTED
        ):
            raise ValueError("external execution must begin with a started receipt")
        if (
            type(max_receipts_per_run) is not int
            or not 1 <= max_receipts_per_run <= 256
        ):
            raise ValueError("receipt reservation requires a positive run call bound")

        def write(connection: SQLConnection) -> EffectReceipt:
            identity = connection.execute(
                "SELECT data FROM metadata WHERE key = 'identity'"
            ).fetchone()
            run_row = connection.execute(
                "SELECT input, result FROM runs WHERE agent_id = ? AND id = ?",
                (receipt.agent_id, receipt.run_id),
            ).fetchone()
            if (
                identity is None
                or decode_identity(identity[0]).id != receipt.agent_id
                or run_row is None
                or run_row[1] is not None
            ):
                raise EffectReceiptConflictError(
                    "receipt requires its current owned nonterminal run"
                )
            run = decode_run_input(run_row[0])
            _require_effects_unblocked(
                connection,
                receipt.agent_id,
                run_id=receipt.run_id,
                routine_id=receipt.routine_id,
            )
            existing = connection.execute(
                "SELECT id FROM effect_receipts WHERE agent_id = ? AND (id = ? OR (run_id = ? AND call_id = ?) OR operation_key = ?)",
                (
                    receipt.agent_id,
                    receipt.receipt_id,
                    receipt.run_id,
                    receipt.call_id,
                    receipt.operation_key,
                ),
            ).fetchone()
            if existing is not None:
                raise EffectReceiptConflictError(
                    "execution or operation identity already has a receipt"
                )
            reserved_count = required_row(
                connection.execute(
                    "SELECT COUNT(*) FROM effect_receipts WHERE agent_id = ? AND run_id = ?",
                    (receipt.agent_id, receipt.run_id),
                )
            )[0]
            if reserved_count >= max_receipts_per_run:
                raise EffectReceiptConflictError(
                    "the run receipt reservation bound is exhausted"
                )
            scope = run.start.execution_scope if run.start is not None else None
            graph_binding = None if scope is None else scope.graph_task_binding
            if scope is None:
                if grant is not None or task_attempt_guard is not None:
                    raise EffectReceiptConflictError(
                        "foreground receipt cannot carry machine authorization"
                    )
            elif graph_binding is not None:
                if (
                    receipt.routine_id is not None
                    or task_attempt_guard is None
                    or task_attempt_guard.binding != graph_binding
                    or task_attempt_guard.claim_token == ""
                    or task_attempt_guard.run_id != receipt.run_id
                    or graph_binding.agent_id != receipt.agent_id
                ):
                    raise EffectReceiptConflictError(
                        "graph receipt lacks its exact frozen attempt authorization"
                    )
                # Approved at the time: a grant is only checked when one was used.
                if grant is not None and (
                    grant not in scope.capability_grants
                    or grant.capability_id != receipt.capability_id
                    or grant.domain_owner_id != receipt.domain_owner_id
                    or grant.capability_contract_digest
                    != receipt.capability_contract_digest
                    or grant.grant_digest != receipt.capability_grant_digest
                ):
                    raise EffectReceiptConflictError(
                        "graph receipt does not match its exact capability grant"
                    )
                loaded_job = _graph_store._load_job(
                    connection, receipt.agent_id, graph_binding.job_id
                )
                loaded_task = _graph_store._load_task(
                    connection,
                    receipt.agent_id,
                    graph_binding.job_id,
                    graph_binding.task_id,
                )
                loaded_attempt = _graph_store._load_attempt(
                    connection,
                    receipt.agent_id,
                    graph_binding.job_id,
                    graph_binding.task_id,
                    graph_binding.attempt_id,
                )
                if loaded_job is None or loaded_task is None or loaded_attempt is None:
                    raise EffectReceiptConflictError(
                        "graph receipt attempt is unavailable"
                    )
                job = loaded_job[0]
                task = loaded_task[0]
                attempt = loaded_attempt[0]
                now = self._clock()
                claim_digest = (
                    "sha256:" + sha256(attempt.claim_token.encode("utf-8")).hexdigest()
                )
                if (
                    job.specification.authority.digest
                    != graph_binding.root_authority_digest
                    or job.state not in {GraphState.QUEUED, GraphState.ACTIVE}
                    or job.desired_state is not GraphDesiredState.RUN
                    or job.deadline_at <= now
                    or task.state is not TaskState.RUNNING
                    or task.current_attempt_id != graph_binding.attempt_id
                    or task.task_revision < graph_binding.task_revision
                    or task.task_spec_digest != graph_binding.task_spec_digest
                    or task.task_scope_digest != graph_binding.task_scope_digest
                    or task.fencing_epoch != graph_binding.fencing_epoch
                    or attempt.state not in ACTIVE_ATTEMPT_STATES
                    or attempt.run_id != receipt.run_id
                    or attempt.fencing_epoch != graph_binding.fencing_epoch
                    or attempt.claim_token != task_attempt_guard.claim_token
                    or claim_digest != graph_binding.claim_token_digest
                    or attempt.absolute_deadline_at != graph_binding.task_deadline_at
                    or attempt.absolute_deadline_at <= now
                    or (
                        attempt.lease_expires_at is not None
                        and attempt.lease_expires_at <= now
                    )
                ):
                    raise EffectReceiptConflictError("graph receipt attempt is stale")
                if grant is not None:
                    count = required_row(
                        connection.execute(
                            "SELECT COUNT(*) FROM effect_receipts WHERE agent_id = ? AND job_id = ? AND task_id = ? AND grant_digest = ?",
                            (
                                receipt.agent_id,
                                graph_binding.job_id,
                                graph_binding.task_id,
                                grant.grant_digest,
                            ),
                        )
                    )[0]
                    if count >= grant.max_calls_per_occurrence:
                        raise EffectReceiptConflictError(
                            "the graph grant invocation ceiling is exhausted"
                        )
            else:
                routine_id = receipt.routine_id
                if routine_id is None:
                    raise EffectReceiptConflictError(
                        "machine receipt lacks routine or graph authorization"
                    )
                # Approved at the time: a grant is only checked when one was used.
                if grant is not None and (
                    grant not in scope.capability_grants
                    or grant.capability_id != receipt.capability_id
                    or grant.domain_owner_id != receipt.domain_owner_id
                    or grant.capability_contract_digest
                    != receipt.capability_contract_digest
                    or grant.grant_digest != receipt.capability_grant_digest
                ):
                    raise EffectReceiptConflictError(
                        "receipt lacks its exact frozen scope grant"
                    )
                if (
                    scope.routine_id != receipt.routine_id
                    or scope.routine_revision != receipt.routine_revision
                    or scope.occurrence_id != receipt.occurrence_id
                ):
                    raise EffectReceiptConflictError(
                        "receipt does not match its scope authorization"
                    )
                occurrence = _load_routine_occurrence_row(
                    connection, receipt.agent_id, receipt.occurrence_id or ""
                )
                routine = _load_routine_row(connection, receipt.agent_id, routine_id)
                if (
                    occurrence is None
                    or routine is None
                    or (
                        occurrence[0].execution_scope != scope
                        or occurrence[0].reserved_run_id != receipt.run_id
                        or occurrence[0].routine_revision != receipt.routine_revision
                        or occurrence[0].disposition
                        is not RoutineOccurrenceDisposition.RUNNING
                        or occurrence[0].claim_token is None
                        # The lease fences unbound claims. Once bound, the
                        # immutable run/scope and nonterminal run row fence
                        # dispatch; model reasoning may outlast the claim lease.
                        or routine[0].revision != receipt.routine_revision
                        or routine[0].active_occurrence_id != receipt.occurrence_id
                        or routine[0].state is not RoutineState.ACTIVE
                    )
                ):
                    raise EffectReceiptConflictError(
                        "receipt occurrence claim is no longer current"
                    )
                if grant is not None:
                    count = required_row(
                        connection.execute(
                            "SELECT COUNT(*) FROM effect_receipts WHERE agent_id = ? AND occurrence_id = ? AND grant_digest = ?",
                            (
                                receipt.agent_id,
                                receipt.occurrence_id,
                                grant.grant_digest,
                            ),
                        )
                    )[0]
                    if count >= grant.max_calls_per_occurrence:
                        raise EffectReceiptConflictError(
                            "the grant invocation ceiling is exhausted"
                        )
            connection.execute(
                "INSERT INTO effect_receipts(agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, grant_digest, unresolved, job_id, task_id, task_attempt_id, fencing_epoch, task_spec_digest, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.agent_id,
                    receipt.receipt_id,
                    receipt.run_id,
                    receipt.call_id,
                    receipt.operation_key,
                    receipt.routine_id,
                    receipt.occurrence_id,
                    receipt.capability_grant_digest,
                    None if graph_binding is None else graph_binding.job_id,
                    None if graph_binding is None else graph_binding.task_id,
                    None if graph_binding is None else graph_binding.attempt_id,
                    None if graph_binding is None else graph_binding.fencing_epoch,
                    None if graph_binding is None else graph_binding.task_spec_digest,
                    encode_receipt(receipt),
                ),
            )
            return receipt

        return await run_sql_transaction(self._database, write)

    async def finish_effect_receipt(self, receipt: EffectReceipt) -> EffectReceipt:
        if (
            not isinstance(receipt, EffectReceipt)
            or receipt.outcome is EffectOutcome.STARTED
            or receipt.resolution is not None
        ):
            raise ValueError(
                "finish requires one terminal observation without human resolution"
            )

        def write(connection: SQLConnection) -> EffectReceipt:
            row = connection.execute(
                "SELECT data, job_id, task_id, task_attempt_id, fencing_epoch, task_spec_digest FROM effect_receipts WHERE agent_id = ? AND id = ?",
                (receipt.agent_id, receipt.receipt_id),
            ).fetchone()
            if row is None:
                raise EffectReceiptConflictError("started receipt does not exist")
            current = decode_receipt(row[0])
            if current.outcome is not EffectOutcome.STARTED:
                if replace(current, resolution=None) == receipt:
                    return current
                raise EffectReceiptConflictError("terminal receipt is immutable")
            if current != receipt.as_started():
                raise EffectReceiptConflictError(
                    "terminal receipt does not match its started identity"
                )
            connection.execute(
                "UPDATE effect_receipts SET data = ?, unresolved = ? WHERE agent_id = ? AND id = ?",
                (
                    encode_receipt(receipt),
                    int(receipt.unresolved),
                    receipt.agent_id,
                    receipt.receipt_id,
                ),
            )
            if receipt.unresolved:
                _pause_effect_routine(
                    connection, receipt, receipt.finished_at or self._clock()
                )
                if row[1] is not None:
                    _open_graph_effect_uncertain_control(
                        connection,
                        receipt=receipt,
                        job_id=str(row[1]),
                        task_id=str(row[2]),
                        attempt_id=str(row[3]),
                        fencing_epoch=int(row[4]),
                        task_spec_digest=str(row[5]),
                        opened_at=receipt.finished_at or self._clock(),
                    )
            return receipt

        return await run_sql_transaction(self._database, write)

    async def list_effect_receipts_for_graph_attempt(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        attempt_id: str,
    ) -> tuple[EffectReceipt, ...]:
        def read() -> tuple[EffectReceipt, ...]:
            with _connect_read_only(self._database) as connection:
                rows = connection.execute(
                    "SELECT data FROM effect_receipts WHERE agent_id = ? AND job_id = ? AND task_id = ? AND task_attempt_id = ? ORDER BY id",
                    (agent_id, job_id, task_id, attempt_id),
                )
                return tuple(decode_receipt(row[0]) for row in rows)

        return await settled_thread(read)

    async def reconcile_graph_effect_attempt(
        self,
        agent_id: str,
        job_id: str,
        task_id: str,
        attempt_id: str,
    ) -> bool:
        def write(connection: SQLConnection) -> bool:
            row = connection.execute(
                "SELECT data, fencing_epoch, task_spec_digest FROM effect_receipts WHERE agent_id = ? AND job_id = ? AND task_id = ? AND task_attempt_id = ? ORDER BY id LIMIT 1",
                (agent_id, job_id, task_id, attempt_id),
            ).fetchone()
            if row is None:
                return False
            receipt = decode_receipt(row[0])
            _open_graph_effect_uncertain_control(
                connection,
                receipt=receipt,
                job_id=job_id,
                task_id=task_id,
                attempt_id=attempt_id,
                fencing_epoch=int(row[1]),
                task_spec_digest=str(row[2]),
                opened_at=max(
                    receipt.finished_at or receipt.started_at,
                    self._clock(),
                ),
            )
            return True

        return await run_sql_transaction(self._database, write)

    async def resolve_effect_receipt(
        self, agent_id: str, resolution: EffectResolution
    ) -> EffectReceipt:
        if not isinstance(resolution, EffectResolution):
            raise TypeError("effect recovery requires an exact human resolution")

        def write(connection: SQLConnection) -> EffectReceipt:
            row = connection.execute(
                "SELECT data FROM effect_receipts WHERE agent_id = ? AND id = ?",
                (agent_id, resolution.receipt_id),
            ).fetchone()
            if row is None:
                raise EffectReceiptConflictError(
                    "the exact owned receipt is unavailable"
                )
            receipt = decode_receipt(row[0])
            if receipt.receipt_digest != resolution.receipt_digest:
                raise EffectReceiptConflictError(
                    "the receipt observation changed before recovery"
                )
            if receipt.resolution is not None:
                if receipt.resolution == resolution:
                    return receipt
                raise EffectReceiptConflictError(
                    "an existing human resolution is immutable"
                )
            if receipt.outcome is not EffectOutcome.UNCERTAIN:
                raise EffectReceiptConflictError(
                    "only terminal uncertainty requires recovery"
                )
            resolved = replace(receipt, resolution=resolution)
            connection.execute(
                "UPDATE effect_receipts SET data = ?, unresolved = 0 WHERE agent_id = ? AND id = ?",
                (encode_receipt(resolved), agent_id, receipt.receipt_id),
            )
            if receipt.routine_id is not None:
                loaded = _load_routine_row(connection, agent_id, receipt.routine_id)
                if loaded is None:
                    raise EffectReceiptConflictError(
                        "the producing routine is unavailable"
                    )
                routine, data = loaded
                state = (
                    RoutineState.DISABLED
                    if resolution.decision
                    is EffectResolutionDecision.CLOSE_WITHOUT_RETRY
                    else RoutineState.PAUSED
                )
                _replace_routine_row(
                    connection,
                    data,
                    replace(routine, state=state, updated_at=resolution.resolved_at),
                )
            return resolved

        return await run_sql_transaction(self._database, write)

    async def detach_source(
        self, agent_id: str, source_id: str, detached_at: datetime
    ) -> SourceRegistration:
        def write(connection: SQLConnection) -> SourceRegistration:
            row = connection.execute(
                "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                (agent_id, source_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown source: {source_id}")
            current = decode_source(row[0])
            detached = current if not current.active else current.detach(detached_at)
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
                """DELETE FROM relational_write_scopes
                   WHERE agent_id = ? AND source_id = ?""",
                (agent_id, source_id),
            )
            return detached

        detached = await run_sql_transaction(self._database, write)
        async with self._decoded_catalog_snapshot_lock:
            self._evict_decoded_catalog_source(agent_id, source_id)
        return detached

    async def record_sync(self, sync: CatalogSync) -> CatalogSync:
        def write(connection: SQLConnection) -> CatalogSync:
            connection.execute(
                f"""INSERT INTO syncs(agent_id, id, source_id, data)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
                (
                    sync.agent_id,
                    sync.id,
                    sync.source_id,
                    encode_catalog_sync(sync),
                ),
            )
            return sync

        return await run_sql_transaction(self._database, write)

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

        def write(connection: SQLConnection) -> SourceCatalogSnapshot:
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
            connection.execute(
                f"""INSERT INTO syncs(agent_id, id, source_id, data)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
                (
                    sync.agent_id,
                    sync.id,
                    sync.source_id,
                    encode_catalog_sync(sync),
                ),
            )
            connection.execute(
                f"""INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, source_id')}) DO UPDATE SET
                     sync_id = excluded.sync_id, data = excluded.data""",
                (
                    sync.agent_id,
                    sync.source_id,
                    sync.id,
                    encode_catalog_snapshot(snapshot),
                ),
            )
            return snapshot

        committed = await run_sql_transaction(
            self._database, write, commit=_commit_catalog_transaction
        )
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

        def write(connection: SQLConnection) -> SourceCatalogSnapshot:
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
                        raise ValueError("replacement connection is already attached")
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
                    """DELETE FROM relational_write_scopes
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
                f"""INSERT INTO syncs(agent_id, id, source_id, data)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
                (
                    sync.agent_id,
                    sync.id,
                    sync.source_id,
                    encode_catalog_sync(sync),
                ),
            )
            connection.execute(
                f"""INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, source_id')}) DO UPDATE SET
                     sync_id = excluded.sync_id, data = excluded.data""",
                (
                    sync.agent_id,
                    sync.source_id,
                    sync.id,
                    encode_catalog_snapshot(snapshot),
                ),
            )
            return snapshot

        committed = await run_sql_transaction(
            self._database, write, commit=_commit_catalog_transaction
        )
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
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

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
                current_sync_id = await settled_thread(
                    _current_snapshot_sync_id,
                    self._database,
                    ref.agent_id,
                    ref.source_id,
                )
                if current_sync_id != ref.sync_id:
                    self._decoded_catalog_snapshots.pop(key, None)
                    return None
                return cached

            row = await settled_thread(
                _current_snapshot_row,
                self._database,
                ref.agent_id,
                ref.source_id,
            )
            if row is None or row[0] != ref.sync_id:
                return None
            snapshot = await settled_thread(_decode_catalog_snapshot, row[1])
            if (
                snapshot.sync.agent_id != ref.agent_id
                or snapshot.sync.source_id != ref.source_id
                or snapshot.sync.id != ref.sync_id
            ):
                raise CatalogStoreError(
                    "stored catalog snapshot does not match its exact reference"
                )
            current_sync_id = await settled_thread(
                _current_snapshot_sync_id,
                self._database,
                ref.agent_id,
                ref.source_id,
            )
            if current_sync_id != ref.sync_id:
                return None
            self._publish_decoded_catalog_snapshot(snapshot)
            return snapshot

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None:
        def read() -> CatalogSync | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT data FROM syncs WHERE agent_id = ? AND id = ?",
                    (agent_id, sync_id),
                ).fetchone()
            return None if row is None else decode_catalog_sync(row[0])

        return await settled_thread(read)

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

    async def save_catalog_link(self, link: CatalogLink) -> None:
        """Create or replace one declared link within the per-agent bound."""

        if not isinstance(link, CatalogLink):
            raise TypeError("link must be a CatalogLink")

        def write(connection: SQLConnection) -> None:
            exists = connection.execute(
                "SELECT 1 FROM catalog_links WHERE agent_id = ? AND id = ?",
                (link.agent_id, link.id),
            ).fetchone()
            count = required_row(
                connection.execute(
                    "SELECT COUNT(*) FROM catalog_links WHERE agent_id = ?",
                    (link.agent_id,),
                )
            )[0]
            if exists is None and count >= CATALOG_MAX_LINKS:
                raise CatalogStoreError(
                    f"an agent can declare at most {CATALOG_MAX_LINKS} catalog links"
                )
            connection.execute(
                f"""INSERT INTO catalog_links(agent_id, id, data)
                   VALUES (?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
                (link.agent_id, link.id, encode_catalog_link(link)),
            )

        await run_sql_transaction(self._database, write)

    async def delete_catalog_link(self, agent_id: str, link_id: str) -> bool:
        def write(connection: SQLConnection) -> bool:
            cursor = connection.execute(
                "DELETE FROM catalog_links WHERE agent_id = ? AND id = ?",
                (agent_id, link_id),
            )
            return cursor.rowcount == 1

        return await run_sql_transaction(self._database, write)

    async def list_catalog_links(self, agent_id: str) -> tuple[CatalogLink, ...]:
        def read() -> tuple[CatalogLink, ...]:
            with _connect_read_only(self._database) as connection:
                rows = connection.execute(
                    "SELECT data FROM catalog_links WHERE agent_id = ? ORDER BY id",
                    (agent_id,),
                ).fetchall()
            if len(rows) > CATALOG_MAX_LINKS:
                raise CatalogStoreError("stored catalog link collection exceeds bound")
            return tuple(decode_catalog_link(data) for (data,) in rows)

        return await settled_thread(read)

    async def list_semantic_annotations(
        self,
        agent_id: str,
    ) -> tuple[SemanticAnnotation, ...]:
        """Return one bounded deterministic agent-isolated semantic collection."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")

        def read() -> tuple[SemanticAnnotation, ...]:
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

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
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    """SELECT data FROM semantic_annotations
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, annotation_id),
                ).fetchone()
            return None if row is None else decode_semantic_annotation(row[0])

        return await settled_thread(read)

    async def preflight_semantic_save(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        expected_sha256: str | None,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one exact semantic create or replacement."""

        _validate_semantic_owner(agent_id, annotation)

        def read() -> FrozenJsonObject:
            with _connect_read_only(self._database) as connection:
                return _semantic_save_fingerprint(
                    connection,
                    agent_id,
                    annotation,
                    expected_sha256,
                )

        return await settled_thread(read)

    async def save_semantic_annotation(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        *,
        expected_sha256: str | None = None,
    ) -> bool:
        """Atomically create, digest-replace, or digest-supersede one annotation."""

        _validate_semantic_owner(agent_id, annotation)

        def write(connection: SQLConnection) -> bool:
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
            changed = row is None or decode_semantic_annotation(row[0]) != annotation
            connection.execute(
                f"""INSERT INTO semantic_annotations(agent_id, id, data)
                   VALUES (?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
                (agent_id, annotation.id, encode_semantic_annotation(annotation)),
            )
            return changed

        changed = await run_sql_transaction(self._database, write)
        return changed

    async def preflight_semantic_delete(
        self,
        agent_id: str,
        annotation_id: str,
        expected_sha256: str,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one digest-protected semantic deletion."""

        def read() -> FrozenJsonObject:
            with _connect_read_only(self._database) as connection:
                return _semantic_delete_fingerprint(
                    connection,
                    agent_id,
                    annotation_id,
                    expected_sha256,
                )

        return await settled_thread(read)

    async def delete_semantic_annotation(
        self,
        agent_id: str,
        annotation_id: str,
        *,
        expected_sha256: str,
    ) -> bool:
        """Atomically delete one annotation only when its rendered digest matches."""

        def write(connection: SQLConnection) -> bool:
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
            return True

        deleted = await run_sql_transaction(self._database, write)
        return deleted

    _UNSPECIFIED_PREDECESSOR = object()

    async def start(
        self,
        run: RunInput,
        *,
        predecessor=_UNSPECIFIED_PREDECESSOR,
    ) -> Transcript:
        if run.conversation_id is None:
            raise ValueError("run conversation_id must be resolved before persistence")

        def write(connection: SQLConnection) -> Transcript:
            try:
                row = connection.execute(
                    """SELECT id, turn_index, input, result
                           FROM runs
                           WHERE agent_id = ? AND conversation_id = ?
                           ORDER BY turn_index DESC
                           LIMIT 1""",
                    (run.agent_id, run.conversation_id),
                ).fetchone()
                if predecessor is self._UNSPECIFIED_PREDECESSOR:
                    turn_index = 0 if row is None else int(row[1]) + 1
                elif predecessor is None:
                    if row is not None:
                        raise ValueError(
                            "conversation predecessor changed before run start"
                        )
                    turn_index = 0
                else:
                    if not isinstance(predecessor, ConversationPredecessor):
                        raise TypeError("conversation predecessor is invalid")
                    if (
                        row is None
                        or row[0] != predecessor.run_id
                        or int(row[1]) != predecessor.turn_index
                        or row[3] is None
                    ):
                        raise ValueError(
                            "conversation predecessor changed before run start"
                        )
                    current_predecessor = ConversationPredecessor.from_run(
                        ConversationRun(
                            turn_index=int(row[1]),
                            transcript=Transcript(run=decode_run_input(row[2])),
                            result=decode_loop_exit(row[3]),
                        )
                    )
                    if current_predecessor != predecessor:
                        raise ValueError(
                            "conversation predecessor changed before run start"
                        )
                    turn_index = predecessor.turn_index + 1
                connection.execute(
                    """INSERT INTO runs(
                               id, agent_id, conversation_id, turn_index, input
                           ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        run.id,
                        run.agent_id,
                        run.conversation_id,
                        turn_index,
                        encode_run_input(run),
                    ),
                )
            except StateIntegrityError as error:
                raise ValueError(f"run already exists: {run.id}") from error
            return Transcript(run=run)

        return await run_sql_transaction(self._database, write)

    async def append(self, run_id: str, message: CanonicalMessage) -> None:
        def position() -> int:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                assert row is not None
                return int(row[0])

        await self.append_at(run_id, await settled_thread(position), message)

    async def append_at(
        self,
        run_id: str,
        position: int,
        message: CanonicalMessage,
    ) -> None:
        def write(connection: SQLConnection) -> None:
            run_row = connection.execute(
                "SELECT result FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                raise KeyError(f"unknown run: {run_id}")
            if run_row[0] is not None:
                raise ValueError(f"run is already terminal: {run_id}")
            row = connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            assert row is not None
            if type(position) is not int or int(row[0]) != position:
                raise ValueError("transcript append position is out of order")
            connection.execute(
                "INSERT INTO messages(run_id, position, data) VALUES (?, ?, ?)",
                (run_id, position, encode_message(message)),
            )

        await run_sql_transaction(self._database, write)

    async def finish(self, result: LoopExit) -> None:
        if result.kind is LoopExitKind.COMPLETED:
            raise ValueError("completed runs require atomic transcript completion")

        def write(connection: SQLConnection) -> None:
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

        await run_sql_transaction(self._database, write)

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

        def write(connection: SQLConnection) -> None:
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

        await run_sql_transaction(self._database, write)

    async def recover_unfinished_runs(
        self,
        agent_id: str,
        *,
        created_at: datetime,
    ) -> tuple[LoopExit, ...]:
        """Terminalize runs left unfinished by a previously admitted host."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")

        def write(connection: SQLConnection) -> tuple[LoopExit, ...]:
            rows = connection.execute(
                """SELECT id, conversation_id, input
                   FROM runs
                   WHERE agent_id = ? AND result IS NULL
                   ORDER BY conversation_id, turn_index""",
                (agent_id,),
            ).fetchall()
            recovered: list[LoopExit] = []
            for run_id, conversation_id, run_data in rows:
                run = decode_run_input(run_data)
                messages = connection.execute(
                    "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                    (run_id,),
                ).fetchall()
                decoded = tuple(decode_message(row[0]) for row in messages)
                steps = sum(
                    message.role is MessageRole.ASSISTANT for message in decoded
                )
                refs: dict[str, ArtifactRef] = {}
                sensitivity = max(
                    run.history_sensitivity,
                    # Machine content is bounded by its approved ceiling.
                    # An interrupted foreground request has no persisted
                    # request classification, so unknown content stays private.
                    (
                        run.execution_scope.sensitivity_ceiling
                        if run.execution_scope is not None
                        else ModelSensitivity.RESTRICTED
                    ),
                    key=lambda item: item.routing_rank,
                )
                for message in decoded:
                    if message.role is not MessageRole.TOOL:
                        continue
                    for block in message.content:
                        if not isinstance(block, ToolResultBlock):
                            continue
                        if block.sensitivity is not None:
                            sensitivity = max(
                                sensitivity,
                                block.sensitivity,
                                key=lambda item: item.routing_rank,
                            )
                        value = block.output.get("artifact")
                        if block.is_error or not isinstance(value, Mapping):
                            continue
                        ref = artifact_ref_from_mapping(value)
                        if (
                            ref.run_id != run_id
                            or ref.conversation_id != conversation_id
                            or ref.call_id != block.call_id
                            or (
                                ref.artifact_id in refs and refs[ref.artifact_id] != ref
                            )
                        ):
                            raise RuntimeError(
                                "recovered artifact identity does not match its run"
                            )
                        refs[ref.artifact_id] = ref
                result = LoopExit(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    kind=LoopExitKind.INTERRUPTED,
                    reason="previous_process_terminated",
                    steps=steps,
                    created_at=created_at,
                    sensitivity=sensitivity,
                    artifacts=tuple(refs.values()),
                )
                cursor = connection.execute(
                    "UPDATE runs SET result = ? WHERE id = ? AND result IS NULL",
                    (encode_loop_exit(result), run_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("unfinished run changed during recovery")
                recovered.append(result)
            return tuple(recovered)

        return await run_sql_transaction(self._database, write)

    async def load(self, run_id: str) -> Transcript:
        def read() -> Transcript:
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

    async def result(self, run_id: str) -> LoopExit | None:
        def read() -> LoopExit | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT result FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            return None if row[0] is None else decode_loop_exit(row[0])

        return await settled_thread(read)

    async def get_artifact_record(self, artifact_id: str) -> ArtifactRecord | None:
        return await _run_graph_read(
            self._database,
            lambda connection: _get_artifact_record(connection, artifact_id),
        )

    async def list_artifact_records(
        self,
        agent_id: str,
        *,
        state: ArtifactState | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        caller_principal_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        after: tuple[datetime, str] | None = None,
        newest_first: bool = False,
    ) -> tuple[ArtifactRecord, ...]:
        if not isinstance(newest_first, bool):
            raise ValueError("artifact newest_first must be a boolean")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 1000):
            raise ValueError("artifact page limit must be between 1 and 1000")
        if type(offset) is not int or offset < 0:
            raise ValueError("artifact offset must be non-negative")
        if after is not None:
            if newest_first:
                raise ValueError("artifact cursor requires creation order")
            if offset or limit is None:
                raise ValueError(
                    "artifact cursor requires a bounded page without offset"
                )
            if (
                not isinstance(after, tuple)
                or len(after) != 2
                or not isinstance(after[0], datetime)
                or after[0].tzinfo is None
                or not isinstance(after[1], str)
                or not after[1]
            ):
                raise ValueError("artifact cursor must contain timestamp and identity")

        def read(connection: SQLConnection) -> tuple[ArtifactRecord, ...]:
            clauses = ["agent_id = ?"]
            parameters: list[object] = [agent_id]
            for column, value in (
                ("state", state.value if state is not None else None),
                ("run_id", run_id),
                ("conversation_id", conversation_id),
                ("caller_principal_id", caller_principal_id),
            ):
                if value is not None:
                    clauses.append(column + " = ?")
                    parameters.append(value)
            if after is not None:
                clauses.append("(created_at_us, artifact_id) > (?, ?)")
                parameters.extend((_datetime_us(after[0]), after[1]))
            return tuple(
                _artifact_record_from_row(row)
                for row in connection.execute(
                    "SELECT artifact_id, agent_id, run_id, conversation_id, caller_principal_id, state, byte_size, created_at_us, data FROM artifacts WHERE "
                    + " AND ".join(clauses)
                    + (
                        " ORDER BY created_at_us DESC, artifact_id DESC LIMIT ? OFFSET ?"
                        if newest_first
                        else " ORDER BY created_at_us, artifact_id LIMIT ? OFFSET ?"
                    ),
                    (
                        *parameters,
                        limit if limit is not None else 9223372036854775807,
                        offset,
                    ),
                )
            )

        return await _run_graph_read(self._database, read)

    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
        caller_principal_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        after: tuple[datetime, str] | None = None,
        newest_first: bool = False,
    ) -> tuple[ArtifactRef, ...]:
        return tuple(
            record.ref
            for record in await self.list_artifact_records(
                agent_id,
                state=ArtifactState.READY,
                run_id=run_id,
                conversation_id=conversation_id,
                caller_principal_id=caller_principal_id,
                limit=limit,
                offset=offset,
                after=after,
                newest_first=newest_first,
            )
        )

    async def begin_artifact_creation(
        self, record: ArtifactRecord, *, reserved: bool = False
    ) -> None:
        if record.state is not ArtifactState.CREATING:
            raise ValueError("new artifact must be creating")

        def write(connection: SQLConnection) -> None:
            identity_row = connection.execute(
                "SELECT data FROM metadata WHERE key = 'identity'"
            ).fetchone()
            if (
                identity_row is None
                or decode_identity(identity_row[0]).id != record.agent_id
            ):
                raise ArtifactError(
                    "artifact_missing", "Artifact belongs to another agent.", {}
                )
            run_row = connection.execute(
                "SELECT agent_id, input FROM runs WHERE id = ?", (record.ref.run_id,)
            ).fetchone()
            if run_row is not None:
                run = decode_run_input(run_row[1])
                if (
                    run_row[0] != record.agent_id
                    or run.caller_principal_id != record.caller_principal_id
                    or (run.conversation_id or run.id) != record.ref.conversation_id
                ):
                    raise ArtifactError(
                        "artifact_missing", "Artifact producer ownership differs.", {}
                    )
            if _get_artifact_record(connection, record.ref.artifact_id) is not None:
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact identity is already registered.",
                    {"stage": "identity"},
                )
            attempt_row = connection.execute(
                "SELECT a.attempt_id, a.state, j.data FROM job_task_attempts AS a "
                "JOIN job_runs AS j ON j.agent_id = a.agent_id AND j.job_id = a.job_id "
                "WHERE a.agent_id = ? AND a.run_id = ?",
                (record.agent_id, record.ref.run_id),
            ).fetchone()
            if reserved and not _active_artifact_reservation(connection, record):
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact reservation is no longer active.",
                    {"stage": "reservation"},
                )
            if attempt_row is not None:
                job = decode_graph_job(attempt_row[2])
                if (
                    job.specification.principal_id != record.caller_principal_id
                    or job.conversation_id != record.ref.conversation_id
                ):
                    raise ArtifactError(
                        "artifact_missing", "Artifact job ownership differs.", {}
                    )
            rows = tuple(
                connection.execute(
                    "SELECT run_id, byte_size FROM artifacts WHERE agent_id = ?",
                    (record.agent_id,),
                )
            )
            run_rows = tuple(row for row in rows if row[0] == record.ref.run_id)
            for scope, kind, used, added, ceiling in (
                ("agent", "count", len(rows), 1, MAX_ARTIFACTS_PER_AGENT),
                (
                    "agent",
                    "bytes",
                    sum(row[1] for row in rows),
                    record.ref.byte_size,
                    MAX_ARTIFACT_BYTES_PER_AGENT,
                ),
                ("run", "count", len(run_rows), 1, MAX_ARTIFACTS_PER_RUN),
                (
                    "run",
                    "bytes",
                    sum(row[1] for row in run_rows),
                    record.ref.byte_size,
                    MAX_ARTIFACT_BYTES_PER_RUN,
                ),
            ):
                if used + added > ceiling:
                    raise ArtifactError(
                        "artifact_quota_exceeded",
                        "The artifact exceeds a storage limit.",
                        {
                            "scope": scope,
                            "limit_kind": kind,
                            "limit": ceiling,
                            "attempted": used + added,
                        },
                    )
            connection.execute(
                "INSERT INTO artifacts (artifact_id, agent_id, run_id, conversation_id, caller_principal_id, state, byte_size, created_at_us, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _artifact_record_row(record),
            )

        await run_sql_transaction(self._database, write)

    async def transition_artifact(
        self, record: ArtifactRecord, state: ArtifactState
    ) -> ArtifactRecord:
        if not (
            (record.state is ArtifactState.CREATING and state is ArtifactState.READY)
            or (
                record.state is not ArtifactState.DELETING
                and state is ArtifactState.DELETING
            )
        ):
            raise ValueError("artifact lifecycle transition is invalid")
        updated = replace(record, state=state)

        def write(connection: SQLConnection) -> ArtifactRecord:
            if (
                record.state is ArtifactState.READY
                and state is ArtifactState.DELETING
                and _active_artifact_reservation(connection, record)
            ):
                raise ArtifactError(
                    "artifact_busy",
                    "An active job still owns this artifact reservation.",
                    {},
                )
            changed = connection.execute(
                "UPDATE artifacts SET state = ?, data = ? WHERE artifact_id = ? AND data = ?",
                (
                    state.value,
                    encode_artifact_record(updated),
                    record.ref.artifact_id,
                    encode_artifact_record(record),
                ),
            ).rowcount
            if changed != 1:
                raise ArtifactError(
                    "artifact_missing", "The artifact lifecycle changed.", {}
                )
            return updated

        return await run_sql_transaction(self._database, write)

    async def finish_artifact_deletion(self, record: ArtifactRecord) -> None:
        if record.state is not ArtifactState.DELETING:
            raise ValueError("artifact deletion must be pending")

        def write(connection: SQLConnection) -> None:
            if (
                connection.execute(
                    "DELETE FROM artifacts WHERE artifact_id = ? AND data = ?",
                    (record.ref.artifact_id, encode_artifact_record(record)),
                ).rowcount
                != 1
            ):
                raise ArtifactError(
                    "artifact_missing", "The artifact lifecycle changed.", {}
                )

        await run_sql_transaction(self._database, write)

    async def conversation_run_page(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        after_turn_index: int = -1,
        limit: int = 100,
    ) -> tuple[ConversationRun, ...]:
        """Read at most 100 turns after an exclusive stable cursor in one snapshot."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("conversation page limit must be between 1 and 100")
        if type(after_turn_index) is not int or after_turn_index < -1:
            raise ValueError("conversation cursor must be at least -1")
        return await _run_graph_read(
            self._database,
            lambda connection: _conversation_records(
                connection,
                agent_id,
                conversation_id,
                after_turn_index=after_turn_index,
                limit=limit,
            ),
        )

    async def conversation_runs(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        """Compatibility full-history read; interactive callers should use pages."""
        return await _run_graph_read(
            self._database,
            lambda connection: _conversation_records(
                connection, agent_id, conversation_id
            ),
        )

    async def conversation_access(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        caller_principal_id: str | None = None,
    ) -> tuple[bool, bool]:
        """Return existence and whole-conversation caller access without messages."""

        def read(connection: SQLConnection) -> tuple[bool, bool]:
            row = connection.execute(
                f"""SELECT
                    EXISTS(SELECT 1 FROM runs WHERE agent_id = ? AND conversation_id = ?),
                    EXISTS(SELECT 1 FROM runs WHERE agent_id = ? AND conversation_id = ?
                        AND {connection.caller_expression('input')} {connection.distinct_operator} ?)
                """,
                (
                    agent_id,
                    conversation_id,
                    agent_id,
                    conversation_id,
                    caller_principal_id,
                ),
            ).fetchone()
            assert row is not None
            exists = bool(row[0])
            return exists, exists and (caller_principal_id is None or not row[1])

        return await _run_graph_read(self._database, read)

    async def conversation_exists(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> bool:
        """Return one agent-scoped existence fact without loading transcript data."""

        def read() -> bool:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    """SELECT 1 FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       LIMIT 1""",
                    (agent_id, conversation_id),
                ).fetchone()
            return row is not None

        return await settled_thread(read)

    async def record_analysis_evidence(
        self, agent_id: str, evidence: AnalysisEvidence
    ) -> None:
        encoded = encode_analysis_evidence(evidence)

        def write(connection: SQLConnection) -> None:
            row = connection.execute(
                "SELECT agent_id, result FROM runs WHERE id = ?", (evidence.run_id,)
            ).fetchone()
            if row is None or row[0] != agent_id or row[1] is not None:
                raise ValueError("Analysis evidence requires an exact live owned run")
            summary = connection.execute(
                "SELECT data FROM analysis_evidence WHERE run_id = ? AND evidence_id = 'run-summary'",
                (evidence.run_id,),
            ).fetchone()
            trace_records = (
                evidence.facts.get("trace_records", 256)
                if evidence.kind == "run" and evidence.facts["status"] == "admitted"
                else (
                    decode_analysis_evidence(summary[0]).facts.get("trace_records", 256)
                    if summary
                    else 256
                )
            )
            if type(trace_records) is not int or not 4 <= trace_records <= 256:
                raise ValueError(
                    "Analysis trace admission requires a bounded record ceiling"
                )
            existing = connection.execute(
                "SELECT data FROM analysis_evidence WHERE run_id = ? AND evidence_id = ?",
                (evidence.run_id, evidence.evidence_id),
            ).fetchone()
            if existing is not None:
                prior = decode_analysis_evidence(existing[0])
                if evidence.kind == "generation" and existing[0] == encoded:
                    return  # Lost commit acknowledgement: exact settlement only.
                disposal = (
                    prior.kind == evidence.kind == "generation"
                    and prior.facts["status"] == "closed"
                    and dict(evidence.facts) == {**prior.facts, "proof_disposed": True}
                )
                if (
                    prior.kind != evidence.kind
                    or not disposal
                    and prior.facts.get("status")
                    not in {
                        "admitted",
                        "running",
                    }
                ):
                    raise ValueError(
                        "Analysis evidence cannot be replayed or overwritten"
                    )
                for identity in (
                    "parent_call_id",
                    "sequence",
                    "tool_name",
                    "arguments",
                    "code",
                    "scratch",
                    "closure_nonce",
                ):
                    if identity in prior.facts and prior.facts[
                        identity
                    ] != evidence.facts.get(identity):
                        raise ValueError("Analysis evidence identity changed")
            else:
                counted = connection.execute(
                    "SELECT COUNT(*) FROM analysis_evidence WHERE run_id = ?",
                    (evidence.run_id,),
                ).fetchone()
                assert counted is not None
                count = counted[0]
                if count >= trace_records:
                    raise AnalysisTraceCapacityError(
                        "Analysis evidence capacity exhausted"
                    )
            credit = (
                max(len(encoded.encode()), 270_000)
                if evidence.facts.get("status")
                in {"admitted", "running", "cleanup_failed"}
                or evidence.kind == "generation"
                and evidence.facts.get("proof_disposed") is not True
                else len(encoded.encode())
            )
            occupied = connection.execute(
                "SELECT COALESCE(SUM(reserved_bytes), 0) FROM analysis_evidence WHERE run_id = ? AND evidence_id != ?",
                (evidence.run_id, evidence.evidence_id),
            ).fetchone()
            assert occupied is not None
            trace_limit = (
                evidence.facts.get("trace_bytes", 4 * 1024 * 1024)
                if evidence.kind == "run" and evidence.facts["status"] == "admitted"
                else (
                    decode_analysis_evidence(summary[0]).facts.get(
                        "trace_bytes", 4 * 1024 * 1024
                    )
                    if summary
                    else 4 * 1024 * 1024
                )
            )
            if (
                type(trace_limit) is not int
                or not 1024 * 1024 <= trace_limit <= 4 * 1024 * 1024
            ):
                raise ValueError(
                    "Analysis trace admission requires a bounded byte ceiling"
                )
            if occupied[0] + credit > trace_limit:
                raise AnalysisTraceCapacityError(
                    "Analysis trace byte reservation exhausted before dispatch"
                )
            connection.execute(
                "INSERT INTO analysis_evidence(agent_id, run_id, evidence_id, kind, data, reserved_bytes) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, evidence_id) DO UPDATE SET data = excluded.data, reserved_bytes = excluded.reserved_bytes",
                (
                    agent_id,
                    evidence.run_id,
                    evidence.evidence_id,
                    evidence.kind,
                    encoded,
                    credit,
                ),
            )

        await run_sql_transaction(self._database, write)

    async def load_analysis_evidence(
        self, agent_id: str, run_id: str
    ) -> tuple[AnalysisEvidence, ...]:
        def read(connection: SQLConnection) -> tuple[AnalysisEvidence, ...]:
            return tuple(
                decode_analysis_evidence(row[0])
                for row in connection.execute(
                    "SELECT data FROM analysis_evidence WHERE agent_id = ? AND run_id = ? ORDER BY evidence_id LIMIT 256",
                    (agent_id, run_id),
                )
            )

        return await _run_graph_read(self._database, read)

    async def pending_analysis_evidence(
        self, agent_id: str, *, after: tuple[str, str] = ("", "")
    ) -> tuple[AnalysisEvidence, ...]:
        """Page candidate evidence, including terminal-run cleanup, before filtering.

        The last returned identity is the next cursor even for wholly closed pages.
        """

        def read(connection: SQLConnection) -> tuple[AnalysisEvidence, ...]:
            return tuple(
                decode_analysis_evidence(row[0])
                for row in connection.execute(
                    "SELECT data FROM analysis_evidence WHERE agent_id = ? "
                    "AND (run_id > ? OR (run_id = ? AND evidence_id > ?)) "
                    "ORDER BY run_id, evidence_id LIMIT 256",
                    (agent_id, after[0], after[0], after[1]),
                )
            )

        return await _run_graph_read(self._database, read)

    async def recover_analysis_evidence(
        self,
        agent_id: str,
        expected: AnalysisEvidence,
        recovery: Mapping[str, object] | None = None,
        *,
        proof_disposed: bool = False,
    ) -> AnalysisEvidence:
        """Append once-only authenticated cleanup facts, without rewriting history.

        Only composition's native recovery calls this operation. Exact encoded CAS
        binds the evidence identity and its retained proof, including terminal runs.
        No native I/O is performed under the transaction.
        """
        before = encode_analysis_evidence(expected)
        facts = dict(expected.facts)
        if recovery is not None:
            if expected.kind != "generation" or facts.get("recovery") is not None:
                raise ValueError("Recovery is once-only native generation evidence")
            if facts["status"] not in {"admitted", "running", "cleanup_failed"}:
                raise ValueError("Recovery requires unresolved generation closure")
            for name in (
                "scratch",
                "scratch_device",
                "scratch_inode",
                "closure_path",
                "closure_device",
                "closure_inode",
                "closure_nonce",
            ):
                if recovery.get(name) != facts.get(name):
                    raise ValueError("Recovery proof identity changed")
            facts["recovery"] = {
                name: recovery[name] for name in ("status", "pid", "usage", "cleanup")
            }
        if proof_disposed:
            proved = facts.get("recovery", facts)
            if expected.kind != "generation" or not isinstance(proved, Mapping):
                raise ValueError("Proof disposal requires generation closure")
            cleanup, usage = proved.get("cleanup"), proved.get("usage")
            if (
                not isinstance(cleanup, Mapping)
                or not isinstance(usage, Mapping)
                or cleanup.get("process_reaped") is not True
                or cleanup.get("scratch_deleted") is not True
                or usage.get("cpu_complete") is not True
            ):
                raise ValueError("Proof disposal requires durable complete closure")
            facts["proof_disposed"] = True
        updated = AnalysisEvidence(
            expected.run_id, expected.evidence_id, expected.kind, facts
        )
        encoded = encode_analysis_evidence(updated)

        def write(connection: SQLConnection) -> None:
            row = connection.execute(
                "SELECT a.data, a.reserved_bytes FROM analysis_evidence a JOIN runs r ON r.id = a.run_id "
                "WHERE a.agent_id = ? AND r.agent_id = ? AND a.run_id = ? AND a.evidence_id = ?",
                (agent_id, agent_id, expected.run_id, expected.evidence_id),
            ).fetchone()
            if row is None or row[0] not in {before, encoded}:
                raise ValueError("Recovery evidence CAS identity changed")
            if len(encoded.encode()) > row[1]:
                raise AnalysisTraceCapacityError(
                    "Recovery exceeded its pre-dispatch reservation"
                )
            connection.execute(
                "UPDATE analysis_evidence SET data = ?, reserved_bytes = ? WHERE agent_id = ? AND run_id = ? AND evidence_id = ? AND data = ?",
                (
                    encoded,
                    len(encoded.encode()) if proof_disposed else row[1],
                    agent_id,
                    expected.run_id,
                    expected.evidence_id,
                    before,
                ),
            )

        await run_sql_transaction(self._database, write)
        return updated

    async def clear_conversations(self, agent_id: str) -> int:
        """Delete transcripts, keeping runs that active work still needs."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")

        def write(connection: SQLConnection) -> int:
            protected_run_ids = {
                str(row[0])
                for row in connection.execute(
                    "SELECT run_id FROM job_task_attempts "
                    "WHERE agent_id = ? AND state IN ('claimed', 'running')",
                    (agent_id,),
                )
            }
            protected_run_ids.update(
                record.run_id
                for row in connection.execute(
                    "SELECT data FROM analysis_evidence WHERE agent_id = ? AND kind = 'generation'",
                    (agent_id,),
                )
                if (record := decode_analysis_evidence(row[0])).facts.get(
                    "proof_disposed"
                )
                is not True
            )
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
                    RoutineOccurrenceDisposition.RUNNING,
                    (RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION),
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
            return len(run_ids)

        cleared = await run_sql_transaction(self._database, write)
        return cleared

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
            with _connect_read_only(self._database) as connection:
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

        return await settled_thread(read)

    async def latest_terminal_conversation_run(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> ConversationRun | None:
        """Return the exact latest terminal revision used for writer CAS binding."""

        def read() -> ConversationRun | None:
            connection = _connect_read_only(self._database)
            try:
                row = connection.execute(
                    """SELECT id, turn_index, input, result
                       FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       ORDER BY turn_index DESC LIMIT 1""",
                    (agent_id, conversation_id),
                ).fetchone()
                if row is None:
                    return None
                run_id, turn_index, input_data, result_data = row
                if result_data is None:
                    raise ValueError("latest conversation run is not terminal")
                messages = tuple(
                    decode_message(message_row[0])
                    for message_row in connection.execute(
                        "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                        (run_id,),
                    )
                )
                return ConversationRun(
                    turn_index=int(turn_index),
                    transcript=Transcript(
                        run=decode_run_input(input_data),
                        messages=messages,
                    ),
                    result=decode_loop_exit(result_data),
                )
            finally:
                connection.close()

        return await settled_thread(read)

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
    path: SQLDatabase,
    agent_id: str,
    source_id: str,
) -> str | None:
    with _connect_read_only(path) as connection:
        row = connection.execute(
            "SELECT sync_id FROM snapshots WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchone()
    return None if row is None else str(row[0])


def _current_snapshot_row(
    path: SQLDatabase,
    agent_id: str,
    source_id: str,
) -> tuple[str, str] | None:
    with _connect_read_only(path) as connection:
        row = connection.execute(
            "SELECT sync_id, data FROM snapshots WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1])


def _current_source_snapshot(
    connection: SQLConnection,
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
    connection: SQLConnection,
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
    connection: SQLConnection,
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
    connection: SQLConnection,
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


async def settled_thread(
    function: Callable[..., _T],
    /,
    *arguments: object,
) -> _T:
    """Run bounded state work in a thread that settles before cancellation.

    A cancelled ``asyncio.to_thread`` caller returns while its thread keeps a
    database connection open. Waiting for the thread keeps every connection a
    task opened released before that task finishes, so owners that join their
    tasks before closing storage leave no connection behind. The worker's own
    result or failure is discarded when the caller was cancelled.
    """
    worker = asyncio.ensure_future(asyncio.to_thread(function, *arguments))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not worker.cancelled():
            worker.exception()  # Retrieve it so the loop does not report it.
        raise asyncio.CancelledError
    return worker.result()


async def run_sql_transaction(
    database: SQLDatabase,
    callback: Callable[[SQLConnection], _T],
    *,
    commit: Callable[[SQLConnection], None] | None = None,
) -> _T:
    """Run one short adapter transaction through the shared cancellation owner.

    Cancellation before admission prevents the write; after admission the worker
    settles exactly once and returns its result or storage failure. The callback
    runs in a worker thread and must perform only bounded database work. No model,
    source, artifact or other external I/O belongs inside this transaction.
    ``commit`` permits an explicit commit boundary; it must never replay work.
    """
    gate = _CatalogCommitGate()
    cancelled_sentinel = object()

    def write() -> _T | object:
        connection = _connect(database)
        try:
            if not gate.start(connection):
                return cancelled_sentinel
            result = callback(connection)
            if commit is None:
                connection.commit()
            else:
                commit(connection)
            return result
        except BaseException as error:
            connection.rollback()
            raise database.normalize_error(error) from None
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


def _conversation_records(
    connection: SQLConnection,
    agent_id: str,
    conversation_id: str,
    *,
    after_turn_index: int = -1,
    limit: int | None = None,
) -> tuple[ConversationRun, ...]:
    # Both statements use the caller's read snapshot. Joining the selected turns
    # fetches all messages in one query instead of one query per turn.
    selection = """SELECT id, turn_index, input, result FROM runs
        WHERE agent_id = ? AND conversation_id = ? AND turn_index > ?
        ORDER BY turn_index LIMIT ?"""
    parameters = (
        agent_id,
        conversation_id,
        after_turn_index,
        limit if limit is not None else 9223372036854775807,
    )
    rows = connection.execute(selection, parameters).fetchall()
    messages: dict[str, list[CanonicalMessage]] = {row[0]: [] for row in rows}
    if rows:
        for run_id, data in connection.execute(
            "WITH selected AS (" + selection + ") "
            "SELECT m.run_id, m.data FROM messages AS m JOIN selected AS s ON s.id = m.run_id "
            "ORDER BY s.turn_index, m.position",
            parameters,
        ):
            messages[run_id].append(decode_message(data))
    return tuple(
        ConversationRun(
            turn_index=int(turn_index),
            transcript=Transcript(
                run=decode_run_input(input_data), messages=tuple(messages[run_id])
            ),
            result=None if result_data is None else decode_loop_exit(result_data),
        )
        for run_id, turn_index, input_data, result_data in rows
    )


async def _run_graph_read(
    path: SQLDatabase,
    callback: Callable[[SQLConnection], _T],
    *,
    commit: Callable[[SQLConnection], None] | None = None,
) -> _T:
    def read() -> _T:
        try:
            with _connect_read_only(path) as connection:
                connection.begin()
                return callback(connection)
        except Exception as error:
            raise path.normalize_error(error) from None

    return await settled_thread(read)


def _validate_current_mcp_binding_bounds(connection: SQLConnection) -> None:
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


def _active_artifact_reservation(
    connection: SQLConnection, record: ArtifactRecord
) -> bool:
    row = connection.execute(
        "SELECT attempt_id, state FROM job_task_attempts WHERE agent_id = ? AND run_id = ?",
        (record.agent_id, record.ref.run_id),
    ).fetchone()
    return (
        row is not None
        and row[1] in {"claimed", "running"}
        and reserved_artifact_id(row[0]) == record.ref.artifact_id
    )


def _artifact_record_row(record: ArtifactRecord) -> tuple[object, ...]:
    ref = record.ref
    return (
        ref.artifact_id,
        record.agent_id,
        ref.run_id,
        ref.conversation_id,
        record.caller_principal_id,
        record.state.value,
        ref.byte_size,
        int(ref.created_at.timestamp() * 1_000_000),
        encode_artifact_record(record),
    )


def _artifact_record_from_row(row: tuple) -> ArtifactRecord:
    record = decode_artifact_record(row[-1])
    if tuple(row) != _artifact_record_row(record):
        raise ValueError("stored artifact registry projection differs from its record")
    return record


def _get_artifact_record(
    connection: SQLConnection, artifact_id: str
) -> ArtifactRecord | None:
    row = connection.execute(
        "SELECT artifact_id, agent_id, run_id, conversation_id, caller_principal_id, state, byte_size, created_at_us, data FROM artifacts WHERE artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    return None if row is None else _artifact_record_from_row(row)


def _validate_current_records(connection: SQLConnection) -> AgentIdentity | None:
    _validate_current_mcp_binding_bounds(connection)
    identity: AgentIdentity | None = None
    for key, data in connection.execute("SELECT key, data FROM metadata"):
        if key == "identity":
            if identity is not None:
                raise ValueError("state contains duplicate agent identity")
            identity = decode_identity(data)
        else:
            raise ValueError("state metadata key is unsupported")

    records = tuple(
        _artifact_record_from_row(row)
        for row in connection.execute(
            "SELECT artifact_id, agent_id, run_id, conversation_id, caller_principal_id, state, byte_size, created_at_us, data FROM artifacts"
        )
    )
    if len(records) > MAX_ARTIFACTS_PER_AGENT:
        raise ValueError("stored artifact registry exceeds its fixed bound")
    if records and (
        identity is None or any(record.agent_id != identity.id for record in records)
    ):
        raise ValueError("stored artifact belongs to another agent")

    analytical_counts: dict[str, int] = {}
    analytical_bytes: dict[str, int] = {}
    for (
        agent_id,
        run_id,
        evidence_id,
        kind,
        data,
        reserved_bytes,
        run_owner,
    ) in connection.execute(
        "SELECT a.agent_id, a.run_id, a.evidence_id, a.kind, a.data, a.reserved_bytes, r.agent_id "
        "FROM analysis_evidence a LEFT JOIN runs r ON r.id = a.run_id"
    ):
        evidence = decode_analysis_evidence(data)
        if (
            agent_id != run_owner
            or identity is None
            or agent_id != identity.id
            or (evidence.run_id, evidence.evidence_id, evidence.kind)
            != (run_id, evidence_id, kind)
            or type(reserved_bytes) is not int
            or reserved_bytes < len(data.encode())
        ):
            raise ValueError(
                "Stored analytical evidence identity or byte reservation is invalid"
            )
        analytical_counts[run_id] = analytical_counts.get(run_id, 0) + 1
        analytical_bytes[run_id] = analytical_bytes.get(run_id, 0) + reserved_bytes
        if (
            analytical_counts[run_id] > 256
            or analytical_bytes[run_id] > 4 * 1024 * 1024
        ):
            raise ValueError("Stored analytical evidence exceeds its run bounds")

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
           FROM relational_write_scopes"""
    ):
        scope_registration = sources.get((agent_id, source_id))
        if scope_registration is None or not scope_registration.active:
            raise ValueError("stored PostgreSQL update scope is foreign")
        decode_relational_write_scope(
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
    run_inputs: dict[str, RunInput] = {}
    run_results: dict[str, LoopExit] = {}
    run_messages: dict[str, list[CanonicalMessage]] = {}
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
        if identity is not None and run_input.agent_id != identity.id:
            raise ValueError("stored run belongs to another agent")
        if result is not None:
            exit_record = decode_loop_exit(result)
            if (
                exit_record.run_id != run_id
                or exit_record.conversation_id != conversation_id
            ):
                raise ValueError("stored run result ownership is invalid")
            run_results[run_id] = exit_record
        run_ids.add(run_id)
        run_inputs[run_id] = run_input
    for run_id, position, data in connection.execute(
        "SELECT run_id, position, data FROM messages ORDER BY run_id, position"
    ):
        if run_id not in run_ids:
            raise ValueError("stored message belongs to an unknown run")
        message = decode_message(data)
        message_positions.setdefault(run_id, []).append(position)
        run_messages.setdefault(run_id, []).append(message)
    if any(
        positions != list(range(len(positions)))
        for positions in message_positions.values()
    ):
        raise ValueError("stored transcript message positions are not contiguous")
    for run_id, result in run_results.items():
        if result.kind is LoopExitKind.COMPLETED:
            validate_completed_transcript(
                Transcript(
                    run=run_inputs[run_id],
                    messages=tuple(run_messages.get(run_id, ())),
                ),
                result,
            )

    for (
        agent_id,
        receipt_id,
        run_id,
        call_id,
        operation_key,
        routine_id,
        occurrence_id,
        grant_digest,
        unresolved,
        data,
    ) in connection.execute(
        """SELECT agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, grant_digest, unresolved, data
           FROM effect_receipts"""
    ):
        receipt = decode_receipt(data)
        if (
            receipt.agent_id != agent_id
            or receipt.receipt_id != receipt_id
            or receipt.run_id != run_id
            or receipt.call_id != call_id
            or receipt.operation_key != operation_key
            or receipt.routine_id != routine_id
            or receipt.occurrence_id != occurrence_id
            or receipt.capability_grant_digest != grant_digest
            or int(receipt.unresolved) != unresolved
        ):
            raise ValueError(
                "stored effect receipt identity or indexed state is invalid"
            )
        if identity is not None and receipt.agent_id != identity.id:
            raise ValueError("stored effect receipt belongs to another agent")

    for agent_id, annotation_id, data in connection.execute(
        "SELECT agent_id, id, data FROM semantic_annotations"
    ):
        annotation = decode_semantic_annotation(data)
        if annotation.agent_id != agent_id or annotation.id != annotation_id:
            raise ValueError("stored semantic annotation ownership is invalid")
        if identity is not None and annotation.agent_id != identity.id:
            raise ValueError("stored semantic annotation belongs to another agent")
    for agent_id, link_id, data in connection.execute(
        "SELECT agent_id, id, data FROM catalog_links"
    ):
        link = decode_catalog_link(data)
        if link.agent_id != agent_id or link.id != link_id:
            raise ValueError("stored catalog link ownership is invalid")
        if identity is not None and link.agent_id != identity.id:
            raise ValueError("stored catalog link belongs to another agent")

    # Keep only job records: later checks need nothing else, so history size
    # does not bound how many jobs an agent may have.
    graph_jobs: dict[tuple[str, str], GraphJob] = {}
    for agent_id, job_id in connection.execute("SELECT agent_id, job_id FROM job_runs"):
        inspection = _graph_store.inspect_graph(connection, agent_id, job_id)
        if inspection is None:
            raise ValueError("stored graph job is unavailable")
        if identity is not None and inspection.job.agent_id != identity.id:
            raise ValueError("stored graph job belongs to another agent")
        graph_jobs[(agent_id, job_id)] = inspection.job

    graph_attention_producers: dict[tuple[str, str], tuple[GraphJob, str, datetime]] = (
        {}
    )
    for (
        event_id,
        agent_id,
        job_id,
        task_id,
        attempt_id,
        kind,
        created_at_us,
        data,
    ) in connection.execute(
        """SELECT event_id, agent_id, job_id, task_id, attempt_id, kind,
                  created_at_us, data
           FROM job_graph_events
           WHERE kind IN (
               'task_control_opened',
               'task_retry_circuit_opened',
               'task_review_changes_requested'
           )"""
    ):
        graph_job = graph_jobs.get((agent_id, job_id))
        if graph_job is None:
            raise ValueError("stored graph attention event has no owned job")
        created_at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
            microseconds=int(created_at_us)
        )
        event = decode_graph_event(
            data,
            event_id=int(event_id),
            agent_id=agent_id,
            job_id=job_id,
            task_id=task_id,
            attempt_id=attempt_id,
            kind=kind,
            created_at=created_at,
        )
        transition_kind: str
        transition_identity: object
        if event.kind == "task_control_opened":
            control_kind = event.payload.get("kind")
            if control_kind in {
                ControlKind.NEEDS_REPLAN.value,
                ControlKind.REVIEW_REQUESTED.value,
            }:
                continue
            transition_kind = "control_opened"
            transition_identity = event.payload.get("control_id")
        elif event.kind == "task_retry_circuit_opened":
            transition_kind = "retry_circuit_open"
            transition_identity = event.payload.get("control_id")
        else:
            transition_kind = "review_changes_requested"
            transition_identity = event.payload.get("changes_control_id")
        if not isinstance(transition_identity, str) or not transition_identity:
            raise ValueError("stored graph attention event identity is invalid")
        subject = (
            f"{event.job_id}/event/{event.event_id}/{transition_kind}/"
            f"{transition_identity}"
        )
        digest = canonical_digest(
            {
                "job_id": event.job_id,
                "event_id": event.event_id,
                "transition_kind": transition_kind,
                "transition_identity": transition_identity,
            }
        )
        key = (event.agent_id, subject)
        if key in graph_attention_producers:
            raise ValueError("stored graph attention producer identity is duplicated")
        graph_attention_producers[key] = (graph_job, digest, created_at)

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
    for routine in routines.values():
        if routine.active_occurrence_id is None:
            continue
        active = occurrences.get((routine.agent_id, routine.active_occurrence_id))
        if active is None or active.routine_id != routine.routine_id:
            raise ValueError("stored routine active occurrence is invalid")

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
        if subject_kind == "graph_job":
            graph_delivery = decode_graph_job_delivery(
                data,
                agent_id=agent_id,
                delivery_id=delivery_id,
                conversation_id=conversation_id,
                subject_kind=subject_kind,
                subject_id=subject_id,
                logical_key=logical_key,
                state=state,
            )
            if not isinstance(graph_delivery, GraphJobDelivery):
                raise TypeError("current graph delivery did not decode to its record")
            if (
                graph_delivery.target.target_fingerprint != target_fingerprint_value
                or _datetime_us(graph_delivery.created_at) != created_at_us
                or (agent_id, subject_id) not in graph_jobs
            ):
                raise ValueError("stored graph delivery projection is invalid")
            delivery_counts[agent_id] = delivery_counts.get(agent_id, 0) + 1
            if delivery_counts[agent_id] > MAX_DELIVERIES_PER_AGENT:
                raise ValueError("stored delivery count exceeds its fixed bound")
            continue
        delivery = decode_delivery(data, agent_id=agent_id, delivery_id=delivery_id)
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
        if delivery.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE:
            producer = occurrences.get((agent_id, subject_id))
            producer_references_delivery = (
                producer is not None and delivery_id in producer.delivery_ids
            )
        elif delivery.subject_kind is DeliverySubjectKind.GRAPH_ATTENTION:
            attention = graph_attention_producers.get((agent_id, subject_id))
            if attention is None:
                raise ValueError("stored graph attention producer is invalid")
            attention_job, transition_digest, transition_at = attention
            producer_references_delivery = (
                delivery.conversation_id == attention_job.conversation_id
                and delivery.outcome.conclusion_id == subject_id
                and delivery.outcome.conclusion_digest == transition_digest
                and delivery.outcome.provenance_digest == transition_digest
                and delivery.outcome.observed_at == transition_at
                and delivery.outcome.resulting_run_id is None
                and not delivery.outcome.artifact_references
                and not delivery.outcome.effect_receipt_ids
                and delivery.delivery_id
                == "delivery-" + sha256(subject_id.encode()).hexdigest()[:32]
                and attention_job.specification.distribution_plan_digest
                == distribution_plan_digest(
                    targets=(delivery.target,), required_target_count=1
                )
            )
        else:
            raise ValueError("stored delivery retained a legacy producer")
        if not producer_references_delivery:
            raise ValueError("stored delivery producer reference is invalid")
        delivery_counts[agent_id] = delivery_counts.get(agent_id, 0) + 1
        if delivery_counts[agent_id] > MAX_DELIVERIES_PER_AGENT:
            raise ValueError("stored delivery count exceeds its fixed bound")
    return identity


def _pause_effect_routine(
    connection: SQLConnection, receipt: EffectReceipt, changed_at: datetime
) -> None:
    if receipt.routine_id is None:
        return
    loaded = _load_routine_row(connection, receipt.agent_id, receipt.routine_id)
    if loaded is None:
        raise EffectReceiptConflictError("the producing routine is unavailable")
    routine, encoded = loaded
    _replace_routine_row(
        connection,
        encoded,
        replace(
            routine,
            state=RoutineState.PAUSED,
            updated_at=max(changed_at, routine.updated_at),
        ),
    )


def _open_graph_effect_uncertain_control(
    connection: SQLConnection,
    *,
    receipt: EffectReceipt,
    job_id: str,
    task_id: str,
    attempt_id: str,
    fencing_epoch: int,
    task_spec_digest: str,
    opened_at: datetime,
) -> TaskControl | None:
    loaded_task = _graph_store._load_task(connection, receipt.agent_id, job_id, task_id)
    loaded_attempt = _graph_store._load_attempt(
        connection, receipt.agent_id, job_id, task_id, attempt_id
    )
    if loaded_task is None or loaded_attempt is None:
        return None
    task = loaded_task[0]
    attempt = loaded_attempt[0]
    if (
        task.state is not TaskState.RUNNING
        or task.current_attempt_id != attempt_id
        or task.task_spec_digest != task_spec_digest
        or task.fencing_epoch != fencing_epoch
        or attempt.state not in ACTIVE_ATTEMPT_STATES
        or attempt.fencing_epoch != fencing_epoch
    ):
        return None
    control_id = (
        "control-"
        + sha256(f"{receipt.receipt_id}:effect-uncertain".encode()).hexdigest()[:32]
    )
    existing = connection.execute(
        "SELECT data FROM job_task_controls WHERE agent_id = ? AND job_id = ? AND task_id = ? AND control_id = ?",
        (receipt.agent_id, job_id, task_id, control_id),
    ).fetchone()
    if existing is not None:
        return None
    if receipt.outcome is EffectOutcome.SUCCEEDED:
        message = (
            "The external operation succeeded, but its graph result was not committed; "
            "the operation will not be replayed."
        )
    elif receipt.outcome is EffectOutcome.NOT_APPLIED:
        message = "The reserved external operation was not applied and will not be replayed automatically."
    else:
        message = (
            "The external operation outcome is uncertain and will not be replayed."
        )
    payload = {
        "message": message,
        "details": {
            "receipt_id": receipt.receipt_id,
            "receipt_digest": receipt.receipt_digest,
            "outcome": receipt.outcome.value,
            "evidence_basis": receipt.evidence_basis.value,
            "resolution_does_not_retry": True,
            "resolution_does_not_mark_success": True,
        },
    }
    control = TaskControl(
        agent_id=receipt.agent_id,
        job_id=job_id,
        task_id=task_id,
        control_id=control_id,
        kind=ControlKind.EFFECT_UNCERTAIN,
        state=ControlState.OPEN,
        requesting_attempt_id=attempt_id,
        payload=payload,
        created_at=opened_at,
        payload_digest=canonical_digest(payload),
    )
    return _graph_store.open_control(
        connection,
        control,
        claim_token=attempt.claim_token,
        fencing_epoch=fencing_epoch,
    )


def _recover_started_effect_receipts(
    connection: SQLConnection, agent_id: str, completed_at: datetime
) -> None:
    rows = tuple(
        connection.execute(
            "SELECT agent_id, id, data, job_id, task_id, task_attempt_id, fencing_epoch, task_spec_digest FROM effect_receipts WHERE agent_id = ?",
            (agent_id,),
        )
    )
    started = tuple(
        (
            agent_id,
            receipt_id,
            receipt,
            job_id,
            task_id,
            attempt_id,
            fencing_epoch,
            task_spec_digest,
        )
        for (
            agent_id,
            receipt_id,
            data,
            job_id,
            task_id,
            attempt_id,
            fencing_epoch,
            task_spec_digest,
        ) in rows
        if (receipt := decode_receipt(data)).outcome is EffectOutcome.STARTED
    )
    for (
        agent_id,
        receipt_id,
        receipt,
        job_id,
        task_id,
        attempt_id,
        fencing_epoch,
        task_spec_digest,
    ) in started:
        recovered = receipt.finish(
            EffectObservation(EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN),
            finished_at=completed_at,
        )
        result = connection.execute(
            """UPDATE effect_receipts SET data = ?
               WHERE agent_id = ? AND id = ? AND data = ?""",
            (
                encode_receipt(recovered),
                agent_id,
                receipt_id,
                encode_receipt(receipt),
            ),
        )
        if result.rowcount != 1:
            raise RuntimeError("effect receipt changed during startup recovery")
        _pause_effect_routine(connection, recovered, completed_at)
        if job_id is not None:
            _open_graph_effect_uncertain_control(
                connection,
                receipt=recovered,
                job_id=str(job_id),
                task_id=str(task_id),
                attempt_id=str(attempt_id),
                fencing_epoch=int(fencing_epoch),
                task_spec_digest=str(task_spec_digest),
                opened_at=completed_at,
            )


def _commit_catalog_transaction(connection: SQLConnection) -> None:
    connection.commit()


def _connect(database: SQLDatabase) -> SQLConnection:
    return database.connect()


def _connect_read_only(database: SQLDatabase) -> SQLConnection:
    return database.connect(read_only=True)
