"""Describe durable state operations independently of a database implementation.

This is the complete composition-boundary contract. Domain owners continue to
accept their existing narrower protocols. Construction, home upgrades, filesystem
paths and schema revision checks belong to backend-specific admission code.

Each mutation is one atomic operation: a backend must preserve the existing
compare-and-set, idempotency, fencing, ordering and budget-conservation semantics.
A successful await means durable commit; cancellation or lost acknowledgement
must not cause blind replay of external effects. Identity, payload validation,
bounds and domain errors have the same meaning for every backend.

The SQLite implementation is the current reference. This protocol alone does not
enable remote homes, distributed writer fencing or a Postgres backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from .._json import FrozenJsonObject
    from ..adapters.mcp import MCPServerBinding
    from ..adapters.models import SourceRegistration
    from ..artifacts.models import ArtifactRecord, ArtifactRef, ArtifactState
    from ..capabilities import CapabilityGrant, ExecutionScope, TaskAttemptGuard
    from ..catalog.models import (
        CatalogFacet,
        CatalogRelationship,
        CatalogResource,
        CatalogResourceRevision,
        CatalogSnapshotRef,
        CatalogSummary,
        CatalogSync,
        RelationshipKind,
        SourceCatalogSnapshot,
    )
    from ..distribution.models import (
        Delivery,
        GraphJobDelivery,
        OutcomeArtifactReference,
    )
    from ..identity import AgentIdentity
    from ..jobs.graph.models import (
        AttemptBudgetReservation,
        AttemptState,
        BudgetAmount,
        BudgetLedger,
        ControlState,
        GraphAdmission,
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
    )
    from ..learning_candidates import (
        LearningCandidate,
        LearningCandidateRejectionReason,
        LearningCandidateReviewStamp,
        LearningReviewRunTail,
    )
    from ..llm.models import CanonicalMessage
    from ..loop.models import ConversationRun, LoopExit, RunInput, Transcript
    from ..loop.transcripts import ConversationPredecessor
    from ..routines.models import (
        ResourceRevisionObservation,
        RoutineOccurrence,
        RoutineState,
        ScheduledRoutine,
    )
    from ..semantics import SemanticAnnotation
    from .sqlite_records import (
        EffectReceipt,
        EffectResolution,
        RelationalWriteScope,
        SourceReadScope,
    )


class StateStore(Protocol):
    """Durable state required by agent composition and execution supervisors."""

    async def admit_graph(self, admission: GraphAdmission) -> GraphJob: ...

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
    ) -> GraphJob: ...

    async def inspect_graph(
        self, agent_id: str, job_id: str
    ) -> GraphInspection | None: ...

    async def list_graph_jobs(
        self, agent_id: str, *, states: frozenset[GraphState] = ..., limit: int = ...
    ) -> tuple[GraphJob, ...]: ...

    async def apply_graph_mutation(
        self, request: GraphMutationRequest
    ) -> GraphMutation: ...

    async def request_graph_cancel(
        self,
        agent_id: str,
        job_id: str,
        *,
        requested_at: datetime,
        requested_by_id: str,
    ) -> GraphJob | None: ...

    async def list_ready_graph_tasks(
        self, agent_id: str, *, now: datetime, limit: int = ...
    ) -> tuple[GraphTask, ...]: ...

    async def expire_due_graphs(
        self, agent_id: str, *, expired_at: datetime, limit: int = ...
    ) -> tuple[GraphJob, ...]: ...

    async def list_stale_graph_attempts(
        self, agent_id: str, *, now: datetime, limit: int = ...
    ) -> tuple[TaskAttempt, ...]: ...

    async def list_active_graph_attempts(
        self, agent_id: str, *, limit: int = ...
    ) -> tuple[TaskAttempt, ...]: ...

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
    ) -> TaskAttempt | None: ...

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
    ) -> TaskAttempt | None: ...

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
        lease_seconds: int = ...,
    ) -> TaskAttempt | None: ...

    async def checkpoint_graph_attempt(
        self, checkpoint: TaskCheckpoint, *, claim_token: str
    ) -> TaskCheckpoint: ...

    async def add_graph_comment(
        self,
        comment: TaskComment,
        *,
        attempt_id: str | None = ...,
        claim_token: str | None = ...,
        fencing_epoch: int | None = ...,
    ) -> TaskComment: ...

    async def complete_graph_attempt(
        self,
        result: TaskResult,
        *,
        claim_token: str,
        fencing_epoch: int,
        usage: tuple[BudgetAmount, ...] | None,
    ) -> TaskResult: ...

    async def finalize_graph_attempt(
        self,
        result: TaskResult,
        delivery: GraphJobDelivery,
        *,
        claim_token: str,
        fencing_epoch: int,
        usage: tuple[BudgetAmount, ...] | None,
    ) -> TaskResult: ...

    async def list_graph_deliveries(
        self, agent_id: str, *, job_id: str | None = ..., limit: int = ...
    ) -> tuple[GraphJobDelivery, ...]: ...

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
    ) -> TaskAttempt | None: ...

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
        attempt_state: AttemptState = ...,
    ) -> TaskAttempt | None: ...

    async def open_graph_control(
        self,
        control: TaskControl,
        *,
        claim_token: str,
        fencing_epoch: int,
        replan_task: GraphTask | None = ...,
        reviewer_task: GraphTask | None = ...,
    ) -> TaskControl: ...

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
        expected_control_digest: str | None = ...,
        expected_task_revision: int | None = ...,
    ) -> TaskControl | None: ...

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
        reviewer_result: TaskResult | None = ...,
        claim_token: str | None = ...,
        fencing_epoch: int | None = ...,
    ) -> TaskResult: ...

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
        reviewer_result: TaskResult | None = ...,
        claim_token: str | None = ...,
        fencing_epoch: int | None = ...,
    ) -> TaskControl: ...

    async def list_graph_events(
        self,
        agent_id: str,
        job_id: str,
        *,
        after_event_id: int = ...,
        limit: int = ...,
        task_id: str | None = ...,
    ) -> GraphEventPage: ...

    async def list_graph_budget_ledgers(
        self, agent_id: str, job_id: str
    ) -> tuple[BudgetLedger, ...]: ...

    async def list_graph_attempt_reservations(
        self, agent_id: str, job_id: str, task_id: str, attempt_id: str
    ) -> tuple[AttemptBudgetReservation, ...]: ...

    async def close(self) -> None: ...

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity: ...

    async def load_identity(self) -> AgentIdentity | None: ...

    async def load_deletion_credential_inventory(
        self, agent_id: str
    ) -> tuple[AgentIdentity | None, tuple[str, ...], tuple[str, ...]]: ...

    async def load_mcp_binding(
        self, agent_id: str, binding_id: str
    ) -> MCPServerBinding | None: ...

    async def list_mcp_bindings(
        self, agent_id: str
    ) -> tuple[MCPServerBinding, ...]: ...

    async def store_mcp_binding(
        self, binding: MCPServerBinding, *, expected_revision: int | None
    ) -> MCPServerBinding: ...

    async def update_mcp_discovery(
        self,
        agent_id: str,
        binding_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> MCPServerBinding: ...

    async def update_source_discovery(
        self,
        agent_id: str,
        source_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> SourceRegistration: ...

    async def admit_scheduled_routine(
        self, routine: ScheduledRoutine
    ) -> ScheduledRoutine: ...

    async def load_scheduled_routine(
        self, agent_id: str, routine_id: str
    ) -> ScheduledRoutine | None: ...

    async def list_scheduled_routines(
        self, agent_id: str, *, states: frozenset[RoutineState] = ..., limit: int = ...
    ) -> tuple[ScheduledRoutine, ...]: ...

    async def list_routine_occurrences(
        self, agent_id: str, routine_id: str, *, limit: int = ...
    ) -> tuple[RoutineOccurrence, ...]: ...

    async def revise_scheduled_routine(
        self, routine: ScheduledRoutine, *, expected_revision: int
    ) -> ScheduledRoutine | None: ...

    async def transition_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        state: RoutineState,
        transitioned_at: datetime,
    ) -> ScheduledRoutine | None: ...

    async def next_routine_deadline(self, agent_id: str) -> datetime | None: ...

    async def load_routine_occurrence(
        self, agent_id: str, occurrence_id: str
    ) -> RoutineOccurrence | None: ...

    async def claim_due_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        expected_due_at: datetime,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None: ...

    async def claim_manual_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        authorized_control_call_id: str,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None: ...

    async def bind_routine_occurrence_run(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        claim_token: str,
        run_id: str,
        execution_scope: ExecutionScope,
        bound_at: datetime,
        precheck_observation: ResourceRevisionObservation | None = ...,
    ) -> RoutineOccurrence | None: ...

    async def mark_routine_occurrence_run_terminal(
        self, agent_id: str, occurrence_id: str, *, run_id: str, terminal_at: datetime
    ) -> RoutineOccurrence | None: ...

    async def finalize_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
        skipped_no_change_observation: ResourceRevisionObservation | None = ...,
        failure_code: str | None = ...,
        artifact_references: tuple[OutcomeArtifactReference, ...] = ...,
        outcome_contract_failure_code: str | None = ...,
    ) -> tuple[RoutineOccurrence, Delivery | None] | None: ...

    async def recover_stale_routine_occurrences(
        self,
        agent_id: str,
        *,
        recovered_at: datetime,
        claim_token_factory: Callable[[str], str],
    ) -> tuple[RoutineOccurrence, ...]: ...

    async def load_delivery(
        self, agent_id: str, delivery_id: str
    ) -> Delivery | None: ...

    async def list_deliveries(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = ...,
        include_acknowledged: bool = ...,
        limit: int = ...,
    ) -> tuple[Delivery, ...]: ...

    async def acknowledge_delivery(
        self, agent_id: str, delivery_id: str, *, acknowledged_at: datetime
    ) -> Delivery | None: ...

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration: ...

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None: ...

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]: ...

    async def load_source_read_scope(
        self, agent_id: str, source_id: str
    ) -> SourceReadScope | None: ...

    async def list_relational_write_scopes(
        self, agent_id: str, source_id: str
    ) -> tuple[RelationalWriteScope, ...]: ...

    async def replace_source_permission_scopes(
        self,
        read_scope: SourceReadScope,
        update_scopes: tuple[RelationalWriteScope, ...],
    ) -> SourceRegistration: ...

    async def load_effect_receipt(
        self, agent_id: str, receipt_id: str
    ) -> EffectReceipt | None: ...

    async def load_effect_receipt_for_call(
        self, agent_id: str, run_id: str, call_id: str
    ) -> EffectReceipt | None: ...

    async def load_effect_receipt_for_operation(
        self, agent_id: str, operation_key: str
    ) -> EffectReceipt | None: ...

    async def list_effect_receipts(
        self,
        agent_id: str,
        *,
        run_id: str | None = ...,
        routine_id: str | None = ...,
        unresolved_only: bool = ...,
        limit: int = ...,
        offset: int = ...,
        caller_principal_id: str | None = ...,
    ) -> tuple[EffectReceipt, ...]: ...

    async def require_effects_unblocked(
        self, agent_id: str, *, run_id: str | None = ..., routine_id: str | None = ...
    ) -> None: ...

    async def start_effect_receipt(
        self,
        receipt: EffectReceipt,
        *,
        grant: CapabilityGrant | None = ...,
        task_attempt_guard: TaskAttemptGuard | None = ...,
        max_receipts_per_run: int = ...,
    ) -> EffectReceipt: ...

    async def finish_effect_receipt(self, receipt: EffectReceipt) -> EffectReceipt: ...

    async def list_effect_receipts_for_graph_attempt(
        self, agent_id: str, job_id: str, task_id: str, attempt_id: str
    ) -> tuple[EffectReceipt, ...]: ...

    async def reconcile_graph_effect_attempt(
        self, agent_id: str, job_id: str, task_id: str, attempt_id: str
    ) -> bool: ...

    async def resolve_effect_receipt(
        self, agent_id: str, resolution: EffectResolution
    ) -> EffectReceipt: ...

    async def detach_source(
        self, agent_id: str, source_id: str, detached_at: datetime
    ) -> SourceRegistration: ...

    async def record_sync(self, sync: CatalogSync) -> CatalogSync: ...

    async def commit_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration | None = ...,
    ) -> SourceCatalogSnapshot: ...

    async def commit_source_edit(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration,
        replaced_source_id: str,
        replaced_at: datetime,
        read_scope: SourceReadScope,
    ) -> SourceCatalogSnapshot: ...

    async def list_current_snapshot_refs(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> tuple[CatalogSnapshotRef, ...]: ...

    async def load_current_snapshot(
        self, ref: CatalogSnapshotRef
    ) -> SourceCatalogSnapshot | None: ...

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None: ...

    async def summarize_catalog(
        self, agent_id: str, active_source_ids: tuple[str, ...]
    ) -> CatalogSummary: ...

    async def load_resource(
        self, agent_id: str, resource_id: str
    ) -> CatalogResource | None: ...

    async def load_revision(
        self, agent_id: str, resource_id: str, revision: str
    ) -> CatalogResourceRevision | None: ...

    async def list_resources(
        self, agent_id: str, source_id: str | None = ...
    ) -> tuple[CatalogResource, ...]: ...

    async def load_facets(
        self, agent_id: str, resource_id: str, revision: str | None = ...
    ) -> tuple[CatalogFacet, ...]: ...

    async def load_incident_relationships(
        self,
        agent_id: str,
        resource_id: str,
        *,
        relationship_kinds: tuple[RelationshipKind, ...] = ...,
        limit: int = ...,
    ) -> tuple[CatalogRelationship, ...]: ...

    async def list_semantic_annotations(
        self, agent_id: str
    ) -> tuple[SemanticAnnotation, ...]: ...

    async def load_semantic_annotation(
        self, agent_id: str, annotation_id: str
    ) -> SemanticAnnotation | None: ...

    async def preflight_semantic_save(
        self, agent_id: str, annotation: SemanticAnnotation, expected_sha256: str | None
    ) -> FrozenJsonObject: ...

    async def save_semantic_annotation(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        *,
        expected_sha256: str | None = ...,
    ) -> bool: ...

    async def preflight_semantic_delete(
        self, agent_id: str, annotation_id: str, expected_sha256: str
    ) -> FrozenJsonObject: ...

    async def delete_semantic_annotation(
        self, agent_id: str, annotation_id: str, *, expected_sha256: str
    ) -> bool: ...

    async def recent_completed_runs(
        self, agent_id: str, *, limit: int
    ) -> LearningReviewRunTail: ...

    async def list_learning_candidates(
        self, agent_id: str
    ) -> tuple[LearningCandidate, ...]: ...

    async def load_learning_candidate(
        self, agent_id: str, candidate_id: str
    ) -> LearningCandidate | None: ...

    async def learning_candidate_review_stamps(
        self, agent_id: str
    ) -> tuple[LearningCandidateReviewStamp, ...]: ...

    async def save_learning_candidate_review(
        self,
        agent_id: str,
        *,
        stamps: tuple[LearningCandidateReviewStamp, ...],
        candidates: tuple[LearningCandidate, ...],
    ) -> tuple[LearningCandidate, ...]: ...

    async def edit_learning_candidate(
        self, agent_id: str, candidate: LearningCandidate, *, expected_fingerprint: str
    ) -> LearningCandidate: ...

    async def reject_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
        *,
        expected_fingerprint: str,
        reason: LearningCandidateRejectionReason,
        rejected_at: datetime,
    ) -> LearningCandidate: ...

    async def accept_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
        *,
        expected_fingerprint: str,
        accepted_at: datetime,
    ) -> LearningCandidate: ...

    async def clear_rejected_learning_candidates(self, agent_id: str) -> int: ...

    async def start(
        self, run: RunInput, *, predecessor: ConversationPredecessor | None = ...
    ) -> Transcript:
        """Admit a turn; explicit None asserts an empty conversation.

        An omitted predecessor preserves legacy unchecked continuation. A supplied
        predecessor must match the latest terminal turn atomically with admission.
        """
        ...

    async def append(self, run_id: str, message: CanonicalMessage) -> None: ...

    async def append_at(
        self, run_id: str, position: int, message: CanonicalMessage
    ) -> None: ...

    async def finish(self, result: LoopExit) -> None: ...

    async def complete(
        self, result: LoopExit, final_message: CanonicalMessage
    ) -> None: ...

    async def recover_unfinished_runs(
        self, agent_id: str, *, created_at: datetime
    ) -> tuple[LoopExit, ...]: ...

    async def load(self, run_id: str) -> Transcript: ...

    async def result(self, run_id: str) -> LoopExit | None: ...

    async def get_artifact_record(self, artifact_id: str) -> ArtifactRecord | None: ...

    async def list_artifact_records(
        self,
        agent_id: str,
        *,
        state: ArtifactState | None = ...,
        run_id: str | None = ...,
        conversation_id: str | None = ...,
        caller_principal_id: str | None = ...,
        limit: int | None = ...,
        offset: int = ...,
    ) -> tuple[ArtifactRecord, ...]: ...

    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = ...,
        conversation_id: str | None = ...,
        caller_principal_id: str | None = ...,
        limit: int | None = ...,
        offset: int = ...,
    ) -> tuple[ArtifactRef, ...]: ...

    async def begin_artifact_creation(
        self, record: ArtifactRecord, *, reserved: bool = ...
    ) -> None: ...

    async def transition_artifact(
        self, record: ArtifactRecord, state: ArtifactState
    ) -> ArtifactRecord: ...

    async def finish_artifact_deletion(self, record: ArtifactRecord) -> None: ...

    async def conversation_runs(
        self, agent_id: str, conversation_id: str
    ) -> tuple[ConversationRun, ...]: ...

    async def conversation_exists(
        self, agent_id: str, conversation_id: str
    ) -> bool: ...

    async def clear_conversations(self, agent_id: str) -> int: ...

    async def completed_conversation_tail(
        self, agent_id: str, conversation_id: str, *, limit: int = ...
    ) -> tuple[bool, tuple[ConversationRun, ...], bool]: ...

    async def latest_terminal_conversation_run(
        self, agent_id: str, conversation_id: str
    ) -> ConversationRun | None: ...
