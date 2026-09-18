"""Graph model-task lifecycle capabilities for the unreleased Phase 4 slice."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..._json import FrozenJsonObject
from ...artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactProvenance,
)
from ...capabilities import (
    AccessMode,
    ArtifactPolicy,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    MachineRunDirectiveKind,
    TaskAttemptGuard,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from ...capability_runtime import CapabilityFailure, SideEffectPlan
from ...catalog.models import Sensitivity
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.models import RunInput, RunOrigin, Transcript
from ..owner import JobError, JobOwner
from .models import (
    ControlKind,
    ControlState,
    GraphInspection,
    GraphTask,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskResult,
    canonical_digest,
)

GRAPH_TASK_DOMAIN_OWNER_ID = "jobs.graph_task"
TASK_CHECKPOINT_CAPABILITY_ID = "jobs.graph.task_checkpoint"
TASK_COMMENT_CAPABILITY_ID = "jobs.graph.task_comment"
TASK_COMPLETE_CAPABILITY_ID = "jobs.graph.task_complete"
TASK_BLOCK_CAPABILITY_ID = "jobs.graph.task_block"
TASK_REQUEST_REVIEW_CAPABILITY_ID = "jobs.graph.task_request_review"
GRAPH_RESULT_FINALIZE_CAPABILITY_ID = "jobs.graph.result_finalize"

TASK_CHECKPOINT_TOOL_NAME = "task_checkpoint"
TASK_COMMENT_TOOL_NAME = "task_comment"
TASK_COMPLETE_TOOL_NAME = "task_complete"
TASK_BLOCK_TOOL_NAME = "task_block"
TASK_REQUEST_REVIEW_TOOL_NAME = "task_request_review"


class TaskTranscriptReader(Protocol):
    async def load(self, run_id: str) -> Transcript: ...


@dataclass(frozen=True, slots=True)
class GraphTaskCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class _LifecycleExecutor:
    def __init__(
        self,
        owner: JobOwner,
        transcripts: TaskTranscriptReader,
        *,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self._owner = owner
        self._transcripts = transcripts
        self._clock = clock
        self._id_factory = id_factory

    async def _current(
        self, request: ToolExecution
    ) -> tuple[GraphInspection, GraphTask, TaskAttempt]:
        guard = request.task_attempt_guard
        if guard is None or guard.run_id != request.run_id:
            raise CapabilityInputError(
                "task_attempt_binding_missing",
                "The lifecycle operation lacks its exact task-attempt binding.",
            )
        binding = guard.binding
        inspection = await self._owner.inspect_graph(binding.job_id)
        if inspection is None:
            raise CapabilityInputError(
                "stale_task_attempt", "The graph task attempt is unavailable."
            )
        task = next(
            (item for item in inspection.tasks if item.task_id == binding.task_id),
            None,
        )
        attempt = next(
            (
                item
                for item in inspection.attempts
                if item.task_id == binding.task_id
                and item.attempt_id == binding.attempt_id
            ),
            None,
        )
        if task is None or attempt is None or attempt.run_id != request.run_id:
            raise CapabilityInputError(
                "stale_task_attempt", "The graph task attempt is unavailable."
            )
        return inspection, task, attempt


class TaskCheckpointExecutor(_LifecycleExecutor):
    executor_id = "jobs.graph.task_checkpoint.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        _inspection, _task, attempt = await self._current(request)
        guard = request.task_attempt_guard
        assert guard is not None
        payload = request.arguments.get("payload", {})
        milestone = request.arguments["milestone"]
        assert isinstance(payload, Mapping) and isinstance(milestone, str)
        checkpoint = TaskCheckpoint(
            agent_id=guard.binding.agent_id,
            job_id=guard.binding.job_id,
            task_id=guard.binding.task_id,
            attempt_id=guard.binding.attempt_id,
            checkpoint_id=self._id_factory("checkpoint"),
            fencing_epoch=guard.binding.fencing_epoch,
            ordinal=len(attempt.checkpoint_ids) + 1,
            milestone=milestone,
            payload=payload,
            created_at=self._clock(),
            payload_digest=canonical_digest(payload),
        )
        await guard.revalidate(
            capability_id=request.capability_id,
            point="before_checkpoint_commit",
        )
        stored = await self._owner.checkpoint_graph_task(
            checkpoint, claim_token=guard.claim_token
        )
        return _lifecycle_output(
            "graph.task_checkpoint",
            stored.checkpoint_id,
            request.request_sensitivity,
            guard.binding.digest,
        )


class TaskCommentExecutor(_LifecycleExecutor):
    executor_id = "jobs.graph.task_comment.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        await self._current(request)
        guard = request.task_attempt_guard
        assert guard is not None
        body = request.arguments["body"]
        assert isinstance(body, str)
        comment = TaskComment(
            agent_id=guard.binding.agent_id,
            job_id=guard.binding.job_id,
            task_id=guard.binding.task_id,
            comment_id=self._id_factory("comment"),
            author_kind="task_attempt",
            author_id=guard.binding.attempt_id,
            sensitivity=request.request_sensitivity,
            body=body,
            created_at=self._clock(),
            body_digest=canonical_digest({"body": body}),
        )
        await guard.revalidate(
            capability_id=request.capability_id,
            point="before_comment_commit",
        )
        stored = await self._owner.comment_graph_task(
            comment,
            attempt_id=guard.binding.attempt_id,
            claim_token=guard.claim_token,
            fencing_epoch=guard.binding.fencing_epoch,
        )
        return _lifecycle_output(
            "graph.task_comment",
            stored.comment_id,
            request.request_sensitivity,
            guard.binding.digest,
        )


class TaskCompleteExecutor(_LifecycleExecutor):
    executor_id = "jobs.graph.task_complete.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        _inspection, task, attempt = await self._current(request)
        guard = request.task_attempt_guard
        assert guard is not None
        transcript = await self._transcripts.load(request.run_id)
        raw_evidence_ids = request.arguments.get("evidence_call_ids")
        raw_artifact_ids = request.arguments.get("artifact_ids")
        if not isinstance(raw_evidence_ids, tuple) or not isinstance(
            raw_artifact_ids, tuple
        ):
            raise CapabilityInputError(
                "task_result_evidence_invalid", "Task evidence references are invalid."
            )
        evidence_ids: tuple[object, ...] = raw_evidence_ids
        artifacts = tuple(sorted(str(item) for item in raw_artifact_ids))
        authenticated, evidence_sensitivity, authenticated_artifacts = (
            _authenticate_evidence(transcript, evidence_ids)
        )
        if not set(artifacts) <= authenticated_artifacts:
            raise CapabilityInputError(
                "task_result_artifact_unverified",
                "A task result artifact is not authenticated by its cited tool evidence.",
            )
        result_kind = request.arguments["result_kind"]
        summary = request.arguments["summary"]
        payload = request.arguments["payload"]
        residual_risk = request.arguments.get("residual_risk")
        downstream = request.arguments.get("downstream_constraints", {})
        assert isinstance(result_kind, str)
        assert isinstance(summary, str)
        assert isinstance(payload, Mapping)
        assert residual_risk is None or isinstance(residual_risk, str)
        assert isinstance(downstream, Mapping)
        expected_kind = task.specification.expected_result_contract.get("result_kind")
        if isinstance(expected_kind, str) and result_kind != expected_kind:
            raise CapabilityInputError(
                "task_result_contract_mismatch",
                "The proposed result kind differs from the frozen task contract.",
            )
        completed_at = self._clock()
        result_id = self._id_factory("result")
        sensitivity = max(
            request.request_sensitivity,
            task.specification.authority.sensitivity,
            evidence_sensitivity,
            key=lambda item: item.routing_rank,
        )
        schema_digest = canonical_digest(task.specification.expected_result_contract)
        provenance = {
            "authority": "authenticated_graph_task_transcript",
            "task_binding_digest": guard.binding.digest,
            "evidence": authenticated,
            "model_authored_payload": True,
        }
        verification = {
            "transcript_run_id": request.run_id,
            "evidence_call_ids": evidence_ids,
            "evidence_authenticated": True,
            "artifact_ids_authenticated": artifacts,
        }
        material = {
            "agent_id": guard.binding.agent_id,
            "job_id": guard.binding.job_id,
            "task_id": guard.binding.task_id,
            "result_id": result_id,
            "attempt_id": guard.binding.attempt_id,
            "run_id": request.run_id,
            "result_kind": result_kind,
            "schema_digest": schema_digest,
            "payload": payload,
            "summary": summary,
            "sensitivity": sensitivity.value,
            "provenance": provenance,
            "artifact_ids": artifacts,
            "effect_receipt_ids": (),
            "verification": verification,
            "residual_risk": residual_risk,
            "downstream_constraints": downstream,
            "completed_at": completed_at.isoformat(),
        }
        result = TaskResult(
            agent_id=guard.binding.agent_id,
            job_id=guard.binding.job_id,
            task_id=guard.binding.task_id,
            result_id=result_id,
            attempt_id=guard.binding.attempt_id,
            run_id=request.run_id,
            result_kind=result_kind,
            schema_digest=schema_digest,
            payload=payload,
            summary=summary,
            sensitivity=sensitivity,
            provenance=provenance,
            artifact_ids=artifacts,
            effect_receipt_ids=(),
            verification=verification,
            residual_risk=residual_risk,
            downstream_constraints=downstream,
            completed_at=completed_at,
            result_digest=canonical_digest(material),
        )
        await guard.revalidate(
            capability_id=request.capability_id,
            point="before_result_commit",
        )
        stored = await self._owner.complete_graph_task(
            result,
            claim_token=guard.claim_token,
            fencing_epoch=guard.binding.fencing_epoch,
            usage=None,
        )
        return await _termination_output(
            self._owner,
            "graph.task_complete",
            stored.result_id,
            sensitivity,
            guard,
        )


class _TaskControlExecutor(_LifecycleExecutor):
    control_kind: ControlKind
    output_kind: str

    async def execute(self, request: ToolExecution) -> ToolOutput:
        await self._current(request)
        guard = request.task_attempt_guard
        assert guard is not None
        message = request.arguments["message"]
        details = request.arguments.get("details", {})
        assert isinstance(message, str) and isinstance(details, Mapping)
        kind = self.control_kind
        if self.control_kind is not ControlKind.REVIEW_REQUESTED:
            raw_kind = request.arguments["kind"]
            assert isinstance(raw_kind, str)
            kind = ControlKind(raw_kind)
        payload = {"message": message, "details": details}
        control = TaskControl(
            agent_id=guard.binding.agent_id,
            job_id=guard.binding.job_id,
            task_id=guard.binding.task_id,
            control_id=self._id_factory("control"),
            kind=kind,
            state=ControlState.OPEN,
            requesting_attempt_id=guard.binding.attempt_id,
            payload=payload,
            created_at=self._clock(),
            payload_digest=canonical_digest(payload),
        )
        await guard.revalidate(
            capability_id=request.capability_id,
            point="before_control_commit",
        )
        stored = await self._owner.open_graph_task_control(
            control,
            claim_token=guard.claim_token,
            fencing_epoch=guard.binding.fencing_epoch,
        )
        return await _termination_output(
            self._owner,
            self.output_kind,
            stored.control_id,
            request.request_sensitivity,
            guard,
        )


class TaskBlockExecutor(_TaskControlExecutor):
    executor_id = "jobs.graph.task_block.executor"
    control_kind = ControlKind.NEEDS_INPUT
    output_kind = "graph.task_block"


class TaskRequestReviewExecutor(_TaskControlExecutor):
    executor_id = "jobs.graph.task_request_review.executor"
    control_kind = ControlKind.REVIEW_REQUESTED
    output_kind = "graph.task_review_request"


class GraphResultFinalizeExecutor:
    executor_id = "jobs.graph.result_finalize.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        work_results = request.arguments["work_results"]
        assert isinstance(job_id, str) and isinstance(work_results, tuple)
        document = {
            "kind": "graph_result",
            "job_id": job_id,
            "accepted_results": work_results,
            "trust_classification": "model_derived_authenticated_task_results",
        }
        from ..._json import canonical_json

        content = canonical_json(document).encode("utf-8")
        return ToolOutput(
            kind="graph.result_finalized",
            data={"job_id": job_id, "accepted_result_count": len(work_results)},
            artifact=ArtifactDraft(
                content=content,
                suggested_filename=f"graph-result-{job_id}.json",
                media_type="application/json",
                sensitivity=Sensitivity(request.request_sensitivity.value),
                provenance=ArtifactProvenance(
                    authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
                ),
            ),
            sensitivity=request.request_sensitivity,
            sensitivity_provenance={
                "authority": "authenticated_graph_task_results",
                "job_id": job_id,
            },
        )


class GraphTaskCapabilityDomain:
    domain_owner_id = GRAPH_TASK_DOMAIN_OWNER_ID

    def __init__(self, declarations: CapabilityDeclarations, owner: JobOwner) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("graph task declarations have the wrong owner")
        self._declarations = declarations
        self._owner = owner

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        if run.origin is not RunOrigin.JOB_TASK or run.agent_id != self._owner.agent_id:
            return ()
        scope = run.execution_scope
        if scope is None or scope.graph_task_binding is None:
            return ()
        return tuple(
            view.name
            for view in self._declarations.tool_views
            if view.capability_id in scope.allowed_capability_ids
        )

    def normalize_arguments(
        self, capability: Capability, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        del call, request_sensitivity
        if capability.id == GRAPH_RESULT_FINALIZE_CAPABILITY_ID:
            return arguments
        if run.origin is not RunOrigin.JOB_TASK:
            raise CapabilityInputError(
                "task_lifecycle_scope_required",
                "Task lifecycle tools require a graph-task run.",
            )
        return arguments

    async def prepare_automation_grant(
        self,
        capability: Capability,
        constraints: FrozenJsonObject,
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> FrozenJsonObject:
        raise CapabilityInputError(
            "automation_grant_unsupported",
            "Graph task lifecycle capabilities do not admit standing grants.",
        )

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        raise ValueError("graph task lifecycle capabilities are effect-free")

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> ToolOutput:
        return output

    def normalize_error(
        self, call: ToolCall, error: BaseException
    ) -> CapabilityFailure | None:
        del call
        if isinstance(error, JobError):
            return CapabilityFailure(error.code, str(error))
        return None


def graph_task_capability_declarations(
    owner: JobOwner,
    transcripts: TaskTranscriptReader,
    *,
    clock: Callable[[], datetime],
    id_factory: Callable[[str], str],
) -> GraphTaskCapabilityDeclarations:
    common_output = {
        "type": "object",
        "properties": {
            "record_id": {"type": "string", "minLength": 1},
            "job_id": {"type": "string", "minLength": 1},
            "task_id": {"type": "string", "minLength": 1},
            "attempt_id": {"type": "string", "minLength": 1},
            "fencing_epoch": {"type": "integer", "minimum": 1},
            "committed_task_revision": {"type": "integer", "minimum": 2},
        },
        "required": ["record_id"],
        "additionalProperties": False,
    }
    checkpoint = Capability(
        id=TASK_CHECKPOINT_CAPABILITY_ID,
        description="Persist one bounded progress checkpoint for this exact task attempt.",
        input_schema={
            "type": "object",
            "properties": {
                "milestone": {"type": "string", "minLength": 1, "maxLength": 512},
                "payload": {"type": "object"},
            },
            "required": ["milestone", "payload"],
            "additionalProperties": False,
        },
        output_kind="graph.task_checkpoint",
        output_schema=common_output,
        executor_id=TaskCheckpointExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    comment = Capability(
        id=TASK_COMMENT_CAPABILITY_ID,
        description="Persist one bounded comment on this exact graph task.",
        input_schema={
            "type": "object",
            "properties": {
                "body": {"type": "string", "minLength": 1, "maxLength": 8192}
            },
            "required": ["body"],
            "additionalProperties": False,
        },
        output_kind="graph.task_comment",
        output_schema=common_output,
        executor_id=TaskCommentExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    complete = Capability(
        id=TASK_COMPLETE_CAPABILITY_ID,
        description="Accept one authenticated result and complete this exact task attempt.",
        input_schema={
            "type": "object",
            "properties": {
                "result_kind": {"type": "string", "minLength": 1, "maxLength": 256},
                "summary": {"type": "string", "minLength": 1, "maxLength": 16384},
                "payload": {"type": "object"},
                "evidence_call_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 64,
                    "uniqueItems": True,
                },
                "artifact_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 64,
                    "uniqueItems": True,
                },
                "residual_risk": {"type": ["string", "null"], "maxLength": 16384},
                "downstream_constraints": {"type": "object"},
            },
            "required": [
                "result_kind",
                "summary",
                "payload",
                "evidence_call_ids",
                "artifact_ids",
            ],
            "additionalProperties": False,
        },
        output_kind="graph.task_complete",
        output_schema=common_output,
        executor_id=TaskCompleteExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        machine_run_directive_kind=MachineRunDirectiveKind.COMPLETE,
    )
    block = Capability(
        id=TASK_BLOCK_CAPABILITY_ID,
        description="Block this task with one durable typed control request.",
        input_schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        ControlKind.NEEDS_INPUT.value,
                        ControlKind.NEEDS_AUTHORIZATION.value,
                        ControlKind.NEEDS_REPLAN.value,
                        ControlKind.CAPABILITY_UNAVAILABLE.value,
                        ControlKind.SOURCE_OR_CONTRACT_DRIFT.value,
                        ControlKind.BUDGET_EXHAUSTED.value,
                    ],
                },
                "message": {"type": "string", "minLength": 1, "maxLength": 16384},
                "details": {"type": "object"},
            },
            "required": ["kind", "message", "details"],
            "additionalProperties": False,
        },
        output_kind="graph.task_block",
        output_schema=common_output,
        executor_id=TaskBlockExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        machine_run_directive_kind=MachineRunDirectiveKind.BLOCK,
    )
    review = Capability(
        id=TASK_REQUEST_REVIEW_CAPABILITY_ID,
        description="Request bounded human review and terminate this exact task attempt.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 1, "maxLength": 16384},
                "details": {"type": "object"},
            },
            "required": ["message", "details"],
            "additionalProperties": False,
        },
        output_kind="graph.task_review_request",
        output_schema=common_output,
        executor_id=TaskRequestReviewExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        machine_run_directive_kind=MachineRunDirectiveKind.REQUEST_REVIEW,
    )
    finalizer = Capability(
        id=GRAPH_RESULT_FINALIZE_CAPABILITY_ID,
        description="Aggregate authenticated accepted task results for final delivery.",
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "minLength": 1},
                "work_results": {"type": "array", "maxItems": 63},
            },
            "required": ["job_id", "work_results"],
            "additionalProperties": False,
        },
        output_kind="graph.result_finalized",
        output_schema={"type": "object", "additionalProperties": True},
        executor_id=GraphResultFinalizeExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"application/json"}),
            allowed_authorships=frozenset({ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS}),
            allowed_extensions=(("application/json", (".json",)),),
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=1024 * 1024,
            max_total_bytes_per_call=1024 * 1024,
        ),
    )
    capabilities = (checkpoint, comment, complete, block, review, finalizer)
    # The owner is deliberately shared by all executors; each command still
    # crosses JobOwner before reaching the graph store.
    owner_value = owner
    executors: tuple[Executor, ...] = (
        TaskCheckpointExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        TaskCommentExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        TaskCompleteExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        TaskBlockExecutor(owner_value, transcripts, clock=clock, id_factory=id_factory),
        TaskRequestReviewExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        GraphResultFinalizeExecutor(),
    )
    names = {
        TASK_CHECKPOINT_CAPABILITY_ID: TASK_CHECKPOINT_TOOL_NAME,
        TASK_COMMENT_CAPABILITY_ID: TASK_COMMENT_TOOL_NAME,
        TASK_COMPLETE_CAPABILITY_ID: TASK_COMPLETE_TOOL_NAME,
        TASK_BLOCK_CAPABILITY_ID: TASK_BLOCK_TOOL_NAME,
        TASK_REQUEST_REVIEW_CAPABILITY_ID: TASK_REQUEST_REVIEW_TOOL_NAME,
    }
    views = tuple(
        ToolView(
            name=names[item.id],
            capability_id=item.id,
            description=item.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.JOBS,
                load_mode=ToolLoadMode.PINNED,
                text_trust=ToolTextTrust.CODE,
                summary=item.description,
                when_to_use="Use only for the current exact graph task attempt.",
                keywords=("task", "lifecycle", item.id.rsplit("_", 1)[-1]),
            ),
        )
        for item in capabilities
        if item.id != GRAPH_RESULT_FINALIZE_CAPABILITY_ID
    )
    return GraphTaskCapabilityDeclarations(capabilities, executors, views)


def _authenticate_evidence(
    transcript: Transcript, evidence_call_ids: tuple[object, ...]
) -> tuple[tuple[dict[str, object], ...], ModelSensitivity, set[str]]:
    if any(not isinstance(item, str) for item in evidence_call_ids):
        raise CapabilityInputError(
            "task_result_evidence_invalid", "Task evidence call IDs are invalid."
        )
    requested = tuple(str(item) for item in evidence_call_ids)
    pairs = {call.id: (call, result) for call, result in transcript.tool_pairs}
    authenticated: list[dict[str, object]] = []
    sensitivity = ModelSensitivity.PUBLIC
    artifact_ids: set[str] = set()
    for call_id in requested:
        pair = pairs.get(call_id)
        if pair is None or pair[1] is None or pair[1].is_error:
            raise CapabilityInputError(
                "task_result_evidence_invalid",
                "A cited task result is missing, failed, or belongs to another run.",
            )
        call, result = pair
        assert result is not None
        if result.capability_id is None or result.output_sha256 is None:
            raise CapabilityInputError(
                "task_result_evidence_invalid",
                "A cited task result lacks authenticated execution lineage.",
            )
        sensitivity = max(
            sensitivity,
            result.sensitivity or ModelSensitivity.PUBLIC,
            key=lambda item: item.routing_rank,
        )
        artifact = result.output.get("artifact")
        if isinstance(artifact, Mapping):
            artifact_id = artifact.get("artifact_id")
            if isinstance(artifact_id, str):
                artifact_ids.add(artifact_id)
        authenticated.append(
            {
                "call_id": call_id,
                "tool_name": call.name,
                "capability_id": result.capability_id,
                "executor_id": result.executor_id,
                "output_sha256": result.output_sha256,
            }
        )
    return tuple(authenticated), sensitivity, artifact_ids


def _lifecycle_output(
    kind: str,
    record_id: str,
    sensitivity: ModelSensitivity,
    binding_digest: str,
) -> ToolOutput:
    return ToolOutput(
        kind=kind,
        data={"record_id": record_id},
        sensitivity=sensitivity,
        sensitivity_provenance={
            "authority": "fenced_graph_task_lifecycle",
            "task_binding_digest": binding_digest,
        },
    )


async def _termination_output(
    owner: JobOwner,
    kind: str,
    record_id: str,
    sensitivity: ModelSensitivity,
    guard: TaskAttemptGuard,
) -> ToolOutput:
    inspection = await owner.inspect_graph(guard.binding.job_id)
    task = (
        None
        if inspection is None
        else next(
            (
                item
                for item in inspection.tasks
                if item.task_id == guard.binding.task_id
            ),
            None,
        )
    )
    if task is None or task.current_attempt_id is not None:
        raise CapabilityInputError(
            "task_termination_not_committed",
            "The lifecycle record was not committed with its exact task transition.",
        )
    return ToolOutput(
        kind=kind,
        data={
            "record_id": record_id,
            "job_id": guard.binding.job_id,
            "task_id": guard.binding.task_id,
            "attempt_id": guard.binding.attempt_id,
            "fencing_epoch": guard.binding.fencing_epoch,
            "committed_task_revision": task.task_revision,
        },
        sensitivity=sensitivity,
        sensitivity_provenance={
            "authority": "fenced_graph_task_lifecycle",
            "task_binding_digest": guard.binding.digest,
        },
    )


__all__ = [
    "GRAPH_TASK_DOMAIN_OWNER_ID",
    "GRAPH_RESULT_FINALIZE_CAPABILITY_ID",
    "GraphTaskCapabilityDeclarations",
    "GraphTaskCapabilityDomain",
    "TASK_BLOCK_CAPABILITY_ID",
    "TASK_CHECKPOINT_CAPABILITY_ID",
    "TASK_COMMENT_CAPABILITY_ID",
    "TASK_COMPLETE_CAPABILITY_ID",
    "TASK_REQUEST_REVIEW_CAPABILITY_ID",
    "graph_task_capability_declarations",
]
