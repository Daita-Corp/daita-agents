"""Graph model-task lifecycle capabilities for the unreleased Phase 4 slice."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
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
    ExecutionContractBindings,
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
    BudgetAmount,
    ControlKind,
    ControlState,
    EdgeKind,
    GraphAuthority,
    GraphInspection,
    GraphMutationRequest,
    GraphTask,
    GraphTaskSpecification,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskDependency,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
    canonical_digest,
)

GRAPH_TASK_DOMAIN_OWNER_ID = "jobs.graph_task"
TASK_CHECKPOINT_CAPABILITY_ID = "jobs.graph.task_checkpoint"
TASK_COMMENT_CAPABILITY_ID = "jobs.graph.task_comment"
TASK_COMPLETE_CAPABILITY_ID = "jobs.graph.task_complete"
TASK_BLOCK_CAPABILITY_ID = "jobs.graph.task_block"
TASK_REQUEST_REVIEW_CAPABILITY_ID = "jobs.graph.task_request_review"
GRAPH_RESULT_FINALIZE_CAPABILITY_ID = "jobs.graph.result_finalize"
PLANNER_LIST_TASKS_CAPABILITY_ID = "jobs.graph.planner_list_tasks"
PLANNER_INSPECT_TASK_CAPABILITY_ID = "jobs.graph.planner_inspect_task"
PLANNER_CREATE_CHILDREN_CAPABILITY_ID = "jobs.graph.planner_create_children"
PLANNER_ADD_DEPENDENCIES_CAPABILITY_ID = "jobs.graph.planner_add_dependencies"
PLANNER_SUPERSEDE_CAPABILITY_ID = "jobs.graph.planner_supersede_unstarted"
PLANNER_REQUEST_INPUT_CAPABILITY_ID = "jobs.graph.planner_request_input"
PLANNER_CAPABILITY_IDS = (
    PLANNER_ADD_DEPENDENCIES_CAPABILITY_ID,
    PLANNER_CREATE_CHILDREN_CAPABILITY_ID,
    PLANNER_INSPECT_TASK_CAPABILITY_ID,
    PLANNER_LIST_TASKS_CAPABILITY_ID,
    PLANNER_REQUEST_INPUT_CAPABILITY_ID,
    PLANNER_SUPERSEDE_CAPABILITY_ID,
)

TASK_CHECKPOINT_TOOL_NAME = "task_checkpoint"
TASK_COMMENT_TOOL_NAME = "task_comment"
TASK_COMPLETE_TOOL_NAME = "task_complete"
TASK_BLOCK_TOOL_NAME = "task_block"
TASK_REQUEST_REVIEW_TOOL_NAME = "task_request_review"
PLANNER_LIST_TASKS_TOOL_NAME = "graph_list_tasks"
PLANNER_INSPECT_TASK_TOOL_NAME = "graph_inspect_task"
PLANNER_CREATE_CHILDREN_TOOL_NAME = "graph_create_children"
PLANNER_ADD_DEPENDENCIES_TOOL_NAME = "graph_add_dependencies"
PLANNER_SUPERSEDE_TOOL_NAME = "graph_supersede_unstarted"
PLANNER_REQUEST_INPUT_TOOL_NAME = "graph_request_input"


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
        _inspection, _task, _attempt = await self._current(request)
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
        _inspection, task, _attempt = await self._current(request)
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
        kind = self.control_kind
        if self.control_kind is not ControlKind.REVIEW_REQUESTED:
            raw_kind = request.arguments["kind"]
            assert isinstance(raw_kind, str)
            kind = ControlKind(raw_kind)
        return await self._execute_kind(request, kind)

    async def _execute_kind(
        self, request: ToolExecution, kind: ControlKind
    ) -> ToolOutput:
        inspection, _task, _attempt = await self._current(request)
        guard = request.task_attempt_guard
        assert guard is not None
        message = request.arguments["message"]
        details = request.arguments.get("details", {})
        assert isinstance(message, str) and isinstance(details, Mapping)
        payload: dict[str, object] = {"message": message, "details": details}
        if kind is ControlKind.NEEDS_INPUT:
            payload.update(
                {
                    "question": message,
                    "response_schema": details.get(
                        "response_schema",
                        {"type": "object", "additionalProperties": True},
                    ),
                    "choices": details.get("choices", ()),
                    "why_blocked": details.get("why_blocked", message),
                    "affected_downstream_task_ids": details.get(
                        "affected_downstream_task_ids", ()
                    ),
                    "sensitivity": request.request_sensitivity.value,
                    "evidence_references": details.get("evidence_references", ()),
                    "expires_at": inspection.job.deadline_at.isoformat(),
                    "default_behavior": "remain_blocked",
                }
            )
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


class PlannerRequestInputExecutor(_TaskControlExecutor):
    executor_id = "jobs.graph.planner_request_input.executor"
    control_kind = ControlKind.NEEDS_INPUT
    output_kind = "graph.planner_input_request"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        return await self._execute_kind(request, ControlKind.NEEDS_INPUT)


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


class _PlannerExecutor(_LifecycleExecutor):
    async def _planner_current(
        self, request: ToolExecution
    ) -> tuple[GraphInspection, GraphTask, TaskAttempt]:
        inspection, task, attempt = await self._current(request)
        if task.role is not TaskRole.PLANNER:
            raise CapabilityInputError(
                "planner_role_required", "This graph operation requires a planner task."
            )
        return inspection, task, attempt


class PlannerListTasksExecutor(_PlannerExecutor):
    executor_id = "jobs.graph.planner_list_tasks.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        inspection, _task, _attempt = await self._planner_current(request)
        tasks = tuple(
            {
                "task_id": item.task_id,
                "role": item.role.value,
                "state": item.state.value,
                "title": item.specification.title,
                "task_revision": item.task_revision,
                "attempt_count": item.attempt_count,
                "failure_streak": item.failure_streak,
                "superseded_by_task_id": item.superseded_by_task_id,
                "latest_control_id": item.latest_control_id,
                "latest_result_id": item.latest_result_id,
            }
            for item in inspection.tasks[:64]
        )
        return _planner_output(
            "graph.task_list",
            {"graph_revision": inspection.graph.revision, "tasks": tasks},
            request,
        )


class PlannerInspectTaskExecutor(_PlannerExecutor):
    executor_id = "jobs.graph.planner_inspect_task.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        inspection, _task, _attempt = await self._planner_current(request)
        task_id = request.arguments["task_id"]
        assert isinstance(task_id, str)
        task = next(
            (item for item in inspection.tasks if item.task_id == task_id), None
        )
        if task is None:
            raise CapabilityInputError(
                "unknown_graph_task", "The requested task is not in this graph."
            )
        results = tuple(
            {
                "result_id": item.result_id,
                "result_kind": item.result_kind,
                "summary": item.summary,
                "result_digest": item.result_digest,
                "artifact_ids": item.artifact_ids,
            }
            for item in inspection.results
            if item.task_id == task_id
        )
        controls = tuple(
            {
                "control_id": item.control_id,
                "kind": item.kind.value,
                "state": item.state.value,
                "payload": item.payload,
                "payload_digest": item.payload_digest,
            }
            for item in inspection.controls
            if item.task_id == task_id
        )[-16:]
        return _planner_output(
            "graph.task_inspection",
            {
                "graph_revision": inspection.graph.revision,
                "task": {
                    "task_id": task.task_id,
                    "role": task.role.value,
                    "state": task.state.value,
                    "specification": task.specification.digest_material(),
                    "attempt_count": task.attempt_count,
                    "failure_streak": task.failure_streak,
                    "latest_checkpoint_id": task.latest_checkpoint_id,
                },
                "results": results,
                "controls": controls,
            },
            request,
        )


class _PlannerMutationExecutor(_PlannerExecutor):
    async def _commit(
        self,
        request: ToolExecution,
        inspection: GraphInspection,
        task: GraphTask,
        attempt: TaskAttempt,
        *,
        tasks: tuple[GraphTask, ...] = (),
        dependencies: tuple[TaskDependency, ...] = (),
        supersessions: tuple[tuple[str, str], ...] = (),
    ) -> ToolOutput:
        guard = request.task_attempt_guard
        assert guard is not None
        idempotency_key = request.arguments["idempotency_key"]
        expected_revision = request.arguments["expected_revision"]
        assert isinstance(idempotency_key, str) and isinstance(expected_revision, int)
        mutation = GraphMutationRequest(
            agent_id=task.agent_id,
            job_id=task.job_id,
            mutation_id=_stable_graph_id(
                "mutation",
                job_id=task.job_id,
                actor_key=task.task_id,
                idempotency_key=idempotency_key,
                discriminator="mutation",
            ),
            actor_kind="planner_attempt",
            actor_key=task.task_id,
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            created_at=self._clock(),
            tasks=tasks,
            dependencies=dependencies,
            supersessions=supersessions,
            creator_task_id=task.task_id,
            creator_attempt_id=attempt.attempt_id,
            claim_token=guard.claim_token,
            fencing_epoch=attempt.fencing_epoch,
        )
        await guard.revalidate(
            capability_id=request.capability_id,
            point="before_graph_mutation_commit",
        )
        committed = await self._owner.mutate_graph(mutation)
        return _planner_output(
            "graph.mutation",
            {
                "record_id": committed.mutation_id,
                "decision": committed.decision.value,
                "failure_code": committed.failure_code,
                "expected_revision": committed.expected_revision,
                "committed_revision": committed.committed_revision,
                "task_ids": committed.resulting_task_ids,
                "edges": committed.resulting_edges,
            },
            request,
        )


class PlannerCreateChildrenExecutor(_PlannerMutationExecutor):
    executor_id = "jobs.graph.planner_create_children.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        inspection, planner, attempt = await self._planner_current(request)
        raw_children = request.arguments["children"]
        idempotency_key = request.arguments["idempotency_key"]
        assert isinstance(raw_children, tuple)
        assert isinstance(idempotency_key, str)
        created_at = self._clock()
        tasks: list[GraphTask] = []
        edges: list[TaskDependency] = []
        client_ids: dict[str, str] = {}
        for raw in raw_children:
            if not isinstance(raw, Mapping):
                raise CapabilityInputError(
                    "planner_child_invalid", "A planner child proposal is malformed."
                )
            client_key = raw["client_key"]
            assert isinstance(client_key, str)
            if client_key in client_ids:
                raise CapabilityInputError(
                    "planner_child_invalid", "Planner child keys must be unique."
                )
            child_id = _stable_graph_id(
                "task",
                job_id=planner.job_id,
                actor_key=planner.task_id,
                idempotency_key=idempotency_key,
                discriminator=f"child:{client_key}",
            )
            client_ids[client_key] = child_id
            child = _planner_child_task(
                inspection,
                planner,
                raw,
                task_id=child_id,
                created_at=created_at,
            )
            tasks.append(child)
            parent_ids = raw.get("parent_task_ids", ())
            assert isinstance(parent_ids, tuple)
            edges.extend(
                TaskDependency(
                    agent_id=planner.agent_id,
                    job_id=planner.job_id,
                    upstream_task_id=str(parent_id),
                    downstream_task_id=child_id,
                    edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                    created_at=created_at,
                    creator_key=planner.task_id,
                )
                for parent_id in parent_ids
            )
        output = await self._commit(
            request,
            inspection,
            planner,
            attempt,
            tasks=tuple(tasks),
            dependencies=tuple(edges),
        )
        return ToolOutput(
            kind=output.kind,
            data={**dict(output.data), "client_task_ids": client_ids},
            sensitivity=output.sensitivity,
            sensitivity_provenance=output.sensitivity_provenance,
        )


class PlannerAddDependenciesExecutor(_PlannerMutationExecutor):
    executor_id = "jobs.graph.planner_add_dependencies.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        inspection, planner, attempt = await self._planner_current(request)
        raw_edges = request.arguments["dependencies"]
        assert isinstance(raw_edges, tuple)
        created_at = self._clock()
        edges = tuple(
            TaskDependency(
                agent_id=planner.agent_id,
                job_id=planner.job_id,
                upstream_task_id=str(raw["upstream_task_id"]),
                downstream_task_id=str(raw["downstream_task_id"]),
                edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                created_at=created_at,
                creator_key=planner.task_id,
            )
            for raw in raw_edges
            if isinstance(raw, Mapping)
        )
        if len(edges) != len(raw_edges):
            raise CapabilityInputError(
                "planner_dependency_invalid", "A dependency proposal is malformed."
            )
        return await self._commit(
            request, inspection, planner, attempt, dependencies=edges
        )


class PlannerSupersedeExecutor(_PlannerMutationExecutor):
    executor_id = "jobs.graph.planner_supersede_unstarted.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        inspection, planner, attempt = await self._planner_current(request)
        replaced_id = request.arguments["task_id"]
        raw = request.arguments["replacement"]
        idempotency_key = request.arguments["idempotency_key"]
        assert isinstance(replaced_id, str) and isinstance(raw, Mapping)
        assert isinstance(idempotency_key, str)
        created_at = self._clock()
        replacement_id = _stable_graph_id(
            "task",
            job_id=planner.job_id,
            actor_key=planner.task_id,
            idempotency_key=idempotency_key,
            discriminator=f"supersede:{replaced_id}",
        )
        original_parent_ids = {
            edge.upstream_task_id
            for edge in inspection.dependencies
            if edge.downstream_task_id == replaced_id
        }
        proposed_parent_ids = raw.get("parent_task_ids", ())
        assert isinstance(proposed_parent_ids, tuple)
        parent_ids = tuple(sorted(original_parent_ids | set(proposed_parent_ids)))
        replacement_input = {**dict(raw), "parent_task_ids": parent_ids}
        replacement = _planner_child_task(
            inspection,
            planner,
            replacement_input,
            task_id=replacement_id,
            created_at=created_at,
        )
        replacement = replace(replacement, supersedes_task_id=replaced_id)
        edges = tuple(
            TaskDependency(
                agent_id=planner.agent_id,
                job_id=planner.job_id,
                upstream_task_id=str(parent_id),
                downstream_task_id=replacement_id,
                edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
                created_at=created_at,
                creator_key=planner.task_id,
            )
            for parent_id in parent_ids
        )
        return await self._commit(
            request,
            inspection,
            planner,
            attempt,
            tasks=(replacement,),
            dependencies=edges,
            supersessions=((replaced_id, replacement_id),),
        )


def _planner_output(
    kind: str,
    data: Mapping[str, object],
    request: ToolExecution,
) -> ToolOutput:
    guard = request.task_attempt_guard
    assert guard is not None
    return ToolOutput(
        kind=kind,
        data=data,
        sensitivity=request.request_sensitivity,
        sensitivity_provenance={
            "authority": "fenced_graph_planner",
            "task_binding_digest": guard.binding.digest,
        },
    )


def _stable_graph_id(
    prefix: str,
    *,
    job_id: str,
    actor_key: str,
    idempotency_key: str,
    discriminator: str,
) -> str:
    digest = canonical_digest(
        {
            "job_id": job_id,
            "actor_key": actor_key,
            "idempotency_key": idempotency_key,
            "discriminator": discriminator,
        }
    )
    return f"{prefix}-{digest.removeprefix('sha256:')[:32]}"


def _planner_child_task(
    inspection: GraphInspection,
    planner: GraphTask,
    raw: Mapping[str, object],
    *,
    task_id: str,
    created_at: datetime,
) -> GraphTask:
    title = raw.get("title")
    description = raw.get("description")
    result_kind = raw.get("result_kind")
    reduction = raw.get("reduction", False)
    capability_id = raw.get("capability_id")
    arguments = raw.get("arguments", {})
    input_result_ids = raw.get("input_result_ids", ())
    parent_ids = raw.get("parent_task_ids", ())
    if (
        not isinstance(title, str)
        or not isinstance(description, str)
        or not isinstance(result_kind, str)
        or not isinstance(reduction, bool)
        or not isinstance(arguments, Mapping)
        or not isinstance(input_result_ids, tuple)
        or any(not isinstance(item, str) for item in input_result_ids)
        or not isinstance(parent_ids, tuple)
        or any(not isinstance(item, str) for item in parent_ids)
    ):
        raise CapabilityInputError(
            "planner_child_invalid", "A planner child proposal is malformed."
        )
    if len(parent_ids) > inspection.job.specification.limits.max_direct_parents:
        raise CapabilityInputError(
            "parent_limit", "The proposed task exceeds the direct-parent bound."
        )
    lifecycle_ids = tuple(
        capability_id
        for capability_id in (
            TASK_CHECKPOINT_CAPABILITY_ID,
            TASK_COMMENT_CAPABILITY_ID,
            TASK_COMPLETE_CAPABILITY_ID,
            TASK_BLOCK_CAPABILITY_ID,
            TASK_REQUEST_REVIEW_CAPABILITY_ID,
        )
        if capability_id in planner.specification.authority.capability_ids
    )
    if reduction:
        if capability_id is not None or len(parent_ids) < 2:
            raise CapabilityInputError(
                "planner_reduction_invalid",
                "A reduction task needs at least two parents and no executor call.",
            )
        child_capability_ids = lifecycle_ids
    else:
        if (
            not isinstance(capability_id, str)
            or capability_id.startswith("jobs.graph.")
            or capability_id not in planner.specification.authority.capability_ids
        ):
            raise CapabilityInputError(
                "planner_capability_invalid",
                "The child capability is outside the planner's immutable ceiling.",
            )
        child_capability_ids = tuple(sorted((*lifecycle_ids, capability_id)))
    planner_authority = planner.specification.authority
    material = planner_authority.contract_bindings
    raw_capabilities = material.get("capability_contracts", {})
    raw_resources = material.get("resource_revisions", {})
    raw_routes = material.get("model_routes", {})
    raw_origins = material.get("tool_origins", {})
    if not isinstance(raw_capabilities, Mapping):
        raise CapabilityInputError(
            "planner_contract_invalid", "The planner contract ceiling is malformed."
        )
    if not isinstance(raw_resources, Mapping):
        raise CapabilityInputError(
            "planner_contract_invalid", "The planner contract ceiling is malformed."
        )
    if not isinstance(raw_routes, Mapping):
        raise CapabilityInputError(
            "planner_contract_invalid", "The planner contract ceiling is malformed."
        )
    if not isinstance(raw_origins, Mapping):
        raise CapabilityInputError(
            "planner_contract_invalid", "The planner contract ceiling is malformed."
        )
    if not planner_authority.model_route_ids:
        raise CapabilityInputError(
            "planner_contract_invalid", "The planner has no admitted model route."
        )
    route_id = planner_authority.model_route_ids[0]
    missing_capabilities = tuple(
        item for item in child_capability_ids if item not in raw_capabilities
    )
    if missing_capabilities or route_id not in raw_routes:
        raise CapabilityInputError(
            "planner_contract_invalid",
            "The proposed child is not covered by the planner contract ceiling.",
        )
    bindings = ExecutionContractBindings(
        capability_contracts={
            item: str(raw_capabilities[item]) for item in child_capability_ids
        },
        resource_revisions=dict(raw_resources),
        model_routes={route_id: str(raw_routes[route_id])},
        tool_origins={
            item: str(value)
            for item, value in raw_origins.items()
            if item in child_capability_ids
        },
    )
    authority = GraphAuthority(
        source_ids=planner_authority.source_ids,
        resource_ids=planner_authority.resource_ids,
        connector_ids=planner_authority.connector_ids,
        capability_ids=child_capability_ids,
        access_modes=planner_authority.access_modes,
        operational_effects=("none",),
        model_route_ids=(route_id,),
        sensitivity=planner_authority.sensitivity,
        contract_bindings=bindings.material(),
    )
    expected: dict[str, object] = {
        "kind": "model_task",
        "result_kind": result_kind,
        "model_route_id": route_id,
        "per_run_max_tokens": planner.specification.expected_result_contract[
            "per_run_max_tokens"
        ],
        "per_run_max_cost_usd": planner.specification.expected_result_contract[
            "per_run_max_cost_usd"
        ],
        "attempt_budgets": {"work_units": 1},
        "input_result_ids": input_result_ids,
        "reduction": reduction,
    }
    if not reduction:
        expected["initial_call"] = {
            "capability_id": capability_id,
            "arguments": arguments,
        }
    spec = GraphTaskSpecification(
        title=title,
        description=description,
        expected_result_contract=expected,
        authority=authority,
        budgets=(BudgetAmount("work_units", 3),),
        max_steps=planner.specification.max_steps,
        max_wall_time_seconds=planner.specification.max_wall_time_seconds,
        created_by=planner.task_id,
    )
    return GraphTask(
        agent_id=planner.agent_id,
        job_id=planner.job_id,
        task_id=task_id,
        state=TaskState.PENDING if parent_ids else TaskState.READY,
        role=TaskRole.WORKER,
        execution_kind=TaskExecutionKind.MODEL,
        priority=100,
        not_before=None,
        current_attempt_id=None,
        task_revision=1,
        specification=spec,
        task_spec_digest=spec.digest,
        task_scope_digest=authority.digest,
        attempt_count=0,
        failure_streak=0,
        fencing_epoch=0,
        created_at=created_at,
        updated_at=created_at,
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
    planner_read_output = {"type": "object", "additionalProperties": True}
    planner_list = Capability(
        id=PLANNER_LIST_TASKS_CAPABILITY_ID,
        description="List the bounded tasks in this planner's own graph.",
        input_schema={"type": "object", "additionalProperties": False},
        output_kind="graph.task_list",
        output_schema=planner_read_output,
        executor_id=PlannerListTasksExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    planner_inspect = Capability(
        id=PLANNER_INSPECT_TASK_CAPABILITY_ID,
        description="Inspect one exact task in this planner's own graph.",
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string", "minLength": 1}},
            "required": ["task_id"],
            "additionalProperties": False,
        },
        output_kind="graph.task_inspection",
        output_schema=planner_read_output,
        executor_id=PlannerInspectTaskExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    child_schema = {
        "type": "object",
        "properties": {
            "client_key": {"type": "string", "minLength": 1, "maxLength": 128},
            "title": {"type": "string", "minLength": 1, "maxLength": 512},
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": 16384,
            },
            "result_kind": {"type": "string", "minLength": 1, "maxLength": 256},
            "capability_id": {"type": ["string", "null"], "minLength": 1},
            "arguments": {"type": "object"},
            "parent_task_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 16,
                "uniqueItems": True,
            },
            "input_result_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 16,
                "uniqueItems": True,
            },
            "reduction": {"type": "boolean"},
        },
        "required": [
            "client_key",
            "title",
            "description",
            "result_kind",
            "arguments",
            "parent_task_ids",
            "input_result_ids",
            "reduction",
        ],
        "additionalProperties": False,
    }
    mutation_common = {
        "expected_revision": {"type": "integer", "minimum": 0},
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 256},
    }
    planner_create = Capability(
        id=PLANNER_CREATE_CHILDREN_CAPABILITY_ID,
        description="Atomically create a bounded batch of child tasks and dependencies.",
        input_schema={
            "type": "object",
            "properties": {
                **mutation_common,
                "children": {
                    "type": "array",
                    "items": child_schema,
                    "minItems": 1,
                    "maxItems": 16,
                },
            },
            "required": ["expected_revision", "idempotency_key", "children"],
            "additionalProperties": False,
        },
        output_kind="graph.mutation",
        output_schema=planner_read_output,
        executor_id=PlannerCreateChildrenExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    planner_dependencies = Capability(
        id=PLANNER_ADD_DEPENDENCIES_CAPABILITY_ID,
        description="Atomically add bounded same-graph dependencies to pending work.",
        input_schema={
            "type": "object",
            "properties": {
                **mutation_common,
                "dependencies": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "upstream_task_id": {"type": "string", "minLength": 1},
                            "downstream_task_id": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "required": ["upstream_task_id", "downstream_task_id"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["expected_revision", "idempotency_key", "dependencies"],
            "additionalProperties": False,
        },
        output_kind="graph.mutation",
        output_schema=planner_read_output,
        executor_id=PlannerAddDependenciesExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    planner_supersede = Capability(
        id=PLANNER_SUPERSEDE_CAPABILITY_ID,
        description="Replace one non-running task without rewriting its specification.",
        input_schema={
            "type": "object",
            "properties": {
                **mutation_common,
                "task_id": {"type": "string", "minLength": 1},
                "replacement": child_schema,
            },
            "required": [
                "expected_revision",
                "idempotency_key",
                "task_id",
                "replacement",
            ],
            "additionalProperties": False,
        },
        output_kind="graph.mutation",
        output_schema=planner_read_output,
        executor_id=PlannerSupersedeExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    planner_input = Capability(
        id=PLANNER_REQUEST_INPUT_CAPABILITY_ID,
        description="Request bounded human input for this exact planner attempt.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 1, "maxLength": 16384},
                "details": {"type": "object"},
            },
            "required": ["message", "details"],
            "additionalProperties": False,
        },
        output_kind="graph.planner_input_request",
        output_schema=common_output,
        executor_id=PlannerRequestInputExecutor.executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        machine_run_directive_kind=MachineRunDirectiveKind.BLOCK,
    )
    capabilities = (
        checkpoint,
        comment,
        complete,
        block,
        review,
        finalizer,
        planner_list,
        planner_inspect,
        planner_create,
        planner_dependencies,
        planner_supersede,
        planner_input,
    )
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
        PlannerListTasksExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        PlannerInspectTaskExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        PlannerCreateChildrenExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        PlannerAddDependenciesExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        PlannerSupersedeExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
        PlannerRequestInputExecutor(
            owner_value, transcripts, clock=clock, id_factory=id_factory
        ),
    )
    names = {
        TASK_CHECKPOINT_CAPABILITY_ID: TASK_CHECKPOINT_TOOL_NAME,
        TASK_COMMENT_CAPABILITY_ID: TASK_COMMENT_TOOL_NAME,
        TASK_COMPLETE_CAPABILITY_ID: TASK_COMPLETE_TOOL_NAME,
        TASK_BLOCK_CAPABILITY_ID: TASK_BLOCK_TOOL_NAME,
        TASK_REQUEST_REVIEW_CAPABILITY_ID: TASK_REQUEST_REVIEW_TOOL_NAME,
        PLANNER_LIST_TASKS_CAPABILITY_ID: PLANNER_LIST_TASKS_TOOL_NAME,
        PLANNER_INSPECT_TASK_CAPABILITY_ID: PLANNER_INSPECT_TASK_TOOL_NAME,
        PLANNER_CREATE_CHILDREN_CAPABILITY_ID: PLANNER_CREATE_CHILDREN_TOOL_NAME,
        PLANNER_ADD_DEPENDENCIES_CAPABILITY_ID: PLANNER_ADD_DEPENDENCIES_TOOL_NAME,
        PLANNER_SUPERSEDE_CAPABILITY_ID: PLANNER_SUPERSEDE_TOOL_NAME,
        PLANNER_REQUEST_INPUT_CAPABILITY_ID: PLANNER_REQUEST_INPUT_TOOL_NAME,
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
                keywords=("task", "lifecycle", item.id),
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
    "GRAPH_RESULT_FINALIZE_CAPABILITY_ID",
    "GRAPH_TASK_DOMAIN_OWNER_ID",
    "PLANNER_CAPABILITY_IDS",
    "TASK_BLOCK_CAPABILITY_ID",
    "TASK_CHECKPOINT_CAPABILITY_ID",
    "TASK_COMMENT_CAPABILITY_ID",
    "TASK_COMPLETE_CAPABILITY_ID",
    "TASK_REQUEST_REVIEW_CAPABILITY_ID",
    "GraphTaskCapabilityDeclarations",
    "GraphTaskCapabilityDomain",
    "graph_task_capability_declarations",
]
