"""Declare and execute bounded job list, inspection, result, and cancellation tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .._json import FrozenJsonObject
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    OperationalEffect,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from ..capability_runtime import CapabilityFailure, SideEffectPlan
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput
from .graph.models import GraphInspection, GraphJob, GraphState, TaskResult, TaskState
from .owner import JobError, JobOwner

JOB_DOMAIN_OWNER_ID = "jobs"
JOB_LIST_CAPABILITY_ID = "jobs.list"
JOB_LIST_EXECUTOR_ID = "jobs.list.executor"
JOB_LIST_TOOL_NAME = "job_list"
JOB_INSPECT_CAPABILITY_ID = "jobs.inspect"
JOB_INSPECT_EXECUTOR_ID = "jobs.inspect.executor"
JOB_INSPECT_TOOL_NAME = "job_inspect"
JOB_READ_RESULTS_CAPABILITY_ID = "jobs.read_results"
JOB_READ_RESULTS_EXECUTOR_ID = "jobs.read_results.executor"
JOB_READ_RESULTS_TOOL_NAME = "job_read_results"
JOB_CANCEL_CAPABILITY_ID = "jobs.cancel"
JOB_CANCEL_EXECUTOR_ID = "jobs.cancel.executor"
JOB_CANCEL_TOOL_NAME = "job_cancel"
MAX_JOB_LIST_PAGE_SIZE = 50


@dataclass(frozen=True, slots=True)
class JobCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class _JobExecutor:
    def __init__(self, owner: JobOwner) -> None:
        self._owner = owner


class JobListExecutor(_JobExecutor):
    executor_id = JOB_LIST_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        raw_states = request.arguments.get("states", ())
        assert isinstance(raw_states, tuple)
        states = frozenset(GraphState(item) for item in raw_states)
        jobs = await self._owner.list(
            states=states,
            limit=MAX_JOB_LIST_PAGE_SIZE,
        )
        sensitivity = _job_sensitivity(jobs)
        return ToolOutput(
            kind="job.list",
            data={
                "jobs": tuple(_job_payload(item) for item in jobs),
                "count": len(jobs),
            },
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": "job_owner_agent_scope",
                "agent_id": self._owner.agent_id,
            },
        )


class JobInspectExecutor(_JobExecutor):
    executor_id = JOB_INSPECT_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        inspection = await self._owner.inspect(job_id)
        if inspection is None:
            raise CapabilityInputError(
                "job_not_found",
                "The requested job is not owned by this agent.",
            )
        return ToolOutput(
            kind="job.inspection",
            data=_inspection_payload(inspection),
            sensitivity=inspection.job.specification.authority.sensitivity,
            sensitivity_provenance={
                "authority": "job_owner_agent_scope",
                "agent_id": self._owner.agent_id,
                "job_id": job_id,
            },
        )


class JobReadResultsExecutor(_JobExecutor):
    executor_id = JOB_READ_RESULTS_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        result = await self._owner.read_result(job_id)
        if result is None:
            inspection = await self._owner.inspect(job_id)
            if inspection is None:
                raise CapabilityInputError(
                    "job_not_found",
                    "The requested job is not owned by this agent.",
                )
            raise CapabilityInputError(
                "job_result_not_ready",
                "The requested job does not have a successful result.",
                {"state": inspection.job.state.value},
            )
        return ToolOutput(
            kind="job.result",
            data=_result_payload(result),
            sensitivity=result.sensitivity,
            sensitivity_provenance={
                "authority": "job_owner_agent_scope",
                "agent_id": self._owner.agent_id,
                "job_id": job_id,
                "result_provenance": result.provenance,
            },
        )


class JobCancelExecutor(_JobExecutor):
    executor_id = JOB_CANCEL_EXECUTOR_ID

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        inspection = await self._owner.inspect(job_id)
        if inspection is None:
            raise CapabilityInputError(
                "job_not_found",
                "The requested job is not owned by this agent.",
            )
        return FrozenJsonObject.from_mapping(
            {
                "job_id": job_id,
                "state": inspection.job.state.value,
                "desired_state": inspection.job.desired_state.value,
                "updated_at": inspection.job.updated_at.isoformat(),
                "specification_digest": inspection.job.specification_digest,
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        job = await self._owner.cancel(job_id)
        if job is None:
            raise CapabilityInputError(
                "job_not_found",
                "The requested job is not owned by this agent.",
            )
        return ToolOutput(
            kind="job.cancel_receipt",
            data={
                "job_id": job_id,
                "state": job.state.value,
                "desired_state": job.desired_state.value,
                "updated_at": job.updated_at.isoformat(),
            },
            sensitivity=job.specification.authority.sensitivity,
            sensitivity_provenance={
                "authority": "job_owner_agent_scope",
                "agent_id": self._owner.agent_id,
                "job_id": job_id,
            },
        )


class JobCapabilityDomain:
    domain_owner_id = JOB_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        owner: JobOwner,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("job declarations have the wrong owner")
        self._declarations = declarations
        self._owner = owner
        self._views = tuple(declarations.tool_views)

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        if run.agent_id != self._owner.agent_id:
            return ()
        names = [JOB_LIST_TOOL_NAME]
        if await self._owner.list(limit=1):
            names.extend((JOB_INSPECT_TOOL_NAME, JOB_READ_RESULTS_TOOL_NAME))
        if await self._owner.list(
            states=frozenset(
                {
                    GraphState.QUEUED,
                    GraphState.ACTIVE,
                    GraphState.BLOCKED,
                    GraphState.NEEDS_ATTENTION,
                }
            ),
            limit=1,
        ):
            names.append(JOB_CANCEL_TOOL_NAME)
        projected = frozenset(names)
        return tuple(item.name for item in self._views if item.name in projected)

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
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
        if run.agent_id != self._owner.agent_id:
            raise CapabilityInputError(
                "job_owner_mismatch",
                "The job lifecycle owner does not match this agent.",
            )
        scope = run.execution_scope
        if scope is not None and capability.id in {
            JOB_INSPECT_CAPABILITY_ID,
            JOB_READ_RESULTS_CAPABILITY_ID,
            JOB_CANCEL_CAPABILITY_ID,
        }:
            requested_job_id = arguments.get("job_id")
            if scope.job_id is None or requested_job_id != scope.job_id:
                raise CapabilityInputError(
                    "execution_scope_job_violation",
                    "This run may inspect only its exact bound job.",
                    {"requested_job_id": requested_job_id},
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
            "This domain does not admit unattended external effects.",
        )

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        if capability.id != JOB_CANCEL_CAPABILITY_ID:
            raise ValueError("job domain received an unsupported operational effect")
        return SideEffectPlan(approval_required=False, recheck_after_approval=True)

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
        del run, call, capability, arguments, request_sensitivity
        return output

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        del call
        if isinstance(error, JobError):
            return CapabilityFailure(error.code, str(error))
        return None


def job_capability_declarations(owner: JobOwner) -> JobCapabilityDeclarations:
    list_capability = Capability(
        id=JOB_LIST_CAPABILITY_ID,
        description="List bounded jobs and past profiles owned by this agent.",
        input_schema={
            "type": "object",
            "properties": {
                "states": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [item.value for item in GraphState],
                    },
                    "maxItems": len(GraphState),
                    "uniqueItems": True,
                }
            },
            "additionalProperties": False,
        },
        output_kind="job.list",
        output_schema=_list_schema(),
        executor_id=JOB_LIST_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    inspect_capability = Capability(
        id=JOB_INSPECT_CAPABILITY_ID,
        description="Inspect one exact durable job owned by the current agent.",
        input_schema=_job_id_schema(),
        output_kind="job.inspection",
        output_schema=_object_output_schema(),
        executor_id=JOB_INSPECT_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    result_capability = Capability(
        id=JOB_READ_RESULTS_CAPABILITY_ID,
        description=(
            "Read a successful job's bounded validated stored result and artifact references."
        ),
        input_schema=_job_id_schema(),
        output_kind="job.result",
        output_schema=_object_output_schema(),
        executor_id=JOB_READ_RESULTS_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    cancel_capability = Capability(
        id=JOB_CANCEL_CAPABILITY_ID,
        description="Request cancellation of one exact durable job.",
        input_schema=_job_id_schema(),
        output_kind="job.cancel_receipt",
        output_schema=_cancel_schema(),
        executor_id=JOB_CANCEL_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.CANCEL_JOB,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    capabilities = (
        list_capability,
        inspect_capability,
        result_capability,
        cancel_capability,
    )
    executors: tuple[Executor, ...] = (
        JobListExecutor(owner),
        JobInspectExecutor(owner),
        JobReadResultsExecutor(owner),
        JobCancelExecutor(owner),
    )
    summaries = {
        JOB_LIST_CAPABILITY_ID: (
            "List this agent's jobs.",
            "Find prior profiles or an unknown job ID.",
            ("job", "list", "status", "profile"),
        ),
        JOB_INSPECT_CAPABILITY_ID: (
            "Inspect one durable job lifecycle.",
            "Use only for status, attempts, failures, or execution details.",
            ("job", "inspect", "attempt"),
        ),
        JOB_READ_RESULTS_CAPABILITY_ID: (
            "Read one successful durable job result.",
            "Use first for a known job ID's stored result, not a fresh recomputation.",
            ("job", "result", "artifact", "profile", "previous"),
        ),
        JOB_CANCEL_CAPABILITY_ID: (
            "Cancel one exact durable job.",
            "Use when the current user wants queued or running work to stop.",
            ("job", "cancel", "stop"),
        ),
    }
    names = {
        JOB_LIST_CAPABILITY_ID: JOB_LIST_TOOL_NAME,
        JOB_INSPECT_CAPABILITY_ID: JOB_INSPECT_TOOL_NAME,
        JOB_READ_RESULTS_CAPABILITY_ID: JOB_READ_RESULTS_TOOL_NAME,
        JOB_CANCEL_CAPABILITY_ID: JOB_CANCEL_TOOL_NAME,
    }
    views = tuple(
        ToolView(
            name=names[item.id],
            capability_id=item.id,
            description=item.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.JOBS,
                load_mode=(
                    ToolLoadMode.ON_DEMAND
                    if item.id == JOB_CANCEL_CAPABILITY_ID
                    else ToolLoadMode.PINNED
                ),
                text_trust=ToolTextTrust.CODE,
                summary=summaries[item.id][0],
                when_to_use=summaries[item.id][1],
                keywords=summaries[item.id][2],
            ),
        )
        for item in capabilities
    )
    return JobCapabilityDeclarations(capabilities, executors, views)


def _job_id_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"job_id": {"type": "string", "minLength": 1}},
        "required": ["job_id"],
        "additionalProperties": False,
    }


def _object_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


def _list_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "jobs": {"type": "array"},
            "count": {"type": "integer"},
        },
        "required": ["jobs", "count"],
        "additionalProperties": False,
    }


def _cancel_schema() -> dict[str, object]:
    properties = {
        "job_id": {"type": "string"},
        "state": {"type": "string"},
        "desired_state": {"type": "string"},
        "updated_at": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _job_payload(value: GraphJob) -> dict[str, object]:
    authority = value.specification.authority
    return {
        "job_id": value.job_id,
        "conversation_id": value.conversation_id,
        "state": value.state.value,
        "desired_state": value.desired_state.value,
        "objective": value.specification.objective,
        "source_ids": authority.source_ids,
        "resource_ids": authority.resource_ids,
        "sensitivity": authority.sensitivity.value,
        "created_at": value.created_at.isoformat(),
        "updated_at": value.updated_at.isoformat(),
        "deadline_at": value.deadline_at.isoformat(),
        "result_available": value.terminal_result_id is not None,
        "failure_code": value.failure_code,
    }


def _inspection_payload(value: GraphInspection) -> dict[str, object]:
    task_counts = {
        state.value: sum(task.state is state for task in value.tasks)
        for state in TaskState
    }
    return {
        **_job_payload(value.job),
        "specification_digest": value.job.specification_digest,
        "graph_revision": value.graph.revision,
        "task_count": value.graph.task_count,
        "edge_count": value.graph.edge_count,
        "active_attempt_count": value.graph.active_attempt_count,
        "task_counts": task_counts,
        "tasks": tuple(
            {
                "task_id": task.task_id,
                "title": task.specification.title,
                "state": task.state.value,
                "role": task.role.value,
                "execution_kind": task.execution_kind.value,
                "attempt_count": task.attempt_count,
                "latest_result_id": task.latest_result_id,
                "latest_control_id": task.latest_control_id,
            }
            for task in value.tasks
        ),
        "dependencies": tuple(
            {
                "upstream_task_id": edge.upstream_task_id,
                "downstream_task_id": edge.downstream_task_id,
                "edge_kind": edge.edge_kind.value,
            }
            for edge in value.dependencies
        ),
        "attempts": tuple(
            {
                "attempt_id": attempt.attempt_id,
                "task_id": attempt.task_id,
                "ordinal": attempt.ordinal,
                "fencing_epoch": attempt.fencing_epoch,
                "state": attempt.state.value,
                "run_id": attempt.run_id,
                "lease_expires_at": (
                    None
                    if attempt.lease_expires_at is None
                    else attempt.lease_expires_at.isoformat()
                ),
                "absolute_deadline_at": attempt.absolute_deadline_at.isoformat(),
                "started_at": (
                    None
                    if attempt.started_at is None
                    else attempt.started_at.isoformat()
                ),
                "ended_at": (
                    None if attempt.ended_at is None else attempt.ended_at.isoformat()
                ),
                "heartbeat_at": (
                    None
                    if attempt.heartbeat_at is None
                    else attempt.heartbeat_at.isoformat()
                ),
                "error_code": attempt.error_code,
            }
            for attempt in value.attempts
        ),
        "open_controls": tuple(
            control.control_id
            for control in value.controls
            if control.state.value == "open"
        ),
        "delivery_ids": value.delivery_ids,
        "terminal_at": (
            None if value.job.terminal_at is None else value.job.terminal_at.isoformat()
        ),
    }


def _result_payload(value: TaskResult) -> dict[str, object]:
    return {
        "job_id": value.job_id,
        "task_id": value.task_id,
        "result_id": value.result_id,
        "attempt_id": value.attempt_id,
        "run_id": value.run_id,
        "result_kind": value.result_kind,
        "summary": value.summary,
        "payload": value.payload,
        "sensitivity": value.sensitivity.value,
        "provenance": value.provenance,
        "artifact_ids": value.artifact_ids,
        "verification": value.verification,
        "residual_risk": value.residual_risk,
        "completed_at": value.completed_at.isoformat(),
        "result_digest": value.result_digest,
    }


def _job_sensitivity(values: tuple[GraphJob, ...]) -> ModelSensitivity:
    order = {
        ModelSensitivity.PUBLIC: 0,
        ModelSensitivity.INTERNAL: 1,
        ModelSensitivity.CONFIDENTIAL: 2,
        ModelSensitivity.RESTRICTED: 3,
    }
    return max(
        (item.specification.authority.sensitivity for item in values),
        default=ModelSensitivity.INTERNAL,
        key=order.__getitem__,
    )


__all__ = [
    "JOB_CANCEL_CAPABILITY_ID",
    "JOB_CANCEL_TOOL_NAME",
    "JOB_DOMAIN_OWNER_ID",
    "JOB_INSPECT_CAPABILITY_ID",
    "JOB_INSPECT_TOOL_NAME",
    "JOB_LIST_CAPABILITY_ID",
    "JOB_LIST_TOOL_NAME",
    "JOB_READ_RESULTS_CAPABILITY_ID",
    "JOB_READ_RESULTS_TOOL_NAME",
    "JobCapabilityDeclarations",
    "JobCapabilityDomain",
    "job_capability_declarations",
]
