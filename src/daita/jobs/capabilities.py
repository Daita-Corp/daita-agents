"""Common bounded lifecycle capabilities over the one Stage B job owner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .._json import FrozenJsonObject
from ..artifacts.models import artifact_ref_to_mapping
from ..capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    OperationalEffect,
    ToolDiscoveryMetadata,
    ToolExecution,
    ToolExposureClass,
    ToolOutput,
    ToolView,
)
from ..capability_runtime import CapabilityFailure, SideEffectPlan
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput
from .models import (
    JobInspection,
    JobResultView,
    JobStatus,
    JobSummary,
    MAX_JOB_LIST_PAGE_SIZE,
)
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
        raw_statuses = request.arguments.get("statuses", ())
        assert isinstance(raw_statuses, tuple)
        statuses = frozenset(JobStatus(item) for item in raw_statuses)
        summaries = await self._owner.list(
            statuses=statuses,
            limit=MAX_JOB_LIST_PAGE_SIZE,
        )
        sensitivity = _summary_sensitivity(summaries)
        return ToolOutput(
            kind="job.list",
            data={
                "jobs": tuple(_summary_payload(item) for item in summaries),
                "count": len(summaries),
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
            sensitivity=inspection.summary.sensitivity,
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
                {"status": inspection.summary.status.value},
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
                "status": inspection.summary.status.value,
                "desired_state": inspection.desired_state.value,
                "updated_at": inspection.summary.updated_at.isoformat(),
                "specification_digest": inspection.specification_digest,
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        inspection = await self._owner.cancel(job_id)
        if inspection is None:
            raise CapabilityInputError(
                "job_not_found",
                "The requested job is not owned by this agent.",
            )
        return ToolOutput(
            kind="job.cancel_receipt",
            data={
                "job_id": job_id,
                "status": inspection.summary.status.value,
                "desired_state": inspection.desired_state.value,
                "cancel_requested_at": (
                    None
                    if inspection.cancel_requested_at is None
                    else inspection.cancel_requested_at.isoformat()
                ),
            },
            sensitivity=inspection.summary.sensitivity,
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
            statuses=frozenset({JobStatus.QUEUED, JobStatus.RUNNING}),
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
        del call, capability, request_sensitivity
        if run.agent_id != self._owner.agent_id:
            raise CapabilityInputError(
                "job_owner_mismatch",
                "The job lifecycle owner does not match this agent.",
            )
        return arguments

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
        description="List bounded jobs owned by this agent.",
        input_schema={
            "type": "object",
            "properties": {
                "statuses": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [item.value for item in JobStatus],
                    },
                    "maxItems": len(JobStatus),
                    "uniqueItems": True,
                }
            },
            "additionalProperties": False,
        },
        output_kind="job.list",
        output_schema=_list_schema(),
        executor_id=JOB_LIST_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
    )
    inspect_capability = Capability(
        id=JOB_INSPECT_CAPABILITY_ID,
        description="Inspect one exact durable job owned by the current agent.",
        input_schema=_job_id_schema(),
        output_kind="job.inspection",
        output_schema=_object_output_schema(),
        executor_id=JOB_INSPECT_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
    )
    result_capability = Capability(
        id=JOB_READ_RESULTS_CAPABILITY_ID,
        description="Read the bounded validated result of one successful durable job.",
        input_schema=_job_id_schema(),
        output_kind="job.result",
        output_schema=_object_output_schema(),
        executor_id=JOB_READ_RESULTS_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
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
            "List durable jobs owned by this agent.",
            "Use for job inventory or when the exact job ID is unknown.",
            ("job", "list", "status"),
        ),
        JOB_INSPECT_CAPABILITY_ID: (
            "Inspect one durable job lifecycle.",
            "Use only for status, attempts, failures, or execution details.",
            ("job", "inspect", "attempt"),
        ),
        JOB_READ_RESULTS_CAPABILITY_ID: (
            "Read one successful durable job result.",
            "Use first when a known job ID needs its validated result references.",
            ("job", "result", "artifact"),
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
            discovery=ToolDiscoveryMetadata(
                summary=summaries[item.id][0],
                when_to_use=summaries[item.id][1],
                keywords=summaries[item.id][2],
                exposure_class=(
                    ToolExposureClass.DEFERRED
                    if item.id == JOB_CANCEL_CAPABILITY_ID
                    else ToolExposureClass.CORE
                ),
                eager_priority=(
                    0
                    if item.id == JOB_CANCEL_CAPABILITY_ID
                    else {
                        JOB_LIST_CAPABILITY_ID: 980,
                        JOB_READ_RESULTS_CAPABILITY_ID: 970,
                        JOB_INSPECT_CAPABILITY_ID: 960,
                    }[item.id]
                ),
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
        "status": {"type": "string"},
        "desired_state": {"type": "string"},
        "cancel_requested_at": {"type": ["string", "null"]},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _summary_payload(value: JobSummary) -> dict[str, object]:
    return {
        "job_id": value.job_id,
        "origin_conversation_id": value.origin_conversation_id,
        "job_kind": value.job_kind,
        "status": value.status.value,
        "execution_mode": value.execution_mode.value,
        "source_ids": value.source_ids,
        "resource_ids": value.resource_ids,
        "sensitivity": value.sensitivity.value,
        "created_at": value.created_at.isoformat(),
        "updated_at": value.updated_at.isoformat(),
        "result_available": value.result_available,
    }


def _inspection_payload(value: JobInspection) -> dict[str, object]:
    return {
        **_summary_payload(value.summary),
        "origin_run_id": value.origin_run_id,
        "specification_digest": value.specification_digest,
        "execution_capability_id": value.execution_capability_id,
        "execution_contract_digest": value.execution_contract_digest,
        "desired_state": value.desired_state.value,
        "deadline_at": value.deadline_at.isoformat(),
        "attempts": tuple(
            {
                "number": item.number,
                "fencing_epoch": item.fencing_epoch,
                "status": item.status.value,
                "claimed_at": item.claimed_at.isoformat(),
                "completed_at": (
                    None if item.completed_at is None else item.completed_at.isoformat()
                ),
                "error_code": item.error_code,
                "external_intents": tuple(
                    {
                        "kind": intent.kind.value,
                        "disposition": intent.disposition.value,
                        "requested_at": intent.requested_at.isoformat(),
                        "completed_at": (
                            None
                            if intent.completed_at is None
                            else intent.completed_at.isoformat()
                        ),
                        "external_job_id": intent.external_job_id,
                        "reason_code": intent.reason_code,
                    }
                    for intent in item.external_intents
                ),
                "external_observations": tuple(
                    {
                        "sequence": observation.sequence,
                        "status": observation.status.value,
                        "observed_at": observation.observed_at.isoformat(),
                        "observation_digest": observation.observation_digest,
                        "external_job_id": observation.external_job_id,
                    }
                    for observation in item.external_observations
                ),
            }
            for item in value.attempts
        ),
        "cancel_requested_at": (
            None
            if value.cancel_requested_at is None
            else value.cancel_requested_at.isoformat()
        ),
        "terminal_at": (
            None if value.terminal_at is None else value.terminal_at.isoformat()
        ),
        "failure_code": value.failure_code,
        "external_executor": (
            None
            if value.external_executor is None
            else {
                "profile_id": value.external_executor.profile_id,
                "binding_id": value.external_executor.binding_id,
                "execution_identity": value.external_executor.execution_identity,
                "contract_digest": value.external_executor.contract_digest,
                "revision": value.external_executor.revision,
            }
        ),
    }


def _result_payload(value: JobResultView) -> dict[str, object]:
    return {
        "job_id": value.job_id,
        "result_id": value.result_id,
        "summary": value.summary,
        "sensitivity": value.sensitivity.value,
        "provenance": value.provenance,
        "artifacts": tuple(
            artifact_ref_to_mapping(item) for item in value.artifact_refs
        ),
        "completed_at": value.completed_at.isoformat(),
    }


def _summary_sensitivity(values: tuple[JobSummary, ...]) -> ModelSensitivity:
    order = {
        ModelSensitivity.PUBLIC: 0,
        ModelSensitivity.INTERNAL: 1,
        ModelSensitivity.CONFIDENTIAL: 2,
        ModelSensitivity.RESTRICTED: 3,
    }
    return max(
        (item.sensitivity for item in values),
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
