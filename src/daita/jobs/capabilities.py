"""Declare and execute the job tools: four for people, three for a job's own run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from .._json import FrozenJsonObject, canonical_json
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    MachineRunDirectiveKind,
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
from ..loop.models import RunInput, RunOrigin
from ..scope import scope_refs
from .models import (
    LIVE_JOB_STATES,
    MAX_CHECKLIST_ITEMS,
    MAX_DELEGATION_CONTEXT_CHARACTERS,
    MAX_GOAL_CHARACTERS,
    MAX_ITEM_NOTE_CHARACTERS,
    MAX_ITEM_TITLE_CHARACTERS,
    MAX_JOB_LIST_PAGE_SIZE,
    MAX_JOB_SOURCES,
    MAX_OBJECTIVE_CHARACTERS,
    MAX_QUESTION_CHARACTERS,
    ChecklistStatus,
    Job,
    JobError,
    JobState,
    clean_text,
)
from .owner import JobOwner

if TYPE_CHECKING:
    from .supervisor import JobSupervisor

JOB_DOMAIN_OWNER_ID = "jobs"
JOB_RUN_DOMAIN_OWNER_ID = "job_runs"
JOB_START_TOOL_NAME = "job_start"
JOB_LIST_TOOL_NAME = "job_list"
JOB_INSPECT_TOOL_NAME = "job_inspect"
JOB_CANCEL_TOOL_NAME = "job_cancel"
CHECKLIST_WRITE_TOOL_NAME = "checklist_write"
DELEGATE_TOOL_NAME = "delegate"
ASK_USER_TOOL_NAME = "ask_user"
JOB_START_CAPABILITY_ID = "jobs.start"
JOB_LIST_CAPABILITY_ID = "jobs.list"
JOB_INSPECT_CAPABILITY_ID = "jobs.inspect"
JOB_CANCEL_CAPABILITY_ID = "jobs.cancel"
CHECKLIST_WRITE_CAPABILITY_ID = "job_runs.checklist_write"
DELEGATE_CAPABILITY_ID = "job_runs.delegate"
ASK_USER_CAPABILITY_ID = "job_runs.ask_user"
_ITEM_ID_SCHEMA = {
    "type": "string",
    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$",
}
_SOURCES_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1, "maxLength": 2048},
    "maxItems": MAX_JOB_SOURCES,
    "uniqueItems": True,
}


@dataclass(frozen=True, slots=True)
class JobCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


def job_payload(job: Job, *, detailed: bool) -> dict[str, object]:
    """One job as the job tools return it; model-written text is sanitized."""

    done, total = job.progress
    summary: dict[str, object] = {
        "job_id": job.job_id,
        "state": job.state.value,
        "objective": job.objective,
        "progress": f"{done} of {total} done",
        "question": job.question,
        "failure_code": job.failure_code,
        "run_count": job.run_count,
        "updated_at": job.updated_at.isoformat(),
    }
    if not detailed:
        return summary
    return {
        **summary,
        "source_ids": job.source_ids,
        "checklist": tuple(
            {
                "id": item.id,
                "title": item.title,
                "status": item.status.value,
                "after": item.after,
                "note": item.note,
            }
            for item in job.checklist
        ),
        "delegations": tuple(
            {
                "id": item.id,
                "goal": item.goal,
                "state": item.state.value,
                "item": item.item,
                "summary": item.summary,
            }
            for item in job.delegations
        ),
        "answers": tuple(
            {"question": item.question, "answer": item.text} for item in job.answers
        ),
        "tokens_used": job.tokens_used,
        "final_summary": job.final_summary,
        "created_at": job.created_at.isoformat(),
    }


class _Executor:
    def __init__(self, owner: JobOwner) -> None:
        self._owner = owner


def _scope_job_id(request: ToolExecution) -> str:
    scope = request.execution_scope
    if scope is None or scope.job_id is None:
        raise CapabilityInputError(
            "job_run_required", "This tool works only inside a job's own run."
        )
    return scope.job_id


def _output(kind: str, data: Mapping[str, object], job: Job | None) -> ToolOutput:
    return ToolOutput(
        kind=kind,
        data=data,
        sensitivity=(job.sensitivity if job is not None else ModelSensitivity.INTERNAL),
        sensitivity_provenance={"authority": "job_record"},
    )


class JobStartExecutor(_Executor):
    executor_id = "jobs.start.executor"

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "objective_digest": "sha256:"
                + sha256(canonical_json(request.arguments).encode("utf-8")).hexdigest()
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        if request.conversation_id is None:
            raise CapabilityInputError(
                "job_conversation_required", "A job delivers to one conversation."
            )
        objective = request.arguments["objective"]
        sources = request.arguments.get("sources", ())
        assert isinstance(objective, str) and isinstance(sources, tuple)
        job = await self._owner.start(
            objective,
            conversation_id=request.conversation_id,
            sources=tuple(str(item) for item in sources),
            refs=scope_refs(request.source_scope),
        )
        return _output(
            "job.started",
            {
                "job_id": job.job_id,
                "state": job.state.value,
                "note": "The job runs in the background and shows on the Work "
                "board; its result arrives in this conversation's Inbox.",
            },
            job,
        )


class JobListExecutor(_Executor):
    executor_id = "jobs.list.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        raw = request.arguments.get("states", ())
        assert isinstance(raw, tuple)
        jobs = await self._owner.list(
            states=frozenset(JobState(item) for item in raw),
            limit=MAX_JOB_LIST_PAGE_SIZE,
        )
        return ToolOutput(
            kind="job.list",
            data={
                "jobs": tuple(job_payload(job, detailed=False) for job in jobs),
                "count": len(jobs),
            },
            sensitivity=max(
                (job.sensitivity for job in jobs),
                default=ModelSensitivity.INTERNAL,
                key=lambda item: item.routing_rank,
            ),
            sensitivity_provenance={"authority": "job_record"},
        )


class JobInspectExecutor(_Executor):
    executor_id = "jobs.inspect.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        job = await self._owner.inspect(job_id)
        if job is None:
            raise CapabilityInputError(
                "job_not_found", "The requested job is not owned by this agent."
            )
        return _output("job.inspection", job_payload(job, detailed=True), job)


class JobCancelExecutor(_Executor):
    executor_id = "jobs.cancel.executor"

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        job = await self._owner.inspect(job_id)
        if job is None:
            raise CapabilityInputError(
                "job_not_found", "The requested job is not owned by this agent."
            )
        return FrozenJsonObject.from_mapping(
            {"job_id": job_id, "state": job.state.value, "revision": job.revision}
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = request.arguments["job_id"]
        assert isinstance(job_id, str)
        job = await self._owner.cancel(job_id)
        if job is None:
            raise CapabilityInputError(
                "job_not_found", "The requested job is not owned by this agent."
            )
        return _output(
            "job.cancel_receipt", {"job_id": job_id, "state": job.state.value}, job
        )


class ChecklistWriteExecutor(_Executor):
    executor_id = "job_runs.checklist_write.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        items = request.arguments["items"]
        merge = request.arguments.get("merge", False)
        assert isinstance(items, tuple) and isinstance(merge, bool)
        job = await self._owner.write_checklist(
            _scope_job_id(request),
            tuple(item for item in items if isinstance(item, Mapping)),
            merge=merge,
        )
        done, total = job.progress
        return _output(
            "job.checklist",
            {
                "progress": f"{done} of {total} done",
                "checklist": job_payload(job, detailed=True)["checklist"],
            },
            job,
        )


class DelegateExecutor(_Executor):
    executor_id = "job_runs.delegate.executor"

    def __init__(self, owner: JobOwner) -> None:
        super().__init__(owner)
        self._supervisor: JobSupervisor | None = None

    def bind_supervisor(self, supervisor: JobSupervisor) -> None:
        self._supervisor = supervisor

    async def execute(self, request: ToolExecution) -> ToolOutput:
        job_id = _scope_job_id(request)
        assert request.execution_scope is not None
        if self._supervisor is None:
            raise CapabilityInputError(
                "job_supervisor_unavailable", "Sub-runs cannot start here."
            )
        goal, context = request.arguments["goal"], request.arguments["context"]
        item = request.arguments.get("item")
        sources = request.arguments.get("sources", ())
        assert isinstance(goal, str) and isinstance(context, str)
        assert isinstance(sources, tuple) and (item is None or isinstance(item, str))
        source_ids = await self._owner.resolve_sources(
            tuple(str(value) for value in sources), scope_refs(request.source_scope)
        )
        outside = sorted(
            set(source_ids) - set(request.execution_scope.allowed_source_ids)
        )
        if outside:
            raise CapabilityInputError(
                "job_source_outside_run",
                "A sub-run can read only the sources this run reads.",
            )
        delegation, sensitivity = await self._supervisor.delegate(
            job_id=job_id,
            parent_scope=request.execution_scope,
            goal=goal,
            context=clean_text(context, MAX_DELEGATION_CONTEXT_CHARACTERS, lines=True),
            item=item,
            source_ids=source_ids,
        )
        return ToolOutput(
            kind="job.delegation",
            data={
                "delegation_id": delegation.id,
                "state": delegation.state.value,
                "item": delegation.item,
                "summary": delegation.summary,
            },
            sensitivity=sensitivity,
            sensitivity_provenance={"authority": "job_subrun"},
        )


class AskUserExecutor(_Executor):
    executor_id = "job_runs.ask_user.executor"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        question = request.arguments["question"]
        assert isinstance(question, str)
        job = await self._owner.ask(_scope_job_id(request), question)
        return _output(
            "job.waiting",
            {
                "state": job.state.value,
                "note": "This run ends here; the job continues in a new run "
                "once the person answers.",
            },
            job,
        )


class _JobDomainBase:
    domain_owner_id = ""
    _origins: frozenset[RunOrigin] = frozenset()

    def __init__(self, declarations: CapabilityDeclarations, owner: JobOwner) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("job declarations have the wrong owner")
        self._declarations = declarations
        self._owner = owner

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

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
        del call, capability, request_sensitivity
        if run.agent_id != self._owner.agent_id or run.origin not in self._origins:
            raise CapabilityInputError(
                "job_tool_unavailable", "This job tool is not available in this run."
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
        # Starting or cancelling a job is the person's own request in a
        # foreground run; the job shows on the Work board at once.
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
        self, call: ToolCall, error: BaseException
    ) -> CapabilityFailure | None:
        del call
        if isinstance(error, JobError):
            return CapabilityFailure(error.code, str(error), error.details)
        return None


class JobCapabilityDomain(_JobDomainBase):
    """job_start, job_list, job_inspect and job_cancel, in foreground runs."""

    domain_owner_id = JOB_DOMAIN_OWNER_ID
    _origins = frozenset({RunOrigin.USER})

    async def project(self, run: RunInput) -> tuple[str, ...]:
        if run.agent_id != self._owner.agent_id or run.origin is not RunOrigin.USER:
            return ()
        names = [JOB_START_TOOL_NAME, JOB_LIST_TOOL_NAME]
        if await self._owner.list(limit=1):
            names.append(JOB_INSPECT_TOOL_NAME)
        if await self._owner.list(states=LIVE_JOB_STATES, limit=1):
            names.append(JOB_CANCEL_TOOL_NAME)
        return tuple(names)


class JobRunCapabilityDomain(_JobDomainBase):
    """checklist_write, delegate and ask_user, in a job's main run only."""

    domain_owner_id = JOB_RUN_DOMAIN_OWNER_ID
    _origins = frozenset({RunOrigin.JOB})

    async def project(self, run: RunInput) -> tuple[str, ...]:
        if run.agent_id != self._owner.agent_id or run.origin is not RunOrigin.JOB:
            return ()
        return (CHECKLIST_WRITE_TOOL_NAME, DELEGATE_TOOL_NAME, ASK_USER_TOOL_NAME)


def _view(
    name: str,
    capability: Capability,
    *,
    summary: str,
    when_to_use: str,
    keywords: tuple[str, ...],
) -> ToolView:
    return ToolView(
        name=name,
        capability_id=capability.id,
        description=capability.description,
        presentation=ToolPresentation(
            toolbox_id=ToolboxId.JOBS,
            # Effectful tools load on demand; the runtime refuses them pinned.
            load_mode=(
                ToolLoadMode.PINNED
                if capability.operational_effect is OperationalEffect.NONE
                else ToolLoadMode.ON_DEMAND
            ),
            text_trust=ToolTextTrust.CODE,
            summary=summary,
            when_to_use=when_to_use,
            keywords=keywords,
        ),
    )


def _capability(
    capability_id: str,
    description: str,
    input_schema: dict[str, object],
    output_kind: str,
    executor_id: str,
    **options: object,
) -> Capability:
    return Capability(
        id=capability_id,
        description=description,
        input_schema=input_schema,
        output_kind=output_kind,
        output_schema={"type": "object", "additionalProperties": True},
        executor_id=executor_id,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        **options,  # type: ignore[arg-type]
    )


def _job_id_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"job_id": {"type": "string", "minLength": 1}},
        "required": ["job_id"],
        "additionalProperties": False,
    }


def job_capability_declarations(owner: JobOwner) -> JobCapabilityDeclarations:
    """The people-facing job tools."""

    start = _capability(
        JOB_START_CAPABILITY_ID,
        "Start a background job that keeps working on an objective in the host.",
        {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_OBJECTIVE_CHARACTERS,
                },
                "sources": _SOURCES_SCHEMA,
            },
            "required": ["objective"],
            "additionalProperties": False,
        },
        "job.started",
        JobStartExecutor.executor_id,
        operational_effect=OperationalEffect.START_JOB,
        durable_foreground_entrypoint=True,
    )
    listing = _capability(
        JOB_LIST_CAPABILITY_ID,
        "List this agent's background jobs with their state and progress.",
        {
            "type": "object",
            "properties": {
                "states": {
                    "type": "array",
                    "items": {"type": "string", "enum": [s.value for s in JobState]},
                    "maxItems": len(JobState),
                    "uniqueItems": True,
                }
            },
            "additionalProperties": False,
        },
        "job.list",
        JobListExecutor.executor_id,
    )
    inspect = _capability(
        JOB_INSPECT_CAPABILITY_ID,
        "Inspect one job: checklist, delegations, question and final result.",
        _job_id_schema(),
        "job.inspection",
        JobInspectExecutor.executor_id,
    )
    cancel = _capability(
        JOB_CANCEL_CAPABILITY_ID,
        "Cancel one background job; its runs stop.",
        _job_id_schema(),
        "job.cancel_receipt",
        JobCancelExecutor.executor_id,
        operational_effect=OperationalEffect.CANCEL_JOB,
    )
    return JobCapabilityDeclarations(
        capabilities=(start, listing, inspect, cancel),
        executors=(
            JobStartExecutor(owner),
            JobListExecutor(owner),
            JobInspectExecutor(owner),
            JobCancelExecutor(owner),
        ),
        tool_views=(
            _view(
                JOB_START_TOOL_NAME,
                start,
                summary="Start a background job.",
                when_to_use="Use for work that should keep going without the "
                "person: long, multi-part or slow.",
                keywords=("job", "background", "durable", "start"),
            ),
            _view(
                JOB_LIST_TOOL_NAME,
                listing,
                summary="List this agent's jobs.",
                when_to_use="Find a job or its state.",
                keywords=("job", "list", "status"),
            ),
            _view(
                JOB_INSPECT_TOOL_NAME,
                inspect,
                summary="Inspect one job and its result.",
                when_to_use="Read a job's checklist, question or final result.",
                keywords=("job", "inspect", "result"),
            ),
            _view(
                JOB_CANCEL_TOOL_NAME,
                cancel,
                summary="Cancel one job.",
                when_to_use="Use when the person wants a job to stop.",
                keywords=("job", "cancel", "stop"),
            ),
        ),
    )


def job_run_capability_declarations(
    owner: JobOwner,
) -> tuple[JobCapabilityDeclarations, DelegateExecutor]:
    """The tools of a job's own main run; sub-runs get none of them."""

    checklist = _capability(
        CHECKLIST_WRITE_CAPABILITY_ID,
        "Write this job's checklist: replace it, or merge items by id. Returns "
        "the full list. An item can start or finish only after every item it "
        "comes after is done.",
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": MAX_CHECKLIST_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": _ITEM_ID_SCHEMA,
                            "title": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_ITEM_TITLE_CHARACTERS,
                            },
                            "status": {
                                "type": "string",
                                "enum": [s.value for s in ChecklistStatus],
                            },
                            "after": {
                                "type": "array",
                                "items": _ITEM_ID_SCHEMA,
                                "maxItems": MAX_CHECKLIST_ITEMS,
                                "uniqueItems": True,
                            },
                            "note": {
                                "type": "string",
                                "maxLength": MAX_ITEM_NOTE_CHARACTERS,
                            },
                        },
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                },
                "merge": {"type": "boolean"},
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "job.checklist",
        ChecklistWriteExecutor.executor_id,
    )
    delegate = _capability(
        DELEGATE_CAPABILITY_ID,
        "Start a sub-run for one independent piece, wait for it and return its "
        "summary. Several delegate calls in one step run in parallel. With item, "
        "the sub-run works on that checklist item and code marks it.",
        {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_GOAL_CHARACTERS,
                },
                "context": {
                    "type": "string",
                    "maxLength": MAX_DELEGATION_CONTEXT_CHARACTERS,
                },
                "item": _ITEM_ID_SCHEMA,
                "sources": _SOURCES_SCHEMA,
            },
            "required": ["goal", "context"],
            "additionalProperties": False,
        },
        "job.delegation",
        DelegateExecutor.executor_id,
    )
    ask = _capability(
        ASK_USER_CAPABILITY_ID,
        "Ask the person a question. This run ends and the job waits; their "
        "answer continues it in a new run. Call it alone in its step.",
        {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUESTION_CHARACTERS,
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "job.waiting",
        AskUserExecutor.executor_id,
        machine_run_directive_kind=MachineRunDirectiveKind.ASK_USER,
    )
    delegate_executor = DelegateExecutor(owner)
    declarations = JobCapabilityDeclarations(
        capabilities=(checklist, delegate, ask),
        executors=(
            ChecklistWriteExecutor(owner),
            delegate_executor,
            AskUserExecutor(owner),
        ),
        tool_views=(
            _view(
                CHECKLIST_WRITE_TOOL_NAME,
                checklist,
                summary="Write the job's checklist.",
                when_to_use="Plan the job and keep item statuses current.",
                keywords=("checklist", "plan", "todo"),
            ),
            _view(
                DELEGATE_TOOL_NAME,
                delegate,
                summary="Hand a piece of the job to a sub-run.",
                when_to_use="Use for independent pieces; call several at once "
                "to run them in parallel.",
                keywords=("delegate", "subrun", "parallel"),
            ),
            _view(
                ASK_USER_TOOL_NAME,
                ask,
                summary="Ask the person and wait for the answer.",
                when_to_use="Use only when the job cannot continue without them.",
                keywords=("ask", "question", "person"),
            ),
        ),
    )
    return declarations, delegate_executor


__all__ = [
    "ASK_USER_TOOL_NAME",
    "JOB_CANCEL_CAPABILITY_ID",
    "JOB_INSPECT_CAPABILITY_ID",
    "JOB_LIST_CAPABILITY_ID",
    "JOB_START_CAPABILITY_ID",
    "CHECKLIST_WRITE_TOOL_NAME",
    "DELEGATE_TOOL_NAME",
    "JOB_CANCEL_TOOL_NAME",
    "JOB_DOMAIN_OWNER_ID",
    "JOB_INSPECT_TOOL_NAME",
    "JOB_LIST_TOOL_NAME",
    "JOB_RUN_DOMAIN_OWNER_ID",
    "JOB_START_TOOL_NAME",
    "DelegateExecutor",
    "JobCapabilityDeclarations",
    "JobCapabilityDomain",
    "JobRunCapabilityDomain",
    "job_capability_declarations",
    "job_payload",
    "job_run_capability_declarations",
]
