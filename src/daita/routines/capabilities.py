"""Static foreground tools for bounded scheduled-routine management."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import TypedDict

from .._json import FrozenJsonObject
from ..artifacts.models import ArtifactAuthorship
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    EffectEvidenceBasis,
    Executor,
    OperationalEffect,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
    resource_id_schema,
    source_id_schema,
)
from ..capability_runtime import CapabilityFailure, SideEffectPlan
from ..distribution.models import (
    MAX_OUTCOME_ARTIFACT_BYTES,
    MAX_OUTCOME_ARTIFACT_REFERENCES,
    MAX_OUTCOME_ARTIFACT_REQUIREMENTS,
    MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
    ArtifactRequirement,
    EffectRequirement,
    OutcomeContract,
    requirements_need_exact_source_bindings,
)
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput, RunOrigin
from .models import (
    MAX_ROUTINE_CONSECUTIVE_FAILURES,
    MAX_ROUTINE_CUMULATIVE_COST_USD,
    MAX_ROUTINE_CUMULATIVE_TOKENS,
    MAX_ROUTINE_IDENTITY_ITEMS,
    MAX_ROUTINE_INSTRUCTION_BYTES,
    MAX_ROUTINE_INTERVAL_SECONDS,
    MAX_ROUTINE_LIST_PAGE_SIZE,
    MAX_ROUTINE_OCCURRENCES,
    MAX_ROUTINE_PER_RUN_COST_USD,
    MAX_ROUTINE_PER_RUN_TOKENS,
    MAX_ROUTINE_SKILL_BINDINGS,
    MAX_ROUTINE_TITLE_CHARACTERS,
    MIN_ROUTINE_INTERVAL_SECONDS,
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    MisfirePolicy,
    NonexistentTimePolicy,
    OnceSchedule,
    ReportingMode,
    RequestedCapabilityGrant,
    ResourceRevisionPrecheck,
    RoutineControlAction,
    RoutineOccurrence,
    RoutineSchedule,
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
)
from .owner import (
    MAX_SCHEDULE_PREVIEW_OCCURRENCES,
    RoutineError,
    RoutineOwner,
    _routine_proposal_payload,
    _schedule_payload,
    routine_approval_arguments,
    schedule_preview,
)
from .temporal import next_weekday_utc

ROUTINE_DOMAIN_OWNER_ID = "routines"
ROUTINE_LIST_CAPABILITY_ID = "routines.list"
ROUTINE_LIST_EXECUTOR_ID = "routines.list.executor"
ROUTINE_LIST_TOOL_NAME = "routine_list"
ROUTINE_INSPECT_CAPABILITY_ID = "routines.inspect"
ROUTINE_INSPECT_EXECUTOR_ID = "routines.inspect.executor"
ROUTINE_INSPECT_TOOL_NAME = "routine_inspect"
ROUTINE_CREATE_CAPABILITY_ID = "routines.create"
ROUTINE_CREATE_EXECUTOR_ID = "routines.create.executor"
ROUTINE_UPDATE_CAPABILITY_ID = "routines.update"
ROUTINE_UPDATE_EXECUTOR_ID = "routines.update.executor"
ROUTINE_SAVE_TOOL_NAME = "routine_save"
ROUTINE_SAVE_DESCRIPTION = (
    "Save one finite routine with frozen grants from a self-contained "
    "instruction. Omit routine_id to create; pass routine_id and "
    "expected_revision to replace that exact revision. Relative timing "
    "(after_seconds, anchor_after_seconds, once_next_weekday, "
    "expires_after_seconds) and run_immediately are create-only; set "
    "expires_at or expires_after_seconds. Omitted options default to "
    "misfire_policy skip, this conversation's Inbox, bindings of allowed "
    "capabilities, and no grants, skills, precheck or dollar ceilings. The Inbox is report delivery only; an action's target goes "
    "in its grant fixed_arguments."
)
ROUTINE_SAVE_SUMMARIZED_ARGUMENTS = (
    "requested_capability_grants",
    "precheck",
    "allowed_connector_binding_ids",
    "per_run_max_cost_usd",
    "cumulative_max_cost_usd",
    "basis_run_id",
)
ROUTINE_CONTROL_CAPABILITY_ID = "routines.control"
ROUTINE_CONTROL_EXECUTOR_ID = "routines.control.executor"
ROUTINE_CONTROL_TOOL_NAME = "routine_control"
MAX_RELATIVE_SCHEDULE_SECONDS = 30 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class RoutineCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class _RoutineExecutor:
    def __init__(self, owner: RoutineOwner) -> None:
        self._owner = owner


class RoutineListExecutor(_RoutineExecutor):
    executor_id = ROUTINE_LIST_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        raw_states = request.arguments.get("states", ())
        assert isinstance(raw_states, tuple)
        summaries = await self._owner.list(
            states=frozenset(RoutineState(item) for item in raw_states),
            limit=MAX_ROUTINE_LIST_PAGE_SIZE,
        )
        return ToolOutput(
            kind="routine.list",
            sensitivity=max(
                (
                    request.request_sensitivity,
                    *(item.sensitivity_ceiling for item in summaries),
                ),
                key=lambda item: item.routing_rank,
            ),
            sensitivity_provenance={"authority": "agent_owned_routine_summaries"},
            data={
                "routines": tuple(_summary_payload(item) for item in summaries),
                "count": len(summaries),
            },
        )


class RoutineInspectExecutor(_RoutineExecutor):
    executor_id = ROUTINE_INSPECT_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        routine_id = _string(request.arguments, "routine_id")
        inspection = await self._owner.inspect(routine_id)
        if inspection is None:
            raise CapabilityInputError(
                "routine_not_found",
                "The requested routine is not owned by this agent.",
            )
        return ToolOutput(
            kind="routine.inspection",
            sensitivity=max(
                request.request_sensitivity,
                inspection.routine.sensitivity_ceiling,
                key=lambda item: item.routing_rank,
            ),
            sensitivity_provenance={
                "authority": "agent_owned_routine",
                "routine_id": routine_id,
            },
            data=routine_inspection_projection(inspection),
        )


class RoutineCreateExecutor(_RoutineExecutor):
    executor_id = ROUTINE_CREATE_EXECUTOR_ID

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        proposal = await _create_proposal(self._owner, request)
        return await self._owner.proposal_authority_snapshot(proposal)

    async def execute(self, request: ToolExecution) -> ToolOutput:
        proposal = await _create_proposal(self._owner, request, commit_time=True)
        if _has_temporal_intent(request.arguments):
            schedule_intent = _mapping(request.arguments, "schedule")
            first_at = (
                _first_absolute_slot(proposal.schedule)
                if "after_seconds" in schedule_intent
                or "anchor_after_seconds" in schedule_intent
                or schedule_intent.get("kind") == "once_next_weekday"
                else None
            )
            now = self._owner.current_time()
            if proposal.expires_at <= now or (first_at is not None and first_at <= now):
                raise RoutineError(
                    "routine_schedule_passed",
                    "The approved timing passed before admission; choose a new time.",
                )
        stored = await self._owner.admit(proposal)
        return ToolOutput(
            kind="routine.receipt",
            sensitivity=max(
                request.request_sensitivity,
                stored.sensitivity_ceiling,
                key=lambda item: item.routing_rank,
            ),
            sensitivity_provenance={
                "authority": "agent_owned_routine",
                "routine_id": stored.routine_id,
            },
            data=_mutation_receipt("create", stored),
        )


class RoutineUpdateExecutor(_RoutineExecutor):
    executor_id = ROUTINE_UPDATE_EXECUTOR_ID

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        proposal, _expected = await _update_proposal(self._owner, request)
        return await self._owner.proposal_authority_snapshot(proposal)

    async def execute(self, request: ToolExecution) -> ToolOutput:
        proposal, expected = await _update_proposal(self._owner, request)
        stored = await self._owner.revise(proposal, expected_revision=expected)
        return ToolOutput(
            kind="routine.receipt",
            sensitivity=max(
                request.request_sensitivity,
                stored.sensitivity_ceiling,
                key=lambda item: item.routing_rank,
            ),
            sensitivity_provenance={
                "authority": "agent_owned_routine",
                "routine_id": stored.routine_id,
            },
            data=_mutation_receipt("update", stored),
        )


class RoutineControlExecutor(_RoutineExecutor):
    executor_id = ROUTINE_CONTROL_EXECUTOR_ID

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        routine_id = _string(request.arguments, "routine_id")
        expected_revision = _integer(request.arguments, "expected_revision")
        action = RoutineControlAction(_string(request.arguments, "action"))
        inspection = await self._owner.inspect(routine_id)
        if inspection is None or inspection.routine.revision != expected_revision:
            raise RoutineError(
                "routine_revision_changed",
                "The routine changed or is no longer owned by this agent.",
            )
        if action in {RoutineControlAction.RESUME, RoutineControlAction.RUN_NOW}:
            await self._owner.authority_snapshot(inspection.routine)
        return FrozenJsonObject.from_mapping(
            {
                "routine_id": routine_id,
                "expected_revision": expected_revision,
                "action": action.value,
                "current_state": inspection.routine.state.value,
                "current_next_due_at": (
                    None
                    if inspection.routine.next_due_at is None
                    else inspection.routine.next_due_at.isoformat()
                ),
                "instruction_digest": inspection.routine.instruction_digest,
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        routine = await self._owner.control(
            _string(request.arguments, "routine_id"),
            expected_revision=_integer(request.arguments, "expected_revision"),
            action=RoutineControlAction(_string(request.arguments, "action")),
            authorized_control_call_id=request.call_id,
        )
        return ToolOutput(
            kind="routine.receipt",
            sensitivity=max(
                request.request_sensitivity,
                routine.sensitivity_ceiling,
                key=lambda item: item.routing_rank,
            ),
            sensitivity_provenance={
                "authority": "agent_owned_routine",
                "routine_id": routine.routine_id,
            },
            data=_mutation_receipt(_string(request.arguments, "action"), routine),
        )


class RoutineCapabilityDomain:
    domain_owner_id = ROUTINE_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        owner: RoutineOwner,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("routine declarations have the wrong owner")
        self._declarations = declarations
        self._owner = owner
        self._views = tuple(declarations.tool_views)

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        if run.agent_id != self._owner.agent_id or run.origin is not RunOrigin.USER:
            return ()
        names = {ROUTINE_LIST_TOOL_NAME, ROUTINE_SAVE_TOOL_NAME}
        if await self._owner.list(limit=1):
            names.update({ROUTINE_INSPECT_TOOL_NAME, ROUTINE_CONTROL_TOOL_NAME})
        return tuple(view.name for view in self._views if view.name in names)

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        if capability.id not in {
            ROUTINE_CREATE_CAPABILITY_ID,
            ROUTINE_UPDATE_CAPABILITY_ID,
        }:
            return arguments
        return _without_derived_outcome_rules(arguments)

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        del call
        if (
            run.agent_id != self._owner.agent_id
            or run.origin is not RunOrigin.USER
            or run.conversation_id is None
        ):
            raise CapabilityInputError(
                "routine_foreground_required",
                "Routine management requires an exact foreground conversation.",
            )
        if capability.id in {
            ROUTINE_CREATE_CAPABILITY_ID,
            ROUTINE_UPDATE_CAPABILITY_ID,
        }:
            ceiling = ModelSensitivity(_string(arguments, "sensitivity_ceiling"))
            if request_sensitivity.routing_rank > ceiling.routing_rank:
                raise CapabilityInputError(
                    "routine_instruction_sensitivity_exceeded",
                    "The saved instruction must retain the full request sensitivity.",
                )
        if capability.id == ROUTINE_UPDATE_CAPABILITY_ID:
            _require_exact_revision_timing(arguments)
        if capability.id == ROUTINE_CREATE_CAPABILITY_ID and _has_temporal_intent(
            arguments
        ):
            # Capture a stable reference for both approval preflights. Relative
            # delays themselves are rebased once after approval, at creation.
            return FrozenJsonObject.from_mapping(
                {
                    **arguments,
                    "_timing_reference_at": self._owner.current_time().isoformat(),
                }
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
        del run, call
        if (
            capability.operational_effect
            is not OperationalEffect.MANAGE_SCHEDULED_ROUTINE
        ):
            raise ValueError("routine domain received an unsupported effect")
        if capability.id == ROUTINE_CREATE_CAPABILITY_ID:
            reason = "Approve this assignment, its standing actions, and its schedule?"
        elif capability.id == ROUTINE_UPDATE_CAPABILITY_ID:
            reason = "Replace this routine with the exact proposed revision once?"
        else:
            reason = "Apply this exact routine control action once?"
        approval_arguments = (
            routine_approval_arguments(fingerprint)
            if "routine" in fingerprint
            else fingerprint
        )
        if capability.id == ROUTINE_CREATE_CAPABILITY_ID and _has_temporal_intent(
            execution.arguments
        ):
            approval_arguments = FrozenJsonObject.from_mapping(
                {
                    **approval_arguments,
                    "timing_intent": {
                        "schedule": execution.arguments["schedule"],
                        "expires_after_seconds": execution.arguments.get(
                            "expires_after_seconds"
                        ),
                    },
                    "timing_note": (
                        "Relative delays start when approval completes and the "
                        "routine is created; displayed absolute times for them are "
                        "estimates. A named local instant is frozen at approval."
                    ),
                }
            )
        return SideEffectPlan(
            approval_arguments=approval_arguments,
            approval_reason=reason,
            recheck_after_approval=True,
        )

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
        if isinstance(error, RoutineError):
            return CapabilityFailure(error.code, str(error))
        return None


def routine_capability_declarations(
    owner: RoutineOwner,
) -> RoutineCapabilityDeclarations:
    list_capability = Capability(
        id=ROUTINE_LIST_CAPABILITY_ID,
        description="List bounded scheduled routines owned by this agent.",
        input_schema={
            "type": "object",
            "properties": {
                "states": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [item.value for item in RoutineState],
                    },
                    "maxItems": len(RoutineState),
                    "uniqueItems": True,
                }
            },
            "additionalProperties": False,
        },
        output_kind="routine.list",
        output_schema=_object_output_schema(("routines", "count")),
        executor_id=ROUTINE_LIST_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    inspect_capability = Capability(
        id=ROUTINE_INSPECT_CAPABILITY_ID,
        description="Inspect one exact routine and its bounded recent occurrences.",
        input_schema=_routine_id_schema(),
        output_kind="routine.inspection",
        output_schema=_object_output_schema(("routine", "recent_occurrences")),
        executor_id=ROUTINE_INSPECT_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    create_capability = Capability(
        id=ROUTINE_CREATE_CAPABILITY_ID,
        description=(
            "Create one finite routine with frozen action grants from a self-contained instruction "
            "and typed absolute, relative, or user-local schedule."
        ),
        input_schema=_spec_schema(update=False),
        delegated_arguments=_GRANT_DELEGATION,
        output_kind="routine.receipt",
        output_schema=_mutation_receipt_schema(),
        executor_id=ROUTINE_CREATE_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.MANAGE_SCHEDULED_ROUTINE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    update_capability = Capability(
        id=ROUTINE_UPDATE_CAPABILITY_ID,
        description="Replace one routine's material contract with an exact new revision.",
        input_schema=_spec_schema(update=True),
        delegated_arguments=_GRANT_DELEGATION,
        output_kind="routine.receipt",
        output_schema=_mutation_receipt_schema(),
        executor_id=ROUTINE_UPDATE_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.MANAGE_SCHEDULED_ROUTINE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    control_capability = Capability(
        id=ROUTINE_CONTROL_CAPABILITY_ID,
        description="Pause, resume, run now, or permanently disable one exact routine revision.",
        input_schema={
            "type": "object",
            "properties": {
                "routine_id": {"type": "string", "minLength": 1, "maxLength": 1024},
                "expected_revision": {"type": "integer", "minimum": 1},
                "action": {
                    "type": "string",
                    "enum": [item.value for item in RoutineControlAction],
                },
            },
            "required": ["routine_id", "expected_revision", "action"],
            "additionalProperties": False,
        },
        output_kind="routine.receipt",
        output_schema=_mutation_receipt_schema(),
        executor_id=ROUTINE_CONTROL_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.MANAGE_SCHEDULED_ROUTINE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    capabilities = (
        list_capability,
        inspect_capability,
        create_capability,
        update_capability,
        control_capability,
    )
    executors: tuple[Executor, ...] = (
        RoutineListExecutor(owner),
        RoutineInspectExecutor(owner),
        RoutineCreateExecutor(owner),
        RoutineUpdateExecutor(owner),
        RoutineControlExecutor(owner),
    )

    def presentation(
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
        load_mode: ToolLoadMode = ToolLoadMode.ON_DEMAND,
    ) -> ToolPresentation:
        return ToolPresentation(
            toolbox_id=ToolboxId.ROUTINES,
            load_mode=load_mode,
            text_trust=ToolTextTrust.CODE,
            summary=summary,
            when_to_use=when_to_use,
            keywords=keywords,
        )

    views = (
        ToolView(
            name=ROUTINE_LIST_TOOL_NAME,
            capability_id=list_capability.id,
            description=list_capability.description,
            presentation=presentation(
                "List scheduled routines.",
                "Use for routine inventory.",
                ("routine", "list"),
                ToolLoadMode.PINNED,
            ),
        ),
        ToolView(
            name=ROUTINE_INSPECT_TOOL_NAME,
            capability_id=inspect_capability.id,
            description=inspect_capability.description,
            presentation=presentation(
                "Inspect one scheduled routine.",
                "Use for exact schedule, scope, budget, and recent occurrence details.",
                ("routine", "inspect", "history"),
            ),
        ),
        ToolView(
            name=ROUTINE_SAVE_TOOL_NAME,
            capability_id=create_capability.id,
            description=ROUTINE_SAVE_DESCRIPTION,
            presentation=presentation(
                "Create or revise one scheduled assignment with exact action limits.",
                "Discover assignment capabilities and their automation_contract in toolbox_search, and the Inbox destination. Load routine_save to author the schedule and exact grants; relative once and interval timing need no clock query. Assignment execution tools need not be loaded.",
                ("routine", "schedule", "create", "update", "revise"),
            ),
            mode_capability_ids=(
                ("create", create_capability.id),
                ("update", update_capability.id),
            ),
            mode_argument="routine_id",
            mode_by_presence=True,
            summarized_arguments=ROUTINE_SAVE_SUMMARIZED_ARGUMENTS,
        ),
        ToolView(
            name=ROUTINE_CONTROL_TOOL_NAME,
            capability_id=control_capability.id,
            description=control_capability.description,
            presentation=presentation(
                "Control one scheduled routine.",
                "Use to pause, resume, run now, or disable an exact revision.",
                ("routine", "pause", "resume", "disable"),
            ),
        ),
    )
    return RoutineCapabilityDeclarations(capabilities, executors, views)


async def _create_proposal(
    owner: RoutineOwner, request: ToolExecution, *, commit_time: bool = False
) -> ScheduledRoutine:
    if request.conversation_id is None:
        raise RoutineError(
            "routine_conversation_required",
            "Routine creation requires an exact conversation.",
        )
    reference_text = request.arguments.get("_timing_reference_at")
    reference_at = (
        _datetime(reference_text) if isinstance(reference_text, str) else None
    )
    relative_at = (
        owner.current_time()
        if commit_time and _has_relative_timing(request.arguments)
        else reference_at
    )
    values = _parsed_spec(
        request.arguments,
        reference_at=reference_at,
        relative_at=relative_at,
    )
    if (
        commit_time
        and _mapping(request.arguments, "schedule").get("kind") == "once_next_weekday"
        and isinstance(values["schedule"], OnceSchedule)
        and values["schedule"].exact_at <= owner.current_time()
    ):
        raise RoutineError(
            "routine_schedule_passed",
            "The approved local time passed before creation; choose a new time.",
        )
    return await owner.prepare_create(
        run_id=request.run_id,
        conversation_id=request.conversation_id,
        call_id=request.call_id,
        **values,
    )


async def _update_proposal(
    owner: RoutineOwner, request: ToolExecution
) -> tuple[ScheduledRoutine, int]:
    if request.conversation_id is None:
        raise RoutineError(
            "routine_conversation_required",
            "Routine revision requires an exact conversation.",
        )
    _require_exact_revision_timing(request.arguments)
    routine_id = _string(request.arguments, "routine_id")
    expected = _integer(request.arguments, "expected_revision")
    inspection = await owner.inspect(routine_id)
    if inspection is None or inspection.routine.revision != expected:
        raise RoutineError(
            "routine_revision_changed",
            "The routine changed or is no longer owned by this agent.",
        )
    values = _parsed_spec(request.arguments)
    revised = await owner.prepare_revision(
        inspection.routine,
        run_id=request.run_id,
        conversation_id=request.conversation_id,
        **values,
    )
    return revised, expected


class _ParsedSpec(TypedDict):
    title: str
    authorized_instruction: str
    schedule: RoutineSchedule
    misfire_policy: MisfirePolicy
    reporting_mode: ReportingMode
    precheck: ResourceRevisionPrecheck | None
    allowed_source_ids: tuple[str, ...]
    allowed_connector_binding_ids: tuple[str, ...] | None
    allowed_resource_ids: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    sensitivity_ceiling: ModelSensitivity
    outcome_contract: OutcomeContract
    distribution_destination_id: str | None
    eligible_model_routes: tuple[str, ...]
    per_run_max_tokens: int
    per_run_max_cost_usd: Decimal | None
    cumulative_max_tokens: int
    cumulative_max_cost_usd: Decimal | None
    cumulative_max_attempts: int
    cumulative_max_occurrences: int
    maximum_consecutive_failures: int
    expires_at: datetime
    skill_names: tuple[str, ...]
    requested_capability_grants: tuple[RequestedCapabilityGrant, ...]
    run_immediately: bool
    basis_run_id: str | None


def _parsed_spec(
    arguments: Mapping[str, object],
    *,
    reference_at: datetime | None = None,
    relative_at: datetime | None = None,
) -> _ParsedSpec:
    return {
        "title": _string(arguments, "title"),
        "requested_capability_grants": _parse_requested_grants(
            arguments.get("requested_capability_grants", ())
        ),
        "run_immediately": (
            _boolean(arguments, "run_immediately")
            if "run_immediately" in arguments
            else False
        ),
        "authorized_instruction": _string(arguments, "authorized_instruction"),
        "schedule": _parse_schedule(
            _mapping(arguments, "schedule"),
            reference_at=reference_at,
            relative_at=relative_at,
        ),
        "misfire_policy": MisfirePolicy(
            _string(arguments, "misfire_policy")
            if "misfire_policy" in arguments
            else MisfirePolicy.SKIP.value
        ),
        "reporting_mode": ReportingMode(_string(arguments, "reporting_mode")),
        "precheck": _parse_precheck(arguments.get("precheck")),
        "allowed_source_ids": _strings(arguments, "allowed_source_ids"),
        "allowed_connector_binding_ids": (
            _strings(arguments, "allowed_connector_binding_ids")
            if "allowed_connector_binding_ids" in arguments
            else None
        ),
        "allowed_resource_ids": _strings(arguments, "allowed_resource_ids"),
        "allowed_capability_ids": _strings(arguments, "allowed_capability_ids"),
        "sensitivity_ceiling": ModelSensitivity(
            _string(arguments, "sensitivity_ceiling")
        ),
        "outcome_contract": _parse_outcome_contract(
            _mapping(arguments, "outcome_contract")
        ),
        "distribution_destination_id": (
            _string(arguments, "distribution_destination_id")
            if "distribution_destination_id" in arguments
            else None
        ),
        "eligible_model_routes": _strings(arguments, "eligible_model_routes"),
        "per_run_max_tokens": _integer(arguments, "per_run_max_tokens"),
        "per_run_max_cost_usd": _optional_decimal(arguments, "per_run_max_cost_usd"),
        "cumulative_max_tokens": _integer(arguments, "cumulative_max_tokens"),
        "cumulative_max_cost_usd": _optional_decimal(
            arguments, "cumulative_max_cost_usd"
        ),
        "cumulative_max_attempts": _integer(arguments, "cumulative_max_attempts"),
        "cumulative_max_occurrences": _integer(arguments, "cumulative_max_occurrences"),
        "maximum_consecutive_failures": _integer(
            arguments, "maximum_consecutive_failures"
        ),
        "expires_at": _expires_at(arguments, relative_at),
        "skill_names": (
            _strings(arguments, "skill_names") if "skill_names" in arguments else ()
        ),
        "basis_run_id": (
            None
            if arguments.get("basis_run_id") is None
            else _string(arguments, "basis_run_id")
        ),
    }


def _expires_at(
    arguments: Mapping[str, object], relative_at: datetime | None
) -> datetime:
    if ("expires_at" in arguments) == ("expires_after_seconds" in arguments):
        raise CapabilityInputError(
            "routine_expiration_invalid",
            "Set exactly one of expires_at or expires_after_seconds.",
        )
    if "expires_after_seconds" in arguments:
        return _relative_instant(arguments, "expires_after_seconds", relative_at)
    return _datetime(_string(arguments, "expires_at"))


def _parse_schedule(
    value: Mapping[str, object],
    *,
    reference_at: datetime | None = None,
    relative_at: datetime | None = None,
) -> RoutineSchedule:
    kind = _string(value, "kind")
    unexpected = set(value) - {"kind"} - _SCHEDULE_KEYS.get(kind, frozenset())
    if kind not in _SCHEDULE_KEYS or unexpected:
        raise CapabilityInputError(
            "routine_schedule_invalid",
            (
                f"Schedule kind {kind!r} does not accept: {', '.join(sorted(unexpected))}."
                if kind in _SCHEDULE_KEYS
                else "Schedule kind must be once, interval, once_next_weekday, or calendar."
            ),
        )
    if kind == "once" and ("exact_at" in value) == ("after_seconds" in value):
        raise CapabilityInputError(
            "routine_schedule_invalid",
            "A once schedule needs exact_at or after_seconds.",
        )
    if kind == "interval" and ("anchor_at" in value) == (
        "anchor_after_seconds" in value
    ):
        raise CapabilityInputError(
            "routine_schedule_invalid",
            "An interval schedule needs anchor_at or anchor_after_seconds.",
        )
    try:
        return _typed_schedule(
            value, kind, reference_at=reference_at, relative_at=relative_at
        )
    except CapabilityInputError:
        raise
    except (TypeError, ValueError) as error:
        message = str(error)
        if kind == "calendar":
            message += (
                ". Correct the schedule: day_selector weekdays takes weekdays, "
                "month_days takes month_days, every_day takes neither."
            )
        raise CapabilityInputError("routine_schedule_invalid", message) from error


def _typed_schedule(
    value: Mapping[str, object],
    kind: str,
    *,
    reference_at: datetime | None = None,
    relative_at: datetime | None = None,
) -> RoutineSchedule:
    if kind == "once":
        if "after_seconds" in value:
            return OnceSchedule(_relative_instant(value, "after_seconds", relative_at))
        return OnceSchedule(_datetime(_string(value, "exact_at")))
    if kind == "interval":
        return IntervalSchedule(
            _integer(value, "interval_seconds"),
            (
                _relative_instant(value, "anchor_after_seconds", relative_at)
                if "anchor_after_seconds" in value
                else _datetime(_string(value, "anchor_at"))
            ),
        )
    if kind == "once_next_weekday":
        if reference_at is None:
            raise CapabilityInputError(
                "routine_time_unavailable", "A trusted current time is required."
            )
        try:
            exact_at = next_weekday_utc(
                now=reference_at,
                timezone=_string(value, "timezone"),
                weekday=_integer(value, "weekday"),
                hour=_integer(value, "hour"),
                minute=_integer(value, "minute"),
            )
        except ValueError as error:
            raise CapabilityInputError(
                "routine_local_time_invalid", str(error)
            ) from error
        return OnceSchedule(exact_at)
    if kind == "calendar":
        return CalendarSchedule(
            timezone=_string(value, "timezone"),
            hour=_integer(value, "hour"),
            minute=_integer(value, "minute"),
            day_selector=_calendar_day_selector(value),
            weekdays=_optional_integers(value, "weekdays"),
            month_days=_optional_integers(value, "month_days"),
            months=_optional_integers(value, "months"),
            nonexistent_time_policy=NonexistentTimePolicy(
                str(value.get("nonexistent_time_policy", "skip"))
            ),
            ambiguous_time_policy=AmbiguousTimePolicy(
                str(value.get("ambiguous_time_policy", "first"))
            ),
        )
    raise CapabilityInputError(
        "routine_schedule_invalid", "Schedule kind must be once, interval, or calendar."
    )


def _calendar_day_selector(value: Mapping[str, object]) -> CalendarDaySelector:
    """An omitted selector follows the day filters given; none means every day."""

    if "day_selector" in value:
        return CalendarDaySelector(_string(value, "day_selector"))
    weekdays, month_days = value.get("weekdays"), value.get("month_days")
    if weekdays and not month_days:
        return CalendarDaySelector.WEEKDAYS
    if month_days and not weekdays:
        return CalendarDaySelector.MONTH_DAYS
    return CalendarDaySelector.EVERY_DAY


def _relative_instant(
    values: Mapping[str, object], name: str, reference_at: datetime | None
) -> datetime:
    seconds = _integer(values, name)
    if not 1 <= seconds <= MAX_RELATIVE_SCHEDULE_SECONDS:
        raise CapabilityInputError(
            "routine_relative_time_invalid", "Relative timing is outside its bound."
        )
    if reference_at is None:
        raise CapabilityInputError(
            "routine_time_unavailable", "A trusted current time is required."
        )
    if reference_at.tzinfo is None or reference_at.utcoffset() is None:
        raise CapabilityInputError(
            "routine_time_unavailable", "A timezone-aware current time is required."
        )
    return reference_at.astimezone(UTC) + timedelta(seconds=seconds)


def _has_relative_timing(arguments: Mapping[str, object]) -> bool:
    schedule = arguments.get("schedule")
    return bool(
        "expires_after_seconds" in arguments
        or (
            isinstance(schedule, Mapping)
            and ("after_seconds" in schedule or "anchor_after_seconds" in schedule)
        )
    )


def _require_exact_revision_timing(arguments: Mapping[str, object]) -> None:
    """A replacement keeps exact timing; creation alone resolves relative intent."""

    if _has_temporal_intent(arguments) or arguments.get("run_immediately") is True:
        raise CapabilityInputError(
            "routine_timing_create_only",
            "Relative or next-weekday timing, expires_after_seconds and "
            "run_immediately are create-only; a replacement uses exact "
            "timestamps and expires_at.",
        )


def _has_temporal_intent(arguments: Mapping[str, object]) -> bool:
    schedule = arguments.get("schedule")
    return _has_relative_timing(arguments) or (
        isinstance(schedule, Mapping) and schedule.get("kind") == "once_next_weekday"
    )


def _first_absolute_slot(schedule: RoutineSchedule) -> datetime | None:
    if isinstance(schedule, OnceSchedule):
        return schedule.exact_at
    if isinstance(schedule, IntervalSchedule):
        return schedule.anchor_at
    return None


def _parse_precheck(value: object) -> ResourceRevisionPrecheck | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CapabilityInputError(
            "routine_precheck_invalid", "Precheck must be an exact object."
        )
    return ResourceRevisionPrecheck(
        capability_id=_string(value, "capability_id"),
        contract_digest=_string(value, "contract_digest"),
        source_id=_string(value, "source_id"),
        resource_id=_string(value, "resource_id"),
    )


def _parse_requested_grants(value: object) -> tuple[RequestedCapabilityGrant, ...]:
    if not isinstance(value, tuple):
        raise CapabilityInputError(
            "routine_grant_request_invalid", "Requested grants must be an array."
        )
    requested = []
    for item in value:
        if not isinstance(item, Mapping):
            raise CapabilityInputError(
                "routine_grant_request_invalid",
                "Requested grants must contain exact objects.",
            )
        requested.append(
            RequestedCapabilityGrant(
                capability_id=_string(item, "capability_id"),
                constraints=FrozenJsonObject.from_mapping(
                    _mapping(item, "constraints")
                ),
                max_calls_per_occurrence=_integer(item, "max_calls_per_occurrence"),
            )
        )
    return tuple(requested)


def _parse_outcome_contract(value: Mapping[str, object]) -> OutcomeContract:
    requirements_value = value.get("artifact_requirements")
    if not isinstance(requirements_value, tuple):
        raise CapabilityInputError(
            "routine_outcome_contract_invalid",
            "artifact_requirements must be an exact array.",
        )
    requirements: list[ArtifactRequirement] = []
    for item in requirements_value:
        if not isinstance(item, Mapping):
            raise CapabilityInputError(
                "routine_outcome_contract_invalid",
                "Each artifact requirement must be an exact object.",
            )
        raw_authorships = _strings(item, "allowed_authorships")
        try:
            requirements.append(
                ArtifactRequirement(
                    required=_boolean(item, "required"),
                    minimum_count=_integer(item, "minimum_count"),
                    maximum_count=_integer(item, "maximum_count"),
                    allowed_media_types=_strings(item, "allowed_media_types"),
                    allowed_authorships=tuple(
                        ArtifactAuthorship(authorship) for authorship in raw_authorships
                    ),
                    allowed_producer_capability_ids=_strings(
                        item, "allowed_producer_capability_ids"
                    ),
                    maximum_artifact_bytes=_integer(item, "maximum_artifact_bytes"),
                    maximum_total_bytes=_integer(item, "maximum_total_bytes"),
                    maximum_sensitivity=ModelSensitivity(
                        _string(item, "maximum_sensitivity")
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise CapabilityInputError(
                "routine_outcome_contract_invalid",
                "An artifact requirement is invalid.",
            ) from error
    try:
        effects_value = value.get("effect_requirements", ())
        if not isinstance(effects_value, tuple) or any(
            not isinstance(item, Mapping) for item in effects_value
        ):
            raise ValueError("effect requirements must be exact objects")
        return OutcomeContract(
            effect_requirements=tuple(
                EffectRequirement(
                    capability_id=_string(item, "capability_id"),
                    minimum_successful_calls=_integer(item, "minimum_successful_calls"),
                    accepted_evidence_bases=frozenset(
                        EffectEvidenceBasis(basis)
                        for basis in _strings(item, "accepted_evidence_bases")
                    ),
                )
                for item in effects_value
            ),
            require_terminal_conclusion=True,
            artifact_requirements=tuple(requirements),
            maximum_total_artifact_bytes=_integer(
                value, "maximum_total_artifact_bytes"
            ),
            maximum_effective_sensitivity=ModelSensitivity(
                _string(value, "maximum_effective_sensitivity")
            ),
            require_current_run_provenance=True,
            require_exact_source_bindings=requirements_need_exact_source_bindings(
                tuple(requirements)
            ),
        )
    except (TypeError, ValueError) as error:
        raise CapabilityInputError(
            "routine_outcome_contract_invalid",
            "The outcome contract is invalid.",
        ) from error


_SCHEDULE_KEYS: dict[str, frozenset[str]] = {
    "once": frozenset({"exact_at", "after_seconds"}),
    "interval": frozenset({"interval_seconds", "anchor_at", "anchor_after_seconds"}),
    "once_next_weekday": frozenset({"timezone", "weekday", "hour", "minute"}),
    "calendar": frozenset(
        {
            "timezone",
            "hour",
            "minute",
            "day_selector",
            "weekdays",
            "month_days",
            "months",
            "nonexistent_time_policy",
            "ambiguous_time_policy",
        }
    ),
}


def _schedule_schema() -> dict[str, object]:
    """One flat typed schedule; the owner checks which keys each kind uses."""

    def days(maximum: int) -> dict[str, object]:
        return {
            "type": "array",
            "uniqueItems": True,
            "maxItems": maximum,
            "items": {"type": "integer", "minimum": 1, "maximum": maximum},
        }

    relative = {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_RELATIVE_SCHEDULE_SECONDS,
    }
    instant = {"type": "string", "minLength": 1, "maxLength": 64}
    return {
        "type": "object",
        "description": (
            "once: exact_at or after_seconds. interval: interval_seconds with "
            "anchor_at or anchor_after_seconds. once_next_weekday: timezone, "
            "weekday (Monday=1), hour, minute. calendar: timezone, hour, minute, "
            "sorted weekdays or month_days (neither means every day), optional "
            "months. Omit keys a kind does not use. Instants are ISO 8601 with "
            "timezone; timezones are IANA. No cron or RRULE. Calendar policies "
            "default to skip and first."
        ),
        "properties": {
            "kind": {"type": "string", "enum": list(_SCHEDULE_KEYS)},
            "exact_at": instant,
            "after_seconds": relative,
            "interval_seconds": {
                "type": "integer",
                "minimum": MIN_ROUTINE_INTERVAL_SECONDS,
                "maximum": MAX_ROUTINE_INTERVAL_SECONDS,
            },
            "anchor_at": instant,
            "anchor_after_seconds": relative,
            "timezone": {"type": "string", "minLength": 1, "maxLength": 256},
            "weekday": {"type": "integer", "minimum": 1, "maximum": 7},
            "hour": {"type": "integer", "minimum": 0, "maximum": 23},
            "minute": {"type": "integer", "minimum": 0, "maximum": 59},
            "day_selector": {
                "type": "string",
                "enum": [item.value for item in CalendarDaySelector],
            },
            "weekdays": days(7),
            "month_days": days(31),
            "months": days(12),
            "nonexistent_time_policy": {
                "type": "string",
                "enum": [item.value for item in NonexistentTimePolicy],
            },
            "ambiguous_time_policy": {
                "type": "string",
                "enum": [item.value for item in AmbiguousTimePolicy],
            },
        },
        "required": ["kind"],
        "additionalProperties": False,
    }


def _spec_schema(*, update: bool) -> dict[str, object]:
    """Create and update share every common rule so routine_save can merge them.

    Optional arguments have code-owned defaults that never widen scope, grants
    or budgets: misfire skip, no connector bindings, grants, skills, precheck,
    dollar ceilings or immediate run. Relative timing is create-only.
    """

    properties: dict[str, object] = {
        "title": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_ROUTINE_TITLE_CHARACTERS,
        },
        "authorized_instruction": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_ROUTINE_INSTRUCTION_BYTES,
        },
        "schedule": _schedule_schema(),
        "run_immediately": {
            "type": "boolean",
            "description": "Create only: true adds one immediate occurrence.",
        },
        "requested_capability_grants": {
            "type": "array",
            "maxItems": MAX_ROUTINE_IDENTITY_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "capability_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024,
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Use the exact search automation_contract, or direct load if needed. Grants are for effects; reads use allowed_capability_ids.",
                    },
                    "max_calls_per_occurrence": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 256,
                    },
                },
                "required": [
                    "capability_id",
                    "constraints",
                    "max_calls_per_occurrence",
                ],
                "additionalProperties": False,
            },
        },
        "misfire_policy": {
            "type": "string",
            "enum": [item.value for item in MisfirePolicy],
        },
        "reporting_mode": {
            "type": "string",
            "enum": [item.value for item in ReportingMode],
        },
        "precheck": {
            "type": "object",
            "description": "changes_only requires this; always/actions/MCP forbid it. Tracks structure, not rows. Use capability contract_digest.",
            "properties": {
                "capability_id": {"type": "string", "minLength": 1, "maxLength": 1024},
                "contract_digest": {
                    "type": "string",
                    "pattern": "^sha256:[0-9a-f]{64}$",
                },
                "source_id": source_id_schema(),
                "resource_id": resource_id_schema(),
            },
            "required": [
                "capability_id",
                "contract_digest",
                "source_id",
                "resource_id",
            ],
            "additionalProperties": False,
        },
        "allowed_source_ids": _identity_array_schema(source_id_schema()),
        "allowed_resource_ids": _identity_array_schema(resource_id_schema()),
        "allowed_capability_ids": _identity_array_schema(minimum=1),
        "sensitivity_ceiling": {
            "type": "string",
            "enum": [item.value for item in ModelSensitivity],
        },
        "outcome_contract": _outcome_contract_schema(),
        "distribution_destination_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1024,
            "description": "Daita Inbox for each run's report; omit for this conversation's Inbox. Never a connector argument.",
        },
        "allowed_connector_binding_ids": {
            **_identity_array_schema(),
            "description": "Omit to derive from allowed_capability_ids.",
        },
        "eligible_model_routes": _identity_array_schema(minimum=1),
        "per_run_max_tokens": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_ROUTINE_PER_RUN_TOKENS,
        },
        "per_run_max_cost_usd": _money_schema(MAX_ROUTINE_PER_RUN_COST_USD),
        "cumulative_max_tokens": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_ROUTINE_CUMULATIVE_TOKENS,
        },
        "cumulative_max_cost_usd": _money_schema(MAX_ROUTINE_CUMULATIVE_COST_USD),
        "cumulative_max_attempts": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_ROUTINE_OCCURRENCES * 3,
        },
        "cumulative_max_occurrences": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_ROUTINE_OCCURRENCES,
        },
        "maximum_consecutive_failures": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_ROUTINE_CONSECUTIVE_FAILURES,
        },
        "expires_at": {"type": "string", "minLength": 1, "maxLength": 64},
        "expires_after_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_RELATIVE_SCHEDULE_SECONDS,
            "description": "Create only, instead of expires_at: seconds after creation.",
        },
        "skill_names": {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9-]{0,63}$",
                "maxLength": 64,
            },
            "maxItems": MAX_ROUTINE_SKILL_BINDINGS,
            "uniqueItems": True,
        },
        "basis_run_id": {"type": "string", "minLength": 1, "maxLength": 1024},
    }
    required = [
        "title",
        "authorized_instruction",
        "schedule",
        "reporting_mode",
        "allowed_source_ids",
        "allowed_resource_ids",
        "allowed_capability_ids",
        "sensitivity_ceiling",
        "outcome_contract",
        "eligible_model_routes",
        "per_run_max_tokens",
        "cumulative_max_tokens",
        "cumulative_max_attempts",
        "cumulative_max_occurrences",
        "maximum_consecutive_failures",
    ]
    if update:
        properties.update(
            {
                "routine_id": {"type": "string", "minLength": 1, "maxLength": 1024},
                "expected_revision": {"type": "integer", "minimum": 1},
            }
        )
        required.extend(("routine_id", "expected_revision"))
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _routine_id_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "routine_id": {"type": "string", "minLength": 1, "maxLength": 1024}
        },
        "required": ["routine_id"],
        "additionalProperties": False,
    }


# Requested grant constraints carry the named capability's own source/resource
# identities, so their run-scoped handles resolve before validation.
_GRANT_DELEGATION = (("requested_capability_grants[]", "constraints", "grant"),)


def _identity_array_schema(
    items: Mapping[str, object] | None = None, *, minimum: int = 0
) -> dict[str, object]:
    return {
        "type": "array",
        "items": items or {"type": "string", "minLength": 1, "maxLength": 1024},
        "minItems": minimum,
        "maxItems": MAX_ROUTINE_IDENTITY_ITEMS,
        "uniqueItems": True,
    }


_DERIVED_OUTCOME_KEYS = frozenset(
    {
        "require_terminal_conclusion",
        "require_current_run_provenance",
        "require_exact_source_bindings",
    }
)


def _without_derived_outcome_rules(
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    """Drop the outcome rules code derives; a copied routine view still has them."""

    contract = arguments.get("outcome_contract")
    if not isinstance(contract, Mapping):
        return arguments
    return {
        **arguments,
        "outcome_contract": {
            key: value
            for key, value in contract.items()
            if key not in _DERIVED_OUTCOME_KEYS
        },
    }


def _outcome_contract_schema() -> dict[str, object]:
    identity_array = _identity_array_schema()
    media_array = {
        "type": "array",
        "items": {"type": "string", "minLength": 3, "maxLength": 128},
        "maxItems": 16,
        "uniqueItems": True,
    }
    requirement = {
        "type": "object",
        "properties": {
            "required": {"type": "boolean"},
            "minimum_count": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_OUTCOME_ARTIFACT_REFERENCES,
            },
            "maximum_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_OUTCOME_ARTIFACT_REFERENCES,
            },
            "allowed_media_types": media_array,
            "allowed_authorships": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [item.value for item in ArtifactAuthorship],
                },
                "maxItems": len(ArtifactAuthorship),
                "uniqueItems": True,
            },
            "allowed_producer_capability_ids": identity_array,
            "maximum_artifact_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_OUTCOME_ARTIFACT_BYTES,
            },
            "maximum_total_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
            },
            "maximum_sensitivity": {
                "type": "string",
                "enum": [item.value for item in ModelSensitivity],
            },
        },
        "required": [
            "required",
            "minimum_count",
            "maximum_count",
            "allowed_media_types",
            "allowed_authorships",
            "allowed_producer_capability_ids",
            "maximum_artifact_bytes",
            "maximum_total_bytes",
            "maximum_sensitivity",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "effect_requirements": {
                "type": "array",
                "maxItems": MAX_ROUTINE_IDENTITY_ITEMS,
                "items": {
                    "type": "object",
                    "properties": {
                        "capability_id": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1024,
                        },
                        "minimum_successful_calls": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 256,
                        },
                        "accepted_evidence_bases": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["adapter_verified", "server_reported"],
                            },
                            "minItems": 1,
                            "maxItems": 2,
                            "uniqueItems": True,
                        },
                    },
                    "required": [
                        "capability_id",
                        "minimum_successful_calls",
                        "accepted_evidence_bases",
                    ],
                    "additionalProperties": False,
                },
            },
            "artifact_requirements": {
                "type": "array",
                "items": requirement,
                "maxItems": MAX_OUTCOME_ARTIFACT_REQUIREMENTS,
            },
            "maximum_total_artifact_bytes": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
            },
            "maximum_effective_sensitivity": {
                "type": "string",
                "enum": [item.value for item in ModelSensitivity],
            },
        },
        "required": [
            "artifact_requirements",
            "maximum_total_artifact_bytes",
            "maximum_effective_sensitivity",
        ],
        "additionalProperties": False,
    }


def _money_schema(maximum: Decimal) -> dict[str, object]:
    del maximum
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": 32,
        "pattern": r"^[0-9]+(?:\.[0-9]+)?$",
    }


def _object_output_schema(required: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {name: {} for name in required},
        "required": list(required),
        "additionalProperties": False,
    }


def _summary_payload(item: ScheduledRoutineSummary) -> dict[str, object]:
    return {
        "routine_id": item.routine_id,
        "title": item.title,
        "state": item.state.value,
        "schedule_kind": item.schedule_kind.value,
        "next_due_at": (
            None if item.next_due_at is None else item.next_due_at.isoformat()
        ),
        "revision": item.revision,
        "occurrence_count": item.occurrence_count,
        "consecutive_failures": item.consecutive_failures,
    }


def _mutation_receipt(action: str, item: ScheduledRoutine) -> dict[str, object]:
    """Confirm the committed mutation; execution evidence belongs to occurrences.

    Approval and enforcement use the complete retained contract. Returning that
    contract again after every mutation needlessly duplicates it in the exact
    transcript. Inspection remains the explicit full-record read.
    """
    return {
        "action": action,
        "routine": {
            "routine_id": item.routine_id,
            "title": item.title,
            "revision": item.revision,
            "state": item.state.value,
            "schedule": _schedule_payload(item.schedule),
            "next_due_at": (
                None if item.next_due_at is None else item.next_due_at.isoformat()
            ),
            "reserved_occurrences": item.occurrence_count,
            "reserved_attempts": item.attempt_count,
            "upcoming": schedule_preview(item, not_before=item.next_due_at),
        },
    }


def _mutation_receipt_schema() -> dict[str, object]:
    fields: dict[str, object] = {
        "routine_id": {"type": "string", "minLength": 1, "maxLength": 1024},
        "title": {"type": "string", "maxLength": MAX_ROUTINE_TITLE_CHARACTERS},
        "revision": {"type": "integer", "minimum": 1},
        "state": {"type": "string", "enum": [item.value for item in RoutineState]},
        "schedule": _schedule_schema(),
        "next_due_at": {"type": ["string", "null"], "maxLength": 64},
        "reserved_occurrences": {"type": "integer", "minimum": 0},
        "reserved_attempts": {"type": "integer", "minimum": 0},
        "upcoming": _schedule_preview_schema(),
    }
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create",
                    "update",
                    *(item.value for item in RoutineControlAction),
                ],
            },
            "routine": {
                "type": "object",
                "properties": fields,
                "required": list(fields),
                "additionalProperties": False,
            },
        },
        "required": ["action", "routine"],
        "additionalProperties": False,
    }


def _schedule_preview_schema() -> dict[str, object]:
    text = {"type": "string", "maxLength": 512}
    fields: dict[str, object] = {
        "timezone": {"type": "string", "maxLength": 256},
        "first_occurrences": {
            "type": "array",
            "maxItems": MAX_SCHEDULE_PREVIEW_OCCURRENCES,
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "string", "maxLength": 64},
                    "local": {"type": "string", "maxLength": 128},
                },
                "required": ["at", "local"],
                "additionalProperties": False,
            },
        },
        "immediate_first_run": {"type": "boolean"},
        "occurrences_before_expiry": {"type": "integer", "minimum": 0},
        "warnings": {"type": "array", "maxItems": 4, "items": text},
    }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def routine_projection(item: ScheduledRoutine) -> dict[str, object]:
    payload = _routine_proposal_payload(item)
    payload.update(
        {
            "state": item.state.value,
            "next_due_at": (
                None if item.next_due_at is None else item.next_due_at.isoformat()
            ),
            "occurrence_count": item.occurrence_count,
            "attempt_count": item.attempt_count,
            "charged_tokens": item.charged_tokens,
            "charged_cost_usd": str(item.charged_cost_usd),
            "consecutive_failures": item.consecutive_failures,
        }
    )
    return payload


def _occurrence_payload(item: RoutineOccurrence) -> dict[str, object]:
    return {
        "occurrence_id": item.occurrence_id,
        "routine_revision": item.routine_revision,
        "slot_kind": item.slot_kind.value,
        "scheduled_for": item.scheduled_for.isoformat(),
        "disposition": item.disposition.value,
        "reserved_run_id": item.reserved_run_id,
        "terminal_run_id": item.terminal_run_id,
        "delivery_ids": item.delivery_ids,
        "effect_receipt_ids": item.effect_receipt_ids,
        "failure_code": item.failure_code,
        "attempt_count": item.attempt_count,
        "updated_at": item.updated_at.isoformat(),
    }


def routine_inspection_projection(
    item: ScheduledRoutineInspection,
) -> dict[str, object]:
    return {
        "routine": routine_projection(item.routine),
        "recent_occurrences": tuple(
            _occurrence_payload(occurrence) for occurrence in item.recent_occurrences
        ),
    }


def _string(values: Mapping[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise CapabilityInputError("routine_argument_invalid", f"{name} must be text.")
    return value


def _integer(values: Mapping[str, object], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CapabilityInputError(
            "routine_argument_invalid", f"{name} must be an integer."
        )
    return value


def _boolean(values: Mapping[str, object], name: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise CapabilityInputError(
            "routine_argument_invalid", f"{name} must be a boolean."
        )
    return value


def _mapping(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = values.get(name)
    if not isinstance(value, Mapping):
        raise CapabilityInputError(
            "routine_argument_invalid", f"{name} must be an object."
        )
    return value


def _strings(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = values.get(name)
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise CapabilityInputError(
            "routine_argument_invalid", f"{name} must be an array of text."
        )
    return tuple(value)


def _optional_integers(values: Mapping[str, object], name: str) -> tuple[int, ...]:
    value = values.get(name, ())
    if not isinstance(value, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise CapabilityInputError(
            "routine_argument_invalid", f"{name} must be an integer array."
        )
    return tuple(value)


def _decimal(values: Mapping[str, object], name: str) -> Decimal:
    try:
        return Decimal(_string(values, name))
    except InvalidOperation as error:
        raise CapabilityInputError(
            "routine_argument_invalid", f"{name} must be decimal text."
        ) from error


def _optional_decimal(values: Mapping[str, object], name: str) -> Decimal | None:
    return _decimal(values, name) if name in values else None


def _datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CapabilityInputError(
            "routine_datetime_invalid", "Datetime must be ISO-8601 text."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CapabilityInputError(
            "routine_datetime_invalid", "Datetime must include an exact UTC offset."
        )
    return parsed.astimezone(UTC)


__all__ = [
    "ROUTINE_CONTROL_CAPABILITY_ID",
    "ROUTINE_CREATE_CAPABILITY_ID",
    "ROUTINE_DOMAIN_OWNER_ID",
    "ROUTINE_INSPECT_CAPABILITY_ID",
    "ROUTINE_LIST_CAPABILITY_ID",
    "ROUTINE_UPDATE_CAPABILITY_ID",
    "RoutineCapabilityDomain",
    "routine_capability_declarations",
]
