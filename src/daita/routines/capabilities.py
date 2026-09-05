"""Static foreground tools for bounded scheduled-routine management."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import TypedDict

from .._json import FrozenJsonObject
from ..artifacts.models import ArtifactAuthorship
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
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
from ..distribution.models import (
    MAX_OUTCOME_ARTIFACT_BYTES,
    MAX_OUTCOME_ARTIFACT_REFERENCES,
    MAX_OUTCOME_ARTIFACT_REQUIREMENTS,
    MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
    ArtifactRequirement,
    OutcomeContract,
)
from ..llm.models import ModelSensitivity, ToolCall
from ..loop.models import RunInput, RunOrigin
from .models import (
    MAX_ROUTINE_CONSECUTIVE_FAILURES,
    MAX_ROUTINE_CUMULATIVE_COST_USD,
    MAX_ROUTINE_CUMULATIVE_TOKENS,
    MAX_ROUTINE_IDENTITY_ITEMS,
    MAX_ROUTINE_INSTRUCTION_BYTES,
    MAX_ROUTINE_LIST_PAGE_SIZE,
    MAX_ROUTINE_OCCURRENCES,
    MAX_ROUTINE_PER_RUN_COST_USD,
    MAX_ROUTINE_PER_RUN_TOKENS,
    MAX_ROUTINE_SKILL_BINDINGS,
    MAX_ROUTINE_TITLE_CHARACTERS,
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    MisfirePolicy,
    NonexistentTimePolicy,
    OnceSchedule,
    ReportingMode,
    ResourceRevisionPrecheck,
    RoutineControlAction,
    RoutineOccurrence,
    RoutineSchedule,
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
)
from .owner import RoutineError, RoutineOwner, _routine_proposal_payload

ROUTINE_DOMAIN_OWNER_ID = "routines"
ROUTINE_LIST_CAPABILITY_ID = "routines.list"
ROUTINE_LIST_EXECUTOR_ID = "routines.list.executor"
ROUTINE_LIST_TOOL_NAME = "routine_list"
ROUTINE_INSPECT_CAPABILITY_ID = "routines.inspect"
ROUTINE_INSPECT_EXECUTOR_ID = "routines.inspect.executor"
ROUTINE_INSPECT_TOOL_NAME = "routine_inspect"
ROUTINE_CREATE_CAPABILITY_ID = "routines.create"
ROUTINE_CREATE_EXECUTOR_ID = "routines.create.executor"
ROUTINE_CREATE_TOOL_NAME = "routine_create"
ROUTINE_UPDATE_CAPABILITY_ID = "routines.update"
ROUTINE_UPDATE_EXECUTOR_ID = "routines.update.executor"
ROUTINE_UPDATE_TOOL_NAME = "routine_update"
ROUTINE_CONTROL_CAPABILITY_ID = "routines.control"
ROUTINE_CONTROL_EXECUTOR_ID = "routines.control.executor"
ROUTINE_CONTROL_TOOL_NAME = "routine_control"


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
            data=_inspection_payload(inspection),
        )


class RoutineCreateExecutor(_RoutineExecutor):
    executor_id = ROUTINE_CREATE_EXECUTOR_ID

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        proposal = await _create_proposal(self._owner, request)
        return await self._owner.proposal_authority_snapshot(proposal)

    async def execute(self, request: ToolExecution) -> ToolOutput:
        proposal = await _create_proposal(self._owner, request)
        stored = await self._owner.admit(proposal)
        return ToolOutput(
            kind="routine.receipt",
            data={"action": "create", "routine": _routine_payload(stored)},
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
            data={"action": "update", "routine": _routine_payload(stored)},
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
            data={
                "action": _string(request.arguments, "action"),
                "routine": _routine_payload(routine),
            },
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
        names = {ROUTINE_LIST_TOOL_NAME, ROUTINE_CREATE_TOOL_NAME}
        if await self._owner.list(limit=1):
            names.update(
                {
                    ROUTINE_INSPECT_TOOL_NAME,
                    ROUTINE_UPDATE_TOOL_NAME,
                    ROUTINE_CONTROL_TOOL_NAME,
                }
            )
        return tuple(view.name for view in self._views if view.name in names)

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        del capability
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
        if (
            run.agent_id != self._owner.agent_id
            or run.origin is not RunOrigin.USER
            or run.conversation_id is None
        ):
            raise CapabilityInputError(
                "routine_foreground_required",
                "Routine management requires an exact foreground conversation.",
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
        del run, call, execution
        if (
            capability.operational_effect
            is not OperationalEffect.MANAGE_SCHEDULED_ROUTINE
        ):
            raise ValueError("routine domain received an unsupported effect")
        if capability.id == ROUTINE_CREATE_CAPABILITY_ID:
            reason = "Create this exact scheduled read routine once?"
        elif capability.id == ROUTINE_UPDATE_CAPABILITY_ID:
            reason = "Replace this routine with the exact proposed revision once?"
        else:
            reason = "Apply this exact routine control action once?"
        proposal = fingerprint.get("routine")
        approval_arguments = (
            FrozenJsonObject.from_mapping({"proposal": proposal})
            if isinstance(proposal, Mapping)
            else fingerprint
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
            "Create one exact, finite, read-only routine from a self-contained instruction "
            "and typed schedule."
        ),
        input_schema=_spec_schema(update=False),
        output_kind="routine.receipt",
        output_schema=_object_output_schema(("action", "routine")),
        executor_id=ROUTINE_CREATE_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.MANAGE_SCHEDULED_ROUTINE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    update_capability = Capability(
        id=ROUTINE_UPDATE_CAPABILITY_ID,
        description="Replace one routine's material contract with an exact new revision.",
        input_schema=_spec_schema(update=True),
        output_kind="routine.receipt",
        output_schema=_object_output_schema(("action", "routine")),
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
        output_schema=_object_output_schema(("action", "routine")),
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
    names = {
        ROUTINE_LIST_CAPABILITY_ID: ROUTINE_LIST_TOOL_NAME,
        ROUTINE_INSPECT_CAPABILITY_ID: ROUTINE_INSPECT_TOOL_NAME,
        ROUTINE_CREATE_CAPABILITY_ID: ROUTINE_CREATE_TOOL_NAME,
        ROUTINE_UPDATE_CAPABILITY_ID: ROUTINE_UPDATE_TOOL_NAME,
        ROUTINE_CONTROL_CAPABILITY_ID: ROUTINE_CONTROL_TOOL_NAME,
    }
    guidance = {
        ROUTINE_LIST_CAPABILITY_ID: (
            "List scheduled routines.",
            "Use for routine inventory.",
            ("routine", "list"),
        ),
        ROUTINE_INSPECT_CAPABILITY_ID: (
            "Inspect one scheduled routine.",
            "Use for exact schedule, scope, budget, and recent occurrence details.",
            ("routine", "inspect", "history"),
        ),
        ROUTINE_CREATE_CAPABILITY_ID: (
            "Create one scheduled read routine.",
            "Use only after expressing a self-contained instruction and exact typed schedule.",
            ("routine", "schedule", "create"),
        ),
        ROUTINE_UPDATE_CAPABILITY_ID: (
            "Revise one scheduled routine.",
            "Use to replace a routine contract at an exact revision.",
            ("routine", "update", "revise"),
        ),
        ROUTINE_CONTROL_CAPABILITY_ID: (
            "Control one scheduled routine.",
            "Use to pause, resume, run now, or disable an exact revision.",
            ("routine", "pause", "resume", "disable"),
        ),
    }
    views = tuple(
        ToolView(
            name=names[capability.id],
            capability_id=capability.id,
            description=capability.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ROUTINES,
                load_mode=(
                    ToolLoadMode.PINNED
                    if capability.id == ROUTINE_LIST_CAPABILITY_ID
                    else ToolLoadMode.ON_DEMAND
                ),
                text_trust=ToolTextTrust.CODE,
                summary=guidance[capability.id][0],
                when_to_use=guidance[capability.id][1],
                keywords=guidance[capability.id][2],
            ),
        )
        for capability in capabilities
    )
    return RoutineCapabilityDeclarations(capabilities, executors, views)


async def _create_proposal(
    owner: RoutineOwner, request: ToolExecution
) -> ScheduledRoutine:
    if request.conversation_id is None:
        raise RoutineError(
            "routine_conversation_required",
            "Routine creation requires an exact conversation.",
        )
    values = _parsed_spec(request.arguments)
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
    allowed_connector_binding_ids: tuple[str, ...]
    allowed_resource_ids: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    sensitivity_ceiling: ModelSensitivity
    outcome_contract: OutcomeContract
    distribution_destination_id: str
    eligible_model_routes: tuple[str, ...]
    per_run_max_tokens: int
    per_run_max_cost_usd: Decimal
    cumulative_max_tokens: int
    cumulative_max_cost_usd: Decimal
    cumulative_max_attempts: int
    cumulative_max_occurrences: int
    maximum_consecutive_failures: int
    expires_at: datetime
    skill_names: tuple[str, ...]
    basis_run_id: str | None


def _parsed_spec(arguments: Mapping[str, object]) -> _ParsedSpec:
    return {
        "title": _string(arguments, "title"),
        "authorized_instruction": _string(arguments, "authorized_instruction"),
        "schedule": _parse_schedule(_mapping(arguments, "schedule")),
        "misfire_policy": MisfirePolicy(_string(arguments, "misfire_policy")),
        "reporting_mode": ReportingMode(_string(arguments, "reporting_mode")),
        "precheck": _parse_precheck(arguments.get("precheck")),
        "allowed_source_ids": _strings(arguments, "allowed_source_ids"),
        "allowed_connector_binding_ids": _strings(
            arguments, "allowed_connector_binding_ids"
        ),
        "allowed_resource_ids": _strings(arguments, "allowed_resource_ids"),
        "allowed_capability_ids": _strings(arguments, "allowed_capability_ids"),
        "sensitivity_ceiling": ModelSensitivity(
            _string(arguments, "sensitivity_ceiling")
        ),
        "outcome_contract": _parse_outcome_contract(
            _mapping(arguments, "outcome_contract")
        ),
        "distribution_destination_id": _string(
            arguments, "distribution_destination_id"
        ),
        "eligible_model_routes": _strings(arguments, "eligible_model_routes"),
        "per_run_max_tokens": _integer(arguments, "per_run_max_tokens"),
        "per_run_max_cost_usd": _decimal(arguments, "per_run_max_cost_usd"),
        "cumulative_max_tokens": _integer(arguments, "cumulative_max_tokens"),
        "cumulative_max_cost_usd": _decimal(arguments, "cumulative_max_cost_usd"),
        "cumulative_max_attempts": _integer(arguments, "cumulative_max_attempts"),
        "cumulative_max_occurrences": _integer(arguments, "cumulative_max_occurrences"),
        "maximum_consecutive_failures": _integer(
            arguments, "maximum_consecutive_failures"
        ),
        "expires_at": _datetime(_string(arguments, "expires_at")),
        "skill_names": _strings(arguments, "skill_names"),
        "basis_run_id": (
            None
            if arguments.get("basis_run_id") is None
            else _string(arguments, "basis_run_id")
        ),
    }


def _parse_schedule(value: Mapping[str, object]) -> RoutineSchedule:
    kind = _string(value, "kind")
    if kind == "once":
        return OnceSchedule(_datetime(_string(value, "exact_at")))
    if kind == "interval":
        return IntervalSchedule(
            _integer(value, "interval_seconds"),
            _datetime(_string(value, "anchor_at")),
        )
    if kind == "calendar":
        return CalendarSchedule(
            timezone=_string(value, "timezone"),
            hour=_integer(value, "hour"),
            minute=_integer(value, "minute"),
            day_selector=CalendarDaySelector(_string(value, "day_selector")),
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
        return OutcomeContract(
            require_terminal_conclusion=_boolean(value, "require_terminal_conclusion"),
            artifact_requirements=tuple(requirements),
            maximum_total_artifact_bytes=_integer(
                value, "maximum_total_artifact_bytes"
            ),
            maximum_effective_sensitivity=ModelSensitivity(
                _string(value, "maximum_effective_sensitivity")
            ),
            require_current_run_provenance=_boolean(
                value, "require_current_run_provenance"
            ),
            require_exact_source_bindings=_boolean(
                value, "require_exact_source_bindings"
            ),
        )
    except (TypeError, ValueError) as error:
        raise CapabilityInputError(
            "routine_outcome_contract_invalid",
            "The outcome contract is invalid.",
        ) from error


def _spec_schema(*, update: bool) -> dict[str, object]:
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
        "schedule": {"type": "object"},
        "misfire_policy": {
            "type": "string",
            "enum": [item.value for item in MisfirePolicy],
        },
        "reporting_mode": {
            "type": "string",
            "enum": [item.value for item in ReportingMode],
        },
        "precheck": {"type": "object"},
        "allowed_source_ids": _identity_array_schema(),
        "allowed_connector_binding_ids": _identity_array_schema(),
        "allowed_resource_ids": _identity_array_schema(),
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
        "misfire_policy",
        "reporting_mode",
        "allowed_source_ids",
        "allowed_connector_binding_ids",
        "allowed_resource_ids",
        "allowed_capability_ids",
        "sensitivity_ceiling",
        "outcome_contract",
        "distribution_destination_id",
        "eligible_model_routes",
        "per_run_max_tokens",
        "per_run_max_cost_usd",
        "cumulative_max_tokens",
        "cumulative_max_cost_usd",
        "cumulative_max_attempts",
        "cumulative_max_occurrences",
        "maximum_consecutive_failures",
        "expires_at",
        "skill_names",
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


def _identity_array_schema(*, minimum: int = 0) -> dict[str, object]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 1024},
        "minItems": minimum,
        "maxItems": MAX_ROUTINE_IDENTITY_ITEMS,
        "uniqueItems": True,
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
            "require_terminal_conclusion": {"type": "boolean"},
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
            "require_current_run_provenance": {"type": "boolean"},
            "require_exact_source_bindings": {"type": "boolean"},
        },
        "required": [
            "require_terminal_conclusion",
            "artifact_requirements",
            "maximum_total_artifact_bytes",
            "maximum_effective_sensitivity",
            "require_current_run_provenance",
            "require_exact_source_bindings",
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


def _routine_payload(item: ScheduledRoutine) -> dict[str, object]:
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
        "failure_code": item.failure_code,
        "attempt_count": item.attempt_count,
        "updated_at": item.updated_at.isoformat(),
    }


def _inspection_payload(item: ScheduledRoutineInspection) -> dict[str, object]:
    return {
        "routine": _routine_payload(item.routine),
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
    return parsed


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
