"""Static foreground tools for scheduled-routine management."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

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
from ..loop.models import RunInput, RunOrigin
from .models import (
    MAX_ROUTINE_LIST_PAGE_SIZE,
    MAX_ROUTINE_TITLE_CHARACTERS,
    RoutineControlAction,
    RoutineIntent,
    RoutineOccurrence,
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
)
from .owner import (
    PreparedRoutine,
    RoutineError,
    RoutineOwner,
    schedule_payload,
    schedule_text,
)

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
    "Save a routine: a self-contained prompt run on a schedule, optionally with "
    "only named sources. Omit routine_id to create; with it, omitted fields keep "
    "their values. Runs ask before writes."
)
ROUTINE_CONTROL_CAPABILITY_ID = "routines.control"
ROUTINE_CONTROL_EXECUTOR_ID = "routines.control.executor"
ROUTINE_CONTROL_TOOL_NAME = "routine_control"
# Routine prompts are a person's retained instructions: never treated as public.
_ROUTINE_SENSITIVITY_FLOOR = ModelSensitivity.INTERNAL


@dataclass(frozen=True, slots=True)
class RoutineCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class _RoutineExecutor:
    def __init__(self, owner: RoutineOwner) -> None:
        self._owner = owner


def _output(
    kind: str, request: ToolExecution, data: Mapping[str, object]
) -> ToolOutput:
    return ToolOutput(
        kind=kind,
        sensitivity=max(
            request.request_sensitivity,
            _ROUTINE_SENSITIVITY_FLOOR,
            key=lambda item: item.routing_rank,
        ),
        sensitivity_provenance={"authority": "agent_owned_routines"},
        data=data,
    )


class RoutineListExecutor(_RoutineExecutor):
    executor_id = ROUTINE_LIST_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        raw_states = request.arguments.get("states", ())
        assert isinstance(raw_states, tuple)
        summaries = await self._owner.list(
            states=frozenset(RoutineState(item) for item in raw_states),
            limit=MAX_ROUTINE_LIST_PAGE_SIZE,
        )
        return _output(
            "routine.list",
            request,
            {
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
        return _output(
            "routine.inspection", request, routine_inspection_projection(inspection)
        )


class _RoutineSaveExecutor(_RoutineExecutor):
    action = "create"

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        return (await self._prepare(request, commit=False)).card

    async def execute(self, request: ToolExecution) -> ToolOutput:
        # Relative timing counts from now, once approval has completed.
        stored = await self._owner.save(await self._prepare(request, commit=True))
        return _output(
            "routine.receipt", request, _mutation_receipt(self.action, stored)
        )

    async def _prepare(
        self, request: ToolExecution, *, commit: bool
    ) -> PreparedRoutine:
        arguments = request.arguments
        if request.conversation_id is None:
            raise RoutineError(
                "routine_conversation_required",
                "Routine management requires an exact conversation.",
            )
        reference = arguments.get("_timing_reference_at")
        try:
            intent = RoutineIntent.from_mapping(
                {
                    key: value
                    for key, value in arguments.items()
                    if key not in {"routine_id", "expected_revision"}
                    and not key.startswith("_")
                }
            )
        except (TypeError, ValueError) as error:
            raise RoutineError("routine_argument_invalid", str(error)) from error
        routine_id = arguments.get("routine_id")
        expected = arguments.get("expected_revision")
        return await self._owner.prepare(
            intent,
            conversation_id=request.conversation_id,
            identity=f"{request.run_id}\x00{request.call_id}",
            routine_id=routine_id if isinstance(routine_id, str) else None,
            expected_revision=expected if isinstance(expected, int) else None,
            origin_run_id=request.run_id,
            reference_at=(
                datetime.fromisoformat(reference)
                if isinstance(reference, str)
                else None
            ),
            relative_at=self._owner.current_time() if commit else None,
        )


class RoutineCreateExecutor(_RoutineSaveExecutor):
    executor_id = ROUTINE_CREATE_EXECUTOR_ID


class RoutineUpdateExecutor(_RoutineSaveExecutor):
    executor_id = ROUTINE_UPDATE_EXECUTOR_ID
    action = "update"


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
        return FrozenJsonObject.from_mapping(
            {
                "routine_id": routine_id,
                "title": inspection.routine.title,
                "expected_revision": expected_revision,
                "action": action.value,
                "current_state": inspection.routine.state.value,
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        routine = await self._owner.control(
            _string(request.arguments, "routine_id"),
            expected_revision=_integer(request.arguments, "expected_revision"),
            action=RoutineControlAction(_string(request.arguments, "action")),
            authorized_control_call_id=request.call_id,
        )
        return _output(
            "routine.receipt",
            request,
            _mutation_receipt(_string(request.arguments, "action"), routine),
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
        del call, request_sensitivity
        if (
            run.agent_id != self._owner.agent_id
            or run.origin is not RunOrigin.USER
            or run.conversation_id is None
        ):
            raise CapabilityInputError(
                "routine_foreground_required",
                "Routine management requires an exact foreground conversation.",
            )
        if (
            capability.id
            in {ROUTINE_CREATE_CAPABILITY_ID, ROUTINE_UPDATE_CAPABILITY_ID}
            and "schedule" in arguments
        ):
            # One reference instant for the card and its post-approval recheck;
            # the save itself counts relative timing from when it commits.
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
        del run, call, execution
        if (
            capability.operational_effect
            is not OperationalEffect.MANAGE_SCHEDULED_ROUTINE
        ):
            raise ValueError("routine domain received an unsupported effect")
        reason = {
            ROUTINE_CREATE_CAPABILITY_ID: "Save this routine? Each run uses the "
            "agent's current model and asks before any write.",
            ROUTINE_UPDATE_CAPABILITY_ID: "Save these changes to the routine?",
        }.get(capability.id, "Apply this routine control action once?")
        return SideEffectPlan(
            approval_arguments=fingerprint,
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
            return CapabilityFailure(error.code, str(error), error.details)
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
        description="Inspect one routine and its recent occurrences.",
        input_schema=_routine_id_schema(),
        output_kind="routine.inspection",
        output_schema=_object_output_schema(("routine", "recent_occurrences")),
        executor_id=ROUTINE_INSPECT_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    create_capability = Capability(
        id=ROUTINE_CREATE_CAPABILITY_ID,
        description="Create one routine from a title, prompt and schedule.",
        input_schema=_save_schema(update=False),
        output_kind="routine.receipt",
        output_schema=_mutation_receipt_schema(),
        executor_id=ROUTINE_CREATE_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.MANAGE_SCHEDULED_ROUTINE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    update_capability = Capability(
        id=ROUTINE_UPDATE_CAPABILITY_ID,
        description="Change one routine; omitted fields keep their values.",
        input_schema=_save_schema(update=True),
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
                "Use for a routine's prompt, schedule and recent occurrences.",
                ("routine", "inspect", "history"),
            ),
        ),
        ToolView(
            name=ROUTINE_SAVE_TOOL_NAME,
            capability_id=create_capability.id,
            description=ROUTINE_SAVE_DESCRIPTION,
            presentation=presentation(
                "Create or change a scheduled routine.",
                "Use to run a prompt later or repeatedly; relative timing needs no clock query.",
                (
                    "routine",
                    "schedule",
                    "recurring",
                    "report",
                    "create",
                    "update",
                    "revise",
                ),
            ),
            mode_capability_ids=(
                ("create", create_capability.id),
                ("update", update_capability.id),
            ),
            mode_argument="routine_id",
            mode_by_presence=True,
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


def _schedule_schema() -> dict[str, object]:
    """One flat schedule shape; the owner checks which keys each kind uses."""

    days = {"type": "array", "items": {"type": "integer"}}
    return {
        "type": "object",
        "description": (
            "once: exact_at|after_seconds. interval: interval_seconds, "
            "anchor_at|anchor_after_seconds. once_next_weekday: timezone, weekday "
            "(Mon=1), hour, minute. calendar: timezone, hour, minute, optional "
            "weekdays/month_days/months. ISO 8601 with offset; IANA zones."
        ),
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["once", "interval", "once_next_weekday", "calendar"],
            },
            "exact_at": {"type": "string"},
            "after_seconds": {"type": "integer"},
            "interval_seconds": {"type": "integer"},
            "anchor_at": {"type": "string"},
            "anchor_after_seconds": {"type": "integer"},
            "timezone": {"type": "string"},
            "weekday": {"type": "integer"},
            "hour": {"type": "integer"},
            "minute": {"type": "integer"},
            "weekdays": days,
            "month_days": days,
            "months": days,
        },
        "required": ["kind"],
        "additionalProperties": False,
    }


def _save_schema(*, update: bool) -> dict[str, object]:
    """Create and update share every common rule so routine_save can merge them."""

    # Bounds are checked when the routine is saved; the schema stays small.
    properties: dict[str, object] = {
        "title": {"type": "string"},
        "prompt": {"type": "string"},
        "schedule": _schedule_schema(),
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names or ids; default all.",
        },
    }
    if update:
        properties["routine_id"] = {"type": "string"}
        properties["expected_revision"] = {"type": "integer"}
        required = ["routine_id"]
    else:
        properties["run_immediately"] = {"type": "boolean"}
        required = ["title", "prompt", "schedule"]
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
    return {
        "action": action,
        "routine": {
            "routine_id": item.routine_id,
            "title": item.title,
            "revision": item.revision,
            "state": item.state.value,
            "schedule": schedule_text(item.schedule),
            "next_due_at": (
                None if item.next_due_at is None else item.next_due_at.isoformat()
            ),
        },
    }


def _mutation_receipt_schema() -> dict[str, object]:
    fields: dict[str, object] = {
        "routine_id": {"type": "string", "minLength": 1, "maxLength": 1024},
        "title": {"type": "string", "maxLength": MAX_ROUTINE_TITLE_CHARACTERS},
        "revision": {"type": "integer", "minimum": 1},
        "state": {"type": "string", "enum": [item.value for item in RoutineState]},
        "schedule": {"type": "string", "maxLength": 512},
        "next_due_at": {"type": ["string", "null"], "maxLength": 64},
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


def routine_projection(item: ScheduledRoutine) -> dict[str, object]:
    return {
        "routine_id": item.routine_id,
        "conversation_id": item.conversation_id,
        "title": item.title,
        "prompt": item.authorized_instruction,
        "instruction_digest": item.instruction_digest,
        "schedule": schedule_payload(item.schedule),
        "schedule_text": schedule_text(item.schedule),
        "source_ids": item.source_ids,
        "revision": item.revision,
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
