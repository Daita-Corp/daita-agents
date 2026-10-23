"""Static foreground logical-delivery read capabilities."""

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
    MAX_DELIVERY_LIST_PAGE_SIZE,
    delivery_inspection_projection,
    delivery_view_projection,
)
from .owner import DistributionOwner

DISTRIBUTION_DOMAIN_OWNER_ID = "distribution"

DELIVERY_LIST_CAPABILITY_ID = "distribution.delivery.list"
DELIVERY_INSPECT_CAPABILITY_ID = "distribution.delivery.inspect"

DELIVERY_LIST_EXECUTOR_ID = "distribution.delivery.list.executor"
DELIVERY_INSPECT_EXECUTOR_ID = "distribution.delivery.inspect.executor"

DELIVERY_LIST_TOOL_NAME = "delivery_list"
DELIVERY_INSPECT_TOOL_NAME = "delivery_inspect"


@dataclass(frozen=True, slots=True)
class DistributionCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class DeliveryListExecutor:
    executor_id = DELIVERY_LIST_EXECUTOR_ID

    def __init__(self, owner: DistributionOwner) -> None:
        self._owner = owner

    async def execute(self, request: ToolExecution) -> ToolOutput:
        conversation_id = request.arguments.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise ValueError("delivery conversation_id is invalid")
        include_acknowledged = request.arguments.get("include_acknowledged", False)
        limit = request.arguments.get("limit", MAX_DELIVERY_LIST_PAGE_SIZE)
        if not isinstance(include_acknowledged, bool):
            raise ValueError("delivery acknowledgment filter is invalid")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("delivery list limit is invalid")
        values = await self._owner.list(
            conversation_id=conversation_id,
            include_acknowledged=include_acknowledged,
            limit=limit,
        )
        sensitivity = _maximum_sensitivity(
            tuple(item.effective_sensitivity for item in values)
        )
        return ToolOutput(
            kind="distribution.delivery_list",
            data={
                "deliveries": [delivery_view_projection(item) for item in values],
                "count": len(values),
            },
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": "distribution_owner_delivery_projection",
                "agent_id": self._owner.agent_id,
            },
        )


class DeliveryInspectExecutor:
    executor_id = DELIVERY_INSPECT_EXECUTOR_ID

    def __init__(self, owner: DistributionOwner) -> None:
        self._owner = owner

    async def execute(self, request: ToolExecution) -> ToolOutput:
        delivery_id = request.arguments.get("delivery_id")
        if not isinstance(delivery_id, str):
            raise ValueError("delivery_id is invalid")
        inspection = await self._owner.inspect(delivery_id)
        if inspection is None:
            raise ValueError("delivery_not_found")
        return ToolOutput(
            kind="distribution.delivery_inspection",
            data=delivery_inspection_projection(inspection),
            sensitivity=inspection.delivery.outcome.effective_sensitivity,
            sensitivity_provenance={
                "authority": "distribution_owner_delivery_projection",
                "agent_id": self._owner.agent_id,
                "delivery_id": delivery_id,
            },
        )


class DistributionCapabilityDomain:
    """Implement the exact interactive-only distribution read surface."""

    domain_owner_id = DISTRIBUTION_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        owner: DistributionOwner,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("distribution declarations have the wrong owner")
        self._declarations = declarations
        self._owner = owner
        self._views = tuple(declarations.tool_views)

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        if run.agent_id != self._owner.agent_id or run.origin is not RunOrigin.USER:
            return ()
        return tuple(item.name for item in self._views)

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
        if run.agent_id != self._owner.agent_id or run.origin is not RunOrigin.USER:
            raise ValueError("distribution_foreground_required")
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
        del run, call, capability, execution, fingerprint
        raise ValueError("distribution capabilities are effect-free")

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
        if isinstance(error, ValueError) and str(error) == "delivery_not_found":
            return CapabilityFailure(
                "delivery_not_found",
                "The requested delivery is not owned by this agent.",
            )
        return None


def distribution_capability_declarations(
    owner: DistributionOwner,
) -> DistributionCapabilityDeclarations:
    capabilities = (
        Capability(
            id=DELIVERY_LIST_CAPABILITY_ID,
            description="List bounded logical deliveries owned by this agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "include_acknowledged": {"type": "boolean"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_DELIVERY_LIST_PAGE_SIZE,
                    },
                },
                "additionalProperties": False,
            },
            output_kind="distribution.delivery_list",
            output_schema=_list_output_schema("deliveries"),
            executor_id=DELIVERY_LIST_EXECUTOR_ID,
            access_mode=AccessMode.NONE,
            automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        ),
        Capability(
            id=DELIVERY_INSPECT_CAPABILITY_ID,
            description=(
                "Inspect one exact logical delivery, including immutable artifact "
                "and provenance facts."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "delivery_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1_024,
                    }
                },
                "required": ["delivery_id"],
                "additionalProperties": False,
            },
            output_kind="distribution.delivery_inspection",
            output_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
            executor_id=DELIVERY_INSPECT_EXECUTOR_ID,
            access_mode=AccessMode.NONE,
            automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        ),
    )
    executors: tuple[Executor, ...] = (
        DeliveryListExecutor(owner),
        DeliveryInspectExecutor(owner),
    )
    by_id = {
        DELIVERY_LIST_CAPABILITY_ID: (
            DELIVERY_LIST_TOOL_NAME,
            "List logical deliveries owned by this agent.",
            "Use when a delivery ID is unknown or recent outcomes are requested.",
            ("delivery", "inbox", "outcome"),
            ToolLoadMode.PINNED,
        ),
        DELIVERY_INSPECT_CAPABILITY_ID: (
            DELIVERY_INSPECT_TOOL_NAME,
            "Inspect one logical delivery.",
            "Use for exact conclusion, artifact, checksum, or provenance facts.",
            ("delivery", "inspect", "artifact", "checksum"),
            ToolLoadMode.ON_DEMAND,
        ),
    }
    views = tuple(
        ToolView(
            name=by_id[item.id][0],
            capability_id=item.id,
            description=item.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ROUTINES,
                load_mode=by_id[item.id][4],
                text_trust=ToolTextTrust.CODE,
                summary=by_id[item.id][1],
                when_to_use=by_id[item.id][2],
                keywords=by_id[item.id][3],
            ),
        )
        for item in capabilities
    )
    return DistributionCapabilityDeclarations(capabilities, executors, views)


def _list_output_schema(name: str) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "array"},
            "count": {"type": "integer"},
        },
        "required": [name, "count"],
        "additionalProperties": False,
    }


def _maximum_sensitivity(
    values: tuple[ModelSensitivity, ...],
) -> ModelSensitivity:
    if not values:
        return ModelSensitivity.PUBLIC
    return max(values, key=lambda item: item.routing_rank)


__all__ = [
    "DELIVERY_INSPECT_CAPABILITY_ID",
    "DELIVERY_INSPECT_EXECUTOR_ID",
    "DELIVERY_INSPECT_TOOL_NAME",
    "DELIVERY_LIST_CAPABILITY_ID",
    "DELIVERY_LIST_EXECUTOR_ID",
    "DELIVERY_LIST_TOOL_NAME",
    "DISTRIBUTION_DOMAIN_OWNER_ID",
    "DistributionCapabilityDeclarations",
    "DistributionCapabilityDomain",
    "distribution_capability_declarations",
]
