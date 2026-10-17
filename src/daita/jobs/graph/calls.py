"""Shared admission of the one exact call a graph worker task is frozen to.

Owner admission (``start_graph_job``) and planner mutations
(``graph_create_children`` and ``graph_supersede_unstarted``) create worker tasks
whose first call is fixed before any attempt runs. The worker must later make
exactly that call, so the frozen call is admitted here once, against current
registry truth, in the same normalised form the runtime gives the worker's own
call: unset nulls removed, the capability input schema satisfied, durable
graph eligibility and target bounds checked, referenced targets inside the
task's resource scope, and at least one model-visible tool able to make it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..._json import FrozenJsonObject
from ...capabilities import (
    CapabilityInputError,
    CapabilityRegistry,
    ExecutionAdmissionPolicy,
    ExecutionPreference,
    RecoveryKind,
    ToolRecovery,
    omit_unset_arguments,
)


@dataclass(frozen=True, slots=True)
class AdmittedGraphCall:
    """One validated call, frozen exactly as a graph worker must make it."""

    capability_id: str
    arguments: FrozenJsonObject
    callable_tool_names: tuple[str, ...]
    execution_policy: ExecutionAdmissionPolicy
    contract_digest: str

    def material(self) -> dict[str, object]:
        """Return the effect-free ``initial_call`` record."""

        return graph_call_material(
            self.capability_id, self.callable_tool_names, self.arguments
        )


def graph_call_material(
    capability_id: str,
    callable_tool_names: tuple[str, ...],
    arguments: Mapping[str, object],
) -> dict[str, object]:
    """Return the one effect-free ``initial_call`` record shape."""

    return {
        "capability_id": capability_id,
        "callable_tool_names": callable_tool_names,
        "arguments": arguments,
    }


def graph_callable_tool_names(
    registry: CapabilityRegistry, capability_id: str
) -> tuple[str, ...]:
    """Return every model-visible tool that can make one capability call."""

    names = tuple(
        sorted(
            name
            for name in registry.tool_names
            if capability_id in registry.tool_capability_ids(name)
        )
    )
    if not names:
        raise CapabilityInputError(
            "execution_not_supported_for_requested_shape",
            "The proposed capability has no model-visible tool in this host.",
        )
    return names


def graph_call_target_ids(
    policy: ExecutionAdmissionPolicy, arguments: Mapping[str, object]
) -> tuple[str, ...]:
    """Return the resource targets the policy's declared target argument names."""

    if policy.target_count_argument is None:
        return ()
    raw = arguments.get(policy.target_count_argument, ())
    if not isinstance(raw, (tuple, list)):
        return ()
    return tuple(sorted({item for item in raw if isinstance(item, str)}))


async def admit_graph_call(
    registry: CapabilityRegistry,
    capability_id: str,
    arguments: Mapping[str, object],
    *,
    argument_capability_id: str | None = None,
    resource_ids: Iterable[str] | None = None,
) -> AdmittedGraphCall:
    """Validate and freeze one graph worker call against current registry truth.

    ``argument_capability_id`` names a different capability whose input schema
    the arguments follow (a native write's preview). ``resource_ids`` is the
    task's resource scope; ``None`` skips the scope check for a caller that
    performs its own.
    """

    if not isinstance(capability_id, str) or not capability_id:
        raise CapabilityInputError(
            "execution_not_supported_for_requested_shape",
            "The proposed task needs one exact capability_id.",
        )
    if not isinstance(arguments, Mapping):
        raise CapabilityInputError(
            "invalid_argument_type",
            "The proposed task arguments must be an object.",
            recovery=ToolRecovery(
                RecoveryKind.CORRECT_ARGUMENTS,
                f"Send the arguments object for {capability_id}.",
            ),
        )
    schema_capability_id = argument_capability_id or capability_id
    try:
        capability = registry.capability(capability_id)
        schema_capability = registry.capability(schema_capability_id)
    except KeyError as error:
        raise CapabilityInputError(
            "execution_not_supported_for_requested_shape",
            "The proposed capability is not registered for this host.",
        ) from error
    normalized = omit_unset_arguments(arguments, (schema_capability.input_schema,))
    try:
        validated = await registry.validate_arguments_async(
            schema_capability_id, normalized
        )
    except CapabilityInputError as error:
        raise CapabilityInputError(
            error.code,
            f"The arguments for {capability_id} do not satisfy its input schema: "
            f"{error}",
            error.details,
        ) from None
    policy = capability.execution_admission_policy
    if (
        policy is None
        or policy.admission_error(validated, ExecutionPreference.DURABLE)
        != "durable_execution_required"
    ):
        raise CapabilityInputError(
            "execution_not_supported_for_requested_shape",
            "The proposed task is not eligible for the released graph contract.",
        )
    if resource_ids is not None:
        allowed = frozenset(resource_ids)
        outside = tuple(
            item
            for item in graph_call_target_ids(policy, validated)
            if item not in allowed
        )
        if outside:
            raise CapabilityInputError(
                "source_scope_violation",
                "The proposed task references resources outside its exact scope.",
                {"resource_ids": outside[:16]},
                recovery=ToolRecovery(
                    RecoveryKind.REDUCE_SCOPE,
                    "Use only resources inside the task's admitted scope.",
                ),
            )
    return AdmittedGraphCall(
        capability_id=capability_id,
        arguments=validated,
        callable_tool_names=graph_callable_tool_names(registry, capability_id),
        execution_policy=policy,
        contract_digest=registry.contract_digest(capability_id),
    )


class GraphCallAdmission:
    """Late-bound registry access for graph executors built before the registry.

    The capability registry is composed from the executors that need it, so the
    host binds the composed registry here, and rebinds it whenever it publishes
    a replacement catalog.
    """

    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self._registry: CapabilityRegistry | None = None
        if registry is not None:
            self.bind_capability_registry(registry)

    def bind_capability_registry(self, registry: CapabilityRegistry) -> None:
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("graph call admission requires CapabilityRegistry")
        self._registry = registry

    @property
    def registry(self) -> CapabilityRegistry:
        if self._registry is None:
            raise CapabilityInputError(
                "graph_call_admission_unavailable",
                "Graph call admission is not available in this host.",
            )
        return self._registry

    async def admit(
        self,
        capability_id: str,
        arguments: Mapping[str, object],
        *,
        resource_ids: Iterable[str] | None = None,
    ) -> AdmittedGraphCall:
        return await admit_graph_call(
            self.registry,
            capability_id,
            arguments,
            resource_ids=resource_ids,
        )


__all__ = [
    "AdmittedGraphCall",
    "GraphCallAdmission",
    "admit_graph_call",
    "graph_call_material",
    "graph_call_target_ids",
    "graph_callable_tool_names",
]
