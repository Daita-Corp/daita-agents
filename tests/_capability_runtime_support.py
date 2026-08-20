"""Focused static domain used only to exercise common runtime mechanics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from daita._json import FrozenJsonObject
from daita.capabilities import (
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    Executor,
    ToolExecution,
    ToolOutput,
    ToolView,
)
from daita.capability_runtime import CapabilityFailure, SideEffectPlan
from daita.llm.models import ModelSensitivity, ToolCall
from daita.loop.models import RunInput


class StaticTestDomain:
    def __init__(
        self,
        capabilities: tuple[Capability, ...],
        tool_views: tuple[ToolView, ...],
        *,
        domain_owner_id: str = "test",
        recheck_after_approval: bool = True,
    ) -> None:
        self.domain_owner_id = domain_owner_id
        self._declarations = CapabilityDeclarations(
            domain_owner_id=domain_owner_id,
            capabilities=capabilities,
            executor_ids=tuple(item.executor_id for item in capabilities),
            tool_views=tool_views,
        )
        self._recheck_after_approval = recheck_after_approval

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(self, run: RunInput) -> tuple[str, ...]:
        return tuple(item.name for item in self._declarations.tool_views)

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
        del request_sensitivity
        return arguments

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        return SideEffectPlan(
            recheck_after_approval=self._recheck_after_approval,
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
        del request_sensitivity
        if output.sensitivity is not None:
            return output
        return replace(
            output,
            sensitivity=ModelSensitivity.INTERNAL,
            sensitivity_provenance={
                "authority": "test_static_domain",
                "capability_id": capability.id,
            },
        )

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        return None


def static_registry(
    domain: StaticTestDomain,
    executors: tuple[Executor, ...],
):
    from daita.capabilities import CapabilityRegistry

    return CapabilityRegistry(
        declarations=(domain.declarations,),
        executors=executors,
    )
