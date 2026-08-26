"""Focused static domain used only to exercise common runtime mechanics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityDeclarations,
    Executor,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from daita.capability_runtime import (
    CapabilityFailure,
    CapabilityRuntime,
    RunToolCatalog,
    SideEffectPlan,
    StepToolProjection,
)
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import LoopLimits, RunInput


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


def presentation_metadata(
    *,
    toolbox_id: ToolboxId = ToolboxId.SOURCES,
    load_mode: ToolLoadMode = ToolLoadMode.PINNED,
) -> ToolPresentation:
    return ToolPresentation(
        toolbox_id=toolbox_id,
        load_mode=load_mode,
        text_trust=ToolTextTrust.CODE,
        summary="Trusted test capability.",
        when_to_use="Use to exercise the declared test contract.",
        keywords=("test",),
    )


async def execute_projected(
    runtime,
    run: RunInput,
    calls: tuple[ToolCall, ...],
    *,
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
):
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    on_demand_names = tuple(
        sorted(
            {
                call.name
                for call in calls
                for entry in catalog.entries
                if entry.view.name == call.name
                and entry.load_mode is ToolLoadMode.ON_DEMAND
            }
        )
    )
    messages: tuple[CanonicalMessage, ...]
    if on_demand_names:
        load = ToolCall(
            id="test-toolbox-load",
            name="toolbox_load",
            arguments={"tool_names": on_demand_names},
        )
        load_outcome = await runtime.execute_all(
            run,
            (load,),
            projection=projection,
            messages=(),
            sensitivity=sensitivity,
        )
        messages = (
            CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(load,)),
            CanonicalMessage(
                MessageRole.TOOL,
                content=(load_outcome.ordered_results[0],),
            ),
        )
        projection = runtime.project(catalog, messages)
    else:
        messages = ()
    return await runtime.execute_all(
        run,
        calls,
        projection=projection,
        messages=messages,
        sensitivity=sensitivity,
    )


class _ContextExecutor:
    def __init__(self, executor_id: str) -> None:
        self.executor_id = executor_id

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        raise AssertionError("context-only test tools must never execute")


class ContextToolProjectionAdapter:
    """Construct context-test catalogs through the real registry/runtime path."""

    def __init__(
        self,
        definitions: tuple[ToolDefinition, ...],
        *,
        capability_ids: tuple[str, ...] | None = None,
        limits: LoopLimits = LoopLimits(),
    ) -> None:
        definitions = tuple(definitions)
        ids = capability_ids or tuple(
            f"test.{definition.name}" for definition in definitions
        )
        if len(ids) != len(definitions):
            raise ValueError("capability_ids must match definitions")
        capabilities = tuple(
            Capability(
                id=capability_id,
                description=definition.description,
                input_schema=definition.input_schema,
                output_kind="test.output",
                output_schema={"type": "object", "properties": {}},
                executor_id=f"{capability_id}.executor",
                access_mode=AccessMode.READ,
            )
            for definition, capability_id in zip(definitions, ids, strict=True)
        )
        views = tuple(
            ToolView(
                name=definition.name,
                capability_id=capability_id,
                description=definition.description,
                presentation=presentation_metadata(),
            )
            for definition, capability_id in zip(definitions, ids, strict=True)
        )
        executors = tuple(
            _ContextExecutor(capability.executor_id) for capability in capabilities
        )
        domain = StaticTestDomain(capabilities, views)
        self._runtime = CapabilityRuntime(
            static_registry(domain, executors),
            (domain,),
            limits=limits,
        )

    async def prepare_run(self, run: RunInput) -> RunToolCatalog:
        return await self._runtime.prepare_run(run)

    def project(
        self,
        catalog: RunToolCatalog,
        messages: tuple[CanonicalMessage, ...],
    ) -> StepToolProjection:
        return self._runtime.project(catalog, messages)


__all__ = [
    "ContextToolProjectionAdapter",
    "StaticTestDomain",
    "execute_projected",
    "presentation_metadata",
    "static_registry",
]
