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
    ToolDiscoveryMetadata,
    ToolExecution,
    ToolExposureClass,
    ToolOutput,
    ToolView,
)
from daita.capability_runtime import (
    CapabilityFailure,
    DomainToolManifestEntry,
    RunToolCatalog,
    RunToolCatalogEntry,
    SideEffectPlan,
    StepToolProjection,
    ToolInvocationMode,
)
from daita.llm.models import (
    ModelSensitivity,
    ToolCall,
    ToolDefinition,
)
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


def discovery_metadata(
    *,
    exposure_class: ToolExposureClass = ToolExposureClass.CORE,
    priority: int = 500,
) -> ToolDiscoveryMetadata:
    return ToolDiscoveryMetadata(
        summary="Trusted test capability.",
        when_to_use="Use to exercise the declared test contract.",
        keywords=("test",),
        exposure_class=exposure_class,
        eager_priority=priority,
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
    return await runtime.execute_all(
        run,
        calls,
        projection=projection,
        sensitivity=sensitivity,
    )


def context_tool_catalog(
    run: RunInput,
    definitions: tuple[ToolDefinition, ...],
    *,
    capability_ids: tuple[str, ...] | None = None,
) -> RunToolCatalog:
    definitions = tuple(definitions)
    ids = capability_ids or tuple(f"test.{item.name}" for item in definitions)
    if len(ids) != len(definitions):
        raise ValueError("capability_ids must match definitions")
    entries = tuple(
        RunToolCatalogEntry(
            view=ToolView(
                name=definition.name,
                capability_id=capability_id,
                description=definition.description,
                discovery=discovery_metadata(),
            ),
            capability=Capability(
                id=capability_id,
                description=definition.description,
                input_schema=definition.input_schema,
                output_kind="test.output",
                output_schema={"type": "object", "properties": {}},
                executor_id=f"{capability_id}.executor",
                access_mode=AccessMode.READ,
            ),
            domain_owner_id="test",
            executor_id=f"{capability_id}.executor",
            input_schema_digest="sha256:" + "1" * 64,
            origin_revision_digest="sha256:" + "2" * 64,
            invocation_mode=ToolInvocationMode.DIRECT,
        )
        for definition, capability_id in zip(definitions, ids, strict=True)
    )
    manifest = (
        ()
        if not entries
        else (
            DomainToolManifestEntry(
                domain_owner_id="test",
                summary="Applicable trusted test capabilities.",
                direct_count=len(entries),
                deferred_count=0,
            ),
        )
    )
    return RunToolCatalog(
        run_id=run.id,
        agent_id=run.agent_id,
        execution_scope_digest="sha256:" + "3" * 64,
        registry_digest="sha256:" + "4" * 64,
        catalog_digest="sha256:" + "5" * 64,
        entries=entries,
        domain_manifest=manifest,
        provider_definitions=definitions,
        aggregate_bytes=1,
        manifest_bytes=1,
        manifest_token_limit=4_000,
    )


def context_step_projection(
    catalog: RunToolCatalog,
) -> StepToolProjection:
    return StepToolProjection(
        run_id=catalog.run_id,
        catalog_digest=catalog.catalog_digest,
        projection_digest="sha256:" + "6" * 64,
        provider_definitions=catalog.provider_definitions,
        catalog_entries=catalog.entries,
        direct_resolution_entries=catalog.entries,
        described_deferred_references=(),
        described_schema_bytes=0,
    )
