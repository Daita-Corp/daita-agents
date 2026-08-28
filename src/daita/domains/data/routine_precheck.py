"""The one internal data-owned resource-revision observation capability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ...capabilities import (
    AccessMode,
    AutomationEligibility,
    Capability,
    Executor,
    ToolExecution,
    ToolOutput,
)
from ...llm.models import ModelSensitivity

RESOURCE_REVISION_OBSERVATION_CAPABILITY_ID = "data.resource_revision_observation"
RESOURCE_REVISION_OBSERVATION_EXECUTOR_ID = (
    "data.resource_revision_observation.executor"
)
RESOURCE_REVISION_OBSERVATION_OUTPUT_KIND = "data.resource_revision_observation"


class ResourceRevisionCatalog(Protocol):
    async def resource_revision_fact(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> tuple[str, str, ModelSensitivity] | None: ...


class ResourceRevisionObservationExecutor:
    executor_id = RESOURCE_REVISION_OBSERVATION_EXECUTOR_ID

    def __init__(
        self,
        *,
        agent_id: str,
        catalog: ResourceRevisionCatalog,
        clock: Callable[[], datetime],
    ) -> None:
        self._agent_id = agent_id
        self._catalog = catalog
        self._clock = clock

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = request.arguments["source_id"]
        resource_id = request.arguments["resource_id"]
        assert isinstance(source_id, str) and isinstance(resource_id, str)
        fact = await self._catalog.resource_revision_fact(
            self._agent_id,
            source_id,
            resource_id,
        )
        if fact is None:
            raise ValueError("routine_precheck_resource_unavailable")
        resource_revision, catalog_revision, sensitivity = fact
        return ToolOutput(
            kind=RESOURCE_REVISION_OBSERVATION_OUTPUT_KIND,
            data={
                "source_id": source_id,
                "resource_id": resource_id,
                "resource_revision": resource_revision,
                "catalog_revision": catalog_revision,
                "observed_at": self._clock().isoformat(),
            },
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": "current_catalog_resource_revision",
                "source_id": source_id,
                "resource_id": resource_id,
            },
        )


@dataclass(frozen=True, slots=True)
class ResourceRevisionObservationDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]


def resource_revision_observation_declarations(
    *,
    agent_id: str,
    catalog: ResourceRevisionCatalog,
    clock: Callable[[], datetime],
) -> ResourceRevisionObservationDeclarations:
    capability = Capability(
        id=RESOURCE_REVISION_OBSERVATION_CAPABILITY_ID,
        description=(
            "Observe the exact current catalog and resource revision for one "
            "admitted readable resource."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "minLength": 1},
                "resource_id": {"type": "string", "minLength": 1},
            },
            "required": ["source_id", "resource_id"],
            "additionalProperties": False,
        },
        output_kind=RESOURCE_REVISION_OBSERVATION_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "resource_id": {"type": "string"},
                "resource_revision": {"type": "string"},
                "catalog_revision": {"type": "string"},
                "observed_at": {"type": "string"},
            },
            "required": [
                "source_id",
                "resource_id",
                "resource_revision",
                "catalog_revision",
                "observed_at",
            ],
            "additionalProperties": False,
        },
        executor_id=RESOURCE_REVISION_OBSERVATION_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.SCHEDULED_DIRECT,
    )
    return ResourceRevisionObservationDeclarations(
        capabilities=(capability,),
        executors=(
            ResourceRevisionObservationExecutor(
                agent_id=agent_id,
                catalog=catalog,
                clock=clock,
            ),
        ),
    )


__all__ = [
    "RESOURCE_REVISION_OBSERVATION_CAPABILITY_ID",
    "RESOURCE_REVISION_OBSERVATION_EXECUTOR_ID",
    "RESOURCE_REVISION_OBSERVATION_OUTPUT_KIND",
    "ResourceRevisionObservationDeclarations",
    "resource_revision_observation_declarations",
]
