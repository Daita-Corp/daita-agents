"""
Extension declarations for LineagePlugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    RiskLevel,
    Task,
)

from .manifest import PluginKind, PluginManifest

LINEAGE_MANIFEST = PluginManifest(
    id="lineage",
    display_name="Lineage",
    version="2.0.0",
    kind=PluginKind.DOMAIN_SERVICE,
    domains=frozenset({"db", "cloud", "file"}),
    provides=frozenset({"lineage", "impact", "relationships"}),
)


def lineage_capabilities() -> tuple[Capability, ...]:
    common_schema = {"type": "object"}
    return (
        Capability(
            id="lineage.trace",
            owner="lineage",
            description="Trace upstream and downstream lineage for an entity.",
            domains=frozenset({"db", "cloud", "file"}),
            operation_types=frozenset({"lineage.trace", "data.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"lineage.trace"}),
            executor="lineage.trace",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="lineage.impact.analyze",
            owner="lineage",
            description="Analyze downstream impact for an entity change.",
            domains=frozenset({"db", "cloud", "file"}),
            operation_types=frozenset({"lineage.trace", "impact.analyze"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"lineage.impact"}),
            executor="lineage.impact.analyze",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="lineage.flow.register",
            owner="lineage",
            description="Register a lineage flow between two entities.",
            domains=frozenset({"db", "cloud", "file"}),
            operation_types=frozenset({"lineage.register"}),
            access=AccessMode.WRITE,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"lineage.flow_registered"}),
            executor="lineage.flow.register",
            runtime_only=True,
            side_effecting=True,
        ),
        Capability(
            id="lineage.path.find",
            owner="lineage",
            description="Find lineage paths between two entities.",
            domains=frozenset({"db", "cloud", "file"}),
            operation_types=frozenset({"lineage.trace", "path.find"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"lineage.path"}),
            executor="lineage.path.find",
            runtime_only=True,
            side_effecting=False,
        ),
    )


def lineage_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    object_schema = {"type": "object"}
    return (
        EvidenceSchema(
            kind="lineage.trace",
            owner="lineage",
            json_schema=object_schema,
            description="Lineage trace result.",
        ),
        EvidenceSchema(
            kind="lineage.impact",
            owner="lineage",
            json_schema=object_schema,
            description="Lineage impact analysis result.",
        ),
        EvidenceSchema(
            kind="lineage.flow_registered",
            owner="lineage",
            json_schema=object_schema,
            description="Registered lineage flow result.",
        ),
        EvidenceSchema(
            kind="lineage.path",
            owner="lineage",
            json_schema=object_schema,
            description="Lineage path search result.",
        ),
    )


@dataclass(frozen=True)
class LineageExecutor:
    """Executor that delegates one task to a LineagePlugin method."""

    id: str
    capability_ids: frozenset[str]
    evidence_kind: str
    handler: Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        payload = await self.handler(task.input)
        return [
            Evidence(
                kind=self.evidence_kind,
                owner="lineage",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]
