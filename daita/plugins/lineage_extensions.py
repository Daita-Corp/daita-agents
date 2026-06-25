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
    ToolView,
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
            model_visible=True,
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
            model_visible=True,
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
            model_visible=True,
            side_effecting=False,
        ),
    )


def lineage_tool_views() -> tuple[ToolView, ...]:
    """Return model-visible read-only lineage tool views."""
    edge_types = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional lineage edge types to include.",
    }
    return (
        ToolView(
            name="trace_lineage",
            capability_id="lineage.trace",
            description="Trace upstream, downstream, or both lineage for an entity.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Lineage entity id to trace.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["upstream", "downstream", "both"],
                        "description": "Traversal direction.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum lineage depth to traverse.",
                    },
                    "edge_types": edge_types,
                },
                "required": ["entity_id"],
                "additionalProperties": False,
            },
        ),
        ToolView(
            name="find_lineage_paths",
            capability_id="lineage.path.find",
            description="Find read-only lineage paths between two entities.",
            parameters={
                "type": "object",
                "properties": {
                    "from_entity": {
                        "type": "string",
                        "description": "Starting lineage entity id.",
                    },
                    "to_entity": {
                        "type": "string",
                        "description": "Destination lineage entity id.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum path depth to search.",
                    },
                    "max_paths": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum paths to return.",
                    },
                    "edge_types": edge_types,
                },
                "required": ["from_entity", "to_entity"],
                "additionalProperties": False,
            },
        ),
        ToolView(
            name="analyze_impact",
            capability_id="lineage.impact.analyze",
            description="Analyze downstream impact for a proposed entity change.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Lineage entity id that may change.",
                    },
                    "change_type": {
                        "type": "string",
                        "description": "Type of change to analyze.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum downstream depth to analyze.",
                    },
                },
                "required": ["entity_id"],
                "additionalProperties": False,
            },
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
