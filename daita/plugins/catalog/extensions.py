"""
Extension declarations for CatalogPlugin.

The catalog package owns these runtime contracts because it owns schema
registration, profiling, search, relationships, comparison, and diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from daita.runtime import (
    AccessMode,
    Capability,
    ContextAudience,
    ContextBlock,
    Evidence,
    EvidenceSchema,
    Operation,
    RiskLevel,
    Task,
    ToolView,
    Worker,
)

from ..manifest import PluginKind, PluginManifest

CATALOG_MANIFEST = PluginManifest(
    id="catalog",
    display_name="Catalog",
    version="2.0.0",
    kind=PluginKind.DOMAIN_SERVICE,
    domains=frozenset({"db", "cloud", "file"}),
    provides=frozenset({"schema", "relationships", "discovery"}),
)


def catalog_capabilities() -> tuple[Capability, ...]:
    """Return catalog runtime capabilities."""
    common_schema = {"type": "object"}
    return (
        Capability(
            id="catalog.source.register",
            owner="catalog",
            description="Register a normalized or raw schema in the catalog.",
            domains=frozenset({"db", "cloud", "file"}),
            operation_types=frozenset({"schema.register", "source.profile"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"catalog.source_registered"}),
            executor="catalog.register_source",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.source.profile",
            owner="catalog",
            description="Profile discovered catalog sources.",
            domains=frozenset({"db", "cloud", "file"}),
            operation_types=frozenset({"source.profile", "schema.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"catalog.profile"}),
            executor="catalog.profile_source",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.schema.search",
            owner="catalog",
            description="Search catalog schema assets and fields.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.query", "data.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.search_result"}),
            executor="catalog.search_schema",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.asset.inspect",
            owner="catalog",
            description="Inspect one bounded catalog asset.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.query", "data.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.asset_profile"}),
            executor="catalog.inspect_asset",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.relationship_paths.find",
            owner="catalog",
            description="Find relationship paths between catalog assets.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.query", "data.query", "lineage.trace"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.relationship_path"}),
            executor="catalog.find_relationship_paths",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.column_values.register",
            owner="catalog",
            description="Register normalized column value profiles in the catalog.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"source.profile", "schema.query", "data.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.column_value_profile"}),
            executor="catalog.register_column_values",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.column_values.search",
            owner="catalog",
            description="Search stored catalog column value profiles.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.query", "data.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.column_value_search_result"}),
            executor="catalog.search_column_values",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.column_value_hints.resolve",
            owner="catalog",
            description="Resolve prompt-scoped filter value hints from catalog profiles.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "query.plan"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.column_value_hint"}),
            executor="catalog.resolve_column_value_hints",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.infrastructure.discover",
            owner="catalog",
            description="Discover infrastructure sources known to catalog discoverers.",
            domains=frozenset({"db", "cloud", "file"}),
            operation_types=frozenset({"infrastructure.discover", "source.profile"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"catalog.infrastructure_inventory"}),
            executor="catalog.discover_infrastructure",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.schema.compare",
            owner="catalog",
            description="Compare two schemas and return structural differences.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.compare", "schema.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.comparison"}),
            executor="catalog.compare_schema",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="catalog.diagram.export",
            owner="catalog",
            description="Export a catalog schema diagram.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.query", "report.generate"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.diagram"}),
            executor="catalog.export_diagram",
            runtime_only=True,
            side_effecting=False,
        ),
    )


def catalog_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    """Return catalog evidence schemas."""
    object_schema = {"type": "object"}
    return (
        EvidenceSchema(
            kind="catalog.source_registered",
            owner="catalog",
            json_schema=object_schema,
            description="A catalog source was registered.",
        ),
        EvidenceSchema(
            kind="catalog.profile",
            owner="catalog",
            json_schema=object_schema,
            description="A catalog source profile or summary.",
        ),
        EvidenceSchema(
            kind="schema.search_result",
            owner="catalog",
            json_schema=object_schema,
            description="Schema search results.",
        ),
        EvidenceSchema(
            kind="schema.asset_profile",
            owner="catalog",
            json_schema=object_schema,
            description="Bounded profile of a schema asset.",
        ),
        EvidenceSchema(
            kind="schema.relationship_path",
            owner="catalog",
            json_schema=object_schema,
            description="Relationship paths between schema assets.",
        ),
        EvidenceSchema(
            kind="column_values.profile",
            owner="catalog",
            json_schema=object_schema,
            description="Raw connector-produced bounded column value profile.",
        ),
        EvidenceSchema(
            kind="schema.column_value_profile",
            owner="catalog",
            json_schema=object_schema,
            description="Canonical catalog column value profile.",
        ),
        EvidenceSchema(
            kind="schema.column_value_search_result",
            owner="catalog",
            json_schema=object_schema,
            description="Search results over catalog column value profiles.",
        ),
        EvidenceSchema(
            kind="schema.column_value_hint",
            owner="catalog",
            json_schema=object_schema,
            description="Prompt-scoped column value hints derived from catalog profiles.",
        ),
        EvidenceSchema(
            kind="catalog.infrastructure_inventory",
            owner="catalog",
            json_schema=object_schema,
            description="Discovered catalog infrastructure inventory.",
        ),
        EvidenceSchema(
            kind="schema.comparison",
            owner="catalog",
            json_schema=object_schema,
            description="Schema comparison result.",
        ),
        EvidenceSchema(
            kind="schema.diagram",
            owner="catalog",
            json_schema=object_schema,
            description="Exported schema diagram.",
        ),
    )


def catalog_tool_views() -> tuple[ToolView, ...]:
    """Return optional model-visible tool views over catalog capabilities."""
    parameters = {"type": "object"}
    return (
        ToolView(
            name="catalog_search_schema",
            capability_id="catalog.schema.search",
            description="Search catalog schema assets and fields.",
            parameters=parameters,
        ),
        ToolView(
            name="catalog_inspect_asset",
            capability_id="catalog.asset.inspect",
            description="Inspect one catalog asset.",
            parameters=parameters,
        ),
        ToolView(
            name="catalog_find_relationship_paths",
            capability_id="catalog.relationship_paths.find",
            description="Find relationship paths between catalog assets.",
            parameters=parameters,
        ),
    )


def catalog_workers() -> tuple[Worker, ...]:
    return ()


@dataclass(frozen=True)
class CatalogExecutor:
    """Executor that delegates one task to a CatalogPlugin method."""

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
                owner="catalog",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]


@dataclass(frozen=True)
class CatalogSummaryContextProvider:
    """Render compact catalog context for runtime audiences."""

    plugin: Any
    id: str = "catalog.summary"
    owner: str = "catalog"
    audiences: frozenset[ContextAudience] = frozenset(
        {
            ContextAudience.PRIMARY_MODEL,
            ContextAudience.OPERATION_INSPECTOR,
        }
    )

    async def render(
        self,
        context: Mapping[str, Any],
        audience: ContextAudience,
        token_budget: int,
    ) -> ContextBlock | None:
        if audience not in self.audiences:
            return None
        store_count = len(getattr(self.plugin, "_discovered_stores", {}))
        schema_count = len(getattr(self.plugin, "_schemas", {}))
        last_scan = getattr(self.plugin, "_last_scan", None) or "never"
        if store_count == 0 and schema_count == 0:
            content = "Catalog has no registered stores or profiled schemas."
        else:
            content = (
                f"Catalog has {store_count} known stores and {schema_count} "
                f"profiled schemas. Last scan: {last_scan}."
            )
        return ContextBlock(
            id=self.id,
            owner=self.owner,
            audience=audience,
            content=content[: max(token_budget, 0)] if token_budget else content,
            priority=10,
            metadata={"store_count": store_count, "schema_count": schema_count},
        )
