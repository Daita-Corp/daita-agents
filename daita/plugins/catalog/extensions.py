"""
Extension declarations for CatalogPlugin.

The catalog package owns these runtime contracts because it owns schema
registration, profiling, search, relationships, comparison, and diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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
            operation_types=frozenset(
                {"schema.query", "schema.relationship_query", "data.query"}
            ),
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
            operation_types=frozenset(
                {"schema.query", "schema.relationship_query", "data.query"}
            ),
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
            operation_types=frozenset(
                {
                    "schema.query",
                    "schema.relationship_query",
                    "data.query",
                    "lineage.trace",
                }
            ),
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
            model_visible=True,
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
            id="catalog.value_grounding.plan",
            owner="catalog",
            description="Plan catalog-owned value grounding targets without connector reads.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "query.plan", "schema.query"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"catalog.value_grounding.plan"}),
            executor="catalog.plan_value_grounding",
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
    value_grounding_plan_schema = {
        "type": "object",
        "properties": {
            "store_id": {"type": "string"},
            "prompt": {"type": "string"},
            "targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "column": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                        "requires_profile_read": {"type": "boolean"},
                        "source": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "object"},
                            ]
                        },
                    },
                    "required": [
                        "table",
                        "column",
                        "reason",
                        "confidence",
                        "requires_profile_read",
                        "source",
                    ],
                    "additionalProperties": False,
                },
            },
            "skipped": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "column": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["table", "column", "reason"],
                    "additionalProperties": False,
                },
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "profile_budget": {"type": "integer"},
                    "target_count": {"type": "integer"},
                    "skipped_count": {"type": "integer"},
                },
                "required": ["profile_budget", "target_count", "skipped_count"],
                "additionalProperties": False,
            },
        },
        "required": ["store_id", "prompt", "targets", "skipped", "diagnostics"],
        "additionalProperties": False,
    }
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
            kind="catalog.value_grounding.plan",
            owner="catalog",
            json_schema=value_grounding_plan_schema,
            description="Catalog-owned value grounding target plan.",
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
    """Return the catalog-owned portion of the Phase 2 DB operation surface."""
    search_parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language or schema term to search for.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum number of matching assets to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    inspect_parameters = {
        "type": "object",
        "properties": {
            "asset_ref": {
                "type": "string",
                "description": "Asset name or reference to inspect.",
            },
            "fields": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 200,
                "description": "Exact field names to return.",
            },
            "field_glob": {
                "type": "string",
                "minLength": 1,
                "description": "One optional case-insensitive field-name glob.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Zero-based field offset for pagination.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Maximum number of fields to return.",
            },
        },
        "required": ["asset_ref"],
        "additionalProperties": False,
    }
    relationship_parameters = {
        "type": "object",
        "properties": {
            "from_assets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Starting asset names or references.",
            },
            "to_assets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Destination asset names or references.",
            },
            "max_hops": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
                "description": "Maximum relationship hops to traverse.",
            },
            "max_paths": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
                "description": "Maximum relationship paths to return.",
            },
        },
        "required": ["from_assets", "to_assets"],
        "additionalProperties": False,
    }
    column_value_parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Value or phrase to ground against fresh profiles.",
            },
            "tables": {
                "type": "array",
                "items": {"type": "string"},
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    runtime_bound = [
        "store_id",
        "allowed_tables",
        "blocked_tables",
        "blocked_columns",
    ]
    return (
        ToolView(
            name="search_schema",
            capability_id="catalog.schema.search",
            description="Search catalog schema assets and fields.",
            parameters=search_parameters,
            metadata={
                "db_slim_phase": 2,
                "runtime_bound_arguments": runtime_bound,
            },
        ),
        ToolView(
            name="inspect_asset",
            capability_id="catalog.asset.inspect",
            description="Inspect one catalog asset.",
            parameters=inspect_parameters,
            metadata={
                "db_slim_phase": 2,
                "runtime_bound_arguments": runtime_bound,
            },
        ),
        ToolView(
            name="find_relationships",
            capability_id="catalog.relationship_paths.find",
            description="Find relationship paths between catalog assets.",
            parameters=relationship_parameters,
            metadata={
                "db_slim_phase": 2,
                "runtime_bound_arguments": runtime_bound,
            },
        ),
        ToolView(
            name="search_column_values",
            capability_id="catalog.column_values.search",
            description="Search fresh, bounded catalog column-value profiles.",
            parameters=column_value_parameters,
            metadata={
                "db_slim_phase": 2,
                "runtime_bound_arguments": [
                    *runtime_bound,
                    "max_profile_age_seconds",
                ],
            },
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
        max_chars = min(12_000, max(1_000, max(0, int(token_budget)) * 4))
        if getattr(self.plugin, "_runtime_source_binding", None) is None:
            store_count = len(getattr(self.plugin, "_discovered_stores", {}))
            schema_count = len(getattr(self.plugin, "_schemas", {}))
            last_scan = getattr(self.plugin, "_last_scan", None) or "never"
            content = (
                "Catalog has no registered stores or profiled schemas."
                if store_count == 0 and schema_count == 0
                else (
                    f"Catalog has {store_count} known stores and {schema_count} "
                    f"profiled schemas. Last scan: {last_scan}."
                )
            )
            return ContextBlock(
                id=self.id,
                owner=self.owner,
                audience=audience,
                content=content[:max_chars],
                priority=10,
                metadata={
                    "store_count": store_count,
                    "schema_count": schema_count,
                    "context_chars": min(len(content), max_chars),
                    "context_limit": max_chars,
                    "truncated": len(content) > max_chars,
                },
            )
        policy_summary = context.get("policy_summary")
        policy_summary = policy_summary if isinstance(policy_summary, Mapping) else {}
        safety_frame = context.get("safety_frame")
        safety_frame = safety_frame if isinstance(safety_frame, Mapping) else {}
        allowed_tables = tuple(policy_summary.get("allowed_tables") or ())
        source_scope = tuple(context.get("source_scope") or ())
        requested_tables = tuple(
            str(item)
            for item in source_scope
            if str(item) != str(context.get("source_owner") or "")
        )
        if requested_tables:
            if allowed_tables:
                wanted = {item.lower() for item in requested_tables}
                allowed_tables = tuple(
                    item
                    for item in allowed_tables
                    if str(item).lower() in wanted
                    or str(item).split(".")[-1].lower() in wanted
                )
            else:
                allowed_tables = requested_tables
        projection = await self.plugin.runtime_relevant_projection(
            str(context.get("prompt") or ""),
            max_chars=max(256, max_chars - 512),
            policy_scope={
                "allowed_tables": allowed_tables,
                "allowed_tables_restricted": bool(
                    policy_summary.get("allowed_tables_restricted", False)
                    or requested_tables
                ),
                "blocked_tables": tuple(
                    dict.fromkeys(
                        (
                            *tuple(policy_summary.get("blocked_tables") or ()),
                            *tuple(safety_frame.get("blocked_tables") or ()),
                        )
                    )
                ),
                "blocked_columns": tuple(
                    dict.fromkeys(
                        (
                            *tuple(policy_summary.get("blocked_columns") or ()),
                            *tuple(safety_frame.get("blocked_columns") or ()),
                        )
                    )
                ),
            },
        )
        content = json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        metadata = {
            "context_chars": len(content),
            "context_limit": max_chars,
            "truncated": bool(projection.get("truncated", False)),
            "freshness": (
                projection.get("freshness", {}).get("status")
                if isinstance(projection.get("freshness"), Mapping)
                else projection.get("freshness")
            ),
            "data_boundary": "untrusted_catalog_data",
        }
        serialized_chars = _context_block_serialized_chars(content, metadata)
        if serialized_chars > max_chars:
            content = json.dumps(
                {
                    "status": "ready",
                    "freshness": projection.get("freshness"),
                    "assets": [],
                    "truncated": True,
                    "truncation": {
                        "character_limit": max_chars,
                        "character_limit_reached": True,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            metadata.update(
                {
                    "context_chars": len(content),
                    "truncated": True,
                }
            )
            serialized_chars = _context_block_serialized_chars(content, metadata)
        metadata["serialized_chars"] = serialized_chars
        final_size = _context_block_serialized_chars(content, metadata)
        metadata["serialized_chars"] = final_size
        if _context_block_serialized_chars(content, metadata) > max_chars:
            raise RuntimeError("catalog context serialization exceeded its hard limit")
        return ContextBlock(
            id=self.id,
            owner=self.owner,
            audience=audience,
            content=content,
            priority=10,
            metadata=metadata,
        )


def _context_block_serialized_chars(
    content: str,
    metadata: Mapping[str, Any],
) -> int:
    return len(
        json.dumps(
            {"content": content, "metadata": metadata},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
