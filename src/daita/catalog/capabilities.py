"""Catalog capability declarations and runtime executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .._json import FrozenJsonObject
from ..capabilities import (
    AccessMode,
    Capability,
    Executor,
    ToolApplicability,
    ToolExecution,
    ToolOutput,
    ToolView,
)
from .models import (
    CATALOG_MAX_LIMIT,
    CATALOG_RESOURCE_ID_MAX_CHARACTERS,
    CATALOG_SCHEMA_DEFAULT_JOIN_DEPTH,
    CATALOG_SCHEMA_MAX_RESOURCE_IDS,
    CATALOG_SOURCE_ID_MAX_CHARACTERS,
    CATALOG_TOOL_DEFAULT_LIMIT,
    CATALOG_TOOL_QUERY_MAX_CHARACTERS,
    CATALOG_TRAVERSAL_DEFAULT_DEPTH,
    CATALOG_TRAVERSAL_DEFAULT_EDGES,
    CATALOG_TRAVERSAL_DEFAULT_NODES,
    CATALOG_TRAVERSAL_DEFAULT_PATHS,
    CATALOG_TRAVERSAL_MAX_DEPTH,
    CATALOG_TRAVERSAL_MAX_EDGES,
    CATALOG_TRAVERSAL_MAX_NODES,
    CATALOG_TRAVERSAL_MAX_PATHS,
    CATALOG_TRAVERSAL_MAX_RESOURCE_IDS,
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogTraversalRequest,
    RelationshipKind,
    ResourceKind,
)

CATALOG_SEARCH_CAPABILITY_ID = "catalog.search"
CATALOG_SCHEMA_CAPABILITY_ID = "catalog.schema"
CATALOG_INSPECT_CAPABILITY_ID = "catalog.inspect"
CATALOG_TRAVERSE_CAPABILITY_ID = "catalog.traverse"
CATALOG_SEARCH_EVIDENCE_KIND = "catalog.search_result"
CATALOG_SCHEMA_EVIDENCE_KIND = "catalog.schema_slice"
CATALOG_INSPECT_EVIDENCE_KIND = "catalog.resource_snapshot"
CATALOG_TRAVERSE_EVIDENCE_KIND = "catalog.traversal_result"


@dataclass(frozen=True, slots=True)
class CatalogDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class CatalogProjection(Protocol):
    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult: ...

    async def inspect_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> FrozenJsonObject: ...

    async def schema_slice(self, request: CatalogSchemaRequest) -> FrozenJsonObject: ...

    async def traverse(
        self,
        request: CatalogTraversalRequest,
    ) -> FrozenJsonObject: ...


class CatalogSearchExecutor:
    executor_id = "catalog.search.executor"

    def __init__(self, agent_id: str, service: CatalogProjection) -> None:
        self._agent_id = agent_id
        self._service = service

    async def execute(self, request: ToolExecution) -> ToolOutput:
        query = _string_argument(request, "query")
        source_id = request.arguments.get("source_id")
        if source_id is not None and not isinstance(source_id, str):
            raise TypeError("source_id must be a string")
        resource_kinds = tuple(
            ResourceKind(value)
            for value in _string_tuple_argument(request, "resource_kinds", ())
        )
        limit = _integer_argument(request, "limit", CATALOG_TOOL_DEFAULT_LIMIT)
        result = await self._service.search(
            CatalogSearchRequest(
                agent_id=self._agent_id,
                query=query,
                source_ids=() if source_id is None else (source_id,),
                resource_kinds=resource_kinds,
                limit=limit,
            )
        )
        return ToolOutput(
            kind=CATALOG_SEARCH_EVIDENCE_KIND,
            data={
                "hits": [
                    {
                        "kind": hit.kind.value,
                        "matched_fields": hit.matched_fields,
                        "match_reasons": hit.match_reasons,
                        "name": hit.name,
                        "resource_id": hit.resource_id,
                        "revision": hit.revision,
                        "sensitivity": hit.sensitivity.value,
                        "source_id": hit.source_id,
                    }
                    for hit in result.hits
                ],
                "query": query,
                "total_matches": result.total_matches,
                "truncated": result.truncated,
                "trust_classification": "untrusted_external_data",
            },
        )


class CatalogInspectExecutor:
    executor_id = "catalog.inspect.executor"

    def __init__(self, agent_id: str, service: CatalogProjection) -> None:
        self._agent_id = agent_id
        self._service = service

    async def execute(self, request: ToolExecution) -> ToolOutput:
        resource_id = _string_argument(request, "resource_id")
        projection = await self._service.inspect_resource(
            self._agent_id,
            resource_id,
        )
        return ToolOutput(
            kind=CATALOG_INSPECT_EVIDENCE_KIND,
            data=projection,
        )


class CatalogSchemaExecutor:
    executor_id = "catalog.schema.executor"

    def __init__(self, agent_id: str, service: CatalogProjection) -> None:
        self._agent_id = agent_id
        self._service = service

    async def execute(self, request: ToolExecution) -> ToolOutput:
        query = request.arguments.get("query")
        if query is not None and not isinstance(query, str):
            raise TypeError("query must be a string")
        source_id = request.arguments.get("source_id")
        if source_id is not None and not isinstance(source_id, str):
            raise TypeError("source_id must be a string")
        include_relationships = request.arguments.get(
            "include_relationships",
            True,
        )
        if not isinstance(include_relationships, bool):
            raise TypeError("include_relationships must be a boolean")
        projection = await self._service.schema_slice(
            CatalogSchemaRequest(
                agent_id=self._agent_id,
                query=query,
                resource_ids=_string_tuple_argument(
                    request,
                    "resource_ids",
                    (),
                ),
                source_id=source_id,
                limit=_integer_argument(request, "limit", CATALOG_TOOL_DEFAULT_LIMIT),
                include_relationships=include_relationships,
                max_join_depth=_integer_argument(
                    request,
                    "max_join_depth",
                    CATALOG_SCHEMA_DEFAULT_JOIN_DEPTH,
                ),
            )
        )
        return ToolOutput(
            kind=CATALOG_SCHEMA_EVIDENCE_KIND,
            data=projection,
        )


class CatalogTraverseExecutor:
    executor_id = "catalog.traverse.executor"

    def __init__(self, agent_id: str, service: CatalogProjection) -> None:
        self._agent_id = agent_id
        self._service = service

    async def execute(self, request: ToolExecution) -> ToolOutput:
        from_resource_ids = _string_tuple_argument(request, "from_resource_ids")
        to_resource_ids = _string_tuple_argument(request, "to_resource_ids")
        relationship_kinds = tuple(
            RelationshipKind(value)
            for value in _string_tuple_argument(request, "relationship_kinds", ())
        )
        traversal_request = CatalogTraversalRequest(
            agent_id=self._agent_id,
            from_resource_ids=from_resource_ids,
            to_resource_ids=to_resource_ids,
            relationship_kinds=relationship_kinds,
            max_depth=_integer_argument(
                request,
                "max_depth",
                CATALOG_TRAVERSAL_DEFAULT_DEPTH,
            ),
            max_paths=_integer_argument(
                request,
                "max_paths",
                CATALOG_TRAVERSAL_DEFAULT_PATHS,
            ),
            max_nodes=_integer_argument(
                request,
                "max_nodes",
                CATALOG_TRAVERSAL_DEFAULT_NODES,
            ),
            max_edges=_integer_argument(
                request,
                "max_edges",
                CATALOG_TRAVERSAL_DEFAULT_EDGES,
            ),
        )
        projection = await self._service.traverse(traversal_request)
        return ToolOutput(
            kind=CATALOG_TRAVERSE_EVIDENCE_KIND,
            data=projection,
        )


def catalog_declarations(
    agent_id: str,
    service: CatalogProjection,
) -> CatalogDeclarations:
    search_executor = CatalogSearchExecutor(agent_id, service)
    schema_executor = CatalogSchemaExecutor(agent_id, service)
    inspect_executor = CatalogInspectExecutor(agent_id, service)
    traverse_executor = CatalogTraverseExecutor(agent_id, service)
    search = Capability(
        id=CATALOG_SEARCH_CAPABILITY_ID,
        description="Find catalog IDs by name or field; use catalog_schema next.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_TOOL_QUERY_MAX_CHARACTERS,
                },
                "source_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_SOURCE_ID_MAX_CHARACTERS,
                },
                "resource_kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [kind.value for kind in ResourceKind],
                    },
                    "maxItems": len(ResourceKind),
                    "uniqueItems": True,
                    "default": [],
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_MAX_LIMIT,
                    "default": CATALOG_TOOL_DEFAULT_LIMIT,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_kind=CATALOG_SEARCH_EVIDENCE_KIND,
        output_schema=_search_output_schema(),
        executor_id=search_executor.executor_id,
        access_mode=AccessMode.READ,
        side_effecting=False,
    )
    schema = Capability(
        id=CATALOG_SCHEMA_CAPABILITY_ID,
        description=(
            "SQL schema, bridges, paths, and exact join fields. Provide a non-empty "
            "query or resource_ids. Do not use with catalog_traverse."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_TOOL_QUERY_MAX_CHARACTERS,
                },
                "resource_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": CATALOG_RESOURCE_ID_MAX_CHARACTERS,
                    },
                    "maxItems": CATALOG_SCHEMA_MAX_RESOURCE_IDS,
                    "uniqueItems": True,
                    "default": [],
                },
                "source_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_SOURCE_ID_MAX_CHARACTERS,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_MAX_LIMIT,
                    "default": CATALOG_TOOL_DEFAULT_LIMIT,
                },
                "include_relationships": {
                    "type": "boolean",
                    "default": True,
                },
                "max_join_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_TRAVERSAL_MAX_DEPTH,
                    "default": CATALOG_SCHEMA_DEFAULT_JOIN_DEPTH,
                },
            },
            "additionalProperties": False,
        },
        output_kind=CATALOG_SCHEMA_EVIDENCE_KIND,
        output_schema=_schema_output_schema(),
        executor_id=schema_executor.executor_id,
        access_mode=AccessMode.READ,
        side_effecting=False,
    )
    inspect = Capability(
        id=CATALOG_INSPECT_CAPABILITY_ID,
        description="Get full catalog facets, freshness, containment, or diagnostics.",
        input_schema={
            "type": "object",
            "properties": {
                "resource_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_RESOURCE_ID_MAX_CHARACTERS,
                }
            },
            "required": ["resource_id"],
            "additionalProperties": False,
        },
        output_kind=CATALOG_INSPECT_EVIDENCE_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "facets": {"type": "array"},
                "incident_relationships": {"type": "array"},
                "incident_relationships_truncated": {"type": "boolean"},
                "neighbors": {"type": "array"},
                "resource": {"type": "object"},
                "selection_facts": {"type": "object"},
                "trust_classification": {"type": "string"},
            },
            "required": [
                "facets",
                "incident_relationships",
                "incident_relationships_truncated",
                "neighbors",
                "resource",
                "selection_facts",
                "trust_classification",
            ],
            "additionalProperties": False,
        },
        executor_id=inspect_executor.executor_id,
        access_mode=AccessMode.READ,
        side_effecting=False,
    )
    traverse = Capability(
        id=CATALOG_TRAVERSE_CAPABILITY_ID,
        description=(
            "Use on a later step only when a prior catalog_schema result leaves a "
            "multi-hop path unresolved. Do not call alongside catalog_schema."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "from_resource_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": CATALOG_RESOURCE_ID_MAX_CHARACTERS,
                    },
                    "minItems": 1,
                    "maxItems": CATALOG_TRAVERSAL_MAX_RESOURCE_IDS,
                    "uniqueItems": True,
                },
                "to_resource_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": CATALOG_RESOURCE_ID_MAX_CHARACTERS,
                    },
                    "minItems": 1,
                    "maxItems": CATALOG_TRAVERSAL_MAX_RESOURCE_IDS,
                    "uniqueItems": True,
                },
                "relationship_kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [kind.value for kind in RelationshipKind],
                    },
                    "maxItems": len(RelationshipKind),
                    "uniqueItems": True,
                    "default": [],
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_TRAVERSAL_MAX_DEPTH,
                    "default": CATALOG_TRAVERSAL_DEFAULT_DEPTH,
                },
                "max_paths": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_TRAVERSAL_MAX_PATHS,
                    "default": CATALOG_TRAVERSAL_DEFAULT_PATHS,
                },
                "max_nodes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_TRAVERSAL_MAX_NODES,
                    "default": CATALOG_TRAVERSAL_DEFAULT_NODES,
                },
                "max_edges": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_TRAVERSAL_MAX_EDGES,
                    "default": CATALOG_TRAVERSAL_DEFAULT_EDGES,
                },
            },
            "required": ["from_resource_ids", "to_resource_ids"],
            "additionalProperties": False,
        },
        output_kind=CATALOG_TRAVERSE_EVIDENCE_KIND,
        output_schema=_traverse_output_schema(),
        executor_id=traverse_executor.executor_id,
        access_mode=AccessMode.READ,
        side_effecting=False,
    )
    return CatalogDeclarations(
        capabilities=(search, schema, inspect, traverse),
        executors=(
            search_executor,
            schema_executor,
            inspect_executor,
            traverse_executor,
        ),
        tool_views=(
            ToolView(
                name="catalog_search",
                capability_id=search.id,
                description=search.description,
                applicability=ToolApplicability(minimum_active_sources=1),
            ),
            ToolView(
                name="catalog_schema",
                capability_id=schema.id,
                description=schema.description,
                applicability=ToolApplicability(minimum_active_sources=1),
            ),
            ToolView(
                name="catalog_inspect",
                capability_id=inspect.id,
                description=inspect.description,
                applicability=ToolApplicability(minimum_active_sources=1),
            ),
            ToolView(
                name="catalog_traverse",
                capability_id=traverse.id,
                description=traverse.description,
                applicability=ToolApplicability(minimum_active_sources=1),
            ),
        ),
    )


def _search_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "hits": {"type": "array"},
            "query": {"type": "string"},
            "total_matches": {"type": "integer"},
            "truncated": {"type": "boolean"},
            "trust_classification": {"type": "string"},
        },
        "required": [
            "hits",
            "query",
            "total_matches",
            "truncated",
            "trust_classification",
        ],
        "additionalProperties": False,
    }


def _schema_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "bounds": {"type": "object"},
            "include_relationships": {"type": "boolean"},
            "paths": {"type": "array"},
            "relationships": {"type": "array"},
            "resources": {"type": "array"},
            "selection": {"type": "object"},
            "sources": {"type": "array"},
            "total_matches": {"type": "integer"},
            "truncation": {"type": "object"},
            "trust_classification": {"type": "string"},
        },
        "required": [
            "bounds",
            "include_relationships",
            "paths",
            "relationships",
            "resources",
            "selection",
            "sources",
            "total_matches",
            "truncation",
            "trust_classification",
        ],
        "additionalProperties": False,
    }


def _traverse_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "paths": {"type": "array"},
            "reachable": {"type": "boolean"},
            "request": {"type": "object"},
            "truncated": {"type": "boolean"},
            "truncation_reason": {
                "type": ["string", "null"],
                "enum": ["depth", "nodes", "edges", "paths", None],
            },
            "trust_classification": {"type": "string"},
            "visited_edges": {"type": "integer"},
            "visited_nodes": {"type": "integer"},
        },
        "required": [
            "paths",
            "reachable",
            "request",
            "truncated",
            "truncation_reason",
            "trust_classification",
            "visited_edges",
            "visited_nodes",
        ],
        "additionalProperties": False,
    }


def _integer_argument(
    request: ToolExecution,
    name: str,
    default: int,
) -> int:
    value = request.arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _string_argument(request: ToolExecution, name: str) -> str:
    value = request.arguments[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _string_tuple_argument(
    request: ToolExecution,
    name: str,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if default is None:
        value = request.arguments[name]
    else:
        value = request.arguments.get(name, default)
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be an array of strings")
    return value


__all__ = [
    "CATALOG_INSPECT_CAPABILITY_ID",
    "CATALOG_INSPECT_EVIDENCE_KIND",
    "CATALOG_SCHEMA_CAPABILITY_ID",
    "CATALOG_SCHEMA_EVIDENCE_KIND",
    "CATALOG_SEARCH_CAPABILITY_ID",
    "CATALOG_SEARCH_EVIDENCE_KIND",
    "CATALOG_TRAVERSE_CAPABILITY_ID",
    "CATALOG_TRAVERSE_EVIDENCE_KIND",
    "CatalogDeclarations",
    "CatalogInspectExecutor",
    "CatalogSchemaExecutor",
    "CatalogSearchExecutor",
    "CatalogTraverseExecutor",
    "catalog_declarations",
]
