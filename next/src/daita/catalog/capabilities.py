"""Catalog capability declarations and runtime executors."""

from __future__ import annotations

from dataclasses import dataclass

from ..capabilities import (
    AccessMode,
    Capability,
    EvidenceCandidate,
    ExecutionRequest,
    Executor,
    RiskLevel,
    ToolApplicability,
    ToolView,
)
from .models import (
    CatalogSearchRequest,
    CatalogTraversalRequest,
    RelationshipKind,
    ResourceKind,
)
from .service import CatalogService

CATALOG_SEARCH_CAPABILITY_ID = "catalog.search"
CATALOG_INSPECT_CAPABILITY_ID = "catalog.inspect"
CATALOG_TRAVERSE_CAPABILITY_ID = "catalog.traverse"
CATALOG_SEARCH_EVIDENCE_KIND = "catalog.search_result"
CATALOG_INSPECT_EVIDENCE_KIND = "catalog.resource_snapshot"
CATALOG_TRAVERSE_EVIDENCE_KIND = "catalog.traversal_result"


@dataclass(frozen=True, slots=True)
class CatalogDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class CatalogSearchExecutor:
    executor_id = "catalog.search.executor"

    def __init__(self, agent_id: str, service: CatalogService) -> None:
        self._agent_id = agent_id
        self._service = service

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        query = _string_argument(request, "query")
        source_id = request.arguments.get("source_id")
        if source_id is not None and not isinstance(source_id, str):
            raise TypeError("source_id must be a string")
        resource_kinds = tuple(
            ResourceKind(value)
            for value in _string_tuple_argument(request, "resource_kinds", ())
        )
        limit = _integer_argument(request, "limit", 12)
        result = await self._service.search(
            CatalogSearchRequest(
                agent_id=self._agent_id,
                query=query,
                source_ids=() if source_id is None else (source_id,),
                resource_kinds=resource_kinds,
                limit=limit,
            )
        )
        return EvidenceCandidate(
            kind=CATALOG_SEARCH_EVIDENCE_KIND,
            schema_version=1,
            payload={
                "hits": [
                    {
                        "kind": hit.kind.value,
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

    def __init__(self, agent_id: str, service: CatalogService) -> None:
        self._agent_id = agent_id
        self._service = service

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        resource_id = _string_argument(request, "resource_id")
        projection = await self._service.inspect_resource(
            self._agent_id,
            resource_id,
        )
        return EvidenceCandidate(
            kind=CATALOG_INSPECT_EVIDENCE_KIND,
            schema_version=2,
            payload=projection,
        )


class CatalogTraverseExecutor:
    executor_id = "catalog.traverse.executor"

    def __init__(self, agent_id: str, service: CatalogService) -> None:
        self._agent_id = agent_id
        self._service = service

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
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
            max_depth=_integer_argument(request, "max_depth", 4),
            max_paths=_integer_argument(request, "max_paths", 5),
            max_nodes=_integer_argument(request, "max_nodes", 100),
            max_edges=_integer_argument(request, "max_edges", 200),
        )
        projection = await self._service.traverse(traversal_request)
        return EvidenceCandidate(
            kind=CATALOG_TRAVERSE_EVIDENCE_KIND,
            schema_version=2,
            payload=projection,
        )


def catalog_declarations(
    agent_id: str,
    service: CatalogService,
) -> CatalogDeclarations:
    search_executor = CatalogSearchExecutor(agent_id, service)
    inspect_executor = CatalogInspectExecutor(agent_id, service)
    traverse_executor = CatalogTraverseExecutor(agent_id, service)
    search = Capability(
        id=CATALOG_SEARCH_CAPABILITY_ID,
        owner="catalog",
        description=(
            "Search attached resource metadata before inspecting or querying; "
            "inspect freshness-sensitive candidates before selecting one."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source_id": {"type": "string"},
                "resource_kinds": {"type": "array"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_evidence_kind=CATALOG_SEARCH_EVIDENCE_KIND,
        output_schema_version=1,
        output_schema=_search_output_schema(),
        executor_id=search_executor.executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    inspect = Capability(
        id=CATALOG_INSPECT_CAPABILITY_ID,
        owner="catalog",
        description=(
            "Inspect current typed facets, freshness, neighbors, and incident "
            "relationships for one catalog resource."
        ),
        input_schema={
            "type": "object",
            "properties": {"resource_id": {"type": "string"}},
            "required": ["resource_id"],
            "additionalProperties": False,
        },
        output_evidence_kind=CATALOG_INSPECT_EVIDENCE_KIND,
        output_schema_version=2,
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
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    traverse = Capability(
        id=CATALOG_TRAVERSE_CAPABILITY_ID,
        owner="catalog",
        description=(
            "Find bounded current relationship paths between catalog resources "
            "with endpoint revisions and field-pair provenance."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "from_resource_ids": {"type": "array"},
                "to_resource_ids": {"type": "array"},
                "relationship_kinds": {"type": "array"},
                "max_depth": {"type": "integer"},
                "max_paths": {"type": "integer"},
                "max_nodes": {"type": "integer"},
                "max_edges": {"type": "integer"},
            },
            "required": ["from_resource_ids", "to_resource_ids"],
            "additionalProperties": False,
        },
        output_evidence_kind=CATALOG_TRAVERSE_EVIDENCE_KIND,
        output_schema_version=2,
        output_schema=_traverse_output_schema(),
        executor_id=traverse_executor.executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    return CatalogDeclarations(
        capabilities=(search, inspect, traverse),
        executors=(search_executor, inspect_executor, traverse_executor),
        tool_views=(
            ToolView(
                name="catalog_search",
                capability_id=search.id,
                description=search.description,
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


def _traverse_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "paths": {"type": "array"},
            "reachable": {"type": "boolean"},
            "request": {"type": "object"},
            "truncated": {"type": "boolean"},
            "trust_classification": {"type": "string"},
            "visited_edges": {"type": "integer"},
            "visited_nodes": {"type": "integer"},
        },
        "required": [
            "paths",
            "reachable",
            "request",
            "truncated",
            "trust_classification",
            "visited_edges",
            "visited_nodes",
        ],
        "additionalProperties": False,
    }


def _integer_argument(
    request: ExecutionRequest,
    name: str,
    default: int,
) -> int:
    value = request.arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _string_argument(request: ExecutionRequest, name: str) -> str:
    value = request.arguments[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _string_tuple_argument(
    request: ExecutionRequest,
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
    "CATALOG_SEARCH_CAPABILITY_ID",
    "CATALOG_SEARCH_EVIDENCE_KIND",
    "CATALOG_TRAVERSE_CAPABILITY_ID",
    "CATALOG_TRAVERSE_EVIDENCE_KIND",
    "CatalogDeclarations",
    "CatalogInspectExecutor",
    "CatalogSearchExecutor",
    "CatalogTraverseExecutor",
    "catalog_declarations",
]
