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
    ToolView,
)
from .models import CatalogSearchRequest
from .service import CatalogService

CATALOG_SEARCH_CAPABILITY_ID = "catalog.search"
CATALOG_INSPECT_CAPABILITY_ID = "catalog.inspect"
CATALOG_SEARCH_EVIDENCE_KIND = "catalog.search_result"
CATALOG_INSPECT_EVIDENCE_KIND = "catalog.resource_snapshot"


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
        query = request.arguments["query"]
        source_id = request.arguments.get("source_id")
        limit = request.arguments.get("limit", 12)
        assert isinstance(query, str)
        assert source_id is None or isinstance(source_id, str)
        assert isinstance(limit, int) and not isinstance(limit, bool)
        result = await self._service.search(
            CatalogSearchRequest(
                agent_id=self._agent_id,
                query=query,
                source_ids=() if source_id is None else (source_id,),
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
        resource_id = request.arguments["resource_id"]
        assert isinstance(resource_id, str)
        projection = await self._service.inspect_resource(
            self._agent_id,
            resource_id,
        )
        return EvidenceCandidate(
            kind=CATALOG_INSPECT_EVIDENCE_KIND,
            schema_version=1,
            payload=projection,
        )


def catalog_declarations(
    agent_id: str,
    service: CatalogService,
) -> CatalogDeclarations:
    search_executor = CatalogSearchExecutor(agent_id, service)
    inspect_executor = CatalogInspectExecutor(agent_id, service)
    search = Capability(
        id=CATALOG_SEARCH_CAPABILITY_ID,
        owner="catalog",
        description="Search attached resource metadata before inspecting or querying.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source_id": {"type": "string"},
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
        description="Inspect the current typed schema for one catalog resource.",
        input_schema={
            "type": "object",
            "properties": {"resource_id": {"type": "string"}},
            "required": ["resource_id"],
            "additionalProperties": False,
        },
        output_evidence_kind=CATALOG_INSPECT_EVIDENCE_KIND,
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {
                "facets": {"type": "array"},
                "resource": {"type": "object"},
                "trust_classification": {"type": "string"},
            },
            "required": ["facets", "resource", "trust_classification"],
            "additionalProperties": False,
        },
        executor_id=inspect_executor.executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    return CatalogDeclarations(
        capabilities=(search, inspect),
        executors=(search_executor, inspect_executor),
        tool_views=(
            ToolView(
                name="catalog_search",
                capability_id=search.id,
                description=search.description,
            ),
            ToolView(
                name="catalog_inspect",
                capability_id=inspect.id,
                description=inspect.description,
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


__all__ = [
    "CATALOG_INSPECT_CAPABILITY_ID",
    "CATALOG_INSPECT_EVIDENCE_KIND",
    "CATALOG_SEARCH_CAPABILITY_ID",
    "CATALOG_SEARCH_EVIDENCE_KIND",
    "CatalogDeclarations",
    "CatalogInspectExecutor",
    "CatalogSearchExecutor",
    "catalog_declarations",
]
