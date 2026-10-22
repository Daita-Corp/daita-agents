"""Declare and execute catalog search and description tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .._json import FrozenJsonObject
from ..capabilities import (
    AccessMode,
    AutomationEligibility,
    Capability,
    Executor,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
    resource_id_schema,
    source_id_schema,
)
from .models import (
    CATALOG_DESCRIBE_CURSOR_MAX_CHARACTERS,
    CATALOG_DESCRIBE_IDENTITY_MAX_LIMIT,
    CATALOG_MATCH_CANDIDATE_BINDING_LIMIT,
    CATALOG_MAX_LIMIT,
    CATALOG_RESOURCE_ID_MAX_CHARACTERS,
    CATALOG_SCHEMA_MAX_RESOURCE_IDS,
    CATALOG_SOURCE_ID_MAX_CHARACTERS,
    CATALOG_TOOL_DEFAULT_LIMIT,
    CATALOG_TOOL_QUERY_MAX_CHARACTERS,
    CatalogDescribeDetail,
    CatalogDescribeRequest,
    CatalogSearchRequest,
    CatalogSearchResult,
    ResourceKind,
)

CATALOG_SEARCH_CAPABILITY_ID = "catalog.search"
CATALOG_DESCRIBE_CAPABILITY_ID = "catalog.describe"
CATALOG_SEARCH_EVIDENCE_KIND = "catalog.search_result"
CATALOG_DESCRIBE_EVIDENCE_KIND = "catalog.description"


@dataclass(frozen=True, slots=True)
class CatalogDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class CatalogProjection(Protocol):
    async def search(
        self,
        request: CatalogSearchRequest,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> CatalogSearchResult: ...

    async def describe(
        self,
        request: CatalogDescribeRequest,
        *,
        readable_resource_ids: frozenset[str] | None = None,
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
        cursor = request.arguments.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise TypeError("cursor must be a string")
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
                cursor=cursor,
                run_id=request.run_id,
            ),
            **(
                {}
                if request.source_scope is None
                else {"readable_resource_ids": request.source_scope.resource_ids}
            ),
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
                        "match_status": (
                            "unmatched_fallback"
                            if "unmatched_fallback" in hit.match_reasons
                            else "matched"
                        ),
                    }
                    for hit in result.hits
                ],
                "query": query,
                "match_outcome": result.match_outcome.to_payload(),
                "total_matches": result.total_matches,
                "total_candidates": result.total_candidates,
                "next_cursor": result.next_cursor,
                "returned_count": result.returned_count,
                "truncated": result.truncated,
                "trust_classification": "untrusted_external_data",
            },
        )


class CatalogDescribeExecutor:
    executor_id = "catalog.describe.executor"

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
        cursor = request.arguments.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise TypeError("cursor must be a string")
        limit = request.arguments.get("limit")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool)
        ):
            raise TypeError("limit must be an integer")
        projection = await self._service.describe(
            CatalogDescribeRequest(
                agent_id=self._agent_id,
                detail=CatalogDescribeDetail(
                    _string_argument(
                        request, "detail", CatalogDescribeDetail.SCHEMA.value
                    )
                ),
                query=query,
                resource_ids=_string_tuple_argument(request, "resource_ids", ()),
                source_id=source_id,
                limit=limit,
                cursor=cursor,
            ),
            **(
                {}
                if request.source_scope is None
                else {"readable_resource_ids": request.source_scope.resource_ids}
            ),
        )
        return ToolOutput(kind=CATALOG_DESCRIBE_EVIDENCE_KIND, data=projection)


def catalog_declarations(
    agent_id: str,
    service: CatalogProjection,
) -> CatalogDeclarations:
    search_executor = CatalogSearchExecutor(agent_id, service)
    describe_executor = CatalogDescribeExecutor(agent_id, service)
    search = Capability(
        id=CATALOG_SEARCH_CAPABILITY_ID,
        description=(
            "Find catalog tables or fields by name or meaning when the target is "
            "missing or ambiguous. Returns ranked names and match_outcome (unique, "
            "ambiguous or no_match), computed over every candidate before paging; "
            "a hit is a candidate, not authorization. Limits: metadata only, no "
            "rows; up to 50 per page. "
            'Example: {"query": "customer invoices"}. '
            "Mistakes: searching again for exact names already supplied; querying a "
            "hit before catalog_describe confirms its columns."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_TOOL_QUERY_MAX_CHARACTERS,
                },
                "source_id": source_id_schema(),
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
                "cursor": {"type": "string", "minLength": 1, "maxLength": 80},
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
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
    )
    describe = Capability(
        id=CATALOG_DESCRIBE_CAPABILITY_ID,
        description=(
            "Describe catalog structure before SQL, without touching the source. "
            "detail=identity with only a source lists every admitted "
            "table (names, labeled row estimates, no columns). detail=schema "
            "(default) adds columns, keys and join paths; detail=relationships "
            "returns only join paths, multi-hop resolved in code. Reports coverage "
            "and truncation; page with cursor. "
            'Example: {"source": "Shop", "tables": ["main.orders"]}. '
            "Mistakes: joining on matching names; repeating it when current "
            "evidence suffices; reading estimates as exact counts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "detail": {
                    "type": "string",
                    "enum": [detail.value for detail in CatalogDescribeDetail],
                    "default": CatalogDescribeDetail.SCHEMA.value,
                },
                "source_id": source_id_schema(),
                "resource_ids": {
                    "type": "array",
                    "items": resource_id_schema(),
                    "maxItems": CATALOG_SCHEMA_MAX_RESOURCE_IDS,
                    "uniqueItems": True,
                    "default": [],
                },
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_TOOL_QUERY_MAX_CHARACTERS,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": CATALOG_DESCRIBE_IDENTITY_MAX_LIMIT,
                    "description": (
                        f"identity: up to {CATALOG_DESCRIBE_IDENTITY_MAX_LIMIT} "
                        f"(default); schema/relationships: up to {CATALOG_MAX_LIMIT} "
                        f"(default {CATALOG_TOOL_DEFAULT_LIMIT})."
                    ),
                },
                "cursor": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CATALOG_DESCRIBE_CURSOR_MAX_CHARACTERS,
                },
            },
            "additionalProperties": False,
        },
        output_kind=CATALOG_DESCRIBE_EVIDENCE_KIND,
        output_schema=_describe_output_schema(),
        executor_id=describe_executor.executor_id,
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
    )
    return CatalogDeclarations(
        capabilities=(search, describe),
        executors=(search_executor, describe_executor),
        tool_views=(
            ToolView(
                name="catalog_search",
                capability_id=search.id,
                description=search.description,
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.SOURCES,
                    load_mode=ToolLoadMode.PINNED,
                    text_trust=ToolTextTrust.CODE,
                    summary=(
                        "Find catalog resources and fields with a deterministic "
                        "unique, ambiguous, or no-match assessment."
                    ),
                    when_to_use=(
                        "Use when exact source or resource identifiers are not yet "
                        "known; do not treat the first ambiguous hit as selected."
                    ),
                    keywords=("catalog", "find", "resource", "schema"),
                ),
            ),
            ToolView(
                name="catalog_describe",
                capability_id=describe.id,
                description=describe.description,
                presentation=ToolPresentation(
                    toolbox_id=ToolboxId.SOURCES,
                    load_mode=ToolLoadMode.PINNED,
                    text_trust=ToolTextTrust.CODE,
                    summary=(
                        "List table identities, or read SQL-ready schemas, "
                        "relationships and resolved join paths."
                    ),
                    when_to_use=(
                        "Use identity to enumerate tables, schema before composing "
                        "SQL, and relationships for join paths."
                    ),
                    keywords=(
                        "catalog",
                        "describe",
                        "tables",
                        "schema",
                        "columns",
                        "sql",
                        "join",
                        "relationships",
                        "estimates",
                    ),
                ),
            ),
        ),
    )


def _search_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "hits": {"type": "array"},
            "match_outcome": _match_outcome_output_schema(),
            "query": {"type": "string"},
            "total_matches": {"type": "integer"},
            "total_candidates": {"type": "integer"},
            "next_cursor": {"type": ["string", "null"], "maxLength": 80},
            "returned_count": {"type": "integer"},
            "truncated": {"type": "boolean"},
            "trust_classification": {"type": "string"},
        },
        "required": [
            "hits",
            "match_outcome",
            "query",
            "total_matches",
            "total_candidates",
            "next_cursor",
            "returned_count",
            "truncated",
            "trust_classification",
        ],
        "additionalProperties": False,
    }


def _match_outcome_output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "binding_status": {
                "type": "string",
                "enum": ["unique", "ambiguous", "no_match"],
            },
            "source_status": {
                "type": "string",
                "enum": ["unique", "ambiguous", "no_match"],
            },
            "evidence_tier": {
                "type": "string",
                "enum": [
                    "explicit_resource_ids",
                    "exact_resource",
                    "catalog_metadata",
                    "source_hint",
                    "none",
                ],
            },
            "candidate_count": {"type": "integer", "minimum": 0},
            "candidate_bindings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": CATALOG_SOURCE_ID_MAX_CHARACTERS,
                        },
                        "resource_id": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": CATALOG_RESOURCE_ID_MAX_CHARACTERS,
                        },
                    },
                    "required": ["source_id", "resource_id"],
                    "additionalProperties": False,
                },
                "maxItems": CATALOG_MATCH_CANDIDATE_BINDING_LIMIT,
                "uniqueItems": True,
            },
            "omitted_candidate_count": {"type": "integer", "minimum": 0},
            "ambiguity_reasons": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["multiple_sources", "multiple_resources"],
                },
                "maxItems": 2,
                "uniqueItems": True,
            },
            "assessment_provenance": {
                "type": "string",
                "enum": ["catalog_service"],
            },
            "trust_classification": {
                "type": "string",
                "enum": ["untrusted_external_data"],
            },
        },
        "required": [
            "binding_status",
            "source_status",
            "evidence_tier",
            "candidate_count",
            "candidate_bindings",
            "omitted_candidate_count",
            "ambiguity_reasons",
            "assessment_provenance",
            "trust_classification",
        ],
        "additionalProperties": False,
    }


def _describe_output_schema() -> dict[str, object]:
    names = {
        "bounds": "object",
        "coverage": "object",
        "detail": "string",
        "estimates": "object",
        "include_relationships": "boolean",
        "next_cursor": ["string", "null"],
        "paths": "array",
        "relationships": "array",
        "resources": "array",
        "selection": "object",
        "sources": "array",
        "total_matches": "integer",
        "truncated": "boolean",
        "truncation": "object",
        "trust_classification": "string",
    }
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in names.items()},
        "required": ["detail", "resources", "sources", "trust_classification"],
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


def _string_argument(
    request: ToolExecution,
    name: str,
    default: str | None = None,
) -> str:
    value = (
        request.arguments[name]
        if default is None
        else request.arguments.get(name, default)
    )
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
    "CATALOG_DESCRIBE_CAPABILITY_ID",
    "CATALOG_DESCRIBE_EVIDENCE_KIND",
    "CATALOG_SEARCH_CAPABILITY_ID",
    "CATALOG_SEARCH_EVIDENCE_KIND",
    "CatalogDeclarations",
    "CatalogDescribeExecutor",
    "CatalogSearchExecutor",
    "catalog_declarations",
]
