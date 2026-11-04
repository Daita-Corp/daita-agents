"""Declare and execute catalog search and description tools."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from ..capabilities import (
    AccessMode,
    Capability,
    CapabilityInputError,
    Executor,
    RecoveryKind,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolRecovery,
    ToolTextTrust,
    ToolView,
    source_id_schema,
)
from ..errors import DaitaError
from ..scope import EffectiveSourceScope
from .models import (
    CATALOG_MAX_LIMIT,
    CATALOG_TOOL_DEFAULT_LIMIT,
    CATALOG_TOOL_QUERY_MAX_CHARACTERS,
    CatalogSearchRequest,
    ResourceKind,
    catalog_resource_id,
)
from .protocols import CatalogStoreError

CATALOG_SEARCH_CAPABILITY_ID = "catalog.search"
CATALOG_DESCRIBE_CAPABILITY_ID = "catalog.describe"
CATALOG_SEARCH_EVIDENCE_KIND = "catalog.search_result"
CATALOG_DESCRIBE_EVIDENCE_KIND = "catalog.description"

_NAME_MAX_CHARACTERS = 1_024
_MAX_QUERY_WORDS = 8
_COMMON_WORDS = frozenset(
    "all and are for from how many much per the was what which with".split()
)
# What a source's own failure can raise; a lookup then says so instead of failing.
_LOOKUP_FAILURES = (DaitaError, ImportError, OSError, TimeoutError)
_TRUST = "untrusted_external_data"

RefreshRequest = Callable[[str, tuple[str, ...]], None]


@dataclass(frozen=True, slots=True)
class CatalogDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class LiveLookup(Protocol):
    """Bounded name search and description one source type answers itself."""

    async def search_names(
        self, *, agent_id: str, source_id: str, words: tuple[str, ...], limit: int
    ) -> tuple[Mapping[str, object], ...]: ...

    async def describe_name(
        self, *, agent_id: str, source_id: str, name: str
    ) -> Mapping[str, object] | None: ...


class CatalogProjection(Protocol):
    @property
    def session_source_id(self) -> str | None: ...

    async def lookup_sources(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[tuple[str, str, str | None], ...]: ...

    async def catalog_hits(
        self, request: CatalogSearchRequest
    ) -> tuple[Mapping[str, object], ...]: ...

    async def cached_description(
        self, agent_id: str, source_id: str, name: str
    ) -> Mapping[str, object] | None: ...


def _query_words(query: str) -> tuple[str, ...]:
    words = tuple(dict.fromkeys(re.findall(r"\w{2,}", query.lower())))
    meaningful = tuple(word for word in words if word not in _COMMON_WORDS)
    return (meaningful or words)[:_MAX_QUERY_WORDS]


def _visible(
    scope: EffectiveSourceScope | None,
    session_source_id: str | None,
    source_id: str,
    kind: object,
    name: str,
) -> bool:
    """The run's frozen scope decides; an ID hashes its source, kind and name."""

    if (
        scope is None
        or source_id == session_source_id
        or source_id in scope.unrestricted_source_ids
    ):
        return True
    return catalog_resource_id(source_id, ResourceKind(str(kind)), name) in (
        scope.resource_ids
    )


class CatalogSearchExecutor:
    executor_id = "catalog.search.executor"

    def __init__(
        self,
        agent_id: str,
        service: CatalogProjection,
        live: Mapping[str, LiveLookup],
        request_refresh: RefreshRequest = lambda source_id, names: None,
    ) -> None:
        self._agent_id = agent_id
        self._service = service
        self._live = live
        self._request_refresh = request_refresh

    async def execute(self, request: ToolExecution) -> ToolOutput:
        query = _string_argument(request, "query")
        limit = _integer_argument(request, "limit", CATALOG_TOOL_DEFAULT_LIMIT)
        kinds = _string_tuple_argument(request, "kinds", ())
        named = request.arguments.get("source_id")
        scope = request.source_scope
        sources = await self._service.lookup_sources(
            self._agent_id,
            (
                (cast(str, named),)
                if named is not None
                else () if scope is None else tuple(sorted(scope.source_ids))
            ),
        )
        dialects = {source_id: dialect for source_id, _, dialect in sources}
        words = _query_words(query)
        unreachable: list[str] = []

        async def live_hits(
            source_id: str, adapter_id: str
        ) -> tuple[tuple[str, Mapping[str, object]], ...]:
            backend = self._live.get(adapter_id)
            if backend is None or not words:
                return ()
            try:
                found = await backend.search_names(
                    agent_id=self._agent_id,
                    source_id=source_id,
                    words=words,
                    limit=limit,
                )
            except _LOOKUP_FAILURES:
                unreachable.append(source_id)
                return ()
            return tuple((source_id, hit) for hit in found)

        # The source and the catalog are always both asked, so a stale catalog
        # cannot hide a table and a failing one cannot stop the search.
        found = await asyncio.gather(
            *(live_hits(source_id, adapter) for source_id, adapter, _ in sources)
        )
        catalog_failed = False
        cataloged: tuple[Mapping[str, object], ...] = ()
        if sources:
            try:
                cataloged = await self._service.catalog_hits(
                    CatalogSearchRequest(
                        agent_id=self._agent_id,
                        query=query,
                        source_ids=tuple(source_id for source_id, _, _ in sources),
                        resource_kinds=tuple(ResourceKind(kind) for kind in kinds),
                        limit=CATALOG_MAX_LIMIT,
                        run_id=request.run_id,
                    )
                )
            except CatalogStoreError:
                catalog_failed = True
        merged: dict[tuple[str, str], dict[str, object]] = {}
        session = self._service.session_source_id
        for source_id, hit in (pair for hits in found for pair in hits):
            if (not kinds or hit["kind"] in kinds) and _visible(
                scope, session, source_id, hit["kind"], cast(str, hit["name"])
            ):
                merged[(source_id, cast(str, hit["name"]))] = {
                    **hit,
                    "source_id": source_id,
                    "dialect": dialects.get(source_id),
                    "seen_live": True,
                }
        for hit in cataloged:
            source_id = cast(str, hit["source_id"])
            name = cast(str, hit["name"])
            current = merged.get((source_id, name))
            if current is not None:
                for key in ("columns", "row_estimate"):
                    current[key] = current.get(key) or hit.get(key)
            elif _visible(scope, session, source_id, hit["kind"], name):
                merged[(source_id, name)] = {
                    **hit,
                    "dialect": dialects.get(source_id),
                    "seen_live": False,
                }
        # A hit only the source knows means the catalog is behind; a failed
        # catalog search says nothing about that.
        in_catalog = {
            (cast(str, hit["source_id"]), cast(str, hit["name"])) for hit in cataloged
        }
        missed: dict[str, list[str]] = {}
        for (source_id, name), hit in merged.items():
            if hit["seen_live"] and (source_id, name) not in in_catalog:
                missed.setdefault(source_id, []).append(name)
        if not catalog_failed:
            for source_id, names in missed.items():
                if source_id != session:
                    self._request_refresh(source_id, tuple(names))
        hits = list(merged.values())
        return ToolOutput(
            kind=CATALOG_SEARCH_EVIDENCE_KIND,
            data={
                "query": query,
                "hits": hits[:limit],
                "returned_count": min(len(hits), limit),
                "truncated": len(hits) > limit,
                "live_unavailable_source_ids": tuple(sorted(unreachable)),
                "catalog_available": not catalog_failed,
                "trust_classification": _TRUST,
            },
        )


class CatalogDescribeExecutor:
    executor_id = "catalog.describe.executor"

    def __init__(
        self,
        agent_id: str,
        service: CatalogProjection,
        live: Mapping[str, LiveLookup],
        request_refresh: RefreshRequest = lambda source_id, names: None,
    ) -> None:
        self._agent_id = agent_id
        self._service = service
        self._live = live
        self._request_refresh = request_refresh

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = _string_argument(request, "source_id")
        name = _string_argument(request, "name")
        ((_, adapter_id, dialect),) = await self._service.lookup_sources(
            self._agent_id, (source_id,)
        )
        backend = self._live.get(adapter_id)
        live: Mapping[str, object] | None = None
        unreachable = False
        if backend is not None:
            try:
                live = await backend.describe_name(
                    agent_id=self._agent_id, source_id=source_id, name=name
                )
            except _LOOKUP_FAILURES:
                unreachable = True
        cached: Mapping[str, object] | None = None
        catalog_failed = False
        try:
            cached = await self._service.cached_description(
                self._agent_id, source_id, name
            )
        except CatalogStoreError:
            catalog_failed = True
        if backend is not None and not unreachable and not catalog_failed:
            stale = _disagreement(live, cached)
            if stale is not None and source_id != self._service.session_source_id:
                self._request_refresh(source_id, (stale,))
        # A cached answer never overrides a live one, and a table the live source
        # no longer has is gone even if the catalog remembers it.
        if live is not None:
            answer = {**live, "answered_from": "live"}
            if cached is not None and cached["name"] == live["name"]:
                answer["revision"] = cached.get("revision")
                answer["relationships"] = _merged_relationships(
                    cast(Sequence[Mapping[str, object]], live.get("relationships", ())),
                    cast(
                        Sequence[Mapping[str, object]], cached.get("relationships", ())
                    ),
                )
        elif cached is not None and (unreachable or backend is None):
            answer = {**cached, "answered_from": "cache"}
            if unreachable:
                answer["note"] = (
                    "The source could not be reached; this is the catalog's last "
                    "description and may be stale."
                )
        elif unreachable:
            raise CapabilityInputError(
                "source_unreachable",
                "The source could not be reached and the catalog has no description "
                f"of {name!r}.",
                recovery=ToolRecovery(
                    RecoveryKind.CORRECT_ARGUMENTS,
                    "Report that the source is unavailable, or try another source.",
                ),
            )
        else:
            answer = {}
        visible = self._scoped(request.source_scope, source_id, answer)
        if visible is None:
            raise CapabilityInputError(
                "table_unknown",
                f"No table, view or file named {name!r} is visible in this source.",
                {"name": "name", "value": name},
                recovery=ToolRecovery(
                    RecoveryKind.CORRECT_ARGUMENTS,
                    "Use catalog_search to find the name, or qualify it with its schema.",
                ),
            )
        return ToolOutput(
            kind=CATALOG_DESCRIBE_EVIDENCE_KIND,
            data={
                **visible,
                "source_id": source_id,
                **({} if dialect is None else {"dialect": dialect}),
                "trust_classification": _TRUST,
            },
        )

    def _scoped(
        self,
        scope: EffectiveSourceScope | None,
        source_id: str,
        answer: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Show only what this run may read, so names outside it cannot leak."""

        if not answer:
            return None
        session = self._service.session_source_id
        kind = answer["kind"]
        name = cast(str, answer["name"])
        scoped = {key: value for key, value in answer.items() if value is not None}
        if "tables" in answer:
            tables = cast(Sequence[Mapping[str, object]], answer["tables"])
            shown = [
                table
                for table in tables
                if _visible(
                    scope, session, source_id, table["kind"], cast(str, table["name"])
                )
            ]
            scoped["tables"] = shown
            if "total_tables" in scoped and len(shown) != len(tables):
                scoped["total_tables"] = len(shown)
            if kind == "ambiguous" and not shown:
                return None
            return scoped
        if not _visible(scope, session, source_id, kind, name):
            return None
        scoped["relationships"] = [
            {
                key: value
                for key, value in relation.items()
                if key not in {"from_kind", "to_kind"}
            }
            for relation in cast(
                Sequence[Mapping[str, object]], answer.get("relationships", ())
            )
            if all(
                _visible(
                    scope,
                    session,
                    cast(str, relation.get(f"{end}_source_id", source_id)),
                    relation.get(f"{end}_kind", "table"),
                    cast(str, relation[end]),
                )
                for end in ("from", "to")
            )
        ]
        return scoped


def _disagreement(
    live: Mapping[str, object] | None, cached: Mapping[str, object] | None
) -> str | None:
    """The relation the live source and the catalog disagree about, if any.

    Only a table, view or file counts: an ambiguous name or a schema listing is
    not one relation. Columns are compared by name and type, which both produce.
    """

    if live is None:
        return None if cached is None else cast(str, cached["name"])
    if live["kind"] == "ambiguous" or "tables" in live:
        return None
    name = cast(str, live["name"])
    if cached is None:
        return name
    return (
        name
        if cached["name"] == name and _column_types(live) != _column_types(cached)
        else None
    )


def _column_types(description: Mapping[str, object]) -> list[tuple[object, object]]:
    return [
        (column["name"], column["type"])
        for column in cast(
            Sequence[Mapping[str, object]], description.get("columns", ())
        )
    ]


def _merged_relationships(
    live: Sequence[Mapping[str, object]], cached: Sequence[Mapping[str, object]]
) -> list[Mapping[str, object]]:
    """Live keys, plus the links only the catalog knows, such as declared ones."""

    seen = {(item["from"], item["to"], repr(item["columns"])) for item in live}
    return [
        *live,
        *(
            item
            for item in cached
            if (item["from"], item["to"], repr(item["columns"])) not in seen
        ),
    ]


def catalog_declarations(
    agent_id: str,
    service: CatalogProjection,
    live: Mapping[str, LiveLookup] | None = None,
    request_refresh: RefreshRequest = lambda source_id, names: None,
) -> CatalogDeclarations:
    search_executor = CatalogSearchExecutor(
        agent_id, service, live or {}, request_refresh
    )
    describe_executor = CatalogDescribeExecutor(
        agent_id, service, live or {}, request_refresh
    )
    search = Capability(
        id=CATALOG_SEARCH_CAPABILITY_ID,
        description=(
            "Find tables, views and files by words in their table or column names. "
            "Every search checks the live source as well as the catalog. Returns "
            "compact hits: full name, kind, leading columns, row estimate, dialect "
            "and whether it was seen live. A hit is a candidate, not a choice. "
            'Limits: no rows; up to 50 hits. Example: {"query": "customer invoices"}. '
            "Mistakes: searching again for a name you already have."
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
                "kinds": {
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
    )
    describe = Capability(
        id=CATALOG_DESCRIBE_CAPABILITY_ID,
        description=(
            "Describe one table, view or file from the live source: columns with "
            "native types, enum values, keys, partitions as one table with a child "
            "count, comments, row estimate and relationships. A schema or folder "
            "name lists what it holds, capped at 100. answered_from says live or "
            'cache. Example: {"source": "Shop", "name": "sales.orders"}. '
            "Mistakes: describing many tables when one query answers; reading "
            "estimates as exact counts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": source_id_schema(),
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _NAME_MAX_CHARACTERS,
                },
            },
            "required": ["source_id", "name"],
            "additionalProperties": False,
        },
        output_kind=CATALOG_DESCRIBE_EVIDENCE_KIND,
        output_schema=_describe_output_schema(),
        executor_id=describe_executor.executor_id,
        access_mode=AccessMode.READ,
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
                        "Find tables and fields by name, in the live source and "
                        "the catalog together."
                    ),
                    when_to_use=(
                        "Use when the exact table is not yet known; choose among "
                        "hits from their evidence."
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
                        "Read one table's live columns, keys and relationships, or "
                        "list a schema."
                    ),
                    when_to_use="Use before composing SQL against an unfamiliar table.",
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
            "query": {"type": "string"},
            "returned_count": {"type": "integer"},
            "truncated": {"type": "boolean"},
            "live_unavailable_source_ids": {"type": "array"},
            "catalog_available": {"type": "boolean"},
            "trust_classification": {"type": "string"},
        },
        "required": [
            "hits",
            "query",
            "returned_count",
            "truncated",
            "live_unavailable_source_ids",
            "catalog_available",
            "trust_classification",
        ],
        "additionalProperties": False,
    }


def _describe_output_schema() -> dict[str, object]:
    names = {
        "answered_from": "string",
        "columns": "array",
        "columns_truncated": "boolean",
        "comment": "string",
        "dialect": "string",
        "kind": "string",
        "name": "string",
        "note": "string",
        "partitions": "integer",
        "primary_key": "array",
        "relationships": "array",
        "revision": "string",
        "row_estimate": "integer",
        "source_id": "string",
        "tables": "array",
        "total_tables": "integer",
        "trust_classification": "string",
        "unique_keys": "array",
    }
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in names.items()},
        "required": [
            "answered_from",
            "kind",
            "name",
            "source_id",
            "trust_classification",
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
    "LiveLookup",
    "catalog_declarations",
]
