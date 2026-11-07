"""Search, inspect, and traverse the stored catalog through indexed lookups."""

from __future__ import annotations

import hmac
import json
import re
import secrets
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

from .._json import FrozenJsonObject, canonical_json
from .models import (
    CATALOG_MAX_LIMIT,
    CATALOG_SCHEMA_DEFAULT_JOIN_DEPTH,
    CATALOG_SCHEMA_MAX_RELATIONSHIPS,
    CATALOG_TRAVERSAL_MAX_EDGES,
    CATALOG_TRAVERSAL_MAX_NODES,
    CATALOG_TRAVERSAL_MAX_PATHS,
    CatalogEntry,
    CatalogFacet,
    CatalogLink,
    CatalogLinkStaleReason,
    CatalogLinkStatus,
    CatalogPath,
    CatalogPathStep,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSchemaRequest,
    CatalogSearchHit,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSearchRow,
    CatalogSummary,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    CatalogTraversalResult,
    CatalogTraversalTruncationReason,
    FacetKind,
    RelationshipDirection,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    TabularFacet,
    catalog_facet_revision,
    catalog_link_id,
)
from .protocols import CatalogResourceNotFoundError, CatalogStore, CatalogStoreError
from .search_text import (
    SEARCH_STOP_WORDS as _SEARCH_STOP_WORDS,
    NormalizedText as _NormalizedText,
    _is_identifier_character,
    contains_bounded_identifier as _contains_bounded_identifier,
    normalize_search_text as _normalize_search_text,
    search_postings,
)

if TYPE_CHECKING:
    from ..adapters.protocols import SourceStore

_INSPECT_INCIDENT_RELATIONSHIP_LIMIT = 50
_INCIDENT_RELATIONSHIP_LIMIT = CATALOG_TRAVERSAL_MAX_EDGES + 1
_SCHEMA_COLUMN_LIMIT = 256
_SCHEMA_KEY_LIMIT = 64
_SCHEMA_RELATIONSHIP_LIMIT = CATALOG_SCHEMA_MAX_RELATIONSHIPS
_SCHEMA_STRUCTURAL_FACT_LIMIT = 32
_SCHEMA_JOIN_MAX_EDGES = CATALOG_TRAVERSAL_MAX_EDGES
_SCHEMA_JOIN_MAX_NODES = CATALOG_TRAVERSAL_MAX_NODES
TABULAR_KINDS = frozenset({ResourceKind.TABLE, ResourceKind.VIEW, ResourceKind.FILE})
# An identity mention longer than this, or beyond this many distinct
# boundary-delimited substrings of one query, is not looked up by name.
_ANCHOR_KEY_LENGTH = 512
_ANCHOR_KEY_LIMIT = 2_000
_GENERATION_CHANGED = "catalog snapshot generation changed repeatedly"


@dataclass(frozen=True, slots=True)
class _IndexedPosting:
    resource_id: str
    field_kind: str
    field_name: str
    normalized_value: str
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _IndexedEdge:
    neighbor_resource_id: str
    relationship_id: str
    direction: RelationshipDirection


@dataclass(frozen=True, slots=True)
class _ParentStep:
    parent_resource_id: str
    relationship_id: str
    direction: RelationshipDirection


@dataclass(frozen=True, slots=True)
class _LinkGraph:
    """Current, readable declared links as edges over the visible resources."""

    links_by_id: Mapping[str, CatalogLink]
    adjacency_by_resource_id: Mapping[str, tuple[_IndexedEdge, ...]]


_NO_LINKS = _LinkGraph(MappingProxyType({}), MappingProxyType({}))


class ReadableScope(frozenset[str]):
    """Readable resource IDs: those chosen, plus every resource of whole sources.

    An ID alone cannot say which source it is in, so a resource is admitted
    through its source or its ID: ``_allows``.
    """

    open_sources: frozenset[str]

    def __new__(
        cls, resource_ids: Iterable[str], open_sources: Iterable[str]
    ) -> "ReadableScope":
        scope = super().__new__(cls, resource_ids)
        scope.open_sources = frozenset(open_sources)
        return scope


def _allows(readable: frozenset[str] | None, resource_id: str, source_id: str) -> bool:
    return (
        readable is None
        or resource_id in readable
        or source_id in getattr(readable, "open_sources", ())
    )


@dataclass(frozen=True, slots=True)
class _Found:
    """What ranking and hit construction need of one resource.

    A search row carries no sensitivity; the returned page fills it in.
    """

    id: str
    source_id: str
    kind: ResourceKind
    name: str
    native_identity: str
    revision: str
    sensitivity: Sensitivity


def _found_from_row(row: CatalogSearchRow) -> _Found:
    return _Found(
        row.resource_id,
        row.source_id,
        row.kind,
        row.name,
        row.native_identity,
        row.revision,
        Sensitivity.UNKNOWN,
    )


def _found_from_resource(resource: CatalogResource) -> _Found:
    return _Found(
        resource.id,
        resource.source_id,
        resource.kind,
        resource.name,
        resource.native_identity,
        resource.current_revision,
        resource.sensitivity,
    )


def _decode_tabular_facet(facet: CatalogFacet) -> TabularFacet:
    if facet.kind is not FacetKind.TABULAR:
        raise CatalogStoreError("catalog facet is not tabular")
    try:
        decoded = TabularFacet.from_payload(facet.payload)
        expected_revision = catalog_facet_revision(
            FacetKind.TABULAR,
            decoded.structural_payload(),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CatalogStoreError(
            "catalog tabular facet has invalid structure"
        ) from error
    if facet.revision != expected_revision:
        raise CatalogStoreError(
            "catalog tabular facet revision does not match its structure"
        )
    return decoded


@dataclass(frozen=True, slots=True)
class CatalogTabularResource:
    """Exact current catalog records needed for one validation schema."""

    sync: CatalogSync
    resource: CatalogResource
    revision: CatalogResourceRevision
    facet: CatalogFacet
    tabular: TabularFacet

    def __post_init__(self) -> None:
        if not isinstance(self.sync, CatalogSync):
            raise TypeError("tabular resource sync must be a CatalogSync")
        if not isinstance(self.resource, CatalogResource):
            raise TypeError("tabular resource must be a CatalogResource")
        if not isinstance(self.revision, CatalogResourceRevision):
            raise TypeError(
                "tabular resource revision must be a CatalogResourceRevision"
            )
        if not isinstance(self.facet, CatalogFacet):
            raise TypeError("tabular resource facet must be a CatalogFacet")
        if not isinstance(self.tabular, TabularFacet):
            raise TypeError("tabular resource decoded facet must be a TabularFacet")
        if self.resource.kind not in {
            ResourceKind.TABLE,
            ResourceKind.VIEW,
            ResourceKind.FILE,
        }:
            raise ValueError("tabular resource has a non-tabular resource kind")
        if self.facet.kind is not FacetKind.TABULAR:
            raise ValueError("tabular resource requires a tabular facet")
        if self.facet.revision != catalog_facet_revision(
            FacetKind.TABULAR,
            self.tabular.structural_payload(),
        ):
            raise ValueError(
                "tabular resource facet revision does not match decoded structure"
            )
        if (
            self.resource.agent_id != self.sync.agent_id
            or self.resource.source_id != self.sync.source_id
            or self.revision.resource_id != self.resource.id
            or self.revision.revision != self.resource.current_revision
            or self.facet.resource_id != self.resource.id
            or self.facet.revision not in self.revision.facet_revisions
        ):
            raise ValueError("tabular resource records are not exact-current bindings")


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    hit: CatalogSearchHit
    matched_terms: tuple[str, ...]
    rank_key: tuple[int, int, int, int, int, int, int, int, int, str, str, str]


@dataclass(frozen=True, slots=True)
class _CatalogHitSelection:
    exact_anchors: tuple[CatalogSearchHit, ...]
    broad_matches: tuple[CatalogSearchHit, ...]
    source_hint_matches: tuple[CatalogSearchHit, ...]
    relationship_neighbors: tuple[CatalogSearchHit, ...]

    @property
    def ordered_hits(self) -> tuple[CatalogSearchHit, ...]:
        return (
            *self.exact_anchors,
            *self.broad_matches,
            *self.source_hint_matches,
            *self.relationship_neighbors,
        )


@dataclass(frozen=True, slots=True)
class _SchemaJoinSelection:
    paths: tuple[CatalogPath, ...]
    bridge_resource_ids: tuple[str, ...]
    relationship_ids: tuple[str, ...]
    unresolved_reasons: tuple[str, ...]
    truncation_reason: str | None
    visited_nodes: int
    visited_edges: int
    # Each source's join group: sources joined by current links share one.
    source_groups: Mapping[str, str]


class _CatalogGenerationChanged(RuntimeError):
    """Signal one bounded retry when a source's current sync changed mid-read."""


class _CatalogReader:
    """Current-row lookups for one request; nothing here outlives the call."""

    def __init__(
        self,
        store: CatalogStore,
        agent_id: str,
        source_ids: Collection[str],
        readable_resource_ids: frozenset[str] | None,
    ) -> None:
        self._store = store
        self._agent_id = agent_id
        self._source_ids = frozenset(source_ids)
        self._readable = readable_resource_ids
        self.entries: dict[str, CatalogEntry] = {}
        self.relationships: dict[str, CatalogRelationship] = {}
        self._absent: set[str] = set()
        self._incident: dict[
            tuple[str, tuple[RelationshipKind, ...]],
            tuple[CatalogRelationship, ...],
        ] = {}
        self._edges: dict[
            tuple[str, tuple[RelationshipKind, ...]], tuple[_IndexedEdge, ...]
        ] = {}

    async def load(self, resource_ids: Iterable[str]) -> None:
        """Load readable entries of the scoped sources; others stay absent."""

        wanted = tuple(
            dict.fromkeys(
                resource_id
                for resource_id in resource_ids
                if resource_id not in self.entries and resource_id not in self._absent
            )
        )
        if not wanted:
            return
        found = {
            entry.resource.id: entry
            for entry in await self._store.load_entries(self._agent_id, wanted)
        }
        for resource_id in wanted:
            entry = found.get(resource_id)
            if (
                entry is None
                or entry.resource.agent_id != self._agent_id
                or entry.resource.source_id not in self._source_ids
                or not _allows(self._readable, resource_id, entry.resource.source_id)
            ):
                self._absent.add(resource_id)
            else:
                self.entries[resource_id] = entry

    async def prefetch(
        self,
        resource_ids: Iterable[str],
        kinds: tuple[RelationshipKind, ...] = (),
        *,
        endpoints: bool = True,
    ) -> None:
        """Load incident relationships (and their endpoints) for many resources at once."""

        resource_ids = tuple(resource_ids)

        wanted = tuple(
            dict.fromkeys(i for i in resource_ids if (i, kinds) not in self._incident)
        )
        if wanted:
            found = await self._store.load_incident_relationships(
                self._agent_id,
                wanted,
                relationship_kinds=kinds,
                limit=_INCIDENT_RELATIONSHIP_LIMIT,
            )
            for resource_id in wanted:
                self._incident[(resource_id, kinds)] = found.get(resource_id, ())
        if endpoints:
            await self.load(
                endpoint
                for resource_id in resource_ids
                for relationship in self._readable_incident(resource_id, kinds)
                for endpoint in (
                    relationship.from_resource_id,
                    relationship.to_resource_id,
                )
            )

    def fetched(self, resource_id: str, kinds: tuple[RelationshipKind, ...]) -> bool:
        return (resource_id, kinds) in self._incident

    def _readable_incident(
        self, resource_id: str, kinds: tuple[RelationshipKind, ...]
    ) -> tuple[CatalogRelationship, ...]:
        incident = self._incident[(resource_id, kinds)]
        if self._readable is None:
            return incident
        return tuple(
            relationship
            for relationship in incident
            if _allows(
                self._readable, relationship.from_resource_id, relationship.source_id
            )
            and _allows(
                self._readable, relationship.to_resource_id, relationship.source_id
            )
        )

    async def incident(
        self,
        resource_id: str,
        kinds: tuple[RelationshipKind, ...] = (),
    ) -> tuple[CatalogRelationship, ...]:
        if (resource_id, kinds) not in self._incident:
            await self.prefetch((resource_id,), kinds)
        return self._incident[(resource_id, kinds)]

    async def edges(
        self,
        resource_id: str,
        kinds: tuple[RelationshipKind, ...] = (),
    ) -> tuple[_IndexedEdge, ...]:
        """Edges to loaded neighbours, in the order traversal expands them."""

        key = (resource_id, kinds)
        cached = self._edges.get(key)
        if cached is not None:
            return cached
        await self.prefetch((resource_id,), kinds)
        incident = self._readable_incident(resource_id, kinds)
        found: list[tuple[CatalogRelationship, _IndexedEdge]] = []
        for relationship in incident:
            if (
                relationship.from_resource_id not in self.entries
                or relationship.to_resource_id not in self.entries
            ):
                continue
            self.relationships[relationship.id] = relationship
            if relationship.from_resource_id == resource_id:
                found.append(
                    (
                        relationship,
                        _IndexedEdge(
                            relationship.to_resource_id,
                            relationship.id,
                            RelationshipDirection.FORWARD,
                        ),
                    )
                )
            if relationship.to_resource_id == resource_id:
                found.append(
                    (
                        relationship,
                        _IndexedEdge(
                            relationship.from_resource_id,
                            relationship.id,
                            RelationshipDirection.REVERSE,
                        ),
                    )
                )

        def sort_key(item: tuple[CatalogRelationship, _IndexedEdge]) -> tuple[str, ...]:
            relationship, edge = item
            neighbor = self.entries[edge.neighbor_resource_id].resource
            return (
                relationship.kind.value,
                relationship.provenance.value,
                _normalize_search_text(neighbor.native_identity).complete,
                relationship.id,
                edge.direction.value,
                edge.neighbor_resource_id,
            )

        edges = tuple(edge for _, edge in sorted(found, key=sort_key))
        self._edges[key] = edges
        return edges


class CatalogService:
    """Keep catalog reads and their bounded projections behind one owner."""

    def __init__(self, store: CatalogStore, sources: SourceStore) -> None:
        if not isinstance(store, CatalogStore):
            raise TypeError("store must implement CatalogStore")
        if not callable(getattr(sources, "load_source", None)) or not callable(
            getattr(sources, "list_sources", None)
        ):
            raise TypeError("sources must provide source registration reads")
        self._store = store
        self._sources = sources
        self._search_cursor_key = secrets.token_bytes(32)

    async def summary(self, agent_id: str) -> CatalogSummary:
        """Project active source counts from their current committed snapshots."""

        active_source_ids = tuple(
            sorted(
                registration.id
                for registration in await self._sources.list_sources(agent_id)
                if registration.agent_id == agent_id and registration.active
            )
        )
        return await self._store.summarize_catalog(agent_id, active_source_ids)

    async def preview(
        self,
        agent_id: str,
        *,
        limit: int,
    ) -> tuple[CatalogResource, ...]:
        """Project a bounded deterministic view of active current resources."""

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= CATALOG_MAX_LIMIT
        ):
            raise ValueError(
                f"catalog preview limit must be from 1 through {CATALOG_MAX_LIMIT}"
            )
        active_source_ids = tuple(
            sorted(
                registration.id
                for registration in await self._sources.list_sources(agent_id)
                if registration.agent_id == agent_id and registration.active
            )
        )
        if not active_source_ids:
            return ()
        rows = await self._store.search_catalog(agent_id, active_source_ids, terms=None)
        page = sorted(rows, key=lambda row: (row.name, row.resource_id))[:limit]
        entries = await self._store.load_entries(
            agent_id, tuple(row.resource_id for row in page)
        )
        return tuple(
            sorted(
                (entry.resource for entry in entries),
                key=lambda resource: (resource.name, resource.id),
            )
        )

    async def declare_link(
        self,
        agent_id: str,
        from_resource_id: str,
        to_resource_id: str,
        field_pairs: tuple[RelationshipFieldPair, ...],
        *,
        declared_at: datetime,
        note: str | None = None,
    ) -> CatalogLink:
        """Validate both endpoints and fields against current truth, then persist."""

        endpoints: list[tuple[CatalogResource, frozenset[str]]] = []
        for resource_id in (from_resource_id, to_resource_id):
            entry = await self._active_entry(agent_id, resource_id)
            endpoints.append((entry.resource, _tabular_field_names(entry.facets)))
        (from_resource, from_fields), (to_resource, to_fields) = endpoints
        for pair in field_pairs:
            for field_name, resource, fields in (
                (pair.source_field, from_resource, from_fields),
                (pair.target_field, to_resource, to_fields),
            ):
                if field_name not in fields:
                    raise ValueError(
                        f"catalog link field {field_name!r} is not a current "
                        f"column of {resource.native_identity}"
                    )
        link = CatalogLink(
            id=catalog_link_id(
                from_resource_id=from_resource_id,
                to_resource_id=to_resource_id,
                field_pairs=field_pairs,
            ),
            agent_id=agent_id,
            from_source_id=from_resource.source_id,
            from_resource_id=from_resource_id,
            to_source_id=to_resource.source_id,
            to_resource_id=to_resource_id,
            field_pairs=field_pairs,
            declared_at=declared_at,
            declared_revisions=(
                from_resource.current_revision,
                to_resource.current_revision,
            ),
            note=note,
        )
        await self._store.save_catalog_link(link)
        return link

    async def remove_link(self, agent_id: str, link_id: str) -> bool:
        return await self._store.delete_catalog_link(agent_id, link_id)

    async def link_statuses(self, agent_id: str) -> tuple[CatalogLinkStatus, ...]:
        """Return every declared link, current or stale with its reason."""

        links = await self._store.list_catalog_links(agent_id)
        active_source_ids = await self._active_source_ids(agent_id)
        indexed = tuple(
            sync.source_id
            for sync in await self._current_syncs(agent_id, active_source_ids)
        )
        reader = _CatalogReader(self._store, agent_id, indexed, None)
        await reader.load(
            resource_id
            for link in links
            for resource_id in (link.from_resource_id, link.to_resource_id)
        )
        return tuple(
            CatalogLinkStatus(
                link,
                _link_stale_reason(link, active_source_ids, indexed, reader.entries),
            )
            for link in links
        )

    async def _current_syncs(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> tuple[CatalogSync, ...]:
        # An empty id tuple means every source to the store, never "none".
        if not source_ids:
            return ()
        return await self._store.current_syncs(agent_id, source_ids)

    async def _current_link_graph(
        self,
        reader: _CatalogReader,
        agent_id: str,
        source_ids: Collection[str],
        readable_resource_ids: frozenset[str] | None,
    ) -> _LinkGraph:
        """Keep links current over these sources whose endpoints are readable."""

        links = tuple(
            link
            for link in await self._store.list_catalog_links(agent_id)
            if link.from_source_id in source_ids
            and link.to_source_id in source_ids
            and (
                _allows(
                    readable_resource_ids, link.from_resource_id, link.from_source_id
                )
                and _allows(
                    readable_resource_ids, link.to_resource_id, link.to_source_id
                )
            )
        )
        await reader.load(
            resource_id
            for link in links
            for resource_id in (link.from_resource_id, link.to_resource_id)
        )
        links_by_id: dict[str, CatalogLink] = {}
        adjacency: dict[str, list[_IndexedEdge]] = {}
        for link in links:
            if _link_stale_reason(link, source_ids, source_ids, reader.entries):
                continue
            links_by_id[link.id] = link
            adjacency.setdefault(link.from_resource_id, []).append(
                _IndexedEdge(
                    link.to_resource_id, link.id, RelationshipDirection.FORWARD
                )
            )
            adjacency.setdefault(link.to_resource_id, []).append(
                _IndexedEdge(
                    link.from_resource_id, link.id, RelationshipDirection.REVERSE
                )
            )
        return _LinkGraph(
            MappingProxyType(links_by_id),
            MappingProxyType({key: tuple(edges) for key, edges in adjacency.items()}),
        )

    async def tabular_resources(
        self,
        agent_id: str,
        source_id: str,
        *,
        resource_ids: tuple[str, ...] | None = None,
        identities: tuple[str, ...] | None = None,
    ) -> tuple[CatalogTabularResource, ...]:
        """The tabular resources of one exact current source generation.

        ``resource_ids`` or ``identities`` (native names) read only those rows;
        with neither, the whole source is read, which only rare paths should do.
        """

        for attempt in range(2):
            await self._active_source(agent_id, source_id)
            syncs = await self._store.current_syncs(agent_id, (source_id,))
            if any(
                sync.agent_id != agent_id or sync.source_id != source_id
                for sync in syncs
            ):
                raise CatalogStoreError("catalog sync escaped validation source scope")
            if len(syncs) > 1:
                raise CatalogStoreError(
                    "catalog validation source has multiple current generations"
                )
            records: list[CatalogTabularResource] = []
            if syncs:
                if resource_ids is not None:
                    entries = await self._store.load_entries(agent_id, resource_ids)
                elif identities is not None:
                    entries = await self._store.load_entries_by_identity(
                        agent_id, source_id, identities
                    )
                else:
                    entries = await self._store.list_entries(agent_id, source_id)
                for entry in entries:
                    resource = entry.resource
                    if resource_ids is not None and resource.source_id != source_id:
                        continue  # asked by ID across sources; not this source's
                    if resource.agent_id != agent_id or resource.source_id != source_id:
                        raise CatalogStoreError(
                            "catalog validation entry escaped source scope"
                        )
                    if resource.kind not in TABULAR_KINDS:
                        continue
                    tabular_facets = tuple(
                        facet
                        for facet in entry.facets
                        if facet.kind is FacetKind.TABULAR
                    )
                    if not tabular_facets:
                        continue
                    if len(tabular_facets) != 1:
                        raise CatalogStoreError(
                            "catalog resource has multiple current tabular facets"
                        )
                    try:
                        records.append(
                            CatalogTabularResource(
                                sync=syncs[0],
                                resource=resource,
                                revision=entry.revision,
                                facet=tabular_facets[0],
                                tabular=_decode_tabular_facet(tabular_facets[0]),
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise CatalogStoreError(
                            "catalog tabular resource has inconsistent current facts"
                        ) from exc

            try:
                await self._active_source(agent_id, source_id)
            except CatalogStoreError:
                if attempt == 0:
                    continue
                break
            if await self._store.current_syncs(agent_id, (source_id,)) == syncs:
                return tuple(
                    sorted(
                        records, key=lambda item: (item.resource.name, item.resource.id)
                    )
                )
            if attempt == 1:
                break
        raise CatalogStoreError(_GENERATION_CHANGED)

    async def search(
        self,
        request: CatalogSearchRequest,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> CatalogSearchResult:
        if not isinstance(request, CatalogSearchRequest):
            raise TypeError("request must be a CatalogSearchRequest record")
        for attempt in range(2):
            active_source_ids = await self._active_source_ids(request.agent_id)
            syncs = await self._current_syncs(request.agent_id, active_source_ids)
            scoped_source_ids = tuple(
                sync.source_id
                for sync in syncs
                if not request.source_ids or sync.source_id in request.source_ids
            )
            reader = _CatalogReader(
                self._store,
                request.agent_id,
                scoped_source_ids,
                readable_resource_ids,
            )
            try:
                result = await self._search_current(
                    request,
                    reader,
                    scoped_source_ids,
                    readable_resource_ids,
                    source_hints=await self._source_hints(
                        request.agent_id,
                        frozenset(scoped_source_ids),
                    ),
                    link_graph=await self._current_link_graph(
                        reader,
                        request.agent_id,
                        scoped_source_ids,
                        readable_resource_ids,
                    ),
                )
            except _CatalogGenerationChanged:
                if attempt == 0:
                    continue
                break
            if await self._same_generation(request.agent_id, active_source_ids, syncs):
                return result
            if attempt == 1:
                break
        raise CatalogStoreError(_GENERATION_CHANGED)

    async def _same_generation(
        self,
        agent_id: str,
        active_source_ids: tuple[str, ...],
        syncs: tuple[CatalogSync, ...],
    ) -> bool:
        current_active = await self._active_source_ids(agent_id)
        return (
            current_active == active_source_ids
            and await self._current_syncs(agent_id, current_active) == syncs
        )

    async def _search_current(
        self,
        request: CatalogSearchRequest,
        reader: _CatalogReader,
        source_ids: tuple[str, ...],
        readable_resource_ids: frozenset[str] | None,
        *,
        include_fallback: bool = True,
        source_hints: tuple[Mapping[str, object], ...] = (),
        link_graph: _LinkGraph = _NO_LINKS,
    ) -> CatalogSearchResult:
        selection, known = await self._select_hits(
            request,
            reader,
            source_ids,
            readable_resource_ids,
            include_fallback=include_fallback,
            source_hints=source_hints,
            link_graph=link_graph,
        )
        matched_hits = selection.ordered_hits
        matched_ids = {hit.resource_id for hit in matched_hits}
        # Unmatched resources stay light until a page needs them as hits.
        fallback = (
            sorted(
                (
                    found
                    for found in (known or {}).values()
                    if found.id not in matched_ids
                    and _allows(readable_resource_ids, found.id, found.source_id)
                    and (
                        not request.resource_kinds
                        or found.kind in request.resource_kinds
                    )
                ),
                key=_resource_search_tie_key,
            )
            if include_fallback
            else []
        )
        total_candidates = len(matched_hits) + len(fallback)
        context = (
            canonical_json(
                {
                    "agent_id": request.agent_id,
                    "run_id": request.run_id,
                    "query": request.query,
                    "source_ids": request.source_ids,
                    "resource_kinds": tuple(
                        kind.value for kind in request.resource_kinds
                    ),
                    "source_hints": tuple(
                        sorted(source_hints, key=lambda hint: str(hint["source_id"]))
                    ),
                }
            )
            + json.dumps(
                [
                    (hit.resource_id, hit.revision, hit.score, hit.match_reasons)
                    for hit in matched_hits
                ]
                + [
                    (found.id, found.revision, 0, ("unmatched_fallback",))
                    for found in fallback
                ],
                separators=(",", ":"),
            )
        ).encode("utf-8")

        def cursor_for(position: int) -> str:
            return (
                f"{position}."
                + hmac.new(
                    self._search_cursor_key, context + f":{position}".encode(), "sha256"
                ).hexdigest()
            )

        def _page(start: int) -> tuple[CatalogSearchHit, ...]:
            ranked = matched_hits[start : start + request.limit]
            first = max(0, start - len(matched_hits))
            room = request.limit - len(ranked)
            return (
                *ranked,
                *(
                    CatalogSearchHit(
                        resource_id=found.id,
                        source_id=found.source_id,
                        kind=found.kind,
                        name=found.name,
                        revision=found.revision,
                        sensitivity=found.sensitivity,
                        score=0,
                        match_reasons=("unmatched_fallback",),
                    )
                    for found in fallback[first : first + room]
                ),
            )

        position = 0
        if request.cursor is not None:
            if re.fullmatch(r"[1-9][0-9]{0,5}\.[0-9a-f]{64}", request.cursor) is None:
                raise CatalogStoreError("catalog search cursor is invalid or stale")
            position = int(request.cursor.partition(".")[0])
            if position >= total_candidates or not hmac.compare_digest(
                request.cursor, cursor_for(position)
            ):
                raise CatalogStoreError("catalog search cursor is invalid or stale")
        page = _page(position)
        # Only the returned page needs full entries, for sensitivity.
        await reader.load(hit.resource_id for hit in page)
        hits: list[CatalogSearchHit] = []
        for hit in page:
            entry = reader.entries.get(hit.resource_id)
            if entry is None or entry.resource.current_revision != hit.revision:
                raise _CatalogGenerationChanged
            hits.append(replace(hit, sensitivity=entry.resource.sensitivity))
        truncated = position + len(hits) < total_candidates
        return CatalogSearchResult(
            request=request,
            hits=tuple(hits),
            total_matches=len(matched_hits),
            returned_count=len(hits),
            truncated=truncated,
            total_candidates=total_candidates,
            position=position,
            next_cursor=cursor_for(position + len(hits)) if truncated else None,
        )

    async def _select_hits(
        self,
        request: CatalogSearchRequest,
        reader: _CatalogReader,
        source_ids: tuple[str, ...],
        readable_resource_ids: frozenset[str] | None,
        *,
        include_fallback: bool,
        source_hints: tuple[Mapping[str, object], ...],
        link_graph: _LinkGraph,
    ) -> tuple[_CatalogHitSelection, dict[str, _Found] | None]:
        """Select hits, plus every resource of the scope when fallback needs it."""

        normalized_query = _normalize_search_text(request.query)
        significant_terms = tuple(
            token
            for token in normalized_query.tokens
            if token not in _SEARCH_STOP_WORDS
        )
        resource_kinds = set(request.resource_kinds)

        def admitted(found: _Found) -> bool:
            return (_allows(readable_resource_ids, found.id, found.source_id)) and (
                not resource_kinds or found.kind in resource_kinds
            )

        # An empty id tuple means every source to the store, never "none".
        known: dict[str, _Found] | None = None
        if include_fallback or not normalized_query.complete:
            known = (
                {
                    row.resource_id: _found_from_row(row)
                    for row in await self._store.search_catalog(
                        request.agent_id, source_ids, terms=None
                    )
                }
                if source_ids
                else {}
            )

        if not normalized_query.complete:
            assert known is not None
            candidates = tuple(
                _inventory_candidate(found)
                for found in sorted(
                    (found for found in known.values() if admitted(found)),
                    key=_resource_search_tie_key,
                )
            )
            exact_anchors: tuple[_RankedCandidate, ...] = ()
            broad_candidates = candidates
        else:
            anchor_keys = _anchor_keys(normalized_query.complete)
            rows = (
                await self._store.search_catalog(
                    request.agent_id,
                    source_ids,
                    terms=significant_terms,
                    anchor_keys=anchor_keys,
                )
                if source_ids and (significant_terms or anchor_keys)
                else ()
            )
            postings_by_candidate: dict[str, set[_IndexedPosting]] = {}
            anchor_postings_by_candidate: dict[str, set[_IndexedPosting]] = {}
            candidate_resources: dict[str, _Found] = {}
            for row in rows:
                found = _found_from_row(row)
                if not admitted(found):
                    continue
                anchors, matching = _row_matches(
                    row, normalized_query.complete, significant_terms
                )
                if not anchors and not matching:
                    continue
                candidate_resources[found.id] = found
                postings_by_candidate[found.id] = anchors | matching
                if anchors:
                    anchor_postings_by_candidate[found.id] = anchors
            ranked_candidates = tuple(
                _rank_index_candidate(
                    candidate_resources[resource_id],
                    tuple(sorted(postings, key=_posting_sort_key)),
                    normalized_query,
                    significant_terms,
                )
                for resource_id, postings in sorted(postings_by_candidate.items())
            )
            anchor_resource_ids = frozenset(anchor_postings_by_candidate)
            exact_anchors = tuple(
                sorted(
                    (
                        _embedded_exact_anchor_candidate(
                            candidate,
                            tuple(
                                sorted(
                                    anchor_postings_by_candidate[
                                        candidate.hit.resource_id
                                    ],
                                    key=_posting_sort_key,
                                )
                            ),
                            whole_query_exact=(
                                any(
                                    posting.normalized_value
                                    == normalized_query.complete
                                    for posting in anchor_postings_by_candidate[
                                        candidate.hit.resource_id
                                    ]
                                )
                            ),
                        )
                        for candidate in ranked_candidates
                        if candidate.hit.resource_id in anchor_resource_ids
                    ),
                    key=lambda candidate: candidate.rank_key,
                )
            )
            broad_candidates = _diversify_candidates(
                tuple(
                    candidate
                    for candidate in ranked_candidates
                    if candidate.hit.resource_id not in anchor_resource_ids
                ),
                significant_terms,
            )

        candidates = (*exact_anchors, *broad_candidates)

        direct_ids = {candidate.hit.resource_id for candidate in candidates}
        neighbor_by_id: dict[
            str,
            tuple[tuple[int, str, str, str], CatalogSearchHit],
        ] = {}
        # With every resource of the scope in hand, neighbours that no filter
        # admits can be ruled out before any relationship lookup.
        may_have_neighbors = known is None or any(
            found.id not in direct_ids and admitted(found) for found in known.values()
        )
        if may_have_neighbors:
            await reader.prefetch(
                (candidate.hit.resource_id for candidate in candidates),
                endpoints=False,
            )
            if known is None:
                neighbor_ids = [
                    end
                    for candidate in candidates
                    for relationship in await reader.incident(candidate.hit.resource_id)
                    for end in (
                        relationship.from_resource_id,
                        relationship.to_resource_id,
                    )
                ]
                neighbor_ids.extend(
                    edge.neighbor_resource_id
                    for candidate in candidates
                    for edge in link_graph.adjacency_by_resource_id.get(
                        candidate.hit.resource_id, ()
                    )
                )
                await reader.load(
                    neighbor for neighbor in neighbor_ids if neighbor not in direct_ids
                )
        for direct_position, candidate in enumerate(
            candidates if may_have_neighbors else ()
        ):
            resource_id = candidate.hit.resource_id
            edges: list[tuple[_IndexedEdge, bool]] = []
            for relationship in await reader.incident(resource_id):
                if relationship.from_resource_id == resource_id:
                    edges.append(
                        (
                            _IndexedEdge(
                                relationship.to_resource_id,
                                relationship.id,
                                RelationshipDirection.FORWARD,
                            ),
                            False,
                        )
                    )
                if relationship.to_resource_id == resource_id:
                    edges.append(
                        (
                            _IndexedEdge(
                                relationship.from_resource_id,
                                relationship.id,
                                RelationshipDirection.REVERSE,
                            ),
                            False,
                        )
                    )
            edges.extend(
                (edge, True)
                for edge in link_graph.adjacency_by_resource_id.get(resource_id, ())
            )
            if known is None:
                await reader.load(
                    edge.neighbor_resource_id
                    for edge, _ in edges
                    if edge.neighbor_resource_id not in direct_ids
                )
            for edge, _ in edges:
                neighbor = (
                    known.get(edge.neighbor_resource_id)
                    if known is not None
                    else (
                        _found_from_resource(
                            reader.entries[edge.neighbor_resource_id].resource
                        )
                        if edge.neighbor_resource_id in reader.entries
                        else None
                    )
                )
                if (
                    neighbor is None
                    or not admitted(neighbor)
                    or neighbor.id in direct_ids
                ):
                    continue
                order = (
                    direct_position,
                    edge.relationship_id,
                    _normalize_search_text(neighbor.native_identity).complete,
                    neighbor.id,
                )
                hit = CatalogSearchHit(
                    resource_id=neighbor.id,
                    source_id=neighbor.source_id,
                    kind=neighbor.kind,
                    name=neighbor.name,
                    revision=neighbor.revision,
                    sensitivity=neighbor.sensitivity,
                    score=0.0,
                    matched_fields=(f"relationship:{edge.relationship_id}",),
                    match_reasons=(
                        ("relationship_neighbor",)
                        if edge.relationship_id not in link_graph.links_by_id
                        else (
                            "relationship_neighbor",
                            "declared_link",
                            *(
                                ("cross_source",)
                                if neighbor.source_id != candidate.hit.source_id
                                else ()
                            ),
                        )
                    ),
                )
                existing = neighbor_by_id.get(neighbor.id)
                if existing is None or order < existing[0]:
                    neighbor_by_id[neighbor.id] = (order, hit)

        exact_hits = tuple(candidate.hit for candidate in exact_anchors)
        broad_hits = tuple(candidate.hit for candidate in broad_candidates)
        direct_ids = {hit.resource_id for hit in (*exact_hits, *broad_hits)}
        query_terms = (
            set(_normalize_search_text(request.query).tokens) - _SEARCH_STOP_WORDS
        )
        hinted_sources = {
            hint["source_id"]
            for hint in source_hints
            if query_terms
            & set(
                _normalize_search_text(
                    canonical_json(
                        {
                            key: value
                            for key, value in hint.items()
                            if key != "source_id"
                        }
                    )
                ).tokens
            )
        }
        hinted_hits = tuple(
            CatalogSearchHit(
                resource_id=found.id,
                source_id=found.source_id,
                kind=found.kind,
                name=found.name,
                revision=found.revision,
                sensitivity=found.sensitivity,
                score=1,
                match_reasons=("source_hint",),
            )
            for found in sorted(
                (
                    found
                    for found in (known or {}).values()
                    if found.source_id in hinted_sources
                    and found.id not in direct_ids
                    and admitted(found)
                ),
                key=_resource_search_tie_key,
            )
        )
        hinted_ids = {hit.resource_id for hit in hinted_hits}
        neighbors = tuple(
            hit
            for _, hit in sorted(neighbor_by_id.values(), key=lambda item: item[0])
            if hit.resource_id not in hinted_ids
        )
        return (
            _CatalogHitSelection(
                exact_anchors=exact_hits,
                broad_matches=broad_hits,
                source_hint_matches=hinted_hits,
                relationship_neighbors=neighbors,
            ),
            known,
        )

    async def _source_hints(
        self,
        agent_id: str,
        source_ids: frozenset[str],
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "source_id": source.id,
                "label": source.display_name,
                "summary": source.summary,
                "when_to_use": source.when_to_use,
                "keywords": source.keywords,
            }
            for source in await self._sources.list_sources(agent_id)
            if source.agent_id == agent_id and source.active and source.id in source_ids
        )

    async def schema_slice(
        self,
        request: CatalogSchemaRequest,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> FrozenJsonObject:
        """Project bounded current structural truth without source connector I/O."""

        if not isinstance(request, CatalogSchemaRequest):
            raise TypeError("request must be a CatalogSchemaRequest record")
        for attempt in range(2):
            active_source_ids = await self._active_source_ids(request.agent_id)
            if request.source_id is not None and request.source_id not in (
                active_source_ids
            ):
                raise CatalogStoreError(
                    "unknown active catalog source for "
                    f"{request.agent_id}: {request.source_id}"
                )
            syncs = await self._current_syncs(request.agent_id, active_source_ids)
            try:
                projection = await self._schema_slice_current(
                    request,
                    syncs,
                    readable_resource_ids=readable_resource_ids,
                )
            except _CatalogGenerationChanged:
                if attempt == 0:
                    continue
                break
            if await self._same_generation(request.agent_id, active_source_ids, syncs):
                return projection
            if attempt == 1:
                break
        raise CatalogStoreError(_GENERATION_CHANGED)

    async def _schema_slice_current(
        self,
        request: CatalogSchemaRequest,
        syncs: tuple[CatalogSync, ...],
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> FrozenJsonObject:
        indexed_source_ids = tuple(sync.source_id for sync in syncs)
        sync_by_source_id = {sync.source_id: sync for sync in syncs}
        narrowed_source_ids = tuple(
            source_id
            for source_id in indexed_source_ids
            if request.source_id in (None, source_id)
        )
        reader = _CatalogReader(
            self._store,
            request.agent_id,
            indexed_source_ids,
            readable_resource_ids,
        )
        link_graph = await self._current_link_graph(
            reader, request.agent_id, narrowed_source_ids, readable_resource_ids
        )
        # A source_id narrows which resources are described, not which current
        # declared links touch them: links leaving that source are still listed,
        # with their far endpoint unselected.
        listed_links = (
            link_graph
            if request.source_id is None
            else await self._current_link_graph(
                reader, request.agent_id, indexed_source_ids, readable_resource_ids
            )
        )

        match_by_resource_id: dict[str, CatalogSearchHit] = {}
        normalized_query = _normalize_search_text(request.query or "")
        significant_terms = tuple(
            token
            for token in normalized_query.tokens
            if token not in _SEARCH_STOP_WORDS
        )
        if request.query is not None:
            search_request = CatalogSearchRequest(
                agent_id=request.agent_id,
                query=request.query,
                source_ids=(() if request.source_id is None else (request.source_id,)),
                resource_kinds=(
                    ResourceKind.TABLE,
                    ResourceKind.VIEW,
                    ResourceKind.FILE,
                ),
                limit=CATALOG_MAX_LIMIT,
            )
            search = await self._search_current(
                search_request,
                reader,
                narrowed_source_ids,
                readable_resource_ids,
                include_fallback=False,
            )
            match_by_resource_id = {hit.resource_id: hit for hit in search.hits}
        else:
            search = None

        if request.resource_ids:
            await reader.load(request.resource_ids)
            selected_resources_by_id: dict[str, CatalogResource] = {}
            for resource_id in request.resource_ids:
                candidate_entry = reader.entries.get(resource_id)
                if candidate_entry is None:
                    raise CatalogResourceNotFoundError(request.agent_id, resource_id)
                candidate_resource = candidate_entry.resource
                if (
                    request.source_id is not None
                    and candidate_resource.source_id != request.source_id
                ):
                    raise CatalogStoreError(
                        "catalog schema resource is outside requested source scope"
                    )
                selected_resources_by_id[candidate_resource.id] = candidate_resource
            candidates = tuple(
                sorted(
                    selected_resources_by_id.values(),
                    key=lambda item: (
                        item.native_identity.casefold(),
                        item.native_identity,
                        item.id,
                    ),
                )
            )
            total_matches = len(candidates)
        else:
            assert search is not None
            candidates_list: list[CatalogResource] = []
            for hit in search.hits:
                if hit.match_reasons == ("relationship_neighbor",):
                    continue
                hit_entry = reader.entries.get(hit.resource_id)
                if (
                    hit_entry is None
                    or hit_entry.resource.current_revision != hit.revision
                    or hit_entry.resource.source_id != hit.source_id
                ):
                    raise _CatalogGenerationChanged
                hit_resource = hit_entry.resource
                if (
                    request.source_id is not None
                    and hit_resource.source_id != request.source_id
                ):
                    raise CatalogStoreError(
                        "catalog schema search escaped requested source scope"
                    )
                if not _is_tabular_entry(hit_entry):
                    continue
                candidates_list.append(hit_resource)
            candidates = tuple(candidates_list)
            total_matches = search.total_matches

        if request.resource_ids:
            seed_limit = request.limit
        elif request.limit == 1:
            seed_limit = 1
        else:
            # Reserve room for the bridges of typical joins only. A deeper
            # traversal bound must not crowd out the tables the query names.
            bridge_reserve = max(
                0, min(request.max_join_depth, CATALOG_SCHEMA_DEFAULT_JOIN_DEPTH) - 1
            )
            seed_limit = min(request.limit, max(2, request.limit - bridge_reserve))
        seeds = candidates[:seed_limit]
        seed_ids = {resource.id for resource in seeds}
        seed_rank = {resource.id: rank for rank, resource in enumerate(seeds)}
        await reader.prefetch(seed_ids)
        matched_terms_by_resource_id = {
            resource.id: await self._schema_matched_terms(
                reader,
                resource.id,
                significant_terms,
            )
            for resource in seeds
        }
        covered_terms = tuple(
            term
            for term in significant_terms
            if any(
                term in matched_terms
                for matched_terms in matched_terms_by_resource_id.values()
            )
        )
        unresolved_terms = tuple(
            term for term in significant_terms if term not in covered_terms
        )

        join_selection = _SchemaJoinSelection((), (), (), (), None, 0, 0, {})
        if request.include_relationships and len(seeds) >= 2:
            join_selection = await self._select_schema_joins(
                request,
                seeds,
                reader,
                indexed_source_ids,
                link_graph=link_graph,
            )

        included_ids = set(seed_ids)
        included_ids.update(join_selection.bridge_resource_ids)
        optional_neighbors_truncated = False
        if request.include_relationships and len(seeds) == 1:
            seed = seeds[0]
            neighbor_candidates = tuple(
                edge.neighbor_resource_id
                for edge in (
                    *await reader.edges(seed.id),
                    *link_graph.adjacency_by_resource_id.get(seed.id, ()),
                )
                if (
                    edge.relationship_id in link_graph.links_by_id
                    or _is_sql_joinable(
                        reader, reader.relationships[edge.relationship_id]
                    )
                )
                and edge.neighbor_resource_id in reader.entries
            )
            for neighbor_id in neighbor_candidates:
                if neighbor_id in included_ids:
                    continue
                if len(included_ids) >= request.limit:
                    optional_neighbors_truncated = True
                    break
                included_ids.add(neighbor_id)

        selected = tuple(
            sorted(
                (reader.entries[resource_id].resource for resource_id in included_ids),
                key=lambda item: (
                    _normalize_search_text(item.native_identity).complete,
                    item.native_identity,
                    item.id,
                ),
            )
        )
        selected_resources = {resource.id: resource for resource in selected}
        selected_syncs_by_id: dict[str, CatalogSync] = {}
        resource_payloads: list[dict[str, object]] = []
        columns_truncated = False
        primary_keys_truncated = False
        unique_keys_truncated = False
        structural_facts_truncated = False
        for resource in selected:
            resource_sync = sync_by_source_id.get(resource.source_id)
            if (
                resource_sync is None
                or resource_sync.agent_id != request.agent_id
                or resource_sync.status is not CatalogSyncStatus.SUCCEEDED
            ):
                raise CatalogStoreError("catalog resource sync is not current")
            selected_syncs_by_id[resource_sync.id] = resource_sync
            (
                structural,
                resource_columns_truncated,
                resource_primary_keys_truncated,
                resource_unique_keys_truncated,
            ) = self._schema_resource_structure(reader.entries[resource.id].facets)
            match_hit = match_by_resource_id.get(resource.id)
            matched_fields = () if match_hit is None else match_hit.matched_fields
            match_reasons = () if match_hit is None else match_hit.match_reasons
            if resource.id in seed_ids:
                selection_role = "seed"
            elif resource.id in join_selection.bridge_resource_ids:
                selection_role = "bridge"
            else:
                selection_role = "neighbor"
            resource_structural_facts_truncated = (
                len(matched_fields) > _SCHEMA_STRUCTURAL_FACT_LIMIT
                or len(match_reasons) > _SCHEMA_STRUCTURAL_FACT_LIMIT
            )
            resource_payloads.append(
                {
                    "columns": structural["columns"],
                    "kind": resource.kind.value,
                    "matched_terms": matched_terms_by_resource_id.get(resource.id, ()),
                    "name": resource.native_identity,
                    "primary_key_fields": structural["primary_key_fields"],
                    "resource_id": resource.id,
                    "revision": resource.current_revision,
                    "selection_role": selection_role,
                    "source_id": resource.source_id,
                    "structural_facts": {
                        "match_reasons": match_reasons[:_SCHEMA_STRUCTURAL_FACT_LIMIT],
                        "matched_fields": matched_fields[
                            :_SCHEMA_STRUCTURAL_FACT_LIMIT
                        ],
                    },
                    "sync_id": resource_sync.id,
                    "unique_key_fields": structural["unique_key_fields"],
                }
            )
            columns_truncated = columns_truncated or resource_columns_truncated
            primary_keys_truncated = (
                primary_keys_truncated or resource_primary_keys_truncated
            )
            unique_keys_truncated = (
                unique_keys_truncated or resource_unique_keys_truncated
            )
            structural_facts_truncated = (
                structural_facts_truncated or resource_structural_facts_truncated
            )

        relationship_payloads: list[dict[str, object]] = []
        relationships_truncated = (
            join_selection.truncation_reason == "relationship_limit"
        )
        if request.include_relationships:
            selected_relationship_ids = set(join_selection.relationship_ids)
            await reader.prefetch(resource.id for resource in selected)
            for resource in selected:
                selected_relationship_ids.update(
                    edge.relationship_id
                    for edge in (
                        *await reader.edges(resource.id),
                        *listed_links.adjacency_by_resource_id.get(resource.id, ()),
                    )
                    if (
                        len(seeds) == 1
                        or edge.neighbor_resource_id in selected_resources
                        or (
                            resource.id in seed_ids
                            and edge.relationship_id in listed_links.links_by_id
                            and edge.relationship_id not in link_graph.links_by_id
                        )
                    )
                )
            # Preserve every selected join path before optional adjacent edges.
            # A compact tool projection must not point at an omitted relationship.
            required_ids = set(join_selection.relationship_ids)
            ordered_ids = (
                *sorted(required_ids),
                *sorted(selected_relationship_ids - required_ids),
            )
            relationship_limit = max(
                len(required_ids),
                min(request.relationship_limit, _SCHEMA_RELATIONSHIP_LIMIT),
            )
            if len(ordered_ids) > relationship_limit:
                relationships_truncated = True
            selected_ids = set(selected_resources)
            for relationship_id in ordered_ids[:relationship_limit]:
                link = listed_links.links_by_id.get(relationship_id)
                edge: CatalogRelationship | CatalogLink
                if link is None:
                    edge = reader.relationships[relationship_id]
                    from_resource, to_resource = _relationship_endpoints(
                        edge, reader.entries
                    )
                else:
                    edge = link
                    from_resource = reader.entries[link.from_resource_id].resource
                    to_resource = reader.entries[link.to_resource_id].resource
                unselected_endpoints = tuple(
                    _schema_endpoint_payload(endpoint)
                    for endpoint in (from_resource, to_resource)
                    if endpoint.id not in selected_ids
                )
                relationship_payloads.append(
                    {
                        "field_pairs": tuple(
                            {
                                "source_field": pair.source_field,
                                "target_field": pair.target_field,
                            }
                            for pair in edge.field_pairs
                        ),
                        "from_resource_id": edge.from_resource_id,
                        "from_resource_revision": from_resource.current_revision,
                        "kind": edge.kind.value,
                        "provenance": edge.provenance.value,
                        "relationship_id": edge.id,
                        "to_resource_id": edge.to_resource_id,
                        "to_resource_revision": to_resource.current_revision,
                        "unselected_endpoints": unselected_endpoints,
                        **(
                            {}
                            if link is None
                            else {
                                "cross_source": from_resource.source_id
                                != to_resource.source_id
                            }
                        ),
                    }
                )

        unresolved_reasons = set(join_selection.unresolved_reasons)
        candidate_resources_truncated = len(candidates) > len(seeds)
        resources_truncated = (
            candidate_resources_truncated
            or optional_neighbors_truncated
            or join_selection.truncation_reason == "resource_limit"
        )
        if candidate_resources_truncated:
            unresolved_reasons.add("resource_limit")
        if relationships_truncated:
            unresolved_reasons.add("relationship_limit")
        truncation_reason = join_selection.truncation_reason
        if truncation_reason is None and relationships_truncated:
            truncation_reason = "relationship_limit"
        elif truncation_reason is None and (
            candidate_resources_truncated or optional_neighbors_truncated
        ):
            truncation_reason = "resource_limit"
        ordered_unresolved_reasons = _ordered_schema_unresolved_reasons(
            unresolved_reasons
        )

        path_payloads = []
        for path in join_selection.paths:
            path_seed_ids = tuple(
                sorted(
                    (
                        resource_id
                        for resource_id in path.resource_ids
                        if resource_id in seed_ids
                    ),
                    key=lambda resource_id: seed_rank[resource_id],
                )
            )
            if len(path_seed_ids) == 1:
                groups = join_selection.source_groups
                terminal_source_id = reader.entries[path_seed_ids[0]].resource.source_id
                terminal_group = groups.get(terminal_source_id, terminal_source_id)
                root_seed_id = next(
                    (
                        resource.id
                        for resource in seeds
                        if groups.get(resource.source_id, resource.source_id)
                        == terminal_group
                        and resource.id != path_seed_ids[0]
                    ),
                    None,
                )
                if root_seed_id is not None:
                    path_seed_ids = tuple(
                        sorted(
                            (root_seed_id, path_seed_ids[0]),
                            key=lambda resource_id: seed_rank[resource_id],
                        )
                    )
            path_payloads.append(
                {
                    "relationship_ids": tuple(
                        step.relationship_id for step in path.steps
                    ),
                    "resource_ids": path.resource_ids,
                    "seed_resource_ids": path_seed_ids,
                }
            )

        source_payloads = tuple(
            {
                "source_id": sync.source_id,
                "source_revision": sync.source_revision,
                "sync_id": sync.id,
            }
            for sync in sorted(
                selected_syncs_by_id.values(),
                key=lambda item: (item.source_id, item.id),
            )
        )
        return FrozenJsonObject.from_mapping(
            {
                "bounds": {
                    "columns_per_resource": _SCHEMA_COLUMN_LIMIT,
                    "primary_key_fields_per_resource": _SCHEMA_KEY_LIMIT,
                    "relationships": (
                        relationship_limit
                        if request.include_relationships
                        else request.relationship_limit
                    ),
                    "resources": request.limit,
                    "join_depth": request.max_join_depth,
                    "join_graph_edges": _SCHEMA_JOIN_MAX_EDGES,
                    "join_graph_nodes": _SCHEMA_JOIN_MAX_NODES,
                    "structural_facts_per_resource": _SCHEMA_STRUCTURAL_FACT_LIMIT,
                    "unique_key_fields_per_resource": _SCHEMA_KEY_LIMIT,
                },
                "include_relationships": request.include_relationships,
                "paths": path_payloads,
                "relationships": relationship_payloads,
                "resources": resource_payloads,
                "selection": {
                    "bridge_resource_ids": tuple(join_selection.bridge_resource_ids),
                    "covered_terms": covered_terms,
                    "seed_resource_ids": tuple(resource.id for resource in seeds),
                    "unresolved_reasons": ordered_unresolved_reasons,
                    "unresolved_terms": unresolved_terms,
                },
                "sources": source_payloads,
                "total_matches": total_matches,
                "truncation": {
                    "columns": columns_truncated,
                    "paths": join_selection.truncation_reason is not None,
                    "primary_key_fields": primary_keys_truncated,
                    "reason": truncation_reason,
                    "relationships": relationships_truncated,
                    "resources": resources_truncated,
                    "structural_facts": structural_facts_truncated,
                    "unique_key_fields": unique_keys_truncated,
                },
                "trust_classification": "untrusted_external_data",
            }
        )

    async def _schema_matched_terms(
        self,
        reader: _CatalogReader,
        resource_id: str,
        significant_terms: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not significant_terms:
            return ()
        postings = search_postings(
            reader.entries[resource_id],
            await reader.incident(resource_id),
        )
        tokens = tuple(
            token
            for posting in postings
            for token in _normalize_search_text(posting.value).tokens
        )
        return tuple(
            term
            for term in significant_terms
            if any(token.startswith(term) for token in tokens)
        )

    async def _select_schema_joins(
        self,
        request: CatalogSchemaRequest,
        seeds: tuple[CatalogResource, ...],
        reader: _CatalogReader,
        indexed_source_ids: tuple[str, ...],
        *,
        link_graph: _LinkGraph = _NO_LINKS,
    ) -> _SchemaJoinSelection:
        # Sources joined by a current link form one join group; seeds in
        # different groups have no path between them.
        source_groups = {source_id: source_id for source_id in indexed_source_ids}

        def group_of(source_id: str) -> str:
            while source_groups[source_id] != source_id:
                source_id = source_groups[source_id]
            return source_id

        for link in link_graph.links_by_id.values():
            first, second = sorted(
                (group_of(link.from_source_id), group_of(link.to_source_id))
            )
            source_groups[second] = first
        source_groups = {source_id: group_of(source_id) for source_id in source_groups}
        groups: dict[str, list[CatalogResource]] = {}
        for seed in seeds:
            groups.setdefault(source_groups[seed.source_id], []).append(seed)

        unresolved_reasons: set[str] = set()
        if len(groups) > 1:
            unresolved_reasons.add("cross_source_unsupported")
        selected_paths: list[CatalogPath] = []
        bridge_ids: set[str] = set()
        selected_relationship_ids: set[str] = set()
        visited_nodes = 0
        visited_edges = 0
        truncation_reason: str | None = None
        seed_ids = {seed.id for seed in seeds}
        seed_rank = {seed.id: rank for rank, seed in enumerate(seeds)}

        for group_id in sorted(groups):
            source_seeds = tuple(groups[group_id])
            joinable_seeds = tuple(
                seed
                for seed in source_seeds
                if _is_tabular_entry(reader.entries[seed.id])
            )
            if len(joinable_seeds) != len(source_seeds):
                unresolved_reasons.add("non_tabular_seed")
            if len(joinable_seeds) < 2:
                continue

            group_link_ids = frozenset(
                link.id
                for link in link_graph.links_by_id.values()
                if source_groups[link.from_source_id] == group_id
            )

            def eligible(
                relationship: CatalogRelationship | CatalogLink,
                group_link_ids: frozenset[str] = group_link_ids,
            ) -> bool:
                if isinstance(relationship, CatalogLink):
                    return relationship.id in group_link_ids
                return _is_sql_joinable(reader, relationship)

            tree_resource_ids = {joinable_seeds[0].id}
            remaining = list(joinable_seeds[1:])
            while remaining:
                candidates: list[
                    tuple[
                        tuple[object, ...],
                        CatalogPath,
                        CatalogResource,
                    ]
                ] = []
                saw_depth = False
                work_stopped = False
                ordered_tree_ids = tuple(
                    sorted(
                        tree_resource_ids,
                        key=lambda resource_id: _resource_join_key(
                            reader.entries[resource_id].resource
                        ),
                    )
                )
                for tree_resource_id in ordered_tree_ids:
                    for terminal in remaining:
                        remaining_nodes = _SCHEMA_JOIN_MAX_NODES - visited_nodes
                        remaining_edges = _SCHEMA_JOIN_MAX_EDGES - visited_edges
                        if remaining_nodes < 2:
                            unresolved_reasons.add("graph_node_limit")
                            truncation_reason = "graph_node_limit"
                            work_stopped = True
                            break
                        if remaining_edges < 1:
                            unresolved_reasons.add("graph_edge_limit")
                            truncation_reason = "graph_edge_limit"
                            work_stopped = True
                            break
                        result = await self._traverse_graph(
                            CatalogTraversalRequest(
                                agent_id=request.agent_id,
                                from_resource_ids=(tree_resource_id,),
                                to_resource_ids=(terminal.id,),
                                relationship_kinds=(RelationshipKind.REFERENCES,),
                                max_depth=request.max_join_depth,
                                max_paths=CATALOG_TRAVERSAL_MAX_PATHS,
                                max_nodes=remaining_nodes,
                                max_edges=remaining_edges,
                            ),
                            reader,
                            link_graph=link_graph,
                            eligible=eligible,
                            stop_at_shortest_target=True,
                        )
                        visited_nodes += result.visited_nodes
                        visited_edges += result.visited_edges
                        if result.truncation_reason in {
                            CatalogTraversalTruncationReason.NODES,
                            CatalogTraversalTruncationReason.EDGES,
                        }:
                            reason = (
                                "graph_node_limit"
                                if result.truncation_reason
                                is CatalogTraversalTruncationReason.NODES
                                else "graph_edge_limit"
                            )
                            unresolved_reasons.add(reason)
                            truncation_reason = reason
                            work_stopped = True
                            break
                        if (
                            result.truncation_reason
                            is CatalogTraversalTruncationReason.DEPTH
                        ):
                            saw_depth = True
                        for path in result.paths:
                            new_bridges = {
                                resource_id
                                for resource_id in path.resource_ids
                                if resource_id not in seed_ids
                                and resource_id not in tree_resource_ids
                            }
                            candidates.append(
                                (
                                    (
                                        len(path.steps),
                                        len(new_bridges),
                                        tuple(
                                            step.relationship_id for step in path.steps
                                        ),
                                        tuple(
                                            _normalize_search_text(
                                                reader.entries[
                                                    resource_id
                                                ].resource.native_identity
                                            ).complete
                                            for resource_id in path.resource_ids
                                        ),
                                        seed_rank[terminal.id],
                                    ),
                                    path,
                                    terminal,
                                )
                            )
                    if work_stopped:
                        break
                if work_stopped:
                    break
                if not candidates:
                    unresolved_reasons.add("max_join_depth" if saw_depth else "no_path")
                    if saw_depth:
                        truncation_reason = "max_join_depth"
                    break

                _, chosen_path, terminal = min(candidates, key=lambda item: item[0])
                new_bridge_ids = {
                    resource_id
                    for resource_id in chosen_path.resource_ids
                    if resource_id not in seed_ids
                    and resource_id not in tree_resource_ids
                }
                new_relationship_ids = {
                    step.relationship_id for step in chosen_path.steps
                } - selected_relationship_ids
                if len(seed_ids | bridge_ids | new_bridge_ids) > request.limit:
                    unresolved_reasons.add("resource_limit")
                    truncation_reason = "resource_limit"
                    break
                if (
                    len(selected_relationship_ids | new_relationship_ids)
                    > _SCHEMA_RELATIONSHIP_LIMIT
                ):
                    unresolved_reasons.add("relationship_limit")
                    truncation_reason = "relationship_limit"
                    break
                selected_paths.append(chosen_path)
                bridge_ids.update(new_bridge_ids)
                selected_relationship_ids.update(new_relationship_ids)
                tree_resource_ids.update(chosen_path.resource_ids)
                remaining = [
                    seed for seed in remaining if seed.id not in tree_resource_ids
                ]

        return _SchemaJoinSelection(
            paths=tuple(selected_paths),
            bridge_resource_ids=tuple(
                sorted(
                    bridge_ids,
                    key=lambda resource_id: _resource_join_key(
                        reader.entries[resource_id].resource
                    ),
                )
            ),
            relationship_ids=tuple(sorted(selected_relationship_ids)),
            unresolved_reasons=_ordered_schema_unresolved_reasons(unresolved_reasons),
            truncation_reason=truncation_reason,
            visited_nodes=visited_nodes,
            visited_edges=visited_edges,
            source_groups=source_groups,
        )

    async def _active_source_ids(self, agent_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                registration.id
                for registration in await self._sources.list_sources(agent_id)
                if registration.agent_id == agent_id and registration.active
            )
        )

    async def inspect_resource(
        self,
        agent_id: str,
        resource_id: str,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> FrozenJsonObject:
        entry = await self._active_entry(agent_id, resource_id)
        resource = entry.resource
        if not _allows(readable_resource_ids, resource_id, resource.source_id):
            raise CatalogResourceNotFoundError(agent_id, resource_id)
        facets = entry.facets
        incident = (
            await self._store.load_incident_relationships(
                agent_id,
                (resource.id,),
                limit=CATALOG_TRAVERSAL_MAX_EDGES + 1,
            )
        ).get(resource.id, ())
        incident = tuple(
            relationship
            for relationship in incident
            if _allows(
                readable_resource_ids,
                relationship.from_resource_id,
                relationship.source_id,
            )
            and _allows(
                readable_resource_ids,
                relationship.to_resource_id,
                relationship.source_id,
            )
        )
        incident_truncated = len(incident) > _INSPECT_INCIDENT_RELATIONSHIP_LIMIT
        incident = incident[:_INSPECT_INCIDENT_RELATIONSHIP_LIMIT]
        resources = {resource.id: resource}
        entries = {resource.id: entry}
        relationships_payload: list[dict[str, object]] = []
        neighbor_payloads: list[dict[str, object]] = []
        seen_neighbor_ids: set[str] = set()
        for relationship in incident:
            endpoints = await self._current_relationship_endpoints(
                agent_id,
                relationship,
                entries,
            )
            resources.update((item.id, item) for item in endpoints)
            if resource.id == relationship.from_resource_id:
                direction = RelationshipDirection.FORWARD
                neighbor = endpoints[1]
            elif resource.id == relationship.to_resource_id:
                direction = RelationshipDirection.REVERSE
                neighbor = endpoints[0]
            else:
                raise CatalogStoreError(
                    "catalog incident relationship does not include the resource"
                )
            relationships_payload.append(
                _relationship_payload(
                    relationship,
                    endpoints[0],
                    endpoints[1],
                    direction=direction,
                )
            )
            if neighbor.id not in seen_neighbor_ids:
                seen_neighbor_ids.add(neighbor.id)
                neighbor_payloads.append(_neighbor_payload(neighbor))

        child_file_freshness: dict[str, dict[str, object]] = {}
        if resource.kind is ResourceKind.FOLDER:
            for relationship in incident:
                if (
                    relationship.kind is not RelationshipKind.CONTAINS
                    or relationship.from_resource_id != resource.id
                ):
                    continue
                child = resources[relationship.to_resource_id]
                if child.kind is not ResourceKind.FILE:
                    continue
                child_file_freshness[child.id] = _file_freshness(
                    entries[child.id].facets
                )

        selection_facts = _selection_facts(
            resource,
            facets,
            incident,
            resources,
            child_file_freshness,
            incident_truncated=incident_truncated,
        )
        return FrozenJsonObject.from_mapping(
            {
                "resource": _resource_payload(resource),
                "facets": [_facet_payload(facet) for facet in facets],
                "incident_relationships": relationships_payload,
                "incident_relationships_truncated": incident_truncated,
                "neighbors": neighbor_payloads,
                "selection_facts": selection_facts,
                "trust_classification": "untrusted_external_data",
            }
        )

    async def traverse(
        self,
        request: CatalogTraversalRequest,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> FrozenJsonObject:
        if not isinstance(request, CatalogTraversalRequest):
            raise TypeError("request must be a CatalogTraversalRequest record")
        for attempt in range(2):
            active_source_ids = await self._active_source_ids(request.agent_id)
            syncs = await self._current_syncs(request.agent_id, active_source_ids)
            indexed_source_ids = tuple(sync.source_id for sync in syncs)
            reader = _CatalogReader(
                self._store,
                request.agent_id,
                indexed_source_ids,
                readable_resource_ids,
            )
            link_graph = await self._current_link_graph(
                reader, request.agent_id, indexed_source_ids, readable_resource_ids
            )
            result = await self._traverse_graph(request, reader, link_graph=link_graph)
            projection = self._traversal_projection(result, reader, link_graph)
            if await self._same_generation(request.agent_id, active_source_ids, syncs):
                return projection
            if attempt == 1:
                break
        raise CatalogStoreError(_GENERATION_CHANGED)

    async def _traverse_graph(
        self,
        request: CatalogTraversalRequest,
        reader: _CatalogReader,
        *,
        link_graph: _LinkGraph = _NO_LINKS,
        eligible: Callable[[CatalogRelationship | CatalogLink], bool] | None = None,
        stop_at_shortest_target: bool = False,
    ) -> CatalogTraversalResult:
        await reader.load((*request.from_resource_ids, *request.to_resource_ids))
        for resource_id in (*request.from_resource_ids, *request.to_resource_ids):
            if resource_id not in reader.entries:
                raise CatalogResourceNotFoundError(request.agent_id, resource_id)

        def resource_key(resource_id: str) -> tuple[str, str, str]:
            resource = reader.entries[resource_id].resource
            return (
                _normalize_search_text(resource.native_identity).complete,
                resource.source_id,
                resource.id,
            )

        def relationship_of(relationship_id: str) -> CatalogRelationship | CatalogLink:
            return (
                link_graph.links_by_id.get(relationship_id)
                or reader.relationships[relationship_id]
            )

        ordered_sources = tuple(sorted(request.from_resource_ids, key=resource_key))
        target_ids = set(request.to_resource_ids)
        admitted_sources = ordered_sources[: request.max_nodes]
        distance_by_resource = {resource_id: 0 for resource_id in admitted_sources}
        parents_by_resource: dict[str, list[_ParentStep]] = {
            resource_id: [] for resource_id in admitted_sources
        }
        frontier = deque(admitted_sources)
        examined_edges = 0
        shortest_target_distance: int | None = None
        truncation_reason = (
            CatalogTraversalTruncationReason.NODES
            if len(admitted_sources) < len(ordered_sources)
            else None
        )
        allowed_kinds = set(request.relationship_kinds)

        while frontier and truncation_reason is None:
            # One round trip per level: the whole queued frontier is fetched at once.
            if not reader.fetched(frontier[0], request.relationship_kinds):
                await reader.prefetch(frontier, request.relationship_kinds)
            current_resource_id = frontier.popleft()
            current_distance = distance_by_resource[current_resource_id]
            if (
                stop_at_shortest_target
                and shortest_target_distance is not None
                and current_distance >= shortest_target_distance
            ):
                break
            if current_resource_id in target_ids and current_distance > 0:
                continue
            for edge in (
                *await reader.edges(current_resource_id, request.relationship_kinds),
                *link_graph.adjacency_by_resource_id.get(current_resource_id, ()),
            ):
                relationship = relationship_of(edge.relationship_id)
                if allowed_kinds and relationship.kind not in allowed_kinds:
                    continue
                if eligible is not None and not eligible(relationship):
                    continue
                if examined_edges >= request.max_edges:
                    truncation_reason = CatalogTraversalTruncationReason.EDGES
                    break
                examined_edges += 1
                known_distance = distance_by_resource.get(edge.neighbor_resource_id)
                if current_distance >= request.max_depth:
                    if known_distance is None:
                        truncation_reason = CatalogTraversalTruncationReason.DEPTH
                        break
                    continue

                candidate_distance = current_distance + 1
                parent = _ParentStep(
                    parent_resource_id=current_resource_id,
                    relationship_id=edge.relationship_id,
                    direction=edge.direction,
                )
                if known_distance is None:
                    if len(distance_by_resource) >= request.max_nodes:
                        truncation_reason = CatalogTraversalTruncationReason.NODES
                        break
                    distance_by_resource[edge.neighbor_resource_id] = candidate_distance
                    parents_by_resource[edge.neighbor_resource_id] = [parent]
                    frontier.append(edge.neighbor_resource_id)
                    if (
                        stop_at_shortest_target
                        and edge.neighbor_resource_id in target_ids
                        and shortest_target_distance is None
                    ):
                        shortest_target_distance = candidate_distance
                elif known_distance == candidate_distance:
                    parents = parents_by_resource[edge.neighbor_resource_id]
                    if parent not in parents:
                        parents.append(parent)

        def parent_key(parent: _ParentStep) -> tuple[object, ...]:
            relationship = relationship_of(parent.relationship_id)
            return (
                *resource_key(parent.parent_resource_id),
                relationship.kind.value,
                relationship.provenance.value,
                relationship.id,
                parent.direction.value,
            )

        ordered_parents = {
            resource_id: tuple(sorted(parents, key=parent_key))
            for resource_id, parents in parents_by_resource.items()
        }
        reachable_targets = tuple(
            sorted(
                (
                    resource_id
                    for resource_id in target_ids
                    if distance_by_resource.get(resource_id, 0) > 0
                ),
                key=resource_key,
            )
        )
        path_count_limit = request.max_paths + 1
        path_counts: dict[str, int] = {}

        def bounded_path_count(resource_id: str) -> int:
            cached = path_counts.get(resource_id)
            if cached is not None:
                return cached
            if distance_by_resource[resource_id] == 0:
                path_counts[resource_id] = 1
                return 1
            total = 0
            for parent in ordered_parents[resource_id]:
                total += bounded_path_count(parent.parent_resource_id)
                if total >= path_count_limit:
                    total = path_count_limit
                    break
            path_counts[resource_id] = total
            return total

        available_path_count = 0
        for target_id in reachable_targets:
            available_path_count += bounded_path_count(target_id)
            if available_path_count >= path_count_limit:
                available_path_count = path_count_limit
                break

        paths: list[CatalogPath] = []
        reverse_resource_ids: list[str] = []
        reverse_steps: list[CatalogPathStep] = []

        def reconstruct(resource_id: str) -> None:
            if len(paths) >= request.max_paths:
                return
            reverse_resource_ids.append(resource_id)
            if distance_by_resource[resource_id] == 0:
                paths.append(
                    CatalogPath(
                        resource_ids=tuple(reversed(reverse_resource_ids)),
                        steps=tuple(reversed(reverse_steps)),
                    )
                )
            else:
                for parent in ordered_parents[resource_id]:
                    if len(paths) >= request.max_paths:
                        break
                    reverse_steps.append(
                        CatalogPathStep(
                            relationship_id=parent.relationship_id,
                            from_resource_id=parent.parent_resource_id,
                            to_resource_id=resource_id,
                            direction=parent.direction,
                        )
                    )
                    reconstruct(parent.parent_resource_id)
                    reverse_steps.pop()
            reverse_resource_ids.pop()

        for target_id in reachable_targets:
            if len(paths) >= request.max_paths:
                break
            reconstruct(target_id)

        if truncation_reason is None and available_path_count > request.max_paths:
            truncation_reason = CatalogTraversalTruncationReason.PATHS
        return CatalogTraversalResult(
            request=request,
            paths=tuple(paths),
            reachable=bool(paths),
            visited_nodes=len(distance_by_resource),
            visited_edges=examined_edges,
            truncated=truncation_reason is not None,
            truncation_reason=truncation_reason,
        )

    def _traversal_projection(
        self,
        result: CatalogTraversalResult,
        reader: _CatalogReader,
        link_graph: _LinkGraph,
    ) -> FrozenJsonObject:
        request = result.request
        paths_payload: list[dict[str, object]] = []
        for path in result.paths:
            steps_payload: list[dict[str, object]] = []
            for step in path.steps:
                relationship: CatalogRelationship | CatalogLink | None = (
                    link_graph.links_by_id.get(step.relationship_id)
                    or reader.relationships.get(step.relationship_id)
                )
                if relationship is None:
                    raise CatalogStoreError(
                        "catalog traversal references a non-current relationship"
                    )
                endpoints = (
                    _relationship_endpoints(relationship, reader.entries)
                    if isinstance(relationship, CatalogRelationship)
                    else (
                        reader.entries[relationship.from_resource_id].resource,
                        reader.entries[relationship.to_resource_id].resource,
                    )
                )
                if step.direction is RelationshipDirection.FORWARD:
                    expected = (
                        relationship.from_resource_id,
                        relationship.to_resource_id,
                    )
                else:
                    expected = (
                        relationship.to_resource_id,
                        relationship.from_resource_id,
                    )
                if expected != (step.from_resource_id, step.to_resource_id):
                    raise CatalogStoreError(
                        "catalog traversal step disagrees with its relationship"
                    )
                payload = (
                    _relationship_payload(
                        relationship,
                        endpoints[0],
                        endpoints[1],
                        direction=step.direction,
                    )
                    if isinstance(relationship, CatalogRelationship)
                    else _link_payload(
                        relationship,
                        endpoints[0],
                        endpoints[1],
                        direction=step.direction,
                    )
                )
                payload["path_from_resource_id"] = step.from_resource_id
                payload["path_to_resource_id"] = step.to_resource_id
                steps_payload.append(payload)
            paths_payload.append(
                {
                    "resource_ids": path.resource_ids,
                    "steps": steps_payload,
                }
            )

        return FrozenJsonObject.from_mapping(
            {
                "request": {
                    "from_resource_ids": request.from_resource_ids,
                    "max_depth": request.max_depth,
                    "max_edges": request.max_edges,
                    "max_nodes": request.max_nodes,
                    "max_paths": request.max_paths,
                    "relationship_kinds": tuple(
                        kind.value for kind in request.relationship_kinds
                    ),
                    "to_resource_ids": request.to_resource_ids,
                },
                "paths": paths_payload,
                "reachable": result.reachable,
                "truncated": result.truncated,
                "truncation_reason": (
                    None
                    if result.truncation_reason is None
                    else result.truncation_reason.value
                ),
                "visited_edges": result.visited_edges,
                "visited_nodes": result.visited_nodes,
                "trust_classification": "untrusted_external_data",
            }
        )

    async def _active_entry(
        self,
        agent_id: str,
        resource_id: str,
    ) -> CatalogEntry:
        entry = await self._store.load_entry(agent_id, resource_id)
        if entry is None or entry.resource.agent_id != agent_id:
            raise CatalogResourceNotFoundError(agent_id, resource_id)
        registration = await self._sources.load_source(
            agent_id, entry.resource.source_id
        )
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != entry.resource.source_id
            or not registration.active
        ):
            raise CatalogResourceNotFoundError(agent_id, resource_id)
        return entry

    async def _active_source(self, agent_id: str, source_id: str) -> object:
        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
        ):
            raise CatalogStoreError(
                f"unknown active catalog source for {agent_id}: {source_id}"
            )
        return registration

    def _schema_resource_structure(
        self,
        facets: tuple[CatalogFacet, ...],
    ) -> tuple[dict[str, object], bool, bool, bool]:
        tabular = next(
            (facet for facet in facets if facet.kind is FacetKind.TABULAR),
            None,
        )
        if tabular is None:
            return (
                {
                    "columns": (),
                    "primary_key_fields": (),
                    "unique_key_fields": (),
                },
                False,
                False,
                False,
            )
        decoded = _decode_tabular_facet(tabular)
        columns: list[dict[str, object]] = []
        primary_fields: list[tuple[int, str]] = []
        for column in decoded.columns:
            columns.append(
                {
                    "name": column.name,
                    "nullable": column.nullable,
                    "type": column.native_type,
                }
            )
            if column.primary_key_ordinal is not None:
                primary_fields.append((column.primary_key_ordinal, column.name))
        ordered_primary = tuple(
            name for _, name in sorted(primary_fields, key=lambda item: item)
        )
        unique_keys: set[tuple[str, ...]] = set()
        for index in decoded.indexes:
            if not index.unique or index.predicate is not None:
                continue
            fields = index.columns
            if fields != ordered_primary:
                unique_keys.add(fields)
        ordered_unique_keys = tuple(sorted(unique_keys))
        return (
            {
                "columns": tuple(columns[:_SCHEMA_COLUMN_LIMIT]),
                "primary_key_fields": ordered_primary[:_SCHEMA_KEY_LIMIT],
                "unique_key_fields": ordered_unique_keys[:_SCHEMA_KEY_LIMIT],
            },
            len(columns) > _SCHEMA_COLUMN_LIMIT,
            len(ordered_primary) > _SCHEMA_KEY_LIMIT,
            len(ordered_unique_keys) > _SCHEMA_KEY_LIMIT,
        )

    async def _current_relationship_endpoints(
        self,
        agent_id: str,
        relationship: CatalogRelationship,
        entries: dict[str, CatalogEntry],
    ) -> tuple[CatalogResource, CatalogResource]:
        for resource_id in (
            relationship.from_resource_id,
            relationship.to_resource_id,
        ):
            if resource_id not in entries:
                entries[resource_id] = await self._active_entry(agent_id, resource_id)
        return _relationship_endpoints(relationship, entries)


def _resource_join_key(resource: CatalogResource) -> tuple[str, str, str]:
    return (
        _normalize_search_text(resource.native_identity).complete,
        resource.native_identity,
        resource.id,
    )


def _tabular_field_names(facets: tuple[CatalogFacet, ...]) -> frozenset[str]:
    tabular = next(
        (facet for facet in facets if facet.kind is FacetKind.TABULAR),
        None,
    )
    if tabular is None:
        return frozenset()
    return frozenset(column.name for column in _decode_tabular_facet(tabular).columns)


def _ordered_schema_unresolved_reasons(
    reasons: set[str] | tuple[str, ...],
) -> tuple[str, ...]:
    order = {
        "cross_source_unsupported": 0,
        "non_tabular_seed": 1,
        "no_path": 2,
        "max_join_depth": 3,
        "graph_node_limit": 4,
        "graph_edge_limit": 5,
        "resource_limit": 6,
        "relationship_limit": 7,
    }
    return tuple(
        sorted(set(reasons), key=lambda reason: (order.get(reason, 99), reason))
    )


def _resource_payload(resource: CatalogResource) -> dict[str, object]:
    return {
        "current_sync_id": resource.current_sync_id,
        "external_uri": resource.external_uri,
        "kind": resource.kind.value,
        "last_observed_at": resource.last_observed_at.isoformat(),
        "name": resource.name,
        "native_identity": resource.native_identity,
        "resource_id": resource.id,
        "revision": resource.current_revision,
        "sensitivity": resource.sensitivity.value,
        "source_id": resource.source_id,
    }


def _facet_payload(facet: CatalogFacet) -> dict[str, object]:
    return {
        "kind": facet.kind.value,
        "observed_at": facet.observed_at.isoformat(),
        "payload": facet.payload,
        "revision": facet.revision,
        "sync_id": facet.sync_id,
    }


def _relationship_endpoints(
    relationship: CatalogRelationship,
    entries: Mapping[str, CatalogEntry],
) -> tuple[CatalogResource, CatalogResource]:
    """Both endpoints, each current for this relationship's revision."""

    endpoints: list[CatalogResource] = []
    for resource_id in (
        relationship.from_resource_id,
        relationship.to_resource_id,
    ):
        entry = entries.get(resource_id)
        if entry is None:
            raise CatalogStoreError("catalog relationship has a non-current endpoint")
        if (
            entry.resource.source_id != relationship.source_id
            or relationship.revision not in entry.revision.relationship_revisions
        ):
            raise CatalogStoreError(
                "catalog relationship is not current for both endpoints"
            )
        endpoints.append(entry.resource)
    return endpoints[0], endpoints[1]


def _is_tabular_entry(entry: CatalogEntry) -> bool:
    return entry.resource.kind in TABULAR_KINDS and any(
        facet.kind is FacetKind.TABULAR for facet in entry.facets
    )


def _is_sql_joinable(reader: _CatalogReader, relationship: CatalogRelationship) -> bool:
    """Whether a connector reference joins two readable tables on real columns."""

    if (
        relationship.kind is not RelationshipKind.REFERENCES
        or relationship.provenance is not RelationshipProvenance.CONNECTOR
        or not relationship.field_pairs
    ):
        return False
    from_entry = reader.entries.get(relationship.from_resource_id)
    to_entry = reader.entries.get(relationship.to_resource_id)
    if (
        from_entry is None
        or to_entry is None
        or any(
            entry.resource.source_id != relationship.source_id
            or not _is_tabular_entry(entry)
            for entry in (from_entry, to_entry)
        )
    ):
        return False
    from_fields = _tabular_field_names(from_entry.facets)
    to_fields = _tabular_field_names(to_entry.facets)
    return all(
        pair.source_field in from_fields and pair.target_field in to_fields
        for pair in relationship.field_pairs
    )


def _anchor_keys(normalized_query: str) -> tuple[str, ...]:
    """Every substring a bounded identity mention in this query could be."""

    starts = [
        index
        for index in range(len(normalized_query))
        if index == 0 or not _is_identifier_character(normalized_query[index - 1])
    ]
    ends = [
        index
        for index in range(1, len(normalized_query) + 1)
        if index == len(normalized_query)
        or not _is_identifier_character(normalized_query[index])
    ]
    keys: dict[str, None] = {}
    for start in starts:
        for end in ends:
            if end <= start:
                continue
            if end - start > _ANCHOR_KEY_LENGTH:
                break
            key = normalized_query[start:end]
            if key == key.strip():
                keys[key] = None
            if len(keys) >= _ANCHOR_KEY_LIMIT:
                return tuple(keys)
    return tuple(keys)


def _row_matches(
    row: CatalogSearchRow, query: str, terms: tuple[str, ...]
) -> tuple[set[_IndexedPosting], set[_IndexedPosting]]:
    """Split a row's postings into whole-identity anchors and term matches."""

    anchors: set[_IndexedPosting] = set()
    matching: set[_IndexedPosting] = set()
    for posting in row.postings:
        normalized = _normalize_search_text(posting.value)
        anchored = posting.field_kind in {
            "resource_name",
            "native_identity",
        } and _contains_bounded_identifier(query, normalized.complete)
        matched = bool(terms) and any(
            token.startswith(terms) for token in normalized.tokens
        )
        if not anchored and not matched:
            continue
        indexed = _IndexedPosting(
            resource_id=row.resource_id,
            field_kind=posting.field_kind,
            field_name=posting.field_name,
            normalized_value=normalized.complete,
            tokens=normalized.tokens,
        )
        if anchored:
            anchors.add(indexed)
        if matched:
            matching.add(indexed)
    return anchors, matching


def _posting_sort_key(posting: _IndexedPosting) -> tuple[str, str, str, str]:
    return (
        posting.resource_id,
        posting.field_kind,
        posting.field_name.casefold(),
        posting.field_name,
    )


def _resource_search_tie_key(
    resource: _Found,
) -> tuple[str, str, str, str]:
    normalized = _normalize_search_text(resource.native_identity).complete
    return (normalized, resource.native_identity, resource.name, resource.id)


def _inventory_candidate(resource: _Found) -> _RankedCandidate:
    hit = CatalogSearchHit(
        resource_id=resource.id,
        source_id=resource.source_id,
        kind=resource.kind,
        name=resource.name,
        revision=resource.revision,
        sensitivity=resource.sensitivity,
        score=0.0,
        matched_fields=(),
        match_reasons=("metadata_contains",),
    )
    normalized = _normalize_search_text(resource.native_identity).complete
    return _RankedCandidate(
        hit=hit,
        matched_terms=(),
        rank_key=(0, 0, 0, 0, 0, 0, 0, 0, 0, normalized, resource.name, resource.id),
    )


def _embedded_exact_anchor_candidate(
    candidate: _RankedCandidate,
    anchor_postings: tuple[_IndexedPosting, ...],
    *,
    whole_query_exact: bool,
) -> _RankedCandidate:
    if whole_query_exact:
        return candidate
    exact_native = int(
        any(posting.field_kind == "native_identity" for posting in anchor_postings)
    )
    exact_name = int(
        any(posting.field_kind == "resource_name" for posting in anchor_postings)
    )
    anchor_fields = tuple(
        posting.field_name
        for posting in sorted(
            anchor_postings,
            key=lambda posting: (
                0 if posting.field_kind == "resource_name" else 1,
                posting.field_name.casefold(),
                posting.field_name,
            ),
        )
    )
    matched_fields = tuple(
        dict.fromkeys((*anchor_fields, *candidate.hit.matched_fields))
    )[:32]
    return _RankedCandidate(
        hit=CatalogSearchHit(
            resource_id=candidate.hit.resource_id,
            source_id=candidate.hit.source_id,
            kind=candidate.hit.kind,
            name=candidate.hit.name,
            revision=candidate.hit.revision,
            sensitivity=candidate.hit.sensitivity,
            score=max(
                candidate.hit.score,
                float(exact_native * 200_000 + exact_name * 180_000),
            ),
            matched_fields=matched_fields,
            match_reasons=("resource_name_exact_mention",),
        ),
        matched_terms=candidate.matched_terms,
        rank_key=(
            -exact_native,
            -exact_name,
            *candidate.rank_key[2:],
        ),
    )


_FIELD_PRIORITY = {
    "resource_name": 0,
    "native_identity": 1,
    "column": 2,
    "index_field": 3,
    "relationship_field": 4,
    "kind": 5,
}


def _rank_index_candidate(
    resource: _Found,
    postings: tuple[_IndexedPosting, ...],
    query: _NormalizedText,
    significant_terms: tuple[str, ...],
) -> _RankedCandidate:
    """Build deterministic rank evidence for one posting-derived candidate."""

    tokens = [token for posting in postings for token in posting.tokens]
    matched_terms = tuple(
        term
        for term in significant_terms
        if any(token.startswith(term) for token in tokens)
    )
    resource_postings = tuple(
        posting
        for posting in postings
        if posting.field_kind in {"resource_name", "native_identity"}
    )
    column_postings = tuple(
        posting
        for posting in postings
        if posting.field_kind in {"column", "index_field"}
    )
    relationship_postings = tuple(
        posting for posting in postings if posting.field_kind == "relationship_field"
    )
    kind_postings = tuple(
        posting for posting in postings if posting.field_kind == "kind"
    )

    def exact_terms(fields: tuple[_IndexedPosting, ...]) -> set[str]:
        return {
            term
            for term in significant_terms
            if any(term in posting.tokens for posting in fields)
        }

    def prefix_terms(fields: tuple[_IndexedPosting, ...]) -> set[str]:
        return {
            term
            for term in significant_terms
            if any(
                token.startswith(term) and token != term
                for posting in fields
                for token in posting.tokens
            )
        }

    exact_native = int(
        any(
            posting.field_kind == "native_identity"
            and posting.normalized_value == query.complete
            for posting in resource_postings
        )
    )
    exact_name = int(
        any(
            posting.field_kind == "resource_name"
            and posting.normalized_value == query.complete
            for posting in resource_postings
        )
    )
    resource_exact_terms = exact_terms(resource_postings)
    column_exact_terms = exact_terms(column_postings)
    resource_prefix_terms = prefix_terms(resource_postings)
    column_prefix_terms = prefix_terms(column_postings)
    relationship_terms = exact_terms(relationship_postings) | prefix_terms(
        relationship_postings
    )
    kind_terms = exact_terms(kind_postings) | prefix_terms(kind_postings)

    reason: str
    if exact_native or exact_name:
        reason = "resource_name_exact"
    elif (
        any(
            posting.normalized_value.startswith(query.complete)
            for posting in resource_postings
        )
        or resource_prefix_terms
    ):
        reason = "resource_name_prefix"
    elif resource_postings:
        reason = "resource_name_contains"
    elif any(posting.normalized_value == query.complete for posting in column_postings):
        reason = "structural_field_exact"
    elif column_postings or relationship_postings:
        reason = "structural_field_contains"
    else:
        reason = "metadata_contains"

    field_priority = _FIELD_PRIORITY
    matched_fields = tuple(
        dict.fromkeys(
            posting.field_name
            for posting in sorted(
                postings,
                key=lambda item: (
                    field_priority[item.field_kind],
                    item.field_name.casefold(),
                    item.field_name,
                ),
            )
        )
    )[:32]
    score = float(
        exact_native * 200_000
        + exact_name * 180_000
        + len(matched_terms) * 5_000
        + len(resource_exact_terms) * 500
        + len(column_exact_terms) * 200
        + len(resource_prefix_terms) * 100
        + len(column_prefix_terms) * 50
        + len(relationship_terms) * 20
        + len(kind_terms) * 10
    )
    normalized_native = _normalize_search_text(resource.native_identity).complete
    rank_key = (
        -exact_native,
        -exact_name,
        -len(matched_terms),
        -len(resource_exact_terms),
        -len(column_exact_terms),
        -len(resource_prefix_terms),
        -len(column_prefix_terms),
        -len(relationship_terms),
        -len(kind_terms),
        normalized_native,
        resource.native_identity,
        resource.id,
    )
    return _RankedCandidate(
        hit=CatalogSearchHit(
            resource_id=resource.id,
            source_id=resource.source_id,
            kind=resource.kind,
            name=resource.name,
            revision=resource.revision,
            sensitivity=resource.sensitivity,
            score=score,
            matched_fields=matched_fields,
            match_reasons=(reason,),
        ),
        matched_terms=matched_terms,
        rank_key=rank_key,
    )


def _diversify_candidates(
    candidates: tuple[_RankedCandidate, ...],
    significant_terms: tuple[str, ...],
) -> tuple[_RankedCandidate, ...]:
    remaining = list(sorted(candidates, key=lambda candidate: candidate.rank_key))
    selected: list[_RankedCandidate] = []
    covered: set[str] = set()
    significant = set(significant_terms)
    while remaining and covered != significant:
        best = min(
            remaining,
            key=lambda candidate: (
                -len(set(candidate.matched_terms) - covered),
                candidate.rank_key,
            ),
        )
        new_terms = set(best.matched_terms) - covered
        if not new_terms:
            break
        selected.append(best)
        covered.update(new_terms)
        remaining.remove(best)
    selected.extend(sorted(remaining, key=lambda candidate: candidate.rank_key))
    return tuple(selected)


def _neighbor_payload(resource: CatalogResource) -> dict[str, object]:
    return {
        "kind": resource.kind.value,
        "name": resource.name,
        "resource_id": resource.id,
        "revision": resource.current_revision,
        "source_id": resource.source_id,
    }


def _schema_endpoint_payload(resource: CatalogResource) -> dict[str, object]:
    return {
        "kind": resource.kind.value,
        "name": resource.native_identity,
        "resource_id": resource.id,
        "revision": resource.current_revision,
        "source_id": resource.source_id,
    }


def _relationship_payload(
    relationship: CatalogRelationship,
    from_resource: CatalogResource,
    to_resource: CatalogResource,
    *,
    direction: RelationshipDirection,
) -> dict[str, object]:
    return {
        "confidence": relationship.confidence,
        "direction": direction.value,
        "field_pairs": tuple(pair.to_payload() for pair in relationship.field_pairs),
        "from_resource_id": relationship.from_resource_id,
        "from_resource_revision": from_resource.current_revision,
        "kind": relationship.kind.value,
        "observed_at": relationship.observed_at.isoformat(),
        "provenance": relationship.provenance.value,
        "relationship_id": relationship.id,
        "revision": relationship.revision,
        "source_id": relationship.source_id,
        "sync_id": relationship.sync_id,
        "to_resource_id": relationship.to_resource_id,
        "to_resource_revision": to_resource.current_revision,
    }


def _link_payload(
    link: CatalogLink,
    from_resource: CatalogResource,
    to_resource: CatalogResource,
    *,
    direction: RelationshipDirection,
) -> dict[str, object]:
    return {
        "confidence": link.confidence,
        "cross_source": from_resource.source_id != to_resource.source_id,
        "declared_at": link.declared_at.isoformat(),
        "direction": direction.value,
        "field_pairs": tuple(pair.to_payload() for pair in link.field_pairs),
        "from_resource_id": link.from_resource_id,
        "from_resource_revision": from_resource.current_revision,
        "kind": link.kind.value,
        "note": link.note,
        "provenance": link.provenance.value,
        "relationship_id": link.id,
        "to_resource_id": link.to_resource_id,
        "to_resource_revision": to_resource.current_revision,
    }


def _link_stale_reason(
    link: CatalogLink,
    active_source_ids: Collection[str],
    indexed_source_ids: Collection[str],
    entries: Mapping[str, CatalogEntry],
) -> CatalogLinkStaleReason | None:
    """Compute currency from fields, never from whole-resource revisions."""

    endpoints = (
        (
            link.from_source_id,
            link.from_resource_id,
            {pair.source_field for pair in link.field_pairs},
        ),
        (
            link.to_source_id,
            link.to_resource_id,
            {pair.target_field for pair in link.field_pairs},
        ),
    )
    if any(source_id not in active_source_ids for source_id, _, _ in endpoints):
        return CatalogLinkStaleReason.SOURCE_INACTIVE
    for source_id, resource_id, _ in endpoints:
        entry = entries.get(resource_id)
        if (
            source_id not in indexed_source_ids
            or entry is None
            or entry.resource.source_id != source_id
        ):
            return CatalogLinkStaleReason.RESOURCE_MISSING
    for _, resource_id, fields in endpoints:
        if not fields <= _tabular_field_names(entries[resource_id].facets):
            return CatalogLinkStaleReason.FIELD_MISSING
    return None


def _selection_facts(
    resource: CatalogResource,
    facets: tuple[CatalogFacet, ...],
    relationships: tuple[CatalogRelationship, ...],
    resources: dict[str, CatalogResource],
    child_file_freshness: dict[str, dict[str, object]],
    *,
    incident_truncated: bool,
) -> dict[str, object]:
    facts: dict[str, object] = {}
    if resource.kind is ResourceKind.FILE:
        facts["freshness"] = _file_freshness(facets)
    if resource.kind is ResourceKind.FOLDER:
        child_ids = tuple(
            relationship.to_resource_id
            for relationship in relationships
            if relationship.kind is RelationshipKind.CONTAINS
            and relationship.from_resource_id == resource.id
        )
        facts["hierarchy"] = {
            "authority": "catalog_relationship",
            "basis": "contains",
            "child_count": len(child_ids),
            "children": tuple(
                _neighbor_payload(resources[child_id]) for child_id in child_ids
            ),
            "truncated": incident_truncated,
        }
        facts["newest_child_file"] = _newest_child_file_selection(
            child_file_freshness,
            truncated=incident_truncated,
        )
    return facts


def _file_freshness(facets: tuple[CatalogFacet, ...]) -> dict[str, object]:
    file_facets = tuple(facet for facet in facets if facet.kind is FacetKind.FILE)
    facet = file_facets[0] if len(file_facets) == 1 else None
    value: str | None = None
    if facet is not None:
        raw_value = facet.payload.get("modified_at")
        if isinstance(raw_value, str):
            try:
                parsed = datetime.fromisoformat(raw_value)
            except ValueError:
                pass
            else:
                if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                    value = parsed.isoformat()
    return {
        "authority": "connector_metadata",
        "available": value is not None,
        "basis": "file.modified_at",
        "facet_revision": None if facet is None else facet.revision,
        "observed_at": None if facet is None else facet.observed_at.isoformat(),
        "sync_id": None if facet is None else facet.sync_id,
        "value": value,
    }


def _newest_child_file_selection(
    freshness_by_resource_id: dict[str, dict[str, object]],
    *,
    truncated: bool,
) -> dict[str, object]:
    candidates: list[tuple[str, datetime, str]] = []
    missing_resource_ids: list[str] = []
    candidate_values: list[dict[str, object]] = []
    for resource_id in sorted(freshness_by_resource_id):
        freshness = freshness_by_resource_id[resource_id]
        value = freshness["value"]
        available = freshness["available"] is True and isinstance(value, str)
        candidate_values.append(
            {
                "available": available,
                "resource_id": resource_id,
                "value": value if available else None,
            }
        )
        if not available:
            missing_resource_ids.append(resource_id)
            continue
        assert isinstance(value, str)
        parsed = datetime.fromisoformat(value)
        candidates.append((resource_id, parsed, value))

    if not freshness_by_resource_id and not truncated:
        status = "no_candidates"
        tied_resource_ids: tuple[str, ...] = ()
        ambiguity_reasons: tuple[str, ...] = ()
        selected_resource_id: str | None = None
    else:
        greatest = max((candidate[1] for candidate in candidates), default=None)
        tied_resource_ids = tuple(
            resource_id
            for resource_id, timestamp, _ in candidates
            if timestamp == greatest
        )
        reasons: list[str] = []
        if truncated:
            reasons.append("candidate_set_truncated")
        if missing_resource_ids:
            reasons.append("missing_freshness")
        if len(tied_resource_ids) > 1:
            reasons.append("equal_greatest_values")
        ambiguity_reasons = tuple(reasons)
        if reasons or not tied_resource_ids:
            status = "ambiguous"
            selected_resource_id = None
        else:
            status = "selected"
            selected_resource_id = tied_resource_ids[0]

    return {
        "ambiguity_reasons": ambiguity_reasons,
        "basis": "file.modified_at",
        "candidate_values": tuple(candidate_values),
        "missing_freshness_resource_ids": tuple(missing_resource_ids),
        "selected_resource_id": selected_resource_id,
        "status": status,
        "tied_resource_ids": tied_resource_ids,
        "truncated": truncated,
    }


__all__ = ["CatalogService"]
