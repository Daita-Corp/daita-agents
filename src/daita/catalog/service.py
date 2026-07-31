"""Catalog-owned projections consumed by tools and the data domain."""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import re
from types import MappingProxyType
from typing import TYPE_CHECKING
import unicodedata

from .._json import FrozenJsonObject
from .models import (
    CatalogFacet,
    CatalogPath,
    CatalogPathStep,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSchemaRequest,
    CatalogSnapshotRef,
    CatalogSearchHit,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSummary,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    CatalogTraversalResult,
    CatalogTraversalTruncationReason,
    FacetKind,
    RelationshipDirection,
    RelationshipKind,
    ResourceKind,
    SourceCatalogSnapshot,
)
from .protocols import CatalogResourceNotFoundError, CatalogStore, CatalogStoreError

if TYPE_CHECKING:
    from ..adapters.protocols import SourceStore

_INSPECT_INCIDENT_RELATIONSHIP_LIMIT = 50
_SCHEMA_COLUMN_LIMIT = 256
_SCHEMA_KEY_LIMIT = 64
_SCHEMA_RELATIONSHIP_LIMIT = 200
_SCHEMA_STRUCTURAL_FACT_LIMIT = 32
_SEARCH_TOKEN_LIMIT = 64
_SEARCH_TOKEN_LENGTH_LIMIT = 128
_SEARCH_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "by",
        "can",
        "for",
        "from",
        "get",
        "give",
        "how",
        "in",
        "is",
        "me",
        "of",
        "on",
        "show",
        "the",
        "to",
        "what",
        "which",
    }
)
_CAMEL_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_CAMEL_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True, slots=True)
class _NormalizedText:
    complete: str
    tokens: tuple[str, ...]


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
class _SourceCatalogIndex:
    agent_id: str
    source_id: str
    sync_id: str
    snapshot: SourceCatalogSnapshot
    resources_by_id: Mapping[str, CatalogResource]
    revisions_by_resource_id: Mapping[str, CatalogResourceRevision]
    facets_by_resource_id: Mapping[str, tuple[CatalogFacet, ...]]
    relationships_by_id: Mapping[str, CatalogRelationship]
    adjacency_by_resource_id: Mapping[str, tuple[_IndexedEdge, ...]]
    exact_resource_names: Mapping[str, tuple[str, ...]]
    token_postings: Mapping[str, tuple[_IndexedPosting, ...]]
    posting_tokens: tuple[str, ...]
    postings_by_resource_id: Mapping[str, tuple[_IndexedPosting, ...]]


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    hit: CatalogSearchHit
    matched_terms: tuple[str, ...]
    rank_key: tuple[int, int, int, int, int, int, int, int, int, str, str, str]


class _CatalogGenerationChanged(RuntimeError):
    """Signal one bounded schema-slice retry from fresh snapshot references."""


class CatalogService:
    """Keep catalog truth and its bounded projections behind one owner."""

    def __init__(self, store: CatalogStore, sources: SourceStore) -> None:
        if not isinstance(store, CatalogStore):
            raise TypeError("store must implement CatalogStore")
        if not callable(getattr(sources, "load_source", None)) or not callable(
            getattr(sources, "list_sources", None)
        ):
            raise TypeError("sources must provide source registration reads")
        self._store = store
        self._sources = sources
        self._source_indexes: dict[tuple[str, str, str], _SourceCatalogIndex] = {}
        self._source_index_lock = asyncio.Lock()

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
            or not 1 <= limit <= 50
        ):
            raise ValueError("catalog preview limit must be from 1 through 50")
        active_source_ids = tuple(
            sorted(
                registration.id
                for registration in await self._sources.list_sources(agent_id)
                if registration.agent_id == agent_id and registration.active
            )
        )
        resources: list[CatalogResource] = []
        for source_id in active_source_ids:
            resources.extend(await self._store.list_resources(agent_id, source_id))
        return tuple(
            sorted(resources, key=lambda resource: (resource.name, resource.id))[:limit]
        )

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        if not isinstance(request, CatalogSearchRequest):
            raise TypeError("request must be a CatalogSearchRequest record")
        for attempt in range(2):
            active_source_ids = await self._active_source_ids(request.agent_id)
            scoped_source_ids = (
                tuple(
                    source_id
                    for source_id in active_source_ids
                    if source_id in request.source_ids
                )
                if request.source_ids
                else active_source_ids
            )
            refs = (
                ()
                if not active_source_ids
                else await self._store.list_current_snapshot_refs(
                    request.agent_id,
                    active_source_ids,
                )
            )
            await self._evict_stale_indexes(
                request.agent_id,
                active_source_ids,
                refs,
            )
            selected_refs = tuple(
                ref for ref in refs if ref.source_id in scoped_source_ids
            )
            indexes: list[_SourceCatalogIndex] = []
            generation_changed = False
            for ref in selected_refs:
                try:
                    indexes.append(await self._index_for_ref(ref))
                except _CatalogGenerationChanged:
                    generation_changed = True
                    break
            if generation_changed:
                if attempt == 0:
                    continue
                break
            result = self._search_indexes(request, tuple(indexes))
            current_active_source_ids = await self._active_source_ids(request.agent_id)
            current_refs = (
                ()
                if not current_active_source_ids
                else await self._store.list_current_snapshot_refs(
                    request.agent_id,
                    current_active_source_ids,
                )
            )
            if current_active_source_ids == active_source_ids and current_refs == refs:
                return result
            if attempt == 1:
                break
        raise CatalogStoreError("catalog snapshot generation changed repeatedly")

    async def _index_for_ref(
        self,
        ref: CatalogSnapshotRef,
    ) -> _SourceCatalogIndex:
        if not isinstance(ref, CatalogSnapshotRef):
            raise TypeError("ref must be a CatalogSnapshotRef record")
        key = (ref.agent_id, ref.source_id, ref.sync_id)
        cached = self._source_indexes.get(key)
        if cached is not None:
            return cached
        snapshot = await self._store.load_current_snapshot(ref)
        if snapshot is None:
            raise _CatalogGenerationChanged
        if (
            snapshot.sync.agent_id,
            snapshot.sync.source_id,
            snapshot.sync.id,
        ) != key:
            raise CatalogStoreError("catalog snapshot disagrees with its reference")
        return await self._index_for_snapshot(snapshot)

    async def _index_for_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
    ) -> _SourceCatalogIndex:
        sync = snapshot.sync
        key = (sync.agent_id, sync.source_id, sync.id)
        cached = self._source_indexes.get(key)
        if cached is not None:
            return cached
        async with self._source_index_lock:
            cached = self._source_indexes.get(key)
            if cached is not None:
                return cached
            try:
                compiled = _compile_source_index(snapshot)
            except CatalogStoreError:
                raise
            except Exception as exc:
                raise CatalogStoreError(
                    "catalog index compilation failed for "
                    f"{sync.agent_id}/{sync.source_id}/{sync.id}"
                ) from exc
            self._source_indexes[key] = compiled
            return compiled

    async def _evict_stale_indexes(
        self,
        agent_id: str,
        active_source_ids: tuple[str, ...],
        refs: tuple[CatalogSnapshotRef, ...],
    ) -> None:
        active = set(active_source_ids)
        current_sync_by_source = {
            ref.source_id: ref.sync_id for ref in refs if ref.agent_id == agent_id
        }
        async with self._source_index_lock:
            for key in tuple(self._source_indexes):
                cached_agent_id, source_id, sync_id = key
                if cached_agent_id != agent_id:
                    continue
                if (
                    source_id not in active
                    or current_sync_by_source.get(source_id) != sync_id
                ):
                    del self._source_indexes[key]

    def _search_indexes(
        self,
        request: CatalogSearchRequest,
        indexes: tuple[_SourceCatalogIndex, ...],
    ) -> CatalogSearchResult:
        for index in indexes:
            if index.agent_id != request.agent_id or (
                request.source_ids and index.source_id not in request.source_ids
            ):
                raise CatalogStoreError("catalog search index escaped request scope")

        normalized_query = _normalize_search_text(request.query)
        significant_terms = tuple(
            token
            for token in normalized_query.tokens
            if token not in _SEARCH_STOP_WORDS
        )
        resource_kinds = set(request.resource_kinds)
        index_by_source_id = {index.source_id: index for index in indexes}

        if not normalized_query.complete:
            resources_by_id = {
                resource.id: resource
                for index in indexes
                for resource in index.resources_by_id.values()
                if not resource_kinds or resource.kind in resource_kinds
            }
            candidates = tuple(
                _inventory_candidate(resource)
                for resource in sorted(
                    resources_by_id.values(),
                    key=_resource_search_tie_key,
                )
            )
        else:
            postings_by_candidate: dict[str, set[_IndexedPosting]] = {}
            candidate_resources: dict[str, CatalogResource] = {}
            for index in indexes:
                exact_ids = index.exact_resource_names.get(
                    normalized_query.complete,
                    (),
                )
                for resource_id in exact_ids:
                    resource = index.resources_by_id[resource_id]
                    if resource_kinds and resource.kind not in resource_kinds:
                        continue
                    candidate_resources[resource_id] = resource
                    matching = tuple(
                        posting
                        for posting in index.postings_by_resource_id.get(
                            resource_id,
                            (),
                        )
                        if posting.normalized_value == normalized_query.complete
                    )
                    postings_by_candidate.setdefault(resource_id, set()).update(
                        matching
                    )
                for term in significant_terms:
                    token_position = bisect_left(index.posting_tokens, term)
                    while token_position < len(index.posting_tokens):
                        indexed_token = index.posting_tokens[token_position]
                        if not indexed_token.startswith(term):
                            break
                        postings = index.token_postings[indexed_token]
                        for posting in postings:
                            resource = index.resources_by_id[posting.resource_id]
                            if resource_kinds and resource.kind not in resource_kinds:
                                continue
                            candidate_resources[posting.resource_id] = resource
                            postings_by_candidate.setdefault(
                                posting.resource_id,
                                set(),
                            ).add(posting)
                        token_position += 1
            candidates = tuple(
                _rank_index_candidate(
                    candidate_resources[resource_id],
                    tuple(sorted(postings, key=_posting_sort_key)),
                    normalized_query,
                    significant_terms,
                )
                for resource_id, postings in sorted(postings_by_candidate.items())
            )
            candidates = _diversify_candidates(candidates, significant_terms)

        direct_ids = {candidate.hit.resource_id for candidate in candidates}
        neighbor_by_id: dict[
            str,
            tuple[tuple[int, str, str, str], CatalogSearchHit],
        ] = {}
        for direct_position, candidate in enumerate(candidates):
            index = index_by_source_id[candidate.hit.source_id]
            for edge in index.adjacency_by_resource_id.get(
                candidate.hit.resource_id,
                (),
            ):
                neighbor = index.resources_by_id.get(edge.neighbor_resource_id)
                if (
                    neighbor is None
                    or neighbor.id in direct_ids
                    or (resource_kinds and neighbor.kind not in resource_kinds)
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
                    revision=neighbor.current_revision,
                    sensitivity=neighbor.sensitivity,
                    score=0.0,
                    matched_fields=(f"relationship:{edge.relationship_id}",),
                    match_reasons=("relationship_neighbor",),
                )
                existing = neighbor_by_id.get(neighbor.id)
                if existing is None or order < existing[0]:
                    neighbor_by_id[neighbor.id] = (order, hit)

        direct_hits = tuple(candidate.hit for candidate in candidates)
        neighbors = tuple(
            hit for _, hit in sorted(neighbor_by_id.values(), key=lambda item: item[0])
        )
        all_hits = (*direct_hits, *neighbors)
        hits = tuple(all_hits[: request.limit])
        return CatalogSearchResult(
            request=request,
            hits=hits,
            total_matches=len(all_hits),
            truncated=len(all_hits) > len(hits),
        )

    async def schema_slice(
        self,
        request: CatalogSchemaRequest,
    ) -> FrozenJsonObject:
        """Project bounded current structural truth without source connector I/O."""

        if not isinstance(request, CatalogSchemaRequest):
            raise TypeError("request must be a CatalogSchemaRequest record")
        for attempt in range(2):
            active_source_ids = await self._active_source_ids(request.agent_id)
            if (
                request.source_id is not None
                and request.source_id not in active_source_ids
            ):
                raise CatalogStoreError(
                    f"unknown active catalog source for {request.agent_id}: "
                    f"{request.source_id}"
                )
            refs = (
                ()
                if not active_source_ids
                else await self._store.list_current_snapshot_refs(
                    request.agent_id,
                    active_source_ids,
                )
            )
            snapshots: list[SourceCatalogSnapshot] = []
            generation_changed = False
            for ref in refs:
                snapshot = await self._store.load_current_snapshot(ref)
                if snapshot is None:
                    generation_changed = True
                    break
                snapshots.append(snapshot)
            if generation_changed:
                if attempt == 0:
                    continue
                break
            await self._evict_stale_indexes(
                request.agent_id,
                active_source_ids,
                refs,
            )
            indexes = (
                tuple(
                    [await self._index_for_snapshot(snapshot) for snapshot in snapshots]
                )
                if request.query is not None
                else ()
            )
            try:
                projection = await self._schema_slice_from_snapshots(
                    request,
                    tuple(snapshots),
                    active_source_ids,
                    indexes,
                )
            except _CatalogGenerationChanged:
                if attempt == 0:
                    continue
                break
            current_source_ids = await self._active_source_ids(request.agent_id)
            current_refs = (
                ()
                if not current_source_ids
                else await self._store.list_current_snapshot_refs(
                    request.agent_id,
                    current_source_ids,
                )
            )
            if current_source_ids == active_source_ids and current_refs == refs:
                return projection
            if attempt == 1:
                break
        raise CatalogStoreError("catalog snapshot generation changed repeatedly")

    async def _schema_slice_from_snapshots(
        self,
        request: CatalogSchemaRequest,
        snapshots: tuple[SourceCatalogSnapshot, ...],
        active_source_ids: tuple[str, ...],
        indexes: tuple[_SourceCatalogIndex, ...],
    ) -> FrozenJsonObject:
        resources_by_id: dict[str, CatalogResource] = {}
        revisions_by_resource_id: dict[str, CatalogResourceRevision] = {}
        facets_by_resource_id: dict[str, list[CatalogFacet]] = {}
        relationships_by_id: dict[str, CatalogRelationship] = {}
        incident_by_resource_id: dict[str, list[CatalogRelationship]] = {}
        all_syncs_by_id: dict[str, CatalogSync] = {}
        for snapshot in snapshots:
            sync = snapshot.sync
            if (
                sync.agent_id != request.agent_id
                or sync.source_id not in active_source_ids
                or sync.id in all_syncs_by_id
            ):
                raise CatalogStoreError("catalog snapshot set has invalid ownership")
            all_syncs_by_id[sync.id] = sync
            for resource in snapshot.resources:
                if resource.id in resources_by_id:
                    raise CatalogStoreError("catalog snapshot set repeats a resource")
                resources_by_id[resource.id] = resource
            for revision in snapshot.revisions:
                if revision.resource_id in revisions_by_resource_id:
                    raise CatalogStoreError("catalog snapshot set repeats a revision")
                revisions_by_resource_id[revision.resource_id] = revision
            for facet in snapshot.facets:
                facets_by_resource_id.setdefault(facet.resource_id, []).append(facet)
            for relationship in snapshot.relationships:
                if relationship.id in relationships_by_id:
                    raise CatalogStoreError(
                        "catalog snapshot set repeats a relationship"
                    )
                relationships_by_id[relationship.id] = relationship
                incident_by_resource_id.setdefault(
                    relationship.from_resource_id,
                    [],
                ).append(relationship)
                incident_by_resource_id.setdefault(
                    relationship.to_resource_id,
                    [],
                ).append(relationship)

        match_by_resource_id: dict[str, CatalogSearchHit] = {}
        if request.query is not None:
            search_request = CatalogSearchRequest(
                agent_id=request.agent_id,
                query=request.query,
                source_ids=(() if request.source_id is None else (request.source_id,)),
                limit=50,
            )
            search = self._search_indexes(
                search_request,
                tuple(
                    index
                    for index in indexes
                    if request.source_id is None or index.source_id == request.source_id
                ),
            )
            match_by_resource_id = {hit.resource_id: hit for hit in search.hits}
        else:
            search = None

        if request.resource_ids:
            selected_resources_by_id: dict[str, CatalogResource] = {}
            for resource_id in request.resource_ids:
                candidate_resource = resources_by_id.get(resource_id)
                if candidate_resource is None:
                    raise CatalogResourceNotFoundError(request.agent_id, resource_id)
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
                hit_resource = resources_by_id.get(hit.resource_id)
                if (
                    hit_resource is None
                    or hit_resource.current_revision != hit.revision
                    or hit_resource.source_id != hit.source_id
                ):
                    raise _CatalogGenerationChanged
                if (
                    request.source_id is not None
                    and hit_resource.source_id != request.source_id
                ):
                    raise CatalogStoreError(
                        "catalog schema search escaped requested source scope"
                    )
                candidates_list.append(hit_resource)
            candidates = tuple(candidates_list)
            total_matches = search.total_matches

        selected = candidates[: request.limit]
        selected_resources = {resource.id: resource for resource in selected}
        selected_syncs_by_id: dict[str, CatalogSync] = {}
        resource_payloads: list[dict[str, object]] = []
        columns_truncated = False
        primary_keys_truncated = False
        unique_keys_truncated = False
        structural_facts_truncated = False
        for resource in selected:
            resource_sync = all_syncs_by_id.get(resource.current_sync_id)
            if (
                resource_sync is None
                or resource_sync.agent_id != request.agent_id
                or resource_sync.source_id != resource.source_id
                or resource_sync.status is not CatalogSyncStatus.SUCCEEDED
            ):
                raise CatalogStoreError("catalog resource sync is not current")
            selected_syncs_by_id[resource_sync.id] = resource_sync
            (
                structural,
                resource_columns_truncated,
                resource_primary_keys_truncated,
                resource_unique_keys_truncated,
            ) = self._schema_resource_structure(
                tuple(facets_by_resource_id.get(resource.id, ())),
            )
            match_hit = match_by_resource_id.get(resource.id)
            matched_fields = () if match_hit is None else match_hit.matched_fields
            match_reasons = () if match_hit is None else match_hit.match_reasons
            resource_structural_facts_truncated = (
                len(matched_fields) > _SCHEMA_STRUCTURAL_FACT_LIMIT
                or len(match_reasons) > _SCHEMA_STRUCTURAL_FACT_LIMIT
            )
            resource_payloads.append(
                {
                    "columns": structural["columns"],
                    "kind": resource.kind.value,
                    "name": resource.native_identity,
                    "primary_key_fields": structural["primary_key_fields"],
                    "resource_id": resource.id,
                    "revision": resource.current_revision,
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
        relationships_truncated = False
        if request.include_relationships:
            selected_relationships_by_id: dict[str, CatalogRelationship] = {}
            for resource in selected:
                incident = tuple(
                    sorted(
                        incident_by_resource_id.get(resource.id, ()),
                        key=lambda item: item.id,
                    )[: _SCHEMA_RELATIONSHIP_LIMIT + 1]
                )
                if len(incident) > _SCHEMA_RELATIONSHIP_LIMIT:
                    relationships_truncated = True
                for relationship in incident:
                    selected_relationships_by_id[relationship.id] = relationship
            relationships = tuple(
                sorted(
                    selected_relationships_by_id.values(),
                    key=lambda item: item.id,
                )
            )
            if len(relationships) > _SCHEMA_RELATIONSHIP_LIMIT:
                relationships_truncated = True
            selected_ids = set(selected_resources)
            for relationship in relationships[:_SCHEMA_RELATIONSHIP_LIMIT]:
                from_resource, to_resource = self._snapshot_relationship_endpoints(
                    relationship,
                    resources_by_id,
                    revisions_by_resource_id,
                )
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
                            for pair in relationship.field_pairs
                        ),
                        "from_resource_id": relationship.from_resource_id,
                        "from_resource_revision": from_resource.current_revision,
                        "kind": relationship.kind.value,
                        "provenance": relationship.provenance.value,
                        "relationship_id": relationship.id,
                        "to_resource_id": relationship.to_resource_id,
                        "to_resource_revision": to_resource.current_revision,
                        "unselected_endpoints": unselected_endpoints,
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
                    "relationships": _SCHEMA_RELATIONSHIP_LIMIT,
                    "resources": request.limit,
                    "structural_facts_per_resource": _SCHEMA_STRUCTURAL_FACT_LIMIT,
                    "unique_key_fields_per_resource": _SCHEMA_KEY_LIMIT,
                },
                "include_relationships": request.include_relationships,
                "relationships": relationship_payloads,
                "resources": resource_payloads,
                "sources": source_payloads,
                "total_matches": total_matches,
                "truncation": {
                    "columns": columns_truncated,
                    "primary_key_fields": primary_keys_truncated,
                    "relationships": relationships_truncated,
                    "resources": total_matches > len(selected),
                    "structural_facts": structural_facts_truncated,
                    "unique_key_fields": unique_keys_truncated,
                },
                "trust_classification": "untrusted_external_data",
            }
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
    ) -> FrozenJsonObject:
        resource = await self._active_resource(agent_id, resource_id)
        facets = await self._store.load_facets(
            agent_id,
            resource.id,
            resource.current_revision,
        )
        incident = await self._store.load_incident_relationships(
            agent_id,
            resource.id,
            limit=_INSPECT_INCIDENT_RELATIONSHIP_LIMIT + 1,
        )
        incident_truncated = len(incident) > _INSPECT_INCIDENT_RELATIONSHIP_LIMIT
        incident = incident[:_INSPECT_INCIDENT_RELATIONSHIP_LIMIT]
        resources = {resource.id: resource}
        relationships_payload: list[dict[str, object]] = []
        neighbor_payloads: list[dict[str, object]] = []
        seen_neighbor_ids: set[str] = set()
        for relationship in incident:
            endpoints = await self._current_relationship_endpoints(
                agent_id,
                relationship,
                resources,
            )
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
                child_facets = await self._store.load_facets(
                    agent_id,
                    child.id,
                    child.current_revision,
                )
                child_file_freshness[child.id] = _file_freshness(child_facets)

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
    ) -> FrozenJsonObject:
        if not isinstance(request, CatalogTraversalRequest):
            raise TypeError("request must be a CatalogTraversalRequest record")
        for attempt in range(2):
            active_source_ids = await self._active_source_ids(request.agent_id)
            refs = (
                ()
                if not active_source_ids
                else await self._store.list_current_snapshot_refs(
                    request.agent_id,
                    active_source_ids,
                )
            )
            await self._evict_stale_indexes(
                request.agent_id,
                active_source_ids,
                refs,
            )
            indexes: list[_SourceCatalogIndex] = []
            generation_changed = False
            for ref in refs:
                try:
                    indexes.append(await self._index_for_ref(ref))
                except _CatalogGenerationChanged:
                    generation_changed = True
                    break
            if generation_changed:
                if attempt == 0:
                    continue
                break
            resolved_indexes = tuple(indexes)
            result = self._traverse_indexes(request, resolved_indexes)
            projection = self._traversal_projection(result, resolved_indexes)
            current_active_source_ids = await self._active_source_ids(request.agent_id)
            current_refs = (
                ()
                if not current_active_source_ids
                else await self._store.list_current_snapshot_refs(
                    request.agent_id,
                    current_active_source_ids,
                )
            )
            if current_active_source_ids == active_source_ids and current_refs == refs:
                return projection
            if attempt == 1:
                break
        raise CatalogStoreError("catalog snapshot generation changed repeatedly")

    def _traverse_indexes(
        self,
        request: CatalogTraversalRequest,
        indexes: tuple[_SourceCatalogIndex, ...],
    ) -> CatalogTraversalResult:
        resources_by_id: dict[str, CatalogResource] = {}
        relationships_by_id: dict[str, CatalogRelationship] = {}
        index_by_resource_id: dict[str, _SourceCatalogIndex] = {}
        for index in indexes:
            if index.agent_id != request.agent_id:
                raise CatalogStoreError("catalog traversal index escaped agent scope")
            duplicate_resources = resources_by_id.keys() & index.resources_by_id.keys()
            duplicate_relationships = (
                relationships_by_id.keys() & index.relationships_by_id.keys()
            )
            if duplicate_resources or duplicate_relationships:
                raise CatalogStoreError("catalog traversal index identities overlap")
            resources_by_id.update(index.resources_by_id)
            relationships_by_id.update(index.relationships_by_id)
            index_by_resource_id.update(
                (resource_id, index) for resource_id in index.resources_by_id
            )

        for resource_id in (*request.from_resource_ids, *request.to_resource_ids):
            if resource_id not in resources_by_id:
                raise CatalogResourceNotFoundError(request.agent_id, resource_id)

        def resource_key(resource_id: str) -> tuple[str, str, str]:
            resource = resources_by_id[resource_id]
            return (
                _normalize_search_text(resource.native_identity).complete,
                resource.source_id,
                resource.id,
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
        truncation_reason = (
            CatalogTraversalTruncationReason.NODES
            if len(admitted_sources) < len(ordered_sources)
            else None
        )
        allowed_kinds = set(request.relationship_kinds)

        while frontier and truncation_reason is None:
            current_resource_id = frontier.popleft()
            current_distance = distance_by_resource[current_resource_id]
            if current_resource_id in target_ids and current_distance > 0:
                continue
            index = index_by_resource_id[current_resource_id]
            for edge in index.adjacency_by_resource_id.get(current_resource_id, ()):
                relationship = relationships_by_id[edge.relationship_id]
                if allowed_kinds and relationship.kind not in allowed_kinds:
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
                elif known_distance == candidate_distance:
                    parents = parents_by_resource[edge.neighbor_resource_id]
                    if parent not in parents:
                        parents.append(parent)

        def parent_key(parent: _ParentStep) -> tuple[object, ...]:
            relationship = relationships_by_id[parent.relationship_id]
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
        indexes: tuple[_SourceCatalogIndex, ...],
    ) -> FrozenJsonObject:
        request = result.request
        resources_by_id = {
            resource_id: resource
            for index in indexes
            for resource_id, resource in index.resources_by_id.items()
        }
        revisions_by_resource_id = {
            resource_id: revision
            for index in indexes
            for resource_id, revision in index.revisions_by_resource_id.items()
        }
        relationships_by_id = {
            relationship_id: relationship
            for index in indexes
            for relationship_id, relationship in index.relationships_by_id.items()
        }

        paths_payload: list[dict[str, object]] = []
        for path in result.paths:
            steps_payload: list[dict[str, object]] = []
            for step in path.steps:
                relationship = relationships_by_id.get(step.relationship_id)
                if relationship is None:
                    raise CatalogStoreError(
                        "catalog traversal references a non-current relationship"
                    )
                endpoints = self._snapshot_relationship_endpoints(
                    relationship,
                    resources_by_id,
                    revisions_by_resource_id,
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
                payload = _relationship_payload(
                    relationship,
                    endpoints[0],
                    endpoints[1],
                    direction=step.direction,
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

    async def _active_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> CatalogResource:
        resource = await self._store.load_resource(agent_id, resource_id)
        if resource is None or resource.agent_id != agent_id:
            raise CatalogResourceNotFoundError(agent_id, resource_id)
        registration = await self._sources.load_source(agent_id, resource.source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != resource.source_id
            or not registration.active
        ):
            raise CatalogResourceNotFoundError(agent_id, resource_id)
        return resource

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

    async def _current_resource_sync(
        self,
        agent_id: str,
        resource: CatalogResource,
        sync_by_id: dict[str, CatalogSync],
    ) -> CatalogSync:
        sync = sync_by_id.get(resource.current_sync_id)
        if sync is None:
            loaded = await self._store.load_sync(
                agent_id,
                resource.current_sync_id,
            )
            if loaded is None:
                raise CatalogStoreError("catalog resource lacks its current sync")
            sync = loaded
            sync_by_id[sync.id] = sync
        if (
            sync.agent_id != agent_id
            or sync.source_id != resource.source_id
            or sync.id != resource.current_sync_id
            or sync.status is not CatalogSyncStatus.SUCCEEDED
        ):
            raise CatalogStoreError("catalog resource sync is not current")
        return sync

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
        raw_columns = tabular.payload.get("columns", ())
        raw_indexes = tabular.payload.get("indexes", ())
        if not isinstance(raw_columns, tuple) or not isinstance(raw_indexes, tuple):
            raise CatalogStoreError("catalog tabular facet has invalid structure")

        columns: list[dict[str, object]] = []
        primary_fields: list[tuple[int, str]] = []
        for raw_column in raw_columns:
            if not isinstance(raw_column, FrozenJsonObject):
                raise CatalogStoreError("catalog tabular column has invalid structure")
            name = raw_column.get("name")
            native_type = raw_column.get("native_type")
            nullable = raw_column.get("nullable")
            if (
                not isinstance(name, str)
                or not isinstance(native_type, str)
                or not isinstance(nullable, bool)
            ):
                raise CatalogStoreError("catalog tabular column is incomplete")
            columns.append(
                {
                    "name": name,
                    "nullable": nullable,
                    "type": native_type,
                }
            )
            primary_ordinal = raw_column.get("primary_key_ordinal")
            if (
                isinstance(primary_ordinal, int)
                and not isinstance(primary_ordinal, bool)
                and primary_ordinal > 0
            ):
                primary_fields.append((primary_ordinal, name))
        ordered_primary = tuple(
            name for _, name in sorted(primary_fields, key=lambda item: item)
        )
        unique_keys: set[tuple[str, ...]] = set()
        for raw_index in raw_indexes:
            if not isinstance(raw_index, FrozenJsonObject):
                raise CatalogStoreError("catalog tabular index has invalid structure")
            raw_fields = raw_index.get("columns")
            if (
                raw_index.get("unique") is not True
                or raw_index.get("predicate") is not None
                or not isinstance(raw_fields, tuple)
                or not raw_fields
                or any(not isinstance(field, str) for field in raw_fields)
            ):
                continue
            fields = tuple(field for field in raw_fields if isinstance(field, str))
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

    def _snapshot_relationship_endpoints(
        self,
        relationship: CatalogRelationship,
        resources_by_id: dict[str, CatalogResource],
        revisions_by_resource_id: dict[str, CatalogResourceRevision],
    ) -> tuple[CatalogResource, CatalogResource]:
        endpoints: list[CatalogResource] = []
        for resource_id in (
            relationship.from_resource_id,
            relationship.to_resource_id,
        ):
            endpoint = resources_by_id.get(resource_id)
            if endpoint is None:
                raise CatalogStoreError(
                    "catalog relationship has a non-current endpoint"
                )
            current_revision = revisions_by_resource_id.get(endpoint.id)
            if (
                endpoint.source_id != relationship.source_id
                or endpoint.current_sync_id != relationship.sync_id
                or current_revision is None
                or current_revision.revision != endpoint.current_revision
                or current_revision.sync_id != endpoint.current_sync_id
                or relationship.revision not in current_revision.relationship_revisions
            ):
                raise CatalogStoreError(
                    "catalog relationship is not current for both endpoints"
                )
            endpoints.append(endpoint)
        return endpoints[0], endpoints[1]

    async def _current_relationship_endpoints(
        self,
        agent_id: str,
        relationship: CatalogRelationship,
        resources: dict[str, CatalogResource],
    ) -> tuple[CatalogResource, CatalogResource]:
        endpoints: list[CatalogResource] = []
        for resource_id in (
            relationship.from_resource_id,
            relationship.to_resource_id,
        ):
            endpoint = resources.get(resource_id)
            if endpoint is None:
                endpoint = await self._active_resource(agent_id, resource_id)
                resources[resource_id] = endpoint
            current_revision = await self._store.load_revision(
                agent_id,
                endpoint.id,
                endpoint.current_revision,
            )
            if (
                endpoint.source_id != relationship.source_id
                or endpoint.current_sync_id != relationship.sync_id
                or current_revision is None
                or current_revision.sync_id != endpoint.current_sync_id
                or relationship.revision not in current_revision.relationship_revisions
            ):
                raise CatalogStoreError(
                    "catalog relationship is not current for both endpoints"
                )
            endpoints.append(endpoint)
        return endpoints[0], endpoints[1]

    async def catalog_context(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int,
        source_ids: tuple[str, ...] = (),
        resource_ids: tuple[str, ...] = (),
    ) -> FrozenJsonObject:
        result = await self.search(
            CatalogSearchRequest(
                agent_id=agent_id,
                query=query,
                source_ids=source_ids,
                limit=(50 if resource_ids else min(limit, 50)),
            )
        )
        hits = tuple(
            hit
            for hit in result.hits
            if not resource_ids or hit.resource_id in resource_ids
        )[:limit]
        sync_by_id: dict[str, CatalogSync] = {}
        current_by_id: dict[str, tuple[CatalogResource, CatalogSync]] = {}
        for hit in hits:
            resource = await self._active_resource(agent_id, hit.resource_id)
            sync = await self._current_resource_sync(
                agent_id,
                resource,
                sync_by_id,
            )
            current_by_id[hit.resource_id] = (resource, sync)
        return FrozenJsonObject.from_mapping(
            {
                "resources": [
                    {
                        "kind": hit.kind.value,
                        "name": hit.name,
                        "resource_id": hit.resource_id,
                        "revision": hit.revision,
                        "sensitivity": hit.sensitivity.value,
                        "source_id": hit.source_id,
                        "source_revision": current_by_id[hit.resource_id][
                            1
                        ].source_revision,
                        "sync_id": current_by_id[hit.resource_id][0].current_sync_id,
                    }
                    for hit in hits
                ],
                "total_matches": len(hits),
                "truncated": result.truncated,
                "trust_classification": "untrusted_external_data",
            }
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


def _compile_source_index(snapshot: SourceCatalogSnapshot) -> _SourceCatalogIndex:
    """Compile one immutable, exact-generation catalog search shard."""

    if not isinstance(snapshot, SourceCatalogSnapshot):
        raise TypeError("snapshot must be a SourceCatalogSnapshot record")
    sync = snapshot.sync
    resources_by_id = {resource.id: resource for resource in snapshot.resources}
    revisions_by_resource_id = {
        revision.resource_id: revision for revision in snapshot.revisions
    }
    facets_by_resource_id: dict[str, list[CatalogFacet]] = {}
    for facet in snapshot.facets:
        facets_by_resource_id.setdefault(facet.resource_id, []).append(facet)
    relationships_by_id = {
        relationship.id: relationship for relationship in snapshot.relationships
    }

    postings_by_resource_id: dict[str, set[_IndexedPosting]] = {
        resource_id: set() for resource_id in resources_by_id
    }
    exact_resource_names: dict[str, set[str]] = {}

    def add_posting(
        resource_id: str,
        field_kind: str,
        field_name: str,
        value: str,
    ) -> None:
        normalized = _normalize_search_text(value)
        posting = _IndexedPosting(
            resource_id=resource_id,
            field_kind=field_kind,
            field_name=field_name,
            normalized_value=normalized.complete,
            tokens=normalized.tokens,
        )
        postings_by_resource_id[resource_id].add(posting)

    for resource in sorted(resources_by_id.values(), key=lambda item: item.id):
        for field_kind, field_name, value in (
            ("resource_name", "name", resource.name),
            ("native_identity", "native_identity", resource.native_identity),
        ):
            add_posting(resource.id, field_kind, field_name, value)
            normalized = _normalize_search_text(value).complete
            exact_resource_names.setdefault(normalized, set()).add(resource.id)
        add_posting(
            resource.id,
            "kind",
            "kind",
            f"{resource.kind.value} {resource.kind.value}s",
        )

        for facet in sorted(
            facets_by_resource_id.get(resource.id, ()),
            key=lambda item: (item.kind.value, item.revision),
        ):
            if facet.kind is not FacetKind.TABULAR:
                continue
            raw_columns = facet.payload.get("columns", ())
            raw_indexes = facet.payload.get("indexes", ())
            if not isinstance(raw_columns, tuple) or not isinstance(
                raw_indexes,
                tuple,
            ):
                raise CatalogStoreError("catalog tabular facet has invalid structure")
            for raw_column in raw_columns:
                if not isinstance(raw_column, FrozenJsonObject):
                    raise CatalogStoreError(
                        "catalog tabular column has invalid structure"
                    )
                name = raw_column.get("name")
                if not isinstance(name, str) or not name:
                    raise CatalogStoreError("catalog tabular column is incomplete")
                add_posting(resource.id, "column", f"column:{name}", name)
            for raw_index in raw_indexes:
                if not isinstance(raw_index, FrozenJsonObject):
                    raise CatalogStoreError(
                        "catalog tabular index has invalid structure"
                    )
                raw_fields = raw_index.get("columns")
                if not isinstance(raw_fields, tuple) or any(
                    not isinstance(field, str) or not field for field in raw_fields
                ):
                    raise CatalogStoreError(
                        "catalog tabular index fields have invalid structure"
                    )
                for field in raw_fields:
                    assert isinstance(field, str)
                    add_posting(
                        resource.id,
                        "index_field",
                        f"index_field:{field}",
                        field,
                    )

    adjacency_by_resource_id: dict[str, list[_IndexedEdge]] = {}
    for relationship in sorted(
        relationships_by_id.values(),
        key=lambda item: item.id,
    ):
        adjacency_by_resource_id.setdefault(
            relationship.from_resource_id,
            [],
        ).append(
            _IndexedEdge(
                neighbor_resource_id=relationship.to_resource_id,
                relationship_id=relationship.id,
                direction=RelationshipDirection.FORWARD,
            )
        )
        adjacency_by_resource_id.setdefault(
            relationship.to_resource_id,
            [],
        ).append(
            _IndexedEdge(
                neighbor_resource_id=relationship.from_resource_id,
                relationship_id=relationship.id,
                direction=RelationshipDirection.REVERSE,
            )
        )
        for pair in relationship.field_pairs:
            add_posting(
                relationship.from_resource_id,
                "relationship_field",
                f"relationship_field:{pair.source_field}",
                pair.source_field,
            )
            add_posting(
                relationship.to_resource_id,
                "relationship_field",
                f"relationship_field:{pair.target_field}",
                pair.target_field,
            )

    token_postings: dict[str, set[_IndexedPosting]] = {}
    for postings in postings_by_resource_id.values():
        for posting in postings:
            for token in posting.tokens:
                token_postings.setdefault(token, set()).add(posting)

    def edge_sort_key(
        resource_id: str,
        edge: _IndexedEdge,
    ) -> tuple[str, str, str, str, str, str]:
        relationship = relationships_by_id[edge.relationship_id]
        neighbor = resources_by_id[edge.neighbor_resource_id]
        return (
            relationship.kind.value,
            relationship.provenance.value,
            _normalize_search_text(neighbor.native_identity).complete,
            relationship.id,
            edge.direction.value,
            edge.neighbor_resource_id,
        )

    ordered_token_postings = {
        token: tuple(sorted(postings, key=_posting_sort_key))
        for token, postings in sorted(token_postings.items())
    }
    return _SourceCatalogIndex(
        agent_id=sync.agent_id,
        source_id=sync.source_id,
        sync_id=sync.id,
        snapshot=snapshot,
        resources_by_id=MappingProxyType(dict(sorted(resources_by_id.items()))),
        revisions_by_resource_id=MappingProxyType(
            dict(sorted(revisions_by_resource_id.items()))
        ),
        facets_by_resource_id=MappingProxyType(
            {
                resource_id: tuple(
                    sorted(
                        facets,
                        key=lambda item: (item.kind.value, item.revision),
                    )
                )
                for resource_id, facets in sorted(facets_by_resource_id.items())
            }
        ),
        relationships_by_id=MappingProxyType(dict(sorted(relationships_by_id.items()))),
        adjacency_by_resource_id=MappingProxyType(
            {
                resource_id: tuple(
                    sorted(
                        edges,
                        key=lambda edge: edge_sort_key(resource_id, edge),
                    )
                )
                for resource_id, edges in sorted(adjacency_by_resource_id.items())
            }
        ),
        exact_resource_names=MappingProxyType(
            {
                value: tuple(sorted(resource_ids))
                for value, resource_ids in sorted(exact_resource_names.items())
            }
        ),
        token_postings=MappingProxyType(ordered_token_postings),
        posting_tokens=tuple(ordered_token_postings),
        postings_by_resource_id=MappingProxyType(
            {
                resource_id: tuple(sorted(postings, key=_posting_sort_key))
                for resource_id, postings in sorted(postings_by_resource_id.items())
            }
        ),
    )


def _normalize_search_text(value: str) -> _NormalizedText:
    if not isinstance(value, str):
        raise TypeError("catalog search text must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    split = _CAMEL_ACRONYM_BOUNDARY.sub(" ", normalized)
    split = _CAMEL_WORD_BOUNDARY.sub(" ", split)
    complete = normalized.casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in split.casefold():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    bounded_tokens = tuple(
        dict.fromkeys(token[:_SEARCH_TOKEN_LENGTH_LIMIT] for token in tokens if token)
    )[:_SEARCH_TOKEN_LIMIT]
    return _NormalizedText(complete=complete, tokens=bounded_tokens)


def _posting_sort_key(posting: _IndexedPosting) -> tuple[str, str, str, str]:
    return (
        posting.resource_id,
        posting.field_kind,
        posting.field_name.casefold(),
        posting.field_name,
    )


def _resource_search_tie_key(
    resource: CatalogResource,
) -> tuple[str, str, str, str]:
    normalized = _normalize_search_text(resource.native_identity).complete
    return (normalized, resource.native_identity, resource.name, resource.id)


def _inventory_candidate(resource: CatalogResource) -> _RankedCandidate:
    hit = CatalogSearchHit(
        resource_id=resource.id,
        source_id=resource.source_id,
        kind=resource.kind,
        name=resource.name,
        revision=resource.current_revision,
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


def _rank_index_candidate(
    resource: CatalogResource,
    postings: tuple[_IndexedPosting, ...],
    query: _NormalizedText,
    significant_terms: tuple[str, ...],
) -> _RankedCandidate:
    """Build deterministic rank evidence for one posting-derived candidate."""

    matched_terms = tuple(
        term
        for term in significant_terms
        if any(
            token.startswith(term) for posting in postings for token in posting.tokens
        )
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

    field_priority = {
        "resource_name": 0,
        "native_identity": 1,
        "column": 2,
        "index_field": 3,
        "relationship_field": 4,
        "kind": 5,
    }
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
            revision=resource.current_revision,
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
