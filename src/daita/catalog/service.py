"""Catalog-owned projections consumed by tools and the data domain."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from .._json import FrozenJsonObject
from .models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSchemaRequest,
    CatalogSearchHit,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSummary,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
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
        active_source_ids = set(await self._active_source_ids(request.agent_id))
        if request.source_ids:
            active_source_ids.intersection_update(request.source_ids)
        return await self._search_active_sources(
            request,
            tuple(sorted(active_source_ids)),
        )

    async def _search_active_sources(
        self,
        request: CatalogSearchRequest,
        active_source_ids: tuple[str, ...],
    ) -> CatalogSearchResult:
        if not active_source_ids:
            return CatalogSearchResult(
                request=request,
                hits=(),
                total_matches=0,
                truncated=False,
            )

        scoped_results = []
        for offset in range(0, len(active_source_ids), 64):
            scoped_results.append(
                await self._store.search(
                    CatalogSearchRequest(
                        agent_id=request.agent_id,
                        query=request.query,
                        source_ids=active_source_ids[offset : offset + 64],
                        resource_kinds=request.resource_kinds,
                        limit=request.limit,
                    )
                )
            )
        ranked_hits = sorted(
            (hit for result in scoped_results for hit in result.hits),
            key=lambda hit: (
                _match_reason_rank(hit.match_reasons),
                -hit.score,
                hit.name,
                hit.resource_id,
            ),
        )
        hits = tuple(ranked_hits[: request.limit])
        total_matches = sum(result.total_matches for result in scoped_results)
        return CatalogSearchResult(
            request=request,
            hits=hits,
            total_matches=total_matches,
            truncated=total_matches > len(hits),
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
            try:
                projection = await self._schema_slice_from_snapshots(
                    request,
                    tuple(snapshots),
                    active_source_ids,
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
            scoped_source_ids = (
                active_source_ids if request.source_id is None else (request.source_id,)
            )
            search_request = CatalogSearchRequest(
                agent_id=request.agent_id,
                query=request.query,
                source_ids=(() if request.source_id is None else (request.source_id,)),
                limit=50,
            )
            search = await self._search_active_sources(
                search_request,
                scoped_source_ids,
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
        resources: dict[str, CatalogResource] = {}
        for resource_id in (*request.from_resource_ids, *request.to_resource_ids):
            resources[resource_id] = await self._active_resource(
                request.agent_id,
                resource_id,
            )

        result = await self._store.traverse(request)
        relationship_ids = tuple(
            dict.fromkeys(
                step.relationship_id for path in result.paths for step in path.steps
            )
        )
        relationships = await self._store.load_relationships(
            request.agent_id,
            relationship_ids,
        )
        relationships_by_id = {
            relationship.id: relationship for relationship in relationships
        }
        if tuple(relationships_by_id) != relationship_ids:
            raise CatalogStoreError(
                "catalog traversal references a non-current relationship"
            )

        paths_payload: list[dict[str, object]] = []
        for path in result.paths:
            steps_payload: list[dict[str, object]] = []
            for step in path.steps:
                relationship = relationships_by_id[step.relationship_id]
                endpoints = await self._current_relationship_endpoints(
                    request.agent_id,
                    relationship,
                    resources,
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


def _match_reason_rank(match_reasons: tuple[str, ...]) -> int:
    for reason, rank in (
        ("resource_name_exact", 0),
        ("resource_name_prefix", 1),
        ("resource_name_contains", 2),
        ("structural_field_exact", 3),
        ("structural_field_contains", 4),
        ("metadata_contains", 5),
        ("relationship_neighbor", 6),
        ("lexical_exact", 0),
        ("lexical_prefix", 1),
        ("lexical_contains", 5),
    ):
        if reason in match_reasons:
            return rank
    return 7


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
