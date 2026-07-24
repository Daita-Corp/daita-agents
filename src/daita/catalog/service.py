"""Catalog-owned projections consumed by tools and the data domain."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from .._json import FrozenJsonObject
from .models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSummary,
    CatalogTraversalRequest,
    FacetKind,
    RelationshipDirection,
    RelationshipKind,
    ResourceKind,
)
from .protocols import CatalogResourceNotFoundError, CatalogStore, CatalogStoreError

if TYPE_CHECKING:
    from ..adapters.protocols import SourceStore

_INSPECT_INCIDENT_RELATIONSHIP_LIMIT = 50


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
        active_source_ids = {
            registration.id
            for registration in await self._sources.list_sources(request.agent_id)
            if registration.agent_id == request.agent_id and registration.active
        }
        if request.source_ids:
            active_source_ids.intersection_update(request.source_ids)
        if not active_source_ids:
            return CatalogSearchResult(
                request=request,
                hits=(),
                total_matches=0,
                truncated=False,
            )

        ordered_source_ids = tuple(sorted(active_source_ids))
        scoped_results = []
        for offset in range(0, len(ordered_source_ids), 64):
            scoped_results.append(
                await self._store.search(
                    CatalogSearchRequest(
                        agent_id=request.agent_id,
                        query=request.query,
                        source_ids=ordered_source_ids[offset : offset + 64],
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
    if "lexical_exact" in match_reasons:
        return 0
    if "lexical_prefix" in match_reasons:
        return 1
    return 2


def _neighbor_payload(resource: CatalogResource) -> dict[str, object]:
    return {
        "kind": resource.kind.value,
        "name": resource.name,
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
