"""Catalog-owned projections consumed by tools and the data domain."""

from __future__ import annotations

from .._json import FrozenJsonObject
from .models import (
    CatalogFacet,
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
)
from .protocols import CatalogResourceNotFoundError, CatalogStore


class CatalogService:
    """Keep catalog truth and its bounded projections behind one owner."""

    def __init__(self, store: CatalogStore) -> None:
        if not isinstance(store, CatalogStore):
            raise TypeError("store must implement CatalogStore")
        self._store = store

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        return await self._store.search(request)

    async def inspect_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> FrozenJsonObject:
        resource = await self._store.load_resource(agent_id, resource_id)
        if resource is None:
            raise CatalogResourceNotFoundError(agent_id, resource_id)
        facets = await self._store.load_facets(
            agent_id,
            resource.id,
            resource.current_revision,
        )
        return FrozenJsonObject.from_mapping(
            {
                "resource": _resource_payload(resource),
                "facets": [_facet_payload(facet) for facet in facets],
                "trust_classification": "untrusted_external_data",
            }
        )

    async def catalog_context(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int,
    ) -> FrozenJsonObject:
        result = await self.search(
            CatalogSearchRequest(
                agent_id=agent_id,
                query=query,
                limit=min(limit, 50),
            )
        )
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
                    for hit in result.hits
                ],
                "total_matches": result.total_matches,
                "truncated": result.truncated,
                "trust_classification": "untrusted_external_data",
            }
        )


def _resource_payload(resource: CatalogResource) -> dict[str, object]:
    return {
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
        "schema_version": facet.schema_version,
    }


__all__ = ["CatalogService"]
