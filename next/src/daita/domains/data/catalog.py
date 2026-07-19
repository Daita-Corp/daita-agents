"""Data-domain projections over catalog-owned structural truth."""

from __future__ import annotations

from collections.abc import Mapping

from ..._json import FrozenJsonObject
from ...catalog.models import FacetKind, ResourceKind
from ...catalog.protocols import CatalogStore
from ...catalog.service import CatalogService
from .sql import ResourceSchema


class CatalogDataView:
    """Translate catalog records without creating a second catalog owner."""

    def __init__(self, store: CatalogStore, service: CatalogService) -> None:
        if not isinstance(store, CatalogStore):
            raise TypeError("store must implement CatalogStore")
        if not isinstance(service, CatalogService):
            raise TypeError("service must be a CatalogService")
        self._store = store
        self._service = service

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        resources = await self._store.list_resources(agent_id, source_id)
        schemas: list[ResourceSchema] = []
        source_revision_by_sync: dict[str, str | None] = {}
        for resource in resources:
            if resource.kind not in {
                ResourceKind.TABLE,
                ResourceKind.VIEW,
                ResourceKind.FILE,
            }:
                continue
            if resource.current_sync_id not in source_revision_by_sync:
                sync = await self._store.load_sync(
                    agent_id,
                    resource.current_sync_id,
                )
                source_revision_by_sync[resource.current_sync_id] = (
                    None
                    if sync is None or sync.source_id != resource.source_id
                    else sync.source_revision
                )
            facets = await self._store.load_facets(
                agent_id,
                resource.id,
                resource.current_revision,
            )
            tabular = next(
                (facet for facet in facets if facet.kind is FacetKind.TABULAR),
                None,
            )
            if tabular is None:
                continue
            raw_columns = tabular.payload.get("columns", ())
            if not isinstance(raw_columns, tuple):
                continue
            columns = tuple(
                name
                for column in raw_columns
                if isinstance(column, Mapping)
                and isinstance((name := column.get("name")), str)
            )
            aliases = (
                ()
                if resource.native_identity == resource.name
                else (resource.native_identity,)
            )
            schemas.append(
                ResourceSchema(
                    resource_id=resource.id,
                    source_id=resource.source_id,
                    name=resource.name,
                    columns=columns,
                    aliases=aliases,
                    revision=resource.current_revision,
                    source_revision=source_revision_by_sync[resource.current_sync_id],
                )
            )
        return tuple(sorted(schemas, key=lambda item: (item.name, item.resource_id)))

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool:
        resource = await self._store.load_resource(agent_id, resource_id)
        if (
            resource is None
            or resource.agent_id != agent_id
            or resource.source_id != source_id
            or resource.kind is not ResourceKind.FILE
        ):
            return False
        facets = await self._store.load_facets(
            agent_id,
            resource.id,
            resource.current_revision,
        )
        return any(facet.kind is FacetKind.TABULAR for facet in facets)

    async def catalog_context(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int,
    ) -> FrozenJsonObject:
        return await self._service.catalog_context(
            agent_id,
            query,
            limit=limit,
        )


__all__ = ["CatalogDataView"]
