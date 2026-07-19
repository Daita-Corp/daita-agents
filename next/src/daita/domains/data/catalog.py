"""Data-domain projections over catalog-owned structural truth."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from ..._json import FrozenJsonObject
from ...adapters.models import SourceRegistration
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
            columns: tuple[str, ...] = tuple(
                name
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
            )
            column_declared_types: tuple[tuple[str, str], ...] = tuple(
                (name, native_type)
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
                and isinstance((native_type := column.get("native_type")), str)
            )
            primary_key_columns: tuple[tuple[int, str], ...] = tuple(
                (ordinal, name)
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
                and isinstance(
                    (ordinal := column.get("primary_key_ordinal")),
                    int,
                )
                and not isinstance(ordinal, bool)
                and ordinal > 0
            )
            unique_key_columns: list[str] = []
            if len(primary_key_columns) == 1 and primary_key_columns[0][0] == 1:
                unique_key_columns.append(primary_key_columns[0][1])
            raw_indexes = tabular.payload.get("indexes", ())
            if isinstance(raw_indexes, tuple):
                for index in raw_indexes:
                    if (
                        not isinstance(index, FrozenJsonObject)
                        or index.get("unique") is not True
                        or index.get("predicate") is not None
                    ):
                        continue
                    index_columns = index.get("columns")
                    if (
                        isinstance(index_columns, tuple)
                        and len(index_columns) == 1
                        and isinstance(index_columns[0], str)
                        and index_columns[0] in columns
                        and index_columns[0] not in unique_key_columns
                    ):
                        unique_key_columns.append(index_columns[0])
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
                    resource_kind=resource.kind.value,
                    sensitivity_class=resource.sensitivity.value,
                    writable=resource.kind is ResourceKind.TABLE,
                    unique_key_columns=tuple(unique_key_columns),
                    column_declared_types=column_declared_types,
                )
            )
        return tuple(sorted(schemas, key=lambda item: (item.name, item.resource_id)))

    async def source_adapter_id(
        self,
        agent_id: str,
        source_id: str,
    ) -> str | None:
        """Return the durable attached-source adapter identity, if current."""

        load_source = getattr(self._store, "load_source", None)
        if not callable(load_source):
            return None
        typed_load_source = cast(
            Callable[[str, str], Awaitable[SourceRegistration | None]],
            load_source,
        )
        registration = await typed_load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
        ):
            return None
        return registration.adapter_id

    async def is_writable_sqlite_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> bool:
        """Project explicit write admission from the durable source registration."""

        load_source = getattr(self._store, "load_source", None)
        if not callable(load_source):
            return False
        typed_load_source = cast(
            Callable[[str, str], Awaitable[SourceRegistration | None]],
            load_source,
        )
        registration = await typed_load_source(agent_id, source_id)
        return bool(
            registration is not None
            and registration.agent_id == agent_id
            and registration.id == source_id
            and registration.active
            and registration.adapter_id == "sqlite"
            and registration.configuration.get("write_access") is True
        )

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
