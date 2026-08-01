"""Data-domain projections over catalog-owned structural truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._json import FrozenJsonObject
from ...capabilities import ToolApplicability
from ...catalog.models import FacetKind, ResourceKind
from ...catalog.protocols import CatalogStore
from ...catalog.service import CatalogService
from ...semantics import SemanticResourceFact
from .sql import ResourceSchema

if TYPE_CHECKING:
    from ...adapters.protocols import SourceStore


class CatalogDataView:
    """Translate catalog records without creating a second catalog owner."""

    def __init__(
        self,
        store: CatalogStore,
        service: CatalogService,
        sources: SourceStore,
    ) -> None:
        if not isinstance(store, CatalogStore):
            raise TypeError("store must implement CatalogStore")
        if not isinstance(service, CatalogService):
            raise TypeError("service must be a CatalogService")
        if not callable(getattr(sources, "load_source", None)) or not callable(
            getattr(sources, "list_sources", None)
        ):
            raise TypeError("sources must provide source registration reads")
        self._store = store
        self._service = service
        self._sources = sources

    async def source_routing_facts(
        self,
        agent_id: str,
        configuration_flags: tuple[str, ...],
        source_ids: tuple[str, ...] = (),
    ) -> tuple[FrozenJsonObject, ...]:
        """Project only active, declared source-routing control facts."""

        requested_flags = ToolApplicability(
            required_configuration_flags=configuration_flags
        ).required_configuration_flags
        selected_source_ids = frozenset(source_ids)
        registrations = await self._sources.list_sources(agent_id)
        return tuple(
            FrozenJsonObject.from_mapping(
                {
                    "source_id": registration.id,
                    "adapter_id": registration.adapter_id,
                    "configuration_flags": {
                        flag: registration.configuration.get(flag) is True
                        for flag in requested_flags
                    },
                }
            )
            for registration in sorted(
                registrations,
                key=lambda item: (item.id, item.adapter_id),
            )
            if registration.agent_id == agent_id
            and registration.active
            and (not selected_source_ids or registration.id in selected_source_ids)
        )

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        tabular_resources = await self._service.tabular_resources(agent_id, source_id)
        schemas: list[ResourceSchema] = []
        for item in tabular_resources:
            resource = item.resource
            tabular = item.facet
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
                    source_revision=item.sync.source_revision,
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

        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
        ):
            return None
        return registration.adapter_id

    async def resource_identity(
        self,
        agent_id: str,
        resource_id: str,
    ) -> tuple[str, str, str] | None:
        """Project source, kind, and revision for an active current resource."""

        resource = await self._store.load_resource(agent_id, resource_id)
        if (
            resource is None
            or resource.agent_id != agent_id
            or resource.id != resource_id
            or await self.source_adapter_id(agent_id, resource.source_id) is None
        ):
            return None
        return (
            resource.source_id,
            resource.kind.value,
            resource.current_revision,
        )

    async def semantic_resource_facts(
        self,
        agent_id: str,
        resource_ids: tuple[str, ...],
    ) -> tuple[SemanticResourceFact, ...]:
        """Project current bounded structure for semantic validation and recall."""

        requested = tuple(sorted(set(resource_ids)))
        if len(requested) > 2_048 or any(
            not isinstance(item, str) or not item for item in requested
        ):
            raise ValueError("semantic resource IDs must be bounded non-empty strings")
        registrations = {
            item.id: item
            for item in await self._sources.list_sources(agent_id)
            if item.agent_id == agent_id and item.active
        }
        resources = {
            item.id: item
            for item in await self._store.list_resources(agent_id)
            if item.id in requested and item.source_id in registrations
        }
        fields_by_id: dict[str, tuple[str, ...]] = {}
        for source_id in sorted({item.source_id for item in resources.values()}):
            for schema in await self.resource_schemas(agent_id, source_id):
                fields_by_id[schema.resource_id] = schema.columns
        return tuple(
            SemanticResourceFact(
                resource_id=resource.id,
                source_id=resource.source_id,
                revision=resource.current_revision,
                field_names=fields_by_id.get(resource.id, ()),
            )
            for resource in sorted(resources.values(), key=lambda item: item.id)
        )

    async def is_writable_sqlite_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> bool:
        """Project explicit write admission from the durable source registration."""

        registration = await self._sources.load_source(agent_id, source_id)
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
        source_ids: tuple[str, ...] = (),
        resource_ids: tuple[str, ...] = (),
    ) -> FrozenJsonObject:
        return await self._service.catalog_context(
            agent_id,
            query,
            limit=limit,
            source_ids=source_ids,
            resource_ids=resource_ids,
        )


__all__ = ["CatalogDataView"]
