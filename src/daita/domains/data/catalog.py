"""Data-domain projections over catalog-owned structural truth."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..._json import FrozenJsonObject
from ...catalog.models import (
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogTraversalRequest,
    FacetKind,
    ResourceKind,
)
from ...catalog.protocols import CatalogResourceNotFoundError, CatalogStore
from ...catalog.service import CatalogService
from ...semantics import SemanticResourceFact
from ...storage.sqlite_records import (
    PostgreSQLUpdateScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    postgresql_update_authorization_fingerprint,
)
from .sql import ResourceSchema

if TYPE_CHECKING:
    from ...adapters.protocols import SourceStore

    class CatalogPermissionStore(SourceStore, Protocol):
        async def load_source_read_scope(
            self,
            agent_id: str,
            source_id: str,
        ) -> SourceReadScope | None: ...

        async def list_postgresql_update_scopes(
            self,
            agent_id: str,
            source_id: str,
        ) -> tuple[PostgreSQLUpdateScope, ...]: ...


class CatalogDataView:
    """Translate catalog records without creating a second catalog owner."""

    def __init__(
        self,
        store: CatalogStore,
        service: CatalogService,
        sources: CatalogPermissionStore,
    ) -> None:
        if not isinstance(store, CatalogStore):
            raise TypeError("store must implement CatalogStore")
        if not isinstance(service, CatalogService):
            raise TypeError("service must be a CatalogService")
        if not callable(getattr(sources, "load_source", None)) or not callable(
            getattr(sources, "list_sources", None)
        ):
            raise TypeError("sources must provide source registration reads")
        for method_name in (
            "load_source_read_scope",
            "list_postgresql_update_scopes",
        ):
            if not callable(getattr(sources, method_name, None)):
                raise TypeError(f"sources must provide {method_name}")
        self._store = store
        self._service = service
        self._sources = sources

    async def source_routing_facts(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[FrozenJsonObject, ...]:
        """Project only active, declared source-routing control facts."""

        selected_source_ids = frozenset(source_ids)
        registrations = await self._sources.list_sources(agent_id)
        return tuple(
            FrozenJsonObject.from_mapping(
                {
                    "source_id": registration.id,
                    "adapter_id": registration.adapter_id,
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

    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]:
        """Resolve current readable IDs without changing complete catalog truth."""

        selected_source_ids = frozenset(source_ids)
        registrations = tuple(
            registration
            for registration in await self._sources.list_sources(agent_id)
            if registration.agent_id == agent_id
            and registration.active
            and (not selected_source_ids or registration.id in selected_source_ids)
        )
        readable: set[str] = set()
        for registration in registrations:
            scope = await self._sources.load_source_read_scope(
                agent_id,
                registration.id,
            )
            if scope is None:
                raise SourcePermissionStateError(
                    "active source is missing its read scope"
                )
            self._require_owned_read_scope(registration.id, agent_id, scope)
            if scope.mode is SourceReadMode.NONE:
                continue
            current = tuple(
                resource
                for resource in await self._store.list_resources(
                    agent_id,
                    registration.id,
                )
                if resource.agent_id == agent_id
                and resource.source_id == registration.id
            )
            current_ids = {resource.id for resource in current}
            if scope.mode is SourceReadMode.ALL:
                readable.update(current_ids)
            else:
                readable.update(current_ids & set(scope.resource_ids))
        return frozenset(readable)

    @staticmethod
    def _require_owned_read_scope(
        source_id: str,
        agent_id: str,
        scope: SourceReadScope,
    ) -> None:
        if (
            not isinstance(scope, SourceReadScope)
            or scope.agent_id != agent_id
            or scope.source_id != source_id
        ):
            raise SourcePermissionStateError(
                "active source read scope ownership is invalid"
            )

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        readable = await self.readable_resource_ids(
            request.agent_id,
            request.source_ids,
        )
        return await self._service.search(
            request,
            readable_resource_ids=readable,
        )

    async def schema_slice(self, request: CatalogSchemaRequest) -> FrozenJsonObject:
        readable = await self.readable_resource_ids(
            request.agent_id,
            (() if request.source_id is None else (request.source_id,)),
        )
        return await self._service.schema_slice(
            request,
            readable_resource_ids=readable,
        )

    async def inspect_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> FrozenJsonObject:
        readable = await self.readable_resource_ids(agent_id)
        return await self._service.inspect_resource(
            agent_id,
            resource_id,
            readable_resource_ids=readable,
        )

    async def traverse(
        self,
        request: CatalogTraversalRequest,
    ) -> FrozenJsonObject:
        readable = await self.readable_resource_ids(request.agent_id)
        return await self._service.traverse(
            request,
            readable_resource_ids=readable,
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
            ordered_primary_key_columns = tuple(
                name for _, name in sorted(primary_key_columns)
            )
            column_nullability: tuple[tuple[str, bool], ...] = tuple(
                (name, nullable)
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
                and isinstance((nullable := column.get("nullable")), bool)
            )
            column_type_provenance: tuple[tuple[str, str, str], ...] = tuple(
                (name, namespace, native_name)
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
                and isinstance((namespace := column.get("native_type_namespace")), str)
                and isinstance((native_name := column.get("native_type_name")), str)
            )
            identity_columns = tuple(
                name
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
                and column.get("identity") is True
            )
            generated_columns = tuple(
                name
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
                and column.get("generated") is True
            )
            updatable_columns = tuple(
                name
                for column in raw_columns
                if isinstance(column, FrozenJsonObject)
                and isinstance((name := column.get("name")), str)
                and column.get("updatable") is True
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
                    primary_key_columns=ordered_primary_key_columns,
                    column_declared_types=column_declared_types,
                    column_nullability=column_nullability,
                    column_type_provenance=column_type_provenance,
                    identity_columns=identity_columns,
                    generated_columns=generated_columns,
                    updatable_columns=updatable_columns,
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
        readable = await self.readable_resource_ids(agent_id)
        registrations = {
            item.id: item
            for item in await self._sources.list_sources(agent_id)
            if item.agent_id == agent_id and item.active
        }
        resources = {
            item.id: item
            for item in await self._store.list_resources(agent_id)
            if item.id in requested
            and item.id in readable
            and item.source_id in registrations
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

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool:
        if resource_id not in await self.readable_resource_ids(
            agent_id,
            (source_id,),
        ):
            return False
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
        readable = await self.readable_resource_ids(agent_id, source_ids)
        if resource_ids and any(
            resource_id not in readable for resource_id in resource_ids
        ):
            raise CatalogResourceNotFoundError(agent_id, resource_ids[0])
        return await self._service.catalog_context(
            agent_id,
            query,
            limit=limit,
            source_ids=source_ids,
            resource_ids=resource_ids,
            readable_resource_ids=readable,
        )

    async def postgresql_update_scope_issue(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> tuple[str, str] | None:
        """Validate one exact current update authorization without source I/O."""

        readable = await self.readable_resource_ids(agent_id, (source_id,))
        if resource_id not in readable:
            return (
                "resource_read_not_allowed",
                "The requested resource is not available for reading.",
            )
        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
            or registration.adapter_id != "postgresql"
        ):
            return (
                "resource_update_not_allowed",
                "The requested resource is not authorized for PostgreSQL updates.",
            )
        current = next(
            (
                item
                for item in await self._service.tabular_resources(agent_id, source_id)
                if item.resource.id == resource_id
                and item.resource.kind is ResourceKind.TABLE
            ),
            None,
        )
        if current is None:
            return (
                "resource_update_not_allowed",
                "The requested resource is not authorized for PostgreSQL updates.",
            )
        scopes = await self._sources.list_postgresql_update_scopes(
            agent_id,
            source_id,
        )
        scope = next(
            (item for item in scopes if item.resource_id == resource_id),
            None,
        )
        if scope is None:
            return (
                "resource_update_not_allowed",
                "The requested resource is not authorized for PostgreSQL updates.",
            )
        if (
            not isinstance(scope, PostgreSQLUpdateScope)
            or scope.agent_id != agent_id
            or scope.source_id != source_id
        ):
            raise SourcePermissionStateError(
                "stored PostgreSQL update scope ownership is invalid"
            )
        requested_columns = frozenset(assignment_columns)
        if not requested_columns <= frozenset(scope.allowed_assignment_columns):
            return (
                "update_column_not_allowed",
                "One or more assignment columns are not authorized for this table.",
            )
        try:
            expected = postgresql_update_authorization_fingerprint(
                source=registration,
                resource=current.resource,
                facet=current.facet,
                allowed_assignment_columns=scope.allowed_assignment_columns,
            )
        except (TypeError, ValueError):
            expected = None
        if scope.authorization_fingerprint != expected:
            return (
                "resource_update_scope_stale",
                "The PostgreSQL update scope is stale; configure source permissions again.",
            )
        return None

    async def postgresql_update_applicable_source_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]:
        selected = frozenset(source_ids)
        applicable: set[str] = set()
        for registration in await self._sources.list_sources(agent_id):
            if (
                registration.agent_id != agent_id
                or not registration.active
                or registration.adapter_id != "postgresql"
                or (selected and registration.id not in selected)
            ):
                continue
            for scope in await self._sources.list_postgresql_update_scopes(
                agent_id,
                registration.id,
            ):
                issue = await self.postgresql_update_scope_issue(
                    agent_id,
                    registration.id,
                    scope.resource_id,
                    scope.allowed_assignment_columns,
                )
                if issue is None:
                    applicable.add(registration.id)
                    break
        return frozenset(applicable)


__all__ = ["CatalogDataView"]
