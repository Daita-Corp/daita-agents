"""Project catalog resources and relationships into data-domain lookup views."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import cast

from ..._json import FrozenJsonObject
from ...adapters.folder import SessionFolderSource
from ...adapters.models import SourceRegistration
from ...adapters.protocols import (
    DESCRIBE_MAX_COLUMNS,
    SUMMARY_MAX_COLUMNS,
    SourcePermissionStore,
    column_summary,
)
from ...adapters.registry import SourceOperation, SourceTypeRegistry
from ...catalog.models import (
    CATALOG_SCHEMA_TOOL_DEFAULT_RELATIONSHIPS,
    CatalogResource,
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogTraversalRequest,
    ResourceKind,
    Sensitivity,
    catalog_resource_id,
)
from ...catalog.protocols import CatalogStore, CatalogStoreError
from ...catalog.service import CatalogService
from ...llm.models import ModelSensitivity
from ...semantics import SemanticResourceFact
from ...storage.sqlite_records import (
    RelationalWriteScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
)
from .sql import ResourceSchema

_CATALOG_PARTIAL = (
    "The catalog holds only part of this source, so the object count is a floor. "
    "catalog_search and catalog_describe read the source live."
)


# The source operation each relational write-scope operation requires.
class CatalogDataView:
    """Translate catalog records without creating a second catalog owner."""

    def __init__(
        self,
        store: CatalogStore,
        service: CatalogService,
        sources: SourcePermissionStore,
        source_types: SourceTypeRegistry,
        session_source: SessionFolderSource | None = None,
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
            "list_relational_write_scopes",
        ):
            if not callable(getattr(sources, method_name, None)):
                raise TypeError(f"sources must provide {method_name}")
        if not isinstance(source_types, SourceTypeRegistry):
            raise TypeError("source_types must be a SourceTypeRegistry")
        self._store = store
        self._service = service
        self._sources = sources
        self._source_types = source_types
        self._session_source = session_source

    @property
    def session_source_id(self) -> str | None:
        """The live, never-stored session folder source, if this host has one."""

        return None if self._session_source is None else self._session_source.source_id

    def _available(self, registration: SourceRegistration, agent_id: str) -> bool:
        """An active owned source whose type is registered in this composition."""

        return (
            registration.agent_id == agent_id
            and registration.active
            and self._source_types.get(registration.adapter_id) is not None
        )

    async def source_routing_facts(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[FrozenJsonObject, ...]:
        """Project only active, declared source-routing control facts.

        The session folder source is always readable by its foreground user
        and sorts after every stored source.
        """

        selected_source_ids = frozenset(source_ids)
        registrations = await self._sources.list_sources(agent_id)
        readable_sources: set[str] = set()
        for registration in registrations:
            if not self._available(registration, agent_id):
                continue
            if selected_source_ids and registration.id not in selected_source_ids:
                continue
            scope = await self._sources.load_source_read_scope(
                agent_id, registration.id
            )
            if scope is None:
                raise SourcePermissionStateError(
                    "active source is missing its read scope"
                )
            self._require_owned_read_scope(registration.id, agent_id, scope)
            if scope.mode is not SourceReadMode.NONE:
                readable_sources.add(registration.id)
        return tuple(
            FrozenJsonObject.from_mapping(
                {
                    "source_id": registration.id,
                    "adapter_id": registration.adapter_id,
                    "display_name": registration.display_name,
                    "summary": registration.summary,
                    "when_to_use": registration.when_to_use,
                    "keywords": registration.keywords,
                    "presentation_sensitivity": registration.presentation_sensitivity.value,
                }
            )
            for registration in sorted(
                registrations,
                key=lambda item: (item.id, item.adapter_id),
            )
            if self._available(registration, agent_id)
            and registration.id in readable_sources
            and (not selected_source_ids or registration.id in selected_source_ids)
        ) + (
            (self._session_source.routing_fact(),)
            if self._session_source is not None
            and (
                not selected_source_ids
                or self._session_source.source_id in selected_source_ids
            )
            else ()
        )

    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]:
        """Resolve current readable IDs without changing complete catalog truth."""

        resources = await self._admitted_resources(agent_id, source_ids)
        return frozenset(resource.id for resource in resources)

    async def non_table_resource_ids(self, agent_id: str) -> Mapping[str, str]:
        """Each view or file's ID, keyed by the ID its name hashes to as a table."""

        return {
            catalog_resource_id(
                resource.source_id, ResourceKind.TABLE, resource.native_identity
            ): resource.id
            for resource in await self._store.list_resources(agent_id)
            if resource.kind is not ResourceKind.TABLE
        }

    async def source_read_access(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
    ) -> Mapping[str, frozenset[str] | None]:
        """Each source's stored read scope: None when it may read in full.

        Reads permission records only, so a failing catalog cannot affect it.
        """

        access: dict[str, frozenset[str] | None] = {}
        for registration in await self._active_registrations(agent_id, source_ids):
            scope = await self._sources.load_source_read_scope(
                agent_id, registration.id
            )
            if scope is None:
                raise SourcePermissionStateError(
                    "active source is missing its read scope"
                )
            self._require_owned_read_scope(registration.id, agent_id, scope)
            access[registration.id] = (
                None
                if scope.mode is SourceReadMode.ALL
                else frozenset(scope.resource_ids)
            )
        return access

    async def resource_names(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> Mapping[str, str]:
        """Map each current readable resource ID to the native name the model uses."""

        resources = await self._admitted_resources(agent_id, source_ids)
        return {resource.id: resource.native_identity for resource in resources}

    async def admitted_model_sensitivity(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> ModelSensitivity | None:
        """Classify the exact current readable scope, or fail closed with None."""

        try:
            await self.source_read_access(agent_id, source_ids)
        except SourcePermissionStateError:
            return None
        sensitivities: list[ModelSensitivity] = []
        sensitivities.extend(
            registration.presentation_sensitivity
            for registration in await self._sources.list_sources(agent_id)
            if registration.agent_id == agent_id
            and registration.active
            and (not source_ids or registration.id in source_ids)
        )
        if self._session_source is not None and (
            not source_ids or self._session_source.source_id in source_ids
        ):
            sensitivities.append(self._session_source.presentation_sensitivity)
        # A source's classification covers every table in it, cataloged or not.
        if not sensitivities:
            return ModelSensitivity.PUBLIC
        return max(sensitivities, key=lambda item: item.routing_rank)

    async def _admitted_resources(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
    ) -> tuple[CatalogResource, ...]:
        """Resolve the cataloged resources catalog tools and routing may show."""

        admitted: list[CatalogResource] = []
        for registration in await self._active_registrations(agent_id, source_ids):
            scope = await self._sources.load_source_read_scope(
                agent_id,
                registration.id,
            )
            if scope is None:
                raise SourcePermissionStateError(
                    "active source is missing its read scope"
                )
            self._require_owned_read_scope(registration.id, agent_id, scope)
            current = tuple(
                resource
                for resource in await self._store.list_resources(
                    agent_id,
                    registration.id,
                )
                if resource.agent_id == agent_id
                and resource.source_id == registration.id
            )
            if scope.mode is SourceReadMode.ALL:
                admitted.extend(current)
            else:
                # A stored ID for a table that no longer exists simply never matches.
                selected = frozenset(scope.resource_ids)
                admitted.extend(item for item in current if item.id in selected)
        return tuple(admitted)

    async def _active_registrations(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> tuple[SourceRegistration, ...]:
        # The uncataloged session source has no registration or stored scope.
        selected_source_ids = frozenset(source_ids) - {self.session_source_id}
        if source_ids and not selected_source_ids:
            return ()
        registrations = tuple(
            registration
            for registration in await self._sources.list_sources(agent_id)
            if self._available(registration, agent_id)
            and (not selected_source_ids or registration.id in selected_source_ids)
        )
        if selected_source_ids and selected_source_ids != {
            item.id for item in registrations
        }:
            raise SourcePermissionStateError(
                "selected source scope is not active for this agent"
            )
        return registrations

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

    async def lookup_sources(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> tuple[tuple[str, str, str | None], ...]:
        """(source ID, adapter ID, SQL dialect) of each current readable source."""

        facts = await self.source_routing_facts(agent_id, source_ids)
        return tuple(
            (
                cast(str, fact["source_id"]),
                adapter := cast(str, fact["adapter_id"]),
                (
                    None
                    if (source_type := self._source_types.get(adapter)) is None
                    else source_type.sql_dialect
                ),
            )
            for fact in facts
        )

    async def _catalog_extent(
        self, agent_id: str, source_id: str
    ) -> tuple[int | None, bool]:
        """(object count, whether the catalog holds only part of the source)."""

        count = 0
        partial = False
        try:
            for ref in await self._store.list_current_snapshot_refs(
                agent_id, (source_id,)
            ):
                snapshot = await self._store.load_current_snapshot(ref)
                if snapshot is None:  # a sync replaced it while reading
                    return None, False
                count += len(snapshot.resources)
                partial = partial or snapshot.sync.truncated
        except CatalogStoreError:
            return None, False
        return count, partial

    async def source_directory(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> tuple[FrozenJsonObject, ...]:
        """Each source's routing facts plus type, dialect, schemas and object count.

        The count advises: a failing catalog leaves it, and the note that the
        catalog holds only part of the source, unknown.
        """

        directory: list[FrozenJsonObject] = []
        for fact in await self.source_routing_facts(agent_id, source_ids):
            source_id = cast(str, fact["source_id"])
            source_type = self._source_types.get(cast(str, fact["adapter_id"]))
            count, partial = await self._catalog_extent(agent_id, source_id)
            directory.append(
                FrozenJsonObject.from_mapping(
                    {
                        **fact.to_dict(),
                        "type": None if source_type is None else source_type.label,
                        "dialect": (
                            None if source_type is None else source_type.sql_dialect
                        ),
                        "schemas": await self.source_schemas(agent_id, source_id),
                        "approximate_objects": count,
                        **({"catalog_partial": _CATALOG_PARTIAL} if partial else {}),
                    }
                )
            )
        return tuple(directory)

    async def catalog_hits(
        self, request: CatalogSearchRequest
    ) -> tuple[Mapping[str, object], ...]:
        """Ranked matches from the catalog snapshot, with their column summaries."""

        readable = await self.readable_resource_ids(
            request.agent_id, request.source_ids
        )
        result = await self._service.search(request, readable_resource_ids=readable)
        names = await self.resource_names(request.agent_id, request.source_ids)
        matched = [
            hit for hit in result.hits if "unmatched_fallback" not in hit.match_reasons
        ]
        tabular = {
            item.resource.id: item.tabular
            for source_id in {hit.source_id for hit in matched}
            for item in await self._service.tabular_resources(
                request.agent_id, source_id
            )
        }
        hits: list[Mapping[str, object]] = []
        for hit in matched:
            facet = tabular.get(hit.resource_id)
            columns = () if facet is None else facet.columns
            hits.append(
                {
                    "source_id": hit.source_id,
                    "name": names.get(hit.resource_id, hit.name),
                    "kind": hit.kind.value,
                    "columns": column_summary(
                        ", ".join(
                            f"{column.name} {column.native_type}"
                            for column in columns[:SUMMARY_MAX_COLUMNS]
                        ),
                        len(columns),
                    ),
                    "row_estimate": (
                        None if facet is None else facet.row_count_estimate
                    ),
                }
            )
        return tuple(hits)

    async def cached_description(
        self, agent_id: str, source_id: str, name: str
    ) -> Mapping[str, object] | None:
        """The catalog's last description of one readable table, or None."""

        if source_id == self.session_source_id:
            return None
        readable = await self.readable_resource_ids(agent_id, (source_id,))
        folded = name.lower()
        current = tuple(
            item
            for item in await self._service.tabular_resources(agent_id, source_id)
            if item.resource.id in readable
        )
        found = next(
            (
                item
                for spelling in (name, folded, f"main.{folded}")
                for item in current
                if item.resource.native_identity == spelling
            ),
            None,
        )
        if found is None:
            suffixed = [
                item
                for item in current
                if item.resource.native_identity.endswith(f".{name}")
            ]
            if len(suffixed) != 1:
                return None
            found = suffixed[0]
        facet = found.tabular
        return {
            "name": found.resource.native_identity,
            "kind": found.resource.kind.value,
            "revision": found.resource.current_revision,
            "comment": None,
            "row_estimate": facet.row_count_estimate,
            "partitions": 0,
            "columns": [
                {
                    "name": column.name,
                    "type": column.native_type,
                    "nullable": column.nullable,
                    **({"comment": column.comment} if column.comment else {}),
                }
                for column in facet.columns[:DESCRIBE_MAX_COLUMNS]
            ],
            "columns_truncated": len(facet.columns) > DESCRIBE_MAX_COLUMNS,
            "primary_key": [
                column.name
                for column in sorted(
                    (item for item in facet.columns if item.primary_key_ordinal),
                    key=lambda item: cast(int, item.primary_key_ordinal),
                )
            ],
            "unique_keys": [
                list(index.columns)
                for index in facet.indexes
                if index.unique and index.predicate is None
            ],
            "relationships": await self._cataloged_relationships(
                agent_id, source_id, found.resource.id
            ),
        }

    async def _cataloged_relationships(
        self, agent_id: str, source_id: str, resource_id: str
    ) -> list[Mapping[str, object]]:
        """Relationships by name, including links declared across sources."""

        # Every source's readable set: a declared link needs its far endpoint.
        projection = await self._service.schema_slice(
            CatalogSchemaRequest(
                agent_id=agent_id,
                resource_ids=(resource_id,),
                source_id=source_id,
                limit=1,
                include_relationships=True,
                max_join_depth=1,
                relationship_limit=CATALOG_SCHEMA_TOOL_DEFAULT_RELATIONSHIPS,
            ),
            readable_resource_ids=await self.readable_resource_ids(agent_id),
        )
        known = {
            item.id: item
            for item in await self._store.list_resources(agent_id)
            if item.agent_id == agent_id
        }
        relationships: list[Mapping[str, object]] = []
        for raw in cast(
            tuple[Mapping[str, object], ...], projection.get("relationships", ())
        ):
            start = known.get(cast(str, raw.get("from_resource_id")))
            end = known.get(cast(str, raw.get("to_resource_id")))
            if start is None or end is None:
                continue
            relationships.append(
                {
                    "from": start.native_identity,
                    "from_source_id": start.source_id,
                    "from_kind": start.kind.value,
                    "to": end.native_identity,
                    "to_source_id": end.source_id,
                    "to_kind": end.kind.value,
                    "columns": [
                        [pair["source_field"], pair["target_field"]]
                        for pair in cast(
                            tuple[Mapping[str, object], ...],
                            raw.get("field_pairs", ()),
                        )
                    ],
                    "provenance": raw.get("provenance"),
                }
            )
        return relationships

    async def schema_slice(
        self,
        request: CatalogSchemaRequest,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> FrozenJsonObject:
        readable = await self.readable_resource_ids(
            request.agent_id,
            (() if request.source_id is None else (request.source_id,)),
        )
        if readable_resource_ids is not None:
            readable &= readable_resource_ids
        return await self._service.schema_slice(
            request,
            readable_resource_ids=readable,
        )

    async def inspect_resource(
        self,
        agent_id: str,
        resource_id: str,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> FrozenJsonObject:
        readable = await self.readable_resource_ids(agent_id)
        if readable_resource_ids is not None:
            readable &= readable_resource_ids
        return await self._service.inspect_resource(
            agent_id,
            resource_id,
            readable_resource_ids=readable,
        )

    async def traverse(
        self,
        request: CatalogTraversalRequest,
        *,
        readable_resource_ids: frozenset[str] | None = None,
    ) -> FrozenJsonObject:
        readable = await self.readable_resource_ids(request.agent_id)
        if readable_resource_ids is not None:
            readable &= readable_resource_ids
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
            tabular = item.tabular
            columns = tuple(column.name for column in tabular.columns)
            column_declared_types = tuple(
                (column.name, column.native_type) for column in tabular.columns
            )
            primary_key_columns = tuple(
                (column.primary_key_ordinal, column.name)
                for column in tabular.columns
                if column.primary_key_ordinal is not None
            )
            ordered_primary_key_columns = tuple(
                name for _, name in sorted(primary_key_columns)
            )
            column_nullability = tuple(
                (column.name, column.nullable) for column in tabular.columns
            )
            column_type_provenance = tuple(
                (
                    column.name,
                    column.native_type_namespace,
                    column.native_type_name,
                )
                for column in tabular.columns
                if column.native_type_namespace is not None
                and column.native_type_name is not None
            )
            identity_column_names = [
                column.name for column in tabular.columns if column.identity
            ]
            generated_column_names = [
                column.name for column in tabular.columns if column.generated
            ]
            updatable_column_names = [
                column.name for column in tabular.columns if column.updatable
            ]
            unique_key_columns: list[str] = []
            if len(primary_key_columns) == 1 and primary_key_columns[0][0] == 1:
                unique_key_columns.append(primary_key_columns[0][1])
            for index in tabular.indexes:
                if not index.unique or index.predicate is not None:
                    continue
                if (
                    len(index.columns) == 1
                    and index.columns[0] in columns
                    and index.columns[0] not in unique_key_columns
                ):
                    unique_key_columns.append(index.columns[0])
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
                    conflict_keys=tuple(
                        index.columns
                        for index in tabular.indexes
                        if index.unique
                        and index.predicate is None
                        and index.write_conflict_supported
                    ),
                    column_defaults=tuple(
                        (column.name, column.default_expression)
                        for column in tabular.columns
                    ),
                    column_collations=tuple(
                        (column.name, column.collation) for column in tabular.columns
                    ),
                    identity_columns=tuple(identity_column_names),
                    generated_columns=tuple(generated_column_names),
                    updatable_columns=tuple(updatable_column_names),
                )
            )
        return tuple(sorted(schemas, key=lambda item: (item.name, item.resource_id)))

    async def source_adapter_id(
        self,
        agent_id: str,
        source_id: str,
    ) -> str | None:
        """Return the source's adapter identity, if current."""

        if self._session_source is not None and source_id == self.session_source_id:
            return self._session_source.adapter_id
        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.id != source_id
            or not self._available(registration, agent_id)
        ):
            return None
        return registration.adapter_id

    async def source_schemas(self, agent_id: str, source_id: str) -> tuple[str, ...]:
        """The source's configured schemas in search order; empty when it has none."""

        registration = await self._sources.load_source(agent_id, source_id)
        schemas = (
            None if registration is None else registration.configuration.get("schemas")
        )
        return (
            tuple(cast(tuple[str, ...], schemas)) if isinstance(schemas, tuple) else ()
        )

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

    async def resource_revision_fact(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> tuple[str, str, ModelSensitivity] | None:
        """Return one exact current readable resource-revision observation fact."""

        readable = await self.readable_resource_ids(agent_id, (source_id,))
        if resource_id not in readable:
            return None
        resource = await self._store.load_resource(agent_id, resource_id)
        if (
            resource is None
            or resource.agent_id != agent_id
            or resource.source_id != source_id
            or resource.sensitivity is Sensitivity.UNKNOWN
        ):
            return None
        return (
            resource.current_revision,
            "sha256:" + sha256(resource.current_sync_id.encode("utf-8")).hexdigest(),
            ModelSensitivity(resource.sensitivity.value),
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

    async def relational_write_scopes(
        self, agent_id: str, source_id: str
    ) -> Mapping[str, RelationalWriteScope]:
        """One source's stored write scopes by table ID: permission records only."""

        return {
            scope.resource_id: scope
            for scope in await self._sources.list_relational_write_scopes(
                agent_id, source_id
            )
        }

    async def relational_write_applicable_source_ids(
        self, agent_id: str, source_ids: tuple[str, ...] = ()
    ) -> frozenset[str]:
        """Active writable sources with at least one stored write scope."""

        selected = frozenset(source_ids)
        applicable: set[str] = set()
        for registration in await self._sources.list_sources(agent_id):
            if (
                self._available(registration, agent_id)
                and self._source_types.supports(
                    registration.adapter_id, SourceOperation.RELATIONAL_WRITE
                )
                and (not selected or registration.id in selected)
                and await self._sources.list_relational_write_scopes(
                    agent_id, registration.id
                )
            ):
                applicable.add(registration.id)
        return frozenset(applicable)


__all__ = ["CatalogDataView"]
