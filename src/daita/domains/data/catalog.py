"""Project catalog resources and relationships into data-domain lookup views."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from difflib import get_close_matches
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
from ...catalog.search_text import normalize_search_text, search_key
from ...catalog.service import TABULAR_KINDS, CatalogService, ReadableScope
from ...llm.models import ModelSensitivity
from ...semantics import (
    SemanticResourceFact,
    SemanticStore,
    inspect_semantic_annotations,
)
from ...storage.sqlite_records import (
    RelationalWriteScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
)
from .sql import ResourceSchema

_CATALOG_PARTIAL = (
    "The catalog lists this source's objects but describes only some of them. "
    "catalog_search and catalog_describe read the source live."
)


_KEY_SUFFIXES = ("_id", "_key", "_code", "_uuid")
_INTEGERS = frozenset(
    "int integer tinyint smallint mediumint bigint int2 int4 int8 serial bigserial "
    "smallserial numeric decimal number".split()
)
_TEXTS = frozenset("text varchar char character nvarchar nchar string citext".split())


def _type_family(native_type: str) -> str | None:
    """Integers and numerics together, text together, anything else exact."""

    folded = re.sub(r"\(.*?\)", "", native_type.lower()).strip()
    word = folded.split(" ", 1)[0]
    if not folded or folded == "unknown":
        return None
    if word in _INTEGERS:
        return "numeric"
    return "text" if word in _TEXTS else folded


def _compatible(left: str, right: str) -> bool:
    family = _type_family(left)
    return family is not None and family == _type_family(right)


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
        resource_ids: tuple[str, ...] | None = None,
        first: int | None = None,
    ) -> frozenset[str]:
        """Resolve current readable IDs without changing complete catalog truth.

        Every readable ID, only those of ``resource_ids`` that are readable, or
        the ``first`` smallest readable IDs.
        """

        found = sorted(
            item
            for item, _ in await self._admitted(
                agent_id, source_ids, resource_ids, first
            )
        )
        return frozenset(found if first is None else found[:first])

    async def non_table_resource_ids(self, agent_id: str) -> Mapping[str, str]:
        """Each view or file's ID, keyed by the ID its name hashes to as a table."""

        return {
            catalog_resource_id(source_id, ResourceKind.TABLE, name): resource_id
            for resource_id, source_id, name in await self._store.list_resource_names(
                agent_id, (), other_than=ResourceKind.TABLE
            )
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
        resource_ids: tuple[str, ...] | None = None,
    ) -> Mapping[str, str]:
        """Map current readable resource IDs to the native name the model uses.

        Every readable resource, or only ``resource_ids`` when given.
        """

        return dict(await self._admitted(agent_id, source_ids, resource_ids))

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

    async def _admitted(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
        resource_ids: tuple[str, ...] | None = None,
        first: int | None = None,
    ) -> list[tuple[str, str]]:
        """(ID, native name) of the readable resources, or of those of ``resource_ids``.

        A source read in full is listed by columns alone; a source with chosen
        tables loads only those, and a stale chosen ID never matches.
        """

        access = await self.source_read_access(agent_id, source_ids)
        admitted: list[tuple[str, str]] = []
        chosen = resource_ids
        if chosen is None:
            open_sources = tuple(
                sorted(item for item, ids in access.items() if ids is None)
            )
            if open_sources:
                admitted.extend(
                    (resource_id, name)
                    for resource_id, _, name in await self._store.list_resource_names(
                        agent_id, open_sources, limit=first
                    )
                )
            chosen = tuple(
                sorted({item for ids in access.values() if ids for item in ids})
            )
        for entry in await self._store.load_entries(agent_id, chosen) if chosen else ():
            resource = entry.resource
            if (
                resource.agent_id == agent_id
                and resource.source_id in access
                and (
                    (resource_ids is not None and access[resource.source_id] is None)
                    or (
                        access[resource.source_id] is not None
                        and resource.id
                        in cast(frozenset[str], access[resource.source_id])
                    )
                )
            ):
                admitted.append((resource.id, resource.native_identity))
        return admitted

    async def _service_readable(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> frozenset[str] | None:
        """The readable scope the catalog service filters by: None when nothing
        is restricted, otherwise the chosen tables plus the sources read in full
        (never a listing of those)."""

        access = await self.source_read_access(agent_id, source_ids)
        open_sources = frozenset(item for item, ids in access.items() if ids is None)
        if (
            access
            and len(open_sources) == len(access)
            and (
                source_ids
                or all(
                    registration.id in access
                    for registration in await self._sources.list_sources(agent_id)
                    if registration.agent_id == agent_id and registration.active
                )
            )
        ):
            return None
        chosen = tuple(sorted({item for ids in access.values() if ids for item in ids}))
        return ReadableScope(
            (
                entry.resource.id
                for entry in (
                    await self._store.load_entries(agent_id, chosen) if chosen else ()
                )
                if entry.resource.agent_id == agent_id
                and entry.resource.id in (access.get(entry.resource.source_id) or ())
            ),
            open_sources,
        )

    @staticmethod
    def _admits(
        access: Mapping[str, frozenset[str] | None], resource: CatalogResource
    ) -> bool:
        if resource.source_id not in access:
            return False
        ids = access[resource.source_id]
        return ids is None or resource.id in ids

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

        try:
            syncs = await self._store.current_syncs(agent_id, (source_id,))
            summary = await self._store.summarize_catalog(agent_id, (source_id,))
        except CatalogStoreError:
            return None, False
        return summary.resource_count, any(sync.truncated for sync in syncs)

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

        readable = await self._service_readable(request.agent_id, request.source_ids)
        result = await self._service.search(request, readable_resource_ids=readable)
        matched = [
            hit for hit in result.hits if "unmatched_fallback" not in hit.match_reasons
        ]
        names = await self.resource_names(
            request.agent_id,
            request.source_ids,
            tuple(hit.resource_id for hit in matched),
        )
        tabular = {
            item.resource.id: item.tabular
            for source_id in {hit.source_id for hit in matched}
            for item in await self._service.tabular_resources(
                request.agent_id,
                source_id,
                resource_ids=tuple(
                    hit.resource_id for hit in matched if hit.source_id == source_id
                ),
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
        access = await self.source_read_access(agent_id, (source_id,))
        folded = name.lower()
        spellings = (name, folded, f"main.{folded}")
        current = tuple(
            item
            for item in await self._service.tabular_resources(
                agent_id, source_id, identities=spellings
            )
            if self._admits(access, item.resource)
        )
        found = next(
            (
                item
                for spelling in spellings
                for item in current
                if item.resource.native_identity == spelling
            ),
            None,
        )
        if found is None:
            # An unqualified name sits in some schema: its resource is named so.
            rows = await self._store.search_catalog(
                agent_id, (source_id,), terms=(), anchor_keys=(search_key(name),)
            )
            ids = tuple(
                row.resource_id
                for row in rows
                if row.detailed and row.native_identity.endswith(f".{name}")
            )
            suffixed = [
                item
                for item in (
                    await self._service.tabular_resources(
                        agent_id, source_id, resource_ids=ids
                    )
                    if ids
                    else ()
                )
                if self._admits(access, item.resource)
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

    async def link_candidates(
        self,
        agent_id: str,
        source_id: str,
        description: Mapping[str, object],
        known: Sequence[Mapping[str, object]],
    ) -> list[Mapping[str, object]]:
        """Unconfirmed joins from a table's key-like columns to other sources.

        A hint only: it grants no read authority and is never declared or stored.
        """

        name = cast(str, description["name"])
        columns = {
            cast(str, column["name"]).lower(): (
                cast(str, column["name"]),
                cast(str, column["type"]),
            )
            for column in cast(Sequence[Mapping[str, object]], description["columns"])
        }
        keyed = {
            column.lower()
            for key in (
                description.get("primary_key", ()),
                *cast(Sequence[Sequence[str]], description.get("unique_keys", ())),
            )
            for column in cast(Sequence[str], key)
        } - {"id"}
        key_like = {
            column
            for column in columns
            if column.endswith(_KEY_SUFFIXES) or column in keyed
        }
        if not key_like:
            return []
        declared = set()
        for relation in known:
            for left, right in cast(Sequence[Sequence[str]], relation["columns"]):
                if relation["from"] == name and (
                    relation.get("from_source_id", source_id) == source_id
                ):
                    declared.add(
                        (left, relation.get("to_source_id", source_id), relation["to"])
                        + (right,)
                    )
                elif relation["to"] == name and (
                    relation.get("to_source_id", source_id) == source_id
                ):
                    declared.add(
                        (
                            right,
                            relation.get("from_source_id", source_id),
                            relation["from"],
                            left,
                        )
                    )
        access = await self.source_read_access(agent_id, ())
        # Where a match can be: a table named for the column's stem, or a column
        # that starts the way the key column does (found by the search table).
        stems = {own.removesuffix("_id") for own in key_like if own.endswith("_id")}
        bare_names = {stem + plural for stem in stems for plural in ("", "s")}
        anchors = tuple(dict.fromkeys(search_key(name) for name in bare_names))
        terms = tuple(
            dict.fromkeys(
                "".join(normalize_search_text(own).tokens)[:3] for own in key_like
            )
        )
        found: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
        for registration in await self._active_registrations(agent_id, ()):
            if registration.id == source_id:
                continue
            try:
                rows = await self._store.search_catalog(
                    agent_id, (registration.id,), terms=terms, anchor_keys=anchors
                )
                ids = tuple(
                    row.resource_id
                    for row in rows
                    if row.detailed
                    and (
                        row.native_identity.rsplit(".", 1)[-1].lower() in bare_names
                        or any(
                            posting.field_kind == "column"
                            and posting.value.lower() in key_like
                            for posting in row.postings
                        )
                    )
                )
                resources = (
                    await self._service.tabular_resources(
                        agent_id, registration.id, resource_ids=ids
                    )
                    if ids
                    else ()
                )
            except CatalogStoreError:
                continue
            for item in resources:
                if not self._admits(access, item.resource):
                    continue
                target = item.resource.native_identity
                bare = target.rsplit(".", 1)[-1].lower()
                primary = [
                    column.name
                    for column in item.tabular.columns
                    if column.primary_key_ordinal
                ]
                for column in item.tabular.columns:
                    for own in key_like:
                        by_name = own == column.name.lower()
                        stem = own.removesuffix("_id")
                        by_stem = (
                            own.endswith("_id")
                            and bare in (stem, f"{stem}s")
                            and (
                                column.name.lower() == "id" or primary == [column.name]
                            )
                        )
                        own_name, own_type = columns[own]
                        if not (by_name or by_stem) or not _compatible(
                            own_type, column.native_type
                        ):
                            continue
                        if (own_name, registration.id, target, column.name) in declared:
                            continue
                        estimate = item.tabular.row_count_estimate
                        found[(own_name, registration.id, target, column.name)] = {
                            "column": own_name,
                            "to": target,
                            "to_source_id": registration.id,
                            "to_kind": item.resource.kind.value,
                            "to_column": column.name,
                            **(
                                {}
                                if estimate is None
                                else {"to_row_estimate": estimate}
                            ),
                            "status": "unconfirmed",
                            "_exact": by_name,
                        }
        ordered = sorted(
            found.items(),
            key=lambda entry: (not entry[1]["_exact"], entry[0][1:], entry[0][0]),
        )
        return [
            {key: value for key, value in candidate.items() if key != "_exact"}
            for _, candidate in ordered
        ]

    async def _cataloged_relationships(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> list[Mapping[str, object]]:
        """Relationships by name, including links declared across sources."""

        # Every source's readable set: a declared link needs its far endpoint.
        readable = await self._service_readable(agent_id, ())
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
            readable_resource_ids=readable,
        )
        relationships: list[Mapping[str, object]] = []
        raws = cast(
            tuple[Mapping[str, object], ...], projection.get("relationships", ())
        )
        known = {
            entry.resource.id: entry.resource
            for entry in await self._store.load_entries(
                agent_id,
                tuple(
                    {
                        cast(str, raw.get(key))
                        for raw in raws
                        for key in ("from_resource_id", "to_resource_id")
                    }
                ),
            )
            if entry.resource.agent_id == agent_id
        }
        # One lookup per source holding an end; a resource past its catalog
        # budget has none.
        access = await self.source_read_access(agent_id, ())
        estimates: dict[str, int | None] = {}
        for end_source in {end.source_id for end in known.values()}:
            try:
                estimates.update(
                    (item.resource.id, item.tabular.row_count_estimate)
                    for item in await self._service.tabular_resources(
                        agent_id,
                        end_source,
                        resource_ids=tuple(
                            end.id
                            for end in known.values()
                            if end.source_id == end_source
                        ),
                    )
                    if self._admits(access, item.resource)
                )
            except CatalogStoreError:
                if end_source == source_id:
                    raise
        for raw in raws:
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
                    **{
                        f"{side}_row_estimate": estimate
                        for side, resource in (("from", start), ("to", end))
                        if (estimate := estimates.get(resource.id)) is not None
                    },
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
        resource_ids: tuple[str, ...] | None = None,
    ) -> tuple[ResourceSchema, ...]:
        """One source's tabular schemas, or only those of ``resource_ids``."""

        tabular_resources = await self._service.tabular_resources(
            agent_id, source_id, resource_ids=resource_ids
        )
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

    async def close_table_names(
        self,
        agent_id: str,
        source_id: str,
        relations: Sequence[str],
        *,
        count: int,
        cutoff: float,
    ) -> tuple[str, ...]:
        """Readable table names close to each of ``relations``, best first.

        For a refusal: the source's names are listed by columns alone, and only
        the close ones are checked against their descriptions.
        """

        access = await self.source_read_access(agent_id, (source_id,))
        if source_id not in access:
            return ()
        if (allowed := access[source_id]) is None:
            rows = await self._store.search_catalog(agent_id, (source_id,), terms=None)
            listed = [
                (row.resource_id, row.name, row.native_identity)
                for row in rows
                if row.detailed and row.kind in TABULAR_KINDS
            ]
        else:
            listed = [
                (entry.resource.id, entry.resource.name, entry.resource.native_identity)
                for entry in await self._store.load_entries(
                    agent_id, tuple(sorted(allowed))
                )
                if entry.detailed
                and entry.resource.source_id == source_id
                and entry.resource.kind in TABULAR_KINDS
            ]
        # A name like "schema.table" shows in full, any other by its own name.
        pool = {
            resource_id: native if "." in native and native != name else name
            for resource_id, name, native in listed
        }
        suggestions: dict[str, None] = {}
        for relation in relations:
            candidates = dict(pool)
            while candidates:
                close = get_close_matches(
                    relation, list(candidates.values()), n=count, cutoff=cutoff
                )
                held = [i for i, name in candidates.items() if name in close]
                described = {
                    item.resource.id
                    for item in await self._service.tabular_resources(
                        agent_id, source_id, resource_ids=tuple(held)
                    )
                }
                if all(i in described for i in held):
                    suggestions.update(dict.fromkeys(close))
                    break
                # A name without a table description is no suggestion; look on.
                for resource_id in held:
                    if resource_id not in described:
                        del candidates[resource_id]
        return tuple(suggestions)

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

        resource = await self._store.load_resource(agent_id, resource_id)
        access = await self.source_read_access(agent_id, (source_id,))
        if (
            resource is None
            or resource.agent_id != agent_id
            or resource.source_id != source_id
            or not self._admits(access, resource)
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
        access = await self.source_read_access(agent_id, ())
        registrations = {
            item.id: item
            for item in await self._sources.list_sources(agent_id)
            if item.agent_id == agent_id and item.active
        }
        # A name-only entry has no fields to hold a note, so it is unknown here.
        resources = {
            entry.resource.id: entry.resource
            for entry in await self._store.load_entries(agent_id, requested)
            if entry.detailed
            and entry.resource.agent_id == agent_id
            and self._admits(access, entry.resource)
            and entry.resource.source_id in registrations
        }
        fields_by_id: dict[str, tuple[str, ...]] = {}
        types_by_id: dict[str, tuple[tuple[str, str], ...]] = {}
        for source_id in sorted({item.source_id for item in resources.values()}):
            for schema in await self.resource_schemas(
                agent_id,
                source_id,
                tuple(
                    item.id
                    for item in resources.values()
                    if item.source_id == source_id
                ),
            ):
                fields_by_id[schema.resource_id] = schema.columns
                types_by_id[schema.resource_id] = schema.column_declared_types
        return tuple(
            SemanticResourceFact(
                resource_id=resource.id,
                source_id=resource.source_id,
                revision=resource.current_revision,
                field_names=fields_by_id.get(resource.id, ()),
                field_types=types_by_id.get(resource.id, ()),
            )
            for resource in sorted(resources.values(), key=lambda item: item.id)
        )

    async def table_notes(
        self, agent_id: str, source_id: str
    ) -> Mapping[str, tuple[str, ...]]:
        """Statements of current notes about one table of a source, by table name."""

        # The composed state store is also the semantic store.
        stored = await cast(SemanticStore, self._store).list_semantic_annotations(
            agent_id
        )
        annotations = tuple(
            item
            for item in stored
            if item.subject.source_ids == (source_id,)
            and len(item.subject.resource_ids) == 1
            and not item.subject.fields
        )
        if not annotations:
            return {}
        resource_ids = tuple(
            sorted({item.subject.resource_ids[0] for item in annotations})
        )
        facts = await self.semantic_resource_facts(agent_id, resource_ids)
        names = await self.resource_names(agent_id, (source_id,), resource_ids)
        notes: dict[str, list[str]] = {}
        for view in inspect_semantic_annotations(annotations, facts):
            name = names.get(view.annotation.subject.resource_ids[0])
            if name is not None and view.usable_as_current_meaning:
                notes.setdefault(name, []).append(view.annotation.statement)
        return {name: tuple(statements) for name, statements in notes.items()}

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
