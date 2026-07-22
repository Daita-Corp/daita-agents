"""Read-only SQLite source discovery and inspection adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading

from ..catalog.models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    TabularIndex,
    catalog_resource_id,
)
from ..capabilities import ExtensionDeclarations
from ..domains.data.capabilities import sqlite_query_extension_declarations
from .models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from .protocols import (
    DiscoveryLimitError,
    ResourceAdapterError,
    ResourceNotFoundError,
    SourceClosedError,
    StaleResourceError,
)


class SQLiteSourceError(ResourceAdapterError):
    def __init__(self, code: str, message: str, *, source_id: str = "sqlite:unopened"):
        super().__init__(source_id, code, message)


@dataclass(frozen=True, slots=True)
class SQLiteSource:
    path: str | Path
    name: str | None = None

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock: Callable[[], datetime],
    ) -> SQLiteResourceAdapter:
        path = _safe_source_path(self.path)
        display_name = self.name or path.stem
        configuration: dict[str, object] = {"path": str(path)}
        registration = SourceRegistration.build(
            agent_id=agent_id,
            adapter_id="sqlite",
            native_identity=str(path),
            display_name=display_name,
            configuration=configuration,
            attached_at=attached_at,
        )
        try:
            connection = await asyncio.to_thread(_open_read_only, path)
        except sqlite3.Error as error:
            raise SQLiteSourceError(
                "sqlite_open_failed",
                "SQLite source could not be opened as a read-only database.",
                source_id=registration.id,
            ) from error
        return SQLiteResourceAdapter(
            registration=registration,
            connection=connection,
            clock=clock,
        )


class SQLiteResourceAdapter:
    def __init__(
        self,
        *,
        registration: SourceRegistration,
        connection: sqlite3.Connection,
        clock: Callable[[], datetime],
    ) -> None:
        self._registration = registration
        self._connection = connection
        self._clock = clock
        self._lock = asyncio.Lock()
        self._closed = False
        self._latest: SourceCatalogSnapshot | None = None

    @property
    def registration(self) -> SourceRegistration:
        return self._registration

    def declarations(self) -> ExtensionDeclarations:
        return sqlite_query_extension_declarations()

    async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        self._require_request(request)
        async with self._lock:
            self._require_open()
            cancellation = threading.Event()
            worker = asyncio.create_task(
                asyncio.to_thread(
                    _discover,
                    self._connection,
                    self._registration,
                    request,
                    self._clock(),
                    cancellation,
                )
            )
            try:
                snapshot = await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancellation.set()
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                raise
            self._latest = snapshot
            assert snapshot.sync.completed_at is not None
            return DiscoveryResult(
                request=request,
                snapshot=snapshot,
                completed_at=snapshot.sync.completed_at,
            )

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
        if not isinstance(resource, ResourceRef):
            raise TypeError("resource must be a ResourceRef")
        async with self._lock:
            self._require_open()
            if (
                resource.agent_id != self._registration.agent_id
                or resource.source_id != self._registration.id
            ):
                raise ResourceNotFoundError(
                    self._registration.id,
                    resource.resource_id,
                )
            snapshot = self._latest
            if snapshot is None:
                raise ResourceNotFoundError(
                    self._registration.id,
                    resource.resource_id,
                )
            current = next(
                (
                    item
                    for item in snapshot.resources
                    if item.id == resource.resource_id
                ),
                None,
            )
            if current is None:
                raise ResourceNotFoundError(
                    self._registration.id,
                    resource.resource_id,
                )
            if resource.revision is not None and (
                resource.revision != current.current_revision
            ):
                raise StaleResourceError(
                    self._registration.id,
                    resource.resource_id,
                )
            revision = next(
                item for item in snapshot.revisions if item.resource_id == current.id
            )
            return ResourceSnapshot(
                reference=resource,
                resource=current,
                revision=revision,
                facets=tuple(
                    facet
                    for facet in snapshot.facets
                    if facet.resource_id == current.id
                ),
                relationships=tuple(
                    relationship
                    for relationship in snapshot.relationships
                    if current.id
                    in {
                        relationship.from_resource_id,
                        relationship.to_resource_id,
                    }
                ),
                inspected_at=self._clock(),
                source_revision=revision.source_revision,
            )

    async def health(self) -> SourceHealth:
        async with self._lock:
            checked_at = self._clock()
            if self._closed:
                return SourceHealth(
                    agent_id=self._registration.agent_id,
                    source_id=self._registration.id,
                    adapter_id="sqlite",
                    healthy=False,
                    checked_at=checked_at,
                    error_code="source_closed",
                )
            try:
                schema_version, query_only = await asyncio.to_thread(
                    _health,
                    self._connection,
                )
            except sqlite3.Error:
                return SourceHealth(
                    agent_id=self._registration.agent_id,
                    source_id=self._registration.id,
                    adapter_id="sqlite",
                    healthy=False,
                    checked_at=checked_at,
                    error_code="sqlite_health_failed",
                )
            return SourceHealth(
                agent_id=self._registration.agent_id,
                source_id=self._registration.id,
                adapter_id="sqlite",
                healthy=True,
                checked_at=checked_at,
                source_revision=f"schema_version:{schema_version}",
                details={"query_only": bool(query_only)},
            )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            await asyncio.to_thread(self._connection.close)
            self._closed = True

    def _require_request(self, request: DiscoveryRequest) -> None:
        if not isinstance(request, DiscoveryRequest):
            raise TypeError("request must be a DiscoveryRequest")
        if (
            request.agent_id != self._registration.agent_id
            or request.source_id != self._registration.id
        ):
            raise ResourceAdapterError(
                self._registration.id,
                "source_scope_mismatch",
                "discovery request does not match this source",
            )

    def _require_open(self) -> None:
        if self._closed:
            raise SourceClosedError(self._registration.id)


@dataclass(frozen=True, slots=True)
class _TableFacts:
    name: str
    kind: ResourceKind
    facet: CatalogFacet


def _safe_source_path(
    value: str | Path,
) -> Path:
    lexical = Path(os.path.abspath(os.fspath(value)))
    try:
        resolved = lexical.resolve(strict=True)
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise SQLiteSourceError(
            "sqlite_path_invalid",
            "SQLite source must be an existing unaliased regular file.",
        ) from error
    try:
        if lexical != resolved or not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SQLiteSourceError(
                "sqlite_path_invalid",
                "SQLite source must be an existing unaliased regular file.",
            )
    finally:
        os.close(descriptor)
    return resolved


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro",
        uri=True,
        check_same_thread=False,
        timeout=5.0,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute(
            "SELECT schema_version FROM pragma_schema_version"
        ).fetchone()
        return connection
    except BaseException:
        connection.close()
        raise


def _health(connection: sqlite3.Connection) -> tuple[int, int]:
    schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    query_only = connection.execute("PRAGMA query_only").fetchone()[0]
    return int(schema_version), int(query_only)


def _discover(
    connection: sqlite3.Connection,
    registration: SourceRegistration,
    request: DiscoveryRequest,
    completed_at: datetime,
    cancellation: threading.Event,
) -> SourceCatalogSnapshot:
    if completed_at < request.requested_at:
        completed_at = request.requested_at
    connection.set_progress_handler(lambda: int(cancellation.is_set()), 500)
    try:
        connection.execute("BEGIN")
        schema_version = int(connection.execute("PRAGMA schema_version").fetchone()[0])
        rows = connection.execute("PRAGMA table_list").fetchall()
        visible = tuple(
            sorted(
                (
                    row
                    for row in rows
                    if row["schema"] == "main"
                    and row["type"] in {"table", "view"}
                    and not str(row["name"]).startswith("sqlite_")
                ),
                key=lambda row: (str(row["name"]), str(row["type"])),
            )
        )
        if len(visible) > request.max_resources:
            raise DiscoveryLimitError(
                registration.id,
                "SQLite discovery resource limit exceeded",
            )

        facts: list[_TableFacts] = []
        for row in visible:
            name = str(row["name"])
            kind = ResourceKind.TABLE if row["type"] == "table" else ResourceKind.VIEW
            resource_id = catalog_resource_id(
                registration.id,
                kind,
                f"main.{name}",
            )
            columns = _columns(connection, name, request, registration.id)
            indexes = _indexes(
                connection,
                name,
                {column.name for column in columns},
                request,
                registration.id,
            )
            facet = CatalogFacet.from_tabular(
                resource_id=resource_id,
                sync_id=request.sync_id,
                observed_at=request.requested_at,
                facet=TabularFacet(columns=columns, indexes=indexes),
            )
            facts.append(_TableFacts(name=name, kind=kind, facet=facet))

        facts_by_name = {fact.name: fact for fact in facts}
        relationships = _relationships(
            connection,
            registration,
            request,
            facts_by_name,
        )
        if len(relationships) > request.max_relationships:
            raise DiscoveryLimitError(
                registration.id,
                "SQLite discovery relationship limit exceeded",
            )
        source_revision = f"schema_version:{schema_version}"
        revisions: list[CatalogResourceRevision] = []
        resources: list[CatalogResource] = []
        for fact in facts:
            resource_id = fact.facet.resource_id
            linked_relationships = tuple(
                relationship.revision
                for relationship in relationships
                if resource_id
                in {
                    relationship.from_resource_id,
                    relationship.to_resource_id,
                }
            )
            revision = CatalogResourceRevision.build(
                resource_id=resource_id,
                sync_id=request.sync_id,
                observed_at=request.requested_at,
                facet_revisions=(fact.facet.revision,),
                relationship_revisions=linked_relationships,
                source_revision=source_revision,
            )
            revisions.append(revision)
            resources.append(
                CatalogResource.build(
                    agent_id=request.agent_id,
                    source_id=request.source_id,
                    native_identity=f"main.{fact.name}",
                    external_uri=f"sqlite://{request.source_id}/main/{fact.name}",
                    kind=fact.kind,
                    name=fact.name,
                    sensitivity=Sensitivity.INTERNAL,
                    revision=revision,
                    first_observed_at=request.requested_at,
                    last_observed_at=request.requested_at,
                )
            )
        sync = CatalogSync(
            id=request.sync_id,
            agent_id=request.agent_id,
            source_id=request.source_id,
            adapter_id="sqlite",
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=request.requested_at,
            completed_at=completed_at,
            source_revision=source_revision,
            resource_count=len(resources),
            relationship_count=len(relationships),
        )
        return SourceCatalogSnapshot(
            sync=sync,
            resources=tuple(resources),
            revisions=tuple(revisions),
            facets=tuple(fact.facet for fact in facts),
            relationships=relationships,
        )
    finally:
        connection.set_progress_handler(None, 0)
        if connection.in_transaction:
            connection.execute("ROLLBACK")


def _columns(
    connection: sqlite3.Connection,
    table: str,
    request: DiscoveryRequest,
    source_id: str,
) -> tuple[TabularColumn, ...]:
    rows = connection.execute(
        'SELECT cid, name, type, "notnull", dflt_value, pk, hidden '
        "FROM pragma_table_xinfo(?) ORDER BY cid",
        (table,),
    ).fetchall()
    visible_rows = tuple(row for row in rows if int(row["hidden"]) != 1)
    if len(visible_rows) > request.max_columns_per_resource:
        raise DiscoveryLimitError(
            source_id,
            "SQLite discovery column limit exceeded",
        )
    return tuple(
        TabularColumn(
            name=str(row["name"]),
            native_type=str(row["type"] or "UNKNOWN"),
            ordinal=int(row["cid"]),
            nullable=not bool(row["notnull"]) and not bool(row["pk"]),
            primary_key_ordinal=(int(row["pk"]) or None),
            default_expression=(
                None if row["dflt_value"] is None else str(row["dflt_value"])
            ),
        )
        for row in visible_rows
    )


def _indexes(
    connection: sqlite3.Connection,
    table: str,
    known_columns: set[str],
    request: DiscoveryRequest,
    source_id: str,
) -> tuple[TabularIndex, ...]:
    rows = connection.execute(
        'SELECT seq, name, "unique", origin, partial '
        "FROM pragma_index_list(?) ORDER BY name",
        (table,),
    ).fetchall()
    if len(rows) > request.max_indexes_per_resource:
        raise DiscoveryLimitError(
            source_id,
            "SQLite discovery index limit exceeded",
        )
    indexes: list[TabularIndex] = []
    for row in rows:
        name = str(row["name"])
        column_rows = connection.execute(
            "SELECT seqno, cid, name FROM pragma_index_info(?) ORDER BY seqno",
            (name,),
        ).fetchall()
        columns = tuple(
            str(column["name"]) for column in column_rows if column["name"] is not None
        )
        if not columns or len(columns) != len(column_rows):
            continue
        if any(column not in known_columns for column in columns):
            continue
        sql_row = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'index' AND name = ?",
            (name,),
        ).fetchone()
        predicate = _index_predicate(None if sql_row is None else sql_row["sql"])
        indexes.append(
            TabularIndex(
                name=name,
                kind="btree",
                columns=columns,
                unique=bool(row["unique"]),
                predicate=predicate,
            )
        )
    return tuple(indexes)


def _index_predicate(sql: object) -> str | None:
    if not isinstance(sql, str):
        return None
    match = re.search(r"\bWHERE\s+(.+)\Z", sql.strip(), re.IGNORECASE | re.DOTALL)
    return None if match is None else match.group(1).strip()


def _relationships(
    connection: sqlite3.Connection,
    registration: SourceRegistration,
    request: DiscoveryRequest,
    facts_by_name: dict[str, _TableFacts],
) -> tuple[CatalogRelationship, ...]:
    relationships: list[CatalogRelationship] = []
    for table, fact in sorted(facts_by_name.items()):
        if fact.kind is not ResourceKind.TABLE:
            continue
        rows = connection.execute(
            'SELECT id, seq, "table", "from", "to", on_update, '
            "on_delete, match FROM pragma_foreign_key_list(?) ORDER BY id, seq",
            (table,),
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(int(row["id"]), []).append(row)
        for group in grouped.values():
            target_name = str(group[0]["table"])
            target = facts_by_name.get(target_name)
            if target is None or target.kind is not ResourceKind.TABLE:
                continue
            target_fields = [row["to"] for row in group]
            if any(field is None for field in target_fields):
                primary = _primary_key_columns(connection, target_name)
                if len(primary) != len(group):
                    continue
                target_fields = list(primary)
            pairs = tuple(
                RelationshipFieldPair(
                    source_field=str(row["from"]),
                    target_field=str(target_field),
                    ordinal=index,
                )
                for index, (row, target_field) in enumerate(
                    zip(group, target_fields, strict=True)
                )
            )
            relationships.append(
                CatalogRelationship.build(
                    source_id=registration.id,
                    from_resource_id=fact.facet.resource_id,
                    to_resource_id=target.facet.resource_id,
                    kind=RelationshipKind.REFERENCES,
                    provenance=RelationshipProvenance.CONNECTOR,
                    confidence=1.0,
                    sync_id=request.sync_id,
                    observed_at=request.requested_at,
                    field_pairs=pairs,
                    attributes={
                        "match": str(group[0]["match"]),
                        "on_delete": str(group[0]["on_delete"]),
                        "on_update": str(group[0]["on_update"]),
                    },
                )
            )
    return tuple(sorted(relationships, key=lambda item: item.id))


def _primary_key_columns(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT name, pk FROM pragma_table_xinfo(?) WHERE pk > 0 ORDER BY pk",
        (table,),
    ).fetchall()
    return tuple(str(row["name"]) for row in rows)


__all__ = ["SQLiteResourceAdapter", "SQLiteSource", "SQLiteSourceError"]
