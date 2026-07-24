"""Read-only PostgreSQL source discovery and inspection adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from importlib import import_module
from itertools import islice
import re
from typing import Any
from urllib.parse import quote

from .._installation import PIPX_REPAIR_GUIDANCE
from .._json import canonical_json
from ..capabilities import ExtensionDeclarations
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
from ..domains.data.capabilities import postgresql_query_extension_declarations
from ..security import (
    SecretProvider,
    SecretReference,
    default_secret_provider,
)
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

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$]{0,62}\Z")
_SSL_MODES = frozenset(
    {"disable", "prefer", "allow", "require", "verify-ca", "verify-full"}
)
_DEFAULT_MAX_RESOURCES = 1_000
_DEFAULT_MAX_COLUMNS = 512
_DEFAULT_MAX_INDEXES = 256
_DEFAULT_MAX_RELATIONSHIPS = 2_000
_MAX_PROBE_SCHEMAS = 100
_PROBE_QUERY_LIMIT = _MAX_PROBE_SCHEMAS + 1
_MAX_PROBE_ROW_FIELDS = 16
_CONNECT_TIMEOUT_SECONDS = 10.0
_COMMAND_TIMEOUT_SECONDS = 10.0
_CLEANUP_TIMEOUT_SECONDS = 1.0
_SUPPORTED_QUERY_TYPE_NAMES = frozenset(
    {
        "bit",
        "bool",
        "bpchar",
        "bytea",
        "cidr",
        "date",
        "float4",
        "float8",
        "inet",
        "int2",
        "int4",
        "int8",
        "interval",
        "json",
        "jsonb",
        "name",
        "numeric",
        "text",
        "time",
        "timestamp",
        "timestamptz",
        "timetz",
        "uuid",
        "varbit",
        "varchar",
    }
)

_RESOURCES_SQL = """
/* daita:postgresql.resources */
SELECT
    table_schema AS schema_name,
    table_name AS resource_name,
    CASE table_type WHEN 'VIEW' THEN 'view' ELSE 'table' END AS resource_kind
FROM information_schema.tables
WHERE table_schema = ANY($1::text[])
  AND table_type = 'BASE TABLE'
ORDER BY table_schema, table_name
LIMIT $2
"""

_COLUMNS_SQL = """
/* daita:postgresql.columns */
SELECT
    c.column_name,
    format_type(a.atttypid, a.atttypmod) AS native_type,
    type_ns.nspname AS type_schema,
    type_def.typname AS type_name,
    c.ordinal_position AS ordinal,
    (c.is_nullable = 'YES') AS nullable,
    c.column_default AS default_expression,
    pk.primary_key_ordinal
FROM information_schema.columns AS c
JOIN pg_namespace AS n ON n.nspname = c.table_schema
JOIN pg_class AS rel ON rel.relnamespace = n.oid AND rel.relname = c.table_name
JOIN pg_attribute AS a
  ON a.attrelid = rel.oid
 AND a.attname = c.column_name
 AND a.attnum > 0
 AND NOT a.attisdropped
JOIN pg_type AS type_def ON type_def.oid = a.atttypid
JOIN pg_namespace AS type_ns ON type_ns.oid = type_def.typnamespace
LEFT JOIN (
    SELECT
        con.conrelid,
        key.attnum,
        key.ordinality::integer AS primary_key_ordinal
    FROM pg_constraint AS con
    CROSS JOIN LATERAL unnest(con.conkey)
        WITH ORDINALITY AS key(attnum, ordinality)
    WHERE con.contype = 'p'
) AS pk ON pk.conrelid = rel.oid AND pk.attnum = a.attnum
WHERE c.table_schema = $1 AND c.table_name = $2
ORDER BY c.ordinal_position
LIMIT $3
"""

_INDEXES_SQL = """
/* daita:postgresql.indexes */
SELECT
    idx.relname AS index_name,
    am.amname AS index_kind,
    array_agg(att.attname ORDER BY key.ordinality)
        FILTER (WHERE att.attname IS NOT NULL) AS columns,
    bool_and(att.attname IS NOT NULL) AS simple_columns,
    ind.indisunique AS is_unique,
    pg_get_expr(ind.indpred, ind.indrelid) AS predicate
FROM pg_namespace AS n
JOIN pg_class AS rel ON rel.relnamespace = n.oid
JOIN pg_index AS ind ON ind.indrelid = rel.oid
JOIN pg_class AS idx ON idx.oid = ind.indexrelid
JOIN pg_am AS am ON am.oid = idx.relam
CROSS JOIN LATERAL unnest(ind.indkey)
    WITH ORDINALITY AS key(attnum, ordinality)
LEFT JOIN pg_attribute AS att
  ON att.attrelid = rel.oid AND att.attnum = key.attnum
WHERE n.nspname = $1
  AND rel.relname = $2
  AND key.ordinality <= ind.indnkeyatts
GROUP BY idx.relname, am.amname, ind.indisunique, ind.indpred, ind.indrelid
ORDER BY idx.relname
LIMIT $3
"""

_RELATIONSHIPS_SQL = """
/* daita:postgresql.relationships */
SELECT
    con.conname AS constraint_name,
    src_ns.nspname AS source_schema,
    src.relname AS source_table,
    dst_ns.nspname AS target_schema,
    dst.relname AS target_table,
    array_agg(src_att.attname ORDER BY keys.ordinality) AS source_columns,
    array_agg(dst_att.attname ORDER BY keys.ordinality) AS target_columns,
    CASE con.confmatchtype WHEN 'f' THEN 'FULL' WHEN 'p' THEN 'PARTIAL'
         ELSE 'SIMPLE' END AS match_type,
    CASE con.confupdtype WHEN 'a' THEN 'NO ACTION' WHEN 'r' THEN 'RESTRICT'
         WHEN 'c' THEN 'CASCADE' WHEN 'n' THEN 'SET NULL'
         WHEN 'd' THEN 'SET DEFAULT' END AS on_update,
    CASE con.confdeltype WHEN 'a' THEN 'NO ACTION' WHEN 'r' THEN 'RESTRICT'
         WHEN 'c' THEN 'CASCADE' WHEN 'n' THEN 'SET NULL'
         WHEN 'd' THEN 'SET DEFAULT' END AS on_delete
FROM pg_constraint AS con
JOIN pg_class AS src ON src.oid = con.conrelid
JOIN pg_namespace AS src_ns ON src_ns.oid = src.relnamespace
JOIN pg_class AS dst ON dst.oid = con.confrelid
JOIN pg_namespace AS dst_ns ON dst_ns.oid = dst.relnamespace
CROSS JOIN LATERAL unnest(con.conkey, con.confkey)
    WITH ORDINALITY AS keys(source_attnum, target_attnum, ordinality)
JOIN pg_attribute AS src_att
  ON src_att.attrelid = src.oid AND src_att.attnum = keys.source_attnum
JOIN pg_attribute AS dst_att
  ON dst_att.attrelid = dst.oid AND dst_att.attnum = keys.target_attnum
WHERE con.contype = 'f'
  AND src_ns.nspname = ANY($1::text[])
  AND dst_ns.nspname = ANY($1::text[])
GROUP BY con.oid, con.conname, src_ns.nspname, src.relname,
         dst_ns.nspname, dst.relname, con.confmatchtype,
         con.confupdtype, con.confdeltype
ORDER BY src_ns.nspname, src.relname, con.conname
LIMIT $2
"""

_SCHEMA_PROBE_SQL = """
/* daita:postgresql.schema_probe */
SELECT
    namespace.nspname AS schema_name,
    EXISTS (
        SELECT 1
        FROM information_schema.tables AS table_info
        WHERE table_info.table_schema = namespace.nspname
          AND table_info.table_type = 'BASE TABLE'
    ) AS has_base_tables
FROM pg_namespace AS namespace
WHERE namespace.nspname <> 'information_schema'
  AND namespace.nspname NOT LIKE 'pg\\_%' ESCAPE '\\'
ORDER BY namespace.nspname
LIMIT $1
"""


class PostgreSQLSourceError(ResourceAdapterError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        source_id: str = "postgresql:unopened",
    ) -> None:
        super().__init__(source_id, code, message)


@dataclass(frozen=True, slots=True)
class PostgreSQLProbeSchema:
    """One bounded non-system schema returned by a connection probe."""

    name: str
    has_base_tables: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _IDENTIFIER.fullmatch(self.name) is None:
            raise ValueError("probe schema name must be a safe PostgreSQL identifier")
        if _is_system_schema(self.name):
            raise ValueError("probe schema cannot be a PostgreSQL system schema")
        if not isinstance(self.has_base_tables, bool):
            raise TypeError("probe has_base_tables must be a boolean")


@dataclass(frozen=True, slots=True)
class PostgreSQLProbeResult:
    """Bounded control-plane facts from one non-persisting connection probe."""

    schemas: tuple[PostgreSQLProbeSchema, ...]
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schemas, tuple):
            raise TypeError("probe schemas must be a tuple")
        if not isinstance(self.truncated, bool):
            raise TypeError("probe truncation indicator must be a boolean")
        if len(self.schemas) > _MAX_PROBE_SCHEMAS:
            raise ValueError("probe schema result exceeds its fixed bound")
        if any(not isinstance(item, PostgreSQLProbeSchema) for item in self.schemas):
            raise TypeError("probe schemas must contain probe schema records")
        names = tuple(item.name for item in self.schemas)
        if len(names) != len(set(names)):
            raise ValueError("probe schema names cannot repeat")
        if names != tuple(sorted(names)):
            raise ValueError("probe schema names must be sorted")

    @classmethod
    def build(
        cls,
        schemas: Sequence[tuple[str, bool]],
        *,
        truncated: bool = False,
    ) -> PostgreSQLProbeResult:
        if isinstance(schemas, (str, bytes)):
            raise TypeError("probe schemas must be a sequence")
        records = tuple(
            PostgreSQLProbeSchema(name=name, has_base_tables=has_base_tables)
            for name, has_base_tables in schemas
        )
        return cls(
            tuple(sorted(records, key=lambda item: item.name)),
            truncated=truncated,
        )


@dataclass(frozen=True, slots=True)
class PostgreSQLSource:
    host: str
    database: str
    username: str
    credential: SecretReference | None = None
    port: int = 5432
    schemas: tuple[str, ...] = ("public",)
    ssl_mode: str = "require"
    name: str | None = None
    secret_provider: SecretProvider = field(
        default_factory=default_secret_provider,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _bounded_text(self.host, "host", maximum=253)
        if (
            "://" in self.host
            or "/" in self.host
            or "@" in self.host
            or any(character.isspace() for character in self.host)
        ):
            raise ValueError("host must be a hostname or address")
        _bounded_text(self.database, "database", maximum=128)
        _bounded_text(self.username, "username", maximum=128)
        if _IDENTIFIER.fullmatch(self.database) is None:
            raise ValueError("database must be a safe PostgreSQL identifier")
        if _IDENTIFIER.fullmatch(self.username) is None:
            raise ValueError("username must be a safe PostgreSQL identifier")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65_535
        ):
            raise ValueError("port must be from 1 through 65535")
        schemas = _schemas(self.schemas)
        if self.ssl_mode not in _SSL_MODES:
            raise ValueError("ssl_mode is not supported")
        if self.credential is not None and not isinstance(
            self.credential, SecretReference
        ):
            raise TypeError("credential must be a SecretReference")
        if not isinstance(self.secret_provider, SecretProvider):
            raise TypeError("secret_provider must implement SecretProvider")
        if self.name is not None:
            _bounded_text(self.name, "name", maximum=512)
        object.__setattr__(self, "schemas", schemas)

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock: Callable[[], datetime],
    ) -> PostgreSQLResourceAdapter:
        configuration = _source_configuration(self)
        native_identity = "postgresql:" + canonical_json(
            {
                "database": self.database,
                "host": self.host,
                "port": self.port,
                "schemas": self.schemas,
                "username": self.username,
            }
        )
        registration = SourceRegistration.build(
            agent_id=agent_id,
            adapter_id="postgresql",
            native_identity=native_identity,
            display_name=self.name or self.database,
            configuration=configuration,
            attached_at=attached_at,
        )
        connection = await _connect(
            registration,
            self.secret_provider,
            error_type=PostgreSQLSourceError,
        )
        return PostgreSQLResourceAdapter(
            registration=registration,
            connection=connection,
            clock=clock,
        )

    async def probe(self) -> PostgreSQLProbeResult:
        """Test one connection and list bounded non-system schemas.

        This method owns no source registration or catalog store and therefore
        cannot persist either probe state or structural metadata.
        """

        connection = await _connect_configuration(
            _source_configuration(self),
            self.secret_provider,
            source_id="postgresql:probe",
            error_type=PostgreSQLSourceError,
        )
        result: PostgreSQLProbeResult | None = None
        probe_failed = False
        try:
            rows = await connection.fetch(
                _SCHEMA_PROBE_SQL,
                _PROBE_QUERY_LIMIT,
            )
            result = _probe_result(rows)
        except asyncio.CancelledError:
            raise
        except PostgreSQLSourceError:
            raise
        except Exception:
            probe_failed = True
        finally:
            await _close_postgresql_connection(
                connection,
                timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
            )
        if probe_failed:
            raise PostgreSQLSourceError(
                "postgresql_probe_failed",
                "PostgreSQL schema probe failed.",
                source_id="postgresql:probe",
            )
        if result is None:
            raise AssertionError("PostgreSQL probe completed without a result")
        return result


class PostgreSQLResourceAdapter:
    def __init__(
        self,
        *,
        registration: SourceRegistration,
        connection: Any,
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
        return postgresql_query_extension_declarations()

    async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        self._require_request(request)
        async with self._lock:
            self._require_open()
            transaction = None
            transaction_started = False
            transaction_finished = False
            discovery_failed = False
            snapshot: SourceCatalogSnapshot | None = None
            completed_at: datetime | None = None
            try:
                transaction = self._connection.transaction(
                    isolation="repeatable_read",
                    readonly=True,
                )
                await transaction.start()
                transaction_started = True
                await self._connection.execute(
                    "SELECT set_config('search_path', $1, true)",
                    "pg_catalog",
                )
                structure = await _load_structure(
                    self._connection,
                    self._registration,
                    max_resources=request.max_resources,
                    max_columns=request.max_columns_per_resource,
                    max_indexes=request.max_indexes_per_resource,
                    max_relationships=request.max_relationships,
                )
                completed_at = max(self._clock(), request.requested_at)
                snapshot = _catalog_snapshot(
                    self._registration,
                    request,
                    structure,
                    completed_at,
                )
                await transaction.commit()
                transaction_finished = True
            except asyncio.CancelledError:
                raise
            except ResourceAdapterError:
                raise
            except Exception:
                # Server and driver diagnostics can contain secrets or
                # attacker-controlled identifiers.  Normalize only after the
                # original exception context has ended.
                discovery_failed = True
            finally:
                if (
                    transaction is not None
                    and transaction_started
                    and not transaction_finished
                ):
                    try:
                        rolled_back = await _rollback_postgresql_transaction(
                            transaction,
                            self._connection,
                            timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
                        )
                    except asyncio.CancelledError:
                        self._closed = True
                        raise
                    if not rolled_back:
                        self._closed = True
            if discovery_failed:
                raise PostgreSQLSourceError(
                    "postgresql_discovery_failed",
                    "PostgreSQL source discovery failed.",
                    source_id=self._registration.id,
                )
            if snapshot is None or completed_at is None:
                raise AssertionError(
                    "PostgreSQL discovery completed without a snapshot"
                )
            self._latest = snapshot
            return DiscoveryResult(
                request=request,
                snapshot=snapshot,
                completed_at=completed_at,
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
                    adapter_id="postgresql",
                    healthy=False,
                    checked_at=checked_at,
                    error_code="source_closed",
                )
            try:
                healthy = await self._connection.fetchval("SELECT 1") == 1
            except asyncio.CancelledError:
                raise
            except Exception:
                healthy = False
            return SourceHealth(
                agent_id=self._registration.agent_id,
                source_id=self._registration.id,
                adapter_id="postgresql",
                healthy=healthy,
                checked_at=checked_at,
                error_code=None if healthy else "postgresql_health_failed",
            )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            try:
                await _close_postgresql_connection(
                    self._connection,
                    timeout_seconds=_CLEANUP_TIMEOUT_SECONDS,
                )
            finally:
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
class _TableStructure:
    schema: str
    name: str
    kind: ResourceKind
    columns: tuple[TabularColumn, ...]
    indexes: tuple[TabularIndex, ...]

    def payload(self) -> dict[str, object]:
        return {
            "columns": tuple(column.to_payload() for column in self.columns),
            "indexes": tuple(index.to_payload() for index in self.indexes),
            "kind": self.kind.value,
            "name": self.name,
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True)
class _RelationshipStructure:
    name: str
    source_schema: str
    source_table: str
    target_schema: str
    target_table: str
    pairs: tuple[RelationshipFieldPair, ...]
    match_type: str
    on_update: str
    on_delete: str

    def payload(self) -> dict[str, object]:
        return {
            "match": self.match_type,
            "name": self.name,
            "on_delete": self.on_delete,
            "on_update": self.on_update,
            "pairs": tuple(pair.to_payload() for pair in self.pairs),
            "source": f"{self.source_schema}.{self.source_table}",
            "target": f"{self.target_schema}.{self.target_table}",
        }


@dataclass(frozen=True, slots=True)
class PostgreSQLStructure:
    tables: tuple[_TableStructure, ...]
    relationships: tuple[_RelationshipStructure, ...]
    source_revision: str


async def _load_structure(
    connection: Any,
    registration: SourceRegistration,
    *,
    max_resources: int,
    max_columns: int,
    max_indexes: int,
    max_relationships: int,
) -> PostgreSQLStructure:
    schemas = _configuration_schemas(registration.configuration)
    resource_rows = await connection.fetch(
        _RESOURCES_SQL,
        list(schemas),
        max_resources + 1,
    )
    if len(resource_rows) > max_resources:
        raise DiscoveryLimitError(
            registration.id,
            "PostgreSQL discovery resource limit exceeded",
        )
    tables: list[_TableStructure] = []
    for row in resource_rows:
        schema = _row_text(row, "schema_name")
        name = _row_text(row, "resource_name")
        kind_value = _row_text(row, "resource_kind")
        kind = ResourceKind.VIEW if kind_value == "view" else ResourceKind.TABLE
        column_rows = await connection.fetch(
            _COLUMNS_SQL,
            schema,
            name,
            max_columns + 1,
        )
        if len(column_rows) > max_columns:
            raise DiscoveryLimitError(
                registration.id,
                "PostgreSQL discovery column limit exceeded",
            )
        columns = tuple(_column(row) for row in column_rows)
        if any(not _is_supported_query_type(column.native_type) for column in columns):
            # Custom/extension type output functions are executable database
            # code.  Until their provenance is a durable catalog contract,
            # omit the entire table rather than advertise an unsafe partial
            # schema as queryable.
            continue
        index_rows = await connection.fetch(
            _INDEXES_SQL,
            schema,
            name,
            max_indexes + 1,
        )
        if len(index_rows) > max_indexes:
            raise DiscoveryLimitError(
                registration.id,
                "PostgreSQL discovery index limit exceeded",
            )
        indexes = tuple(
            index
            for row in index_rows
            if (index := _index(row, {column.name for column in columns})) is not None
        )
        tables.append(
            _TableStructure(
                schema=schema,
                name=name,
                kind=kind,
                columns=columns,
                indexes=indexes,
            )
        )
    relationship_rows = await connection.fetch(
        _RELATIONSHIPS_SQL,
        list(schemas),
        max_relationships + 1,
    )
    if len(relationship_rows) > max_relationships:
        raise DiscoveryLimitError(
            registration.id,
            "PostgreSQL discovery relationship limit exceeded",
        )
    known = {(table.schema, table.name) for table in tables}
    relationships = tuple(
        relation
        for row in relationship_rows
        if (relation := _relationship(row, known)) is not None
    )
    ordered_tables = tuple(sorted(tables, key=lambda item: (item.schema, item.name)))
    ordered_relationships = tuple(
        sorted(
            relationships,
            key=lambda item: (
                item.source_schema,
                item.source_table,
                item.name,
            ),
        )
    )
    encoded = canonical_json(
        {
            "relationships": tuple(item.payload() for item in ordered_relationships),
            "tables": tuple(item.payload() for item in ordered_tables),
        }
    ).encode("utf-8")
    return PostgreSQLStructure(
        tables=ordered_tables,
        relationships=ordered_relationships,
        source_revision="catalog:sha256:" + sha256(encoded).hexdigest(),
    )


def _catalog_snapshot(
    registration: SourceRegistration,
    request: DiscoveryRequest,
    structure: PostgreSQLStructure,
    completed_at: datetime,
) -> SourceCatalogSnapshot:
    facets: dict[tuple[str, str], CatalogFacet] = {}
    resource_ids: dict[tuple[str, str], tuple[str, ResourceKind]] = {}
    for table in structure.tables:
        native_identity = f"{table.schema}.{table.name}"
        resource_id = catalog_resource_id(
            registration.id,
            table.kind,
            native_identity,
        )
        resource_ids[(table.schema, table.name)] = (resource_id, table.kind)
        facets[(table.schema, table.name)] = CatalogFacet.from_tabular(
            resource_id=resource_id,
            sync_id=request.sync_id,
            observed_at=request.requested_at,
            facet=TabularFacet(columns=table.columns, indexes=table.indexes),
        )
    relationships: list[CatalogRelationship] = []
    for relation in structure.relationships:
        source = resource_ids.get((relation.source_schema, relation.source_table))
        target = resource_ids.get((relation.target_schema, relation.target_table))
        if source is None or target is None:
            continue
        relationships.append(
            CatalogRelationship.build(
                source_id=registration.id,
                from_resource_id=source[0],
                to_resource_id=target[0],
                kind=RelationshipKind.REFERENCES,
                provenance=RelationshipProvenance.CONNECTOR,
                confidence=1.0,
                sync_id=request.sync_id,
                observed_at=request.requested_at,
                field_pairs=relation.pairs,
                attributes={
                    "constraint_name": relation.name,
                    "match": relation.match_type,
                    "on_delete": relation.on_delete,
                    "on_update": relation.on_update,
                },
            )
        )
    ordered_relationships = tuple(sorted(relationships, key=lambda item: item.id))
    revisions: list[CatalogResourceRevision] = []
    resources: list[CatalogResource] = []
    for table in structure.tables:
        resource_id, kind = resource_ids[(table.schema, table.name)]
        facet = facets[(table.schema, table.name)]
        linked = tuple(
            relationship.revision
            for relationship in ordered_relationships
            if resource_id
            in {relationship.from_resource_id, relationship.to_resource_id}
        )
        revision = CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=request.sync_id,
            observed_at=request.requested_at,
            facet_revisions=(facet.revision,),
            relationship_revisions=linked,
            source_revision=structure.source_revision,
        )
        revisions.append(revision)
        resources.append(
            CatalogResource.build(
                agent_id=request.agent_id,
                source_id=request.source_id,
                native_identity=f"{table.schema}.{table.name}",
                external_uri=(
                    f"postgresql://{request.source_id}/"
                    f"{quote(table.schema, safe='')}/{quote(table.name, safe='')}"
                ),
                kind=kind,
                name=table.name,
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
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=request.requested_at,
        completed_at=completed_at,
        source_revision=structure.source_revision,
        resource_count=len(resources),
        relationship_count=len(ordered_relationships),
    )
    return SourceCatalogSnapshot(
        sync=sync,
        resources=tuple(resources),
        revisions=tuple(revisions),
        facets=tuple(facets[key] for key in sorted(facets)),
        relationships=ordered_relationships,
    )


def _column(row: Mapping[str, object]) -> TabularColumn:
    ordinal = _row_int(row, "ordinal")
    primary = row.get("primary_key_ordinal")
    type_schema = _row_text(row, "type_schema")
    type_name = _row_text(row, "type_name")
    display_type = _row_text(row, "native_type")
    return TabularColumn(
        name=_row_text(row, "column_name"),
        native_type=f"{type_schema}.{type_name}|{display_type}",
        ordinal=ordinal - 1,
        nullable=_row_bool(row, "nullable"),
        primary_key_ordinal=(
            None if primary is None else _positive_int(primary, "primary_key_ordinal")
        ),
        default_expression=_optional_row_text(row, "default_expression"),
    )


def _is_supported_query_type(native_type: str) -> bool:
    provenance = native_type.partition("|")[0]
    namespace, separator, name = provenance.partition(".")
    if separator != "." or namespace != "pg_catalog" or not name:
        return False
    element = name[1:] if name.startswith("_") else name
    return element in _SUPPORTED_QUERY_TYPE_NAMES


def _index(
    row: Mapping[str, object],
    known_columns: set[str],
) -> TabularIndex | None:
    if row.get("simple_columns") is not True:
        return None
    columns = _row_text_sequence(row, "columns", maximum_items=64)
    if not columns or any(column not in known_columns for column in columns):
        return None
    return TabularIndex(
        name=_row_text(row, "index_name"),
        kind=_row_text(row, "index_kind"),
        columns=columns,
        unique=_row_bool(row, "is_unique"),
        predicate=_optional_row_text(row, "predicate"),
    )


def _relationship(
    row: Mapping[str, object],
    known: set[tuple[str, str]],
) -> _RelationshipStructure | None:
    source = (_row_text(row, "source_schema"), _row_text(row, "source_table"))
    target = (_row_text(row, "target_schema"), _row_text(row, "target_table"))
    if source not in known or target not in known:
        return None
    source_columns = _row_text_sequence(
        row,
        "source_columns",
        maximum_items=64,
    )
    target_columns = _row_text_sequence(
        row,
        "target_columns",
        maximum_items=64,
    )
    if len(source_columns) != len(target_columns) or not source_columns:
        return None
    pairs = tuple(
        RelationshipFieldPair(
            source_field=str(source_column),
            target_field=str(target_column),
            ordinal=index,
        )
        for index, (source_column, target_column) in enumerate(
            zip(source_columns, target_columns, strict=True)
        )
    )
    return _RelationshipStructure(
        name=_row_text(row, "constraint_name"),
        source_schema=source[0],
        source_table=source[1],
        target_schema=target[0],
        target_table=target[1],
        pairs=pairs,
        match_type=_row_text(row, "match_type"),
        on_update=_row_text(row, "on_update"),
        on_delete=_row_text(row, "on_delete"),
    )


def _source_configuration(source: PostgreSQLSource) -> dict[str, object]:
    configuration: dict[str, object] = {
        "database": source.database,
        "host": source.host,
        "port": source.port,
        "schemas": source.schemas,
        "ssl_mode": source.ssl_mode,
        "username": source.username,
    }
    if source.credential is not None:
        configuration["credential_ref"] = source.credential.to_uri()
    return configuration


def _probe_result(rows: Sequence[object]) -> PostgreSQLProbeResult:
    if isinstance(rows, (str, bytes)) or len(rows) > _PROBE_QUERY_LIMIT:
        raise PostgreSQLSourceError(
            "postgresql_probe_result_invalid",
            "PostgreSQL returned an invalid schema probe result.",
            source_id="postgresql:probe",
        )
    schemas: list[tuple[str, bool]] = []
    try:
        for row in rows:
            normalized = _probe_row_mapping(row)
            name = _row_text(normalized, "schema_name")
            if _is_system_schema(name):
                continue
            if _IDENTIFIER.fullmatch(name) is None:
                raise ValueError("probe schema name is invalid")
            schemas.append((name, _row_bool(normalized, "has_base_tables")))
        truncated = len(schemas) > _MAX_PROBE_SCHEMAS
        return PostgreSQLProbeResult.build(
            schemas[:_MAX_PROBE_SCHEMAS],
            truncated=truncated,
        )
    except PostgreSQLSourceError as error:
        if error.code == "postgresql_probe_result_invalid":
            raise
    except (TypeError, ValueError):
        pass
    raise PostgreSQLSourceError(
        "postgresql_probe_result_invalid",
        "PostgreSQL returned an invalid schema probe result.",
        source_id="postgresql:probe",
    ) from None


def _probe_row_mapping(row: object) -> Mapping[str, object]:
    if isinstance(row, Mapping):
        return row
    items_method = getattr(row, "items", None)
    if not callable(items_method):
        raise TypeError("probe row must expose a mapping interface")
    raw_items = items_method()
    if not isinstance(raw_items, Iterable):
        raise TypeError("probe row items must be iterable")
    items = tuple(islice(raw_items, _MAX_PROBE_ROW_FIELDS + 1))
    if len(items) > _MAX_PROBE_ROW_FIELDS:
        raise ValueError("probe row contains too many fields")
    normalized = dict(items)
    if len(normalized) != len(items):
        raise ValueError("probe row fields must be unique")
    return normalized


def _load_asyncpg() -> Any:
    try:
        return import_module("asyncpg")
    except ImportError as error:
        raise ImportError(
            "Daita's PostgreSQL runtime dependency is unavailable. "
            f"{PIPX_REPAIR_GUIDANCE}"
        ) from error


async def _connect(
    registration: SourceRegistration,
    secret_provider: SecretProvider,
    *,
    error_type: type[PostgreSQLSourceError] = PostgreSQLSourceError,
) -> Any:
    return await _connect_configuration(
        registration.configuration,
        secret_provider,
        source_id=registration.id,
        error_type=error_type,
    )


async def _connect_configuration(
    configuration: Mapping[str, object],
    secret_provider: SecretProvider,
    *,
    source_id: str,
    error_type: type[PostgreSQLSourceError],
) -> Any:
    asyncpg = _load_asyncpg()
    credential_ref = configuration.get("credential_ref")
    password = None
    if credential_ref is not None:
        if not isinstance(credential_ref, str):
            raise error_type(
                "postgresql_configuration_invalid",
                "PostgreSQL source credential reference is invalid.",
                source_id=source_id,
            )
        resolution_failed = False
        try:
            password = await secret_provider.resolve(
                SecretReference.parse(credential_ref)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # A custom provider can fail with arbitrary, possibly secret-bearing
            # text. Cross the boundary with a fresh normalized error only after
            # leaving that exception context, so diagnostics cannot retain it.
            resolution_failed = True
        if resolution_failed:
            raise error_type(
                "postgresql_credential_unavailable",
                "PostgreSQL credential resolution failed.",
                source_id=source_id,
            )
        if not isinstance(password, str) or not password:
            raise error_type(
                "postgresql_credential_invalid",
                "PostgreSQL credential resolution returned an invalid value.",
                source_id=source_id,
            )
    ssl_mode = _configuration_text(configuration, "ssl_mode")
    connect_failed = False
    try:
        return await asyncpg.connect(
            host=_configuration_text(configuration, "host"),
            port=_configuration_port(configuration),
            database=_configuration_text(configuration, "database"),
            user=_configuration_text(configuration, "username"),
            password=password,
            ssl=False if ssl_mode == "disable" else ssl_mode,
            timeout=_CONNECT_TIMEOUT_SECONDS,
            command_timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        # Connector exceptions can include DSNs, credentials, server banners,
        # or query fragments.  Raise only after leaving the exception context
        # so neither chaining nor traceback formatting can retain that data.
        connect_failed = True
    if connect_failed:
        raise error_type(
            "postgresql_connect_failed",
            "PostgreSQL source could not be opened.",
            source_id=source_id,
        )
    raise AssertionError("PostgreSQL connection attempt returned no result")


async def _rollback_postgresql_transaction(
    transaction: Any,
    connection: Any,
    *,
    timeout_seconds: float,
) -> bool:
    """Bound rollback and synchronously terminate on an uncertain connection."""

    try:
        rollback = transaction.rollback()
    except Exception:
        _terminate_postgresql_connection(connection)
        return False
    try:
        completed = await _await_postgresql_cleanup(
            rollback,
            timeout_seconds=timeout_seconds,
        )
    except asyncio.CancelledError:
        _terminate_postgresql_connection(connection)
        raise
    if not completed:
        _terminate_postgresql_connection(connection)
    return completed


async def _close_postgresql_connection(
    connection: Any,
    *,
    timeout_seconds: float,
) -> bool:
    """Bound graceful close and fall back to synchronous termination."""

    try:
        close = connection.close()
    except Exception:
        _terminate_postgresql_connection(connection)
        return False
    try:
        completed = await _await_postgresql_cleanup(
            close,
            timeout_seconds=timeout_seconds,
        )
    except asyncio.CancelledError:
        _terminate_postgresql_connection(connection)
        raise
    if not completed:
        _terminate_postgresql_connection(connection)
    return completed


async def _await_postgresql_cleanup(
    awaitable: Awaitable[object],
    *,
    timeout_seconds: float,
) -> bool:
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise ValueError("PostgreSQL cleanup timeout must be positive")
    cleanup = asyncio.ensure_future(awaitable)
    try:
        done, _pending = await asyncio.wait(
            (cleanup,),
            timeout=float(timeout_seconds),
        )
    except asyncio.CancelledError:
        cleanup.cancel()
        cleanup.add_done_callback(_consume_cleanup_result)
        raise
    if cleanup not in done:
        cleanup.cancel()
        cleanup.add_done_callback(_consume_cleanup_result)
        return False
    try:
        cleanup.result()
    except asyncio.CancelledError:
        return False
    except Exception:
        return False
    return True


def _terminate_postgresql_connection(connection: Any) -> None:
    terminate = getattr(connection, "terminate", None)
    if not callable(terminate):
        return
    try:
        terminate()
    except Exception:
        pass


def _consume_cleanup_result(cleanup: asyncio.Future[object]) -> None:
    try:
        cleanup.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def _configuration_text(configuration: Mapping[str, object], key: str) -> str:
    value = configuration.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PostgreSQLSourceError(
            "postgresql_configuration_invalid",
            "PostgreSQL source configuration is invalid.",
        )
    return value


def _configuration_port(configuration: Mapping[str, object]) -> int:
    value = configuration.get("port")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 65_535
    ):
        raise PostgreSQLSourceError(
            "postgresql_configuration_invalid",
            "PostgreSQL source configuration is invalid.",
        )
    return value


def _configuration_schemas(configuration: Mapping[str, object]) -> tuple[str, ...]:
    value = configuration.get("schemas")
    if not isinstance(value, tuple):
        raise PostgreSQLSourceError(
            "postgresql_configuration_invalid",
            "PostgreSQL source schema configuration is invalid.",
        )
    return _schemas(value)


def _schemas(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("schemas must be a sequence of identifiers")
    schemas = tuple(values)
    if not schemas or len(schemas) > 32:
        raise ValueError("schemas must contain from 1 through 32 identifiers")
    if len(schemas) != len(set(schemas)):
        raise ValueError("schemas cannot contain duplicates")
    for schema in schemas:
        if not isinstance(schema, str) or _IDENTIFIER.fullmatch(schema) is None:
            raise ValueError("schemas must contain safe PostgreSQL identifiers")
        if _is_system_schema(schema):
            raise ValueError("system PostgreSQL schemas cannot be attached")
    return tuple(sorted(schemas))


def _is_system_schema(value: str) -> bool:
    folded = value.casefold()
    return folded.startswith("pg_") or folded == "information_schema"


def _bounded_text(value: str, name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise ValueError(f"{name} must be a bounded non-empty string")


def _row_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value or len(value) > 4_096:
        raise PostgreSQLSourceError(
            "postgresql_metadata_invalid",
            "PostgreSQL returned invalid structural metadata.",
        )
    return value


def _optional_row_text(
    row: Mapping[str, object],
    key: str,
) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    return _row_text(row, key)


def _row_text_sequence(
    row: Mapping[str, object],
    key: str,
    *,
    maximum_items: int,
) -> tuple[str, ...]:
    values = row.get(key)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) > maximum_items
    ):
        raise PostgreSQLSourceError(
            "postgresql_metadata_invalid",
            "PostgreSQL returned invalid structural metadata.",
        )
    projected: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 4_096:
            raise PostgreSQLSourceError(
                "postgresql_metadata_invalid",
                "PostgreSQL returned invalid structural metadata.",
            )
        projected.append(value)
    return tuple(projected)


def _row_int(row: Mapping[str, object], key: str) -> int:
    return _positive_int(row.get(key), key)


def _positive_int(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PostgreSQLSourceError(
            "postgresql_metadata_invalid",
            f"PostgreSQL returned invalid {key} metadata.",
        )
    return value


def _row_bool(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise PostgreSQLSourceError(
            "postgresql_metadata_invalid",
            "PostgreSQL returned invalid structural metadata.",
        )
    return value


__all__ = [
    "PostgreSQLProbeResult",
    "PostgreSQLProbeSchema",
    "PostgreSQLResourceAdapter",
    "PostgreSQLSource",
    "PostgreSQLSourceError",
]
