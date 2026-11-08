"""Discover and inspect read-admitted PostgreSQL schemas and tables."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import re
import socket
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, replace
from datetime import datetime
from hashlib import sha256
from importlib import import_module
from itertools import islice
from typing import Any, cast
from urllib.parse import quote

from .._installation import repair_guidance
from .._json import canonical_json
from ..catalog.models import (
    CatalogEntry,
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogTier,
    FacetKind,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceBatch,
    ResourceKind,
    Sensitivity,
    TabularColumn,
    TabularFacet,
    TabularIndex,
    catalog_resource_id,
)
from ..security import (
    SecretProvider,
    SecretReference,
    default_secret_provider,
)
from .models import (
    DiscoveryRequest,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from .protocols import (
    ResourceAdapterError,
    ResourceNotFoundError,
    SourceClosedError,
    latest_snapshot,
)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$]{0,62}\Z")
_ROLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.$]{0,127}\Z")
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
_CONNECT_TIMEOUT_SECONDS = 8.0
_COMMAND_TIMEOUT_SECONDS = 8.0
_CLEANUP_TIMEOUT_SECONDS = 1.0
# Bounds reading table detail; a slow source keeps what was read.
_DISCOVERY_SECONDS = 60.0
# Each cursor fetch is a round trip, so the batch sets how latency scales with size.
_BULK_FETCH_ROWS = 5_000

# The system catalogs list everything, whether or not this role can read it, and
# a name it cannot use only misleads. Every metadata query goes through these.
# Tables r, p and f map to TABLE; views v and m to VIEW.
READABLE_RELATION_SQL = (
    "c.relkind IN ('r', 'p', 'f', 'v', 'm') "
    "AND pg_catalog.has_any_column_privilege(c.oid, 'SELECT') "
    "AND pg_catalog.has_schema_privilege(c.relnamespace, 'USAGE')"
)
# A partition folds under an ancestor the role can read in the source's schemas
# ($1), which is what queries read; without one it is listed as its own table.
LISTED_RELATION_SQL = """(NOT c.relispartition OR NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_partition_ancestors(c.oid) AS ancestor(relid)
    JOIN pg_catalog.pg_class AS parent ON parent.oid = ancestor.relid
    JOIN pg_catalog.pg_namespace AS parent_ns ON parent_ns.oid = parent.relnamespace
    WHERE ancestor.relid <> c.oid
      AND parent_ns.nspname = ANY($1::text[])
      AND pg_catalog.has_any_column_privilege(parent.oid, 'SELECT')
      AND pg_catalog.has_schema_privilege(parent.relnamespace, 'USAGE')
))"""
READABLE_COLUMN_SQL = (
    "a.attnum > 0 AND NOT a.attisdropped "
    "AND pg_catalog.has_column_privilege(a.attrelid, a.attnum, 'SELECT')"
)
# Each foreign key with its columns in order. Close the query with
# READABLE_FOREIGN_KEY_SQL after its GROUP BY: a key is readable only whole.
FOREIGN_KEYS_FROM_SQL = """FROM pg_constraint AS con
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
  AND pg_catalog.has_schema_privilege(src.relnamespace, 'USAGE')
  AND pg_catalog.has_schema_privilege(dst.relnamespace, 'USAGE')"""
READABLE_FOREIGN_KEY_SQL = """HAVING bool_and(
    pg_catalog.has_column_privilege(src.oid, src_att.attnum, 'SELECT')
    AND pg_catalog.has_column_privilege(dst.oid, dst_att.attnum, 'SELECT')
)"""
ROW_ESTIMATE_SQL = """CASE WHEN c.relkind = 'p' THEN (
        SELECT sum(GREATEST(child.reltuples, 0))
        FROM pg_inherits AS part
        JOIN pg_class AS child ON child.oid = part.inhrelid
        WHERE part.inhparent = c.oid
    ) ELSE c.reltuples END::double precision"""

_RESOURCES_SQL = f"""
/* daita:postgresql.resources */
SELECT
    n.nspname AS schema_name,
    c.relname AS resource_name,
    CASE WHEN c.relkind IN ('v', 'm') THEN 'view' ELSE 'table' END AS resource_kind,
    c.oid::text AS source_object_id
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE n.nspname = ANY($1::text[])
  AND {READABLE_RELATION_SQL}
  AND {LISTED_RELATION_SQL}
ORDER BY n.nspname, c.relname
LIMIT $2
"""

# The single-table and bulk statements share these fragments so they cannot drift.
_COLUMN_FIELDS_SQL = """    a.attname AS column_name,
    format_type(a.atttypid, a.atttypmod) AS native_type,
    type_ns.nspname AS type_schema,
    type_def.typname AS type_name,
    a.attnum::integer AS ordinal,
    NOT (a.attnotnull OR (type_def.typtype = 'd' AND type_def.typnotnull)) AS nullable,
    CASE WHEN a.attgenerated = '' THEN pg_get_expr(def.adbin, def.adrelid) END
        AS default_expression,
    CASE WHEN coll_ns.nspname = 'pg_catalog' AND coll.collisdeterministic
         THEN coll.collname ELSE NULL END AS collation,
    (a.attidentity <> '') AS is_identity,
    (a.attgenerated <> '') AS is_generated,
    pg_column_is_updatable(rel.oid, a.attnum, false) AS is_updatable,
    pk.primary_key_ordinal,
    pk.primary_key_size,
    col_description(rel.oid, a.attnum) AS comment,
    (
        SELECT array_agg(e.enumlabel::text ORDER BY e.enumsortorder)
        FROM pg_enum AS e
        WHERE e.enumtypid = CASE WHEN type_def.typtype = 'd'
                                 THEN type_def.typbasetype ELSE a.atttypid END
    ) AS enum_values"""
_COLUMN_SOURCES_SQL = f"""FROM pg_namespace AS n
JOIN pg_class AS rel ON rel.relnamespace = n.oid
JOIN pg_attribute AS a
  ON a.attrelid = rel.oid
 AND {READABLE_COLUMN_SQL}
JOIN pg_type AS type_def ON type_def.oid = a.atttypid
JOIN pg_namespace AS type_ns ON type_ns.oid = type_def.typnamespace
LEFT JOIN pg_attrdef AS def ON def.adrelid = rel.oid AND def.adnum = a.attnum
LEFT JOIN pg_collation AS coll ON coll.oid = a.attcollation
LEFT JOIN pg_namespace AS coll_ns ON coll_ns.oid = coll.collnamespace
LEFT JOIN (
    SELECT
        con.conrelid,
        key.attnum,
        key.ordinality::integer AS primary_key_ordinal,
        cardinality(con.conkey) AS primary_key_size
    FROM pg_constraint AS con
    CROSS JOIN LATERAL unnest(con.conkey)
        WITH ORDINALITY AS key(attnum, ordinality)
    WHERE con.contype = 'p'
      AND NOT EXISTS (
          SELECT 1 FROM unnest(con.conkey) AS part(attnum)
          WHERE NOT pg_catalog.has_column_privilege(con.conrelid, part.attnum, 'SELECT')
      )
) AS pk ON pk.conrelid = rel.oid AND pk.attnum = a.attnum"""

COLUMNS_SQL = f"""
/* daita:postgresql.columns */
SELECT
{_COLUMN_FIELDS_SQL}
{_COLUMN_SOURCES_SQL}
WHERE n.nspname = $1 AND rel.relname = $2
  AND pg_catalog.has_schema_privilege(n.oid, 'USAGE')
ORDER BY a.attnum
LIMIT $3
"""

_BULK_COLUMNS_SQL = f"""
/* daita:postgresql.bulk_columns */
SELECT * FROM (
    SELECT
        n.nspname AS schema_name,
        rel.relname AS resource_name,
{_COLUMN_FIELDS_SQL},
        row_number() OVER (PARTITION BY rel.oid ORDER BY a.attnum) AS column_rank
{_COLUMN_SOURCES_SQL}
    JOIN unnest($1::text[], $2::text[]) AS wanted(schema_name, resource_name)
      ON wanted.schema_name = n.nspname AND wanted.resource_name = rel.relname
    WHERE pg_catalog.has_schema_privilege(n.oid, 'USAGE')
) AS ranked
WHERE column_rank <= $3
ORDER BY schema_name, resource_name, ordinal
"""

_INDEX_FIELDS_SQL = """    idx.relname AS index_name,
    am.amname AS index_kind,
    array_agg(att.attname ORDER BY key.ordinality)
        FILTER (WHERE att.attname IS NOT NULL) AS columns,
    bool_and(att.attname IS NOT NULL) AS simple_columns,
    ind.indisunique AS is_unique,
    (ind.indisunique AND ind.indisvalid AND ind.indisready AND ind.indimmediate
     AND ind.indpred IS NULL AND ind.indexprs IS NULL AND am.amname = 'btree'
     AND bool_and(opc.opcdefault AND opc_ns.nspname = 'pg_catalog'
                  AND ind.indcollation[key.ordinality - 1] = att.attcollation))
        AS write_conflict_supported,
    pg_get_expr(ind.indpred, ind.indrelid) AS predicate"""
_INDEX_SOURCES_SQL = """FROM pg_namespace AS n
JOIN pg_class AS rel ON rel.relnamespace = n.oid
JOIN pg_index AS ind ON ind.indrelid = rel.oid
JOIN pg_class AS idx ON idx.oid = ind.indexrelid
JOIN pg_am AS am ON am.oid = idx.relam
CROSS JOIN LATERAL unnest(ind.indkey)
    WITH ORDINALITY AS key(attnum, ordinality)
LEFT JOIN pg_attribute AS att
  ON att.attrelid = rel.oid AND att.attnum = key.attnum
JOIN pg_opclass AS opc ON opc.oid = ind.indclass[key.ordinality - 1]
JOIN pg_namespace AS opc_ns ON opc_ns.oid = opc.opcnamespace"""
# The bulk statement selects and partitions by schema, table name and oid.
_INDEX_GROUP_SQL = """GROUP BY n.nspname, rel.relname, rel.oid, idx.relname, am.amname,
         ind.indisunique, ind.indpred, ind.indrelid, ind.indisvalid, ind.indisready,
         ind.indimmediate, ind.indexprs
HAVING bool_and(
    att.attnum IS NULL
    OR pg_catalog.has_column_privilege(rel.oid, att.attnum, 'SELECT')
)"""

INDEXES_SQL = f"""
/* daita:postgresql.indexes */
SELECT
{_INDEX_FIELDS_SQL}
{_INDEX_SOURCES_SQL}
WHERE n.nspname = $1
  AND rel.relname = $2
  AND key.ordinality <= ind.indnkeyatts
{_INDEX_GROUP_SQL}
ORDER BY idx.relname
LIMIT $3
"""

_BULK_INDEXES_SQL = f"""
/* daita:postgresql.bulk_indexes */
SELECT * FROM (
    SELECT
        n.nspname AS schema_name,
        rel.relname AS resource_name,
{_INDEX_FIELDS_SQL},
        row_number() OVER (PARTITION BY rel.oid ORDER BY idx.relname) AS index_rank
{_INDEX_SOURCES_SQL}
    JOIN unnest($1::text[], $2::text[]) AS wanted(schema_name, resource_name)
      ON wanted.schema_name = n.nspname AND wanted.resource_name = rel.relname
    WHERE key.ordinality <= ind.indnkeyatts
{_INDEX_GROUP_SQL}
) AS ranked
WHERE index_rank <= $3
ORDER BY schema_name, resource_name, index_name
"""

_RELATIONSHIPS_SQL = f"""
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
{FOREIGN_KEYS_FROM_SQL}
  AND src_ns.nspname = ANY($1::text[])
  AND dst_ns.nspname = ANY($1::text[])
GROUP BY con.oid, con.conname, src_ns.nspname, src.relname,
         dst_ns.nspname, dst.relname, con.confmatchtype,
         con.confupdtype, con.confdeltype
{READABLE_FOREIGN_KEY_SQL}
ORDER BY src_ns.nspname, src.relname, con.conname
LIMIT $2
"""

# One token per configured schema the role can use: a hash of what detail would
# store for its listed, readable relations. It reuses the discovery fragments, so
# a grant, revoke or role change that alters what is readable moves it, and
# nothing it hashes (estimates, statistics, table comments, enum labels) is
# something a revision does not store.
_CHANGE_PROBE_SQL = f"""
/* daita:postgresql.change_probe */
WITH rels AS MATERIALIZED (
    SELECT c.oid, n.nspname, c.relname, c.relkind,
        pg_catalog.has_table_privilege(c.oid, 'SELECT') AS whole
    FROM pg_catalog.pg_class AS c
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    WHERE n.nspname = ANY($1::text[])
      AND {READABLE_RELATION_SQL}
      AND {LISTED_RELATION_SQL}
), cols AS MATERIALIZED (
    -- Order-free: a sum of per-column hashes needs no sort. A table-level SELECT
    -- grant makes every column readable, so has_column_privilege runs only for
    -- tables without one (the same readable columns as READABLE_COLUMN_SQL).
    SELECT a.attrelid AS oid,
        sum(hashtextextended(ROW(
            a.attnum, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod),
            a.attnotnull, t.typtype = 'd' AND t.typnotnull, a.attidentity,
            a.attgenerated, a.attcollation, note.description,
            CASE WHEN a.attgenerated = ''
                 THEN pg_catalog.pg_get_expr(def.adbin, def.adrelid) END,
            CASE WHEN rels.relkind IN ('v', 'f')
                 THEN pg_catalog.pg_column_is_updatable(a.attrelid, a.attnum, false)
            END)::text, 0)::numeric) AS d
    FROM rels
    JOIN pg_catalog.pg_attribute AS a ON a.attrelid = rels.oid
    JOIN pg_catalog.pg_type AS t ON t.oid = a.atttypid
    LEFT JOIN pg_catalog.pg_attrdef AS def
      ON def.adrelid = a.attrelid AND def.adnum = a.attnum
    LEFT JOIN pg_catalog.pg_description AS note
      ON note.objoid = a.attrelid AND note.objsubid = a.attnum
     AND note.classoid = 'pg_catalog.pg_class'::regclass
    WHERE a.attnum > 0 AND NOT a.attisdropped
      AND (rels.whole OR pg_catalog.has_column_privilege(a.attrelid, a.attnum, 'SELECT'))
    GROUP BY a.attrelid
), pks AS MATERIALIZED (
    SELECT con.conrelid AS oid, con.conkey::text AS d
    FROM pg_catalog.pg_constraint AS con
    JOIN rels ON rels.oid = con.conrelid
    WHERE con.contype = 'p'
      AND (rels.whole OR NOT EXISTS (
          SELECT 1 FROM unnest(con.conkey) AS part(attnum)
          WHERE NOT pg_catalog.has_column_privilege(con.conrelid, part.attnum, 'SELECT')
      ))
), idxs AS MATERIALIZED (
    SELECT oid, string_agg(ROW(index_name, index_kind, columns, is_unique,
        write_conflict_supported, predicate)::text, ';' ORDER BY index_name) AS d
    FROM (
        SELECT rel.oid AS oid,
{_INDEX_FIELDS_SQL}
{_INDEX_SOURCES_SQL}
        JOIN rels ON rels.oid = rel.oid
        WHERE key.ordinality <= ind.indnkeyatts
{_INDEX_GROUP_SQL}
    ) AS listed_indexes
    WHERE simple_columns
    GROUP BY oid
), fkeys AS MATERIALIZED (
    SELECT src.oid AS src_oid, dst.oid AS dst_oid,
        ROW(con.conname, src_ns.nspname, src.relname, dst_ns.nspname, dst.relname,
            array_agg(src_att.attname ORDER BY keys.ordinality),
            array_agg(dst_att.attname ORDER BY keys.ordinality),
            con.confmatchtype, con.confupdtype, con.confdeltype)::text AS d
    {FOREIGN_KEYS_FROM_SQL}
      AND src.oid IN (SELECT oid FROM rels) AND dst.oid IN (SELECT oid FROM rels)
    GROUP BY con.oid, con.conname, src.oid, src_ns.nspname, src.relname,
             dst.oid, dst_ns.nspname, dst.relname, con.confmatchtype,
             con.confupdtype, con.confdeltype
    {READABLE_FOREIGN_KEY_SQL}
), fks AS MATERIALIZED (
    SELECT oid, string_agg(d, ';' ORDER BY d) AS d
    FROM (
        SELECT src_oid AS oid, d FROM fkeys
        UNION ALL
        SELECT dst_oid, d FROM fkeys WHERE dst_oid <> src_oid
    ) AS ends
    GROUP BY oid
)
SELECT n.nspname AS scope,
    md5(coalesce(string_agg(
        ROW(rels.oid, rels.relname, rels.relkind, cols.d, pks.d, idxs.d, fks.d)::text,
        E'\\n' ORDER BY rels.relname) FILTER (WHERE rels.oid IS NOT NULL), ''))
        AS token
FROM pg_catalog.pg_namespace AS n
LEFT JOIN rels ON rels.nspname = n.nspname
LEFT JOIN cols ON cols.oid = rels.oid
LEFT JOIN pks ON pks.oid = rels.oid
LEFT JOIN idxs ON idxs.oid = rels.oid
LEFT JOIN fks ON fks.oid = rels.oid
WHERE n.nspname = ANY($1::text[]) AND pg_catalog.has_schema_privilege(n.oid, 'USAGE')
GROUP BY n.nspname
"""


async def probe_tokens(
    connection: Any, schemas: Sequence[str], *, timeout: float = 5.0
) -> dict[str, str]:
    """One statement: a token for each of ``schemas`` the connected role can use."""

    rows = await connection.fetch(_CHANGE_PROBE_SQL, list(schemas), timeout=timeout)
    return {_row_text(row, "scope"): _row_text(row, "token") for row in rows}


# Planner statistics are approximate and nonstructural. They are read once at
# discovery, never during query revalidation, and never change a revision.
_ESTIMATES_SQL = f"""
/* daita:postgresql.estimates */
SELECT
    n.nspname AS schema_name,
    c.relname AS resource_name,
    {ROW_ESTIMATE_SQL} AS reltuples,
    pg_total_relation_size(c.oid) AS size_bytes,
    GREATEST(
        s.last_vacuum,
        s.last_autovacuum,
        s.last_analyze,
        s.last_autoanalyze
    ) AS statistics_as_of
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
JOIN unnest($1::text[], $2::text[]) AS known(schema_name, resource_name)
  ON known.schema_name = n.nspname AND known.resource_name = c.relname
LEFT JOIN pg_stat_user_tables AS s ON s.relid = c.oid
WHERE c.relkind IN ('r', 'p', 'f', 'm')
ORDER BY n.nspname, c.relname
LIMIT $3
"""

_SCHEMA_PROBE_SQL = f"""
/* daita:postgresql.schema_probe */
SELECT
    namespace.nspname AS schema_name,
    EXISTS (
        SELECT 1
        FROM pg_class AS c
        WHERE c.relnamespace = namespace.oid
          AND {READABLE_RELATION_SQL}
    ) AS has_base_tables
FROM pg_namespace AS namespace
WHERE namespace.nspname <> 'information_schema'
  AND namespace.nspname NOT LIKE 'pg\\_%' ESCAPE '\\'
ORDER BY namespace.nspname
LIMIT $1
"""


# What this role could change, for the person who attached it. Joining pg_roles
# on the built-in names keeps servers that lack one of them from erroring.
_HEALTH_SQL = """
/* daita:postgresql.health */
SELECT
    current_user::text AS role_name,
    me.rolsuper,
    me.rolbypassrls,
    me.rolcreaterole,
    me.rolcreatedb,
    COALESCE((
        SELECT pg_catalog.array_agg(grp.rolname::text ORDER BY grp.rolname)
        FROM pg_catalog.pg_roles AS grp
        WHERE grp.rolname IN (
            'pg_read_all_data', 'pg_write_all_data', 'pg_read_server_files',
            'pg_write_server_files', 'pg_execute_server_program'
        )
          AND pg_catalog.pg_has_role(current_user, grp.oid, 'MEMBER')
    ), ARRAY[]::text[]) AS builtin_roles,
    (
        SELECT pg_catalog.count(*)
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = ANY($1::text[])
          AND c.relkind IN ('r', 'p', 'f')
          AND pg_catalog.has_schema_privilege(n.oid, 'USAGE')
          AND (
              pg_catalog.has_table_privilege(
                  c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE')
              OR pg_catalog.has_any_column_privilege(c.oid, 'INSERT, UPDATE')
          )
    )::integer AS writable_tables,
    (
        SELECT pg_catalog.count(*)
        FROM pg_catalog.pg_namespace AS n
        WHERE n.nspname = ANY($1::text[])
          AND pg_catalog.has_schema_privilege(n.oid, 'CREATE')
    )::integer AS creatable_schemas
FROM pg_catalog.pg_roles AS me
WHERE me.rolname = current_user
"""
_BUILTIN_ROLES = (
    "pg_read_all_data",
    "pg_write_all_data",
    "pg_read_server_files",
    "pg_write_server_files",
    "pg_execute_server_program",
)
_WRITING_BUILTIN_ROLES = frozenset(
    ("pg_write_all_data", "pg_write_server_files", "pg_execute_server_program")
)

POSTGRESQL_ADAPTER_ID = "postgresql"


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
    public_network_only: bool = False
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
        if _ROLE_NAME.fullmatch(self.username) is None:
            raise ValueError("username must be a safe PostgreSQL role name")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65_535
        ):
            raise ValueError("port must be from 1 through 65535")
        schemas = _schemas(self.schemas)
        if self.ssl_mode not in _SSL_MODES:
            raise ValueError("ssl_mode is not supported")
        if not isinstance(self.public_network_only, bool):
            raise TypeError("public_network_only must be a boolean")
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


def postgresql_source_from_registration(
    registration: SourceRegistration,
    *,
    secret_provider: SecretProvider,
) -> PostgreSQLSource:
    """Reconstruct one admitted PostgreSQL source from its exact configuration."""

    configuration = registration.configuration
    required = {"database", "host", "port", "schemas", "ssl_mode", "username"}
    fields = set(configuration)
    if not required <= fields or not fields <= required | {
        "credential_ref",
        "public_network_only",
    }:
        raise ValueError("PostgreSQL source configuration is invalid")
    raw_schemas = configuration["schemas"]
    if not isinstance(raw_schemas, tuple) or any(
        not isinstance(schema, str) for schema in raw_schemas
    ):
        raise ValueError("PostgreSQL source configuration is invalid")
    raw_reference = configuration.get("credential_ref")
    if raw_reference is not None and not isinstance(raw_reference, str):
        raise ValueError("PostgreSQL source configuration is invalid")
    texts = {}
    for name in ("host", "database", "username", "ssl_mode"):
        value = configuration[name]
        if not isinstance(value, str) or not value:
            raise ValueError("source configuration text is invalid")
        texts[name] = value
    port = configuration["port"]
    if not isinstance(port, int) or isinstance(port, bool):
        raise ValueError("source configuration integer is invalid")
    public_network_only = configuration.get("public_network_only", False)
    if not isinstance(public_network_only, bool):
        raise ValueError("source configuration boolean is invalid")
    try:
        return PostgreSQLSource(
            host=texts["host"],
            port=port,
            database=texts["database"],
            username=texts["username"],
            credential=(
                None if raw_reference is None else SecretReference.parse(raw_reference)
            ),
            schemas=raw_schemas,
            ssl_mode=texts["ssl_mode"],
            public_network_only=public_network_only,
            name=registration.display_name,
            secret_provider=secret_provider,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("PostgreSQL source configuration is invalid") from error


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
        self._latest: ResourceBatch | None = None

    @property
    def registration(self) -> SourceRegistration:
        return self._registration

    async def reconcile(
        self, request: DiscoveryRequest
    ) -> AsyncIterator[ResourceBatch]:
        """Yield the inventory, then the detail; all reads finish before the first yield."""

        self._require_request(request)
        async with self._lock:
            self._require_open()
            transaction = None
            transaction_started = False
            transaction_finished = False
            discovery_failed = False
            batches: tuple[ResourceBatch, ResourceBatch] | None = None
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
                    max_inventory=request.max_inventory,
                    max_resources=request.max_resources,
                    max_columns=request.max_columns_per_resource,
                    max_indexes=request.max_indexes_per_resource,
                    max_relationships=request.max_relationships,
                    scopes=request.scopes,
                    previous=request.previous,
                    fill=request.fill,
                )
                estimates = await _load_estimates(self._connection, structure)
                batches = _batches(request, structure, estimates)
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
            if batches is None:
                raise AssertionError("PostgreSQL discovery completed without batches")
            self._latest = batches[1]
        for batch in batches:
            yield batch

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
            return latest_snapshot(self._latest, resource, self._clock())

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
                row = await self._connection.fetchrow(
                    _HEALTH_SQL,
                    list(_configuration_schemas(self._registration.configuration)),
                )
                details = _health_details(row)
            except asyncio.CancelledError:
                raise
            except Exception:
                details = None
            return SourceHealth(
                agent_id=self._registration.agent_id,
                source_id=self._registration.id,
                adapter_id="postgresql",
                healthy=details is not None,
                checked_at=checked_at,
                error_code=None if details else "postgresql_health_failed",
                details=details or {},
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
    # Every listed relation (schema, name, kind, pg_class oid), detailed or not.
    inventory: tuple[tuple[str, str, ResourceKind, str | None], ...] = ()
    inventory_truncated: bool = False
    truncated: bool = False


async def _read_by_table(
    connection: Any,
    sql: str,
    wanted: Sequence[tuple[str, str]],
    limit: int,
    deadline: float,
    build: Callable[[tuple[str, str], list[Any]], None],
) -> int:
    """Stream one bulk statement and hand `build` each table's rows once all arrived.

    Only one table's raw rows are held. Returns how many leading `wanted` tables
    finished; the budget cutting the stream discards the table in progress.
    """

    loop = asyncio.get_running_loop()
    if not wanted or (remaining := deadline - loop.time()) <= 0:
        return 0
    position = {key: index for index, key in enumerate(wanted)}
    current: tuple[str, str] | None = None
    rows: list[Any] = []
    try:
        # A fetch that outlasts the budget aborts only this savepoint, so the
        # relationships and estimates reads can still run.
        async with connection.transaction():
            cursor = await connection.cursor(
                sql,
                [schema for schema, _ in wanted],
                [name for _, name in wanted],
                limit + 1,
                timeout=remaining,
            )
            while (remaining := deadline - loop.time()) > 0:
                batch = await cursor.fetch(_BULK_FETCH_ROWS, timeout=remaining)
                for row in batch:
                    key = (
                        _row_text(row, "schema_name"),
                        _row_text(row, "resource_name"),
                    )
                    if key != current:
                        if current is not None:
                            build(current, rows)
                        current, rows = key, []
                    rows.append(row)
                if len(batch) < _BULK_FETCH_ROWS:
                    if current is not None:
                        build(current, rows)
                    return len(wanted)
    except TimeoutError:
        pass
    return position[current] if current is not None else 0


async def _load_structure(
    connection: Any,
    registration: SourceRegistration,
    *,
    max_resources: int,
    max_columns: int,
    max_indexes: int,
    max_relationships: int,
    max_inventory: int = 1_000_000,
    scopes: tuple[str, ...] = (),
    previous: tuple[CatalogEntry, ...] | None = None,
    fill: bool = False,
) -> PostgreSQLStructure:
    """Read the structure the request allows; whatever a limit cuts is truncated."""

    schemas = _configuration_schemas(registration.configuration)
    resource_rows = await connection.fetch(
        _RESOURCES_SQL,
        list(schemas),
        max_inventory + 1,
    )
    inventory_truncated = len(resource_rows) > max_inventory
    resource_rows = resource_rows[:max_inventory]
    # A fill describes what lacks detail, up to max_resources of it; any other
    # pass describes a prefix of the listing.
    truncated = not fill and len(resource_rows) > max_resources
    # A table outside the refreshed schemas keeps what the previous pass read.
    kept = (
        {entry.resource.id: entry for entry in previous if entry.detailed}
        if (scopes or fill) and previous is not None
        else {}
    )
    fresh = 0
    inventory: list[tuple[str, str, ResourceKind, str | None]] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _DISCOVERY_SECONDS
    listed: list[tuple[str, str, ResourceKind, TabularFacet | None]] = []
    wanted: list[tuple[str, str]] = []
    for position, row in enumerate(resource_rows):
        schema = _row_text(row, "schema_name")
        name = _row_text(row, "resource_name")
        kind_value = _row_text(row, "resource_kind")
        kind = ResourceKind.VIEW if kind_value == "view" else ResourceKind.TABLE
        inventory.append(
            (schema, name, kind, _optional_row_text(row, "source_object_id"))
        )
        if not fill and position >= max_resources:
            continue
        previous_entry = (
            None
            if schema in scopes and not fill
            else kept.get(
                catalog_resource_id(registration.id, kind, f"{schema}.{name}")
            )
        )
        facet = next(
            (
                item
                for item in (() if previous_entry is None else previous_entry.facets)
                if item.kind is FacetKind.TABULAR
            ),
            None,
        )
        tabular = (
            TabularFacet.from_payload(facet.payload) if facet is not None else None
        )
        if tabular is not None:
            # The previous read may have been cut at these limits.
            truncated |= (
                len(tabular.columns) >= max_columns
                or len(tabular.indexes) >= max_indexes
            )
        else:
            if fill and fresh >= max_resources:
                truncated = True
                continue
            fresh += 1
            wanted.append((schema, name))
        listed.append((schema, name, kind, tabular))

    columns: dict[tuple[str, str], tuple[TabularColumn, ...]] = {}
    indexes: dict[tuple[str, str], tuple[TabularIndex, ...]] = {}

    def build_columns(key: tuple[str, str], rows: list[Any]) -> None:
        nonlocal truncated
        columns[key] = columns_from_rows(rows, max_columns)
        truncated |= len(rows) > max_columns

    def build_indexes(key: tuple[str, str], rows: list[Any]) -> None:
        nonlocal truncated
        known_columns = {column.name for column in columns.get(key, ())}
        indexes[key] = tuple(
            index
            for row in rows[:max_indexes]
            if (index := _index(row, known_columns)) is not None
        )
        truncated |= len(rows) > max_indexes

    # A quarter of the budget is left so tables whose columns arrived can get indexes.
    read = await _read_by_table(
        connection,
        _BULK_COLUMNS_SQL,
        wanted,
        max_columns,
        deadline - _DISCOVERY_SECONDS / 4,
        build_columns,
    )
    done = await _read_by_table(
        connection,
        _BULK_INDEXES_SQL,
        wanted[:read],
        max_indexes,
        deadline,
        build_indexes,
    )
    truncated |= done < len(wanted)
    read_keys = set(wanted[:done])
    tables: list[_TableStructure] = []
    for schema, name, kind, tabular in listed:
        key = (schema, name)
        if tabular is not None:
            table_columns, table_indexes = tabular.columns, tabular.indexes
        elif key in read_keys:
            table_columns = columns.get(key, ())
            table_indexes = indexes.get(key, ())
        else:
            continue
        tables.append(
            _TableStructure(
                schema=schema,
                name=name,
                kind=kind,
                columns=table_columns,
                indexes=table_indexes,
            )
        )
    relationship_rows = await connection.fetch(
        _RELATIONSHIPS_SQL,
        list(schemas),
        max_relationships + 1,
    )
    truncated |= len(relationship_rows) > max_relationships
    known = {
        (table.schema, table.name): {column.name for column in table.columns}
        for table in tables
    }
    relationships = tuple(
        relation
        for row in relationship_rows[:max_relationships]
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
        inventory=tuple(inventory),
        inventory_truncated=inventory_truncated,
        truncated=truncated,
    )


@dataclass(frozen=True, slots=True)
class _TableEstimates:
    """Approximate planner statistics; never exact row counts."""

    row_count: int | None
    size_bytes: int | None
    statistics_as_of: datetime | None


async def _load_estimates(
    connection: Any,
    structure: PostgreSQLStructure,
) -> dict[tuple[str, str], _TableEstimates]:
    if not structure.tables:
        return {}
    rows = await connection.fetch(
        _ESTIMATES_SQL,
        [table.schema for table in structure.tables],
        [table.name for table in structure.tables],
        len(structure.tables) + 1,
    )
    known = {(table.schema, table.name) for table in structure.tables}
    estimates: dict[tuple[str, str], _TableEstimates] = {}
    for row in rows:
        key = (_row_text(row, "schema_name"), _row_text(row, "resource_name"))
        if key not in known:
            continue
        estimates[key] = _table_estimates(row)
    return estimates


def _table_estimates(row: Mapping[str, object]) -> _TableEstimates:
    raw_tuples = row.get("reltuples")
    raw_size = row.get("size_bytes")
    raw_as_of = row.get("statistics_as_of")
    as_of = (
        raw_as_of
        if isinstance(raw_as_of, datetime) and raw_as_of.utcoffset() is not None
        else None
    )
    row_count: int | None = None
    if (
        isinstance(raw_tuples, (int, float))
        and not isinstance(raw_tuples, bool)
        and math.isfinite(raw_tuples)
        and raw_tuples >= 0
        # PostgreSQL before 14 stores 0 for a never-analyzed table. Without a
        # statistics timestamp that zero is unknown, not an empty table.
        and (raw_tuples > 0 or as_of is not None)
    ):
        row_count = int(round(raw_tuples))
    size_bytes = (
        raw_size
        if isinstance(raw_size, int)
        and not isinstance(raw_size, bool)
        and raw_size >= 0
        else None
    )
    return _TableEstimates(row_count, size_bytes, as_of)


def _external_uri(source_id: str, schema: str, name: str) -> str:
    return f"postgresql://{source_id}/{quote(schema, safe='')}/{quote(name, safe='')}"


def _batches(
    request: DiscoveryRequest,
    structure: PostgreSQLStructure,
    estimates: Mapping[tuple[str, str], _TableEstimates] | None = None,
) -> tuple[ResourceBatch, ResourceBatch]:
    """The inventory of every listed relation and the detail that was read."""

    inventory = ResourceBatch(
        tier=CatalogTier.INVENTORY,
        entries=tuple(
            request.inventory_entry(
                kind=kind,
                native_identity=f"{schema}.{name}",
                name=name,
                external_uri=_external_uri(request.source_id, schema, name),
                source_object_id=oid,
            )
            for schema, name, kind, oid in structure.inventory
        ),
        truncated=structure.inventory_truncated,
    )
    registration_id = request.source_id
    oids = {(schema, name): oid for schema, name, _, oid in structure.inventory}
    facets: dict[tuple[str, str], CatalogFacet] = {}
    resource_ids: dict[tuple[str, str], tuple[str, ResourceKind]] = {}
    for table in structure.tables:
        native_identity = f"{table.schema}.{table.name}"
        resource_id = catalog_resource_id(
            registration_id,
            table.kind,
            native_identity,
        )
        resource_ids[(table.schema, table.name)] = (resource_id, table.kind)
        estimate = (estimates or {}).get((table.schema, table.name))
        facets[(table.schema, table.name)] = CatalogFacet.from_tabular(
            resource_id=resource_id,
            sync_id=request.sync_id,
            observed_at=request.requested_at,
            facet=TabularFacet(
                columns=table.columns,
                indexes=table.indexes,
                row_count_estimate=None if estimate is None else estimate.row_count,
                size_bytes_estimate=(None if estimate is None else estimate.size_bytes),
                statistics_as_of=(
                    None if estimate is None else estimate.statistics_as_of
                ),
            ),
        )
    relationships: list[CatalogRelationship] = []
    for relation in structure.relationships:
        source = resource_ids.get((relation.source_schema, relation.source_table))
        target = resource_ids.get((relation.target_schema, relation.target_table))
        if source is None or target is None:
            continue
        relationships.append(
            CatalogRelationship.build(
                source_id=registration_id,
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
    entries: list[CatalogEntry] = []
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
        entries.append(
            CatalogEntry(
                resource=CatalogResource.build(
                    agent_id=request.agent_id,
                    source_id=request.source_id,
                    native_identity=f"{table.schema}.{table.name}",
                    external_uri=_external_uri(
                        request.source_id, table.schema, table.name
                    ),
                    kind=kind,
                    name=table.name,
                    sensitivity=Sensitivity.INTERNAL,
                    revision=revision,
                    first_observed_at=request.requested_at,
                    last_observed_at=request.requested_at,
                ),
                revision=revision,
                facets=(facet,),
                detailed=True,
                source_object_id=oids.get((table.schema, table.name)),
            )
        )
    detail = ResourceBatch(
        tier=CatalogTier.DETAIL,
        entries=tuple(entries),
        relationships=ordered_relationships,
        truncated=structure.truncated,
        source_revision=structure.source_revision,
    )
    return inventory, detail


def columns_from_rows(
    rows: Sequence[Mapping[str, object]], limit: int
) -> tuple[TabularColumn, ...]:
    """The first ``limit`` columns; a primary key the cut splits shows not at all."""

    kept = rows[:limit]
    columns = tuple(column_from_row(row) for row in kept)
    keyed = [row for row in kept if row.get("primary_key_ordinal") is not None]
    if keyed and len(keyed) < _row_int(keyed[0], "primary_key_size"):
        return tuple(replace(column, primary_key_ordinal=None) for column in columns)
    return columns


def column_from_row(row: Mapping[str, object]) -> TabularColumn:
    ordinal = _row_int(row, "ordinal")
    primary = row.get("primary_key_ordinal")
    type_schema = _row_text(row, "type_schema")
    type_name = _row_text(row, "type_name")
    display_type = _row_text(row, "native_type")
    return TabularColumn(
        name=_row_text(row, "column_name"),
        native_type=display_type,
        native_type_namespace=type_schema,
        native_type_name=type_name,
        ordinal=ordinal - 1,
        nullable=_row_bool(row, "nullable"),
        primary_key_ordinal=(
            None if primary is None else _positive_int(primary, "primary_key_ordinal")
        ),
        default_expression=_optional_row_text(row, "default_expression"),
        collation=_optional_row_text(row, "collation"),
        identity=_row_bool(row, "is_identity"),
        generated=_row_bool(row, "is_generated"),
        updatable=_row_bool(row, "is_updatable"),
        comment=(
            comment[:2_048] if (comment := _optional_row_text(row, "comment")) else None
        ),
    )


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
        write_conflict_supported=row.get("write_conflict_supported") is True,
        predicate=_optional_row_text(row, "predicate"),
    )


def _relationship(
    row: Mapping[str, object],
    known: Mapping[tuple[str, str], set[str]],
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
    # A key naming a column the column limit cut would point outside its table.
    if not (
        set(source_columns) <= known[source] and set(target_columns) <= known[target]
    ):
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
    if source.public_network_only:
        configuration["public_network_only"] = True
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


def _health_details(row: Mapping[str, object]) -> dict[str, object]:
    """The role, whether it can only read, and a sentence for each broad grant."""

    # Only the fixed names are repeated, never server text.
    held = set(_BUILTIN_ROLES).intersection(cast(Sequence[str], row["builtin_roles"]))
    writable = cast(int, row["writable_tables"])
    creatable = cast(int, row["creatable_schemas"])
    superuser = cast(bool, row["rolsuper"])
    warnings: list[str] = []
    if superuser:
        warnings.append("This role is a superuser, so it can read and change anything.")
    if row["rolbypassrls"]:
        warnings.append("This role bypasses row-level security.")
    if row["rolcreaterole"] or row["rolcreatedb"]:
        warnings.append("This role can create roles or databases.")
    # A superuser implicitly holds every built-in role; its own warning says so.
    if held and not superuser:
        warnings.append(
            f"This role belongs to broad built-in roles: {', '.join(sorted(held))}."
        )
    if writable:
        warnings.append(
            f"This role can change data in {writable} table"
            f"{'' if writable == 1 else 's'}. Daita only writes what is enabled "
            "and approved, but a read-only role is safer unless writes are planned."
        )
    if creatable:
        warnings.append("This role can create objects in a configured schema.")
    return {
        "role": str(row["role_name"]),
        "read_only": not (
            superuser or writable or creatable or held & _WRITING_BUILTIN_ROLES
        ),
        "warnings": tuple(warnings),
    }


def _load_asyncpg() -> Any:
    try:
        return import_module("asyncpg")
    except ImportError as error:
        raise ImportError(
            "Daita's PostgreSQL runtime dependency is unavailable. "
            f"{repair_guidance()}"
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
    host = _configuration_text(configuration, "host")
    port = _configuration_port(configuration)
    public_network_only = configuration.get("public_network_only", False)
    if not isinstance(public_network_only, bool):
        raise error_type(
            "postgresql_configuration_invalid",
            "PostgreSQL source network policy is invalid.",
            source_id=source_id,
        )
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
        async with asyncio.timeout(_CONNECT_TIMEOUT_SECONDS):
            if public_network_only:
                await _require_public_network_destination(
                    host,
                    port,
                    source_id=source_id,
                    error_type=error_type,
                )
            return await asyncpg.connect(
                host=host,
                port=port,
                database=_configuration_text(configuration, "database"),
                user=_configuration_text(configuration, "username"),
                password=password,
                ssl=False if ssl_mode == "disable" else ssl_mode,
                timeout=_CONNECT_TIMEOUT_SECONDS,
                command_timeout=_COMMAND_TIMEOUT_SECONDS,
            )
    except asyncio.CancelledError:
        raise
    except error_type:
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


async def _require_public_network_destination(
    host: str,
    port: int,
    *,
    source_id: str,
    error_type: type[PostgreSQLSourceError],
) -> None:
    resolution_failed = False
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        resolution_failed = True
        addresses = []
    if resolution_failed or not addresses:
        raise error_type(
            "postgresql_host_resolution_failed",
            "PostgreSQL host could not be resolved.",
            source_id=source_id,
        )
    if any(not _is_public_ip(str(item[4][0])) for item in addresses):
        raise error_type(
            "postgresql_network_disallowed",
            "PostgreSQL host must resolve only to public network addresses.",
            source_id=source_id,
        )


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return address.is_global and not address.is_multicast


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
    "POSTGRESQL_ADAPTER_ID",
    "PostgreSQLProbeResult",
    "PostgreSQLProbeSchema",
    "PostgreSQLResourceAdapter",
    "PostgreSQLSource",
    "PostgreSQLSourceError",
    "postgresql_source_from_registration",
]
