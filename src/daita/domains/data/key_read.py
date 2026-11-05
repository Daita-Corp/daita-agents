"""Read the distinct values of key columns once, and compare two such reads.

This is one bounded read on the normal read path: SQL is validated against the
source's real scope before any I/O, exactly as ``data_query`` does, and no
catalog state is consulted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from ..._json import canonical_json, thaw_json
from ...catalog.models import ResourceKind, catalog_resource_id
from ...errors import DaitaError
from .capabilities import (
    ReadTabularRequest,
    SqlReadBackend,
    TabularReadBackend,
    _execute_read,
)
from .sql import CheckedRead, validate_read
from .sql.contracts import _SqlDialect
from .sql.duckdb_read import DuckDBReadValidationError, validate_duckdb_read


@dataclass(frozen=True, slots=True)
class KeyEndpoint:
    source_id: str
    table: str  # native name; for a folder source, the path pattern
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KeyRead:
    keys: frozenset[tuple[object, ...]]
    truncated: bool
    read_at: datetime


@dataclass(frozen=True, slots=True)
class KeyComparison:
    left_keys: int
    right_keys: int
    matched_left: int
    matched_right: int
    left_key_types: tuple[str, ...]
    right_key_types: tuple[str, ...]
    match_rate: float | None


class KeyReadError(DaitaError):
    """A key read that policy refused or the source could not serve."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message, error_code=code)


def _key_sql(endpoint: KeyEndpoint, *, folder: bool) -> str:
    def quote(name: str) -> str:  # ANSI quoting: valid in SQLite, PostgreSQL and DuckDB
        return '"' + name.replace('"', '""') + '"'

    if folder:
        relation, owner = "data", "data"
    else:
        # The catalog names a SQL source's tables schema.table (SQLite's is main).
        schema, dot, name = endpoint.table.partition(".")
        if dot:
            relation, owner = f"{quote(schema)}.{quote(name)}", quote(name)
        else:
            relation = owner = quote(endpoint.table)
    # Qualified, because SQLite reads an unknown quoted name as a string literal.
    columns = [f"{owner}.{quote(column)}" for column in endpoint.columns]
    return (
        f"SELECT DISTINCT {', '.join(columns)} FROM {relation} WHERE "
        + " AND ".join(f"{column} IS NOT NULL" for column in columns)
    )


def checked_key_read(
    endpoint: KeyEndpoint,
    dialect: str | None,
    *,
    access: frozenset[str] | None,
    schemas: tuple[str, ...],
) -> CheckedRead | None:
    """Validate the key read against the source's scope before any I/O.

    A source without a SQL dialect is read by column name: its table must be in
    scope, and None is returned.
    """

    if dialect is None:
        resource_id = catalog_resource_id(
            endpoint.source_id, ResourceKind.TABLE, endpoint.table
        )
        if access is not None and resource_id not in access:
            raise KeyReadError(
                "key_read_out_of_scope", "The key read was refused before reading."
            )
        return None
    sql = _key_sql(endpoint, folder=dialect == "duckdb")
    allowed = None if access is None else tuple(sorted(access))
    if dialect == "duckdb":
        try:
            folder_read = validate_duckdb_read(sql)
        except DuckDBReadValidationError as error:
            raise KeyReadError("key_read_refused", error.message) from None
        return CheckedRead(
            canonical_sql=folder_read.canonical_sql,
            sql_fingerprint=folder_read.sql_fingerprint,
            path_pattern=endpoint.table,
            allowed_resource_ids=allowed,
        )
    result = validate_read(
        sql,
        dialect=cast(_SqlDialect, dialect),
        source_id=endpoint.source_id,
        allowed_resource_ids=access,
        schemas=schemas,
    )
    if not result.valid or result.analysis is None:
        codes = {item.code for item in result.issues}
        raise KeyReadError(
            (
                "key_read_out_of_scope"
                if codes <= {"resource_out_of_scope", "schema_not_configured"}
                else "key_read_refused"
            ),
            "The key read was refused before reading.",
        )
    return CheckedRead(
        canonical_sql=result.analysis.canonical_sql,
        sql_fingerprint=result.analysis.sql_fingerprint,
        function_lookups=result.function_lookups,
        attribute_lookups=result.attribute_lookups,
        written_calls=result.written_calls,
        unqualified_relations=result.unqualified_relations,
        resource_ids=result.resource_ids,
        resource_names=result.resource_names,
        allowed_resource_ids=allowed,
    )


def _value(raw: object) -> object:
    """Equal values compare equal across sources; a source's tagged decimal is exact."""

    value = thaw_json(raw)  # type: ignore[arg-type]
    if isinstance(value, dict):
        text = value.get("value")
        if value.get("type") == "decimal" and isinstance(text, str):
            return Decimal(text)
        return canonical_json(value)  # any other tagged value compares as its text
    return value


async def read_keys(
    endpoint: KeyEndpoint,
    *,
    agent_id: str,
    read: CheckedRead | None,
    sql_backend: SqlReadBackend | None,
    tabular_backend: TabularReadBackend | None,
    max_keys: int,
    max_bytes: int,
    timeout_seconds: float,
) -> KeyRead:
    """Read ``endpoint``'s distinct non-null keys, at most ``max_keys`` of them.

    ``read`` is the validated SQL (``checked_key_read``) for a source with a SQL
    dialect, or None to read by column name through a tabular backend.
    ``truncated`` says the bound was hit.
    """

    read_at = datetime.now(UTC)
    if read is not None and sql_backend is not None:
        result = await _execute_read(
            sql_backend,
            lambda source_id, names: None,
            agent_id=agent_id,
            source_id=endpoint.source_id,
            read=read,
            max_rows=max_keys,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
        )
        projection, columns = result.projection, result.columns
    elif tabular_backend is not None:
        projection = await tabular_backend.read_tabular(
            ReadTabularRequest(
                agent_id=agent_id,
                source_id=endpoint.source_id,
                resource_id=catalog_resource_id(
                    endpoint.source_id, ResourceKind.TABLE, endpoint.table
                ),
                columns=endpoint.columns,
                filters=(),
                limit=max_keys,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
            )
        )
        columns = endpoint.columns
    else:
        raise KeyReadError("key_read_unsupported", "This source cannot be key-read.")
    keys = frozenset(
        key
        for row in projection.rows
        for key in (tuple(_value(cast(Mapping[str, object], row)[c]) for c in columns),)
        if None not in key
    )
    return KeyRead(keys=keys, truncated=projection.truncated, read_at=read_at)


def _types(keys: frozenset[tuple[object, ...]]) -> tuple[str, ...]:
    return tuple(sorted({type(part).__name__ for key in keys for part in key}))


def compare_keys(left: KeyRead, right: KeyRead) -> KeyComparison:
    matched_left = sum(1 for key in left.keys if key in right.keys)
    matched_right = sum(1 for key in right.keys if key in left.keys)
    return KeyComparison(
        left_keys=len(left.keys),
        right_keys=len(right.keys),
        matched_left=matched_left,
        matched_right=matched_right,
        left_key_types=_types(left.keys),
        right_key_types=_types(right.keys),
        match_rate=matched_left / len(left.keys) if left.keys else None,
    )
