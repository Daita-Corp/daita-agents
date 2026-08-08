"""Guarded PostgreSQL read backend invoked only by the runtime executor."""

from __future__ import annotations

import asyncio
import base64
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
from typing import Any, cast
from uuid import UUID

from .._json import canonical_json
from ..artifacts.models import ArtifactError
from ..artifacts.renderers import (
    ExactCsvRenderer,
    ExactXlsxProvenance,
    ExactXlsxRenderer,
)
from ..domains.data.capabilities import PostgreSQLReadResult
from ..domains.data.controller import CatalogSchemaReader
from ..domains.data.export_capabilities import (
    ExactTabularExportResult,
    ExactTabularProgress,
    resolved_exact_export_sensitivity,
)
from ..domains.data.results import project_result_rows
from ..domains.data.sql import validate_postgresql_read
from ..errors import PluginError
from ..security import SecretProvider, default_secret_provider
from .postgresql import (
    _DEFAULT_MAX_COLUMNS,
    _DEFAULT_MAX_INDEXES,
    _DEFAULT_MAX_RELATIONSHIPS,
    _DEFAULT_MAX_RESOURCES,
    PostgreSQLSourceError,
    _close_postgresql_connection,
    _connect,
    _load_structure,
    _rollback_postgresql_transaction,
)
from .protocols import SourceStore

_MAX_QUERY_ROWS = 1_000
_MAX_QUERY_BYTES = 16 * 1_024 * 1_024
_MAX_RESULT_COLUMNS = 512
_MAX_VALUE_DEPTH = 32
_BOUNDED_RESULT_MARKER = "/* daita:postgresql.bounded_result */"


class PostgreSQLQueryError(PluginError):
    """Normalized query-boundary failure without connector error leakage."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("query error code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("query error message must be a non-empty string")
        self.code = code
        super().__init__(message, plugin_id="postgresql", error_code=code)


class PostgreSQLQueryBackend:
    """Revalidate catalog scope and structure, then run one bounded read."""

    def __init__(
        self,
        sources: SourceStore,
        catalog: CatalogSchemaReader,
        secret_provider: SecretProvider | None = None,
        *,
        statement_timeout_seconds: float = 5.0,
        cleanup_timeout_seconds: float = 1.0,
    ) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        if not callable(getattr(catalog, "resource_schemas", None)):
            raise TypeError("catalog must provide resource_schemas")
        provider = default_secret_provider(secret_provider)
        if not isinstance(provider, SecretProvider):
            raise TypeError("secret_provider must implement SecretProvider")
        if (
            not isinstance(statement_timeout_seconds, (int, float))
            or isinstance(statement_timeout_seconds, bool)
            or not 0 < float(statement_timeout_seconds) <= 60
        ):
            raise ValueError("statement_timeout_seconds must be from 0 through 60")
        if (
            not isinstance(cleanup_timeout_seconds, (int, float))
            or isinstance(cleanup_timeout_seconds, bool)
            or not 0 < float(cleanup_timeout_seconds) <= 10
        ):
            raise ValueError("cleanup_timeout_seconds must be from 0 through 10")
        self._sources = sources
        self._catalog = catalog
        self._secret_provider = provider
        self._statement_timeout_seconds = float(statement_timeout_seconds)
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)

    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_bytes: int,
    ) -> PostgreSQLReadResult:
        if (
            not isinstance(max_rows, int)
            or isinstance(max_rows, bool)
            or not 1 <= max_rows <= _MAX_QUERY_ROWS
        ):
            raise ValueError("max_rows must be from one through 1000")
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or not 2 <= max_bytes <= _MAX_QUERY_BYTES
        ):
            raise ValueError("max_bytes must be from two through 16777216")
        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
        ):
            raise PostgreSQLQueryError(
                "source_not_available",
                "The PostgreSQL source is not attached to this agent.",
            )
        if registration.adapter_id != "postgresql":
            raise PostgreSQLQueryError(
                "source_adapter_mismatch",
                "The selected source is not a PostgreSQL source.",
            )
        resources = await self._catalog.resource_schemas(agent_id, source_id)
        validation = validate_postgresql_read(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
        )
        if not validation.valid or validation.analysis is None:
            codes = ",".join(validation.issue_codes[:8]) or "invalid_sql"
            raise PostgreSQLQueryError(
                "query_revalidation_failed",
                f"PostgreSQL query failed deterministic revalidation: {codes}",
            )
        if not validation.resource_ids:
            raise PostgreSQLQueryError(
                "query_resource_scope_empty",
                "PostgreSQL query must reference a current catalog resource.",
            )
        if validation.source_revision is None or len(
            validation.resource_revisions
        ) != len(validation.resource_ids):
            raise PostgreSQLQueryError(
                "catalog_provenance_missing",
                "PostgreSQL query requires complete current catalog provenance.",
            )

        connection = None
        transaction = None
        transaction_finished = False
        connection_failure: tuple[str, str] | None = None
        query_failed = False
        columns: tuple[str, ...] = ()
        rows: tuple[Mapping[str, object], ...] = ()
        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read",
                readonly=True,
            )
            await transaction.start()
            timeout_milliseconds = max(
                1,
                int(self._statement_timeout_seconds * 1_000),
            )
            await connection.execute(
                "SELECT set_config('statement_timeout', $1, true)",
                f"{timeout_milliseconds}ms",
            )
            await connection.execute(
                "SELECT set_config('search_path', $1, true)",
                "pg_catalog",
            )
            structure = await _load_structure(
                connection,
                registration,
                max_resources=_DEFAULT_MAX_RESOURCES,
                max_columns=_DEFAULT_MAX_COLUMNS,
                max_indexes=_DEFAULT_MAX_INDEXES,
                max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
            )
            if structure.source_revision != validation.source_revision:
                raise PostgreSQLQueryError(
                    "catalog_source_stale",
                    "PostgreSQL source structure changed after its catalog snapshot.",
                )
            columns, rows, value_limited = await _execute_user_query(
                connection,
                validation.analysis.canonical_sql,
                parameters,
                max_rows=max_rows,
                max_bytes=max_bytes,
                timeout_seconds=self._statement_timeout_seconds,
            )
            await transaction.commit()
            transaction_finished = True
        except asyncio.CancelledError:
            raise
        except ImportError:
            # Optional-package admission must retain its actionable install hint.
            raise
        except PostgreSQLQueryError:
            raise
        except PostgreSQLSourceError as error:
            # The connection boundary has already removed connector diagnostics.
            # Retain only its safe code and message, then raise a fresh query error
            # after the original exception context has ended.
            connection_failure = (error.code, str(error))
        except Exception:
            # Driver and server errors can carry SQL, bound values, DSNs, or
            # credentials.  Cross the boundary only after the original
            # exception context has ended so diagnostics retain none of it.
            query_failed = True
        finally:
            try:
                if transaction is not None and not transaction_finished:
                    await _rollback_postgresql_transaction(
                        transaction,
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
            finally:
                if connection is not None:
                    await _close_postgresql_connection(
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )

        if connection_failure is not None:
            raise PostgreSQLQueryError(*connection_failure)
        if query_failed:
            raise PostgreSQLQueryError(
                "postgresql_query_failed",
                "PostgreSQL could not complete the bounded read.",
            )

        projection = project_result_rows(
            rows,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )
        if value_limited:
            reasons = tuple(
                dict.fromkeys((*projection.truncation_reasons, "byte_limit"))
            )
            projection = replace(
                projection,
                total_rows=projection.total_rows + 1,
                truncated=True,
                truncation_reasons=reasons,
            )
        return PostgreSQLReadResult(
            source_id=source_id,
            canonical_sql=validation.analysis.canonical_sql,
            sql_fingerprint=validation.analysis.sql_fingerprint,
            resource_ids=validation.resource_ids,
            resource_revisions=validation.resource_revisions,
            source_revision=validation.source_revision,
            columns=columns,
            projection=projection,
        )

    async def execute_exact_tabular(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        format_name: str,
        parameters_sha256: str,
        created_at: datetime,
        max_rows: int,
        max_columns: int,
        max_bytes: int,
        timeout_seconds: float,
        progress: ExactTabularProgress | None = None,
    ) -> ExactTabularExportResult:
        """Run one fresh read through the selected fixed exact renderer."""

        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
        ):
            raise PostgreSQLQueryError(
                "source_not_available",
                "The PostgreSQL source is not attached to this agent.",
            )
        if registration.adapter_id != "postgresql":
            raise PostgreSQLQueryError(
                "source_adapter_mismatch",
                "The selected source is not a PostgreSQL source.",
            )
        resources = await self._catalog.resource_schemas(agent_id, source_id)
        validation = validate_postgresql_read(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
        )
        if not validation.valid or validation.analysis is None:
            codes = ",".join(validation.issue_codes[:8]) or "invalid_sql"
            raise PostgreSQLQueryError(
                "query_revalidation_failed",
                f"PostgreSQL query failed deterministic revalidation: {codes}",
            )
        if not validation.resource_ids:
            raise PostgreSQLQueryError(
                "query_resource_scope_empty",
                "PostgreSQL query must reference a current catalog resource.",
            )
        if validation.source_revision is None or len(
            validation.resource_revisions
        ) != len(validation.resource_ids):
            raise PostgreSQLQueryError(
                "catalog_provenance_missing",
                "PostgreSQL query requires complete current catalog provenance.",
            )

        connection = None
        transaction = None
        transaction_finished = False
        connection_failure: tuple[str, str] | None = None
        query_failed = False
        content = b""
        columns: tuple[str, ...] = ()
        row_count = 0
        completed = {"rows": 0, "columns": 0, "bytes": 0}
        sensitivity = resolved_exact_export_sensitivity(
            tuple(
                item.sensitivity_class
                for item in resources
                if item.resource_id in validation.resource_ids
            )
        )
        xlsx_provenance = (
            ExactXlsxProvenance(
                source_id=source_id,
                source_revision=validation.source_revision,
                resource_revisions=validation.resource_revisions,
                sql_fingerprint=validation.analysis.sql_fingerprint,
                parameters_sha256=parameters_sha256,
                sensitivity=sensitivity,
                created_at=created_at,
            )
            if format_name == "xlsx"
            else None
        )

        def record_progress(rows: int, column_count: int, byte_count: int) -> None:
            completed.update(
                rows=rows,
                columns=column_count,
                bytes=byte_count,
            )
            if progress is not None:
                progress(rows, column_count, byte_count)

        try:
            connection = await _connect(registration, self._secret_provider)
            transaction = connection.transaction(
                isolation="repeatable_read",
                readonly=True,
            )
            await transaction.start()
            timeout_milliseconds = max(1, int(timeout_seconds * 1_000))
            await connection.execute(
                "SELECT set_config('statement_timeout', $1, true)",
                f"{timeout_milliseconds}ms",
            )
            await connection.execute(
                "SELECT set_config('search_path', $1, true)",
                "pg_catalog",
            )
            structure = await _load_structure(
                connection,
                registration,
                max_resources=_DEFAULT_MAX_RESOURCES,
                max_columns=_DEFAULT_MAX_COLUMNS,
                max_indexes=_DEFAULT_MAX_INDEXES,
                max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
            )
            if structure.source_revision != validation.source_revision:
                raise PostgreSQLQueryError(
                    "catalog_source_stale",
                    "PostgreSQL source structure changed after its catalog snapshot.",
                )
            async with asyncio.timeout(timeout_seconds):
                content, columns, row_count = await _execute_exact_tabular_query(
                    connection,
                    validation.analysis.canonical_sql,
                    parameters,
                    format_name=format_name,
                    xlsx_provenance=xlsx_provenance,
                    max_rows=max_rows,
                    max_columns=max_columns,
                    max_bytes=max_bytes,
                    timeout_seconds=timeout_seconds,
                    progress=record_progress,
                )
            await transaction.commit()
            transaction_finished = True
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            raise ArtifactError(
                "artifact_incomplete_export",
                "The exact tabular export exceeded its execution-time limit.",
                {
                    "reason": "time_limit",
                    "completed_rows": completed["rows"],
                    "completed_columns": completed["columns"],
                    "completed_bytes": completed["bytes"],
                },
            ) from error
        except ImportError:
            raise
        except (ArtifactError, PostgreSQLQueryError):
            raise
        except PostgreSQLSourceError as error:
            connection_failure = (error.code, str(error))
        except Exception:
            query_failed = True
        finally:
            try:
                if transaction is not None and not transaction_finished:
                    await _rollback_postgresql_transaction(
                        transaction,
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
            finally:
                if connection is not None:
                    await _close_postgresql_connection(
                        connection,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
        if connection_failure is not None:
            raise PostgreSQLQueryError(*connection_failure)
        if query_failed:
            raise PostgreSQLQueryError(
                "postgresql_query_failed",
                "PostgreSQL could not complete the exact bounded read.",
            )
        return ExactTabularExportResult(
            format=format_name,
            source_id=source_id,
            source_revision=validation.source_revision,
            sql_fingerprint=validation.analysis.sql_fingerprint,
            resource_revisions=validation.resource_revisions,
            columns=columns,
            row_count=row_count,
            content=content,
            sensitivity=sensitivity,
        )


async def _execute_user_query(
    connection: Any,
    sql: str,
    parameters: tuple[object, ...],
    *,
    max_rows: int,
    max_bytes: int,
    timeout_seconds: float,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...], bool]:
    shape_statement = await connection.prepare(sql, timeout=timeout_seconds)
    attributes = tuple(shape_statement.get_attributes())
    if len(attributes) > _MAX_RESULT_COLUMNS:
        raise PostgreSQLQueryError(
            "query_result_too_wide",
            "PostgreSQL result exceeds the bounded column limit.",
        )
    raw_names = tuple(str(attribute.name) for attribute in attributes)
    columns = _unique_columns(raw_names)
    bounded_sql = _bounded_result_sql(
        sql,
        column_count=len(columns),
        max_rows=max_rows,
        max_bytes=max_bytes,
    )
    statement = await connection.prepare(bounded_sql, timeout=timeout_seconds)
    cursor = statement.cursor(
        *parameters,
        prefetch=max_rows + 1,
        timeout=timeout_seconds,
    )
    rows: list[Mapping[str, object]] = []
    projected_bytes = 2
    value_limited = False
    async for record in cursor:
        values: tuple[object, ...]
        values_method = getattr(record, "values", None)
        if callable(values_method):
            # asyncpg.Record intentionally implements its mapping API without
            # registering as collections.abc.Mapping.  Consume that public
            # interface without importing the optional SDK type.
            parsed_values: tuple[object, ...] | None = None
            try:
                parsed_values = tuple(cast(Iterable[object], values_method()))
            except TypeError:
                pass
            if parsed_values is None:
                raise PostgreSQLQueryError(
                    "query_result_invalid",
                    "PostgreSQL returned an invalid tabular row.",
                )
            values = parsed_values
        elif isinstance(record, Sequence):
            values = tuple(cast(Sequence[object], record))
        else:
            raise PostgreSQLQueryError(
                "query_result_invalid",
                "PostgreSQL returned an invalid tabular row.",
            )
        if len(values) != len(columns) + 1 or not isinstance(values[-1], bool):
            raise PostgreSQLQueryError(
                "query_result_invalid",
                "PostgreSQL returned an invalid tabular row.",
            )
        if values[-1] is not True:
            value_limited = True
            break
        row = {
            column: _json_value(value)
            for column, value in zip(columns, values[:-1], strict=True)
        }
        rows.append(row)
        encoded_row_bytes = len(canonical_json(row).encode("utf-8"))
        projected_bytes += encoded_row_bytes + (1 if len(rows) > 1 else 0)
        if len(rows) >= max_rows + 1:
            break
        if projected_bytes > max_bytes:
            break
    return columns, tuple(rows), value_limited


async def _execute_exact_tabular_query(
    connection: Any,
    sql: str,
    parameters: tuple[object, ...],
    *,
    format_name: str,
    xlsx_provenance: ExactXlsxProvenance | None,
    max_rows: int,
    max_columns: int,
    max_bytes: int,
    timeout_seconds: float,
    progress: ExactTabularProgress | None = None,
) -> tuple[bytes, tuple[str, ...], int]:
    shape_statement = await connection.prepare(sql, timeout=timeout_seconds)
    attributes = tuple(shape_statement.get_attributes())
    raw_names = tuple(attribute.name for attribute in attributes)
    if format_name == "csv":
        renderer: ExactCsvRenderer | ExactXlsxRenderer = ExactCsvRenderer(
            raw_names,
            max_rows=max_rows,
            max_columns=max_columns,
            max_bytes=max_bytes,
            max_seconds=timeout_seconds,
        )
    elif format_name == "xlsx" and xlsx_provenance is not None:
        renderer = ExactXlsxRenderer(
            raw_names,
            provenance=xlsx_provenance,
            max_rows=max_rows,
            max_columns=max_columns,
            max_bytes=max_bytes,
            max_seconds=timeout_seconds,
        )
    else:
        raise ValueError("exact tabular format and provenance are invalid")
    if progress is not None:
        progress(renderer.row_count, len(renderer.columns), renderer.byte_count)
    bounded_sql = _exact_tabular_bounded_result_sql(
        sql,
        column_names=renderer.columns,
        max_rows=max_rows,
        max_bytes=max_bytes,
    )
    statement = await connection.prepare(bounded_sql, timeout=timeout_seconds)
    cursor = statement.cursor(
        *parameters,
        prefetch=min(1_000, max_rows + 1),
        timeout=timeout_seconds,
    )
    async for record in cursor:
        values_method = getattr(record, "values", None)
        values: tuple[object, ...] | None = None
        if callable(values_method):
            try:
                values = tuple(cast(Iterable[object], values_method()))
            except TypeError:
                pass
        elif isinstance(record, Sequence):
            values = tuple(cast(Sequence[object], record))
        if (
            values is None
            or len(values) != len(renderer.columns) + 1
            or not isinstance(values[-1], bool)
        ):
            raise PostgreSQLQueryError(
                "query_result_invalid",
                "PostgreSQL returned an invalid exact tabular row.",
            )
        if values[-1] is not True:
            renderer.incomplete("byte_limit")
        renderer.append(values[:-1])
        if progress is not None:
            progress(renderer.row_count, len(renderer.columns), renderer.byte_count)
    return renderer.finish(), renderer.columns, renderer.row_count


def _bounded_result_sql(
    sql: str,
    *,
    column_count: int,
    max_rows: int,
    max_bytes: int,
) -> str:
    aliases = tuple(f"__daita_column_{index}" for index in range(column_count))
    row_alias = "__daita_bounded_row"
    size_check = (
        f"pg_catalog.pg_column_size({row_alias}) <= {max_bytes} AND "
        f"pg_catalog.octet_length(pg_catalog.to_jsonb({row_alias})::text) "
        f"<= {max_bytes}"
    )
    selected = [
        (
            f'CASE WHEN {size_check} THEN {row_alias}."{alias}" ELSE NULL END '
            f'AS "{alias}"'
        )
        for alias in aliases
    ]
    selected.append(f'{size_check} AS "__daita_within_result_limit"')
    alias_clause = (
        "" if not aliases else " (" + ", ".join(f'"{item}"' for item in aliases) + ")"
    )
    return (
        f"{_BOUNDED_RESULT_MARKER} SELECT {', '.join(selected)} "
        f"FROM ({sql}) AS {row_alias}{alias_clause} LIMIT {max_rows + 1}"
    )


def _exact_tabular_bounded_result_sql(
    sql: str,
    *,
    column_names: tuple[str, ...],
    max_rows: int,
    max_bytes: int,
) -> str:
    aliases = tuple(
        f"__daita_exact_column_{index}" for index in range(len(column_names))
    )
    row_alias = "__daita_exact_row"
    size_check = f"pg_catalog.pg_column_size({row_alias}) <= {max_bytes}"
    selected = [
        (
            f'CASE WHEN {size_check} THEN {row_alias}."{alias}" ELSE NULL END '
            f"AS {_postgresql_identifier(column_name)}"
        )
        for alias, column_name in zip(aliases, column_names, strict=True)
    ]
    selected.append(f'{size_check} AS "__daita_exact_within_result_limit"')
    alias_clause = " (" + ", ".join(f'"{item}"' for item in aliases) + ")"
    return (
        "/* daita:postgresql.exact_tabular */ "
        f"SELECT {', '.join(selected)} FROM ({sql}) AS {row_alias}{alias_clause} "
        f"LIMIT {max_rows + 1}"
    )


def _postgresql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _unique_columns(values: tuple[str, ...]) -> tuple[str, ...]:
    bases = tuple(
        value.strip() if value.strip() else f"column_{index}"
        for index, value in enumerate(values)
    )
    reserved = set(bases)
    result: list[str] = []
    used: set[str] = set()
    for base in bases:
        # PostgreSQL quoted output names are case-sensitive JSON keys.  Only
        # exact duplicates require disambiguation, but a generated suffix must
        # never collide with another real or generated output name.
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
            while candidate in reserved and candidate != base:
                candidate = f"{base}__{suffix}"
                suffix += 1
        used.add(candidate)
        result.append(candidate)
    return tuple(result)


def _json_value(value: object, *, _depth: int = 0) -> object:
    if _depth > _MAX_VALUE_DEPTH:
        raise PostgreSQLQueryError(
            "query_value_unsupported",
            "PostgreSQL returned a value outside the bounded JSON projection.",
        )
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PostgreSQLQueryError(
                "query_value_unsupported",
                "PostgreSQL returned a value outside the bounded JSON projection.",
            )
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise PostgreSQLQueryError(
                "query_value_unsupported",
                "PostgreSQL returned a value outside the bounded JSON projection.",
            )
        # JSON numbers cannot preserve PostgreSQL NUMERIC precision.  Keep the
        # exact decimal text in an explicit tagged value so accepted evidence
        # never silently changes a factual result.
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {
            "type": "interval",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, (IPv4Address, IPv6Address, IPv4Network, IPv6Network)):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "value": base64.b64encode(value).decode("ascii"),
        }
    value_type = type(value)
    if (
        value_type.__module__ == "asyncpg.pgproto.types"
        and value_type.__name__ == "BitString"
    ):
        as_string = getattr(value, "as_string", None)
        bits = as_string() if callable(as_string) else None
        if (
            not isinstance(bits, str)
            or not bits
            or any(character not in "01" for character in bits)
        ):
            raise PostgreSQLQueryError(
                "query_value_unsupported",
                "PostgreSQL returned a value outside the bounded JSON projection.",
            )
        return {"type": "bit_string", "value": bits}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PostgreSQLQueryError(
                "query_value_unsupported",
                "PostgreSQL returned a value outside the bounded JSON projection.",
            )
        return {
            key: _json_value(item, _depth=_depth + 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, _depth=_depth + 1) for item in value]
    raise PostgreSQLQueryError(
        "query_value_unsupported",
        "PostgreSQL returned a value outside the bounded JSON projection.",
    )


__all__ = ["PostgreSQLQueryBackend", "PostgreSQLQueryError"]
