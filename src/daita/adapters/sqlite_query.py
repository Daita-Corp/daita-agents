"""Guarded SQLite read backend invoked only by the runtime query executor."""

from __future__ import annotations

import asyncio
import base64
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..artifacts.models import ArtifactError
from ..artifacts.renderers import (
    ExactCsvRenderer,
    ExactXlsxProvenance,
    ExactXlsxRenderer,
)
from ..domains.data.capabilities import SQLiteReadResult
from ..domains.data.controller import ReadScopedCatalogReader
from ..domains.data.export_capabilities import (
    ExactTabularExportResult,
    ExactTabularProgress,
    resolved_exact_export_sensitivity,
)
from ..domains.data.results import project_result_rows
from ..domains.data.sql import validate_sqlite_read
from ..errors import PluginError
from ..storage.sqlite_records import SourcePermissionStateError
from .protocols import SourceStore


class SQLiteQueryError(RuntimeError):
    """Normalized query-boundary failure without connector error leakage."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _SQLiteReadPermissionError(PluginError):
    """Stable model-visible failure for a permission recheck at source I/O."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message, plugin_id="sqlite", error_code=code)


@dataclass(frozen=True, slots=True)
class _AdmittedSQLiteFile:
    path: Path
    device: int
    inode: int


class SQLiteQueryBackend:
    """Revalidate catalog scope, then perform one bounded read-only query."""

    def __init__(self, sources: SourceStore, catalog: ReadScopedCatalogReader) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        for method_name in ("resource_schemas", "readable_resource_ids"):
            if not callable(getattr(catalog, method_name, None)):
                raise TypeError(f"catalog must provide {method_name}")
        self._sources = sources
        self._catalog = catalog

    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_bytes: int,
    ) -> SQLiteReadResult:
        registration = await self._sources.load_source(agent_id, source_id)
        if registration is None or not registration.active:
            raise SQLiteQueryError(
                "source_not_available",
                "The SQLite source is not attached to this agent.",
            )
        if registration.adapter_id != "sqlite":
            raise SQLiteQueryError(
                "source_adapter_mismatch",
                "The selected source is not a SQLite source.",
            )
        resources = await self._catalog.resource_schemas(agent_id, source_id)
        try:
            readable = await self._catalog.readable_resource_ids(
                agent_id,
                (source_id,),
            )
        except SourcePermissionStateError:
            raise _SQLiteReadPermissionError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from None
        validation = validate_sqlite_read(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if not validation.valid or validation.analysis is None:
            if "resource_out_of_scope" in validation.issue_codes:
                raise _SQLiteReadPermissionError(
                    "resource_read_not_allowed",
                    "One or more requested resources are not available for reading.",
                )
            codes = ",".join(validation.issue_codes[:8]) or "invalid_sql"
            raise SQLiteQueryError(
                "query_revalidation_failed",
                f"SQLite query failed deterministic revalidation: {codes}",
            )
        if not validation.resource_ids:
            raise SQLiteQueryError(
                "query_resource_scope_empty",
                "SQLite query must reference a current catalog resource.",
            )
        if validation.source_revision is None or len(
            validation.resource_revisions
        ) != len(validation.resource_ids):
            raise SQLiteQueryError(
                "catalog_provenance_missing",
                "SQLite query requires complete current catalog provenance.",
            )
        expected_schema_version = _expected_schema_version(validation.source_revision)
        configured_path = registration.configuration.get("path")
        if not isinstance(configured_path, str):
            raise SQLiteQueryError(
                "source_configuration_invalid",
                "SQLite source configuration is missing its path.",
            )
        path = _regular_unaliased_path(configured_path)
        columns, rows, live_source_revision = await _run_query(
            path,
            validation.analysis.canonical_sql,
            parameters,
            expected_schema_version=expected_schema_version,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )
        projection = project_result_rows(
            rows,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )
        return SQLiteReadResult(
            source_id=source_id,
            canonical_sql=validation.analysis.canonical_sql,
            sql_fingerprint=validation.analysis.sql_fingerprint,
            resource_ids=validation.resource_ids,
            resource_revisions=validation.resource_revisions,
            source_revision=live_source_revision,
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
        """Run one fresh exact read through the selected fixed renderer."""

        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
        ):
            raise SQLiteQueryError(
                "source_not_available",
                "The SQLite source is not attached to this agent.",
            )
        if registration.adapter_id != "sqlite":
            raise SQLiteQueryError(
                "source_adapter_mismatch",
                "The selected source is not a SQLite source.",
            )
        resources = await self._catalog.resource_schemas(agent_id, source_id)
        try:
            readable = await self._catalog.readable_resource_ids(
                agent_id,
                (source_id,),
            )
        except SourcePermissionStateError:
            raise _SQLiteReadPermissionError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from None
        validation = validate_sqlite_read(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if not validation.valid or validation.analysis is None:
            if "resource_out_of_scope" in validation.issue_codes:
                raise _SQLiteReadPermissionError(
                    "resource_read_not_allowed",
                    "One or more requested resources are not available for reading.",
                )
            codes = ",".join(validation.issue_codes[:8]) or "invalid_sql"
            raise SQLiteQueryError(
                "query_revalidation_failed",
                f"SQLite query failed deterministic revalidation: {codes}",
            )
        if not validation.resource_ids:
            raise SQLiteQueryError(
                "query_resource_scope_empty",
                "SQLite query must reference a current catalog resource.",
            )
        if validation.source_revision is None or len(
            validation.resource_revisions
        ) != len(validation.resource_ids):
            raise SQLiteQueryError(
                "catalog_provenance_missing",
                "SQLite query requires complete current catalog provenance.",
            )
        configured_path = registration.configuration.get("path")
        if not isinstance(configured_path, str):
            raise SQLiteQueryError(
                "source_configuration_invalid",
                "SQLite source configuration is missing its path.",
            )
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
        content, columns, row_count, live_source_revision = await _run_exact_tabular(
            _regular_unaliased_path(configured_path),
            validation.analysis.canonical_sql,
            parameters,
            format_name=format_name,
            xlsx_provenance=xlsx_provenance,
            expected_schema_version=_expected_schema_version(
                validation.source_revision
            ),
            max_rows=max_rows,
            max_columns=max_columns,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            progress=progress,
        )
        return ExactTabularExportResult(
            format=format_name,
            source_id=source_id,
            source_revision=live_source_revision,
            sql_fingerprint=validation.analysis.sql_fingerprint,
            resource_revisions=validation.resource_revisions,
            columns=columns,
            row_count=row_count,
            content=content,
            sensitivity=sensitivity,
        )


def _expected_schema_version(source_revision: str) -> int:
    prefix = "schema_version:"
    if not source_revision.startswith(prefix):
        raise SQLiteQueryError(
            "catalog_source_revision_invalid",
            "SQLite catalog source revision is malformed.",
        )
    raw_version = source_revision.removeprefix(prefix)
    if not raw_version.isascii() or not raw_version.isdecimal():
        raise SQLiteQueryError(
            "catalog_source_revision_invalid",
            "SQLite catalog source revision is malformed.",
        )
    version = int(raw_version)
    if version < 0 or str(version) != raw_version:
        raise SQLiteQueryError(
            "catalog_source_revision_invalid",
            "SQLite catalog source revision is malformed.",
        )
    return version


def _regular_unaliased_path(value: str) -> _AdmittedSQLiteFile:
    lexical = Path(os.path.abspath(value))
    try:
        resolved = lexical.resolve(strict=True)
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise SQLiteQueryError(
            "source_path_invalid",
            "SQLite source path is unavailable or unsafe.",
        ) from error
    try:
        identity = os.fstat(descriptor)
        if lexical != resolved or not stat.S_ISREG(identity.st_mode):
            raise SQLiteQueryError(
                "source_path_invalid",
                "SQLite source path is unavailable or unsafe.",
            )
    finally:
        os.close(descriptor)
    return _AdmittedSQLiteFile(resolved, identity.st_dev, identity.st_ino)


async def _run_query(
    path: _AdmittedSQLiteFile,
    sql: str,
    parameters: tuple[object, ...],
    *,
    expected_schema_version: int,
    max_rows: int,
    max_bytes: int,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...], str]:
    cancellation = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _run_query_sync,
            path,
            sql,
            parameters,
            expected_schema_version,
            max_rows,
            max_bytes,
            cancellation,
        )
    )
    try:
        return await asyncio.shield(worker)
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


async def _run_exact_tabular(
    path: _AdmittedSQLiteFile,
    sql: str,
    parameters: tuple[object, ...],
    *,
    format_name: str,
    xlsx_provenance: ExactXlsxProvenance | None,
    expected_schema_version: int,
    max_rows: int,
    max_columns: int,
    max_bytes: int,
    timeout_seconds: float,
    progress: ExactTabularProgress | None = None,
) -> tuple[bytes, tuple[str, ...], int, str]:
    cancellation = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _run_exact_tabular_sync,
            path,
            sql,
            parameters,
            format_name,
            xlsx_provenance,
            expected_schema_version,
            max_rows,
            max_columns,
            max_bytes,
            timeout_seconds,
            cancellation,
            progress,
        )
    )
    try:
        return await asyncio.shield(worker)
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


def _run_exact_tabular_sync(
    path: _AdmittedSQLiteFile,
    sql: str,
    parameters: tuple[object, ...],
    format_name: str,
    xlsx_provenance: ExactXlsxProvenance | None,
    expected_schema_version: int,
    max_rows: int,
    max_columns: int,
    max_bytes: int,
    timeout_seconds: float,
    cancellation: threading.Event,
    progress: ExactTabularProgress | None = None,
) -> tuple[bytes, tuple[str, ...], int, str]:
    connection: sqlite3.Connection | None = None
    admitted_descriptor: int | None = None
    timed_out = threading.Event()
    deadline = time.monotonic() + timeout_seconds
    renderer: ExactCsvRenderer | ExactXlsxRenderer | None = None

    def interrupted() -> int:
        if cancellation.is_set():
            return 1
        if time.monotonic() >= deadline:
            timed_out.set()
            return 1
        return 0

    try:
        connection, admitted_descriptor = _connect_admitted_read_only(path)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.enable_load_extension(False)
        connection.set_progress_handler(interrupted, 500)
        if hasattr(connection, "setlimit"):
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max(4_096, max_bytes))
        connection.execute("BEGIN")
        version_row = connection.execute("PRAGMA schema_version").fetchone()
        if version_row is None or not isinstance(version_row[0], int):
            raise SQLiteQueryError(
                "source_revision_unavailable",
                "SQLite source revision could not be read.",
            )
        live_schema_version = version_row[0]
        if live_schema_version != expected_schema_version:
            raise SQLiteQueryError(
                "catalog_source_stale",
                "SQLite source schema changed after its catalog snapshot.",
            )
        connection.set_authorizer(_read_only_authorizer)
        cursor = connection.execute(sql, parameters)
        if cursor.description is None:
            raise SQLiteQueryError(
                "query_result_missing",
                "SQLite read did not return tabular results.",
            )
        columns = tuple(item[0] for item in cursor.description)
        if format_name == "csv":
            renderer = ExactCsvRenderer(
                columns,
                max_rows=max_rows,
                max_columns=max_columns,
                max_bytes=max_bytes,
                max_seconds=timeout_seconds,
            )
        elif format_name == "xlsx" and xlsx_provenance is not None:
            renderer = ExactXlsxRenderer(
                columns,
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
        while True:
            batch = cursor.fetchmany(256)
            if not batch:
                break
            for row in batch:
                renderer.append(row)
                if progress is not None:
                    progress(
                        renderer.row_count,
                        len(renderer.columns),
                        renderer.byte_count,
                    )
        content = renderer.finish()
        connection.set_authorizer(None)
        connection.execute("COMMIT")
        return (
            content,
            renderer.columns,
            renderer.row_count,
            f"schema_version:{live_schema_version}",
        )
    except ArtifactError:
        raise
    except SQLiteQueryError:
        raise
    except sqlite3.Error as error:
        if timed_out.is_set():
            completed_rows = 0 if renderer is None else renderer.row_count
            completed_columns = 0 if renderer is None else len(renderer.columns)
            completed_bytes = 0 if renderer is None else renderer.byte_count
            raise ArtifactError(
                "artifact_incomplete_export",
                "The exact tabular export exceeded its execution-time limit.",
                {
                    "reason": "time_limit",
                    "completed_rows": completed_rows,
                    "completed_columns": completed_columns,
                    "completed_bytes": completed_bytes,
                },
            ) from error
        raise SQLiteQueryError(
            "sqlite_query_failed",
            "SQLite could not complete the exact bounded read.",
        ) from error
    finally:
        if connection is not None:
            connection.set_authorizer(None)
            connection.set_progress_handler(None, 0)
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()
        if admitted_descriptor is not None:
            os.close(admitted_descriptor)


def _run_query_sync(
    path: _AdmittedSQLiteFile,
    sql: str,
    parameters: tuple[object, ...],
    expected_schema_version: int,
    max_rows: int,
    max_bytes: int,
    cancellation: threading.Event,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...], str]:
    connection: sqlite3.Connection | None = None
    admitted_descriptor: int | None = None
    try:
        connection, admitted_descriptor = _connect_admitted_read_only(path)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.enable_load_extension(False)
        connection.set_progress_handler(lambda: 1 if cancellation.is_set() else 0, 500)
        if hasattr(connection, "setlimit"):
            connection.setlimit(
                sqlite3.SQLITE_LIMIT_LENGTH,
                max(4_096, max_bytes),
            )
        connection.execute("BEGIN")
        version_row = connection.execute("PRAGMA schema_version").fetchone()
        if version_row is None or not isinstance(version_row[0], int):
            raise SQLiteQueryError(
                "source_revision_unavailable",
                "SQLite source revision could not be read.",
            )
        live_schema_version = version_row[0]
        if live_schema_version != expected_schema_version:
            raise SQLiteQueryError(
                "catalog_source_stale",
                "SQLite source schema changed after its catalog snapshot.",
            )
        connection.set_authorizer(_read_only_authorizer)
        columns, rows = _execute_user_query(
            connection,
            sql,
            parameters,
            max_rows,
        )
        connection.set_authorizer(None)
        connection.execute("COMMIT")
        return columns, rows, f"schema_version:{live_schema_version}"
    except SQLiteQueryError:
        raise
    except sqlite3.Error as error:
        raise SQLiteQueryError(
            "sqlite_query_failed",
            "SQLite could not complete the bounded read.",
        ) from error
    finally:
        if connection is not None:
            connection.set_authorizer(None)
            connection.set_progress_handler(None, 0)
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()
        if admitted_descriptor is not None:
            os.close(admitted_descriptor)


def _connect_admitted_read_only(
    admitted: _AdmittedSQLiteFile,
) -> tuple[sqlite3.Connection, int]:
    descriptor: int | None = None
    connection: sqlite3.Connection | None = None
    try:
        try:
            current_mode = os.lstat(admitted.path).st_mode
        except OSError as error:
            raise SQLiteQueryError(
                "source_path_changed",
                "SQLite source path changed after admission.",
            ) from error
        if stat.S_ISLNK(current_mode):
            raise SQLiteQueryError(
                "source_path_changed",
                "SQLite source path changed after admission.",
            )
        descriptor = os.open(
            admitted.path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_dev != admitted.device
            or identity.st_ino != admitted.inode
        ):
            raise SQLiteQueryError(
                "source_path_changed",
                "SQLite source path changed after admission.",
            )
        _reject_wal_format(descriptor)
        descriptor_path = next(
            (
                candidate
                for directory in ("/dev/fd", "/proc/self/fd")
                if (candidate := Path(directory) / str(descriptor)).exists()
            ),
            None,
        )
        if descriptor_path is None:
            raise SQLiteQueryError(
                "source_descriptor_unavailable",
                "SQLite source identity cannot be bound safely on this platform.",
            )
        try:
            connection = sqlite3.connect(
                f"{descriptor_path.as_uri()}?mode=ro",
                uri=True,
                timeout=5.0,
            )
        except sqlite3.Error:
            # Some platforms fail while SQLite opens the admitted descriptor if
            # its original path was replaced. Recheck the path identity before
            # normalizing that connector error so substitution remains explicit.
            _require_admitted_identity(admitted)
            raise
        _require_admitted_identity(admitted)
        return connection, descriptor
    except BaseException:
        if connection is not None:
            connection.close()
        if descriptor is not None:
            os.close(descriptor)
        raise


def _reject_wal_format(descriptor: int) -> None:
    try:
        header = os.read(descriptor, 20)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise SQLiteQueryError(
            "source_path_invalid",
            "SQLite source header could not be read safely.",
        ) from error
    if (
        len(header) == 20
        and header.startswith(b"SQLite format 3\x00")
        and 2 in header[18:20]
    ):
        raise SQLiteQueryError(
            "source_wal_not_supported",
            "SQLite WAL-mode sources are not supported by the inode-bound read "
            "boundary.",
        )


def _require_admitted_identity(admitted: _AdmittedSQLiteFile) -> None:
    descriptor: int | None = None
    try:
        if stat.S_ISLNK(os.lstat(admitted.path).st_mode):
            raise SQLiteQueryError(
                "source_path_changed",
                "SQLite source path changed after admission.",
            )
        descriptor = os.open(
            admitted.path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        identity = os.fstat(descriptor)
    except OSError as error:
        raise SQLiteQueryError(
            "source_path_changed",
            "SQLite source path changed after admission.",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not stat.S_ISREG(identity.st_mode)
        or identity.st_dev != admitted.device
        or identity.st_ino != admitted.inode
    ):
        raise SQLiteQueryError(
            "source_path_changed",
            "SQLite source path changed after admission.",
        )


def _execute_user_query(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...],
    max_rows: int,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
    cursor = connection.execute(sql, parameters)
    if cursor.description is None:
        raise SQLiteQueryError(
            "query_result_missing",
            "SQLite read did not return tabular results.",
        )
    columns = _unique_columns(tuple(item[0] for item in cursor.description))
    raw_rows = cursor.fetchmany(max_rows + 1)
    rows = tuple(
        {column: _json_value(value) for column, value in zip(columns, raw, strict=True)}
        for raw in raw_rows
    )
    return columns, rows


def _read_only_authorizer(
    action: int,
    _arg1: str | None,
    _arg2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    denied_names = (
        "SQLITE_ALTER_TABLE",
        "SQLITE_ANALYZE",
        "SQLITE_ATTACH",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DELETE",
        "SQLITE_DETACH",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_INSERT",
        "SQLITE_PRAGMA",
        "SQLITE_REINDEX",
        "SQLITE_TRANSACTION",
        "SQLITE_UPDATE",
    )
    denied = {
        value
        for name in denied_names
        if isinstance((value := getattr(sqlite3, name, None)), int)
    }
    return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK


def _unique_columns(values: tuple[str, ...]) -> tuple[str, ...]:
    bases = tuple(
        value.strip() if isinstance(value, str) and value.strip() else f"column_{index}"
        for index, value in enumerate(values)
    )
    reserved = {base.casefold() for base in bases}
    result: list[str] = []
    used: set[str] = set()
    for base in bases:
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
            while (
                candidate.casefold() in reserved
                and candidate.casefold() != base.casefold()
            ):
                candidate = f"{base}__{suffix}"
                suffix += 1
        used.add(candidate.casefold())
        result.append(candidate)
    return tuple(result)


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise SQLiteQueryError(
        "query_value_unsupported",
        "SQLite returned a value outside the bounded JSON projection.",
    )


__all__ = ["SQLiteQueryBackend", "SQLiteQueryError"]
