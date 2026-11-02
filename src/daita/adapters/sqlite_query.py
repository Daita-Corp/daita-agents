"""Execute policy-checked, bounded reads against admitted SQLite files."""

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
    ExactArrowRenderer,
    ExactCsvRenderer,
    ExactXlsxProvenance,
    ExactXlsxRenderer,
    arrow_schema,
)
from ..domains.data.capabilities import SQLiteReadResult, refused_read_details
from ..domains.data.controller import ReadScopedCatalogReader
from ..domains.data.export_capabilities import (
    ExactTabularExportResult,
    ExactTabularProgress,
    resolved_exact_export_sensitivity,
)
from ..domains.data.results import project_result_rows
from ..domains.data.sql import CheckedRead, denied_relations
from ..errors import DaitaError
from ..storage.sqlite_records import SourcePermissionStateError
from .protocols import SourceStore

_DEFAULT_STATEMENT_TIMEOUT_SECONDS = 5.0
_MAX_STATEMENT_TIMEOUT_SECONDS = 60.0


class SQLiteQueryError(DaitaError):
    """Normalized query-boundary failure without connector error leakage."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.code = code
        self.details = dict(details or {})
        super().__init__(message, error_code=code)


@dataclass(frozen=True, slots=True)
class _AdmittedSQLiteFile:
    path: Path
    device: int
    inode: int


class SQLiteQueryBackend:
    """Perform one bounded read-only query that already passed policy."""

    def __init__(self, sources: SourceStore, catalog: ReadScopedCatalogReader) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        if not callable(getattr(catalog, "readable_resource_ids", None)):
            raise TypeError("catalog must provide readable_resource_ids")
        self._sources = sources
        self._catalog = catalog

    async def _configured_path(
        self, agent_id: str, source_id: str, read: CheckedRead
    ) -> str:
        """Recheck current permission, since a read can queue after it was prepared."""

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
        configured_path = registration.configuration.get("path")
        if not isinstance(configured_path, str):
            raise SQLiteQueryError(
                "source_configuration_invalid",
                "SQLite source configuration is missing its path.",
            )
        try:
            readable = await self._catalog.readable_resource_ids(agent_id, (source_id,))
        except SourcePermissionStateError:
            raise SQLiteQueryError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from None
        denied = denied_relations(read, frozenset(readable))
        if denied:
            raise SQLiteQueryError(
                "resource_read_not_allowed",
                f"{', '.join(denied)} is not available for reading in this run.",
            )
        return configured_path

    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        read: CheckedRead,
        max_rows: int,
        max_bytes: int,
        timeout_seconds: float | None = None,
    ) -> SQLiteReadResult:
        statement_timeout = (
            _DEFAULT_STATEMENT_TIMEOUT_SECONDS
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if not 0 < statement_timeout <= _MAX_STATEMENT_TIMEOUT_SECONDS:
            raise ValueError("timeout_seconds must be from 0 through 60")
        path = _regular_unaliased_path(
            await self._configured_path(agent_id, source_id, read)
        )
        columns, rows, live_source_revision = await _run_query(
            path,
            read.canonical_sql,
            read.parameters,
            max_rows=max_rows,
            max_bytes=max_bytes,
            timeout_seconds=statement_timeout,
        )
        projection = project_result_rows(
            rows,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )
        return SQLiteReadResult(
            source_id=source_id,
            canonical_sql=read.canonical_sql,
            sql_fingerprint=read.sql_fingerprint,
            resource_ids=read.resource_ids,
            resource_revisions=read.resource_revisions,
            source_revision=live_source_revision,
            columns=columns,
            projection=projection,
            resource_names=read.resource_names,
        )

    async def execute_exact_tabular(
        self,
        *,
        agent_id: str,
        source_id: str,
        read: CheckedRead,
        format_name: str,
        parameters_sha256: str,
        created_at: datetime,
        max_rows: int,
        max_columns: int,
        max_bytes: int,
        timeout_seconds: float,
        progress: ExactTabularProgress | None = None,
    ) -> ExactTabularExportResult:
        """Run one fresh read through the selected fixed renderer."""

        configured_path = await self._configured_path(agent_id, source_id, read)
        sensitivity = resolved_exact_export_sensitivity(read.sensitivity_classes)
        xlsx_provenance = (
            ExactXlsxProvenance(
                source_id=source_id,
                source_revision=read.source_revision,
                resource_revisions=read.resource_revisions,
                sql_fingerprint=read.sql_fingerprint,
                parameters_sha256=parameters_sha256,
                sensitivity=sensitivity,
                created_at=created_at,
            )
            if format_name == "xlsx"
            else None
        )
        content, columns, row_count, live_source_revision = await _run_exact_tabular(
            _regular_unaliased_path(configured_path),
            read.canonical_sql,
            read.parameters,
            format_name=format_name,
            xlsx_provenance=xlsx_provenance,
            max_rows=max_rows,
            max_columns=max_columns,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            progress=progress,
        )
        return ExactTabularExportResult(
            format=format_name,
            adapter_id="sqlite",
            source_id=source_id,
            source_revision=live_source_revision,
            sql_fingerprint=read.sql_fingerprint,
            resource_revisions=read.resource_revisions,
            columns=columns,
            row_count=row_count,
            content=content,
            sensitivity=sensitivity,
            column_types=arrow_schema(content)[1] if format_name == "arrow" else (),
        )


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
    max_rows: int,
    max_bytes: int,
    timeout_seconds: float = 5.0,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...], str]:
    cancellation = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _run_query_sync,
            path,
            sql,
            parameters,
            max_rows,
            max_bytes,
            cancellation,
            timeout_seconds,
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
    renderer: ExactArrowRenderer | ExactCsvRenderer | ExactXlsxRenderer | None = None

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
        elif format_name == "arrow":
            renderer = ExactArrowRenderer(
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
        raise _classified_failure(error) from None
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
    max_rows: int,
    max_bytes: int,
    cancellation: threading.Event,
    timeout_seconds: float = 5.0,
) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...], str]:
    deadline = time.monotonic() + timeout_seconds
    connection: sqlite3.Connection | None = None
    admitted_descriptor: int | None = None
    try:
        connection, admitted_descriptor = _connect_admitted_read_only(path)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.enable_load_extension(False)
        # SQLite has no statement timeout; the progress handler interrupts the
        # statement at its deadline as well as on cancellation.
        connection.set_progress_handler(
            lambda: int(cancellation.is_set() or time.monotonic() >= deadline),
            500,
        )
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
        if not cancellation.is_set() and time.monotonic() >= deadline:
            raise SQLiteQueryError(
                "sqlite_query_timeout",
                "The bounded SQLite read exceeded its statement deadline of "
                f"{timeout_seconds:g} seconds.",
            ) from None
        raise _classified_failure(error) from None
    finally:
        if connection is not None:
            connection.set_authorizer(None)
            connection.set_progress_handler(None, 0)
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()
        if admitted_descriptor is not None:
            os.close(admitted_descriptor)


def _classified_failure(error: sqlite3.Error) -> SQLiteQueryError:
    """Sort one SQLite failure; only names and syntax text cross the boundary."""

    name = str(getattr(error, "sqlite_errorname", "") or "")
    details: dict[str, object] = {"sqlite_error": name} if name else {}
    text = str(error)[:1_024]
    if name == "SQLITE_AUTH" or "not authorized" in text:
        return SQLiteQueryError("not_permitted", text, details)
    if name.startswith("SQLITE_READONLY") or "readonly" in text:
        return SQLiteQueryError(
            "read_only_violation", "A read cannot modify data.", details
        )
    if name == "SQLITE_ERROR":
        # Unknown tables and columns, ambiguity and syntax: names, never values.
        return SQLiteQueryError("query_invalid", text, refused_read_details(details))
    return SQLiteQueryError(
        "sqlite_query_failed",
        "SQLite could not complete the bounded read" + (f" ({name})." if name else "."),
        details,
    )


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
    arg2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
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


__all__ = [
    "SQLiteQueryBackend",
    "SQLiteQueryError",
]
