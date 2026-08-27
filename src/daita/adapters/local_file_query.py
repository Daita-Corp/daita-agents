"""Run one bounded DuckDB local-file query in one private child process."""

from __future__ import annotations

import asyncio
import base64
import ctypes
import json
import math
import multiprocessing
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .._installation import repair_guidance
from .._json import FrozenJsonObject, canonical_json

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from .local_workspace import LocalFileQueryManifest

_MIB = 1_024 * 1_024
_GIB = 1_024 * _MIB
_PROTOCOL = "daita.local_file_query.v1"
_DUCKDB_VERSION = "1.5.5"
_MAX_REQUEST_BYTES = 512 * 1_024
_MAX_RESPONSE_BYTES = 64 * 1_024
_MAX_COLUMN_NAME_BYTES = 256
_MAX_VALUE_DEPTH = 16
_POLL_SECONDS = 0.01


class LocalFileQueryError(RuntimeError):
    """One normalized query-boundary failure without engine path leakage."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("file-query error code must be non-empty text")
        if not isinstance(message, str) or not message:
            raise ValueError("file-query error message must be non-empty text")
        self.code = code
        self.message = message
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LocalFileQueryLimits:
    """Code-owned worker, engine, spill, and model-projection limits."""

    max_query_seconds: float = 30.0
    max_result_rows: int = 100
    max_result_bytes: int = 48 * 1_024
    max_result_columns: int = 128
    duckdb_memory_bytes: int = 256 * _MIB
    max_spill_bytes: int = 2 * _GIB
    max_worker_rss_bytes: int = 1_536 * _MIB
    duckdb_threads: int = 2

    def __post_init__(self) -> None:
        integer_bounds = (
            (self.max_result_rows, "max_result_rows", 1, 100),
            (self.max_result_bytes, "max_result_bytes", 1_024, 48 * 1_024),
            (self.max_result_columns, "max_result_columns", 1, 512),
            (
                self.duckdb_memory_bytes,
                "duckdb_memory_bytes",
                64 * _MIB,
                512 * _MIB,
            ),
            (self.max_spill_bytes, "max_spill_bytes", _MIB, 2 * _GIB),
            (
                self.max_worker_rss_bytes,
                "max_worker_rss_bytes",
                256 * _MIB,
                2 * _GIB,
            ),
            (self.duckdb_threads, "duckdb_threads", 1, 4),
        )
        for value, name, minimum, maximum in integer_bounds:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} is outside its code-owned bound")
        if (
            not isinstance(self.max_query_seconds, (int, float))
            or isinstance(self.max_query_seconds, bool)
            or not 0.01 <= float(self.max_query_seconds) <= 30.0
        ):
            raise ValueError("max_query_seconds is outside its code-owned bound")


@dataclass(frozen=True, slots=True)
class LocalFileQueryResult:
    data: FrozenJsonObject
    sensitivity_provenance: FrozenJsonObject


class LocalFileQueryBackend:
    """One-agent admission and lifecycle owner for private query workers."""

    def __init__(
        self,
        *,
        scratch_parent: Path,
        limits: LocalFileQueryLimits | None = None,
    ) -> None:
        if not isinstance(scratch_parent, Path) or not scratch_parent.is_absolute():
            raise TypeError("scratch_parent must be one absolute pathlib.Path")
        self._scratch_parent = scratch_parent
        self._limits = limits or LocalFileQueryLimits()
        self._gate = asyncio.Lock()
        self._active_query = False
        self._active: dict[asyncio.Task[dict[str, object]], threading.Event] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def limits(self) -> LocalFileQueryLimits:
        return self._limits

    async def query(
        self,
        *,
        manifest: LocalFileQueryManifest,
        canonical_sql: str,
        sql_fingerprint: str,
    ) -> LocalFileQueryResult:
        if self._closed:
            raise LocalFileQueryError(
                "workspace_unavailable", "The local file-query backend is closed."
            )
        if not isinstance(canonical_sql, str) or not canonical_sql:
            raise TypeError("canonical_sql must be non-empty text")
        if not isinstance(sql_fingerprint, str) or not sql_fingerprint.startswith(
            "sha256:"
        ):
            raise TypeError("sql_fingerprint must use sha256")
        async with self._gate:
            if self._closed:
                raise LocalFileQueryError(
                    "workspace_unavailable",
                    "The local file-query backend is closed.",
                )
            if self._active_query:
                raise LocalFileQueryError(
                    "file_query_limited",
                    "Only one local structured-file query may run at a time.",
                    {"concurrent_query_limit": 1},
                )
            self._active_query = True
        cancellation = threading.Event()
        worker: asyncio.Task[dict[str, object]] = asyncio.create_task(
            asyncio.to_thread(
                _run_private_worker,
                manifest,
                canonical_sql,
                sql_fingerprint,
                self._scratch_parent,
                self._limits,
                cancellation,
            )
        )
        self._active[worker] = cancellation
        try:
            try:
                payload = await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancellation.set()
                await _settle_worker(worker)
                raise
            manifest.revalidate()
            data = {
                **payload,
                "path_pattern": manifest.path_pattern,
                "format": manifest.format,
                "input_file_count": len(manifest.bindings),
                "input_bytes": manifest.input_bytes,
                "manifest_bytes": manifest.encoded_bytes,
                "manifest_sha256": manifest.manifest_sha256,
                "input_bindings": tuple(
                    item.result_mapping() for item in manifest.bindings
                ),
            }
            return LocalFileQueryResult(
                data=FrozenJsonObject.from_mapping(data),
                sensitivity_provenance=FrozenJsonObject.from_mapping(
                    manifest.provenance_mapping()
                ),
            )
        finally:
            self._active.pop(worker, None)
            async with self._gate:
                self._active_query = False

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_once())
        await asyncio.shield(self._close_task)

    async def _close_once(self) -> None:
        for cancellation in tuple(self._active.values()):
            cancellation.set()
        for worker in tuple(self._active):
            await _settle_worker(worker)
        self._active.clear()
        try:
            self._scratch_parent.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # A crash-created non-empty directory is not deleted as part of
            # ordinary close; each live call still removes its exact scratch.
            pass


async def _settle_worker(worker: asyncio.Task[object]) -> None:
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except BaseException:  # noqa: BLE001 - settling must absorb worker failure
            break


def _run_private_worker(
    manifest: LocalFileQueryManifest,
    canonical_sql: str,
    sql_fingerprint: str,
    scratch_parent: Path,
    limits: LocalFileQueryLimits,
    cancellation: threading.Event,
) -> dict[str, object]:
    scratch = _create_private_scratch(scratch_parent)
    context = multiprocessing.get_context("spawn")
    parent: Connection | None = None
    process: Any | None = None
    termination: str | None = None
    peak_spill = 0
    peak_rss = 0
    final: dict[str, object] | None = None
    try:
        manifest.revalidate()
        if not (sys.platform == "darwin" or sys.platform.startswith("linux")):
            raise LocalFileQueryError(
                "dependency_unavailable",
                "This platform cannot monitor private file-query worker RSS.",
            )
        request = _request_mapping(manifest, canonical_sql, sql_fingerprint, limits)
        request_bytes = canonical_json(request).encode("utf-8")
        if len(request_bytes) > _MAX_REQUEST_BYTES:
            raise LocalFileQueryError(
                "file_pattern_too_broad",
                "The complete private file-query request exceeds its byte bound.",
                {"limit": _MAX_REQUEST_BYTES, "observed": len(request_bytes)},
            )
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_private_worker_entry,
            args=(child, os.fspath(scratch)),
            name="daita-local-file-query",
        )
        started = time.monotonic()
        process.start()
        child.close()
        parent.send_bytes(request_bytes)
        from multiprocessing.reduction import send_handle

        assert process.pid is not None
        for binding in manifest.bindings:
            send_handle(parent, binding.descriptor, process.pid)
        while process.is_alive():
            now = time.monotonic()
            peak_spill = max(peak_spill, _directory_bytes(scratch))
            observed_rss = _process_rss_bytes(process.pid)
            if observed_rss is not None:
                peak_rss = max(peak_rss, observed_rss)
            final = _receive_messages(parent, final)
            if cancellation.is_set():
                termination = "cancelled"
            elif now - started >= limits.max_query_seconds:
                termination = "timeout"
            elif peak_spill > limits.max_spill_bytes:
                termination = "spill_limit"
            elif peak_rss > limits.max_worker_rss_bytes:
                termination = "rss_limit"
            if termination is not None:
                process.terminate()
                break
            process.join(timeout=_POLL_SECONDS)
        if process.is_alive():
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        process.join()
        final = _receive_messages(parent, final)
        peak_spill = max(peak_spill, _directory_bytes(scratch))
        if termination == "cancelled":
            raise _WorkerCancelled()
        if termination == "timeout":
            raise LocalFileQueryError(
                "file_query_timeout",
                "The local structured-file query exceeded its bounded wall time.",
                {"limit_seconds": limits.max_query_seconds},
            )
        if termination == "spill_limit":
            raise LocalFileQueryError(
                "file_query_limited",
                "The local structured-file query exceeded its private spill bound.",
                {"spill_limit_bytes": limits.max_spill_bytes},
            )
        if termination == "rss_limit":
            raise LocalFileQueryError(
                "file_query_limited",
                "The private file-query worker exceeded its monitored RSS bound.",
                {"rss_limit_bytes": limits.max_worker_rss_bytes},
            )
        if peak_rss <= 0:
            raise LocalFileQueryError(
                "dependency_unavailable",
                "Private file-query worker RSS monitoring is unavailable.",
            )
        if final is None:
            raise LocalFileQueryError(
                "file_query_invalid",
                "The private file-query worker ended without a bounded result.",
            )
        if final.get("status") != "success":
            code = final.get("code")
            message = final.get("message")
            details = final.get("details")
            raise LocalFileQueryError(
                code if isinstance(code, str) else "file_query_invalid",
                (
                    message
                    if isinstance(message, str)
                    else "The local structured-file query could not be executed safely."
                ),
                dict(details) if isinstance(details, dict) else None,
            )
        payload = final.get("data")
        if not isinstance(payload, dict):
            raise LocalFileQueryError(
                "file_query_invalid",
                "The private file-query worker returned an invalid result.",
            )
        manifest.revalidate()
        return payload
    except _WorkerCancelled as error:
        raise asyncio.CancelledError from error
    except LocalFileQueryError:
        raise
    except (EOFError, OSError, ValueError) as error:
        raise LocalFileQueryError(
            "file_query_invalid",
            "The private file-query worker could not complete safely.",
        ) from error
    finally:
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            try:
                process.close()
            except (OSError, ValueError):
                pass
        if parent is not None:
            parent.close()
        _remove_private_scratch(scratch)


class _WorkerCancelled(RuntimeError):
    pass


class _WorkerFailure(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


def _private_worker_entry(
    connection: Connection,
    scratch_text: str,
) -> None:
    descriptors: list[int] = []
    duckdb_connection: Any | None = None
    final: dict[str, object]
    try:
        _set_worker_limits()
        request = _validate_request(connection.recv_bytes(maxlength=_MAX_REQUEST_BYTES))
        _revalidate_canonical_sql(request)
        from multiprocessing.reduction import recv_handle

        for _binding in request["bindings"]:
            descriptors.append(int(recv_handle(connection)))
        descriptor_paths = tuple(_descriptor_path(item) for item in descriptors)
        _revalidate_descriptors(descriptors, request["bindings"])
        duckdb = _load_duckdb()
        duckdb_connection = duckdb.connect(":memory:")
        _apply_security_profile(
            duckdb_connection,
            descriptor_paths,
            Path(scratch_text),
            request,
        )
        schema = _bind_data(
            duckdb_connection,
            format_name=str(request["format"]),
            descriptor_paths=descriptor_paths,
        )
        _send_message(
            connection,
            {
                "kind": "ready",
                "duckdb_version": duckdb.__version__,
                "schema": schema,
            },
        )
        data = _execute_bounded_query(
            duckdb_connection,
            str(request["canonical_sql"]),
            str(request["sql_fingerprint"]),
            max_rows=int(request["max_result_rows"]),
            max_bytes=int(request["max_result_bytes"]),
            max_columns=int(request["max_result_columns"]),
        )
        _revalidate_descriptors(descriptors, request["bindings"])
        final = {"kind": "final", "status": "success", "data": data}
    except ImportError as error:
        final = {
            "kind": "final",
            "status": "error",
            "code": "dependency_unavailable",
            "message": str(error),
            "details": {},
        }
    except _WorkerFailure as error:
        final = {
            "kind": "final",
            "status": "error",
            "code": error.code,
            "message": error.message,
            "details": error.details,
        }
    except BaseException:  # noqa: BLE001 - child boundary must return sanitized errors
        final = {
            "kind": "final",
            "status": "error",
            "code": "file_query_invalid",
            "message": "The local structured-file query could not be executed safely.",
            "details": {},
        }
    finally:
        if duckdb_connection is not None:
            try:
                duckdb_connection.close()
            except BaseException:  # noqa: BLE001,S110 - best-effort child cleanup
                pass
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            _send_message(connection, final)
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


def _request_mapping(
    manifest: LocalFileQueryManifest,
    canonical_sql: str,
    sql_fingerprint: str,
    limits: LocalFileQueryLimits,
) -> dict[str, object]:
    return {
        "protocol": _PROTOCOL,
        "duckdb_version": _DUCKDB_VERSION,
        "format": manifest.format,
        "workspace_id": manifest.workspace_id,
        "path_pattern": manifest.path_pattern,
        "canonical_sql": canonical_sql,
        "sql_fingerprint": sql_fingerprint,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_bytes": manifest.encoded_bytes,
        "bindings": [item.provenance_mapping() for item in manifest.bindings],
        "query_seconds": float(limits.max_query_seconds),
        "max_result_rows": limits.max_result_rows,
        "max_result_bytes": limits.max_result_bytes,
        "max_result_columns": limits.max_result_columns,
        "memory_bytes": limits.duckdb_memory_bytes,
        "spill_bytes": limits.max_spill_bytes,
        "rss_bytes": limits.max_worker_rss_bytes,
        "threads": limits.duckdb_threads,
    }


def _validate_request(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _WorkerFailure(
            "file_query_invalid", "The private file-query request is malformed."
        ) from error
    expected = {
        "protocol",
        "duckdb_version",
        "format",
        "workspace_id",
        "path_pattern",
        "canonical_sql",
        "sql_fingerprint",
        "manifest_sha256",
        "manifest_bytes",
        "bindings",
        "query_seconds",
        "max_result_rows",
        "max_result_bytes",
        "max_result_columns",
        "memory_bytes",
        "spill_bytes",
        "rss_bytes",
        "threads",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise _WorkerFailure(
            "file_query_invalid", "The private file-query request is invalid."
        )
    if value["protocol"] != _PROTOCOL or value["duckdb_version"] != _DUCKDB_VERSION:
        raise _WorkerFailure(
            "dependency_unavailable",
            "The local structured-file query protocol is incompatible.",
        )
    if value["format"] not in {
        "csv",
        "tsv",
        "json_records",
        "ndjson",
        "parquet",
    }:
        raise _WorkerFailure(
            "format_unsupported", "The structured-file format is unsupported."
        )
    if (
        not isinstance(value["workspace_id"], str)
        or not value["workspace_id"].startswith("workspace:sha256:")
        or not isinstance(value["path_pattern"], str)
        or not value["path_pattern"]
        or value["path_pattern"].startswith("/")
        or "\\" in value["path_pattern"]
        or "\x00" in value["path_pattern"]
    ):
        raise _WorkerFailure(
            "file_query_invalid", "The bound file-query manifest is invalid."
        )
    bindings = value["bindings"]
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 1_000:
        raise _WorkerFailure(
            "file_pattern_too_broad", "The bound file-query manifest is invalid."
        )
    required_binding = {
        "path",
        "format",
        "physical_revision",
        "device",
        "inode",
        "mode",
        "uid",
        "gid",
        "link_count",
        "size_bytes",
        "modified_ns",
        "changed_ns",
        "observed_at",
    }
    if any(
        not isinstance(item, dict)
        or set(item) != required_binding
        or item.get("format") != value["format"]
        for item in bindings
    ):
        raise _WorkerFailure(
            "file_query_invalid", "The bound file-query manifest is invalid."
        )
    manifest_material = {
        "protocol": "daita.local_file_query_manifest.v1",
        "workspace_id": value["workspace_id"],
        "path_pattern": value["path_pattern"],
        "format": value["format"],
        "bindings": bindings,
    }
    encoded_manifest = canonical_json(manifest_material).encode("utf-8")
    if (
        not isinstance(value["manifest_bytes"], int)
        or isinstance(value["manifest_bytes"], bool)
        or value["manifest_bytes"] != len(encoded_manifest)
        or value["manifest_sha256"] != "sha256:" + sha256(encoded_manifest).hexdigest()
    ):
        raise _WorkerFailure(
            "file_query_invalid", "The bound file-query manifest changed."
        )
    integer_names = (
        "max_result_rows",
        "max_result_bytes",
        "max_result_columns",
        "memory_bytes",
        "spill_bytes",
        "rss_bytes",
        "threads",
    )
    if any(
        not isinstance(value[name], int) or isinstance(value[name], bool)
        for name in integer_names
    ) or (
        not isinstance(value["query_seconds"], (int, float))
        or isinstance(value["query_seconds"], bool)
    ):
        raise _WorkerFailure(
            "file_query_invalid", "The private file-query limits are invalid."
        )
    try:
        LocalFileQueryLimits(
            max_query_seconds=value["query_seconds"],
            max_result_rows=value["max_result_rows"],
            max_result_bytes=value["max_result_bytes"],
            max_result_columns=value["max_result_columns"],
            duckdb_memory_bytes=value["memory_bytes"],
            max_spill_bytes=value["spill_bytes"],
            max_worker_rss_bytes=value["rss_bytes"],
            duckdb_threads=value["threads"],
        )
    except (TypeError, ValueError) as error:
        raise _WorkerFailure(
            "file_query_invalid", "The private file-query limits are invalid."
        ) from error
    canonical_sql = value["canonical_sql"]
    if not isinstance(canonical_sql, str) or not 1 <= len(canonical_sql) <= 8 * 1_024:
        raise _WorkerFailure(
            "file_query_invalid", "The canonical file query is invalid."
        )
    fingerprint = "sha256:" + sha256(canonical_sql.encode("utf-8")).hexdigest()
    if value["sql_fingerprint"] != fingerprint:
        raise _WorkerFailure(
            "file_query_invalid", "The canonical file-query fingerprint changed."
        )
    return value


def _load_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as error:
        raise ImportError(
            "Daita's local structured-file query dependency is unavailable. "
            f"{repair_guidance()}"
        ) from error
    if duckdb.__version__ != _DUCKDB_VERSION:
        raise ImportError(
            "Daita's local structured-file query dependency is incompatible. "
            f"{repair_guidance()}"
        )
    return duckdb


def _revalidate_canonical_sql(request: dict[str, Any]) -> None:
    from ..domains.data.sql.duckdb_read import (
        DuckDBReadValidationError,
        validate_duckdb_read,
    )

    try:
        validated = validate_duckdb_read(str(request["canonical_sql"]))
    except DuckDBReadValidationError as error:
        raise _WorkerFailure(
            "file_query_invalid",
            "The private worker rejected non-canonical file-query SQL.",
            {"reason": error.reason},
        ) from error
    if (
        validated.canonical_sql != request["canonical_sql"]
        or validated.sql_fingerprint != request["sql_fingerprint"]
    ):
        raise _WorkerFailure(
            "file_query_invalid",
            "The private worker rejected changed file-query SQL.",
        )


def _apply_security_profile(
    connection: Any,
    descriptor_paths: tuple[str, ...],
    scratch: Path,
    request: dict[str, Any],
) -> None:
    extension = connection.execute(
        "SELECT loaded, installed FROM duckdb_extensions() "
        "WHERE extension_name = 'httpfs'"
    ).fetchone()
    if extension is None or bool(extension[0]) or bool(extension[1]):
        raise _WorkerFailure(
            "dependency_unavailable",
            "DuckDB's network filesystem extension is not safely absent.",
        )
    settings = {
        str(name): value
        for name, value in connection.execute(
            "SELECT name, value FROM duckdb_settings()"
        ).fetchall()
    }
    required = {
        "allowed_paths",
        "allowed_directories",
        "threads",
        "memory_limit",
        "temp_directory",
        "max_temp_directory_size",
        "autoinstall_known_extensions",
        "autoload_known_extensions",
        "allow_community_extensions",
        "allow_persistent_secrets",
        "enable_external_access",
        "allowed_configs",
        "lock_configuration",
    }
    if not required <= set(settings):
        raise _WorkerFailure(
            "dependency_unavailable",
            "DuckDB cannot provide Daita's required locked security profile.",
        )
    _set(connection, "allowed_paths", _sql_list(descriptor_paths))
    configured_paths = connection.execute(
        "SELECT current_setting('allowed_paths')"
    ).fetchone()
    if (
        configured_paths is None
        or not isinstance(configured_paths[0], list)
        or len(configured_paths[0]) != len(descriptor_paths)
        or any(
            not isinstance(item, str)
            or not item.startswith(("/dev/fd/", "/proc/self/fd/"))
            for item in configured_paths[0]
        )
    ):
        raise _WorkerFailure(
            "dependency_unavailable",
            "DuckDB did not admit the exact file-query descriptor set.",
        )
    expected_paths = tuple(configured_paths[0])
    _set(connection, "allowed_directories", _sql_list((os.fspath(scratch),)))
    _set(connection, "threads", str(request["threads"]))
    _set(
        connection,
        "memory_limit",
        _sql_string(f"{int(request['memory_bytes']) // _MIB}MiB"),
    )
    _set(connection, "temp_directory", _sql_string(os.fspath(scratch)))
    _set(
        connection,
        "max_temp_directory_size",
        _sql_string(f"{int(request['spill_bytes']) // _MIB}MiB"),
    )
    if "preserve_insertion_order" in settings:
        _set(connection, "preserve_insertion_order", "false")
    if "secret_directory" in settings:
        _set(
            connection,
            "secret_directory",
            _sql_string(os.fspath(scratch / "secrets-disabled")),
        )
    _set(connection, "autoinstall_known_extensions", "false")
    _set(connection, "autoload_known_extensions", "false")
    _set(connection, "allow_community_extensions", "false")
    if "allow_unsigned_extensions" in settings:
        _set(connection, "allow_unsigned_extensions", "false")
    _set(connection, "allow_persistent_secrets", "false")
    if "enable_global_s3_configuration" in settings:
        _set(connection, "enable_global_s3_configuration", "false")
    elif bool(extension[0]) or bool(extension[1]):  # pragma: no cover - guarded above
        raise _WorkerFailure(
            "dependency_unavailable",
            "DuckDB's network filesystem extension is not safely absent.",
        )
    _set(connection, "enable_external_access", "false")
    _set(connection, "allowed_configs", "[]")
    _set(connection, "lock_configuration", "true")

    expected = {
        "threads": int(request["threads"]),
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        "allow_community_extensions": False,
        "allow_persistent_secrets": False,
        "enable_external_access": False,
        "allowed_configs": [],
        "lock_configuration": True,
    }
    for name, value in expected.items():
        observed = connection.execute("SELECT current_setting(?)", [name]).fetchone()
        if observed is None or observed[0] != value:
            raise _WorkerFailure(
                "dependency_unavailable",
                "DuckDB did not retain Daita's locked security profile.",
                {"setting": name},
            )
    observed_memory = connection.execute(
        "SELECT current_setting('memory_limit')"
    ).fetchone()
    observed_temp = connection.execute(
        "SELECT current_setting('temp_directory')"
    ).fetchone()
    observed_spill = connection.execute(
        "SELECT current_setting('max_temp_directory_size')"
    ).fetchone()
    if (
        observed_memory is None
        or _duckdb_size_bytes(observed_memory[0]) != int(request["memory_bytes"])
        or observed_temp is None
        or Path(str(observed_temp[0])).resolve() != scratch.resolve()
        or observed_spill is None
        or _duckdb_size_bytes(observed_spill[0]) != int(request["spill_bytes"])
    ):
        raise _WorkerFailure(
            "dependency_unavailable",
            "DuckDB did not retain Daita's bounded execution settings.",
        )
    if "enable_global_s3_configuration" in settings:
        observed_s3 = connection.execute(
            "SELECT current_setting('enable_global_s3_configuration')"
        ).fetchone()
        if observed_s3 is None or observed_s3[0] is not False:
            raise _WorkerFailure(
                "dependency_unavailable",
                "DuckDB did not disable global S3 configuration.",
            )
    observed_paths = connection.execute(
        "SELECT current_setting('allowed_paths')"
    ).fetchone()
    observed_directories = connection.execute(
        "SELECT current_setting('allowed_directories')"
    ).fetchone()
    observed_path_values = observed_paths[0] if observed_paths is not None else None
    paths_match = (
        isinstance(observed_path_values, list)
        and tuple(observed_path_values) == expected_paths
    )
    if (
        not paths_match
        or observed_directories is None
        or not isinstance(observed_directories[0], list)
        or len(observed_directories[0]) != 1
        or Path(str(observed_directories[0][0])).resolve() != scratch.resolve()
    ):
        raise _WorkerFailure(
            "dependency_unavailable",
            "DuckDB did not retain the exact file-query path restrictions.",
        )


def _set(connection: Any, name: str, value: str) -> None:
    try:
        connection.execute(f"SET {name} = {value}")
    except BaseException as error:
        raise _WorkerFailure(
            "dependency_unavailable",
            "DuckDB cannot provide Daita's required locked security profile.",
            {"setting": name},
        ) from error


def _duckdb_size_bytes(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?) ([KMGT]iB)", value)
    if match is None:
        return None
    multipliers = {
        "KiB": 1_024,
        "MiB": _MIB,
        "GiB": _GIB,
        "TiB": 1_024 * _GIB,
    }
    return int(float(match.group(1)) * multipliers[match.group(2)])


def _bind_data(
    connection: Any,
    *,
    format_name: str,
    descriptor_paths: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    schemas: list[tuple[tuple[object, ...], ...]] = []
    for path in descriptor_paths:
        reader = _reader_sql(format_name, (path,))
        try:
            schema = tuple(
                tuple(item)
                for item in connection.execute(
                    f"DESCRIBE SELECT * FROM {reader}"
                ).fetchall()
            )
        except BaseException as error:
            raise _WorkerFailure(
                "file_query_invalid",
                "A matched structured file is malformed or unreadable.",
                {"stage": "schema"},
            ) from error
        schemas.append(schema)
    if any(item != schemas[0] for item in schemas[1:]):
        raise _WorkerFailure(
            "file_query_invalid",
            "Matched files do not have one compatible logical schema.",
            {"stage": "schema_compatibility"},
        )
    try:
        connection.execute(
            "CREATE TEMP VIEW data AS SELECT * FROM "
            + _reader_sql(format_name, descriptor_paths)
        )
    except BaseException as error:
        raise _WorkerFailure(
            "file_query_invalid",
            "The homogeneous structured dataset could not be bound safely.",
            {"stage": "binding"},
        ) from error
    return tuple({"name": str(item[0]), "type": str(item[1])} for item in schemas[0])


def _reader_sql(format_name: str, paths: tuple[str, ...]) -> str:
    encoded = _sql_list(paths)
    if format_name == "csv":
        return (
            f"read_csv({encoded}, header=true, auto_detect=true, sample_size=-1, "
            "union_by_name=false)"
        )
    if format_name == "tsv":
        return (
            f"read_csv({encoded}, header=true, delim='\\t', auto_detect=true, "
            "sample_size=-1, union_by_name=false)"
        )
    if format_name == "json_records":
        return f"read_json({encoded}, format='array', union_by_name=false)"
    if format_name == "ndjson":
        return f"read_json({encoded}, format='newline_delimited', union_by_name=false)"
    if format_name == "parquet":
        return f"read_parquet({encoded}, union_by_name=false)"
    raise _WorkerFailure(
        "format_unsupported", "The structured-file format is unsupported."
    )


def _execute_bounded_query(
    connection: Any,
    sql: str,
    sql_fingerprint: str,
    *,
    max_rows: int,
    max_bytes: int,
    max_columns: int,
) -> dict[str, object]:
    try:
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise _WorkerFailure(
                "file_query_invalid", "The validated file query returned no rows."
            )
        raw_names = tuple(str(item[0]) for item in cursor.description)
        if len(raw_names) > max_columns:
            raise _WorkerFailure(
                "file_query_limited",
                "The file query returned too many columns.",
                {"column_limit": max_columns},
            )
        if any(
            not name or len(name.encode("utf-8")) > _MAX_COLUMN_NAME_BYTES
            for name in raw_names
        ):
            raise _WorkerFailure(
                "file_query_limited",
                "A file-query result column name exceeded its bound.",
            )
        columns = _unique_columns(raw_names)
        column_types = tuple(str(item[1]) for item in cursor.description)
        raw_rows = cursor.fetchmany(max_rows + 1)
    except _WorkerFailure:
        raise
    except BaseException as error:
        raise _WorkerFailure(
            "file_query_invalid",
            "The validated local structured-file query failed.",
            {"stage": "query"},
        ) from error

    rows: list[dict[str, object]] = []
    byte_limited = False
    for raw in raw_rows[:max_rows]:
        row = {
            column: _json_value(value)
            for column, value in zip(columns, raw, strict=True)
        }
        candidate = [*rows, row]
        if len(canonical_json(candidate).encode("utf-8")) > max_bytes:
            byte_limited = True
            break
        rows.append(row)
    reasons: list[str] = []
    if len(raw_rows) > max_rows:
        reasons.append("row_limit")
    if byte_limited:
        reasons.append("byte_limit")
    encoded_rows = len(canonical_json(rows).encode("utf-8"))
    data = {
        "canonical_sql": sql,
        "sql_fingerprint": sql_fingerprint,
        "columns": columns,
        "column_types": column_types,
        "rows": rows,
        "observed_rows": len(raw_rows),
        "returned_rows": len(rows),
        "utf8_bytes": encoded_rows,
        "row_limit": max_rows,
        "byte_limit": max_bytes,
        "truncated": bool(reasons),
        "truncation_reasons": reasons,
        "trust_classification": "untrusted_external_data",
    }
    probe = canonical_json({"kind": "final", "status": "success", "data": data}).encode(
        "utf-8"
    )
    if len(probe) > _MAX_RESPONSE_BYTES:
        raise _WorkerFailure(
            "file_query_limited", "The file-query result exceeded its byte bound."
        )
    return data


def _json_value(value: object, *, depth: int = 0) -> object:
    if depth > _MAX_VALUE_DEPTH:
        raise _WorkerFailure(
            "file_query_limited",
            "A file-query value exceeded the bounded JSON projection.",
        )
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _WorkerFailure(
                "file_query_limited",
                "A file-query value exceeded the bounded JSON projection.",
            )
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise _WorkerFailure(
                "file_query_limited",
                "A file-query value exceeded the bounded JSON projection.",
            )
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {
            "type": "interval",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 1_024:
            raise _WorkerFailure(
                "file_query_limited",
                "A file-query value exceeded the bounded JSON projection.",
            )
        return [_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 1_024 or any(not isinstance(key, str) for key in value):
            raise _WorkerFailure(
                "file_query_limited",
                "A file-query value exceeded the bounded JSON projection.",
            )
        return {key: _json_value(item, depth=depth + 1) for key, item in value.items()}
    raise _WorkerFailure(
        "file_query_limited",
        "A file-query value exceeded the bounded JSON projection.",
    )


def _unique_columns(values: tuple[str, ...]) -> tuple[str, ...]:
    reserved = set(values)
    result: list[str] = []
    used: set[str] = set()
    for base in values:
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


def _revalidate_descriptors(
    descriptors: list[int], bindings: list[dict[str, Any]]
) -> None:
    for descriptor, binding in zip(descriptors, bindings, strict=True):
        try:
            facts = os.fstat(descriptor)
        except OSError as error:
            raise _WorkerFailure(
                "file_changed", "A bound workspace file became unavailable."
            ) from error
        if not stat.S_ISREG(facts.st_mode) or _physical_revision(facts) != binding.get(
            "physical_revision"
        ):
            raise _WorkerFailure(
                "file_changed", "A bound workspace file changed during the query."
            )


def _physical_revision(value: os.stat_result) -> str:
    material = {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "link_count": int(value.st_nlink),
        "size_bytes": int(value.st_size),
        "modified_ns": int(value.st_mtime_ns),
        "changed_ns": int(value.st_ctime_ns),
        "flags": int(getattr(value, "st_flags", 0)),
    }
    return "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()


def _descriptor_path(descriptor: int) -> str:
    for root in ("/dev/fd", "/proc/self/fd"):
        if os.path.isdir(root):
            return f"{root}/{descriptor}"
    raise _WorkerFailure(
        "dependency_unavailable",
        "This platform cannot expose exact descriptors to the private query worker.",
    )


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_sql_string(item) for item in values) + "]"


def _send_message(connection: Connection, value: dict[str, object]) -> None:
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > _MAX_RESPONSE_BYTES:
        encoded = canonical_json(
            {
                "kind": "final",
                "status": "error",
                "code": "file_query_limited",
                "message": "The private file-query result exceeded its byte bound.",
                "details": {},
            }
        ).encode("utf-8")
    connection.send_bytes(encoded)


def _receive_messages(
    connection: Connection, final: dict[str, object] | None
) -> dict[str, object] | None:
    while connection.poll(0):
        try:
            raw = connection.recv_bytes(maxlength=_MAX_RESPONSE_BYTES)
        except EOFError:
            break
        try:
            message = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise LocalFileQueryError(
                "file_query_invalid",
                "The private file-query worker returned a malformed result.",
            ) from error
        if not isinstance(message, dict) or message.get("kind") not in {
            "ready",
            "final",
        }:
            raise LocalFileQueryError(
                "file_query_invalid",
                "The private file-query worker returned an invalid result.",
            )
        if message.get("kind") == "final":
            final = message
    return final


def _create_private_scratch(parent: Path) -> Path:
    try:
        parent.mkdir(mode=0o700, parents=False, exist_ok=True)
        facts = parent.lstat()
        if (
            not stat.S_ISDIR(facts.st_mode)
            or stat.S_ISLNK(facts.st_mode)
            or int(facts.st_uid) != os.getuid()
        ):
            raise OSError("unsafe scratch parent")
        path = Path(tempfile.mkdtemp(prefix="query-", dir=parent))
        path.chmod(0o700)
        return path
    except OSError as error:
        raise LocalFileQueryError(
            "workspace_unavailable",
            "Private file-query scratch space is unavailable.",
        ) from error


def _remove_private_scratch(path: Path) -> None:
    try:
        facts = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(facts.st_mode) or stat.S_ISLNK(facts.st_mode):
        raise LocalFileQueryError(
            "workspace_unavailable",
            "Private file-query scratch state changed unexpectedly.",
        )
    shutil.rmtree(path)


def _directory_bytes(path: Path) -> int:
    total = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except FileNotFoundError:
            continue
        for entry in entries:
            try:
                facts = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(facts.st_mode):
                total += int(facts.st_size)
            elif stat.S_ISDIR(facts.st_mode) and not stat.S_ISLNK(facts.st_mode):
                pending.append(Path(entry.path))
    return total


def _process_rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (FileNotFoundError, OSError, ValueError, IndexError):
            return None
    if sys.platform == "darwin":
        return _darwin_process_rss_bytes(pid)
    return None


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32),
        ("threadnum", ctypes.c_int32),
        ("numrunning", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


def _darwin_process_rss_bytes(pid: int) -> int | None:
    function = _darwin_proc_pidinfo()
    if function is None:
        return None
    try:
        info = _ProcTaskInfo()
        size = ctypes.sizeof(info)
        returned = function(pid, 4, 0, ctypes.byref(info), size)
        return int(info.resident_size) if returned == size else None
    except (AttributeError, OSError):
        return None


@lru_cache(maxsize=1)
def _darwin_proc_pidinfo() -> Any | None:
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        function = library.proc_pidinfo
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        return function
    except (AttributeError, OSError):
        return None


def _set_worker_limits() -> None:
    try:
        import resource
    except ImportError:
        return
    requested = {
        "RLIMIT_CPU": (35, 36),
        "RLIMIT_FSIZE": (2 * _GIB, 2 * _GIB),
        "RLIMIT_NOFILE": (2_048, 2_048),
        "RLIMIT_CORE": (0, 0),
    }
    for name, desired in requested.items():
        resource_id = getattr(resource, name, None)
        if resource_id is None:
            continue
        try:
            current = resource.getrlimit(resource_id)
            hard = (
                desired[1]
                if current[1] == resource.RLIM_INFINITY
                else min(desired[1], current[1])
            )
            soft = min(desired[0], hard)
            resource.setrlimit(resource_id, (soft, hard))
        except (OSError, ValueError):
            if name in {"RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_CORE"}:
                raise _WorkerFailure(
                    "dependency_unavailable",
                    "The private file-query worker limits are unavailable.",
                    {"limit": name},
                )


__all__ = [
    "LocalFileQueryBackend",
    "LocalFileQueryError",
    "LocalFileQueryLimits",
    "LocalFileQueryResult",
]
