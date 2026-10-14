"""Execute catalog-validated, bounded read queries against PostgreSQL."""

from __future__ import annotations

import asyncio
import base64
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
from typing import Any, cast
from uuid import UUID

from .._json import FrozenJsonObject, canonical_json
from ..artifacts.models import ArtifactError
from ..artifacts.renderers import (
    ExactArrowRenderer,
    ExactCsvRenderer,
    ExactXlsxProvenance,
    ExactXlsxRenderer,
    arrow_schema,
)
from ..config import PostgreSQLReadLimits
from ..domains.data.capabilities import PostgreSQLReadResult
from ..domains.data.controller import ReadScopedCatalogReader
from ..domains.data.export_capabilities import (
    ExactTabularExportResult,
    ExactTabularProgress,
    resolved_exact_export_sensitivity,
)
from ..domains.data.results import project_result_rows
from ..domains.data.sql import validate_postgresql_read
from ..errors import DaitaError, ErrorRetryability
from ..observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely
from ..security import SecretProvider, SecretReference, default_secret_provider
from ..storage.sqlite_records import SourcePermissionStateError
from .models import SourceRegistration
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
_PERMANENT_QUERY_ERRORS = frozenset(
    {
        "source_not_available",
        "source_adapter_mismatch",
        "source_permission_state_invalid",
        "resource_read_not_allowed",
        "query_revalidation_failed",
        "query_resource_scope_empty",
        "catalog_provenance_missing",
        "catalog_source_stale",
        "query_result_too_wide",
        "query_result_invalid",
        "query_value_unsupported",
    }
)


class PostgreSQLQueryError(DaitaError):
    """Normalized query-boundary failure without connector error leakage."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("query error code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("query error message must be a non-empty string")
        self.code = code
        super().__init__(
            message,
            error_code=code,
            retryability=(
                ErrorRetryability.PERMANENT
                if code in _PERMANENT_QUERY_ERRORS
                else ErrorRetryability.UNKNOWN
            ),
        )


@dataclass(frozen=True, slots=True)
class PostgreSQLReadDiagnostics:
    limits: PostgreSQLReadLimits
    pool_size_per_source: int
    cached_sources: int
    reserved_connections: int
    high_water_reserved_connections: int
    waiting: int
    attempts: int
    acquired: int
    completed: int
    active: int
    high_water: int
    timeouts: int
    cancellations: int
    connections_created: int
    pools_created: int
    pools_retired: int
    discarded: int
    queue_seconds: float
    structural_seconds: float
    query_seconds: float
    cleanup_seconds: float


@dataclass(eq=False, slots=True)
class _ReadPool:
    registration: SourceRegistration
    readable: frozenset[str]
    password: str | None = field(repr=False)
    size: int = 1
    pool: Any = None
    users: int = 1
    ready: bool = False
    retired: bool = False
    closing: bool = False
    last_used: float = 0.0


@dataclass(frozen=True, slots=True)
class _ResolvedReadSecret:
    reference: str | None
    password: str | None = field(repr=False)

    async def resolve(self, reference: SecretReference) -> str:
        if reference.to_uri() != self.reference or self.password is None:
            raise ValueError("The pooled credential admission is unavailable")
        return self.password


class PostgreSQLQueryBackend:
    """Revalidate catalog scope and structure, then run one bounded read."""

    def __init__(
        self,
        sources: SourceStore,
        catalog: ReadScopedCatalogReader,
        secret_provider: SecretProvider | None = None,
        *,
        statement_timeout_seconds: float = 5.0,
        cleanup_timeout_seconds: float = 1.0,
        read_limits: PostgreSQLReadLimits | None = None,
        source_capacity: int = 2,
        deadline_reader: Callable[[], float | None] | None = None,
        observer: AgentObserver | None = None,
    ) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        for method_name in ("resource_schemas", "readable_resource_ids"):
            if not callable(getattr(catalog, method_name, None)):
                raise TypeError(f"catalog must provide {method_name}")
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
        if read_limits is not None and not isinstance(
            read_limits, PostgreSQLReadLimits
        ):
            raise TypeError("read_limits must be PostgreSQLReadLimits")
        self._read_limits = read_limits or PostgreSQLReadLimits(
            statement_timeout_seconds=statement_timeout_seconds,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
        )
        self._read_limits.__post_init__()
        if type(source_capacity) is not int or not 1 <= source_capacity <= 8:
            raise ValueError(
                "source_capacity must be an integer from one through eight"
            )
        self._pool_size = min(
            source_capacity,
            self._read_limits.pool_size_per_source,
            self._read_limits.max_connections,
        )
        self._deadline_reader = deadline_reader
        self._observer = observer
        self._pool_condition = asyncio.Condition()
        self._pools: dict[tuple[str, str], _ReadPool] = {}
        self._owned_pools: set[_ReadPool] = set()
        self._pool_closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._operation_deadline: ContextVar[float | None] = ContextVar(
            "postgresql_read_deadline", default=None
        )
        self._checkout_waiting = 0
        self._high_water_connections = 0
        self._metrics: dict[str, Any] = {
            name: 0
            for name in (
                "attempts",
                "acquired",
                "completed",
                "active",
                "high_water",
                "timeouts",
                "cancellations",
                "connections_created",
                "pools_created",
                "pools_retired",
                "discarded",
            )
        }
        self._metrics.update(
            {
                name: 0.0
                for name in (
                    "queue_seconds",
                    "structural_seconds",
                    "query_seconds",
                    "cleanup_seconds",
                )
            }
        )
        self._sources = sources
        self._catalog = catalog
        self._secret_provider = provider
        self._statement_timeout_seconds = float(
            self._read_limits.statement_timeout_seconds
        )
        self._cleanup_timeout_seconds = float(self._read_limits.cleanup_timeout_seconds)

    def diagnostics(self) -> PostgreSQLReadDiagnostics:
        return PostgreSQLReadDiagnostics(
            limits=self._read_limits,
            pool_size_per_source=self._pool_size,
            cached_sources=len(self._owned_pools),
            reserved_connections=sum(entry.size for entry in self._owned_pools),
            high_water_reserved_connections=self._high_water_connections,
            waiting=self._checkout_waiting,
            **self._metrics,
        )

    @asynccontextmanager
    async def _operation(self, seconds: float):
        if not self._read_limits.pool_enabled:
            yield None
            return
        deadline = (
            asyncio.get_running_loop().time()
            + seconds
            + self._read_limits.acquire_timeout_seconds
        )
        host_deadline = (
            None if self._deadline_reader is None else self._deadline_reader()
        )
        if host_deadline is not None:
            deadline = min(deadline, host_deadline)
        token = self._operation_deadline.set(deadline)
        try:
            async with asyncio.timeout_at(deadline):
                yield deadline
        finally:
            self._operation_deadline.reset(token)

    def _remaining(self, maximum: float) -> float:
        deadline = self._operation_deadline.get()
        remaining = (
            maximum
            if deadline is None
            else min(maximum, deadline - asyncio.get_running_loop().time())
        )
        if remaining <= 0:
            raise TimeoutError("The admitted PostgreSQL read deadline expired")
        return remaining

    async def _password(self, registration: SourceRegistration) -> str | None:
        reference = registration.configuration.get("credential_ref")
        if reference is None:
            return None
        failed = False
        try:
            password = await self._secret_provider.resolve(
                SecretReference.parse(cast(str, reference))
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
        if failed:
            raise PostgreSQLQueryError(
                "postgresql_credential_unavailable",
                "PostgreSQL credential resolution failed.",
            )
        if not isinstance(password, str) or not password:
            raise PostgreSQLQueryError(
                "postgresql_credential_invalid",
                "PostgreSQL credential resolution returned an invalid value.",
            )
        return password

    def _retire_locked(self, entry: _ReadPool) -> None:
        if not entry.retired:
            entry.retired = True
            self._metrics["pools_retired"] += 1
        key = (entry.registration.agent_id, entry.registration.id)
        if self._pools.get(key) is entry:
            del self._pools[key]

    async def _close_entry(self, entry: _ReadPool) -> None:
        # The short bookkeeping lock is never held over native pool I/O.
        try:
            if entry.pool is not None:
                try:
                    async with asyncio.timeout(
                        self._read_limits.cleanup_timeout_seconds
                    ):
                        await entry.pool.close()
                except (Exception, asyncio.CancelledError):
                    try:
                        entry.pool.terminate()
                    except Exception:
                        pass  # min_size=0 initialization owns no connections before ready
        finally:
            entry.password = None
            async with self._pool_condition:
                self._owned_pools.discard(entry)
                self._pool_condition.notify_all()

    async def _drop_user(self, entry: _ReadPool) -> None:
        async with self._pool_condition:
            entry.users -= 1
            entry.last_used = asyncio.get_running_loop().time()
            close = entry.retired and entry.users == 0 and not entry.closing
            if close:
                entry.closing = True
            self._pool_condition.notify_all()
        if close:
            await self._close_entry(entry)

    async def _still_admitted(self, entry: _ReadPool) -> bool:
        registration = entry.registration
        current = await self._sources.load_source(
            registration.agent_id, registration.id
        )
        readable = await self._catalog.readable_resource_ids(
            registration.agent_id, (registration.id,)
        )
        return (
            current == registration
            and registration.active
            and frozenset(readable) == entry.readable
            and await self._password(registration) == entry.password
        )

    async def _connection(
        self, registration: SourceRegistration, readable: frozenset[str]
    ):
        if self._pool_closed:
            raise PostgreSQLQueryError(
                "postgresql_pool_closed", "The PostgreSQL read backend is closed."
            )
        if not self._read_limits.pool_enabled:
            return await _connect(registration, self._secret_provider), None
        started = asyncio.get_running_loop().time()
        self._metrics["attempts"] += 1
        self._checkout_waiting += 1
        entry = None
        connection = None
        try:
            deadline = started + self._remaining(
                self._read_limits.acquire_timeout_seconds
            )
            async with asyncio.timeout_at(deadline):
                password = await self._password(registration)
                key = (registration.agent_id, registration.id)
                while True:
                    closing = []
                    create = False
                    async with self._pool_condition:
                        if self._pool_closed:
                            raise PostgreSQLQueryError(
                                "postgresql_pool_closed",
                                "The PostgreSQL read backend is closed.",
                            )
                        now = asyncio.get_running_loop().time()
                        current = self._pools.get(key)
                        for cached in tuple(self._pools.values()):
                            changed = cached is current and (
                                cached.registration != registration
                                or cached.password != password
                                or cached.readable != readable
                            )
                            expired = (
                                cached.users == 0
                                and now - cached.last_used
                                >= self._read_limits.idle_timeout_seconds
                            )
                            if changed or expired:
                                self._retire_locked(cached)
                        current = self._pools.get(key)
                        for cached in self._owned_pools:
                            if (
                                cached.retired
                                and cached.users == 0
                                and not cached.closing
                            ):
                                cached.closing = True
                                closing.append(cached)
                        if current is not None and current.ready:
                            current.users += 1
                            entry = current
                        elif current is None and not closing:
                            reserved = sum(cached.size for cached in self._owned_pools)
                            if (
                                len(self._owned_pools)
                                < self._read_limits.max_cached_sources
                                and reserved + self._pool_size
                                <= self._read_limits.max_connections
                            ):
                                entry = _ReadPool(
                                    registration,
                                    readable,
                                    password,
                                    self._pool_size,
                                    last_used=now,
                                )
                                self._owned_pools.add(entry)
                                self._pools[key] = entry
                                self._high_water_connections = max(
                                    self._high_water_connections, reserved + entry.size
                                )
                                create = True
                            else:
                                idle = [
                                    cached
                                    for cached in self._owned_pools
                                    if cached.users == 0 and not cached.closing
                                ]
                                if idle:
                                    cached = min(
                                        idle, key=lambda value: value.last_used
                                    )
                                    self._retire_locked(cached)
                                    cached.closing = True
                                    closing.append(cached)
                        if entry is None and not closing:
                            await self._pool_condition.wait()
                    for cached in closing:
                        await self._close_entry(cached)
                    if entry is not None:
                        break
                if create:
                    from .postgresql import _load_asyncpg

                    snapshot = _ResolvedReadSecret(
                        cast(
                            str | None, registration.configuration.get("credential_ref")
                        ),
                        password,
                    )

                    async def connect(*args: object, **kwargs: object):
                        del args, kwargs
                        created = await _connect(registration, snapshot)
                        self._metrics["connections_created"] += 1
                        return created

                    entry.pool = _load_asyncpg().create_pool(
                        min_size=0,
                        max_size=entry.size,
                        max_inactive_connection_lifetime=self._read_limits.idle_timeout_seconds,
                        connect=connect,
                    )
                    await entry.pool
                    async with self._pool_condition:
                        entry.ready = True
                        self._metrics["pools_created"] += 1
                        self._pool_condition.notify_all()
                connection = await entry.pool.acquire(
                    timeout=max(0.001, deadline - asyncio.get_running_loop().time())
                )
                self._metrics["acquired"] += 1
                self._metrics["active"] += 1
                self._metrics["high_water"] = max(
                    self._metrics["high_water"], self._metrics["active"]
                )
                if entry.retired or not await self._still_admitted(entry):
                    async with self._pool_condition:
                        self._retire_locked(entry)
                    raise PostgreSQLQueryError(
                        "postgresql_admission_changed",
                        "PostgreSQL source or credential admission changed before dispatch.",
                    )
                return connection, entry
        except BaseException as error:
            if isinstance(error, PostgreSQLQueryError) and error.code in {
                "postgresql_credential_unavailable",
                "postgresql_credential_invalid",
                "postgresql_admission_changed",
            }:
                await self.retire_source(registration.agent_id, registration.id)
            if isinstance(error, TimeoutError):
                self._metrics["timeouts"] += 1
            if isinstance(error, asyncio.CancelledError):
                self._metrics[
                    (
                        "timeouts"
                        if asyncio.get_running_loop().time() >= deadline
                        else "cancellations"
                    )
                ] += 1
            if entry is not None:
                if connection is not None:
                    await self._release_pooled(connection, entry, discard=True)
                else:
                    if not entry.ready:
                        async with self._pool_condition:
                            self._retire_locked(entry)
                    await self._drop_user(entry)
            raise
        finally:
            self._checkout_waiting -= 1
            self._metrics["queue_seconds"] += (
                asyncio.get_running_loop().time() - started
            )

    async def _release_pooled(
        self,
        connection: Any,
        entry: _ReadPool,
        *,
        discard: bool,
        deadline: float | None = None,
    ) -> None:
        async def settle():
            started = asyncio.get_running_loop().time()
            try:
                timeout = self._remaining(self._read_limits.cleanup_timeout_seconds)
                if deadline is not None:
                    timeout = min(timeout, deadline - asyncio.get_running_loop().time())
                    if timeout <= 0:
                        raise TimeoutError("The read deadline expired before reset")
            except TimeoutError:
                timeout = self._read_limits.cleanup_timeout_seconds
                discard_connection = True
            else:
                discard_connection = discard or entry.retired
            try:
                if discard_connection:
                    connection.terminate()
                    self._metrics["discarded"] += 1
                try:
                    await entry.pool.release(connection, timeout=timeout)
                except (Exception, asyncio.CancelledError):
                    connection.terminate()
                    if not discard_connection:
                        self._metrics["discarded"] += 1
            finally:
                self._metrics["active"] -= 1
                self._metrics["completed"] += 1
                self._metrics["cleanup_seconds"] += (
                    asyncio.get_running_loop().time() - started
                )
                await self._drop_user(entry)
                if self._observer is not None:
                    _emit_safely(
                        self._observer,
                        AgentEvent(
                            kind=AgentEventKind.ADMISSION_UPDATED,
                            occurred_at=datetime.now(UTC),
                            run_id="host",
                            conversation_id="host",
                            data=FrozenJsonObject.from_mapping(
                                {"permit_kind": "postgresql_checkout", **self._metrics}
                            ),
                        ),
                    )

        task = asyncio.create_task(settle())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def retire_source(self, agent_id: str, source_id: str) -> None:
        async with self._pool_condition:
            entries = [
                entry
                for entry in self._owned_pools
                if (entry.registration.agent_id, entry.registration.id)
                == (agent_id, source_id)
            ]
            closing = []
            for entry in entries:
                self._retire_locked(entry)
                if entry.users == 0 and not entry.closing:
                    entry.closing = True
                    closing.append(entry)
            self._pool_condition.notify_all()
        for entry in closing:
            await self._close_entry(entry)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        async with self._pool_condition:
            self._pool_closed = True
            entries = tuple(self._owned_pools)
            for entry in entries:
                self._retire_locked(entry)
                entry.closing = True
            self._pool_condition.notify_all()
        for entry in entries:
            await self._close_entry(entry)

    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_bytes: int,
        expected_resource_revisions: tuple[tuple[str, str], ...] | None = None,
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
            await self.retire_source(agent_id, source_id)
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
        if expected_resource_revisions is not None:
            current = {item.resource_id: item.revision for item in resources}
            if any(
                current.get(resource_id) != revision
                for resource_id, revision in expected_resource_revisions
            ):
                raise PostgreSQLQueryError(
                    "count_binding_stale",
                    "The bound count resource revision changed before source dispatch.",
                )
        try:
            readable = await self._catalog.readable_resource_ids(
                agent_id,
                (source_id,),
            )
        except SourcePermissionStateError:
            raise PostgreSQLQueryError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from None
        if self._read_limits.pool_enabled:
            cached = self._pools.get((agent_id, source_id))
            if cached is not None and (
                cached.registration != registration
                or cached.readable != frozenset(readable)
            ):
                await self.retire_source(agent_id, source_id)
        validation = validate_postgresql_read(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if not validation.valid or validation.analysis is None:
            if "resource_out_of_scope" in validation.issue_codes:
                raise PostgreSQLQueryError(
                    "resource_read_not_allowed",
                    "One or more requested resources are not available for reading.",
                )
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
        pool_entry = None
        read_deadline = None
        transaction = None
        transaction_finished = False
        connection_failure: tuple[str, str] | None = None
        query_failed = False
        columns: tuple[str, ...] = ()
        rows: tuple[Mapping[str, object], ...] = ()
        try:
            async with self._operation(
                self._statement_timeout_seconds
            ) as read_deadline:
                connection, pool_entry = await self._connection(
                    registration, frozenset(readable)
                )
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
                structural_started = asyncio.get_running_loop().time()
                structure = await _load_structure(
                    connection,
                    registration,
                    max_resources=_DEFAULT_MAX_RESOURCES,
                    max_columns=_DEFAULT_MAX_COLUMNS,
                    max_indexes=_DEFAULT_MAX_INDEXES,
                    max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
                )
                if self._read_limits.pool_enabled:
                    self._metrics["structural_seconds"] += (
                        asyncio.get_running_loop().time() - structural_started
                    )
                if structure.source_revision != validation.source_revision:
                    raise PostgreSQLQueryError(
                        "catalog_source_stale",
                        "PostgreSQL source structure changed after its catalog snapshot.",
                    )
                query_started = asyncio.get_running_loop().time()
                columns, rows, value_limited = await _execute_user_query(
                    connection,
                    validation.analysis.canonical_sql,
                    parameters,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    timeout_seconds=self._statement_timeout_seconds,
                )
                if self._read_limits.pool_enabled:
                    self._metrics["query_seconds"] += (
                        asyncio.get_running_loop().time() - query_started
                    )
                await transaction.commit()
                transaction_finished = True
        except asyncio.CancelledError:
            if pool_entry is not None:
                self._metrics["cancellations"] += 1
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
            if pool_entry is not None and connection is not None:
                await self._release_pooled(
                    connection,
                    pool_entry,
                    discard=not transaction_finished,
                    deadline=read_deadline,
                )
            else:
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
            await self.retire_source(agent_id, source_id)
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
        try:
            readable = await self._catalog.readable_resource_ids(
                agent_id,
                (source_id,),
            )
        except SourcePermissionStateError:
            raise PostgreSQLQueryError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from None
        if self._read_limits.pool_enabled:
            cached = self._pools.get((agent_id, source_id))
            if cached is not None and (
                cached.registration != registration
                or cached.readable != frozenset(readable)
            ):
                await self.retire_source(agent_id, source_id)
        validation = validate_postgresql_read(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if not validation.valid or validation.analysis is None:
            if "resource_out_of_scope" in validation.issue_codes:
                raise PostgreSQLQueryError(
                    "resource_read_not_allowed",
                    "One or more requested resources are not available for reading.",
                )
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
        pool_entry = None
        read_deadline = None
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
            async with self._operation(timeout_seconds) as read_deadline:
                connection, pool_entry = await self._connection(
                    registration, frozenset(readable)
                )
                transaction = connection.transaction(
                    isolation="repeatable_read",
                    readonly=True,
                )
                await transaction.start()
                query_timeout = (
                    min(
                        timeout_seconds,
                        self._statement_timeout_seconds,
                        self._remaining(timeout_seconds),
                    )
                    if self._read_limits.pool_enabled
                    else timeout_seconds
                )
                timeout_milliseconds = max(1, int(query_timeout * 1_000))
                await connection.execute(
                    "SELECT set_config('statement_timeout', $1, true)",
                    f"{timeout_milliseconds}ms",
                )
                await connection.execute(
                    "SELECT set_config('search_path', $1, true)",
                    "pg_catalog",
                )
                structural_started = asyncio.get_running_loop().time()
                structure = await _load_structure(
                    connection,
                    registration,
                    max_resources=_DEFAULT_MAX_RESOURCES,
                    max_columns=_DEFAULT_MAX_COLUMNS,
                    max_indexes=_DEFAULT_MAX_INDEXES,
                    max_relationships=_DEFAULT_MAX_RELATIONSHIPS,
                )
                if self._read_limits.pool_enabled:
                    self._metrics["structural_seconds"] += (
                        asyncio.get_running_loop().time() - structural_started
                    )
                if structure.source_revision != validation.source_revision:
                    raise PostgreSQLQueryError(
                        "catalog_source_stale",
                        "PostgreSQL source structure changed after its catalog snapshot.",
                    )
                async with asyncio.timeout(timeout_seconds):
                    query_started = asyncio.get_running_loop().time()
                content, columns, row_count = await _execute_exact_tabular_query(
                    connection,
                    validation.analysis.canonical_sql,
                    parameters,
                    format_name=format_name,
                    xlsx_provenance=xlsx_provenance,
                    max_rows=max_rows,
                    max_columns=max_columns,
                    max_bytes=max_bytes,
                    timeout_seconds=query_timeout,
                    progress=record_progress,
                )
                if self._read_limits.pool_enabled:
                    self._metrics["query_seconds"] += (
                        asyncio.get_running_loop().time() - query_started
                    )
                await transaction.commit()
                transaction_finished = True
        except asyncio.CancelledError:
            if pool_entry is not None:
                self._metrics["cancellations"] += 1
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
            if pool_entry is not None and connection is not None:
                await self._release_pooled(
                    connection,
                    pool_entry,
                    discard=not transaction_finished,
                    deadline=read_deadline,
                )
            else:
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
            adapter_id="postgresql",
            source_id=source_id,
            source_revision=validation.source_revision,
            sql_fingerprint=validation.analysis.sql_fingerprint,
            resource_revisions=validation.resource_revisions,
            columns=columns,
            row_count=row_count,
            content=content,
            sensitivity=sensitivity,
            column_types=arrow_schema(content)[1] if format_name == "arrow" else (),
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
        renderer: ExactArrowRenderer | ExactCsvRenderer | ExactXlsxRenderer = (
            ExactCsvRenderer(
                raw_names,
                max_rows=max_rows,
                max_columns=max_columns,
                max_bytes=max_bytes,
                max_seconds=timeout_seconds,
            )
        )
    elif format_name == "arrow":
        renderer = ExactArrowRenderer(
            raw_names,
            max_rows=max_rows,
            max_columns=max_columns,
            max_bytes=max_bytes,
            max_seconds=timeout_seconds,
            projected_types=tuple(
                getattr(getattr(attribute, "type", None), "name", None)
                for attribute in attributes
            ),
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
