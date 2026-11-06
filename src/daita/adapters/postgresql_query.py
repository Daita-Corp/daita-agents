"""Execute policy-checked, bounded read queries against PostgreSQL."""

from __future__ import annotations

import asyncio
import base64
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
from typing import Any, TypeVar, cast
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
from ..domains.data.capabilities import PostgreSQLReadResult, refused_read_details
from ..domains.data.export_capabilities import (
    ExactTabularExportResult,
    ExactTabularProgress,
    resolved_exact_export_sensitivity,
)
from ..domains.data.results import project_result_rows
from ..domains.data.sql import CheckedRead, denied_relations, resource_id_in_scope
from ..errors import DaitaError, ErrorRetryability
from ..observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely
from ..security import SecretProvider, SecretReference, default_secret_provider
from ..storage.sqlite_records import (
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
)
from .models import SourceRegistration
from .postgresql import (
    COLUMNS_SQL,
    FOREIGN_KEYS_FROM_SQL,
    INDEXES_SQL,
    LISTED_RELATION_SQL,
    READABLE_COLUMN_SQL,
    READABLE_FOREIGN_KEY_SQL,
    READABLE_RELATION_SQL,
    ROW_ESTIMATE_SQL,
    PostgreSQLSourceError,
    _close_postgresql_connection,
    _configuration_schemas,
    _connect,
    _rollback_postgresql_transaction,
    columns_from_rows,
)
from .protocols import (
    DESCRIBE_MAX_COLUMNS,
    DESCRIBE_MAX_LISTED_TABLES,
    SourceReadScopeStore,
    column_summary,
)

_T = TypeVar("_T")
_MAX_QUERY_ROWS = 100_000
_MAX_QUERY_BYTES = 64 * 1_024 * 1_024
_MAX_RESULT_COLUMNS = 512
_MAX_VALUE_DEPTH = 32
_BOUNDED_RESULT_MARKER = "/* daita:postgresql.bounded_result */"
_FUNCTION_POLICY_LOOKUP = """/* daita:postgresql.function_policy */
SELECT p.proname, n.nspname, p.provolatile,
  p.pronargs >= 1
    AND p.pronargs - p.pronargdefaults <= 1
    AND argument.typtype IN ('c', 'd', 'p') AS takes_row
FROM pg_catalog.pg_proc AS p
JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
LEFT JOIN pg_catalog.pg_type AS argument ON argument.oid = p.proargtypes[0]
WHERE p.proname = ANY($1::text[])
  AND n.nspname = ANY(pg_catalog.current_schemas(true))"""
_UNQUALIFIED_RELATION_LOOKUP = f"""/* daita:postgresql.unqualified_relation */
SELECT n.nspname, c.relname
FROM pg_catalog.pg_class AS c
JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
WHERE c.relname = ANY($1::text[])
  AND n.nspname = ANY($2::text[])
  AND {READABLE_RELATION_SQL}
ORDER BY 1, 2
LIMIT 32"""
_LOOKUP_SECONDS = 5.0
_LIVE_SEARCH = f"""/* daita:postgresql.live_search */
WITH words AS (
  SELECT * FROM unnest($2::text[], $3::text[]) AS w(word, pattern)
), matched AS (
  SELECT c.oid, n.nspname, c.relname, c.relkind,
    {ROW_ESTIMATE_SQL} AS estimate,
    (SELECT count(*) FROM words WHERE lower(c.relname) = word) AS exact,
    (SELECT count(*) FROM words WHERE c.relname ILIKE pattern) AS named,
    (SELECT count(*) FROM words WHERE EXISTS (
       SELECT 1 FROM pg_attribute AS a
       WHERE a.attrelid = c.oid AND {READABLE_COLUMN_SQL}
         AND a.attname ILIKE pattern)) AS columned
  FROM pg_class AS c
  JOIN pg_namespace AS n ON n.oid = c.relnamespace
  WHERE n.nspname = ANY($1::text[]) AND {READABLE_RELATION_SQL} AND {LISTED_RELATION_SQL}
)
SELECT m.nspname, m.relname, m.relkind, m.estimate,
  (SELECT string_agg(a.attname || ' ' || format_type(a.atttypid, a.atttypmod), ', ')
   FROM (SELECT * FROM pg_attribute AS a
         WHERE a.attrelid = m.oid AND {READABLE_COLUMN_SQL}
         ORDER BY a.attnum LIMIT 12) AS a) AS columns,
  (SELECT count(*) FROM pg_attribute AS a
   WHERE a.attrelid = m.oid AND {READABLE_COLUMN_SQL}) AS column_count,
  obj_description(m.oid, 'pg_class') AS comment,
  (SELECT count(*) FROM pg_constraint AS fk
   JOIN pg_class AS fk_src ON fk_src.oid = fk.conrelid
   JOIN pg_namespace AS fk_ns ON fk_ns.oid = fk_src.relnamespace
   WHERE fk.contype = 'f' AND fk.conparentid = 0 AND fk.confrelid = m.oid
     AND fk_ns.nspname = ANY($1::text[])) AS inbound_foreign_keys,
  (SELECT greatest(s.last_analyze, s.last_autoanalyze)
   FROM pg_stat_all_tables AS s WHERE s.relid = m.oid) AS last_analyzed
FROM matched AS m
WHERE m.named + m.columned > 0
ORDER BY m.exact DESC, m.named DESC, m.columned DESC, m.estimate DESC NULLS LAST,
         m.nspname, m.relname
LIMIT $4"""
_LIVE_RELATIONS = f"""/* daita:postgresql.live_relations */
SELECT c.oid, n.nspname, c.relname, c.relkind, obj_description(c.oid, 'pg_class'),
  {ROW_ESTIMATE_SQL} AS estimate,
  (SELECT count(*) FROM pg_inherits WHERE inhparent = c.oid) AS children
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE n.nspname = ANY($1::text[]) AND c.relname = ANY($2::text[]) AND {READABLE_RELATION_SQL}
ORDER BY c.relname = $3 DESC, n.nspname
LIMIT 8"""
_LIVE_SCHEMA = f"""/* daita:postgresql.live_schema */
SELECT n.nspname, c.relname, c.relkind, {ROW_ESTIMATE_SQL} AS estimate,
  count(*) OVER () AS total
FROM pg_class AS c
JOIN pg_namespace AS n ON n.oid = c.relnamespace
WHERE n.nspname = $2 AND {READABLE_RELATION_SQL} AND {LISTED_RELATION_SQL}
ORDER BY c.relname
LIMIT $3"""
_LIVE_RELATIONSHIPS = f"""/* daita:postgresql.live_relationships */
SELECT src_ns.nspname || '.' || src.relname, dst_ns.nspname || '.' || dst.relname,
  array_agg(src_att.attname::text ORDER BY keys.ordinality),
  array_agg(dst_att.attname::text ORDER BY keys.ordinality),
  EXISTS (
    SELECT 1 FROM pg_index AS ix
    WHERE ix.indrelid = con.conrelid AND ix.indisunique
      AND ix.indpred IS NULL AND ix.indexprs IS NULL
      AND ix.indkey::int2[] <@ con.conkey
  ) AS referencing_unique
{FOREIGN_KEYS_FROM_SQL}
  AND con.conparentid = 0
  AND (con.conrelid = $1 OR con.confrelid = $1)
  AND src_ns.nspname = ANY($2::text[]) AND dst_ns.nspname = ANY($2::text[])
GROUP BY con.oid, con.conrelid, con.conkey, src_ns.nspname, src.relname,
  dst_ns.nspname, dst.relname
{READABLE_FOREIGN_KEY_SQL}
ORDER BY 1, 2
LIMIT 32"""
# Volatile in pg_catalog, yet they only read a clock or a random source.
_VOLATILE_FUNCTION_EXCEPTIONS = frozenset(
    {"random", "clock_timestamp", "timeofday", "gen_random_uuid", "timezone"}
)
_MAX_SERVER_TEXT = 1_024
# Undefined table, column and object, and an invalid schema name.
_UNDEFINED_OBJECT_STATES = frozenset({"42P01", "42703", "42704", "3F000"})
_PERMANENT_QUERY_ERRORS = frozenset(
    {
        "source_not_available",
        "source_adapter_mismatch",
        "source_permission_state_invalid",
        "function_not_allowed",
        "unqualified_relation",
        "resource_read_not_allowed",
        "not_permitted",
        "query_invalid",
        "query_data_error",
        "read_only_violation",
        "query_result_too_wide",
        "query_result_invalid",
        "query_value_unsupported",
    }
)


class PostgreSQLQueryError(DaitaError):
    """Normalized query-boundary failure without connector error leakage."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
        *,
        undefined_object: bool = False,
    ) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("query error code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("query error message must be a non-empty string")
        self.code = code
        self.details = dict(details or {})
        # The source says a table or column is missing; the catalog may be stale.
        self.undefined_object = undefined_object
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
    query_seconds: float
    cleanup_seconds: float


@dataclass(eq=False, slots=True)
class _ReadPool:
    registration: SourceRegistration
    read_scope: SourceReadScope | None
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
    """Run one bounded read that already passed policy; the source resolves names."""

    def __init__(
        self,
        sources: SourceReadScopeStore,
        secret_provider: SecretProvider | None = None,
        *,
        statement_timeout_seconds: float = 5.0,
        cleanup_timeout_seconds: float = 1.0,
        read_limits: PostgreSQLReadLimits | None = None,
        source_capacity: int = 2,
        deadline_reader: Callable[[], float | None] | None = None,
        observer: AgentObserver | None = None,
    ) -> None:
        if not isinstance(sources, SourceReadScopeStore):
            raise TypeError("sources must implement SourceReadScopeStore")
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
                    "query_seconds",
                    "cleanup_seconds",
                )
            }
        )
        self._sources = sources
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
        """One wall-clock limit around everything a read does, pooled or not."""

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
        read_scope = await self._sources.load_source_read_scope(
            registration.agent_id, registration.id
        )
        return (
            current == registration
            and registration.active
            and read_scope == entry.read_scope
            and await self._password(registration) == entry.password
        )

    async def _connection(
        self, registration: SourceRegistration, read_scope: SourceReadScope | None
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
                                or cached.read_scope != read_scope
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
                                    read_scope,
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
        read: CheckedRead,
        max_rows: int,
        max_bytes: int,
        timeout_seconds: float | None = None,
    ) -> PostgreSQLReadResult:
        statement_timeout = self._statement_timeout_seconds
        if timeout_seconds is not None:
            if not 0 < float(timeout_seconds) <= 60:
                raise ValueError("timeout_seconds must be from 0 through 60")
            statement_timeout = min(statement_timeout, float(timeout_seconds))
        if (
            not isinstance(max_rows, int)
            or isinstance(max_rows, bool)
            or not 1 <= max_rows <= _MAX_QUERY_ROWS
        ):
            raise ValueError("max_rows must be from one through 100000")
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or not 2 <= max_bytes <= _MAX_QUERY_BYTES
        ):
            raise ValueError("max_bytes must be from two through 67108864")
        registration, read_scope = await self._admit(agent_id, source_id, read)

        connection = None
        pool_entry = None
        read_deadline = None
        transaction = None
        transaction_finished = False
        failure: PostgreSQLQueryError | None = None
        columns: tuple[str, ...] = ()
        rows: tuple[Mapping[str, object], ...] = ()
        try:
            async with self._operation(statement_timeout) as read_deadline:
                connection, pool_entry = await self._connection(
                    registration, read_scope
                )
                transaction = connection.transaction(
                    isolation="repeatable_read",
                    readonly=True,
                )
                await transaction.start()
                await _prepare_read_transaction(
                    connection, registration, read, statement_timeout
                )
                query_started = asyncio.get_running_loop().time()
                columns, rows, value_limited = await _execute_user_query(
                    connection,
                    read.canonical_sql,
                    read.parameters,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    timeout_seconds=statement_timeout,
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
            failure = PostgreSQLQueryError(error.code, str(error))
        except Exception as error:
            # Driver and server errors can carry bound values, DSNs or credentials.
            # Only their category crosses, after the original exception context
            # has ended, so diagnostics retain nothing else.
            failure = _classified_failure(error, read, statement_timeout)
        finally:
            await self._finish_read(
                connection,
                pool_entry,
                transaction,
                transaction_finished,
                read_deadline,
            )
        if failure is not None:
            raise failure

        projection = await asyncio.to_thread(
            project_result_rows, rows, max_rows=max_rows, max_bytes=max_bytes
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
            canonical_sql=read.canonical_sql,
            sql_fingerprint=read.sql_fingerprint,
            resource_ids=read.resource_ids,
            columns=columns,
            projection=projection,
            resource_names=read.resource_names,
        )

    async def _admit(
        self, agent_id: str, source_id: str, read: CheckedRead
    ) -> tuple[SourceRegistration, SourceReadScope | None]:
        """Recheck current permission, since a read can queue after it was prepared.

        Also retires a pool the current registration no longer matches.
        """

        registration, read_scope = await self._current_source(agent_id, source_id)
        denied = denied_relations(read, read_scope)
        if denied is None or denied:
            raise PostgreSQLQueryError(
                "resource_read_not_allowed",
                f"{', '.join(denied or ('This source',))} is not available for "
                "reading in this run.",
            )
        return registration, read_scope

    async def _current_source(
        self, agent_id: str, source_id: str
    ) -> tuple[SourceRegistration, SourceReadScope | None]:
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
        try:
            read_scope = await self._sources.load_source_read_scope(agent_id, source_id)
        except SourcePermissionStateError:
            raise PostgreSQLQueryError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from None
        if self._read_limits.pool_enabled:
            cached = self._pools.get((agent_id, source_id))
            if cached is not None and (
                cached.registration != registration or cached.read_scope != read_scope
            ):
                await self.retire_source(agent_id, source_id)
        return registration, read_scope

    async def _lookup(
        self,
        agent_id: str,
        source_id: str,
        work: Callable[[Any, tuple[str, ...]], Awaitable[_T]],
    ) -> _T:
        """Run code-owned catalog queries in one bounded read-only transaction."""

        registration, read_scope = await self._current_source(agent_id, source_id)
        if read_scope is None or read_scope.mode is SourceReadMode.NONE:
            raise PostgreSQLQueryError(
                "resource_read_not_allowed",
                "This source is not available for reading in this run.",
            )
        connection = None
        pool_entry = None
        read_deadline = None
        transaction = None
        finished = False
        try:
            async with self._operation(_LOOKUP_SECONDS) as read_deadline:
                connection, pool_entry = await self._connection(
                    registration, read_scope
                )
                transaction = connection.transaction(
                    isolation="repeatable_read", readonly=True
                )
                await transaction.start()
                await configure_transaction(
                    connection,
                    schemas=("pg_catalog",),
                    statement_timeout_seconds=_LOOKUP_SECONDS,
                )
                result = await work(
                    connection, _configuration_schemas(registration.configuration)
                )
                await transaction.commit()
                finished = True
                return result
        except (asyncio.CancelledError, ImportError, PostgreSQLQueryError):
            raise
        except Exception:
            # Driver and server text can carry DSNs; none of it crosses.
            raise PostgreSQLQueryError(
                "source_unreachable", "The PostgreSQL source could not be inspected."
            ) from None
        finally:
            await self._finish_read(
                connection, pool_entry, transaction, finished, read_deadline
            )

    async def search_names(
        self, *, agent_id: str, source_id: str, words: tuple[str, ...], limit: int
    ) -> tuple[Mapping[str, object], ...]:
        """Match words against relation and column names in the configured schemas."""

        async def work(
            connection: Any, schemas: tuple[str, ...]
        ) -> tuple[Mapping[str, object], ...]:
            rows = await connection.fetch(
                _LIVE_SEARCH,
                list(schemas),
                list(words),
                [f"%{_like_literal(word)}%" for word in words],
                limit,
            )
            return tuple(
                {
                    "name": f"{row[0]}.{row[1]}",
                    "kind": _relation_kind(row[2]),
                    "columns": column_summary(row[4], int(row[5])),
                    "row_estimate": _row_estimate(row[3]),
                    **({"comment": row[6]} if row[6] else {}),
                    "inbound_foreign_keys": int(row[7]),
                    **({"last_analyzed": row[8].isoformat()} if row[8] else {}),
                }
                for row in rows
            )

        return await self._lookup(agent_id, source_id, work)

    async def describe_name(
        self, *, agent_id: str, source_id: str, name: str
    ) -> Mapping[str, object] | None:
        """Describe one relation or one configured schema live; None when absent."""

        async def work(
            connection: Any, schemas: tuple[str, ...]
        ) -> Mapping[str, object] | None:
            parts = _name_parts(name)
            bare = parts[-1]
            searched = (
                tuple(schema for schema in schemas if schema in {parts[0]})
                or tuple(s for s in schemas if s == parts[0].lower())
                if len(parts) == 2
                else schemas
            )
            rows = (
                await connection.fetch(
                    _LIVE_RELATIONS,
                    list(searched),
                    list({bare, bare.lower()}),
                    bare,
                )
                if len(parts) <= 2 and searched
                else ()
            )
            if rows:
                exact = [row for row in rows if row[2] == bare] or rows
                if len({row[1] for row in exact}) > 1:
                    return {
                        "name": name,
                        "kind": "ambiguous",
                        "tables": [
                            {
                                "name": f"{row[1]}.{row[2]}",
                                "kind": _relation_kind(row[3]),
                                "row_estimate": _row_estimate(row[5]),
                            }
                            for row in exact
                        ],
                    }
                return await _describe_relation(connection, exact[0], schemas)
            schema = next(
                (item for item in schemas if item in {name, name.lower()}), None
            )
            if schema is None:
                return None
            listed = await connection.fetch(
                _LIVE_SCHEMA, list(schemas), schema, DESCRIBE_MAX_LISTED_TABLES
            )
            return {
                "name": schema,
                "kind": "schema",
                "tables": [
                    {
                        "name": f"{row[0]}.{row[1]}",
                        "kind": _relation_kind(row[2]),
                        "row_estimate": _row_estimate(row[3]),
                    }
                    for row in listed
                ],
                "total_tables": int(listed[0][4]) if listed else 0,
            }

        return await self._lookup(agent_id, source_id, work)

    async def _finish_read(
        self,
        connection: Any,
        pool_entry: _ReadPool | None,
        transaction: Any,
        transaction_finished: bool,
        read_deadline: float | None,
    ) -> None:
        if pool_entry is not None and connection is not None:
            await self._release_pooled(
                connection,
                pool_entry,
                discard=not transaction_finished,
                deadline=read_deadline,
            )
            return
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
        """Run one fresh read through the selected fixed exact renderer."""

        registration, read_scope = await self._admit(agent_id, source_id, read)

        connection = None
        pool_entry = None
        read_deadline = None
        transaction = None
        transaction_finished = False
        failure: PostgreSQLQueryError | None = None
        content = b""
        columns: tuple[str, ...] = ()
        row_count = 0
        query_timeout = timeout_seconds
        completed = {"rows": 0, "columns": 0, "bytes": 0}
        sensitivity = resolved_exact_export_sensitivity(
            (registration.presentation_sensitivity.value,)
        )
        xlsx_provenance = (
            ExactXlsxProvenance(
                source_id=source_id,
                resource_ids=read.resource_ids,
                sql_fingerprint=read.sql_fingerprint,
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
                    registration, read_scope
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
                await _prepare_read_transaction(
                    connection, registration, read, query_timeout
                )
                async with asyncio.timeout(timeout_seconds):
                    query_started = asyncio.get_running_loop().time()
                content, columns, row_count = await _execute_exact_tabular_query(
                    connection,
                    read.canonical_sql,
                    read.parameters,
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
            failure = PostgreSQLQueryError(error.code, str(error))
        except Exception as error:
            failure = _classified_failure(error, read, query_timeout)
        finally:
            await self._finish_read(
                connection,
                pool_entry,
                transaction,
                transaction_finished,
                read_deadline,
            )
        if failure is not None:
            raise failure
        return ExactTabularExportResult(
            format=format_name,
            adapter_id="postgresql",
            source_id=source_id,
            sql_fingerprint=read.sql_fingerprint,
            resource_ids=read.resource_ids,
            columns=columns,
            row_count=row_count,
            content=content,
            sensitivity=sensitivity,
            column_types=arrow_schema(content)[1] if format_name == "arrow" else (),
        )


def _like_literal(word: str) -> str:
    return word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _relation_kind(relkind: object) -> str:
    return "view" if relkind in (b"v", b"m", "v", "m") else "table"


def _row_estimate(value: object) -> int | None:
    # PostgreSQL reports 0 or -1 until a table has been analyzed: unknown, not empty.
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _name_parts(name: str) -> tuple[str, ...]:
    """Split a dotted name: quoted parts are exact, others keep their spelling."""

    return tuple(
        quoted.replace('""', '"') or bare
        for quoted, bare in re.findall(r'"((?:[^"]|"")+)"|([^.]+)', name)
    )


async def _describe_relation(
    connection: Any, row: Any, schemas: tuple[str, ...]
) -> Mapping[str, object]:
    oid, schema, relation, relkind, comment, estimate, children = row
    column_rows = await connection.fetch(
        COLUMNS_SQL, schema, relation, DESCRIBE_MAX_COLUMNS + 1
    )
    index_rows = await connection.fetch(INDEXES_SQL, schema, relation, 64)
    foreign_keys = await connection.fetch(_LIVE_RELATIONSHIPS, oid, list(schemas))
    columns = columns_from_rows(column_rows, DESCRIBE_MAX_COLUMNS)
    enums = {item["column_name"]: item["enum_values"] for item in column_rows}
    primary_key = [
        column.name
        for column in sorted(
            (item for item in columns if item.primary_key_ordinal),
            key=lambda item: cast(int, item.primary_key_ordinal),
        )
    ]
    return {
        "name": f"{schema}.{relation}",
        "kind": _relation_kind(relkind),
        "comment": comment,
        "row_estimate": _row_estimate(estimate),
        "partitions": int(children),
        "columns": [
            {
                "name": column.name,
                "type": column.native_type,
                "nullable": column.nullable,
                **({"comment": column.comment} if column.comment else {}),
                **(
                    {"enum_values": list(enums[column.name])}
                    if enums.get(column.name)
                    else {}
                ),
            }
            for column in columns
        ],
        "columns_truncated": len(column_rows) > DESCRIBE_MAX_COLUMNS,
        "primary_key": primary_key,
        "unique_keys": [
            list(item["columns"])
            for item in index_rows
            if item["is_unique"]
            and item["predicate"] is None
            and item["columns"]
            and list(item["columns"]) != primary_key
        ],
        "relationships": [
            {
                "from": item[0],
                "to": item[1],
                "columns": [list(pair) for pair in zip(item[2], item[3], strict=True)],
                "cardinality": "one_to_one" if item[4] else "many_to_one",
            }
            for item in foreign_keys
        ],
    }


async def configure_transaction(
    connection: Any,
    *,
    schemas: tuple[str, ...],
    statement_timeout_seconds: float,
    lock_timeout_seconds: float | None = None,
) -> None:
    """Bound one transaction and pin how the server reads names and strings."""

    statement_milliseconds = f"{max(1, int(statement_timeout_seconds * 1_000))}ms"
    settings = [("statement_timeout", statement_milliseconds)]
    if lock_timeout_seconds is not None:
        settings += [
            ("lock_timeout", f"{max(1, int(lock_timeout_seconds * 1_000))}ms"),
            ("idle_in_transaction_session_timeout", statement_milliseconds),
        ]
    settings += [
        # pg_catalog stays implicitly first, so a configured schema cannot shadow a
        # built-in operator or a system catalog.
        ("search_path", ", ".join(_postgresql_identifier(item) for item in schemas)),
        # The parser reads a plain string's backslash as literal; so must the server.
        ("standard_conforming_strings", "on"),
    ]
    for name, value in settings:
        await connection.execute(f"SELECT set_config('{name}', $1, true)", value)


async def _prepare_read_transaction(
    connection: Any,
    registration: SourceRegistration,
    read: CheckedRead,
    timeout_seconds: float,
) -> None:
    """Bound the transaction, pin name resolution, then apply the function rule."""

    await configure_transaction(
        connection,
        schemas=_configuration_schemas(registration.configuration),
        statement_timeout_seconds=timeout_seconds,
    )
    if read.unqualified_relations:
        await refuse_unqualified(
            connection,
            registration,
            read.unqualified_relations,
            read.allowed_resource_ids,
        )
    if read.function_lookups or read.written_calls or read.attribute_lookups:
        await check_functions(
            connection,
            read.function_lookups,
            read.attribute_lookups,
            read.written_calls,
        )


async def refuse_unqualified(
    connection: Any,
    registration: SourceRegistration,
    relations: tuple[str, ...],
    allowed_resource_ids: tuple[str, ...] | None,
) -> None:
    """Several schemas could hold the name, so list where it exists and refuse.

    PostgreSQL would silently use the first match. The list keeps to this run's
    scope, so out-of-scope names cannot leak.
    """

    schemas = _configuration_schemas(registration.configuration)
    rows = await connection.fetch(
        _UNQUALIFIED_RELATION_LOOKUP, list(relations), list(schemas)
    )
    allowed = None if allowed_resource_ids is None else frozenset(allowed_resource_ids)
    candidates = tuple(
        name
        for row in rows
        if resource_id_in_scope(registration.id, name := f"{row[0]}.{row[1]}", allowed)
    )
    raise PostgreSQLQueryError(
        "unqualified_relation",
        f"{', '.join(relations)} is unqualified and this source has several schemas; "
        + (
            f"qualify it: {', '.join(candidates)}."
            if candidates
            else f"it is in none of {', '.join(schemas)} that this run can read."
        ),
        refused_read_details({"candidates": candidates, "schemas": schemas}),
    )


async def check_functions(
    connection: Any,
    called: tuple[str, ...],
    attributes: tuple[str, ...] = (),
    written: tuple[str, ...] = (),
) -> bool:
    """Let the server say what each name resolves to on the pinned path.

    Returns whether any called name can give a different result on another call.
    """

    bare = {name: name.removeprefix("PG_CATALOG.").lower() for name in called}
    rows = await connection.fetch(
        _FUNCTION_POLICY_LOOKUP, sorted({*bare.values(), *attributes, *written})
    )
    builtin: dict[str, list[bytes | str]] = {}
    shadowed: set[str] = set()
    row_functions: set[str] = set()
    calls = {*bare.values(), *written}
    varying = False
    for row in rows:
        name, schema, volatility, takes_row = row[0], row[1], row[2], row[3]
        volatile = _is_volatile(name, volatility)
        varying = varying or (name in calls and volatility not in (b"i", "i"))
        if schema == "pg_catalog":
            builtin.setdefault(name, []).append(volatility)
        else:
            shadowed.add(name)
        # x.name is name(x) unless x has that column. A real column of the same
        # name is refused too: the parser cannot tell them apart.
        if takes_row and (schema != "pg_catalog" or volatile):
            row_functions.add(name)
    refused = tuple(
        sorted(
            {
                *(
                    bare_name
                    for name, bare_name in bare.items()
                    if bare_name not in builtin
                    or (not name.startswith("PG_CATALOG.") and bare_name in shadowed)
                    or any(_is_volatile(bare_name, item) for item in builtin[bare_name])
                ),
                *(name for name in attributes if name in row_functions),
                # What the parser renamed: a word that names no function is left
                # to the server, like any other name.
                *(
                    name
                    for name in written
                    if name in shadowed
                    or any(_is_volatile(name, item) for item in builtin.get(name, ()))
                ),
            }
        )
    )
    if refused:
        raise PostgreSQLQueryError(
            "function_not_allowed",
            f"Reads cannot call {', '.join(refused[:8])}: only functions in "
            "pg_catalog that do not change or reach outside the database are "
            "allowed.",
            refused_read_details({"functions": refused[:8]}),
        )
    return varying


def _is_volatile(name: str, volatility: bytes | str) -> bool:
    return volatility in (b"v", "v") and name not in _VOLATILE_FUNCTION_EXCEPTIONS


def _classified_failure(
    error: Exception, read: CheckedRead, timeout_seconds: float
) -> PostgreSQLQueryError:
    """Sort one driver or server failure by SQLSTATE; only safe text crosses."""

    sqlstate = getattr(error, "sqlstate", None)
    state = sqlstate if isinstance(sqlstate, str) else ""
    details: dict[str, object] = {"sqlstate": state} if state else {}
    if isinstance(error, TimeoutError) or state == "57014":
        return PostgreSQLQueryError(
            "postgresql_query_timeout",
            "The bounded PostgreSQL read exceeded its deadline of "
            f"{timeout_seconds:g} seconds.",
            details,
        )
    if type(error).__name__ == "UnsupportedClientFeatureError":
        return PostgreSQLQueryError(
            "query_value_unsupported",
            "A selected value has a type this read cannot decode; select its "
            "columns or cast it to text.",
        )
    if state == "42501":
        return PostgreSQLQueryError(
            "not_permitted", _server_text(error, "message"), details
        )
    if state.startswith("42"):
        for name in ("hint", "position"):
            value = _server_text(error, name)
            if value:
                details[name] = value
        details["canonical_sql"] = read.canonical_sql
        return PostgreSQLQueryError(
            "query_invalid",
            _server_text(error, "message"),
            refused_read_details(details),
            undefined_object=state in _UNDEFINED_OBJECT_STATES,
        )
    if state == "25006":
        return PostgreSQLQueryError(
            "read_only_violation", "A read cannot modify data.", details
        )
    if state.startswith("22"):
        return PostgreSQLQueryError(
            "query_data_error",
            f"PostgreSQL could not process a value in the query (SQLSTATE {state}).",
            details,
        )
    if state.startswith("08"):
        return PostgreSQLQueryError(
            "postgresql_connection_failed",
            "The connection to PostgreSQL failed during the read.",
            details,
        )
    return PostgreSQLQueryError(
        "postgresql_query_failed",
        "PostgreSQL could not complete the bounded read"
        + (f" (SQLSTATE {state})." if state else "."),
        details,
        undefined_object=state in _UNDEFINED_OBJECT_STATES,
    )


def _server_text(error: Exception, name: str) -> str:
    value = getattr(error, name, None)
    return "" if value is None else str(value)[:_MAX_SERVER_TEXT]


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
        # asyncpg buffers a whole prefetch batch before the byte tally sees a row.
        # shortcut: one batch can still hold 256 rows past the budget, upgrade to size-adaptive cursor.fetch(n) if wide rows prove a problem
        prefetch=min(256, max_rows + 1),
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
        f"FROM (\n{sql}\n) AS {row_alias}{alias_clause} LIMIT {max_rows + 1}"
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
        f"SELECT {', '.join(selected)} FROM (\n{sql}\n) AS {row_alias}{alias_clause} "
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
    if value_type.__name__ == "Range" and value_type.__module__.startswith("asyncpg"):
        range_value = cast(Any, value)
        return {
            "lower": _json_value(range_value.lower, _depth=_depth + 1),
            "upper": _json_value(range_value.upper, _depth=_depth + 1),
            "lower_inclusive": bool(range_value.lower_inc),
            "upper_inclusive": bool(range_value.upper_inc),
            "empty": bool(range_value.isempty),
        }
    # asyncpg.Record has the mapping API without registering as a Mapping.
    if value_type.__name__ == "Record" and value_type.__module__.startswith("asyncpg"):
        return {
            str(key): _json_value(item, _depth=_depth + 1)
            for key, item in cast(Any, value).items()
        }
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
