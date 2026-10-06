"""PostgreSQL state backend using the same operations as local SQLite homes.

The runtime role must be restricted to explicitly provisioned namespaces. All
settings are transaction-local, prepared statements are disabled, and an uncertain
commit is never retried. Configuration and artifact bytes remain separate owners.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from .errors import (
    StorageCommitUnknownError,
    StorageError,
    StorageOwnershipLostError,
    StorageUnavailableError,
)
from .home_migrations import CURRENT_HOME_REVISION
from .postgres_admin import postgres_schema_checksum, schema_identifier
from .postgres_schema import POSTGRES_TABLES
from .sql import SQLStateStore
from .sql_connection import StateIntegrityError


def _load_driver():
    try:
        import psycopg
        import psycopg_pool
    except ImportError as error:
        raise ImportError(
            "PostgreSQL state storage is unavailable. Run: pipx reinstall daita-agents"
        ) from error
    return psycopg, psycopg_pool


@dataclass(frozen=True)
class PostgresStateConfig:
    """Connection information is private and deliberately excluded from repr."""

    conninfo: str = field(repr=False)
    namespace_id: UUID
    schema: str = "daita_state"
    max_connections: int = 2
    timeout_seconds: int = 15

    def __post_init__(self) -> None:
        schema_identifier(self.schema)
        if not isinstance(self.namespace_id, UUID):
            raise TypeError("namespace_id must be UUID")
        if not isinstance(self.conninfo, str) or not self.conninfo:
            raise ValueError("conninfo must be nonempty text")
        for label, value, maximum in (
            ("max_connections", self.max_connections, 16),
            ("timeout_seconds", self.timeout_seconds, 120),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{label} must be between 1 and {maximum}")


class PostgresStateStore(SQLStateStore):
    """A passive state handle, scoped by a database role and a fencing epoch.

    Opening does not run recovery, claim execution ownership or initialize data.
    A host must own its writer lease before recovery/admission. Once it fences a
    previous writer, every mutation on the old handle is rejected by the database.
    """

    def __init__(
        self,
        database: _PostgresDatabase,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(database, clock=clock)
        self._postgres = database

    @classmethod
    async def open(
        cls, config: PostgresStateConfig, *, clock: Callable[[], datetime] | None = None
    ) -> PostgresStateStore:
        if not isinstance(config, PostgresStateConfig):
            raise TypeError("PostgreSQL state requires PostgresStateConfig")
        database = _PostgresDatabase(config)
        worker = asyncio.create_task(asyncio.to_thread(database.admit))
        cancelled = False
        try:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancelled = True
            worker.result()
            if cancelled:
                raise asyncio.CancelledError
            return cls(database, clock=clock)
        except BaseException as error:
            await asyncio.to_thread(database.close)
            if isinstance(error, database.driver.OperationalError):
                raise StorageUnavailableError(
                    "PostgreSQL state admission unavailable"
                ) from None
            raise

    @property
    def fencing_epoch(self) -> int:
        return self._postgres.epoch

    async def advance_fence(self) -> int:
        """CAS to a new writer generation after the host's ownership handover.

        This is fencing, not an execution lease. A caller cannot silently steal a
        newer generation: a stale handle must fail and return to its host owner.
        """
        from .sql import _run_cancellation_safe_transaction

        def advance(connection):
            row = connection.execute(
                "UPDATE namespaces SET fencing_epoch = fencing_epoch + 1 "
                "WHERE namespace_id = ? AND fencing_epoch = ? RETURNING fencing_epoch",
                (self._postgres.config.namespace_id, self._postgres.epoch),
            ).fetchone()
            if row is None:
                raise StorageOwnershipLostError("state writer generation changed")
            return int(row[0])

        epoch = await _run_cancellation_safe_transaction(self._postgres, advance)
        self._postgres.epoch = epoch
        return epoch

    async def close(self) -> None:
        worker = asyncio.create_task(asyncio.to_thread(self._postgres.close))
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        worker.result()
        await super().close()


class _PostgresDatabase:
    def __init__(self, config: PostgresStateConfig) -> None:
        self.config = config
        self.schema = schema_identifier(config.schema)
        self.driver, pool_module = _load_driver()
        from psycopg.conninfo import conninfo_to_dict

        options = conninfo_to_dict(config.conninfo)
        if options.get("sslmode") != "verify-full":
            raise ValueError("PostgreSQL state requires sslmode=verify-full")
        self.pool = pool_module.ConnectionPool(
            config.conninfo,
            min_size=0,
            max_size=config.max_connections,
            timeout=config.timeout_seconds,
            max_waiting=64,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,
                "connect_timeout": config.timeout_seconds,
                "keepalives": 1,
                "keepalives_idle": config.timeout_seconds,
                "keepalives_interval": config.timeout_seconds,
                "keepalives_count": 2,
                "tcp_user_timeout": config.timeout_seconds * 1000,
            },
            open=False,
        )
        self.epoch = 0
        self._closed = False
        self._poisoned = False
        self._condition = threading.Condition()
        self._writer_lock = threading.Lock()
        self._active = 0

    def admit(self) -> None:
        self.pool.open()
        with self.pool.connection() as connection:
            with connection.transaction():
                self._settings(connection)
                role = connection.execute(
                    "SELECT rolsuper, rolbypassrls, rolcreaterole FROM pg_roles WHERE rolname = session_user"
                ).fetchone()
                owner = connection.execute(
                    "SELECT pg_has_role(session_user, nspowner, 'MEMBER') OR has_schema_privilege(session_user, oid, 'CREATE') FROM pg_namespace WHERE nspname = %s",
                    (self.config.schema,),
                ).fetchone()
                if role != (False, False, False) or owner != (False,):
                    raise ValueError(
                        "PostgreSQL state requires a non-owner login without privileged role membership"
                    )
                unsafe = connection.execute(
                    "SELECT has_table_privilege(session_user, %s, 'INSERT,UPDATE,DELETE,TRUNCATE') "
                    "OR has_table_privilege(session_user, %s, 'INSERT,UPDATE,DELETE,TRUNCATE') "
                    "OR has_table_privilege(session_user, %s, 'INSERT,DELETE,TRUNCATE') "
                    "OR has_column_privilege(session_user, %s, 'namespace_id', 'UPDATE')",
                    tuple(
                        f"{self.schema}.{table}"
                        for table in (
                            "schema_version",
                            "namespace_roles",
                            "namespaces",
                            "namespaces",
                        )
                    ),
                ).fetchone()
                if unsafe != (False,):
                    raise ValueError(
                        "PostgreSQL runtime can modify administrative state"
                    )
                history = connection.execute(
                    f"SELECT version, checksum FROM {self.schema}.schema_version"
                ).fetchall()
                expected = [(CURRENT_HOME_REVISION, postgres_schema_checksum())]
                if history != expected:
                    raise ValueError(
                        "PostgreSQL state schema history is unsupported or changed"
                    )
                row = connection.execute(
                    f"SELECT fencing_epoch FROM {self.schema}.namespaces WHERE namespace_id = %s",
                    (self.config.namespace_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        "PostgreSQL state namespace is absent or not authorized"
                    )
                tables = connection.execute(
                    "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, pg_has_role(session_user, c.relowner, 'MEMBER') OR has_table_privilege(session_user, c.oid, 'TRUNCATE') FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = %s AND c.relkind = 'r' AND c.relname = ANY(%s)",
                    (self.config.schema, list(POSTGRES_TABLES)),
                ).fetchall()
                if len(tables) != len(POSTGRES_TABLES) or any(
                    not enabled or not forced or owner
                    for _, enabled, forced, owner in tables
                ):
                    raise ValueError(
                        "PostgreSQL state table isolation is missing or unsafe"
                    )
                self.epoch = int(row[0])

    def _settings(self, connection, *, epoch: int | None = None) -> None:
        connection.execute(
            "SELECT set_config('search_path', %s, true), set_config('daita.store_id', %s, true), set_config('daita.fencing_epoch', %s, true), set_config('statement_timeout', %s, true), set_config('lock_timeout', %s, true), set_config('idle_in_transaction_session_timeout', %s, true)",
            (
                f"{self.schema}, pg_catalog",
                str(self.config.namespace_id),
                str(self.epoch if epoch is None else epoch),
                str(self.config.timeout_seconds * 1000),
                str(self.config.timeout_seconds * 1000),
                str(self.config.timeout_seconds * 1000),
            ),
        )

    def connect(self, *, read_only: bool = False):
        with self._condition:
            if self._closed:
                raise StorageError("PostgreSQL state store is closed")
            if self._poisoned:
                raise StorageCommitUnknownError(
                    "state store has an unresolved commit; close and inspect durable state"
                )
            self._active += 1
        try:
            connection = self.pool.getconn()
        except Exception:
            self._released()
            raise StorageUnavailableError(
                "PostgreSQL state connection unavailable"
            ) from None
        return _PostgresConnection(self, connection, read_only=read_only)

    def _released(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def normalize_error(self, error: BaseException) -> BaseException:
        # Called only after rollback, never to reinterpret an unknown COMMIT.
        if isinstance(error, (StorageError, StateIntegrityError)):
            return error
        if getattr(error, "sqlstate", None) in {"40001", "40P01", "55P03", "57014"}:
            return StorageUnavailableError("PostgreSQL state transaction rolled back")
        if isinstance(error, self.driver.Error):
            return StorageError("PostgreSQL state operation failed")
        return error

    def close(self) -> None:
        with self._condition:
            self._closed = True
            # Every server operation is bounded. Never abandon a committing
            # worker merely because its caller was cancelled.
            self._condition.wait_for(lambda: self._active == 0)
        self.pool.close()


_SQL_TOKENS = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\?")


def _parameters(query: str) -> str:
    """Adapt qmark bindings, leaving quoted text and identifiers untouched."""
    return _SQL_TOKENS.sub(
        lambda match: "%s" if match[0] == "?" else match[0], query.replace("%", "%%")
    )


class _PostgresConnection:
    insertion_order = "insertion_id"
    distinct_operator = "IS DISTINCT FROM"

    def __init__(
        self, database: _PostgresDatabase, connection, *, read_only: bool
    ) -> None:
        self.database = database
        self.connection = connection
        self.read_only = read_only
        self._begun = False
        self._closed = False
        self._cursors: list[Any] = []
        self._writer_locked = False

    def begin(self, *, write: bool = False) -> None:
        if self._begun:
            return
        if self.read_only and write:
            raise StorageError("read-only state transaction cannot mutate")
        if not self.read_only:
            # Keep acknowledgement handling ahead of the next queued mutation,
            # including when the server released its lock before we lost COMMIT.
            self._writer_locked = self.database._writer_lock.acquire(
                timeout=self.database.config.timeout_seconds
            )
            if not self._writer_locked:
                raise StorageUnavailableError("PostgreSQL state writer is busy")
        if self.database._poisoned:
            raise StorageCommitUnknownError(
                "state store has an unresolved commit; close and inspect durable state"
            )
        epoch = self.database.epoch
        self.connection.execute(
            "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"
            if self.read_only
            else "BEGIN"
        )
        self._begun = True
        self.database._settings(self.connection, epoch=epoch)
        if not self.read_only:
            row = self.connection.execute(
                f"SELECT fencing_epoch FROM {self.database.schema}.namespaces WHERE namespace_id = %s FOR UPDATE",
                (self.database.config.namespace_id,),
            ).fetchone()
            if row is None or row[0] != epoch:
                raise StorageOwnershipLostError("state writer generation changed")

    def execute(self, query, parameters=()):
        self.begin()
        try:
            cursor = self.connection.execute(
                _parameters(query),
                tuple(int(v) if isinstance(v, bool) else v for v in parameters),
                prepare=False,
            )
            self._cursors.append(cursor)
            return cursor
        except self.database.driver.IntegrityError as error:
            raise StateIntegrityError(
                "state relational constraint rejected the operation"
            ) from error

    def executemany(self, query, parameters):
        # Avoid Psycopg's executemany pipeline: transaction poolers do not all
        # support pipeline mode. These bounded batches remain one transaction.
        cursor = None
        for values in parameters:
            cursor = self.execute(query, values)
        if cursor is None:
            cursor = self.execute("SELECT 1 WHERE false")
        return cursor

    def commit(self) -> None:
        if not self._begun:
            return
        try:
            self.connection.execute("COMMIT", prepare=False)
        except (
            self.database.driver.OperationalError,
            self.database.driver.InterfaceError,
        ) as error:
            if self.read_only or getattr(error, "sqlstate", None) in {
                "40001",
                "40P01",
                "55P03",
                "57014",
            }:
                raise StorageUnavailableError(
                    "PostgreSQL state transaction did not commit a mutation"
                ) from None
            self.database._poisoned = True
            raise StorageCommitUnknownError(
                "PostgreSQL commit acknowledgement was lost"
            ) from None
        self._begun = False

    def rollback(self) -> None:
        if self._begun:
            try:
                self.connection.execute("ROLLBACK", prepare=False)
            except self.database.driver.Error:
                self.connection.close()
            self._begun = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.rollback()
            for cursor in self._cursors:
                cursor.close()
            self.database.pool.putconn(self.connection)
        finally:
            if self._writer_locked:
                self._writer_locked = False
                self.database._writer_lock.release()
            self.database._released()

    def conflict(self, columns: str) -> str:
        return "namespace_id, " + columns

    def caller_expression(self, column: str) -> str:
        return f"({column}::jsonb #>> '{{fields,caller_principal_id}}')"

    def table_exists(self, table: str) -> bool:
        return self.execute(
            "SELECT to_regclass(?) IS NOT NULL", (f"{self.database.schema}.{table}",)
        ).fetchone()[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
                if exc is not None:
                    normalized = self.database.normalize_error(exc)
                    if normalized is not exc:
                        raise normalized from None
        finally:
            self.close()
