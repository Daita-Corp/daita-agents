from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ..artifacts.models import (
    ArtifactRecord,
)
from ..errors import StateCompatibilityCode, StateCompatibilityError
from ..identity import AgentIdentity
from .errors import StorageUnavailableError
from .graph_schema import ClosingSQLiteConnection
from .home_migrations import (
    CURRENT_HOME_REVISION,
    HomeMigrationJournalError,
    HomeMigrationJournalNewerError,
    create_current_database,
    inspect_home_revision,
)
from .sql import (
    SQLStateStore,
    _artifact_record_from_row,
    _validate_current_records as _validate_records,
)
from .sql_connection import SQLConnection, StateIntegrityError
from .sqlite_records import (
    EffectOutcome,
    EffectReceipt,
    EffectReceiptConflictError,
    RelationalWriteScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    effect_receipt_id,
    relational_write_authorization_fingerprint,
)
from .sqlite_schema import (
    CURRENT_SCHEMA,
    require_schema,
    table_names,
)


class SQLiteStateStore(SQLStateStore):
    """SQLite admission and connections for the shared state operations."""

    current_revision = str(CURRENT_HOME_REVISION)

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = path
        super().__init__(_SQLiteDatabase(path), clock=clock)

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        current_home_validated: bool = False,
        **_: object,
    ) -> SQLiteStateStore:
        resolved = Path(path).resolve()
        resolved_clock = clock or (lambda: datetime.now(UTC))

        def admit() -> None:
            if current_home_validated:
                if not resolved.is_file():
                    raise ValueError("validated current state database is missing")
            else:
                _initialize(resolved)

        worker = asyncio.create_task(asyncio.to_thread(admit))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return cls(resolved, clock=resolved_clock)

    async def close(self) -> None:
        async with self._decoded_catalog_snapshot_lock:
            self._decoded_catalog_snapshots.clear()
        await asyncio.to_thread(_checkpoint_wal_for_close, self.path)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path,
        timeout=30,
        factory=ClosingSQLiteConnection,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def load_current_artifact_inventory(
    path: Path, agent_id: str
) -> tuple[ArtifactRecord, ...]:
    """Read authoritative lifecycle rows from an already validated current database."""
    with _connect_read_only(path) as connection:
        return tuple(
            _artifact_record_from_row(row)
            for row in connection.execute(
                "SELECT * FROM artifacts WHERE agent_id = ? ORDER BY created_at_us, artifact_id",
                (agent_id,),
            )
        )


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        validate_current_state_database(path)
        os.chmod(path, 0o600)
        return
    with _connect(path) as connection:
        create_current_database(connection)
    os.chmod(path, 0o600)


def _checkpoint_wal_for_close(path: Path) -> None:
    connection = _connect(path)
    try:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if (
            checkpoint is None
            or int(checkpoint[0]) != 0
            or int(checkpoint[1]) != int(checkpoint[2])
        ):
            raise RuntimeError("state database WAL checkpoint did not complete")
    finally:
        connection.close()


def _damaged_state_error(
    path: Path,
    found_revision: str | None,
) -> StateCompatibilityError:
    return StateCompatibilityError(
        StateCompatibilityCode.DAMAGED,
        path,
        (
            "This agent state database is damaged or does not match its declared "
            "Daita revision. No state was changed. Reinstall the matching Daita "
            "release or restore the database through your normal recovery process."
        ),
        current_revision=str(CURRENT_HOME_REVISION),
        found_revision=found_revision,
    )


def validate_current_state_database(path: Path) -> AgentIdentity | None:
    """Validate the entire canonical database without changing it."""

    try:
        with _connect_read_only(path) as connection:
            if "agent_home_migrations" not in table_names(connection):
                raise StateCompatibilityError(
                    StateCompatibilityCode.LEGACY,
                    path,
                    "This database must be admitted by the agent-home upgrade owner.",
                    current_revision=str(CURRENT_HOME_REVISION),
                    found_revision="preproduction",
                )
            revision = inspect_home_revision(connection)
            if revision != CURRENT_HOME_REVISION:
                raise StateCompatibilityError(
                    StateCompatibilityCode.REVISION_UNSUPPORTED,
                    path,
                    "This agent-home revision is not current.",
                    current_revision=str(CURRENT_HOME_REVISION),
                    found_revision=str(revision),
                )
            require_schema(connection, CURRENT_SCHEMA)
            return _validate_current_records(connection)
    except HomeMigrationJournalNewerError as error:
        raise StateCompatibilityError(
            StateCompatibilityCode.NEWER_REVISION,
            path,
            "This local state was created by a newer Daita release. No state was changed.",
            current_revision=str(CURRENT_HOME_REVISION),
            found_revision=(
                None if error.found_revision is None else str(error.found_revision)
            ),
        ) from None
    except HomeMigrationJournalError as error:
        raise StateCompatibilityError(
            StateCompatibilityCode.REVISION_UNSUPPORTED,
            path,
            "This local state has an invalid agent-home migration history. No state was changed.",
            current_revision=str(CURRENT_HOME_REVISION),
            found_revision=(
                "invalid-journal"
                if error.found_revision is None
                else str(error.found_revision)
            ),
        ) from None
    except StateCompatibilityError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise _damaged_state_error(path, None) from None


def _storage_error(error: BaseException) -> BaseException:
    if isinstance(error, sqlite3.OperationalError) and (
        getattr(error, "sqlite_errorcode", 0) & 255
    ) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return StorageUnavailableError("SQLite storage is temporarily busy")
    return error


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro",
        timeout=30,
        uri=True,
        factory=ClosingSQLiteConnection,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA query_only = ON")
    return connection


__all__ = [
    "EffectOutcome",
    "EffectReceipt",
    "EffectReceiptConflictError",
    "RelationalWriteScope",
    "SQLiteStateStore",
    "SourcePermissionStateError",
    "SourceReadMode",
    "SourceReadScope",
    "StateCompatibilityCode",
    "StateCompatibilityError",
    "effect_receipt_id",
    "load_current_artifact_inventory",
    "relational_write_authorization_fingerprint",
    "validate_current_state_database",
]


class _SQLiteDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self, *, read_only: bool = False) -> SQLConnection:
        return _SQLiteConnection(
            _connect_read_only(self.path) if read_only else _connect(self.path)
        )

    def normalize_error(self, error: BaseException) -> BaseException:
        return _storage_error(error)


class _SQLiteConnection:
    insertion_order = "rowid"
    distinct_operator = "IS NOT"

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def execute(self, query, parameters=()):
        try:
            return self.connection.execute(query, parameters)
        except sqlite3.IntegrityError as error:
            raise StateIntegrityError(str(error)) from error

    def executemany(self, query, parameters):
        try:
            return self.connection.executemany(query, parameters)
        except sqlite3.IntegrityError as error:
            raise StateIntegrityError(str(error)) from error

    def begin(self, *, write: bool = False) -> None:
        self.connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def close(self) -> None:
        self.connection.close()

    def conflict(self, columns: str) -> str:
        return columns

    def caller_expression(self, column: str) -> str:
        return f"json_extract({column}, '$.fields.caller_principal_id')"

    def table_exists(self, table: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            is not None
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.connection.__exit__(exc_type, exc, traceback)


def _validate_current_records(connection: sqlite3.Connection) -> AgentIdentity | None:
    return _validate_records(_SQLiteConnection(connection))
