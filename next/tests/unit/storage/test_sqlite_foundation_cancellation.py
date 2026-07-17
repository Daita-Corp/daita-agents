from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import sqlite3
import threading
from typing import Literal

import pytest

from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import (
    DAITA_V2_APPLICATION_ID,
    SQLiteOperationStore,
)


async def _wait_until_set(event: threading.Event) -> None:
    """Wait for a worker-thread checkpoint without blocking the event loop."""

    assert await asyncio.to_thread(event.wait, 2.0)


def _database_state(
    connect: Callable[[Path], sqlite3.Connection],
    path: Path,
) -> tuple[int, int, tuple[tuple[int, str, str], ...]]:
    connection = connect(path)
    try:
        application_id_row = connection.execute("PRAGMA application_id").fetchone()
        user_version_row = connection.execute("PRAGMA user_version").fetchone()
        migration_rows = connection.execute(
            "SELECT version, name, checksum " "FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert application_id_row is not None
        assert user_version_row is not None
        return (
            int(application_id_row[0]),
            int(user_version_row[0]),
            tuple(
                (int(version), str(name), str(checksum))
                for version, name, checksum in migration_rows
            ),
        )
    finally:
        connection.close()


async def test_cancelled_public_open_waits_for_migration_and_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    migration_blocked = threading.Event()
    release_migration = threading.Event()
    caller_returned = threading.Event()
    gate_timed_out = threading.Event()
    statements_after_return: list[str] = []
    traced_statements: list[str] = []
    created_connection_ids: list[int] = []
    closed_connection_ids: list[int] = []
    real_connect = sqlite3.connect

    def trace(statement: str) -> None:
        traced_statements.append(statement)
        if caller_returned.is_set():
            statements_after_return.append(statement)
        if statement.startswith("INSERT INTO schema_migrations"):
            migration_blocked.set()
            if not release_migration.wait(2.0):
                gate_timed_out.set()
            if caller_returned.is_set():
                statements_after_return.append(statement)

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            closed_connection_ids.append(id(self))
            super().close()

    def tracking_connect(
        database: str | bytes | Path,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: (
            Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None
        ) = "DEFERRED",
        check_same_thread: bool = True,
        factory: type[sqlite3.Connection] = TrackingConnection,
        cached_statements: int = 128,
        uri: bool = False,
    ) -> sqlite3.Connection:
        del factory
        connection = real_connect(
            database,
            timeout=timeout,
            detect_types=detect_types,
            isolation_level=isolation_level,
            check_same_thread=check_same_thread,
            factory=TrackingConnection,
            cached_statements=cached_statements,
            uri=uri,
        )
        created_connection_ids.append(id(connection))
        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(sqlite_owner.sqlite3, "connect", tracking_connect)
    opening = asyncio.create_task(SQLiteOperationStore.open(path))

    try:
        await _wait_until_set(migration_blocked)
        opening.cancel()
        await asyncio.sleep(0)

        # Cancellation cannot let the public call return while SQLite is still
        # able to mutate the database in its worker thread.
        assert opening.done() is False

        release_migration.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        caller_returned.set()
    finally:
        release_migration.set()
        if not opening.done():
            opening.cancel()
            with pytest.raises(asyncio.CancelledError):
                await opening

    assert gate_timed_out.is_set() is False
    assert created_connection_ids
    assert sorted(closed_connection_ids) == sorted(created_connection_ids)
    assert statements_after_return == []

    statements_at_return = tuple(traced_statements)
    await asyncio.sleep(0)
    assert tuple(traced_statements) == statements_at_return

    monkeypatch.setattr(sqlite_owner.sqlite3, "connect", real_connect)
    state_at_return = _database_state(real_connect, path)
    await asyncio.to_thread(lambda: None)
    assert _database_state(real_connect, path) == state_at_return

    expected_migrations = tuple(
        (migration.version, migration.name, migration.checksum)
        for migration in sqlite_owner._MIGRATIONS
    )
    assert state_at_return == (
        DAITA_V2_APPLICATION_ID,
        len(expected_migrations),
        expected_migrations,
    )

    reopened = await SQLiteOperationStore.open(path)
    assert reopened.closed is False
    await reopened.close()
    assert reopened.closed is True


async def test_cancelled_inspection_waits_for_its_sqlite_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteOperationStore.open(tmp_path / "state.db")
    inspection_blocked = threading.Event()
    release_inspection = threading.Event()
    inspection_finished = threading.Event()
    real_inspect = sqlite_owner._inspect_foundation

    def blocking_inspect(connection: sqlite3.Connection) -> object:
        inspection_blocked.set()
        assert release_inspection.wait(2.0)
        result = real_inspect(connection)
        inspection_finished.set()
        return result

    monkeypatch.setattr(sqlite_owner, "_inspect_foundation", blocking_inspect)
    inspecting = asyncio.create_task(store.inspect_foundation())

    try:
        await _wait_until_set(inspection_blocked)
        inspecting.cancel()
        await asyncio.sleep(0)
        assert inspecting.done() is False

        release_inspection.set()
        with pytest.raises(asyncio.CancelledError):
            await inspecting
    finally:
        release_inspection.set()
        if not inspecting.done():
            inspecting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await inspecting

    assert inspection_finished.is_set()
    assert store.closed is False

    monkeypatch.setattr(sqlite_owner, "_inspect_foundation", real_inspect)
    foundation = await store.inspect_foundation()
    assert foundation.application_id == DAITA_V2_APPLICATION_ID
    await store.close()


async def test_cancelled_close_waits_until_the_connection_is_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_armed = threading.Event()
    close_blocked = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()
    created_connection_ids: list[int] = []
    closed_connection_ids: list[int] = []
    real_connect = sqlite3.connect

    class BlockingCloseConnection(sqlite3.Connection):
        def close(self) -> None:
            if close_armed.is_set():
                close_blocked.set()
                assert release_close.wait(2.0)
            super().close()
            closed_connection_ids.append(id(self))
            close_finished.set()

    def tracking_connect(
        database: str | bytes | Path,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: (
            Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None
        ) = "DEFERRED",
        check_same_thread: bool = True,
        factory: type[sqlite3.Connection] = BlockingCloseConnection,
        cached_statements: int = 128,
        uri: bool = False,
    ) -> sqlite3.Connection:
        del factory
        connection = real_connect(
            database,
            timeout=timeout,
            detect_types=detect_types,
            isolation_level=isolation_level,
            check_same_thread=check_same_thread,
            factory=BlockingCloseConnection,
            cached_statements=cached_statements,
            uri=uri,
        )
        created_connection_ids.append(id(connection))
        return connection

    monkeypatch.setattr(sqlite_owner.sqlite3, "connect", tracking_connect)
    store = await SQLiteOperationStore.open(tmp_path / "state.db")
    close_armed.set()
    closing = asyncio.create_task(store.close())

    try:
        await _wait_until_set(close_blocked)
        closing.cancel()
        await asyncio.sleep(0)
        assert closing.done() is False

        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
    finally:
        release_close.set()
        if not closing.done():
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing

    assert close_finished.is_set()
    assert store.closed is True
    assert sorted(closed_connection_ids) == sorted(created_connection_ids)
