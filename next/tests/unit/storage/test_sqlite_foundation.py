from __future__ import annotations

import asyncio
import inspect
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import (
    DAITA_V2_APPLICATION_ID,
    SQLiteCompatibilityError,
    SQLiteMigrationError,
    SQLiteOperationStore,
)

MIGRATION_ONE = sqlite_owner._SQLiteMigration(
    version=1,
    name="create_alpha",
    statements=("CREATE TABLE alpha (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",),
)
MIGRATION_TWO = sqlite_owner._SQLiteMigration(
    version=2,
    name="create_beta",
    statements=(
        "CREATE TABLE beta (id INTEGER PRIMARY KEY, alpha_id INTEGER NOT NULL "
        "REFERENCES alpha(id))",
    ),
)
MIGRATIONS = (MIGRATION_ONE, MIGRATION_TWO)


async def _open(
    path: Path,
    *,
    migrations: tuple[sqlite_owner._SQLiteMigration, ...] = MIGRATIONS,
    busy_timeout_ms: int = 5_000,
    backup_path: Path | None = None,
) -> SQLiteOperationStore:
    """Exercise arbitrary histories without exposing SQL in the public API."""

    return await sqlite_owner._open_with_migrations(
        path,
        migrations=migrations,
        busy_timeout_ms=busy_timeout_ms,
        backup_path=backup_path,
    )


def _execute(path: Path, statement: str, parameters: tuple[object, ...] = ()) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


def _scalar(path: Path, statement: str) -> object:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(statement).fetchone()
        assert row is not None
        return row[0]
    finally:
        connection.close()


def _migration_rows(path: Path) -> list[tuple[int, str, str]]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT version, name, checksum " "FROM schema_migrations ORDER BY version"
        ).fetchall()
        return [
            (int(version), str(name), str(checksum)) for version, name, checksum in rows
        ]
    finally:
        connection.close()


def _table_names(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        connection.close()


async def test_async_open_close_and_reopen_preserve_the_foundation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"

    first = await _open(path)
    assert path.is_file()
    assert first.closed is False

    await first.close()
    assert first.closed is True

    second = await _open(path)
    assert second.closed is False
    assert _migration_rows(path) == [
        (migration.version, migration.name, migration.checksum)
        for migration in MIGRATIONS
    ]

    await second.close()
    assert second.closed is True


async def test_public_open_uses_the_package_owned_migration_plan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"

    assert "migrations" not in inspect.signature(SQLiteOperationStore.open).parameters

    store = await SQLiteOperationStore.open(path)
    await store.close()

    assert _migration_rows(path) == [
        (migration.version, migration.name, migration.checksum)
        for migration in sqlite_owner._MIGRATIONS
    ]


async def test_open_applies_the_v2_marker_and_required_connection_pragmas(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await _open(
        path,
        busy_timeout_ms=2_750,
    )

    foundation = await store.inspect_foundation()

    assert foundation.application_id == DAITA_V2_APPLICATION_ID
    assert foundation.journal_mode.lower() == "wal"
    assert foundation.foreign_keys is True
    assert foundation.busy_timeout_ms == 2_750
    assert foundation.synchronous == "FULL"

    await store.close()

    assert _scalar(path, "PRAGMA application_id") == DAITA_V2_APPLICATION_ID
    assert str(_scalar(path, "PRAGMA journal_mode")).lower() == "wal"


async def test_schema_migrations_are_ordered_durable_and_checksummed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await _open(path)
    await store.close()

    rows = _migration_rows(path)

    assert rows == [
        (migration.version, migration.name, migration.checksum)
        for migration in MIGRATIONS
    ]
    assert [version for version, _, _ in rows] == [1, 2]
    assert all(re.fullmatch(r"[0-9a-f]{64}", checksum) for _, _, checksum in rows)


@pytest.mark.parametrize("marker", [0, 0x13572468], ids=["missing", "wrong"])
async def test_existing_non_v2_database_is_rejected_without_mutation(
    tmp_path: Path,
    marker: int,
) -> None:
    path = tmp_path / "state.db"
    _execute(path, "CREATE TABLE legacy_state (value TEXT NOT NULL)")
    if marker:
        _execute(path, f"PRAGMA application_id = {marker}")
    before_tables = _table_names(path)

    with pytest.raises(SQLiteCompatibilityError):
        await _open(path)

    assert _scalar(path, "PRAGMA application_id") == marker
    assert _table_names(path) == before_tables
    assert "schema_migrations" not in _table_names(path)


@pytest.mark.parametrize(
    ("statement", "parameters"),
    [
        (
            "INSERT INTO schema_migrations(version, name, checksum) "
            "VALUES (?, ?, ?)",
            (3, "future_migration", "f" * 64),
        ),
        (
            "UPDATE schema_migrations SET name = ? WHERE version = ?",
            ("unknown_migration", 1),
        ),
        ("DELETE FROM schema_migrations WHERE version = ?", (1,)),
        (
            "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
            ("0" * 64, 1),
        ),
    ],
    ids=["future", "unknown", "gapped", "drifted"],
)
async def test_incompatible_migration_history_is_rejected_without_repair(
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
) -> None:
    path = tmp_path / "state.db"
    backup_path = tmp_path / "must-not-be-created.db"
    store = await _open(path)
    await store.close()
    _execute(path, statement, parameters)
    incompatible_rows = _migration_rows(path)

    with pytest.raises(SQLiteCompatibilityError):
        await _open(
            path,
            backup_path=backup_path,
        )

    assert _migration_rows(path) == incompatible_rows
    assert not backup_path.exists()


async def test_recognized_upgrade_uses_connection_backup_before_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    backup_path = tmp_path / "state.before-v2.db"
    old_store = await _open(
        path,
        migrations=(MIGRATION_ONE,),
    )
    await old_store.close()
    _execute(path, "INSERT INTO alpha(id, value) VALUES (?, ?)", (1, "durable"))

    backup_observations: list[tuple[Path, tuple[int, ...], bool]] = []
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def backup(
            self,
            target: sqlite3.Connection,
            *,
            pages: int = -1,
            progress: Callable[[int, int, int], object] | None = None,
            name: str = "main",
            sleep: float = 0.250,
        ) -> None:
            destination = Path(
                str(target.execute("PRAGMA database_list").fetchone()[2])
            ).resolve()
            versions = tuple(
                int(row[0])
                for row in self.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
            )
            beta_exists = (
                self.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'beta'"
                ).fetchone()
                is not None
            )
            backup_observations.append((destination, versions, beta_exists))
            super().backup(
                target,
                pages=pages,
                progress=progress,
                name=name,
                sleep=sleep,
            )

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
        return real_connect(
            database,
            timeout=timeout,
            detect_types=detect_types,
            isolation_level=isolation_level,
            check_same_thread=check_same_thread,
            factory=TrackingConnection,
            cached_statements=cached_statements,
            uri=uri,
        )

    monkeypatch.setattr(sqlite_owner.sqlite3, "connect", tracking_connect)

    upgraded = await _open(
        path,
        backup_path=backup_path,
    )
    await upgraded.close()

    assert backup_observations == [(backup_path.resolve(), (1,), False)]
    assert _migration_rows(path) == [
        (migration.version, migration.name, migration.checksum)
        for migration in MIGRATIONS
    ]
    assert _migration_rows(backup_path) == [
        (MIGRATION_ONE.version, MIGRATION_ONE.name, MIGRATION_ONE.checksum)
    ]
    assert "beta" in _table_names(path)
    assert "beta" not in _table_names(backup_path)
    assert _scalar(backup_path, "SELECT value FROM alpha WHERE id = 1") == "durable"


async def test_failed_migration_rolls_back_schema_data_and_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    backup_path = tmp_path / "state.before-failed-upgrade.db"
    old_store = await _open(
        path,
        migrations=(MIGRATION_ONE,),
    )
    await old_store.close()
    _execute(path, "INSERT INTO alpha(id, value) VALUES (?, ?)", (1, "durable"))
    failing_migration = sqlite_owner._SQLiteMigration(
        version=2,
        name="broken_upgrade",
        statements=(
            "CREATE TABLE partial_upgrade (value TEXT NOT NULL)",
            "INSERT INTO partial_upgrade(value) VALUES ('must roll back')",
            "INSERT INTO table_that_does_not_exist(value) VALUES ('fail')",
        ),
    )

    with pytest.raises(SQLiteMigrationError):
        await _open(
            path,
            migrations=(MIGRATION_ONE, failing_migration),
            backup_path=backup_path,
        )

    assert _migration_rows(path) == [
        (MIGRATION_ONE.version, MIGRATION_ONE.name, MIGRATION_ONE.checksum)
    ]
    assert "partial_upgrade" not in _table_names(path)
    assert _scalar(path, "SELECT value FROM alpha WHERE id = 1") == "durable"
    assert backup_path.is_file()

    reopened = await _open(
        path,
        migrations=(MIGRATION_ONE,),
    )
    await reopened.close()


async def test_failed_migration_names_the_actual_later_pending_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    failing_migration = sqlite_owner._SQLiteMigration(
        version=3,
        name="broken_third_migration",
        statements=(
            "CREATE TABLE partial_third_upgrade (value TEXT NOT NULL)",
            "INSERT INTO table_that_does_not_exist(value) VALUES ('fail')",
        ),
    )

    with pytest.raises(
        SQLiteMigrationError,
        match=r"migration 3 \(broken_third_migration\) failed",
    ):
        await _open(
            path,
            migrations=(*MIGRATIONS, failing_migration),
        )

    assert path.exists() is False


async def test_transient_backup_destination_cannot_satisfy_backup_before_migrate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    old_store = await _open(path, migrations=(MIGRATION_ONE,))
    await old_store.close()

    with pytest.raises(ValueError, match="durable filesystem path"):
        await _open(path, backup_path=Path(":memory:"))

    assert _migration_rows(path) == [
        (MIGRATION_ONE.version, MIGRATION_ONE.name, MIGRATION_ONE.checksum)
    ]
    assert "beta" not in _table_names(path)


async def test_close_resists_cancellation_while_waiting_for_the_connection_lock(
    tmp_path: Path,
) -> None:
    store = await _open(tmp_path / "state.db")
    await store._lock.acquire()
    close = asyncio.create_task(store.close())
    await asyncio.sleep(0)

    close.cancel()
    await asyncio.sleep(0)

    assert close.done() is False
    assert store.closed is False

    store._lock.release()
    with pytest.raises(asyncio.CancelledError):
        await close
    assert store.closed is True


@pytest.mark.parametrize("preexisting_empty_file", [False, True])
async def test_failed_initial_migration_leaves_a_retryable_fresh_path(
    tmp_path: Path,
    preexisting_empty_file: bool,
) -> None:
    path = tmp_path / "state.db"
    if preexisting_empty_file:
        path.touch()
    failing = sqlite_owner._SQLiteMigration(
        version=1,
        name="failed_initialization",
        statements=(
            "CREATE TABLE partial_initialization (value TEXT NOT NULL)",
            "INSERT INTO missing_table(value) VALUES ('fail')",
        ),
    )

    with pytest.raises(SQLiteMigrationError):
        await _open(path, migrations=(failing,))

    assert path.exists() is preexisting_empty_file
    if preexisting_empty_file:
        assert path.stat().st_size == 0

    recovered = await _open(path, migrations=(MIGRATION_ONE,))
    await recovered.close()
    assert "alpha" in _table_names(path)
    assert "partial_initialization" not in _table_names(path)


def test_migration_plan_rejects_transaction_and_connection_control_sql() -> None:
    for statement in (
        "BEGIN IMMEDIATE",
        "COMMIT",
        "END",
        "ROLLBACK",
        "SAVEPOINT unsafe",
        "RELEASE unsafe",
        "ATTACH DATABASE ':memory:' AS unsafe",
        "DETACH DATABASE unsafe",
        "PRAGMA foreign_keys = OFF",
        "VACUUM",
    ):
        with pytest.raises(ValueError, match="transaction or connection control"):
            sqlite_owner._SQLiteMigration(
                version=1,
                name="unsafe_migration",
                statements=(statement,),
            )


def test_raw_connection_cannot_bypass_the_open_boundary(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        with pytest.raises(TypeError, match="SQLiteOperationStore.open"):
            SQLiteOperationStore(
                tmp_path / "bypass.db",
                connection,
                busy_timeout_ms=5_000,
            )
    finally:
        connection.close()


async def test_existing_verified_default_backup_allows_migration_retry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    old_store = await _open(path, migrations=(MIGRATION_ONE,))
    await old_store.close()
    backup_path = path.with_name(f"{path.name}.before-v1.bak")
    source = sqlite3.connect(path)
    destination = sqlite3.connect(backup_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()

    upgraded = await _open(path)
    await upgraded.close()

    assert "beta" in _table_names(path)
    assert _migration_rows(backup_path) == [
        (MIGRATION_ONE.version, MIGRATION_ONE.name, MIGRATION_ONE.checksum)
    ]


async def test_marker_without_migration_history_is_rejected_consistently(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    _execute(path, f"PRAGMA application_id = {DAITA_V2_APPLICATION_ID}")
    backup_path = tmp_path / "must-not-exist.db"

    with pytest.raises(SQLiteCompatibilityError, match="migration history"):
        await _open(path, backup_path=backup_path)

    assert not backup_path.exists()
    assert _table_names(path) == set()
