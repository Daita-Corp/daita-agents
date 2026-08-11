from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sqlite3
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from daita import Agent
from daita import cli
from daita.adapters.models import SourceRegistration
from daita.security import SecretReference
from daita.storage import sqlite as sqlite_storage
from daita.storage.sqlite import (
    STATE_FORMAT_VERSION,
    DatabaseWriteOutcome,
    DatabaseWriteReceipt,
    SQLiteStateStore,
    StateCompatibilityCode,
    StateCompatibilityError,
)

FIXTURE = Path(__file__).parent / "fixtures" / "state" / "state-format-1.json"
NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _state_format(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute("PRAGMA user_version").fetchone()
    assert row is not None
    return int(row[0])


def _tables(path: Path) -> tuple[str, ...]:
    with sqlite3.connect(path) as connection:
        return tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )


def _rows(
    path: Path, tables: tuple[str, ...]
) -> dict[str, tuple[tuple[object, ...], ...]]:
    with sqlite3.connect(path) as connection:
        return {
            table: tuple(connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
            for table in tables
        }


def _schema_objects(path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY type, name"
        )
        return [
            {"type": kind, "name": name, "table": table, "sql": sql}
            for kind, name, table, sql in rows
        ]


def _make_previous_format(path: Path, *, marker: int = 0) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE database_write_receipts")
        connection.execute(f"PRAGMA user_version = {marker}")


def _mark_format_two(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 2")


async def _current_database(path: Path) -> None:
    store = await SQLiteStateStore.open(path)
    await store.close()


async def test_new_state_records_current_format_and_reopens_without_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _current_database(path)
    assert _state_format(path) == STATE_FORMAT_VERSION
    before = _sha256(path)

    store = await SQLiteStateStore.open(path)
    await store.close()

    assert _sha256(path) == before


@pytest.mark.parametrize("marker", (0, 1))
async def test_actual_predecessor_shape_migrates_atomically_and_preserves_all_rows(
    tmp_path: Path,
    marker: int,
) -> None:
    agent = await Agent.create("upgrade", root=tmp_path)
    path = agent.home / "state.db"
    agent_id = agent.id
    await agent.close()
    _make_previous_format(path, marker=marker)
    previous_tables = _tables(path)
    previous_rows = _rows(path, previous_tables)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["user_version"] == 0
    assert fixture["package_version"] == "1.0.0"
    assert _schema_objects(path) == fixture["objects"]

    migrated = await Agent.open("upgrade", root=tmp_path)
    try:
        assert migrated.id == agent_id
    finally:
        await migrated.close()

    assert _state_format(path) == STATE_FORMAT_VERSION
    assert set(_tables(path)) == {*previous_tables, "database_write_receipts"}
    assert _rows(path, previous_tables) == previous_rows
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM database_write_receipts"
        ).fetchone() == (0,)


async def test_previous_format_preserves_source_credential_reference_and_active_source(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("credential-upgrade", root=tmp_path)
    path = agent.home / "state.db"
    reference = SecretReference.environment("DAITA_UPGRADE_TEST_PASSWORD")
    registration = SourceRegistration.build(
        agent_id=agent.id,
        adapter_id="postgresql",
        native_identity="postgresql://upgrade-host:5432/example",
        display_name="Upgrade warehouse",
        configuration={"credential_ref": reference.to_uri(), "write_access": True},
        attached_at=NOW,
    )
    await agent._embedded._store.register_source(registration)
    await agent._embedded._store.set_active_source_id(agent.id, registration.id)
    await agent.close()
    _make_previous_format(path)

    reopened = await Agent.open("credential-upgrade", root=tmp_path)
    try:
        sources = await reopened.list_sources()
        active = await reopened.active_source()
        assert sources == (registration,)
        assert active == registration
        assert sources[0].configuration["credential_ref"] == reference.to_uri()
    finally:
        await reopened.close()

    persisted = path.read_text(encoding="utf-8", errors="ignore")
    assert reference.to_uri() in persisted
    assert "secret-value" not in persisted


@pytest.mark.parametrize("write_access", (None, False, True))
async def test_format_two_canonicalizes_postgresql_write_access_without_losing_state(
    tmp_path: Path,
    write_access: bool | None,
) -> None:
    agent = await Agent.create("postgresql-upgrade", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    reference = SecretReference.environment("DAITA_UPGRADE_TEST_PASSWORD")
    configuration: dict[str, object] = {
        "credential_ref": reference.to_uri(),
        "database": "warehouse",
        "host": "db.example.test",
        "port": 5432,
        "schemas": ("analytics", "public"),
        "ssl_mode": "require",
        "username": "daita_writer",
    }
    if write_access is not None:
        configuration["write_access"] = write_access
    registration = SourceRegistration.build(
        agent_id=agent.id,
        adapter_id="postgresql",
        native_identity="postgresql:upgrade-state-format-two",
        display_name="Upgrade warehouse",
        configuration=configuration,
        attached_at=NOW,
    )
    await agent._embedded._store.register_source(registration)
    await agent._embedded._store.set_active_source_id(agent.id, registration.id)
    await agent.close()
    _mark_format_two(path)
    tables = _tables(path)
    rows_before = _rows(path, tables)

    reopened = await Agent.open("postgresql-upgrade", root=tmp_path, clock=lambda: NOW)
    try:
        expected = SourceRegistration(
            id=registration.id,
            agent_id=registration.agent_id,
            adapter_id=registration.adapter_id,
            native_identity=registration.native_identity,
            display_name=registration.display_name,
            configuration=(
                {**dict(registration.configuration), "write_access": False}
                if write_access is None
                else registration.configuration
            ),
            attached_at=registration.attached_at,
            detached_at=registration.detached_at,
        )
        assert await reopened.list_sources() == (expected,)
        assert await reopened.active_source() == expected
        assert expected.configuration["credential_ref"] == reference.to_uri()
        assert expected.configuration["write_access"] is (
            False if write_access is None else write_access
        )
    finally:
        await reopened.close()

    assert _state_format(path) == STATE_FORMAT_VERSION
    rows_after = _rows(path, tables)
    assert {
        table: rows for table, rows in rows_after.items() if table != "sources"
    } == {table: rows for table, rows in rows_before.items() if table != "sources"}
    assert len(rows_after["sources"]) == 1
    old_source_data = json.loads(str(rows_before["sources"][0][2]))
    if write_access is None:
        old_source_data["fields"]["configuration"]["write_access"] = False
    expected_source_data = json.dumps(
        old_source_data,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert rows_after["sources"][0][2] == expected_source_data
    persisted = path.read_text(encoding="utf-8", errors="ignore")
    assert reference.to_uri() in persisted
    assert "secret-value" not in persisted


async def test_format_two_invalid_postgresql_write_access_rolls_back_atomically(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "invalid-postgresql-upgrade",
        root=tmp_path,
        clock=lambda: NOW,
    )
    path = agent.home / "state.db"
    registration = SourceRegistration.build(
        agent_id=agent.id,
        adapter_id="postgresql",
        native_identity="postgresql:invalid-upgrade-state",
        display_name="Invalid upgrade warehouse",
        configuration={
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "daita_writer",
            "write_access": "yes",
        },
        attached_at=NOW,
    )
    await agent._embedded._store.register_source(registration)
    await agent.close()
    _mark_format_two(path)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.open("invalid-postgresql-upgrade", root=tmp_path)

    assert raised.value.code is StateCompatibilityCode.MIGRATION_FAILED
    assert raised.value.to_mapping()["state_changed"] is False
    assert _state_format(path) == 2
    assert _sha256(path) == before


async def test_unversioned_current_format_preserves_terminal_write_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path, clock=lambda: NOW)
    started = DatabaseWriteReceipt.start(
        agent_id="agent-upgrade-receipt",
        run_id="run-upgrade-receipt",
        call_id="call-upgrade-receipt",
        capability_id="data.postgresql.update",
        source_id="source:sha256:" + "1" * 64,
        resource_id="catalog-resource:sha256:" + "2" * 64,
        intent_sha256="sha256:" + "3" * 64,
        preview_fingerprint="sha256:" + "4" * 64,
        started_at=NOW,
    )
    terminal = started.finish(
        DatabaseWriteOutcome.COMMITTED,
        completed_at=NOW + timedelta(seconds=1),
        affected_rows=1,
        normalized_error_code=None,
    )
    await store.start_database_write_receipt(started)
    await store.finish_database_write_receipt(terminal)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 0")

    reopened = await SQLiteStateStore.open(path, clock=lambda: NOW)
    try:
        assert (
            await reopened.load_database_write_receipt(
                terminal.agent_id, terminal.receipt_id
            )
            == terminal
        )
    finally:
        await reopened.close()


async def test_unversioned_current_candidate_is_stamped_without_changing_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _current_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 0")
    current_tables = _tables(path)
    current_rows = _rows(path, current_tables)

    store = await SQLiteStateStore.open(path)
    await store.close()

    assert _state_format(path) == STATE_FORMAT_VERSION
    assert _tables(path) == current_tables
    assert _rows(path, current_tables) == current_rows


async def test_failed_migration_rolls_back_schema_marker_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    await _current_database(path)
    _make_previous_format(path)
    tables_before = _tables(path)
    rows_before = _rows(path, tables_before)
    hash_before = _sha256(path)

    original = sqlite_storage._migrate_v1_to_v2

    def fail_after_ddl(connection: sqlite3.Connection) -> None:
        original(connection)
        raise RuntimeError("injected migration failure")

    monkeypatch.setitem(
        sqlite_storage._STATE_MIGRATIONS,
        1,
        replace(sqlite_storage._STATE_MIGRATIONS[1], apply=fail_after_ddl),
    )

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.MIGRATION_FAILED
    assert raised.value.to_mapping()["state_changed"] is False
    assert _state_format(path) == 0
    assert _tables(path) == tables_before
    assert _rows(path, tables_before) == rows_before
    assert _sha256(path) == hash_before


async def test_cancellation_waits_for_atomic_migration_to_settle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    await _current_database(path)
    _make_previous_format(path)
    previous_tables = _tables(path)
    previous_rows = _rows(path, previous_tables)
    entered = threading.Event()
    release = threading.Event()
    original = sqlite_storage._migrate_v1_to_v2

    def controlled_migration(connection: sqlite3.Connection) -> None:
        original(connection)
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setitem(
        sqlite_storage._STATE_MIGRATIONS,
        1,
        replace(sqlite_storage._STATE_MIGRATIONS[1], apply=controlled_migration),
    )
    opening = asyncio.create_task(SQLiteStateStore.open(path))
    assert await asyncio.to_thread(entered.wait, 5)

    opening.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await opening

    assert _state_format(path) == STATE_FORMAT_VERSION
    assert _rows(path, previous_tables) == previous_rows
    assert set(_tables(path)) == {*previous_tables, "database_write_receipts"}


async def test_newer_format_is_a_downgrade_error_and_is_never_changed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _current_database(path)
    newer = STATE_FORMAT_VERSION + 1
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {newer}")
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.NEWER_FORMAT
    assert raised.value.found_format == newer
    assert "newer Daita release" in str(raised.value)
    assert _sha256(path) == before


async def test_recognizable_pre_1_state_is_distinguished_from_damage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE operations(id TEXT PRIMARY KEY, data TEXT)")
        connection.execute("CREATE TABLE tasks(id TEXT PRIMARY KEY, data TEXT)")
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.LEGACY_FORMAT
    assert "pre-1.0" in str(raised.value)
    assert _sha256(path) == before


async def test_pre_1_root_layout_is_rejected_without_mixing_new_agent_state(
    tmp_path: Path,
) -> None:
    legacy_session = tmp_path / "sessions" / "workspace" / "session.json"
    legacy_session.parent.mkdir(parents=True)
    legacy_session.write_text('{"history":[]}', encoding="utf-8")

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.list(root=tmp_path)

    assert raised.value.code is StateCompatibilityCode.LEGACY_FORMAT
    assert legacy_session.read_text(encoding="utf-8") == '{"history":[]}'
    assert not (tmp_path / "agents").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "DROP TABLE syncs",
        "ALTER TABLE runs ADD COLUMN future_value TEXT",
        "DROP INDEX runs_conversation_turn",
        "CREATE VIEW unexpected_view AS SELECT key FROM metadata",
    ),
)
async def test_damaged_current_state_fails_before_any_write(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "state.db"
    await _current_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert raised.value.found_format == STATE_FORMAT_VERSION
    assert _sha256(path) == before


async def test_existing_empty_state_is_damaged_and_not_initialized_in_place(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    path.touch()
    before = path.read_bytes()

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert path.read_bytes() == before


async def test_agent_open_preserves_damaged_home_and_releases_writer_lock(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("incompatible-upgrade", root=tmp_path)
    home = agent.home
    await agent.close()
    state_path = home / "state.db"
    manifest_before = (home / "agent.toml").read_bytes()
    with sqlite3.connect(state_path) as connection:
        connection.execute("DROP INDEX runs_conversation_turn")
    state_before = state_path.read_bytes()

    for _attempt in range(2):
        with pytest.raises(StateCompatibilityError) as raised:
            await Agent.open("incompatible-upgrade", root=tmp_path)
        assert raised.value.code is StateCompatibilityCode.DAMAGED
        assert (home / "agent.toml").read_bytes() == manifest_before
        assert state_path.read_bytes() == state_before


def test_headless_cli_returns_structured_newer_format_diagnostic(
    tmp_path: Path,
) -> None:
    created = asyncio.run(Agent.create("newer", root=tmp_path))
    path = created.home / "state.db"
    asyncio.run(created.close())
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {STATE_FORMAT_VERSION + 1}")
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(["--root", str(tmp_path), "sources", "newer"])

    assert code == 1
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    error = payload["error"]
    assert error["code"] == "state_format_newer"
    assert error["current_format"] == STATE_FORMAT_VERSION
    assert error["found_format"] == STATE_FORMAT_VERSION + 1
    assert error["state_changed"] is False
    assert error["state_path"] == str(path)
    assert "newer Daita release" in error["message"]


def test_interactive_tui_renders_human_compatibility_diagnostic(tmp_path: Path) -> None:
    created = asyncio.run(Agent.create("newer", root=tmp_path))
    path = created.home / "state.db"
    asyncio.run(created.close())
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {STATE_FORMAT_VERSION + 1}")
    stdin = _TTYBuffer()
    stdout = _TTYBuffer()
    stderr = _TTYBuffer()

    with (
        patch.object(sys, "stdin", stdin),
        patch.object(sys, "stdout", stdout),
        patch.object(sys, "stderr", stderr),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        code = cli.main(["--root", str(tmp_path), "--agent", "newer"])

    rendered = stderr.getvalue()
    assert code == 1
    assert stdout.getvalue() == ""
    assert rendered.startswith("Daita could not open this local agent state.\n")
    assert "newer Daita release" in rendered
    assert f"State: {path}" in rendered
    assert (
        f"Format: {STATE_FORMAT_VERSION + 1} "
        f"(this release: {STATE_FORMAT_VERSION})" in rendered
    )
    assert not rendered.lstrip().startswith("{")
