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

from daita import Agent, cli
from daita.adapters.models import SourceRegistration
from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.security import SecretReference
from daita.storage.sqlite import (
    DatabaseWriteOutcome,
    DatabaseWriteReceipt,
    SQLiteStateStore,
)
from daita.storage.sqlite_migrations import migration_rows, runner as migration_runner
from daita.storage.sqlite_migrations.postgresql_write_admission import (
    MIGRATION as ADMISSION_MIGRATION,
)
from daita.storage.sqlite_migrations.scoped_source_permissions import (
    MIGRATION as SCOPED_PERMISSION_MIGRATION,
)
from daita.storage.sqlite_schema import (
    ADMISSION_TABLE_SQL,
    INITIAL_TABLES,
    RECEIPT_TABLES,
    require_schema,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "state" / "preledger-supported-shapes.json"
)
NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _journal(path: Path) -> tuple[tuple[int, str, str], ...]:
    with sqlite3.connect(path) as connection:
        return tuple(connection.execute("""SELECT ordinal, migration_id, checksum
                   FROM state_migrations ORDER BY ordinal"""))


def _embedded_admission(
    connection: sqlite3.Connection,
    source_id: str,
    value: object,
) -> None:
    row = connection.execute(
        "SELECT agent_id, data FROM sources WHERE id = ?", (source_id,)
    ).fetchone()
    assert row is not None
    agent_id, data = row
    payload = json.loads(data)
    configuration = payload["fields"]["configuration"]
    if value is _MISSING:
        configuration.pop("write_access", None)
    else:
        configuration["write_access"] = value
    connection.execute(
        "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            agent_id,
            source_id,
        ),
    )


_MISSING = object()


class _RejectSecretProvider:
    async def resolve(self, reference: SecretReference) -> str:
        raise AssertionError(f"upgrade must not resolve secret reference {reference}")


def _make_preledger(
    path: Path,
    *,
    receipt_era: bool,
    marker: int,
    embedded_admissions: dict[str, object] | None = None,
) -> None:
    with sqlite3.connect(path) as connection:
        for source_id, value in (embedded_admissions or {}).items():
            _embedded_admission(connection, source_id, value)
        connection.execute("DROP TABLE postgresql_update_scopes")
        connection.execute("DROP TABLE source_read_scopes")
        connection.execute("DROP TABLE state_migrations")
        if not receipt_era:
            connection.execute("DROP TABLE database_write_receipts")
        connection.execute(f"PRAGMA user_version = {marker}")


def _make_journal_prefix(path: Path, source_id: str, *, admission: object) -> None:
    with sqlite3.connect(path) as connection:
        _embedded_admission(connection, source_id, admission)
        connection.execute("DROP TABLE postgresql_update_scopes")
        connection.execute("DROP TABLE source_read_scopes")
        connection.execute("DELETE FROM state_migrations WHERE ordinal >= 2")


def _make_admission_prefix(
    path: Path,
    source_id: str,
    *,
    admitted: bool,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE postgresql_update_scopes")
        connection.execute("DROP TABLE source_read_scopes")
        connection.execute(ADMISSION_TABLE_SQL)
        if admitted:
            agent_id = connection.execute(
                "SELECT agent_id FROM sources WHERE id = ?", (source_id,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO postgresql_write_admissions(agent_id, source_id) "
                "VALUES (?, ?)",
                (agent_id, source_id),
            )
        connection.execute("DELETE FROM state_migrations WHERE ordinal = 3")


def _registration(agent_id: str) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="postgresql",
        native_identity="postgresql:upgrade-contract",
        display_name="Upgrade warehouse",
        configuration={
            "credential_ref": SecretReference.environment(
                "DAITA_UPGRADE_TEST_PASSWORD"
            ).to_uri(),
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("analytics", "public"),
            "ssl_mode": "require",
            "username": "daita_writer",
            "custom_key": "preserved",
            "write_access": False,
        },
        attached_at=NOW,
    )


async def test_fresh_state_has_exact_journal_and_reopens_without_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()

    assert _journal(path) == migration_rows()
    assert {
        "state_migrations",
        "source_read_scopes",
        "postgresql_update_scopes",
        "database_write_receipts",
    } <= set(_tables(path))
    assert "postgresql_write_admissions" not in _tables(path)
    before = _sha256(path)

    reopened = await SQLiteStateStore.open(path)
    await reopened.close()

    assert _sha256(path) == before


def test_supported_preledger_fixture_matches_the_bounded_bridge(tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["provenance"] == {
        "pre_receipt": "retained 1.0.0 pre-receipt wheel schema fixture",
        "receipt_era_checkpoint": "d9faeba",
        "receipt_era_package_version": "1.0.0",
        "receipt_era_wheel_sha256": (
            "9af300e3cd7eaf6a177ae1380d9f36c22fafbf60a20695354cdf1994e61f645a"
        ),
    }
    assert fixture["pre_receipt"]["user_version_markers"] == [0, 1]
    assert fixture["receipt_era"] == {
        "observed_user_version": 3,
        "user_version_markers": [0, 2, 3],
    }
    assert fixture["removal_gate"] == (
        "remove only after the minimum supported release is guaranteed "
        "to contain state_migrations"
    )

    for name, expected_schema in (
        ("pre_receipt", INITIAL_TABLES),
        ("receipt_era", RECEIPT_TABLES),
    ):
        objects = list(fixture["base_objects"])
        if name == "receipt_era":
            objects.extend(fixture["receipt_era_extra_objects"])
        path = tmp_path / f"{name}.db"
        with sqlite3.connect(path) as connection:
            for item in sorted(objects, key=lambda value: value["type"] == "index"):
                connection.execute(item["sql"])
            require_schema(connection, expected_schema)


@pytest.mark.parametrize(
    ("receipt_era", "marker"),
    ((False, 0), (False, 1), (True, 0), (True, 2), (True, 3)),
)
async def test_every_supported_preledger_shape_upgrades_and_preserves_rows(
    tmp_path: Path,
    receipt_era: bool,
    marker: int,
) -> None:
    agent = await Agent.create("supported-preledger", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    registration = _registration(agent.id)
    registered = await agent._embedded._store.register_source(registration)
    await agent._embedded._store.set_active_source_id(agent.id, registration.id)
    if receipt_era:
        receipt = DatabaseWriteReceipt.start(
            agent_id=agent.id,
            run_id="run-upgrade-receipt",
            call_id="call-upgrade-receipt",
            capability_id="data.postgresql.update",
            source_id=registration.id,
            resource_id="catalog-resource:sha256:" + "2" * 64,
            intent_sha256="sha256:" + "3" * 64,
            preview_fingerprint="sha256:" + "4" * 64,
            started_at=NOW,
        ).finish(
            DatabaseWriteOutcome.COMMITTED,
            completed_at=NOW + timedelta(seconds=1),
            affected_rows=1,
            normalized_error_code=None,
        )
        await agent._embedded._store.start_database_write_receipt(receipt.as_started())
        await agent._embedded._store.finish_database_write_receipt(receipt)
    agent_id = agent.id
    await agent.close()
    _make_preledger(
        path,
        receipt_era=receipt_era,
        marker=marker,
        embedded_admissions={registration.id: True},
    )
    old_tables = _tables(path)
    old_rows = _rows(path, old_tables)

    reopened = await Agent.open(
        "supported-preledger",
        root=tmp_path,
        clock=lambda: NOW,
        secret_provider=_RejectSecretProvider(),
    )
    try:
        assert reopened.id == agent_id
        sources = await reopened.list_sources()
        assert sources == (registered,)
        assert await reopened.active_source() == sources[0]
        assert (
            sources[0].configuration["credential_ref"]
            == registered.configuration["credential_ref"]
        )
        assert sources[0].configuration["custom_key"] == "preserved"
    finally:
        await reopened.close()

    assert _journal(path) == migration_rows()
    after = _rows(path, _tables(path))
    for table, rows in old_rows.items():
        if table == "sources":
            continue
        assert after[table] == rows
    with sqlite3.connect(path) as connection:
        source_data = connection.execute("SELECT data FROM sources").fetchone()[0]
        assert "write_access" not in json.loads(source_data)["fields"]["configuration"]
        assert connection.execute(
            "SELECT agent_id, source_id FROM source_read_scopes"
        ).fetchone() == (agent_id, registration.id)
        assert connection.execute(
            "SELECT COUNT(*) FROM postgresql_update_scopes"
        ).fetchone() == (0,)


@pytest.mark.parametrize("legacy_value", (_MISSING, False, True))
async def test_preledger_admission_cutover_is_fail_closed(
    tmp_path: Path,
    legacy_value: object,
) -> None:
    agent = await Agent.create("admission-cutover", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    await agent.close()
    _make_preledger(
        path,
        receipt_era=True,
        marker=2,
        embedded_admissions={registration.id: legacy_value},
    )

    reopened = await Agent.open("admission-cutover", root=tmp_path, clock=lambda: NOW)
    try:
        assert (await reopened.list_sources())[0].configuration["write_access"] is False
        assert (
            await reopened._embedded._store.list_postgresql_update_scopes(
                reopened.id,
                registration.id,
            )
            == ()
        )
    finally:
        await reopened.close()


async def test_invalid_preledger_admission_rolls_back_byte_for_byte(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("invalid-admission", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    await agent.close()
    _make_preledger(
        path,
        receipt_era=True,
        marker=2,
        embedded_admissions={registration.id: "yes"},
    )
    before = _sha256(path)
    rows_before = _rows(path, _tables(path))

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.open("invalid-admission", root=tmp_path)

    assert raised.value.code is StateCompatibilityCode.UPGRADE_FAILED
    assert raised.value.to_mapping()["state_changed"] is False
    assert _sha256(path) == before
    assert _rows(path, _tables(path)) == rows_before


async def test_detached_preledger_source_never_inherits_embedded_admission(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("detached-cutover", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    detached = await agent._embedded._store.detach_source(
        agent.id,
        registration.id,
        NOW + timedelta(seconds=1),
    )
    await agent.close()
    _make_preledger(
        path,
        receipt_era=True,
        marker=3,
        embedded_admissions={registration.id: True},
    )

    reopened = await Agent.open("detached-cutover", root=tmp_path, clock=lambda: NOW)
    try:
        sources = await reopened.list_sources()
        assert sources == (detached,)
        assert await reopened.active_source() is None
        with sqlite3.connect(path) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM source_read_scopes"
            ).fetchone() == (0,)
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    ("mutation", "found"),
    (
        (
            "UPDATE state_migrations SET checksum = '0' WHERE ordinal = 1",
            "20260810_database_write_receipts",
        ),
        (
            "UPDATE state_migrations SET migration_id = '20990101_unknown' WHERE ordinal = 2",
            "20990101_unknown",
        ),
        (
            "UPDATE state_migrations SET ordinal = 4 WHERE ordinal = 2",
            "20260812_scoped_source_permissions",
        ),
    ),
)
async def test_checksum_unknown_id_and_gap_are_refused_without_write(
    tmp_path: Path,
    mutation: str,
    found: str,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
    assert raised.value.found_revision == found
    assert _sha256(path) == before


async def test_skipped_release_journal_prefix_applies_pending_migration(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("journal-prefix", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    await agent.close()
    _make_journal_prefix(path, registration.id, admission=True)

    reopened = await Agent.open("journal-prefix", root=tmp_path, clock=lambda: NOW)
    try:
        assert (await reopened.list_sources())[0].configuration["write_access"] is False
        assert (
            await reopened._embedded._store.list_postgresql_update_scopes(
                reopened.id,
                registration.id,
            )
            == ()
        )
    finally:
        await reopened.close()
    assert _journal(path) == migration_rows()


async def test_cancellation_waits_for_atomic_journal_upgrade_to_settle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = await Agent.create("cancelled-upgrade", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    await agent.close()
    _make_admission_prefix(path, registration.id, admitted=True)
    entered = threading.Event()
    release = threading.Event()

    def controlled_apply(connection: sqlite3.Connection) -> None:
        SCOPED_PERMISSION_MIGRATION.apply(connection)
        entered.set()
        assert release.wait(timeout=5)

    controlled = replace(SCOPED_PERMISSION_MIGRATION, apply=controlled_apply)
    monkeypatch.setattr(
        migration_runner,
        "MIGRATIONS",
        (*migration_runner.MIGRATIONS[:2], controlled),
    )
    opening = asyncio.create_task(SQLiteStateStore.open(path))
    assert await asyncio.to_thread(entered.wait, 5)

    opening.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await opening

    assert _journal(path) == migration_rows()[:2]
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT source_id FROM postgresql_write_admissions"
        ).fetchone() == (registration.id,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'source_read_scopes'"
            ).fetchone()
            is None
        )


async def test_scoped_permission_migration_failure_rolls_back_every_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = await Agent.create("failed-scope-upgrade", root=tmp_path, clock=lambda: NOW)
    path = agent.home / "state.db"
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    await agent.close()
    _make_admission_prefix(path, registration.id, admitted=True)
    rows_before = _rows(path, _tables(path))

    def failing_apply(connection: sqlite3.Connection) -> None:
        SCOPED_PERMISSION_MIGRATION.apply(connection)
        raise RuntimeError("controlled migration failure")

    controlled = replace(SCOPED_PERMISSION_MIGRATION, apply=failing_apply)
    monkeypatch.setattr(
        migration_runner,
        "MIGRATIONS",
        (*migration_runner.MIGRATIONS[:2], controlled),
    )
    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.UPGRADE_FAILED
    assert _journal(path) == migration_rows()[:2]
    assert _rows(path, _tables(path)) == rows_before


async def test_newer_preledger_state_is_refused_without_write(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    _make_preledger(path, receipt_era=True, marker=4)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.NEWER_REVISION
    assert "newer Daita release" in str(raised.value)
    assert _sha256(path) == before


async def test_newer_journal_extension_is_a_downgrade_refusal_without_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    future_revision = "20270101_future_revision"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO state_migrations(ordinal, migration_id, checksum) "
            "VALUES (4, ?, ?)",
            (future_revision, "f" * 64),
        )
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.NEWER_REVISION
    assert raised.value.found_revision == future_revision
    assert "newer Daita release" in str(raised.value)
    assert _sha256(path) == before


async def test_invalid_later_journal_entry_is_unsupported_without_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO state_migrations(ordinal, migration_id, checksum) "
            "VALUES (5, 'gapped-future', ?)",
            ("f" * 64,),
        )
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
    assert _sha256(path) == before


@pytest.mark.parametrize("marker", (0, 99))
async def test_recognizable_pre_1_state_is_distinguished_from_damage(
    tmp_path: Path,
    marker: int,
) -> None:
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE operations(id TEXT PRIMARY KEY, data TEXT)")
        connection.execute("CREATE TABLE tasks(id TEXT PRIMARY KEY, data TEXT)")
        connection.execute(f"PRAGMA user_version = {marker}")
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.LEGACY
    assert _sha256(path) == before


async def test_unrecognized_high_marker_is_damage_not_newer_daita_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated(id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 99")
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert _sha256(path) == before


async def test_pre_1_root_layout_is_rejected_without_mixing_state(
    tmp_path: Path,
) -> None:
    legacy_session = tmp_path / "sessions" / "workspace" / "session.json"
    legacy_session.parent.mkdir(parents=True)
    legacy_session.write_text('{"history":[]}', encoding="utf-8")

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.list(root=tmp_path)

    assert raised.value.code is StateCompatibilityCode.LEGACY
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
    store = await SQLiteStateStore.open(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert _sha256(path) == before


@pytest.mark.parametrize(
    "rewrite",
    (
        """
        ALTER TABLE state_migrations RENAME TO malformed_state_migrations;
        CREATE TABLE state_migrations (
            ordinal INTEGER NOT NULL,
            migration_id TEXT NOT NULL PRIMARY KEY,
            checksum TEXT NOT NULL
        );
        INSERT INTO state_migrations SELECT * FROM malformed_state_migrations;
        DROP TABLE malformed_state_migrations;
        """,
        """
        ALTER TABLE database_write_receipts
            RENAME TO malformed_database_write_receipts;
        CREATE TABLE database_write_receipts (
            agent_id TEXT NOT NULL,
            id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY(agent_id, id)
        );
        INSERT INTO database_write_receipts
            SELECT * FROM malformed_database_write_receipts;
        DROP TABLE malformed_database_write_receipts;
        """,
    ),
)
async def test_missing_required_unique_constraint_is_damaged_without_write(
    tmp_path: Path,
    rewrite: str,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.executescript(rewrite)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert _sha256(path) == before


async def test_existing_empty_state_is_damaged_and_not_initialized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    path.touch()
    before = path.read_bytes()

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert path.read_bytes() == before


def test_headless_cli_returns_descriptive_revision_diagnostic(tmp_path: Path) -> None:
    created = asyncio.run(Agent.create("unsupported", root=tmp_path))
    path = created.home / "state.db"
    asyncio.run(created.close())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE state_migrations SET migration_id = '20990101_unknown' "
            "WHERE ordinal = 2"
        )
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(["--root", str(tmp_path), "sources", "unsupported"])

    assert code == 1
    assert stdout.getvalue() == ""
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "state_revision_unsupported"
    assert error["current_revision"] == SQLiteStateStore.current_revision
    assert error["found_revision"] == "20990101_unknown"
    assert error["state_changed"] is False
    assert error["state_path"] == str(path)
    assert "current_format" not in error
    assert "found_format" not in error


def test_interactive_tui_renders_human_safe_upgrade_diagnostic(
    tmp_path: Path,
) -> None:
    created = asyncio.run(Agent.create("unsupported", root=tmp_path))
    path = created.home / "state.db"
    asyncio.run(created.close())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE state_migrations SET migration_id = '20990101_unknown' "
            "WHERE ordinal = 2"
        )
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
        code = cli.main(["--root", str(tmp_path), "--agent", "unsupported"])

    rendered = stderr.getvalue()
    assert code == 1
    assert stdout.getvalue() == ""
    assert rendered.startswith("Daita could not open this local agent state.\n")
    assert f"State: {path}" in rendered
    assert "Local data changed: no" in rendered
    assert "Format:" not in rendered
    assert "current_revision" not in rendered
    assert not rendered.lstrip().startswith("{")


def test_migration_checksums_are_stable_sha256_values() -> None:
    assert migration_rows() == (
        (
            1,
            "20260810_database_write_receipts",
            "0cf5d23bf0426851e51c24450d1f8febd221880e74c78fc39648b6a1dd015b84",
        ),
        (
            2,
            "20260811_postgresql_write_admission",
            "451840240521fe5ad424d43e0bc5b7df2d124b3261b35be64b53bd36e08431d0",
        ),
        (
            3,
            "20260812_scoped_source_permissions",
            "2ed3f7017f9d4c683ee17a0ba43c88ad4452c5af1b06223343cd43248f699d95",
        ),
    )


def test_migration_checksum_binds_identity_schemas_transform_and_validation() -> None:
    def alternate_apply(connection: sqlite3.Connection) -> None:
        connection.execute("SELECT 1")

    def alternate_validation(connection: sqlite3.Connection) -> None:
        connection.execute("SELECT 1")

    variants = (
        replace(ADMISSION_MIGRATION, ordinal=3),
        replace(ADMISSION_MIGRATION, migration_id="alternate_migration"),
        replace(ADMISSION_MIGRATION, definition="alternate definition"),
        replace(ADMISSION_MIGRATION, source_schema={}),
        replace(ADMISSION_MIGRATION, target_schema={}),
        replace(ADMISSION_MIGRATION, apply=alternate_apply),
        replace(ADMISSION_MIGRATION, validate_target=alternate_validation),
        replace(ADMISSION_MIGRATION, validate_target=None),
    )

    assert all(variant.checksum != ADMISSION_MIGRATION.checksum for variant in variants)


def test_obsolete_numeric_and_reflection_owners_are_absent() -> None:
    production = Path(__file__).parents[1] / "src" / "daita"
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in production.rglob("*.py")
    )
    for obsolete in (
        "STATE_FORMAT_VERSION",
        "_UNVERSIONED_STATE_FORMAT",
        "_StateMigration",
        "_STATE_MIGRATIONS",
        "_state_migration_path",
        "_unversioned_state_format",
        "_migrate_existing_state",
        "_migrate_v1_to_v2",
        "_migrate_v2_to_v3",
        "_require_current_source_records",
        "_UNVERSIONED_STATE_SCHEMAS",
        "_RECORD_TYPES",
        "_ENUM_TYPES",
        "def _pack(",
        "def _unpack(",
        "def _dumps(",
        "def _loads(",
    ):
        assert obsolete not in text
    pragma_owners = {
        path.relative_to(production).as_posix()
        for path in production.rglob("*.py")
        if "PRAGMA user_version" in path.read_text(encoding="utf-8")
    }
    assert pragma_owners == {"storage/sqlite_migrations/preledger.py"}
