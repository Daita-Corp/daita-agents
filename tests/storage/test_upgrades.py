from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from daita import Agent, AgentConfig, SQLiteSource, cli
from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.hosting import home_upgrade as coordinator
from daita.hosting.embedded import (
    _encode_model_profile,
    _model_route,
    _write_model_configuration,
)
from daita.llm.models import CanonicalMessage, MessageRole, TextBlock
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.storage import sqlite as sqlite_store
from daita.storage.home_migrations import (
    HOME_MIGRATIONS,
    HomeMigration,
    registry as migration_registry,
)
from daita.storage.home_migrations.revision_0001 import REVISION_1
from daita.storage.home_migrations.revision_0001_schema import (
    REVISION_1_DATABASE_SQL,
    SCHEMA_REVISION_1,
)
from daita.storage.home_migrations.revision_0002 import REVISION_2
from daita.storage.schema_contract import require_healthy, require_schema
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_schema import CURRENT_SCHEMA
from tests.support.paths import REPO_ROOT
from tests.support.workspace import workspace_for


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path) -> tuple[tuple[int, str, str], ...]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(
                "SELECT revision, migration_id, checksum "
                "FROM agent_home_migrations ORDER BY revision"
            )
        )


def _validate_synthetic_home(
    source_home: Path,
    candidate_home: Path,
    affected_paths: frozenset[str],
) -> None:
    state_home = candidate_home if "state.db" in affected_paths else source_home
    memory_home = candidate_home if "MEMORY.md" in affected_paths else source_home
    with sqlite3.connect(state_home / "state.db") as connection:
        require_schema(connection, CURRENT_SCHEMA)
        require_healthy(connection)
        assert sqlite_store._validate_current_records(connection) is not None
        assert tuple(
            connection.execute(
                "SELECT revision, migration_id, checksum "
                "FROM agent_home_migrations ORDER BY revision"
            )
        ) == tuple(
            (item.revision, item.migration_id, item.checksum)
            for item in migration_registry.HOME_MIGRATIONS
        )
    assert (
        (memory_home / "MEMORY.md")
        .read_text(encoding="utf-8")
        .endswith("revision three\n")
    )


def _synthetic_next_apply(staged_home: Path, source_shape: str | None) -> None:
    assert source_shape is None
    with (staged_home / "MEMORY.md").open("a", encoding="utf-8") as file:
        file.write("revision three\n")


def _synthetic_failure(staged_home: Path, source_shape: str | None) -> None:
    _synthetic_next_apply(staged_home, source_shape)
    raise RuntimeError("controlled migration failure")


def _synthetic_next_migration(*, apply=_synthetic_next_apply) -> HomeMigration:
    return HomeMigration(
        revision=3,
        migration_id="test_only_agent_home_revision_3",
        definition="test-only whole-home revision",
        affected_paths=("state.db", "MEMORY.md"),
        target_schema=CURRENT_SCHEMA,
        apply=apply,
        implementation_material=("test-only-material",),
    )


def _patch_next_migration(
    monkeypatch: pytest.MonkeyPatch,
    migration: HomeMigration,
) -> None:
    migrations = (REVISION_1, REVISION_2, migration)
    monkeypatch.setattr(migration_registry, "HOME_MIGRATIONS", migrations)
    monkeypatch.setattr(migration_registry, "CURRENT_HOME_REVISION", 3)
    monkeypatch.setattr(coordinator, "HOME_MIGRATIONS", migrations)
    monkeypatch.setattr(coordinator, "CURRENT_HOME_REVISION", 3)
    monkeypatch.setattr(sqlite_store, "CURRENT_HOME_REVISION", 3)
    monkeypatch.setattr(SQLiteStateStore, "current_revision", "3")


async def _create_rich_home(tmp_path: Path, name: str = "atlas") -> tuple[Path, str]:
    database = tmp_path / f"{name}-source.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE facts(id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO facts(value) VALUES ('preserved')")
    route = _model_route(
        "ollama",
        "test-model",
        base_url=None,
        secret_reference=None,
        context_window_tokens=8_192,
        max_output_tokens=1_024,
    )
    agent = await Agent.create(
        name,
        root=tmp_path,
        config=AgentConfig(model_route=route),
        workspace=workspace_for(tmp_path),
    )
    await agent.set_memory("Durable memory.")
    await agent.set_user_profile("Durable user profile.")
    await agent.save_skill(
        "preserved-skill",
        "A durable procedure.",
        "Read the current catalog first.",
    )
    await agent.attach(SQLiteSource(database, name="Preserved source"))
    home = agent.home
    agent_id = agent.id
    await agent.close()
    _write_model_configuration(home, AgentConfig(model_route=route))

    now = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    run = RunInput(
        id="run-preserved",
        agent_id=agent_id,
        message="Preserve this question.",
        created_at=now,
        conversation_id="conversation-preserved",
    )
    user = CanonicalMessage(
        role=MessageRole.USER,
        content=(TextBlock("Preserve this question."),),
    )
    assistant = CanonicalMessage(
        role=MessageRole.ASSISTANT,
        content=(TextBlock("Preserved answer."),),
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id=run.conversation_id or "",
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=now,
        final_text="Preserved answer.",
    )
    store = await SQLiteStateStore.open(home / "state.db")
    await store.start(run)
    await store.append(run.id, user)
    await store.complete(result, assistant)
    await store.close()
    return home, agent_id


def _durable_snapshot(home: Path) -> dict[str, object]:
    tables = (
        "messages",
        "metadata",
        "runs",
        "snapshots",
        "source_read_scopes",
        "sources",
        "syncs",
    )
    with sqlite3.connect(home / "state.db") as connection:
        rows = {
            table: tuple(connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
            for table in tables
        }
    return {
        "config": json.loads((home / "config.json").read_text(encoding="utf-8")),
        "memory": (home / "MEMORY.md").read_bytes(),
        "rows": rows,
        "skill": (home / "skills/preserved-skill/SKILL.md").read_bytes(),
        "user": (home / "USER.md").read_bytes(),
    }


def _make_preproduction_home(home: Path, shape: str) -> None:
    current_path = home / "state.db"
    revision_1_path = home / ".revision-1-source.db"
    with (
        sqlite3.connect(current_path) as source,
        sqlite3.connect(revision_1_path) as target,
    ):
        target.executescript(REVISION_1_DATABASE_SQL)
        source_tables = {
            str(row[0])
            for row in source.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        for table, definitions in SCHEMA_REVISION_1.tables.items():
            if table not in source_tables or table == "agent_home_migrations":
                continue
            columns = tuple(str(column[0]) for column in definitions)
            projection = ", ".join(f'"{column}"' for column in columns)
            rows = tuple(source.execute(f'SELECT {projection} FROM "{table}"'))
            if rows:
                placeholders = ", ".join("?" for _ in columns)
                target.executemany(
                    f'INSERT INTO "{table}" ({projection}) VALUES ({placeholders})',
                    rows,
                )
        target.execute(
            "INSERT INTO agent_home_migrations VALUES (?, ?, ?)",
            (REVISION_1.revision, REVISION_1.migration_id, REVISION_1.checksum),
        )
    revision_1_path.replace(current_path)
    current_path.with_name(current_path.name + "-wal").unlink(missing_ok=True)
    current_path.with_name(current_path.name + "-shm").unlink(missing_ok=True)
    config_path = home / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    route = _model_route(
        "ollama",
        "test-model",
        base_url=None,
        secret_reference=None,
        context_window_tokens=8_192,
        max_output_tokens=1_024,
    )
    for raw, candidate in zip(
        config["model_route"]["candidates"], route.candidates, strict=True
    ):
        raw["profile"] = _encode_model_profile(candidate.profile)
        del raw["profile_limits"]
    retry = config["model_route"]["retry_policy"]
    config["model_route"]["retry_policy"] = {
        "attempts": retry["max_attempts_per_candidate"],
        "backoff_seconds": retry["backoff_seconds"],
    }
    del config["model_call_policy"]
    config_path.write_text(
        json.dumps(config, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )

    versioned_tables = (
        "autonomous_followups",
        "deliveries",
        "job_runs",
        "mcp_server_bindings",
        "relational_write_scopes",
        "routine_occurrences",
        "scheduled_routines",
        "source_read_scopes",
    )
    with sqlite3.connect(home / "state.db") as connection:
        for table in versioned_tables:
            rows = tuple(connection.execute(f'SELECT rowid, data FROM "{table}"'))
            for rowid, encoded in rows:
                payload = json.loads(encoded)
                payload["fields"]["version"] = 1
                connection.execute(
                    f'UPDATE "{table}" SET data = ? WHERE rowid = ?',
                    (json.dumps(payload, separators=(",", ":"), sort_keys=True), rowid),
                )
        connection.execute("DROP TABLE agent_home_migrations")
        connection.execute(
            "CREATE TABLE state_migrations("
            "ordinal INTEGER NOT NULL UNIQUE,"
            "migration_id TEXT NOT NULL PRIMARY KEY,"
            "checksum TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO state_migrations VALUES (1, 'development_baseline', ?)",
            ("a" * 64,),
        )
        if shape in {"preproduction_routines", "preproduction_inbox"}:
            connection.execute("DROP TABLE effect_receipts")
            connection.execute("DROP TABLE relational_write_scopes")
            connection.execute(
                "CREATE TABLE database_write_receipts("
                "agent_id TEXT NOT NULL, id TEXT NOT NULL, run_id TEXT NOT NULL,"
                "call_id TEXT NOT NULL, data TEXT NOT NULL,"
                "PRIMARY KEY(agent_id, id), UNIQUE(agent_id, run_id, call_id))"
            )
            connection.execute(
                "CREATE TABLE postgresql_update_scopes("
                "agent_id TEXT NOT NULL, source_id TEXT NOT NULL,"
                "resource_id TEXT NOT NULL,"
                "authorization_fingerprint TEXT NOT NULL, data TEXT NOT NULL,"
                "PRIMARY KEY(agent_id, source_id, resource_id),"
                "FOREIGN KEY(agent_id, source_id) REFERENCES sources(agent_id, id) "
                "ON DELETE CASCADE)"
            )
        if shape == "preproduction_inbox":
            connection.execute("DROP TABLE routine_occurrences")
            connection.execute("DROP TABLE scheduled_routines")
            connection.execute("DROP TABLE deliveries")
            connection.execute("DROP TABLE mcp_server_bindings")
            connection.execute(
                "CREATE TABLE mcp_server_bindings("
                "agent_id TEXT NOT NULL, binding_id TEXT NOT NULL,"
                "data TEXT NOT NULL, PRIMARY KEY(agent_id, binding_id))"
            )
            connection.execute(
                "CREATE TABLE conversation_inbox("
                "agent_id TEXT NOT NULL, delivery_id TEXT NOT NULL,"
                "conversation_id TEXT NOT NULL, subject_kind TEXT NOT NULL,"
                "subject_id TEXT NOT NULL, logical_key TEXT NOT NULL,"
                "data TEXT NOT NULL, PRIMARY KEY(agent_id, delivery_id),"
                "UNIQUE(agent_id, logical_key),"
                "UNIQUE(agent_id, subject_kind, subject_id))"
            )


async def test_generic_engine_upgrades_database_and_owned_file_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    before_memory = (home / "MEMORY.md").read_text(encoding="utf-8")
    migration = _synthetic_next_migration()
    _patch_next_migration(monkeypatch, migration)

    result = coordinator.upgrade_agent_home(
        home,
        validate_home=_validate_synthetic_home,
    )

    assert result.source_revision == 2
    assert result.target_revision == 3
    assert result.upgraded
    assert _journal(home / "state.db") == (
        (1, REVISION_1.migration_id, REVISION_1.checksum),
        (2, REVISION_2.migration_id, REVISION_2.checksum),
        (3, migration.migration_id, migration.checksum),
    )
    assert (home / "MEMORY.md").read_text(encoding="utf-8") == (
        before_memory + "revision three\n"
    )
    assert result.rollback_path is not None
    assert (result.rollback_path / "state.db").is_file()
    assert (result.rollback_path / "MEMORY.md").is_file()


async def test_disk_preflight_refuses_before_creating_upgrade_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    before = {name: _sha256(home / name) for name in ("state.db", "MEMORY.md")}
    _patch_next_migration(monkeypatch, _synthetic_next_migration())

    class NoFreeSpace:
        free = 0

    monkeypatch.setattr(coordinator.shutil, "disk_usage", lambda _path: NoFreeSpace())

    with pytest.raises(StateCompatibilityError) as captured:
        coordinator.upgrade_agent_home(home, validate_home=_validate_synthetic_home)

    assert captured.value.code is StateCompatibilityCode.UPGRADE_FAILED
    assert {name: _sha256(home / name) for name in before} == before
    assert not (home / ".home-upgrade").exists()


async def test_failed_staged_home_migration_leaves_active_home_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    before = {name: _sha256(home / name) for name in ("state.db", "MEMORY.md")}
    _patch_next_migration(
        monkeypatch,
        _synthetic_next_migration(apply=_synthetic_failure),
    )

    with pytest.raises(StateCompatibilityError) as captured:
        coordinator.upgrade_agent_home(home, validate_home=_validate_synthetic_home)

    assert captured.value.code is StateCompatibilityCode.UPGRADE_FAILED
    assert {name: _sha256(home / name) for name in before} == before
    assert (home / ".home-upgrade/journal.json").is_file()
    assert not (home / ".home-rollbacks").exists()


async def test_publish_failure_restores_the_complete_source_then_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    before = _durable_snapshot(home)
    before_journal = _journal(home / "state.db")
    migration = _synthetic_next_migration()
    _patch_next_migration(monkeypatch, migration)
    original_publish = coordinator._atomic_publish
    failed = False

    def fail_database_publish_once(source: Path, destination: Path) -> None:
        nonlocal failed
        if destination.name == "state.db" and not failed:
            failed = True
            raise OSError("controlled publish failure")
        original_publish(source, destination)

    monkeypatch.setattr(coordinator, "_atomic_publish", fail_database_publish_once)

    with pytest.raises(StateCompatibilityError) as captured:
        coordinator.upgrade_agent_home(home, validate_home=_validate_synthetic_home)

    assert captured.value.code is StateCompatibilityCode.UPGRADE_FAILED
    assert failed
    assert _durable_snapshot(home) == before
    assert _journal(home / "state.db") == before_journal

    result = coordinator.upgrade_agent_home(
        home,
        validate_home=_validate_synthetic_home,
    )
    assert result.recovered
    assert result.upgraded
    assert _journal(home / "state.db")[-1][0] == 3


@pytest.mark.parametrize("phase", ("prepared", "committing", "committed"))
async def test_upgrade_recovers_after_each_durable_commit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    migration = _synthetic_next_migration()
    _patch_next_migration(monkeypatch, migration)

    def crash(observed: str) -> None:
        if observed == phase:
            raise RuntimeError(f"simulated crash at {phase}")

    with pytest.raises(StateCompatibilityError):
        coordinator.upgrade_agent_home(
            home,
            validate_home=_validate_synthetic_home,
            phase_hook=crash,
        )
    assert (home / ".home-upgrade/journal.json").is_file()

    recovered = coordinator.upgrade_agent_home(
        home,
        validate_home=_validate_synthetic_home,
    )

    assert recovered.recovered
    assert _journal(home / "state.db")[-1] == (
        3,
        migration.migration_id,
        migration.checksum,
    )
    assert not (home / ".home-upgrade").exists()
    assert len(tuple((home / ".home-rollbacks").iterdir())) == 1


async def test_recovery_finishes_a_partially_published_whole_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    migration = _synthetic_next_migration()
    _patch_next_migration(monkeypatch, migration)

    def stop_prepared(observed: str) -> None:
        if observed == "prepared":
            raise RuntimeError("stop before publish")

    with pytest.raises(StateCompatibilityError):
        coordinator.upgrade_agent_home(
            home,
            validate_home=_validate_synthetic_home,
            phase_hook=stop_prepared,
        )
    upgrade = home / ".home-upgrade"
    shutil.copyfile(upgrade / "stage/MEMORY.md", home / "MEMORY.md")
    journal_path = upgrade / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["phase"] = "committing"
    journal_path.write_text(
        json.dumps(journal, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )

    result = coordinator.upgrade_agent_home(
        home,
        validate_home=_validate_synthetic_home,
    )

    assert result.recovered
    assert _journal(home / "state.db")[-1][0] == 3
    assert not upgrade.exists()


@pytest.mark.parametrize(
    ("mutation", "found"),
    (
        (
            "UPDATE agent_home_migrations SET checksum = '0' WHERE revision = 1",
            "1",
        ),
        (
            (
                "UPDATE agent_home_migrations SET migration_id = 'unknown' "
                "WHERE revision = 1"
            ),
            "unknown",
        ),
        (
            "UPDATE agent_home_migrations SET revision = 3 WHERE revision = 2",
            "3",
        ),
    ),
)
async def test_changed_release_journal_is_refused_without_write(
    tmp_path: Path,
    mutation: str,
    found: str,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    path = home / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))

    assert raised.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
    assert raised.value.found_revision == found
    assert _sha256(path) == before


async def test_newer_home_revision_is_a_downgrade_refusal_without_write(
    tmp_path: Path,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    path = home / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO agent_home_migrations VALUES (3, 'future_revision', ?)",
            ("f" * 64,),
        )
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))

    assert raised.value.code is StateCompatibilityCode.NEWER_REVISION
    assert raised.value.found_revision == "3"
    assert _sha256(path) == before


@pytest.mark.parametrize(
    "mutation",
    (
        "DROP TABLE syncs",
        "ALTER TABLE runs ADD COLUMN future_value TEXT",
        "DROP INDEX runs_conversation_turn",
        "CREATE VIEW unexpected_view AS SELECT key FROM metadata",
    ),
)
async def test_damaged_current_home_fails_before_any_write(
    tmp_path: Path,
    mutation: str,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    path = home / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert _sha256(path) == before


@pytest.mark.parametrize(
    "mutation",
    (
        "UPDATE messages SET data = '{}' WHERE run_id = 'run-preserved'",
        "UPDATE runs SET result = '{}' WHERE id = 'run-preserved'",
    ),
)
async def test_corrupt_current_records_fail_before_any_write(
    tmp_path: Path,
    mutation: str,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    path = home / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))

    assert raised.value.code is StateCompatibilityCode.DAMAGED
    assert _sha256(path) == before


async def test_newer_unfinished_upgrade_is_refused_without_write(
    tmp_path: Path,
) -> None:
    home, _ = await _create_rich_home(tmp_path)
    upgrade = home / ".home-upgrade"
    upgrade.mkdir()
    journal = {
        "files": [],
        "kind": "daita_agent_home_upgrade",
        "migrations": [],
        "operation_id": "a" * 32,
        "phase": "staging",
        "source_kind": "production",
        "source_revision": 2,
        "target_revision": 3,
    }
    (upgrade / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    before = _sha256(home / "state.db")

    with pytest.raises(StateCompatibilityError) as raised:
        await Agent.open("atlas", root=tmp_path, workspace=workspace_for(tmp_path))

    assert raised.value.code is StateCompatibilityCode.NEWER_REVISION
    assert _sha256(home / "state.db") == before
    assert upgrade.is_dir()


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
    "shape",
    ("preproduction_current", "preproduction_routines", "preproduction_inbox"),
)
async def test_revision_1_bridge_preserves_complete_observed_preproduction_homes(
    tmp_path: Path,
    shape: str,
) -> None:
    home, agent_id = await _create_rich_home(tmp_path)
    expected = _durable_snapshot(home)
    _make_preproduction_home(home, shape)
    expected_config = expected["config"]
    assert isinstance(expected_config, dict)
    expected_route = expected_config["model_route"]
    assert isinstance(expected_route, dict)
    expected_retry = expected_route["retry_policy"]
    assert isinstance(expected_retry, dict)
    expected_retry["max_total_attempts"] = 2

    reopened = await Agent.open(
        "atlas",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    try:
        assert reopened.id == agent_id
        assert await reopened.read_memory() == "Durable memory."
        assert await reopened.read_user_profile() == "Durable user profile."
        skill = await reopened.read_skill("preserved-skill")
        assert skill is not None
        assert skill.instructions == "Read the current catalog first."
        assert len(await reopened.list_sources()) == 1
        transcript = await reopened.transcript("run-preserved")
        assert transcript.messages[-1].content == (TextBlock("Preserved answer."),)
    finally:
        await reopened.close()

    assert _durable_snapshot(home) == expected
    assert _journal(home / "state.db") == (
        (1, REVISION_1.migration_id, REVISION_1.checksum),
        (2, REVISION_2.migration_id, REVISION_2.checksum),
    )
    assert not (home / ".home-upgrade").exists()
    assert len(tuple((home / ".home-rollbacks").iterdir())) == 1


def test_headless_cli_reports_revision_status_and_safe_failures(tmp_path: Path) -> None:
    created = asyncio.run(
        Agent.create("status", root=tmp_path, workspace=workspace_for(tmp_path))
    )
    path = created.home / "state.db"
    asyncio.run(created.close())
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(["--root", str(tmp_path), "state", "status", "status"])

    assert code == 0
    assert stderr.getvalue() == ""
    status = json.loads(stdout.getvalue())
    assert status == {
        "current_revision": 2,
        "found_revision": 2,
        "minimum_supported_revision": 1,
        "recovery_required": False,
        "source_kind": "production",
        "upgrade_required": False,
    }

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE agent_home_migrations SET migration_id = 'unknown' "
            "WHERE revision = 1"
        )
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(["--root", str(tmp_path), "sources", "status"])

    assert code == 1
    assert stdout.getvalue() == ""
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "state_revision_unsupported"
    assert error["current_revision"] == "2"
    assert error["found_revision"] == "unknown"
    assert error["state_changed"] is False
    assert error["state_path"] == str(path)


def test_interactive_tui_renders_human_safe_upgrade_diagnostic(tmp_path: Path) -> None:
    created = asyncio.run(
        Agent.create("unsupported", root=tmp_path, workspace=workspace_for(tmp_path))
    )
    path = created.home / "state.db"
    asyncio.run(created.close())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE agent_home_migrations SET migration_id = 'unknown' "
            "WHERE revision = 1"
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
    assert "current_revision" not in rendered


def test_released_home_migration_checksum_binds_its_exact_definition() -> None:
    def alternate_apply(staged_home: Path, source_shape: str | None) -> None:
        del staged_home, source_shape

    variants = (
        replace(REVISION_1, revision=2),
        replace(REVISION_1, migration_id="alternate"),
        replace(REVISION_1, definition="alternate definition"),
        replace(REVISION_1, affected_paths=("state.db",)),
        replace(REVISION_1, apply=alternate_apply),
        replace(REVISION_1, implementation_material=("alternate",)),
    )

    assert len(REVISION_1.checksum) == 64
    assert all(variant.checksum != REVISION_1.checksum for variant in variants)


def test_runtime_has_one_home_revision_owner_and_no_legacy_version_gates() -> None:
    production = REPO_ROOT / "src" / "daita"
    assert not (production / "storage" / "sqlite_migrations").exists()
    assert {
        path.name for path in (production / "storage/home_migrations").glob("*.py")
    } == {
        "__init__.py",
        "baseline.py",
        "models.py",
        "registry.py",
        "revision_0001.py",
        "revision_0001_schema.py",
        "revision_0002.py",
        "revision_0002_conversion.py",
        "revision_0002_legacy_autonomy.py",
        "revision_0002_legacy_autonomy_codecs.py",
        "revision_0002_legacy_delivery.py",
        "revision_0002_legacy_job_codecs.py",
        "revision_0002_legacy_jobs.py",
    }
    assert HOME_MIGRATIONS == (REVISION_1, REVISION_2)
    current_code = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (production / "storage/sqlite_codecs").glob("*.py")
    )
    assert 'fields["version"]' not in current_code
    assert "version is unsupported" not in current_code
    historical = (production / "storage/home_migrations/revision_0001.py").read_text(
        encoding="utf-8"
    )
    assert "development_baseline" in historical
    assert 'fields["version"]' in historical


def test_revision_2_home_reopens_without_importing_revision_1_job_decoders(
    tmp_path: Path,
) -> None:
    agent = asyncio.run(
        Agent.create("current-home", root=tmp_path, workspace=workspace_for(tmp_path))
    )
    asyncio.run(agent.close())
    code = """
import asyncio
import sys
from pathlib import Path
from daita import Agent
from tests.support.workspace import workspace_for

async def main():
    root = Path(sys.argv[1])
    agent = await Agent.open(
        "current-home", root=root, workspace=workspace_for(root)
    )
    await agent.close()
    forbidden = sorted(
        name for name in sys.modules
        if name.endswith("revision_0002_conversion")
        or ".revision_0002_legacy" in name
    )
    if forbidden:
        raise AssertionError(forbidden)

asyncio.run(main())
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "src"), str(REPO_ROOT))
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
