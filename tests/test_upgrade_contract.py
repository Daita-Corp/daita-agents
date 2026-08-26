from __future__ import annotations

from _workspace_support import workspace_for

import asyncio
import hashlib
import io
import json
import re
import sqlite3
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from daita import Agent, cli
from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.storage import sqlite as sqlite_store
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_migrations import (
    DEVELOPMENT_BASELINE,
    migration_rows,
    runner as migration_runner,
)
from daita.storage.sqlite_schema import CURRENT_TABLES


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path) -> tuple[tuple[int, str, str], ...]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(
                "SELECT ordinal, migration_id, checksum "
                "FROM state_migrations ORDER BY ordinal"
            )
        )


def _synthetic_next_apply(connection: sqlite3.Connection) -> None:
    connection.execute("SELECT 1")


def _synthetic_failure(connection: sqlite3.Connection) -> None:
    connection.execute("SELECT 1")
    raise RuntimeError("controlled migration failure")


def _synthetic_next_migration():
    return replace(
        DEVELOPMENT_BASELINE,
        ordinal=2,
        migration_id="test_only_next_revision",
        definition="test-only next revision",
        source_schema=CURRENT_TABLES,
        target_schema=CURRENT_TABLES,
        apply=_synthetic_next_apply,
    )


def _patch_next_migration(monkeypatch: pytest.MonkeyPatch, migration) -> None:
    migrations = (DEVELOPMENT_BASELINE, migration)
    monkeypatch.setattr(migration_runner, "MIGRATIONS", migrations)
    monkeypatch.setattr(sqlite_store, "MIGRATIONS", migrations)


async def test_generic_engine_upgrades_a_known_prefix_on_a_staged_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    original_journal = _journal(path)
    migration = _synthetic_next_migration()
    _patch_next_migration(monkeypatch, migration)

    reopened = await SQLiteStateStore.open(path)
    await reopened.close()

    assert _journal(path) == (
        original_journal[0],
        (migration.ordinal, migration.migration_id, migration.checksum),
    )
    rollback_points = tuple(tmp_path.glob("state.db.rollback-*"))
    assert len(rollback_points) == 1
    assert _journal(rollback_points[0]) == original_journal


async def test_failed_staged_migration_leaves_active_database_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    before = _sha256(path)
    migration = replace(_synthetic_next_migration(), apply=_synthetic_failure)
    _patch_next_migration(monkeypatch, migration)

    with pytest.raises(StateCompatibilityError) as captured:
        await SQLiteStateStore.open(path)

    assert captured.value.code is StateCompatibilityCode.UPGRADE_FAILED
    assert captured.value.to_mapping()["state_changed"] is False
    assert _sha256(path) == before
    assert tuple(tmp_path.glob("state.db.rollback-*")) == ()
    assert tuple(tmp_path.glob(".state.db.*.db")) == ()


@pytest.mark.parametrize(
    ("mutation", "found"),
    (
        (
            "UPDATE state_migrations SET checksum = '0' WHERE ordinal = 1",
            "development_baseline",
        ),
        (
            "UPDATE state_migrations SET migration_id = 'unknown' WHERE ordinal = 1",
            "unknown",
        ),
        (
            "UPDATE state_migrations SET ordinal = 2 WHERE ordinal = 1",
            "development_baseline",
        ),
    ),
)
async def test_changed_baseline_journal_is_refused_without_write(
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


async def test_newer_journal_extension_is_a_downgrade_refusal_without_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    future_revision = "future_revision"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO state_migrations(ordinal, migration_id, checksum) "
            "VALUES (?, ?, ?)",
            (len(migration_rows()) + 1, future_revision, "f" * 64),
        )
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.NEWER_REVISION
    assert raised.value.found_revision == future_revision
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
            "VALUES (?, 'gapped-future', ?)",
            (len(migration_rows()) + 2, "f" * 64),
        )
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as raised:
        await SQLiteStateStore.open(path)

    assert raised.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
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


async def test_state_without_the_current_baseline_is_damaged_without_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated(id TEXT PRIMARY KEY)")
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


def test_headless_cli_returns_descriptive_revision_diagnostic(tmp_path: Path) -> None:
    created = asyncio.run(
        Agent.create("unsupported", root=tmp_path, workspace=workspace_for(tmp_path))
    )
    path = created.home / "state.db"
    asyncio.run(created.close())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE state_migrations SET migration_id = 'unknown' WHERE ordinal = 1"
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
    assert error["found_revision"] == "unknown"
    assert error["state_changed"] is False
    assert error["state_path"] == str(path)


def test_interactive_tui_renders_human_safe_upgrade_diagnostic(
    tmp_path: Path,
) -> None:
    created = asyncio.run(
        Agent.create("unsupported", root=tmp_path, workspace=workspace_for(tmp_path))
    )
    path = created.home / "state.db"
    asyncio.run(created.close())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE state_migrations SET migration_id = 'unknown' WHERE ordinal = 1"
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


def test_development_baseline_checksum_binds_its_exact_definition() -> None:
    def alternate_apply(connection: sqlite3.Connection) -> None:
        connection.execute("SELECT 2")

    variants = (
        replace(DEVELOPMENT_BASELINE, ordinal=2),
        replace(DEVELOPMENT_BASELINE, migration_id="alternate"),
        replace(DEVELOPMENT_BASELINE, definition="alternate definition"),
        replace(DEVELOPMENT_BASELINE, source_schema={}),
        replace(DEVELOPMENT_BASELINE, target_schema={}),
        replace(DEVELOPMENT_BASELINE, apply=alternate_apply),
    )

    assert re.fullmatch(r"[0-9a-f]{64}", DEVELOPMENT_BASELINE.checksum)
    assert all(
        variant.checksum != DEVELOPMENT_BASELINE.checksum for variant in variants
    )


def test_preproduction_tree_has_no_unreleased_compatibility_history() -> None:
    production = Path(__file__).parents[1] / "src" / "daita"
    migration_files = {
        path.name
        for path in (production / "storage" / "sqlite_migrations").glob("*.py")
    }
    assert migration_files == {
        "__init__.py",
        "baseline.py",
        "models.py",
        "runner.py",
    }
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in production.rglob("*.py")
    )
    for obsolete in (
        "decode_preledger_source",
        "PreledgerShape",
        "_decode_tool_v1",
        "_migrate_v1_to_v2",
        "_migrate_v2_to_v3",
        "20260810_database_write_receipts",
        "20260811_postgresql_write_admission",
        "20260812_scoped_source_permissions",
        "20260814_generalized_postgresql_updates",
        "20260819_mcp_server_bindings",
    ):
        assert obsolete not in text
