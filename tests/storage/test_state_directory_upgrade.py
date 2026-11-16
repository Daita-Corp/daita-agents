"""A bare state directory upgrades through the same migrations as a local home."""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

import daita.hosting.home_upgrade as home_upgrade
from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.hosting.home_upgrade import upgrade_state_directory
from daita.storage.home_migrations import (
    HOME_MIGRATIONS,
    insert_migration_row,
    revision_database,
)
from daita.storage.schema_contract import require_schema
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.integration

UPGRADE = REPO_ROOT / "tests/fixtures/state-upgrades/revision-3-to-4"


def _materialize(fixture: Path, directory: Path) -> None:
    directory.mkdir()
    with closing(sqlite3.connect(directory / "state.db")) as connection:
        connection.executescript((fixture / "state.sql").read_text(encoding="utf-8"))
    shutil.copyfile(fixture / "config.json", directory / "config.json")


def _records(connection: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    tables = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {
        table: sorted(connection.execute(f'SELECT * FROM "{table}"'), key=repr)
        for (table,) in tables
    }


@pytest.mark.parametrize("migration", HOME_MIGRATIONS, ids=lambda item: item.revision)
def test_each_revision_database_creates_the_schema_its_migration_checks(
    migration,
) -> None:
    sql, schema = revision_database(migration.revision)
    assert schema == migration.target_schema
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(sql)
        require_schema(connection, schema)


@pytest.mark.parametrize("revision", (0, len(HOME_MIGRATIONS) + 1, True, "1"))
def test_an_unknown_revision_has_no_database(revision) -> None:
    with pytest.raises(ValueError, match="unknown agent-home revision"):
        revision_database(revision)


def _export(store: sqlite3.Connection, directory: Path, revision: int) -> None:
    """The documented round trip's export, with a SQLite database as the store."""

    sql, _ = revision_database(revision)
    directory.mkdir()
    with closing(sqlite3.connect(directory / "state.db")) as connection:
        connection.executescript(sql)
        for migration in HOME_MIGRATIONS[:revision]:
            insert_migration_row(connection, migration)
        for table, rows in _records(store).items():
            if table != "agent_home_migrations" and rows:
                marks = ", ".join("?" * len(rows[0]))
                connection.executemany(f'INSERT INTO "{table}" VALUES ({marks})', rows)
        connection.commit()


def test_revision_3_store_round_trips_to_the_published_records(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "state"
    with closing(sqlite3.connect(":memory:")) as store:
        store.executescript((UPGRADE / "before/state.sql").read_text(encoding="utf-8"))
        _export(store, directory, 3)
    shutil.copyfile(UPGRADE / "before/config.json", directory / "config.json")

    result = upgrade_state_directory(directory)

    assert (result.source_revision, result.target_revision) == (3, 4)
    assert result.upgraded and result.rollback_path is not None
    expected = sqlite3.connect(":memory:")
    with closing(expected), closing(sqlite3.connect(directory / "state.db")) as actual:
        expected.executescript(
            (UPGRADE / "after/state.sql").read_text(encoding="utf-8")
        )
        assert _records(actual) == _records(expected)
    assert json.loads((directory / "config.json").read_text()) == json.loads(
        (UPGRADE / "after/config.json").read_text()
    )
    assert not upgrade_state_directory(directory).upgraded


def test_a_migration_that_changes_another_home_file_is_refused_unstaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "state"
    _materialize(UPGRADE / "before", directory)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    latest = HOME_MIGRATIONS[-1]
    monkeypatch.setattr(
        home_upgrade,
        "HOME_MIGRATIONS",
        (
            *HOME_MIGRATIONS[:-1],
            replace(latest, affected_paths=(*latest.affected_paths, "MEMORY.md")),
        ),
    )

    with pytest.raises(StateCompatibilityError, match="MEMORY.md") as raised:
        upgrade_state_directory(directory)

    assert raised.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
