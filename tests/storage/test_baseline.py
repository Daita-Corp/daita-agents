from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.storage.home_migrations import (
    CURRENT_HOME_REVISION,
    HOME_MIGRATIONS,
    migration_rows,
)
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_schema import CURRENT_SCHEMA, table_names


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


async def test_fresh_state_uses_the_current_production_home_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        assert table_names(connection) == set(CURRENT_SCHEMA.tables)
    assert CURRENT_HOME_REVISION == 2
    assert SQLiteStateStore.current_revision == "2"
    assert migration_rows() == (
        (1, "agent_home_revision_1", HOME_MIGRATIONS[0].checksum),
        (2, "agent_home_revision_2", HOME_MIGRATIONS[1].checksum),
    )
    assert _journal(path) == migration_rows()


async def test_current_state_open_is_validation_only(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    before = _sha256(path)

    reopened = await SQLiteStateStore.open(path)
    await reopened.close()

    assert _sha256(path) == before
    assert tuple(tmp_path.glob("state.db.rollback-*")) == ()


async def test_changed_released_checksum_is_rejected_without_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE agent_home_migrations SET checksum = ? WHERE revision = 1",
            ("0" * 64,),
        )
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as captured:
        await SQLiteStateStore.open(path)

    assert captured.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
    assert _sha256(path) == before
