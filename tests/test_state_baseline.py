from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_migrations import (
    CURRENT_REVISION,
    DEVELOPMENT_BASELINE,
    migration_rows,
)
from daita.storage.sqlite_schema import CURRENT_TABLES, table_names


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


async def test_fresh_state_uses_one_current_development_baseline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        assert table_names(connection) == set(CURRENT_TABLES)
    assert migration_rows() == (
        (
            1,
            "development_baseline",
            DEVELOPMENT_BASELINE.checksum,
        ),
    )
    assert CURRENT_REVISION == "development_baseline"
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


async def test_changed_development_baseline_is_rejected_without_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE state_migrations SET checksum = ?", ("0" * 64,))
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as captured:
        await SQLiteStateStore.open(path)

    assert captured.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
    assert _sha256(path) == before
    assert tuple(tmp_path.glob("state.db.rollback-*")) == ()
