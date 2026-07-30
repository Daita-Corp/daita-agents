from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import pytest

from daita import Agent
from daita.storage.sqlite import SQLiteStateStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _v1_database(path: Path) -> None:
    store = await SQLiteStateStore.open(path)
    await store.close()


async def test_existing_v1_state_is_admitted_without_writing(tmp_path: Path):
    path = tmp_path / "state.db"
    await _v1_database(path)
    before = _sha256(path)

    store = await SQLiteStateStore.open(path)
    await store.close()

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
async def test_incompatible_existing_state_fails_before_any_write(
    tmp_path: Path,
    mutation: str,
):
    path = tmp_path / "state.db"
    await _v1_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = _sha256(path)

    with pytest.raises(
        RuntimeError,
        match="not compatible.*agent home was preserved",
    ):
        await SQLiteStateStore.open(path)

    assert _sha256(path) == before


async def test_existing_empty_state_is_not_initialized_in_place(tmp_path: Path):
    path = tmp_path / "state.db"
    path.touch()
    before = path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match="not compatible.*agent home was preserved",
    ):
        await SQLiteStateStore.open(path)

    assert path.read_bytes() == before


async def test_agent_open_preserves_incompatible_home_and_releases_writer_lock(
    tmp_path: Path,
):
    agent = await Agent.create("incompatible-upgrade", root=tmp_path)
    home = agent.home
    await agent.close()
    state_path = home / "state.db"
    manifest_before = (home / "agent.toml").read_bytes()
    with sqlite3.connect(state_path) as connection:
        connection.execute("DROP INDEX runs_conversation_turn")
    state_before = state_path.read_bytes()

    for _attempt in range(2):
        with pytest.raises(
            RuntimeError,
            match="not compatible.*agent home was preserved",
        ):
            await Agent.open("incompatible-upgrade", root=tmp_path)
        assert (home / "agent.toml").read_bytes() == manifest_before
        assert state_path.read_bytes() == state_before
