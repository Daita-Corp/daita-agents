from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from daita.errors import StateCompatibilityError
from daita.storage.draft_graph_schema import (
    DRAFT_GRAPH_NAMED_INDEXES,
    DRAFT_GRAPH_TABLE_NAMES,
    connect_draft_graph,
    initialize_draft_graph_database,
    require_draft_graph_schema,
)
from daita.storage.home_migrations import CURRENT_HOME_REVISION, HOME_MIGRATIONS
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_schema import CURRENT_SCHEMA

pytestmark = pytest.mark.contract


def test_draft_schema_is_complete_but_revision_one_contract_is_unchanged(
    tmp_path: Path,
):
    path = tmp_path / "draft.db"
    initialize_draft_graph_database(path)

    with connect_draft_graph(path, read_only=True) as connection:
        require_draft_graph_schema(connection)
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert DRAFT_GRAPH_TABLE_NAMES <= tables
        assert "autonomous_followups" not in tables
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone() == (1000,)

    assert CURRENT_HOME_REVISION == 1
    assert tuple(item.revision for item in HOME_MIGRATIONS) == (1,)
    assert "job_graphs" not in CURRENT_SCHEMA.tables
    assert "autonomous_followups" in CURRENT_SCHEMA.tables


def test_ready_and_stale_access_paths_are_covered_by_named_indexes(tmp_path: Path):
    path = tmp_path / "draft.db"
    initialize_draft_graph_database(path)

    with connect_draft_graph(path, read_only=True) as connection:
        assert DRAFT_GRAPH_NAMED_INDEXES["job_tasks_ready"] == (
            "agent_id",
            "state",
            "not_before_us",
            "priority",
            "updated_at_us",
        )
        assert DRAFT_GRAPH_NAMED_INDEXES["job_task_attempts_stale"] == (
            "agent_id",
            "state",
            "lease_expires_at_us",
        )
        ready_plan = " ".join(
            str(item)
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT task_id FROM job_tasks "
                "WHERE agent_id = ? AND state = 'ready' AND not_before_us IS NULL",
                ("agent-1",),
            )
            for item in row
        )
        stale_plan = " ".join(
            str(item)
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT attempt_id FROM job_task_attempts "
                "WHERE agent_id = ? AND state IN ('claimed','running') "
                "AND lease_expires_at_us <= ?",
                ("agent-1", 1),
            )
            for item in row
        )
        assert "job_tasks_ready" in ready_plan
        assert "job_task_attempts_stale" in stale_plan


@pytest.mark.asyncio
async def test_current_store_rejects_a_draft_database(tmp_path: Path):
    path = tmp_path / "draft.db"
    initialize_draft_graph_database(path)

    with pytest.raises(StateCompatibilityError):
        await SQLiteStateStore.open(path)

    connection = sqlite3.connect(path)
    try:
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert "job_graphs" in names
