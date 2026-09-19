"""Exact current SQLite schema and conversation grouping contract."""

from __future__ import annotations

import sqlite3

from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_schema import CURRENT_SCHEMA


async def test_current_state_schema_and_conversation_index_are_exact(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {
            table: tuple(
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            )
            for table in tables
        }
        named_indexes = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT name, tbl_name FROM sqlite_master "
                "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
            )
        }

        assert tables == set(CURRENT_SCHEMA.tables)
        assert columns == {
            table: tuple(str(column[0]) for column in definitions)
            for table, definitions in CURRENT_SCHEMA.tables.items()
        }
        assert named_indexes == {
            name: definition[0]
            for name, definition in CURRENT_SCHEMA.named_indexes.items()
        }
        assert "job_graphs" in tables
        assert "job_task_attempts" in tables
        assert "job_task_results" in tables
        assert "autonomous_followups" not in tables
        run_indexes = {
            str(row[1]): bool(row[2])
            for row in connection.execute("PRAGMA index_list(runs)")
            if not str(row[1]).startswith("sqlite_autoindex")
        }
        assert run_indexes == {"runs_conversation_turn": True}
        index_columns = tuple(
            str(row[2])
            for row in connection.execute("PRAGMA index_info(runs_conversation_turn)")
        )
        assert index_columns == ("agent_id", "conversation_id", "turn_index")
