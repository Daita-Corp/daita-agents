"""Exact current SQLite schema and conversation grouping contract."""

from __future__ import annotations

import sqlite3

from daita.storage.sqlite import SQLiteStateStore


async def test_current_state_schema_and_conversation_index_are_exact(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {
            table: tuple(
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            )
            for table in tables
        }
        named_indexes = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT name, tbl_name FROM sqlite_master "
                "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
            )
        }

        assert tables == {
            "effect_receipts",
            "learning_candidates",
            "job_runs",
            "autonomous_followups",
            "deliveries",
            "mcp_server_bindings",
            "messages",
            "metadata",
            "relational_write_scopes",
            "runs",
            "routine_occurrences",
            "scheduled_routines",
            "semantic_annotations",
            "snapshots",
            "source_read_scopes",
            "sources",
            "state_migrations",
            "syncs",
        }
        assert columns == {
            "effect_receipts": (
                "agent_id",
                "id",
                "run_id",
                "call_id",
                "operation_key",
                "routine_id",
                "occurrence_id",
                "grant_digest",
                "unresolved",
                "data",
            ),
            "learning_candidates": ("agent_id", "id", "data"),
            "job_runs": ("agent_id", "job_id", "data"),
            "autonomous_followups": (
                "agent_id",
                "followup_id",
                "job_id",
                "event_id",
                "data",
            ),
            "deliveries": (
                "agent_id",
                "delivery_id",
                "conversation_id",
                "subject_kind",
                "subject_id",
                "logical_key",
                "target_kind",
                "target_fingerprint",
                "state",
                "created_at_us",
                "data",
            ),
            "messages": ("run_id", "position", "data"),
            "metadata": ("key", "data"),
            "mcp_server_bindings": ("agent_id", "binding_id", "data"),
            "relational_write_scopes": (
                "agent_id",
                "source_id",
                "resource_id",
                "authorization_fingerprint",
                "data",
            ),
            "runs": (
                "id",
                "agent_id",
                "conversation_id",
                "turn_index",
                "input",
                "result",
            ),
            "routine_occurrences": (
                "agent_id",
                "occurrence_id",
                "routine_id",
                "routine_revision",
                "slot_key",
                "state",
                "lease_expires_at_us",
                "reserved_run_id",
                "data",
            ),
            "scheduled_routines": (
                "agent_id",
                "routine_id",
                "conversation_id",
                "state",
                "next_due_at_us",
                "data",
            ),
            "semantic_annotations": ("agent_id", "id", "data"),
            "snapshots": ("agent_id", "source_id", "sync_id", "data"),
            "source_read_scopes": ("agent_id", "source_id", "data"),
            "sources": ("agent_id", "id", "data"),
            "state_migrations": ("ordinal", "migration_id", "checksum"),
            "syncs": ("agent_id", "id", "source_id", "data"),
        }
        assert named_indexes == {
            "effect_receipts_unresolved": "effect_receipts",
            "effect_receipts_grant_reservations": "effect_receipts",
            "deliveries_conversation_history": "deliveries",
            "routine_occurrences_stale": "routine_occurrences",
            "runs_conversation_turn": "runs",
            "scheduled_routines_due": "scheduled_routines",
        }
        run_indexes = {
            row[1]: bool(row[2])
            for row in connection.execute("PRAGMA index_list(runs)")
            if not str(row[1]).startswith("sqlite_autoindex")
        }
        assert run_indexes == {"runs_conversation_turn": True}
        index_columns = tuple(
            row[2]
            for row in connection.execute("PRAGMA index_info(runs_conversation_turn)")
        )
        assert index_columns == ("agent_id", "conversation_id", "turn_index")
