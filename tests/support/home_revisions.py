"""Rebuild released agent-home schemas from a current home."""

from __future__ import annotations

import sqlite3

from daita.storage.graph_schema import GRAPH_RECEIPT_SQL, GRAPH_TABLE_SQL

_RECEIPT_COLUMNS = (
    "agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, "
    "grant_digest, unresolved, job_id, data"
)


def undo_revision_4_schema(connection: sqlite3.Connection) -> None:
    """Restore revision 3's tables; callers own record and ledger edits.

    Receipts keep their rows; a current home's jobs have no revision-3 form.
    """

    if connection.execute("SELECT 1 FROM jobs").fetchone() is not None:
        raise ValueError("a home with checklist jobs has no revision-3 form")
    receipts = connection.execute(
        f"SELECT {_RECEIPT_COLUMNS} FROM effect_receipts"
    ).fetchall()
    connection.execute("DROP TABLE effect_receipts")
    connection.execute("DROP TABLE jobs")
    connection.execute("DROP TABLE analysis_evidence")
    connection.execute("DROP TABLE catalog_links")
    connection.execute("""CREATE TABLE learning_candidates (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
)""")
    connection.executescript(GRAPH_RECEIPT_SQL + ";\n" + GRAPH_TABLE_SQL)
    connection.executemany(
        f"INSERT INTO effect_receipts({_RECEIPT_COLUMNS}) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        receipts,
    )
