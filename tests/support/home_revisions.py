"""Rebuild released agent-home schemas from a current home."""

from __future__ import annotations

import sqlite3


def undo_revision_4_schema(connection: sqlite3.Connection) -> None:
    """Restore revision 3's tables; callers own record and ledger edits."""

    connection.execute("DROP TABLE analysis_evidence")
    connection.execute("DROP TABLE catalog_links")
    connection.execute("""CREATE TABLE learning_candidates (
    agent_id TEXT NOT NULL,
    id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, id)
)""")
