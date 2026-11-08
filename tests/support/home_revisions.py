"""Rebuild released agent-home schemas from a current home."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

from daita.catalog.models import SourceCatalogSnapshot
from daita.storage.graph_schema import GRAPH_RECEIPT_SQL, GRAPH_TABLE_SQL
from daita.storage.sqlite_codecs import (
    decode_catalog_entry,
    decode_catalog_relationship,
    decode_catalog_source_state,
    decode_catalog_sync,
    encode_catalog_snapshot,
)

_RECEIPT_COLUMNS = (
    "agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, "
    "grant_digest, unresolved, job_id, data"
)


_CATALOG_TABLES = (
    "catalog_search",
    "catalog_changes",
    "catalog_source_state",
    "catalog_relationships",
    "catalog_resources",
)
_SNAPSHOTS_SQL = """CREATE TABLE snapshots (
    agent_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    sync_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY(agent_id, source_id)
)"""


def catalog_snapshot_rows(
    connection: sqlite3.Connection,
    *,
    truncated_flag: bool = False,
) -> list[tuple[str, str, str, str]]:
    """Rebuild each source's revision-3 snapshot row from the per-resource tables."""

    rows = []
    for agent_id, source_id, encoded_state in connection.execute(
        "SELECT agent_id, source_id, data FROM catalog_source_state"
    ).fetchall():
        sync_id = decode_catalog_source_state(encoded_state).current_sync_id
        (encoded_sync,) = connection.execute(
            "SELECT data FROM syncs WHERE agent_id = ? AND id = ?",
            (agent_id, sync_id),
        ).fetchone()
        sync_record = json.loads(encoded_sync)
        # A caller may already have rewritten the sync as revision 3 stored it.
        sync_record["fields"].setdefault("truncated", False)
        entries = [
            decode_catalog_entry(data)
            for (data,) in connection.execute(
                "SELECT data FROM catalog_resources "
                "WHERE agent_id = ? AND source_id = ? ORDER BY id",
                (agent_id, source_id),
            )
        ]
        relationships = [
            replace(decode_catalog_relationship(data), sync_id=sync_id)
            for (data,) in connection.execute(
                "SELECT data FROM catalog_relationships "
                "WHERE agent_id = ? AND source_id = ? ORDER BY id",
                (agent_id, source_id),
            )
        ]
        snapshot = SourceCatalogSnapshot(
            sync=decode_catalog_sync(json.dumps(sync_record)),
            resources=tuple(
                replace(entry.resource, current_sync_id=sync_id) for entry in entries
            ),
            revisions=tuple(
                replace(entry.revision, sync_id=sync_id) for entry in entries
            ),
            facets=tuple(
                replace(facet, sync_id=sync_id)
                for entry in entries
                for facet in entry.facets
            ),
            relationships=tuple(relationships),
        )
        payload = json.loads(encode_catalog_snapshot(snapshot))
        # Revision 3 syncs predate the truncated flag.
        if not truncated_flag:
            del payload["fields"]["sync"]["fields"]["truncated"]
        rows.append(
            (
                agent_id,
                source_id,
                sync_id,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
        )
    return rows


def restore_catalog_snapshots(
    connection: sqlite3.Connection, *, truncated_flag: bool = False
) -> None:
    """Replace the per-resource catalog tables with revision 3's snapshots table.

    A caller that runs the snapshot conversion alone keeps the truncated flag,
    which the full upgrade adds before converting.
    """

    rows = catalog_snapshot_rows(connection, truncated_flag=truncated_flag)
    for table in _CATALOG_TABLES:
        connection.execute(f"DROP TABLE {table}")
    connection.execute(_SNAPSHOTS_SQL)
    connection.executemany(
        "INSERT INTO snapshots (agent_id, source_id, sync_id, data) "
        "VALUES (?, ?, ?, ?)",
        rows,
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
    restore_catalog_snapshots(connection)
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
