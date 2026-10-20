"""Add foreground analytical evidence, declared catalog links and sync truncation."""

import json
import sqlite3
from pathlib import Path

from ..sqlite_schema import (
    ANALYSIS_EVIDENCE_SQL,
    CATALOG_LINKS_SQL,
    SCHEMA_REVISION_4,
)
from .models import HomeMigration


def _dump(record: object) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _untruncated_sync(record: dict[str, object]) -> dict[str, object]:
    if record["__record__"] != "CatalogSync":
        raise ValueError("Historical catalog sync record is invalid")
    fields = record["fields"]
    assert isinstance(fields, dict)
    fields["truncated"] = False
    return record


def _apply_revision_4(home: Path, source_shape: str | None) -> None:
    if source_shape is not None:
        raise ValueError("Unknown analysis migration source")
    with sqlite3.connect(home / "state.db") as connection:
        connection.executescript(ANALYSIS_EVIDENCE_SQL + CATALOG_LINKS_SQL)
        for artifact_id, encoded in connection.execute(
            "SELECT artifact_id, data FROM artifacts"
        ):
            record = json.loads(encoded)
            if record["__record__"] != "ArtifactRecord":
                raise ValueError("Historical artifact record is invalid")
            record["fields"]["computation_evidence"] = {}
            connection.execute(
                "UPDATE artifacts SET data = ? WHERE artifact_id = ?",
                (_dump(record), artifact_id),
            )
        # Every historical discovery was complete: none was ever truncated.
        for agent_id, sync_id, encoded in connection.execute(
            "SELECT agent_id, id, data FROM syncs"
        ).fetchall():
            connection.execute(
                "UPDATE syncs SET data = ? WHERE agent_id = ? AND id = ?",
                (_dump(_untruncated_sync(json.loads(encoded))), agent_id, sync_id),
            )
        for agent_id, source_id, encoded in connection.execute(
            "SELECT agent_id, source_id, data FROM snapshots"
        ).fetchall():
            snapshot = json.loads(encoded)
            if snapshot["__record__"] != "SourceCatalogSnapshot":
                raise ValueError("Historical catalog snapshot record is invalid")
            _untruncated_sync(snapshot["fields"]["sync"])
            connection.execute(
                "UPDATE snapshots SET data = ? WHERE agent_id = ? AND source_id = ?",
                (_dump(snapshot), agent_id, source_id),
            )
        connection.commit()


REVISION_4 = HomeMigration(
    revision=4,
    migration_id="0004_foreground_analysis_evidence",
    definition=(
        "Add bounded run-owned child, cell and native cleanup evidence; historical records never authorize computation. "
        "Native allocation retains an inherited exclusive closure guard and authenticated launch phases. "
        "Durable allocations retain named proof; standalone guards are anonymous. "
        "Generation reservations retain capacity for once-only measured or proved non-spawn recovery facts and proof disposal. "
        "Recovery appends exact-identity CAS facts without replacing original observations or terminal runs. "
        "Add an empty agent-owned declared catalog link table; link currency is computed, never stored. "
        "Catalog syncs record whether a bounded discovery was truncated; every historical sync was complete."
    ),
    affected_paths=("state.db",),
    target_schema=SCHEMA_REVISION_4,
    apply=_apply_revision_4,
)
