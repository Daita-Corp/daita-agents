"""Add foreground analytical evidence without changing released run formats."""

import json
import sqlite3
from pathlib import Path

from ..sqlite_schema import ANALYSIS_EVIDENCE_SQL, SCHEMA_REVISION_4
from .models import HomeMigration


def _apply_revision_4(home: Path, source_shape: str | None) -> None:
    if source_shape is not None:
        raise ValueError("Unknown analysis migration source")
    with sqlite3.connect(home / "state.db") as connection:
        connection.executescript(ANALYSIS_EVIDENCE_SQL)
        for artifact_id, encoded in connection.execute(
            "SELECT artifact_id, data FROM artifacts"
        ):
            record = json.loads(encoded)
            if record["__record__"] != "ArtifactRecord":
                raise ValueError("Historical artifact record is invalid")
            record["fields"]["computation_evidence"] = {}
            connection.execute(
                "UPDATE artifacts SET data = ? WHERE artifact_id = ?",
                (
                    json.dumps(record, sort_keys=True, separators=(",", ":")),
                    artifact_id,
                ),
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
        "Recovery appends exact-identity CAS facts without replacing original observations or terminal runs."
    ),
    affected_paths=("state.db",),
    target_schema=SCHEMA_REVISION_4,
    apply=_apply_revision_4,
)
