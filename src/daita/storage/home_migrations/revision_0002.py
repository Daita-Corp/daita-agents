"""Add the caller-owned target posture to retained run inputs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..sqlite_schema import SCHEMA_REVISION_2
from .models import HomeMigration

_REVISION_1_RUN_FIELDS = frozenset(
    {
        "id",
        "agent_id",
        "message",
        "created_at",
        "conversation_id",
        "source_scope_ids",
        "resolved_source_scope",
        "history_sensitivity",
        "start",
    }
)


def _add_target_posture(encoded: object) -> str:
    if not isinstance(encoded, str):
        raise TypeError("revision 1 run input is invalid")
    value = json.loads(encoded)
    if (
        not isinstance(value, dict)
        or set(value) != {"__record__", "fields"}
        or value.get("__record__") != "RunInput"
        or not isinstance(value.get("fields"), dict)
    ):
        raise ValueError("revision 1 run input is invalid")
    fields = value["fields"]
    if set(fields) != _REVISION_1_RUN_FIELDS:
        raise ValueError("revision 1 run input fields are invalid")
    fields["target_posture"] = "single_target"
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def apply(staged_home: Path, source_shape: str | None) -> None:
    if source_shape is not None:
        raise ValueError("revision 2 requires a production revision 1 source")
    with sqlite3.connect(staged_home / "state.db") as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = tuple(connection.execute("SELECT id, input FROM runs ORDER BY id"))
        connection.executemany(
            "UPDATE runs SET input = ? WHERE id = ?",
            tuple((_add_target_posture(encoded), run_id) for run_id, encoded in rows),
        )


REVISION_2 = HomeMigration(
    revision=2,
    migration_id="agent_home_revision_2",
    definition=(
        "Adds the required caller-owned target posture to every retained run "
        "input. Runs written before this field existed retain the conservative "
        "single-target posture."
    ),
    affected_paths=("state.db",),
    target_schema=SCHEMA_REVISION_2,
    apply=apply,
)


__all__ = ["REVISION_2"]
