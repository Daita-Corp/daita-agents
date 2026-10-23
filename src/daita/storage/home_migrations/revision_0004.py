"""Add foreground analytical evidence, declared catalog links and sync truncation."""

import json
import sqlite3
from pathlib import Path

from ..sqlite_schema import (
    ANALYSIS_EVIDENCE_SQL,
    CATALOG_LINKS_SQL,
    RETIRE_LEARNING_CANDIDATES_SQL,
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


_ROUTINE_FIELDS = (
    "conversation_id",
    "owner_principal_id",
    "title",
    "authorized_instruction",
    "instruction_digest",
    "schedule",
    "schedule_interpreter_revision",
    "charged_tokens",
    "charged_cost_usd",
    "attempt_count",
    "occurrence_count",
    "consecutive_failures",
    "next_due_at",
    "active_occurrence_id",
    "last_occurrence_id",
    "last_delivery_ids",
    "state",
    "revision",
    "created_at",
    "updated_at",
)
_DROPPED_OCCURRENCE_FIELDS = (
    "precheck_observation",
    "reserved_tokens",
    "reserved_cost_usd",
)
# Prechecks are gone: a claim mid-precheck is an ordinary claim, and an
# occurrence skipped as unchanged ended without a run, like a completed one.
_OCCURRENCE_DISPOSITIONS = {"prechecking": "claimed", "skipped_no_change": "completed"}


def _simple_routine(record: dict[str, object]) -> dict[str, object]:
    """Keep what a person asked for; drop the frozen contract around it."""

    if record["__record__"] != "ScheduledRoutine":
        raise ValueError("Historical routine record is invalid")
    fields = record["fields"]
    assert isinstance(fields, dict)
    simple = {name: fields[name] for name in _ROUTINE_FIELDS}
    simple["source_ids"] = fields["allowed_source_ids"]
    return {"__record__": "ScheduledRoutine", "fields": simple}


def _simple_occurrence(record: dict[str, object]) -> dict[str, object]:
    if record["__record__"] != "RoutineOccurrence":
        raise ValueError("Historical routine occurrence record is invalid")
    fields = record["fields"]
    assert isinstance(fields, dict)
    for name in _DROPPED_OCCURRENCE_FIELDS:
        del fields[name]
    disposition = fields["disposition"]
    assert isinstance(disposition, str)
    fields["disposition"] = _OCCURRENCE_DISPOSITIONS.get(disposition, disposition)
    return record


def _apply_revision_4(home: Path, source_shape: str | None) -> None:
    if source_shape is not None:
        raise ValueError("Unknown analysis migration source")
    with sqlite3.connect(home / "state.db") as connection:
        connection.executescript(
            ANALYSIS_EVIDENCE_SQL + CATALOG_LINKS_SQL + RETIRE_LEARNING_CANDIDATES_SQL
        )
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
        for agent_id, routine_id, encoded in connection.execute(
            "SELECT agent_id, routine_id, data FROM scheduled_routines"
        ).fetchall():
            connection.execute(
                "UPDATE scheduled_routines SET data = ? WHERE agent_id = ? AND routine_id = ?",
                (_dump(_simple_routine(json.loads(encoded))), agent_id, routine_id),
            )
        for agent_id, occurrence_id, encoded in connection.execute(
            "SELECT agent_id, occurrence_id, data FROM routine_occurrences"
        ).fetchall():
            occurrence = _simple_occurrence(json.loads(encoded))
            fields = occurrence["fields"]
            assert isinstance(fields, dict)
            connection.execute(
                "UPDATE routine_occurrences SET data = ?, state = ? "
                "WHERE agent_id = ? AND occurrence_id = ?",
                (_dump(occurrence), fields["disposition"], agent_id, occurrence_id),
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
        "Catalog syncs record whether a bounded discovery was truncated; every historical sync was complete. "
        "Retire inactive learning candidates and their review stamps; approved knowledge is unaffected. "
        "Routines keep their title, prompt, schedule, named sources, counters and state; their frozen "
        "access, grants, routes, sensitivity, budgets, expiry, outcome, precheck, skills and provenance "
        "are dropped, and their runs ask before writes. Occurrences drop prechecks and reservations."
    ),
    affected_paths=("state.db",),
    target_schema=SCHEMA_REVISION_4,
    apply=_apply_revision_4,
)
