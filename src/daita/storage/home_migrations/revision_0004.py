"""Add foreground analytical evidence, declared catalog links and sync truncation."""

import json
import re
import sqlite3
from hashlib import sha256
from pathlib import Path

from ..sqlite_schema import (
    ANALYSIS_EVIDENCE_SQL,
    CATALOG_LINKS_SQL,
    JOBS_SQL,
    RETIRE_GRAPH_SQL,
    RETIRE_LEARNING_CANDIDATES_SQL,
    SCHEMA_REVISION_4,
)
from .models import HomeMigration


def _dump(record: object) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    material = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + sha256(material.encode("utf-8")).hexdigest()


# Finished graph jobs keep their accepted result; unfinished ones fail with
# this summary and show as Needs you. A requested cancellation is honoured.
_UPGRADED_SUMMARY = "Created by an older version of Daita; start it again."
_FINISHED_GRAPH_STATES = {
    "succeeded": "done",
    "failed": "failed",
    "cancelled": "cancelled",
    "cancel_requested": "cancelled",
}
# The default per-run token limit times the code-owned job allowance.
_UPGRADED_TOKEN_LIMIT = 10 * 100_000


def _bounded(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[: maximum - 1].rstrip() + "…"


def _checklist_job(
    record: dict[str, object],
    results: dict[str, str],
    *,
    agent_id: str,
    job_id: str,
) -> tuple[dict[str, object], str]:
    """Rewrite one stored graph job as the checklist-job record it becomes."""

    if record["__record__"] != "GraphJob":
        raise ValueError("Historical graph job record is invalid")
    fields = record["fields"]
    assert isinstance(fields, dict)
    specification = fields["specification"]["fields"]
    authority = specification["authority"]["fields"]
    state = _FINISHED_GRAPH_STATES.get(fields["state"], "failed")
    finished = fields["state"] in _FINISHED_GRAPH_STATES
    summary: str | None = None
    failure_code: str | None = None
    if state == "done":
        summary = _bounded(results.get(fields["terminal_result_id"], ""), 16_384)
    elif not finished:
        summary, failure_code = _UPGRADED_SUMMARY, "job_upgraded"
    elif state == "failed":
        code = fields["failure_code"]
        failure_code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code)
            else "job_failed"
        )
    updated_at = fields["terminal_at"] or fields["updated_at"]
    job = {
        "job_id": job_id,
        "agent_id": agent_id,
        "conversation_id": fields["conversation_id"],
        "objective": _bounded(specification["objective"], 16_384),
        "source_ids": sorted(authority["source_ids"]),
        "state": state,
        "checklist": [],
        "delegations": [],
        "question": None,
        "answers": [],
        "current_run_id": None,
        "run_count": 0,
        "tokens_used": 0,
        "run_limit": 20,
        "token_limit": _UPGRADED_TOKEN_LIMIT,
        "final_summary": summary,
        "failure_code": failure_code,
        "sensitivity": authority["sensitivity"],
        "revision": 1,
        "created_at": fields["created_at"],
        "updated_at": updated_at,
    }
    return {"__record__": "Job", "fields": job}, state


def _job_delivery(record: dict[str, object], *, agent_id: str) -> dict[str, object]:
    """A graph job's result delivery becomes the job's ordinary delivery."""

    fields = record["fields"]
    assert isinstance(fields, dict)
    if fields["visibility_state"] == "acknowledged" and not fields["acknowledged_at"]:
        fields["acknowledged_at"] = fields["updated_at"]
    fields["subject_kind"] = "job"
    fields["migration_provenance"] = {}
    fields["logical_key"] = "delivery:" + _digest(
        {
            "agent_id": agent_id,
            "subject_kind": "job",
            "subject_id": fields["subject_id"],
            "target_fingerprint": fields["target"]["fields"]["target_fingerprint"],
        }
    )
    return record


def _convert_graph_jobs(connection: sqlite3.Connection) -> None:
    """Replace the task graph with one checklist-job row per graph job.

    Deliveries and receipts keep pointing at the same job id. Graph attention
    notices described graph transitions that no longer exist and are dropped.
    """

    results = {
        result_id: json.loads(data)["fields"]["summary"]
        for result_id, data in connection.execute(
            "SELECT result_id, data FROM job_task_results"
        )
    }
    jobs = []
    for (
        agent_id,
        job_id,
        created_us,
        updated_us,
        terminal_us,
        encoded,
    ) in connection.execute(
        "SELECT agent_id, job_id, created_at_us, updated_at_us, terminal_at_us, "
        "data FROM job_runs"
    ).fetchall():
        job, state = _checklist_job(
            json.loads(encoded), results, agent_id=agent_id, job_id=job_id
        )
        jobs.append(
            (
                agent_id,
                job_id,
                job["fields"]["conversation_id"],  # type: ignore[index]
                state,
                created_us,
                max(updated_us, terminal_us or updated_us),
                _dump(job),
            )
        )
    receipts = connection.execute(
        "SELECT agent_id, id, run_id, call_id, operation_key, routine_id, "
        "occurrence_id, grant_digest, unresolved, job_id, data FROM effect_receipts"
    ).fetchall()
    connection.execute("DELETE FROM deliveries WHERE subject_kind = 'graph_attention'")
    for agent_id, delivery_id, encoded in connection.execute(
        "SELECT agent_id, delivery_id, data FROM deliveries "
        "WHERE subject_kind = 'graph_job' ORDER BY created_at_us, delivery_id"
    ).fetchall():
        delivery = _job_delivery(json.loads(encoded), agent_id=agent_id)
        fields = delivery["fields"]
        assert isinstance(fields, dict)
        duplicate = connection.execute(
            "SELECT 1 FROM deliveries WHERE agent_id = ? AND subject_kind = 'job' "
            "AND subject_id = ? AND target_fingerprint = ?",
            (agent_id, fields["subject_id"], _fingerprint(fields)),
        ).fetchone()
        if duplicate is not None:
            connection.execute(
                "DELETE FROM deliveries WHERE agent_id = ? AND delivery_id = ?",
                (agent_id, delivery_id),
            )
            continue
        connection.execute(
            "UPDATE deliveries SET subject_kind = 'job', logical_key = ?, state = ?, "
            "data = ? WHERE agent_id = ? AND delivery_id = ?",
            (
                fields["logical_key"],
                fields["visibility_state"],
                _dump(delivery),
                agent_id,
                delivery_id,
            ),
        )
    connection.executescript(RETIRE_GRAPH_SQL + JOBS_SQL)
    connection.executemany(
        "INSERT INTO jobs(agent_id, job_id, conversation_id, state, revision, "
        "created_at_us, updated_at_us, data) VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
        jobs,
    )
    connection.executemany(
        "INSERT INTO effect_receipts(agent_id, id, run_id, call_id, operation_key, "
        "routine_id, occurrence_id, grant_digest, unresolved, job_id, data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        receipts,
    )


def _fingerprint(fields: dict[str, object]) -> str:
    target = fields["target"]
    assert isinstance(target, dict)
    return str(target["fields"]["target_fingerprint"])


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
        _convert_graph_jobs(connection)
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
        "are dropped, and their runs ask before writes. Occurrences drop prechecks and reservations. "
        "The task graph is replaced by one checklist-job record per job: finished jobs keep their "
        "accepted result, unfinished jobs fail with an upgrade message, result deliveries and "
        "receipts keep their job id, graph attention notices are dropped, and the graph tables go."
    ),
    affected_paths=("state.db",),
    target_schema=SCHEMA_REVISION_4,
    apply=_apply_revision_4,
)
