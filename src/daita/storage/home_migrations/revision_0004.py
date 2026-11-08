"""Add foreground analytical evidence, declared catalog links and sync truncation."""

import json
import os
import re
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

# shortcut: imports current catalog codecs and search text, so revision 4's output follows them; freeze copies here before any later revision changes them.
from ...catalog.models import (
    CatalogEntry,
    CatalogSourceState,
    FacetKind,
    TabularFacet,
)
from ...catalog.search_text import search_key, search_postings, search_words
from ..sqlite_codecs import (
    decode_catalog_entry,
    decode_catalog_snapshot,
    encode_catalog_entry,
    encode_catalog_relationship,
    encode_catalog_source_state,
    encode_search_postings,
)
from ..sqlite_codecs.common import datetime_encode
from ..sqlite_schema import (
    ANALYSIS_EVIDENCE_SQL,
    CATALOG_ENTRIES_SQL,
    CATALOG_LINKS_SQL,
    JOBS_SQL,
    RETIRE_GRAPH_SQL,
    RETIRE_LEARNING_CANDIDATES_SQL,
    RETIRE_SNAPSHOTS_SQL,
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
    # The occurrence's run row already records the scope that run started with.
    "execution_scope",
    "execution_scope_digest",
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


def _askable_mcp_binding(record: dict[str, object]) -> dict[str, object]:
    """MCP actions ask at the time; tools no longer record unattended eligibility."""

    if record["__record__"] != "MCPServerBinding":
        raise ValueError("Historical MCP binding is invalid")
    fields = record["fields"]
    assert isinstance(fields, dict)
    for tool in fields["tools"]:
        if tool["__record__"] != "MCPToolBinding":
            raise ValueError("Historical MCP tool binding is invalid")
        del tool["fields"]["automation_eligibility"]
    return record


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _replaced(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replaced(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replaced(item, replacements) for key, item in value.items()}
    return value


def _folded_snapshot(
    source_id: str, fields: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Fold one SQLite snapshot's table names, recomputing its chain of hashes.

    A resource ID hashes its name, a relationship ID its endpoints, a relationship
    revision that ID, and a resource revision its ID and relationship revisions.
    Returns the old-to-new resource IDs and revisions; both empty if nothing folds.
    """

    ids: dict[str, str] = {}
    identities: dict[str, str] = {}
    for record in fields["resources"]:
        resource = record["fields"]
        folded = str(resource["native_identity"]).translate(_ASCII_LOWER)
        ids[resource["id"]] = "catalog-resource:" + _digest(
            {
                "kind": resource["kind"]["value"],
                "native_identity": folded,
                "source_id": source_id,
            }
        )
        identities[resource["id"]] = folded
    if len(set(ids.values())) != len(ids):
        raise ValueError("Historical SQLite catalog has tables differing only by case")
    if all(old == new for old, new in ids.items()):
        return {}, {}

    for record in fields["facets"]:
        record["fields"]["resource_id"] = ids[record["fields"]["resource_id"]]
    relationship_revisions: dict[str, str] = {}
    for record in fields["relationships"]:
        relationship = record["fields"]
        relationship["from_resource_id"] = ids[relationship["from_resource_id"]]
        relationship["to_resource_id"] = ids[relationship["to_resource_id"]]
        relationship["id"] = "catalog-relationship:" + _digest(
            {
                "field_pairs": [
                    item["fields"]
                    for item in sorted(
                        relationship["field_pairs"],
                        key=lambda item: item["fields"]["ordinal"],
                    )
                ],
                "from_resource_id": relationship["from_resource_id"],
                "kind": relationship["kind"]["value"],
                "provenance": relationship["provenance"]["value"],
                "source_id": source_id,
                "to_resource_id": relationship["to_resource_id"],
            }
        )
        revision = _digest(
            {
                "attributes": relationship["attributes"],
                "confidence": float(relationship["confidence"]),
                "relationship_id": relationship["id"],
            }
        )
        relationship_revisions[relationship["revision"]] = revision
        relationship["revision"] = revision
    revisions: dict[str, str] = {}
    for record in fields["revisions"]:
        revision_fields = record["fields"]
        revision_fields["resource_id"] = ids[revision_fields["resource_id"]]
        revision_fields["relationship_revisions"] = sorted(
            relationship_revisions[item]
            for item in revision_fields["relationship_revisions"]
        )
        revision = _digest(
            {
                "facet_revisions": sorted(revision_fields["facet_revisions"]),
                "relationship_revisions": revision_fields["relationship_revisions"],
                "resource_id": revision_fields["resource_id"],
            }
        )
        revisions[revision_fields["revision"]] = revision
        revision_fields["revision"] = revision
    for record in fields["resources"]:
        resource = record["fields"]
        resource["native_identity"] = identities[resource["id"]]
        resource["id"] = ids[resource["id"]]
        resource["current_revision"] = revisions[resource["current_revision"]]
    return ids, revisions


def _fold_sqlite_identities(connection: sqlite3.Connection) -> None:
    """Give SQLite tables one identity per name, as SQLite itself does.

    What must keep matching moves with them: read scopes, semantic annotations and
    catalog snapshots. Artifacts, runs, messages and evidence stay as history.
    """

    sqlite_sources = {
        (agent_id, source_id)
        for agent_id, source_id, encoded in connection.execute(
            "SELECT agent_id, id, data FROM sources"
        )
        if json.loads(encoded)["fields"]["adapter_id"] == "sqlite"
    }
    moved: dict[str, dict[str, str]] = {}
    for agent_id, source_id, encoded in connection.execute(
        "SELECT agent_id, source_id, data FROM snapshots"
    ).fetchall():
        if (agent_id, source_id) not in sqlite_sources:
            continue
        snapshot = json.loads(encoded)
        ids, revisions = _folded_snapshot(source_id, snapshot["fields"])
        if not ids:
            continue
        moved.setdefault(agent_id, {}).update(ids)
        moved[agent_id].update(revisions)
        connection.execute(
            "UPDATE snapshots SET data = ? WHERE agent_id = ? AND source_id = ?",
            (_dump(snapshot), agent_id, source_id),
        )
        scope_row = connection.execute(
            "SELECT data FROM source_read_scopes WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchone()
        if scope_row is not None:
            scope = _replaced(json.loads(scope_row[0]), ids)
            assert isinstance(scope, dict)
            scope["fields"]["resource_ids"] = sorted(scope["fields"]["resource_ids"])
            connection.execute(
                "UPDATE source_read_scopes SET data = ? "
                "WHERE agent_id = ? AND source_id = ?",
                (_dump(scope), agent_id, source_id),
            )
    for agent_id, annotation_id, encoded in connection.execute(
        "SELECT agent_id, id, data FROM semantic_annotations"
    ).fetchall():
        replacements = moved.get(agent_id)
        if not replacements:
            continue
        annotation = _replaced(json.loads(encoded), replacements)
        assert isinstance(annotation, dict)
        fields = annotation["fields"]
        subject = fields["subject"]["fields"]
        subject["resource_ids"] = sorted(subject["resource_ids"])
        subject["fields"].sort(
            key=lambda item: (
                item["fields"]["resource_id"],
                item["fields"]["field_name"],
            )
        )
        fields["catalog_revisions"].sort(
            key=lambda item: (item["fields"]["resource_id"], item["fields"]["revision"])
        )
        connection.execute(
            "UPDATE semantic_annotations SET data = ? WHERE agent_id = ? AND id = ?",
            (_dump(annotation), agent_id, annotation_id),
        )


def _converted_write_scope(
    agent_id: str, source: dict[str, Any], record: dict[str, Any], table: str
) -> tuple[dict[str, object], str]:
    """One stored scope in its new shape, with the fingerprint that binds it.

    update stays update; upsert becomes insert and update. Delete is never granted.
    """

    if record["__record__"] != "RelationalWriteScope":
        raise ValueError("Historical write scope record is invalid")
    fields = record["fields"]
    held = set(fields["allowed_operations"])
    operations = sorted({"update"} | ({"insert"} if "upsert" in held else set()))
    converted = {
        "table": table,
        "allowed_operations": operations,
        "allowed_insert_columns": fields["allowed_insert_columns"],
        "allowed_update_columns": fields["allowed_update_columns"],
        "max_rows": fields["max_rows"],
    }
    return converted, _digest(
        {
            "agent_id": agent_id,
            "adapter_id": source["adapter_id"],
            "source_id": source["id"],
            "resource_id": record["resource_id"],
            "authority": converted,
        }
    )


def _convert_write_scopes(connection: sqlite3.Connection) -> None:
    """Bind stored write scopes to table and column names, not catalog revisions.

    A scope whose table left the catalog, or changed since it was granted, was
    already refused as stale, so it is dropped rather than given authority back.
    """

    tables: dict[tuple[str, str], tuple[str, str]] = {}
    for agent_id, encoded in connection.execute("SELECT agent_id, data FROM snapshots"):
        for resource in json.loads(encoded)["fields"]["resources"]:
            fields = resource["fields"]
            tables[(agent_id, fields["id"])] = (
                fields["native_identity"],
                fields["current_revision"],
            )
    sources = {
        (agent_id, source_id): json.loads(encoded)["fields"]
        for agent_id, source_id, encoded in connection.execute(
            "SELECT agent_id, id, data FROM sources"
        )
    }
    for agent_id, source_id, resource_id, encoded in connection.execute(
        "SELECT agent_id, source_id, resource_id, data FROM relational_write_scopes"
    ).fetchall():
        key = (agent_id, source_id, resource_id)
        record = json.loads(encoded)
        table, revision = tables.get((agent_id, resource_id), (None, None))
        if table is None or record["fields"]["resource_revision"] != revision:
            connection.execute(
                "DELETE FROM relational_write_scopes "
                "WHERE agent_id = ? AND source_id = ? AND resource_id = ?",
                key,
            )
            continue
        converted, fingerprint = _converted_write_scope(
            agent_id,
            sources[(agent_id, source_id)],
            {**record, "resource_id": resource_id},
            table,
        )
        connection.execute(
            "UPDATE relational_write_scopes "
            "SET data = ?, authorization_fingerprint = ? "
            "WHERE agent_id = ? AND source_id = ? AND resource_id = ?",
            (
                _dump({"__record__": "RelationalWriteScope", "fields": converted}),
                fingerprint,
                *key,
            ),
        )


_REVISION_3_CANDIDATE_FIELDS = {
    "allowed_sensitivities",
    "base_url",
    "profile_limits",
    "provider_id",
    "secret_reference",
}


def _default_reasoning_effort(path: Path) -> None:
    """Give every saved route candidate the provider-default reasoning effort."""

    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError("Historical model configuration path is invalid")
    value = json.loads(path.read_bytes().decode("utf-8"))
    route = value.get("model_route") if isinstance(value, dict) else None
    candidates = route.get("candidates") if isinstance(route, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("Historical model configuration is invalid")
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != (
            _REVISION_3_CANDIDATE_FIELDS
        ):
            raise ValueError("Historical model route candidate is invalid")
        candidate["reasoning_effort"] = None
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("model configuration write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_MAX_PARAMETERS = 900


def _insert_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: str,
    rows: list[tuple[str | None, ...]],
) -> None:
    """Insert rows in multi-row statements that stay under the parameter limit."""

    if not rows:
        return
    width = len(rows[0])
    per_statement = _MAX_PARAMETERS // width
    group = "(" + ", ".join("?" * width) + ")"
    for start in range(0, len(rows), per_statement):
        chunk = rows[start : start + per_statement]
        connection.execute(
            f"INSERT INTO {table} ({columns}) VALUES "
            + ", ".join([group] * len(chunk)),
            [value for row in chunk for value in row],
        )


def _timestamp(value: object) -> str:
    stored = datetime_encode(value)["__datetime__"]  # type: ignore[arg-type]
    assert isinstance(stored, str)
    return stored


def _convert_catalog_snapshots(connection: sqlite3.Connection) -> None:
    """Turn each stored source snapshot into per-resource rows, then drop it.

    Every resource keeps its ID, revision and facets and counts as fully
    described; the source's one state row names the snapshot's sync as current.
    A snapshot that does not decode stops the upgrade.
    """

    connection.executescript(CATALOG_ENTRIES_SQL)
    for agent_id, source_id, encoded in connection.execute(
        "SELECT agent_id, source_id, data FROM snapshots"
    ).fetchall():
        snapshot = decode_catalog_snapshot(encoded)
        sync_id = snapshot.sync.id
        revisions = {item.resource_id: item for item in snapshot.revisions}
        facets: dict[str, list] = {}
        for facet in snapshot.facets:
            facets.setdefault(facet.resource_id, []).append(facet)
        incident: dict[str, list] = {item.id: [] for item in snapshot.resources}
        for relationship in snapshot.relationships:
            incident[relationship.from_resource_id].append(relationship)
            if relationship.to_resource_id != relationship.from_resource_id:
                incident[relationship.to_resource_id].append(relationship)
        resource_rows: list[tuple[str | None, ...]] = []
        search_rows: list[tuple[str | None, ...]] = []
        for resource in snapshot.resources:
            entry = CatalogEntry(
                resource=resource,
                revision=revisions[resource.id],
                facets=tuple(
                    sorted(
                        facets.get(resource.id, ()),
                        key=lambda item: (item.kind.value, item.revision),
                    )
                ),
                detailed=True,
            )
            resource_rows.append(
                (
                    agent_id,
                    source_id,
                    resource.id,
                    resource.kind.value,
                    resource.native_identity,
                    resource.name,
                    resource.current_revision,
                    None,
                    "complete",
                    sync_id,
                    _timestamp(resource.first_observed_at),
                    _timestamp(resource.last_observed_at),
                    encode_catalog_entry(entry),
                )
            )
            postings = search_postings(entry, incident[resource.id])
            search_rows.append(
                (
                    agent_id,
                    source_id,
                    resource.id,
                    search_key(resource.name),
                    search_key(resource.native_identity),
                    search_words(postings),
                    encode_search_postings(postings),
                )
            )
        _insert_rows(
            connection,
            "catalog_resources",
            "agent_id, source_id, id, kind, native_identity, name, revision, "
            "source_object_id, detail, last_seen_sync, first_observed_at, "
            "last_observed_at, data",
            resource_rows,
        )
        _insert_rows(
            connection,
            "catalog_relationships",
            "agent_id, source_id, id, from_resource_id, to_resource_id, revision, data",
            [
                (
                    agent_id,
                    source_id,
                    item.id,
                    item.from_resource_id,
                    item.to_resource_id,
                    item.revision,
                    encode_catalog_relationship(item),
                )
                for item in snapshot.relationships
            ],
        )
        _insert_rows(
            connection,
            "catalog_search",
            "agent_id, source_id, resource_id, name_key, identity_key, words, data",
            search_rows,
        )
        connection.execute(
            "INSERT INTO catalog_source_state (agent_id, source_id, data) "
            "VALUES (?, ?, ?)",
            (
                agent_id,
                source_id,
                encode_catalog_source_state(
                    CatalogSourceState(
                        agent_id=agent_id,
                        source_id=source_id,
                        current_sync_id=sync_id,
                    )
                ),
            ),
        )
    connection.executescript(RETIRE_SNAPSHOTS_SQL)


def _backfill_note_field_types(connection: sqlite3.Connection) -> None:
    """Save the field types a note was written against, where they are still known.

    A binding whose resource is still at its bound revision gets the native types
    of the fields the note names. Any other binding keeps comparing revisions.
    """

    for agent_id, annotation_id, encoded in connection.execute(
        "SELECT agent_id, id, data FROM semantic_annotations"
    ).fetchall():
        annotation = json.loads(encoded)
        fields = annotation["fields"]
        named: dict[str, set[str]] = {}
        for item in fields["subject"]["fields"]["fields"]:
            named.setdefault(item["fields"]["resource_id"], set()).add(
                item["fields"]["field_name"]
            )
        changed = False
        for binding in fields["catalog_revisions"]:
            bound = binding["fields"]
            row = connection.execute(
                "SELECT revision, data FROM catalog_resources "
                "WHERE agent_id = ? AND id = ?",
                (agent_id, bound["resource_id"]),
            ).fetchone()
            if row is None or row[0] != bound["revision"]:
                continue
            tabular = next(
                (
                    facet
                    for facet in decode_catalog_entry(row[1]).facets
                    if facet.kind is FacetKind.TABULAR
                ),
                None,
            )
            columns = (
                TabularFacet.from_payload(tabular.payload).columns if tabular else ()
            )
            wanted = named.get(bound["resource_id"], set())
            bound["field_types"] = sorted(
                [column.name, column.native_type]
                for column in columns
                if column.name in wanted
            )
            changed = True
        if changed:
            connection.execute(
                "UPDATE semantic_annotations SET data = ? "
                "WHERE agent_id = ? AND id = ?",
                (_dump(annotation), agent_id, annotation_id),
            )


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
        for agent_id, binding_id, encoded in connection.execute(
            "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
        ).fetchall():
            connection.execute(
                "UPDATE mcp_server_bindings SET data = ? WHERE agent_id = ? AND binding_id = ?",
                (
                    _dump(_askable_mcp_binding(json.loads(encoded))),
                    agent_id,
                    binding_id,
                ),
            )
        _convert_graph_jobs(connection)
        _fold_sqlite_identities(connection)
        _convert_write_scopes(connection)
        _convert_catalog_snapshots(connection)
        _backfill_note_field_types(connection)
        connection.commit()
    _default_reasoning_effort(home / "config.json")


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
        "are dropped, and their runs ask before writes. Occurrences drop prechecks, reservations "
        "and their copy of the run's scope. MCP tool bindings drop unattended eligibility; every "
        "MCP action asks at the time. "
        "The task graph is replaced by one checklist-job record per job: finished jobs keep their "
        "accepted result, unfinished jobs fail with an upgrade message, result deliveries and "
        "receipts keep their job id, graph attention notices are dropped, and the graph tables go. "
        "Stored write scopes bind to table and column names, not catalog revisions: update stays "
        "update, upsert becomes insert and update, delete is never granted, conflict keys and "
        "identity columns go, and a scope whose table left the catalog is dropped. "
        "Each saved model route candidate gains a null reasoning effort, which keeps the "
        "provider's default. "
        "Each source's catalog snapshot becomes one row per resource, relationship and search "
        "entry plus a source state row; resource ids and revisions are unchanged, every "
        "resource counts as fully described, and the snapshot table goes. "
        "Each semantic note whose resource is still at the revision it was saved against "
        "gains the native types of the fields it names; other notes keep comparing revisions."
    ),
    affected_paths=("state.db", "config.json"),
    target_schema=SCHEMA_REVISION_4,
    apply=_apply_revision_4,
)
