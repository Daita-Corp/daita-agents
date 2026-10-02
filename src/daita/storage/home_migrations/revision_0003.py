"""Convert released revision-2 records to caller-aware records and independent artifact storage."""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from pathlib import Path

from ...artifacts.models import ArtifactRecord, ArtifactState, artifact_ref_from_mapping
from ..schema_contract import require_healthy, require_schema
from ..sqlite_codecs.artifacts import encode_artifact_record
from ..sqlite_schema import ARTIFACT_REGISTRY_SQL, SCHEMA_REVISION_2, SCHEMA_REVISION_3
from .models import HomeMigration

# Exact format pushed in 91e0f8ce before the artifact registry was added. This is
# a migration input, never a runtime fallback or an exemption for other checksums.
PRE_REGISTRY_REVISION_3_SOURCE = "development_revision_3_without_artifact_registry"
PRE_REGISTRY_REVISION_3_CHECKSUM = (
    "db1e58e2ed98008dcf9764ff9115d555e298820cf5d910d01adad4ce7ee398b3"
)


def is_pre_registry_revision_3(
    connection: sqlite3.Connection, released_prefix: tuple[tuple[int, str, str], ...]
) -> bool:
    rows = tuple(
        connection.execute(
            "SELECT revision, migration_id, checksum FROM agent_home_migrations ORDER BY revision"
        )
    )
    if rows != (
        *released_prefix,
        (3, "0003_framework_caller_authority", PRE_REGISTRY_REVISION_3_CHECKSUM),
    ):
        return False
    require_schema(connection, SCHEMA_REVISION_2)
    require_healthy(connection)
    return True


def _apply_revision_3(home: Path, source_shape: str | None) -> None:
    # Already caller-aware development homes preserve their MCP contracts exactly.
    if source_shape not in {None, PRE_REGISTRY_REVISION_3_SOURCE}:
        raise ValueError("revision-3 migration source is unknown")
    with sqlite3.connect(home / "state.db") as connection:
        if source_shape != PRE_REGISTRY_REVISION_3_SOURCE:
            for run_id, agent_id, encoded in connection.execute(
                "SELECT id, agent_id, input FROM runs"
            ):
                record = json.loads(encoded)
                fields = record["fields"]
                if record["__record__"] != "RunInput":
                    raise ValueError("revision-2 run record is invalid")
                if "caller_principal_id" not in fields:
                    scope = fields["start"]["fields"]["execution_scope"]
                    principal = (
                        agent_id if scope is None else scope["fields"]["principal_id"]
                    )
                    fields["caller_principal_id"] = principal
                fields.setdefault("caller_principal_verified", True)
                connection.execute(
                    "UPDATE runs SET input = ? WHERE id = ?",
                    (json.dumps(record, sort_keys=True, separators=(",", ":")), run_id),
                )
            for agent_id, binding_id, encoded in connection.execute(
                "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
            ):
                record = json.loads(encoded)
                fields = record["fields"]
                if record["__record__"] != "MCPServerBinding":
                    raise ValueError("revision-2 MCP binding is invalid")
                if "connection_id" not in fields:
                    fields["owner_principal_id"] = agent_id
                    fields.update(
                        connection_id=None,
                        resource_uri=None,
                        required_scopes=[],
                    )
                fields.setdefault("protocol_capabilities_digest", None)
                for tool in fields["tools"]:
                    # Released records retain the admitted assertion projection and
                    # original digest, but not the discarded remote annotations.
                    # Do not invent a raw schema or rehash retained authority.
                    tool["fields"]["raw_input_schema"] = None
                connection.execute(
                    "UPDATE mcp_server_bindings SET data = ? WHERE agent_id = ? AND binding_id = ?",
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")),
                        agent_id,
                        binding_id,
                    ),
                )
        connection.executescript(ARTIFACT_REGISTRY_SQL)
        _register_retained_artifacts(home, connection)
        connection.commit()


def _register_retained_artifacts(home: Path, connection: sqlite3.Connection) -> None:
    """Consume prior artifact roots once; history is not a runtime inventory."""
    runs = {
        run_id: (agent_id, json.loads(encoded)["fields"])
        for run_id, agent_id, encoded in connection.execute(
            "SELECT id, agent_id, input FROM runs"
        )
    }
    owners = {
        agent_id: json.loads(encoded)["fields"]["id"]
        for agent_id, encoded in connection.execute(
            "SELECT json_extract(data, '$.fields.id'), data FROM metadata WHERE key = 'identity'"
        )
    }
    jobs = {
        (agent_id, job_id): json.loads(encoded)["fields"]
        for agent_id, job_id, encoded in connection.execute(
            "SELECT agent_id, job_id, data FROM job_runs"
        )
    }
    retained: dict[str, ArtifactRecord] = {}

    def add(raw: dict, agent_id: str, principal: str) -> None:
        ref = artifact_ref_from_mapping(raw)
        record = ArtifactRecord(ref, agent_id, principal, ArtifactState.READY)
        old = retained.setdefault(ref.artifact_id, record)
        if old != record:
            raise ValueError("revision-2 artifact identity is ambiguous")

    for run_id, conversation_id, encoded in connection.execute(
        "SELECT r.id, r.conversation_id, m.data FROM runs AS r "
        "JOIN messages AS m ON m.run_id = r.id"
    ):
        message = json.loads(encoded)["fields"]
        if message["role"]["value"] != "tool":
            continue
        agent_id, run = runs[run_id]
        for block in message["content"]:
            if block["__record__"] != "ToolResultBlock":
                continue
            result = block["fields"]
            raw = result["output"].get("artifact")
            if result["is_error"] or raw is None:
                continue
            if (
                raw["run_id"] != run_id
                or raw["conversation_id"] != conversation_id
                or raw["call_id"] != result["call_id"]
            ):
                raise ValueError("revision-2 artifact run binding differs")
            add(raw, agent_id, run["caller_principal_id"])

    for agent_id, job_id, encoded in connection.execute(
        "SELECT agent_id, job_id, data FROM job_task_results"
    ):
        result = json.loads(encoded)["fields"]
        job = jobs[(agent_id, job_id)]
        raw_refs = result["provenance"].get("artifact_refs", [])
        if sorted(raw["artifact_id"] for raw in raw_refs) != list(
            result["artifact_ids"]
        ):
            raise ValueError("revision-2 task artifact identities differ")
        for raw in raw_refs:
            if (
                raw["run_id"] != result["run_id"]
                or raw["conversation_id"] != job["conversation_id"]
            ):
                raise ValueError("revision-2 task artifact binding differs")
            principal = runs.get(raw["run_id"], (agent_id, {}))[1].get(
                "caller_principal_id", job["specification"]["fields"]["principal_id"]
            )
            add(raw, agent_id, principal)

    def manifest(run_id: str, artifact_id: str) -> dict:
        # Only fixed contained paths from historical producer records are read.
        import re

        if not re.fullmatch(r"run-[0-9a-f]{32}", run_id) or not re.fullmatch(
            r"artifact-[0-9a-f]{32}", artifact_id
        ):
            raise ValueError("revision-2 artifact path identity is invalid")
        path = home / "artifacts" / run_id / artifact_id / "manifest.json"
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("revision-2 artifact manifest is not contained")
        if path.stat().st_size > 64 * 1024:
            raise ValueError("revision-2 artifact manifest is oversized")
        return json.loads(path.read_text(encoding="utf-8"))

    for agent_id, subject_kind, subject_id, encoded in connection.execute(
        "SELECT agent_id, subject_kind, subject_id, data FROM deliveries"
    ):
        delivery = json.loads(encoded)["fields"]
        for encoded_ref in delivery["outcome"]["fields"]["artifact_references"]:
            expected = encoded_ref["fields"]
            raw = manifest(expected["producing_run_id"], expected["artifact_id"])
            for field, target in (
                ("artifact_id", "artifact_id"),
                ("run_id", "producing_run_id"),
                ("call_id", "producing_call_id"),
                ("capability_id", "producer_capability_id"),
                ("sha256", "sha256"),
                ("media_type", "media_type"),
                ("byte_size", "byte_size"),
            ):
                if raw[field] != expected[target]:
                    raise ValueError("revision-2 delivery artifact binding differs")
            provenance_digest = (
                "sha256:"
                + sha256(
                    json.dumps(
                        raw["provenance"],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest()
            )
            if (
                raw["sensitivity"] != expected["sensitivity"]
                or raw["provenance"]["authorship"] != expected["authorship"]
                or provenance_digest != expected["provenance_digest"]
            ):
                raise ValueError("revision-2 delivery artifact provenance differs")
            producer_principal = agent_id
            if subject_kind == "routine_occurrence":
                occurrence_row = connection.execute(
                    "SELECT data FROM routine_occurrences WHERE agent_id = ? AND occurrence_id = ?",
                    (agent_id, subject_id),
                ).fetchone()
                if occurrence_row is None:
                    raise ValueError("artifact delivery occurrence is missing")
                scope = json.loads(occurrence_row[0])["fields"]["execution_scope"]
                if scope is not None:
                    producer_principal = scope["fields"]["principal_id"]
            principal = runs.get(raw["run_id"], (agent_id, {}))[1].get(
                "caller_principal_id", producer_principal
            )
            if raw["artifact_id"] in retained:
                principal = retained[raw["artifact_id"]].caller_principal_id
            add(raw, agent_id, principal)

    # Preserve files belonging to exact active restart reservations as well.
    for agent_id, job_id, attempt_id, run_id in connection.execute(
        "SELECT agent_id, job_id, attempt_id, run_id FROM job_task_attempts "
        "WHERE state IN ('claimed', 'running')"
    ):
        artifact_id = "artifact-" + sha256(attempt_id.encode()).hexdigest()[:32]
        path = home / "artifacts" / run_id / artifact_id
        if path.exists():
            add(
                manifest(run_id, artifact_id),
                agent_id,
                jobs[(agent_id, job_id)]["specification"]["fields"]["principal_id"],
            )

    for record in retained.values():
        if record.agent_id not in owners:
            raise ValueError("revision-2 artifact belongs to another agent")
        ref = record.ref
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ref.artifact_id,
                record.agent_id,
                ref.run_id,
                ref.conversation_id,
                record.caller_principal_id,
                record.state.value,
                ref.byte_size,
                int(ref.created_at.timestamp() * 1_000_000),
                encode_artifact_record(record),
            ),
        )


REVISION_3 = HomeMigration(
    revision=3,
    migration_id="0003_framework_caller_authority",
    definition=(
        "Add caller provenance, opaque personal MCP connection claims and SDK "
        "protocol facts and exact external schema contracts while preserving released revision-2 identities, grants, "
        "receipts and data. Register retained artifacts with independent ownership and lifecycle state."
        " Convert the exact previously pushed pre-registry development format without rewriting caller or MCP records."
    ),
    affected_paths=("state.db",),
    read_only_globs=("artifacts/*/*/manifest.json",),
    target_schema=SCHEMA_REVISION_3,
    apply=_apply_revision_3,
)

__all__ = ["REVISION_3"]
