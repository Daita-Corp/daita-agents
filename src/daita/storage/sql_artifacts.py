"""Artifact-registry persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

from datetime import datetime

from ..artifacts.models import (
    MAX_ARTIFACT_BYTES_PER_AGENT,
    MAX_ARTIFACT_BYTES_PER_RUN,
    MAX_ARTIFACTS_PER_AGENT,
    MAX_ARTIFACTS_PER_RUN,
    ArtifactError,
    ArtifactRecord,
    ArtifactState,
)
from .sql_connection import SQLConnection
from .sql_routines import _datetime_us
from .sqlite_codecs import decode_identity, decode_run_input
from .sqlite_codecs.artifacts import decode_artifact_record, encode_artifact_record

_COLUMNS = "artifact_id, agent_id, run_id, conversation_id, caller_principal_id, state, byte_size, created_at_us, data"


def get_record(connection: SQLConnection, artifact_id: str) -> ArtifactRecord | None:
    row = connection.execute(
        f"SELECT {_COLUMNS} FROM artifacts WHERE artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    return None if row is None else record_from_row(row)


def list_records(
    connection: SQLConnection,
    agent_id: str,
    *,
    state: ArtifactState | None,
    run_id: str | None,
    conversation_id: str | None,
    caller_principal_id: str | None,
    limit: int | None,
    offset: int,
    after: tuple[datetime, str] | None,
    newest_first: bool,
) -> tuple[ArtifactRecord, ...]:
    clauses = ["agent_id = ?"]
    parameters: list[object] = [agent_id]
    for column, value in (
        ("state", state.value if state is not None else None),
        ("run_id", run_id),
        ("conversation_id", conversation_id),
        ("caller_principal_id", caller_principal_id),
    ):
        if value is not None:
            clauses.append(column + " = ?")
            parameters.append(value)
    if after is not None:
        clauses.append("(created_at_us, artifact_id) > (?, ?)")
        parameters.extend((_datetime_us(after[0]), after[1]))
    return tuple(
        record_from_row(row)
        for row in connection.execute(
            f"SELECT {_COLUMNS} FROM artifacts WHERE "
            + " AND ".join(clauses)
            + (
                " ORDER BY created_at_us DESC, artifact_id DESC LIMIT ? OFFSET ?"
                if newest_first
                else " ORDER BY created_at_us, artifact_id LIMIT ? OFFSET ?"
            ),
            (
                *parameters,
                limit if limit is not None else 9223372036854775807,
                offset,
            ),
        )
    )


def begin_creation(
    connection: SQLConnection, record: ArtifactRecord, reserved: bool
) -> None:
    identity_row = connection.execute(
        "SELECT data FROM metadata WHERE key = 'identity'"
    ).fetchone()
    if identity_row is None or decode_identity(identity_row[0]).id != record.agent_id:
        raise ArtifactError(
            "artifact_missing", "Artifact belongs to another agent.", {}
        )
    run_row = connection.execute(
        "SELECT agent_id, input FROM runs WHERE id = ?", (record.ref.run_id,)
    ).fetchone()
    if run_row is not None:
        run = decode_run_input(run_row[1])
        if (
            run_row[0] != record.agent_id
            or run.caller_principal_id != record.caller_principal_id
            or (run.conversation_id or run.id) != record.ref.conversation_id
        ):
            raise ArtifactError(
                "artifact_missing", "Artifact producer ownership differs.", {}
            )
    if get_record(connection, record.ref.artifact_id) is not None:
        raise ArtifactError(
            "artifact_storage_failed",
            "Artifact identity is already registered.",
            {"stage": "identity"},
        )
    if reserved:
        # Only graph task attempts reserved artifact identities.
        raise ArtifactError(
            "artifact_storage_failed",
            "Artifact reservation is no longer active.",
            {"stage": "reservation"},
        )
    rows = tuple(
        connection.execute(
            "SELECT run_id, byte_size FROM artifacts WHERE agent_id = ?",
            (record.agent_id,),
        )
    )
    run_rows = tuple(row for row in rows if row[0] == record.ref.run_id)
    for scope, kind, used, added, ceiling in (
        ("agent", "count", len(rows), 1, MAX_ARTIFACTS_PER_AGENT),
        (
            "agent",
            "bytes",
            sum(row[1] for row in rows),
            record.ref.byte_size,
            MAX_ARTIFACT_BYTES_PER_AGENT,
        ),
        ("run", "count", len(run_rows), 1, MAX_ARTIFACTS_PER_RUN),
        (
            "run",
            "bytes",
            sum(row[1] for row in run_rows),
            record.ref.byte_size,
            MAX_ARTIFACT_BYTES_PER_RUN,
        ),
    ):
        if used + added > ceiling:
            raise ArtifactError(
                "artifact_quota_exceeded",
                "The artifact exceeds a storage limit.",
                {
                    "scope": scope,
                    "limit_kind": kind,
                    "limit": ceiling,
                    "attempted": used + added,
                },
            )
    connection.execute(
        f"INSERT INTO artifacts ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        record_row(record),
    )


def transition(
    connection: SQLConnection, record: ArtifactRecord, updated: ArtifactRecord
) -> ArtifactRecord:
    changed = connection.execute(
        "UPDATE artifacts SET state = ?, data = ? WHERE artifact_id = ? AND data = ?",
        (
            updated.state.value,
            encode_artifact_record(updated),
            record.ref.artifact_id,
            encode_artifact_record(record),
        ),
    ).rowcount
    if changed != 1:
        raise ArtifactError("artifact_missing", "The artifact lifecycle changed.", {})
    return updated


def finish_deletion(connection: SQLConnection, record: ArtifactRecord) -> None:
    if (
        connection.execute(
            "DELETE FROM artifacts WHERE artifact_id = ? AND data = ?",
            (record.ref.artifact_id, encode_artifact_record(record)),
        ).rowcount
        != 1
    ):
        raise ArtifactError("artifact_missing", "The artifact lifecycle changed.", {})


def record_row(record: ArtifactRecord) -> tuple[object, ...]:
    ref = record.ref
    return (
        ref.artifact_id,
        record.agent_id,
        ref.run_id,
        ref.conversation_id,
        record.caller_principal_id,
        record.state.value,
        ref.byte_size,
        int(ref.created_at.timestamp() * 1_000_000),
        encode_artifact_record(record),
    )


def record_from_row(row: tuple) -> ArtifactRecord:
    record = decode_artifact_record(row[-1])
    if tuple(row) != record_row(record):
        raise ValueError("stored artifact registry projection differs from its record")
    return record
