"""Analysis-evidence persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

from ..loop.analysis import AnalysisEvidence, AnalysisTraceCapacityError
from .sql_connection import SQLConnection
from .sqlite_codecs.analysis import decode_analysis_evidence


def record_evidence(
    connection: SQLConnection,
    agent_id: str,
    evidence: AnalysisEvidence,
    encoded: str,
) -> None:
    row = connection.execute(
        "SELECT agent_id, result FROM runs WHERE id = ?", (evidence.run_id,)
    ).fetchone()
    if row is None or row[0] != agent_id or row[1] is not None:
        raise ValueError("Analysis evidence requires an exact live owned run")
    summary = connection.execute(
        "SELECT data FROM analysis_evidence WHERE run_id = ? AND evidence_id = 'run-summary'",
        (evidence.run_id,),
    ).fetchone()
    trace_records = (
        evidence.facts.get("trace_records", 256)
        if evidence.kind == "run" and evidence.facts["status"] == "admitted"
        else (
            decode_analysis_evidence(summary[0]).facts.get("trace_records", 256)
            if summary
            else 256
        )
    )
    if type(trace_records) is not int or not 4 <= trace_records <= 256:
        raise ValueError("Analysis trace admission requires a bounded record ceiling")
    existing = connection.execute(
        "SELECT data FROM analysis_evidence WHERE run_id = ? AND evidence_id = ?",
        (evidence.run_id, evidence.evidence_id),
    ).fetchone()
    if existing is not None:
        prior = decode_analysis_evidence(existing[0])
        if evidence.kind == "generation" and existing[0] == encoded:
            return  # Lost commit acknowledgement: exact settlement only.
        disposal = (
            prior.kind == evidence.kind == "generation"
            and prior.facts["status"] == "closed"
            and dict(evidence.facts) == {**prior.facts, "proof_disposed": True}
        )
        if (
            prior.kind != evidence.kind
            or not disposal
            and prior.facts.get("status")
            not in {
                "admitted",
                "running",
            }
        ):
            raise ValueError("Analysis evidence cannot be replayed or overwritten")
        for identity in (
            "parent_call_id",
            "sequence",
            "tool_name",
            "arguments",
            "code",
            "scratch",
            "closure_nonce",
        ):
            if identity in prior.facts and prior.facts[identity] != evidence.facts.get(
                identity
            ):
                raise ValueError("Analysis evidence identity changed")
    else:
        counted = connection.execute(
            "SELECT COUNT(*) FROM analysis_evidence WHERE run_id = ?",
            (evidence.run_id,),
        ).fetchone()
        assert counted is not None
        count = counted[0]
        if count >= trace_records:
            raise AnalysisTraceCapacityError("Analysis evidence capacity exhausted")
    credit = (
        max(len(encoded.encode()), 270_000)
        if evidence.facts.get("status") in {"admitted", "running", "cleanup_failed"}
        or evidence.kind == "generation"
        and evidence.facts.get("proof_disposed") is not True
        else len(encoded.encode())
    )
    occupied = connection.execute(
        "SELECT COALESCE(SUM(reserved_bytes), 0) FROM analysis_evidence WHERE run_id = ? AND evidence_id != ?",
        (evidence.run_id, evidence.evidence_id),
    ).fetchone()
    assert occupied is not None
    trace_limit = (
        evidence.facts.get("trace_bytes", 4 * 1024 * 1024)
        if evidence.kind == "run" and evidence.facts["status"] == "admitted"
        else (
            decode_analysis_evidence(summary[0]).facts.get(
                "trace_bytes", 4 * 1024 * 1024
            )
            if summary
            else 4 * 1024 * 1024
        )
    )
    if (
        type(trace_limit) is not int
        or not 1024 * 1024 <= trace_limit <= 4 * 1024 * 1024
    ):
        raise ValueError("Analysis trace admission requires a bounded byte ceiling")
    if occupied[0] + credit > trace_limit:
        raise AnalysisTraceCapacityError(
            "Analysis trace byte reservation exhausted before dispatch"
        )
    connection.execute(
        "INSERT INTO analysis_evidence(agent_id, run_id, evidence_id, kind, data, reserved_bytes) VALUES (?, ?, ?, ?, ?, ?) "
        f"ON CONFLICT({connection.conflict('run_id, evidence_id')}) DO UPDATE SET data = excluded.data, reserved_bytes = excluded.reserved_bytes",
        (
            agent_id,
            evidence.run_id,
            evidence.evidence_id,
            evidence.kind,
            encoded,
            credit,
        ),
    )


def load_evidence(
    connection: SQLConnection, agent_id: str, run_id: str
) -> tuple[AnalysisEvidence, ...]:
    return tuple(
        decode_analysis_evidence(row[0])
        for row in connection.execute(
            "SELECT data FROM analysis_evidence WHERE agent_id = ? AND run_id = ? ORDER BY evidence_id LIMIT 256",
            (agent_id, run_id),
        )
    )


def pending_evidence(
    connection: SQLConnection, agent_id: str, after: tuple[str, str]
) -> tuple[AnalysisEvidence, ...]:
    return tuple(
        decode_analysis_evidence(row[0])
        for row in connection.execute(
            "SELECT data FROM analysis_evidence WHERE agent_id = ? "
            "AND (run_id > ? OR (run_id = ? AND evidence_id > ?)) "
            "ORDER BY run_id, evidence_id LIMIT 256",
            (agent_id, after[0], after[0], after[1]),
        )
    )


def replace_recovered_evidence(
    connection: SQLConnection,
    agent_id: str,
    expected: AnalysisEvidence,
    before: str,
    encoded: str,
    proof_disposed: bool,
) -> None:
    row = connection.execute(
        "SELECT a.data, a.reserved_bytes FROM analysis_evidence a JOIN runs r ON r.id = a.run_id "
        "WHERE a.agent_id = ? AND r.agent_id = ? AND a.run_id = ? AND a.evidence_id = ?",
        (agent_id, agent_id, expected.run_id, expected.evidence_id),
    ).fetchone()
    if row is None or row[0] not in {before, encoded}:
        raise ValueError("Recovery evidence CAS identity changed")
    if len(encoded.encode()) > row[1]:
        raise AnalysisTraceCapacityError(
            "Recovery exceeded its pre-dispatch reservation"
        )
    connection.execute(
        "UPDATE analysis_evidence SET data = ?, reserved_bytes = ? WHERE agent_id = ? AND run_id = ? AND evidence_id = ? AND data = ?",
        (
            encoded,
            len(encoded.encode()) if proof_disposed else row[1],
            agent_id,
            expected.run_id,
            expected.evidence_id,
            before,
        ),
    )
