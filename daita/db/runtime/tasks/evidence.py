"""Evidence helpers for ``DbRuntime`` tasks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from daita.runtime import Evidence, Operation, Task, TaskDependency

from .common import (
    _evidence_payload_fingerprint,
    _payload_contains,
    _payload_fingerprint,
    _stable_hash,
)
from .context import DbTaskContext


async def persist_verification_result_evidence(
    context: DbTaskContext,
    operation: Operation,
    verification: Any,
    evidence: tuple[Evidence, ...],
) -> Evidence:
    existing = await latest_accepted_evidence(
        context,
        operation.id,
        "verification.result",
        payload={"passed": True},
    )
    if existing is not None:
        return existing
    accepted = tuple(item for item in evidence if item.accepted and item.id)
    evidence_details = [
        {
            "id": item.id,
            "kind": item.kind,
            "owner": item.owner,
            "task_id": item.task_id,
            "payload_fingerprint": item.metadata.get("payload_fingerprint")
            or _payload_fingerprint(item.payload),
        }
        for item in accepted
    ]
    payload = {
        "passed": bool(verification.passed),
        "evidence_refs": [item["id"] for item in evidence_details],
        "evidence_details": evidence_details,
        "warnings": list(verification.warnings),
        "missing_evidence": list(verification.missing_evidence),
        "diagnostics": verification.diagnostics,
        "input_fingerprint": _stable_hash(
            {
                "operation_id": operation.id,
                "evidence": evidence_details,
                "warnings": list(verification.warnings),
                "missing_evidence": list(verification.missing_evidence),
            }
        ),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    verification_evidence = Evidence(
        id=f"evidence-{uuid4()}",
        kind="verification.result",
        owner="db_runtime",
        operation_id=operation.id,
        accepted=True,
        payload=payload,
        metadata={
            "payload_fingerprint": _payload_fingerprint(payload),
            "input_fingerprint": payload["input_fingerprint"],
        },
    )
    await context.store.save_evidence(verification_evidence)
    return verification_evidence


async def accepted_evidence_for_dependency(
    context: DbTaskContext,
    operation_id: str,
    dependency: TaskDependency,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in await context.store.list_evidence(operation_id)
        if evidence.kind == dependency.evidence_kind
        and evidence.accepted is dependency.evidence_accepted
        and (dependency.evidence_id is None or evidence.id == dependency.evidence_id)
        and (
            dependency.evidence_owner is None
            or evidence.owner == dependency.evidence_owner
        )
        and (
            dependency.producer_task_id is None
            or evidence.task_id == dependency.producer_task_id
        )
        and (
            dependency.input_hash is None
            or evidence.metadata.get("task_input_hash") == dependency.input_hash
        )
        and _payload_contains(evidence.payload, dependency.evidence_payload)
        and (
            dependency.payload_fingerprint is None
            or dependency.payload_fingerprint == _evidence_payload_fingerprint(evidence)
        )
    ]
    return matches[-1] if matches else None


async def authoritative_validation_evidence(
    context: DbTaskContext,
    operation: Operation,
    task: Task | None,
) -> tuple[Evidence, ...]:
    if task is None or task.capability_id not in {
        "db.sql.execute_read",
        "db.sql.execute_write",
    }:
        return ()
    if task.metadata.get("monitor_action_role") == "write_execution":
        expected_validation_id = str(task.metadata.get("validation_evidence_id") or "")
        if expected_validation_id:
            matches = [
                evidence
                for evidence in await context.store.list_evidence(operation.id)
                if evidence.kind == "sql.validation"
                and evidence.id == expected_validation_id
                and evidence.accepted
            ]
            if matches:
                return (matches[-1],)
    validation_dependency = next(
        (
            dependency
            for dependency in task.dependencies
            if dependency.kind.value == "evidence"
            and dependency.evidence_kind == "sql.validation"
        ),
        None,
    )
    if validation_dependency is None:
        return ()
    evidence = await accepted_evidence_for_dependency(
        context,
        operation.id,
        validation_dependency,
    )
    return (evidence,) if evidence is not None else ()


async def latest_accepted_evidence(
    context: DbTaskContext,
    operation_id: str,
    kind: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in await context.store.list_evidence(operation_id)
        if evidence.kind == kind
        and evidence.accepted
        and _payload_contains(evidence.payload, payload or {})
    ]
    return matches[-1] if matches else None


async def latest_evidence(
    context: DbTaskContext,
    operation_id: str,
    kind: str,
    *,
    payload: dict[str, Any] | None = None,
    accepted: bool | None = None,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in await context.store.list_evidence(operation_id)
        if evidence.kind == kind
        and (accepted is None or evidence.accepted is accepted)
        and _payload_contains(evidence.payload, payload or {})
    ]
    return matches[-1] if matches else None
