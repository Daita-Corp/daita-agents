"""Pure task dependency helpers for ``DbRuntime``."""

from __future__ import annotations

from daita.runtime import (
    ApprovalStatus,
    Capability,
    Evidence,
    Operation,
    Task,
    TaskDependency,
)

from ...fingerprints import persisted_fingerprint


def _combine_dependencies(
    default_dependencies: tuple[TaskDependency, ...],
    spec_dependencies: tuple[TaskDependency, ...],
) -> tuple[TaskDependency, ...]:
    combined: list[TaskDependency] = []
    seen: set[str] = set()
    for dependency in (*default_dependencies, *spec_dependencies):
        fingerprint = persisted_fingerprint(dependency.to_dict())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        combined.append(dependency)
    return tuple(combined)


def _has_sql_validation_dependency(
    dependencies: tuple[TaskDependency, ...],
) -> bool:
    return any(
        dependency.kind.value == "evidence"
        and dependency.evidence_kind == "sql.validation"
        for dependency in dependencies
    )


def _task_dependencies_for_capability(
    operation: Operation,
    capability: Capability,
    *,
    validation_task: Task | None = None,
) -> tuple[TaskDependency, ...]:
    if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
        return ()
    if capability.id == "db.sql.execute_read" and validation_task is None:
        return ()
    validation_dependency = TaskDependency(
        kind="evidence",
        evidence_kind="sql.validation",
        evidence_owner=(
            validation_task.metadata.get("owner") if validation_task else None
        ),
        producer_task_id=validation_task.id if validation_task else None,
        producer_capability_id=(
            validation_task.capability_id if validation_task else "db.sql.validate"
        ),
        producer_executor_id=(validation_task.executor_id if validation_task else None),
        evidence_payload={"valid": True},
        operation_id=operation.id,
        input_hash=(
            validation_task.metadata.get("input_hash") if validation_task else None
        ),
    )
    if capability.id == "db.sql.execute_read":
        return (validation_dependency,)
    return (
        validation_dependency,
        TaskDependency(
            kind="approval",
            approval_status=ApprovalStatus.APPROVED,
            approval_policy_id="approval_required_for_writes",
            approval_name="human",
            operation_id=operation.id,
        ),
    )


def _dependency_for_evidence(evidence: Evidence) -> TaskDependency:
    return TaskDependency(
        kind="evidence",
        evidence_kind=evidence.kind,
        evidence_id=evidence.id,
        evidence_owner=evidence.owner,
        producer_task_id=evidence.task_id,
        evidence_accepted=True,
        operation_id=evidence.operation_id,
        payload_fingerprint=evidence.metadata.get("payload_fingerprint")
        or persisted_fingerprint(evidence.payload),
    )
