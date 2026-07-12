"""Evidence matching shared by DB monitor extension executors."""

from __future__ import annotations

from typing import Any

from daita.runtime import Evidence, Operation, Task

from ...evidence import load_evidence


async def load_monitor_proposal_evidence(
    runtime: Any,
    operation: Operation,
    task: Task,
    evidence_id: Any,
) -> Evidence | None:
    """Load an explicit, dependency-matched, or latest monitor proposal."""
    explicit = await load_evidence(runtime, operation.id, evidence_id)
    if explicit is not None:
        return explicit
    evidence = await runtime.store.list_evidence(operation.id)
    for dependency in task.dependencies:
        if dependency.kind_value != "evidence":
            continue
        if dependency.evidence_kind != "monitor.proposal":
            continue
        for item in reversed(evidence):
            if evidence_matches_dependency(item, dependency):
                return item
    for item in reversed(evidence):
        if item.kind == "monitor.proposal" and item.accepted:
            return item
    return None


def evidence_matches_dependency(evidence: Evidence, dependency: Any) -> bool:
    """Return whether evidence satisfies the complete dependency predicate."""
    if evidence.kind != dependency.evidence_kind:
        return False
    if dependency.evidence_id is not None and evidence.id != dependency.evidence_id:
        return False
    if (
        dependency.evidence_owner is not None
        and evidence.owner != dependency.evidence_owner
    ):
        return False
    if (
        dependency.producer_task_id is not None
        and evidence.task_id != dependency.producer_task_id
    ):
        return False
    if evidence.accepted is not dependency.evidence_accepted:
        return False
    for key, value in dependency.evidence_payload.items():
        if evidence.payload.get(key) != value:
            return False
    return True
