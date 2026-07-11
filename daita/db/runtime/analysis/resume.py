"""Analysis resume progress helpers."""

from __future__ import annotations

from typing import Any

from daita.runtime import Evidence, OperationSnapshot, Task, TaskStatus

from ...analysis import DbAnalysisPlan
from ..resume import (
    _analysis_synthesis_is_partial,
    _latest_final_analysis_synthesis_from_snapshot,
)


class DbRuntimeAnalysisResumeMixin:
    def _analysis_progress_payload(
        self,
        snapshot: OperationSnapshot | None,
        *,
        plan_evidence: Evidence | None = None,
    ) -> dict[str, Any]:
        if snapshot is None:
            return {}
        plan_evidence = plan_evidence or _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.plan",
        )
        validation_evidence = _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.plan.validation",
            payload={"valid": True},
        )
        plan_steps: tuple[Any, ...] = ()
        budgets: dict[str, Any] = {}
        analysis_id = None
        if plan_evidence is not None:
            try:
                plan = DbAnalysisPlan.from_mapping(plan_evidence.payload)
                plan_steps = plan.steps
                budgets = plan.budgets.to_dict()
                analysis_id = plan.analysis_id
            except Exception:
                analysis_id = str(
                    plan_evidence.payload.get("analysis_id")
                    or plan_evidence.metadata.get("analysis_id")
                    or ""
                )
                budgets = dict(plan_evidence.payload.get("budgets") or {})
        completed_steps = {
            str(item.metadata.get("analysis_step_id"))
            for item in snapshot.evidence
            if item.accepted
            and item.metadata.get("analysis_step_id")
            and item.kind
            not in {
                "analysis.plan",
                "analysis.plan.validation",
                "planning.context",
            }
            and (
                item.kind != "analysis.synthesis"
                or not _analysis_synthesis_is_partial(item)
            )
        }
        running_steps = sorted(
            {
                str(task.metadata.get("analysis_step_id"))
                for task in snapshot.tasks
                if task.status is TaskStatus.RUNNING
                and task.metadata.get("analysis_step_id")
            }
        )
        blocked_steps = sorted(
            {
                str(task.metadata.get("analysis_step_id"))
                for task in snapshot.tasks
                if task.status is TaskStatus.BLOCKED
                and task.metadata.get("analysis_step_id")
            }
        )
        remaining_step_ids = [
            step.id for step in plan_steps if step.id not in completed_steps
        ]
        latest_checkpoint = _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.checkpoint",
        )
        latest_synthesis = _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.synthesis",
        )
        final_synthesis = _latest_final_analysis_synthesis_from_snapshot(snapshot)
        approval_task_ids = {
            event.approval_id: event.task_id
            for event in snapshot.events
            if event.approval_id is not None and event.task_id is not None
        }
        return {
            "operation_id": snapshot.operation.id,
            "operation_status": snapshot.operation.status.value,
            "analysis_id": analysis_id,
            "plan_evidence_id": plan_evidence.id if plan_evidence else None,
            "validation_evidence_id": (
                validation_evidence.id if validation_evidence else None
            ),
            "completed_step_ids": sorted(completed_steps),
            "blocked_step_ids": blocked_steps,
            "running_step_ids": running_steps,
            "remaining_step_ids": remaining_step_ids,
            "budgets": budgets,
            "approvals": [
                {
                    "approval_id": approval.approval_id,
                    "status": approval.to_dict()["status"],
                    "task_id": approval_task_ids.get(approval.approval_id),
                    "policy_id": approval.requested_by_policy_id,
                }
                for approval in snapshot.approval_requests
            ],
            "next_resumable_task_ids": list(snapshot.resumable_task_ids),
            "latest_checkpoint_id": (
                latest_checkpoint.id if latest_checkpoint else None
            ),
            "latest_synthesis_id": latest_synthesis.id if latest_synthesis else None,
            "latest_synthesis_partial": (
                _analysis_synthesis_is_partial(latest_synthesis)
                if latest_synthesis is not None
                else None
            ),
            "final_synthesis_id": (
                final_synthesis.id if final_synthesis is not None else None
            ),
            "task_status_counts": _task_status_counts(snapshot.tasks),
            "evidence_counts": _evidence_kind_counts(snapshot.evidence),
        }


def _latest_accepted_evidence_from_snapshot(
    snapshot: OperationSnapshot,
    kind: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in snapshot.evidence
        if evidence.kind == kind
        and evidence.accepted
        and _payload_contains(evidence.payload, payload or {})
    ]
    return matches[-1] if matches else None


def _task_status_counts(tasks: tuple[Task, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.status.value] = counts.get(task.status.value, 0) + 1
    return counts


def _evidence_kind_counts(evidence: tuple[Evidence, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    return counts


def _payload_contains(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True
