"""Analysis checkpoint execution helpers for ``DbRuntime``."""

from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

from daita.runtime import (
    Capability,
    Evidence,
    GovernanceResult,
    Operation,
    OperationSnapshot,
    Task,
    TaskDependency,
)

if TYPE_CHECKING:
    from daita.plugins import ExtensionRegistry

    from ..types import _AnalysisPlanState

from ...analysis import analysis_metadata
from ...evidence import DbEvidenceStore
from ...fingerprints import persisted_fingerprint


class DbRuntimeAnalysisCheckpointMixin:
    if TYPE_CHECKING:
        registry: ExtensionRegistry

        def _analysis_progress_payload(
            self,
            snapshot: OperationSnapshot | None,
            *,
            plan_evidence: Evidence | None = None,
        ) -> dict[str, Any]: ...

        async def inspect_operation(
            self,
            operation_id: str,
        ) -> OperationSnapshot | None: ...

        async def _analysis_task(
            self,
            operation: Operation,
            capability: Capability,
            *,
            input: dict[str, Any],
            metadata: dict[str, Any],
            dependencies: tuple[TaskDependency, ...],
            sequence: int,
        ) -> Task: ...

        async def execute_task(
            self,
            task: Task,
            operation: Operation,
            context: dict[str, Any] | None = None,
        ) -> tuple[Evidence, ...]: ...

        async def _analysis_plan_state(
            self,
            operation_id: str,
            *,
            plan_evidence: Evidence,
            validation_evidence: Evidence,
        ) -> _AnalysisPlanState: ...

        def _accepted_analysis_step_evidence_map(
            self,
            evidence: Iterable[Evidence],
            *,
            analysis_id: str,
        ) -> dict[str, tuple[Evidence, ...]]: ...

        async def _execute_analysis_synthesis_task(
            self,
            operation: Operation,
            tasks: list[Task],
            evidence_store: DbEvidenceStore,
            *,
            analysis_id: str,
            step_id: str,
            plan_evidence: Evidence,
            cited_evidence: tuple[Evidence, ...],
            partial: bool = False,
            pause_reason: str | None = None,
            remaining_step_ids: tuple[str, ...] = (),
        ) -> Evidence: ...

    async def _execute_analysis_checkpoint_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        analysis_id: str,
        step_id: str,
        plan_evidence: Evidence,
        cited_evidence: tuple[Evidence, ...],
        remaining_step_ids: tuple[str, ...],
        diagnostics: dict[str, Any] | None = None,
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.analysis.checkpoint", owner="db_runtime"
        )
        dependencies = tuple(
            TaskDependency(
                kind="evidence",
                evidence_kind=item.kind,
                evidence_id=item.id,
                evidence_owner=item.owner,
                producer_task_id=item.task_id,
                evidence_accepted=item.accepted,
                operation_id=item.operation_id,
                payload_fingerprint=item.metadata.get("payload_fingerprint")
                or persisted_fingerprint(item.payload),
            )
            for item in cited_evidence
            if item.id
        )
        metadata = analysis_metadata(
            analysis_id=analysis_id,
            step_id=step_id,
            step_kind="checkpoint",
            plan_evidence_id=plan_evidence.id,
        )
        progress = self._analysis_progress_payload(
            await self.inspect_operation(operation.id),
            plan_evidence=plan_evidence,
        )
        checkpoint_diagnostics = {
            "checkpoint_reason": _checkpoint_reason(diagnostics or {}),
            "operation_status": operation.status.value,
            "progress": progress,
            **dict(diagnostics or {}),
        }
        task = await self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "analysis_step_id": step_id,
                "remaining_step_ids": list(remaining_step_ids),
                "diagnostics": checkpoint_diagnostics,
            },
            metadata=metadata,
            dependencies=dependencies,
            sequence=8000 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        checkpoint = next(
            (item for item in evidence if item.kind == "analysis.checkpoint"),
            None,
        )
        if checkpoint is None:
            raise RuntimeError("analysis.checkpoint evidence was not produced")
        return checkpoint

    async def _checkpoint_blocked_analysis_state(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        governance: GovernanceResult,
        evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence | None, Evidence | None]:
        plan_evidence = next(
            (
                item
                for item in reversed(evidence)
                if item.kind == "analysis.plan" and item.accepted
            ),
            None,
        )
        validation_evidence = next(
            (
                item
                for item in reversed(evidence)
                if item.kind == "analysis.plan.validation"
                and item.accepted
                and item.payload.get("valid") is True
            ),
            None,
        )
        if plan_evidence is None or validation_evidence is None:
            return None, None
        try:
            plan_state = await self._analysis_plan_state(
                operation.id,
                plan_evidence=plan_evidence,
                validation_evidence=validation_evidence,
            )
        except Exception:
            return None, None
        plan = plan_state.plan
        analysis_id = plan.analysis_id
        completed_by_step = self._accepted_analysis_step_evidence_map(
            evidence,
            analysis_id=analysis_id,
        )
        cited_evidence = tuple(
            item
            for values in completed_by_step.values()
            for item in values
            if item.accepted and item.id
        )
        remaining_step_ids = tuple(
            step.id for step in plan.steps if step.id not in completed_by_step
        )
        checkpoint = await self._execute_analysis_checkpoint_task(
            operation,
            tasks,
            evidence_store,
            analysis_id=analysis_id,
            step_id="analysis_blocked_checkpoint",
            plan_evidence=plan_state.selected_plan_evidence,
            cited_evidence=cited_evidence,
            remaining_step_ids=remaining_step_ids,
            diagnostics={
                "blocked_reason": "governance",
                "pending_approval": governance.pending_approval,
                "governance": governance.to_dict(),
            },
        )
        synthesis = None
        if cited_evidence:
            synthesis = await self._execute_analysis_synthesis_task(
                operation,
                tasks,
                evidence_store,
                analysis_id=analysis_id,
                step_id="analysis_blocked_partial_synthesis",
                plan_evidence=plan_state.selected_plan_evidence,
                cited_evidence=(*cited_evidence, checkpoint),
                partial=True,
                pause_reason=(
                    "approval_required"
                    if governance.pending_approval
                    else "governance_blocked"
                ),
                remaining_step_ids=remaining_step_ids,
            )
        return checkpoint, synthesis


def _checkpoint_reason(diagnostics: dict[str, object]) -> str:
    if diagnostics.get("budget_exceeded"):
        return "budget_exhausted"
    if diagnostics.get("blocked_reason"):
        return "blocked"
    if diagnostics.get("cancelled") or diagnostics.get("cancelled_reason"):
        return "cancelled"
    if diagnostics.get("interrupted") or diagnostics.get("error"):
        return "interrupted"
    if diagnostics.get("pause_reason"):
        return "paused"
    return "checkpoint"
