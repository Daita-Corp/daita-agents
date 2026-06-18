"""Analysis replan execution and plan revision helpers."""

from __future__ import annotations

from typing import Any

from daita.runtime import Evidence, Operation, Task

from ...analysis import (
    DbAnalysisPlan,
    analysis_metadata,
    validate_analysis_plan_payload,
)
from ...evidence import DbEvidenceStore
from ..types import _AnalysisPlanState
from .materialization import _dependency_for_evidence


class DbRuntimeAnalysisReplanMixin:
    async def _execute_analysis_replan_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        analysis_id: str,
        plan_evidence: Evidence,
        trigger_evidence: tuple[Evidence, ...],
        failed_step_ids: tuple[str, ...],
        budget_usage: dict[str, Any],
        retry_rationale: str,
        replacement_steps: tuple[dict[str, Any], ...] = (),
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.analysis.replan", owner="db_runtime"
        )
        dependencies = (
            _dependency_for_evidence(plan_evidence),
            *tuple(
                _dependency_for_evidence(item)
                for item in trigger_evidence
                if item.id and item.accepted and item.operation_id == operation.id
            ),
        )
        metadata = analysis_metadata(
            analysis_id=analysis_id,
            step_id="analysis_replan",
            phase="replan",
            plan_evidence_id=plan_evidence.id,
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "parent_plan_evidence_id": plan_evidence.id,
                "trigger_evidence_ids": [
                    item.id for item in trigger_evidence if item.id
                ],
                "failed_step_ids": list(failed_step_ids),
                "replacement_steps": [dict(item) for item in replacement_steps],
                "budget_usage": dict(budget_usage),
                "retry_rationale": retry_rationale,
            },
            metadata=metadata,
            dependencies=dependencies,
            sequence=8500 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        revision = next(
            (item for item in evidence if item.kind == "analysis.plan.revision"),
            None,
        )
        if revision is None:
            raise RuntimeError("analysis.plan.revision evidence was not produced")
        return revision

    async def _analysis_plan_state(
        self,
        operation_id: str,
        *,
        plan_evidence: Evidence,
        validation_evidence: Evidence,
    ) -> _AnalysisPlanState:
        parent_plan = DbAnalysisPlan.from_mapping(plan_evidence.payload)
        revision = await self._latest_accepted_plan_revision(
            operation_id,
            parent_plan_evidence_id=plan_evidence.id,
        )
        if revision is None:
            return _AnalysisPlanState(
                plan=parent_plan,
                plan_evidence=plan_evidence,
                validation_evidence=validation_evidence,
            )
        revised_plan = self._compose_analysis_revision_plan(
            parent_plan,
            revision,
        )
        validation = validate_analysis_plan_payload(
            revised_plan.to_dict(),
            plan_evidence=revision,
            registered_capabilities={
                (capability.owner, capability.id): capability
                for capability in self.registry.capabilities
            },
        )
        if not validation.valid:
            raise RuntimeError(
                "accepted analysis.plan.revision is not executable: "
                + ",".join(validation.errors)
            )
        return _AnalysisPlanState(
            plan=revised_plan,
            plan_evidence=plan_evidence,
            validation_evidence=validation_evidence,
            revision_evidence=revision,
        )

    async def _latest_accepted_plan_revision(
        self,
        operation_id: str,
        *,
        parent_plan_evidence_id: str | None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == "analysis.plan.revision"
            and evidence.accepted
            and evidence.payload.get("parent_plan_evidence_id")
            == parent_plan_evidence_id
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _compose_analysis_revision_plan(
        parent_plan: DbAnalysisPlan,
        revision: Evidence,
    ) -> DbAnalysisPlan:
        unchanged_step_ids = {
            str(item) for item in revision.payload.get("unchanged_step_ids") or ()
        }
        replacement_steps = [
            dict(item)
            for item in revision.payload.get("replacement_steps") or ()
            if isinstance(item, dict)
        ]
        parent_steps = [
            step.to_dict()
            for step in parent_plan.steps
            if step.id in unchanged_step_ids
        ]
        payload = {
            **parent_plan.to_dict(),
            "analysis_id": str(
                revision.payload.get("analysis_id") or parent_plan.analysis_id
            ),
            "steps": [*parent_steps, *replacement_steps],
            "budgets": dict(
                revision.payload.get("budgets") or parent_plan.budgets.to_dict()
            ),
            "diagnostics": {
                **parent_plan.diagnostics,
                "revision_evidence_id": revision.id,
                "revision_number": revision.payload.get("revision_number"),
            },
        }
        return DbAnalysisPlan.from_mapping(payload)
