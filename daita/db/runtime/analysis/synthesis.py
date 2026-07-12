"""Analysis synthesis execution helpers for ``DbRuntime``."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from daita.runtime import Capability, Evidence, Operation, Task, TaskDependency

if TYPE_CHECKING:
    from daita.plugins import ExtensionRegistry
    from daita.runtime import RuntimeStore

from ...analysis import analysis_metadata
from ...evidence import DbEvidenceStore
from ..resume import _analysis_synthesis_is_partial
from .materialization import _dependency_for_evidence


class DbRuntimeAnalysisSynthesisMixin:
    if TYPE_CHECKING:
        registry: ExtensionRegistry
        store: RuntimeStore

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
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.analysis.summarize", owner="db_runtime"
        )
        dependencies = tuple(
            _dependency_for_evidence(item)
            for item in cited_evidence
            if item.id and item.accepted and item.operation_id == operation.id
        )
        metadata = analysis_metadata(
            analysis_id=analysis_id,
            step_id=step_id,
            step_kind="synthesis",
            plan_evidence_id=plan_evidence.id,
        )
        task = await self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "analysis_step_id": step_id,
                "partial": partial,
                "pause_reason": pause_reason,
                "remaining_step_ids": list(remaining_step_ids),
            },
            metadata=metadata,
            dependencies=dependencies,
            sequence=9000 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        synthesis = next(
            (item for item in evidence if item.kind == "analysis.synthesis"),
            None,
        )
        if synthesis is None:
            raise RuntimeError("analysis.synthesis evidence was not produced")
        return synthesis

    async def _latest_final_analysis_synthesis(
        self,
        operation_id: str,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == "analysis.synthesis"
            and evidence.accepted
            and not _analysis_synthesis_is_partial(evidence)
        ]
        return matches[-1] if matches else None


def _answer_from_analysis_synthesis_evidence(evidence: Evidence) -> str:
    answer = evidence.payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("accepted analysis.synthesis evidence is missing answer")
    return answer
