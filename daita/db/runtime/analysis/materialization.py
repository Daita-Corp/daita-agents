"""Multi-step analysis materialization for ``DbRuntime``."""

from __future__ import annotations

import asyncio
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any
from uuid import uuid4

from daita.runtime import (
    Capability,
    Evidence,
    Operation,
    OperationStatus,
    RuntimeEventType,
    Task,
    TaskDependency,
)

from ...analysis import (
    DbAnalysisPlan,
    analysis_metadata,
    capability_contract_for_step_kind,
    evidence_ref,
    stable_fingerprint,
)
from ...evidence import DbEvidenceStore
from ...models import (
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from ...query_sql_validation import sql_fingerprint
from ..cache import _from_db_options
from ..types import (
    DbRuntimeGovernanceBlocked,
    _governance_blocked_answer,
    _governance_blocked_warning,
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
        or _payload_fingerprint(evidence.payload),
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return _stable_hash(payload)


from .budgets import DbRuntimeAnalysisBudgetMixin
from .checkpoints import DbRuntimeAnalysisCheckpointMixin
from .replan import DbRuntimeAnalysisReplanMixin
from .resume import DbRuntimeAnalysisResumeMixin
from .synthesis import (
    DbRuntimeAnalysisSynthesisMixin,
    _answer_from_analysis_synthesis_evidence,
)


class DbRuntimeAnalysisMixin(
    DbRuntimeAnalysisCheckpointMixin,
    DbRuntimeAnalysisSynthesisMixin,
    DbRuntimeAnalysisReplanMixin,
    DbRuntimeAnalysisBudgetMixin,
    DbRuntimeAnalysisResumeMixin,
):
    def _should_route_multi_step_analysis(
        self,
        request: DbRequest,
        contract: DbOperationContract,
    ) -> bool:
        granted_lanes = set(contract.metadata.get("granted_lanes") or ())
        if "read" not in granted_lanes and contract.access.value != "read":
            return False
        if request.metadata.get("analysis_mode") == "multi_step":
            return True
        prompt = request.prompt.lower()
        explicit = (
            "multi-step",
            "multiple queries",
            "long-running",
            "deep analysis",
            "investigate",
            "explain why",
            "why did",
            "root cause",
            "break down and compare",
            "compare drivers",
            "then drill",
            "step by step analysis",
        )
        return self.db_llm_service.available and any(
            term in prompt for term in explicit
        )

    async def _run_multi_step_analysis(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        *,
        base_diagnostics: dict[str, Any],
        reuse_existing_plan: bool = False,
    ) -> DbOperationResult:
        evidence_store = DbEvidenceStore()
        tasks: list[Task] = []
        warnings: list[str] = []
        started_at = time.monotonic()
        diagnostics: dict[str, Any] = {
            "planner_strategy": "analysis",
            "phase": "multi_step_analysis",
            "evidence_refs": [],
            "evidence_kinds": [],
        }
        try:
            schema_evidence = await self._inspect_analysis_schema_if_available(
                operation, tasks, evidence_store
            )
            schema = schema_evidence.payload if schema_evidence is not None else {}
            planning_context = await self._execute_analysis_planning_context_task(
                request,
                operation,
                tasks,
                evidence_store,
                schema_evidence=schema_evidence,
                catalog_evidence=tuple(
                    item
                    for item in evidence_store.list()
                    if item.kind.startswith("catalog.")
                    or item.kind in {"schema.search_result", "catalog.source"}
                ),
                relationship_evidence=(),
                analysis_metadata={
                    "analysis_id": f"analysis-{operation.id}",
                    "analysis_step_id": "analysis_context",
                    "analysis_phase": "context",
                },
            )
            plan_evidence = (
                await self._latest_accepted_evidence(operation.id, "analysis.plan")
                if reuse_existing_plan
                else None
            )
            validation_evidence = (
                await self._latest_accepted_evidence(
                    operation.id, "analysis.plan.validation", payload={"valid": True}
                )
                if plan_evidence is not None
                else None
            )
            if plan_evidence is None:
                plan_evidence = await self._execute_analysis_plan_task(
                    operation,
                    tasks,
                    evidence_store,
                    planning_context=planning_context,
                )
            if validation_evidence is None:
                validation_evidence = await self._execute_analysis_validation_task(
                    operation,
                    tasks,
                    evidence_store,
                    plan_evidence=plan_evidence,
                )
            if not plan_evidence.accepted or not validation_evidence.accepted:
                warnings.append("analysis_plan_unavailable_or_invalid")
                checkpoint = await self._execute_analysis_checkpoint_task(
                    operation,
                    tasks,
                    evidence_store,
                    analysis_id=str(
                        plan_evidence.payload.get("analysis_id")
                        or plan_evidence.metadata.get("analysis_id")
                        or f"analysis-{operation.id}"
                    ),
                    step_id="analysis_blocked",
                    plan_evidence=plan_evidence,
                    cited_evidence=(plan_evidence, validation_evidence),
                    remaining_step_ids=(),
                    diagnostics={"blocked_reason": "analysis_plan_invalid"},
                )
                all_evidence = (*evidence_store.list(), checkpoint)
                return await self._record_operation_result(
                    DbOperationResult(
                        operation_id=operation.id,
                        request=request,
                        contract=contract,
                        status=OperationStatus.BLOCKED,
                        answer=str(
                            plan_evidence.payload.get("clarification_question")
                            or "The multi-step analysis plan could not be validated."
                        ),
                        evidence=all_evidence,
                        warnings=tuple(warnings),
                        diagnostics={
                            **base_diagnostics,
                            "execution": {
                                **diagnostics,
                                "tasks": [task.to_dict() for task in tasks],
                            },
                            "analysis_plan_validation": validation_evidence.payload,
                        },
                    ),
                    operation=operation,
                )

            plan_state = await self._analysis_plan_state(
                operation.id,
                plan_evidence=plan_evidence,
                validation_evidence=validation_evidence,
            )
            plan = plan_state.plan
            selected_plan_evidence = plan_state.selected_plan_evidence
            analysis_id = plan.analysis_id
            completed_by_step = self._accepted_analysis_step_evidence_map(
                await self.store.list_evidence(operation.id),
                analysis_id=analysis_id,
            )
            while True:
                budget_failure = self._analysis_budget_failure(
                    plan,
                    tuple(await self.store.list_evidence(operation.id)),
                    started_at=started_at,
                )
                if budget_failure is not None:
                    cited_evidence = tuple(
                        evidence
                        for values in completed_by_step.values()
                        for evidence in values
                        if evidence.accepted and evidence.id
                    )
                    remaining_step_ids = tuple(
                        item.id
                        for item in plan.steps
                        if item.id not in completed_by_step
                    )
                    checkpoint = await self._execute_analysis_checkpoint_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id="budget_checkpoint",
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=cited_evidence,
                        remaining_step_ids=remaining_step_ids,
                        diagnostics=budget_failure,
                    )
                    partial_synthesis = await self._execute_analysis_synthesis_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id="budget_partial_synthesis",
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=(*cited_evidence, checkpoint),
                        partial=True,
                        pause_reason="budget_exhausted",
                        remaining_step_ids=remaining_step_ids,
                    )
                    if _analysis_replan_enabled(self.config.metadata):
                        await self._execute_analysis_replan_task(
                            operation,
                            tasks,
                            evidence_store,
                            analysis_id=analysis_id,
                            plan_evidence=selected_plan_evidence,
                            trigger_evidence=(checkpoint,),
                            failed_step_ids=remaining_step_ids,
                            budget_usage=dict(budget_failure.get("budget_usage") or {}),
                            retry_rationale="budget_exhausted",
                        )
                    all_evidence = tuple(await self.store.list_evidence(operation.id))
                    all_tasks = tuple(await self.store.list_tasks(operation.id))
                    return await self._record_operation_result(
                        DbOperationResult(
                            operation_id=operation.id,
                            request=request,
                            contract=contract,
                            status=OperationStatus.BLOCKED,
                            answer=_answer_from_analysis_synthesis_evidence(
                                partial_synthesis
                            ),
                            evidence=all_evidence,
                            warnings=tuple((*warnings, "analysis_budget_exhausted")),
                            diagnostics={
                                **base_diagnostics,
                                "execution": {
                                    **diagnostics,
                                    "evidence_kinds": [
                                        item.kind for item in all_evidence
                                    ],
                                    "evidence_refs": [item.id for item in all_evidence],
                                    "task_count": len(all_tasks),
                                    "tasks": [task.to_dict() for task in all_tasks],
                                },
                                "analysis": partial_synthesis.payload,
                                "budget": budget_failure,
                            },
                        ),
                        operation=operation,
                    )
                ready_steps = self._select_ready_analysis_steps(
                    plan,
                    completed_by_step,
                    serial=not _analysis_parallel_enabled(self.config.metadata),
                )
                if not ready_steps:
                    break
                parallel_query_steps = tuple(
                    step for step in ready_steps if step.kind == "query"
                )
                if (
                    _analysis_parallel_enabled(self.config.metadata)
                    and len(parallel_query_steps) > 1
                    and len(parallel_query_steps) == len(ready_steps)
                ):
                    parallel_results = (
                        await self._execute_parallel_analysis_query_steps(
                            request,
                            contract,
                            operation,
                            schema=schema,
                            schema_evidence=schema_evidence,
                            plan=plan,
                            plan_evidence=selected_plan_evidence,
                            steps=parallel_query_steps,
                        )
                    )
                    for step, produced, step_tasks in parallel_results:
                        tasks.extend(step_tasks)
                        evidence_store.add_many(produced)
                        completed_by_step[step.id] = produced
                        step_budget_failure = self._analysis_step_budget_failure(
                            step,
                            produced,
                        )
                        checkpoint = await self._execute_analysis_checkpoint_task(
                            operation,
                            tasks,
                            evidence_store,
                            analysis_id=analysis_id,
                            step_id=f"{step.id}_checkpoint",
                            plan_evidence=selected_plan_evidence,
                            cited_evidence=produced,
                            remaining_step_ids=tuple(
                                item.id
                                for item in plan.steps
                                if item.id not in completed_by_step
                            ),
                            diagnostics=step_budget_failure or {"parallel_batch": True},
                        )
                        completed_by_step.setdefault(
                            f"{step.id}_checkpoint",
                            (checkpoint,),
                        )
                    continue
                step = ready_steps[0]
                dependencies = tuple(
                    evidence
                    for dependency_id in step.depends_on
                    for evidence in completed_by_step.get(dependency_id, ())
                    if evidence.accepted and evidence.id
                )
                step_meta = analysis_metadata(
                    analysis_id=analysis_id,
                    step_id=step.id,
                    step_kind=step.kind,
                    plan_evidence_id=selected_plan_evidence.id,
                )
                if step.kind == "query":
                    produced = await self._execute_analysis_query_step(
                        request,
                        contract,
                        operation,
                        tasks,
                        evidence_store,
                        schema=schema,
                        schema_evidence=schema_evidence,
                        step_prompt=f"{request.prompt}\n\nAnalysis step {step.id}: {step.purpose}",
                        step_metadata=step_meta,
                        context_dependencies=await self._analysis_context_dependencies(
                            operation.id,
                            step.context_evidence_refs,
                        ),
                    )
                    verification_evidence = (
                        await self._persist_analysis_verification_result_evidence(
                            operation,
                            contract,
                            produced,
                            tuple(tasks),
                            step_metadata=step_meta,
                        )
                    )
                    evidence_store.add(verification_evidence)
                    produced = (*produced, verification_evidence)
                    completed_by_step[step.id] = produced
                    step_budget_failure = self._analysis_step_budget_failure(
                        step,
                        produced,
                    )
                    checkpoint = await self._execute_analysis_checkpoint_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id=f"{step.id}_checkpoint",
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=produced,
                        remaining_step_ids=tuple(
                            item.id
                            for item in plan.steps
                            if item.id not in completed_by_step
                        ),
                        diagnostics=step_budget_failure or {},
                    )
                    completed_by_step.setdefault(f"{step.id}_checkpoint", (checkpoint,))
                    if step_budget_failure is not None:
                        break
                elif step.kind == "checkpoint":
                    checkpoint = await self._execute_analysis_checkpoint_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id=step.id,
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=dependencies,
                        remaining_step_ids=tuple(
                            item.id
                            for item in plan.steps
                            if item.id not in completed_by_step
                        ),
                    )
                    completed_by_step[step.id] = (checkpoint,)
                elif step.kind == "synthesis":
                    synthesis = await self._execute_analysis_synthesis_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id=step.id,
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=dependencies
                        or tuple(
                            evidence
                            for values in completed_by_step.values()
                            for evidence in values
                            if evidence.accepted and evidence.id
                        ),
                    )
                    completed_by_step[step.id] = (synthesis,)
                elif capability_contract_for_step_kind(step.kind) is not None:
                    produced = await self._execute_analysis_capability_step(
                        operation,
                        tasks,
                        evidence_store,
                        step=step,
                        step_metadata=step_meta,
                        dependency_evidence=dependencies,
                    )
                    completed_by_step[step.id] = produced

            synthesis = await self._latest_final_analysis_synthesis(operation.id)
            if synthesis is None:
                synthesis = await self._execute_analysis_synthesis_task(
                    operation,
                    tasks,
                    evidence_store,
                    analysis_id=analysis_id,
                    step_id="final_synthesis",
                    plan_evidence=selected_plan_evidence,
                    cited_evidence=tuple(
                        evidence
                        for values in completed_by_step.values()
                        for evidence in values
                        if evidence.accepted and evidence.id
                    ),
                )
            all_evidence = tuple(await self.store.list_evidence(operation.id))
            all_tasks = tuple(await self.store.list_tasks(operation.id))
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation.id,
                    request=request,
                    contract=contract,
                    status=OperationStatus.SUCCEEDED,
                    answer=_answer_from_analysis_synthesis_evidence(synthesis),
                    evidence=all_evidence,
                    warnings=tuple(warnings),
                    diagnostics={
                        **base_diagnostics,
                        "execution": {
                            **diagnostics,
                            "evidence_kinds": [item.kind for item in all_evidence],
                            "evidence_refs": [item.id for item in all_evidence],
                            "task_count": len(all_tasks),
                            "tasks": [task.to_dict() for task in all_tasks],
                        },
                        "analysis": synthesis.payload,
                    },
                ),
                operation=operation,
            )
        except DbRuntimeGovernanceBlocked as exc:
            blocked_evidence = tuple(await self.store.list_evidence(operation.id))
            blocked_checkpoint, blocked_synthesis = (
                await self._checkpoint_blocked_analysis_state(
                    operation,
                    tasks,
                    evidence_store,
                    governance=exc.governance,
                    evidence=blocked_evidence,
                )
            )
            final_blocked_evidence = tuple(await self.store.list_evidence(operation.id))
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation.id,
                    request=request,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer=(
                        _answer_from_analysis_synthesis_evidence(blocked_synthesis)
                        if blocked_synthesis is not None
                        else _governance_blocked_answer(exc.governance)
                    ),
                    evidence=final_blocked_evidence,
                    warnings=(_governance_blocked_warning(exc.governance),),
                    diagnostics={
                        **base_diagnostics,
                        "governance": exc.governance.to_dict(),
                        "analysis_checkpoint_id": (
                            blocked_checkpoint.id if blocked_checkpoint else None
                        ),
                        "analysis_partial_synthesis_id": (
                            blocked_synthesis.id if blocked_synthesis else None
                        ),
                    },
                ),
                operation=operation,
            )
        except Exception as exc:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation.id,
                    request=request,
                    contract=contract,
                    status=OperationStatus.FAILED,
                    answer=f"DB analysis failed: {exc}",
                    evidence=tuple(await self.store.list_evidence(operation.id)),
                    warnings=("db_runtime_analysis_failed",),
                    diagnostics={
                        **base_diagnostics,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                ),
                operation=operation,
            )

    async def _execute_analysis_plan_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence,
    ) -> Evidence:
        analysis_id = f"analysis-{operation.id}"
        capability = self.registry.get_capability(
            "db.analysis.plan", owner="db_runtime"
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "planning_context_evidence_id": planning_context.id,
            },
            metadata=analysis_metadata(
                analysis_id=analysis_id,
                step_id="analysis_plan",
                phase="plan",
            ),
            dependencies=(_dependency_for_evidence(planning_context),),
            sequence=100,
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        plan = next((item for item in evidence if item.kind == "analysis.plan"), None)
        if plan is None:
            raise RuntimeError("analysis.plan evidence was not produced")
        return plan

    async def _execute_analysis_validation_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        plan_evidence: Evidence,
    ) -> Evidence:
        analysis_id = str(
            plan_evidence.payload.get("analysis_id")
            or plan_evidence.metadata.get("analysis_id")
            or f"analysis-{operation.id}"
        )
        capability = self.registry.get_capability(
            "db.analysis.plan.validate", owner="db_runtime"
        )
        task = self._analysis_task(
            operation,
            capability,
            input={"analysis_plan_evidence_id": plan_evidence.id},
            metadata=analysis_metadata(
                analysis_id=analysis_id,
                step_id="analysis_plan_validation",
                plan_evidence_id=plan_evidence.id,
                phase="validation",
            ),
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=plan_evidence.kind,
                    evidence_id=plan_evidence.id,
                    evidence_owner=plan_evidence.owner,
                    producer_task_id=plan_evidence.task_id,
                    evidence_accepted=plan_evidence.accepted,
                    operation_id=plan_evidence.operation_id,
                    payload_fingerprint=plan_evidence.metadata.get(
                        "payload_fingerprint"
                    )
                    or _payload_fingerprint(plan_evidence.payload),
                ),
            ),
            sequence=101,
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        validation = next(
            (item for item in evidence if item.kind == "analysis.plan.validation"),
            None,
        )
        if validation is None:
            raise RuntimeError("analysis.plan.validation evidence was not produced")
        return validation

    async def _execute_analysis_query_step(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        schema: dict[str, Any],
        schema_evidence: Evidence | None,
        step_prompt: str,
        step_metadata: dict[str, Any],
        context_dependencies: tuple[TaskDependency, ...] = (),
    ) -> tuple[Evidence, ...]:
        before_ids = {item.id for item in await self.store.list_evidence(operation.id)}
        step_request = dataclass_replace(request, prompt=step_prompt)
        planning_context = await self._execute_analysis_planning_context_task(
            step_request,
            operation,
            tasks,
            evidence_store,
            schema_evidence=schema_evidence,
            catalog_evidence=(),
            relationship_evidence=(),
            analysis_metadata=step_metadata,
            extra_dependencies=context_dependencies,
        )
        plan_evidence, strategy_warnings, _ = (
            await self._execute_analysis_query_plan_task(
                step_request,
                operation,
                schema,
                None,
                planning_context,
                tasks,
                evidence_store,
                analysis_metadata=step_metadata,
                extra_dependencies=context_dependencies,
            )
        )
        validation = await self._execute_analysis_query_validation_task(
            operation,
            tasks,
            evidence_store,
            plan_evidence=plan_evidence,
            planning_context=planning_context,
            analysis_metadata=step_metadata,
            extra_dependencies=context_dependencies,
        )
        sql = validation.payload.get("accepted_sql")
        if sql:
            await self._execute_analysis_validated_read_pair(
                contract,
                operation,
                tasks,
                evidence_store,
                sql,
                plan_validation=validation,
                step_metadata=step_metadata,
                extra_dependencies=context_dependencies,
            )
        produced = tuple(
            item
            for item in await self.store.list_evidence(operation.id)
            if item.id not in before_ids
            and item.metadata.get("analysis_step_id")
            == step_metadata.get("analysis_step_id")
        )
        if strategy_warnings:
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation.id,
                message="Analysis query step produced planner warnings.",
                payload={
                    "analysis_step_id": step_metadata.get("analysis_step_id"),
                    "warnings": list(strategy_warnings),
                },
            )
        return produced

    async def _execute_parallel_analysis_query_steps(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        *,
        schema: dict[str, Any],
        schema_evidence: Evidence | None,
        plan: DbAnalysisPlan,
        plan_evidence: Evidence,
        steps: tuple[Any, ...],
    ) -> tuple[tuple[Any, tuple[Evidence, ...], tuple[Task, ...]], ...]:
        semaphore = asyncio.Semaphore(_analysis_max_concurrency(self.config.metadata))

        async def run_step(
            step: Any,
        ) -> tuple[Any, tuple[Evidence, ...], tuple[Task, ...]]:
            async with semaphore:
                step_tasks: list[Task] = []
                step_evidence_store = DbEvidenceStore()
                step_meta = analysis_metadata(
                    analysis_id=plan.analysis_id,
                    step_id=step.id,
                    step_kind=step.kind,
                    plan_evidence_id=plan_evidence.id,
                )
                produced = await self._execute_analysis_query_step(
                    request,
                    contract,
                    operation,
                    step_tasks,
                    step_evidence_store,
                    schema=schema,
                    schema_evidence=schema_evidence,
                    step_prompt=(
                        f"{request.prompt}\n\nAnalysis step {step.id}: {step.purpose}"
                    ),
                    step_metadata=step_meta,
                    context_dependencies=await self._analysis_context_dependencies(
                        operation.id,
                        step.context_evidence_refs,
                    ),
                )
                verification_evidence = (
                    await self._persist_analysis_verification_result_evidence(
                        operation,
                        contract,
                        produced,
                        tuple(step_tasks),
                        step_metadata=step_meta,
                    )
                )
                produced = (*produced, verification_evidence)
                return step, produced, tuple(step_tasks)

        results = await asyncio.gather(*(run_step(step) for step in steps))
        return tuple(sorted(results, key=lambda item: item[0].id))

    async def _inspect_analysis_schema_if_available(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
    ) -> Evidence | None:
        cached = self.cached_schema_evidence(operation_id=operation.id)
        if cached is not None:
            persisted = await self._persist_analysis_runtime_evidence(
                operation,
                _with_schema_scope(cached, "database"),
            )
            evidence_store.add(persisted)
            return persisted
        persisted_schema = self.persisted_schema_evidence(operation_id=operation.id)
        if persisted_schema is not None:
            persisted = await self._persist_analysis_runtime_evidence(
                operation,
                _with_schema_scope(persisted_schema, "database"),
            )
            evidence_store.add(persisted)
            return persisted
        capability = _first_capability(self.registry.capabilities, "db.schema.inspect")
        if capability is None:
            return None
        task = self._analysis_task(
            operation,
            capability,
            input={},
            metadata={"analysis_phase": "context"},
            dependencies=(),
            sequence=1,
        )
        tasks.append(task)
        try:
            evidence = await self.execute_task(task, operation)
        except Exception as exc:
            fallback = self.stale_persisted_schema_evidence(
                operation_id=operation.id,
                error=exc,
            )
            if fallback is None:
                raise
            persisted = await self._persist_analysis_runtime_evidence(
                operation,
                _with_schema_scope(fallback, "database"),
            )
            evidence_store.add(persisted)
            return persisted
        evidence_store.add_many(evidence)
        schema = next(
            (item for item in evidence if item.kind == "schema.asset_profile"),
            None,
        )
        if schema is None:
            return None
        scoped = _with_schema_scope(schema, "database")
        self.remember_schema_evidence(scoped)
        return scoped

    async def _execute_analysis_planning_context_task(
        self,
        request: DbRequest,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        schema_evidence: Evidence | None,
        catalog_evidence: tuple[Evidence, ...],
        relationship_evidence: tuple[Evidence, ...],
        analysis_metadata: dict[str, Any],
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.planning.context.build", owner="db_runtime"
        )
        dependencies = (
            *(
                _dependency_for_evidence(item)
                for item in (
                    *((schema_evidence,) if schema_evidence is not None else ()),
                    *catalog_evidence,
                    *relationship_evidence,
                )
                if item.id and item.accepted
            ),
            *extra_dependencies,
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                "prompt": request.prompt,
                "schema_evidence_id": schema_evidence.id if schema_evidence else None,
                "catalog_evidence_ids": [
                    item.id for item in catalog_evidence if item.id is not None
                ],
                "relationship_evidence_ids": [
                    item.id for item in relationship_evidence if item.id is not None
                ],
                "memory_recall_evidence_ids": [],
                "memory_recall_diagnostics": {
                    "registered": False,
                    "queried": False,
                    "decision": {"recall": False, "reason": "analysis_phase_7"},
                },
            },
            metadata=analysis_metadata,
            dependencies=dependencies,
            sequence=200 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        context = next(
            (item for item in evidence if item.kind == "planning.context"),
            None,
        )
        if context is None:
            raise RuntimeError("planning.context evidence was not produced")
        return context

    async def _execute_analysis_query_plan_task(
        self,
        request: DbRequest,
        operation: Operation,
        schema: dict[str, Any],
        relationship_payload: dict[str, Any] | None,
        planning_context: Evidence,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        analysis_metadata: dict[str, Any],
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> tuple[Evidence, tuple[str, ...], dict[str, Any]]:
        del schema, relationship_payload
        capability = self.registry.get_capability("db.query.plan", owner="db_runtime")
        task = self._analysis_task(
            operation,
            capability,
            input={
                "planning_context_evidence_id": planning_context.id,
                "prompt": request.prompt,
            },
            metadata=analysis_metadata,
            dependencies=(
                _dependency_for_evidence(planning_context),
                *extra_dependencies,
            ),
            sequence=300 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        plan = next(
            (item for item in evidence if item.kind == "query.plan.proposal"),
            None,
        )
        if plan is None:
            raise RuntimeError("query.plan.proposal evidence was not produced")
        return plan, (), {"strategy": "llm", "planner": "llm"}

    async def _execute_analysis_query_validation_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        plan_evidence: Evidence,
        planning_context: Evidence,
        analysis_metadata: dict[str, Any],
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.query.plan.validate", owner="db_runtime"
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                "plan_evidence_id": plan_evidence.id,
                "planning_context_evidence_id": planning_context.id,
            },
            metadata=analysis_metadata,
            dependencies=(
                _dependency_for_evidence(plan_evidence),
                _dependency_for_evidence(planning_context),
                *extra_dependencies,
            ),
            sequence=400 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        validation = next(
            (item for item in evidence if item.kind == "query.plan.validation"),
            None,
        )
        if validation is None:
            raise RuntimeError("query.plan.validation evidence was not produced")
        return validation

    async def _execute_analysis_validated_read_pair(
        self,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        sql: str,
        *,
        plan_validation: Evidence,
        step_metadata: dict[str, Any],
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> tuple[Evidence, ...]:
        owner = _selected_capability_owner(contract, "db.sql.execute_read")
        validation_dependency = TaskDependency(
            kind="evidence",
            evidence_kind="query.plan.validation",
            evidence_id=plan_validation.id,
            evidence_owner=plan_validation.owner,
            producer_task_id=plan_validation.task_id,
            evidence_payload={"valid": True},
            evidence_accepted=True,
            operation_id=operation.id,
            payload_fingerprint=plan_validation.metadata.get("payload_fingerprint"),
        )
        read_result = await self.execute_validated_read_task_pair(
            operation,
            sql=sql,
            owner=owner,
            reason="analysis_validated_read",
            sequence=500 + len(tasks),
            metadata=step_metadata,
            validation_dependencies=(validation_dependency, *extra_dependencies),
            read_dependencies=extra_dependencies,
            deterministic_key=_stable_hash(
                {
                    "analysis_id": step_metadata.get("analysis_id"),
                    "analysis_step_id": step_metadata.get("analysis_step_id"),
                    "sql_fingerprint": sql_fingerprint(sql),
                }
            ),
        )
        tasks.extend((read_result.validation_task, read_result.read_task))
        evidence_store.add_many(read_result.evidence)
        if read_result.status != "succeeded":
            raise RuntimeError(read_result.block_reason or "analysis_sql_read_failed")
        return read_result.evidence

    async def _persist_analysis_runtime_evidence(
        self,
        operation: Operation,
        evidence: Evidence,
    ) -> Evidence:
        persisted = dataclass_replace(
            evidence,
            id=evidence.id or f"evidence-{uuid4()}",
            operation_id=evidence.operation_id or operation.id,
            metadata={
                **evidence.metadata,
                "payload_fingerprint": _payload_fingerprint(evidence.payload),
            },
        )
        await self.store.save_evidence(persisted)
        return persisted

    async def _persist_analysis_verification_result_evidence(
        self,
        operation: Operation,
        contract: DbOperationContract,
        evidence: tuple[Evidence, ...],
        tasks: tuple[Task, ...],
        *,
        step_metadata: dict[str, Any],
    ) -> Evidence:
        verification = self.verifier.verify(contract, evidence, tasks)
        evidence_details = [
            evidence_ref(item) for item in evidence if item.accepted and item.id
        ]
        payload = {
            "passed": bool(verification.passed),
            "evidence_refs": [item["id"] for item in evidence_details],
            "evidence_details": evidence_details,
            "warnings": list(verification.warnings),
            "missing_evidence": list(verification.missing_evidence),
            "diagnostics": verification.diagnostics,
            "input_fingerprint": stable_fingerprint(evidence_details),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        verification_evidence = Evidence(
            id=f"evidence-{uuid4()}",
            kind="verification.result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=verification.passed,
            payload=payload,
            metadata={
                **step_metadata,
                "payload_fingerprint": stable_fingerprint(payload),
                "input_fingerprint": payload["input_fingerprint"],
            },
        )
        await self.store.save_evidence(verification_evidence)
        return verification_evidence

    async def _execute_analysis_capability_step(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        step: Any,
        step_metadata: dict[str, Any],
        dependency_evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence, ...]:
        if not step.capability_id or not step.capability_owner:
            raise RuntimeError(
                f"analysis capability step {step.id} is missing capability"
            )
        capability = self.registry.get_capability(
            step.capability_id,
            owner=step.capability_owner,
        )
        contract = capability_contract_for_step_kind(step.kind)
        if contract is None or capability.id not in contract["capabilities"]:
            raise RuntimeError(
                f"analysis step {step.id} uses unsupported capability {capability.id}"
            )
        expected = set(step.expected_evidence)
        if expected and not expected <= set(capability.output_evidence):
            raise RuntimeError(
                f"analysis step {step.id} expects evidence not produced by {capability.id}"
            )
        if capability.side_effecting:
            raise RuntimeError(
                f"analysis step {step.id} cannot execute side-effecting capability"
            )
        context_dependencies = await self._analysis_context_dependencies(
            operation.id,
            step.context_evidence_refs,
        )
        dependencies = (
            *tuple(
                _dependency_for_evidence(item)
                for item in dependency_evidence
                if item.id and item.accepted and item.operation_id == operation.id
            ),
            *context_dependencies,
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                **dict(step.input),
                "dependency_evidence_refs": [
                    evidence_ref(item)
                    for item in dependency_evidence
                    if item.id and item.accepted
                ],
            },
            metadata=step_metadata,
            dependencies=dependencies,
            sequence=4000 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        return tuple(
            item
            for item in evidence
            if item.accepted and (not expected or item.kind in expected)
        )

    async def _analysis_context_dependencies(
        self,
        operation_id: str,
        refs: tuple[dict[str, Any], ...],
    ) -> tuple[TaskDependency, ...]:
        dependencies: list[TaskDependency] = []
        evidence = await self.store.list_evidence(operation_id)
        by_id = {item.id: item for item in evidence if item.id}
        for ref in refs:
            evidence_id = ref.get("id")
            if not evidence_id:
                continue
            item = by_id.get(str(evidence_id))
            if item is None or not item.accepted or item.operation_id != operation_id:
                raise RuntimeError(
                    f"analysis context evidence not accepted: {evidence_id}"
                )
            fingerprint = ref.get("payload_fingerprint")
            actual = item.metadata.get("payload_fingerprint") or _payload_fingerprint(
                item.payload
            )
            if fingerprint is not None and str(fingerprint) != actual:
                raise RuntimeError(
                    f"analysis context evidence fingerprint mismatch: {evidence_id}"
                )
            dependencies.append(_dependency_for_evidence(item))
        return tuple(dependencies)

    def _analysis_task(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input: dict[str, Any],
        metadata: dict[str, Any],
        dependencies: tuple[TaskDependency, ...],
        sequence: int,
    ) -> Task:
        return self.materialize_analysis_task(
            operation,
            capability,
            input=input,
            metadata=metadata,
            dependencies=dependencies,
            sequence=sequence,
        )

    def _accepted_analysis_step_evidence_map(
        self,
        evidence: tuple[Evidence, ...],
        *,
        analysis_id: str,
    ) -> dict[str, tuple[Evidence, ...]]:
        grouped: dict[str, list[Evidence]] = {}
        for item in evidence:
            if not item.accepted:
                continue
            if item.metadata.get("analysis_id") != analysis_id:
                continue
            step_id = item.metadata.get("analysis_step_id")
            if not isinstance(step_id, str) or not step_id:
                continue
            if item.kind in {
                "analysis.plan",
                "analysis.plan.validation",
                "planning.context",
            }:
                continue
            grouped.setdefault(step_id, []).append(item)
        return {key: tuple(value) for key, value in grouped.items()}

    @staticmethod
    def _analysis_steps_in_order(plan: DbAnalysisPlan) -> tuple[Any, ...]:
        remaining = {step.id: step for step in plan.steps}
        ordered = []
        while remaining:
            ready = [
                step
                for step in remaining.values()
                if all(dependency not in remaining for dependency in step.depends_on)
            ]
            if not ready:
                return tuple((*ordered, *remaining.values()))
            for step in ready:
                ordered.append(step)
                remaining.pop(step.id, None)
        return tuple(ordered)

    def _select_ready_analysis_steps(
        self,
        plan: DbAnalysisPlan,
        completed_by_step: dict[str, tuple[Evidence, ...]],
        *,
        serial: bool = True,
    ) -> tuple[Any, ...]:
        completed_step_ids = set(completed_by_step)
        ready = tuple(
            step
            for step in self._analysis_steps_in_order(plan)
            if step.id not in completed_step_ids
            and self._analysis_step_dependencies_satisfied(step, completed_by_step)
        )
        if serial:
            return ready[:1]
        return ready

    @staticmethod
    def _analysis_step_dependencies_satisfied(
        step: Any,
        completed_by_step: dict[str, tuple[Evidence, ...]],
    ) -> bool:
        for dependency_id in step.depends_on:
            dependency_evidence = tuple(
                evidence
                for evidence in completed_by_step.get(dependency_id, ())
                if evidence.accepted
                and evidence.id
                and evidence.metadata.get("analysis_step_id") == dependency_id
            )
            if not dependency_evidence:
                return False
        return True


def _analysis_replan_enabled(metadata: dict[str, Any]) -> bool:
    return bool(_from_db_options(metadata).get("analysis_replan_enabled"))


def _analysis_parallel_enabled(metadata: dict[str, Any]) -> bool:
    return bool(_from_db_options(metadata).get("analysis_parallel_enabled"))


def _analysis_max_concurrency(metadata: dict[str, Any]) -> int:
    value = _from_db_options(metadata).get("analysis_max_concurrency", 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _first_capability(capabilities: Any, capability_id: str) -> Capability | None:
    for capability in capabilities:
        if capability.id == capability_id:
            return capability
    return None


def _selected_capability_owner(
    contract: DbOperationContract,
    capability_id: str,
) -> str | None:
    for item in contract.metadata.get("selected_capabilities", ()):
        if item.get("id") == capability_id:
            return str(item.get("owner") or "")
    return None


def _with_schema_scope(evidence: Evidence, scope: str) -> Evidence:
    payload = dict(evidence.payload)
    payload_metadata = dict(payload.get("metadata") or {})
    payload["metadata"] = {**payload_metadata, "scope": scope}
    return dataclass_replace(
        evidence,
        payload=payload,
        metadata={**evidence.metadata, "scope": scope},
    )
