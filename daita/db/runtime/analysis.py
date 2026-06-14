"""Multi-step analysis workflow helpers for ``DbRuntime``."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any
from uuid import uuid4

from daita.runtime import (
    Capability,
    Evidence,
    GovernanceResult,
    Operation,
    OperationSnapshot,
    OperationStatus,
    RuntimeEventType,
    Task,
    TaskDependency,
    TaskStatus,
)

from ..analysis import (
    DbAnalysisPlan,
    analysis_metadata,
    capability_contract_for_step_kind,
    evidence_ref,
    stable_fingerprint,
    validate_analysis_plan_payload,
    with_analysis_evidence_trace,
)
from ..evidence import DbEvidenceStore
from ..execution import DbOperationExecutor
from ..models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from .cache import _from_db_options
from .resume import (
    _analysis_synthesis_is_partial,
    _latest_final_analysis_synthesis_from_snapshot,
)
from .types import (
    _AnalysisPlanState,
    DbRuntimeGovernanceBlocked,
    _governance_blocked_answer,
    _governance_blocked_warning,
)


class DbRuntimeAnalysisMixin:
    def _should_route_multi_step_analysis(
        self,
        request: DbRequest,
        intent: DbIntent,
    ) -> bool:
        if intent.kind not in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
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
        intent: DbIntent,
        contract: DbOperationContract,
        operation: Operation,
        *,
        base_diagnostics: dict[str, Any],
        reuse_existing_plan: bool = False,
    ) -> DbOperationResult:
        executor = DbOperationExecutor(self)
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
            schema_evidence = await executor._inspect_schema_if_available(
                operation, tasks, evidence_store
            )
            schema = schema_evidence.payload if schema_evidence is not None else {}
            planning_context = await executor._build_planning_context(
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
                        intent=intent,
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
                            intent=intent,
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
                            executor,
                            request,
                            intent,
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
                        executor,
                        request,
                        intent,
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
                            intent,
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
                    intent=intent,
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
                    intent=intent,
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
                    intent=intent,
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
        executor: DbOperationExecutor,
        request: DbRequest,
        intent: DbIntent,
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
        from dataclasses import replace as dataclass_replace

        before_ids = {item.id for item in await self.store.list_evidence(operation.id)}
        step_request = dataclass_replace(request, prompt=step_prompt)
        planning_context = await executor._build_planning_context(
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
        plan_evidence, strategy_warnings, _ = await executor._plan_query(
            step_request,
            intent,
            operation,
            schema,
            None,
            planning_context,
            tasks,
            evidence_store,
            analysis_metadata=step_metadata,
            extra_dependencies=context_dependencies,
        )
        validation = await executor._validate_query_plan(
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
            sql_validation = await executor._execute_sql_validation(
                contract,
                operation,
                tasks,
                evidence_store,
                sql,
                plan_validation=validation,
                analysis_metadata=step_metadata,
                extra_dependencies=context_dependencies,
            )
            await executor._execute_validated_read(
                contract,
                operation,
                tasks,
                evidence_store,
                sql_validation,
                analysis_metadata=step_metadata,
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
        executor: DbOperationExecutor,
        request: DbRequest,
        intent: DbIntent,
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
                    executor,
                    request,
                    intent,
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
                        intent,
                        produced,
                        tuple(step_tasks),
                        step_metadata=step_meta,
                    )
                )
                produced = (*produced, verification_evidence)
                return step, produced, tuple(step_tasks)

        results = await asyncio.gather(*(run_step(step) for step in steps))
        return tuple(sorted(results, key=lambda item: item[0].id))

    async def _persist_analysis_verification_result_evidence(
        self,
        operation: Operation,
        contract: DbOperationContract,
        intent: DbIntent,
        evidence: tuple[Evidence, ...],
        tasks: tuple[Task, ...],
        *,
        step_metadata: dict[str, Any],
    ) -> Evidence:
        verification = self.verifier.verify(contract, intent, evidence, tasks)
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
                or _payload_fingerprint(item.payload),
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
        task = self._analysis_task(
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
        task = self._analysis_task(
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
        task_input = {**input}
        input_hash = _stable_hash(task_input)
        return Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**task_input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            dependencies=dependencies,
            metadata=with_analysis_evidence_trace(
                {
                    "owner": capability.owner,
                    "reason": "analysis",
                    "sequence": sequence,
                    "input_hash": input_hash,
                    "idempotency_key": _stable_hash(
                        {
                            "operation_id": operation.id,
                            "capability_id": capability.id,
                            "input": task_input,
                            **metadata,
                        }
                    ),
                    "idempotent": capability.idempotent,
                    "replay_safe": capability.replay_safe,
                    "side_effecting": capability.side_effecting,
                    **metadata,
                }
            ),
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

    def _analysis_budget_failure(
        self,
        plan: DbAnalysisPlan,
        evidence: tuple[Evidence, ...],
        *,
        started_at: float,
    ) -> dict[str, Any] | None:
        usage = _analysis_budget_usage(evidence, started_at=started_at)
        failures = []
        budgets = plan.budgets
        if usage["total_rows"] > budgets.max_total_rows:
            failures.append("budget_max_total_rows_exceeded")
        if usage["llm_calls"] > budgets.max_llm_calls:
            failures.append("budget_max_llm_calls_exceeded")
        if usage["context_chars"] > budgets.max_context_chars:
            failures.append("budget_max_context_chars_exceeded")
        if usage["duration_seconds"] > budgets.max_duration_seconds:
            failures.append("budget_max_duration_seconds_exceeded")
        if not failures:
            return None
        return {
            "budget_exceeded": True,
            "failures": failures,
            "budget_usage": usage,
            "budgets": budgets.to_dict(),
        }

    @staticmethod
    def _analysis_step_budget_failure(
        step: Any,
        evidence: tuple[Evidence, ...],
    ) -> dict[str, Any] | None:
        rows = sum(
            _analysis_query_result_rows(item)
            for item in evidence
            if item.kind == "query.result"
        )
        if rows <= step.budgets.max_rows:
            return None
        return {
            "budget_exceeded": True,
            "failures": ["step_max_rows_exceeded"],
            "budget_usage": {"step_rows": rows},
            "budgets": step.budgets.to_dict(),
        }

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
                    "status": approval.status.value,
                    "task_id": approval.task_id,
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


def _checkpoint_reason(diagnostics: dict[str, Any]) -> str:
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


def _analysis_budget_usage(
    evidence: tuple[Evidence, ...],
    *,
    started_at: float,
) -> dict[str, Any]:
    return {
        "total_rows": sum(
            _analysis_query_result_rows(item)
            for item in evidence
            if item.kind == "query.result" and item.accepted
        ),
        "llm_calls": sum(1 for item in evidence if _analysis_evidence_used_llm(item)),
        "context_chars": sum(
            len(str(item.payload.get("rendered_context") or ""))
            for item in evidence
            if item.kind == "planning.context" and item.accepted
        ),
        "duration_seconds": time.monotonic() - started_at,
    }


def _analysis_query_result_rows(evidence: Evidence) -> int:
    rows = evidence.payload.get("rows")
    if isinstance(rows, list):
        return len(rows)
    try:
        return int(evidence.payload.get("total_rows") or 0)
    except (TypeError, ValueError):
        return 0


def _analysis_evidence_used_llm(evidence: Evidence) -> bool:
    if not evidence.accepted:
        return False
    diagnostics = evidence.payload.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("mode") == "llm":
        return True
    planner_diagnostics = evidence.payload.get("planner_diagnostics")
    return isinstance(planner_diagnostics, dict) and bool(
        planner_diagnostics.get("model") or planner_diagnostics.get("provider")
    )


def _answer_from_analysis_synthesis_evidence(evidence: Evidence) -> str:
    answer = evidence.payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("accepted analysis.synthesis evidence is missing answer")
    return answer


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


def _payload_contains(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if payload.get(key) != value:
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

