"""Model-planned, runtime-governed DB agent loop."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.runtime import (
    Evidence,
    Operation,
    RuntimeKernelGovernanceBlocked,
)

from ..continuation import DbContinuationResolver
from ..evidence import evidence_in_task_plan_order
from ..fingerprints import (
    db_operation_contract_binding_fingerprint,
    persisted_fingerprint,
)
from ..models import DbIntentKind
from ..planner_protocol import (
    DbAgentPlanner,
    DbLoopState,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
    DbPlannerObservation,
)
from ..verification import db_run_finalization_check

from .actions import _has_runtime_continuation_block, _has_terminal_compilation_error
from .compiler import DbActionCompiler
from .contracts import (
    _contract_from_latest_snapshot,
    _contract_snapshot,
    _fallback_contract_for_operation,
    _fallback_intent_for_operation,
    _governance_summary,
    _intent_from_contract,
    _intent_summary,
)
from .grounding import _validation_grounding_runtime_continuation_action
from .execution import DbLoopTaskBatchExecutor
from .memory import (
    _memory_context_for_state,
    _memory_update_runtime_continuation_action,
    _required_memory_recall_runtime_continuation_action,
    _state_is_memory_update_operation,
)
from .progress import (
    _LoopProgressDecision,
    _LoopProgressGuard,
    _LoopProgressSnapshot,
    _blocked_resource_execution_errors,
    _compiled_action_fingerprints,
    _execution_error_fingerprint,
    _is_sql_execution_error,
    _new_evidence_refs,
)
from .summaries import (
    _approval_task_id,
    _capability_summary,
    _evidence_ref,
    _evidence_summary,
    _task_ref,
    _task_summary,
)
from .types import DbActionCompilation, DbLoopResult
from .utils import (
    _json_dict,
    _optional_string,
)


class DbAgentLoop:
    """Serial first implementation of the planner-driven DB runtime loop."""

    def __init__(self, runtime: Any, planner: DbAgentPlanner) -> None:
        self.runtime = runtime
        self.planner = planner
        self.continuation_resolver = DbContinuationResolver()
        self.compiler = DbActionCompiler(self.runtime, self.continuation_resolver)
        self.task_batch = DbLoopTaskBatchExecutor(self.runtime)

    async def run(
        self,
        operation: Operation,
        *,
        safety_frame: Mapping[str, Any] | None = None,
        max_turns: int | None = None,
    ) -> DbLoopResult:
        """Run the injected DB planner until a terminal loop result is reached."""
        if not getattr(self.runtime, "is_setup", False):
            await self.runtime.setup()
        turn_budget = max_turns or int(
            getattr(self.runtime.config.limits, "max_tasks", 20)
        )
        finalizable, finalization = await self._operation_finalizable(operation.id)
        if finalizable:
            return await self._result(
                operation,
                "finished",
                warnings=(),
                diagnostics={
                    "pre_planner_finalization": True,
                    "finalization": finalization,
                },
            )
        warnings: list[str] = []
        last_compilation: DbActionCompilation | None = None
        progress_guard = _LoopProgressGuard()
        for turn in range(1, turn_budget + 1):
            operation = await self._fresh_operation(operation.id)
            progress_before = await self._progress_snapshot(operation.id)
            state = await self.build_loop_state(
                operation,
                safety_frame=safety_frame,
                turn=turn,
                remaining_turns=turn_budget - turn + 1,
            )
            runtime_continuation = _required_memory_recall_runtime_continuation_action(
                state,
                current_action_ids=set(),
            )
            if runtime_continuation is None:
                runtime_continuation = _memory_update_runtime_continuation_action(
                    state,
                    current_action_ids=set(),
                )
            if runtime_continuation is None:
                runtime_continuation = (
                    _validation_grounding_runtime_continuation_action(
                        state,
                        current_action_ids=set(),
                    )
                )
            if runtime_continuation is None:
                decision = await self.planner.plan(state)
            elif str(
                runtime_continuation.metadata.get("continuation") or ""
            ).startswith(("validation_grounding.", "memory.recall.")):
                decision = DbPlannerDecision(
                    status=DbPlannerDecisionStatus.CONTINUE,
                    intent={"operation_type": DbIntentKind.DATA_QUERY.value},
                    actions=(runtime_continuation,),
                    rationale="Runtime continuation for deterministic DB work.",
                    metadata={
                        "runtime_continuation": True,
                        "continuation": runtime_continuation.metadata.get(
                            "continuation"
                        ),
                    },
                )
            else:
                decision = DbPlannerDecision(
                    status=DbPlannerDecisionStatus.CONTINUE,
                    intent={"operation_type": DbIntentKind.MEMORY_UPDATE.value},
                    actions=(runtime_continuation,),
                    rationale=(
                        "Runtime continuation for an accepted DB memory proposal."
                    ),
                    metadata={
                        "runtime_continuation": True,
                        "continuation": "memory.update.commit",
                    },
                )
            await self._persist_planner_decision(operation, decision, turn=turn)

            finish_finalization: tuple[bool, dict[str, Any]] | None = None
            if decision.status is DbPlannerDecisionStatus.FINISH:
                finish_finalization = await self._operation_finalizable(operation.id)

            if (
                decision.status is DbPlannerDecisionStatus.CLARIFY
                and not decision.actions
            ):
                observation = await self._persist_observation(
                    operation,
                    DbPlannerObservation(
                        diagnostics={
                            "status": "clarification_required",
                            "turn": turn,
                        }
                    ),
                    turn=turn,
                )
                return await self._result(
                    operation,
                    "clarification_required",
                    warnings=warnings,
                    diagnostics={
                        "clarification_question": decision.clarification_question,
                        "observation": observation.to_dict(),
                    },
                )
            if decision.status in {
                DbPlannerDecisionStatus.BLOCKED,
                DbPlannerDecisionStatus.FAILED,
            }:
                status = (
                    "configuration_required"
                    if decision.metadata.get("configuration_required")
                    else decision.status.value
                )
                observation = await self._persist_observation(
                    operation,
                    DbPlannerObservation(
                        diagnostics={
                            "status": status,
                            "turn": turn,
                            "planner_metadata": decision.metadata,
                        }
                    ),
                    turn=turn,
                )
                return await self._result(
                    operation,
                    status,
                    warnings=warnings,
                    diagnostics={"observation": observation.to_dict()},
                )
            if (
                decision.status is DbPlannerDecisionStatus.FINISH
                and not decision.actions
            ):
                assert finish_finalization is not None
                finalizable, finalization = finish_finalization
                verification = finalization.get("verification")
                verification = verification if isinstance(verification, Mapping) else {}
                finalization_fingerprint = persisted_fingerprint(
                    {
                        "query_result_required": finalization.get(
                            "query_result_required"
                        ),
                        "query_result_present": finalization.get(
                            "query_result_present"
                        ),
                        "missing_evidence": verification.get("missing_evidence") or (),
                        "warnings": verification.get("warnings") or (),
                        "contract": finalization.get("contract") or {},
                    }
                )
                observation = await self._persist_observation(
                    operation,
                    DbPlannerObservation(
                        diagnostics={
                            "status": (
                                "finish_requested"
                                if finalizable
                                else "finish_requested_not_finalizable"
                            ),
                            "turn": turn,
                            "finalization": finalization,
                            "finalization_fingerprint": finalization_fingerprint,
                        }
                    ),
                    turn=turn,
                )
                if not finalizable and _state_is_memory_update_operation(state):
                    verification_warnings = (
                        verification.get("warnings") if verification else ()
                    )
                    return await self._result(
                        operation,
                        "blocked",
                        warnings=tuple(
                            str(item) for item in (verification_warnings or ())
                        ),
                        diagnostics={
                            "observation": observation.to_dict(),
                            "finalization": finalization,
                        },
                    )
                if finalizable:
                    return await self._result(
                        operation,
                        "finished",
                        warnings=warnings,
                        diagnostics={"observation": observation.to_dict()},
                    )
                premature_finish_count = 1 + sum(
                    1
                    for prior in state.planner_observations
                    if prior.diagnostics.get("status")
                    == "finish_requested_not_finalizable"
                    and prior.diagnostics.get("finalization_fingerprint")
                    == finalization_fingerprint
                )
                if premature_finish_count > 1:
                    warning = "db_agent_loop_unmet_finalization"
                    warnings.append(warning)
                    return await self._result(
                        operation,
                        "failed",
                        warnings=warnings,
                        diagnostics={
                            "error": warning,
                            "observation": observation.to_dict(),
                            "finalization": finalization,
                        },
                    )
                missing_requirements = (
                    verification.get("warnings") if verification else ()
                )
                warnings.extend(
                    str(item)
                    for item in (missing_requirements or ())
                    if str(item).strip()
                )
                if not warnings:
                    warnings.append("finish_requested_not_finalizable")
                continue

            compilation = self.compile_actions(decision, state)
            last_compilation = compilation
            await self._persist_compilation(operation, compilation, decision, turn=turn)
            operation = await self._persist_compiled_contract(operation, compilation)
            if compilation.rejected_action_summaries and not compilation.task_specs:
                progress_after = await self._progress_snapshot(operation.id)
                progress_facts = self._progress_facts(
                    before=progress_before,
                    after=progress_after,
                    decision=decision,
                    compilation=compilation,
                    execution_errors=compilation.rejected_action_summaries,
                )
                progress_decision = progress_guard.evaluate(progress_facts)
                terminal_warning = (
                    progress_decision.warnings[0]
                    if progress_decision.warnings
                    else None
                )
                observation = await self._persist_observation(
                    operation,
                    self._observation_for_compilation(
                        compilation,
                        turn=turn,
                        progress_facts=progress_facts,
                        progress_decision=progress_decision,
                        terminal_warning=terminal_warning,
                    ),
                    turn=turn,
                )
                warnings.extend(
                    str(item.get("error") or "planner_action_rejected")
                    for item in compilation.rejected_action_summaries
                )
                if _has_runtime_continuation_block(compilation):
                    return await self._result(
                        operation,
                        "blocked",
                        warnings=warnings,
                        diagnostics={
                            "compilation": compilation.to_dict(),
                            "progress": progress_facts,
                            "observation": observation.to_dict(),
                        },
                    )
                if _has_terminal_compilation_error(compilation):
                    return await self._result(
                        operation,
                        "failed",
                        warnings=warnings,
                        diagnostics={
                            "compilation": compilation.to_dict(),
                            "progress": progress_facts,
                            "observation": observation.to_dict(),
                        },
                    )
                if progress_decision.terminal_status is not None:
                    warnings.extend(progress_decision.warnings)
                    return await self._result(
                        operation,
                        progress_decision.terminal_status,
                        warnings=warnings,
                        diagnostics={
                            "compilation": compilation.to_dict(),
                            "progress": progress_facts,
                            "progress_guard": progress_decision.diagnostics,
                            "observation": observation.to_dict(),
                        },
                    )
                continue

            try:
                await self.runtime.kernel.evaluate_operation_governance(operation.id)
            except RuntimeKernelGovernanceBlocked as exc:
                governance = exc.governance
                await self._persist_observation(
                    operation,
                    DbPlannerObservation(
                        governance_status=_governance_summary(governance),
                        diagnostics={
                            "status": "governance_blocked",
                            "turn": turn,
                            "error": str(exc),
                        },
                    ),
                    turn=turn,
                )
                return await self._result(
                    operation,
                    "blocked",
                    warnings=warnings,
                    diagnostics={"governance": _governance_summary(governance)},
                )

            task_plan = await self.runtime.plan_task_specs(
                operation,
                compilation.task_specs,
                contract=compilation.compiled_contract_snapshot,
            )
            executed: list[Evidence] = []
            execution_errors: list[dict[str, Any]] = []
            outcomes = await self.task_batch.execute(task_plan.tasks, operation)
            for outcome in outcomes:
                if outcome.governance_error is not None:
                    governance_error = outcome.governance_error
                    await self._persist_observation(
                        operation,
                        DbPlannerObservation(
                            task_statuses=await self._task_summaries(operation.id),
                            governance_status=_governance_summary(
                                governance_error.governance
                            ),
                            diagnostics={"status": "task_governance_blocked"},
                        ),
                        turn=turn,
                    )
                    return await self._result(
                        operation,
                        "blocked",
                        warnings=warnings,
                        diagnostics={
                            "task_id": (
                                governance_error.task.id
                                if governance_error.task
                                else None
                            )
                        },
                    )
                if outcome.readiness_error is not None:
                    readiness_error = outcome.readiness_error
                    execution_errors.append(
                        {
                            "task_id": readiness_error.task.id,
                            "capability_id": readiness_error.task.capability_id,
                            "error": str(readiness_error),
                            "readiness": readiness_error.readiness,
                        }
                    )
                    continue
                if outcome.error is not None:
                    task_error = outcome.error
                    execution_errors.append(
                        {
                            "task_id": outcome.task.id,
                            "capability_id": outcome.task.capability_id,
                            "error": str(task_error),
                            "error_type": type(task_error).__name__,
                        }
                    )
                    continue
                executed.extend(outcome.evidence)

            progress_after = await self._progress_snapshot(operation.id)
            progress_facts = self._progress_facts(
                before=progress_before,
                after=progress_after,
                decision=decision,
                compilation=compilation,
                task_plan=task_plan.to_dict(),
                execution_errors=tuple(execution_errors),
            )
            progress_decision = progress_guard.evaluate(progress_facts)
            terminal_warning = (
                progress_decision.warnings[0] if progress_decision.warnings else None
            )
            observation = await self._persist_observation(
                operation,
                await self._observation_after_execution(
                    operation.id,
                    executed=tuple(executed),
                    execution_errors=tuple(execution_errors),
                    turn=turn,
                    progress_facts=progress_facts,
                    progress_decision=progress_decision,
                    terminal_warning=terminal_warning,
                ),
                turn=turn,
            )
            blocked_resource_errors = _blocked_resource_execution_errors(
                execution_errors
            )
            if blocked_resource_errors:
                warnings.append("db_agent_loop_blocked_resource_validation")
                return await self._result(
                    operation,
                    "blocked",
                    warnings=warnings,
                    diagnostics={
                        "task_plan": task_plan.to_dict(),
                        "progress": progress_facts,
                        "blocked_resource_errors": blocked_resource_errors,
                        "observation": observation.to_dict(),
                    },
                )
            if progress_decision.terminal_status is not None:
                warnings.extend(progress_decision.warnings)
                return await self._result(
                    operation,
                    progress_decision.terminal_status,
                    warnings=warnings,
                    diagnostics={
                        "task_plan": task_plan.to_dict(),
                        "progress": progress_facts,
                        "progress_guard": progress_decision.diagnostics,
                        "observation": observation.to_dict(),
                    },
                )
            if execution_errors:
                warnings.extend(str(item.get("error")) for item in execution_errors)
                continue
            finalizable, finalization = await self._operation_finalizable(operation.id)
            diagnostics = {
                "task_plan": task_plan.to_dict(),
                "observation": observation.to_dict(),
                "finalization": finalization,
            }
            if finalizable:
                return await self._result(
                    operation,
                    "finished",
                    warnings=warnings,
                    diagnostics=diagnostics,
                )
            continue
        budget_diagnostics: dict[str, Any] = {"turn_budget": turn_budget}
        if last_compilation is not None:
            budget_diagnostics["last_compilation"] = last_compilation.to_dict()
        return await self._result(
            operation,
            "budget_exhausted",
            warnings=warnings,
            diagnostics=budget_diagnostics,
        )

    async def build_loop_state(
        self,
        operation: Operation,
        *,
        safety_frame: Mapping[str, Any] | None = None,
        turn: int = 1,
        remaining_turns: int = 1,
    ) -> DbLoopState:
        """Build planner input from persisted operation, task, and evidence state."""
        operation = await self._fresh_operation(operation.id)
        metadata = dict(operation.metadata)
        frame = _json_dict(safety_frame or metadata.get("safety_frame") or {})
        tasks = await self.runtime.store.list_tasks(operation.id)
        evidence = evidence_in_task_plan_order(
            await self.runtime.store.list_evidence(operation.id),
            tasks,
        )
        accepted = tuple(item for item in evidence if item.accepted)
        rejected = tuple(item for item in evidence if not item.accepted)
        planner_observations = tuple(
            DbPlannerObservation.from_dict(item.payload["observation"])
            for item in evidence
            if item.kind == "planner.observation"
            and isinstance(item.payload.get("observation"), dict)
        )
        return DbLoopState(
            operation_id=operation.id,
            normalized_user_request={
                **operation.request,
                "operation_type": operation.operation_type,
            },
            source_scope=tuple(
                str(item)
                for item in (
                    operation.request.get("source_scope")
                    or metadata.get("source_scope")
                    or ()
                )
            ),
            explicit_mode=_optional_string(
                operation.request.get("mode") or metadata.get("mode")
            ),
            explicit_requested_capabilities=tuple(
                str(item)
                for item in (
                    operation.request.get("requested_capabilities")
                    or metadata.get("requested_capabilities")
                    or ()
                )
            ),
            safety_frame=frame,
            latest_compiled_contract_snapshot=metadata.get(
                "latest_compiled_contract_snapshot"
            ),
            available_action_kinds=tuple(DbPlannerActionKind),
            capability_summaries=tuple(
                _capability_summary(capability)
                for capability in self.runtime.registry.capabilities
            ),
            memory_context=_memory_context_for_state(
                self.runtime,
                operation,
                accepted,
                safety_frame=frame,
            ),
            task_summaries=await self._task_summaries(operation.id),
            accepted_evidence_summaries=tuple(
                _evidence_summary(item) for item in accepted
            ),
            rejected_evidence_summaries=tuple(
                _evidence_summary(item) for item in rejected
            ),
            approval_state=await self._approval_state(operation.id),
            validation_summaries=tuple(
                _evidence_summary(item)
                for item in evidence
                if item.kind in {"sql.validation", "query.plan.validation"}
            ),
            execution_error_summaries=tuple(
                _evidence_summary(item)
                for item in rejected
                if item.kind not in {"planner.compilation", "planner.observation"}
            ),
            planner_observations=planner_observations,
            runtime_limits=self.runtime.config.limits.to_dict(),
            remaining_budget={"planner_turns": remaining_turns},
            diagnostics={
                "turn": turn,
                "contract_fingerprint": (
                    db_operation_contract_binding_fingerprint(operation)
                ),
                "session_context_fingerprint": (
                    persisted_fingerprint(operation.request["session_context"])
                    if isinstance(operation.request.get("session_context"), Mapping)
                    else None
                ),
            },
        )

    def compile_actions(
        self,
        decision: DbPlannerDecision,
        state: DbLoopState,
    ) -> DbActionCompilation:
        """Validate planner actions and compile accepted ones to task specs."""
        return self.compiler.compile_actions(decision, state)

    async def _persist_compiled_contract(
        self,
        operation: Operation,
        compilation: DbActionCompilation,
    ) -> Operation:
        if not compilation.accepted_action_summaries and not compilation.task_specs:
            return operation
        metadata = {
            **operation.metadata,
            "latest_compiled_contract_snapshot": compilation.compiled_contract_snapshot,
            "resume_context": {
                **dict(operation.metadata.get("resume_context") or {}),
                "contract": compilation.compiled_contract_snapshot,
                "latest_compiled_contract_snapshot": (
                    compilation.compiled_contract_snapshot
                ),
            },
        }
        updated = replace(operation, metadata=metadata)
        await self.runtime.store.save_operation(updated)
        return updated

    async def _persist_planner_decision(
        self,
        operation: Operation,
        decision: DbPlannerDecision,
        *,
        turn: int,
    ) -> Evidence:
        evidence = Evidence(
            id=f"planner-decision-{operation.id}-{turn}-{uuid4()}",
            kind="planner.decision",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload={
                "turn": turn,
                "decision": decision.to_dict(),
                "fingerprint": persisted_fingerprint(decision.to_dict()),
            },
            metadata={"planner_loop": True},
        )
        await self.runtime.store.save_evidence(evidence)
        return evidence

    async def _persist_compilation(
        self,
        operation: Operation,
        compilation: DbActionCompilation,
        decision: DbPlannerDecision,
        *,
        turn: int,
    ) -> Evidence:
        evidence = Evidence(
            id=f"planner-compilation-{operation.id}-{turn}-{uuid4()}",
            kind="planner.compilation",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=not compilation.rejected_action_summaries,
            payload={
                "turn": turn,
                "decision_fingerprint": persisted_fingerprint(decision.to_dict()),
                "compilation": compilation.to_dict(),
            },
            metadata={"planner_loop": True},
        )
        await self.runtime.store.save_evidence(evidence)
        return evidence

    async def _persist_observation(
        self,
        operation: Operation,
        observation: DbPlannerObservation,
        *,
        turn: int,
    ) -> DbPlannerObservation:
        evidence = Evidence(
            id=f"planner-observation-{operation.id}-{turn}-{uuid4()}",
            kind="planner.observation",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload={"turn": turn, "observation": observation.to_dict()},
            metadata={"planner_loop": True},
        )
        await self.runtime.store.save_evidence(evidence)
        return observation

    async def _progress_snapshot(self, operation_id: str) -> _LoopProgressSnapshot:
        tasks = await self.runtime.store.list_tasks(operation_id)
        evidence = evidence_in_task_plan_order(
            await self.runtime.store.list_evidence(operation_id),
            tasks,
        )
        return _LoopProgressSnapshot(
            task_statuses={task.id: task.status.value for task in tasks},
            accepted_evidence=tuple(
                _evidence_ref(item) for item in evidence if item.accepted
            ),
            rejected_evidence=tuple(
                _evidence_ref(item) for item in evidence if not item.accepted
            ),
        )

    def _progress_facts(
        self,
        *,
        before: _LoopProgressSnapshot,
        after: _LoopProgressSnapshot,
        decision: DbPlannerDecision,
        compilation: DbActionCompilation,
        task_plan: Mapping[str, Any] | None = None,
        execution_errors: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        before_task_ids = set(before.task_statuses)
        after_task_ids = set(after.task_statuses)
        new_task_ids = tuple(sorted(after_task_ids - before_task_ids))
        changed_task_statuses = tuple(
            {
                "task_id": task_id,
                "before": before.task_statuses.get(task_id),
                "after": after.task_statuses.get(task_id),
            }
            for task_id in sorted(before_task_ids & after_task_ids)
            if before.task_statuses.get(task_id) != after.task_statuses.get(task_id)
        )
        new_accepted = _new_evidence_refs(
            before.accepted_evidence,
            after.accepted_evidence,
            include_loop_control=False,
        )
        new_rejected = _new_evidence_refs(
            before.rejected_evidence,
            after.rejected_evidence,
            include_loop_control=False,
        )
        decision_fingerprint = persisted_fingerprint(decision.to_dict())
        compiled_action_fingerprints = _compiled_action_fingerprints(decision)
        task_spec_fingerprints = tuple(
            persisted_fingerprint(spec.to_dict()) for spec in compilation.task_specs
        )
        execution_error_fingerprints = tuple(
            _execution_error_fingerprint(error) for error in execution_errors
        )
        sql_error_fingerprints = tuple(
            fingerprint
            for fingerprint, error in zip(
                execution_error_fingerprints,
                execution_errors,
                strict=False,
            )
            if _is_sql_execution_error(error)
        )
        has_useful_error_observation = bool(execution_errors)
        no_progress = (
            not new_task_ids
            and not new_accepted
            and not changed_task_statuses
            and not new_rejected
            and not has_useful_error_observation
        )
        progress_fingerprint = persisted_fingerprint(
            {
                "decision_fingerprint": decision_fingerprint,
                "compiled_action_fingerprints": compiled_action_fingerprints,
                "task_spec_fingerprints": task_spec_fingerprints,
                "execution_error_fingerprints": execution_error_fingerprints,
                "new_task_ids": new_task_ids,
                "new_accepted_evidence_refs": new_accepted,
                "new_rejected_evidence_refs": new_rejected,
                "changed_task_statuses": changed_task_statuses,
            }
        )
        return {
            "decision_fingerprint": decision_fingerprint,
            "compiled_action_fingerprints": list(compiled_action_fingerprints),
            "task_spec_fingerprints": list(task_spec_fingerprints),
            "execution_error_fingerprints": list(execution_error_fingerprints),
            "sql_error_fingerprints": list(sql_error_fingerprints),
            "accepted_evidence_before": list(before.accepted_evidence),
            "accepted_evidence_after": list(after.accepted_evidence),
            "rejected_evidence_before": list(before.rejected_evidence),
            "rejected_evidence_after": list(after.rejected_evidence),
            "new_task_ids": list(new_task_ids),
            "new_accepted_evidence_refs": list(new_accepted),
            "new_rejected_evidence_refs": list(new_rejected),
            "changed_task_statuses": list(changed_task_statuses),
            "failed_action": bool(execution_errors),
            "no_progress": no_progress,
            "progress_fingerprint": progress_fingerprint,
            "task_plan_diagnostics": dict((task_plan or {}).get("diagnostics") or {}),
        }

    def _observation_for_compilation(
        self,
        compilation: DbActionCompilation,
        *,
        turn: int,
        progress_facts: Mapping[str, Any] | None = None,
        progress_decision: _LoopProgressDecision | None = None,
        terminal_warning: str | None = None,
    ) -> DbPlannerObservation:
        progress_decision = progress_decision or _LoopProgressDecision()
        diagnostics = {
            "status": terminal_warning or "compilation_rejected",
            "turn": turn,
            "compilation": compilation.to_dict(),
        }
        if progress_decision.diagnostics:
            diagnostics["progress_guard"] = progress_decision.diagnostics
        return DbPlannerObservation(
            execution_errors=tuple(
                {
                    "action_id": item.get("action_id"),
                    "kind": item.get("kind"),
                    "error": item.get("error"),
                }
                for item in compilation.rejected_action_summaries
            ),
            retry_facts=progress_decision.retry_facts,
            no_progress_facts=(
                (dict(progress_facts),) if progress_facts is not None else ()
            ),
            diagnostics=diagnostics,
        )

    async def _observation_after_execution(
        self,
        operation_id: str,
        *,
        executed: tuple[Evidence, ...],
        execution_errors: tuple[dict[str, Any], ...],
        turn: int,
        progress_facts: Mapping[str, Any] | None = None,
        progress_decision: _LoopProgressDecision | None = None,
        terminal_warning: str | None = None,
    ) -> DbPlannerObservation:
        progress_decision = progress_decision or _LoopProgressDecision()
        diagnostics = {"status": terminal_warning or "tasks_executed", "turn": turn}
        if progress_decision.diagnostics:
            diagnostics["progress_guard"] = progress_decision.diagnostics
        return DbPlannerObservation(
            accepted_evidence_summaries=tuple(
                _evidence_summary(item) for item in executed if item.accepted
            ),
            rejected_evidence_summaries=tuple(
                _evidence_summary(item) for item in executed if not item.accepted
            ),
            task_statuses=await self._task_summaries(operation_id),
            execution_errors=execution_errors,
            retry_facts=progress_decision.retry_facts,
            no_progress_facts=(
                (dict(progress_facts),) if progress_facts is not None else ()
            ),
            diagnostics=diagnostics,
        )

    async def _task_summaries(self, operation_id: str) -> tuple[dict[str, Any], ...]:
        tasks = await self.runtime.store.list_tasks(operation_id)
        return tuple(_task_summary(task) for task in tasks)

    async def _approval_state(self, operation_id: str) -> dict[str, Any]:
        approvals = await self.runtime.store.list_approval_requests(operation_id)
        return {
            "requests": [
                {
                    "approval_id": item.approval_id,
                    "status": item.status.value,
                    "policy_id": item.requested_by_policy_id,
                    "task_id": _approval_task_id(item),
                }
                for item in approvals
            ]
        }

    async def _operation_finalizable(
        self,
        operation_id: str,
    ) -> tuple[bool, dict[str, Any]]:
        operation = await self._fresh_operation(operation_id)
        fallback_contract = _fallback_contract_for_operation(
            operation,
            limits=self.runtime.config.limits,
        )
        contract = _contract_from_latest_snapshot(operation, fallback_contract)
        fallback_intent = _fallback_intent_for_operation(operation, contract)
        finalization_state = getattr(
            self.runtime,
            "_run_operation_finalization_state",
            None,
        )
        if finalization_state is not None:
            return await finalization_state(
                operation_id,
                fallback_intent=fallback_intent,
                fallback_contract=fallback_contract,
            )
        tasks = tuple(await self.runtime.store.list_tasks(operation_id))
        evidence = evidence_in_task_plan_order(
            await self.runtime.store.list_evidence(operation_id),
            tasks,
        )
        intent = _intent_from_contract(contract, fallback_intent)
        check = db_run_finalization_check(
            operation=operation,
            verifier=self.runtime.verifier,
            contract=contract,
            intent=intent,
            evidence=evidence,
            tasks=tasks,
        )
        return check.finalizable, {
            **check.to_dict(),
            "intent": _intent_summary(intent),
            "contract": _contract_snapshot(contract),
        }

    async def _fresh_operation(self, operation_id: str) -> Operation:
        loaded = await self.runtime.store.load_operation(operation_id)
        if loaded is None:
            raise KeyError(operation_id)
        return loaded

    async def _result(
        self,
        operation: Operation,
        status: str,
        *,
        warnings: Iterable[str],
        diagnostics: Mapping[str, Any],
    ) -> DbLoopResult:
        tasks = await self.runtime.store.list_tasks(operation.id)
        evidence = evidence_in_task_plan_order(
            await self.runtime.store.list_evidence(operation.id),
            tasks,
        )
        return DbLoopResult(
            status=status,
            evidence_refs=tuple(_evidence_ref(item) for item in evidence if item.id),
            task_refs=tuple(_task_ref(item) for item in tasks),
            warnings=tuple(warnings),
            diagnostics=dict(diagnostics),
        )
