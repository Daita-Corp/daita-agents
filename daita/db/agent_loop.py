"""Model-planned, runtime-governed DB agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.runtime import (
    AccessMode,
    Evidence,
    GovernanceResult,
    Operation,
    RuntimeKernelGovernanceBlocked,
    Task,
    TaskDependency,
    TaskStatus,
)

from .continuation import DbContinuationResolver
from .models import DbIntent, DbIntentKind, DbLimits, DbOperationContract
from .memory import (
    db_memory_options_from_runtime_metadata,
    db_memory_planning_recall_decision,
)
from .planning_context import planner_eligible_column_value_hint
from .planner_protocol import (
    DbAgentPlanner,
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
    DbPlannerObservation,
)
from .runtime.tasks import DbTaskSpec
from .runtime.types import DbRuntimeGovernanceBlocked, DbRuntimeTaskNotRunnable
from .verification import (
    DB_FINALIZATION_CONTROL_EVIDENCE_KINDS,
    db_run_finalization_check,
)

_ACCESS_ORDER = {
    AccessMode.NONE.value: 0,
    AccessMode.METADATA_READ.value: 1,
    AccessMode.READ.value: 2,
    AccessMode.WRITE.value: 3,
    AccessMode.ADMIN.value: 4,
}

_SIMPLE_ACTION_CAPABILITIES: dict[DbPlannerActionKind, tuple[str, ...]] = {
    DbPlannerActionKind.INSPECT_SCHEMA: ("db.schema.inspect",),
    DbPlannerActionKind.REGISTER_CATALOG_SOURCE: ("catalog.source.register",),
    DbPlannerActionKind.SEARCH_SCHEMA: ("catalog.schema.search",),
    DbPlannerActionKind.INSPECT_ASSET: ("catalog.asset.inspect",),
    DbPlannerActionKind.FIND_RELATIONSHIP_PATHS: ("catalog.relationship_paths.find",),
    DbPlannerActionKind.SEARCH_COLUMN_VALUES: ("catalog.column_values.search",),
    DbPlannerActionKind.BUILD_PLANNING_CONTEXT: ("db.planning.context.build",),
    DbPlannerActionKind.PROPOSE_SQL_READ: ("db.query.plan",),
    DbPlannerActionKind.REPAIR_QUERY_PLAN: ("db.query.repair",),
    DbPlannerActionKind.RECALL_MEMORY: ("memory.semantic.recall",),
    DbPlannerActionKind.PLAN_MEMORY_UPDATE: ("db.memory.plan_update",),
    DbPlannerActionKind.COMMIT_MEMORY_UPDATE: ("db.memory.commit_update",),
    DbPlannerActionKind.PLAN_ANALYSIS: (
        "db.analysis.plan",
        "db.analysis.plan.validate",
    ),
    DbPlannerActionKind.EXECUTE_ANALYSIS_STEP: ("db.analysis.checkpoint",),
    DbPlannerActionKind.SUMMARIZE_ANALYSIS: ("db.analysis.summarize",),
    DbPlannerActionKind.SYNTHESIZE: ("db.answer.synthesize",),
}

_SQL_QUERY_ACTIONS = {
    DbPlannerActionKind.PROPOSE_SQL_READ,
    DbPlannerActionKind.REPAIR_QUERY_PLAN,
    DbPlannerActionKind.EXECUTE_VALIDATED_READ,
}

_MONITOR_ACTION_CAPABILITIES: dict[DbPlannerActionKind, str] = {
    DbPlannerActionKind.PLAN_MONITOR_CREATE: "db.monitor.plan_create",
    DbPlannerActionKind.COMMIT_MONITOR_CREATE: "db.monitor.commit_create",
    DbPlannerActionKind.PLAN_MONITOR_LIFECYCLE: "db.monitor.plan_lifecycle",
    DbPlannerActionKind.COMMIT_MONITOR_LIFECYCLE: "db.monitor.commit_lifecycle",
    DbPlannerActionKind.READ_MONITOR_STATE: "db.monitor.read",
    DbPlannerActionKind.RESOLVE_MONITOR_APPROVAL: "db.monitor.resolve_approval",
}

_TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.BLOCKED,
    TaskStatus.SKIPPED,
}

_CATALOG_COLUMN_VALUE_GROUNDING_REASON = "catalog_column_value_grounding"


@dataclass(frozen=True)
class DbActionCompilation:
    """Runtime compilation of planner actions to governed DB task specs."""

    accepted_action_summaries: tuple[dict[str, Any], ...] = ()
    rejected_action_summaries: tuple[dict[str, Any], ...] = ()
    task_specs: tuple[DbTaskSpec, ...] = ()
    compiled_contract_snapshot: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_action_summaries",
            tuple(_json_dict(item) for item in self.accepted_action_summaries),
        )
        object.__setattr__(
            self,
            "rejected_action_summaries",
            tuple(_json_dict(item) for item in self.rejected_action_summaries),
        )
        object.__setattr__(self, "task_specs", tuple(self.task_specs))
        object.__setattr__(
            self,
            "compiled_contract_snapshot",
            _json_dict(self.compiled_contract_snapshot),
        )
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_action_summaries": list(self.accepted_action_summaries),
            "rejected_action_summaries": list(self.rejected_action_summaries),
            "task_specs": [spec.to_dict() for spec in self.task_specs],
            "compiled_contract_snapshot": self.compiled_contract_snapshot,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class DbLoopResult:
    """Terminal result returned by the DB agent loop."""

    status: str
    evidence_refs: tuple[dict[str, Any], ...] = ()
    task_refs: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status:
            raise ValueError("status is required")
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(_json_dict(item) for item in self.evidence_refs),
        )
        object.__setattr__(
            self, "task_refs", tuple(_json_dict(item) for item in self.task_refs)
        )
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "task_refs": list(self.task_refs),
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class _LoopProgressSnapshot:
    task_statuses: dict[str, str]
    accepted_evidence: tuple[dict[str, Any], ...] = ()
    rejected_evidence: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _LoopProgressDecision:
    terminal_status: str | None = None
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    retry_facts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _ResolvedSqlInput:
    sql: str
    provenance: str
    query_plan_dependency: TaskDependency | None = None
    source_evidence_id: str | None = None
    source_evidence_kind: str | None = None
    source_evidence_owner: str | None = None
    source_task_id: str | None = None
    source_payload_fingerprint: str | None = None


@dataclass
class _LoopProgressGuard:
    seen_no_progress_fingerprints: set[str] = field(default_factory=set)
    failed_action_counts: dict[str, int] = field(default_factory=dict)
    sql_error_counts: dict[str, int] = field(default_factory=dict)
    no_progress_count: int = 0

    def evaluate(self, facts: Mapping[str, Any]) -> _LoopProgressDecision:
        retry_facts: list[dict[str, Any]] = []
        sql_terminal = False
        for fingerprint in facts.get("sql_error_fingerprints") or ():
            previous = self.sql_error_counts.get(str(fingerprint), 0)
            count = previous + 1
            self.sql_error_counts[str(fingerprint)] = count
            if previous == 1:
                retry_facts.append(
                    {
                        "warning": "db_agent_loop_repeated_sql_failure",
                        "fingerprint": str(fingerprint),
                        "count": count,
                        "message": (
                            "The same SQL validation or execution failure "
                            "repeated; choose a different action or repair the SQL."
                        ),
                    }
                )
                sql_terminal = True
            elif previous >= 2:
                sql_terminal = True

        repeated_failed_action = False
        if facts.get("failed_action") and not facts.get("new_accepted_evidence_refs"):
            for fingerprint in facts.get("compiled_action_fingerprints") or ():
                previous = self.failed_action_counts.get(str(fingerprint), 0)
                self.failed_action_counts[str(fingerprint)] = previous + 1
                if previous >= 1:
                    repeated_failed_action = True

        repeated_no_progress = False
        progress_fingerprint = str(facts.get("progress_fingerprint") or "")
        if facts.get("no_progress"):
            self.no_progress_count += 1
            repeated_no_progress = (
                progress_fingerprint in self.seen_no_progress_fingerprints
                or self.no_progress_count >= 2
            )
            if progress_fingerprint:
                self.seen_no_progress_fingerprints.add(progress_fingerprint)
        else:
            self.no_progress_count = 0

        if retry_facts and not sql_terminal:
            return _LoopProgressDecision(retry_facts=tuple(retry_facts))

        warnings: list[str] = []
        terminal_status: str | None = None
        if sql_terminal:
            terminal_status = "failed"
            warnings.append("db_agent_loop_repeated_sql_failure")
        if repeated_failed_action:
            terminal_status = terminal_status or "failed"
            warnings.append("db_agent_loop_repeated_action")
        if repeated_no_progress:
            terminal_status = terminal_status or "blocked"
            warnings.append("db_agent_loop_no_progress")

        if terminal_status is None:
            return _LoopProgressDecision()

        return _LoopProgressDecision(
            terminal_status=terminal_status,
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics={
                "sql_terminal": sql_terminal,
                "repeated_failed_action": repeated_failed_action,
                "repeated_no_progress": repeated_no_progress,
                "no_progress_count": self.no_progress_count,
            },
            retry_facts=tuple(retry_facts),
        )


class DbAgentLoop:
    """Serial first implementation of the planner-driven DB runtime loop."""

    def __init__(self, runtime: Any, planner: DbAgentPlanner) -> None:
        self.runtime = runtime
        self.planner = planner
        self.continuation_resolver = DbContinuationResolver()

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
            elif (
                runtime_continuation.kind is DbPlannerActionKind.BUILD_PLANNING_CONTEXT
            ):
                decision = DbPlannerDecision(
                    status=DbPlannerDecisionStatus.CONTINUE,
                    intent={"operation_type": DbIntentKind.DATA_QUERY.value},
                    actions=(runtime_continuation,),
                    rationale=(
                        "Runtime continuation for validation-driven value grounding."
                    ),
                    metadata={
                        "runtime_continuation": True,
                        "continuation": "validation_grounding.context_refresh",
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

            if decision.status is DbPlannerDecisionStatus.CLARIFY:
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
                finalizable, finalization = await self._operation_finalizable(
                    operation.id
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
                        }
                    ),
                    turn=turn,
                )
                if not finalizable and _state_is_memory_update_operation(state):
                    verification = finalization.get("verification")
                    verification_warnings = (
                        verification.get("warnings")
                        if isinstance(verification, Mapping)
                        else ()
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
                return await self._result(
                    operation,
                    "finished",
                    warnings=warnings,
                    diagnostics={"observation": observation.to_dict()},
                )

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
            for task in task_plan.tasks:
                if task.status in _TERMINAL_TASK_STATUSES:
                    continue
                try:
                    executed.extend(await self.runtime.execute_task(task, operation))
                except DbRuntimeGovernanceBlocked as exc:
                    await self._persist_observation(
                        operation,
                        DbPlannerObservation(
                            task_statuses=await self._task_summaries(operation.id),
                            governance_status=_governance_summary(exc.governance),
                            diagnostics={"status": "task_governance_blocked"},
                        ),
                        turn=turn,
                    )
                    return await self._result(
                        operation,
                        "blocked",
                        warnings=warnings,
                        diagnostics={"task_id": exc.task.id if exc.task else None},
                    )
                except DbRuntimeTaskNotRunnable as exc:
                    execution_errors.append(
                        {
                            "task_id": exc.task.id,
                            "capability_id": exc.task.capability_id,
                            "error": str(exc),
                            "readiness": exc.readiness,
                        }
                    )
                except Exception as exc:
                    execution_errors.append(
                        {
                            "task_id": task.id,
                            "capability_id": task.capability_id,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )

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
        diagnostics: dict[str, Any] = {"turn_budget": turn_budget}
        if last_compilation is not None:
            diagnostics["last_compilation"] = last_compilation.to_dict()
        return await self._result(
            operation,
            "budget_exhausted",
            warnings=warnings,
            diagnostics=diagnostics,
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
        evidence = await self.runtime.store.list_evidence(operation.id)
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
            diagnostics={"turn": turn},
        )

    def compile_actions(
        self,
        decision: DbPlannerDecision,
        state: DbLoopState,
    ) -> DbActionCompilation:
        """Validate planner actions and compile accepted ones to task specs."""
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        specs: list[DbTaskSpec] = []
        selected_capabilities: list[dict[str, Any]] = []
        runtime_prerequisites: list[dict[str, Any]] = []
        runtime_prerequisite_capabilities: list[dict[str, Any]] = []
        decision_fingerprint = _stable_hash(decision.to_dict())
        actions: list[DbPlannerAction] = []

        for raw_action in decision.actions:
            action, error = _coerce_action(raw_action)
            if error is not None or action is None:
                rejected.append(error or {"error": "invalid_action"})
                continue
            actions.append(action)

        runtime_continuation = _validation_grounding_runtime_continuation_action(
            state,
            current_action_ids={action.action_id for action in actions},
        )
        if runtime_continuation is not None:
            if self.continuation_resolver.blocked_diagnostic(runtime_continuation):
                actions = [runtime_continuation]
            elif not any(
                action.kind is DbPlannerActionKind.BUILD_PLANNING_CONTEXT
                for action in actions
            ):
                actions.insert(0, runtime_continuation)

        if not any(
            action.kind is DbPlannerActionKind.COMMIT_MEMORY_UPDATE
            for action in actions
        ):
            runtime_continuation = _memory_update_runtime_continuation_action(
                state,
                current_action_ids={action.action_id for action in actions},
            )
            if runtime_continuation is not None:
                actions.insert(0, runtime_continuation)

        current_action_ids = {action.action_id for action in actions}
        resolved_actions: list[DbPlannerAction] = []
        for action in actions:
            continuation_error = self.continuation_resolver.blocked_diagnostic(action)
            if continuation_error is not None:
                rejected.append(_continuation_action_error(action, continuation_error))
                continue
            resolved_action = self.continuation_resolver.resolve(
                action,
                state,
                current_action_ids=current_action_ids,
            )
            continuation_error = self.continuation_resolver.blocked_diagnostic(
                resolved_action
            )
            if continuation_error is not None:
                rejected.append(
                    _continuation_action_error(resolved_action, continuation_error)
                )
                continue
            resolved_actions.append(resolved_action)
        actions = resolved_actions

        ordered_actions, dag_errors = _ordered_actions_or_errors(tuple(actions))
        if dag_errors:
            rejected.extend(dag_errors)

        prior_action_specs: dict[str, tuple[DbTaskSpec, ...]] = {}
        for action in ordered_actions:
            planner_dependencies, dependency_errors = self._planner_dependency_specs(
                action,
                prior_action_specs=prior_action_specs,
                operation_id=state.operation_id,
            )
            if dependency_errors:
                rejected.extend(dependency_errors)
                continue
            mode_errors = _mode_action_errors(action, state)
            if mode_errors:
                rejected.extend(mode_errors)
                continue
            action_specs, action_capabilities, action_errors = self._compile_one_action(
                action,
                state=state,
                sequence_start=len(specs) + 1,
                decision_fingerprint=decision_fingerprint,
            )
            if action_errors:
                rejected.extend(action_errors)
                continue
            if planner_dependencies:
                action_specs = _with_spec_dependencies(
                    action_specs, planner_dependencies
                )
            (
                prerequisite_capabilities,
                prerequisite_metadata,
                prerequisite_errors,
            ) = self._catalog_prerequisite_capabilities(action, state=state)
            if prerequisite_errors:
                rejected.extend(prerequisite_errors)
                continue
            specs.extend(action_specs)
            prior_action_specs[action.action_id] = tuple(action_specs)
            selected_capabilities.extend(action_capabilities)
            runtime_prerequisite_capabilities.extend(prerequisite_capabilities)
            runtime_prerequisites.extend(prerequisite_metadata)
            accepted.append(
                {
                    "action_id": action.action_id,
                    "kind": action.kind.value,
                    "capabilities": action_capabilities,
                }
            )

        contract = self._compiled_contract_snapshot(
            decision=decision,
            state=state,
            decision_fingerprint=decision_fingerprint,
            selected_capabilities=tuple(selected_capabilities),
            runtime_prerequisite_capabilities=tuple(runtime_prerequisite_capabilities),
            runtime_prerequisites=tuple(runtime_prerequisites),
            compiled_action_ids=tuple(item["action_id"] for item in accepted),
        )
        return DbActionCompilation(
            accepted_action_summaries=tuple(accepted),
            rejected_action_summaries=tuple(rejected),
            task_specs=tuple(specs),
            compiled_contract_snapshot=contract,
            diagnostics={
                "decision_fingerprint": decision_fingerprint,
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "task_spec_count": len(specs),
            },
        )

    def _planner_dependency_specs(
        self,
        action: DbPlannerAction,
        *,
        prior_action_specs: Mapping[str, tuple[DbTaskSpec, ...]],
        operation_id: str,
    ) -> tuple[tuple[TaskDependency, ...], tuple[dict[str, Any], ...]]:
        dependencies: list[TaskDependency] = []
        errors: list[dict[str, Any]] = []
        for dependency_action_id in dict.fromkeys(action.depends_on):
            producer_specs = prior_action_specs.get(dependency_action_id)
            if not producer_specs:
                errors.append(
                    _action_error(
                        action, f"dependency_not_durable:{dependency_action_id}"
                    )
                )
                continue
            dependency = self._dependency_for_producer_specs(
                action,
                dependency_action_id=dependency_action_id,
                producer_specs=producer_specs,
                operation_id=operation_id,
            )
            if dependency is None:
                errors.append(
                    _action_error(
                        action, f"dependency_not_durable:{dependency_action_id}"
                    )
                )
                continue
            dependencies.append(dependency)
        return tuple(dependencies), tuple(errors)

    def _dependency_for_producer_specs(
        self,
        action: DbPlannerAction,
        *,
        dependency_action_id: str,
        producer_specs: tuple[DbTaskSpec, ...],
        operation_id: str,
    ) -> TaskDependency | None:
        for spec in reversed(producer_specs):
            resolved = self._resolve_capability(spec.capability_id, owner=spec.owner)
            capability = resolved.get("capability")
            if capability is None:
                continue
            evidence_kind = next(iter(sorted(capability.output_evidence)), None)
            if evidence_kind is None:
                continue
            return TaskDependency(
                kind="evidence",
                evidence_kind=evidence_kind,
                evidence_owner=capability.owner,
                producer_task_id=spec.task_id,
                producer_capability_id=capability.id,
                producer_executor_id=capability.executor,
                evidence_accepted=True,
                input_hash=_stable_hash(spec.input),
                operation_id=operation_id,
                metadata={
                    "planner_dependency": True,
                    "producer_action_id": dependency_action_id,
                    "consumer_action_id": action.action_id,
                },
            )
        return None

    def _compile_one_action(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        if action.kind in {DbPlannerActionKind.CLARIFY, DbPlannerActionKind.FINISH}:
            return [], [], []
        if action.kind is DbPlannerActionKind.BUILD_PLANNING_CONTEXT:
            return self._compile_planning_context_action(
                action,
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        if action.kind is DbPlannerActionKind.PROPOSE_SQL_READ:
            return self._compile_sql_read_proposal_action(
                action,
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        if action.kind is DbPlannerActionKind.EXECUTE_VALIDATED_READ:
            return self._compile_validated_sql_action(
                action,
                sql_operation="query",
                execute_capability_id="db.sql.execute_read",
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        if action.kind is DbPlannerActionKind.PROPOSE_SQL_WRITE:
            return self._compile_sql_validation_only_action(
                action,
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        if action.kind is DbPlannerActionKind.EXECUTE_VALIDATED_WRITE:
            return self._compile_validated_sql_action(
                action,
                sql_operation="write",
                execute_capability_id="db.sql.execute_write",
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        if action.kind in _MONITOR_ACTION_CAPABILITIES:
            return self._compile_monitor_action(
                action,
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        if action.kind is DbPlannerActionKind.FIND_RELATIONSHIP_PATHS:
            return self._compile_relationship_paths_action(
                action,
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        if action.kind is DbPlannerActionKind.COMMIT_MEMORY_UPDATE:
            return self._compile_memory_commit_action(
                action,
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
            )
        capability_ids = _SIMPLE_ACTION_CAPABILITIES.get(action.kind)
        if capability_ids is None:
            return [], [], [_action_error(action, "unsupported_action_kind")]
        return self._compile_capability_specs(
            action,
            capability_ids=capability_ids,
            state=state,
            sequence_start=sequence_start,
            decision_fingerprint=decision_fingerprint,
        )

    def _compile_planning_context_action(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        owner = _action_owner(action)
        context_resolved = self._resolve_capability(
            "db.planning.context.build",
            owner=owner,
        )
        context_error = self._capability_error(action, context_resolved)
        if context_error is not None:
            return [], [], [context_error]

        context_capability = context_resolved["capability"]
        capabilities = [context_capability]
        specs: list[DbTaskSpec] = []
        selected = [_capability_selection(context_capability, action)]
        context_dependencies: list[TaskDependency] = []
        context_input = _task_input_for_action(action)
        validation_repair_context = _validation_grounding_repair_context(state)
        if validation_repair_context:
            context_input.setdefault(
                "validation_grounding_repair",
                validation_repair_context,
            )
        memory_decision = state.memory_context.get("recall_decision") or {}
        prerequisite_capabilities = [context_capability]
        schema_dependency: TaskDependency | None = None

        if not _state_has_accepted_evidence(state, "schema.asset_profile"):
            schema_resolved = self._resolve_capability(
                "db.schema.inspect",
                owner=_source_capability_owner(action),
            )
            schema_error = self._capability_error(action, schema_resolved)
            if schema_error is None:
                schema_capability = schema_resolved["capability"]
                prerequisite_capabilities.append(schema_capability)
                access_errors = self._access_errors(
                    action,
                    prerequisite_capabilities,
                    state,
                )
                if access_errors:
                    return [], [], access_errors
                schema_spec = _with_deterministic_task_id(
                    state.operation_id,
                    DbTaskSpec(
                        capability_id=schema_capability.id,
                        owner=schema_capability.owner,
                        input={},
                        reason="runtime_prerequisite:planning_schema_inspect",
                        sequence=sequence_start,
                        metadata={
                            **_action_metadata(action, decision_fingerprint),
                            "runtime_prerequisite": True,
                            "schema_inspect": "planning",
                        },
                        deterministic_key=f"{action.action_id}:db.schema.inspect",
                    ),
                )
                specs.append(schema_spec)
                selected.append(
                    _capability_selection(
                        schema_capability,
                        action,
                        reason="runtime_prerequisite:planning_schema_inspect",
                    )
                )
                schema_dependency = TaskDependency(
                    kind="evidence",
                    evidence_kind="schema.asset_profile",
                    evidence_owner=schema_capability.owner,
                    producer_task_id=schema_spec.task_id,
                    producer_capability_id=schema_capability.id,
                    producer_executor_id=schema_capability.executor,
                    evidence_accepted=True,
                    input_hash=_stable_hash({}),
                    operation_id=state.operation_id,
                    metadata={
                        "runtime_prerequisite": True,
                        "producer_action_id": action.action_id,
                        "consumer_action_id": action.action_id,
                    },
                )
                context_dependencies.append(schema_dependency)

        if _state_should_plan_value_grounding_for_planning(state, action):
            grounding_resolved = self._resolve_capability(
                "catalog.value_grounding.plan",
                owner=None,
            )
            hints_resolved = self._resolve_capability(
                "catalog.column_value_hints.resolve",
                owner=None,
            )
            grounding_error = self._capability_error(action, grounding_resolved)
            hints_error = self._capability_error(action, hints_resolved)
            if grounding_error is None and hints_error is None:
                grounding_capability = grounding_resolved["capability"]
                hints_capability = hints_resolved["capability"]
                prerequisite_capabilities.extend(
                    (grounding_capability, hints_capability)
                )
                access_errors = self._access_errors(
                    action,
                    prerequisite_capabilities,
                    state,
                )
                if access_errors:
                    return [], [], access_errors
                grounding_input = _value_grounding_plan_task_input(action, state)
                grounding_spec = _with_deterministic_task_id(
                    state.operation_id,
                    DbTaskSpec(
                        capability_id=grounding_capability.id,
                        owner=grounding_capability.owner,
                        input=grounding_input,
                        reason="runtime_prerequisite:planning_value_grounding_plan",
                        sequence=sequence_start + len(specs),
                        dependencies=(
                            (schema_dependency,)
                            if schema_dependency is not None
                            else ()
                        ),
                        metadata={
                            **_action_metadata(action, decision_fingerprint),
                            "runtime_prerequisite": True,
                            "value_grounding_plan": "planning",
                        },
                        deterministic_key=(
                            f"{action.action_id}:catalog.value_grounding.plan"
                        ),
                    ),
                )
                specs.append(grounding_spec)
                selected.append(
                    _capability_selection(
                        grounding_capability,
                        action,
                        reason="runtime_prerequisite:planning_value_grounding_plan",
                    )
                )
                grounding_dependency = TaskDependency(
                    kind="evidence",
                    evidence_kind="catalog.value_grounding.plan",
                    evidence_owner=grounding_capability.owner,
                    producer_task_id=grounding_spec.task_id,
                    producer_capability_id=grounding_capability.id,
                    producer_executor_id=grounding_capability.executor,
                    evidence_accepted=True,
                    input_hash=_stable_hash(grounding_input),
                    operation_id=state.operation_id,
                    metadata={
                        "runtime_prerequisite": True,
                        "producer_action_id": action.action_id,
                        "consumer_action_id": action.action_id,
                    },
                )
                hints_input = _column_value_hint_task_input(action, state)
                hints_spec = _with_deterministic_task_id(
                    state.operation_id,
                    DbTaskSpec(
                        capability_id=hints_capability.id,
                        owner=hints_capability.owner,
                        input=hints_input,
                        reason="runtime_prerequisite:planning_column_value_hints",
                        sequence=sequence_start + len(specs),
                        dependencies=(grounding_dependency,),
                        metadata={
                            **_action_metadata(action, decision_fingerprint),
                            "runtime_prerequisite": True,
                            "column_value_hints": "planning",
                        },
                        deterministic_key=(
                            f"{action.action_id}:" "catalog.column_value_hints.resolve"
                        ),
                    ),
                )
                specs.append(hints_spec)
                selected.append(
                    _capability_selection(
                        hints_capability,
                        action,
                        reason="runtime_prerequisite:planning_column_value_hints",
                    )
                )
                context_dependencies.append(
                    TaskDependency(
                        kind="evidence",
                        evidence_kind="schema.column_value_hint",
                        evidence_owner=hints_capability.owner,
                        producer_task_id=hints_spec.task_id,
                        producer_capability_id=hints_capability.id,
                        producer_executor_id=hints_capability.executor,
                        evidence_accepted=True,
                        input_hash=_stable_hash(hints_input),
                        operation_id=state.operation_id,
                        metadata={
                            "runtime_prerequisite": True,
                            "producer_action_id": action.action_id,
                            "consumer_action_id": action.action_id,
                        },
                    )
                )

        if _state_should_recall_memory_for_planning(state):
            recall_resolved = self._resolve_capability(
                "memory.semantic.recall",
                owner="memory",
            )
            recall_error = self._capability_error(action, recall_resolved)
            if recall_error is None:
                recall_capability = recall_resolved["capability"]
                prerequisite_capabilities.append(recall_capability)
                access_errors = self._access_errors(
                    action,
                    prerequisite_capabilities,
                    state,
                )
                if access_errors:
                    return [], [], access_errors
                recall_input = _memory_recall_task_input(state)
                recall_spec = _with_deterministic_task_id(
                    state.operation_id,
                    DbTaskSpec(
                        capability_id=recall_capability.id,
                        owner=recall_capability.owner,
                        input=recall_input,
                        reason="runtime_prerequisite:planning_memory_recall",
                        sequence=sequence_start + len(specs),
                        metadata={
                            **_action_metadata(action, decision_fingerprint),
                            "runtime_prerequisite": True,
                            "memory_recall": "planning",
                        },
                        deterministic_key=f"{action.action_id}:memory.semantic.recall",
                    ),
                )
                specs.append(recall_spec)
                selected.append(
                    _capability_selection(
                        recall_capability,
                        action,
                        reason="runtime_prerequisite:planning_memory_recall",
                    )
                )
                context_dependencies.append(
                    TaskDependency(
                        kind="evidence",
                        evidence_kind="memory.semantic.recall",
                        evidence_owner=recall_capability.owner,
                        producer_task_id=recall_spec.task_id,
                        producer_capability_id=recall_capability.id,
                        producer_executor_id=recall_capability.executor,
                        evidence_accepted=True,
                        input_hash=_stable_hash(recall_input),
                        operation_id=state.operation_id,
                        metadata={
                            "runtime_prerequisite": True,
                            "producer_action_id": action.action_id,
                            "consumer_action_id": action.action_id,
                        },
                    )
                )
                context_input.setdefault(
                    "memory_recall_diagnostics",
                    {
                        "registered": True,
                        "queried": True,
                        "decision": memory_decision,
                    },
                )

        if not specs:
            access_errors = self._access_errors(action, capabilities, state)
            if access_errors:
                return [], [], access_errors

        context_metadata = _action_metadata(action, decision_fingerprint)
        if validation_repair_context:
            context_metadata.update(
                {
                    "validation_grounding_repair_attempted": True,
                    "validation_grounding_targets": validation_repair_context[
                        "targets"
                    ],
                    "validation_grounding_fingerprint": validation_repair_context[
                        "fingerprint"
                    ],
                }
            )
        context_spec = _with_deterministic_task_id(
            state.operation_id,
            DbTaskSpec(
                capability_id=context_capability.id,
                owner=context_capability.owner,
                input=context_input,
                reason=f"planner:{action.kind.value}",
                sequence=sequence_start + len(specs),
                dependencies=tuple(context_dependencies),
                metadata=context_metadata,
                deterministic_key=f"{action.action_id}:db.planning.context.build",
            ),
        )
        specs.append(context_spec)
        return specs, selected, []

    def _compile_sql_read_proposal_action(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        owner = _action_owner(action)
        resolved = self._resolve_capability("db.query.plan", owner=owner)
        query_error = self._capability_error(action, resolved)
        if query_error is not None:
            return [], [], [query_error]
        query_capability = resolved["capability"]

        task_input = _task_input_for_action(action)
        task_input.pop("query_plan_ref", None)
        task_input.pop("plan_evidence_id", None)

        specs: list[DbTaskSpec] = []
        selected: list[dict[str, Any]] = []
        query_dependencies: list[TaskDependency] = []
        planning_context = _latest_accepted_evidence_summary(
            state,
            "planning.context",
        )
        if planning_context is not None:
            task_input["planning_context_evidence_id"] = planning_context["id"]
        else:
            context_specs, context_selected, context_errors = (
                self._compile_planning_context_action(
                    action,
                    state=state,
                    sequence_start=sequence_start,
                    decision_fingerprint=decision_fingerprint,
                )
            )
            if context_errors:
                return context_specs, context_selected, context_errors
            context_spec = next(
                (
                    spec
                    for spec in reversed(context_specs)
                    if spec.capability_id == "db.planning.context.build"
                ),
                None,
            )
            if context_spec is None:
                return [], [], [_action_error(action, "missing_planning_context_task")]
            context_resolved = self._resolve_capability(
                context_spec.capability_id,
                owner=context_spec.owner,
            )
            context_error = self._capability_error(action, context_resolved)
            if context_error is not None:
                return [], [], [context_error]
            context_capability = context_resolved["capability"]
            query_dependencies.append(
                TaskDependency(
                    kind="evidence",
                    evidence_kind="planning.context",
                    evidence_owner=context_capability.owner,
                    producer_task_id=context_spec.task_id,
                    producer_capability_id=context_capability.id,
                    producer_executor_id=context_capability.executor,
                    evidence_accepted=True,
                    input_hash=_stable_hash(context_spec.input),
                    operation_id=state.operation_id,
                    metadata={
                        "runtime_prerequisite": True,
                        "producer_action_id": action.action_id,
                        "consumer_action_id": action.action_id,
                        "prerequisite_for": "db.query.plan",
                    },
                )
            )
            specs.extend(context_specs)
            selected.extend(context_selected)

        access_errors = self._access_errors(action, (query_capability,), state)
        if access_errors:
            return [], [], access_errors
        query_spec = _with_deterministic_task_id(
            state.operation_id,
            DbTaskSpec(
                capability_id=query_capability.id,
                owner=query_capability.owner,
                input=task_input,
                reason=f"planner:{action.kind.value}",
                sequence=sequence_start + len(specs),
                dependencies=tuple(query_dependencies),
                metadata=_action_metadata(action, decision_fingerprint),
                deterministic_key=f"{action.action_id}:{query_capability.id}",
            ),
        )
        specs.append(query_spec)
        selected.append(_capability_selection(query_capability, action))
        return specs, selected, []

    def _compile_validated_sql_action(
        self,
        action: DbPlannerAction,
        *,
        sql_operation: str,
        execute_capability_id: str,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        resolved_sql, sql_error = _resolve_sql_input_for_action(
            action,
            state,
            sql_operation=sql_operation,
        )
        if sql_error is not None or resolved_sql is None:
            return [], [], [_action_error(action, sql_error or "missing_sql")]
        owner = _action_owner(action)
        validation = self._resolve_capability("db.sql.validate", owner=owner)
        execute = self._resolve_capability(execute_capability_id, owner=owner)
        plan_validation = None
        planning_context = _latest_accepted_evidence_summary(
            state,
            "planning.context",
        )
        if (
            resolved_sql.source_evidence_kind == "query.plan.proposal"
            and planning_context is not None
        ):
            plan_validation = self._resolve_capability(
                "db.query.plan.validate",
                owner="db_runtime",
            )
        errors = [
            error
            for error in (
                (
                    self._capability_error(action, plan_validation)
                    if plan_validation is not None
                    else None
                ),
                self._capability_error(action, validation),
                self._capability_error(action, execute),
            )
            if error is not None
        ]
        if errors:
            return [], [], errors
        capabilities = [
            *([plan_validation["capability"]] if plan_validation is not None else []),
            validation["capability"],
            execute["capability"],
        ]
        access_error = self._access_errors(action, capabilities, state)
        if access_error:
            return [], [], access_error
        metadata = {
            **_action_metadata(action, decision_fingerprint),
            "sql_provenance": _sql_provenance_metadata(resolved_sql),
        }
        specs: list[DbTaskSpec] = []
        validation_dependencies: list[TaskDependency] = []
        if plan_validation is not None and planning_context is not None:
            plan_capability = plan_validation["capability"]
            plan_validation_spec = DbTaskSpec(
                capability_id=plan_capability.id,
                owner=plan_capability.owner,
                input={
                    "plan_evidence_id": resolved_sql.source_evidence_id,
                    "planning_context_evidence_id": planning_context["id"],
                },
                reason=f"planner:{action.kind.value}:query_plan_validation",
                sequence=sequence_start,
                dependencies=(
                    (resolved_sql.query_plan_dependency,)
                    if resolved_sql.query_plan_dependency is not None
                    else ()
                ),
                metadata=metadata,
                deterministic_key=f"{action.action_id}:db.query.plan.validate",
            )
            plan_validation_spec = _with_deterministic_task_id(
                state.operation_id,
                plan_validation_spec,
            )
            specs.append(plan_validation_spec)
            validation_dependencies.append(
                TaskDependency(
                    kind="evidence",
                    evidence_kind="query.plan.validation",
                    evidence_owner=plan_capability.owner,
                    producer_task_id=plan_validation_spec.task_id,
                    producer_capability_id=plan_capability.id,
                    producer_executor_id=plan_capability.executor,
                    evidence_accepted=True,
                    operation_id=state.operation_id,
                    metadata={
                        "sql_resolution": True,
                        "provenance": "query_plan_validation",
                    },
                )
            )
        elif resolved_sql.query_plan_dependency is not None:
            validation_dependencies.append(resolved_sql.query_plan_dependency)

        validation_spec = DbTaskSpec(
            capability_id="db.sql.validate",
            owner=validation["capability"].owner,
            input={"sql": resolved_sql.sql, "operation": sql_operation},
            reason=f"planner:{action.kind.value}:validation",
            sequence=sequence_start + len(specs),
            dependencies=tuple(validation_dependencies),
            metadata=metadata,
            deterministic_key=f"{action.action_id}:db.sql.validate",
        )
        validation_spec = _with_deterministic_task_id(
            state.operation_id,
            validation_spec,
        )
        execute_capability = execute["capability"]
        execute_input: dict[str, Any] = {
            "sql_ref": "sql.validation",
            "params": list(action.input.get("params") or ()),
        }
        if action.input.get("param_specs"):
            execute_input["param_specs"] = list(action.input.get("param_specs") or ())
        if action.input.get("focus") is not None:
            execute_input["focus"] = action.input["focus"]
        execute_spec = DbTaskSpec(
            capability_id=execute_capability_id,
            owner=execute_capability.owner,
            input=execute_input,
            reason=f"planner:{action.kind.value}",
            sequence=sequence_start + len(specs) + 1,
            metadata=metadata,
            deterministic_key=f"{action.action_id}:{execute_capability_id}",
        )
        execute_spec = _with_deterministic_task_id(
            state.operation_id,
            execute_spec,
        )
        specs.extend((validation_spec, execute_spec))
        return (
            specs,
            [_capability_selection(item, action) for item in capabilities],
            [],
        )

    def _compile_sql_validation_only_action(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        sql = action.input.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return [], [], [_action_error(action, "missing_sql")]
        owner = _action_owner(action)
        resolved = self._resolve_capability("db.sql.validate", owner=owner)
        error = self._capability_error(action, resolved)
        if error is not None:
            return [], [], [error]
        capability = resolved["capability"]
        access_errors = self._access_errors(action, [capability], state)
        if access_errors:
            return [], [], access_errors
        spec = DbTaskSpec(
            capability_id=capability.id,
            owner=capability.owner,
            input={"sql": sql, "operation": "write"},
            reason=f"planner:{action.kind.value}:validation",
            sequence=sequence_start,
            metadata=_action_metadata(action, decision_fingerprint),
            deterministic_key=f"{action.action_id}:{capability.id}",
        )
        spec = _with_deterministic_task_id(state.operation_id, spec)
        return [spec], [_capability_selection(capability, action)], []

    def _compile_relationship_paths_action(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        owner = _action_owner(action)
        resolved = self._resolve_capability(
            "catalog.relationship_paths.find",
            owner=owner,
        )
        error = self._capability_error(action, resolved)
        if error is not None:
            return [], [], [error]
        capability = resolved["capability"]
        access_errors = self._access_errors(action, [capability], state)
        if access_errors:
            return [], [], access_errors
        task_input, input_errors = _relationship_paths_task_input(action, state)
        if input_errors:
            return [], [], [_action_error(action, item) for item in input_errors]
        spec = DbTaskSpec(
            capability_id=capability.id,
            owner=capability.owner,
            input=task_input,
            reason=f"planner:{action.kind.value}",
            sequence=sequence_start,
            metadata=_action_metadata(action, decision_fingerprint),
            deterministic_key=f"{action.action_id}:{capability.id}",
        )
        spec = _with_deterministic_task_id(state.operation_id, spec)
        return [spec], [_capability_selection(capability, action)], []

    def _compile_monitor_action(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        capability_id = _MONITOR_ACTION_CAPABILITIES[action.kind]
        owner = _action_owner(action)
        resolved = self._resolve_capability(capability_id, owner=owner)
        error = self._capability_error(action, resolved)
        if error is not None:
            return [], [], [error]
        capability = resolved["capability"]
        access_errors = self._access_errors(action, [capability], state)
        if access_errors:
            return [], [], access_errors
        expected_evidence, evidence_error = _monitor_action_output_evidence(action)
        if evidence_error is not None:
            return [], [], [_action_error(action, evidence_error)]
        task_input = _task_input_for_action(action)
        spec = DbTaskSpec(
            capability_id=capability.id,
            owner=capability.owner,
            input=task_input,
            reason=f"planner:{action.kind.value}",
            sequence=sequence_start,
            metadata=_action_metadata(action, decision_fingerprint),
            deterministic_key=f"{action.action_id}:{capability.id}",
        )
        spec = _with_deterministic_task_id(state.operation_id, spec)
        return (
            [spec],
            [
                _capability_selection(
                    capability,
                    action,
                    output_evidence=(expected_evidence,),
                )
            ],
            [],
        )

    def _compile_memory_commit_action(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        owner = _action_owner(action)
        resolved = self._resolve_capability("db.memory.commit_update", owner=owner)
        error = self._capability_error(action, resolved)
        if error is not None:
            return [], [], [error]
        capability = resolved["capability"]
        access_errors = self._access_errors(action, [capability], state)
        if access_errors:
            return [], [], access_errors
        proposal, proposal_error = _resolve_memory_proposal_for_action(action, state)
        if proposal_error is not None or proposal is None:
            return (
                [],
                [],
                [
                    _action_error(
                        action, proposal_error or "missing_accepted_memory_proposal"
                    )
                ],
            )
        proposal_id = str(proposal.get("id") or "")
        payload_fingerprint = str(proposal.get("payload_fingerprint") or "")
        proposal_fingerprint = str(
            proposal.get("proposal_fingerprint") or payload_fingerprint
        )
        task_input = _task_input_for_action(action)
        task_input.setdefault("proposal_evidence_id", proposal_id)
        if proposal_fingerprint:
            task_input.setdefault("proposal_fingerprint", proposal_fingerprint)
        metadata = {
            **_action_metadata(action, decision_fingerprint),
            "proposal_evidence_id": proposal_id,
        }
        if proposal_fingerprint:
            metadata["proposal_fingerprint"] = proposal_fingerprint
        dependency = TaskDependency(
            kind="evidence",
            evidence_kind="db.memory.proposal",
            evidence_id=proposal_id,
            evidence_owner=str(proposal.get("owner") or "db_runtime"),
            producer_task_id=proposal.get("task_id"),
            producer_capability_id="db.memory.plan_update",
            producer_executor_id="db_runtime.memory.plan_update",
            evidence_accepted=True,
            payload_fingerprint=payload_fingerprint or None,
            operation_id=state.operation_id,
            metadata={"memory_update_commit": True},
        )
        spec = DbTaskSpec(
            capability_id=capability.id,
            owner=capability.owner,
            input=task_input,
            reason=f"planner:{action.kind.value}",
            sequence=sequence_start,
            dependencies=(dependency,),
            metadata=metadata,
            deterministic_key=(
                f"{action.action_id}:{capability.id}:"
                f"{proposal_fingerprint or proposal_id}"
            ),
            idempotency_key=_stable_hash(
                {
                    "operation_id": state.operation_id,
                    "capability_id": capability.id,
                    "owner": capability.owner,
                    "proposal_evidence_id": proposal_id,
                    "proposal_fingerprint": proposal_fingerprint,
                }
            ),
        )
        spec = _with_deterministic_task_id(state.operation_id, spec)
        return [spec], [_capability_selection(capability, action)], []

    def _compile_capability_specs(
        self,
        action: DbPlannerAction,
        *,
        capability_ids: tuple[str, ...],
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
    ) -> tuple[list[DbTaskSpec], list[dict[str, Any]], list[dict[str, Any]]]:
        owner = _action_owner(action)
        resolved = [
            self._resolve_capability(capability_id, owner=owner)
            for capability_id in capability_ids
        ]
        errors = [
            error
            for error in (self._capability_error(action, item) for item in resolved)
            if error is not None
        ]
        if errors:
            return [], [], errors
        capabilities = [item["capability"] for item in resolved]
        access_errors = self._access_errors(action, capabilities, state)
        if access_errors:
            return [], [], access_errors
        specs: list[DbTaskSpec] = []
        task_input = _task_input_for_action(action)
        task_dependencies: tuple[TaskDependency, ...] = ()
        if action.kind is DbPlannerActionKind.REPAIR_QUERY_PLAN:
            (
                task_input,
                task_dependencies,
                repair_errors,
            ) = _repair_query_plan_task_input_and_dependencies(
                action,
                state,
                task_input,
            )
            if repair_errors:
                return [], [], repair_errors
        for offset, capability in enumerate(capabilities):
            specs.append(
                _with_deterministic_task_id(
                    state.operation_id,
                    DbTaskSpec(
                        capability_id=capability.id,
                        owner=capability.owner,
                        input=task_input,
                        reason=f"planner:{action.kind.value}",
                        sequence=sequence_start + offset,
                        dependencies=task_dependencies,
                        metadata=_action_metadata(action, decision_fingerprint),
                        deterministic_key=f"{action.action_id}:{capability.id}",
                    ),
                )
            )
        return specs, [_capability_selection(item, action) for item in capabilities], []

    def _catalog_prerequisite_capabilities(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        if action.kind is DbPlannerActionKind.SEARCH_COLUMN_VALUES:
            target_capability_id = "catalog.column_values.search"
            tables, columns = _column_value_scope_for_action(action)
            if not tables or not columns or not _state_allows_read_profile(state):
                return [], [], []
        elif action.kind in {
            DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
            DbPlannerActionKind.PROPOSE_SQL_READ,
        }:
            target_capability_id = "catalog.column_value_hints.resolve"
            tables, columns = _column_value_scope_for_action(action)
            if (
                action.kind is DbPlannerActionKind.PROPOSE_SQL_READ
                and _latest_accepted_evidence_summary(state, "planning.context")
                is not None
            ):
                return [], [], []
            if not _state_should_plan_value_grounding_for_planning(state, action):
                return [], [], []
        else:
            return [], [], []

        selected: list[dict[str, Any]] = []
        prerequisites: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        source_owner = _source_capability_owner(action)
        catalog_owner = (
            _action_owner(action)
            if action.kind is DbPlannerActionKind.SEARCH_COLUMN_VALUES
            else None
        )

        def add_prerequisite(
            capability_id: str,
            *,
            owner: str | None,
        ) -> Any | None:
            resolved = self._resolve_capability(capability_id, owner=owner)
            error = resolved.get("error")
            if error is not None:
                return None
            capability = resolved["capability"]
            access_errors = self._access_errors(action, (capability,), state)
            if access_errors:
                errors.extend(access_errors)
                return None
            selected.append(
                _capability_selection(
                    capability,
                    action,
                    reason=f"runtime_prerequisite:{action.kind.value}",
                )
            )
            prerequisites.append(
                {
                    "for_action_id": action.action_id,
                    "for_action_kind": action.kind.value,
                    "for_capability_id": target_capability_id,
                    "capability_id": capability.id,
                    "owner": capability.owner,
                    "access": capability.access.value,
                    "output_evidence": sorted(capability.output_evidence),
                    "reason": _CATALOG_COLUMN_VALUE_GROUNDING_REASON,
                    "tables": list(tables),
                    "columns": list(columns),
                }
            )
            return capability

        if not _state_has_accepted_evidence(state, "schema.asset_profile"):
            add_prerequisite("db.schema.inspect", owner=source_owner)
        if not _state_has_accepted_evidence(state, "catalog.source_registered"):
            add_prerequisite("catalog.source.register", owner=catalog_owner)
        add_prerequisite(
            "catalog.value_grounding.plan",
            owner=catalog_owner,
        )
        profile_capability = add_prerequisite(
            "db.column_values.profile",
            owner=source_owner,
        )
        if profile_capability is not None:
            add_prerequisite(
                "catalog.column_values.register",
                owner=catalog_owner,
            )
        return selected, prerequisites, errors

    def _resolve_capability(
        self,
        capability_id: str,
        *,
        owner: str | None,
    ) -> dict[str, Any]:
        try:
            capability = self.runtime.registry.get_capability(
                capability_id, owner=owner
            )
        except ValueError:
            return {"error": f"ambiguous_capability:{capability_id}"}
        except KeyError:
            return {"error": f"missing_capability:{capability_id}"}
        return {"capability": capability}

    def _capability_error(
        self,
        action: DbPlannerAction,
        resolved: dict[str, Any],
    ) -> dict[str, Any] | None:
        error = resolved.get("error")
        if error is None:
            return None
        return _action_error(action, str(error))

    def _access_errors(
        self,
        action: DbPlannerAction,
        capabilities: Iterable[Any],
        state: DbLoopState,
    ) -> list[dict[str, Any]]:
        safety_frame = state.safety_frame or {}
        max_access = str(
            safety_frame.get("max_access")
            or safety_frame.get("max_allowed_access")
            or AccessMode.ADMIN.value
        )
        allowed = {
            str(item)
            for item in (
                safety_frame.get("allowed_capabilities")
                or safety_frame.get("capability_allowlist")
                or ()
            )
        }
        denied = {
            str(item)
            for item in (
                safety_frame.get("denied_capabilities")
                or safety_frame.get("capability_denylist")
                or ()
            )
        }
        errors: list[dict[str, Any]] = []
        for capability in capabilities:
            if allowed and capability.id not in allowed:
                errors.append(
                    _action_error(
                        action, f"capability_outside_contract:{capability.id}"
                    )
                )
            if capability.id in denied:
                errors.append(
                    _action_error(action, f"capability_denied:{capability.id}")
                )
            if _access_rank(capability.access.value) > _access_rank(max_access):
                errors.append(
                    _action_error(
                        action,
                        (
                            "access_outside_contract:"
                            f"{capability.id}:{capability.access.value}>{max_access}"
                        ),
                    )
                )
        return errors

    def _compiled_contract_snapshot(
        self,
        *,
        decision: DbPlannerDecision,
        state: DbLoopState,
        decision_fingerprint: str,
        selected_capabilities: tuple[dict[str, Any], ...],
        compiled_action_ids: tuple[str, ...],
        runtime_prerequisite_capabilities: tuple[dict[str, Any], ...] = (),
        runtime_prerequisites: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        max_access = AccessMode.NONE
        required_evidence: set[str] = set()
        contract_capabilities = (
            *selected_capabilities,
            *runtime_prerequisite_capabilities,
        )
        for selected in contract_capabilities:
            access = AccessMode(selected["access"])
            if _access_rank(access.value) > _access_rank(max_access.value):
                max_access = access
            required_evidence.update(str(item) for item in selected["output_evidence"])
        raw_planner_intent = dict(decision.intent)
        operation_type = _explicit_mode_operation_type(state.explicit_mode) or str(
            decision.intent.get("operation_type")
            or decision.intent.get("label")
            or "db.run"
        )
        planner_intent = {**raw_planner_intent, "operation_type": operation_type}
        contract = DbOperationContract(
            operation_type=operation_type,
            required_capabilities=tuple(
                dict.fromkeys(str(item["id"]) for item in contract_capabilities)
            ),
            required_evidence=tuple(sorted(required_evidence)),
            access=max_access,
            limits=self.runtime.config.limits,
            policy_ids=tuple(
                str(item) for item in decision.intent.get("policy_ids") or ()
            ),
            metadata={
                "planner_intent": planner_intent,
                "planner_raw_intent": raw_planner_intent,
                "planner_decision_fingerprint": decision_fingerprint,
                "compiled_action_ids": list(compiled_action_ids),
                "selected_capabilities": list(selected_capabilities),
                "runtime_prerequisite_capabilities": list(
                    runtime_prerequisite_capabilities
                ),
                "runtime_prerequisites": list(runtime_prerequisites),
            },
        )
        return _contract_snapshot(contract)

    async def _persist_compiled_contract(
        self,
        operation: Operation,
        compilation: DbActionCompilation,
    ) -> Operation:
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
                "fingerprint": _stable_hash(decision.to_dict()),
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
                "decision_fingerprint": _stable_hash(decision.to_dict()),
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
        evidence = await self.runtime.store.list_evidence(operation_id)
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
        decision_fingerprint = _stable_hash(decision.to_dict())
        compiled_action_fingerprints = _compiled_action_fingerprints(decision)
        task_spec_fingerprints = tuple(
            _stable_hash(spec.to_dict()) for spec in compilation.task_specs
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
        progress_fingerprint = _stable_hash(
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
        evidence = tuple(await self.runtime.store.list_evidence(operation_id))
        tasks = tuple(await self.runtime.store.list_tasks(operation_id))
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
        evidence = await self.runtime.store.list_evidence(operation.id)
        tasks = await self.runtime.store.list_tasks(operation.id)
        return DbLoopResult(
            status=status,
            evidence_refs=tuple(_evidence_ref(item) for item in evidence if item.id),
            task_refs=tuple(_task_ref(item) for item in tasks),
            warnings=tuple(warnings),
            diagnostics=dict(diagnostics),
        )


def _coerce_action(
    raw_action: Any,
) -> tuple[DbPlannerAction | None, dict[str, Any] | None]:
    if isinstance(raw_action, DbPlannerAction):
        return raw_action, None
    if isinstance(raw_action, Mapping):
        try:
            return DbPlannerAction.from_dict(raw_action), None
        except Exception as exc:
            return None, {
                "action_id": str(raw_action.get("action_id") or ""),
                "kind": str(raw_action.get("kind") or ""),
                "error": f"invalid_action:{exc}",
            }
    return None, {"action_id": "", "kind": "", "error": "invalid_action_shape"}


def _ordered_actions_or_errors(
    actions: tuple[DbPlannerAction, ...],
) -> tuple[tuple[DbPlannerAction, ...], tuple[dict[str, Any], ...]]:
    by_id: dict[str, DbPlannerAction] = {}
    duplicate_ids: set[str] = set()
    for action in actions:
        if action.action_id in by_id:
            duplicate_ids.add(action.action_id)
            continue
        by_id[action.action_id] = action
    if duplicate_ids:
        return (), tuple(
            _action_error(action, "duplicate_action_id")
            for action in actions
            if action.action_id in duplicate_ids
        )

    missing_errors: list[dict[str, Any]] = []
    dependency_sets: dict[str, tuple[str, ...]] = {}
    for action in actions:
        dependencies = tuple(dict.fromkeys(action.depends_on))
        dependency_sets[action.action_id] = dependencies
        for dependency in dependencies:
            if dependency not in by_id:
                missing_errors.append(
                    _action_error(action, f"missing_dependency:{dependency}")
                )
    if missing_errors:
        return (), tuple(missing_errors)

    dependents: dict[str, list[str]] = {action.action_id: [] for action in actions}
    remaining_dependencies: dict[str, int] = {}
    for action in actions:
        dependencies = dependency_sets[action.action_id]
        remaining_dependencies[action.action_id] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(action.action_id)

    ready = [
        action.action_id
        for action in actions
        if remaining_dependencies[action.action_id] == 0
    ]
    ordered: list[DbPlannerAction] = []
    while ready:
        action_id = ready.pop(0)
        ordered.append(by_id[action_id])
        for dependent_id in dependents[action_id]:
            remaining_dependencies[dependent_id] -= 1
            if remaining_dependencies[dependent_id] == 0:
                ready.append(dependent_id)

    if len(ordered) != len(actions):
        cyclic_ids = tuple(
            action.action_id
            for action in actions
            if remaining_dependencies[action.action_id] > 0
        )
        cycle_label = "->".join(cyclic_ids) if cyclic_ids else "unknown"
        return (), tuple(
            _action_error(action, f"dependency_cycle:{cycle_label}")
            for action in actions
            if action.action_id in cyclic_ids
        )
    return tuple(ordered), ()


def _with_spec_dependencies(
    specs: list[DbTaskSpec],
    dependencies: tuple[TaskDependency, ...],
) -> list[DbTaskSpec]:
    if not dependencies:
        return specs
    return [
        replace(
            spec,
            dependencies=_merge_dependencies(spec.dependencies, dependencies),
        )
        for spec in specs
    ]


def _mode_action_errors(
    action: DbPlannerAction,
    state: DbLoopState,
) -> tuple[dict[str, Any], ...]:
    metadata_only_modes = {
        DbIntentKind.SCHEMA_QUERY.value,
        DbIntentKind.SCHEMA_RELATIONSHIP_QUERY.value,
    }
    effective_mode = _explicit_mode_operation_type(state.explicit_mode)
    if effective_mode in metadata_only_modes and action.kind in _SQL_QUERY_ACTIONS:
        return (
            _action_error(
                action,
                f"action_outside_explicit_mode:{action.kind.value}:{effective_mode}",
            ),
        )
    return ()


def _explicit_mode_operation_type(mode: str | None) -> str | None:
    normalized = (mode or "").strip().lower()
    if normalized in {"schema", "schema.query"}:
        return DbIntentKind.SCHEMA_QUERY.value
    if normalized in {
        "relationships",
        "relationship",
        "schema_relationship",
        "schema.relationship_query",
    }:
        return DbIntentKind.SCHEMA_RELATIONSHIP_QUERY.value
    if normalized in {"data", "data.query", "query", "read"}:
        return DbIntentKind.DATA_QUERY.value
    if normalized in {"write", "write.propose"}:
        return DbIntentKind.WRITE_PROPOSE.value
    if normalized in {"write.execute", "write_execute"}:
        return DbIntentKind.WRITE_EXECUTE.value
    if normalized == "admin":
        return DbIntentKind.ADMIN.value
    return None


def _with_deterministic_task_id(
    operation_id: str,
    spec: DbTaskSpec,
) -> DbTaskSpec:
    if spec.task_id:
        return spec
    input_hash = _stable_hash(spec.input)
    idempotency_key = spec.idempotency_key or _stable_hash(
        {
            "operation_id": operation_id,
            "capability_id": spec.capability_id,
            "owner": spec.owner,
            "input_hash": input_hash,
            "sequence": spec.sequence,
            "deterministic_key": spec.deterministic_key,
        }
    )
    task_fingerprint = _stable_hash(
        {
            "operation_id": operation_id,
            "idempotency_key": idempotency_key,
        }
    )
    return replace(spec, task_id=f"db-task-{task_fingerprint[:32]}")


def _merge_dependencies(
    left: tuple[TaskDependency, ...],
    right: tuple[TaskDependency, ...],
) -> tuple[TaskDependency, ...]:
    merged: list[TaskDependency] = []
    seen: set[str] = set()
    for dependency in (*left, *right):
        fingerprint = _stable_hash(dependency.to_dict())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        merged.append(dependency)
    return tuple(merged)


def _has_terminal_compilation_error(compilation: DbActionCompilation) -> bool:
    terminal_prefixes = (
        "duplicate_action_id",
        "missing_dependency:",
        "dependency_cycle:",
        "dependency_not_durable:",
        "ambiguous_continuation:",
        "ambiguous_continuation_evidence:",
    )
    return any(
        str(item.get("error") or "").startswith(terminal_prefixes)
        for item in compilation.rejected_action_summaries
    )


def _has_runtime_continuation_block(compilation: DbActionCompilation) -> bool:
    return any(
        isinstance(item.get("continuation"), Mapping)
        and item["continuation"].get("status") == "blocked"
        and item["continuation"].get("source") == "runtime_continuation"
        for item in compilation.rejected_action_summaries
    )


def _compiled_action_fingerprints(
    decision: DbPlannerDecision,
) -> tuple[str, ...]:
    return tuple(
        _stable_hash(
            {
                "kind": action.kind.value,
                "input": action.input,
                "depends_on": list(action.depends_on),
                "metadata": action.metadata,
            }
        )
        for action in decision.actions
    )


def _execution_error_fingerprint(error: Mapping[str, Any]) -> str:
    return _stable_hash(
        {
            "capability_id": error.get("capability_id"),
            "error": error.get("error"),
            "error_type": error.get("error_type"),
            "readiness": error.get("readiness"),
        }
    )


def _is_sql_execution_error(error: Mapping[str, Any]) -> bool:
    capability_id = str(error.get("capability_id") or "")
    if capability_id in {
        "db.sql.validate",
        "db.sql.execute_read",
        "db.sql.execute_write",
    }:
        return True
    text = str(error.get("error") or "").lower()
    return "sql" in text or "validation_failed" in text


def _blocked_resource_execution_errors(
    errors: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(error) for error in errors if _is_blocked_resource_error(error))


def _is_blocked_resource_error(error: Mapping[str, Any]) -> bool:
    if str(error.get("capability_id") or "") != "db.sql.validate":
        return False
    text = str(error.get("error") or "").lower()
    return any(
        marker in text
        for marker in (
            "sql guardrail rejected blocked table",
            "sql guardrail rejected blocked column",
            "sql guardrail rejected table(s) outside allowlist",
        )
    )


def _new_evidence_refs(
    before: tuple[dict[str, Any], ...],
    after: tuple[dict[str, Any], ...],
    *,
    include_loop_control: bool,
) -> tuple[dict[str, Any], ...]:
    before_ids = {item.get("id") for item in before}
    refs = tuple(item for item in after if item.get("id") not in before_ids)
    if include_loop_control:
        return refs
    return tuple(
        item
        for item in refs
        if item.get("kind") not in DB_FINALIZATION_CONTROL_EVIDENCE_KINDS
    )


def _action_owner(action: DbPlannerAction) -> str | None:
    owner = action.input.get("owner") or action.input.get("capability_owner")
    return str(owner) if owner else None


def _source_capability_owner(action: DbPlannerAction) -> str | None:
    for source in (action.input, action.metadata):
        owner = (
            source.get("source_owner")
            or source.get("db_owner")
            or source.get("connector_owner")
            or source.get("source_capability_owner")
        )
        if owner:
            return str(owner)
    return None


def _task_input_for_action(action: DbPlannerAction) -> dict[str, Any]:
    return {
        key: value
        for key, value in action.input.items()
        if key not in {"owner", "capability_owner"}
    }


def _summary_id(summary: Mapping[str, Any]) -> str:
    return str(summary.get("id") or "").strip()


def _memory_update_runtime_continuation_action(
    state: DbLoopState,
    *,
    current_action_ids: set[str],
) -> DbPlannerAction | None:
    if not _state_is_memory_update_operation(state):
        return None
    proposals = _uncommitted_memory_proposal_summaries(state)
    if not proposals:
        return None
    owner = _memory_commit_capability_owner_for_state(state, proposals)
    action_id = _runtime_memory_commit_action_id(proposals, current_action_ids)
    action_input: dict[str, Any] = {}
    if owner:
        action_input["owner"] = owner
    if len(proposals) > 1:
        diagnostic = {
            "status": "blocked",
            "source": "runtime_continuation",
            "error": "ambiguous_continuation:latest_uncommitted_memory_proposal",
            "role": "latest_uncommitted_memory_proposal",
            "evidence_kind": "db.memory.proposal",
            "candidate_count": len(proposals),
            "candidate_ids": [_summary_id(item) for item in proposals],
        }
        return DbPlannerAction(
            action_id=action_id,
            kind=DbPlannerActionKind.COMMIT_MEMORY_UPDATE,
            input=action_input,
            rationale=("Runtime found multiple accepted uncommitted memory proposals."),
            metadata={
                "runtime_continuation": True,
                "continuation_resolution": diagnostic,
            },
        )

    proposal = proposals[0]
    proposal_id = _summary_id(proposal)
    proposal_fingerprint = str(
        proposal.get("proposal_fingerprint")
        or proposal.get("payload_fingerprint")
        or ""
    ).strip()
    action_input["proposal_evidence_id"] = proposal_id
    if proposal_fingerprint:
        action_input["proposal_fingerprint"] = proposal_fingerprint
    diagnostic = {
        "status": "resolved",
        "source": "runtime_continuation",
        "role": "latest_uncommitted_memory_proposal",
        "evidence_kind": "db.memory.proposal",
        "evidence_id": proposal_id,
        "candidate_count": 1,
    }
    return DbPlannerAction(
        action_id=action_id,
        kind=DbPlannerActionKind.COMMIT_MEMORY_UPDATE,
        input=action_input,
        rationale="Runtime continuation for accepted DB memory proposal.",
        metadata={
            "runtime_continuation": True,
            "continuation_resolution": diagnostic,
        },
    )


def _validation_grounding_runtime_continuation_action(
    state: DbLoopState,
    *,
    current_action_ids: set[str],
) -> DbPlannerAction | None:
    repair_context = _validation_grounding_repair_context(state)
    missing_refs = tuple(
        str(item).strip()
        for item in repair_context.get("missing_target_refs", ())
        if str(item).strip()
    )
    if not missing_refs:
        return None
    if not _state_allows_read_profile(state):
        return None

    fingerprint = str(repair_context.get("fingerprint") or "").strip()
    targets = _safe_target_list(repair_context.get("targets"))
    action_input: dict[str, Any] = {}
    source_owner = _single_source_owner_for_state(state)
    if source_owner:
        action_input["source_owner"] = source_owner

    metadata: dict[str, Any] = {
        "runtime_continuation": True,
        "continuation": "validation_grounding.context_refresh",
        "validation_grounding_fingerprint": fingerprint,
        "validation_grounding_targets": targets,
    }
    if _validation_grounding_context_refresh_exhausted(state, repair_context):
        metadata["continuation_resolution"] = {
            "status": "blocked",
            "source": "runtime_continuation",
            "error": "validation_grounding_context_refresh_exhausted",
            "continuation": "validation_grounding.context_refresh",
            "validation_grounding_fingerprint": fingerprint,
            "validation_grounding_targets": targets,
            "missing_target_refs": list(missing_refs),
        }

    return DbPlannerAction(
        action_id=_runtime_validation_grounding_action_id(
            repair_context,
            current_action_ids,
        ),
        kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
        input=action_input,
        rationale="Runtime continuation for validation-driven value grounding.",
        metadata=metadata,
    )


def _runtime_validation_grounding_action_id(
    repair_context: Mapping[str, Any],
    current_action_ids: set[str],
) -> str:
    fingerprint = str(repair_context.get("fingerprint") or "")
    seed = {
        "fingerprint": fingerprint,
        "target_refs": list(repair_context.get("target_refs") or ()),
    }
    action_id = f"runtime_validation_grounding_{_stable_hash(seed)[:12]}"
    if action_id not in current_action_ids:
        return action_id
    return (
        f"{action_id}_{_stable_hash({'existing_ids': sorted(current_action_ids)})[:8]}"
    )


def _validation_grounding_context_refresh_exhausted(
    state: DbLoopState,
    repair_context: Mapping[str, Any],
) -> bool:
    fingerprint = str(repair_context.get("fingerprint") or "").strip()
    if not fingerprint:
        return False
    missing_refs = {
        str(item).strip().lower()
        for item in repair_context.get("missing_target_refs", ())
        if str(item).strip()
    }
    if not missing_refs:
        return False
    for summary in state.accepted_evidence_summaries:
        if summary.get("kind") != "planning.context":
            continue
        if str(summary.get("validation_grounding_fingerprint") or "") != fingerprint:
            continue
        attempted_refs = {
            str(item).strip().lower()
            for item in summary.get("validation_grounding_target_refs", ())
            if str(item).strip()
        }
        if missing_refs <= attempted_refs:
            return True
    return False


def _single_source_owner_for_state(state: DbLoopState) -> str | None:
    source_scope = _string_list(state.source_scope) or _string_list(
        state.normalized_user_request.get("source_scope")
    )
    unique_scope = tuple(dict.fromkeys(source_scope))
    if len(unique_scope) == 1:
        return unique_scope[0]

    source_capability_owners = tuple(
        dict.fromkeys(
            str(summary.get("owner") or "").strip()
            for summary in state.capability_summaries
            if summary.get("id") in {"db.schema.inspect", "db.column_values.profile"}
            and str(summary.get("owner") or "").strip()
        )
    )
    if len(source_capability_owners) == 1:
        return source_capability_owners[0]
    return None


def _state_is_memory_update_operation(state: DbLoopState) -> bool:
    candidates: list[Any] = [
        state.explicit_mode,
        state.normalized_user_request.get("mode"),
        state.normalized_user_request.get("operation_type"),
    ]
    snapshot = state.latest_compiled_contract_snapshot
    if isinstance(snapshot, Mapping):
        candidates.append(snapshot.get("operation_type"))
        metadata = snapshot.get("metadata")
        if isinstance(metadata, Mapping):
            planner_intent = metadata.get("planner_intent")
            if isinstance(planner_intent, Mapping):
                candidates.append(planner_intent.get("operation_type"))
    for candidate in candidates:
        normalized = str(candidate or "").strip().lower().replace("_", ".")
        if normalized == DbIntentKind.MEMORY_UPDATE.value:
            return True
    return False


def _uncommitted_memory_proposal_summaries(
    state: DbLoopState,
) -> tuple[dict[str, Any], ...]:
    proposals = tuple(
        dict(summary)
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "db.memory.proposal"
        and summary.get("accepted", True) is True
        and _summary_id(summary)
    )
    if not proposals:
        return ()
    committed = _accepted_memory_definition_refs(state)
    uncommitted: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal_id = _summary_id(proposal)
        proposal_fingerprint = str(
            proposal.get("proposal_fingerprint")
            or proposal.get("payload_fingerprint")
            or ""
        ).strip()
        if proposal_id in committed["ids"]:
            continue
        if proposal_fingerprint and proposal_fingerprint in committed["fingerprints"]:
            continue
        uncommitted.append(proposal)
    return tuple(uncommitted)


def _accepted_memory_definition_refs(
    state: DbLoopState,
) -> dict[str, set[str]]:
    proposal_ids: set[str] = set()
    proposal_fingerprints: set[str] = set()
    for summary in state.accepted_evidence_summaries:
        if summary.get("kind") != "db.memory.definition":
            continue
        if summary.get("accepted", True) is not True:
            continue
        proposal_id = str(summary.get("proposal_evidence_id") or "").strip()
        if proposal_id:
            proposal_ids.add(proposal_id)
        proposal_fingerprint = str(summary.get("proposal_fingerprint") or "").strip()
        if proposal_fingerprint:
            proposal_fingerprints.add(proposal_fingerprint)
    return {"ids": proposal_ids, "fingerprints": proposal_fingerprints}


def _single_capability_owner_for_state(
    state: DbLoopState,
    capability_id: str,
) -> str | None:
    owners = tuple(
        str(summary.get("owner") or "").strip()
        for summary in state.capability_summaries
        if summary.get("id") == capability_id
        and str(summary.get("owner") or "").strip()
    )
    unique = tuple(dict.fromkeys(owners))
    if len(unique) == 1:
        return unique[0]
    return None


def _memory_commit_capability_owner_for_state(
    state: DbLoopState,
    proposals: tuple[dict[str, Any], ...],
) -> str | None:
    proposal_owners = tuple(
        str(proposal.get("owner") or "").strip()
        for proposal in proposals
        if str(proposal.get("owner") or "").strip()
    )
    unique_proposal_owners = tuple(dict.fromkeys(proposal_owners))
    capability_owners = {
        str(summary.get("owner") or "").strip()
        for summary in state.capability_summaries
        if summary.get("id") == "db.memory.commit_update"
        and str(summary.get("owner") or "").strip()
    }
    if (
        len(unique_proposal_owners) == 1
        and unique_proposal_owners[0] in capability_owners
    ):
        return unique_proposal_owners[0]
    return _single_capability_owner_for_state(state, "db.memory.commit_update")


def _runtime_memory_commit_action_id(
    proposals: tuple[dict[str, Any], ...],
    current_action_ids: set[str],
) -> str:
    if len(proposals) == 1:
        seed: Mapping[str, Any] = {
            "proposal_evidence_id": _summary_id(proposals[0]),
            "proposal_fingerprint": proposals[0].get("proposal_fingerprint"),
        }
    else:
        seed = {
            "candidate_ids": [_summary_id(proposal) for proposal in proposals],
        }
    action_id = f"runtime_memory_commit_{_stable_hash(seed)[:12]}"
    if action_id not in current_action_ids:
        return action_id
    return (
        f"{action_id}_{_stable_hash({'existing_ids': sorted(current_action_ids)})[:8]}"
    )


def _resolve_memory_proposal_for_action(
    action: DbPlannerAction,
    state: DbLoopState,
) -> tuple[dict[str, Any] | None, str | None]:
    proposal_id = action.input.get("proposal_evidence_id")
    summaries = tuple(
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "db.memory.proposal"
        and summary.get("accepted") is True
    )
    if proposal_id is not None:
        proposal_id = str(proposal_id).strip()
        if not proposal_id:
            return None, "missing_proposal_evidence_id"
        matches = tuple(
            summary for summary in summaries if summary.get("id") == proposal_id
        )
        if len(matches) > 1:
            return None, f"ambiguous_memory_proposal:{proposal_id}"
        if not matches:
            return None, f"memory_proposal_not_found:{proposal_id}"
        return dict(matches[0]), None
    if not summaries:
        return None, "missing_accepted_memory_proposal"
    if len(summaries) > 1:
        return None, "ambiguous_memory_proposal"
    return dict(summaries[-1]), None


def _relationship_paths_task_input(
    action: DbPlannerAction,
    state: DbLoopState,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    base_input = {
        key: value
        for key, value in _task_input_for_action(action).items()
        if key
        not in {
            "from",
            "from_assets",
            "source",
            "source_assets",
            "to",
            "to_assets",
            "target",
            "target_assets",
            "assets",
            "tables",
        }
    }
    from_assets = _first_string_list_from_mappings(
        (action.input, action.metadata),
        "from_assets",
        "from",
        "source_assets",
        "source",
    )
    to_assets = _first_string_list_from_mappings(
        (action.input, action.metadata),
        "to_assets",
        "to",
        "target_assets",
        "target",
    )
    if not from_assets and not to_assets:
        paired_assets = _first_string_list_from_mappings(
            (action.input, action.metadata),
            "assets",
            "tables",
        )
        if len(paired_assets) >= 2:
            from_assets = paired_assets[:1]
            to_assets = paired_assets[1:]
    if from_assets and not to_assets and len(from_assets) >= 2:
        to_assets = from_assets[1:]
        from_assets = from_assets[:1]
    fallback_from, fallback_to = _relationship_assets_from_state(state)
    if not from_assets:
        from_assets = fallback_from
    if not to_assets:
        to_assets = fallback_to
    errors: list[str] = []
    if not from_assets:
        errors.append("missing_from_assets")
    if not to_assets:
        errors.append("missing_to_assets")
    if errors:
        return {}, tuple(errors)
    return {
        **base_input,
        "from_assets": from_assets,
        "to_assets": to_assets,
    }, ()


def _relationship_assets_from_state(
    state: DbLoopState,
) -> tuple[list[str], list[str]]:
    scoped = _string_list(state.source_scope) or _string_list(
        state.normalized_user_request.get("source_scope")
    )
    if len(scoped) >= 2:
        return scoped[:1], scoped[1:]
    session_tables = _session_context_tables(
        state.normalized_user_request.get("session_context")
    )
    if len(session_tables) >= 2:
        return session_tables[:1], session_tables[1:]
    return _relationship_assets_from_text(
        str(state.normalized_user_request.get("prompt") or "")
    )


def _session_context_tables(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    referents = value.get("referents")
    if not isinstance(referents, Mapping):
        return []
    return _string_list(referents.get("tables"))


def _relationship_assets_from_text(text: str) -> tuple[list[str], list[str]]:
    identifier = r"([A-Za-z_][A-Za-z0-9_]*)"
    patterns = (
        rf"\bjoin\s+(?:the\s+)?{identifier}\s+(?:table\s+)?"
        rf"(?:to|with|and)\s+(?:the\s+)?{identifier}\b",
        rf"\b(?:relationship|relationships|path|paths)\s+"
        rf"(?:between|from)\s+(?:the\s+)?{identifier}\s+"
        rf"(?:and|to)\s+(?:the\s+)?{identifier}\b",
        rf"\b{identifier}\s+(?:to|and)\s+{identifier}\s+"
        rf"(?:relationship|relationships|join|joins|path|paths)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            left, right = match.group(1).strip(), match.group(2).strip()
            if left and right:
                return [left], [right]
    return [], []


def _first_string_list_from_mappings(
    mappings: Iterable[Mapping[str, Any]],
    *keys: str,
) -> list[str]:
    for mapping in mappings:
        for key in keys:
            values = _string_list(mapping.get(key))
            if values:
                return values
    return []


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _resolve_sql_input_for_action(
    action: DbPlannerAction,
    state: DbLoopState,
    *,
    sql_operation: str,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    direct_sql = action.input.get("sql")
    has_direct_sql = isinstance(direct_sql, str) and bool(direct_sql.strip())
    has_plan_evidence_id = "plan_evidence_id" in action.input
    has_query_plan_ref = "query_plan_ref" in action.input

    if has_direct_sql and has_plan_evidence_id:
        plan_evidence_id = action.input.get("plan_evidence_id")
        if not isinstance(plan_evidence_id, str) or not plan_evidence_id.strip():
            return None, "missing_plan_evidence_id"
        resolved, error = _resolve_sql_from_plan_evidence_id(
            plan_evidence_id.strip(),
            state,
        )
        if error == f"plan_evidence_not_found:{plan_evidence_id.strip()}":
            resolved, error = _resolve_sql_from_validation_evidence_id(
                plan_evidence_id.strip(),
                state,
                sql_operation=sql_operation,
            )
        if error is not None or resolved is None:
            return None, error
        if not _sql_inputs_match(direct_sql, resolved.sql):
            return None, "ambiguous_sql_input"
        return resolved, None
    if has_direct_sql and has_query_plan_ref:
        query_plan_ref = action.input.get("query_plan_ref")
        if query_plan_ref != "latest_accepted_query_plan":
            return None, f"unsupported_query_plan_ref:{query_plan_ref}"
        resolved, error = _resolve_sql_from_latest_accepted_query_plan(state)
        if error == "missing_accepted_query_plan" and sql_operation == "write":
            return _resolve_matching_sql_from_latest_accepted_validation(
                direct_sql,
                state,
                sql_operation=sql_operation,
            )
        if error is not None or resolved is None:
            return None, error
        if not _sql_inputs_match(direct_sql, resolved.sql):
            return None, "ambiguous_sql_input"
        return resolved, None
    if has_plan_evidence_id and has_query_plan_ref:
        return None, "ambiguous_sql_input"

    if has_direct_sql:
        return (
            _ResolvedSqlInput(
                sql=direct_sql.strip(),
                provenance="direct",
            ),
            None,
        )

    if has_plan_evidence_id:
        plan_evidence_id = action.input.get("plan_evidence_id")
        if not isinstance(plan_evidence_id, str) or not plan_evidence_id.strip():
            return None, "missing_plan_evidence_id"
        resolved, error = _resolve_sql_from_plan_evidence_id(
            plan_evidence_id.strip(),
            state,
        )
        if error == f"plan_evidence_not_found:{plan_evidence_id.strip()}":
            return _resolve_sql_from_validation_evidence_id(
                plan_evidence_id.strip(),
                state,
                sql_operation=sql_operation,
            )
        return resolved, error

    if has_query_plan_ref:
        query_plan_ref = action.input.get("query_plan_ref")
        if query_plan_ref != "latest_accepted_query_plan":
            return None, f"unsupported_query_plan_ref:{query_plan_ref}"
        return _resolve_sql_from_latest_accepted_query_plan(state)

    return None, "missing_sql"


def _sql_inputs_match(left: str, right: str) -> bool:
    return _normalized_sql_input(left) == _normalized_sql_input(right)


def _normalized_sql_input(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()


def _resolve_sql_from_plan_evidence_id(
    plan_evidence_id: str,
    state: DbLoopState,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    accepted_matches = tuple(
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "query.plan.proposal"
        and summary.get("accepted") is True
        and summary.get("id") == plan_evidence_id
    )
    rejected_matches = tuple(
        summary
        for summary in state.rejected_evidence_summaries
        if summary.get("kind") == "query.plan.proposal"
        and summary.get("id") == plan_evidence_id
    )
    if rejected_matches and not accepted_matches:
        return None, f"rejected_plan_evidence:{plan_evidence_id}"
    if rejected_matches or len(accepted_matches) > 1:
        return None, f"ambiguous_plan_evidence:{plan_evidence_id}"
    if not accepted_matches:
        return None, f"plan_evidence_not_found:{plan_evidence_id}"
    return _resolved_sql_from_plan_summary(
        accepted_matches[0],
        state,
        provenance="plan_evidence_id",
    )


def _resolve_sql_from_latest_accepted_query_plan(
    state: DbLoopState,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    summaries = tuple(
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "query.plan.proposal"
        and summary.get("accepted") is True
    )
    for summary in reversed(summaries):
        sql = summary.get("sql")
        if summary.get("valid") is True and isinstance(sql, str) and sql.strip():
            return _resolved_sql_from_plan_summary(
                summary,
                state,
                provenance="latest_accepted_query_plan",
            )
    if summaries:
        return None, "query_plan_evidence_without_sql"
    return None, "missing_accepted_query_plan"


def _resolve_matching_sql_from_latest_accepted_validation(
    sql: str,
    state: DbLoopState,
    *,
    sql_operation: str,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    summaries = tuple(
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "sql.validation" and summary.get("accepted") is True
    )
    valid_summaries = tuple(
        summary for summary in summaries if summary.get("valid") is True
    )
    for summary in reversed(valid_summaries):
        operation = _optional_string(summary.get("operation"))
        if operation is not None and operation != sql_operation:
            continue
        validated_sql = summary.get("sql")
        if not isinstance(validated_sql, str) or not validated_sql.strip():
            continue
        if not _sql_inputs_match(sql, validated_sql):
            continue
        return _resolved_sql_from_validation_summary(
            summary,
            sql=sql,
            provenance="latest_accepted_sql_validation",
        )
    if valid_summaries:
        return None, "ambiguous_sql_input"
    if summaries:
        return None, "missing_valid_sql_validation"
    return None, "missing_accepted_query_plan"


def _resolve_sql_from_validation_evidence_id(
    evidence_id: str,
    state: DbLoopState,
    *,
    sql_operation: str,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    accepted_matches = tuple(
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "sql.validation"
        and summary.get("accepted") is True
        and summary.get("id") == evidence_id
    )
    rejected_matches = tuple(
        summary
        for summary in state.rejected_evidence_summaries
        if summary.get("kind") == "sql.validation" and summary.get("id") == evidence_id
    )
    if rejected_matches and not accepted_matches:
        return None, f"rejected_validation_evidence:{evidence_id}"
    if rejected_matches or len(accepted_matches) > 1:
        return None, f"ambiguous_validation_evidence:{evidence_id}"
    if not accepted_matches:
        return None, f"validation_evidence_not_found:{evidence_id}"
    summary = accepted_matches[0]
    if summary.get("valid") is not True:
        return None, f"invalid_validation_evidence:{evidence_id}"
    operation = _optional_string(summary.get("operation"))
    if operation is not None and operation != sql_operation:
        return None, f"validation_operation_mismatch:{evidence_id}"
    sql = summary.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return None, f"validation_evidence_without_sql:{evidence_id}"
    return _resolved_sql_from_validation_summary(
        summary,
        sql=sql,
        provenance="validation_evidence_id",
    )


def _resolved_sql_from_validation_summary(
    summary: Mapping[str, Any],
    *,
    sql: str,
    provenance: str,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    evidence_id = _optional_string(summary.get("id"))
    return (
        _ResolvedSqlInput(
            sql=sql.strip(),
            provenance=provenance,
            source_evidence_id=evidence_id,
            source_evidence_kind="sql.validation",
            source_evidence_owner=_optional_string(summary.get("owner")),
            source_task_id=_optional_string(summary.get("task_id")),
            source_payload_fingerprint=_optional_string(
                summary.get("payload_fingerprint")
            ),
        ),
        None,
    )


def _resolved_sql_from_plan_summary(
    summary: Mapping[str, Any],
    state: DbLoopState,
    *,
    provenance: str,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    sql = summary.get("sql")
    evidence_id = _optional_string(summary.get("id"))
    if not isinstance(sql, str) or not sql.strip():
        return None, f"plan_evidence_without_sql:{evidence_id or 'unknown'}"
    evidence_owner = _optional_string(summary.get("owner"))
    task_id = _optional_string(summary.get("task_id"))
    payload_fingerprint = _optional_string(summary.get("payload_fingerprint"))
    dependency = TaskDependency(
        kind="evidence",
        evidence_kind="query.plan.proposal",
        evidence_id=evidence_id,
        evidence_owner=evidence_owner,
        producer_task_id=task_id,
        evidence_accepted=True,
        operation_id=state.operation_id,
        payload_fingerprint=payload_fingerprint,
        metadata={
            "sql_resolution": True,
            "provenance": provenance,
        },
    )
    return (
        _ResolvedSqlInput(
            sql=sql.strip(),
            provenance=provenance,
            query_plan_dependency=dependency,
            source_evidence_id=evidence_id,
            source_evidence_kind="query.plan.proposal",
            source_evidence_owner=evidence_owner,
            source_task_id=task_id,
            source_payload_fingerprint=payload_fingerprint,
        ),
        None,
    )


def _sql_provenance_metadata(resolved: _ResolvedSqlInput) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provenance": resolved.provenance,
        "sql_fingerprint": _stable_hash({"sql": resolved.sql}),
    }
    if resolved.source_evidence_id is not None:
        metadata["source_evidence_id"] = resolved.source_evidence_id
    if resolved.source_evidence_kind is not None:
        metadata["source_evidence_kind"] = resolved.source_evidence_kind
    if resolved.source_evidence_owner is not None:
        metadata["source_evidence_owner"] = resolved.source_evidence_owner
    if resolved.source_task_id is not None:
        metadata["source_task_id"] = resolved.source_task_id
    if resolved.source_payload_fingerprint is not None:
        metadata["source_payload_fingerprint"] = resolved.source_payload_fingerprint
    return metadata


def _monitor_action_output_evidence(
    action: DbPlannerAction,
) -> tuple[str, str | None]:
    if action.kind is DbPlannerActionKind.PLAN_MONITOR_CREATE:
        return "monitor.proposal", None
    if action.kind is DbPlannerActionKind.COMMIT_MONITOR_CREATE:
        return "monitor.definition", None
    if action.kind is DbPlannerActionKind.PLAN_MONITOR_LIFECYCLE:
        return "monitor.proposal", None
    if action.kind is DbPlannerActionKind.COMMIT_MONITOR_LIFECYCLE:
        lifecycle_action = _monitor_lifecycle_action_label(
            action.input.get("action") or action.input.get("operation_type")
        )
        if lifecycle_action is None:
            return "", "missing_monitor_lifecycle_action"
        return _monitor_lifecycle_evidence_kind(lifecycle_action), None
    if action.kind is DbPlannerActionKind.READ_MONITOR_STATE:
        read_kind = str(action.input.get("read_kind") or "list").lower()
        evidence_kind = {
            "list": "monitor.listing",
            "inspect": "monitor.inspection",
            "explain_run": "monitor.run_summary",
            "approvals": "monitor.approval_state",
        }.get(read_kind)
        if evidence_kind is None:
            return "", f"unsupported_monitor_read_kind:{read_kind}"
        return evidence_kind, None
    if action.kind is DbPlannerActionKind.RESOLVE_MONITOR_APPROVAL:
        return "monitor.approval_resolution", None
    return "", "unsupported_monitor_action_kind"


def _monitor_lifecycle_action_label(value: Any) -> str | None:
    normalized = str(value or "").removeprefix("monitor.").replace("_", ".").lower()
    if normalized in {"update", "pause", "resume", "delete", "disable"}:
        return normalized
    if normalized == "disabled":
        return "disable"
    return None


def _monitor_lifecycle_evidence_kind(action: str) -> str:
    if action == "delete":
        return "monitor.deleted"
    if action == "disable":
        return "monitor.disabled"
    if action == "pause":
        return "monitor.paused"
    if action == "resume":
        return "monitor.resumed"
    return "monitor.state_update"


def _action_metadata(
    action: DbPlannerAction,
    decision_fingerprint: str,
) -> dict[str, Any]:
    return {
        **action.metadata,
        "planner_action_id": action.action_id,
        "planner_action_kind": action.kind.value,
        "planner_decision_fingerprint": decision_fingerprint,
    }


def _action_error(action: DbPlannerAction, error: str) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "kind": action.kind.value,
        "error": error,
    }


def _continuation_action_error(
    action: DbPlannerAction,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **_action_error(
            action,
            str(diagnostic.get("error") or "continuation_resolution_blocked"),
        ),
        "continuation": dict(diagnostic),
    }


def _capability_selection(
    capability: Any,
    action: DbPlannerAction,
    *,
    output_evidence: tuple[str, ...] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "access": capability.access.value,
        "risk": capability.risk.value,
        "output_evidence": (
            sorted(output_evidence)
            if output_evidence is not None
            else sorted(capability.output_evidence)
        ),
        "action_id": action.action_id,
        "reason": reason or action.kind.value,
    }


def _capability_summary(capability: Any) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "description": capability.description,
        "access": capability.access.value,
        "risk": capability.risk.value,
        "output_evidence": sorted(capability.output_evidence),
        "runtime_only": capability.runtime_only,
    }


def _task_summary(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "capability_id": task.capability_id,
        "executor_id": task.executor_id,
        "status": task.status.value,
        "metadata": {
            key: task.metadata.get(key)
            for key in (
                "owner",
                "reason",
                "sequence",
                "planner_action_id",
                "planner_action_kind",
            )
            if key in task.metadata
        },
    }


def _task_ref(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "capability_id": task.capability_id,
        "status": task.status.value,
    }


def _approval_task_id(approval: Any) -> str | None:
    direct = getattr(approval, "task_id", None)
    if direct:
        return str(direct)
    for source in (
        getattr(approval, "metadata", None),
        getattr(approval, "proposed_action", None),
    ):
        if isinstance(source, Mapping) and source.get("task_id"):
            return str(source["task_id"])
    return None


def _evidence_summary(evidence: Evidence) -> dict[str, Any]:
    summary = {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "accepted": evidence.accepted,
        "task_id": evidence.task_id,
    }
    sql = _sql_from_evidence_payload(evidence.payload)
    if sql:
        summary["sql"] = sql
    if isinstance(evidence.payload, dict):
        if evidence.kind in {
            "sql.validation",
            "query.plan.validation",
            "query.plan.proposal",
        }:
            if "valid" in evidence.payload:
                summary["valid"] = evidence.payload.get("valid") is True
            validation_facts = _safe_validation_items(
                evidence.payload.get("validation_facts")
            )
            validation_warnings = _safe_validation_items(
                evidence.payload.get("warnings")
            )
            validation_errors = _safe_validation_items(evidence.payload.get("errors"))
            if validation_facts:
                summary["validation_facts"] = validation_facts
            if validation_warnings:
                summary["warnings"] = validation_warnings
            if validation_errors:
                summary["validation_warnings"] = validation_errors
        if evidence.kind == "schema.column_value_hint":
            hints = _safe_column_value_hint_summaries(evidence.payload.get("hints"))
            if hints:
                summary["hints"] = hints
        if evidence.kind == "planning.context":
            diagnostics = evidence.payload.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                repair = diagnostics.get("validation_grounding_repair")
                if isinstance(repair, Mapping):
                    fingerprint = str(repair.get("fingerprint") or "").strip()
                    if fingerprint:
                        summary["validation_grounding_fingerprint"] = fingerprint
                    target_refs = [
                        str(item).strip()
                        for item in repair.get("target_refs", ())
                        if str(item).strip()
                    ]
                    if target_refs:
                        summary["validation_grounding_target_refs"] = target_refs
        if evidence.kind == "db.memory.proposal":
            proposal_fingerprint = evidence.payload.get("proposal_fingerprint")
            if isinstance(proposal_fingerprint, str) and proposal_fingerprint.strip():
                summary["proposal_fingerprint"] = proposal_fingerprint.strip()
        if evidence.kind == "db.memory.definition":
            proposal_evidence_id = evidence.payload.get(
                "proposal_evidence_id"
            ) or evidence.metadata.get("proposal_evidence_id")
            if proposal_evidence_id:
                summary["proposal_evidence_id"] = str(proposal_evidence_id)
            proposal_fingerprint = evidence.payload.get(
                "proposal_fingerprint"
            ) or evidence.metadata.get("proposal_fingerprint")
            if isinstance(proposal_fingerprint, str) and proposal_fingerprint.strip():
                summary["proposal_fingerprint"] = proposal_fingerprint.strip()
        operation = _optional_string(evidence.payload.get("operation"))
        if operation is not None:
            summary["operation"] = operation
    payload_fingerprint = evidence.metadata.get("payload_fingerprint")
    if payload_fingerprint:
        summary["payload_fingerprint"] = str(payload_fingerprint)
    return summary


def _safe_validation_items(value: Any) -> list[Any]:
    items: list[Any] = []
    for item in _safe_iterable(value):
        if isinstance(item, Mapping):
            safe = {
                key: item[key]
                for key in (
                    "kind",
                    "table",
                    "table_name",
                    "column",
                    "column_name",
                    "operator",
                    "literal",
                    "value",
                    "filter_literal",
                    "candidates",
                    "source",
                    "reason",
                )
                if key in item
            }
            if safe:
                items.append(safe)
        elif isinstance(item, str) and item.strip():
            items.append(item.strip())
    return _dedupe_json_values(items)


def _safe_column_value_hint_summaries(value: Any) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for item in _safe_iterable(value):
        if not isinstance(item, Mapping):
            continue
        if not planner_eligible_column_value_hint(dict(item)):
            continue
        table = str(item.get("table") or "").strip()
        column = str(item.get("column") or "").strip()
        if not table or not column:
            continue
        hint: dict[str, Any] = {
            "table": table,
            "column": column,
        }
        observed_values = []
        for observed in _safe_iterable(item.get("observed_values")):
            if isinstance(observed, Mapping):
                if observed.get("value") is not None:
                    observed_values.append({"value": observed.get("value")})
            elif observed is not None:
                observed_values.append({"value": observed})
        if observed_values:
            hint["observed_values"] = observed_values[:25]
        candidate_mapping = item.get("candidate_mapping")
        if isinstance(candidate_mapping, Mapping):
            hint["candidate_mapping"] = dict(candidate_mapping)
        hints.append(hint)
    return _dedupe_dicts(hints, keys=("table", "column"))


def _evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "accepted": evidence.accepted,
    }


def _sql_from_evidence_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("sql", "selected_sql"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    structured_plan = payload.get("structured_plan")
    if isinstance(structured_plan, dict):
        selected_sql = structured_plan.get("selected_sql")
        if isinstance(selected_sql, str) and selected_sql.strip():
            return selected_sql.strip()
    return None


def _fallback_contract_for_operation(
    operation: Operation,
    *,
    limits: DbLimits,
) -> DbOperationContract:
    return DbOperationContract(
        operation_type=operation.operation_type,
        required_evidence=tuple(sorted(operation.required_evidence)),
        access=AccessMode.NONE,
        limits=limits,
        policy_ids=(),
        metadata={"source": "operation_state"},
    )


def _contract_from_latest_snapshot(
    operation: Operation,
    fallback: DbOperationContract,
) -> DbOperationContract:
    context = operation.metadata.get("resume_context")
    context = context if isinstance(context, dict) else {}
    snapshot = (
        operation.metadata.get("latest_compiled_contract_snapshot")
        or context.get("latest_compiled_contract_snapshot")
        or context.get("contract")
    )
    if not isinstance(snapshot, dict):
        return fallback
    limits = snapshot.get("limits")
    limits = limits if isinstance(limits, dict) else {}
    metadata = snapshot.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    try:
        access = AccessMode(str(snapshot.get("access") or fallback.access.value))
    except ValueError:
        access = fallback.access
    try:
        db_limits = DbLimits(**limits) if limits else fallback.limits
    except (TypeError, ValueError):
        db_limits = fallback.limits
    return DbOperationContract(
        operation_type=str(snapshot.get("operation_type") or fallback.operation_type),
        required_evidence=_string_tuple(snapshot.get("required_evidence")),
        access=access,
        limits=db_limits,
        policy_ids=_string_tuple(snapshot.get("policy_ids")),
        metadata=metadata,
    )


def _fallback_intent_for_operation(
    operation: Operation,
    contract: DbOperationContract,
) -> DbIntent:
    context = operation.metadata.get("resume_context")
    context = context if isinstance(context, dict) else {}
    intent_context = context.get("intent")
    intent_context = intent_context if isinstance(intent_context, dict) else {}
    kind_value = str(intent_context.get("kind") or operation.operation_type)
    try:
        kind = DbIntentKind(kind_value)
    except ValueError:
        kind = DbIntentKind.CONVERSATIONAL
    return DbIntent(
        kind=kind,
        confidence=1.0,
        access=contract.access,
        evidence_mode="planner_loop",
        diagnostics={
            "source": "operation_state",
            "operation_type": operation.operation_type,
        },
    )


def _intent_from_contract(
    contract: DbOperationContract,
    fallback: DbIntent,
) -> DbIntent:
    intent_metadata = contract.metadata.get("planner_intent")
    intent_metadata = intent_metadata if isinstance(intent_metadata, dict) else {}
    operation_type = str(
        intent_metadata.get("operation_type") or contract.operation_type
    )
    try:
        kind = DbIntentKind(operation_type)
    except ValueError:
        kind = fallback.kind
    return DbIntent(
        kind=kind,
        confidence=1.0,
        access=contract.access,
        evidence_mode="planner_loop",
        diagnostics={
            "source": "planner_compiled_contract",
            "operation_type": operation_type,
            "planner_intent": intent_metadata,
        },
    )


def _intent_summary(intent: DbIntent) -> dict[str, Any]:
    return {
        "kind": intent.kind.value,
        "access": intent.access.value,
        "evidence_mode": intent.evidence_mode,
        "diagnostics": intent.diagnostics,
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _column_value_scope_for_action(
    action: DbPlannerAction,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tables = _first_action_string_list(
        action,
        "tables",
        "table",
        "target_table",
        "target_tables",
        "target",
    )
    columns = _first_action_string_list(
        action,
        "columns",
        "column",
        "target_column",
        "target_columns",
        "field",
    )
    sql_tables, sql_columns = _column_value_scope_from_sql(action.input.get("sql"))
    if not tables:
        tables = list(sql_tables)
    if not columns:
        columns = list(sql_columns)
    normalized_tables: list[str] = []
    normalized_columns: list[str] = list(columns)
    for table in tables:
        if "." in table:
            table_name, column_name = table.split(".", 1)
            if table_name.strip():
                normalized_tables.append(table_name.strip())
            if column_name.strip() and not normalized_columns:
                normalized_columns.append(column_name.strip())
            continue
        normalized_tables.append(table)
    return (
        tuple(_ordered_unique_strings(normalized_tables)),
        tuple(_ordered_unique_strings(normalized_columns)),
    )


def _column_value_scope_from_sql(sql: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(sql, str) or not sql.strip():
        return (), ()
    identifier = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)'
    qualified = rf"{identifier}(?:\s*\.\s*{identifier})*"
    match = re.search(
        rf"(?is)^\s*select\s+(?:distinct\s+)?({qualified})\s+from\s+({qualified})\b",
        sql,
    )
    if match is None:
        return (), ()
    column = _clean_sql_identifier(match.group(1))
    table = _clean_sql_identifier(match.group(2))
    if "." in column:
        column = column.split(".")[-1]
    if not table or not column or column == "*":
        return (), ()
    return (table,), (column,)


def _clean_sql_identifier(value: str) -> str:
    parts = []
    for part in re.split(r"\s*\.\s*", value.strip()):
        part = part.strip()
        if len(part) >= 2 and (
            (part[0] == part[-1] == '"')
            or (part[0] == part[-1] == "`")
            or (part[0] == "[" and part[-1] == "]")
        ):
            part = part[1:-1]
        if part:
            parts.append(part)
    return ".".join(parts)


def _first_action_string_list(
    action: DbPlannerAction,
    *keys: str,
) -> list[str]:
    for source in (action.input, action.metadata):
        for key in keys:
            values = _safe_string_list(source.get(key))
            if values:
                return values
    return []


def _safe_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _ordered_unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _memory_context_for_state(
    runtime: Any,
    operation: Operation,
    accepted: tuple[Evidence, ...],
) -> dict[str, Any]:
    memory_config = db_memory_options_from_runtime_metadata(runtime.config.metadata)
    if not memory_config:
        return {"enabled": False}
    prompt = str(operation.request.get("prompt") or "")
    schema = _latest_schema_payload(accepted)
    decision = db_memory_planning_recall_decision(
        prompt=prompt,
        intent_kind=_memory_recall_intent_kind(operation),
        schema=schema,
        memory_config=memory_config,
    )
    return {
        "enabled": bool(memory_config.get("enabled")),
        "source_identity": memory_config.get("source_identity"),
        "retrieval_mode": memory_config.get("retrieval_mode") or "structured",
        "limit": int(memory_config.get("limit") or 3),
        "char_budget": int(memory_config.get("char_budget") or 800),
        "score_threshold": _float_option(memory_config, "score_threshold", 0.45),
        "recall": memory_config.get("recall") or "auto",
        "recall_decision": decision,
        "has_recall_evidence": any(
            item.kind == "memory.semantic.recall" and item.accepted for item in accepted
        ),
    }


def _latest_schema_payload(accepted: tuple[Evidence, ...]) -> dict[str, Any]:
    for evidence in reversed(accepted):
        if evidence.kind == "schema.asset_profile" and isinstance(
            evidence.payload, dict
        ):
            return dict(evidence.payload)
    return {}


def _memory_recall_intent_kind(operation: Operation) -> str:
    mode = operation.request.get("mode")
    if isinstance(mode, str) and mode.strip() == "memory.update":
        return "memory.update"
    return "data.query"


def _state_should_recall_memory_for_planning(state: DbLoopState) -> bool:
    memory_context = state.memory_context or {}
    decision = memory_context.get("recall_decision")
    if not isinstance(decision, Mapping) or decision.get("recall") is not True:
        return False
    if memory_context.get("has_recall_evidence") is True:
        return False
    return not _state_has_accepted_evidence(state, "memory.semantic.recall")


def _memory_recall_task_input(state: DbLoopState) -> dict[str, Any]:
    memory_context = state.memory_context or {}
    decision = memory_context.get("recall_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    limit = int(memory_context.get("limit") or 3)
    return {
        "query": str(
            decision.get("query") or state.normalized_user_request.get("prompt") or ""
        ),
        "category": "db_semantics",
        "limit": max(limit * 3, limit),
        "score_threshold": _float_option(memory_context, "score_threshold", 0.45),
        "retrieval_mode": str(memory_context.get("retrieval_mode") or "structured"),
        "source_identity": memory_context.get("source_identity"),
    }


def _column_value_hint_task_input(
    action: DbPlannerAction,
    state: DbLoopState,
) -> dict[str, Any]:
    prompt = _planning_value_hint_prompt(action, state)
    task_input: dict[str, Any] = {
        "prompt": prompt,
        "query": prompt,
        "limit": 8,
    }
    for key in (
        "source_owner",
        "db_owner",
        "connector_owner",
        "source_capability_owner",
    ):
        value = action.input.get(key) or action.metadata.get(key)
        if value:
            task_input[key] = str(value)
            break
    tables, columns = _column_value_scope_for_action(action)
    if tables:
        task_input["tables"] = list(tables)
    if columns:
        task_input["columns"] = list(columns)
    validation_facts, validation_warnings = _state_value_grounding_validation_inputs(
        state
    )
    if validation_facts:
        task_input["validation_facts"] = validation_facts
    if validation_warnings:
        task_input["validation_warnings"] = validation_warnings
    return task_input


def _value_grounding_plan_task_input(
    action: DbPlannerAction,
    state: DbLoopState,
) -> dict[str, Any]:
    prompt = _planning_value_hint_prompt(action, state)
    profile_budget = int(
        _first_present(
            action.input,
            action.metadata,
            keys=("profile_budget", "max_profile_budget"),
            default=4,
        )
    )
    task_input: dict[str, Any] = {
        "prompt": prompt,
        "query": prompt,
        "profile_budget": profile_budget,
        "max_profile_budget": profile_budget,
    }
    for key in (
        "source_owner",
        "db_owner",
        "connector_owner",
        "source_capability_owner",
    ):
        value = action.input.get(key) or action.metadata.get(key)
        if value:
            task_input[key] = str(value)
            break
    tables, columns = _column_value_scope_for_action(action)
    if tables:
        task_input["tables"] = list(tables)
    if columns:
        task_input["columns"] = list(columns)
    profile_pairs = _action_profile_pairs(action, tables=tables, columns=columns)
    if profile_pairs:
        task_input["profile_pairs"] = profile_pairs
    targets = _action_value_grounding_targets(action)
    if targets:
        task_input["targets"] = targets
    validation_facts, validation_warnings = _state_value_grounding_validation_inputs(
        state
    )
    if validation_facts:
        task_input["validation_facts"] = validation_facts
    if validation_warnings:
        task_input["validation_warnings"] = validation_warnings
    session_query_scopes = _state_session_query_scopes(state)
    if session_query_scopes:
        task_input["session_query_scopes"] = session_query_scopes
    if state.safety_frame:
        task_input["policy_frame"] = dict(state.safety_frame)
    return task_input


def _action_profile_pairs(
    action: DbPlannerAction,
    *,
    tables: tuple[str, ...],
    columns: tuple[str, ...],
) -> list[dict[str, str]]:
    explicit = _safe_profile_pair_list(
        action.input.get("profile_pairs") or action.metadata.get("profile_pairs")
    )
    if explicit:
        return explicit
    if tables and columns:
        return [
            {"table": table, "column": column} for table in tables for column in columns
        ]
    return []


def _action_value_grounding_targets(action: DbPlannerAction) -> list[dict[str, Any]]:
    targets = _safe_target_list(
        action.input.get("targets") or action.metadata.get("targets")
    )
    if targets:
        return targets
    target = action.input.get("target") or action.metadata.get("target")
    if isinstance(target, str) and target.strip():
        table, column = _split_column_ref(target)
        if table and column:
            return [{"table": table, "column": column}]
    return []


def _safe_profile_pair_list(value: Any) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for item in _safe_iterable(value):
        table = ""
        column = ""
        if isinstance(item, Mapping):
            table = str(item.get("table") or "").strip()
            column = str(item.get("column") or "").strip()
        elif isinstance(item, str):
            table, column = _split_column_ref(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            table = str(item[0]).strip()
            column = str(item[1]).strip()
        if table and column:
            pairs.append({"table": table, "column": column})
    return _dedupe_dicts(pairs, keys=("table", "column"))


def _safe_target_list(value: Any) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in _safe_iterable(value):
        if isinstance(item, Mapping):
            table = str(item.get("table") or item.get("table_name") or "").strip()
            column = str(item.get("column") or item.get("column_name") or "").strip()
            ref = str(item.get("target") or item.get("ref") or "").strip()
            if ref and (not table or not column):
                ref_table, ref_column = _split_column_ref(ref)
                table = table or ref_table
                column = column or ref_column
            if table and column:
                target = dict(item)
                target["table"] = table
                target["column"] = column
                targets.append(target)
        elif isinstance(item, str):
            table, column = _split_column_ref(item)
            if table and column:
                targets.append({"table": table, "column": column})
    return _dedupe_dicts(targets, keys=("table", "column"))


def _state_value_grounding_validation_inputs(
    state: DbLoopState,
) -> tuple[list[Any], list[Any]]:
    facts: list[Any] = []
    warnings: list[Any] = []
    for summary in state.validation_summaries:
        facts.extend(_safe_iterable(summary.get("validation_facts")))
        warnings.extend(_safe_iterable(summary.get("warnings")))
        warnings.extend(_safe_iterable(summary.get("validation_warnings")))
    return _dedupe_json_values(facts), _dedupe_json_values(warnings)


def _validation_grounding_repair_context(
    state: DbLoopState,
) -> dict[str, Any]:
    validation_facts, validation_warnings = _state_value_grounding_validation_inputs(
        state
    )
    targets = _validation_value_grounding_targets(
        (*validation_facts, *validation_warnings)
    )
    if not targets:
        return {}
    target_refs = [f"{target['table']}.{target['column']}" for target in targets]
    satisfied_refs = _state_column_value_hint_refs(state)
    missing_refs = [ref for ref in target_refs if ref.lower() not in satisfied_refs]
    fingerprint = _stable_hash(
        {
            "operation_id": state.operation_id,
            "targets": targets,
        }
    )
    return {
        "attempted": True,
        "fingerprint": fingerprint,
        "target_refs": target_refs,
        "targets": targets,
        "missing_target_refs": missing_refs,
        "satisfied_target_refs": [
            ref for ref in target_refs if ref.lower() in satisfied_refs
        ],
    }


def _validation_value_grounding_targets(values: Iterable[Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, Mapping):
            kind = str(item.get("kind") or "")
            if kind and kind not in {
                "filter_literal_requires_grounding",
                "unobserved_filter_literal",
                "ambiguous_literal_column",
            }:
                continue
            table = str(item.get("table") or item.get("table_name") or "").strip()
            column = str(item.get("column") or item.get("column_name") or "").strip()
            ref = str(item.get("target") or item.get("ref") or "").strip()
            if ref and (not table or not column):
                ref_table, ref_column = _split_column_ref(ref)
                table = table or ref_table
                column = column or ref_column
            literal = item.get("literal")
            if literal is None:
                literal = item.get("value", item.get("filter_literal"))
            if table and column:
                target = {
                    "table": table,
                    "column": column,
                }
                if literal is not None:
                    target["literal"] = str(literal)
                if kind:
                    target["kind"] = kind
                targets.append(target)
            continue
        if isinstance(item, str):
            parsed = _validation_value_grounding_target_from_warning(item)
            if parsed:
                targets.append(parsed)
    return _dedupe_dicts(targets, keys=("table", "column", "literal"))


def _validation_value_grounding_target_from_warning(
    value: str,
) -> dict[str, Any] | None:
    match = re.search(
        r"(filter_literal_requires_grounding|unobserved_filter_literal|"
        r"ambiguous_literal_column):"
        r"\s*([^=\s;]+)\s*=\s*([^;]+)",
        str(value),
    )
    if match is None:
        return None
    table, column = _split_column_ref(match.group(2))
    literal = match.group(3).strip().strip("'\"")
    if not table or not column or not literal:
        return None
    return {
        "kind": match.group(1),
        "table": table,
        "column": column,
        "literal": literal,
    }


def _state_column_value_hint_refs(state: DbLoopState) -> set[str]:
    refs: set[str] = set()
    for summary in state.accepted_evidence_summaries:
        if summary.get("kind") != "schema.column_value_hint":
            continue
        for hint in _safe_iterable(summary.get("hints")):
            if not isinstance(hint, Mapping):
                continue
            table = str(hint.get("table") or "").strip()
            column = str(hint.get("column") or "").strip()
            if table and column:
                refs.add(f"{table}.{column}".lower())
    return refs


def _state_session_query_scopes(state: DbLoopState) -> list[dict[str, Any]]:
    session_context = state.normalized_user_request.get("session_context")
    if not isinstance(session_context, Mapping):
        return []
    scopes = []
    for scope in _safe_iterable(session_context.get("query_scopes")):
        if isinstance(scope, Mapping):
            scopes.append(dict(scope))
    return _dedupe_json_values(scopes)


def _safe_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _first_present(
    *sources: Mapping[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for source in sources:
        for key in keys:
            if key in source and source.get(key) is not None:
                return source[key]
    return default


def _split_column_ref(value: Any) -> tuple[str, str]:
    parts = [
        part.strip().strip('"`[]')
        for part in str(value or "").split(".")
        if part.strip().strip('"`[]')
    ]
    if len(parts) >= 2:
        return ".".join(parts[:-1]), parts[-1]
    return "", ""


def _dedupe_dicts(
    values: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, Any]] = []
    for value in values:
        key = tuple(str(value.get(item) or "") for item in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _dedupe_json_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _float_option(value: Mapping[str, Any], key: str, default: float) -> float:
    raw = value.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _state_has_accepted_evidence(state: DbLoopState, kind: str) -> bool:
    return any(
        item.get("kind") == kind and item.get("accepted", True) is True
        for item in state.accepted_evidence_summaries
    )


def _state_should_plan_value_grounding_for_planning(
    state: DbLoopState,
    action: DbPlannerAction,
) -> bool:
    if not _state_allows_read_profile(state):
        return False
    validation_facts, validation_warnings = _state_value_grounding_validation_inputs(
        state
    )
    validation_targets = _validation_value_grounding_targets(
        (*validation_facts, *validation_warnings)
    )
    if validation_targets:
        satisfied_refs = _state_column_value_hint_refs(state)
        return any(
            f"{target['table']}.{target['column']}".lower() not in satisfied_refs
            for target in validation_targets
        )
    if _state_has_accepted_evidence(state, "schema.column_value_hint"):
        return False
    prompt = _planning_value_hint_prompt(action, state)
    if prompt:
        return True
    return bool(
        validation_facts
        or validation_warnings
        or _state_session_query_scopes(state)
        or _action_value_grounding_targets(action)
    )


def _planning_value_hint_prompt(
    action: DbPlannerAction,
    state: DbLoopState,
) -> str:
    values = (
        action.input.get("prompt"),
        action.input.get("query"),
        action.input.get("goal"),
        state.normalized_user_request.get("prompt"),
        state.normalized_user_request.get("query"),
    )
    return " ".join(str(value).strip() for value in values if str(value or "").strip())


def _latest_accepted_evidence_summary(
    state: DbLoopState,
    kind: str,
) -> dict[str, Any] | None:
    for item in reversed(state.accepted_evidence_summaries):
        if item.get("kind") == kind and item.get("accepted", True) is True:
            return dict(item)
    return None


def _repair_query_plan_task_input_and_dependencies(
    action: DbPlannerAction,
    state: DbLoopState,
    task_input: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[TaskDependency, ...], list[dict[str, Any]]]:
    repair_input = dict(task_input)
    repair_input.pop("query_plan_ref", None)
    plan_alias = repair_input.pop("plan_evidence_id", None)
    if "prior_plan_evidence_id" not in repair_input and str(plan_alias or "").strip():
        repair_input["prior_plan_evidence_id"] = str(plan_alias).strip()

    planning_context_id = str(
        repair_input.get("planning_context_evidence_id") or ""
    ).strip()
    planning_context = _repair_evidence_summary(
        state,
        "planning.context",
        evidence_id=planning_context_id or None,
        accepted=True,
    )
    if planning_context is None and not planning_context_id:
        planning_context = _latest_accepted_evidence_summary(
            state,
            "planning.context",
        )
    if planning_context is None:
        return {}, (), [_action_error(action, "missing_repair_planning_context")]
    repair_input["planning_context_evidence_id"] = planning_context["id"]

    prior_plan_id = str(repair_input.get("prior_plan_evidence_id") or "").strip()
    prior_plan = _repair_evidence_summary(
        state,
        "query.plan.proposal",
        evidence_id=prior_plan_id or None,
        accepted=True,
        valid=True,
        require_sql=True,
    )
    if prior_plan is None and not prior_plan_id:
        prior_plan = _latest_repair_evidence_summary(
            state,
            "query.plan.proposal",
            accepted=True,
            valid=True,
            require_sql=True,
        )
    if prior_plan is None:
        return {}, (), [_action_error(action, "missing_repair_prior_plan")]
    repair_input["prior_plan_evidence_id"] = prior_plan["id"]

    failure_id = str(repair_input.get("failure_evidence_id") or "").strip()
    failure_matches = _repair_failure_evidence_summaries(
        state,
        evidence_id=failure_id or None,
    )
    if failure_id:
        if len(failure_matches) > 1:
            return {}, (), [_action_error(action, "ambiguous_repair_failure_evidence")]
        failure = failure_matches[0] if failure_matches else None
    else:
        failure = _latest_repair_evidence_summary(
            state,
            "query.plan.validation",
            accepted=False,
            valid=False,
        )
        if failure is None:
            failure = _latest_repair_evidence_summary(
                state,
                "sql.validation",
                accepted=False,
                valid=False,
            )
    if failure is None:
        return {}, (), [_action_error(action, "missing_repair_failure_evidence")]
    repair_input["failure_evidence_id"] = failure["id"]

    dependencies = (
        _repair_input_dependency(
            state,
            planning_context,
            input_name="planning_context_evidence_id",
        ),
        _repair_input_dependency(
            state,
            prior_plan,
            input_name="prior_plan_evidence_id",
        ),
        _repair_input_dependency(
            state,
            failure,
            input_name="failure_evidence_id",
        ),
    )
    return repair_input, dependencies, []


def _latest_repair_evidence_summary(
    state: DbLoopState,
    kind: str,
    *,
    accepted: bool,
    valid: bool | None = None,
    require_sql: bool = False,
) -> dict[str, Any] | None:
    summaries = (
        state.accepted_evidence_summaries
        if accepted
        else state.rejected_evidence_summaries
    )
    for item in reversed(summaries):
        if _repair_summary_matches(
            item,
            kind,
            accepted=accepted,
            valid=valid,
            require_sql=require_sql,
        ):
            return dict(item)
    return None


def _repair_evidence_summary(
    state: DbLoopState,
    kind: str,
    *,
    evidence_id: str | None,
    accepted: bool,
    valid: bool | None = None,
    require_sql: bool = False,
) -> dict[str, Any] | None:
    if evidence_id is None:
        return None
    summaries = (
        state.accepted_evidence_summaries
        if accepted
        else state.rejected_evidence_summaries
    )
    matches = [
        dict(item)
        for item in summaries
        if _repair_summary_matches(
            item,
            kind,
            accepted=accepted,
            valid=valid,
            require_sql=require_sql,
        )
        and item.get("id") == evidence_id
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _repair_failure_evidence_summaries(
    state: DbLoopState,
    *,
    evidence_id: str | None,
) -> tuple[dict[str, Any], ...]:
    if evidence_id is None:
        return ()
    matches: list[dict[str, Any]] = []
    for kind in ("query.plan.validation", "sql.validation"):
        item = _repair_evidence_summary(
            state,
            kind,
            evidence_id=evidence_id,
            accepted=False,
            valid=False,
        )
        if item is not None:
            matches.append(item)
    return tuple(matches)


def _repair_summary_matches(
    summary: Mapping[str, Any],
    kind: str,
    *,
    accepted: bool,
    valid: bool | None,
    require_sql: bool,
) -> bool:
    if summary.get("kind") != kind:
        return False
    if summary.get("accepted", True) is not accepted:
        return False
    if not str(summary.get("id") or "").strip():
        return False
    if valid is not None and summary.get("valid") is not valid:
        return False
    if require_sql:
        sql = summary.get("sql")
        if summary.get("valid") is not True or not isinstance(sql, str):
            return False
        if not sql.strip():
            return False
    return True


def _repair_input_dependency(
    state: DbLoopState,
    summary: Mapping[str, Any],
    *,
    input_name: str,
) -> TaskDependency:
    return TaskDependency(
        kind="evidence",
        evidence_kind=str(summary["kind"]),
        evidence_id=str(summary["id"]),
        evidence_owner=_optional_string(summary.get("owner")),
        producer_task_id=_optional_string(summary.get("task_id")),
        evidence_accepted=summary.get("accepted", True) is True,
        operation_id=state.operation_id,
        metadata={
            "repair_input": input_name,
        },
    )


def _state_allows_read_profile(state: DbLoopState) -> bool:
    safety_frame = state.safety_frame or {}
    max_access = str(
        safety_frame.get("max_access")
        or safety_frame.get("max_allowed_access")
        or AccessMode.ADMIN.value
    )
    return _access_rank(max_access) >= _access_rank(AccessMode.READ.value)


def _governance_summary(governance: GovernanceResult | None) -> dict[str, Any]:
    if governance is None:
        return {}
    return governance.to_dict()


def _contract_snapshot(contract: DbOperationContract) -> dict[str, Any]:
    return {
        "operation_type": contract.operation_type,
        "required_capabilities": list(contract.required_capabilities),
        "required_evidence": list(contract.required_evidence),
        "access": contract.access.value,
        "limits": contract.limits.to_dict(),
        "policy_ids": list(contract.policy_ids),
        "metadata": contract.metadata,
    }


def _access_rank(value: str) -> int:
    return _ACCESS_ORDER.get(value, _ACCESS_ORDER[AccessMode.ADMIN.value])


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied, sort_keys=True, default=str)
    except TypeError as exc:
        raise TypeError("DB agent loop mappings must be JSON serializable") from exc
    return copied
