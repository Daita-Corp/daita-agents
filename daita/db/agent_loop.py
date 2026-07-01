"""Model-planned, runtime-governed DB agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
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

from .models import DbIntent, DbIntentKind, DbLimits, DbOperationContract
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
        )


class DbAgentLoop:
    """Serial first implementation of the planner-driven DB runtime loop."""

    def __init__(self, runtime: Any, planner: DbAgentPlanner) -> None:
        self.runtime = runtime
        self.planner = planner

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
            decision = await self.planner.plan(state)
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
            if decision.status is DbPlannerDecisionStatus.FINISH:
                observation = await self._persist_observation(
                    operation,
                    DbPlannerObservation(
                        diagnostics={"status": "finish_requested", "turn": turn}
                    ),
                    turn=turn,
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
                for item in accepted
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
        errors = [
            error
            for error in (
                self._capability_error(action, validation),
                self._capability_error(action, execute),
            )
            if error is not None
        ]
        if errors:
            return [], [], errors
        capabilities = [validation["capability"], execute["capability"]]
        access_error = self._access_errors(action, capabilities, state)
        if access_error:
            return [], [], access_error
        metadata = {
            **_action_metadata(action, decision_fingerprint),
            "sql_provenance": _sql_provenance_metadata(resolved_sql),
        }
        validation_spec = DbTaskSpec(
            capability_id="db.sql.validate",
            owner=capabilities[0].owner,
            input={"sql": resolved_sql.sql, "operation": sql_operation},
            reason=f"planner:{action.kind.value}:validation",
            sequence=sequence_start,
            dependencies=(
                (resolved_sql.query_plan_dependency,)
                if resolved_sql.query_plan_dependency is not None
                else ()
            ),
            metadata=metadata,
            deterministic_key=f"{action.action_id}:db.sql.validate",
        )
        validation_spec = _with_deterministic_task_id(
            state.operation_id,
            validation_spec,
        )
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
            owner=capabilities[1].owner,
            input=execute_input,
            reason=f"planner:{action.kind.value}",
            sequence=sequence_start + 1,
            metadata=metadata,
            deterministic_key=f"{action.action_id}:{execute_capability_id}",
        )
        execute_spec = _with_deterministic_task_id(
            state.operation_id,
            execute_spec,
        )
        return (
            [validation_spec, execute_spec],
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
        if action.kind is not DbPlannerActionKind.SEARCH_COLUMN_VALUES:
            return [], [], []

        tables, columns = _column_value_scope_for_action(action)
        if not tables or not columns or not _state_allows_read_profile(state):
            return [], [], []

        selected: list[dict[str, Any]] = []
        prerequisites: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        target_capability_id = "catalog.column_values.search"
        source_owner = _source_capability_owner(action)

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
            add_prerequisite("catalog.source.register", owner=_action_owner(action))
        profile_capability = add_prerequisite(
            "db.column_values.profile",
            owner=source_owner,
        )
        if profile_capability is not None:
            add_prerequisite(
                "catalog.column_values.register",
                owner=_action_owner(action),
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
                    "policy_id": item.policy_id,
                    "task_id": item.task_id,
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
    if (
        _explicit_mode_operation_type(state.explicit_mode)
        == DbIntentKind.SCHEMA_QUERY.value
        and action.kind in _SQL_QUERY_ACTIONS
    ):
        return (
            _action_error(
                action,
                f"action_outside_explicit_mode:{action.kind.value}:schema.query",
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
    )
    return any(
        str(item.get("error") or "").startswith(terminal_prefixes)
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

    if has_direct_sql and (has_plan_evidence_id or has_query_plan_ref):
        return None, "ambiguous_sql_input"
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
        return _resolve_sql_from_plan_evidence_id(
            plan_evidence_id.strip(),
            state,
        )

    if has_query_plan_ref:
        query_plan_ref = action.input.get("query_plan_ref")
        if query_plan_ref != "latest_accepted_query_plan":
            return None, f"unsupported_query_plan_ref:{query_plan_ref}"
        return _resolve_sql_from_latest_accepted_query_plan(state)

    return None, "missing_sql"


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
        if isinstance(sql, str) and sql.strip():
            return _resolved_sql_from_plan_summary(
                summary,
                state,
                provenance="latest_accepted_query_plan",
            )
    if summaries:
        return None, "query_plan_evidence_without_sql"
    return None, "missing_accepted_query_plan"


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
    payload_fingerprint = evidence.metadata.get("payload_fingerprint")
    if payload_fingerprint:
        summary["payload_fingerprint"] = str(payload_fingerprint)
    return summary


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
    tables = _first_action_string_list(action, "tables", "table", "target")
    columns = _first_action_string_list(action, "columns", "column", "field")
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


def _state_has_accepted_evidence(state: DbLoopState, kind: str) -> bool:
    return any(
        item.get("kind") == kind and item.get("accepted", True) is True
        for item in state.accepted_evidence_summaries
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
