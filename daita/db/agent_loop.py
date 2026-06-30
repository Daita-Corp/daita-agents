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

_TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.BLOCKED,
    TaskStatus.SKIPPED,
}

_LOOP_CONTROL_EVIDENCE_KINDS = {
    "planner.decision",
    "planner.compilation",
    "planner.observation",
    "verification.result",
    "answer.synthesis",
}


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
        warnings: list[str] = []
        last_compilation: DbActionCompilation | None = None
        for turn in range(1, turn_budget + 1):
            operation = await self._fresh_operation(operation.id)
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
                warnings.extend(
                    str(item.get("error") or "planner_action_rejected")
                    for item in compilation.rejected_action_summaries
                )
                await self._persist_observation(
                    operation,
                    self._observation_for_compilation(compilation, turn=turn),
                    turn=turn,
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
                    execution_errors.append({"error": str(exc)})

            observation = await self._persist_observation(
                operation,
                await self._observation_after_execution(
                    operation.id,
                    executed=tuple(executed),
                    execution_errors=tuple(execution_errors),
                    turn=turn,
                ),
                turn=turn,
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
        decision_fingerprint = _stable_hash(decision.to_dict())

        for raw_action in decision.actions:
            action, error = _coerce_action(raw_action)
            if error is not None or action is None:
                rejected.append(error or {"error": "invalid_action"})
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
            specs.extend(action_specs)
            selected_capabilities.extend(action_capabilities)
            accepted.append(
                {
                    "action_id": action.action_id,
                    "kind": action.kind.value,
                    "capabilities": action_capabilities,
                }
            )

        contract = self._compiled_contract_snapshot(
            decision=decision,
            decision_fingerprint=decision_fingerprint,
            selected_capabilities=tuple(selected_capabilities),
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
        sql = action.input.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return [], [], [_action_error(action, "missing_sql")]
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
        metadata = _action_metadata(action, decision_fingerprint)
        validation_spec = DbTaskSpec(
            capability_id="db.sql.validate",
            owner=capabilities[0].owner,
            input={"sql": sql, "operation": sql_operation},
            reason=f"planner:{action.kind.value}:validation",
            sequence=sequence_start,
            metadata=metadata,
            deterministic_key=f"{action.action_id}:db.sql.validate",
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
        for offset, capability in enumerate(capabilities):
            specs.append(
                DbTaskSpec(
                    capability_id=capability.id,
                    owner=capability.owner,
                    input=_task_input_for_action(action),
                    reason=f"planner:{action.kind.value}",
                    sequence=sequence_start + offset,
                    metadata=_action_metadata(action, decision_fingerprint),
                    deterministic_key=f"{action.action_id}:{capability.id}",
                )
            )
        return specs, [_capability_selection(item, action) for item in capabilities], []

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
        decision_fingerprint: str,
        selected_capabilities: tuple[dict[str, Any], ...],
        compiled_action_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        max_access = AccessMode.NONE
        required_evidence: set[str] = set()
        for selected in selected_capabilities:
            access = AccessMode(selected["access"])
            if _access_rank(access.value) > _access_rank(max_access.value):
                max_access = access
            required_evidence.update(str(item) for item in selected["output_evidence"])
        operation_type = str(
            decision.intent.get("operation_type")
            or decision.intent.get("label")
            or "db.run"
        )
        contract = DbOperationContract(
            operation_type=operation_type,
            required_capabilities=tuple(
                dict.fromkeys(str(item["id"]) for item in selected_capabilities)
            ),
            required_evidence=tuple(sorted(required_evidence)),
            access=max_access,
            limits=self.runtime.config.limits,
            policy_ids=tuple(
                str(item) for item in decision.intent.get("policy_ids") or ()
            ),
            metadata={
                "planner_intent": decision.intent,
                "planner_decision_fingerprint": decision_fingerprint,
                "compiled_action_ids": list(compiled_action_ids),
                "selected_capabilities": list(selected_capabilities),
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

    def _observation_for_compilation(
        self,
        compilation: DbActionCompilation,
        *,
        turn: int,
    ) -> DbPlannerObservation:
        return DbPlannerObservation(
            execution_errors=tuple(
                {
                    "action_id": item.get("action_id"),
                    "kind": item.get("kind"),
                    "error": item.get("error"),
                }
                for item in compilation.rejected_action_summaries
            ),
            diagnostics={
                "status": "compilation_rejected",
                "turn": turn,
                "compilation": compilation.to_dict(),
            },
        )

    async def _observation_after_execution(
        self,
        operation_id: str,
        *,
        executed: tuple[Evidence, ...],
        execution_errors: tuple[dict[str, Any], ...],
        turn: int,
    ) -> DbPlannerObservation:
        return DbPlannerObservation(
            accepted_evidence_summaries=tuple(
                _evidence_summary(item) for item in executed if item.accepted
            ),
            rejected_evidence_summaries=tuple(
                _evidence_summary(item) for item in executed if not item.accepted
            ),
            task_statuses=await self._task_summaries(operation_id),
            execution_errors=execution_errors,
            diagnostics={"status": "tasks_executed", "turn": turn},
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
        evidence = tuple(await self.runtime.store.list_evidence(operation_id))
        tasks = tuple(await self.runtime.store.list_tasks(operation_id))
        fallback_contract = _fallback_contract_for_operation(
            operation,
            limits=self.runtime.config.limits,
        )
        contract = _contract_from_latest_snapshot(operation, fallback_contract)
        fallback_intent = _fallback_intent_for_operation(operation, contract)
        intent = _intent_from_contract(contract, fallback_intent)
        verification = self.runtime.verifier.verify(contract, intent, evidence, tasks)
        supporting_evidence = _accepted_synthesis_support_evidence(evidence)
        finalizable = verification.passed and bool(supporting_evidence)
        return finalizable, {
            "finalizable": finalizable,
            "verification": verification.to_dict(),
            "intent": _intent_summary(intent),
            "contract": _contract_snapshot(contract),
            "synthesis_supporting_evidence": tuple(
                _evidence_ref(item) for item in supporting_evidence
            ),
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


def _action_owner(action: DbPlannerAction) -> str | None:
    owner = action.input.get("owner") or action.input.get("capability_owner")
    return str(owner) if owner else None


def _task_input_for_action(action: DbPlannerAction) -> dict[str, Any]:
    return {
        key: value
        for key, value in action.input.items()
        if key not in {"owner", "capability_owner"}
    }


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


def _capability_selection(capability: Any, action: DbPlannerAction) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "access": capability.access.value,
        "risk": capability.risk.value,
        "output_evidence": sorted(capability.output_evidence),
        "action_id": action.action_id,
        "reason": action.kind.value,
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
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "accepted": evidence.accepted,
        "task_id": evidence.task_id,
    }


def _evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "accepted": evidence.accepted,
    }


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


def _accepted_synthesis_support_evidence(
    evidence: tuple[Evidence, ...],
) -> tuple[Evidence, ...]:
    return tuple(
        item
        for item in evidence
        if item.accepted and item.kind not in _LOOP_CONTROL_EVIDENCE_KINDS
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


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
