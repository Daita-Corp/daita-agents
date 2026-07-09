"""Planner action compiler for the DB agent loop."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from daita.runtime import AccessMode, TaskDependency

from ..continuation import DbContinuationResolver
from ..models import DbOperationContract
from ..planner_protocol import (
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
)
from ..runtime.tasks.models import DbTaskSpec
from .actions import (
    _SIMPLE_ACTION_CAPABILITIES,
    _action_error,
    _action_metadata,
    _action_owner,
    _capability_selection,
    _coerce_action,
    _continuation_action_error,
    _dependency_for_prerequisite_spec,
    _durable_action_task_summaries,
    _explicit_mode_operation_type,
    _mode_action_errors,
    _ordered_actions_or_errors,
    _preferred_output_evidence_kind,
    _same_turn_repair_execute_deferral,
    _source_capability_owner,
    _task_input_for_action,
    _validated_sql_action_requires_query_plan_proposal,
    _with_deterministic_task_id,
    _with_spec_dependencies,
)
from .catalog import (
    _catalog_assets_for_action_or_state,
    _catalog_capability_present,
    _catalog_relationship_scope_for_action_or_state,
    _catalog_search_query,
    _planning_context_has_matching_relationship,
    _planning_context_satisfies_catalog_phase2,
    _relationship_paths_task_input,
    _relationship_scope_for_resolved_sql,
    _state_can_use_catalog_structure,
    _state_has_catalog_asset_profile,
    _state_has_catalog_structural_evidence,
    _state_has_matching_relationship_evidence,
)
from .contracts import (
    _access_rank,
    _contract_snapshot,
    _state_allows_read_profile,
)
from .grounding import (
    _CATALOG_COLUMN_VALUE_GROUNDING_REASON,
    _column_value_hint_task_input,
    _column_value_scope_for_action,
    _state_should_plan_value_grounding_for_planning,
    _validation_grounding_repair_context,
    _validation_grounding_runtime_continuation_action,
    _value_grounding_plan_task_input,
)
from .memory import (
    _memory_recall_task_input,
    _memory_update_runtime_continuation_action,
    _resolve_memory_proposal_for_action,
    _state_should_recall_memory_for_planning,
)
from .monitors import (
    _MONITOR_ACTION_CAPABILITIES,
    _monitor_action_output_evidence,
)
from .sql import (
    _repair_query_plan_task_input_and_dependencies,
    _resolve_sql_input_for_action,
    _sql_provenance_metadata,
    _validated_sql_execute_dependencies,
    _validated_sql_execute_deterministic_key,
    _validated_sql_execute_input,
)
from .summaries import (
    _latest_accepted_evidence_summary,
    _state_has_accepted_evidence,
)
from .types import DbActionCompilation
from .utils import _stable_hash


class DbActionCompiler:
    def __init__(
        self,
        runtime: Any,
        continuation_resolver: DbContinuationResolver,
    ) -> None:
        self.runtime = runtime
        self.continuation_resolver = continuation_resolver

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

        durable_action_summaries = _durable_action_task_summaries(state)
        ordered_actions, dag_errors = _ordered_actions_or_errors(
            tuple(actions),
            external_dependency_ids=frozenset(durable_action_summaries),
        )
        if dag_errors:
            rejected.extend(dag_errors)

        same_decision_repair_action_ids = frozenset(
            action.action_id
            for action in ordered_actions
            if action.kind is DbPlannerActionKind.REPAIR_QUERY_PLAN
        )
        prior_action_specs: dict[str, tuple[DbTaskSpec, ...]] = {}
        for action in ordered_actions:
            planner_dependencies, dependency_errors = self._planner_dependency_specs(
                action,
                prior_action_specs=prior_action_specs,
                durable_action_summaries=durable_action_summaries,
                operation_id=state.operation_id,
            )
            if dependency_errors:
                rejected.extend(dependency_errors)
                continue
            mode_errors = _mode_action_errors(action, state)
            if mode_errors:
                rejected.extend(mode_errors)
                continue
            repair_deferral = _same_turn_repair_execute_deferral(
                action,
                planner_dependencies,
                same_decision_repair_action_ids=same_decision_repair_action_ids,
            )
            if repair_deferral is not None:
                rejected.append(repair_deferral)
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
        durable_action_summaries: Mapping[str, tuple[dict[str, Any], ...]],
        operation_id: str,
    ) -> tuple[tuple[TaskDependency, ...], tuple[dict[str, Any], ...]]:
        dependencies: list[TaskDependency] = []
        errors: list[dict[str, Any]] = []
        for dependency_action_id in dict.fromkeys(action.depends_on):
            producer_specs = prior_action_specs.get(dependency_action_id)
            if producer_specs:
                dependency = self._dependency_for_producer_specs(
                    action,
                    dependency_action_id=dependency_action_id,
                    producer_specs=producer_specs,
                    operation_id=operation_id,
                )
            else:
                dependency = self._dependency_for_durable_action_summary(
                    action,
                    dependency_action_id=dependency_action_id,
                    task_summaries=durable_action_summaries.get(
                        dependency_action_id,
                        (),
                    ),
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

    def _dependency_for_durable_action_summary(
        self,
        action: DbPlannerAction,
        *,
        dependency_action_id: str,
        task_summaries: tuple[dict[str, Any], ...],
        operation_id: str,
    ) -> TaskDependency | None:
        for summary in reversed(task_summaries):
            capability_id = str(summary.get("capability_id") or "").strip()
            metadata = summary.get("metadata") if isinstance(summary, Mapping) else {}
            owner = (
                str(metadata.get("owner") or "").strip()
                if isinstance(metadata, Mapping)
                else ""
            )
            resolved = self._resolve_capability(capability_id, owner=owner or None)
            capability = resolved.get("capability")
            if capability is None:
                continue
            evidence_kind = _preferred_output_evidence_kind(capability)
            if evidence_kind is None:
                continue
            task_id = str(summary.get("task_id") or "").strip() or None
            if task_id is None:
                continue
            return TaskDependency(
                kind="evidence",
                evidence_kind=evidence_kind,
                evidence_owner=capability.owner,
                producer_task_id=task_id,
                producer_capability_id=capability.id,
                producer_executor_id=capability.executor,
                evidence_accepted=True,
                operation_id=operation_id,
                metadata={
                    "planner_dependency": True,
                    "durable_prior_action": True,
                    "producer_action_id": dependency_action_id,
                    "consumer_action_id": action.action_id,
                },
            )
        return None

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
            evidence_kind = _preferred_output_evidence_kind(capability)
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

        (
            catalog_specs,
            catalog_selected,
            catalog_dependencies,
            catalog_errors,
        ) = self._catalog_structure_prerequisite_specs(
            action,
            state=state,
            sequence_start=sequence_start + len(specs),
            decision_fingerprint=decision_fingerprint,
            upstream_dependencies=(
                (schema_dependency,) if schema_dependency is not None else ()
            ),
        )
        if catalog_errors:
            return [], [], catalog_errors
        specs.extend(catalog_specs)
        selected.extend(catalog_selected)
        context_dependencies.extend(catalog_dependencies)

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
        if planning_context is not None and _planning_context_satisfies_catalog_phase2(
            state,
            planning_context,
        ):
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
        uses_existing_sql_validation = (
            resolved_sql.sql_validation_dependency is not None
        )
        validation = (
            None
            if uses_existing_sql_validation
            else self._resolve_capability("db.sql.validate", owner=owner)
        )
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
                (
                    self._capability_error(action, validation)
                    if validation is not None
                    else None
                ),
                self._capability_error(action, execute),
            )
            if error is not None
        ]
        if errors:
            return [], [], errors
        capabilities = [
            *([plan_validation["capability"]] if plan_validation is not None else []),
            *([validation["capability"]] if validation is not None else []),
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
        selected_capabilities: list[Any] = []
        if plan_validation is not None:
            selected_capabilities.append(plan_validation["capability"])
        if validation is not None:
            selected_capabilities.append(validation["capability"])
        selected_capabilities.append(execute["capability"])
        validation_dependencies: list[TaskDependency] = []
        refreshed_context_dependency: TaskDependency | None = None
        relationship_dependencies: list[TaskDependency] = []
        relationship_scope = _relationship_scope_for_resolved_sql(resolved_sql, state)
        if (
            plan_validation is not None
            and planning_context is not None
            and relationship_scope[0]
            and relationship_scope[1]
            and not _planning_context_has_matching_relationship(
                planning_context,
                relationship_scope[0],
                relationship_scope[1],
            )
        ):
            (
                relationship_specs,
                relationship_selected,
                relationship_dependencies,
                relationship_errors,
            ) = self._catalog_structure_prerequisite_specs(
                action,
                state=state,
                sequence_start=sequence_start,
                decision_fingerprint=decision_fingerprint,
                force_relationship_scope=relationship_scope,
            )
            if relationship_errors:
                return [], [], relationship_errors
            if not any(
                dependency.evidence_kind == "schema.relationship_path"
                for dependency in relationship_dependencies
            ):
                relationship_specs = []
                relationship_selected = []
                relationship_dependencies = []
            specs.extend(relationship_specs)
            selected_capabilities.extend(
                item
                for item in (
                    self._resolve_capability(
                        selected["id"], owner=selected.get("owner")
                    ).get("capability")
                    for selected in relationship_selected
                )
                if item is not None
            )
        if (
            plan_validation is not None
            and relationship_dependencies
            and planning_context is not None
        ):
            context_resolved = self._resolve_capability(
                "db.planning.context.build",
                owner="db_runtime",
            )
            context_error = self._capability_error(action, context_resolved)
            if context_error is not None:
                return [], [], [context_error]
            context_capability = context_resolved["capability"]
            selected_capabilities.append(context_capability)
            context_spec = _with_deterministic_task_id(
                state.operation_id,
                DbTaskSpec(
                    capability_id=context_capability.id,
                    owner=context_capability.owner,
                    input={},
                    reason="runtime_prerequisite:catalog_relationship_context",
                    sequence=sequence_start + len(specs),
                    dependencies=tuple(relationship_dependencies),
                    metadata={
                        **metadata,
                        "runtime_prerequisite": True,
                        "catalog_relationship_context": True,
                    },
                    deterministic_key=(
                        f"{action.action_id}:db.planning.context.build:"
                        "catalog_relationship"
                    ),
                ),
            )
            specs.append(context_spec)
            refreshed_context_dependency = _dependency_for_prerequisite_spec(
                context_spec,
                capability=context_capability,
                operation_id=state.operation_id,
                action_id=action.action_id,
                consumer_capability_id="db.query.plan.validate",
            )
            planning_context = None

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
        elif plan_validation is not None and refreshed_context_dependency is not None:
            plan_capability = plan_validation["capability"]
            plan_validation_spec = DbTaskSpec(
                capability_id=plan_capability.id,
                owner=plan_capability.owner,
                input={
                    "plan_evidence_id": resolved_sql.source_evidence_id,
                },
                reason=f"planner:{action.kind.value}:query_plan_validation",
                sequence=sequence_start + len(specs),
                dependencies=(
                    *(
                        (resolved_sql.query_plan_dependency,)
                        if resolved_sql.query_plan_dependency is not None
                        else ()
                    ),
                    refreshed_context_dependency,
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
        elif resolved_sql.plan_validation_dependency is not None:
            validation_dependencies.append(resolved_sql.plan_validation_dependency)
        elif resolved_sql.query_plan_dependency is not None:
            validation_dependencies.append(resolved_sql.query_plan_dependency)

        validation_spec: DbTaskSpec | None = None
        if validation is not None:
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
        execute_input = _validated_sql_execute_input(
            action,
            resolved_sql,
            validation_spec=validation_spec,
        )
        execute_dependencies = _validated_sql_execute_dependencies(
            resolved_sql,
            validation_spec=validation_spec,
            operation_id=state.operation_id,
        )
        execute_spec = DbTaskSpec(
            capability_id=execute_capability_id,
            owner=execute_capability.owner,
            input=execute_input,
            reason=f"planner:{action.kind.value}",
            sequence=sequence_start + len(specs) + 1,
            dependencies=execute_dependencies,
            metadata=metadata,
            deterministic_key=_validated_sql_execute_deterministic_key(
                action,
                execute_capability_id=execute_capability_id,
                execute_input=execute_input,
                dependencies=execute_dependencies,
            ),
        )
        execute_spec = _with_deterministic_task_id(
            state.operation_id,
            execute_spec,
        )
        if validation_spec is not None:
            specs.append(validation_spec)
        specs.append(execute_spec)
        return (
            specs,
            [_capability_selection(item, action) for item in selected_capabilities],
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

    def _catalog_structure_prerequisite_specs(
        self,
        action: DbPlannerAction,
        *,
        state: DbLoopState,
        sequence_start: int,
        decision_fingerprint: str,
        upstream_dependencies: tuple[TaskDependency, ...] = (),
        force_relationship_scope: tuple[list[str], list[str]] | None = None,
    ) -> tuple[
        list[DbTaskSpec],
        list[dict[str, Any]],
        list[TaskDependency],
        list[dict[str, Any]],
    ]:
        if not _state_can_use_catalog_structure(state):
            return [], [], [], []
        if not _catalog_capability_present(state, "catalog.schema.search"):
            return [], [], [], []

        specs: list[DbTaskSpec] = []
        selected: list[dict[str, Any]] = []
        dependencies: list[TaskDependency] = []
        errors: list[dict[str, Any]] = []
        prior_dependency: TaskDependency | None = None

        def add_spec(
            capability_id: str,
            *,
            task_input: dict[str, Any],
            reason: str,
            deterministic_key: str,
            extra_dependencies: tuple[TaskDependency, ...] = (),
        ) -> TaskDependency | None:
            resolved = self._resolve_capability(capability_id, owner="catalog")
            error = self._capability_error(action, resolved)
            if error is not None:
                errors.append(error)
                return None
            capability = resolved["capability"]
            access_errors = self._access_errors(action, (capability,), state)
            if access_errors:
                errors.extend(access_errors)
                return None
            spec = _with_deterministic_task_id(
                state.operation_id,
                DbTaskSpec(
                    capability_id=capability.id,
                    owner=capability.owner,
                    input=task_input,
                    reason=reason,
                    sequence=sequence_start + len(specs),
                    dependencies=extra_dependencies,
                    metadata={
                        **_action_metadata(action, decision_fingerprint),
                        "runtime_prerequisite": True,
                        "catalog_structure": "planning",
                        "prerequisite_for": "db.planning.context.build",
                    },
                    deterministic_key=deterministic_key,
                ),
            )
            specs.append(spec)
            selected.append(_capability_selection(capability, action, reason=reason))
            dependency = _dependency_for_prerequisite_spec(
                spec,
                capability=capability,
                operation_id=state.operation_id,
                action_id=action.action_id,
                consumer_capability_id="db.planning.context.build",
            )
            dependencies.append(dependency)
            return dependency

        if not _state_has_catalog_structural_evidence(state, "schema.search_result"):
            search_query = _catalog_search_query(action, state)
            prior_dependency = add_spec(
                "catalog.schema.search",
                task_input={"query": search_query, "limit": 20},
                reason="runtime_prerequisite:catalog_schema_search",
                deterministic_key=f"{action.action_id}:catalog.schema.search",
                extra_dependencies=upstream_dependencies,
            )

        assets = _catalog_assets_for_action_or_state(action, state)
        if assets and not _state_has_catalog_asset_profile(state):
            for asset in assets[:4]:
                asset_dependency = add_spec(
                    "catalog.asset.inspect",
                    task_input={
                        "asset_ref": asset,
                        "limit": 100,
                        "include_relationships": True,
                    },
                    reason="runtime_prerequisite:catalog_asset_inspect",
                    deterministic_key=(
                        f"{action.action_id}:catalog.asset.inspect:{asset}"
                    ),
                    extra_dependencies=(
                        (prior_dependency,)
                        if prior_dependency is not None
                        else upstream_dependencies
                    ),
                )
                if asset_dependency is not None:
                    prior_dependency = asset_dependency

        relationship_scope = force_relationship_scope or (
            _catalog_relationship_scope_for_action_or_state(action, state)
        )
        if (
            relationship_scope[0]
            and relationship_scope[1]
            and not _state_has_matching_relationship_evidence(
                state,
                relationship_scope[0],
                relationship_scope[1],
            )
            and _catalog_capability_present(state, "catalog.relationship_paths.find")
        ):
            add_spec(
                "catalog.relationship_paths.find",
                task_input={
                    "from_assets": relationship_scope[0],
                    "to_assets": relationship_scope[1],
                    "max_hops": 4,
                    "max_paths": 5,
                },
                reason="runtime_prerequisite:catalog_relationship_paths",
                deterministic_key=(
                    f"{action.action_id}:catalog.relationship_paths.find:"
                    f"{','.join(relationship_scope[0])}:"
                    f"{','.join(relationship_scope[1])}"
                ),
                extra_dependencies=(
                    (prior_dependency,)
                    if prior_dependency is not None
                    else upstream_dependencies
                ),
            )

        return specs, selected, dependencies, errors

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
