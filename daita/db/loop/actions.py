"""Planner action and task-spec helpers for the DB agent loop."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from daita.runtime import TaskDependency, TaskStatus

from ..fingerprints import persisted_fingerprint
from ..models import DbIntentKind
from ..planner_protocol import DbLoopState, DbPlannerAction, DbPlannerActionKind
from ..runtime.tasks.models import DbTaskSpec
from .types import DbActionCompilation

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


_TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.BLOCKED,
    TaskStatus.SKIPPED,
}


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


def _durable_action_task_summaries(
    state: DbLoopState,
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in state.task_summaries:
        if str(summary.get("status") or "") != TaskStatus.SUCCEEDED.value:
            continue
        metadata = summary.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        action_id = str(metadata.get("planner_action_id") or "").strip()
        if not action_id:
            continue
        grouped.setdefault(action_id, []).append(dict(summary))
    return {action_id: tuple(items) for action_id, items in grouped.items()}


def _ordered_actions_or_errors(
    actions: tuple[DbPlannerAction, ...],
    *,
    external_dependency_ids: frozenset[str] = frozenset(),
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
            if dependency not in by_id and dependency not in external_dependency_ids:
                missing_errors.append(
                    _action_error(action, f"missing_dependency:{dependency}")
                )
    if missing_errors:
        return (), tuple(missing_errors)

    dependents: dict[str, list[str]] = {action.action_id: [] for action in actions}
    remaining_dependencies: dict[str, int] = {}
    for action in actions:
        dependencies = dependency_sets[action.action_id]
        current_dependencies = tuple(
            dependency for dependency in dependencies if dependency in by_id
        )
        remaining_dependencies[action.action_id] = len(current_dependencies)
        for dependency in current_dependencies:
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


def _same_turn_repair_execute_deferral(
    action: DbPlannerAction,
    planner_dependencies: tuple[TaskDependency, ...],
    *,
    same_decision_repair_action_ids: frozenset[str],
) -> dict[str, Any] | None:
    if action.kind not in {
        DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
    }:
        return None
    if not _validated_sql_action_requires_query_plan_proposal(action):
        return None

    producer_action_ids = tuple(
        dict.fromkeys(
            str(dependency.metadata.get("producer_action_id") or "")
            for dependency in planner_dependencies
            if dependency.evidence_kind == "query.plan.proposal"
            and str(dependency.metadata.get("producer_action_id") or "")
            in same_decision_repair_action_ids
        )
    )
    if not producer_action_ids:
        return None
    return {
        **_action_error(
            action,
            "deferred_until_query_plan_proposal_available",
        ),
        "deferred": {
            "reason": "same_turn_repair_query_plan",
            "required_evidence_kind": "query.plan.proposal",
            "producer_action_ids": list(producer_action_ids),
            "non_terminal": True,
        },
    }


def _validated_sql_action_requires_query_plan_proposal(
    action: DbPlannerAction,
) -> bool:
    if action.input.get("query_plan_ref") == "latest_accepted_query_plan":
        return True
    if str(action.input.get("plan_evidence_id") or "").strip():
        return True
    sql = action.input.get("sql")
    return not (isinstance(sql, str) and sql.strip())


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
    input_hash = persisted_fingerprint(spec.input)
    idempotency_key = spec.idempotency_key or persisted_fingerprint(
        {
            "operation_id": operation_id,
            "capability_id": spec.capability_id,
            "owner": spec.owner,
            "input_hash": input_hash,
            "sequence": spec.sequence,
            "deterministic_key": spec.deterministic_key,
        }
    )
    task_fingerprint = persisted_fingerprint(
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
        fingerprint = persisted_fingerprint(dependency.to_dict())
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


def _dependency_for_prerequisite_spec(
    spec: DbTaskSpec,
    *,
    capability: Any,
    operation_id: str,
    action_id: str,
    consumer_capability_id: str,
) -> TaskDependency:
    evidence_kind = _preferred_output_evidence_kind(capability) or ""
    return TaskDependency(
        kind="evidence",
        evidence_kind=evidence_kind,
        evidence_owner=capability.owner,
        producer_task_id=spec.task_id,
        producer_capability_id=capability.id,
        producer_executor_id=capability.executor,
        evidence_accepted=True,
        input_hash=persisted_fingerprint(spec.input),
        operation_id=operation_id,
        metadata={
            "runtime_prerequisite": True,
            "producer_action_id": action_id,
            "consumer_action_id": action_id,
            "prerequisite_for": consumer_capability_id,
        },
    )


def _preferred_output_evidence_kind(capability: Any) -> str | None:
    output_evidence = set(getattr(capability, "output_evidence", ()) or ())
    if (
        capability.id == "db.planning.context.build"
        and "planning.context" in output_evidence
    ):
        return "planning.context"
    return next(iter(sorted(output_evidence)), None)


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
