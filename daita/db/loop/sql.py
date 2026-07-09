"""SQL resolution and repair helpers for the DB agent loop."""

from __future__ import annotations

import re
from typing import Any, Mapping

from daita.runtime import TaskDependency

from ..planner_protocol import DbLoopState, DbPlannerAction
from ..runtime.tasks import DbTaskSpec
from .actions import _action_error, _merge_dependencies
from .summaries import _latest_accepted_evidence_summary
from .types import _ResolvedSqlInput
from .utils import _optional_string, _stable_hash


def _resolve_sql_input_for_action(
    action: DbPlannerAction,
    state: DbLoopState,
    *,
    sql_operation: str,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    raw_direct_sql = action.input.get("sql")
    direct_sql = (
        raw_direct_sql.strip()
        if isinstance(raw_direct_sql, str) and raw_direct_sql.strip()
        else None
    )
    has_plan_evidence_id = "plan_evidence_id" in action.input
    has_query_plan_ref = "query_plan_ref" in action.input

    if direct_sql is not None and has_plan_evidence_id:
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
    if direct_sql is not None and has_query_plan_ref:
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
        plan_evidence_id = action.input.get("plan_evidence_id")
        if not isinstance(plan_evidence_id, str) or not plan_evidence_id.strip():
            return None, "missing_plan_evidence_id"
        query_plan_ref = action.input.get("query_plan_ref")
        if query_plan_ref != "latest_accepted_query_plan":
            return None, f"unsupported_query_plan_ref:{query_plan_ref}"
        explicit, explicit_error = _resolve_sql_from_plan_evidence_id(
            plan_evidence_id.strip(),
            state,
        )
        if explicit_error == f"plan_evidence_not_found:{plan_evidence_id.strip()}":
            explicit, explicit_error = _resolve_sql_from_validation_evidence_id(
                plan_evidence_id.strip(),
                state,
                sql_operation=sql_operation,
            )
        if explicit_error is not None or explicit is None:
            return None, explicit_error
        latest, latest_error = _resolve_sql_from_latest_accepted_query_plan(state)
        if latest_error is not None or latest is None:
            return None, latest_error
        if (
            explicit.source_evidence_id != latest.source_evidence_id
            or not _sql_inputs_match(explicit.sql, latest.sql)
        ):
            return None, "ambiguous_sql_input"
        return explicit, None

    if direct_sql is not None:
        return (
            _ResolvedSqlInput(
                sql=direct_sql,
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


def _sql_input_fingerprint(sql: str) -> str:
    return _stable_hash({"sql": sql})


def _validated_sql_execute_input(
    action: DbPlannerAction,
    resolved_sql: _ResolvedSqlInput,
    *,
    validation_spec: DbTaskSpec | None,
) -> dict[str, Any]:
    execute_input: dict[str, Any] = {
        "sql_fingerprint": _sql_input_fingerprint(resolved_sql.sql),
        "params": list(action.input.get("params") or ()),
    }
    if validation_spec is not None:
        execute_input["sql_validation_task_id"] = validation_spec.task_id
        execute_input["sql_validation_input_hash"] = _stable_hash(validation_spec.input)
    if resolved_sql.sql_validation_dependency is not None:
        dependency = resolved_sql.sql_validation_dependency
        if dependency.evidence_id:
            execute_input["sql_validation_evidence_id"] = dependency.evidence_id
        if dependency.payload_fingerprint:
            execute_input["sql_validation_payload_fingerprint"] = (
                dependency.payload_fingerprint
            )
        if dependency.producer_task_id:
            execute_input["sql_validation_task_id"] = dependency.producer_task_id
        if dependency.input_hash:
            execute_input["sql_validation_input_hash"] = dependency.input_hash
    if resolved_sql.query_plan_dependency is not None:
        dependency = resolved_sql.query_plan_dependency
        if dependency.evidence_id:
            execute_input["plan_evidence_id"] = dependency.evidence_id
        if dependency.payload_fingerprint:
            execute_input["plan_payload_fingerprint"] = dependency.payload_fingerprint
        if dependency.producer_task_id:
            execute_input["plan_task_id"] = dependency.producer_task_id
    if resolved_sql.plan_validation_dependency is not None:
        dependency = resolved_sql.plan_validation_dependency
        if dependency.evidence_id:
            execute_input["plan_validation_evidence_id"] = dependency.evidence_id
        if dependency.payload_fingerprint:
            execute_input["plan_validation_payload_fingerprint"] = (
                dependency.payload_fingerprint
            )
        if dependency.producer_task_id:
            execute_input["plan_validation_task_id"] = dependency.producer_task_id
    if action.input.get("param_specs"):
        execute_input["param_specs"] = list(action.input.get("param_specs") or ())
    if action.input.get("focus") is not None:
        execute_input["focus"] = action.input["focus"]
    return execute_input


def _validated_sql_execute_dependencies(
    resolved_sql: _ResolvedSqlInput,
    *,
    validation_spec: DbTaskSpec | None,
    operation_id: str,
) -> tuple[TaskDependency, ...]:
    dependencies: list[TaskDependency] = []
    if resolved_sql.query_plan_dependency is not None:
        dependencies.append(resolved_sql.query_plan_dependency)
    if resolved_sql.plan_validation_dependency is not None:
        dependencies.append(resolved_sql.plan_validation_dependency)
    if resolved_sql.sql_validation_dependency is not None:
        dependencies.append(resolved_sql.sql_validation_dependency)
    elif validation_spec is not None:
        dependencies.append(
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                evidence_owner=validation_spec.owner,
                producer_task_id=validation_spec.task_id,
                producer_capability_id=validation_spec.capability_id,
                evidence_accepted=True,
                evidence_payload={"valid": True},
                input_hash=_stable_hash(validation_spec.input),
                operation_id=operation_id,
                metadata={
                    "sql_resolution": True,
                    "provenance": "same_turn_sql_validation",
                },
            )
        )
    return _merge_dependencies((), tuple(dependencies))


def _validated_sql_execute_deterministic_key(
    action: DbPlannerAction,
    *,
    execute_capability_id: str,
    execute_input: Mapping[str, Any],
    dependencies: tuple[TaskDependency, ...],
) -> str:
    identity = {
        "input": execute_input,
        "dependencies": [dependency.to_dict() for dependency in dependencies],
    }
    return f"{action.action_id}:{execute_capability_id}:{_stable_hash(identity)[:16]}"


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
            state=state,
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
        state=state,
    )


def _resolved_sql_from_validation_summary(
    summary: Mapping[str, Any],
    *,
    sql: str,
    provenance: str,
    state: DbLoopState,
) -> tuple[_ResolvedSqlInput | None, str | None]:
    evidence_id = _optional_string(summary.get("id"))
    evidence_owner = _optional_string(summary.get("owner"))
    task_id = _optional_string(summary.get("task_id"))
    payload_fingerprint = _optional_string(summary.get("payload_fingerprint"))
    dependency = TaskDependency(
        kind="evidence",
        evidence_kind="sql.validation",
        evidence_id=evidence_id,
        evidence_owner=evidence_owner,
        producer_task_id=task_id,
        evidence_accepted=True,
        evidence_payload={"valid": True},
        operation_id=state.operation_id,
        payload_fingerprint=payload_fingerprint,
        input_hash=_optional_string(summary.get("task_input_hash")),
        metadata={
            "sql_resolution": True,
            "provenance": provenance,
        },
    )
    return (
        _ResolvedSqlInput(
            sql=sql.strip(),
            provenance=provenance,
            sql_validation_dependency=dependency,
            source_evidence_id=evidence_id,
            source_evidence_kind="sql.validation",
            source_evidence_owner=evidence_owner,
            source_task_id=task_id,
            source_payload_fingerprint=payload_fingerprint,
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
    plan_validation_dependency = _accepted_plan_validation_dependency_for_plan(
        state,
        plan_evidence_id=evidence_id,
    )
    return (
        _ResolvedSqlInput(
            sql=sql.strip(),
            provenance=provenance,
            query_plan_dependency=dependency,
            plan_validation_dependency=plan_validation_dependency,
            source_evidence_id=evidence_id,
            source_evidence_kind="query.plan.proposal",
            source_evidence_owner=evidence_owner,
            source_task_id=task_id,
            source_payload_fingerprint=payload_fingerprint,
        ),
        None,
    )


def _accepted_plan_validation_dependency_for_plan(
    state: DbLoopState,
    *,
    plan_evidence_id: str | None,
) -> TaskDependency | None:
    if not plan_evidence_id:
        return None
    for summary in reversed(state.accepted_evidence_summaries):
        if summary.get("kind") != "query.plan.validation":
            continue
        if summary.get("accepted") is not True or summary.get("valid") is not True:
            continue
        if summary.get("plan_evidence_id") != plan_evidence_id:
            continue
        evidence_id = _optional_string(summary.get("id"))
        if evidence_id is None:
            continue
        return TaskDependency(
            kind="evidence",
            evidence_kind="query.plan.validation",
            evidence_id=evidence_id,
            evidence_owner=_optional_string(summary.get("owner")),
            producer_task_id=_optional_string(summary.get("task_id")),
            evidence_accepted=True,
            operation_id=state.operation_id,
            payload_fingerprint=_optional_string(summary.get("payload_fingerprint")),
            input_hash=_optional_string(summary.get("task_input_hash")),
            metadata={
                "sql_resolution": True,
                "provenance": "query_plan_validation",
            },
        )
    return None


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
