"""Value grounding helpers for the DB agent loop."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from ..fingerprints import persisted_fingerprint
from ..planner_protocol import DbLoopState, DbPlannerAction, DbPlannerActionKind
from .contracts import _state_allows_read_profile
from .memory import _single_source_owner_for_state
from .summaries import _state_has_accepted_evidence
from .utils import (
    _dedupe_dicts,
    _dedupe_json_values,
    _first_present,
    _ordered_unique_strings,
    _safe_iterable,
    _string_list,
    _split_column_ref,
)

_CATALOG_COLUMN_VALUE_GROUNDING_REASON = "catalog_column_value_grounding"


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
            values = _string_list(source.get(key))
            if values:
                return values
    return []


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
    for summary in reversed(state.validation_summaries):
        facts = list(_safe_iterable(summary.get("validation_facts")))
        warnings = list(_safe_iterable(summary.get("warnings")))
        warnings.extend(_safe_iterable(summary.get("validation_warnings")))
        if _validation_value_grounding_targets((*facts, *warnings)):
            return _dedupe_json_values(facts), _dedupe_json_values(warnings)
    return [], []


def _validation_grounding_repair_context(
    state: DbLoopState,
) -> dict[str, Any]:
    failed_validation = next(
        (
            summary
            for summary in reversed(state.validation_summaries)
            if summary.get("kind") in {"query.plan.validation", "sql.validation"}
            and summary.get("valid") is False
            and _validation_value_grounding_targets(
                (
                    *list(_safe_iterable(summary.get("validation_facts"))),
                    *list(_safe_iterable(summary.get("warnings"))),
                    *list(_safe_iterable(summary.get("validation_warnings"))),
                )
            )
        ),
        None,
    )
    if failed_validation is None:
        return {}
    validation_facts = list(_safe_iterable(failed_validation.get("validation_facts")))
    validation_warnings = list(_safe_iterable(failed_validation.get("warnings")))
    validation_warnings.extend(
        _safe_iterable(failed_validation.get("validation_warnings"))
    )
    targets = _validation_value_grounding_targets(
        (*validation_facts, *validation_warnings)
    )
    if not targets:
        return {}
    target_refs = [f"{target['table']}.{target['column']}" for target in targets]
    satisfied_refs = _state_column_value_hint_refs(state)
    missing_refs = [ref for ref in target_refs if ref.lower() not in satisfied_refs]
    fingerprint = persisted_fingerprint(
        {
            "operation_id": state.operation_id,
            "validation_kind": failed_validation.get("kind"),
            "validation_evidence_id": failed_validation.get("id"),
            "plan_evidence_id": failed_validation.get("plan_evidence_id"),
            "planning_context_evidence_id": failed_validation.get(
                "planning_context_evidence_id"
            ),
            "planning_context_fingerprint": failed_validation.get(
                "planning_context_fingerprint"
            ),
            "schema_fingerprint": failed_validation.get("schema_fingerprint"),
            "session_scope_binding_fingerprint": failed_validation.get(
                "session_scope_binding_fingerprint"
            ),
            "session_context_fingerprint": failed_validation.get(
                "session_context_fingerprint"
            ),
            "contract_fingerprint": failed_validation.get("contract_fingerprint"),
            "targets": targets,
        }
    )
    return {
        "attempted": True,
        "validation_kind": failed_validation.get("kind"),
        "fingerprint": fingerprint,
        "target_refs": target_refs,
        "targets": targets,
        "validation_evidence_id": failed_validation.get("id"),
        "plan_evidence_id": failed_validation.get("plan_evidence_id"),
        "planning_context_evidence_id": failed_validation.get(
            "planning_context_evidence_id"
        ),
        "planning_context_fingerprint": failed_validation.get(
            "planning_context_fingerprint"
        ),
        "schema_fingerprint": failed_validation.get("schema_fingerprint"),
        "session_scope_binding_fingerprint": failed_validation.get(
            "session_scope_binding_fingerprint"
        ),
        "session_context_fingerprint": failed_validation.get(
            "session_context_fingerprint"
        ),
        "contract_fingerprint": failed_validation.get("contract_fingerprint"),
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
    if not _state_allows_read_profile(state):
        return None
    fingerprint = str(repair_context.get("fingerprint") or "").strip()
    targets = _safe_target_list(repair_context.get("targets"))
    read_owners = {
        str(summary.get("owner") or "").strip()
        for summary in state.capability_summaries
        if summary.get("id") == "db.sql.execute_read"
        and str(summary.get("owner") or "").strip()
    }
    source_owner = (
        next(iter(read_owners))
        if len(read_owners) == 1
        else _single_source_owner_for_state(state)
    )
    action_input: dict[str, Any] = {}
    if source_owner:
        action_input["owner"] = source_owner

    accepted_plans = tuple(
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("kind") == "query.plan.proposal"
        and summary.get("accepted", True) is True
        and summary.get("valid") is True
        and isinstance(summary.get("sql"), str)
        and bool(str(summary.get("sql") or "").strip())
    )
    latest_plan = accepted_plans[-1] if accepted_plans else None
    latest_plan_id = str((latest_plan or {}).get("id") or "").strip()
    grounding_plan_id = str(repair_context.get("plan_evidence_id") or "").strip()
    grounding_applies = repair_context.get(
        "validation_kind"
    ) == "query.plan.validation" and (
        not grounding_plan_id or grounding_plan_id == latest_plan_id
    )

    if missing_refs and grounding_applies:
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
        context_input = {"source_owner": source_owner} if source_owner else {}
        return DbPlannerAction(
            action_id=_runtime_validation_grounding_action_id(
                {**repair_context, "stage": "context_refresh"},
                current_action_ids,
            ),
            kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
            input=context_input,
            rationale="Runtime continuation for validation-driven value grounding.",
            metadata=metadata,
        )

    if latest_plan is None:
        return None
    if any(
        summary.get("kind") == "query.result"
        and summary.get("accepted", True) is True
        and summary.get("plan_evidence_id") == latest_plan_id
        for summary in state.accepted_evidence_summaries
    ):
        return None

    latest_plan_validation = next(
        (
            summary
            for summary in reversed(state.validation_summaries)
            if summary.get("kind") == "query.plan.validation"
            and summary.get("plan_evidence_id") == latest_plan_id
        ),
        None,
    )
    latest_failed_validation = (
        latest_plan_validation
        if latest_plan_validation is not None
        and latest_plan_validation.get("valid") is False
        else None
    )
    failed_plan_id = str(
        (latest_failed_validation or {}).get("plan_evidence_id") or ""
    ).strip()
    repair_validation_id = str(
        repair_context.get("validation_evidence_id") or ""
    ).strip()

    if (
        latest_failed_validation is not None
        and failed_plan_id == latest_plan_id
        and str(latest_failed_validation.get("id") or "") == repair_validation_id
    ):
        planning_context = next(
            (
                summary
                for summary in reversed(state.accepted_evidence_summaries)
                if summary.get("kind") == "planning.context"
                and summary.get("accepted", True) is True
                and str(summary.get("id") or "").strip()
            ),
            None,
        )
        if planning_context is None:
            return None
        failure_id = str(latest_failed_validation.get("id") or "").strip()
        context_id = str(planning_context.get("id") or "").strip()
        repeated_repair = next(
            (
                summary
                for summary in reversed(state.rejected_evidence_summaries)
                if summary.get("kind") == "query.plan.repair"
                and summary.get("failure_evidence_id") == failure_id
                and summary.get("prior_plan_evidence_id") == latest_plan_id
                and summary.get("planning_context_evidence_id") == context_id
            ),
            None,
        )
        repair_metadata: dict[str, Any] = {
            "runtime_continuation": True,
            "continuation": "validation_grounding.query_plan_repair",
            "validation_grounding_fingerprint": fingerprint,
            "validation_grounding_targets": targets,
        }
        if repeated_repair is not None:
            repair_metadata["continuation_resolution"] = {
                "status": "blocked",
                "source": "runtime_continuation",
                "error": "validation_grounding_repair_exhausted",
                "failure_evidence_id": failure_id,
                "prior_plan_evidence_id": latest_plan_id,
                "planning_context_evidence_id": context_id,
            }
        return DbPlannerAction(
            action_id=_runtime_validation_grounding_action_id(
                {
                    **repair_context,
                    "stage": "repair",
                    "failure_evidence_id": failure_id,
                    "prior_plan_evidence_id": latest_plan_id,
                    "planning_context_evidence_id": context_id,
                },
                current_action_ids,
            ),
            kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
            input={
                "owner": "db_runtime",
                "failure_evidence_id": failure_id,
                "prior_plan_evidence_id": latest_plan_id,
                "planning_context_evidence_id": context_id,
            },
            rationale="Runtime continuation for a grounded query-plan repair.",
            metadata=repair_metadata,
        )

    if (
        latest_plan_validation is not None
        and latest_plan_validation.get("valid") is False
    ):
        return None

    planning_context = next(
        (
            summary
            for summary in reversed(state.accepted_evidence_summaries)
            if summary.get("kind") == "planning.context"
            and summary.get("accepted", True) is True
        ),
        None,
    )
    if planning_context is not None:
        plan_context_id = str(
            latest_plan.get("planning_context_evidence_id") or ""
        ).strip()
        current_context_id = str(planning_context.get("id") or "").strip()
        plan_context_fingerprint = str(
            latest_plan.get("planning_context_fingerprint") or ""
        ).strip()
        current_context_fingerprint = str(
            planning_context.get("payload_fingerprint") or ""
        ).strip()
        plan_schema = str(latest_plan.get("schema_fingerprint") or "").strip()
        current_schema = str(planning_context.get("schema_fingerprint") or "").strip()
        plan_session = str(latest_plan.get("session_context_fingerprint") or "").strip()
        current_session = str(
            planning_context.get("session_context_fingerprint") or ""
        ).strip()
        plan_contract = str(latest_plan.get("contract_fingerprint") or "").strip()
        current_contract = str(
            state.diagnostics.get("contract_fingerprint") or ""
        ).strip()
        stale_reason = None
        if (
            plan_context_fingerprint
            and current_context_fingerprint
            and plan_context_fingerprint != current_context_fingerprint
        ):
            stale_reason = "stale_query_plan_planning_context"
        elif (
            not plan_context_fingerprint
            and not current_context_fingerprint
            and plan_context_id
            and current_context_id
            and plan_context_id != current_context_id
        ):
            stale_reason = "stale_query_plan_planning_context"
        elif plan_schema and current_schema and plan_schema != current_schema:
            stale_reason = "stale_query_plan_schema"
        elif plan_session and current_session and plan_session != current_session:
            stale_reason = "stale_query_plan_session_binding"
        elif plan_contract and current_contract and plan_contract != current_contract:
            stale_reason = "stale_query_plan_contract"
        if stale_reason is not None:
            return DbPlannerAction(
                action_id=_runtime_validation_grounding_action_id(
                    {**repair_context, "stage": "stale", "plan": latest_plan_id},
                    current_action_ids,
                ),
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={"plan_evidence_id": latest_plan_id, **action_input},
                rationale="Runtime continuation rejected a stale query plan.",
                metadata={
                    "runtime_continuation": True,
                    "continuation": "validation_grounding.validated_execution",
                    "continuation_resolution": {
                        "status": "blocked",
                        "source": "runtime_continuation",
                        "error": stale_reason,
                        "plan_evidence_id": latest_plan_id,
                    },
                },
            )

    return DbPlannerAction(
        action_id=_runtime_validation_grounding_action_id(
            {**repair_context, "stage": "execute", "plan": latest_plan_id},
            current_action_ids,
        ),
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"plan_evidence_id": latest_plan_id, **action_input},
        rationale="Runtime continuation for governed validated read execution.",
        metadata={
            "runtime_continuation": True,
            "continuation": "validation_grounding.validated_execution",
            "validation_grounding_fingerprint": fingerprint,
        },
    )


def _runtime_validation_grounding_action_id(
    repair_context: Mapping[str, Any],
    current_action_ids: set[str],
) -> str:
    fingerprint = str(repair_context.get("fingerprint") or "")
    seed = {
        "fingerprint": fingerprint,
        "target_refs": list(repair_context.get("target_refs") or ()),
        "stage": repair_context.get("stage"),
        "failure_evidence_id": repair_context.get("failure_evidence_id"),
        "prior_plan_evidence_id": repair_context.get("prior_plan_evidence_id"),
        "planning_context_evidence_id": repair_context.get(
            "planning_context_evidence_id"
        ),
        "plan": repair_context.get("plan"),
    }
    action_id = f"runtime_validation_grounding_{persisted_fingerprint(seed)[:12]}"
    if action_id not in current_action_ids:
        return action_id
    return f"{action_id}_{persisted_fingerprint({'existing_ids': sorted(current_action_ids)})[:8]}"


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
