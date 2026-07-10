"""Catalog structure and relationship helpers for the DB agent loop."""

from __future__ import annotations

from typing import Any, Mapping

from ..models import DbIntentKind
from ..planner_protocol import DbLoopState, DbPlannerAction
from ..sql_analysis import SqlAnalysisError, analyze_sql
from .actions import _explicit_mode_operation_type, _task_input_for_action
from .summaries import _safe_join_summaries
from .types import _ResolvedSqlInput
from .utils import (
    _first_string_list_from_mappings,
    _ordered_unique_strings,
    _string_list,
)


def _state_can_use_catalog_structure(state: DbLoopState) -> bool:
    effective_mode = _explicit_mode_operation_type(state.explicit_mode)
    operation_type = str(
        effective_mode
        or state.normalized_user_request.get("operation_type")
        or "db.run"
    )
    return operation_type in {
        "db.run",
        DbIntentKind.DATA_QUERY.value,
        DbIntentKind.CATALOG_ASSISTED_DATA_QUERY.value,
        "query.plan",
    }


def _catalog_capability_present(state: DbLoopState, capability_id: str) -> bool:
    return any(
        summary.get("id") == capability_id and summary.get("owner") == "catalog"
        for summary in state.capability_summaries
    )


def _state_has_catalog_structural_evidence(
    state: DbLoopState,
    kind: str | None = None,
) -> bool:
    structural_kinds = {
        "catalog.source_registered",
        "catalog.profile",
        "schema.search_result",
        "schema.asset_profile",
    }
    return any(
        summary.get("kind") in ({kind} if kind else structural_kinds)
        and summary.get("accepted", True) is True
        and (
            summary.get("owner") == "catalog"
            or str(summary.get("kind") or "").startswith("catalog.")
        )
        for summary in state.accepted_evidence_summaries
    )


def _state_has_catalog_asset_profile(state: DbLoopState) -> bool:
    return _state_has_catalog_structural_evidence(state, "schema.asset_profile")


def _planning_context_satisfies_catalog_phase2(
    state: DbLoopState,
    planning_context: Mapping[str, Any],
) -> bool:
    if not _state_can_use_catalog_structure(state):
        return True
    if not _catalog_capability_present(state, "catalog.schema.search"):
        return True
    diagnostics = planning_context.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    structural_source = planning_context.get(
        "structural_schema_source"
    ) or diagnostics.get("structural_schema_source")
    if structural_source == "catalog":
        return True
    structural_refs = _string_list(
        planning_context.get("catalog_structural_evidence_refs")
    ) or _string_list(diagnostics.get("catalog_structural_evidence_refs"))
    if structural_refs:
        return True
    return False


def _catalog_search_query(action: DbPlannerAction, state: DbLoopState) -> str:
    values = (
        action.input.get("query"),
        action.input.get("prompt"),
        action.input.get("goal"),
        state.normalized_user_request.get("query"),
        state.normalized_user_request.get("prompt"),
    )
    return " ".join(str(value).strip() for value in values if str(value or "").strip())


def _catalog_assets_for_action_or_state(
    action: DbPlannerAction,
    state: DbLoopState,
) -> list[str]:
    assets = _first_string_list_from_mappings(
        (action.input, action.metadata),
        "asset_ref",
        "asset",
        "assets",
        "tables",
        "table",
        "selected_tables",
    )
    if assets:
        return _ordered_unique_strings(assets)
    return _source_scope_asset_refs(state)


def _source_scope_asset_refs(state: DbLoopState) -> list[str]:
    scoped = _string_list(state.source_scope) or _string_list(
        state.normalized_user_request.get("source_scope")
    )
    if not scoped:
        return []
    source_owners = {
        str(summary.get("owner") or "").strip()
        for summary in state.capability_summaries
        if summary.get("id")
        in {
            "db.schema.inspect",
            "db.sql.validate",
            "db.sql.execute_read",
            "db.sql.execute_write",
            "db.column_values.profile",
        }
        and str(summary.get("owner") or "").strip()
    }
    reserved = source_owners | {"catalog", "db_runtime", "memory"}
    return [
        item
        for item in _ordered_unique_strings(scoped)
        if item not in reserved and ":" not in item
    ]


def _catalog_relationship_scope_for_action_or_state(
    action: DbPlannerAction,
    state: DbLoopState,
) -> tuple[list[str], list[str]]:
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
    paired_assets = _first_string_list_from_mappings(
        (action.input, action.metadata),
        "assets",
        "tables",
        "selected_tables",
    )
    if not from_assets and not to_assets and len(paired_assets) >= 2:
        from_assets = paired_assets[:1]
        to_assets = paired_assets[1:]
    if from_assets and not to_assets and len(from_assets) >= 2:
        to_assets = from_assets[1:]
        from_assets = from_assets[:1]
    if not from_assets and not to_assets:
        assets = _catalog_assets_for_action_or_state(action, state)
        if len(assets) >= 2:
            from_assets = assets[:1]
            to_assets = assets[1:]
    return _ordered_unique_strings(from_assets), _ordered_unique_strings(to_assets)


def _relationship_scope_for_resolved_sql(
    resolved_sql: _ResolvedSqlInput,
    state: DbLoopState,
) -> tuple[list[str], list[str]]:
    tables = _sql_join_tables(resolved_sql.sql)
    if not tables:
        plan_summary = _accepted_evidence_by_id(
            state,
            resolved_sql.source_evidence_id,
            kind=resolved_sql.source_evidence_kind,
        )
        if plan_summary is not None:
            tables = _string_list(plan_summary.get("selected_tables"))
    if len(tables) < 2:
        return [], []
    return [tables[0]], tables[1:]


def _sql_join_tables(sql: str) -> list[str]:
    if not sql.strip():
        return []
    try:
        analysis = analyze_sql(sql)
    except (ImportError, SqlAnalysisError):
        return []
    tables = [
        str(table.short_key)
        for table in analysis.tables
        if not table.is_cte and str(table.short_key)
    ]
    if len(tables) < 2:
        return []
    return _ordered_unique_strings(tables)


def _accepted_evidence_by_id(
    state: DbLoopState,
    evidence_id: str | None,
    *,
    kind: str | None = None,
) -> dict[str, Any] | None:
    if not evidence_id:
        return None
    matches = [
        summary
        for summary in state.accepted_evidence_summaries
        if summary.get("id") == evidence_id
        and (kind is None or summary.get("kind") == kind)
    ]
    return dict(matches[-1]) if matches else None


def _state_has_matching_relationship_evidence(
    state: DbLoopState,
    from_assets: list[str],
    to_assets: list[str],
) -> bool:
    for summary in state.accepted_evidence_summaries:
        if summary.get("kind") == "schema.relationship_path":
            if _join_summaries_match_scope(
                _safe_join_summaries(summary.get("joins")),
                from_assets,
                to_assets,
            ):
                return True
        if summary.get("kind") == "planning.context":
            if _planning_context_has_matching_relationship(
                summary,
                from_assets,
                to_assets,
            ):
                return True
    return False


def _planning_context_has_matching_relationship(
    planning_context: Mapping[str, Any],
    from_assets: list[str],
    to_assets: list[str],
) -> bool:
    return _join_summaries_match_scope(
        _safe_join_summaries(planning_context.get("relationship_joins")),
        from_assets,
        to_assets,
    )


def _join_summaries_match_scope(
    joins: list[dict[str, Any]],
    from_assets: list[str],
    to_assets: list[str],
) -> bool:
    if not joins:
        return False
    from_keys = {_short_asset_key(item) for item in from_assets if item}
    to_keys = {_short_asset_key(item) for item in to_assets if item}
    if not from_keys or not to_keys:
        return False
    for join in joins:
        left = _short_asset_key(join.get("left_table"))
        right = _short_asset_key(join.get("right_table"))
        if left in from_keys and right in to_keys:
            return True
        if right in from_keys and left in to_keys:
            return True
    return False


def _short_asset_key(value: Any) -> str:
    return str(value or "").split(".")[-1].strip().lower()


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
    fallback_from, fallback_to = _relationship_assets_from_structured_state(state)
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


def _relationship_assets_from_structured_state(
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
    return [], []


def _session_context_tables(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    referents = value.get("referents")
    if not isinstance(referents, Mapping):
        return []
    return _string_list(referents.get("tables"))
