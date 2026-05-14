"""
Structured query planning helpers for ``Agent.from_db()``.

The LLM provides intent. This module enriches that intent with catalog facts
from ``navigation.py`` so downstream SQL validation and compilation work from a
shared shape instead of raw prose.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .navigation import find_join_paths, search_schema
from .state import DbQueryPlan, DbRunState

MAX_CANDIDATE_TABLES = 8
MAX_FIELD_CANDIDATES = 8


def build_query_plan(
    args: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    run_state: Optional[DbRunState] = None,
) -> Dict[str, Any]:
    """Normalize LLM intent and enrich it with schema candidates."""

    plan = DbQueryPlan(
        goal=str(args.get("goal") or "").strip(),
        required_fields=_string_list(args.get("required_fields")),
        candidate_tables=_string_list(args.get("candidate_tables")),
        required_joins=_join_requirements(args.get("required_joins")),
        filters=_string_list(args.get("filters")),
        aggregations=_string_list(args.get("aggregations")),
        grouping=_string_list(args.get("grouping")),
        ordering=_string_list(args.get("ordering")),
        limit=_optional_int(args.get("limit")),
        assumptions=_string_list(args.get("assumptions")),
        answer_checks=_string_list(args.get("answer_checks")),
    )

    search_text = " ".join(
        [plan.goal]
        + plan.required_fields
        + plan.candidate_tables
        + plan.filters
        + plan.aggregations
        + plan.grouping
    ).strip()
    table_candidates = search_schema(
        schema,
        query=search_text or plan.goal,
        limit=MAX_CANDIDATE_TABLES,
    )
    resolved_tables = _resolve_candidate_tables(schema, plan.candidate_tables)
    field_candidates = _field_candidates(
        schema,
        plan.required_fields,
        candidate_tables=resolved_tables["resolved_tables"],
    )
    join_paths = _required_join_paths(schema, plan)

    if not plan.answer_checks:
        plan.answer_checks = [f"include {field}" for field in plan.required_fields]

    result = {
        "ok": True,
        "plan": plan.to_dict(),
        "route": _classify_plan(plan),
        "resolved_tables": resolved_tables["resolved_tables"],
        "ambiguous_tables": resolved_tables["ambiguous_tables"],
        "unknown_tables": resolved_tables["unknown_tables"],
        "table_candidates": table_candidates["tables"],
        "field_candidates": field_candidates,
        "join_paths": join_paths,
        "next_steps": _next_steps(resolved_tables, field_candidates, join_paths),
    }

    if run_state is not None:
        for field_name, candidates in field_candidates.items():
            run_state.record_candidate_columns(field_name, candidates)
        for path_result in join_paths:
            run_state.record_join_paths(path_result)
        run_state.record_plan(plan, result)

    return result


def _resolve_candidate_tables(
    schema: Dict[str, Any], table_names: List[str]
) -> Dict[str, List[Any]]:
    tables = schema.get("tables", []) or []
    resolved: List[str] = []
    ambiguous: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []
    for name in table_names:
        matches = _matching_tables(tables, name)
        if len(matches) == 1:
            table_name = str(matches[0].get("name"))
            if table_name not in resolved:
                resolved.append(table_name)
        elif len(matches) > 1:
            ambiguous.append(
                {
                    "table": name,
                    "matches": [_table_ref(match) for match in matches[:5]],
                }
            )
        else:
            unknown.append(
                {
                    "table": name,
                    "candidates": search_schema(schema, query=name, limit=5)["tables"],
                }
            )
    return {
        "resolved_tables": resolved,
        "ambiguous_tables": ambiguous,
        "unknown_tables": unknown,
    }


def _field_candidates(
    schema: Dict[str, Any],
    field_names: List[str],
    *,
    candidate_tables: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    if not field_names:
        return {}
    table_filter = {name.lower() for name in candidate_tables}
    results: Dict[str, List[Dict[str, Any]]] = {}
    for field_name in field_names:
        tokens = _tokens(field_name)
        matches = []
        for table in schema.get("tables", []) or []:
            table_name = str(table.get("name") or "")
            if table_filter and table_name.lower() not in table_filter:
                continue
            for column in table.get("columns", []) or []:
                column_name = str(column.get("name") or "")
                score = _column_score(column_name, tokens)
                if score <= 0:
                    continue
                matches.append(
                    {
                        "table": table_name,
                        "column": column_name,
                        "type": column.get("type"),
                        "score": score,
                    }
                )
        matches.sort(key=lambda item: (-item["score"], item["table"], item["column"]))
        results[field_name] = matches[:MAX_FIELD_CANDIDATES]
    return results


def _required_join_paths(
    schema: Dict[str, Any], plan: DbQueryPlan
) -> List[Dict[str, Any]]:
    path_results = []
    for join in plan.required_joins:
        from_tables = _string_list(join.get("from_tables") or join.get("from"))
        to_tables = _string_list(join.get("to_tables") or join.get("to"))
        if not from_tables or not to_tables:
            continue
        path_results.append(
            find_join_paths(
                schema,
                from_tables=from_tables,
                to_tables=to_tables,
                max_hops=join.get("max_hops", 4),
                max_paths=join.get("max_paths", 3),
            )
        )
    return path_results


def _next_steps(
    resolved_tables: Dict[str, List[Any]],
    field_candidates: Dict[str, List[Dict[str, Any]]],
    join_paths: List[Dict[str, Any]],
) -> List[str]:
    steps = []
    if resolved_tables["unknown_tables"] or resolved_tables["ambiguous_tables"]:
        steps.append("inspect or search unresolved candidate tables")
    missing_fields = [
        field_name
        for field_name, candidates in field_candidates.items()
        if not candidates
    ]
    if missing_fields:
        steps.append("inspect schema for fields: " + ", ".join(missing_fields[:5]))
    if any(not path.get("reachable") for path in join_paths):
        steps.append("revise required joins or inspect relationships")
    if not steps:
        steps.append("validate SQL generated from this plan before execution")
    return steps


def _classify_plan(plan: DbQueryPlan) -> str:
    if plan.aggregations or plan.grouping:
        return "aggregation"
    if plan.required_joins:
        return "join_query"
    text = plan.goal.lower()
    if any(term in text for term in ("schema", "table", "column", "relationship")):
        return "schema_question"
    return "data_query"


def _matching_tables(
    tables: List[Dict[str, Any]], table_name: str
) -> List[Dict[str, Any]]:
    wanted = table_name.strip().lower()
    if not wanted:
        return []
    exact = [table for table in tables if str(table.get("name", "")).lower() == wanted]
    if exact:
        return exact
    return [
        table
        for table in tables
        if str(table.get("name", "")).split(".")[-1].lower() == wanted
    ]


def _table_ref(table: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": table.get("name"),
        "row_count": table.get("row_count"),
        "column_count": len(table.get("columns", []) or []),
    }


def _join_requirements(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append({"from_tables": [item[0]], "to_tables": [item[1]]})
    return out


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _optional_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _tokens(value: str) -> List[str]:
    raw = re.findall(r"[a-zA-Z0-9_]+", value.lower())
    tokens = []
    for item in raw:
        tokens.extend(part for part in re.split(r"[_\W]+", item) if part)
    return [token for token in tokens if len(token) > 1]


def _column_score(column_name: str, tokens: List[str]) -> int:
    if not tokens:
        return 0
    column = column_name.lower()
    column_parts = set(part for part in re.split(r"[_\W]+", column) if part)
    score = 0
    for token in tokens:
        if token == column:
            score += 6
        elif token in column_parts:
            score += 3
        elif token in column:
            score += 1
    return score
