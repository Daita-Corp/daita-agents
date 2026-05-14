"""
Structured query planning helpers for ``Agent.from_db()``.

The LLM provides intent. This module enriches that intent with catalog facts
from ``navigation.py`` so downstream SQL validation and compilation work from a
shared shape instead of raw prose.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..runtime.state import DbQueryPlan, DbRunState
from ..schema.metadata import (
    matching_tables as _matching_tables,
    schema_table_columns as _schema_table_columns,
    split_identifier as _split_identifier,
)
from ..schema.navigation import find_join_paths, search_schema
from .compiler import compile_query_plan
from .intent import looks_like_count_intent
from .ir_validator import validate_query_plan
from .resolver import resolve_query_plan

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
    plan_warnings = _aggregation_reference_warnings(schema, plan)
    query_resolution = resolve_query_plan(plan, schema)
    query_ir = query_resolution.plan
    ir_validation = (
        validate_query_plan(
            query_ir,
            schema,
            dialect=str(schema.get("database_type") or ""),
        )
        if query_ir is not None
        else {"ok": False, "errors": [], "warnings": []}
    )
    compiled_sql = None
    compiler_warning = None
    if query_ir is not None and ir_validation.get("ok"):
        dialect = str(schema.get("database_type") or "")
        if dialect.lower() in {"postgresql", "postgres", "sqlite"}:
            try:
                compiled_sql = compile_query_plan(query_ir, dialect).sql
            except ValueError as exc:
                compiler_warning = {
                    "type": "query_ir_compile_failed",
                    "message": str(exc),
                }
        else:
            compiler_warning = {
                "type": "query_ir_compiler_unsupported_dialect",
                "dialect": dialect or "unknown",
                "supported_dialects": ["postgresql", "sqlite"],
            }
    if compiler_warning:
        plan_warnings.append(compiler_warning)

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
        "query_ir_resolution": query_resolution.to_dict(),
        "query_ir": query_ir.to_dict() if query_ir is not None else None,
        "compiled_sql": compiled_sql,
        "validation": ir_validation,
        "plan_warnings": plan_warnings,
        "next_steps": _next_steps(
            resolved_tables, field_candidates, join_paths, compiled_sql
        ),
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
        tokens = [token for token in _split_identifier(field_name) if len(token) > 1]
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


def _aggregation_reference_warnings(
    schema: Dict[str, Any], plan: DbQueryPlan
) -> List[Dict[str, Any]]:
    table_columns = _schema_table_columns(schema)
    warnings: List[Dict[str, Any]] = []
    for expression in plan.aggregations:
        for table, column in re.findall(
            r"\b([A-Za-z_][\w.]*)\.([A-Za-z_][\w]*)\b", expression
        ):
            table_key = table.strip().lower()
            column_key = column.strip().lower()
            columns = table_columns.get(table_key)
            if columns is None and "." in table_key:
                columns = table_columns.get(table_key.split(".")[-1])
            if columns is None:
                warnings.append(
                    {
                        "type": "unknown_aggregation_table",
                        "aggregation": expression,
                        "table": table,
                        "suggested_next_tool": "db_search_schema",
                    }
                )
                continue
            if column_key not in columns:
                warning = {
                    "type": "unknown_aggregation_column",
                    "aggregation": expression,
                    "table": table,
                    "column": column,
                    "suggested_next_tool": "db_inspect_table",
                }
                if _looks_like_plan_count_intent(plan, column):
                    warning["guidance"] = (
                        "For count-style questions, count stable rows such as "
                        "COUNT(*) or COUNT(primary_key) and alias the result, "
                        "instead of summing a similarly named column that is not "
                        "present in the fact table."
                    )
                warnings.append(warning)
    return warnings


def _looks_like_plan_count_intent(plan: DbQueryPlan, column_name: str) -> bool:
    text = " ".join(
        [plan.goal, column_name]
        + plan.required_fields
        + plan.aggregations
        + plan.ordering
    )
    return looks_like_count_intent(text)


def _next_steps(
    resolved_tables: Dict[str, List[Any]],
    field_candidates: Dict[str, List[Dict[str, Any]]],
    join_paths: List[Dict[str, Any]],
    compiled_sql: Optional[str] = None,
) -> List[str]:
    steps = []
    if compiled_sql:
        return ["run db_query with compiled_sql"]
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
