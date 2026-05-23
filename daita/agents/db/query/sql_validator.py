"""SQL preflight validation for ``from_db`` agents."""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any

from .metadata import (
    candidate_column_matches_required,
    field_ref_matches_required,
    metric_matches_required,
    normalize_identifier,
    schema_table_columns,
)
from .intent import is_count_metric_name
from .ir import QueryPlan
from .sql_analysis import SqlAnalysis, SqlAnalysisError, analyze_sql


def validate_sql_against_schema(
    sql: str,
    schema: dict[str, Any],
    *,
    dialect: str = "",
    analysis: SqlAnalysis | None = None,
) -> dict[str, Any]:
    if not sql.strip():
        return _validation_error(
            "Missing SQL query",
            error_type="dialect_parse_error",
            sql=sql,
            suggested_next_tool="db_validate_sql",
        )

    if analysis is None:
        try:
            analysis = analyze_sql(
                sql, dialect=dialect or str(schema.get("database_type") or "")
            )
        except SqlAnalysisError as exc:
            return _validation_error(
                str(exc),
                error_type=exc.error_type,
                sql=sql,
                suggested_next_tool="db_validate_sql",
            )

    table_columns = schema_table_columns(schema)
    if not table_columns:
        return {"ok": True, "sql_fingerprint": sql_fingerprint(sql)}

    unknown_tables = _unknown_tables(analysis, table_columns)
    missing_columns = _missing_columns(analysis, table_columns)

    if not unknown_tables and not missing_columns:
        return _validation_ok(sql, analysis)

    inspect_tables = _tables_to_inspect(unknown_tables, missing_columns)
    return {
        "error": "SQL preflight failed against known schema",
        "error_type": "schema_reference_error",
        "repair_required": True,
        "preflight_failed": True,
        "sql_fingerprint": sql_fingerprint(sql),
        "unknown_tables": sorted(unknown_tables),
        "missing_columns": missing_columns,
        "inspect_tables": inspect_tables,
        "available_tables": sorted(table_columns)[:50],
        "available_columns": _available_columns_for_tables(
            table_columns, inspect_tables
        ),
        "column_candidates": _column_candidates(missing_columns, table_columns),
        "table_candidates": _table_candidates(unknown_tables, table_columns),
        "suggested_next_tool": (
            "catalog_inspect_table" if inspect_tables else "catalog_search_schema"
        ),
        "do_not_retry_same_sql": True,
        "guidance": (
            "Do not call db_query again with this exact SQL. Inspect the missing "
            "or ambiguous table, use catalog_find_join_paths for join ambiguity, "
            "then call db_validate_sql or db_query with corrected SQL."
        ),
    }


def required_field_warnings_for_plan(
    plan: QueryPlan | None, required_fields: list[str]
) -> list[dict[str, Any]]:
    if plan is None or not required_fields:
        return []

    metric_aliases = {
        normalize_identifier(metric.name): metric for metric in plan.metrics
    }
    warnings: list[dict[str, Any]] = []
    for field_name in required_fields:
        normalized = normalize_identifier(field_name)
        if not normalized:
            continue
        if any(field_ref_matches_required(field, field_name) for field in plan.grain):
            continue
        metric = metric_aliases.get(normalized)
        if metric is not None:
            continue
        if any(metric_matches_required(metric, field_name) for metric in plan.metrics):
            continue
        if is_count_metric_name(field_name) and any(
            metric.kind == "count" and normalize_identifier(metric.name) == normalized
            for metric in plan.metrics
        ):
            continue
        warnings.append(
            {
                "required_field": field_name,
                "reason": "required field not represented by query plan IR",
            }
        )
    return warnings


def apply_required_field_validation(
    validation: dict[str, Any],
    sql: str,
    run_state: Any,
    *,
    dialect: str = "",
    analysis: SqlAnalysis | None = None,
) -> None:
    warnings = required_field_warnings(
        sql, run_state, dialect=dialect, analysis=analysis
    )
    if not warnings:
        return
    validation["required_field_warnings"] = warnings
    if validation.get("error"):
        validation["guidance"] = (
            f"{validation.get('guidance', '')} Preserve required fields from "
            "the query plan when revising SQL."
        ).strip()
        return
    validation.update(
        {
            "error": "SQL does not preserve required fields from query plan",
            "error_type": "required_field_error",
            "repair_required": True,
            "preflight_failed": True,
            "sql_fingerprint": sql_fingerprint(sql),
            "suggested_next_tool": "db_plan_query",
            "do_not_retry_same_sql": True,
            "guidance": (
                "Revise the SQL so selected expressions use the required fields "
                "from db_plan_query. Do not substitute different metrics while "
                "keeping the original alias."
            ),
        }
    )


def required_field_warnings(
    sql: str,
    run_state: Any,
    *,
    dialect: str = "",
    analysis: SqlAnalysis | None = None,
) -> list[dict[str, Any]]:
    if run_state is None:
        return []
    required_fields = getattr(run_state, "required_answer_fields", None) or []
    candidate_columns = getattr(run_state, "candidate_columns", None) or {}
    if not required_fields or not isinstance(candidate_columns, dict):
        return []

    if analysis is None:
        analysis = analyze_sql(sql, dialect=dialect)
    referenced_columns = analysis.referenced_column_names
    selected_by_alias = {
        normalize_identifier(item.alias): item
        for item in analysis.select_items
        if item.alias
    }
    warnings: list[dict[str, Any]] = []
    for field_name in required_fields:
        candidates = candidate_columns.get(field_name) or []
        required_columns = _high_confidence_required_columns(field_name, candidates)
        if not required_columns:
            continue
        if referenced_columns & required_columns:
            continue
        if _count_alias_satisfies_required_field(field_name, selected_by_alias):
            continue
        warnings.append(
            {
                "required_field": field_name,
                "expected_columns": sorted(required_columns)[:8],
                "reason": "required field not referenced by SQL",
            }
        )
    return warnings


def sql_fingerprint(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", (sql or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _validation_ok(sql: str, analysis: SqlAnalysis) -> dict[str, Any]:
    return {
        "ok": True,
        "sql_fingerprint": sql_fingerprint(sql),
        "referenced_tables": sorted(
            {table.key for table in analysis.tables if not table.is_cte}
        ),
        "referenced_columns": sorted(analysis.referenced_column_names),
        "selected_columns": _selected_output_columns(analysis),
    }


def _selected_output_columns(analysis: SqlAnalysis) -> list[str]:
    columns: list[str] = []
    for item in analysis.select_items:
        name = item.alias or _simple_selected_expression_name(item.expression_sql)
        if name and name not in columns:
            columns.append(name)
    return columns


def _simple_selected_expression_name(expression_sql: str) -> str:
    expression = str(expression_sql or "").strip().strip('"`[]')
    if not expression:
        return ""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", expression):
        return expression.rsplit(".", 1)[-1]
    return ""


def _validation_error(
    message: str, *, error_type: str, sql: str, suggested_next_tool: str
) -> dict[str, Any]:
    return {
        "error": message,
        "error_type": error_type,
        "repair_required": True,
        "preflight_failed": True,
        "suggested_next_tool": suggested_next_tool,
        "sql_fingerprint": sql_fingerprint(sql),
    }


def _unknown_tables(
    analysis: SqlAnalysis, table_columns: dict[str, set[str]]
) -> set[str]:
    known = {name.lower() for name in table_columns}
    unknown: set[str] = set()
    for table in analysis.tables:
        if table.is_cte:
            continue
        if table.key in known or table.short_key in known:
            continue
        unknown.add(table.key)
    return unknown


def _missing_columns(
    analysis: SqlAnalysis, table_columns: dict[str, set[str]]
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    table_count = len([table for table in analysis.tables if not table.is_cte])
    select_aliases = {
        normalize_identifier(item.alias) for item in analysis.select_items if item.alias
    }
    for column in analysis.columns:
        column_name = column.key
        if normalize_identifier(column.name) in select_aliases:
            continue
        if column.qualifier_key:
            table_key = _known_table_key(column.qualifier_key, table_columns)
            if table_key is None:
                continue
            if column_name not in table_columns.get(table_key, set()):
                key = (table_key, column_name)
                if key not in seen:
                    missing.append(
                        {
                            "table": table_key,
                            "column": column.name,
                            "reason": "column not found",
                        }
                    )
                    seen.add(key)
            continue

        if table_count == 1:
            table = next(table for table in analysis.tables if not table.is_cte)
            table_key = _known_table_key(table.key, table_columns)
            if table_key and column_name not in table_columns.get(table_key, set()):
                key = (table_key, column_name)
                if key not in seen:
                    missing.append(
                        {
                            "table": table_key,
                            "column": column.name,
                            "reason": "column not found",
                        }
                    )
                    seen.add(key)
    return missing


def _known_table_key(table_key: str, table_columns: dict[str, set[str]]) -> str | None:
    key = table_key.lower()
    if key in table_columns:
        return key
    short = key.split(".")[-1]
    if short in table_columns:
        return short
    return None


def _count_alias_satisfies_required_field(
    field_name: str, selected_expressions: dict[str, Any]
) -> bool:
    normalized_field = normalize_identifier(field_name)
    selected = selected_expressions.get(normalized_field)
    return bool(selected and is_count_metric_name(field_name) and selected.is_count)


def _high_confidence_required_columns(
    field_name: str, candidates: list[dict[str, Any]]
) -> set[str]:
    out: set[str] = set()
    for candidate in candidates:
        column = str(candidate.get("column") or "").strip().lower()
        if not column:
            continue
        score = int(candidate.get("score") or 0)
        if candidate_column_matches_required(
            field_name, column, min_score=6, score=score
        ):
            out.add(column)
    return out


def _tables_to_inspect(
    unknown_tables: set[str], missing_columns: list[dict[str, str]]
) -> list[str]:
    tables: list[str] = []
    for table in sorted(unknown_tables):
        if table not in tables:
            tables.append(table)
    for item in missing_columns:
        table = item.get("table")
        if table and table not in tables:
            tables.append(table)
    return tables[:5]


def _available_columns_for_tables(
    table_columns: dict[str, set[str]], tables: list[str]
) -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    for table in tables:
        table_l = table.lower()
        if table_l in table_columns:
            available[table] = sorted(table_columns[table_l])[:80]
    return available


def _column_candidates(
    missing_columns: list[dict[str, str]], table_columns: dict[str, set[str]]
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for item in missing_columns:
        table = item.get("table")
        column = item.get("column")
        if not table or not column:
            continue
        columns = sorted(table_columns.get(table.lower(), set()))
        if not columns:
            continue
        matches = difflib.get_close_matches(column.lower(), columns, n=5, cutoff=0.45)
        identity_matches = [
            col
            for col in columns
            if any(term in col for term in ("name", "email", "user", "id"))
        ][:8]
        combined = []
        for candidate in matches + identity_matches:
            if candidate not in combined:
                combined.append(candidate)
        if combined:
            candidates[f"{table}.{column}"] = combined[:8]
    return candidates


def _table_candidates(
    unknown_tables: set[str], table_columns: dict[str, set[str]]
) -> dict[str, list[str]]:
    tables = sorted(table_columns)
    candidates: dict[str, list[str]] = {}
    for table in sorted(unknown_tables):
        matches = difflib.get_close_matches(table, tables, n=5, cutoff=0.45)
        if matches:
            candidates[table] = matches
    return candidates
