"""SQL preflight validation for ``from_db`` agents."""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any

from .metadata import (
    field_ref_matches_required,
    metric_matches_required,
    normalize_identifier,
    schema_table_columns,
)
from .intent import is_count_metric_name
from .ir import QueryPlan
from .requirements import (
    AnswerRequirement,
    output_satisfies_requirement,
    requirement_covers_field,
    requirement_covers_metric,
)
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
    plan: QueryPlan | None, requirements: list[AnswerRequirement]
) -> list[dict[str, Any]]:
    if plan is None or not requirements:
        return []

    metric_aliases = {
        normalize_identifier(metric.name): metric for metric in plan.metrics
    }
    warnings: list[dict[str, Any]] = []
    for requirement in requirements:
        field_name = requirement.raw
        normalized = normalize_identifier(field_name)
        if not normalized:
            continue
        if requirement.kind == "aggregate":
            if any(
                requirement_covers_metric(requirement, metric)
                for metric in plan.metrics
            ):
                continue
            warnings.append(
                {
                    "required_field": field_name,
                    "reason": "required aggregate not represented by query plan IR",
                }
            )
            continue
        if requirement.kind == "field" and any(
            requirement_covers_field(requirement, field) for field in plan.grain
        ):
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
    requirements = getattr(run_state, "answer_requirements", None) or []
    if not requirements:
        return []

    if analysis is None:
        analysis = analyze_sql(sql, dialect=dialect)
    selected_outputs = _selected_output_columns(analysis)
    warnings: list[dict[str, Any]] = []
    for requirement in requirements:
        source_required = bool(requirement.source_columns)
        source_present = (
            _analysis_references_sources(analysis, requirement)
            if source_required
            else True
        )
        output_present = any(
            output_satisfies_requirement(requirement, output)
            for output in selected_outputs
        )
        if source_present and output_present:
            continue
        if (
            source_present
            and requirement.kind == "aggregate"
            and not requirement.output_name
        ):
            continue
        reason = (
            "required source column not referenced by SQL"
            if not source_present
            else "required output field not selected by SQL"
        )
        warnings.append(
            {
                "required_field": requirement.raw,
                "expected_source_columns": [
                    {
                        "table": source.table,
                        "column": source.column,
                    }
                    for source in requirement.source_columns
                ],
                "expected_outputs": list(requirement.acceptable_outputs)
                or [requirement.output_name],
                "reason": reason,
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
    expression = re.sub(r'["`\[\]]', "", str(expression_sql or "").strip())
    if not expression:
        return ""
    if re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
        expression,
    ):
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


def _analysis_references_sources(
    analysis: SqlAnalysis, requirement: AnswerRequirement
) -> bool:
    for source in requirement.source_columns:
        if not source.column:
            continue
        source_column = source.column.lower()
        source_table = source.table.lower()
        for column in analysis.columns:
            if column.key != source_column:
                continue
            if not source_table:
                return True
            qualifier = column.qualifier_key
            if not qualifier:
                return True
            if (
                qualifier == source_table
                or qualifier.split(".")[-1] == source_table.split(".")[-1]
            ):
                return True
        return False
    return True


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
