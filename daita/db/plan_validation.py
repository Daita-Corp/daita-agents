"""Deterministic validation for DB query plan proposals."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .query_plan import DbQueryPlan, DbQueryPlanValidation
from .query_sql_validation import sql_fingerprint
from .sql_analysis import SqlAnalysisError, analyze_sql


class DbQueryPlanValidator:
    """Validate structured plan proposals before SQL validation runs."""

    def validate(
        self,
        plan: DbQueryPlan,
        planning_context: dict[str, Any],
    ) -> DbQueryPlanValidation:
        errors: list[str] = []
        warnings: list[str] = []
        schema = dict(planning_context.get("schema") or {})
        dialect = str(
            planning_context.get("dialect") or schema.get("database_type") or ""
        )
        table_columns = _table_columns(schema)
        sql = plan.selected_sql

        if plan.confidence < 0 or plan.confidence > 1:
            errors.append("confidence_out_of_bounds")
        if plan.operation == "read" and not plan.clarification_question and not sql:
            errors.append("missing_selected_sql")
        if plan.clarification_question:
            warnings.append("clarification_requested")
        missing_tables = sorted(
            table
            for table in plan.selected_tables
            if table.lower() not in table_columns
        )
        if missing_tables:
            errors.append(f"unknown_selected_tables:{','.join(missing_tables)}")
        for join in plan.joins:
            _validate_column(
                table_columns,
                join.left_table,
                join.left_column,
                errors,
                label="join_left",
            )
            _validate_column(
                table_columns,
                join.right_table,
                join.right_column,
                errors,
                label="join_right",
            )
        for filter_spec in plan.filters:
            table, column = _split_column_ref(filter_spec.column)
            if table:
                _validate_column(table_columns, table, column, errors, label="filter")

        if sql:
            try:
                analysis = analyze_sql(sql, dialect=dialect)
            except SqlAnalysisError as exc:
                errors.append(f"sql_parse_failed:{exc.error_type}")
            else:
                if analysis.has_multiple_statements:
                    errors.append("sql_multiple_statements")
                if plan.operation == "read" and (
                    not analysis.is_read or analysis.mutating_statement_types
                ):
                    errors.append("read_plan_contains_non_read_sql")
                sql_tables = sorted(
                    {table.short_key for table in analysis.tables if not table.is_cte}
                )
                unknown_sql_tables = [
                    table for table in sql_tables if table.lower() not in table_columns
                ]
                if unknown_sql_tables:
                    errors.append(
                        "sql_references_unknown_tables:"
                        + ",".join(sorted(unknown_sql_tables))
                    )
                if plan.selected_tables:
                    missing_from_sql = [
                        table
                        for table in plan.selected_tables
                        if table.lower() not in {item.lower() for item in sql_tables}
                    ]
                    if missing_from_sql:
                        errors.append(
                            "selected_tables_not_in_sql:"
                            + ",".join(sorted(missing_from_sql))
                        )
                if (
                    plan.operation == "read"
                    and not analysis.has_limit
                    and not plan.aggregations
                    and not _looks_aggregated(sql)
                ):
                    warnings.append("row_returning_query_without_explicit_limit")

        fingerprint = _fingerprint(plan.to_dict())
        return DbQueryPlanValidation(
            valid=not errors,
            accepted_sql=sql if not errors else None,
            sql_fingerprint=sql_fingerprint(sql) if sql else None,
            errors=tuple(errors),
            warnings=tuple(warnings),
            plan_fingerprint=fingerprint,
            metadata={
                "validator": "deterministic",
                "schema_fingerprint": planning_context.get("schema_fingerprint"),
            },
        )


def _table_columns(schema: dict[str, Any]) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for table in schema.get("tables", []) or []:
        name = str(table.get("name") or "")
        if not name:
            continue
        tables[name.lower()] = {
            str(column.get("name") or "").lower()
            for column in table.get("columns", []) or []
            if column.get("name")
        }
    return tables


def _validate_column(
    table_columns: dict[str, set[str]],
    table: str,
    column: str,
    errors: list[str],
    *,
    label: str,
) -> None:
    table_key = table.lower()
    if table_key not in table_columns:
        errors.append(f"{label}_unknown_table:{table}")
        return
    if column.lower() not in table_columns[table_key]:
        errors.append(f"{label}_unknown_column:{table}.{column}")


def _split_column_ref(value: str) -> tuple[str | None, str]:
    parts = [part.strip('"`[] ') for part in str(value).split(".") if part]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, parts[-1] if parts else ""


def _looks_aggregated(sql: str) -> bool:
    lowered = sql.lower()
    return any(token in lowered for token in ("count(", "sum(", "avg(", "min(", "max("))


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
