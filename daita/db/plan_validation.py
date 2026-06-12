"""Deterministic validation for DB query plan proposals."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .planning_context import planner_eligible_column_value_hint
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
        value_profiles = _value_profiles(planning_context)
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
            _validate_filter_literal(
                value_profiles,
                table,
                column,
                filter_spec.operator,
                filter_spec.value,
                errors,
            )

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
                _validate_declared_filters_present_in_sql(
                    plan.filters,
                    analysis.literal_predicates,
                    sql_tables,
                    errors,
                )
                for predicate in analysis.literal_predicates:
                    _validate_filter_literal(
                        value_profiles,
                        predicate.column.table or None,
                        predicate.column.name,
                        predicate.operator,
                        (
                            list(predicate.values)
                            if predicate.operator == "in"
                            else predicate.values[0]
                        ),
                        errors,
                    )

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


def _validate_filter_literal(
    value_profiles: dict[tuple[str | None, str], set[str]],
    table: str | None,
    column: str,
    operator: str,
    value: Any,
    errors: list[str],
) -> None:
    normalized_operator = str(operator or "").strip().lower()
    if normalized_operator not in {"=", "==", "eq", "in"}:
        return
    if value is None:
        return
    if normalized_operator == "in" and isinstance(value, (list, tuple, set)):
        for item in value:
            _validate_filter_literal(
                value_profiles,
                table,
                column,
                "=",
                item,
                errors,
            )
        return
    literal = str(value)
    candidates = _observed_values(value_profiles, table, column)
    if not candidates or literal.lower() in candidates:
        return
    table_label = table or _single_table_for_column(value_profiles, column) or "unknown"
    errors.append(
        "unobserved_filter_literal:"
        f"{table_label}.{column}={literal};"
        f"candidates={','.join(sorted(candidates))}"
    )


def _validate_declared_filters_present_in_sql(
    filters: tuple[Any, ...],
    predicates: tuple[Any, ...],
    sql_tables: list[str],
    errors: list[str],
) -> None:
    for filter_spec in filters:
        if not _filter_has_sql_literal_shape(filter_spec):
            continue
        if any(
            _predicate_matches_filter(filter_spec, predicate, sql_tables)
            for predicate in predicates
        ):
            continue
        table, column = _split_column_ref(str(filter_spec.column))
        label = f"{table + '.' if table else ''}{column}"
        errors.append(
            "declared_filter_not_in_sql:"
            f"{label}{_normalize_operator(filter_spec.operator)}{filter_spec.value}"
        )


def _filter_has_sql_literal_shape(filter_spec: Any) -> bool:
    column = str(getattr(filter_spec, "column", "") or "")
    operator = _normalize_operator(getattr(filter_spec, "operator", ""))
    if not column or operator not in {"=", "!=", ">", ">=", "<", "<=", "in", "like"}:
        return False
    value = getattr(filter_spec, "value", None)
    if operator == "in":
        return isinstance(value, (list, tuple, set)) and bool(value)
    return value is not None


def _predicate_matches_filter(
    filter_spec: Any,
    predicate: Any,
    sql_tables: list[str],
) -> bool:
    filter_table, filter_column = _split_column_ref(str(filter_spec.column))
    predicate_column = getattr(predicate, "column", None)
    predicate_table = str(getattr(predicate_column, "table", "") or "") or None
    predicate_column_name = str(getattr(predicate_column, "name", "") or "")
    if filter_column.lower() != predicate_column_name.lower():
        return False
    if not _tables_match(filter_table, predicate_table, sql_tables):
        return False
    if _normalize_operator(filter_spec.operator) != _normalize_operator(
        getattr(predicate, "operator", "")
    ):
        return False
    return _filter_values(filter_spec) == _predicate_values(predicate)


def _tables_match(
    filter_table: str | None,
    predicate_table: str | None,
    sql_tables: list[str],
) -> bool:
    if not filter_table:
        return True
    filter_key = _short_table_key(filter_table)
    predicate_key = _short_table_key(predicate_table)
    if predicate_key:
        return predicate_key == filter_key
    return len(sql_tables) == 1 and _short_table_key(sql_tables[0]) == filter_key


def _filter_values(filter_spec: Any) -> tuple[str, ...]:
    value = getattr(filter_spec, "value", None)
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(item).lower() for item in value if item is not None))
    return (str(value).lower(),)


def _predicate_values(predicate: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(value).lower()
            for value in getattr(predicate, "values", ()) or ()
            if value is not None
        )
    )


def _normalize_operator(operator: Any) -> str:
    value = str(operator or "").strip().lower()
    if value in {"==", "eq"}:
        return "="
    if value in {"<>", "neq"}:
        return "!="
    return value


def _observed_values(
    value_profiles: dict[tuple[str | None, str], set[str]],
    table: str | None,
    column: str,
) -> set[str]:
    table_key = _short_table_key(table)
    key = (table_key, column.lower())
    if key in value_profiles:
        return value_profiles[key]
    matches = [
        values
        for (profile_table, profile_column), values in value_profiles.items()
        if profile_column == column.lower()
        and (table_key is None or profile_table in {None, table_key})
    ]
    if len(matches) == 1:
        return set(matches[0])
    return set()


def _single_table_for_column(
    value_profiles: dict[tuple[str | None, str], set[str]],
    column: str,
) -> str | None:
    tables = sorted(
        table
        for (table, profile_column), values in value_profiles.items()
        if profile_column == column.lower() and values and table
    )
    return tables[0] if len(tables) == 1 else None


def _short_table_key(table: str | None) -> str | None:
    if not table:
        return None
    return str(table).split(".")[-1].lower()


def _value_profiles(
    planning_context: dict[str, Any],
) -> dict[tuple[str | None, str], set[str]]:
    profiles: dict[tuple[str | None, str], set[str]] = {}
    for hint in planning_context.get("column_value_hints", []) or []:
        if not isinstance(hint, dict):
            continue
        if not planner_eligible_column_value_hint(hint):
            continue
        table = str(hint.get("table") or "").lower() or None
        column = str(hint.get("column") or "").lower()
        if not column:
            continue
        values = set()
        for item in hint.get("observed_values", []) or []:
            if isinstance(item, dict):
                raw = item.get("value")
            else:
                raw = item
            if raw is not None:
                values.add(str(raw).lower())
        if values:
            profiles[(table, column)] = values
    return profiles


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
