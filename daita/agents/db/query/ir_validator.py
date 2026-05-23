"""Query IR validation for ``from_db`` agents."""

from __future__ import annotations

import re
from typing import Any

from ..catalog_profile import is_numeric_type
from .metadata import column_name, schema_table_columns, table_name
from .ir import FieldRef, QueryPlan


def validate_query_plan(
    plan: QueryPlan,
    schema: dict[str, Any],
    *,
    dialect: str = "",
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    table_columns = schema_table_columns(schema)
    column_types = _schema_column_types(schema)

    for field in _plan_field_refs(plan):
        if field.table.lower() not in table_columns:
            errors.append({"type": "unknown_table", "table": field.table})
            continue
        if field.column.lower() not in table_columns[field.table.lower()]:
            errors.append(
                {
                    "type": "unknown_column",
                    "table": field.table,
                    "column": field.column,
                }
            )

    for metric in plan.metrics:
        table = metric.table.lower()
        if table not in table_columns:
            errors.append({"type": "unknown_metric_table", "table": metric.table})
            continue
        if metric.kind not in {"count", "distinct_count", "sum", "avg", "min", "max"}:
            errors.append(
                {"type": "unsupported_metric_kind", "metric": metric.to_dict()}
            )
            continue
        if metric.kind in {"sum", "avg", "min", "max", "distinct_count"}:
            if not metric.column:
                errors.append(
                    {"type": "metric_requires_column", "metric": metric.to_dict()}
                )
                continue
            if metric.column.lower() not in table_columns[table]:
                errors.append(
                    {
                        "type": "unknown_metric_column",
                        "metric": metric.to_dict(),
                    }
                )
                continue
        if metric.kind in {"sum", "avg"} and metric.column:
            type_name = column_types.get((table, metric.column.lower()))
            if not is_numeric_type(type_name):
                errors.append(
                    {
                        "type": "non_numeric_metric_column",
                        "metric": metric.to_dict(),
                        "column_type": type_name,
                    }
                )
        if metric.kind == "count" and not metric.column:
            warnings.append(
                {
                    "type": "count_uses_star",
                    "metric": metric.name,
                    "guidance": "Prefer a primary key or stable row identifier when available.",
                }
            )

    relation_error = _relation_graph_error(plan)
    if relation_error is not None:
        errors.append(relation_error)

    date_macros = [
        item.value.get("macro")
        for item in plan.filters
        if isinstance(item.value, dict) and item.value.get("macro")
    ]
    unsupported_macros = [
        macro for macro in date_macros if not _date_macro_supported(str(macro), dialect)
    ]
    if unsupported_macros:
        errors.append(
            {
                "type": "unsupported_date_macro",
                "dialect": dialect,
                "macros": unsupported_macros,
            }
        )

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _plan_field_refs(plan: QueryPlan) -> list[FieldRef]:
    fields = list(plan.grain)
    fields.extend(item.field for item in plan.filters)
    for join in plan.joins:
        fields.extend([join.left, join.right])
    return fields


def _relation_graph_error(plan: QueryPlan) -> dict[str, Any] | None:
    referenced_tables = _referenced_tables(plan)
    if len(referenced_tables) <= 1:
        return None
    connected = _connected_join_tables(plan, referenced_tables)
    missing = sorted(table for table in referenced_tables if table not in connected)
    if missing:
        return {
            "type": "unjoined_table_reference",
            "tables": sorted(referenced_tables),
            "missing_tables": missing,
            "guidance": (
                "Every table referenced by selected fields, metrics, filters, "
                "or ordering must be connected by QueryPlan.joins before SQL "
                "compilation."
            ),
        }
    return None


def _referenced_tables(plan: QueryPlan) -> set[str]:
    tables = {field.table for field in _plan_field_refs(plan) if field.table}
    tables.update(metric.table for metric in plan.metrics if metric.table)
    for order in plan.order_by:
        if isinstance(order.field, FieldRef) and order.field.table:
            tables.add(order.field.table)
    return tables


def _connected_join_tables(plan: QueryPlan, referenced_tables: set[str]) -> set[str]:
    if not plan.joins:
        return set()
    adjacency: dict[str, set[str]] = {}
    for join in plan.joins:
        left = join.left.table
        right = join.right.table
        if not left or not right:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    start = next(iter(referenced_tables))
    seen = {start}
    queue = [start]
    while queue:
        table = queue.pop(0)
        for neighbor in adjacency.get(table, set()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def _schema_column_types(schema: dict[str, Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for table in schema.get("tables") or []:
        name = table_name(table).lower()
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            col_name = column_name(column).lower()
            if name and col_name:
                out[(name, col_name)] = str(column.get("type") or "")
    return out


def _date_macro_supported(macro: str, dialect: str) -> bool:
    supported = {
        "postgresql",
        "postgres",
        "sqlite",
        "mysql",
        "snowflake",
    }
    return (dialect or "").lower() in supported and (
        macro
        in {
            "start_of_current_month",
            "start_of_last_month",
            "start_of_next_month",
        }
        or bool(re.match(r"last_[1-9]\d*_days$", macro))
    )
