"""SQL compiler for validated ``from_db`` query IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import FieldRef, Filter, Metric, OrderBy, QueryPlan


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    dialect: str

    def to_dict(self) -> dict[str, Any]:
        return {"sql": self.sql, "dialect": self.dialect}


def compile_query_plan(plan: QueryPlan, dialect: str) -> CompiledQuery:
    """Compile a small, deterministic analytics query plan to SQL."""

    normalized_dialect = _normalize_dialect(dialect)
    tables = _ordered_tables(plan)
    if not tables:
        raise ValueError("QueryPlan must include at least one table")

    select_items = [_field_sql(field) for field in plan.grain]
    select_items.extend(_metric_sql(metric) for metric in plan.metrics)
    if not select_items:
        select_items.append("*")

    base_table = tables[0]
    lines = ["SELECT", "  " + ",\n  ".join(select_items), f"FROM {_quote(base_table)}"]
    for join in plan.joins:
        right_table = join.right.table
        lines.append(
            "JOIN "
            f"{_quote(right_table)} ON "
            f"{_field_sql(join.left)} = {_field_sql(join.right)}"
        )

    if plan.filters:
        lines.append(
            "WHERE "
            + " AND ".join(
                _filter_sql(item, normalized_dialect) for item in plan.filters
            )
        )

    if plan.grain and plan.metrics:
        lines.append("GROUP BY " + ", ".join(_field_sql(field) for field in plan.grain))

    if plan.order_by:
        lines.append(
            "ORDER BY "
            + ", ".join(_order_by_sql(item) for item in plan.order_by if item.field)
        )

    if plan.limit:
        lines.append(f"LIMIT {int(plan.limit)}")

    return CompiledQuery(sql="\n".join(lines), dialect=normalized_dialect)


def _ordered_tables(plan: QueryPlan) -> list[str]:
    tables: list[str] = []
    for metric in plan.metrics:
        _append_unique(tables, metric.table)
    for item in plan.filters:
        _append_unique(tables, item.field.table)
    for field in plan.grain:
        _append_unique(tables, field.table)
    for join in plan.joins:
        _append_unique(tables, join.left.table)
        _append_unique(tables, join.right.table)
    return tables


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _metric_sql(metric: Metric) -> str:
    if metric.kind == "count":
        target = (
            _field_sql(FieldRef(metric.table, metric.column)) if metric.column else "*"
        )
        return f"COUNT({target}) AS {_quote(metric.name)}"
    if metric.kind == "distinct_count":
        if not metric.column:
            raise ValueError("distinct_count metrics require a source column")
        return f"COUNT(DISTINCT {_field_sql(FieldRef(metric.table, metric.column))}) AS {_quote(metric.name)}"
    if metric.kind in {"sum", "avg", "min", "max"}:
        if not metric.column:
            raise ValueError(f"{metric.kind} metrics require a source column")
        return (
            f"{metric.kind.upper()}({_field_sql(FieldRef(metric.table, metric.column))}) "
            f"AS {_quote(metric.name)}"
        )
    raise ValueError(f"Unsupported metric kind: {metric.kind}")


def _filter_sql(item: Filter, dialect: str) -> str:
    field = _field_sql(item.field)
    operator = item.operator.lower()
    value = item.value
    if isinstance(value, dict) and "macro" in value:
        compiled_value = _date_macro_sql(str(value["macro"]), dialect)
    elif operator == "in" and isinstance(value, list):
        compiled_value = "(" + ", ".join(_literal_sql(part) for part in value) + ")"
    elif operator == "between" and isinstance(value, list) and len(value) == 2:
        return f"{field} BETWEEN {_literal_sql(value[0])} AND {_literal_sql(value[1])}"
    else:
        compiled_value = _literal_sql(value)
    return f"{field} {item.operator.upper()} {compiled_value}"


def _order_by_sql(item: OrderBy) -> str:
    field = (
        _field_sql(item.field)
        if isinstance(item.field, FieldRef)
        else _quote(item.field)
    )
    return f"{field} {item.direction.upper()}"


def _field_sql(field: FieldRef) -> str:
    return f"{_quote(field.table)}.{_quote(field.column)}"


def _quote(identifier: str) -> str:
    parts = [part for part in str(identifier or "").split(".") if part]
    return ".".join(f'"{part.replace(chr(34), chr(34) + chr(34))}"' for part in parts)


def _literal_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _date_macro_sql(macro: str, dialect: str) -> str:
    if dialect in {"postgresql", "postgres"}:
        return _postgres_date_macro(macro)
    if dialect == "sqlite":
        return _sqlite_date_macro(macro)
    if dialect == "mysql":
        return _mysql_date_macro(macro)
    if dialect == "snowflake":
        return _snowflake_date_macro(macro)
    raise ValueError(f"Unsupported date macro for dialect: {dialect}")


def _postgres_date_macro(macro: str) -> str:
    if macro == "start_of_current_month":
        return "date_trunc('month', current_date)"
    if macro == "start_of_last_month":
        return "date_trunc('month', current_date) - interval '1 month'"
    if macro == "start_of_next_month":
        return "date_trunc('month', current_date) + interval '1 month'"
    if macro.startswith("last_") and macro.endswith("_days"):
        days = _macro_int(macro, prefix="last_", suffix="_days")
        return f"current_date - interval '{days} days'"
    raise ValueError(f"Unsupported date macro: {macro}")


def _sqlite_date_macro(macro: str) -> str:
    if macro == "start_of_current_month":
        return "date('now', 'start of month')"
    if macro == "start_of_last_month":
        return "date('now', 'start of month', '-1 month')"
    if macro == "start_of_next_month":
        return "date('now', 'start of month', '+1 month')"
    if macro.startswith("last_") and macro.endswith("_days"):
        days = _macro_int(macro, prefix="last_", suffix="_days")
        return f"date('now', '-{days} days')"
    raise ValueError(f"Unsupported date macro: {macro}")


def _mysql_date_macro(macro: str) -> str:
    if macro == "start_of_current_month":
        return "DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')"
    if macro == "start_of_last_month":
        return "DATE_FORMAT(CURRENT_DATE - INTERVAL 1 MONTH, '%Y-%m-01')"
    if macro == "start_of_next_month":
        return "DATE_FORMAT(CURRENT_DATE + INTERVAL 1 MONTH, '%Y-%m-01')"
    if macro.startswith("last_") and macro.endswith("_days"):
        days = _macro_int(macro, prefix="last_", suffix="_days")
        return f"CURRENT_DATE - INTERVAL {days} DAY"
    raise ValueError(f"Unsupported date macro: {macro}")


def _snowflake_date_macro(macro: str) -> str:
    if macro == "start_of_current_month":
        return "DATE_TRUNC('MONTH', CURRENT_DATE())"
    if macro == "start_of_last_month":
        return "DATEADD(month, -1, DATE_TRUNC('MONTH', CURRENT_DATE()))"
    if macro == "start_of_next_month":
        return "DATEADD(month, 1, DATE_TRUNC('MONTH', CURRENT_DATE()))"
    if macro.startswith("last_") and macro.endswith("_days"):
        days = _macro_int(macro, prefix="last_", suffix="_days")
        return f"DATEADD(day, -{days}, CURRENT_DATE())"
    raise ValueError(f"Unsupported date macro: {macro}")


def _macro_int(macro: str, *, prefix: str, suffix: str) -> int:
    value = macro.removeprefix(prefix).removesuffix(suffix)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Date macro must use a positive interval: {macro}")
    return parsed


def _normalize_dialect(dialect: str) -> str:
    value = (dialect or "").lower()
    if value == "postgres":
        return "postgresql"
    return value or "unknown"
