"""Catalog-backed resolver from legacy planning strings to Query IR."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..utils import unique_preserving_order
from .metadata import (
    column_name as _column_name,
    metric_matches_required as _metric_matches_required,
    normalize_identifier as _normalize,
    split_identifier as _split_identifier,
    table_name as _table_name,
)
from .catalog_adapter import find_relationship_paths, search_tables
from .intent import looks_like_count_intent
from .ir import FieldRef, Filter, Join, Metric, OrderBy, QueryPlan


@dataclass
class ResolveResult:
    plan: Optional[QueryPlan] = None
    confidence: str = "none"
    warnings: list[dict[str, Any]] = field(default_factory=list)
    ambiguities: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.plan is not None,
            "confidence": self.confidence,
            "query_ir": self.plan.to_dict() if self.plan else None,
            "warnings": self.warnings,
            "ambiguities": self.ambiguities,
        }


class _SqlPlanningView:
    """Temporary synchronous lookup view over the catalog-approved schema snapshot."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.tables = {
            _table_name(table): table
            for table in schema.get("tables") or []
            if isinstance(table, dict) and _table_name(table)
        }
        self.table_lookup: dict[str, str] = {}
        for name in self.tables:
            self.table_lookup[name.lower()] = name
            self.table_lookup[name.split(".")[-1].lower()] = name

    def resolve_tables(self, names: list[str]) -> list[str]:
        return unique_preserving_order(
            [table for table in (self.resolve_table(name) for name in names) if table]
        )

    def resolve_table(self, name: Any) -> Optional[str]:
        key = str(name or "").strip().lower()
        if not key:
            return None
        return self.table_lookup.get(key)

    def has_column(self, table_name: str, column_ref: str) -> bool:
        return any(
            _column_name(column).lower() == column_ref.lower()
            for column in self._columns(table_name)
            if isinstance(column, dict)
        )

    def resolve_field(
        self, name: str, *, preferred_tables: list[str]
    ) -> Optional[FieldRef]:
        raw = str(name or "").strip()
        if not raw:
            return None
        if "." in raw:
            table_ref, column_ref = raw.rsplit(".", 1)
            table = self.resolve_table(table_ref)
            if table and self.has_column(table, column_ref):
                return FieldRef(table, column_ref)

        tokens = set(_split_identifier(raw))
        matches: list[tuple[int, str, str]] = []
        table_order = preferred_tables + [
            table for table in self.tables if table not in preferred_tables
        ]
        for table in table_order:
            for column in self._columns(table):
                col_name = _column_name(column)
                column_tokens = set(_split_identifier(col_name))
                score = 0
                if _normalize(col_name) == _normalize(raw):
                    score += 8
                score += len(tokens & column_tokens) * 3
                if score:
                    matches.append((score, table, col_name))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], table_order.index(item[1]), item[2]))
        _, table, column = matches[0]
        return FieldRef(table, column)

    def primary_key_or_identity(self, table_name: str) -> Optional[str]:
        columns = self._columns(table_name)
        for column in columns:
            if bool(column.get("is_primary_key")):
                return _column_name(column)
        names = [_column_name(column) for column in columns]
        short = table_name.split(".")[-1].rstrip("s")
        preferred = [f"{short}_id", "id", "uuid", "key"]
        for name in preferred:
            for col_name in names:
                if col_name.lower() == name.lower():
                    return col_name
        for col_name in names:
            lowered = col_name.lower()
            if lowered.endswith("_id") or lowered == "id":
                return col_name
        return None

    def timestamp_column(self, table_name: str) -> Optional[str]:
        names = [_column_name(column) for column in self._columns(table_name)]
        preferred = [
            "created_at",
            "created",
            "timestamp",
            "event_time",
            "event_timestamp",
            "date",
            "updated_at",
        ]
        for name in preferred:
            for col_name in names:
                if col_name.lower() == name:
                    return col_name
        for col_name in names:
            lowered = col_name.lower()
            if any(token in lowered for token in ("time", "date", "created_at")):
                return col_name
        return None

    def _columns(self, table_name: str) -> list[dict[str, Any]]:
        table = self.tables.get(table_name) or {}
        return [
            column for column in table.get("columns") or [] if isinstance(column, dict)
        ]


def resolve_query_plan(
    legacy_plan: Any,
    schema: dict[str, Any],
    *,
    catalog: Any = None,
    store_id: Optional[str] = None,
) -> ResolveResult:
    """Resolve a legacy ``DbQueryPlan`` into a typed QueryPlan when safe."""

    context = _SqlPlanningView(schema)
    candidate_tables = context.resolve_tables(
        getattr(legacy_plan, "candidate_tables", [])
    )
    goal = str(getattr(legacy_plan, "goal", "") or "")
    warnings: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []

    if not candidate_tables:
        candidate_tables = _search_candidate_tables(
            context, schema, goal, catalog=catalog, store_id=store_id
        )
    if not candidate_tables:
        return ResolveResult(
            warnings=[
                {
                    "type": "unresolved_fact_table",
                    "guidance": "Use catalog schema tools to identify the source table before SQL compilation.",
                }
            ]
        )

    metrics = _resolve_metrics(legacy_plan, context, candidate_tables, warnings)
    if not metrics:
        return ResolveResult(warnings=warnings)

    fact_table = metrics[0].table
    grain = _resolve_grain(legacy_plan, context, candidate_tables, fact_table, metrics)
    filters = _resolve_filters(legacy_plan, context, fact_table, candidate_tables)
    joins = _resolve_joins(
        legacy_plan,
        schema,
        fact_table,
        grain,
        warnings,
        ambiguities,
        catalog=catalog,
        store_id=store_id,
    )
    order_by = _resolve_ordering(legacy_plan, goal, metrics, grain)
    limit = _resolve_limit(getattr(legacy_plan, "limit", None), goal, order_by)

    plan = QueryPlan(
        grain=grain,
        metrics=metrics,
        filters=filters,
        joins=joins,
        order_by=order_by,
        limit=limit,
    )
    confidence = "high" if metrics and not ambiguities else "medium"
    if warnings:
        confidence = "medium" if confidence == "high" else "low"
    return ResolveResult(
        plan=plan,
        confidence=confidence,
        warnings=warnings,
        ambiguities=ambiguities,
    )


def _resolve_metrics(
    legacy_plan: Any,
    context: "_SqlPlanningView",
    candidate_tables: list[str],
    warnings: list[dict[str, Any]],
) -> list[Metric]:
    metrics: list[Metric] = []
    for expression in getattr(legacy_plan, "aggregations", []) or []:
        metric = _metric_from_aggregation(
            expression, context, candidate_tables, warnings
        )
        if metric is not None and metric not in metrics:
            metrics.append(metric)

    if metrics:
        return metrics

    text = _intent_text(legacy_plan)
    if looks_like_count_intent(text):
        fact_table = _choose_fact_table(context, candidate_tables, text)
        alias = _count_alias(legacy_plan, fact_table)
        metrics.append(
            Metric(
                name=alias,
                kind="count",
                table=fact_table,
                column=context.primary_key_or_identity(fact_table),
            )
        )
    return metrics


def _metric_from_aggregation(
    expression: str,
    context: "_SqlPlanningView",
    candidate_tables: list[str],
    warnings: list[dict[str, Any]],
) -> Optional[Metric]:
    match = re.search(
        r"\b(count|sum|avg|min|max)\s*\(\s*(distinct\s+)?(?:(?P<table>[A-Za-z_][\w.]*)\.)?(?P<column>[A-Za-z_][\w]*|\*)?\s*\)(?:\s+as\s+(?P<alias>[A-Za-z_][\w]*))?",
        expression or "",
        flags=re.IGNORECASE,
    )
    if not match:
        warnings.append(
            {
                "type": "unsupported_aggregation",
                "aggregation": expression,
                "guidance": "Only count, distinct count, sum, avg, min, and max compile through Query IR today.",
            }
        )
        return None

    raw_kind = match.group(1).lower()
    distinct = bool(match.group(2))
    kind = "distinct_count" if raw_kind == "count" and distinct else raw_kind
    raw_table = match.group("table")
    raw_column = match.group("column")
    alias = match.group("alias")
    table = context.resolve_table(raw_table) if raw_table else candidate_tables[0]
    if not table:
        warnings.append(
            {
                "type": "unresolved_metric_table",
                "aggregation": expression,
                "table": raw_table,
            }
        )
        return None

    column = None if raw_column in (None, "*") else raw_column
    if kind == "count" and not column:
        column = context.primary_key_or_identity(table)
    if column and not context.has_column(table, column):
        if kind == "count":
            column = context.primary_key_or_identity(table)
        else:
            warnings.append(
                {
                    "type": "unresolved_metric_column",
                    "aggregation": expression,
                    "table": table,
                    "column": column,
                }
            )
            return None

    metric_alias = alias or _default_metric_alias(kind, table, column)
    return Metric(name=metric_alias, kind=kind, table=table, column=column)


def _resolve_grain(
    legacy_plan: Any,
    context: "_SqlPlanningView",
    candidate_tables: list[str],
    fact_table: str,
    metrics: list[Metric],
) -> list[FieldRef]:
    grain: list[FieldRef] = []
    names = list(getattr(legacy_plan, "grouping", []) or [])
    names.extend(
        field
        for field in getattr(legacy_plan, "required_fields", []) or []
        if not _required_field_is_metric(field, metrics)
    )

    for name in names:
        preferred_tables = [table for table in candidate_tables if table != fact_table]
        preferred_tables.append(fact_table)
        field = context.resolve_field(name, preferred_tables=preferred_tables)
        if field and field.table != fact_table and field not in grain:
            grain.append(field)
        elif field and "." in name and field not in grain:
            grain.append(field)
    return grain


def _required_field_is_metric(field: Any, metrics: list[Metric]) -> bool:
    return looks_like_count_intent(field) or any(
        _metric_matches_required(metric, str(field or "")) for metric in metrics
    )


def _resolve_time_filters(
    legacy_plan: Any,
    context: "_SqlPlanningView",
    fact_table: str,
) -> list[Filter]:
    text = _intent_text(legacy_plan)
    timestamp_column = context.timestamp_column(fact_table)
    if not timestamp_column:
        return []
    field = FieldRef(fact_table, timestamp_column)
    if re.search(r"\b(this|current)\s+month\b", text):
        return [
            Filter(
                field=field, operator=">=", value={"macro": "start_of_current_month"}
            )
        ]
    if re.search(r"\blast\s+month\b", text):
        return [
            Filter(field=field, operator=">=", value={"macro": "start_of_last_month"}),
            Filter(
                field=field, operator="<", value={"macro": "start_of_current_month"}
            ),
        ]
    match = re.search(r"\blast\s+(\d{1,4})\s+days?\b", text)
    if match:
        return [
            Filter(
                field=field,
                operator=">=",
                value={"macro": f"last_{int(match.group(1))}_days"},
            )
        ]
    return []


def _resolve_filters(
    legacy_plan: Any,
    context: "_SqlPlanningView",
    fact_table: str,
    candidate_tables: list[str],
) -> list[Filter]:
    filters = _resolve_time_filters(legacy_plan, context, fact_table)
    preferred_tables = [fact_table] + [
        table for table in candidate_tables if table != fact_table
    ]
    for expression in getattr(legacy_plan, "filters", []) or []:
        parsed = _filter_from_expression(expression, context, preferred_tables)
        if parsed is not None and parsed not in filters:
            filters.append(parsed)
    return filters


def _filter_from_expression(
    expression: Any,
    context: "_SqlPlanningView",
    preferred_tables: list[str],
) -> Optional[Filter]:
    text = str(expression or "").strip()
    if not text:
        return None
    match = re.match(
        r"^\s*([A-Za-z_][\w.]*?)\s*(=|!=|>=|<=|>|<)\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    field = context.resolve_field(match.group(1), preferred_tables=preferred_tables)
    if field is None:
        return None
    value = _parse_filter_value(match.group(3))
    return Filter(field=field, operator=match.group(2), value=value)


def _parse_filter_value(value: Any) -> Any:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    lowered = text.lower()
    if lowered == "null":
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _resolve_joins(
    legacy_plan: Any,
    schema: dict[str, Any],
    fact_table: str,
    grain: list[FieldRef],
    warnings: list[dict[str, Any]],
    ambiguities: list[dict[str, Any]],
    *,
    catalog: Any = None,
    store_id: Optional[str] = None,
) -> list[Join]:
    target_tables = [field.table for field in grain if field.table != fact_table]
    provided_paths: list[dict[str, Any]] = []
    context = _SqlPlanningView(schema)
    for requirement in getattr(legacy_plan, "required_joins", []) or []:
        for table_ref in _string_list(
            requirement.get("from_tables") or requirement.get("from")
        ):
            resolved = context.resolve_table(table_ref)
            if resolved and resolved != fact_table:
                target_tables.append(resolved)
        for table_ref in _string_list(
            requirement.get("to_tables") or requirement.get("to")
        ):
            resolved = context.resolve_table(table_ref)
            if resolved and resolved != fact_table:
                target_tables.append(resolved)
        for path in requirement.get("paths") or []:
            if isinstance(path, dict) and path.get("joins"):
                provided_paths.append(path)

    joins: list[Join] = []
    resolved_targets: set[str] = set()
    for path in provided_paths:
        path_tables = _string_list(path.get("tables"))
        resolved_targets.update(table for table in path_tables if table != fact_table)
        for join in path.get("joins") or []:
            resolved = _join_from_path_item(join)
            if resolved is not None and resolved not in joins:
                joins.append(resolved)

    for target_table in unique_preserving_order(target_tables, skip_empty=True):
        if target_table in resolved_targets:
            continue
        path_result = find_relationship_paths(
            schema,
            from_tables=[fact_table],
            to_tables=[target_table],
            max_hops=4,
            max_paths=2,
            catalog=catalog,
            store_id=store_id,
        )
        paths = path_result.get("paths") or []
        if not path_result.get("reachable") or not paths:
            warnings.append(
                {
                    "type": "unresolved_join_path",
                    "from_table": fact_table,
                    "to_table": target_table,
                    "suggested_next_tool": "catalog_find_join_paths",
                }
            )
            continue
        if len(paths) > 1:
            ambiguities.append(
                {
                    "type": "multiple_join_paths",
                    "from_table": fact_table,
                    "to_table": target_table,
                    "path_count": len(paths),
                }
            )
        for join in paths[0].get("joins") or []:
            resolved = Join(
                left=FieldRef(join["left_table"], join["left_column"]),
                right=FieldRef(join["right_table"], join["right_column"]),
            )
            if resolved not in joins:
                joins.append(resolved)
    return joins


def _join_from_path_item(item: dict[str, Any]) -> Join | None:
    try:
        left_table = item["left_table"]
        left_column = item["left_column"]
        right_table = item["right_table"]
        right_column = item["right_column"]
    except KeyError:
        return None
    return Join(
        left=FieldRef(str(left_table), str(left_column)),
        right=FieldRef(str(right_table), str(right_column)),
    )


def _resolve_ordering(
    legacy_plan: Any,
    goal: str,
    metrics: list[Metric],
    grain: list[FieldRef],
) -> list[OrderBy]:
    order_text = " ".join(getattr(legacy_plan, "ordering", []) or [])
    combined = f"{goal} {order_text}".lower()
    direction = (
        "asc"
        if re.search(r"\b(least|lowest|smallest|ascending|asc)\b", combined)
        else "desc"
    )
    for metric in metrics:
        if _normalize(metric.name) in _normalize(order_text):
            return [OrderBy(field=metric.name, direction=direction)]
    if metrics and re.search(
        r"\b(most|top|highest|largest|least|lowest|smallest)\b", combined
    ):
        return [OrderBy(field=metrics[0].name, direction=direction)]
    if grain and order_text:
        return [OrderBy(field=grain[0], direction=direction)]
    return []


def _resolve_limit(raw_limit: Any, goal: str, order_by: list[OrderBy]) -> Optional[int]:
    try:
        parsed = int(raw_limit)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    if order_by and re.search(
        r"\b(which|what|who)\b.*\b(most|least|highest|lowest)\b", goal.lower()
    ):
        return 1
    return None


def _search_candidate_tables(
    context: "_SqlPlanningView",
    schema: dict[str, Any],
    goal: str,
    *,
    catalog: Any = None,
    store_id: Optional[str] = None,
) -> list[str]:
    result = search_tables(
        schema, query=goal, limit=5, catalog=catalog, store_id=store_id
    )
    tables = []
    for item in result.get("tables") or []:
        if float(item.get("score") or 0) <= 0:
            continue
        table = context.resolve_table(item.get("name"))
        if table:
            tables.append(table)
    return unique_preserving_order(tables, skip_empty=True)


def _choose_fact_table(
    context: "_SqlPlanningView", candidate_tables: list[str], text: str
) -> str:
    for table in candidate_tables:
        if any(token in text for token in _split_identifier(table)):
            return table
    for table in candidate_tables:
        if context.primary_key_or_identity(table):
            return table
    return candidate_tables[0]


def _count_alias(legacy_plan: Any, fact_table: str) -> str:
    for field in getattr(legacy_plan, "required_fields", []) or []:
        if looks_like_count_intent(field):
            return _snake_case(field)
    return f"total_{_split_identifier(fact_table)[-1] if _split_identifier(fact_table) else 'rows'}"


def _default_metric_alias(kind: str, table: str, column: Optional[str]) -> str:
    if kind == "count":
        return f"total_{_split_identifier(table)[-1] if _split_identifier(table) else 'rows'}"
    source = column or table
    return f"{kind}_{_snake_case(source)}"


def _intent_text(legacy_plan: Any) -> str:
    parts = [getattr(legacy_plan, "goal", "") or ""]
    for attr in ("required_fields", "filters", "aggregations", "grouping", "ordering"):
        parts.extend(getattr(legacy_plan, attr, []) or [])
    return " ".join(str(part) for part in parts).lower()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _snake_case(value: str) -> str:
    tokens = _split_identifier(value)
    return "_".join(tokens) or "metric"
