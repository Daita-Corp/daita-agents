"""Deterministic validation for DB query plan proposals."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .fingerprints import persisted_fingerprint
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
        validation_facts: list[dict[str, Any]] = []
        schema = dict(planning_context.get("schema") or {})
        dialect = str(
            planning_context.get("dialect") or schema.get("database_type") or ""
        )
        schema_columns = _schema_columns(schema)
        table_columns = _table_columns_from_schema_columns(schema_columns)
        value_profiles = _value_profiles(planning_context)
        db_memory_validation = {
            "checked": bool(planning_context.get("db_memory_semantics")),
            "memory_keys": [],
            "passed": True,
            "errors": [],
        }
        catalog_relationship_validation = {
            "checked": False,
            "required_joins": [],
            "relationship_evidence_refs": [],
            "passed": True,
            "errors": [],
        }
        session_scope_validation: dict[str, Any] = {
            "checked": False,
            "source_scope_id": None,
            "source_operation_id": None,
            "binding_status": None,
            "required_filter_count": 0,
            "required_join_count": 0,
            "passed": True,
            "errors": [],
        }
        sql = plan.selected_sql
        analysis_for_relationships: Any | None = None

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
            _validate_join_column(
                table_columns,
                join.left_table,
                join.left_column,
                errors,
                warnings,
                label="join_left",
            )
            _validate_join_column(
                table_columns,
                join.right_table,
                join.right_column,
                errors,
                warnings,
                label="join_right",
            )
        for filter_spec in plan.filters:
            table, column = _split_column_ref(filter_spec.column)
            if table:
                _validate_column(table_columns, table, column, errors, label="filter")
            _validate_filter_literal(
                value_profiles,
                schema_columns,
                plan.selected_tables,
                table,
                column,
                filter_spec.operator,
                filter_spec.value,
                None,
                errors,
                validation_facts,
            )

        if sql:
            try:
                analysis = analyze_sql(sql, dialect=dialect)
            except SqlAnalysisError as exc:
                errors.append(f"sql_parse_failed:{exc.error_type}")
            else:
                analysis_for_relationships = analysis
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
                    predicate_values = list(predicate.values)
                    predicate_kinds = list(getattr(predicate, "value_kinds", ()) or ())
                    _validate_filter_literal(
                        value_profiles,
                        schema_columns,
                        sql_tables,
                        predicate.column.table or None,
                        predicate.column.name,
                        predicate.operator,
                        (
                            predicate_values
                            if predicate.operator == "in"
                            else predicate_values[0]
                        ),
                        predicate_kinds,
                        errors,
                        validation_facts,
                    )
                memory_errors = _validate_db_memory_semantics(
                    plan,
                    analysis,
                    planning_context,
                )
                if memory_errors:
                    errors.extend(memory_errors)
                    db_memory_validation["passed"] = False
                    db_memory_validation["errors"] = memory_errors
                db_memory_validation["memory_keys"] = [
                    str(item.get("key") or item.get("memory_key") or "")
                    for item in planning_context.get("db_memory_semantics", []) or []
                    if isinstance(item, dict) and item.get("enforceable")
                ]
                session_errors, session_facts, session_metadata = (
                    _validate_session_scope_binding(
                        plan,
                        planning_context,
                        analysis,
                        sql_tables,
                    )
                )
                if session_errors:
                    errors.extend(session_errors)
                    validation_facts.extend(session_facts)
                session_scope_validation.update(session_metadata)
        elif plan.clarification_question:
            session_metadata = _session_scope_binding_metadata(planning_context)
            if session_metadata["checked"]:
                session_metadata["binding_status"] = "clarification_requested"
                session_scope_validation.update(session_metadata)

        relationship_errors, relationship_facts, relationship_metadata = (
            _validate_catalog_relationships(
                plan,
                planning_context,
                analysis_for_relationships,
            )
        )
        if relationship_errors:
            errors.extend(relationship_errors)
            validation_facts.extend(relationship_facts)
        catalog_relationship_validation.update(relationship_metadata)

        fingerprint = persisted_fingerprint(plan.to_dict())
        return DbQueryPlanValidation(
            valid=not errors,
            accepted_sql=sql if not errors else None,
            sql_fingerprint=sql_fingerprint(sql) if sql else None,
            errors=tuple(errors),
            warnings=tuple(warnings),
            validation_facts=tuple(_dedupe_validation_facts(validation_facts)),
            plan_fingerprint=fingerprint,
            metadata={
                "validator": "deterministic",
                "schema_fingerprint": planning_context.get("schema_fingerprint"),
                "db_memory_contract_validation": db_memory_validation,
                "catalog_relationship_validation": catalog_relationship_validation,
                "session_scope_binding_validation": session_scope_validation,
            },
        )


def _schema_columns(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for table in schema.get("tables", []) or []:
        name = str(table.get("name") or "")
        if not name:
            continue
        columns = {
            str(column.get("name") or "").lower(): dict(column)
            for column in table.get("columns", []) or []
            if isinstance(column, dict) and column.get("name")
        }
        tables[name.lower()] = {"name": name, "columns": columns}
    return tables


def _table_columns_from_schema_columns(
    schema_columns: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    return {
        table_key: set(table.get("columns", {}))
        for table_key, table in schema_columns.items()
    }


def _table_columns(schema: dict[str, Any]) -> dict[str, set[str]]:
    return _table_columns_from_schema_columns(_schema_columns(schema))


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


def _validate_join_column(
    table_columns: dict[str, set[str]],
    table: str,
    column: str,
    errors: list[str],
    warnings: list[str],
    *,
    label: str,
) -> None:
    if not column:
        table_label = table or "unknown"
        warnings.append(f"{label}_column_not_declared:{table_label}")
        if table and table.lower() not in table_columns:
            errors.append(f"{label}_unknown_table:{table}")
        return
    _validate_column(table_columns, table, column, errors, label=label)


def _validate_filter_literal(
    value_profiles: dict[tuple[str | None, str], set[str]],
    schema_columns: dict[str, dict[str, Any]],
    sql_tables: Iterable[str],
    table: str | None,
    column: str,
    operator: str,
    value: Any,
    literal_kinds: Iterable[str] | None,
    errors: list[str],
    validation_facts: list[dict[str, Any]],
) -> None:
    normalized_operator = str(operator or "").strip().lower()
    if normalized_operator not in {"=", "==", "eq", "in"}:
        return
    if value is None:
        return
    if normalized_operator == "in" and isinstance(value, (list, tuple, set)):
        kinds = list(literal_kinds or ())
        for index, item in enumerate(value):
            _validate_filter_literal(
                value_profiles,
                schema_columns,
                sql_tables,
                table,
                column,
                "=",
                item,
                (kinds[index],) if index < len(kinds) else None,
                errors,
                validation_facts,
            )
        return
    literal = str(value)
    candidates = _observed_values(value_profiles, table, column)
    if candidates and literal.lower() in candidates:
        return
    if not candidates:
        if not _is_string_literal(value, literal_kinds):
            return
        resolved = _resolve_filter_literal_column(
            schema_columns,
            table=table,
            column=column,
            sql_tables=sql_tables,
        )
        if resolved is None:
            return
        table_label, column_label, column_metadata = resolved
        if not _column_eligible_for_value_grounding(column_metadata):
            return
        errors.append(
            "filter_literal_requires_grounding:"
            f"{table_label}.{column_label}={literal}"
        )
        validation_facts.append(
            {
                "kind": "filter_literal_requires_grounding",
                "table": table_label,
                "column": column_label,
                "operator": _normalize_operator(normalized_operator),
                "literal": literal,
                "source": "query.plan.validation",
                "reason": (
                    "proposed_sql_filter_literal_without_accepted_value_evidence"
                ),
            }
        )
        return
    table_label = table or _single_table_for_column(value_profiles, column) or "unknown"
    sorted_candidates = sorted(candidates)
    errors.append(
        "unobserved_filter_literal:"
        f"{table_label}.{column}={literal};"
        f"candidates={','.join(sorted_candidates)}"
    )
    validation_facts.append(
        {
            "kind": "unobserved_filter_literal",
            "table": table_label,
            "column": column,
            "literal": literal,
            "candidates": sorted_candidates,
        }
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


def _is_string_literal(value: Any, literal_kinds: Iterable[str] | None) -> bool:
    kinds = [str(kind).lower() for kind in literal_kinds or () if str(kind)]
    if kinds:
        return all(kind == "string" for kind in kinds)
    return isinstance(value, str) and bool(value.strip())


def _resolve_filter_literal_column(
    schema_columns: dict[str, dict[str, Any]],
    *,
    table: str | None,
    column: str,
    sql_tables: Iterable[str],
) -> tuple[str, str, dict[str, Any]] | None:
    column_key = str(column or "").lower()
    if not column_key:
        return None
    requested_tables = [str(item) for item in sql_tables if str(item or "").strip()]
    if table:
        requested_tables = [str(table), *requested_tables]

    table_items = list(schema_columns.items())
    if requested_tables:
        requested_keys = {str(item).lower() for item in requested_tables}
        requested_short_keys = {
            _short_table_key(item)
            for item in requested_tables
            if _short_table_key(item)
        }
        table_items = [
            (table_key, table_info)
            for table_key, table_info in table_items
            if table_key in requested_keys
            or _short_table_key(str(table_info.get("name") or table_key))
            in requested_keys
            or table_key in requested_short_keys
            or _short_table_key(str(table_info.get("name") or table_key))
            in requested_short_keys
        ]

    matches: list[tuple[str, str, dict[str, Any]]] = []
    for table_key, table_info in table_items:
        columns = table_info.get("columns")
        if not isinstance(columns, dict) or column_key not in columns:
            continue
        column_metadata = dict(columns[column_key])
        table_name = str(table_info.get("name") or table_key)
        column_name = str(column_metadata.get("name") or column)
        matches.append((table_name, column_name, column_metadata))
    return matches[0] if len(matches) == 1 else None


def _column_eligible_for_value_grounding(column: dict[str, Any]) -> bool:
    data_type = str(
        column.get("data_type") or column.get("type") or column.get("db_type") or ""
    ).lower()
    if not data_type:
        return False
    return any(
        token in data_type
        for token in (
            "char",
            "clob",
            "enum",
            "string",
            "text",
            "uuid",
        )
    )


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


def _validate_catalog_relationships(
    plan: DbQueryPlan,
    planning_context: dict[str, Any],
    analysis: Any | None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    required = _required_join_edges(plan, analysis)
    evidence_edges, relationship_refs = _catalog_relationship_edges(planning_context)
    metadata = {
        "checked": bool(required),
        "required_joins": [dict(item) for item in required],
        "relationship_evidence_refs": relationship_refs,
        "passed": True,
        "errors": [],
    }
    if plan.operation != "read" or not required:
        return [], [], metadata

    errors: list[str] = []
    facts: list[dict[str, Any]] = []
    for edge in required:
        if any(
            _relationship_edge_matches(edge, evidence) for evidence in evidence_edges
        ):
            continue
        label = _join_edge_label(edge)
        error = f"missing_catalog_relationship_path:{label}"
        errors.append(error)
        fact = {
            "kind": "missing_catalog_relationship_path",
            "left_table": edge["left_table"],
            "right_table": edge["right_table"],
            "source": edge.get("source") or "query.plan.validation",
            "reason": "joined_data_query_without_catalog_relationship_evidence",
        }
        if edge.get("left_column"):
            fact["left_column"] = edge["left_column"]
        if edge.get("right_column"):
            fact["right_column"] = edge["right_column"]
        facts.append(fact)
    if errors:
        metadata["passed"] = False
        metadata["errors"] = list(errors)
    return errors, facts, metadata


def _required_join_edges(
    plan: DbQueryPlan,
    analysis: Any | None,
) -> tuple[dict[str, str], ...]:
    edges: list[dict[str, str]] = []
    for join in plan.joins:
        edge = _join_edge(
            join.left_table,
            join.left_column,
            join.right_table,
            join.right_column,
            source="query_plan_join",
        )
        if edge is not None:
            edges.append(edge)

    if analysis is not None:
        for predicate in getattr(analysis, "column_predicates", ()) or ():
            if str(getattr(predicate, "operator", "") or "") != "=":
                continue
            left = getattr(predicate, "left", None)
            right = getattr(predicate, "right", None)
            edge = _join_edge(
                getattr(left, "table", "") if left is not None else "",
                getattr(left, "name", "") if left is not None else "",
                getattr(right, "table", "") if right is not None else "",
                getattr(right, "name", "") if right is not None else "",
                source="sql_column_predicate",
            )
            if edge is not None:
                edges.append(edge)

    if not edges:
        tables = _join_candidate_tables(plan, analysis)
        if len(tables) > 1:
            first = tables[0]
            for table in tables[1:]:
                edge = _join_edge(first, "", table, "", source="multi_table_plan")
                if edge is not None:
                    edges.append(edge)
    return _dedupe_join_edges(edges)


def _join_candidate_tables(
    plan: DbQueryPlan,
    analysis: Any | None,
) -> list[str]:
    tables: list[str] = []
    if analysis is not None:
        tables.extend(
            str(table.short_key)
            for table in getattr(analysis, "tables", ()) or ()
            if not getattr(table, "is_cte", False) and str(table.short_key)
        )
    tables.extend(str(table) for table in plan.selected_tables if str(table).strip())
    return list(
        dict.fromkeys(_short_table_key(table) or str(table) for table in tables)
    )


def _join_edge(
    left_table: Any,
    left_column: Any,
    right_table: Any,
    right_column: Any,
    *,
    source: str,
) -> dict[str, str] | None:
    left_table_key = _short_table_key(str(left_table or ""))
    right_table_key = _short_table_key(str(right_table or ""))
    if not left_table_key or not right_table_key or left_table_key == right_table_key:
        return None
    return {
        "left_table": left_table_key,
        "left_column": str(left_column or "").strip().lower(),
        "right_table": right_table_key,
        "right_column": str(right_column or "").strip().lower(),
        "source": source,
    }


def _dedupe_join_edges(edges: Iterable[dict[str, str]]) -> tuple[dict[str, str], ...]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, str]] = []
    for edge in edges:
        key = (
            edge.get("left_table", ""),
            edge.get("left_column", ""),
            edge.get("right_table", ""),
            edge.get("right_column", ""),
        )
        reverse = (key[2], key[3], key[0], key[1])
        if key in seen or reverse in seen:
            continue
        seen.add(key)
        out.append(edge)
    return tuple(out)


def _catalog_relationship_edges(
    planning_context: dict[str, Any],
) -> tuple[tuple[dict[str, str], ...], list[dict[str, str]]]:
    edges: list[dict[str, str]] = []
    refs: list[dict[str, str]] = []
    for detail in planning_context.get("relationship_evidence_details", []) or []:
        if not isinstance(detail, dict):
            continue
        if detail.get("accepted", True) is not True:
            continue
        if detail.get("owner") != "catalog":
            continue
        if detail.get("kind", "schema.relationship_path") != "schema.relationship_path":
            continue
        ref = {
            "id": str(detail.get("id") or ""),
            "kind": "schema.relationship_path",
            "owner": "catalog",
        }
        if detail.get("payload_fingerprint"):
            ref["payload_fingerprint"] = str(detail["payload_fingerprint"])
        refs.append(ref)
        raw_payload = detail.get("payload")
        payload: dict[str, Any] = (
            {str(key): value for key, value in raw_payload.items()}
            if isinstance(raw_payload, dict)
            else {str(key): value for key, value in detail.items()}
        )
        if payload.get("reachable") is False:
            continue
        for path in payload.get("paths", []) or []:
            if not isinstance(path, dict):
                continue
            for join in path.get("joins", []) or path.get("relationships", []) or []:
                if not isinstance(join, dict):
                    continue
                edge = _join_edge(
                    join.get("left_table")
                    or join.get("left_asset")
                    or join.get("source_table")
                    or join.get("source_asset"),
                    join.get("left_column")
                    or join.get("left_field")
                    or join.get("source_column")
                    or join.get("source_field"),
                    join.get("right_table")
                    or join.get("right_asset")
                    or join.get("target_table")
                    or join.get("target_asset"),
                    join.get("right_column")
                    or join.get("right_field")
                    or join.get("target_column")
                    or join.get("target_field"),
                    source="schema.relationship_path",
                )
                if edge is not None:
                    edges.append(edge)
    return _dedupe_join_edges(edges), refs


def _relationship_edge_matches(
    required: dict[str, str],
    evidence: dict[str, str],
) -> bool:
    if _relationship_edge_matches_direction(required, evidence):
        return True
    reversed_required = {
        "left_table": required["right_table"],
        "left_column": required.get("right_column", ""),
        "right_table": required["left_table"],
        "right_column": required.get("left_column", ""),
    }
    return _relationship_edge_matches_direction(reversed_required, evidence)


def _relationship_edge_matches_direction(
    required: dict[str, str],
    evidence: dict[str, str],
) -> bool:
    if required["left_table"] != evidence["left_table"]:
        return False
    if required["right_table"] != evidence["right_table"]:
        return False
    left_column = required.get("left_column", "")
    right_column = required.get("right_column", "")
    if left_column and left_column != evidence.get("left_column", ""):
        return False
    if right_column and right_column != evidence.get("right_column", ""):
        return False
    return True


def _join_edge_label(edge: dict[str, str]) -> str:
    left = edge["left_table"]
    right = edge["right_table"]
    if edge.get("left_column") and edge.get("right_column"):
        left = f"{left}.{edge['left_column']}"
        right = f"{right}.{edge['right_column']}"
    return f"{left}->{right}"


def _validate_db_memory_semantics(
    plan: DbQueryPlan,
    analysis: Any,
    planning_context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    semantics = [
        item
        for item in planning_context.get("db_memory_semantics", []) or []
        if isinstance(item, dict) and item.get("enforceable")
    ]
    if not semantics:
        return errors
    sql = plan.selected_sql or ""
    sql_refs = _sql_column_refs(analysis, planning_context.get("schema") or {})
    for contract in semantics:
        kind = str(contract.get("contract_kind") or contract.get("kind") or "")
        if kind == "metric_definition":
            _validate_metric_contract(contract, plan, analysis, sql, sql_refs, errors)
        elif kind == "unit_convention":
            _validate_unit_contract(contract, analysis, sql, errors)
    return errors


def _validate_metric_contract(
    contract: dict[str, Any],
    plan: DbQueryPlan,
    analysis: Any,
    sql: str,
    sql_refs: set[str],
    errors: list[str],
) -> None:
    memory_key = _memory_key(contract)
    for ref in contract.get("required_refs", []) or []:
        if not _sql_references_ref(str(ref), sql_refs):
            errors.append(f"missing_db_memory_required_ref:{memory_key}:{ref}")
    for relationship in contract.get("required_relationships", []) or []:
        if not _sql_contains_relationship(str(relationship), analysis, sql):
            errors.append(
                f"missing_db_memory_required_join:{memory_key}:{relationship}"
            )
    for filter_spec in contract.get("required_filters", []) or []:
        if not isinstance(filter_spec, dict):
            continue
        if not _sql_contains_contract_filter(filter_spec, analysis):
            errors.append(
                "missing_db_memory_required_filter:"
                f"{memory_key}:{filter_spec.get('ref')}={filter_spec.get('value')}"
            )
    for aggregation in contract.get("required_aggregations", []) or []:
        if not isinstance(aggregation, dict):
            continue
        function = str(aggregation.get("function") or "").lower()
        ref = str(aggregation.get("ref") or "")
        if function and ref and not _sql_contains_aggregation(function, ref, analysis):
            errors.append(
                f"missing_db_memory_required_aggregation:"
                f"{memory_key}:{function}:{ref}"
            )
    result_shape = contract.get("result_shape") or {}
    if result_shape.get("grain") == "single_aggregate":
        if plan.group_by or re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE):
            errors.append(
                f"missing_db_memory_required_result_shape:"
                f"{memory_key}:single_aggregate"
            )
        elif not _looks_aggregated(sql):
            errors.append(
                f"missing_db_memory_required_result_shape:"
                f"{memory_key}:single_aggregate"
            )


def _validate_session_scope_binding(
    plan: DbQueryPlan,
    planning_context: dict[str, Any],
    analysis: Any,
    sql_tables: list[str],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    metadata = _session_scope_binding_metadata(planning_context)
    if not metadata["checked"]:
        return [], [], metadata
    binding = planning_context.get("session_scope_binding")
    if not isinstance(binding, dict):
        return [], [], metadata
    status = str(binding.get("binding_status") or "").strip().lower()
    if status and status not in {"bound", "accepted", "enforced", "required"}:
        return [], [], metadata
    if plan.operation != "read":
        return [], [], metadata

    errors: list[str] = []
    facts: list[dict[str, Any]] = []
    for required in _binding_required_filters(binding):
        if _sql_contains_scope_filter(required, analysis, sql_tables):
            continue
        label = _scope_filter_label(required)
        error = f"missing_session_scope_filter:{label}"
        errors.append(error)
        facts.append(
            {
                "kind": "missing_session_scope_filter",
                "column": str(required.get("column") or required.get("ref") or ""),
                "operator": _normalize_operator(required.get("operator")),
                "values": list(_binding_filter_values(required)),
                "source": "session.scope_binding",
                "reason": "follow_up_sql_missing_bound_scope_filter",
            }
        )

    sql_edges = _required_join_edges(plan, analysis)
    for required in _binding_required_joins(binding):
        if any(_relationship_edge_matches(required, edge) for edge in sql_edges):
            continue
        label = _join_edge_label(required)
        error = f"missing_session_scope_join:{label}"
        errors.append(error)
        fact = {
            "kind": "missing_session_scope_join",
            "left_table": required["left_table"],
            "right_table": required["right_table"],
            "source": "session.scope_binding",
            "reason": "follow_up_sql_missing_bound_scope_join",
        }
        if required.get("left_column"):
            fact["left_column"] = required["left_column"]
        if required.get("right_column"):
            fact["right_column"] = required["right_column"]
        facts.append(fact)

    if errors:
        metadata["passed"] = False
        metadata["errors"] = list(errors)
    return errors, facts, metadata


def _session_scope_binding_metadata(
    planning_context: dict[str, Any],
) -> dict[str, Any]:
    binding = planning_context.get("session_scope_binding")
    metadata: dict[str, Any] = {
        "checked": isinstance(binding, dict),
        "source_scope_id": None,
        "source_operation_id": None,
        "binding_status": None,
        "required_filter_count": 0,
        "required_join_count": 0,
        "passed": True,
        "errors": [],
    }
    if not isinstance(binding, dict):
        return metadata
    metadata.update(
        {
            "source_scope_id": binding.get("source_scope_id"),
            "source_operation_id": binding.get("source_operation_id"),
            "binding_status": binding.get("binding_status"),
            "required_filter_count": len(_binding_required_filters(binding)),
            "required_join_count": len(_binding_required_joins(binding)),
        }
    )
    return metadata


def _binding_required_filters(binding: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    filters = []
    for item in binding.get("required_filters", ()) or ():
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or item.get("ref") or "").strip()
        operator = str(item.get("operator") or item.get("op") or "").strip()
        values = _binding_filter_values(item)
        if column and operator and values:
            filters.append(
                {
                    "column": column,
                    "operator": operator,
                    "values": list(values),
                }
            )
    return tuple(filters)


def _binding_required_joins(binding: dict[str, Any]) -> tuple[dict[str, str], ...]:
    joins = []
    for item in binding.get("required_joins", ()) or ():
        if not isinstance(item, dict):
            continue
        edge = _join_edge(
            item.get("left_table") or item.get("source_table"),
            item.get("left_column") or item.get("source_column"),
            item.get("right_table") or item.get("target_table"),
            item.get("right_column") or item.get("target_column"),
            source="session.scope_binding",
        )
        if edge is not None:
            joins.append(edge)
    return _dedupe_join_edges(joins)


def _sql_contains_scope_filter(
    required: dict[str, Any],
    analysis: Any,
    sql_tables: list[str],
) -> bool:
    required_table, required_column = _split_column_ref(str(required.get("column")))
    required_operator = _normalize_operator(required.get("operator"))
    required_values = set(_binding_filter_values(required))
    if not required_values:
        return False
    for predicate in getattr(analysis, "literal_predicates", ()) or ():
        predicate_column = getattr(predicate, "column", None)
        predicate_table = str(getattr(predicate_column, "table", "") or "") or None
        predicate_column_name = str(getattr(predicate_column, "name", "") or "")
        if required_column.lower() != predicate_column_name.lower():
            continue
        if not _tables_match(required_table, predicate_table, sql_tables):
            continue
        predicate_operator = _normalize_operator(getattr(predicate, "operator", ""))
        if not _scope_operator_matches(required_operator, predicate_operator):
            continue
        predicate_values = {
            str(value).lower() for value in getattr(predicate, "values", ()) or ()
        }
        if {value.lower() for value in required_values} <= predicate_values:
            return True
    return False


def _scope_operator_matches(required: str, actual: str) -> bool:
    if required == actual:
        return True
    if required == "=" and actual == "in":
        return True
    if required == "in" and actual == "=":
        return True
    return False


def _binding_filter_values(filter_spec: dict[str, Any]) -> tuple[str, ...]:
    raw = (
        filter_spec.get("values")
        if "values" in filter_spec
        else filter_spec.get("value")
    )
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = (raw,)
    return tuple(str(item).lower() for item in values if item is not None)


def _scope_filter_label(filter_spec: dict[str, Any]) -> str:
    values = ",".join(_binding_filter_values(filter_spec))
    return (
        f"{filter_spec.get('column')}"
        f"{_normalize_operator(filter_spec.get('operator'))}{values}"
    )


def _validate_unit_contract(
    contract: dict[str, Any],
    analysis: Any,
    sql: str,
    errors: list[str],
) -> None:
    conversion = contract.get("unit_conversion") or {}
    if conversion.get("operator") not in {"divide", "multiply"}:
        return
    required_refs = [str(ref) for ref in contract.get("required_refs", []) or [] if ref]
    if not required_refs:
        return
    memory_key = _memory_key(contract)
    factor = str(conversion.get("factor") or "")
    for ref in required_refs:
        item = _select_item_for_ref(ref, analysis)
        if item is None:
            continue
        expression = str(getattr(item, "expression_sql", "") or "")
        if conversion["operator"] == "divide":
            if re.search(rf"/\s*{re.escape(factor)}(?:\.0+)?\b", expression):
                continue
        if conversion["operator"] == "multiply":
            if re.search(rf"\*\s*{re.escape(factor)}(?:\.0+)?\b", expression):
                continue
        if not _prompted_raw_unit_alias(item, conversion):
            errors.append(
                f"missing_db_memory_required_unit_conversion:"
                f"{memory_key}:{ref}:{conversion['operator']}:{factor}"
            )


def _sql_column_refs(analysis: Any, schema: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    table_columns = _table_columns(schema)
    sql_tables = [
        table.short_key for table in getattr(analysis, "tables", ()) if not table.is_cte
    ]
    for column in getattr(analysis, "columns", ()) or ():
        name = str(getattr(column, "name", "") or "").lower()
        table = _short_table_key(getattr(column, "table", "") or None)
        if table:
            refs.add(f"{table}.{name}")
            continue
        matching_tables = [
            table_name
            for table_name, columns in table_columns.items()
            if name in columns and (not sql_tables or table_name in sql_tables)
        ]
        for table_name in matching_tables:
            refs.add(f"{table_name}.{name}")
        refs.add(name)
    return refs


def _sql_references_ref(ref: str, sql_refs: set[str]) -> bool:
    table, column = _split_column_ref(ref)
    if table and column:
        return f"{_short_table_key(table)}.{column.lower()}" in sql_refs
    return column.lower() in sql_refs


def _sql_contains_contract_filter(filter_spec: dict[str, Any], analysis: Any) -> bool:
    ref = str(filter_spec.get("ref") or "")
    table, column = _split_column_ref(ref)
    expected_operator = _normalize_operator(
        "="
        if str(filter_spec.get("operator") or "").lower() == "semantic_equals"
        else filter_spec.get("operator")
    )
    expected_value = str(filter_spec.get("value") or "").lower()
    for predicate in getattr(analysis, "literal_predicates", ()) or ():
        predicate_column = getattr(predicate, "column", None)
        predicate_table = str(getattr(predicate_column, "table", "") or "") or None
        predicate_column_name = str(getattr(predicate_column, "name", "") or "")
        if column.lower() != predicate_column_name.lower():
            continue
        if table and _short_table_key(table) != _short_table_key(predicate_table):
            continue
        if _normalize_operator(getattr(predicate, "operator", "")) != expected_operator:
            continue
        values = {str(value).lower() for value in getattr(predicate, "values", ())}
        if expected_value in values:
            return True
    return False


def _sql_contains_relationship(relationship: str, analysis: Any, sql: str) -> bool:
    if "->" not in relationship:
        return False
    left, right = [part.strip() for part in relationship.split("->", maxsplit=1)]
    left_variants = _ref_sql_variants(left, analysis)
    right_variants = _ref_sql_variants(right, analysis)
    normalized = _normalized_sql_text(sql)
    for left_variant in left_variants:
        for right_variant in right_variants:
            left_pattern = re.escape(left_variant).replace(r"\.", r"\s*\.\s*")
            right_pattern = re.escape(right_variant).replace(r"\.", r"\s*\.\s*")
            if re.search(rf"{left_pattern}\s*=\s*{right_pattern}", normalized):
                return True
            if re.search(rf"{right_pattern}\s*=\s*{left_pattern}", normalized):
                return True
    return False


def _sql_contains_aggregation(function: str, ref: str, analysis: Any) -> bool:
    for item in getattr(analysis, "select_items", ()) or ():
        expression = str(getattr(item, "expression_sql", "") or "").lower()
        if f"{function.lower()}(" not in expression:
            continue
        if any(
            variant in _normalized_sql_text(expression)
            for variant in _ref_sql_variants(ref, analysis)
        ):
            return True
    return False


def _select_item_for_ref(ref: str, analysis: Any) -> Any | None:
    for item in getattr(analysis, "select_items", ()) or ():
        expression = _normalized_sql_text(getattr(item, "expression_sql", "") or "")
        if any(variant in expression for variant in _ref_sql_variants(ref, analysis)):
            return item
    return None


def _prompted_raw_unit_alias(item: Any, conversion: dict[str, Any]) -> bool:
    alias = str(getattr(item, "alias", "") or "").lower()
    stored_unit = str(conversion.get("stored_unit") or "").lower()
    return bool(stored_unit and stored_unit in alias)


def _ref_sql_variants(ref: str, analysis: Any) -> set[str]:
    table, column = _split_column_ref(ref)
    column_key = column.lower()
    table_key = _short_table_key(table)
    sql_tables = [
        table_ref
        for table_ref in getattr(analysis, "tables", ()) or ()
        if not getattr(table_ref, "is_cte", False)
    ]
    if not table_key:
        return {column_key}
    variants = {f"{table_key}.{column_key}"}
    matching_tables = [
        table_ref
        for table_ref in sql_tables
        if getattr(table_ref, "short_key", "") == table_key
    ]
    if matching_tables or not sql_tables:
        variants.add(column_key)
    for table_ref in matching_tables:
        alias = str(getattr(table_ref, "alias", "") or "").lower()
        if alias:
            variants.add(f"{alias}.{column_key}")
    return variants


def _normalized_sql_text(sql: str) -> str:
    return re.sub(r'["`\[\]]', "", str(sql or "").lower())


def _memory_key(contract: dict[str, Any]) -> str:
    return str(contract.get("memory_key") or contract.get("key") or "db_memory")


def _dedupe_validation_facts(
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for fact in facts:
        key = json.dumps(fact, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped
