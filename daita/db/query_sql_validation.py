"""SQL preflight validation for DB agents."""

from __future__ import annotations

import difflib
import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from daita.db.query_metadata import (
    normalize_identifier,
    schema_table_columns,
)
from daita.db.query_tool_views import (
    CATALOG_INSPECT_TOOL_VIEW,
    CATALOG_RELATIONSHIP_TOOL_VIEW,
    CATALOG_SEARCH_TOOL_VIEW,
    DB_QUERY_TOOL_VIEW,
    DB_VALIDATE_SQL_TOOL_VIEW,
)
from daita.db.sql_analysis import SqlAnalysis, SqlAnalysisError, analyze_sql


def validate_sql_against_schema(
    sql: str,
    schema: dict[str, Any],
    *,
    dialect: str = "",
    analysis: SqlAnalysis | None = None,
    params: list[Any] | tuple[Any, ...] = (),
    groundings: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    source_owner: str = "",
    validator_tool: str = DB_VALIDATE_SQL_TOOL_VIEW,
    catalog_search_tool: str = CATALOG_SEARCH_TOOL_VIEW,
    catalog_inspect_tool: str = CATALOG_INSPECT_TOOL_VIEW,
    relationship_tool: str = CATALOG_RELATIONSHIP_TOOL_VIEW,
    execution_tool: str = DB_QUERY_TOOL_VIEW,
) -> dict[str, Any]:
    if not sql.strip():
        return _validation_error(
            "Missing SQL query",
            error_type="dialect_parse_error",
            sql=sql,
            suggested_next_tool=validator_tool,
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
                suggested_next_tool=validator_tool,
            )

    table_columns = schema_table_columns(schema)
    unknown_tables = (
        _unknown_tables(analysis, table_columns) if table_columns else set()
    )
    missing_columns = _missing_columns(analysis, table_columns) if table_columns else []

    if not unknown_tables and not missing_columns:
        coverage = grounding_coverage_result(
            analysis,
            params=params,
            groundings=groundings,
            schema=schema,
            source_owner=source_owner,
        )
        if coverage["valid"] is not True:
            return {
                "error": "SQL omits or conflicts with accepted catalog grounding.",
                "error_type": "grounding_coverage_error",
                "repair_required": True,
                "preflight_failed": True,
                "sql_fingerprint": sql_fingerprint(sql),
                "statement_facts": sql_statement_facts(sql, analysis),
                "grounding_coverage": coverage,
                "suggested_next_tool": execution_tool,
                "do_not_retry_same_sql": True,
                "guidance": (
                    "Revise the SQL so every enforceable grounding is represented "
                    "by an exact predicate and literal or bound parameter."
                ),
            }
        return _validation_ok(sql, analysis, grounding_coverage=coverage)

    inspect_tables = _tables_to_inspect(unknown_tables, missing_columns)
    return {
        "error": "SQL preflight failed against known schema",
        "error_type": "schema_reference_error",
        "repair_required": True,
        "preflight_failed": True,
        "sql_fingerprint": sql_fingerprint(sql),
        "statement_facts": sql_statement_facts(sql, analysis),
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
            catalog_inspect_tool if inspect_tables else catalog_search_tool
        ),
        "do_not_retry_same_sql": True,
        "guidance": (
            f"Do not call {execution_tool} again with this exact SQL. Inspect "
            f"the missing or ambiguous table, use {relationship_tool} for join "
            f"ambiguity, then call {validator_tool} or {execution_tool} with "
            "corrected SQL."
        ),
    }


def grounding_coverage_result(
    analysis: SqlAnalysis,
    *,
    params: list[Any] | tuple[Any, ...] = (),
    groundings: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    schema: Mapping[str, Any] | None = None,
    source_owner: str = "",
) -> dict[str, Any]:
    """Compare accepted catalog groundings with parsed SQL predicate values."""

    schema = schema if isinstance(schema, Mapping) else {}
    catalog_facts = schema.get("_catalog")
    catalog_facts = catalog_facts if isinstance(catalog_facts, Mapping) else {}
    predicates = _resolved_value_predicates(analysis, params)
    covered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    advisory: list[dict[str, Any]] = []

    for raw in list(groundings or ())[:12]:
        grounding = dict(raw) if isinstance(raw, Mapping) else {}
        reason = _grounding_advisory_reason(
            grounding,
            catalog_facts=catalog_facts,
            source_owner=source_owner,
        )
        grounding_id = str(grounding.get("grounding_id") or "")[:80]
        if reason:
            advisory.append(
                {
                    "grounding_id": grounding_id,
                    "status": "advisory",
                    "reason": reason,
                }
            )
            continue

        table = str(grounding.get("table") or "")
        column = str(grounding.get("column") or "")
        expected = grounding.get("value")
        applicable = [
            predicate
            for predicate in predicates
            if _predicate_targets_grounding(
                predicate,
                table=table,
                column=column,
                analysis=analysis,
                schema=schema,
            )
        ]
        item = {
            "grounding_id": grounding_id,
            "target": f"{table}.{column}",
            "value": expected,
        }
        exact_match = any(
            _predicate_exactly_covers(predicate, expected) for predicate in applicable
        )
        conflicts = [
            predicate
            for predicate in applicable
            if _predicate_conflicts(predicate, expected)
        ]
        if conflicts:
            conflicting.append(
                {
                    **item,
                    "status": "conflicting",
                    "predicate_operators": sorted(
                        {str(predicate["operator"]) for predicate in conflicts}
                    )[:4],
                }
            )
        elif exact_match:
            covered.append({**item, "status": "covered"})
        else:
            missing.append({**item, "status": "missing"})

    status = (
        "conflicting"
        if conflicting
        else "missing" if missing else "covered" if covered else "advisory"
    )
    return {
        "status": status,
        "valid": not missing and not conflicting,
        "applicable_count": len(covered) + len(missing) + len(conflicting),
        "predicate_count": len(predicates),
        "covered": covered,
        "missing": missing,
        "conflicting": conflicting,
        "advisory": advisory,
    }


def sql_fingerprint(sql: str) -> str:
    normalized = re.sub(
        r"\s+",
        " ",
        (sql or "").strip().rstrip(";").rstrip().lower(),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def sql_statement_facts(
    sql: str,
    analysis: SqlAnalysis,
    *,
    guardrail_result: str = "passed",
) -> dict[str, Any]:
    """Return audit-safe statement facts owned by SQL validation."""
    mutating_classes = tuple(sorted(set(analysis.mutating_statement_types)))
    destructive_classes = tuple(
        item
        for item in mutating_classes
        if item in {"ALTER", "DELETE", "DROP", "TRUNCATETABLE", "TRUNCATE"}
    )
    admin_classes = tuple(
        item
        for item in mutating_classes
        if item in {"ALTER", "CREATE", "DROP", "TRUNCATETABLE", "TRUNCATE"}
    )
    target_resources = tuple(
        sorted({table.short_key for table in analysis.tables if not table.is_cte})
    )
    return {
        "statement_type": analysis.statement_type,
        "statement_count": analysis.statement_count,
        "is_read": analysis.is_read,
        "has_limit": analysis.has_limit,
        "mutating_statement_classes": list(mutating_classes),
        "destructive_statement_classes": list(destructive_classes),
        "admin_statement_classes": list(admin_classes),
        "target_resources": list(target_resources),
        "guardrail_result": guardrail_result,
        "sql_fingerprint": sql_fingerprint(sql),
    }


def _validation_ok(
    sql: str,
    analysis: SqlAnalysis,
    *,
    grounding_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "ok": True,
        "sql_fingerprint": sql_fingerprint(sql),
        "statement_facts": sql_statement_facts(sql, analysis),
        "referenced_tables": sorted(
            {table.key for table in analysis.tables if not table.is_cte}
        ),
        "referenced_columns": sorted(analysis.referenced_column_names),
        "selected_columns": _selected_output_columns(analysis),
    }
    if grounding_coverage is not None:
        result["grounding_coverage"] = dict(grounding_coverage)
    return result


def _resolved_value_predicates(
    analysis: SqlAnalysis,
    params: list[Any] | tuple[Any, ...],
) -> list[dict[str, Any]]:
    predicates: list[dict[str, Any]] = []
    for predicate in analysis.literal_predicates:
        predicates.append(
            {
                "table": predicate.column.qualifier_key,
                "column": predicate.column.key,
                "operator": predicate.operator,
                "values": [
                    _typed_sql_literal(value, kind)
                    for value, kind in zip(
                        predicate.values,
                        predicate.value_kinds,
                    )
                ],
                "resolved": True,
                "negated": predicate.negated,
                "disjunctive": predicate.disjunctive,
            }
        )
    for predicate in analysis.parameter_predicates:
        values: list[Any] = [
            _typed_sql_literal(value, kind)
            for value, kind in zip(
                predicate.literal_values,
                predicate.literal_value_kinds,
            )
        ]
        resolved = True
        for index in predicate.parameter_indexes:
            if index < 0 or index >= len(params):
                resolved = False
                continue
            values.append(params[index])
        predicates.append(
            {
                "table": predicate.column.qualifier_key,
                "column": predicate.column.key,
                "operator": predicate.operator,
                "values": values,
                "resolved": resolved,
                "negated": predicate.negated,
                "disjunctive": predicate.disjunctive,
            }
        )
    return predicates


def _typed_sql_literal(value: str, kind: str) -> Any:
    if kind != "number":
        return value
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return value
    if parsed == parsed.to_integral_value():
        return int(parsed)
    return float(parsed)


def _grounding_advisory_reason(
    grounding: Mapping[str, Any],
    *,
    catalog_facts: Mapping[str, Any],
    source_owner: str,
) -> str:
    declared_reason = str(grounding.get("reason") or "")
    if grounding.get("status") != "enforceable":
        return declared_reason or "not_enforceable"
    if grounding.get("policy_eligible") is not True:
        return "policy_ineligible"
    if grounding.get("unambiguous") is not True:
        return "ambiguous"
    if (
        grounding.get("match_kind") != "exact_match"
        or grounding.get("confidence") != 1.0
    ):
        return "not_exact"
    if (
        grounding.get("fresh") is not True
        or grounding.get("value_freshness") != "fresh"
        or grounding.get("profile_status") != "profiled"
        or grounding.get("source_fingerprint_status")
        not in {"authoritative", "best_effort"}
    ):
        return "stale_or_unverified"
    if any(
        bool(grounding.get(key))
        for key in ("sampled", "truncated", "redacted", "blocked")
    ):
        return "ineligible_profile"
    current_owner = str(source_owner or catalog_facts.get("source_owner") or "")
    if not current_owner or grounding.get("source_owner") != current_owner:
        return "source_mismatch"
    if catalog_facts.get("freshness") != "fresh":
        return "catalog_not_fresh"
    if catalog_facts.get("revision_status") != "authoritative":
        return "catalog_revision_unverified"
    for key in ("source_revision", "catalog_revision", "schema_fingerprint"):
        expected = catalog_facts.get(key)
        actual = grounding.get(key)
        if not expected or not actual or str(actual) != str(expected):
            return "revision_mismatch"
    value = grounding.get("value")
    if isinstance(value, str) and (not value or len(value) > 128):
        return "value_not_exactly_comparable"
    if value is None or not isinstance(value, (str, int, float, bool)):
        return "value_not_exactly_comparable"
    if not str(grounding.get("table") or "") or not str(grounding.get("column") or ""):
        return "target_unavailable"
    return ""


def _predicate_targets_grounding(
    predicate: Mapping[str, Any],
    *,
    table: str,
    column: str,
    analysis: SqlAnalysis,
    schema: Mapping[str, Any],
) -> bool:
    if str(predicate.get("column") or "").lower() != column.lower():
        return False
    predicate_table = str(predicate.get("table") or "").lower()
    if predicate_table:
        return _table_refs_match(predicate_table, table)

    table_columns = schema_table_columns(dict(schema))
    matching_tables = []
    for table_ref in analysis.tables:
        if table_ref.is_cte:
            continue
        known_key = _known_table_key(table_ref.key, table_columns)
        if known_key and column.lower() in table_columns.get(known_key, set()):
            matching_tables.append(table_ref.key)
    return len(set(matching_tables)) == 1 and _table_refs_match(
        matching_tables[0], table
    )


def _table_refs_match(left: str, right: str) -> bool:
    left_key = normalize_identifier(left)
    right_key = normalize_identifier(right)
    return left_key == right_key or left_key.split(".")[-1] == right_key.split(".")[-1]


def _predicate_exactly_covers(predicate: Mapping[str, Any], expected: Any) -> bool:
    if (
        predicate.get("resolved") is not True
        or predicate.get("negated") is True
        or predicate.get("disjunctive") is True
    ):
        return False
    operator = str(predicate.get("operator") or "").lower()
    values = list(predicate.get("values") or ())
    return (
        operator in {"=", "in"}
        and len(values) == 1
        and _values_equal(values[0], expected)
    )


def _predicate_conflicts(predicate: Mapping[str, Any], expected: Any) -> bool:
    if predicate.get("resolved") is not True:
        return False
    operator = str(predicate.get("operator") or "").lower()
    values = list(predicate.get("values") or ())
    if predicate.get("negated") is True:
        return operator in {"=", "in"} and any(
            _values_equal(value, expected) for value in values
        )
    if operator in {"=", "in"} and values:
        return len(values) != 1 or not _values_equal(values[0], expected)
    if operator == "!=":
        return any(_values_equal(value, expected) for value in values)
    return False


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return Decimal(str(left)) == Decimal(str(right))
        except InvalidOperation:
            return False
    return type(left) is type(right) and left == right


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
