"""SQL preflight validation for DB agents."""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any

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
    if not table_columns:
        return {
            "ok": True,
            "sql_fingerprint": sql_fingerprint(sql),
            "statement_facts": sql_statement_facts(sql, analysis),
        }

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


def _validation_ok(sql: str, analysis: SqlAnalysis) -> dict[str, Any]:
    return {
        "ok": True,
        "sql_fingerprint": sql_fingerprint(sql),
        "statement_facts": sql_statement_facts(sql, analysis),
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
