"""SQL preflight validation for ``from_db`` agents."""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any

from ..schema.metadata import (
    normalize_identifier,
    schema_table_columns,
)
from .intent import is_count_metric_name


def validate_sql_against_schema(sql: str, schema: dict[str, Any]) -> dict[str, Any]:
    if not sql.strip():
        return {
            "error": "Missing SQL query",
            "repair_required": True,
            "preflight_failed": True,
            "suggested_next_tool": "db_validate_sql",
            "sql_fingerprint": sql_fingerprint(sql),
        }

    table_columns = schema_table_columns(schema)
    if not table_columns:
        return {"ok": True}

    aliases, unknown_tables = _extract_table_aliases(sql, table_columns)
    missing_columns = _missing_column_references(sql, aliases, table_columns)

    if not unknown_tables and not missing_columns:
        return {"ok": True}

    inspect_tables = _tables_to_inspect(unknown_tables, missing_columns)
    return {
        "error": "SQL preflight failed against known schema",
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
            "db_inspect_table" if inspect_tables else "db_search_schema"
        ),
        "do_not_retry_same_sql": True,
        "guidance": (
            "Do not call db_query again with this exact SQL. Inspect the missing "
            "or ambiguous table, use db_find_join_path for join ambiguity, then "
            "call db_validate_sql or db_query with corrected SQL."
        ),
    }


def apply_required_field_validation(
    validation: dict[str, Any], sql: str, run_state: Any
) -> None:
    warnings = required_field_warnings(sql, run_state)
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


def required_field_warnings(sql: str, run_state: Any) -> list[dict[str, Any]]:
    if run_state is None:
        return []
    required_fields = getattr(run_state, "required_answer_fields", None) or []
    candidate_columns = getattr(run_state, "candidate_columns", None) or {}
    if not required_fields or not isinstance(candidate_columns, dict):
        return []

    referenced_columns = _referenced_sql_columns(sql)
    selected_expressions = _selected_expressions_by_alias(sql)
    warnings: list[dict[str, Any]] = []
    for field_name in required_fields:
        candidates = candidate_columns.get(field_name) or []
        required_columns = _high_confidence_required_columns(field_name, candidates)
        if not required_columns:
            continue
        if referenced_columns & required_columns:
            continue
        if _count_alias_satisfies_required_field(field_name, selected_expressions):
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


def _referenced_sql_columns(sql: str) -> set[str]:
    cleaned = _strip_sql_literals(sql)
    columns = {
        column.lower()
        for column in re.findall(r"\b[A-Za-z_][\w]*\.([A-Za-z_][\w]*)\b", cleaned)
    }
    for function_args in re.findall(r"\b[A-Za-z_][\w]*\s*\(([^)]*)\)", cleaned):
        for name in re.findall(r"\b([A-Za-z_][\w]*)\b", function_args):
            if name.lower() not in _SQL_KEYWORDS:
                columns.add(name.lower())
    return columns


def _selected_expressions_by_alias(sql: str) -> dict[str, str]:
    select_body = _select_body(sql)
    if not select_body:
        return {}

    expressions: dict[str, str] = {}
    for item in _split_sql_select_items(select_body):
        alias = _select_item_alias(item)
        if alias:
            expressions[normalize_identifier(alias)] = item
    return expressions


def _select_body(sql: str) -> str:
    cleaned = _strip_sql_literals(sql or "")
    match = re.search(
        r"\bselect\b(.*?)\bfrom\b", cleaned, flags=re.IGNORECASE | re.DOTALL
    )
    if not match:
        return ""
    return match.group(1)


def _split_sql_select_items(select_body: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(select_body):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            item = select_body[start:idx].strip()
            if item:
                items.append(item)
            start = idx + 1
    tail = select_body[start:].strip()
    if tail:
        items.append(tail)
    return items


def _select_item_alias(item: str) -> str:
    match = re.search(r"\bas\s+([A-Za-z_][\w]*)\s*$", item, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    tokens = re.findall(r"[A-Za-z_][\w]*", item)
    if len(tokens) >= 2 and not item.rstrip().endswith(")"):
        return tokens[-1]
    return ""


def _count_alias_satisfies_required_field(
    field_name: str, selected_expressions: dict[str, str]
) -> bool:
    normalized_field = normalize_identifier(field_name)
    expression = selected_expressions.get(normalized_field)
    if not expression:
        return False
    if not is_count_metric_name(field_name):
        return False
    return bool(re.search(r"\bcount\s*\(", expression, flags=re.IGNORECASE))


def _high_confidence_required_columns(
    field_name: str, candidates: list[dict[str, Any]]
) -> set[str]:
    normalized_field = normalize_identifier(field_name)
    out: set[str] = set()
    for candidate in candidates:
        column = str(candidate.get("column") or "").strip().lower()
        if not column:
            continue
        score = int(candidate.get("score") or 0)
        normalized_column = normalize_identifier(column)
        if score >= 6 and (
            normalized_column == normalized_field
            or normalized_column.endswith(normalized_field)
            or normalized_field.endswith(normalized_column)
        ):
            out.add(column)
    return out


def _strip_sql_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "", sql or "")


def _extract_table_aliases(
    sql: str, table_columns: dict[str, set[str]]
) -> tuple[dict[str, str], set[str]]:
    aliases: dict[str, str] = {}
    unknown_tables: set[str] = set()
    stop_words = "|".join(sorted(_SQL_TABLE_ALIAS_STOP_WORDS))
    for match in re.finditer(
        rf"\b(?:from|join)\s+([A-Za-z_][\w.]*)(?:\s+(?:as\s+)?(?!{stop_words}\b)([A-Za-z_][\w]*))?",
        sql,
        flags=re.IGNORECASE,
    ):
        table = match.group(1).strip().strip('"`[]').lower()
        alias = (match.group(2) or table.split(".")[-1]).strip().strip('"`[]').lower()
        if alias in _SQL_TABLE_ALIAS_STOP_WORDS:
            alias = table.split(".")[-1]
        if table not in table_columns:
            unknown_tables.add(table)
            continue
        aliases[alias] = table
        aliases[table] = table
        aliases[table.split(".")[-1]] = table
    return aliases, unknown_tables


def _missing_column_references(
    sql: str, aliases: dict[str, str], table_columns: dict[str, set[str]]
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias, column in re.findall(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", sql):
        alias_l = alias.lower()
        column_l = column.lower()
        if f"{alias_l}.{column_l}" in table_columns:
            continue
        if alias_l in _SQL_QUALIFIER_SKIP:
            continue
        table = aliases.get(alias_l)
        if not table:
            key = (alias_l, column_l)
            if key not in seen:
                missing.append(
                    {
                        "table_or_alias": alias,
                        "column": column,
                        "reason": "unknown table alias",
                    }
                )
                seen.add(key)
            continue
        if column_l not in table_columns.get(table, set()):
            key = (table, column_l)
            if key not in seen:
                missing.append(
                    {"table": table, "column": column, "reason": "column not found"}
                )
                seen.add(key)
    return missing


def _tables_to_inspect(
    unknown_tables: set[str], missing_columns: list[dict[str, str]]
) -> list[str]:
    tables: list[str] = []
    for table in sorted(unknown_tables):
        if table not in tables:
            tables.append(table)
    for item in missing_columns:
        table = item.get("table") or item.get("table_or_alias")
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


_SQL_QUALIFIER_SKIP = {
    "date",
    "time",
    "timestamp",
    "json",
    "jsonb",
}


_SQL_TABLE_ALIAS_STOP_WORDS = {
    "cross",
    "full",
    "group",
    "having",
    "inner",
    "join",
    "left",
    "limit",
    "on",
    "order",
    "outer",
    "right",
    "using",
    "where",
}


_SQL_KEYWORDS = _SQL_TABLE_ALIAS_STOP_WORDS | {
    "and",
    "as",
    "asc",
    "between",
    "by",
    "case",
    "desc",
    "distinct",
    "else",
    "end",
    "from",
    "in",
    "is",
    "not",
    "null",
    "or",
    "select",
    "then",
    "when",
}
