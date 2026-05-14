"""
Generic query tools for ``from_db()`` agents.

Plugin-specific tools still belong to individual database plugins. These
facades provide the LLM with a stable ``from_db`` capability surface while
delegating execution to the active plugin's existing guardrails.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any, Dict, List, Set, Tuple

from ....core.tools import AgentTool
from ..planning import build_query_plan
from ..state import get_db_run_state

SQL_DATABASE_TYPES = {"postgresql", "postgres", "mysql", "sqlite", "snowflake"}


def create_db_query_tools(plugin: Any, schema: Dict[str, Any]) -> List[AgentTool]:
    db_type = str(schema.get("database_type") or getattr(plugin, "sql_dialect", ""))
    db_type = db_type.lower()
    if db_type == "mongodb":
        return _mongo_tools(plugin)
    if db_type in SQL_DATABASE_TYPES or hasattr(plugin, "_tool_query"):
        return _sql_tools(plugin, schema)
    return []


def _sql_tools(plugin: Any, schema: Dict[str, Any]) -> List[AgentTool]:
    query_handler = _preflight_sql_handler(plugin, plugin._tool_query, schema)
    tools = [
        _db_tool(
            plugin,
            name="db_plan_query",
            description=(
                "Convert a database question into a structured query plan and "
                "resolve candidate tables, fields, and required join paths. Use "
                "this before SQL for analytic or multi-table questions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Plain-English objective for the query.",
                    },
                    "required_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields or metrics the final answer must include.",
                    },
                    "candidate_tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Likely tables or collections involved.",
                    },
                    "required_joins": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Join requirements with from_tables and to_tables arrays."
                        ),
                    },
                    "filters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Natural-language filters or SQL-safe clauses.",
                    },
                    "aggregations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Requested metrics such as count, sum, average.",
                    },
                    "grouping": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to group by.",
                    },
                    "ordering": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sort requirements.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows requested, if any.",
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Assumptions needed to proceed.",
                    },
                    "answer_checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Completeness checks for the final answer.",
                    },
                },
                "required": ["goal"],
            },
            handler=_plan_query_tool_handler(plugin, schema),
            timeout_seconds=10,
        ),
        _db_tool(
            plugin,
            name="db_validate_sql",
            description=(
                "Validate a SQL SELECT query against the known schema without "
                "executing it. Use this before retrying SQL after a schema error."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to validate.",
                    }
                },
                "required": ["sql"],
            },
            handler=_validate_sql_tool_handler(plugin, schema),
            timeout_seconds=10,
        ),
        _db_tool(
            plugin,
            name="db_query",
            description=(
                "Run a read-only SQL query against the connected database. Add LIMIT "
                "to control result size; a default LIMIT is applied when omitted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query for the active database dialect.",
                    },
                    "params": {
                        "type": "array",
                        "description": "Optional parameter values for placeholders.",
                        "items": {},
                    },
                },
                "required": ["sql"],
            },
            handler=query_handler,
            timeout_seconds=60,
        ),
        _db_tool(
            plugin,
            name="db_count",
            description="Count rows in a table or collection with an optional filter.",
            parameters={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name to count.",
                    },
                    "filter": {
                        "type": "string",
                        "description": "Optional SQL WHERE clause without the WHERE keyword.",
                    },
                },
                "required": ["table"],
            },
            handler=plugin._tool_count,
            timeout_seconds=30,
        ),
        _db_tool(
            plugin,
            name="db_sample",
            description="Return a small random sample of rows from a table.",
            parameters={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name."},
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to sample. Default 5.",
                    },
                },
                "required": ["table"],
            },
            handler=plugin._tool_sample,
            timeout_seconds=30,
        ),
    ]
    if not getattr(plugin, "read_only", True) and hasattr(plugin, "_tool_execute"):
        tools.append(
            _db_tool(
                plugin,
                name="db_execute",
                description=(
                    "Execute a mutating SQL statement against the connected database. "
                    "Unavailable when the from_db agent is read-only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL statement to execute.",
                        },
                        "params": {
                            "type": "array",
                            "description": "Optional parameter values.",
                            "items": {},
                        },
                    },
                    "required": ["sql"],
                },
                handler=plugin._tool_execute,
                timeout_seconds=60,
            )
        )
    return tools


def _plan_query_tool_handler(plugin: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        return build_query_plan(
            args or {},
            schema,
            run_state=get_db_run_state(plugin),
        )

    return _handler


def _validate_sql_tool_handler(plugin: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        sql = str((args or {}).get("sql") or "")
        validation = _validate_sql_against_schema(sql, schema)
        state = get_db_run_state(plugin)
        _apply_required_field_validation(validation, sql, state)
        if validation.get("error"):
            _record_sql_preflight_failure(plugin, state, validation)
            return validation
        if state is not None:
            state.record_validated_sql(_sql_fingerprint(sql), validation)
        return {
            "ok": True,
            "sql_fingerprint": _sql_fingerprint(sql),
            "message": "SQL passed schema preflight. It has not been executed.",
        }

    return _handler


def _preflight_sql_handler(plugin: Any, handler: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        sql = str((args or {}).get("sql") or "")
        validation = _validate_sql_against_schema(sql, schema)
        run_state = get_db_run_state(plugin)
        _apply_required_field_validation(validation, sql, run_state)
        if validation.get("error"):
            _record_sql_preflight_failure(plugin, run_state, validation)
            return validation
        fingerprint = _sql_fingerprint(sql)
        if run_state is not None:
            run_state.record_validated_sql(fingerprint, validation)
        result = await handler(args)
        if run_state is not None:
            run_state.record_executed_query(
                {
                    "sql_fingerprint": fingerprint,
                    "sql": sql,
                    "row_count": _result_row_count(result),
                    "truncated": (
                        bool(result.get("truncated"))
                        if isinstance(result, dict)
                        else False
                    ),
                }
            )
        return result

    return _handler


def _record_sql_preflight_failure(
    plugin: Any, run_state: Any, validation: Dict[str, Any]
) -> None:
    fingerprint = validation["sql_fingerprint"]
    if run_state is not None:
        attempt_count = run_state.record_failed_sql(fingerprint)
    else:
        failures = getattr(plugin, "_daita_sql_preflight_failures", None)
        if not isinstance(failures, dict):
            failures = {}
            setattr(plugin, "_daita_sql_preflight_failures", failures)
        failures[fingerprint] = int(failures.get(fingerprint, 0)) + 1
        attempt_count = failures[fingerprint]
    validation["attempt_count"] = attempt_count
    if attempt_count > 1:
        validation.pop("error", None)
        validation["blocked_repeat"] = True
        validation["status"] = "repeated_invalid_sql_blocked"
        validation["message"] = (
            "This exact SQL already failed schema preflight. It was not "
            "executed again. Inspect the referenced tables or validate a "
            "different SQL statement before calling db_query."
        )


def _result_row_count(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    if isinstance(result.get("total_rows"), int):
        return result["total_rows"]
    rows = result.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def _apply_required_field_validation(
    validation: Dict[str, Any], sql: str, run_state: Any
) -> None:
    warnings = _required_field_warnings(sql, run_state)
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
            "sql_fingerprint": _sql_fingerprint(sql),
            "suggested_next_tool": "db_plan_query",
            "do_not_retry_same_sql": True,
            "guidance": (
                "Revise the SQL so selected expressions use the required fields "
                "from db_plan_query. Do not substitute different metrics while "
                "keeping the original alias."
            ),
        }
    )


def _required_field_warnings(sql: str, run_state: Any) -> List[Dict[str, Any]]:
    if run_state is None:
        return []
    required_fields = getattr(run_state, "required_answer_fields", None) or []
    candidate_columns = getattr(run_state, "candidate_columns", None) or {}
    if not required_fields or not isinstance(candidate_columns, dict):
        return []

    referenced_columns = _referenced_sql_columns(sql)
    warnings: List[Dict[str, Any]] = []
    for field_name in required_fields:
        candidates = candidate_columns.get(field_name) or []
        required_columns = _high_confidence_required_columns(field_name, candidates)
        if not required_columns:
            continue
        if referenced_columns & required_columns:
            continue
        warnings.append(
            {
                "required_field": field_name,
                "expected_columns": sorted(required_columns)[:8],
                "reason": "required field not referenced by SQL",
            }
        )
    return warnings


def _referenced_sql_columns(sql: str) -> Set[str]:
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


def _high_confidence_required_columns(
    field_name: str, candidates: List[Dict[str, Any]]
) -> Set[str]:
    normalized_field = _normalize_identifier(field_name)
    out: Set[str] = set()
    for candidate in candidates:
        column = str(candidate.get("column") or "").strip().lower()
        if not column:
            continue
        score = int(candidate.get("score") or 0)
        normalized_column = _normalize_identifier(column)
        if score >= 6 and (
            normalized_column == normalized_field
            or normalized_column.endswith(normalized_field)
            or normalized_field.endswith(normalized_column)
        ):
            out.add(column)
    return out


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _strip_sql_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "", sql or "")


def _validate_sql_against_schema(sql: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    if not sql.strip():
        return {
            "error": "Missing SQL query",
            "repair_required": True,
            "preflight_failed": True,
            "suggested_next_tool": "db_validate_sql",
            "sql_fingerprint": _sql_fingerprint(sql),
        }

    table_columns = _schema_table_columns(schema)
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
        "sql_fingerprint": _sql_fingerprint(sql),
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


def _sql_fingerprint(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", (sql or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _schema_table_columns(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    tables: Dict[str, Set[str]] = {}
    for table in schema.get("tables") or []:
        if not isinstance(table, dict):
            continue
        name = str(table.get("name") or table.get("table_name") or "").strip()
        if not name:
            continue
        columns: Set[str] = set()
        for column in table.get("columns") or []:
            col_name = _column_name(column)
            if col_name:
                columns.add(col_name.lower())
        tables[name.lower()] = columns
        if "." in name:
            tables[name.split(".")[-1].lower()] = columns
    return tables


def _column_name(column: Any) -> str:
    if isinstance(column, dict):
        return str(column.get("name") or column.get("column_name") or "").strip()
    if isinstance(column, str):
        return column.split(":", 1)[0].strip()
    return ""


def _extract_table_aliases(
    sql: str, table_columns: Dict[str, Set[str]]
) -> Tuple[Dict[str, str], Set[str]]:
    aliases: Dict[str, str] = {}
    unknown_tables: Set[str] = set()
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
    sql: str, aliases: Dict[str, str], table_columns: Dict[str, Set[str]]
) -> List[Dict[str, str]]:
    missing: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for alias, column in re.findall(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", sql):
        alias_l = alias.lower()
        column_l = column.lower()
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
    unknown_tables: Set[str], missing_columns: List[Dict[str, str]]
) -> List[str]:
    tables: List[str] = []
    for table in sorted(unknown_tables):
        if table not in tables:
            tables.append(table)
    for item in missing_columns:
        table = item.get("table") or item.get("table_or_alias")
        if table and table not in tables:
            tables.append(table)
    return tables[:5]


def _available_columns_for_tables(
    table_columns: Dict[str, Set[str]], tables: List[str]
) -> Dict[str, List[str]]:
    available: Dict[str, List[str]] = {}
    for table in tables:
        table_l = table.lower()
        if table_l in table_columns:
            available[table] = sorted(table_columns[table_l])[:80]
    return available


def _column_candidates(
    missing_columns: List[Dict[str, str]], table_columns: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    candidates: Dict[str, List[str]] = {}
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
    unknown_tables: Set[str], table_columns: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    tables = sorted(table_columns)
    candidates: Dict[str, List[str]] = {}
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


def _mongo_tools(plugin: Any) -> List[AgentTool]:
    tools = [
        _db_tool(
            plugin,
            name="db_find",
            description=(
                "Find documents in a MongoDB collection. Use projection and limit "
                "to keep result size manageable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection name to search.",
                    },
                    "filter": {
                        "type": "object",
                        "description": "MongoDB query filter. Empty object matches all documents.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum documents to return. Default 50.",
                    },
                    "projection": {
                        "type": "object",
                        "description": "Optional field projection.",
                    },
                },
                "required": ["collection"],
            },
            handler=plugin._tool_find,
            timeout_seconds=60,
        ),
        _db_tool(
            plugin,
            name="db_aggregate",
            description="Run a MongoDB aggregation pipeline.",
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection name.",
                    },
                    "pipeline": {
                        "type": "array",
                        "description": "MongoDB aggregation pipeline stages.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["collection", "pipeline"],
            },
            handler=plugin._tool_aggregate,
            timeout_seconds=60,
        ),
        _db_tool(
            plugin,
            name="db_count",
            description="Count documents in a MongoDB collection, optionally filtered.",
            parameters={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection name.",
                    },
                    "filter": {
                        "type": "object",
                        "description": "Optional MongoDB query filter.",
                    },
                },
                "required": ["collection"],
            },
            handler=plugin._tool_count,
            timeout_seconds=30,
        ),
    ]
    return tools


def _db_tool(
    plugin: Any,
    *,
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler: Any,
    timeout_seconds: int,
) -> AgentTool:
    return AgentTool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
        category="database",
        source="from_db",
        plugin_name=plugin.__class__.__name__,
        timeout_seconds=timeout_seconds,
    )
