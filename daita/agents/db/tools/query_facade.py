"""
Generic query tools for ``from_db()`` agents.

Plugin-specific tools still belong to individual database plugins. These
facades provide the LLM with a stable ``from_db`` capability surface while
delegating execution to the active plugin's existing guardrails.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from ....core.tools import AgentTool

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
    query_handler = _preflight_sql_handler(plugin._tool_query, schema)
    tools = [
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


def _preflight_sql_handler(handler: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        sql = str((args or {}).get("sql") or "")
        validation = _validate_sql_against_schema(sql, schema)
        if validation.get("error"):
            return validation
        return await handler(args)

    return _handler


def _validate_sql_against_schema(sql: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    if not sql.strip():
        return {"error": "Missing SQL query", "suggested_next_tool": "db_query"}

    table_columns = _schema_table_columns(schema)
    if not table_columns:
        return {"ok": True}

    aliases, unknown_tables = _extract_table_aliases(sql, table_columns)
    missing_columns = _missing_column_references(sql, aliases, table_columns)

    if not unknown_tables and not missing_columns:
        return {"ok": True}

    return {
        "error": "SQL preflight failed against known schema",
        "unknown_tables": sorted(unknown_tables),
        "missing_columns": missing_columns,
        "available_tables": sorted(table_columns)[:50],
        "suggested_next_tool": "db_inspect_table",
        "guidance": (
            "Inspect the missing or ambiguous table, then retry with exact table "
            "and column names from the schema."
        ),
    }


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
    for match in re.finditer(
        r"\b(?:from|join)\s+([A-Za-z_][\w.]*)(?:\s+(?:as\s+)?([A-Za-z_][\w]*))?",
        sql,
        flags=re.IGNORECASE,
    ):
        table = match.group(1).strip().strip('"`[]').lower()
        alias = (match.group(2) or table.split(".")[-1]).strip().strip('"`[]').lower()
        if alias in {"on", "where", "join", "left", "right", "inner", "outer", "full", "cross"}:
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
    for alias, column in re.findall(
        r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", sql
    ):
        alias_l = alias.lower()
        column_l = column.lower()
        if alias_l in _SQL_QUALIFIER_SKIP:
            continue
        table = aliases.get(alias_l)
        if not table:
            key = (alias_l, column_l)
            if key not in seen:
                missing.append({"table_or_alias": alias, "column": column, "reason": "unknown table alias"})
                seen.add(key)
            continue
        if column_l not in table_columns.get(table, set()):
            key = (table, column_l)
            if key not in seen:
                missing.append({"table": table, "column": column, "reason": "column not found"})
                seen.add(key)
    return missing


_SQL_QUALIFIER_SKIP = {
    "date",
    "time",
    "timestamp",
    "json",
    "jsonb",
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
