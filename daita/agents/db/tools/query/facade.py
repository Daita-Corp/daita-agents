"""
Generic query tools for ``from_db()`` agents.

Plugin-specific tools still belong to individual database plugins. These
facades provide the LLM with a stable ``from_db`` capability surface while
delegating execution to the active plugin's existing guardrails.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .....core.tools import AgentTool
from .execution import (
    compile_and_query_tool_handler,
    plan_query_tool_handler,
    preflight_sql_handler,
    validate_sql_tool_handler,
)
from .intent_schema import query_intent_parameters

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
    query_handler = preflight_sql_handler(plugin, plugin._tool_query, schema)
    tools = [
        _db_tool(
            plugin,
            name="db_plan_query",
            description=(
                "Convert a database question into a structured query plan and "
                "resolve candidate tables, fields, and required join paths. Use "
                "this before SQL for analytic or multi-table questions. When the "
                "result includes suggested_next_tool=db_query, call db_query with "
                "suggested_next_arguments; do not call db_plan_query again for "
                "the same goal."
            ),
            parameters=query_intent_parameters(include_diagnostics=True),
            handler=plan_query_tool_handler(plugin, schema),
            timeout_seconds=10,
        ),
        _db_tool(
            plugin,
            name="db_compile_and_query",
            description=(
                "Plan, compile, validate, and execute a supported read-only "
                "database question in one step. Use for clear single-table "
                "counts, direct lookups, and simple filtered query intents."
            ),
            parameters=query_intent_parameters(mode="compile"),
            handler=compile_and_query_tool_handler(plugin, schema),
            timeout_seconds=60,
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
            handler=validate_sql_tool_handler(plugin, schema),
            timeout_seconds=10,
        ),
        _db_tool(
            plugin,
            name="db_query",
            description=(
                "Run a read-only SQL query against the connected database. Add LIMIT "
                "to control result size; a default LIMIT is applied when omitted. "
                "After db_plan_query returns a validated plan_id, pass that plan_id "
                "here instead of rewriting the SQL."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query for the active database dialect.",
                    },
                    "plan_id": {
                        "type": "string",
                        "description": "Validated plan ID returned by db_plan_query.",
                    },
                    "params": {
                        "type": "array",
                        "description": "Optional parameter values for placeholders.",
                        "items": {},
                    },
                },
                "required": [],
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
                retry_safe=False,
                replay_safe=False,
                side_effecting=True,
            )
        )
    return tools


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
    replay_safe: bool = True,
    retry_safe: bool = True,
    side_effecting: bool = False,
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
        retry_safe=retry_safe,
        replay_safe=replay_safe,
        side_effecting=side_effecting,
    )
