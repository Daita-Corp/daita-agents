"""
LLM-callable schema navigation tools for ``from_db()`` agents.

These tools expose bounded metadata from the schema that ``from_db`` has
already discovered. They do not execute SQL or return raw row data.
"""

from __future__ import annotations

from typing import Any, Dict

from ....core.tools import AgentTool
from ..navigation import (
    describe_relationships,
    inspect_table,
    list_tables,
    search_schema,
)


def create_db_list_tables_tool(schema: Dict[str, Any]) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return list_tables(
            schema,
            pattern=args.get("pattern"),
            limit=args.get("limit", 50),
            offset=args.get("offset", 0),
        )

    return AgentTool(
        name="db_list_tables",
        description=(
            "List bounded table metadata from the discovered schema. Use this before "
            "writing SQL when the schema is large, omitted from the prompt, or table "
            "names are ambiguous. Returns table names, row counts, and column counts; "
            "never returns row data."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Optional case-insensitive substring or glob pattern for table names.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum tables to return, capped at 100. Default 50.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                },
            },
            "required": [],
        },
        handler=handler,
        category="schema",
    )


def create_db_search_schema_tool(schema: Dict[str, Any]) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return search_schema(
            schema,
            query=args.get("query") or "",
            limit=args.get("limit", 20),
        )

    return AgentTool(
        name="db_search_schema",
        description=(
            "Search discovered schema metadata by table or column terms. Use this "
            "to find tables omitted from a large schema prompt before calling query "
            "tools. Returns bounded table matches and matching columns; never returns "
            "row data."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms, e.g. 'event feature account' or a partial table name.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum table matches to return, capped at 50. Default 20.",
                },
            },
            "required": ["query"],
        },
        handler=handler,
        category="schema",
    )


def create_db_inspect_table_tool(schema: Dict[str, Any], plugin: Any) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return inspect_table(
            schema,
            table_name=args.get("table_name") or "",
            column_pattern=args.get("column_pattern"),
            include_columns=args.get("include_columns", True),
            limit=args.get("limit", 100),
            offset=args.get("offset", 0),
            blocked_columns=getattr(plugin, "blocked_columns", set()),
        )

    return AgentTool(
        name="db_inspect_table",
        description=(
            "Inspect bounded metadata for one discovered table, including a paged "
            "column list and FK relationships. Use after db_search_schema or "
            "db_list_tables to identify exact columns before SQL. Never returns rows."
        ),
        parameters={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Exact table name to inspect.",
                },
                "column_pattern": {
                    "type": "string",
                    "description": "Optional case-insensitive substring or glob pattern for column names.",
                },
                "include_columns": {
                    "type": "boolean",
                    "description": "Whether to include column metadata. Default true.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum columns to return, capped at 200. Default 100.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Column pagination offset. Default 0.",
                },
            },
            "required": ["table_name"],
        },
        handler=handler,
        category="schema",
    )


def create_db_describe_relationships_tool(schema: Dict[str, Any]) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return describe_relationships(
            schema,
            table_name=args.get("table_name"),
            limit=args.get("limit", 50),
        )

    return AgentTool(
        name="db_describe_relationships",
        description=(
            "Describe discovered foreign-key relationships, optionally scoped to "
            "one table. Use this before joins in large or ambiguous schemas."
        ),
        parameters={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Optional table name to scope relationships.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum relationships to return, capped at 100. Default 50.",
                },
            },
            "required": [],
        },
        handler=handler,
        category="schema",
    )


def register_schema_navigation_tools(
    agent: Any,
    plugin: Any,
    schema: Dict[str, Any],
) -> None:
    for tool in (
        create_db_list_tables_tool(schema),
        create_db_search_schema_tool(schema),
        create_db_inspect_table_tool(schema, plugin),
        create_db_describe_relationships_tool(schema),
    ):
        agent.tool_registry.register(tool)
