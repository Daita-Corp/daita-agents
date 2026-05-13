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
    find_join_paths,
    inspect_table,
    list_tables,
    search_schema,
)


def create_db_list_tables_tool(schema: Dict[str, Any]) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return list_tables(
            schema,
            pattern=args.get("pattern"),
            limit=args.get("limit", 30),
            offset=args.get("offset", 0),
        )

    return _schema_tool(
        name="db_list_tables",
        description=(
            "List discovered table names with row/column counts. Use when table names "
            "are ambiguous. Never returns row data."
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
                    "description": "Maximum tables to return, capped at 100. Default 30.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                },
            },
            "required": [],
        },
        handler=handler,
    )


def create_db_search_schema_tool(schema: Dict[str, Any]) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return search_schema(
            schema,
            query=args.get("query") or "",
            limit=args.get("limit", 12),
        )

    return _schema_tool(
        name="db_search_schema",
        description=(
            "Search discovered table and column metadata by terms. Use before SQL when "
            "the relevant table is unclear. Never returns row data."
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
                    "description": "Maximum table matches to return, capped at 50. Default 12.",
                },
            },
            "required": ["query"],
        },
        handler=handler,
    )


def create_db_inspect_table_tool(schema: Dict[str, Any], plugin: Any) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return inspect_table(
            schema,
            table_name=args.get("table_name") or "",
            column_pattern=args.get("column_pattern"),
            include_columns=args.get("include_columns", True),
            limit=args.get("limit", 40),
            offset=args.get("offset", 0),
            blocked_columns=getattr(plugin, "blocked_columns", set()),
        )

    return _schema_tool(
        name="db_inspect_table",
        description=(
            "Inspect bounded metadata for one table, including columns and FK links. "
            "Use after db_search_schema. Never returns rows."
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
                    "description": "Maximum columns to return, capped at 200. Default 40.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Column pagination offset. Default 0.",
                },
            },
            "required": ["table_name"],
        },
        handler=handler,
    )


def create_db_describe_relationships_tool(schema: Dict[str, Any]) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return describe_relationships(
            schema,
            table_name=args.get("table_name"),
            limit=args.get("limit", 24),
        )

    return _schema_tool(
        name="db_describe_relationships",
        description=(
            "Describe discovered foreign-key relationships, optionally scoped to one table."
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
                    "description": "Maximum relationships to return, capped at 100. Default 24.",
                },
            },
            "required": [],
        },
        handler=handler,
    )


def create_db_find_join_path_tool(schema: Dict[str, Any]) -> AgentTool:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return find_join_paths(
            schema,
            from_tables=args.get("from_tables") or [],
            to_tables=args.get("to_tables") or [],
            max_hops=args.get("max_hops", 4),
            max_paths=args.get("max_paths", 5),
        )

    return _schema_tool(
        name="db_find_join_path",
        description=(
            "Find SQL-ready foreign-key join paths between tables. Use before SQL "
            "when a question requires joining facts to entities through one or more "
            "relationships. Returns tables, join predicates, confidence, and warnings."
        ),
        parameters={
            "type": "object",
            "properties": {
                "from_tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Starting table names, usually fact/event tables.",
                },
                "to_tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target table names, usually entity/dimension tables.",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "Maximum FK hops to traverse, capped at 6. Default 4.",
                },
                "max_paths": {
                    "type": "integer",
                    "description": "Maximum paths to return, capped at 8. Default 5.",
                },
            },
            "required": ["from_tables", "to_tables"],
        },
        handler=handler,
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
        create_db_find_join_path_tool(schema),
        create_db_describe_relationships_tool(schema),
    ):
        agent.tool_registry.register(tool)


def _schema_tool(
    *,
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler: Any,
) -> AgentTool:
    return AgentTool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
        category="schema",
    )
