"""
Generic query tools for ``from_db()`` agents.

Plugin-specific tools still belong to individual database plugins. These
facades provide the LLM with a stable ``from_db`` capability surface while
delegating execution to the active plugin's existing guardrails.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ....core.tools import AgentTool
from ..query.planner import build_query_plan
from ..query.sql_validator import (
    apply_required_field_validation,
    sql_fingerprint,
    validate_sql_against_schema,
)
from ..query.sql_analysis import SqlAnalysis
from ..runtime.state import get_db_run_state

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
        sql = _normalize_sql_for_plugin(plugin, str((args or {}).get("sql") or ""))
        analysis = _validate_plugin_query_policy(plugin, sql)
        dialect = _schema_or_plugin_dialect(plugin, schema)
        validation = validate_sql_against_schema(
            sql, schema, dialect=dialect, analysis=analysis
        )
        state = get_db_run_state(plugin)
        apply_required_field_validation(
            validation, sql, state, dialect=dialect, analysis=analysis
        )
        if validation.get("error"):
            _record_sql_preflight_failure(plugin, state, validation)
            return validation
        if state is not None:
            state.record_validated_sql(sql_fingerprint(sql), validation)
        return {
            "ok": True,
            "sql_fingerprint": sql_fingerprint(sql),
            "message": "SQL passed schema preflight. It has not been executed.",
        }

    return _handler


def _preflight_sql_handler(plugin: Any, handler: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        args = args or {}
        if args.get("plan_id"):
            return await _execute_plan_id(plugin, handler, schema, args)

        sql = _normalize_sql_for_plugin(plugin, str((args or {}).get("sql") or ""))
        analysis = _validate_plugin_query_policy(plugin, sql)
        dialect = _schema_or_plugin_dialect(plugin, schema)
        validation = validate_sql_against_schema(
            sql, schema, dialect=dialect, analysis=analysis
        )
        run_state = get_db_run_state(plugin)
        apply_required_field_validation(
            validation, sql, run_state, dialect=dialect, analysis=analysis
        )
        if validation.get("error"):
            _record_sql_preflight_failure(plugin, run_state, validation)
            return validation
        fingerprint = sql_fingerprint(sql)
        if run_state is not None:
            run_state.record_validated_sql(fingerprint, validation)
        normalized_args = {**(args or {}), "sql": sql}
        result = await handler(normalized_args)
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


async def _execute_plan_id(
    plugin: Any, handler: Any, schema: Dict[str, Any], args: Dict[str, Any]
) -> Any:
    run_state = get_db_run_state(plugin)
    plan_id = str(args.get("plan_id") or "").strip()
    if run_state is None or not plan_id:
        return {
            "error": "Missing query plan state",
            "error_type": "framework_compiler_error",
            "repair_required": True,
            "suggested_next_tool": "db_plan_query",
        }
    stored = run_state.get_plan(plan_id)
    if stored is None:
        return {
            "error": f"Unknown query plan ID: {plan_id}",
            "error_type": "framework_compiler_error",
            "repair_required": True,
            "suggested_next_tool": "db_plan_query",
        }

    result = stored.get("result") or {}
    validation = result.get("validation") or {}
    sql = str(result.get("compiled_sql") or "")
    if not sql or not validation.get("ok"):
        return {
            "error": "Query plan is not executable",
            "error_type": "framework_compiler_error",
            "repair_required": True,
            "plan_id": plan_id,
            "validation": validation,
            "suggested_next_tool": "db_plan_query",
        }

    sql = _normalize_sql_for_plugin(plugin, sql)
    _validate_plugin_query_policy(plugin, sql)
    fingerprint = sql_fingerprint(sql)
    run_state.record_validated_sql(
        fingerprint,
        {
            "ok": True,
            "plan_id": plan_id,
            "source": "validated_query_ir",
            "sql_fingerprint": fingerprint,
        },
    )
    normalized_args = {**args, "sql": sql}
    normalized_args.pop("plan_id", None)
    result = await handler(normalized_args)
    run_state.record_executed_query(
        {
            "plan_id": plan_id,
            "sql_fingerprint": fingerprint,
            "sql": sql,
            "row_count": _result_row_count(result),
            "truncated": (
                bool(result.get("truncated")) if isinstance(result, dict) else False
            ),
        }
    )
    return result


def _normalize_sql_for_plugin(plugin: Any, sql: str) -> str:
    normalizer = getattr(type(plugin), "_normalize_sql", None)
    if callable(normalizer):
        return str(normalizer(sql))

    instance_normalizer = vars(plugin).get("_normalize_sql")
    if callable(instance_normalizer):
        return str(instance_normalizer(sql))

    return sql


def _validate_plugin_query_policy(plugin: Any, sql: str) -> SqlAnalysis | None:
    validator = getattr(plugin, "_validate_sql_policy", None)
    if callable(validator):
        analysis = validator(sql, operation="query")
        return analysis if isinstance(analysis, SqlAnalysis) else None
    return None


def _schema_or_plugin_dialect(plugin: Any, schema: Dict[str, Any]) -> str:
    return str(schema.get("database_type") or getattr(plugin, "sql_dialect", ""))


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
