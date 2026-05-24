"""Execution handlers for core ``from_db`` query tools."""

from __future__ import annotations

from typing import Any, Dict, List

from .....core.tracing import TraceType
from ...query.evidence import collect_query_evidence
from ...query.planner import build_query_plan
from ...query.sql_analysis import SqlAnalysis
from ...query.sql_validator import (
    apply_required_field_validation,
    sql_fingerprint,
    validate_sql_against_schema,
)
from ...runtime.state import get_db_run_state
from ...utils import string_list


def plan_query_tool_handler(plugin: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        from ...runtime.tracing import db_trace_span

        async with db_trace_span(
            plugin,
            "from_db.plan_query",
            trace_type=TraceType.TOOL_EXECUTION,
        ) as (trace_manager, span_id):
            run_state = get_db_run_state(plugin)
            evidence = await collect_query_evidence(
                str((args or {}).get("goal") or ""),
                args or {},
                schema,
                run_state=run_state,
                catalog=vars(plugin).get("_db_catalog"),
                store_id=vars(plugin).get("_db_catalog_store_id"),
            )
            result = build_query_plan(
                args or {},
                schema,
                run_state=run_state,
                include_diagnostics=_include_plan_diagnostics(args),
                evidence=evidence,
                catalog=vars(plugin).get("_db_catalog"),
                store_id=vars(plugin).get("_db_catalog_store_id"),
            )
            trace_manager.record_output(span_id, _plan_trace_summary(result))
            return result

    return _handler


def compile_and_query_tool_handler(plugin: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        from ...runtime.tracing import db_trace_span

        async with db_trace_span(
            plugin,
            "from_db.compile_and_query",
            trace_type=TraceType.TOOL_EXECUTION,
        ) as (trace_manager, span_id):
            run_state = get_db_run_state(plugin)
            evidence = await collect_query_evidence(
                str((args or {}).get("goal") or ""),
                args or {},
                schema,
                run_state=run_state,
                catalog=vars(plugin).get("_db_catalog"),
                store_id=vars(plugin).get("_db_catalog_store_id"),
            )
            plan_result = build_query_plan(
                args or {},
                schema,
                run_state=run_state,
                include_diagnostics=False,
                evidence=evidence,
                catalog=vars(plugin).get("_db_catalog"),
                store_id=vars(plugin).get("_db_catalog_store_id"),
            )
            sql = str(plan_result.get("compiled_sql") or "").strip()
            validation = plan_result.get("validation") or {}
            if not sql or not validation.get("ok"):
                result = _compile_and_query_repair_payload(plan_result)
                trace_manager.record_output(span_id, result)
                return result

            sql = _normalize_sql_for_plugin(plugin, sql)
            analysis = _validate_plugin_query_policy(plugin, sql)
            validation = await trace_sql_validation(plugin, schema, sql, analysis)
            apply_required_field_validation(
                validation,
                sql,
                run_state,
                dialect=schema_or_plugin_dialect(plugin, schema),
                analysis=analysis,
            )
            if validation.get("error"):
                record_sql_preflight_failure(
                    plugin,
                    run_state,
                    validation,
                    source_tool="db_compile_and_query",
                )
                result = {
                    **validation,
                    "plan_id": plan_result.get("plan_id"),
                    "repair_required": True,
                    "suggested_next_tool": "db_plan_query",
                }
                trace_manager.record_output(span_id, _compile_and_query_trace(result))
                return result

            fingerprint = sql_fingerprint(sql)
            if run_state is not None:
                run_state.record_validated_sql(
                    fingerprint,
                    validation,
                    source_tool="db_compile_and_query",
                )
            query_result = await trace_sql_execution(
                plugin, plugin._tool_query, {"sql": sql}
            )
            if run_state is not None:
                run_state.record_executed_query(
                    {
                        "plan_id": plan_result.get("plan_id"),
                        "sql_fingerprint": fingerprint,
                        "sql": sql,
                        "columns": result_columns(query_result),
                        "selected_columns": validation.get("selected_columns") or [],
                        "row_count": result_row_count(query_result),
                        "truncated": (
                            bool(query_result.get("truncated"))
                            if isinstance(query_result, dict)
                            else False
                        ),
                    },
                    source_tool="db_compile_and_query",
                )
            result = _compile_and_query_result(
                plan_result, validation, sql, query_result, args or {}
            )
            trace_manager.record_output(span_id, _compile_and_query_trace(result))
            return result

    return _handler


def validate_sql_tool_handler(plugin: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        sql = _normalize_sql_for_plugin(plugin, str((args or {}).get("sql") or ""))
        analysis = _validate_plugin_query_policy(plugin, sql)
        dialect = schema_or_plugin_dialect(plugin, schema)
        validation = await trace_sql_validation(plugin, schema, sql, analysis)
        state = get_db_run_state(plugin)
        apply_required_field_validation(
            validation, sql, state, dialect=dialect, analysis=analysis
        )
        if validation.get("error"):
            record_sql_preflight_failure(
                plugin, state, validation, source_tool="db_validate_sql"
            )
            return validation
        if state is not None:
            state.record_validated_sql(
                sql_fingerprint(sql),
                validation,
                source_tool="db_validate_sql",
            )
        return {
            "ok": True,
            "sql_fingerprint": sql_fingerprint(sql),
            "message": "SQL passed schema preflight. It has not been executed.",
        }

    return _handler


def preflight_sql_handler(plugin: Any, handler: Any, schema: Dict[str, Any]) -> Any:
    async def _handler(args: Dict[str, Any]) -> Any:
        args = args or {}
        plan_id = str(args.get("plan_id") or "").strip()
        if plan_id:
            run_state = get_db_run_state(plugin)
            stored_plan = run_state.get_plan(plan_id) if run_state is not None else None
            if stored_plan is not None and _stored_plan_is_executable(stored_plan):
                return await execute_plan_id(plugin, handler, schema, args)
            if not args.get("sql"):
                return await execute_plan_id(plugin, handler, schema, args)
            args = dict(args)
            args.pop("plan_id", None)

        sql = _normalize_sql_for_plugin(plugin, str((args or {}).get("sql") or ""))
        analysis = _validate_plugin_query_policy(plugin, sql)
        dialect = schema_or_plugin_dialect(plugin, schema)
        validation = await trace_sql_validation(plugin, schema, sql, analysis)
        run_state = get_db_run_state(plugin)
        apply_required_field_validation(
            validation, sql, run_state, dialect=dialect, analysis=analysis
        )
        if validation.get("error"):
            record_sql_preflight_failure(plugin, run_state, validation)
            return validation
        fingerprint = sql_fingerprint(sql)
        if run_state is not None:
            run_state.record_validated_sql(
                fingerprint, validation, source_tool="db_query"
            )
        normalized_args = {**(args or {}), "sql": sql}
        result = await trace_sql_execution(plugin, handler, normalized_args)
        if run_state is not None:
            run_state.record_executed_query(
                {
                    "sql_fingerprint": fingerprint,
                    "sql": sql,
                    "columns": result_columns(result),
                    "selected_columns": validation.get("selected_columns") or [],
                    "row_count": result_row_count(result),
                    "truncated": (
                        bool(result.get("truncated"))
                        if isinstance(result, dict)
                        else False
                    ),
                },
                source_tool="db_query",
            )
        return result

    return _handler


def _stored_plan_is_executable(stored: Dict[str, Any]) -> bool:
    result = stored.get("result") or {}
    validation = result.get("validation") or {}
    return bool(str(result.get("compiled_sql") or "").strip() and validation.get("ok"))


async def execute_plan_id(
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
    analysis = _validate_plugin_query_policy(plugin, sql)
    fingerprint = sql_fingerprint(sql)
    validation = await trace_sql_validation(plugin, schema, sql, analysis)
    apply_required_field_validation(
        validation,
        sql,
        run_state,
        dialect=schema_or_plugin_dialect(plugin, schema),
        analysis=analysis,
    )
    if validation.get("error") or not validation.get("ok"):
        record_sql_preflight_failure(
            plugin,
            run_state,
            validation,
            source_tool="db_query",
        )
        return {
            **validation,
            "plan_id": plan_id,
            "repair_required": True,
            "suggested_next_tool": "db_plan_query",
        }
    run_state.record_validated_sql(
        fingerprint,
        {
            **validation,
            "ok": True,
            "plan_id": plan_id,
            "source": "validated_query_ir",
            "sql_fingerprint": fingerprint,
        },
        source_tool="db_query",
    )
    normalized_args = {**args, "sql": sql}
    normalized_args.pop("plan_id", None)
    result = await trace_sql_execution(plugin, handler, normalized_args)
    run_state.record_executed_query(
        {
            "plan_id": plan_id,
            "sql_fingerprint": fingerprint,
            "sql": sql,
            "columns": result_columns(result),
            "selected_columns": validation.get("selected_columns") or [],
            "row_count": result_row_count(result),
            "truncated": (
                bool(result.get("truncated")) if isinstance(result, dict) else False
            ),
        },
        source_tool="db_query",
    )
    return result


async def trace_sql_validation(
    plugin: Any,
    schema: Dict[str, Any],
    sql: str,
    analysis: SqlAnalysis | None,
) -> Dict[str, Any]:
    from ...runtime.tracing import db_trace_span

    dialect = schema_or_plugin_dialect(plugin, schema)
    async with db_trace_span(
        plugin,
        "from_db.validate_sql",
        trace_type=TraceType.TOOL_EXECUTION,
    ) as (trace_manager, span_id):
        validation = validate_sql_against_schema(
            sql, schema, dialect=dialect, analysis=analysis
        )
        trace_manager.record_output(
            span_id,
            {
                "ok": bool(validation.get("ok")),
                "error_type": validation.get("error_type"),
                "sql_fingerprint": validation.get("sql_fingerprint"),
            },
        )
        return validation


async def trace_sql_execution(plugin: Any, handler: Any, args: Dict[str, Any]) -> Any:
    from ...runtime.tracing import db_trace_span

    async with db_trace_span(
        plugin,
        "from_db.execute_sql",
        trace_type=TraceType.TOOL_EXECUTION,
    ) as (trace_manager, span_id):
        result = await handler(args)
        trace_manager.record_output(
            span_id,
            {
                "row_count": result_row_count(result),
                "truncated": (
                    bool(result.get("truncated")) if isinstance(result, dict) else False
                ),
            },
        )
        return result


def schema_or_plugin_dialect(plugin: Any, schema: Dict[str, Any]) -> str:
    return str(schema.get("database_type") or getattr(plugin, "sql_dialect", ""))


def record_sql_preflight_failure(
    plugin: Any,
    run_state: Any,
    validation: Dict[str, Any],
    *,
    source_tool: str = "db_query",
) -> None:
    fingerprint = validation["sql_fingerprint"]
    if run_state is not None:
        attempt_count = run_state.record_failed_sql(
            fingerprint,
            validation,
            source_tool=source_tool,
        )
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


def result_row_count(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    if isinstance(result.get("total_rows"), int):
        return result["total_rows"]
    rows = result.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def result_columns(result: Any) -> List[str]:
    if not isinstance(result, dict):
        return []
    columns = result.get("columns")
    if isinstance(columns, list):
        return [str(column) for column in columns if column]
    rows = result.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return [str(column) for column in rows[0].keys()]
    return []


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


def _plan_trace_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_id": result.get("plan_id"),
        "route": result.get("route"),
        "resolved_table_count": len(result.get("resolved_tables") or []),
        "table_candidate_count": len(result.get("table_candidates") or []),
        "field_candidate_count": len(result.get("field_candidates") or {}),
        "join_path_request_count": len(result.get("join_paths") or []),
        "compiled": bool(result.get("compiled_sql")),
        "validation_ok": bool((result.get("validation") or {}).get("ok")),
        "knowledge_used": bool(result.get("knowledge_used")),
        "payload_tokens_estimate": max(1, (len(str(result)) + 3) // 4),
    }


def _include_plan_diagnostics(args: Any) -> bool:
    if not isinstance(args, dict):
        return False
    return bool(args.get("include_diagnostics") or args.get("debug"))


def _compile_and_query_repair_payload(plan_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": False,
        "plan_id": plan_result.get("plan_id"),
        "route": plan_result.get("route"),
        "resolved_tables": plan_result.get("resolved_tables", []),
        "ambiguous_tables": plan_result.get("ambiguous_tables", []),
        "unknown_tables": plan_result.get("unknown_tables", []),
        "validation": plan_result.get("validation"),
        "plan_warnings": plan_result.get("plan_warnings", []),
        "repair_required": True,
        "suggested_next_tool": "db_plan_query",
        "next_step": plan_result.get("next_step"),
        "message": "Query intent could not be compiled into validated SQL.",
    }


def _compile_and_query_result(
    plan_result: Dict[str, Any],
    validation: Dict[str, Any],
    sql: str,
    query_result: Any,
    intent_args: Dict[str, Any],
) -> Dict[str, Any]:
    if isinstance(query_result, dict):
        payload = dict(query_result)
    else:
        payload = {"result": query_result}
    payload.update(
        {
            "ok": True,
            "plan_id": plan_result.get("plan_id"),
            "route": plan_result.get("route"),
            "sql": sql,
            "resolved_tables": plan_result.get("resolved_tables", []),
            "best_join_path": plan_result.get("best_join_path"),
            "validation": {
                "ok": bool(validation.get("ok")),
                "sql_fingerprint": validation.get("sql_fingerprint"),
            },
            "assumptions": string_list(intent_args.get("assumptions")),
        }
    )
    return payload


def _compile_and_query_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": bool(result.get("ok")),
        "plan_id": result.get("plan_id"),
        "route": result.get("route"),
        "row_count": result_row_count(result),
        "repair_required": bool(result.get("repair_required")),
        "suggested_next_tool": result.get("suggested_next_tool"),
    }
