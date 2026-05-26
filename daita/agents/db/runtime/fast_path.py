"""Deterministic fast path for simple ``from_db`` queries."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..config.route_decision import DbRouteDecision, build_db_route_decision
from ..query.evidence import collect_query_evidence, evidence_table_names
from ..query.catalog_adapter import catalog_schema_snapshot
from ..query.metadata import (
    identity_column,
    short_table_name,
    split_identifier,
    table_name,
)
from .tracing import db_trace_span


async def try_db_fast_path(
    agent: Any,
    prompt: str,
    kwargs: Dict[str, Any],
    *,
    run_state: Any = None,
    route_decision: DbRouteDecision | None = None,
) -> Optional[Dict[str, Any]]:
    """Try a deterministic DB answer before the autonomous LLM loop."""

    if route_decision is None:
        route_decision = build_db_route_decision(agent, prompt)

    async with db_trace_span(
        agent,
        "from_db.fast_path",
        prompt=prompt[:200],
    ) as (trace_manager, span_id):
        fast_path = route_decision.fast_path
        if not fast_path.eligible:
            trace_manager.record_output(
                span_id, {"used": False, "reason": fast_path.reason}
            )
            return None

        intent = await _simple_count_intent(
            agent,
            prompt,
            run_state=run_state,
        )
        if intent is None:
            trace_manager.record_output(
                span_id, {"used": False, "reason": "table_match_unavailable"}
            )
            return None

        tool_name = fast_path.tool_name
        if tool_name is None:
            tool = None
        else:
            tool = agent.tool_registry.get(tool_name)
        if tool is None:
            trace_manager.record_output(
                span_id, {"used": False, "reason": "tool_unavailable"}
            )
            return None

        start = time.time()
        result = await tool.handler(intent)
        duration_ms = int((time.time() - start) * 1000)
        result_ok = isinstance(result, dict) and bool(result.get("ok"))
        trace_manager.record_output(
            span_id,
            {
                "used": result_ok,
                "reason": None if result_ok else "compile_or_query_failed",
                "strategy": fast_path.strategy,
                "tool_duration_ms": duration_ms,
            },
        )
        if not result_ok:
            return None

        tool_record = {
            "tool": tool_name,
            "arguments": intent,
            "result": result,
            "duration_ms": duration_ms,
        }
        agent._db_last_context_metadata = {
            "runtime_context_chars": 0,
            "runtime_context_tokens_estimate": 0,
            "selected_tools": [tool_name],
            "selected_tool_count": 1,
            "prompt_terms": split_identifier(prompt),
            "fast_path": True,
        }
        if hasattr(agent, "_tool_call_history"):
            agent._tool_call_history.append(
                {
                    "name": tool_name,
                    "duration_ms": duration_ms,
                    "input": intent,
                    "output": result,
                }
            )
        return {
            "result": _count_answer(result, intent),
            "tool_calls": [tool_record],
            "iterations": 0,
            "tokens": {},
            "cost": 0.0,
            "processing_time_ms": duration_ms,
            "agent_id": getattr(agent, "agent_id", None),
            "agent_name": getattr(agent, "name", None),
            "from_db_fast_path": {
                "used": True,
                "kind": "simple_count",
                "fallback_reason": None,
            },
            "diagnostics": {
                "db_run_state": (
                    run_state.summary() if hasattr(run_state, "summary") else {}
                )
            },
        }


async def _simple_count_intent(
    agent: Any,
    prompt: str,
    *,
    run_state: Any = None,
) -> Optional[Dict[str, Any]]:
    schema = catalog_schema_snapshot(agent)
    table = await _evidence_table_match(agent, prompt, schema, run_state=run_state)
    if table is None:
        return None
    table_ref = table_name(table)
    pk = identity_column(table, mode="count_stable_row")
    alias = f"total_{short_table_name(table_ref)}"
    return {
        "goal": prompt,
        "required_fields": [alias],
        "candidate_tables": [table_ref],
        "aggregations": [
            f"COUNT({table_ref}.{pk}) AS {alias}" if pk else f"COUNT(*) AS {alias}"
        ],
    }


async def _evidence_table_match(
    agent: Any, prompt: str, schema: Dict[str, Any], *, run_state: Any = None
) -> Optional[Dict[str, Any]]:
    evidence = await collect_query_evidence(
        prompt,
        {"goal": prompt},
        schema,
        run_state=run_state,
        catalog=vars(agent).get("_db_catalog"),
        store_id=vars(agent).get("_db_catalog_store_id"),
    )
    table_names = evidence_table_names(evidence)
    if len(table_names) != 1:
        return None
    wanted = table_names[0].lower()
    matches = [
        table
        for table in schema.get("tables", []) or []
        if table_name(table).lower() == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def _count_answer(result: Dict[str, Any], intent: Dict[str, Any]) -> str:
    alias = (intent.get("required_fields") or ["count"])[0]
    rows = result.get("rows") or []
    if rows and isinstance(rows[0], dict) and alias in rows[0]:
        return f"{alias}: {rows[0][alias]}"
    return str(result)
