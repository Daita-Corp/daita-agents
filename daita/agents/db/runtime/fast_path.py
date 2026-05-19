"""Deterministic fast path for simple ``from_db`` queries."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..query.evidence import collect_query_evidence, evidence_table_names
from ..query.catalog_adapter import (
    catalog_schema_snapshot,
    primary_key_or_identity,
)
from ..query.metadata import (
    short_table_name,
    split_identifier,
    table_name,
)
from ..query.intent import looks_like_count_intent
from .tracing import db_trace_span

WRITE_TERMS = ("insert", "update", "delete", "upsert", "write", "mutate")
SCHEMA_TERMS = ("schema", "column", "columns", "relationship", "relationships")


async def try_db_fast_path(
    agent: Any, prompt: str, kwargs: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Try a deterministic DB answer before the autonomous LLM loop."""

    async with db_trace_span(
        agent,
        "from_db.fast_path",
        prompt=prompt[:200],
    ) as (trace_manager, span_id):
        intent = await _simple_count_intent(agent, prompt)
        if intent is None:
            trace_manager.record_output(span_id, {"used": False})
            return None

        tool = agent.tool_registry.get("db_compile_and_query")
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
                "tool_duration_ms": duration_ms,
            },
        )
        if not result_ok:
            return None

        tool_record = {
            "tool": "db_compile_and_query",
            "arguments": intent,
            "result": result,
            "duration_ms": duration_ms,
        }
        agent._db_last_context_metadata = {
            "runtime_context_chars": 0,
            "runtime_context_tokens_estimate": 0,
            "selected_tools": ["db_compile_and_query"],
            "selected_tool_count": 1,
            "prompt_terms": split_identifier(prompt),
            "fast_path": True,
        }
        if hasattr(agent, "_tool_call_history"):
            agent._tool_call_history.append(
                {
                    "name": "db_compile_and_query",
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
        }


async def _simple_count_intent(agent: Any, prompt: str) -> Optional[Dict[str, Any]]:
    text = str(prompt or "").lower()
    if not looks_like_count_intent(text):
        return None
    if any(term in text for term in WRITE_TERMS + SCHEMA_TERMS):
        return None

    schema = catalog_schema_snapshot(agent)
    table = await _evidence_table_match(agent, prompt, schema)
    if table is None:
        return None
    table_ref = table_name(table)
    pk = primary_key_or_identity(table)
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
    agent: Any, prompt: str, schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    evidence = await collect_query_evidence(
        prompt,
        {"goal": prompt},
        schema,
        catalog=vars(agent).get("_db_catalog"),
        store_id=vars(agent).get("_db_catalog_store_id"),
        graph_backend=vars(agent).get("_db_query_graph_backend"),
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
