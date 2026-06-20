"""Live OpenAI trace presentation test for Agent.from_db().

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/from_db/test_from_db_tracing_telemetry_live.py \
        -m "requires_llm and integration" -v -s
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.core.tracing import TraceType, get_trace_manager
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus

load_dotenv(Path.cwd() / ".env")

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


def _require_live_openai() -> str:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db trace tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")
    return os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini")


async def _seed_trace_demo_db(path) -> None:
    sqlite = SQLitePlugin(path=str(path))
    await sqlite.execute_script(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            tier TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total REAL NOT NULL
        );
        INSERT INTO customers (name, tier)
        VALUES ('Ada', 'enterprise'), ('Linus', 'startup');
        INSERT INTO orders (customer_id, total)
        VALUES (1, 120.50), (2, 42.00);
        """
    )
    await sqlite.disconnect()


def _reset_trace_exporter():
    trace_manager = get_trace_manager()
    trace_manager.flush(timeout_millis=2000)
    return trace_manager


def _present_trace_summary(trace_manager, operation_id: str) -> dict[str, object]:
    trace_manager.flush(timeout_millis=5000)
    spans = []
    for span in trace_manager.get_recent_operations(limit=200):
        metadata = dict(span.get("metadata") or {})
        if metadata.get("operation_id") != operation_id:
            continue
        spans.append(
            {
                "name": span["operation"],
                "trace_type": span["type"],
                "trace_id": span["trace_id"],
                "span_id": span["span_id"],
                "parent_span_id": span["parent_span_id"],
                "attributes": {
                    key: metadata.get(key)
                    for key in (
                        "runtime_id",
                        "runtime_kind",
                        "operation_id",
                        "execution_id",
                        "operation_type",
                        "intent_kind",
                        "task_id",
                        "capability_id",
                        "executor_id",
                        "plugin_id",
                    )
                    if metadata.get(key) is not None
                },
            }
        )
    return {
        "operation_id": operation_id,
        "span_count": len(spans),
        "root_spans": [
            span
            for span in spans
            if span["trace_type"] == TraceType.AGENT_EXECUTION.value
        ],
        "child_spans": [
            span
            for span in spans
            if span["trace_type"] == TraceType.TOOL_EXECUTION.value
        ],
    }


async def test_live_openai_from_db_trace_presentation(tmp_path):
    model = _require_live_openai()
    trace_manager = _reset_trace_exporter()
    db_path = tmp_path / "live-trace-demo.sqlite"
    await _seed_trace_demo_db(db_path)

    agent = await Agent.from_db(
        str(db_path),
        name="LiveTraceTelemetryAgent",
        cache_ttl=0,
        llm_provider="openai",
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )

    try:
        result = await agent.run_detailed(
            "What tables exist?",
            mode="schema.query",
            session_id="live-trace-demo-session",
            user_id="live-trace-demo-user",
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    assert result.diagnostics["trace"] == snapshot.operation.metadata["trace"]
    assert result.telemetry["mode"] == "llm"
    assert result.telemetry["provider"] == "openai"

    summary = _present_trace_summary(trace_manager, result.operation_id)
    assert summary["root_spans"]
    assert summary["child_spans"]

    print("\n=== DAITA FROM_DB LIVE TRACE SUMMARY ===")
    print(
        json.dumps(
            {
                "answer": result.answer,
                "trace": result.diagnostics["trace"],
                "telemetry": result.telemetry,
                "runtime_events": [
                    {
                        "type": event.type.value,
                        "message": event.message,
                        "trace_id": event.trace_id,
                        "span_id": event.span_id,
                        "task_id": event.task_id,
                        "capability_id": event.capability_id,
                        "executor_id": event.executor_id,
                    }
                    for event in snapshot.events
                ],
                "spans": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
