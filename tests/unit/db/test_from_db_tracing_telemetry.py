from __future__ import annotations

from daita.agents.agent import Agent
from daita.core.tracing import TraceType, get_trace_manager
from daita.db import DbRuntime
from daita.db.llm_service import DbLLMResponse
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
    normalize_db_telemetry_diagnostics,
)
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import (
    AccessMode,
    Evidence,
    HostRuntimeContext,
    OperationStatus,
    host_runtime_context,
)


async def _seed_sqlite(path):
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        INSERT INTO customers (name) VALUES ('Ada'), ('Linus');
        """)
    await plugin.disconnect()


class FakeTraceLLMService:
    safe_metadata = {"provider": "fake", "model": "trace-llm-test"}

    @property
    def available(self):
        return True

    async def generate_json(self, messages):
        return await self.generate_synthesis_json(messages)

    async def generate_synthesis_json(self, messages):
        import json

        context = json.loads(messages[-1]["content"])["context"]
        evidence_by_kind = {item["kind"]: item for item in context["evidence"]}
        citations = [
            {
                "id": evidence_by_kind[kind]["id"],
                "kind": kind,
                "purpose": f"cite {kind}",
            }
            for kind in ("schema.asset_profile", "verification.result")
            if kind in evidence_by_kind
        ]
        return DbLLMResponse(
            content=json.dumps(
                {
                    "answer": "The database contains a customers table.",
                    "reasoning_summary": "Used accepted schema evidence.",
                    "cited_evidence_refs": citations,
                    "assumptions": [],
                    "limitations": [],
                    "warnings": [],
                    "follow_up_questions": [],
                    "sufficiency": "answered",
                    "confidence": 0.91,
                    "truncation": context["truncation"],
                    "grounding": {"all_claims_from_evidence": True},
                }
            ),
            diagnostics={
                "provider": "fake",
                "model": "trace-llm-test",
                "tokens": {
                    "prompt_tokens": "12.0",
                    "completion_tokens": "8",
                },
                "estimated_cost_usd": 0.0,
                "latency_ms": "2.5",
            },
        )


class FakeTraceMonitorPlanner:
    def __init__(self):
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "monitor.list"},
            actions=(
                DbPlannerAction(
                    action_id="list_monitors",
                    kind=DbPlannerActionKind.READ_MONITOR_STATE,
                    input={"read_kind": "list"},
                ),
            ),
        )


def _reset_traces():
    trace_manager = get_trace_manager()
    trace_manager.flush(timeout_millis=2000)
    trace_manager._memory_exporter.clear()
    return trace_manager


def _finished_span_dicts(trace_manager):
    trace_manager.flush(timeout_millis=2000)
    return [
        {
            "name": span.name,
            "trace_id": format(span.context.trace_id, "032x"),
            "span_id": format(span.context.span_id, "016x"),
            "parent_span_id": (
                None if span.parent is None else format(span.parent.span_id, "016x")
            ),
            "attributes": dict(span.attributes or {}),
        }
        for span in trace_manager._memory_exporter.get_finished_spans()
    ]


async def test_prompt_monitor_loop_result_has_trace_correlation(tmp_path):
    trace_manager = _reset_traces()
    db_path = tmp_path / "monitor_trace.sqlite"
    await _seed_sqlite(db_path)
    planner = FakeTraceMonitorPlanner()
    host_context = HostRuntimeContext(services={"db_agent_planner": planner})
    with host_runtime_context(host_context):
        agent = await Agent.from_db(str(db_path), name="MonitorTraceTest", cache_ttl=0)

    try:
        result = await agent.run_detailed("List monitors", session_id="monitor-session")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    assert len(planner.states) == 1
    trace = result.diagnostics["trace"]
    assert trace == {
        key: value
        for key, value in snapshot.operation.metadata["trace"].items()
        if key in {"trace_id", "root_span_id"}
    }
    assert snapshot.operation.operation_type == "db.run"
    assert any(item.kind == "monitor.listing" for item in snapshot.evidence)
    assert any(item.kind == "answer.synthesis" for item in snapshot.evidence)
    assert all(event.trace_id == trace["trace_id"] for event in snapshot.events)

    spans = _finished_span_dicts(trace_manager)
    root = next(
        span
        for span in spans
        if span["attributes"].get("daita.operation.id") == result.operation_id
        and span["attributes"].get("daita.trace.type")
        == TraceType.AGENT_EXECUTION.value
    )
    assert root["attributes"]["daita.operation.type"] == "db.run"


async def test_blocked_operation_keeps_trace_correlation(tmp_path):
    _reset_traces()
    db_path = tmp_path / "blocked_trace.sqlite"
    await _seed_sqlite(db_path)
    agent = await Agent.from_db(str(db_path), name="BlockedTraceTest", cache_ttl=0)

    try:
        result = await agent.run_detailed(
            "Hello",
            requested_capabilities=("quality.profile",),
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert result.status is OperationStatus.BLOCKED
    assert snapshot is not None
    assert result.diagnostics["trace"] == {
        key: value
        for key, value in snapshot.operation.metadata["trace"].items()
        if key in {"trace_id", "root_span_id"}
    }
    assert all(
        event.trace_id == result.diagnostics["trace"]["trace_id"]
        for event in snapshot.events
    )


def test_llm_telemetry_normalization_from_synthesis_evidence():
    result = DbOperationResult(
        operation_id="db-op-llm",
        request=DbRequest("Summarize customers"),
        intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        contract=DbOperationContract(
            operation_type="db.query",
            access=AccessMode.READ,
        ),
        status=OperationStatus.SUCCEEDED,
        evidence=(
            Evidence(
                kind="answer.synthesis",
                owner="db_runtime",
                operation_id="db-op-llm",
                accepted=True,
                payload={
                    "answer": "Two customers.",
                    "diagnostics": {
                        "mode": "llm",
                        "provider": "openai",
                        "model": "gpt-test",
                        "latency_ms": 12.5,
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "estimated_cost_usd": 0.001,
                    },
                },
            ),
        ),
    )

    assert result.telemetry == {
        "provider": "openai",
        "model": "gpt-test",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "llm_calls": 1,
        "estimated_cost_usd": 0.001,
        "latency_ms": 12.5,
        "mode": "llm",
    }


def test_analysis_telemetry_prefers_final_synthesis_over_partial():
    base = {
        "answer": "partial",
        "diagnostics": {
            "mode": "llm",
            "provider": "partial-provider",
            "model": "partial-model",
            "input_tokens": 1,
            "output_tokens": 2,
        },
    }
    final = {
        "answer": "final",
        "partial": False,
        "diagnostics": {
            "mode": "llm",
            "provider": "final-provider",
            "model": "final-model",
            "input_tokens": 10,
            "output_tokens": 5,
            "estimated_cost_usd": 0.0,
        },
    }
    result = DbOperationResult(
        operation_id="analysis-op",
        request=DbRequest("Analyze customers"),
        intent=DbIntent(kind=DbIntentKind.ANOMALY_INVESTIGATE, access=AccessMode.READ),
        contract=DbOperationContract(
            operation_type="analysis.run",
            access=AccessMode.READ,
        ),
        status=OperationStatus.SUCCEEDED,
        evidence=(
            Evidence(
                kind="analysis.synthesis",
                owner="db_runtime",
                operation_id="analysis-op",
                accepted=True,
                payload={**base, "partial": True},
            ),
            Evidence(
                kind="analysis.synthesis",
                owner="db_runtime",
                operation_id="analysis-op",
                accepted=True,
                payload=final,
            ),
        ),
    )

    assert result.telemetry["provider"] == "final-provider"
    assert result.telemetry["model"] == "final-model"
    assert result.telemetry["total_tokens"] == 15
    assert result.telemetry["estimated_cost_usd"] == 0.0


def test_telemetry_numeric_normalization_preserves_zero_and_unknown_values():
    assert normalize_db_telemetry_diagnostics(
        {
            "mode": "llm",
            "provider": "fake",
            "model": "m",
            "tokens": {
                "prompt_tokens": "12.0",
                "completion_tokens": "8",
            },
            "estimated_cost_usd": 0.0,
            "latency_ms": "2.5",
        }
    ) == {
        "provider": "fake",
        "model": "m",
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
        "llm_calls": 1,
        "estimated_cost_usd": 0.0,
        "latency_ms": 2.5,
        "mode": "llm",
    }


def test_telemetry_optional_token_mapping_retains_unknown_result_shape():
    assert normalize_db_telemetry_diagnostics(
        {
            "mode": "unknown",
            "tokens": None,
        }
    ) == {
        "provider": "unknown",
        "model": "unknown",
        "input_tokens": "unknown",
        "output_tokens": "unknown",
        "total_tokens": "unknown",
        "llm_calls": "unknown",
        "estimated_cost_usd": "unknown",
        "latency_ms": None,
        "mode": "unknown",
    }
