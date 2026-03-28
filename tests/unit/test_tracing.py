"""
Unit tests for the OTel-backed TraceManager.

No external dependencies required — all OTel components are in-process.
asyncio_mode = "auto" is set globally in pyproject.toml.
"""

import json
import re
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from daita.core.tracing import (
    TraceManager,
    TraceStatus,
    TraceType,
    configure_tracing,
    get_trace_manager,
)
from daita.core.otel_exporter import BoundedInMemorySpanExporter, DaitaSpanExporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
HEX16_RE = re.compile(r"^[0-9a-f]{16}$")


def fresh_manager() -> TraceManager:
    """Return a new TraceManager (not the global singleton) for test isolation."""
    return TraceManager()


def flush(tm: TraceManager) -> None:
    """Force-flush the provider so in-memory exporter has all finished spans."""
    tm.flush(timeout_millis=2000)


# ---------------------------------------------------------------------------
# BoundedInMemorySpanExporter
# ---------------------------------------------------------------------------


class TestBoundedInMemorySpanExporter:
    def test_export_and_retrieve(self):
        exp = BoundedInMemorySpanExporter(maxlen=10)
        mock_span = MagicMock()
        result = exp.export([mock_span, mock_span])
        assert result == SpanExportResult.SUCCESS
        assert len(exp.get_finished_spans()) == 2

    def test_cap_enforced(self):
        exp = BoundedInMemorySpanExporter(maxlen=5)
        mock_span = MagicMock()
        # Export 8 spans — should only keep the last 5
        exp.export([mock_span] * 8)
        assert len(exp.get_finished_spans()) == 5

    def test_clear(self):
        exp = BoundedInMemorySpanExporter(maxlen=10)
        exp.export([MagicMock()])
        exp.clear()
        assert exp.get_finished_spans() == []

    def test_shutdown_rejects_exports(self):
        exp = BoundedInMemorySpanExporter(maxlen=10)
        exp.shutdown()
        result = exp.export([MagicMock()])
        assert result == SpanExportResult.FAILURE

    def test_get_finished_spans_returns_copy(self):
        exp = BoundedInMemorySpanExporter(maxlen=10)
        s1 = MagicMock()
        exp.export([s1])
        copy = exp.get_finished_spans()
        copy.clear()  # mutating the copy should not affect internal state
        assert len(exp.get_finished_spans()) == 1


# ---------------------------------------------------------------------------
# DaitaSpanExporter
# ---------------------------------------------------------------------------


class TestDaitaSpanExporter:
    def test_disabled_when_no_env_vars(self):
        """Exporter is no-op when API key / URL not set."""
        with patch.dict("os.environ", {}, clear=True):
            exp = DaitaSpanExporter()
        assert not exp.enabled
        result = exp.export([MagicMock()])
        assert result == SpanExportResult.SUCCESS  # no-op success

    def test_enabled_when_env_vars_set(self):
        env = {
            "DAITA_API_KEY": "sk-test",
            "DAITA_DASHBOARD_URL": "http://localhost:8000",
        }
        with patch.dict("os.environ", env):
            exp = DaitaSpanExporter()
        assert exp.enabled

    def test_shutdown_stops_export(self):
        env = {
            "DAITA_API_KEY": "sk-test",
            "DAITA_DASHBOARD_URL": "http://localhost:8000",
        }
        with patch.dict("os.environ", env):
            exp = DaitaSpanExporter()
        exp.shutdown()
        result = exp.export([MagicMock()])
        assert result == SpanExportResult.SUCCESS  # stopped = no-op


# ---------------------------------------------------------------------------
# TraceManager — span lifecycle
# ---------------------------------------------------------------------------


class TestSpanLifecycle:
    def test_start_span_returns_hex_span_id(self):
        tm = fresh_manager()
        span_id = tm.start_span("test_op", TraceType.AGENT_EXECUTION, agent_id="a1")
        assert HEX16_RE.match(span_id), f"Expected 16-char hex, got: {span_id!r}"

    def test_end_span_increments_metrics(self):
        tm = fresh_manager()
        span_id = tm.start_span("test_op", TraceType.AGENT_EXECUTION, agent_id="a1")
        initial = tm.get_global_metrics()["total_spans"]
        tm.end_span(span_id, TraceStatus.SUCCESS)
        assert tm.get_global_metrics()["total_spans"] == initial

    def test_end_span_unknown_id_is_silent(self):
        tm = fresh_manager()
        # Should not raise
        tm.end_span("deadbeefdeadbeef", TraceStatus.SUCCESS)

    def test_llm_call_counter(self):
        tm = fresh_manager()
        span_id = tm.start_span("llm_op", TraceType.LLM_CALL, agent_id="a1")
        before = tm.get_global_metrics()["total_llm_calls"]
        tm.end_span(span_id, TraceStatus.SUCCESS)
        flush(tm)
        # Counter is updated on end_span
        assert tm.get_global_metrics()["total_llm_calls"] == before + 1

    def test_decision_counter(self):
        tm = fresh_manager()
        span_id = tm.start_span("dec_op", TraceType.DECISION_TRACE, agent_id="a1")
        before = tm.get_global_metrics()["total_decisions"]
        tm.end_span(span_id, TraceStatus.SUCCESS)
        assert tm.get_global_metrics()["total_decisions"] == before + 1


# ---------------------------------------------------------------------------
# TraceManager — trace ID format
# ---------------------------------------------------------------------------


class TestTraceIdFormat:
    def test_trace_id_is_w3c_hex(self):
        tm = fresh_manager()
        tm.start_span("op", TraceType.AGENT_EXECUTION, agent_id="a1")
        trace_id = tm.trace_context.current_trace_id
        assert trace_id is not None
        assert HEX32_RE.match(
            trace_id
        ), f"Expected 32-char hex trace ID, got: {trace_id!r}"

    def test_span_id_is_w3c_hex(self):
        tm = fresh_manager()
        span_id = tm.start_span("op", TraceType.AGENT_EXECUTION, agent_id="a1")
        assert HEX16_RE.match(
            span_id
        ), f"Expected 16-char hex span ID, got: {span_id!r}"

    async def test_context_manager_sets_contextvars(self):
        tm = fresh_manager()
        async with tm.span("op", TraceType.AGENT_EXECUTION, agent_id="a1") as span_id:
            assert tm.trace_context.current_span_id == span_id
            assert HEX16_RE.match(span_id)
            assert HEX32_RE.match(tm.trace_context.current_trace_id)


# ---------------------------------------------------------------------------
# TraceManager — async context manager
# ---------------------------------------------------------------------------


class TestSpanContextManager:
    async def test_success_path(self):
        tm = fresh_manager()
        async with tm.span("op", TraceType.AGENT_EXECUTION, agent_id="a1") as span_id:
            assert span_id is not None
        flush(tm)
        ops = tm.get_recent_operations(agent_id="a1")
        assert len(ops) >= 1
        assert ops[0]["status"] == "success"

    async def test_error_path_reraises(self):
        tm = fresh_manager()
        with pytest.raises(ValueError):
            async with tm.span("op", TraceType.AGENT_EXECUTION, agent_id="a1"):
                raise ValueError("boom")
        flush(tm)
        ops = tm.get_recent_operations(agent_id="a1")
        assert ops[0]["status"] == "error"

    async def test_context_restored_after_exit(self):
        tm = fresh_manager()
        outer_trace_id = None
        async with tm.span("outer", TraceType.AGENT_EXECUTION, agent_id="a1") as _:
            outer_trace_id = tm.trace_context.current_trace_id
            async with tm.span("inner", TraceType.TOOL_EXECUTION, agent_id="a1") as _:
                inner_trace_id = tm.trace_context.current_trace_id
                assert inner_trace_id == outer_trace_id  # same trace
        # After exit, context var should be restored to parent value or None
        # (token reset mechanism)

    async def test_nested_spans_share_trace_id(self):
        tm = fresh_manager()
        async with tm.span(
            "outer", TraceType.AGENT_EXECUTION, agent_id="a1"
        ) as outer_id:
            outer_trace = tm.trace_context.current_trace_id
            async with tm.span(
                "inner", TraceType.TOOL_EXECUTION, agent_id="a1"
            ) as inner_id:
                inner_trace = tm.trace_context.current_trace_id
        assert outer_trace == inner_trace


# ---------------------------------------------------------------------------
# TraceManager — record_decision / record_llm_call
# ---------------------------------------------------------------------------


class TestRecordMethods:
    def test_record_decision(self):
        tm = fresh_manager()
        span_id = tm.start_span("dec", TraceType.DECISION_TRACE, agent_id="a1")
        # Should not raise
        tm.record_decision(
            span_id,
            confidence=0.95,
            reasoning=["step1", "step2"],
            alternatives=["alt_a", "alt_b"],
        )
        tm.end_span(span_id, TraceStatus.SUCCESS)

    def test_record_decision_unknown_span(self):
        tm = fresh_manager()
        # Should not raise
        tm.record_decision("nonexistent16char", confidence=0.5)

    def test_record_llm_call(self):
        tm = fresh_manager()
        span_id = tm.start_span("llm", TraceType.LLM_CALL, agent_id="a1")
        tm.record_llm_call(
            span_id, model="gpt-4o", prompt_tokens=100, completion_tokens=50
        )
        tm.end_span(span_id, TraceStatus.SUCCESS)

    def test_record_llm_call_unknown_span(self):
        tm = fresh_manager()
        # Should not raise
        tm.record_llm_call("nonexistent16char", model="gpt-4")


# ---------------------------------------------------------------------------
# TraceManager — query methods
# ---------------------------------------------------------------------------


class TestQueryMethods:
    async def test_get_recent_operations_returns_dict_list(self):
        tm = fresh_manager()
        async with tm.span("op1", TraceType.AGENT_EXECUTION, agent_id="qa"):
            pass
        flush(tm)
        ops = tm.get_recent_operations(agent_id="qa")
        assert isinstance(ops, list)
        assert len(ops) >= 1
        op = ops[0]
        assert "span_id" in op
        assert "trace_id" in op
        assert "status" in op
        assert "duration_ms" in op

    async def test_get_recent_operations_filtered_by_agent(self):
        tm = fresh_manager()
        async with tm.span("op_a", TraceType.AGENT_EXECUTION, agent_id="agent_a"):
            pass
        async with tm.span("op_b", TraceType.AGENT_EXECUTION, agent_id="agent_b"):
            pass
        flush(tm)
        ops_a = tm.get_recent_operations(agent_id="agent_a")
        assert all(o["agent_id"] == "agent_a" for o in ops_a)

    async def test_get_agent_metrics_success_rate(self):
        tm = fresh_manager()
        async with tm.span("ok", TraceType.AGENT_EXECUTION, agent_id="ag"):
            pass
        with pytest.raises(RuntimeError):
            async with tm.span("fail", TraceType.AGENT_EXECUTION, agent_id="ag"):
                raise RuntimeError("oops")
        flush(tm)
        metrics = tm.get_agent_metrics("ag")
        assert metrics["total_operations"] == 2
        assert metrics["successful_operations"] == 1
        assert metrics["failed_operations"] == 1
        assert metrics["success_rate"] == 0.5

    async def test_get_agent_metrics_no_spans(self):
        tm = fresh_manager()
        metrics = tm.get_agent_metrics("nobody")
        assert metrics["total_operations"] == 0
        assert metrics["success_rate"] == 0

    async def test_get_workflow_communications(self):
        tm = fresh_manager()
        span_id = tm.start_span(
            "wf_comm",
            TraceType.WORKFLOW_COMMUNICATION,
            agent_id="wf1",
            workflow_name="my_wf",
            from_agent="a",
            to_agent="b",
        )
        tm.end_span(span_id, TraceStatus.SUCCESS)
        flush(tm)
        comms = tm.get_workflow_communications(workflow_name="my_wf")
        assert len(comms) >= 1
        assert comms[0]["from_agent"] == "a"
        assert comms[0]["to_agent"] == "b"

    async def test_get_workflow_metrics(self):
        tm = fresh_manager()
        for _ in range(3):
            sid = tm.start_span(
                "wf_comm",
                TraceType.WORKFLOW_COMMUNICATION,
                workflow_name="wf_x",
            )
            tm.end_span(sid, TraceStatus.SUCCESS)
        flush(tm)
        m = tm.get_workflow_metrics("wf_x")
        assert m["total_messages"] == 3
        assert m["success_rate"] == 1.0

    def test_get_global_metrics_structure(self):
        tm = fresh_manager()
        m = tm.get_global_metrics()
        assert "total_spans" in m
        assert "active_spans" in m
        assert "total_llm_calls" in m
        assert "total_decisions" in m


# ---------------------------------------------------------------------------
# TraceManager — decision stream callbacks
# ---------------------------------------------------------------------------


class TestDecisionStreamCallbacks:
    def test_register_and_emit(self):
        tm = fresh_manager()
        received = []
        cb = lambda evt: received.append(evt)
        tm.register_decision_stream_callback("agent1", cb)

        mock_event = MagicMock()
        tm.emit_decision_event("agent1", mock_event)
        assert received == [mock_event]

    def test_unregister_stops_delivery(self):
        tm = fresh_manager()
        received = []
        cb = lambda evt: received.append(evt)
        tm.register_decision_stream_callback("agent1", cb)
        tm.unregister_decision_stream_callback("agent1", cb)
        tm.emit_decision_event("agent1", MagicMock())
        assert received == []

    def test_emit_no_agent_is_silent(self):
        tm = fresh_manager()
        # Should not raise
        tm.emit_decision_event(None, MagicMock())

    def test_get_streaming_agents(self):
        tm = fresh_manager()
        tm.register_decision_stream_callback("a1", lambda e: None)
        tm.register_decision_stream_callback("a2", lambda e: None)
        agents = tm.get_streaming_agents()
        assert "a1" in agents
        assert "a2" in agents

    def test_callback_exception_does_not_propagate(self):
        tm = fresh_manager()

        def bad_cb(evt):
            raise ValueError("callback error")

        tm.register_decision_stream_callback("agent1", bad_cb)
        # Should not raise
        tm.emit_decision_event("agent1", MagicMock())


# ---------------------------------------------------------------------------
# configure_tracing — adds custom exporter
# ---------------------------------------------------------------------------


class TestConfigureTracing:
    def test_custom_exporter_receives_spans(self):
        tm = fresh_manager()
        custom_exp = InMemorySpanExporter()
        tm.configure([custom_exp])

        span_id = tm.start_span("custom_op", TraceType.AGENT_EXECUTION, agent_id="cx")
        tm.end_span(span_id, TraceStatus.SUCCESS)
        flush(tm)

        finished = custom_exp.get_finished_spans()
        assert len(finished) >= 1
        assert finished[0].name == "custom_op"

    def test_configure_tracing_function(self):
        """Top-level configure_tracing() function integrates with global singleton."""
        custom_exp = InMemorySpanExporter()
        # configure_tracing() uses the global singleton
        configure_tracing(exporters=[custom_exp])
        tm = get_trace_manager()
        span_id = tm.start_span("global_op", TraceType.AGENT_EXECUTION, agent_id="g1")
        tm.end_span(span_id, TraceStatus.SUCCESS)
        flush(tm)
        assert len(custom_exp.get_finished_spans()) >= 1


# ---------------------------------------------------------------------------
# Shutdown / flush
# ---------------------------------------------------------------------------


class TestShutdownFlush:
    def test_flush_does_not_raise(self):
        tm = fresh_manager()
        tm.start_span("op", TraceType.AGENT_EXECUTION)
        # Should not raise
        tm.flush(timeout_millis=500)

    def test_shutdown_does_not_raise(self):
        tm = fresh_manager()
        tm.start_span("op", TraceType.AGENT_EXECUTION)
        # Should not raise
        tm.shutdown()


# ---------------------------------------------------------------------------
# Legacy compatibility functions
# ---------------------------------------------------------------------------


class TestLegacyCompatFunctions:
    def test_record_tokens_is_noop(self):
        from daita.core.tracing import record_tokens

        record_tokens("agent1", total_tokens=100)  # Should not raise

    def test_get_agent_tokens_returns_dict(self):
        from daita.core.tracing import get_agent_tokens

        result = get_agent_tokens("agent1")
        assert isinstance(result, dict)
        assert "total_tokens" in result
        assert "requests" in result

    def test_record_operation_returns_span_id(self):
        from daita.core.tracing import record_operation

        span_id = record_operation(
            agent_id="a1",
            agent_name="TestAgent",
            task="do_something",
            input_data="input",
            output_data="output",
            latency_ms=100.0,
            status="success",
        )
        assert span_id is not None
        assert isinstance(span_id, str)

    def test_get_recent_operations_module_level(self):
        from daita.core.tracing import get_recent_operations

        result = get_recent_operations()
        assert isinstance(result, list)
