"""
Unified TraceManager for Daita Agents

OpenTelemetry-backed tracing that captures all agent operations, LLM calls,
workflow communication, and tool usage.  Zero configuration required —
create an Agent and tracing just works.

Architecture
------------
TraceManager is a thin facade over an OTel TracerProvider:

    TracerProvider
      ├── BatchSpanProcessor → BoundedInMemorySpanExporter   (powers query APIs)
      ├── BatchSpanProcessor → DaitaSpanExporter              (dashboard, when configured)
      └── BatchSpanProcessor → [user exporters]               (Datadog/Jaeger via configure_tracing)

Trace IDs are always W3C hex format (32-char trace ID, 16-char span ID).

Context propagation
-------------------
Three module-level ContextVars shadow the OTel context so that
``agent.trace_id`` and ``agent.current_span_id`` work without requiring
callers to import the OTel API.  They are updated inside ``span()`` (the
async context manager) and by ``start_span()`` / ``end_span()``.
"""

import atexit
import json
import logging
import os
import time
import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from .otel_exporter import BoundedInMemorySpanExporter, DaitaSpanExporter

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public enums / dataclasses (API unchanged)
# ---------------------------------------------------------------------------


class TraceType(str, Enum):
    """Types of traces we capture."""

    AGENT_EXECUTION = "agent_execution"
    LLM_CALL = "llm_call"
    WORKFLOW_COMMUNICATION = "workflow_communication"
    AGENT_LIFECYCLE = "agent_lifecycle"
    DECISION_TRACE = "decision_trace"
    TOOL_EXECUTION = "tool_execution"


class TraceStatus(str, Enum):
    """Status of a trace span."""

    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"


class TraceSpan:
    """
    Lightweight span representation returned by query methods.

    Constructed from OTel ReadableSpan data; mirrors the old dataclass API
    so call-sites like ``span.to_dict()`` continue working.
    """

    def __init__(
        self,
        span_id: str,
        trace_id: str,
        parent_span_id: Optional[str],
        agent_id: Optional[str],
        operation_name: str,
        trace_type: "TraceType",
        start_time: float,
        end_time: Optional[float],
        status: "TraceStatus",
        input_data: Any,
        output_data: Any,
        error_message: Optional[str],
        duration_ms: Optional[float],
        metadata: Dict[str, Any],
        deployment_id: Optional[str] = None,
        environment: str = "",
    ) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.agent_id = agent_id
        self.operation_name = operation_name
        self.trace_type = trace_type
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self.input_data = input_data
        self.output_data = output_data
        self.error_message = error_message
        self.duration_ms = duration_ms
        self.metadata = metadata
        self.deployment_id = deployment_id or os.getenv("DAITA_DEPLOYMENT_ID")
        self.environment = environment or os.getenv("DAITA_ENVIRONMENT", "development")

    @property
    def is_completed(self) -> bool:
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "agent_id": self.agent_id,
            "operation": self.operation_name,
            "type": (
                self.trace_type.value
                if isinstance(self.trace_type, TraceType)
                else self.trace_type
            ),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": (
                self.status.value
                if isinstance(self.status, TraceStatus)
                else self.status
            ),
            "input_preview": self._create_preview(self.input_data),
            "output_preview": self._create_preview(self.output_data),
            "error": self.error_message,
            "metadata": self.metadata,
            "environment": self.environment,
            "deployment_id": self.deployment_id,
        }

    def _create_preview(self, data: Any, max_length: int = 200) -> str:
        if data is None:
            return ""
        try:
            if isinstance(data, str):
                preview = data
            elif isinstance(data, dict):
                preview = json.dumps(data, separators=(",", ":"))
            else:
                preview = str(data)
            if len(preview) > max_length:
                return preview[:max_length] + "..."
            return preview
        except Exception:
            return f"<{type(data).__name__}>"


# ---------------------------------------------------------------------------
# Context vars (one per async task; correctly propagated across await)
# ---------------------------------------------------------------------------

_trace_id_var: ContextVar[Optional[str]] = ContextVar("daita_trace_id", default=None)
_span_id_var: ContextVar[Optional[str]] = ContextVar("daita_span_id", default=None)
_agent_id_var: ContextVar[Optional[str]] = ContextVar("daita_agent_id", default=None)


class TraceContext:
    """Async-native trace context — read by agent.trace_id / current_span_id."""

    @property
    def current_trace_id(self) -> Optional[str]:
        return _trace_id_var.get()

    @property
    def current_span_id(self) -> Optional[str]:
        return _span_id_var.get()

    @property
    def current_agent_id(self) -> Optional[str]:
        return _agent_id_var.get()

    @asynccontextmanager
    async def span_context(
        self, trace_id: str, span_id: str, agent_id: Optional[str] = None
    ):
        """Scope trace/span IDs to the current async task with proper cleanup."""
        trace_token = _trace_id_var.set(trace_id)
        span_token = _span_id_var.set(span_id)
        agent_token = _agent_id_var.set(agent_id) if agent_id is not None else None
        try:
            yield
        finally:
            _trace_id_var.reset(trace_token)
            _span_id_var.reset(span_token)
            if agent_token is not None:
                _agent_id_var.reset(agent_token)


# ---------------------------------------------------------------------------
# Attribute mapping helper
# ---------------------------------------------------------------------------


def _map_metadata_to_attributes(
    trace_type: Any,
    agent_id: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert daita metadata kwargs to OTel attribute dict."""
    attrs: Dict[str, Any] = {}

    # Core daita attributes
    type_val = (
        trace_type.value if isinstance(trace_type, TraceType) else str(trace_type)
    )
    attrs["daita.trace.type"] = type_val

    if agent_id:
        attrs["daita.agent.id"] = agent_id

    # Agent name
    if metadata.get("agent_name"):
        attrs["daita.agent.name"] = str(metadata["agent_name"])

    # OTel GenAI semconv
    if metadata.get("model"):
        attrs["gen_ai.request.model"] = str(metadata["model"])
    if metadata.get("tokens_prompt") is not None:
        attrs["gen_ai.usage.input_tokens"] = int(metadata["tokens_prompt"])
    if metadata.get("tokens_completion") is not None:
        attrs["gen_ai.usage.output_tokens"] = int(metadata["tokens_completion"])
    if metadata.get("tokens_total") is not None:
        attrs["gen_ai.usage.total_tokens"] = int(metadata["tokens_total"])

    # Decision attributes
    if metadata.get("confidence_score") is not None:
        attrs["daita.decision.confidence"] = float(metadata["confidence_score"])
    if metadata.get("reasoning_chain") is not None:
        attrs["daita.decision.reasoning"] = json.dumps(metadata["reasoning_chain"])
    if metadata.get("alternatives") is not None:
        attrs["daita.decision.alternatives"] = json.dumps(metadata["alternatives"])

    # Agent input — captured from prompt kwarg passed to span()
    if metadata.get("prompt"):
        attrs["daita.agent.prompt"] = str(metadata["prompt"])[:2000]

    # Tool attributes
    if metadata.get("tool_name"):
        attrs["daita.tool.name"] = str(metadata["tool_name"])

    # Workflow attributes
    if metadata.get("workflow_name"):
        attrs["daita.workflow.name"] = str(metadata["workflow_name"])
    if metadata.get("from_agent"):
        attrs["daita.comm.from_agent"] = str(metadata["from_agent"])
    if metadata.get("to_agent"):
        attrs["daita.comm.to_agent"] = str(metadata["to_agent"])
    if metadata.get("channel"):
        attrs["daita.comm.channel"] = str(metadata["channel"])

    # Deployment / environment
    if metadata.get("deployment_id"):
        attrs["deployment.id"] = str(metadata["deployment_id"])
    env = metadata.get("environment") or os.getenv("DAITA_ENVIRONMENT", "development")
    attrs["deployment.environment"] = env

    # OTel only allows str/bool/int/float (and sequences thereof) as attribute values.
    # Filter out anything else silently.
    _VALID = (str, bool, int, float)
    return {k: v for k, v in attrs.items() if isinstance(v, _VALID)}


# ---------------------------------------------------------------------------
# TraceManager
# ---------------------------------------------------------------------------


class TraceManager:
    """
    OTel-backed tracing facade.  Public API is identical to the old
    in-memory implementation — all call-sites continue working unchanged.
    """

    def __init__(self) -> None:
        self.trace_context = TraceContext()

        # --- Build the OTel TracerProvider ---
        resource = Resource.create(
            {
                "service.name": "daita-agents",
                "service.version": _get_version(),
            }
        )
        self._provider = TracerProvider(resource=resource)

        # Always-on: bounded in-memory store (powers local query APIs)
        self._memory_exporter = BoundedInMemorySpanExporter(maxlen=500)
        # Use a low schedule_delay so query methods see spans quickly after end_span()
        self._provider.add_span_processor(
            BatchSpanProcessor(
                self._memory_exporter,
                schedule_delay_millis=100,
                max_export_batch_size=512,
            )
        )

        # Dashboard exporter — self-disabling when env vars absent
        self._daita_exporter = DaitaSpanExporter()
        self._provider.add_span_processor(BatchSpanProcessor(self._daita_exporter))

        # Register as the global OTel provider
        otel_trace.set_tracer_provider(self._provider)
        self._tracer = self._provider.get_tracer("daita.agents")

        # Live OTel spans keyed by hex span_id (for manual start/end pattern)
        self._otel_spans: Dict[str, otel_trace.Span] = {}
        self._otel_spans_lock = threading.Lock()

        # Lightweight counters (not span-derived, stay in-process)
        self._metrics: Dict[str, int] = {
            "total_spans": 0,
            "total_llm_calls": 0,
            "total_tokens": 0,
            "total_decisions": 0,
        }

        # Decision streaming callbacks (in-process only, not OTel-related)
        self._decision_stream_callbacks: Dict[str, List[Callable]] = {}

        logger.info("TraceManager initialized (OTel backend)")

    # ------------------------------------------------------------------
    # Span lifecycle — public API (signatures unchanged)
    # ------------------------------------------------------------------

    def start_span(
        self,
        operation_name: str,
        trace_type: TraceType,
        agent_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        **metadata,
    ) -> str:
        """Start a new trace span and return its hex span ID."""
        try:
            if not agent_id:
                agent_id = self.trace_context.current_agent_id

            # Determine OTel parent context
            ctx = None
            if parent_span_id:
                with self._otel_spans_lock:
                    parent_otel = self._otel_spans.get(parent_span_id)
                if parent_otel is not None:
                    ctx = otel_trace.set_span_in_context(parent_otel)
            elif _trace_id_var.get() is not None:
                # Use whatever is currently active in OTel context
                ctx = None  # start_span picks up current context automatically

            attrs = _map_metadata_to_attributes(trace_type, agent_id, metadata)

            span = self._tracer.start_span(
                operation_name,
                context=ctx,
                kind=SpanKind.INTERNAL,
                attributes=attrs,
            )

            span_ctx = span.get_span_context()
            hex_span_id = format(span_ctx.span_id, "016x")
            hex_trace_id = format(span_ctx.trace_id, "032x")

            with self._otel_spans_lock:
                self._otel_spans[hex_span_id] = span

            # Mirror to custom contextvars (only update span_id; trace_id stays from context)
            _trace_id_var.set(hex_trace_id)
            _span_id_var.set(hex_span_id)
            if agent_id:
                _agent_id_var.set(agent_id)

            self._metrics["total_spans"] += 1
            logger.debug(f"Started span {hex_span_id} for '{operation_name}'")
            return hex_span_id

        except Exception as e:
            logger.error(f"Failed to start span: {e}")
            return f"error_{id(e):016x}"

    def end_span(
        self,
        span_id: str,
        status: TraceStatus = TraceStatus.SUCCESS,
        output_data: Any = None,
        error_message: Optional[str] = None,
        **metadata,
    ) -> None:
        """End a trace span."""
        try:
            with self._otel_spans_lock:
                span = self._otel_spans.pop(span_id, None)
            if span is None:
                logger.debug(f"Unknown or already completed span: {span_id}")
                return

            # Set additional attributes from metadata
            extra_attrs = _map_metadata_to_attributes(
                metadata.get("trace_type", ""),
                metadata.get("agent_id"),
                metadata,
            )
            for k, v in extra_attrs.items():
                if k not in ("daita.trace.type", "daita.agent.id"):  # already set
                    span.set_attribute(k, v)

            # Record input/output as span events (they can be large)
            if metadata.get("input_data") is not None:
                _add_data_event(span, "daita.input", metadata["input_data"])
            if output_data is not None:
                _add_data_event(span, "daita.output", output_data)

            # Status
            if status == TraceStatus.ERROR:
                span.set_status(Status(StatusCode.ERROR, error_message or ""))
                if error_message:
                    span.set_attribute("error.message", error_message)
            else:
                span.set_status(Status(StatusCode.OK))

            span.end()

            # Update counters
            attrs = span.attributes or {}
            trace_type_val = attrs.get("daita.trace.type", "")
            if trace_type_val == TraceType.LLM_CALL.value:
                self._metrics["total_llm_calls"] += 1
                self._metrics["total_tokens"] += int(
                    attrs.get("gen_ai.usage.total_tokens", 0)
                    or metadata.get("tokens_total", 0)
                )
            elif trace_type_val == TraceType.DECISION_TRACE.value:
                self._metrics["total_decisions"] += 1

            logger.debug(f"Ended span {span_id} (status={status.value})")

        except Exception as e:
            logger.error(f"Failed to end span {span_id}: {e}")
            with self._otel_spans_lock:
                self._otel_spans.pop(span_id, None)

    def record_decision(
        self,
        span_id: str,
        confidence: float = 0.0,
        reasoning: Optional[List[str]] = None,
        alternatives: Optional[List[str]] = None,
        **factors,
    ) -> None:
        """Record decision metadata onto a live span."""
        try:
            with self._otel_spans_lock:
                span = self._otel_spans.get(span_id)
            if span:
                span.set_attribute("daita.decision.confidence", float(confidence))
                if reasoning is not None:
                    span.set_attribute(
                        "daita.decision.reasoning", json.dumps(reasoning)
                    )
                if alternatives is not None:
                    span.set_attribute(
                        "daita.decision.alternatives", json.dumps(alternatives)
                    )
                if factors:
                    span.set_attribute("daita.decision.factors", json.dumps(factors))
                logger.debug(
                    f"Recorded decision for span {span_id} (confidence={confidence:.2f})"
                )
            else:
                logger.debug(f"Cannot record decision for unknown span: {span_id}")
        except Exception as e:
            logger.error(f"Failed to record decision for span {span_id}: {e}")

    def record_llm_call(
        self,
        span_id: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        **llm_metadata,
    ) -> None:
        """Record LLM call metadata onto a live span."""
        try:
            with self._otel_spans_lock:
                span = self._otel_spans.get(span_id)
            if span:
                span.set_attribute("gen_ai.request.model", model)
                span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
                span.set_attribute(
                    "gen_ai.usage.total_tokens",
                    total_tokens or (prompt_tokens + completion_tokens),
                )
                logger.debug(
                    f"Recorded LLM call for span {span_id} ({total_tokens} tokens)"
                )
            else:
                logger.debug(f"Cannot record LLM call for unknown span: {span_id}")
        except Exception as e:
            logger.error(f"Failed to record LLM call for span {span_id}: {e}")

    @asynccontextmanager
    async def span(
        self,
        operation_name: str,
        trace_type: TraceType,
        agent_id: Optional[str] = None,
        **metadata,
    ):
        """Async context manager for automatic span lifecycle."""
        if not agent_id:
            agent_id = self.trace_context.current_agent_id

        attrs = _map_metadata_to_attributes(trace_type, agent_id, metadata)

        with self._tracer.start_as_current_span(
            operation_name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as otel_span:
            span_ctx = otel_span.get_span_context()
            hex_span_id = format(span_ctx.span_id, "016x")
            hex_trace_id = format(span_ctx.trace_id, "032x")

            # Also register in _otel_spans so callers who stash the span_id can use it
            with self._otel_spans_lock:
                self._otel_spans[hex_span_id] = otel_span

            self._metrics["total_spans"] += 1

            # Mirror to custom contextvars
            async with self.trace_context.span_context(
                hex_trace_id, hex_span_id, agent_id
            ):
                try:
                    yield hex_span_id
                    otel_span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                    otel_span.record_exception(e)
                    raise
                finally:
                    with self._otel_spans_lock:
                        self._otel_spans.pop(hex_span_id, None)

    # Convenience wrappers (API unchanged)

    async def decision_span(
        self, decision_point: str, agent_id: Optional[str] = None, **metadata
    ):
        metadata.update({"decision_point": decision_point, "trace_subtype": "decision"})
        return self.span(
            f"decision_{decision_point}", TraceType.DECISION_TRACE, agent_id, **metadata
        )

    async def tool_span(
        self, tool_name: str, operation: str, agent_id: Optional[str] = None, **metadata
    ):
        metadata.update({"tool_name": tool_name, "tool_operation": operation})
        return self.span(
            f"tool_{tool_name}_{operation}",
            TraceType.TOOL_EXECUTION,
            agent_id,
            **metadata,
        )

    # ------------------------------------------------------------------
    # Query methods — read from BoundedInMemorySpanExporter
    # ------------------------------------------------------------------

    def get_recent_operations(
        self, agent_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent completed operations, most recent first."""
        try:
            spans = self._memory_exporter.get_finished_spans()
            if agent_id:
                spans = [
                    s
                    for s in spans
                    if (s.attributes or {}).get("daita.agent.id") == agent_id
                ]
            result = [_readable_span_to_dict(s) for s in spans[-limit:]]
            return list(reversed(result))
        except Exception as e:
            logger.error(f"Error getting recent operations: {e}")
            return []

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global tracing metrics."""
        with self._otel_spans_lock:
            active = len(self._otel_spans)
        return {
            **self._metrics,
            "active_spans": active,
            "dashboard_reports_sent": 0,  # legacy field — exporter handles internally
            "dashboard_reports_failed": 0,
        }

    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent."""
        try:
            spans = [
                s
                for s in self._memory_exporter.get_finished_spans()
                if (s.attributes or {}).get("daita.agent.id") == agent_id
            ]
            if not spans:
                return {"total_operations": 0, "success_rate": 0}

            total_ops = len(spans)
            successful_ops = sum(
                1 for s in spans if s.status and s.status.status_code == StatusCode.OK
            )
            latencies = []
            for s in spans:
                if s.start_time and s.end_time:
                    latencies.append((s.end_time - s.start_time) / 1_000_000)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            return {
                "total_operations": total_ops,
                "successful_operations": successful_ops,
                "failed_operations": total_ops - successful_ops,
                "success_rate": successful_ops / total_ops,
                "avg_latency_ms": avg_latency,
            }
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return {"total_operations": 0, "success_rate": 0}

    def get_workflow_communications(
        self, workflow_name: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get workflow communication traces (agent-to-agent messages)."""
        try:
            spans = [
                s
                for s in self._memory_exporter.get_finished_spans()
                if (s.attributes or {}).get("daita.trace.type")
                == TraceType.WORKFLOW_COMMUNICATION.value
                and (
                    workflow_name is None
                    or (s.attributes or {}).get("daita.workflow.name") == workflow_name
                )
            ]
            result = []
            for s in reversed(spans[-limit:]):
                d = _readable_span_to_dict(s)
                attrs = s.attributes or {}
                d["from_agent"] = attrs.get("daita.comm.from_agent", "unknown")
                d["to_agent"] = attrs.get("daita.comm.to_agent", "unknown")
                d["channel"] = attrs.get("daita.comm.channel", "unknown")
                d["success"] = s.status and s.status.status_code == StatusCode.OK
                result.append(d)
            return result
        except Exception as e:
            logger.error(f"Error getting workflow communications: {e}")
            return []

    def get_workflow_metrics(self, workflow_name: str) -> Dict[str, Any]:
        """Get metrics for a specific workflow."""
        try:
            spans = [
                s
                for s in self._memory_exporter.get_finished_spans()
                if (s.attributes or {}).get("daita.trace.type")
                == TraceType.WORKFLOW_COMMUNICATION.value
                and (s.attributes or {}).get("daita.workflow.name") == workflow_name
            ]
            if not spans:
                return {"total_messages": 0, "success_rate": 0}
            total = len(spans)
            successful = sum(
                1 for s in spans if s.status and s.status.status_code == StatusCode.OK
            )
            return {
                "workflow_name": workflow_name,
                "total_messages": total,
                "successful_messages": successful,
                "failed_messages": total - successful,
                "success_rate": successful / total,
            }
        except Exception as e:
            logger.error(f"Error getting workflow metrics: {e}")
            return {"total_messages": 0, "success_rate": 0}

    # ------------------------------------------------------------------
    # Decision stream callbacks (in-process, not OTel-related)
    # ------------------------------------------------------------------

    def register_decision_stream_callback(
        self, agent_id: str, callback: Callable
    ) -> None:
        self._decision_stream_callbacks.setdefault(agent_id, []).append(callback)
        logger.debug(f"Registered decision stream callback for agent {agent_id}")

    def unregister_decision_stream_callback(
        self, agent_id: str, callback: Callable
    ) -> None:
        callbacks = self._decision_stream_callbacks.get(agent_id)
        if callbacks:
            try:
                callbacks.remove(callback)
            except ValueError:
                pass
            if not callbacks:
                del self._decision_stream_callbacks[agent_id]
        logger.debug(f"Unregistered decision stream callback for agent {agent_id}")

    def emit_decision_event(
        self, agent_id: Optional[str], decision_event: "DecisionEvent"
    ) -> None:
        if not agent_id:
            return
        for callback in list(self._decision_stream_callbacks.get(agent_id, [])):
            try:
                callback(decision_event)
            except Exception as e:
                logger.warning(
                    f"Decision stream callback failed for agent {agent_id}: {e}"
                )

    def get_streaming_agents(self) -> List[str]:
        return list(self._decision_stream_callbacks.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush pending spans and shut down the tracer provider."""
        try:
            self._provider.shutdown()
        except Exception as e:
            logger.debug(f"TraceManager shutdown error (ignored): {e}")

    def flush(self, timeout_millis: int = 5000) -> None:
        """Force-flush all pending spans to exporters."""
        try:
            self._provider.force_flush(timeout_millis)
        except Exception as e:
            logger.debug(f"TraceManager flush error (ignored): {e}")

    def configure(self, exporters: List["SpanExporter"]) -> None:
        """Add additional span exporters (e.g. OTLP for Datadog/Jaeger)."""
        for exporter in exporters:
            self._provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"Added span exporter: {type(exporter).__name__}")


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_trace_manager: Optional[TraceManager] = None


def get_trace_manager() -> TraceManager:
    """Get the global TraceManager instance (created once)."""
    global _global_trace_manager
    if _global_trace_manager is None:
        _global_trace_manager = TraceManager()
        atexit.register(_global_trace_manager.shutdown)
    return _global_trace_manager


# ---------------------------------------------------------------------------
# configure_tracing() — user-facing API for custom exporters
# ---------------------------------------------------------------------------


def configure_tracing(
    *,
    exporters: Optional[List["SpanExporter"]] = None,
    resource_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add custom OTel exporters to the Daita tracing pipeline.

    Call this **before** creating any Agent instances.

    Args:
        exporters: List of SpanExporter instances (e.g., OTLPSpanExporter for Datadog).
        resource_attributes: Additional OTel Resource attributes to merge in.

    Example::

        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from daita.core.tracing import configure_tracing

        configure_tracing(exporters=[OTLPSpanExporter(endpoint="http://localhost:4317")])
    """
    tm = get_trace_manager()
    if exporters:
        tm.configure(exporters)
    if resource_attributes:
        logger.warning(
            "configure_tracing(resource_attributes=...) has no effect after the "
            "TraceManager is initialized.  Pass resource_attributes before the first "
            "agent is created."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_version() -> str:
    try:
        from importlib.metadata import version

        return version("daita-agents")
    except Exception:
        return "unknown"


def _add_data_event(span: otel_trace.Span, event_name: str, data: Any) -> None:
    """Add input/output data as a span event (truncated to 2000 chars)."""
    try:
        if isinstance(data, str):
            preview = data
        elif isinstance(data, dict):
            preview = json.dumps(data, separators=(",", ":"))
        else:
            preview = str(data)
        if len(preview) > 2000:
            preview = preview[:2000] + "..."
        span.add_event(event_name, {"data": preview})
    except Exception:
        pass


def _readable_span_to_dict(span) -> Dict[str, Any]:
    """Convert an OTel ReadableSpan to the legacy to_dict() format."""
    ctx = span.get_span_context()
    hex_span_id = format(ctx.span_id, "016x") if ctx else "unknown"
    hex_trace_id = format(ctx.trace_id, "032x") if ctx else "unknown"

    parent_span_id = None
    if span.parent:
        parent_span_id = format(span.parent.span_id, "016x")

    attrs = dict(span.attributes) if span.attributes else {}

    start_time = span.start_time / 1e9 if span.start_time else None
    end_time = span.end_time / 1e9 if span.end_time else None
    duration_ms = None
    if span.start_time and span.end_time:
        duration_ms = (span.end_time - span.start_time) / 1_000_000

    status_val = TraceStatus.SUCCESS
    if span.status and span.status.status_code == StatusCode.ERROR:
        status_val = TraceStatus.ERROR

    # Extract input/output from span events
    input_preview = ""
    output_preview = ""
    for event in span.events:
        if event.name == "daita.input":
            input_preview = (event.attributes or {}).get("data", "")
        elif event.name == "daita.output":
            output_preview = (event.attributes or {}).get("data", "")

    error_message = attrs.get("error.message") or (
        span.status.description if span.status else None
    )

    trace_type_str = attrs.get("daita.trace.type", "agent_execution")
    try:
        trace_type = TraceType(trace_type_str)
    except ValueError:
        trace_type = TraceType.AGENT_EXECUTION

    return {
        "span_id": hex_span_id,
        "trace_id": hex_trace_id,
        "parent_span_id": parent_span_id,
        "agent_id": attrs.get("daita.agent.id"),
        "operation": span.name,
        "type": trace_type_str,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
        "status": status_val.value,
        "input_preview": input_preview,
        "output_preview": output_preview,
        "error": error_message,
        "metadata": {
            k: v
            for k, v in attrs.items()
            if not k.startswith("gen_ai.")
            and not k.startswith("daita.")
            and not k.startswith("deployment.")
        },
        "environment": attrs.get("deployment.environment", "development"),
        "deployment_id": attrs.get("deployment.id"),
    }


# ---------------------------------------------------------------------------
# Legacy compatibility functions (none are called anywhere in the codebase,
# but kept to avoid breaking any external code that may import them)
# ---------------------------------------------------------------------------


def record_tokens(
    agent_id: str,
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Legacy token recording — no-op."""
    pass


def get_agent_tokens(agent_id: str) -> Dict[str, int]:
    """Legacy token retrieval."""
    metrics = get_trace_manager().get_agent_metrics(agent_id)
    return {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "requests": metrics.get("total_operations", 0),
    }


def record_operation(
    agent_id: str,
    agent_name: str,
    task: str,
    input_data: Any,
    output_data: Any,
    latency_ms: float,
    status: str = "success",
    **kwargs,
) -> str:
    """Legacy operation recording."""
    tm = get_trace_manager()
    span_id = tm.start_span(
        operation_name=task,
        trace_type=TraceType.AGENT_EXECUTION,
        agent_id=agent_id,
        input_data=input_data,
        agent_name=agent_name,
    )
    trace_status = TraceStatus.SUCCESS if status == "success" else TraceStatus.ERROR
    tm.end_span(
        span_id=span_id,
        status=trace_status,
        output_data=output_data,
        error_message=kwargs.get("error_message"),
    )
    return span_id


def get_recent_operations(
    agent_id: Optional[str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """Legacy function to get recent operations."""
    return get_trace_manager().get_recent_operations(agent_id, limit)
