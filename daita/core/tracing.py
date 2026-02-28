"""
Unified TraceManager for Daita Agents

Automatic tracing system that captures all agent operations, LLM calls,
workflow communication, and tool usage. Zero configuration required.

Uses contextvars for async-native context propagation — trace IDs and span
IDs are correctly scoped per async task and survive await boundaries.
"""

import asyncio
import logging
import time
import uuid
import json
import os
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

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

@dataclass
class TraceSpan:
    """A single trace span - simplified for MVP."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    agent_id: Optional[str]
    operation_name: str
    trace_type: TraceType
    start_time: float
    end_time: Optional[float]
    status: TraceStatus
    
    # Core data
    input_data: Any
    output_data: Any
    error_message: Optional[str]
    
    # Performance
    duration_ms: Optional[float]
    
    # Metadata - simple dict for flexibility
    metadata: Dict[str, Any]
    
    # Environment context
    deployment_id: Optional[str]
    environment: str
    
    def __post_init__(self):
        """Auto-populate deployment context."""
        if self.deployment_id is None:
            self.deployment_id = os.getenv("DAITA_DEPLOYMENT_ID")
        if not self.environment:
            self.environment = os.getenv("DAITA_ENVIRONMENT", "development")
    
    @property
    def is_completed(self) -> bool:
        return self.end_time is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "agent_id": self.agent_id,
            "operation": self.operation_name,
            "type": self.trace_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "input_preview": self._create_preview(self.input_data),
            "output_preview": self._create_preview(self.output_data),
            "error": self.error_message,
            "metadata": self.metadata,
            "environment": self.environment,
            "deployment_id": self.deployment_id
        }
    
    def _create_preview(self, data: Any, max_length: int = 200) -> str:
        """Create a preview string for data."""
        if data is None:
            return ""
        try:
            if isinstance(data, str):
                preview = data
            elif isinstance(data, dict):
                preview = json.dumps(data, separators=(',', ':'))
            else:
                preview = str(data)
            
            if len(preview) > max_length:
                return preview[:max_length] + "..."
            return preview
        except Exception:
            return f"<{type(data).__name__}>"

# Module-level ContextVars — one per async task, correctly propagated across
# await boundaries and isolated between concurrent tasks.
_trace_id_var: ContextVar[Optional[str]] = ContextVar('daita_trace_id', default=None)
_span_id_var: ContextVar[Optional[str]] = ContextVar('daita_span_id', default=None)
_agent_id_var: ContextVar[Optional[str]] = ContextVar('daita_agent_id', default=None)


class TraceContext:
    """Async-native trace context using contextvars for correct propagation across await points."""

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
    async def span_context(self, trace_id: str, span_id: str, agent_id: Optional[str] = None):
        """Context manager that scopes trace/span IDs to the current async task.

        Uses ContextVar.reset() tokens so nested spans correctly restore the
        parent context on exit, even if the coroutine is suspended and resumed.
        """
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

class DashboardReporter:
    """Dashboard reporting with proper dependency management."""
    
    def __init__(self):
        self.api_key = os.getenv("DAITA_API_KEY")
        self.dashboard_url = os.getenv("DAITA_DASHBOARD_URL") or os.getenv("DAITA_DASHBOARD_API") or os.getenv("DAITA_DASHBOARD_API_OVERRIDE") or ""
        self.enabled = bool(self.api_key and self.dashboard_url)
        self.reports_sent = 0
        self.reports_failed = 0
        self._aiohttp_available = None
        
        # Validate configuration
        if self.api_key and not self.dashboard_url:
            self.enabled = False
        
        if self.enabled:
            logger.info(f"Dashboard reporting enabled (URL: {self.dashboard_url})")
        else:
            logger.debug("Dashboard reporting disabled (API key or URL not configured)")
    
    def _check_aiohttp(self) -> bool:
        """Check if aiohttp is available (cached result)."""
        if self._aiohttp_available is None:
            try:
                import aiohttp
                self._aiohttp_available = True
                logger.debug("aiohttp available for dashboard reporting")
            except ImportError:
                self._aiohttp_available = False
                logger.warning("aiohttp not available - dashboard reporting will be skipped")
        return self._aiohttp_available
    
    async def report_span(self, span: TraceSpan) -> bool:
        """Report a single span to dashboard with proper error handling."""
        if not self.enabled:
            return True
            
        if not self._check_aiohttp():
            # Don't log this repeatedly
            return False
        
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "daita-agents/0.1.1"
            }
            
            payload = {
                "spans": [span.to_dict()],
                "environment": span.environment,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.dashboard_url}/v1/traces",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        self.reports_sent += 1
                        logger.debug(f"Successfully reported span {span.span_id}")
                        return True
                    else:
                        self.reports_failed += 1
                        logger.warning(f"Dashboard API error: {response.status} - {await response.text()}")
                        return False
                        
        except asyncio.TimeoutError:
            self.reports_failed += 1
            logger.warning("Dashboard reporting timeout")
            return False
        except Exception as e:
            self.reports_failed += 1
            logger.warning(f"Dashboard reporting failed: {e}")
            return False

class TraceManager:
    """
    Automatic tracing for all agent operations.

    Designed for single event-loop async use. Dict/deque access is safe without
    locking because all mutations happen in synchronous sections (no await between
    read and write). contextvars handle per-task context isolation.
    """

    def __init__(self):
        self.trace_context = TraceContext()
        self.dashboard_reporter = DashboardReporter()

        self._active_spans: Dict[str, TraceSpan] = {}
        self._completed_spans: deque = deque(maxlen=500)
        self._metrics = {
            "total_spans": 0,
            "total_llm_calls": 0,
            "total_tokens": 0,
            "total_decisions": 0,
        }
        self._decision_stream_callbacks: Dict[str, List[Callable]] = {}

        logger.info("TraceManager initialized")
    
    def start_span(
        self,
        operation_name: str,
        trace_type: TraceType,
        agent_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        **metadata
    ) -> str:
        """Start a new trace span."""
        try:
            span_id = str(uuid.uuid4())

            # Determine trace_id — inherit from parent or current context
            if parent_span_id:
                parent = self._active_spans.get(parent_span_id)
                trace_id = parent.trace_id if parent else str(uuid.uuid4())
            elif self.trace_context.current_trace_id:
                trace_id = self.trace_context.current_trace_id
                parent_span_id = self.trace_context.current_span_id
            else:
                trace_id = str(uuid.uuid4())

            if not agent_id:
                agent_id = self.trace_context.current_agent_id

            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                agent_id=agent_id,
                operation_name=operation_name,
                trace_type=trace_type,
                start_time=time.time(),
                end_time=None,
                status=TraceStatus.STARTED,
                input_data=metadata.get('input_data'),
                output_data=None,
                error_message=None,
                duration_ms=None,
                metadata=metadata,
                deployment_id=None,
                environment=""
            )

            self._active_spans[span_id] = span
            self._metrics["total_spans"] += 1
            logger.debug(f"Started span {span_id} for '{operation_name}'")
            return span_id

        except Exception as e:
            logger.error(f"Failed to start span: {e}")
            return f"error_{uuid.uuid4().hex[:8]}"
    
    def end_span(
        self,
        span_id: str,
        status: TraceStatus = TraceStatus.SUCCESS,
        output_data: Any = None,
        error_message: Optional[str] = None,
        **metadata
    ) -> None:
        """End a trace span."""
        try:
            span = self._active_spans.pop(span_id, None)
            if span is None:
                logger.debug(f"Unknown or already completed span: {span_id}")
                return

            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            span.status = status
            span.output_data = output_data
            span.error_message = error_message
            span.metadata.update(metadata)

            self._completed_spans.append(span)

            if span.trace_type == TraceType.LLM_CALL:
                self._metrics["total_llm_calls"] += 1
                self._metrics["total_tokens"] += span.metadata.get("tokens_total", 0)
            elif span.trace_type == TraceType.DECISION_TRACE:
                self._metrics["total_decisions"] += 1

            # Report to dashboard (fire and forget)
            task = asyncio.create_task(self.dashboard_reporter.report_span(span))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

            logger.debug(f"Ended span {span_id} ({span.duration_ms:.1f}ms)")

        except Exception as e:
            logger.error(f"Failed to end span {span_id}: {e}")
            self._active_spans.pop(span_id, None)
    
    def record_decision(
        self,
        span_id: str,
        confidence: float = 0.0,
        reasoning: Optional[List[str]] = None,
        alternatives: Optional[List[str]] = None,
        **factors
    ) -> None:
        """Record decision metadata for a span."""
        try:
            span = self._active_spans.get(span_id)
            if span:
                span.metadata.update({
                    "confidence_score": confidence,
                    "reasoning_chain": reasoning or [],
                    "alternatives": alternatives or [],
                    "decision_factors": factors,
                })
                logger.debug(f"Recorded decision for span {span_id} (confidence: {confidence:.2f})")
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
        **llm_metadata
    ) -> None:
        """Record LLM call metadata for a span."""
        try:
            span = self._active_spans.get(span_id)
            if span:
                span.metadata.update({
                    "model": model,
                    "tokens_prompt": prompt_tokens,
                    "tokens_completion": completion_tokens,
                    "tokens_total": total_tokens or (prompt_tokens + completion_tokens),
                    **llm_metadata,
                })
                logger.debug(f"Recorded LLM call for span {span_id} ({total_tokens} tokens)")
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
        **metadata
    ):
        """Context manager for automatic span lifecycle."""
        span_id = self.start_span(
            operation_name=operation_name,
            trace_type=trace_type,
            agent_id=agent_id,
            **metadata
        )

        try:
            active = self._active_spans.get(span_id)
            if active:
                async with self.trace_context.span_context(active.trace_id, span_id, agent_id):
                    yield span_id
            else:
                yield span_id

            self.end_span(span_id, TraceStatus.SUCCESS)

        except Exception as e:
            self.end_span(span_id, TraceStatus.ERROR, error_message=str(e))
            raise
    
    # Convenience methods for specific trace types
    
    async def decision_span(self, decision_point: str, agent_id: Optional[str] = None, **metadata):
        """Context manager for decision tracing."""
        metadata.update({
            "decision_point": decision_point,
            "trace_subtype": "decision"
        })
        return self.span(f"decision_{decision_point}", TraceType.DECISION_TRACE, agent_id, **metadata)
    
    async def tool_span(self, tool_name: str, operation: str, agent_id: Optional[str] = None, **metadata):
        """Context manager for tool execution tracing."""
        metadata.update({
            "tool_name": tool_name,
            "tool_operation": operation
        })
        return self.span(f"tool_{tool_name}_{operation}", TraceType.TOOL_EXECUTION, agent_id, **metadata)
    
    # Query methods
    
    def get_recent_operations(self, agent_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent completed operations, most recent first."""
        try:
            spans = list(self._completed_spans)
            if agent_id:
                spans = [s for s in spans if s.agent_id == agent_id]
            return [s.to_dict() for s in reversed(spans[-limit:])]
        except Exception as e:
            logger.error(f"Error getting recent operations: {e}")
            return []

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global tracing metrics."""
        return {
            **self._metrics,
            "active_spans": len(self._active_spans),
            "dashboard_reports_sent": self.dashboard_reporter.reports_sent,
            "dashboard_reports_failed": self.dashboard_reporter.reports_failed,
        }

    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent."""
        try:
            spans = [s for s in self._completed_spans if s.agent_id == agent_id]
            if not spans:
                return {"total_operations": 0, "success_rate": 0}

            total_ops = len(spans)
            successful_ops = sum(1 for s in spans if s.status == TraceStatus.SUCCESS)
            latencies = [s.duration_ms for s in spans if s.duration_ms]
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

    def get_workflow_communications(self, workflow_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get workflow communication traces (agent-to-agent messages)."""
        try:
            comm_spans = [
                s for s in self._completed_spans
                if s.trace_type == TraceType.WORKFLOW_COMMUNICATION
                and (workflow_name is None or s.metadata.get('workflow_name') == workflow_name)
            ]
            result = []
            for span in reversed(comm_spans[-limit:]):
                comm_dict = span.to_dict()
                comm_dict['from_agent'] = span.metadata.get('from_agent', 'unknown')
                comm_dict['to_agent'] = span.metadata.get('to_agent', 'unknown')
                comm_dict['channel'] = span.metadata.get('channel', 'unknown')
                comm_dict['message_id'] = span.metadata.get('message_id')
                comm_dict['success'] = span.status == TraceStatus.SUCCESS
                result.append(comm_dict)
            return result
        except Exception as e:
            logger.error(f"Error getting workflow communications: {e}")
            return []

    def get_workflow_metrics(self, workflow_name: str) -> Dict[str, Any]:
        """Get metrics for a specific workflow."""
        try:
            comm_spans = [
                s for s in self._completed_spans
                if s.trace_type == TraceType.WORKFLOW_COMMUNICATION
                and s.metadata.get('workflow_name') == workflow_name
            ]
            if not comm_spans:
                return {"total_messages": 0, "success_rate": 0}

            total = len(comm_spans)
            successful = sum(1 for s in comm_spans if s.status == TraceStatus.SUCCESS)
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
    
    # Streaming decision events support
    
    def register_decision_stream_callback(self, agent_id: str, callback: Callable) -> None:
        """Register a callback for streaming decision events for a specific agent."""
        self._decision_stream_callbacks.setdefault(agent_id, []).append(callback)
        logger.debug(f"Registered decision stream callback for agent {agent_id}")

    def unregister_decision_stream_callback(self, agent_id: str, callback: Callable) -> None:
        """Unregister a decision stream callback for a specific agent."""
        callbacks = self._decision_stream_callbacks.get(agent_id)
        if callbacks:
            try:
                callbacks.remove(callback)
            except ValueError:
                pass
            if not callbacks:
                del self._decision_stream_callbacks[agent_id]
        logger.debug(f"Unregistered decision stream callback for agent {agent_id}")

    def emit_decision_event(self, agent_id: Optional[str], decision_event: 'DecisionEvent') -> None:
        """Emit a decision event to all registered callbacks for the agent."""
        if not agent_id:
            return
        # Copy list before iterating so callback removal during emit is safe
        for callback in list(self._decision_stream_callbacks.get(agent_id, [])):
            try:
                callback(decision_event)
            except Exception as e:
                logger.warning(f"Decision stream callback failed for agent {agent_id}: {e}")

    def get_streaming_agents(self) -> List[str]:
        """Get list of agents that have streaming callbacks registered."""
        return list(self._decision_stream_callbacks.keys())

# Global singleton — initialized once on first access.
_global_trace_manager: Optional[TraceManager] = None


def get_trace_manager() -> TraceManager:
    """Get the global TraceManager instance."""
    global _global_trace_manager
    if _global_trace_manager is None:
        _global_trace_manager = TraceManager()
    return _global_trace_manager

# Legacy compatibility functions (preserved for backward compatibility)
def record_tokens(agent_id: str, total_tokens: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0):
    """Legacy token recording - now handled automatically by LLM tracing."""
    pass

def get_agent_tokens(agent_id: str) -> Dict[str, int]:
    """Legacy token retrieval."""
    metrics = get_trace_manager().get_agent_metrics(agent_id)
    return {
        "total_tokens": 0,  # Legacy format not supported in simplified version
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "requests": metrics.get("total_operations", 0)
    }

def record_operation(agent_id: str, agent_name: str, task: str, input_data: Any, 
                    output_data: Any, latency_ms: float, status: str = "success", **kwargs) -> str:
    """Legacy operation recording."""
    trace_manager = get_trace_manager()
    
    span_id = trace_manager.start_span(
        operation_name=task,
        trace_type=TraceType.AGENT_EXECUTION,
        agent_id=agent_id,
        input_data=input_data,
        agent_name=agent_name
    )
    
    trace_status = TraceStatus.SUCCESS if status == "success" else TraceStatus.ERROR
    trace_manager.end_span(
        span_id=span_id,
        status=trace_status,
        output_data=output_data,
        error_message=kwargs.get("error_message")
    )
    
    return span_id

def get_recent_operations(agent_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Legacy function to get recent operations."""
    return get_trace_manager().get_recent_operations(agent_id, limit)