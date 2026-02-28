"""
Unified TraceManager for Daita Agents - Fixed MVP Version

Streamlined automatic tracing system that captures all agent operations,
LLM calls, workflow communication, and tool usage. Zero configuration required.

FIXED ISSUES:
- Added missing methods (record_decision, record_llm_call)
- Improved error handling with proper logging
- Fixed dependency management for aiohttp
- Added thread safety for concurrent access
- Improved context management to prevent leaks
- Added proper configuration validation
"""

import asyncio
import logging
import time
import uuid
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from collections import deque
from enum import Enum
from contextlib import asynccontextmanager
import threading

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

class TraceContext:
    """Thread-local trace context for automatic correlation."""
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def current_trace_id(self) -> Optional[str]:
        return getattr(self._local, 'trace_id', None)
    
    @property
    def current_span_id(self) -> Optional[str]:
        return getattr(self._local, 'span_id', None)
    
    @property
    def current_agent_id(self) -> Optional[str]:
        return getattr(self._local, 'agent_id', None)
    
    def set_context(self, trace_id: str, span_id: str, agent_id: Optional[str] = None):
        self._local.trace_id = trace_id
        self._local.span_id = span_id
        if agent_id:
            self._local.agent_id = agent_id
    
    def clear_context(self):
        self._local.trace_id = None
        self._local.span_id = None
        self._local.agent_id = None
    
    @asynccontextmanager
    async def span_context(self, trace_id: str, span_id: str, agent_id: Optional[str] = None):
        """Context manager for automatic span context management."""
        old_trace_id = self.current_trace_id
        old_span_id = self.current_span_id
        old_agent_id = self.current_agent_id
        
        try:
            self.set_context(trace_id, span_id, agent_id)
            yield
        finally:
            if old_trace_id:
                self.set_context(old_trace_id, old_span_id, old_agent_id)
            else:
                self.clear_context()

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
    Fixed TraceManager for MVP - automatic tracing with proper error handling.
    """
    
    def __init__(self):
        self.trace_context = TraceContext()
        self.dashboard_reporter = DashboardReporter()
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._active_spans: Dict[str, TraceSpan] = {}
        self._completed_spans: deque = deque(maxlen=500)
        
        # Basic metrics
        self._metrics = {
            "total_spans": 0,
            "total_llm_calls": 0,
            "total_tokens": 0,
            "total_decisions": 0
        }
        
        # Streaming decision events support
        self._decision_stream_callbacks: Dict[str, List[callable]] = {}

        logger.info("TraceManager initialized")
    
    def start_span(
        self,
        operation_name: str,
        trace_type: TraceType,
        agent_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        **metadata
    ) -> str:
        """Start a new trace span with thread safety."""
        try:
            span_id = str(uuid.uuid4())
            
            # Determine trace_id with context fallback
            if parent_span_id:
                with self._lock:
                    parent_span = self._active_spans.get(parent_span_id)
                    trace_id = parent_span.trace_id if parent_span else str(uuid.uuid4())
            elif self.trace_context.current_trace_id:
                trace_id = self.trace_context.current_trace_id
                parent_span_id = self.trace_context.current_span_id
            else:
                trace_id = str(uuid.uuid4())
            
            # Use agent from context if not provided
            if not agent_id:
                agent_id = self.trace_context.current_agent_id
            
            # Create span
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
            
            with self._lock:
                self._active_spans[span_id] = span
                self._metrics["total_spans"] += 1
            
            logger.debug(f"Started span {span_id} for '{operation_name}'")
            return span_id
            
        except Exception as e:
            logger.error(f"Failed to start span: {e}")
            # Return a valid span ID so operations don't break
            return f"error_{uuid.uuid4().hex[:8]}"
    
    def end_span(
        self,
        span_id: str,
        status: TraceStatus = TraceStatus.SUCCESS,
        output_data: Any = None,
        error_message: Optional[str] = None,
        **metadata
    ) -> None:
        """End a trace span with thread safety."""
        try:
            with self._lock:
                if span_id not in self._active_spans:
                    logger.debug(f"Unknown or already completed span: {span_id}")
                    return
                
                span = self._active_spans[span_id]
                
                # Update span
                span.end_time = time.time()
                span.duration_ms = (span.end_time - span.start_time) * 1000
                span.status = status
                span.output_data = output_data
                span.error_message = error_message
                span.metadata.update(metadata)
                
                # Move to completed
                self._completed_spans.append(span)
                del self._active_spans[span_id]
                
                # Update metrics
                if span.trace_type == TraceType.LLM_CALL:
                    self._metrics["total_llm_calls"] += 1
                    if "tokens_total" in span.metadata:
                        self._metrics["total_tokens"] += span.metadata.get("tokens_total", 0)
                elif span.trace_type == TraceType.DECISION_TRACE:
                    self._metrics["total_decisions"] += 1
            
            # Report to dashboard (fire and forget)
            task = asyncio.create_task(self.dashboard_reporter.report_span(span))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            
            logger.debug(f"Ended span {span_id} ({span.duration_ms:.1f}ms)")
            
        except Exception as e:
            logger.error(f"Failed to end span {span_id}: {e}")
            # Clean up active span even if there's an error
            with self._lock:
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
            with self._lock:
                span = self._active_spans.get(span_id)
                if span:
                    span.metadata.update({
                        "confidence_score": confidence,
                        "reasoning_chain": reasoning or [],
                        "alternatives": alternatives or [],
                        "decision_factors": factors
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
            with self._lock:
                span = self._active_spans.get(span_id)
                if span:
                    span.metadata.update({
                        "model": model,
                        "tokens_prompt": prompt_tokens,
                        "tokens_completion": completion_tokens,
                        "tokens_total": total_tokens or (prompt_tokens + completion_tokens),
                        **llm_metadata
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
            with self._lock:
                span = self._active_spans.get(span_id)
            
            if span:
                async with self.trace_context.span_context(span.trace_id, span_id, agent_id):
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
        """Get recent operations with thread safety."""
        try:
            with self._lock:
                spans = list(self._completed_spans)
            
            if agent_id:
                spans = [s for s in spans if s.agent_id == agent_id]
            
            # Most recent first
            spans = spans[-limit:]
            spans.reverse()
            
            return [span.to_dict() for span in spans]
        except Exception as e:
            logger.error(f"Error getting recent operations: {e}")
            return []
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global metrics with thread safety."""
        with self._lock:
            return {
                **self._metrics,
                "active_spans": len(self._active_spans),
                "dashboard_reports_sent": self.dashboard_reporter.reports_sent,
                "dashboard_reports_failed": self.dashboard_reporter.reports_failed
            }
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get basic metrics for an agent with thread safety."""
        try:
            with self._lock:
                spans = [s for s in self._completed_spans if s.agent_id == agent_id]

            if not spans:
                return {"total_operations": 0, "success_rate": 0}

            total_ops = len(spans)
            successful_ops = len([s for s in spans if s.status == TraceStatus.SUCCESS])

            # Average latency
            latencies = [s.duration_ms for s in spans if s.duration_ms]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            return {
                "total_operations": total_ops,
                "successful_operations": successful_ops,
                "failed_operations": total_ops - successful_ops,
                "success_rate": successful_ops / total_ops if total_ops > 0 else 0,
                "avg_latency_ms": avg_latency
            }
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return {"total_operations": 0, "success_rate": 0}

    def get_workflow_communications(self, workflow_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get workflow communication traces.

        Returns spans that represent workflow communications (agent-to-agent messages).
        """
        try:
            with self._lock:
                # Filter for workflow communication spans
                comm_spans = [
                    s for s in self._completed_spans
                    if s.trace_type == TraceType.WORKFLOW_COMMUNICATION
                ]

                # Filter by workflow name if provided
                if workflow_name:
                    comm_spans = [
                        s for s in comm_spans
                        if s.metadata.get('workflow_name') == workflow_name
                    ]

                # Most recent first
                comm_spans = comm_spans[-limit:]
                comm_spans.reverse()

                # Convert to dictionaries with workflow-specific fields
                result = []
                for span in comm_spans:
                    comm_dict = span.to_dict()
                    # Add workflow-specific fields from metadata
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
            with self._lock:
                # Get all communication spans for this workflow
                comm_spans = [
                    s for s in self._completed_spans
                    if s.trace_type == TraceType.WORKFLOW_COMMUNICATION
                    and s.metadata.get('workflow_name') == workflow_name
                ]

                if not comm_spans:
                    return {"total_messages": 0, "success_rate": 0}

                total = len(comm_spans)
                successful = len([s for s in comm_spans if s.status == TraceStatus.SUCCESS])

                return {
                    "workflow_name": workflow_name,
                    "total_messages": total,
                    "successful_messages": successful,
                    "failed_messages": total - successful,
                    "success_rate": successful / total if total > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error getting workflow metrics: {e}")
            return {"total_messages": 0, "success_rate": 0}
    
    # Streaming decision events support
    
    def register_decision_stream_callback(self, agent_id: str, callback: callable) -> None:
        """Register a callback for streaming decision events for a specific agent."""
        try:
            with self._lock:
                if agent_id not in self._decision_stream_callbacks:
                    self._decision_stream_callbacks[agent_id] = []
                self._decision_stream_callbacks[agent_id].append(callback)
            logger.debug(f"Registered decision stream callback for agent {agent_id}")
        except Exception as e:
            logger.error(f"Failed to register decision stream callback: {e}")
    
    def unregister_decision_stream_callback(self, agent_id: str, callback: callable) -> None:
        """Unregister a decision stream callback for a specific agent."""
        try:
            with self._lock:
                if agent_id in self._decision_stream_callbacks:
                    if callback in self._decision_stream_callbacks[agent_id]:
                        self._decision_stream_callbacks[agent_id].remove(callback)
                    if not self._decision_stream_callbacks[agent_id]:
                        del self._decision_stream_callbacks[agent_id]
            logger.debug(f"Unregistered decision stream callback for agent {agent_id}")
        except Exception as e:
            logger.error(f"Failed to unregister decision stream callback: {e}")
    
    def emit_decision_event(self, agent_id: Optional[str], decision_event: 'DecisionEvent') -> None:
        """Emit a decision event to all registered callbacks for the agent."""
        if not agent_id:
            return
            
        try:
            with self._lock:
                callbacks = self._decision_stream_callbacks.get(agent_id, [])
            
            # Call each callback (don't hold the lock during callback execution)
            for callback in callbacks:
                try:
                    callback(decision_event)
                except Exception as e:
                    logger.warning(f"Decision stream callback failed for agent {agent_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to emit decision event: {e}")
    
    def get_streaming_agents(self) -> List[str]:
        """Get list of agents that have streaming callbacks registered."""
        with self._lock:
            return list(self._decision_stream_callbacks.keys())

# Global instance with safer initialization
_global_trace_manager: Optional[TraceManager] = None
_manager_lock = threading.Lock()

def get_trace_manager() -> TraceManager:
    """Get the global trace manager instance with thread safety."""
    global _global_trace_manager
    if _global_trace_manager is None:
        with _manager_lock:
            if _global_trace_manager is None:  # Double-check pattern
                try:
                    _global_trace_manager = TraceManager()
                    logger.info("TraceManager successfully initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize TraceManager: {e}")
                    # Create a no-op manager that doesn't break but logs the issue
                    _global_trace_manager = _create_safe_noop_manager()
    return _global_trace_manager

def _create_safe_noop_manager():
    """Create a safe no-op manager that logs issues but doesn't break."""
    logger.warning("Using no-op TraceManager due to initialization failure")
    
    class SafeNoOpTraceManager:
        def __init__(self):
            self.dashboard_reporter = type('obj', (object,), {
                'enabled': False, 
                'reports_sent': 0, 
                'reports_failed': 0
            })()
        
        def start_span(self, *args, **kwargs):
            return f"noop_{uuid.uuid4().hex[:8]}"
        
        def end_span(self, *args, **kwargs):
            pass
        
        def record_llm_call(self, *args, **kwargs):
            pass
        
        def record_decision(self, *args, **kwargs):
            pass
        
        @asynccontextmanager
        async def span(self, *args, **kwargs):
            yield f"noop_{uuid.uuid4().hex[:8]}"
        
        async def decision_span(self, *args, **kwargs):
            return self.span(*args, **kwargs)
        
        async def tool_span(self, *args, **kwargs):
            return self.span(*args, **kwargs)
        
        def get_recent_operations(self, *args, **kwargs):
            return []
        
        def get_global_metrics(self):
            return {"total_spans": 0, "total_llm_calls": 0, "total_tokens": 0}
        
        def get_agent_metrics(self, *args, **kwargs):
            return {"total_operations": 0, "success_rate": 0}
    
    return SafeNoOpTraceManager()

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