"""
BaseAgent — Infrastructure base class for all Daita agents.

Provides the shared foundation that Agent and any custom agent subclass
builds on. Does not contain LLM or tool-calling logic; that lives in Agent.

Responsibilities:
- Agent identity: ID generation, name, lifecycle state
- Automatic tracing: every operation is traced via TraceManager
- Retry infrastructure: _retry_with_tracing() shared by all subclasses
- Reliability features: opt-in backpressure and task queue management
- Decision tracing: retry decisions recorded with confidence scores
- Health and metrics: real-time stats from the trace system
"""

import asyncio
import logging
import uuid
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config.base import AgentConfig, RetryStrategy
from ..core.interfaces import AgentABC, LLMProvider

from ..core.tracing import get_trace_manager, TraceType
from ..core.decision_tracing import record_decision_point, DecisionType
from ..core.reliability import (
    get_global_task_manager, TaskStatus,
    BackpressureController
)

logger = logging.getLogger(__name__)

class BaseAgent(AgentABC):
    """
    Base implementation for all Daita agents with automatic tracing.
    
    Every operation is automatically traced and sent to the dashboard.
    Users don't need to configure anything - tracing just works.
    
    Features:
    - Automatic operation tracing
    - Retry decision tracing with confidence scores
    - Agent lifecycle tracing
    - LLM integration with automatic token tracking
    - Performance monitoring
    - Error tracking and correlation
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_provider: Optional[LLMProvider] = None,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        enable_reliability: bool = False,
        max_concurrent_tasks: int = 10,
        max_queue_size: int = 100,
    ):
        self.config = config
        self.llm = llm_provider
        self.name = name or config.name
        self.agent_type = config.type
        self.enable_reliability = enable_reliability
        
        # Generate unique ID
        if agent_id:
            self.agent_id = agent_id
        elif self.name:
            slug = self.name.lower().replace(' ', '_').replace('-', '_')
            self.agent_id = f"{slug}_{uuid.uuid4().hex[:8]}"
        else:
            self.agent_id = f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        
        # Runtime state
        self._running = False
        self._tasks = []
        
        # Get trace manager for automatic tracing
        self.trace_manager = get_trace_manager()
        
        # Reliability features (enabled when reliability is configured)
        self.task_manager = get_global_task_manager() if enable_reliability else None
        self.backpressure_controller = None
        if enable_reliability:
            self.backpressure_controller = BackpressureController(
                max_concurrent_tasks=max_concurrent_tasks,
                max_queue_size=max_queue_size,
                agent_id=self.agent_id
            )
        
        # Set agent ID in LLM provider for automatic LLM tracing
        if self.llm:
            self.llm.set_agent_id(self.agent_id)
        
        logger.debug(f"Agent {self.name} ({self.agent_id}) initialized with automatic tracing")
    
    async def start(self) -> None:
        """Start the agent with automatic lifecycle tracing."""
        if self._running:
            return
        
        # Start decision display if enabled
        if hasattr(self, '_decision_display') and self._decision_display:
            self._decision_display.start()
        
        # Automatically trace agent lifecycle
        async with self.trace_manager.span(
            operation_name="agent_start",
            trace_type=TraceType.AGENT_LIFECYCLE,
            agent_id=self.agent_id,
            agent_name=self.name,
            agent_type=self.agent_type.value,
            retry_enabled=str(self.config.retry_enabled)
        ):
            self._running = True
            logger.info(f"Agent {self.name} started")
    
    async def stop(self) -> None:
        """Stop the agent with automatic lifecycle tracing."""
        if not self._running:
            return
        
        # Stop decision display if enabled
        if hasattr(self, '_decision_display') and self._decision_display:
            self._decision_display.stop()
            # Cleanup decision streaming registration
            try:
                from ..core.decision_tracing import unregister_agent_decision_stream
                unregister_agent_decision_stream(
                    agent_id=self.agent_id,
                    callback=self._decision_display.handle_event
                )
            except Exception as e:
                logger.debug(f"Failed to cleanup decision display: {e}")
        
        # Automatically trace agent lifecycle
        async with self.trace_manager.span(
            operation_name="agent_stop",
            trace_type=TraceType.AGENT_LIFECYCLE,
            agent_id=self.agent_id,
            agent_name=self.name,
            tasks_completed=str(len(self._tasks))
        ):
            # Cancel running tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            
            self._running = False
            logger.info(f"Agent {self.name} stopped")
    
    async def _process(
        self,
        task: str,
        data: Any = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        INTERNAL: Override in subclasses to handle tasks.

        Agent overrides this to route through run_detailed().
        Custom BaseAgent subclasses can override this directly.
        Workflow and scaling infrastructure call this as a fallback
        when receive_message() is not available.
        """
        return {
            'result': f'Task received: {task}',
            'task': task,
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def _retry_with_tracing(
        self,
        execute_fn: Any,
        span_name_prefix: str,
        parent_span_id: Optional[str] = None,
    ) -> Any:
        """
        Shared retry scaffold used by both _process_with_retry and Agent._execute_autonomous_with_retry.

        Runs execute_fn(attempt, max_attempts) up to max_retries+1 times, recording
        each attempt as a child trace span with decision tracing on failure.
        Raises the last exception on final failure.

        Args:
            execute_fn: Async callable accepting (attempt: int, max_attempts: int).
            span_name_prefix: Prefix for attempt span names.
            parent_span_id: Optional parent span ID for trace hierarchy.
        """
        retry_policy = self.config.retry_policy
        max_attempts = retry_policy.max_retries + 1
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            async with self.trace_manager.span(
                operation_name=f"{span_name_prefix}_{attempt}",
                trace_type=TraceType.AGENT_EXECUTION,
                agent_id=self.agent_id,
                parent_span_id=parent_span_id,
                attempt=str(attempt),
                max_attempts=str(max_attempts),
                is_retry=str(attempt > 1)
            ) as attempt_span_id:
                try:
                    result = await execute_fn(attempt, max_attempts)
                    if attempt > 1:
                        logger.info(f"Agent {self.name} succeeded on attempt {attempt}")
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        should_retry = await self._should_retry_error_with_tracing(
                            e, attempt, max_attempts, attempt_span_id
                        )
                        if should_retry:
                            delay = self._calculate_retry_delay(attempt - 1, retry_policy)
                            logger.debug(f"Agent {self.name} retrying in {delay:.2f}s")
                            await asyncio.sleep(delay)
                            continue
                    logger.debug(f"Agent {self.name} not retrying: {type(e).__name__}")
                    raise

        raise last_exception or Exception("Unknown error in retry loop")

    async def _should_retry_error_with_tracing(
        self, 
        error: Exception, 
        attempt: int, 
        max_attempts: int,
        span_id: str
    ) -> bool:
        """
        Determine if an error should be retried with decision tracing.
        
        This traces the retry decision-making process including confidence
        scores and reasoning for better observability.
        """
        # Use decision tracing to record retry logic
        async with record_decision_point("retry_decision", DecisionType.VALIDATION, self.agent_id) as decision:
            
            # Import here to avoid circular imports
            from ..core.exceptions import classify_exception
            
            # Classify the error
            error_class = classify_exception(error)
            error_type = type(error).__name__
            
            # Decision logic with reasoning
            reasoning = []
            should_retry = False
            confidence = 0.0
            
            # Check attempt limit
            if attempt >= max_attempts:
                reasoning.append(f"Max attempts reached ({attempt}/{max_attempts})")
                should_retry = False
                confidence = 1.0  # Certain we shouldn't retry
            
            # Error classification logic
            elif error_class == "transient":
                reasoning.append(f"Transient error detected: {error_type}")
                reasoning.append("Transient errors are typically safe to retry")
                should_retry = True
                confidence = 0.9
            
            elif error_class == "retryable":
                reasoning.append(f"Retryable error detected: {error_type}")
                reasoning.append("Error may resolve on retry")
                should_retry = True
                confidence = 0.7
            
            elif error_class == "permanent":
                reasoning.append(f"Permanent error detected: {error_type}")
                reasoning.append("Permanent errors should not be retried")
                should_retry = False
                confidence = 0.95
            
            else:
                # Unknown error - use heuristics
                reasoning.append(f"Unknown error type: {error_type}")
                
                if isinstance(error, (ValueError, TypeError, KeyError)):
                    reasoning.append("Logic/data error - likely permanent")
                    should_retry = False
                    confidence = 0.8
                else:
                    reasoning.append("Unknown error - defaulting to retry")
                    should_retry = True
                    confidence = 0.5
            
            # Record the decision
            decision.set_confidence(confidence)
            for reason in reasoning:
                decision.add_reasoning(reason)
            
            decision.set_factor("error_type", error_type)
            decision.set_factor("error_class", error_class)
            decision.set_factor("attempt", attempt)
            decision.set_factor("max_attempts", max_attempts)
            
            # Add alternatives considered
            decision.add_alternative("retry" if not should_retry else "fail")
            
            logger.debug(f"Retry decision for {error_type}: {should_retry} (confidence: {confidence:.2f})")
            return should_retry
    
    def _calculate_retry_delay(self, attempt: int, retry_policy) -> float:
        """Calculate retry delay with jitter."""
        if hasattr(retry_policy, 'calculate_delay'):
            # Use the RetryPolicy's built-in delay calculation
            return retry_policy.calculate_delay(attempt)
        
        # Legacy fallback for old-style retry policies
        if retry_policy.strategy in [RetryStrategy.IMMEDIATE, "immediate"]:
            delay = 0.0
        elif retry_policy.strategy in [RetryStrategy.FIXED, RetryStrategy.FIXED_DELAY, "fixed", "fixed_delay"]:
            delay = getattr(retry_policy, 'base_delay', getattr(retry_policy, 'initial_delay', 1.0))
        else:  # EXPONENTIAL (default)
            base_delay = getattr(retry_policy, 'base_delay', getattr(retry_policy, 'initial_delay', 1.0))
            delay = base_delay * (2 ** attempt)
        
        # Add small random jitter to prevent thundering herd
        jitter = delay * 0.1 * random.random()
        delay += jitter
        
        return delay
    
    def _format_success_response(
        self,
        result: Any,
        context: Dict[str, Any],
        attempt: int,
        max_attempts: int
    ) -> Dict[str, Any]:
        """Format successful response with tracing metadata (flattened for better DX)."""
        # Build response with framework metadata
        response = {
            'status': 'success',
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'context': context,
            'retry_info': {
                'attempt': attempt,
                'max_attempts': max_attempts,
                'retry_enabled': self.config.retry_enabled
            } if self.config.retry_enabled else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Flatten execution result into top level for better DX
        # If result is a dict, merge it; otherwise add as 'result' key
        if isinstance(result, dict):
            # Merge execution result at top level (result keys won't overwrite framework keys)
            response.update(result)
        else:
            # Non-dict results stored under 'result' key
            response['result'] = result

        return response
    
    def _format_error_response(
        self,
        error: Exception,
        context: Dict[str, Any],
        attempt: int,
        max_attempts: int
    ) -> Dict[str, Any]:
        """Format error response with tracing metadata."""
        return {
            'status': 'error',
            'error': str(error),
            'error_type': error.__class__.__name__,
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'context': context,
            'result': None,  # Ensure result field exists for relay compatibility
            'retry_info': {
                'attempt': attempt,
                'max_attempts': max_attempts,
                'retry_enabled': self.config.retry_enabled,
                'retry_exhausted': attempt >= max_attempts
            } if self.config.retry_enabled else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    @property
    def health(self) -> Dict[str, Any]:
        """Get agent health information from unified tracing system."""
        # Get real-time metrics from trace manager
        metrics = self.trace_manager.get_agent_metrics(self.agent_id)
        
        return {
            'id': self.agent_id,
            'name': self.name,
            'type': self.agent_type.value,
            'running': self._running,
            'metrics': metrics,
            'retry_config': {
                'enabled': self.config.retry_enabled,
                'max_retries': self.config.retry_policy.max_retries if self.config.retry_enabled else None,
                'strategy': self.config.retry_policy.strategy.value if self.config.retry_enabled else None,
            },
            'tracing': {
                'enabled': True,
                'trace_manager_available': self.trace_manager is not None
            }
        }
    
    @property
    def trace_id(self) -> Optional[str]:
        """Get current trace ID for this agent."""
        return self.trace_manager.trace_context.current_trace_id
    
    @property
    def current_span_id(self) -> Optional[str]:
        """Get current span ID for this agent."""
        return self.trace_manager.trace_context.current_span_id
    
    def get_recent_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent operations for this agent from unified tracing."""
        return self.trace_manager.get_recent_operations(agent_id=self.agent_id, limit=limit)
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get comprehensive tracing statistics for this agent."""
        return self.trace_manager.get_agent_metrics(self.agent_id)
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision traces for this agent."""
        from ..core.decision_tracing import get_recent_decisions
        return get_recent_decisions(agent_id=self.agent_id, limit=limit)
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics for this agent."""
        from ..core.decision_tracing import get_decision_stats
        return get_decision_stats(agent_id=self.agent_id)
    
    # Reliability management methods
    
    def enable_reliability_features(
        self,
        max_concurrent_tasks: int = 10,
        max_queue_size: int = 100
    ) -> None:
        """
        Enable reliability features for this agent.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            max_queue_size: Maximum queue size for backpressure control
        """
        if self.enable_reliability:
            logger.warning(f"Reliability already enabled for agent {self.name}")
            return
        
        self.enable_reliability = True
        self.task_manager = get_global_task_manager()
        self.backpressure_controller = BackpressureController(
            max_concurrent_tasks=max_concurrent_tasks,
            max_queue_size=max_queue_size,
            agent_id=self.agent_id
        )
        
        logger.info(f"Enabled reliability features for agent {self.name}")
    
    def disable_reliability_features(self) -> None:
        """Disable reliability features for this agent."""
        self.enable_reliability = False
        self.task_manager = None
        self.backpressure_controller = None
        
        logger.info(f"Disabled reliability features for agent {self.name}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if not self.task_manager:
            return None
        return await self.task_manager.get_task_status(task_id)
    
    async def get_agent_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """Get all tasks for this agent, optionally filtered by status."""
        if not self.task_manager:
            return []
        
        tasks = await self.task_manager.get_agent_tasks(self.agent_id, status)
        return [
            {
                "id": task.id,
                "status": task.status.value,
                "progress": task.progress,
                "error": task.error,
                "duration": task.duration(),
                "age": task.age(),
                "retry_count": task.retry_count
            }
            for task in tasks
        ]
    
    def get_backpressure_stats(self) -> Dict[str, Any]:
        """Get current backpressure statistics."""
        if not self.backpressure_controller:
            return {"enabled": False}
        
        stats = self.backpressure_controller.get_stats()
        stats["enabled"] = True
        return stats
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        if not self.task_manager:
            return False
        return await self.task_manager.cancel_task(task_id)
    
    # Integration helpers
    
    def create_child_agent(self, name: str, config_overrides: Optional[Dict[str, Any]] = None) -> "BaseAgent":
        """Create a child agent that inherits tracing context."""
        # Create new config based on current config
        from ..config.base import AgentConfig
        
        child_config = AgentConfig(
            name=name,
            type=self.config.type,
            enable_retry=self.config.enable_retry,
            retry_policy=self.config.retry_policy,
            **(config_overrides or {})
        )
        
        # Create child agent
        child = self.__class__(
            config=child_config,
            llm_provider=self.llm,
            name=name
        )
        
        logger.debug(f"Created child agent {name} from parent {self.name}")
        return child
    
    def __repr__(self) -> str:
        return f"BaseAgent(name='{self.name}', id='{self.agent_id}', running={self._running})"
    
    def __str__(self) -> str:
        return f"BaseAgent '{self.name}'"