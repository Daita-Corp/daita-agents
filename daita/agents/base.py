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
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..config.base import AgentConfig
from ..core.interfaces import AgentABC, LLMProvider

from ..core.tracing import get_trace_manager, TraceType
from ..core.decision_tracing import record_decision_point, DecisionType
from ..core.reliability import (
    get_global_task_manager,
    TaskStatus,
    BackpressureController,
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
            slug = self.name.lower().replace(" ", "_").replace("-", "_")
            self.agent_id = f"{slug}_{uuid.uuid4().hex[:8]}"
        else:
            self.agent_id = f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

        # Runtime state
        self._running = False
        self._tasks = []

        # Watch system state
        self._watches: list = []
        self._watch_states: dict = {}
        self._watches_started: bool = False
        self._watches_lock: asyncio.Lock = asyncio.Lock()

        # Get trace manager for automatic tracing
        self.trace_manager = get_trace_manager()

        # Reliability features (enabled when reliability is configured)
        self.task_manager = get_global_task_manager() if enable_reliability else None
        self.backpressure_controller = None
        if enable_reliability:
            self.backpressure_controller = BackpressureController(
                max_concurrent_tasks=max_concurrent_tasks,
                max_queue_size=max_queue_size,
                agent_id=self.agent_id,
            )

        # Set agent ID in LLM provider for automatic LLM tracing
        if self.llm:
            self.llm.set_agent_id(self.agent_id)

        logger.debug(
            f"Agent {self.name} ({self.agent_id}) initialized with automatic tracing"
        )

    async def start(self) -> None:
        """Start the agent with automatic lifecycle tracing."""
        if self._running:
            return

        # Start decision display if enabled
        if hasattr(self, "_decision_display") and self._decision_display:
            self._decision_display.start()

        # Automatically trace agent lifecycle
        async with self.trace_manager.span(
            operation_name="agent_start",
            trace_type=TraceType.AGENT_LIFECYCLE,
            agent_id=self.agent_id,
            agent_name=self.name,
            agent_type=self.agent_type.value,
            retry_enabled=str(self.config.retry_enabled),
        ):
            self._running = True
            logger.info(f"Agent {self.name} started")

        # Start watches after _running is True so tasks can check agent state
        await self._start_watches()

    async def stop(self) -> None:
        """Stop the agent with automatic lifecycle tracing."""
        if not self._running:
            return

        # Stop decision display if enabled
        if hasattr(self, "_decision_display") and self._decision_display:
            self._decision_display.stop()
            # Cleanup decision streaming registration
            try:
                from ..core.decision_tracing import unregister_agent_decision_stream

                unregister_agent_decision_stream(
                    agent_id=self.agent_id, callback=self._decision_display.handle_event
                )
            except Exception as e:
                logger.debug(f"Failed to cleanup decision display: {e}")

        # Automatically trace agent lifecycle
        async with self.trace_manager.span(
            operation_name="agent_stop",
            trace_type=TraceType.AGENT_LIFECYCLE,
            agent_id=self.agent_id,
            agent_name=self.name,
            tasks_completed=str(len(self._tasks)),
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

    # ========================================================================
    # WATCH SYSTEM — polling/streaming data source monitoring
    # ========================================================================

    def watch(
        self,
        source: Any = None,
        *,
        condition: Any = None,
        threshold: Optional[Callable[[Any], bool]] = None,
        interval: Optional[Union[str, timedelta]] = None,
        on_resolve: bool = False,
        topic: Optional[str] = None,
        filter: Optional[Callable[[Any], bool]] = None,
        relay_channel: Optional[str] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
        handler_timeout: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Callable:
        """Decorator: register an async handler as a watch on a data source.

        Polling example::

            @agent.watch(source=pg,
                         condition="SELECT COUNT(*) FROM orders",
                         threshold=lambda v: v > 100,
                         interval="10s")
            async def on_spike(event: WatchEvent):
                print(f"Order spike: {event.value}")

        Watches start automatically when ``agent.start()`` is called, or lazily
        on the first ``run()`` / ``stream()`` invocation.

        Args:
            source: Plugin or WatchSource to poll/subscribe.
            condition: SQL string or async callable returning the watched value.
            threshold: Called with the current value; watch fires when it returns True.
            interval: Poll period — timedelta or string like "30s", "5m", "1h", "2d".
            on_resolve: If True, fire again when the condition clears.
            topic: Streaming topic (Phase 2).
            filter: Message filter for streaming sources (Phase 2).
            relay_channel: Relay channel to publish events to (Phase 3).
            on_error: Called when the handler raises; receives the exception.
            name: Override the watch name (defaults to handler.__name__).

        Raises:
            ValueError: If the configuration is invalid.
        """
        from ..core.watch import _parse_interval, WatchConfig, PollingWatchSource

        if on_resolve and interval is None and topic is None:
            raise ValueError(
                "on_resolve=True requires either interval= (polling) or topic= (streaming)"
            )

        if interval is not None and condition is None and topic is None:
            raise ValueError("interval= requires condition= to specify what to poll")

        if isinstance(interval, str):
            interval = _parse_interval(interval)

        def decorator(handler: Callable) -> Callable:
            watch_name = name or handler.__name__

            if self._watches_started:
                raise RuntimeError(
                    f"Cannot register watch '{watch_name}' after the agent has started. "
                    "Register all watches before calling run() or start()."
                )

            if interval is not None:
                watch_source = PollingWatchSource(
                    plugin=source,
                    condition=condition,
                    interval=interval,
                )
            else:
                watch_source = source

            config = WatchConfig(
                handler=handler,
                source=watch_source,
                name=watch_name,
                condition=condition,
                threshold=threshold,
                interval=interval,
                on_resolve=on_resolve,
                topic=topic,
                filter=filter,
                relay_channel=relay_channel,
                on_error=on_error,
                handler_timeout=handler_timeout,
            )
            self._watches.append(config)
            logger.debug(f"Registered watch '{watch_name}' on agent {self.name}")
            return handler

        return decorator

    async def _start_watches(self) -> None:
        """Start all registered watches. Idempotent, concurrency-safe.

        Called from start() and as a lazy fallback from Agent._run_traced().
        The fast path (no lock) handles the common single-caller case; the
        lock + double-check guards concurrent run() calls when reliability
        is enabled (task_manager introduces yield points in _start_watch).
        """
        if self._watches_started or not self._watches:
            return
        async with self._watches_lock:
            if self._watches_started:  # re-check after acquiring
                return
            for config in self._watches:
                await self._start_watch(config)
            self._watches_started = True

    async def _start_watch(self, config: Any) -> None:
        """Create and register a background task for a single watch."""
        self._tasks = [t for t in self._tasks if not t.done()]
        task = asyncio.create_task(
            self._watch_loop(config), name=f"watch:{config.name}"
        )
        self._tasks.append(task)
        if self.task_manager:
            try:
                await self.task_manager.create_task(
                    agent_id=self.agent_id,
                    task_type="watch",
                    data={"watch_name": config.name},
                )
            except Exception as e:
                logger.debug(
                    f"task_manager.create_task failed for watch '{config.name}': {e}"
                )

    async def _watch_loop(self, config: Any) -> None:
        """Core watch runtime: polls/streams the source, fires the handler on trigger."""
        from ..core.watch import WatchState

        state = WatchState(config=config, task=asyncio.current_task(), status="running")
        self._watch_states[config.name] = state

        try:
            async for raw_value in config.source.events():
                try:
                    if self._should_trigger(config, state, raw_value):
                        event = self._build_watch_event(config, state, raw_value)
                        state.triggered = True
                        state._previous_value = raw_value
                        await self._invoke_handler(config.handler, event, config)
                    elif config.on_resolve and state.triggered:
                        state.triggered = False
                        event = self._build_watch_event(
                            config, state, raw_value, resolved=True
                        )
                        state._previous_value = raw_value
                        await self._invoke_handler(config.handler, event, config)
                    else:
                        state._previous_value = raw_value
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    state.last_error = e
                    logger.error(f"Watch '{config.name}' iteration error: {e}")
        except asyncio.CancelledError:
            state.status = "stopped"
        except Exception as e:
            state.status = "error"
            state.last_error = e
            logger.error(
                f"Watch '{config.name}' failed permanently: {type(e).__name__}: {e}",
                exc_info=True,
            )

    def _should_trigger(self, config: Any, state: Any, raw_value: Any) -> bool:
        """Determine whether this value should fire the handler."""
        if config.filter is not None:
            return config.filter(raw_value)
        if config.threshold is None:
            return True
        return config.threshold(raw_value)

    def _build_watch_event(
        self, config: Any, state: Any, raw_value: Any, resolved: bool = False
    ) -> Any:
        """Construct a WatchEvent from the current state."""
        from ..core.watch import WatchEvent

        return WatchEvent(
            value=raw_value,
            triggered_at=datetime.now(timezone.utc),
            source_type="polling" if config.interval else "streaming",
            resolved=resolved,
            previous_value=state._previous_value,
        )

    async def _invoke_handler(self, handler: Callable, event: Any, config: Any) -> None:
        """Call the watch handler, isolating exceptions from the watch loop."""
        try:
            if config.handler_timeout is not None:
                await asyncio.wait_for(handler(event), timeout=config.handler_timeout)
            else:
                await handler(event)
        except asyncio.TimeoutError:
            msg = f"Watch handler '{config.name}' timed out after {config.handler_timeout}s"
            logger.error(msg)
            if config.on_error:
                try:
                    await config.on_error(TimeoutError(msg))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Watch handler '{config.name}' raised: {e}")
            if config.on_error:
                try:
                    await config.on_error(e)
                except Exception:
                    pass

    async def _process(
        self,
        task: str,
        data: Any = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        INTERNAL: Override in subclasses to handle tasks.

        Agent overrides this to route through run(detailed=True).
        Custom BaseAgent subclasses can override this directly.
        Workflow and scaling infrastructure call this as a fallback
        when receive_message() is not available.
        """
        return {
            "result": f"Task received: {task}",
            "task": task,
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
                is_retry=str(attempt > 1),
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
                            delay = retry_policy.calculate_delay(attempt)
                            logger.debug(f"Agent {self.name} retrying in {delay:.2f}s")
                            await asyncio.sleep(delay)
                            continue
                    logger.debug(f"Agent {self.name} not retrying: {type(e).__name__}")
                    raise

        raise last_exception or Exception("Unknown error in retry loop")

    async def _should_retry_error_with_tracing(
        self, error: Exception, attempt: int, max_attempts: int, span_id: str
    ) -> bool:
        """
        Determine if an error should be retried with decision tracing.

        This traces the retry decision-making process including confidence
        scores and reasoning for better observability.
        """
        # Use decision tracing to record retry logic
        async with record_decision_point(
            "retry_decision", DecisionType.VALIDATION, self.agent_id
        ) as decision:

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

            logger.debug(
                f"Retry decision for {error_type}: {should_retry} (confidence: {confidence:.2f})"
            )
            return should_retry

    def watch_status(self) -> Dict[str, Any]:
        """Return the current status of all registered watches.

        Useful for health checks, dashboards, and debugging in production.
        Returns an empty dict if no watches are registered.
        """
        return {
            name: {
                "status": state.status,
                "triggered": state.triggered,
                "last_error": str(state.last_error) if state.last_error else None,
            }
            for name, state in self._watch_states.items()
        }

    @property
    def health(self) -> Dict[str, Any]:
        """Get agent health information from unified tracing system."""
        # Get real-time metrics from trace manager
        metrics = self.trace_manager.get_agent_metrics(self.agent_id)

        return {
            "id": self.agent_id,
            "name": self.name,
            "type": self.agent_type.value,
            "running": self._running,
            "metrics": metrics,
            "watches": self.watch_status(),
            "retry_config": {
                "enabled": self.config.retry_enabled,
                "max_retries": (
                    self.config.retry_policy.max_retries
                    if self.config.retry_enabled
                    else None
                ),
                "strategy": (
                    self.config.retry_policy.strategy.value
                    if self.config.retry_enabled
                    else None
                ),
            },
            "tracing": {
                "enabled": True,
                "trace_manager_available": self.trace_manager is not None,
            },
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
        return self.trace_manager.get_recent_operations(
            agent_id=self.agent_id, limit=limit
        )

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
        self, max_concurrent_tasks: int = 10, max_queue_size: int = 100
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
            agent_id=self.agent_id,
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

    async def get_agent_tasks(
        self, status: Optional[TaskStatus] = None
    ) -> List[Dict[str, Any]]:
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
                "retry_count": task.retry_count,
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

    def __repr__(self) -> str:
        return f"BaseAgent(name='{self.name}', id='{self.agent_id}', running={self._running})"

    def __str__(self) -> str:
        return f"BaseAgent '{self.name}'"
