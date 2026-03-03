"""
Simplified Workflow System for Daita Agents.

Provides orchestration of agents as connected systems with automatic tracing.
All workflow communication is automatically traced through the unified tracing system.

Example:
    ```python
    from daita import Agent, Workflow

    fetcher = Agent(name="Data Fetcher")
    analyzer = Agent(name="Analyzer")

    workflow = Workflow("Data Pipeline")
    workflow.add_agent("fetcher", fetcher)
    workflow.add_agent("analyzer", analyzer)
    workflow.connect("fetcher", "raw_data", "analyzer")

    await workflow.start()
    ```
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from ..core.exceptions import WorkflowError, BackpressureError
from ..core.relay import RelayManager, get_global_relay
from ..core.tracing import get_trace_manager, TraceType, TraceStatus

logger = logging.getLogger(__name__)

class WorkflowStatus(str, Enum):
    """Status of a workflow."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ReliabilityConfig:
    """Configuration for workflow reliability features."""
    acknowledgments: bool = True
    task_tracking: bool = True
    backpressure_control: bool = True

@dataclass
class Connection:
    """Represents a connection between agents via a relay channel."""
    from_agent: str
    channel: str
    to_agent: str
    task: str = "relay_message"
    
    def __str__(self):
        return f"{self.from_agent} -> {self.channel} -> {self.to_agent}"

class Workflow:
    """
    A workflow manages a collection of agents and their connections.
    
    All workflow communication is automatically traced through the unified
    tracing system without any configuration required.
    """
    
    def __init__(
        self,
        name: str,
        project_id: Optional[str] = None,
        relay_manager: Optional[RelayManager] = None
    ):
        """
        Initialize a workflow.
        
        Args:
            name: Workflow name
            project_id: Optional project ID this workflow belongs to
            relay_manager: Relay manager for agent communication
        """
        self.name = name
        self.project_id = project_id
        self.relay_manager = relay_manager or get_global_relay()
        
        # Agent storage: agent_name -> agent_instance
        self.agents: Dict[str, Any] = {}
        
        # Agent pools: pool_name -> AgentPool instance (for horizontal scaling)
        self.agent_pools: Dict[str, Any] = {}
        
        # Connections: list of Connection objects
        self.connections: List[Connection] = []
        
        # Relay channels used by this workflow
        self.channels: Set[str] = set()
        
        # Reliability configuration
        self.reliability_config: Optional[ReliabilityConfig] = None
        self._reliability_enabled = False
        
        # Workflow state
        self.status = WorkflowStatus.CREATED
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None
        self.error: Optional[str] = None
        
        # Subscription tracking for cleanup
        self._subscriptions: List[Tuple[str, Any]] = []

        # Message deduplication (only for reliable mode)
        self._processed_messages: Set[str] = set()
        self._dedup_cleanup_task: Optional[asyncio.Task] = None
        self._dedup_max_size = 10000  # Prevent unbounded growth

        # Get trace manager for automatic workflow communication tracing
        self.trace_manager = get_trace_manager()

        logger.debug(f"Created workflow '{name}' with automatic tracing")
    
    def add_agent(self, name: str, agent: Any) -> "Workflow":
        """
        Add an agent to the workflow.
        
        Args:
            name: Agent name for workflow reference
            agent: Agent instance
            
        Returns:
            Self for method chaining
        """
        if name in self.agents:
            raise WorkflowError(f"Agent '{name}' already exists in workflow")
        
        self.agents[name] = agent
        logger.debug(f"Added agent '{name}' to workflow '{self.name}'")
        return self
    
    def add_agent_pool(
        self,
        name: str,
        agent_factory: Any,
        instances: int = 1
    ) -> "Workflow":
        """
        Add an agent pool to the workflow for horizontal scaling.
        
        Args:
            name: Pool name for workflow reference
            agent_factory: Factory function to create agent instances
            instances: Number of agent instances in the pool
            
        Returns:
            Self for method chaining
            
        Example:
            ```python
            def create_processor():
                return Agent(name="Processor")
            
            workflow.add_agent_pool("processors", create_processor, instances=5)
            ```
        """
        if name in self.agent_pools:
            raise WorkflowError(f"Agent pool '{name}' already exists in workflow")
        
        if name in self.agents:
            raise WorkflowError(f"Name '{name}' already used by an agent in workflow")
        
        # Import AgentPool here to avoid circular imports
        from ..core.scaling import AgentPool
        
        # Create agent pool
        pool = AgentPool(
            agent_factory=agent_factory,
            instances=instances,
            pool_name=f"{self.name}_{name}"
        )
        
        self.agent_pools[name] = pool
        logger.debug(f"Added agent pool '{name}' with {instances} instances to workflow '{self.name}'")
        return self

    def remove_agent(self, name: str) -> bool:
        """
        Remove agent and clean up its connections.

        Args:
            name: Agent name to remove

        Returns:
            True if agent was removed, False if not found
        """
        if name not in self.agents:
            return False

        # Remove agent
        del self.agents[name]

        # Clean up connections involving this agent
        self.connections = [
            c for c in self.connections
            if c.from_agent != name and c.to_agent != name
        ]

        # Note: Subscriptions will be cleaned up in _cleanup_connections when workflow stops

        logger.debug(f"Removed agent '{name}' and cleaned up connections")
        return True

    def connect(self, from_agent: str, channel: str, to_agent: str, task: str = "relay_message") -> "Workflow":
        """
        Connect two agents via a relay channel.

        Args:
            from_agent: Source agent name
            channel: Relay channel name
            to_agent: Destination agent name
            task: Task to execute on destination agent

        Returns:
            Self for method chaining
        """
        # Validate agents exist
        if from_agent not in self.agents:
            raise WorkflowError(f"Source agent '{from_agent}' not found")
        if to_agent not in self.agents:
            raise WorkflowError(f"Destination agent '{to_agent}' not found")

        # Check if connection already exists
        existing = next(
            (c for c in self.connections if c.from_agent == from_agent
             and c.channel == channel and c.to_agent == to_agent),
            None
        )

        if existing:
            logger.warning(f"Connection already exists: {existing}")
            return self

        connection = Connection(from_agent, channel, to_agent, task)
        self.connections.append(connection)
        self.channels.add(channel)

        logger.debug(f"Connected {from_agent} -> {channel} -> {to_agent}")
        return self
    
    def configure_reliability(
        self,
        preset: Optional[str] = None,
        acknowledgments: Optional[bool] = None,
        task_tracking: Optional[bool] = None,
        backpressure_control: Optional[bool] = None
    ) -> "Workflow":
        """
        Configure reliability features for this workflow.

        Args:
            acknowledgments: Enable message acknowledgments
            task_tracking: Enable task lifecycle tracking
            backpressure_control: Enable backpressure control

        Returns:
            Self for method chaining
        """
        config = ReliabilityConfig(
            acknowledgments=acknowledgments if acknowledgments is not None else True,
            task_tracking=task_tracking if task_tracking is not None else True,
            backpressure_control=backpressure_control if backpressure_control is not None else True
        )
        self.reliability_config = config
        self._reliability_enabled = True
        
        # Enable reliability in relay manager
        self.relay_manager.enable_reliability = True
        
        logger.info(f"Configured reliability for workflow '{self.name}': {config}")
        return self

    def validate_connections(self) -> List[str]:
        """
        Validate all workflow connections.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        for conn in self.connections:
            # Check from_agent exists
            if conn.from_agent not in self.agents and conn.from_agent not in self.agent_pools:
                errors.append(f"Source '{conn.from_agent}' not found in workflow")

            # Check to_agent exists
            if conn.to_agent not in self.agents and conn.to_agent not in self.agent_pools:
                errors.append(f"Destination '{conn.to_agent}' not found in workflow")

            # Check for circular dependencies (self-loops)
            if conn.from_agent == conn.to_agent:
                errors.append(f"Circular dependency: {conn.from_agent} -> {conn.to_agent}")

        return errors

    async def start(self) -> None:
        """Start the workflow and all agents with automatic tracing."""
        if self.status in [WorkflowStatus.RUNNING, WorkflowStatus.STARTING]:
            logger.warning(f"Workflow '{self.name}' is already running")
            return
        
        try:
            self.status = WorkflowStatus.STARTING
            logger.info(f"Starting workflow '{self.name}'")

            # Validate connections before starting
            validation_errors = self.validate_connections()
            if validation_errors:
                raise WorkflowError(f"Invalid connections: {'; '.join(validation_errors)}")

            # Trace workflow lifecycle start
            await self._trace_workflow_event("workflow_started", {
                "workflow_name": self.name,
                "agent_count": len(self.agents),
                "agent_pool_count": len(self.agent_pools),
                "connection_count": len(self.connections)
            })

            # Ensure relay manager is running
            if not self.relay_manager._running:
                try:
                    await self.relay_manager.start()
                except Exception as e:
                    raise WorkflowError(f"Failed to start relay manager: {str(e)}")
            
            # Configure agents with reliability features if enabled
            if self._reliability_enabled and self.reliability_config:
                await self._configure_agents_reliability()
            
            # Start all agents
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, 'start'):
                        await agent.start()
                    logger.debug(f"Started agent '{agent_name}'")
                except Exception as e:
                    logger.error(f"Failed to start agent '{agent_name}': {str(e)}")
                    raise WorkflowError(f"Failed to start agent '{agent_name}': {str(e)}")
            
            # Start all agent pools
            for pool_name, pool in self.agent_pools.items():
                try:
                    await pool.start()
                    logger.debug(f"Started agent pool '{pool_name}' with {pool.instance_count} instances")
                except Exception as e:
                    logger.error(f"Failed to start agent pool '{pool_name}': {str(e)}")
                    raise WorkflowError(f"Failed to start agent pool '{pool_name}': {str(e)}")
            
            # Set up relay connections with automatic tracing
            await self._setup_connections()

            # Start dedup cleanup task if reliability enabled
            if self._reliability_enabled:
                self._dedup_cleanup_task = asyncio.create_task(self._cleanup_dedup_cache())

            # Update status
            self.status = WorkflowStatus.RUNNING
            self.started_at = time.time()

            logger.info(f"Workflow '{self.name}' started successfully")
            
        except Exception as e:
            self.status = WorkflowStatus.ERROR
            self.error = str(e)
            logger.error(f"Failed to start workflow '{self.name}': {str(e)}")
            
            # Trace workflow error
            await self._trace_workflow_event("workflow_error", {
                "workflow_name": self.name,
                "error": str(e)
            })
            raise
    
    async def stop(self) -> None:
        """Stop the workflow by stopping all agents and cleaning up connections."""
        if self.status in [WorkflowStatus.STOPPED, WorkflowStatus.STOPPING]:
            logger.warning(f"Workflow '{self.name}' is already stopped")
            return
        
        try:
            self.status = WorkflowStatus.STOPPING
            logger.info(f"Stopping workflow '{self.name}'")

            # Stop dedup cleanup task
            if self._dedup_cleanup_task:
                self._dedup_cleanup_task.cancel()
                try:
                    await self._dedup_cleanup_task
                except asyncio.CancelledError:
                    pass
                self._dedup_cleanup_task = None

            # Clean up relay subscriptions
            await self._cleanup_connections()
            
            # Stop all agents
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, 'stop'):
                        await agent.stop()
                    logger.debug(f"Stopped agent '{agent_name}'")
                except Exception as e:
                    logger.warning(f"Error stopping agent '{agent_name}': {str(e)}")
            
            # Stop all agent pools
            for pool_name, pool in self.agent_pools.items():
                try:
                    await pool.stop()
                    logger.debug(f"Stopped agent pool '{pool_name}'")
                except Exception as e:
                    logger.warning(f"Error stopping agent pool '{pool_name}': {str(e)}")
            
            self.status = WorkflowStatus.STOPPED
            self.stopped_at = time.time()
            
            # Trace workflow lifecycle stop
            await self._trace_workflow_event("workflow_stopped", {
                "workflow_name": self.name,
                "running_time_seconds": self.stopped_at - (self.started_at or self.stopped_at)
            })
            
            logger.info(f"Workflow '{self.name}' stopped")
            
        except Exception as e:
            self.status = WorkflowStatus.ERROR
            self.error = str(e)
            logger.error(f"Error stopping workflow '{self.name}': {str(e)}")
            raise
    
    async def _setup_connections(self) -> None:
        """Set up relay connections between agents with automatic tracing."""
        for connection in self.connections:
            try:
                # Create callback based on reliability configuration
                if self._reliability_enabled and self.reliability_config:
                    callback = self._create_reliable_callback(connection)
                else:
                    callback = self._create_traced_callback(connection)
                
                # Subscribe to the relay channel
                await self.relay_manager.subscribe(connection.channel, callback)
                self._subscriptions.append((connection.channel, callback))
                
                logger.debug(f"Set up connection: {connection} (reliability: {self._reliability_enabled})")
                
            except Exception as e:
                logger.error(f"Failed to set up connection {connection}: {str(e)}")
                raise WorkflowError(f"Failed to set up connection {connection}: {str(e)}")
    
    def _create_traced_callback(self, connection: Connection):
        """Create a callback that automatically traces workflow communication and propagates metadata."""
        async def traced_callback(data: Any, metadata: Optional[Dict[str, Any]] = None):
            """Callback that processes relay data with automatic metadata propagation."""
            try:
                # Automatically trace the workflow communication
                await self._trace_workflow_communication(
                    from_agent=connection.from_agent,
                    to_agent=connection.to_agent,
                    channel=connection.channel,
                    data=data,
                    success=True
                )

                # Build enriched context with metadata propagation
                enriched_context = {
                    'source_agent': connection.from_agent,
                    'channel': connection.channel,
                    'workflow': self.name
                }

                # Add all upstream metadata to context
                if metadata:
                    enriched_context.update(metadata)

                # Process the data with the destination agent or agent pool
                if connection.to_agent in self.agents:
                    # Single agent
                    dest_agent = self.agents[connection.to_agent]

                    # Try new API first
                    if hasattr(dest_agent, 'receive_message'):
                        await dest_agent.receive_message(
                            data=data,
                            source_agent=connection.from_agent,
                            channel=connection.channel,
                            workflow_name=self.name
                        )
                    # Fallback to old API for legacy agents
                    elif hasattr(dest_agent, '_process'):
                        await dest_agent._process(connection.task, data, enriched_context)
                    else:
                        logger.warning(
                            f"Agent '{connection.to_agent}' has no receive_message() or _process() method"
                        )
                
                elif connection.to_agent in self.agent_pools:
                    # Agent pool - submit task to pool with enriched context
                    pool = self.agent_pools[connection.to_agent]
                    await pool.submit_task(connection.task, data, enriched_context)
                
                else:
                    logger.error(f"Destination '{connection.to_agent}' not found in agents or agent pools")
                
            except Exception as e:
                # Trace communication error
                await self._trace_workflow_communication(
                    from_agent=connection.from_agent,
                    to_agent=connection.to_agent,
                    channel=connection.channel,
                    data=data,
                    success=False,
                    error_message=str(e)
                )
                logger.error(f"Error in workflow communication {connection}: {str(e)}")
        
        return traced_callback
    
    async def _configure_agents_reliability(self) -> None:
        """Configure agents with reliability features."""
        if not self.reliability_config:
            return
        
        for agent_name, agent in self.agents.items():
            try:
                # Enable reliability features if agent supports it
                if hasattr(agent, 'enable_reliability_features') and self.reliability_config.task_tracking:
                    agent.enable_reliability_features()
                    logger.debug(f"Enabled reliability features for agent '{agent_name}'")
                
                # Configure backpressure if supported
                if (hasattr(agent, 'backpressure_controller') and 
                    self.reliability_config.backpressure_control and
                    agent.backpressure_controller is None):
                    from ..core.reliability import BackpressureController
                    agent.backpressure_controller = BackpressureController(
                        max_concurrent_tasks=10,
                        max_queue_size=100,
                        agent_id=getattr(agent, 'agent_id', agent_name)
                    )
                    logger.debug(f"Configured backpressure control for agent '{agent_name}'")
                
            except Exception as e:
                logger.warning(f"Failed to configure reliability for agent '{agent_name}': {e}")
    
    def _create_reliable_callback(self, connection: Connection):
        """Create a callback with reliability features and metadata propagation."""
        async def reliable_callback(data: Any, metadata: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None):
            """Callback that processes relay data with reliability features and metadata."""
            start_time = time.time()

            # Deduplication check (only if message_id provided)
            if message_id:
                if message_id in self._processed_messages:
                    logger.debug(f"Skipping duplicate message {message_id}")
                    # Still ACK it since we processed it before (idempotency)
                    if self.reliability_config and self.reliability_config.acknowledgments:
                        await self.relay_manager.ack_message(message_id)
                    return

                # Mark as processing
                self._processed_messages.add(message_id)

            try:
                # Automatically trace the workflow communication
                await self._trace_workflow_communication(
                    from_agent=connection.from_agent,
                    to_agent=connection.to_agent,
                    channel=connection.channel,
                    data=data,
                    success=True,
                    message_id=message_id
                )

                # Build enriched context with metadata propagation
                enriched_context = {
                    'source_agent': connection.from_agent,
                    'channel': connection.channel,
                    'workflow': self.name,
                    'message_id': message_id,
                    'reliability_enabled': True
                }

                # Add all upstream metadata to context
                if metadata:
                    enriched_context.update(metadata)

                # Process the data with the destination agent
                dest_agent = self.agents[connection.to_agent]

                # Try new API first
                if hasattr(dest_agent, 'receive_message'):
                    # Handle backpressure if enabled
                    if (self.reliability_config and
                        self.reliability_config.backpressure_control and
                        hasattr(dest_agent, 'backpressure_controller') and
                        dest_agent.backpressure_controller):

                        # Check if agent can handle the task
                        if not await dest_agent.backpressure_controller.acquire_processing_slot():
                            raise BackpressureError(
                                f"Agent {connection.to_agent} queue is full",
                                agent_id=getattr(dest_agent, 'agent_id', connection.to_agent)
                            )

                        try:
                            await dest_agent.receive_message(
                                data=data,
                                source_agent=connection.from_agent,
                                channel=connection.channel,
                                workflow_name=self.name
                            )
                        finally:
                            dest_agent.backpressure_controller.release_processing_slot()
                    else:
                        await dest_agent.receive_message(
                            data=data,
                            source_agent=connection.from_agent,
                            channel=connection.channel,
                            workflow_name=self.name
                        )

                    # Acknowledge message if reliability is enabled and message_id provided
                    if message_id and self.reliability_config and self.reliability_config.acknowledgments:
                        await self.relay_manager.ack_message(message_id)

                # Fallback to old API for legacy agents
                elif hasattr(dest_agent, '_process'):
                    context = enriched_context

                    # Handle backpressure if enabled
                    if (self.reliability_config and
                        self.reliability_config.backpressure_control and
                        hasattr(dest_agent, 'backpressure_controller') and
                        dest_agent.backpressure_controller):

                        # Check if agent can handle the task
                        if not await dest_agent.backpressure_controller.acquire_processing_slot():
                            raise BackpressureError(
                                f"Agent {connection.to_agent} queue is full",
                                agent_id=getattr(dest_agent, 'agent_id', connection.to_agent)
                            )

                        try:
                            await dest_agent._process(connection.task, data, context)
                        finally:
                            dest_agent.backpressure_controller.release_processing_slot()
                    else:
                        await dest_agent._process(connection.task, data, context)

                    # Acknowledge message if reliability is enabled and message_id provided
                    if message_id and self.reliability_config and self.reliability_config.acknowledgments:
                        await self.relay_manager.ack_message(message_id)

                else:
                    logger.warning(f"Agent '{connection.to_agent}' has no receive_message() or _process() method")
                    # NACK the message since we can't process it
                    if message_id and self.reliability_config and self.reliability_config.acknowledgments:
                        await self.relay_manager.nack_message(message_id, "Agent has no receive_message() or _process() method")
                
            except Exception as e:
                # Remove from processed on error so it can be retried
                if message_id:
                    self._processed_messages.discard(message_id)

                # Trace communication error
                await self._trace_workflow_communication(
                    from_agent=connection.from_agent,
                    to_agent=connection.to_agent,
                    channel=connection.channel,
                    data=data,
                    success=False,
                    error_message=str(e),
                    message_id=message_id
                )

                # NACK the message on error
                if message_id and self.reliability_config and self.reliability_config.acknowledgments:
                    await self.relay_manager.nack_message(message_id, str(e))

                # Enhanced error propagation
                error_context = {
                    'connection': str(connection),
                    'processing_time': time.time() - start_time,
                    'message_id': message_id,
                    'workflow': self.name
                }

                workflow_error = WorkflowError(
                    f"Agent {connection.to_agent} failed processing {connection.task}: {str(e)}",
                    workflow_name=self.name,
                    context=error_context
                )

                logger.error(f"Error in reliable workflow communication {connection}: {str(e)}")

                raise workflow_error
        
        return reliable_callback
    
    async def _cleanup_connections(self) -> None:
        """Clean up relay subscriptions."""
        for channel, callback in self._subscriptions:
            try:
                self.relay_manager.unsubscribe(channel, callback)
            except Exception as e:
                logger.warning(f"Error cleaning up subscription for channel '{channel}': {str(e)}")

        self._subscriptions.clear()
        logger.debug("Cleaned up relay subscriptions")

    async def _cleanup_dedup_cache(self) -> None:
        """Periodic cleanup of dedup cache to prevent memory leaks."""
        while self.status == WorkflowStatus.RUNNING:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Simple size-based cleanup
                if len(self._processed_messages) > self._dedup_max_size:
                    logger.warning(
                        f"Dedup cache exceeded {self._dedup_max_size} entries, clearing"
                    )
                    self._processed_messages.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dedup cleanup: {e}")

    async def inject_data(self, agent_name: str, data: Any, task: str = "inject") -> None:
        """
        Inject data into a specific agent to trigger workflow processing.
        
        Args:
            agent_name: Name of the agent to inject data into
            data: Data to inject
            task: Task to execute
        """
        if self.status != WorkflowStatus.RUNNING:
            raise WorkflowError(f"Cannot inject data - workflow is not running (status: {self.status})")
        
        if agent_name not in self.agents:
            raise WorkflowError(f"Agent '{agent_name}' not found in workflow")
        
        agent = self.agents[agent_name]
        
        try:
            # Trace data injection
            await self._trace_workflow_event("data_injected", {
                "workflow_name": self.name,
                "target_agent": agent_name,
                "task": task,
                "data_type": type(data).__name__
            })
            
            # Process data with the agent
            if hasattr(agent, 'process'):
                await agent.process(task, data, {'workflow': self.name, 'injection': True})
                logger.debug(f"Injected data into agent '{agent_name}' in workflow '{self.name}'")
            else:
                logger.warning(f"Agent '{agent_name}' has no process method")
                
        except Exception as e:
            # Log the error but don't raise it - allow workflow to continue
            logger.error(f"Error injecting data into agent '{agent_name}': {str(e)}")
    
    # Tracing methods for workflow events
    
    async def _trace_workflow_communication(
        self,
        from_agent: str,
        to_agent: str,
        channel: str,
        data: Any,
        success: bool,
        error_message: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Trace workflow communication using the unified tracing system."""
        try:
            # Create a communication span
            span_id = self.trace_manager.start_span(
                operation_name=f"workflow_communication",
                trace_type=TraceType.WORKFLOW_COMMUNICATION,
                input_data={
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "channel": channel,
                    "data_preview": str(data)[:200] if data else None
                },
                workflow_name=self.name,
                from_agent=from_agent,
                to_agent=to_agent,
                channel=channel,
                data_type=type(data).__name__,
                message_id=message_id,
                reliability_enabled=str(self._reliability_enabled)
            )
            
            # End the span with the result
            self.trace_manager.end_span(
                span_id=span_id,
                status=TraceStatus.SUCCESS if success else TraceStatus.ERROR,
                output_data={"communication_processed": success},
                error_message=error_message
            )
            
        except Exception as e:
            logger.warning(f"Failed to trace workflow communication: {e}")
    
    async def _trace_workflow_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trace general workflow events using the unified tracing system."""
        try:
            span_id = self.trace_manager.start_span(
                operation_name=f"workflow_{event_type}",
                trace_type=TraceType.WORKFLOW_COMMUNICATION,
                input_data=event_data,
                metadata={
                    "workflow_name": self.name,
                    "event_type": event_type
                }
            )
            
            self.trace_manager.end_span(
                span_id=span_id,
                status=TraceStatus.SUCCESS,
                output_data={"event_recorded": True}
            )
            
        except Exception as e:
            logger.warning(f"Failed to trace workflow event: {e}")
    
    # Simplified query methods using unified tracing
    
    def get_recent_communication(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent workflow communication from the unified tracing system.
        
        Args:
            limit: Maximum number of communication events to return
            
        Returns:
            List of recent workflow communication events
        """
        try:
            # Get recent operations from unified system
            return self.trace_manager.get_recent_operations(limit=limit)
        except Exception as e:
            logger.warning(f"Failed to get recent communication: {e}")
            return []
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics from the unified tracing system."""
        try:
            # Get workflow-specific metrics
            workflow_metrics = self.trace_manager.get_workflow_metrics(self.name)
            return workflow_metrics
        except Exception as e:
            logger.warning(f"Failed to get workflow stats: {e}")
            return {}
    
    # Basic workflow information methods
    
    def get_agent(self, name: str) -> Optional[Any]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all agent names in the workflow."""
        return list(self.agents.keys())
    
    def list_connections(self) -> List[str]:
        """List all connections as strings."""
        return [str(conn) for conn in self.connections]
    
    def get_channel_data(self, channel: str, count: int = 1) -> List[Any]:
        """Get latest data from a relay channel."""
        return self.relay_manager.get_latest(channel, count)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics including reliability metrics."""
        running_time = None
        if self.started_at:
            end_time = self.stopped_at or time.time()
            running_time = end_time - self.started_at
        
        stats = {
            'name': self.name,
            'project_id': self.project_id,
            'status': self.status.value,
            'agent_count': len(self.agents),
            'connection_count': len(self.connections),
            'channel_count': len(self.channels),
            'created_at': self.created_at,
            'started_at': self.started_at,
            'stopped_at': self.stopped_at,
            'running_time': running_time,
            'error': self.error,
            'agents': list(self.agents.keys()),
            'channels': list(self.channels),
            'reliability_enabled': self._reliability_enabled
        }
        
        # Add reliability statistics if enabled
        if self._reliability_enabled and self.reliability_config:
            stats['reliability_config'] = {
                'acknowledgments': self.reliability_config.acknowledgments,
                'task_tracking': self.reliability_config.task_tracking,
                'backpressure_control': self.reliability_config.backpressure_control
            }
            
            # Get relay manager reliability stats
            relay_stats = self.relay_manager.get_stats()
            if 'pending_messages' in relay_stats:
                stats['pending_messages'] = relay_stats['pending_messages']
                stats['active_timeouts'] = relay_stats['active_timeouts']
        
        return stats
    
    # Reliability management methods
    
    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get list of pending messages waiting for acknowledgment."""
        if not self._reliability_enabled:
            return []
        return self.relay_manager.get_pending_messages()
    
    async def get_agent_reliability_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get reliability statistics for a specific agent."""
        if agent_name not in self.agents:
            return {}
        
        agent = self.agents[agent_name]
        stats = {'agent_name': agent_name, 'reliability_enabled': False}
        
        if hasattr(agent, 'enable_reliability') and agent.enable_reliability:
            stats['reliability_enabled'] = True
            
            # Get task management stats
            if hasattr(agent, 'get_agent_tasks'):
                try:
                    tasks = await agent.get_agent_tasks()
                    stats['total_tasks'] = len(tasks)
                    stats['tasks_by_status'] = {}
                    for task in tasks:
                        status = task.get('status', 'unknown')
                        stats['tasks_by_status'][status] = stats['tasks_by_status'].get(status, 0) + 1
                except Exception as e:
                    logger.warning(f"Failed to get task stats for agent {agent_name}: {e}")
            
            # Get backpressure stats
            if hasattr(agent, 'get_backpressure_stats'):
                try:
                    bp_stats = agent.get_backpressure_stats()
                    stats['backpressure'] = bp_stats
                except Exception as e:
                    logger.warning(f"Failed to get backpressure stats for agent {agent_name}: {e}")
        
        return stats
    
    def is_reliability_enabled(self) -> bool:
        """Check if reliability features are enabled for this workflow."""
        return self._reliability_enabled
    
    def get_reliability_config(self) -> Optional[ReliabilityConfig]:
        """Get the current reliability configuration."""
        return self.reliability_config

    def get_token_usage(self) -> Dict[str, Any]:
        """
        Aggregate token usage from all agents in the workflow.

        Returns:
            Dictionary containing aggregated token usage across all agents
        """
        total_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
            "models_used": [],
            "agents_with_usage": []
        }

        for agent_name, agent in self.agents.items():
            try:
                # Method 1: Check if agent has get_token_usage()
                if hasattr(agent, 'get_token_usage'):
                    usage = agent.get_token_usage()
                    if usage and isinstance(usage, dict):
                        tokens = usage.get("total_tokens", 0)
                        if tokens > 0:
                            total_usage["total_tokens"] += tokens
                            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                            total_usage["llm_calls"] += usage.get("total_calls", usage.get("llm_calls", 0))
                            total_usage["agents_with_usage"].append(agent_name)
                            logger.debug(f"Workflow: Agent '{agent_name}' used {tokens} tokens")

                # Method 2: Check if agent has llm_provider with usage tracking
                elif hasattr(agent, 'llm_provider') and hasattr(agent.llm_provider, 'get_token_usage'):
                    llm_usage = agent.llm_provider.get_token_usage()
                    if llm_usage and isinstance(llm_usage, dict):
                        tokens = llm_usage.get("total_tokens", 0)
                        if tokens > 0:
                            total_usage["total_tokens"] += tokens
                            total_usage["prompt_tokens"] += llm_usage.get("prompt_tokens", 0)
                            total_usage["completion_tokens"] += llm_usage.get("completion_tokens", 0)
                            total_usage["llm_calls"] += 1
                            total_usage["agents_with_usage"].append(agent_name)
                            logger.debug(f"Workflow: Agent '{agent_name}' (via llm_provider) used {tokens} tokens")

            except Exception as e:
                logger.warning(f"Failed to get token usage from agent '{agent_name}': {e}")

        if total_usage["total_tokens"] > 0:
            logger.info(f"Workflow '{self.name}' total token usage: {total_usage['total_tokens']} tokens across {len(total_usage['agents_with_usage'])} agents")
            return total_usage
        else:
            logger.debug(f"Workflow '{self.name}' has no token usage")
            return None

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive workflow health check.

        Returns:
            Dictionary containing health status and any issues found
        """
        health = {
            'status': self.status.value,
            'healthy': self.status == WorkflowStatus.RUNNING,
            'agents': {},
            'issues': []
        }

        # Check each agent (with safe attribute access)
        for agent_name, agent in self.agents.items():
            agent_health = {
                'name': agent_name,
                'has_receive_message': hasattr(agent, 'receive_message'),
                'running': self.status == WorkflowStatus.RUNNING
            }

            # Check if agent has health method (optional)
            if hasattr(agent, 'get_health'):
                try:
                    agent_health.update(agent.get_health())
                except Exception as e:
                    agent_health['health_error'] = str(e)

            health['agents'][agent_name] = agent_health

            if not agent_health['has_receive_message']:
                health['issues'].append(f"Agent '{agent_name}' has no receive_message method")
                health['healthy'] = False

        # Check subscriptions for potential memory leaks
        subscription_count = len(self._subscriptions)
        health['subscription_count'] = subscription_count
        if subscription_count > 1000:
            health['issues'].append(f"High subscription count: {subscription_count}")
            health['healthy'] = False

        # Check pending messages (if reliability enabled)
        if self._reliability_enabled:
            pending = self.get_pending_messages()
            health['pending_message_count'] = len(pending)
            if len(pending) > 100:
                health['issues'].append(f"High pending message count: {len(pending)}")
                health['healthy'] = False

        return health

    # Context manager support
    async def __aenter__(self) -> "Workflow":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()