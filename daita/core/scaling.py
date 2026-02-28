"""
Agent Pool Scaling for Daita Agents.

Provides manual horizontal scaling of agent instances for handling concurrent workloads.
Uses a simple, MVP-focused approach without complex auto-scaling logic.

Features:
- Manual agent pool management
- Round-robin load balancing
- Simple instance lifecycle management  
- Integration with existing reliability features
- Async-safe concurrent task execution

Example:
    ```python
    from daita.core.scaling import AgentPool
    from daita.agents.agent import Agent
    
    # Create agent factory
    def create_processor():
        return Agent(name="Processor", preset="analysis")
    
    # Create agent pool with 5 instances
    pool = AgentPool(
        agent_factory=create_processor,
        instances=5,
        pool_name="processors"
    )
    
    await pool.start()
    
    # Submit tasks to pool (load balanced)
    result = await pool.submit_task("analyze", data={"text": "Hello"})
    
    await pool.stop()
    ```
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class PoolStatus(str, Enum):
    """Status of an agent pool."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass 
class AgentInstance:
    """Represents an agent instance in a pool."""
    id: str
    agent: Any
    created_at: float = field(default_factory=time.time)
    task_count: int = 0
    last_task_at: Optional[float] = None
    current_tasks: int = 0
    status: str = "idle"
    
    def is_available(self) -> bool:
        """Check if instance is available for new tasks."""
        return self.status == "idle" and self.current_tasks == 0

class LoadBalancer:
    """Simple round-robin load balancer for agent instances."""
    
    def __init__(self):
        self.current_index = 0
    
    def select_instance(self, instances: List[AgentInstance]) -> Optional[AgentInstance]:
        """
        Select next available agent instance using round-robin.
        
        Args:
            instances: List of agent instances
            
        Returns:
            Selected agent instance or None if none available
        """
        if not instances:
            return None
        
        # Try round-robin selection first
        available_instances = [inst for inst in instances if inst.is_available()]
        
        if not available_instances:
            return None
        
        # Select using round-robin
        selected = available_instances[self.current_index % len(available_instances)]
        self.current_index += 1
        
        return selected

class AgentPool:
    """
    Agent pool for horizontal scaling with manual instance management.
    
    Provides load balancing across multiple agent instances to handle
    concurrent workloads. Uses simple round-robin balancing and manual
    instance count management.
    """
    
    def __init__(
        self,
        agent_factory: Callable[[], Any],
        instances: int = 1,
        pool_name: Optional[str] = None,
        max_concurrent_per_instance: int = 1
    ):
        """
        Initialize agent pool.
        
        Args:
            agent_factory: Factory function to create agent instances
            instances: Number of agent instances to create
            pool_name: Optional name for the pool (for logging)
            max_concurrent_per_instance: Max concurrent tasks per agent (default: 1)
        """
        self.agent_factory = agent_factory
        self.instance_count = max(1, instances)  # At least 1 instance
        self.pool_name = pool_name or f"pool_{uuid.uuid4().hex[:8]}"
        self.max_concurrent_per_instance = max_concurrent_per_instance
        
        # Pool state
        self.status = PoolStatus.CREATED
        self.instances: List[AgentInstance] = []
        self.load_balancer = LoadBalancer()
        
        # Statistics
        self.total_tasks = 0
        self.failed_tasks = 0
        self.created_at = time.time()
        
        # Async locks
        self._pool_lock = asyncio.Lock()
        
        logger.debug(f"AgentPool '{self.pool_name}' created with {self.instance_count} instances")
    
    async def start(self) -> None:
        """Start the agent pool and create all agent instances."""
        if self.status != PoolStatus.CREATED:
            logger.warning(f"Pool '{self.pool_name}' already started or in invalid state")
            return
        
        self.status = PoolStatus.STARTING
        logger.info(f"Starting agent pool '{self.pool_name}' with {self.instance_count} instances")
        
        try:
            async with self._pool_lock:
                # Create all agent instances
                for i in range(self.instance_count):
                    await self._create_instance(f"{self.pool_name}_instance_{i}")
            
            self.status = PoolStatus.RUNNING
            logger.info(f"Agent pool '{self.pool_name}' started successfully")
            
        except Exception as e:
            self.status = PoolStatus.ERROR
            logger.error(f"Failed to start agent pool '{self.pool_name}': {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the agent pool and cleanup all instances."""
        if self.status in [PoolStatus.STOPPED, PoolStatus.STOPPING]:
            return
        
        self.status = PoolStatus.STOPPING
        logger.info(f"Stopping agent pool '{self.pool_name}'")
        
        try:
            async with self._pool_lock:
                # Stop all agent instances
                for instance in self.instances:
                    try:
                        if hasattr(instance.agent, 'stop'):
                            await instance.agent.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping instance {instance.id}: {e}")
                
                self.instances.clear()
            
            self.status = PoolStatus.STOPPED
            logger.info(f"Agent pool '{self.pool_name}' stopped")
            
        except Exception as e:
            self.status = PoolStatus.ERROR
            logger.error(f"Error stopping agent pool '{self.pool_name}': {e}")
            raise
    
    async def _create_instance(self, instance_id: str) -> AgentInstance:
        """
        Create and start a new agent instance.
        
        Args:
            instance_id: Unique ID for the instance
            
        Returns:
            Created agent instance
        """
        try:
            # Create agent using factory
            agent = self.agent_factory()
            
            # Start the agent if it has a start method
            if hasattr(agent, 'start'):
                await agent.start()
            
            # Create instance record
            instance = AgentInstance(
                id=instance_id,
                agent=agent
            )
            
            self.instances.append(instance)
            logger.debug(f"Created agent instance {instance_id}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create agent instance {instance_id}: {e}")
            raise
    
    async def submit_task(
        self, 
        task: str, 
        data: Any = None, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Submit task to an available agent instance.
        
        Args:
            task: Task name/type
            data: Task data
            context: Optional task context
            **kwargs: Additional task parameters
            
        Returns:
            Task result
        """
        if self.status != PoolStatus.RUNNING:
            raise RuntimeError(f"Pool '{self.pool_name}' is not running (status: {self.status})")
        
        # Select available instance
        instance = self.load_balancer.select_instance(self.instances)
        if not instance:
            raise RuntimeError(f"No available agent instances in pool '{self.pool_name}'")
        
        # Track task execution
        self.total_tasks += 1
        instance.task_count += 1
        instance.current_tasks += 1
        instance.last_task_at = time.time()
        instance.status = "busy"
        
        try:
            # Execute task on selected agent
            logger.debug(f"Submitting task '{task}' to instance {instance.id}")

            # Agent pools typically used for workflow parallelization
            # Check task type and route appropriately
            if task == "relay_message" and hasattr(instance.agent, 'receive_message'):
                # Workflow relay message
                result = await instance.agent.receive_message(
                    data=data,
                    source_agent=context.get('source_agent', 'pool') if context else 'pool',
                    channel=context.get('channel', 'default') if context else 'default',
                    workflow_name=context.get('workflow') if context else None
                )
            elif hasattr(instance.agent, '_process'):
                # Fallback to internal API
                result = await instance.agent._process(task, data, context or {}, **kwargs)
            else:
                # Last resort: try run_detailed
                prompt = context.get('prompt', str(data)) if context else str(data)
                result = await instance.agent.run_detailed(prompt)

            logger.debug(f"Task '{task}' completed on instance {instance.id}")
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Task '{task}' failed on instance {instance.id}: {e}")
            raise
        finally:
            # Update instance state
            instance.current_tasks = max(0, instance.current_tasks - 1)
            if instance.current_tasks == 0:
                instance.status = "idle"
    
    async def resize(self, new_instance_count: int) -> None:
        """
        Resize the agent pool (manual scaling).
        
        Args:
            new_instance_count: New number of instances
        """
        if self.status != PoolStatus.RUNNING:
            raise RuntimeError(f"Cannot resize pool '{self.pool_name}' - not running")
        
        new_instance_count = max(1, new_instance_count)  # At least 1 instance
        current_count = len(self.instances)
        
        if new_instance_count == current_count:
            logger.debug(f"Pool '{self.pool_name}' already has {current_count} instances")
            return
        
        async with self._pool_lock:
            if new_instance_count > current_count:
                # Scale up - add instances
                instances_to_add = new_instance_count - current_count
                logger.info(f"Scaling up pool '{self.pool_name}' from {current_count} to {new_instance_count} instances")
                
                for i in range(instances_to_add):
                    instance_id = f"{self.pool_name}_instance_{current_count + i}"
                    await self._create_instance(instance_id)
            
            else:
                # Scale down - remove instances
                instances_to_remove = current_count - new_instance_count
                logger.info(f"Scaling down pool '{self.pool_name}' from {current_count} to {new_instance_count} instances")
                
                # Remove least busy instances
                instances_by_load = sorted(self.instances, key=lambda x: x.current_tasks)
                
                for _ in range(instances_to_remove):
                    if instances_by_load:
                        instance = instances_by_load.pop(0)
                        
                        # Wait for current tasks to complete (with timeout)
                        timeout_seconds = 30
                        wait_start = time.time()
                        
                        while instance.current_tasks > 0 and (time.time() - wait_start) < timeout_seconds:
                            await asyncio.sleep(0.1)
                        
                        # Stop and remove instance
                        try:
                            if hasattr(instance.agent, 'stop'):
                                await instance.agent.stop()
                        except Exception as e:
                            logger.warning(f"Error stopping instance {instance.id}: {e}")
                        
                        self.instances.remove(instance)
                        logger.debug(f"Removed instance {instance.id}")
        
        self.instance_count = new_instance_count
        logger.info(f"Pool '{self.pool_name}' resized to {new_instance_count} instances")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent pool statistics."""
        if not self.instances:
            return {
                "pool_name": self.pool_name,
                "status": self.status.value,
                "instance_count": 0,
                "total_tasks": self.total_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": 0.0,
                "uptime_seconds": time.time() - self.created_at
            }
        
        # Calculate instance statistics
        busy_instances = sum(1 for inst in self.instances if inst.current_tasks > 0)
        total_current_tasks = sum(inst.current_tasks for inst in self.instances)
        avg_tasks_per_instance = sum(inst.task_count for inst in self.instances) / len(self.instances)
        
        success_rate = 0.0
        if self.total_tasks > 0:
            success_rate = ((self.total_tasks - self.failed_tasks) / self.total_tasks) * 100
        
        return {
            "pool_name": self.pool_name,
            "status": self.status.value,
            "instance_count": len(self.instances),
            "busy_instances": busy_instances,
            "idle_instances": len(self.instances) - busy_instances,
            "total_current_tasks": total_current_tasks,
            "total_tasks": self.total_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": round(success_rate, 2),
            "avg_tasks_per_instance": round(avg_tasks_per_instance, 2),
            "uptime_seconds": round(time.time() - self.created_at, 2)
        }
    
    def get_instance_stats(self) -> List[Dict[str, Any]]:
        """Get detailed statistics for each instance."""
        return [
            {
                "id": inst.id,
                "status": inst.status,
                "task_count": inst.task_count,
                "current_tasks": inst.current_tasks,
                "last_task_at": inst.last_task_at,
                "uptime_seconds": round(time.time() - inst.created_at, 2),
                "is_available": inst.is_available()
            }
            for inst in self.instances
        ]
    
    # Context manager support
    async def __aenter__(self) -> "AgentPool":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

# Utility functions for pool management

def create_agent_pool(
    agent_factory: Callable[[], Any],
    instances: int = 1,
    pool_name: Optional[str] = None
) -> AgentPool:
    """
    Create an agent pool with the specified configuration.
    
    Args:
        agent_factory: Factory function to create agent instances
        instances: Number of agent instances
        pool_name: Optional pool name
        
    Returns:
        Configured AgentPool instance
        
    Example:
        ```python
        from daita.core.scaling import create_agent_pool
        from daita.agents.agent import Agent
        
        # Create pool factory
        def make_processor():
            return Agent(name="Processor")
        
        pool = create_agent_pool(make_processor, instances=5, pool_name="processors")
        
        async with pool:
            result = await pool.submit_task("analyze", data={"text": "Hello"})
        ```
    """
    return AgentPool(
        agent_factory=agent_factory,
        instances=instances,
        pool_name=pool_name
    )