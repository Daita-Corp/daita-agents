"""
Core Reliability Infrastructure for Daita Agents.

Provides task management, retry policies, circuit breakers, and other
reliability patterns for production-grade inter-agent communication.

Key Components:
- TaskManager: Track task lifecycle and state
- CircuitBreaker: Prevent cascading failures
- BackpressureController: Manage agent queue capacity

Note: RetryPolicy has been moved to config.base for better integration with configuration system.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Task Management

class TaskStatus(str, Enum):
    """Status of a task in the system."""
    QUEUED = "queued"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Represents a task being processed by an agent."""
    id: str
    agent_id: str
    task_type: str
    data: Any
    status: TaskStatus = TaskStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    progress: float = 0.0
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def duration(self) -> Optional[float]:
        """Calculate task duration if started."""
        if not self.started_at:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
    
    def age(self) -> float:
        """Calculate task age since creation."""
        return time.time() - self.created_at

class TaskManager:
    """
    Manages task lifecycle and state tracking.
    
    Integrates with existing tracing system for automatic task visibility.
    """
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._agent_tasks: Dict[str, List[str]] = {}  # agent_id -> task_ids
        self._lock = asyncio.Lock()
    
    async def create_task(
        self, 
        agent_id: str, 
        task_type: str, 
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task and return its ID."""
        task_id = uuid.uuid4().hex
        task = Task(
            id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            data=data,
            context=context or {}
        )
        
        async with self._lock:
            self._tasks[task_id] = task
            if agent_id not in self._agent_tasks:
                self._agent_tasks[agent_id] = []
            self._agent_tasks[agent_id].append(task_id)
        
        logger.debug(f"Created task {task_id} for agent {agent_id}")
        return task_id
    
    async def update_status(
        self, 
        task_id: str, 
        status: TaskStatus,
        error: Optional[str] = None,
        progress: Optional[float] = None
    ) -> bool:
        """Update task status and metadata."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            task.status = status
            if error:
                task.error = error
            if progress is not None:
                task.progress = progress
            
            # Update timestamps
            if status == TaskStatus.RUNNING and not task.started_at:
                task.started_at = time.time()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task.completed_at = time.time()
        
        logger.debug(f"Updated task {task_id} status to {status}")
        return True
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status summary."""
        task = await self.get_task(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "status": task.status.value,
            "progress": task.progress,
            "error": task.error,
            "duration": task.duration(),
            "age": task.age(),
            "retry_count": task.retry_count
        }
    
    async def get_agent_tasks(self, agent_id: str, status: Optional[TaskStatus] = None) -> List[Task]:
        """Get all tasks for an agent, optionally filtered by status."""
        async with self._lock:
            task_ids = self._agent_tasks.get(agent_id, [])
            tasks = [self._tasks[tid] for tid in task_ids if tid in self._tasks]
            
            if status:
                tasks = [t for t in tasks if t.status == status]
            
            return tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's not already completed."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
        
        logger.debug(f"Cancelled task {task_id}")
        return True
    
    async def cleanup_old_tasks(self, max_age_seconds: int = 3600) -> int:
        """Remove old completed tasks to prevent memory leaks."""
        cutoff_time = time.time() - max_age_seconds
        removed_count = 0
        
        async with self._lock:
            to_remove = []
            for task_id, task in self._tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                    and task.created_at < cutoff_time):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                task = self._tasks.pop(task_id, None)
                if task:
                    # Remove from agent tasks
                    agent_tasks = self._agent_tasks.get(task.agent_id, [])
                    if task_id in agent_tasks:
                        agent_tasks.remove(task_id)
                    removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old tasks")
        
        return removed_count

# Circuit Breaker

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by temporarily stopping calls to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if (self.state == CircuitState.OPEN and 
                self.last_failure_time and
                time.time() - self.last_failure_time > self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.debug("Circuit breaker transitioning to HALF_OPEN")
        
        # Reject requests if circuit is OPEN
        if self.state == CircuitState.OPEN:
            try:
                from ..core.exceptions import CircuitBreakerOpenError
            except ImportError:
                from core.exceptions import CircuitBreakerOpenError
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful operation."""
        async with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info("Circuit breaker closed after successful recovery")
            elif self.state == CircuitState.CLOSED:
                pass  # Already in good state
    
    async def _on_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure during recovery test - go back to OPEN
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker opened after failed recovery attempt")
            elif (self.state == CircuitState.CLOSED and 
                  self.failure_count >= self.failure_threshold):
                # Too many failures - open the circuit
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }

# Backpressure Control

class BackpressureController:
    """
    Controls backpressure for agent task queues.
    
    Prevents memory exhaustion by limiting concurrent tasks and queue sizes.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        max_queue_size: int = 100,
        agent_id: Optional[str] = None
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.agent_id = agent_id
        
        self.current_tasks = 0
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._lock = asyncio.Lock()
    
    async def submit_task(self, task_data: Any) -> bool:
        """Submit task for processing. Returns False if queue is full."""
        try:
            self.task_queue.put_nowait(task_data)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full for agent {self.agent_id}, rejecting task")
            return False
    
    async def get_next_task(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Get next task from queue with optional timeout."""
        try:
            if timeout:
                return await asyncio.wait_for(self.task_queue.get(), timeout=timeout)
            else:
                return await self.task_queue.get()
        except asyncio.TimeoutError:
            return None
    
    async def acquire_processing_slot(self) -> bool:
        """Acquire a processing slot for concurrent task execution."""
        try:
            await self._semaphore.acquire()
            async with self._lock:
                self.current_tasks += 1
            return True
        except Exception:
            return False
    
    def release_processing_slot(self):
        """Release a processing slot after task completion."""
        try:
            self._semaphore.release()
            asyncio.create_task(self._decrement_current_tasks())
        except Exception as e:
            logger.error(f"Error releasing processing slot: {e}")
    
    async def _decrement_current_tasks(self):
        """Safely decrement current task count."""
        async with self._lock:
            self.current_tasks = max(0, self.current_tasks - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current backpressure statistics."""
        return {
            "current_tasks": self.current_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "queue_utilization": self.task_queue.qsize() / self.max_queue_size,
            "concurrency_utilization": self.current_tasks / self.max_concurrent_tasks
        }

# Global instances for shared use
_global_task_manager: Optional[TaskManager] = None

def get_global_task_manager() -> TaskManager:
    """Get global task manager instance."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = TaskManager()
    return _global_task_manager