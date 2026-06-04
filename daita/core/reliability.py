"""
Core Reliability Infrastructure for Daita Agents.

Provides small reliability helper patterns that do not own runtime task state.

Key Components:
- CircuitBreaker: Prevent cascading failures

Note: RetryPolicy has been moved to config.base for better integration with configuration system.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

# Circuit Breaker


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by temporarily stopping calls to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
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
            if (
                self.state == CircuitState.OPEN
                and self.last_failure_time
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
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
            elif (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.failure_threshold
            ):
                # Too many failures - open the circuit
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }
