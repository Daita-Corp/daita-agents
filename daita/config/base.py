"""
Configuration models for Daita Agents.
"""

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class YamlSerializableMixin:
    """Mixin for YAML serialization support."""

    def model_dump_yaml_safe(self) -> Dict[str, Any]:
        """Export to YAML-safe dictionary."""
        return self.model_dump(mode="json", exclude_none=True)


class AgentType(str, Enum):
    """Types of agents available."""

    STANDARD = "standard"
    # Additional agent types can be added here


class RetryStrategy(str, Enum):
    """Retry strategy types - unified from core/reliability.py."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    # Legacy aliases for backward compatibility
    EXPONENTIAL_BACKOFF = "exponential"
    FIXED_DELAY = "fixed"
    IMMEDIATE = "fixed"


class RetryPolicy(YamlSerializableMixin, BaseModel):
    """Unified retry configuration compatible with core/reliability.py."""

    # Core fields (compatible with dataclass version)
    max_retries: int = Field(
        default=3, ge=0, le=20, description="Maximum number of retry attempts"
    )
    base_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Base delay in seconds"
    )
    max_delay: float = Field(
        default=60.0, ge=1.0, le=300.0, description="Maximum delay in seconds"
    )
    strategy: RetryStrategy = Field(
        default=RetryStrategy.EXPONENTIAL, description="Retry timing strategy"
    )
    jitter: bool = Field(
        default=True, description="Add jitter to prevent thundering herd"
    )

    # Additional fields for configuration support
    permanent_errors: List[str] = Field(
        default=[
            "AuthenticationError",
            "PermissionError",
            "ValidationError",
            "InvalidDataError",
            "NotFoundError",
            "BadRequestError",
            "PermanentError",
        ],
        description="Error types that should not be retried",
    )

    # Legacy compatibility field
    initial_delay: Optional[float] = Field(
        default=None, description="Legacy field - use base_delay instead"
    )

    @model_validator(mode="after")
    def handle_legacy_fields(self):
        """Handle legacy initial_delay field."""
        if self.initial_delay is not None and hasattr(self, "base_delay"):
            self.base_delay = self.initial_delay
        return self

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given retry attempt."""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        else:  # EXPONENTIAL
            delay = self.base_delay * (2 ** (attempt - 1))

        # Apply max delay limit
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5

        return delay

    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry policy."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt + 1)
                    await asyncio.sleep(delay)
                else:
                    break

        # If we get here, all retries failed
        raise last_error


class AgentConfig(YamlSerializableMixin, BaseModel):
    """Simplified agent configuration."""

    name: str
    type: AgentType = AgentType.STANDARD
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)

    # Simple retry configuration - optional
    enable_retry: bool = Field(
        default=False, description="Enable retry behavior (default: fail-fast)"
    )
    retry_policy: Optional[RetryPolicy] = Field(
        default=None, description="Retry policy - only used if enable_retry=True"
    )

    @model_validator(mode="after")
    def setup_retry_policy(self):
        """Set up retry policy if retry is enabled but no policy provided."""
        if self.enable_retry and self.retry_policy is None:
            self.retry_policy = RetryPolicy()
        return self

    @property
    def retry_enabled(self) -> bool:
        """Check if retry behavior is enabled."""
        return self.enable_retry and self.retry_policy is not None


class DaitaConfig(YamlSerializableMixin, BaseModel):
    """Main configuration for Daita framework."""

    version: str = "1.0.0"
    agents: List[AgentConfig] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
