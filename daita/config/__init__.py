"""
Configuration system for Daita Agents.

Core configuration classes:
- AgentConfig: Configuration for individual agents
- DaitaConfig: Overall framework configuration
- RetryPolicy: Simple retry behavior configuration

Enums and types:
- AgentType: Types of available agents
- RetryStrategy: Retry timing strategies

Focus filtering is now handled by the Focus DSL — see daita.core.focus.
"""

from .base import (
    # Enums
    AgentType,
    RetryStrategy,
    # Configuration classes
    RetryPolicy,
    AgentConfig,
    DaitaConfig,
)

from .settings import settings

__all__ = [
    # Enums
    "AgentType",
    "RetryStrategy",
    # Configuration classes
    "RetryPolicy",
    "AgentConfig",
    "DaitaConfig",
    # Runtime settings
    "settings",
]
