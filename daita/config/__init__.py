"""
Configuration system for Daita Agents.

Simplified configuration classes focused on essential functionality.

Core configuration classes:
- AgentConfig: Configuration for individual agents
  
- DaitaConfig: Overall framework configuration
- RetryPolicy: Simple retry behavior configuration

Enums and types:
- AgentType: Types of available agents
- FocusType: Types of focus selectors
- RetryStrategy: Retry timing strategies

Usage:
    ```python
    from daita.config import AgentConfig, RetryPolicy
    
    # Simple agent configuration
    config = AgentConfig(name="My Agent")
    
    # Agent with retry enabled
    config = AgentConfig(
        name="My Agent",
        enable_retry=True,
        retry_policy=RetryPolicy(max_retries=5)
    )
    ```
"""

from .base import (
    # Enums
    AgentType,
    FocusType,
    RetryStrategy,
    
    # Configuration classes
    FocusConfig,
    RetryPolicy,
    AgentConfig,
    DaitaConfig,
)

from .settings import settings

__all__ = [
    # Enums
    "AgentType",
    "FocusType", 
    "RetryStrategy",
    
    # Configuration classes
    "FocusConfig",
    "RetryPolicy",
    "AgentConfig",
    "DaitaConfig",
    
    # Runtime settings
    "settings",
]