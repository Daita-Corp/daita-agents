"""
Agent implementations for Daita Agents.

This module provides the core agent implementations:
- BaseAgent: Foundation class with retry logic and error handling
- Agent: Flexible agent that can be customized with handlers and presets

The agent system is designed around the Agent as the primary interface,
with preset configurations for common patterns like analysis and transformation.

Usage:
    ```python
    from daita.agents import Agent

    # Direct instantiation (recommended)
    agent = Agent(name="My Agent", llm_provider="openai", model="gpt-4")

    # Or with configuration object (backward compatibility)
    from daita.config.base import AgentConfig
    config = AgentConfig(name="My Agent")
    agent = Agent(config=config)
    ```
"""

# Core agent classes
from .base import BaseAgent
from .agent import Agent

# Export all agent functionality
__all__ = [
    "BaseAgent",
    "Agent",
]