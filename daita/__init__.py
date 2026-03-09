"""
Daita Agents — An open-source framework for building production-ready AI agents.

Quick start:
    from daita import Agent, tool

    @tool
    def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    agent = Agent(name="weather", tools=[get_weather])
    result = await agent.run("What's the weather in Tokyo?")

Key components:
- Agent / BaseAgent      — Build single or custom agents
- Workflow / RelayManager — Multi-agent orchestration
- tool / AgentTool       — Define tools agents can call
- Plugins               — Database and API integrations (postgresql, s3, slack, ...)
- AgentConfig           — Configure retry policies, LLM settings, and more
"""

__version__ = "0.9.0"

# ---------------------------------------------------------------------------
# Core — what 95% of users need
# ---------------------------------------------------------------------------
from .agents.agent import Agent
from .agents.base import BaseAgent
from .agents.conversation import ConversationHistory

from .core.tools import tool, AgentTool, ToolRegistry

from .core.workflow import Workflow
from .core.relay import RelayManager

# ---------------------------------------------------------------------------
# Plugins — database and API integrations
# ---------------------------------------------------------------------------
from .plugins import postgresql, mysql, mongodb, rest, s3, slack, elasticsearch
from .plugins.redis_messaging import redis_messaging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from .config.base import AgentConfig, RetryPolicy, RetryStrategy

# ---------------------------------------------------------------------------
# Focus DSL — pre-filter tool results before the LLM sees them
# ---------------------------------------------------------------------------
from .core.focus import apply_focus

# ---------------------------------------------------------------------------
# Exceptions — for error handling in user code
# ---------------------------------------------------------------------------
from .core.exceptions import (
    DaitaError,
    AgentError,
    LLMError,
    ConfigError,
    PluginError,
    WorkflowError,
    TransientError,
    RetryableError,
    PermanentError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    FocusDSLError,
)

# ---------------------------------------------------------------------------
# LLM factory — for explicit provider construction
# ---------------------------------------------------------------------------
from .llm.factory import create_llm_provider

# ---------------------------------------------------------------------------
# Advanced — reliability, scaling, tracing (importable from submodules too)
# ---------------------------------------------------------------------------
# from daita.core.reliability import TaskManager, CircuitBreaker, BackpressureController
# from daita.core.scaling import AgentPool, LoadBalancer, create_agent_pool
# from daita.core.tracing import get_trace_manager
# from daita.core.interfaces import AgentABC, LLMProvider

__all__ = [
    # Primary interfaces
    "Agent",
    "BaseAgent",
    "ConversationHistory",
    "Workflow",
    "RelayManager",
    # Tool system
    "tool",
    "AgentTool",
    "ToolRegistry",
    # Plugins
    "postgresql",
    "mysql",
    "mongodb",
    "rest",
    "s3",
    "slack",
    "elasticsearch",
    "redis_messaging",
    # Configuration
    "AgentConfig",
    "RetryPolicy",
    "RetryStrategy",
    # Focus DSL
    "apply_focus",
    # Exceptions
    "DaitaError",
    "AgentError",
    "LLMError",
    "ConfigError",
    "PluginError",
    "WorkflowError",
    "TransientError",
    "RetryableError",
    "PermanentError",
    "RateLimitError",
    "AuthenticationError",
    "ValidationError",
    "FocusDSLError",
    # LLM
    "create_llm_provider",
    # Version
    "__version__",
]
