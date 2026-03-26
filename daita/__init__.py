"""
Daita Agents — A data focused framework for building production ready AI agents.

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

__version__ = "0.12.1"

# ---------------------------------------------------------------------------
# Core
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
from .plugins import postgresql, mysql, mongodb, rest, s3, slack, elasticsearch, sqlite
from .plugins.redis_messaging import redis_messaging
from .plugins.base import BasePlugin, LifecyclePlugin

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from .config.base import AgentConfig, RetryPolicy, RetryStrategy

# ---------------------------------------------------------------------------
# Watch system — data source monitoring via @agent.watch()
# ---------------------------------------------------------------------------
from .core.watch import WatchEvent

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
    DataQualityError,
)

# ---------------------------------------------------------------------------
# Data quality enforcement — ItemAssertion + query_checked()
# ---------------------------------------------------------------------------
from .core.assertions import ItemAssertion

# ---------------------------------------------------------------------------
# LLM factory — for explicit provider construction
# ---------------------------------------------------------------------------
from .llm.factory import create_llm_provider

# ---------------------------------------------------------------------------
# Advanced — reliability, scaling, tracing (importable from submodules too)
# ---------------------------------------------------------------------------
# from daita.core.reliability import TaskManager, CircuitBreaker, BackpressureController
# from daita.core.scaling import AgentPool, LoadBalancer, create_agent_pool
# from daita.core.interfaces import AgentABC, LLMProvider
from .core.tracing import get_trace_manager, configure_tracing

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
    # Tracing
    "configure_tracing",
    "get_trace_manager",
    # Plugins
    "postgresql",
    "mysql",
    "mongodb",
    "sqlite",
    "rest",
    "s3",
    "slack",
    "elasticsearch",
    "redis_messaging",
    "BasePlugin",
    "LifecyclePlugin",
    # Configuration
    "AgentConfig",
    "RetryPolicy",
    "RetryStrategy",
    # Watch system
    "WatchEvent",
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
    "DataQualityError",
    # Assertions
    "ItemAssertion",
    # LLM
    "create_llm_provider",
    # Version
    "__version__",
]
