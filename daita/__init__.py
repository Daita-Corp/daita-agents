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
- tool                  — Define local model-visible functions
- Plugins               — Extension-first database/API integrations
- AgentConfig           — Configure retry policies, LLM settings, and more
"""

__version__ = "0.20.0"

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
from .agents.agent import Agent
from .agents.base import BaseAgent
from .agents.conversation import ConversationHistory

from .core.tools import tool

# ---------------------------------------------------------------------------
# Plugins — extension-first database and API integrations
# ---------------------------------------------------------------------------
from .plugins import postgresql, mysql, mongodb, rest, s3, slack, elasticsearch, sqlite
from .plugins.redis_messaging import redis_messaging
from .plugins.base import (
    BasePlugin,
    ConnectorPlugin,
    DomainServicePlugin,
    EmptySecretProvider,
    ObservabilityPlugin,
    PluginContext,
    RuntimeExtensionPlugin,
    SecretProvider,
    ServiceRegistry,
    SkillPlugin,
    WorkerProviderPlugin,
)
from .plugins.manifest import PluginKind, PluginManifest
from .plugins.registry import ExtensionRegistry, RegistryDiagnostic

# ---------------------------------------------------------------------------
# Skills — composable units of agent capability
# ---------------------------------------------------------------------------
from .skills import BaseSkill, Skill

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
    SkillError,
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
# Embedding providers — subclass BaseEmbeddingProvider for custom providers
# ---------------------------------------------------------------------------
from .embeddings import BaseEmbeddingProvider

from .core.tracing import get_trace_manager, configure_tracing, set_trace_context

__all__ = [
    # Primary interfaces
    "Agent",
    "BaseAgent",
    "ConversationHistory",
    # Tool system
    "tool",
    # Tracing
    "configure_tracing",
    "get_trace_manager",
    "set_trace_context",
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
    "ConnectorPlugin",
    "DomainServicePlugin",
    "EmptySecretProvider",
    "ObservabilityPlugin",
    "PluginContext",
    "PluginKind",
    "PluginManifest",
    "RuntimeExtensionPlugin",
    "SecretProvider",
    "ServiceRegistry",
    "SkillPlugin",
    "WorkerProviderPlugin",
    "ExtensionRegistry",
    "RegistryDiagnostic",
    "BaseSkill",
    "Skill",
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
    "SkillError",
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
    # Embeddings
    "BaseEmbeddingProvider",
    # Version
    "__version__",
]
