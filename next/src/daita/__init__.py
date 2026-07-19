"""Daita autonomous-agent v2 replacement package."""

from .agent import Agent
from .adapters import LocalDirectorySource, PostgreSQLSource, SQLiteSource
from .hosting import AgentHost
from .llm import create_llm_provider
from .security import (
    EnvironmentSecretProvider,
    SecretProvider,
    SecretReference,
    SecretResolutionError,
)

__version__ = "2.0.0a0"

__all__ = [
    "Agent",
    "AgentHost",
    "EnvironmentSecretProvider",
    "LocalDirectorySource",
    "PostgreSQLSource",
    "SecretProvider",
    "SecretReference",
    "SecretResolutionError",
    "SQLiteSource",
    "__version__",
    "create_llm_provider",
]
