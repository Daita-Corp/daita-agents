"""Daita autonomous-agent v2 replacement package."""

from .agent import Agent
from .adapters import LocalDirectorySource, SQLiteSource
from .hosting import AgentHost

__version__ = "2.0.0a0"

__all__ = [
    "Agent",
    "AgentHost",
    "LocalDirectorySource",
    "SQLiteSource",
    "__version__",
]
