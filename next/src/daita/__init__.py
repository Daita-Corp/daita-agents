"""Daita autonomous-agent v2 replacement package."""

from .agent import Agent
from .adapters import LocalDirectorySource, SQLiteSource

__version__ = "2.0.0a0"

__all__ = ["Agent", "LocalDirectorySource", "SQLiteSource", "__version__"]
