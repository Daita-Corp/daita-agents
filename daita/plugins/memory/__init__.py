"""
Memory plugin for DAITA agents.

Provides persistent, semantic memory with automatic local/cloud detection.
"""

from .memory_plugin import MemoryPlugin, memory
from .local_backend import LocalMemoryBackend

__all__ = ["MemoryPlugin", "LocalMemoryBackend", "memory"]
