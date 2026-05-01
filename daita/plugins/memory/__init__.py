"""
Memory plugin for DAITA agents.

Provides persistent, semantic memory with automatic local/cloud detection.
"""

from .memory_plugin import MemoryPlugin, memory
from .local_backend import LocalMemoryBackend
from .working_memory import WorkingMemory
from .memory_graph import MemoryGraph
from .graph_models import (
    MemoryEdgeType,
    MemoryGraphEdge,
    MemoryGraphNode,
    MemoryNodeType,
)

__all__ = [
    "MemoryPlugin",
    "LocalMemoryBackend",
    "WorkingMemory",
    "MemoryGraph",
    "MemoryGraphNode",
    "MemoryGraphEdge",
    "MemoryNodeType",
    "MemoryEdgeType",
    "memory",
]
