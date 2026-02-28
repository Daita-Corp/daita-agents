"""
Agent graph system — core infrastructure for graph-based persistence.

Used by lineage.py and catalog.py to persist data flows and schema metadata
across agent runs. Developers do not interact with this module directly;
LineagePlugin and CatalogPlugin consume it automatically.
"""
from .models import AgentGraphNode, AgentGraphEdge, NodeType, EdgeType
from .backend import GraphBackend, auto_select_backend
from .local_backend import LocalGraphBackend
from .algorithms import traverse, impact_analysis, find_paths

__all__ = [
    "AgentGraphNode",
    "AgentGraphEdge",
    "NodeType",
    "EdgeType",
    "GraphBackend",
    "auto_select_backend",
    "LocalGraphBackend",
    "traverse",
    "impact_analysis",
    "find_paths",
]
