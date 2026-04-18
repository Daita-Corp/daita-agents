"""
Agent graph system — core infrastructure for graph-based persistence.

Used by lineage.py and catalog.py to persist data flows and schema metadata
across agent runs. Developers do not interact with this module directly;
LineagePlugin and CatalogPlugin consume it automatically.

Generic graph-query tools (``graph_subgraph`` / ``graph_shortest_path``) are
opt-in — call ``register_graph_tools(agent, ...)`` to expose them.
"""

from .algorithms import (
    LINEAGE_EDGE_TYPES,
    ancestors,
    connected_component,
    default_subgraph,
    descendants,
    find_paths,
    impact_analysis,
    shortest_path,
    traverse,
)
from .backend import GraphBackend, auto_select_backend
from .local_backend import LocalGraphBackend
from .models import AgentGraphEdge, AgentGraphNode, EdgeType, NodeType
from .tools import build_graph_tools, register_graph_tools

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
    "shortest_path",
    "connected_component",
    "ancestors",
    "descendants",
    "default_subgraph",
    "LINEAGE_EDGE_TYPES",
    "build_graph_tools",
    "register_graph_tools",
]
