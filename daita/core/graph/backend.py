"""
GraphBackend protocol and backend auto-selection.

The backend is chosen automatically based on runtime environment:
- Local dev / daita test -> LocalGraphBackend (NetworkX + .daita/graph/*.json)
- Daita cloud (Lambda)  -> DynamoGraphBackend (not yet implemented; falls back to local)

Developers never call auto_select_backend() directly. LineagePlugin calls it
during initialize() if no backend was provided at construction time.
"""

from typing import List, Optional, Protocol, runtime_checkable

import networkx as nx

from .models import AgentGraphNode, AgentGraphEdge


@runtime_checkable
class GraphBackend(Protocol):
    """
    Storage protocol for the agent graph system.

    Implement this protocol to provide a custom storage backend.
    All graph traversal logic runs on top of NetworkX regardless of
    which backend is used for persistence.
    """

    async def add_node(self, node: AgentGraphNode) -> None:
        """Add or update a node. Upsert semantics — safe to call repeatedly."""
        ...

    async def add_edge(self, edge: AgentGraphEdge) -> None:
        """Add or update an edge. Upsert semantics — safe to call repeatedly."""
        ...

    async def get_node(self, node_id: str) -> Optional[AgentGraphNode]:
        """Retrieve a single node by ID. Returns None if not found."""
        ...

    async def get_edges(
        self,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
    ) -> List[AgentGraphEdge]:
        """
        Retrieve edges filtered by from_node_id and/or to_node_id.
        Both parameters are optional — omitting both returns all edges.
        """
        ...

    async def load_graph(self) -> nx.MultiDiGraph:
        """
        Load the full graph into a NetworkX MultiDiGraph for algorithm execution.
        Each node carries its AgentGraphNode as the 'data' attribute.
        Each edge carries its AgentGraphEdge as the 'data' attribute.
        Multiple edge types between the same node pair are stored as distinct edges.
        """
        ...

    async def delete_node(self, node_id: str) -> None:
        """Remove a node and all its connected edges."""
        ...

    async def update_node_properties(self, node_id: str, properties: dict) -> None:
        """Merge new properties into an existing node."""
        ...

    async def prune_stale(self, max_age_seconds: int) -> dict:
        """
        Remove nodes and edges whose last_seen is older than max_age_seconds.

        Intended to be called at the end of a full scan run. Any node/edge not
        touched during that run (last_seen not refreshed) will be older than the
        scan interval and gets removed.

        Returns a summary: {"removed_nodes": [...], "removed_edges": [...]}
        """
        ...


def auto_select_backend(graph_type: str = "lineage") -> "GraphBackend":
    """
    Select the appropriate storage backend based on the runtime environment.

    Currently only LocalGraphBackend is implemented. DynamoGraphBackend will
    be added once the front-end visualization surface is ready.
    """
    from .local_backend import LocalGraphBackend

    return LocalGraphBackend(graph_type=graph_type)
