"""
GraphBackend protocol, registry, and backend auto-selection.

Default backend (no configuration required):
  LocalGraphBackend — NetworkX + .daita/graph/{graph_type}.json

To use a different backend, register a factory once at application startup:

    from daita.core.graph.backend import register_backend_factory
    from my_backends import Neo4jBackend

    register_backend_factory(lambda graph_type: Neo4jBackend(graph_type))

Developers never call auto_select_backend() directly. LineagePlugin and CatalogPlugin
call it during initialize() if no backend was provided at construction time.
"""

from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import networkx as nx

from .models import AgentGraphNode, AgentGraphEdge

# Module-level registry. Set once at startup via register_backend_factory().
# None means use LocalGraphBackend.
_BACKEND_FACTORY: Optional[Callable[[str], "GraphBackend"]] = None


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

    async def load_graph(self) -> "nx.MultiDiGraph":
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


def register_backend_factory(
    factory: Optional[Callable[[str], "GraphBackend"]],
) -> None:
    """
    Register a factory that creates graph backends.

    Called once at application startup to inject a backend. Pass None to reset
    to the default (LocalGraphBackend), which is useful in tests.

    Args:
        factory: Callable that receives graph_type (str) and returns a GraphBackend.
                 Pass None to clear the registered factory and revert to default.

    Example — use a custom backend everywhere without modifying agent code:
        register_backend_factory(lambda graph_type: Neo4jBackend(graph_type))
    """
    global _BACKEND_FACTORY
    _BACKEND_FACTORY = factory


def auto_select_backend(graph_type: str = "lineage") -> "GraphBackend":
    """
    Return the appropriate graph backend for the current environment.

    Uses the factory registered via register_backend_factory() if one has been
    set, otherwise returns LocalGraphBackend (the default for local development).

    Developers never call this directly. LineagePlugin and CatalogPlugin call
    it during initialize() if no backend was passed to the constructor.
    """
    if _BACKEND_FACTORY is not None:
        return _BACKEND_FACTORY(graph_type)

    from .local_backend import LocalGraphBackend

    return LocalGraphBackend(graph_type=graph_type)
