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

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import networkx as nx

from .models import AgentGraphNode, AgentGraphEdge, EdgeType, NodeType

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

    Core primitives are ``iter_nodes`` and ``iter_edges`` (streaming).
    ``find_nodes`` and ``get_edges`` have default implementations that
    materialize the iterators into lists — backends may override either
    form, but only the iterators are strictly required beyond the basic
    CRUD methods.
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

    async def iter_edges(
        self,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
        edge_types: Optional[Iterable[EdgeType]] = None,
        properties_match: Optional[dict[str, Any]] = None,
        page_size: int = 500,
    ) -> AsyncIterator[AgentGraphEdge]:
        """
        Streaming variant of get_edges. Yields one edge at a time; the
        backend handles pagination internally.

        Filter semantics:
          * ``from_node_id`` / ``to_node_id`` — endpoint constraints (either or both).
          * ``edge_types`` — accept only edges whose ``edge_type`` is in this set.
          * ``properties_match`` — accept only edges whose ``properties`` map
            contains all the given key/value pairs.

        Use ``get_edges`` when the result set is known to fit in memory.
        """
        ...

    async def get_edges(
        self,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
        edge_types: Optional[Iterable[EdgeType]] = None,
        properties_match: Optional[dict[str, Any]] = None,
    ) -> List[AgentGraphEdge]:
        """
        Retrieve edges matching the given filters.

        Default implementation materializes ``iter_edges`` into a list;
        backends may override for bulk-fetch optimizations.
        """
        ...

    async def load_graph(self) -> "nx.MultiDiGraph":
        """
        Load the full graph into a NetworkX MultiDiGraph for algorithm execution.

        This is the "offline analytics" path — prune passes, diagram export,
        full-graph reporting. Per-call traversal should use ``subgraph``
        instead, which touches only the nodes reachable within ``max_depth``.
        """
        ...

    async def subgraph(
        self,
        root: str,
        direction: str = "both",
        edge_types: Optional[Iterable[EdgeType]] = None,
        max_depth: int = 5,
    ) -> "nx.MultiDiGraph":
        """
        Load a bounded-radius subgraph around ``root`` into NetworkX.

        Replaces the ``load_graph()`` + in-memory BFS pattern for per-call
        traversals. Backend-native implementations (DynamoDB) can perform a
        breadth-first expansion with per-level edge filtering, pulling only
        the nodes reachable within ``max_depth`` rather than the whole
        partition. The default implementation in ``algorithms.subgraph`` is
        correct for any backend, just not the most efficient one.
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

    async def iter_nodes(
        self,
        node_type: NodeType,
        properties_match: Optional[dict[str, Any]] = None,
        page_size: int = 500,
    ) -> AsyncIterator[AgentGraphNode]:
        """
        Streaming variant of find_nodes. Yields one node at a time;
        the backend handles pagination internally. Use when the result
        set may exceed memory (e.g. "every Column node in the graph").
        """
        ...

    async def find_nodes(
        self,
        node_type: NodeType,
        properties_match: Optional[dict[str, Any]] = None,
    ) -> List[AgentGraphNode]:
        """
        Return every node of ``node_type`` whose properties match all key/value
        pairs in ``properties_match``. Used by the resolution layer to look up
        Table nodes by bare name across stores.

        Default implementation materializes ``iter_nodes`` into a list.
        """
        ...

    async def promote_node(self, old_id: str, new_id: str) -> None:
        """
        Rewrite every edge pointing at ``old_id`` to point at ``new_id`` and
        delete the node at ``old_id``. No-op if ``old_id`` does not exist.

        Used by unresolved-reference reconciliation: when a placeholder
        ``table:__unresolved__.<name>`` node is replaced by a canonical
        ``table:<store>.<name>`` node, its incident edges migrate atomically
        so downstream traversals see the canonical node.
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
