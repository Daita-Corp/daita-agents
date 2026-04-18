"""
Graph traversal and analysis algorithms.

All functions operate on a nx.MultiDiGraph returned by backend.load_graph() or
backend.subgraph(). Algorithms are stateless — they take a graph and return
results with no backend dependency.

Every traversal accepts an optional ``edge_types`` filter. When supplied, only
edges whose ``edge_type`` falls within the set are followed. The filter is
implemented once in ``_filter_by_edge_types`` as a NetworkX edge-subgraph view
(O(|E|) with no copy).
"""

from __future__ import annotations

from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

if TYPE_CHECKING:
    import networkx as nx

    from .backend import GraphBackend
    from .models import AgentGraphEdge, AgentGraphNode

from .models import EdgeType

# Default edge-type set for lineage-semantic traversals. Excludes structural
# edges (HAS_COLUMN, INDEXED_BY, COVERS, REFERENCES, PART_OF) so lineage tools
# don't accidentally walk into Column / Index nodes. REFERENCES is structural
# in catalog terms even though it implies data flow — callers that want FK-aware
# lineage pass a custom ``edge_types`` including REFERENCES.
LINEAGE_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {
        EdgeType.READS,
        EdgeType.WRITES,
        EdgeType.TRANSFORMS,
        EdgeType.SYNCS_TO,
        EdgeType.DERIVED_FROM,
        EdgeType.TRIGGERS,
        EdgeType.CALLS,
        EdgeType.PRODUCES,
    }
)


def _normalize_edge_types(
    edge_types: Optional[Iterable[EdgeType | str]],
) -> Optional[Set[str]]:
    """Normalize an edge-type filter to a set of string values or None."""
    if edge_types is None:
        return None
    out: Set[str] = set()
    for et in edge_types:
        if isinstance(et, EdgeType):
            out.add(et.value)
        else:
            out.add(str(et))
    return out


def _filter_by_edge_types(
    graph: "nx.MultiDiGraph",
    edge_types: Optional[Iterable[EdgeType | str]],
) -> "nx.MultiDiGraph":
    """
    Return a read-only edge-subgraph view restricted to the given edge types.

    Returns the original graph unchanged when ``edge_types`` is None.
    """
    wanted = _normalize_edge_types(edge_types)
    if wanted is None:
        return graph

    keep = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_type = data.get("data", {}).get("edge_type")
        if edge_type in wanted:
            keep.append((u, v, key))
    return graph.edge_subgraph(keep)


def traverse(
    graph: "nx.MultiDiGraph",
    start_node_id: str,
    direction: str = "downstream",
    max_depth: int = 10,
    edge_types: Optional[Iterable[EdgeType | str]] = None,
) -> Any:
    """
    BFS traversal from a starting node.

    direction: "downstream" (follows edges forward), "upstream" (follows edges
               backward), "both" (returns dict with both directions)
    edge_types: Optional iterable of EdgeType values. When set, traversal only
                follows edges whose edge_type is in the set. Nodes reachable
                only via excluded edge types are not visited.

    Returns list of node data dicts in traversal order (excluding start node),
    or {"downstream": [...], "upstream": [...]} when direction is "both".
    """
    g = _filter_by_edge_types(graph, edge_types)

    if direction == "both":
        return {
            "downstream": _bfs(g, start_node_id, reverse=False, max_depth=max_depth),
            "upstream": _bfs(g, start_node_id, reverse=True, max_depth=max_depth),
        }

    return _bfs(
        g, start_node_id, reverse=(direction == "upstream"), max_depth=max_depth
    )


def _bfs(
    graph: "nx.MultiDiGraph",
    start: str,
    reverse: bool,
    max_depth: int,
) -> List[Dict[str, Any]]:
    # copy=False returns a read-only view — O(1) vs O(N+E) for a full copy
    g = graph.reverse(copy=False) if reverse else graph
    if start not in g:
        return []
    visited: Set[str] = set()
    queue: deque = deque([(start, 0)])
    results = []

    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited or depth > max_depth:
            continue
        visited.add(node_id)
        if node_id != start:
            node_data = g.nodes[node_id].get("data", {"node_id": node_id})
            results.append(node_data)
        for neighbor in g.successors(node_id):
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

    return results


def impact_analysis(
    graph: "nx.MultiDiGraph",
    node_id: str,
    edge_types: Optional[Iterable[EdgeType | str]] = None,
) -> Dict[str, Any]:
    """
    Find all downstream nodes affected by a change to node_id.
    Returns nodes ranked by cumulative impact_weight along the path.

    Uses a single-pass shortest-path computation (one BFS from node_id) rather
    than a separate nx.shortest_path call per descendant. For MultiDiGraph edges
    between the same node pair, the maximum impact_weight is used (conservative
    — assumes the highest-impact relationship drives the risk score).

    When ``edge_types`` is set, only edges of those types contribute to the
    impact — useful for scoping impact to lineage-only edges (the default
    LineagePlugin behavior).
    """
    import networkx as nx

    g = _filter_by_edge_types(graph, edge_types)
    affected = []

    try:
        all_paths = nx.single_source_shortest_path(g, node_id)
    except nx.NodeNotFound:
        all_paths = {}

    for successor, path in all_paths.items():
        if successor == node_id:
            continue

        cumulative_impact = 1.0
        for i in range(len(path) - 1):
            # graph[u][v] returns {key: attr_dict} for all edges between u and v.
            # Take the maximum impact_weight across all edge types on this hop.
            edges_on_hop = g[path[i]][path[i + 1]]
            weight = max(
                (
                    attrs.get("data", {}).get("impact_weight", 1.0)
                    for attrs in edges_on_hop.values()
                ),
                default=1.0,
            )
            cumulative_impact *= weight

        node_data = g.nodes[successor].get("data", {"node_id": successor})
        affected.append(
            {
                "node": node_data,
                "cumulative_impact": round(cumulative_impact, 3),
                "path_length": len(path) - 1,
            }
        )

    affected.sort(key=lambda x: x["cumulative_impact"], reverse=True)

    risk = "LOW"
    if any(a["cumulative_impact"] > 0.7 for a in affected):
        risk = "HIGH"
    elif any(a["cumulative_impact"] > 0.4 for a in affected):
        risk = "MEDIUM"

    return {
        "source_node": node_id,
        "affected_count": len(affected),
        "risk_level": risk,
        "affected_nodes": affected,
    }


def find_paths(
    graph: "nx.MultiDiGraph",
    from_node_id: str,
    to_node_id: str,
    edge_types: Optional[Iterable[EdgeType | str]] = None,
    cutoff: Optional[int] = None,
) -> List[List[str]]:
    """Find all simple paths between two nodes.

    ``cutoff`` caps path length (number of edges). NetworkX's default behavior
    (no cutoff) can explode on dense graphs, so callers reaching for this tool
    typically want a bound.
    """
    import networkx as nx

    g = _filter_by_edge_types(graph, edge_types)
    try:
        return list(nx.all_simple_paths(g, from_node_id, to_node_id, cutoff=cutoff))
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return []


def shortest_path(
    graph: "nx.MultiDiGraph",
    from_node_id: str,
    to_node_id: str,
    edge_types: Optional[Iterable[EdgeType | str]] = None,
) -> Optional[List[str]]:
    """
    Single shortest path from ``from_node_id`` to ``to_node_id``.

    Returns the sequence of node IDs (inclusive of endpoints) or None when
    unreachable under the given ``edge_types`` filter.
    """
    import networkx as nx

    g = _filter_by_edge_types(graph, edge_types)
    try:
        return nx.shortest_path(g, from_node_id, to_node_id)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return None


def connected_component(
    graph: "nx.MultiDiGraph",
    node_id: str,
    edge_types: Optional[Iterable[EdgeType | str]] = None,
) -> Set[str]:
    """
    Every node reachable from ``node_id`` in either direction, restricted to
    the given ``edge_types``. Weakly-connected component under the filter.
    """
    import networkx as nx

    g = _filter_by_edge_types(graph, edge_types)
    if node_id not in g:
        return set()
    undirected = g.to_undirected(as_view=True)
    return set(nx.node_connected_component(undirected, node_id))


def ancestors(
    graph: "nx.MultiDiGraph",
    node_id: str,
    edge_types: Optional[Iterable[EdgeType | str]] = None,
    max_depth: Optional[int] = None,
) -> Set[str]:
    """
    Transitive closure upstream of ``node_id``.

    When ``max_depth`` is None, returns all ancestors (equivalent to
    ``nx.ancestors`` on the filtered graph). Otherwise bounded BFS.
    """
    g = _filter_by_edge_types(graph, edge_types)
    if node_id not in g:
        return set()

    if max_depth is None:
        import networkx as nx

        return set(nx.ancestors(g, node_id))

    visited: Set[str] = set()
    queue: deque = deque([(node_id, 0)])
    while queue:
        cur, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for pred in g.predecessors(cur):
            if pred not in visited and pred != node_id:
                visited.add(pred)
                queue.append((pred, depth + 1))
    return visited


def descendants(
    graph: "nx.MultiDiGraph",
    node_id: str,
    edge_types: Optional[Iterable[EdgeType | str]] = None,
    max_depth: Optional[int] = None,
) -> Set[str]:
    """
    Transitive closure downstream of ``node_id``.

    When ``max_depth`` is None, returns all descendants. Otherwise bounded BFS.
    """
    g = _filter_by_edge_types(graph, edge_types)
    if node_id not in g:
        return set()

    if max_depth is None:
        import networkx as nx

        return set(nx.descendants(g, node_id))

    visited: Set[str] = set()
    queue: deque = deque([(node_id, 0)])
    while queue:
        cur, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for succ in g.successors(cur):
            if succ not in visited and succ != node_id:
                visited.add(succ)
                queue.append((succ, depth + 1))
    return visited


# ---------------------------------------------------------------------------
# Backend-agnostic subgraph default
# ---------------------------------------------------------------------------


async def default_subgraph(
    backend: "GraphBackend",
    root: str,
    direction: str = "both",
    edge_types: Optional[Iterable[EdgeType | str]] = None,
    max_depth: int = 5,
) -> "nx.MultiDiGraph":
    """
    Default backend-agnostic implementation of ``subgraph``.

    Performs a breadth-first expansion using ``backend.get_edges`` at each hop.
    Backends with native subgraph support (e.g. DynamoDB's partition-scoped
    BFS) should override ``subgraph`` for efficiency; this implementation is
    correct for any backend but does one ``get_edges`` call per frontier node.
    """
    import networkx as nx

    g = nx.MultiDiGraph()
    frontier: Set[str] = {root}
    visited: Set[str] = set()

    # Ensure the root node is included even if it has no incident edges in scope.
    root_node = await backend.get_node(root)
    if root_node is not None:
        g.add_node(root, data=root_node.model_dump())
    else:
        g.add_node(root, data={"node_id": root})

    for _ in range(max_depth):
        next_frontier: Set[str] = set()
        for node_id in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)

            if direction in ("downstream", "both"):
                out_edges = await backend.get_edges(
                    from_node_id=node_id, edge_types=edge_types
                )
                for edge in out_edges:
                    if edge.to_node_id not in g:
                        nbr = await backend.get_node(edge.to_node_id)
                        data = nbr.model_dump() if nbr else {"node_id": edge.to_node_id}
                        g.add_node(edge.to_node_id, data=data)
                    g.add_edge(
                        edge.from_node_id,
                        edge.to_node_id,
                        key=edge.edge_id,
                        data=edge.model_dump(),
                    )
                    if edge.to_node_id not in visited:
                        next_frontier.add(edge.to_node_id)

            if direction in ("upstream", "both"):
                in_edges = await backend.get_edges(
                    to_node_id=node_id, edge_types=edge_types
                )
                for edge in in_edges:
                    if edge.from_node_id not in g:
                        nbr = await backend.get_node(edge.from_node_id)
                        data = (
                            nbr.model_dump() if nbr else {"node_id": edge.from_node_id}
                        )
                        g.add_node(edge.from_node_id, data=data)
                    g.add_edge(
                        edge.from_node_id,
                        edge.to_node_id,
                        key=edge.edge_id,
                        data=edge.model_dump(),
                    )
                    if edge.from_node_id not in visited:
                        next_frontier.add(edge.from_node_id)

        frontier = next_frontier - visited
        if not frontier:
            break

    return g
