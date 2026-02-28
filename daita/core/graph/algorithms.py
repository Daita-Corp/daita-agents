"""
Graph traversal and analysis algorithms.

All functions operate on a nx.MultiDiGraph returned by backend.load_graph().
Algorithms are stateless — they take a graph and return results with no
backend dependency.
"""
from collections import deque
from typing import Any, Dict, List, Set

import networkx as nx


def traverse(
    graph: nx.MultiDiGraph,
    start_node_id: str,
    direction: str = "downstream",
    max_depth: int = 10,
) -> Any:
    """
    BFS traversal from a starting node.

    direction: "downstream" (follows edges forward), "upstream" (follows edges
               backward), "both" (returns dict with both directions)

    Returns list of node data dicts in traversal order (excluding start node),
    or {"downstream": [...], "upstream": [...]} when direction is "both".
    """
    if direction == "both":
        return {
            "downstream": _bfs(graph, start_node_id, reverse=False, max_depth=max_depth),
            "upstream": _bfs(graph, start_node_id, reverse=True, max_depth=max_depth),
        }

    return _bfs(graph, start_node_id, reverse=(direction == "upstream"), max_depth=max_depth)


def _bfs(
    graph: nx.MultiDiGraph,
    start: str,
    reverse: bool,
    max_depth: int,
) -> List[Dict[str, Any]]:
    # copy=False returns a read-only view — O(1) vs O(N+E) for a full copy
    g = graph.reverse(copy=False) if reverse else graph
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
    graph: nx.MultiDiGraph,
    node_id: str,
) -> Dict[str, Any]:
    """
    Find all downstream nodes affected by a change to node_id.
    Returns nodes ranked by cumulative impact_weight along the path.

    Uses a single-pass shortest-path computation (one BFS from node_id) rather
    than a separate nx.shortest_path call per descendant. For MultiDiGraph edges
    between the same node pair, the maximum impact_weight is used (conservative
    — assumes the highest-impact relationship drives the risk score).
    """
    affected = []

    try:
        all_paths = nx.single_source_shortest_path(graph, node_id)
    except nx.NodeNotFound:
        all_paths = {}

    for successor, path in all_paths.items():
        if successor == node_id:
            continue

        cumulative_impact = 1.0
        for i in range(len(path) - 1):
            # graph[u][v] returns {key: attr_dict} for all edges between u and v.
            # Take the maximum impact_weight across all edge types on this hop.
            edges_on_hop = graph[path[i]][path[i + 1]]
            weight = max(
                (attrs.get("data", {}).get("impact_weight", 1.0) for attrs in edges_on_hop.values()),
                default=1.0,
            )
            cumulative_impact *= weight

        node_data = graph.nodes[successor].get("data", {"node_id": successor})
        affected.append({
            "node": node_data,
            "cumulative_impact": round(cumulative_impact, 3),
            "path_length": len(path) - 1,
        })

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
    graph: nx.MultiDiGraph,
    from_node_id: str,
    to_node_id: str,
) -> List[List[str]]:
    """Find all simple paths between two nodes."""
    try:
        return list(nx.all_simple_paths(graph, from_node_id, to_node_id))
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return []
