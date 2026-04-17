"""
Generic graph-query tools for agents.

These are edge-type-agnostic primitives for structural graph exploration —
use them when the lineage-specific defaults in LineagePlugin aren't what
you want (e.g. walking only :REFERENCES edges across a catalog, or asking
"is X connected to Y at all" regardless of edge type).

Registration is opt-in: an agent only gains these tools if you explicitly
call ``register_graph_tools(agent, backend=...)``. This keeps the agent's
tool list focused for workflows that don't need low-level graph primitives.

Exposed tools:

  * ``graph_subgraph(root, depth, edge_types?, direction?)`` —
    Return nodes + edges reachable within ``depth`` hops of ``root``.
    Single exploration primitive; covers neighbors (depth=1) and larger
    bounded expansions.

  * ``graph_shortest_path(from_id, to_id, edge_types?)`` —
    Return the shortest path between two nodes, or null when unreachable.
    Answers "are X and Y connected, and how?" across any edge type.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ..exceptions import ValidationError
from .models import EdgeType

if TYPE_CHECKING:
    from ..tools import AgentTool
    from .backend import GraphBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge-type parsing
# ---------------------------------------------------------------------------


def _parse_edge_types(
    raw: Optional[Iterable[str]],
) -> Optional[List[EdgeType]]:
    """Parse an optional list of edge-type strings into EdgeType values.

    Returns None when ``raw`` is empty / None so callers see "no filter"
    rather than "filter matching nothing".
    """
    if not raw:
        return None
    out: List[EdgeType] = []
    for value in raw:
        try:
            out.append(EdgeType(value))
        except ValueError:
            raise ValidationError(
                f"Unknown edge_type '{value}'. Valid values: "
                f"{sorted(e.value for e in EdgeType)}"
            )
    return out


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _subgraph_to_dict(graph: Any) -> Dict[str, Any]:
    """Convert a NetworkX MultiDiGraph into a JSON-safe dict of nodes + edges."""
    nodes = []
    for node_id in graph.nodes():
        data = graph.nodes[node_id].get("data") or {"node_id": node_id}
        nodes.append(data)

    edges = []
    for u, v, key, edge_data in graph.edges(keys=True, data=True):
        raw = edge_data.get("data")
        if raw is not None:
            edges.append(raw)
        else:
            edges.append({"from_node_id": u, "to_node_id": v, "edge_id": key})

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_graph_subgraph(
    backend: "GraphBackend",
    default_edge_types: Optional[List[EdgeType]],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    root = args.get("root")
    if not root:
        raise ValidationError("root is required")

    direction = args.get("direction", "both")
    if direction not in ("upstream", "downstream", "both"):
        raise ValidationError(
            f"direction must be one of 'upstream', 'downstream', 'both' (got {direction!r})"
        )

    max_depth = int(args.get("max_depth") or 3)
    edge_types = _parse_edge_types(args.get("edge_types")) or default_edge_types

    graph = await backend.subgraph(
        root=root,
        direction=direction,
        edge_types=edge_types,
        max_depth=max_depth,
    )
    payload = _subgraph_to_dict(graph)
    payload.update(
        {
            "root": root,
            "direction": direction,
            "max_depth": max_depth,
            "node_count": len(payload["nodes"]),
            "edge_count": len(payload["edges"]),
        }
    )
    return payload


async def _handle_graph_shortest_path(
    backend: "GraphBackend",
    default_edge_types: Optional[List[EdgeType]],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    from_id = args.get("from_id")
    to_id = args.get("to_id")
    if not from_id or not to_id:
        raise ValidationError("from_id and to_id are required")

    edge_types = _parse_edge_types(args.get("edge_types")) or default_edge_types

    # Bound the search radius by materializing a subgraph around the source.
    # This keeps the tool tractable on very large graphs while still returning
    # None for genuinely unreachable pairs.
    max_depth = int(args.get("max_depth") or 10)
    graph = await backend.subgraph(
        root=from_id,
        direction="downstream",
        edge_types=edge_types,
        max_depth=max_depth,
    )

    from .algorithms import shortest_path as _shortest_path

    path = _shortest_path(graph, from_id, to_id, edge_types=edge_types)
    return {
        "from_id": from_id,
        "to_id": to_id,
        "path": path,
        "length": (len(path) - 1) if path else None,
        "reachable": path is not None,
    }


# ---------------------------------------------------------------------------
# Tool builder + registration helper
# ---------------------------------------------------------------------------


_GRAPH_TOOL_DEFAULTS = {
    "category": "graph",
    "source": "core",
    "plugin_name": "GraphQuery",
}


def build_graph_tools(
    backend: "GraphBackend",
    default_edge_types: Optional[Iterable[EdgeType]] = None,
) -> List["AgentTool"]:
    """Build the list of generic graph-query AgentTools bound to ``backend``.

    ``default_edge_types`` sets the filter used when a tool invocation omits
    the ``edge_types`` argument. Pass ``LINEAGE_EDGE_TYPES`` to make these
    tools lineage-defaulted, or leave as None to follow every edge.
    """
    from ..tools import AgentTool

    default_list: Optional[List[EdgeType]] = (
        list(default_edge_types) if default_edge_types else None
    )

    edge_type_enum = sorted(e.value for e in EdgeType)

    return [
        AgentTool(
            name="graph_subgraph",
            description=(
                "Return every node and edge reachable within a bounded radius "
                "around a root node. Covers neighbor queries (depth=1) and "
                "larger bounded expansions. Use to explore structural "
                "relationships (FK graphs, schema hierarchies) that lineage "
                "tools exclude by default."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Root node ID (e.g. 'table:orders', 'column:orders.id').",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["upstream", "downstream", "both"],
                        "description": "Traversal direction. Default 'both'.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum hops from the root (default 3).",
                    },
                    "edge_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": edge_type_enum},
                        "description": (
                            "Optional allowlist of edge types to follow. "
                            "Omit to follow every edge."
                        ),
                    },
                },
                "required": ["root"],
            },
            handler=lambda args, b=backend, d=default_list: _handle_graph_subgraph(
                b, d, args
            ),
            timeout_seconds=60,
            **_GRAPH_TOOL_DEFAULTS,
        ),
        AgentTool(
            name="graph_shortest_path",
            description=(
                "Find the shortest path between two nodes, restricted to "
                "optional edge types. Returns the ordered list of node IDs, "
                "or null when the target is unreachable within max_depth hops."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "from_id": {
                        "type": "string",
                        "description": "Starting node ID.",
                    },
                    "to_id": {
                        "type": "string",
                        "description": "Target node ID.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum search radius (default 10).",
                    },
                    "edge_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": edge_type_enum},
                        "description": (
                            "Optional allowlist of edge types to follow. "
                            "Omit to follow every edge."
                        ),
                    },
                },
                "required": ["from_id", "to_id"],
            },
            handler=lambda args, b=backend, d=default_list: _handle_graph_shortest_path(
                b, d, args
            ),
            timeout_seconds=60,
            **_GRAPH_TOOL_DEFAULTS,
        ),
    ]


def register_graph_tools(
    agent: Any,
    backend: Optional["GraphBackend"] = None,
    default_edge_types: Optional[Iterable[EdgeType]] = None,
    graph_type: str = "lineage",
) -> List["AgentTool"]:
    """Attach the generic graph-query tools to ``agent``.

    Args:
        agent: An ``Agent`` instance (or anything exposing a
            ``tool_registry`` with ``register_many``).
        backend: Graph backend the tools should query. Defaults to
            ``auto_select_backend(graph_type)``.
        default_edge_types: Optional default filter applied when a tool
            invocation omits ``edge_types``. Pass ``LINEAGE_EDGE_TYPES``
            for lineage-defaulted behavior.
        graph_type: Graph namespace used when ``backend`` is None.

    Returns:
        The list of registered AgentTool instances.
    """
    if backend is None:
        from .backend import auto_select_backend

        backend = auto_select_backend(graph_type=graph_type)

    tools = build_graph_tools(backend, default_edge_types=default_edge_types)

    registry = getattr(agent, "tool_registry", None)
    if registry is None or not hasattr(registry, "register_many"):
        raise TypeError("register_graph_tools requires an agent with a tool_registry.")
    registry.register_many(tools)
    logger.info("Registered %d graph-query tools on agent", len(tools))
    return tools
