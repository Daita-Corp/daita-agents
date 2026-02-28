"""
Local graph backend using NetworkX with JSON file persistence.

Stores the graph at .daita/graph/{graph_type}.json relative to the current
working directory (the daita project root). The file is created automatically
on first write.

Not suitable for concurrent writes from multiple processes. This is acceptable
for local development. Use DynamoGraphBackend in production (when available).
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import networkx as nx

from .models import AgentGraphNode, AgentGraphEdge

logger = logging.getLogger(__name__)


class LocalGraphBackend:
    """
    NetworkX graph with JSON file persistence.

    Uses MultiDiGraph so multiple edge types (READS, WRITES, TRANSFORMS, etc.)
    can coexist between the same pair of nodes. Each edge is stored under its
    deterministic edge_id as the graph key.

    The in-memory graph is cached after first load. All mutations hold an
    asyncio.Lock to prevent concurrent coroutines from racing on the
    read-modify-write cycle. All mutations write through to disk immediately
    so state survives across agent runs.
    """

    def __init__(self, graph_type: str = "lineage"):
        if not re.fullmatch(r"[a-zA-Z0-9_]+", graph_type):
            raise ValueError(
                f"Invalid graph_type {graph_type!r}. Only alphanumeric characters and underscores are allowed."
            )
        self.graph_type = graph_type
        self._graph_path = Path(".daita") / "graph" / f"{graph_type}.json"
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph: Optional[nx.MultiDiGraph] = None
        self._lock = asyncio.Lock()
        self._dirty = False

    def _load(self) -> nx.MultiDiGraph:
        """Load graph from disk into memory. Returns empty graph if file missing."""
        if self._graph is not None:
            return self._graph
        if not self._graph_path.exists():
            self._graph = nx.MultiDiGraph()
            return self._graph
        try:
            with open(self._graph_path, "r") as f:
                data = json.load(f)
            # directed=True, multigraph=True ensures correct type even when
            # loading legacy files that were saved as DiGraph (multigraph=false).
            self._graph = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Could not load graph from {self._graph_path}: {exc}. Starting fresh.")
            self._graph = nx.MultiDiGraph()
        return self._graph

    def _save(self, graph: nx.MultiDiGraph) -> None:
        """Persist graph to disk."""
        data = nx.node_link_data(graph, edges="links")
        with open(self._graph_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    async def add_node(self, node: AgentGraphNode) -> None:
        async with self._lock:
            graph = self._load()
            now = datetime.now(timezone.utc)

            if node.node_id in graph:
                existing = graph.nodes[node.node_id].get("data", {})
                incoming = node.model_dump()
                # Preserve created_at from first registration
                incoming["created_at"] = existing.get("created_at", incoming["created_at"])
                # Preserve health_score and tags unless the incoming node explicitly sets them
                if node.health_score is None:
                    incoming["health_score"] = existing.get("health_score")
                if not node.tags:
                    incoming["tags"] = existing.get("tags", [])
                # Merge properties — don't overwrite keys set by other agents
                merged_props = dict(existing.get("properties", {}))
                merged_props.update(incoming.get("properties", {}))
                incoming["properties"] = merged_props
                # Always advance updated_at and last_seen
                incoming["updated_at"] = now
                incoming["last_seen"] = now
                graph.nodes[node.node_id]["data"] = incoming
            else:
                data = node.model_dump()
                data["last_seen"] = now
                graph.add_node(node.node_id, data=data)

            self._dirty = True
            logger.debug(f"Graph: upserted node {node.node_id}")

    async def add_edge(self, edge: AgentGraphEdge) -> None:
        async with self._lock:
            graph = self._load()
            # Ensure both endpoint nodes exist as minimal stubs if not already present
            if edge.from_node_id not in graph:
                graph.add_node(edge.from_node_id, data={"node_id": edge.from_node_id})
            if edge.to_node_id not in graph:
                graph.add_node(edge.to_node_id, data={"node_id": edge.to_node_id})

            incoming = edge.model_dump()
            # Use edge_id as the MultiDiGraph key so different edge types between
            # the same node pair are stored as distinct edges, not overwritten.
            key = edge.edge_id

            existing_data = graph.get_edge_data(edge.from_node_id, edge.to_node_id, key=key)
            if existing_data is not None:
                existing = existing_data.get("data", {})
                # Preserve the original registration timestamp
                incoming["timestamp"] = existing.get("timestamp", incoming["timestamp"])
                # Merge properties
                merged_props = dict(existing.get("properties", {}))
                merged_props.update(incoming.get("properties", {}))
                incoming["properties"] = merged_props
                graph.edges[edge.from_node_id, edge.to_node_id, key]["data"] = incoming
            else:
                graph.add_edge(edge.from_node_id, edge.to_node_id, key=key, data=incoming)

            self._dirty = True
            logger.debug(f"Graph: upserted edge {edge.edge_id}")

    async def get_node(self, node_id: str) -> Optional[AgentGraphNode]:
        graph = self._load()
        if node_id not in graph:
            return None
        raw = graph.nodes[node_id].get("data", {})
        if not raw.get("node_type"):
            return None
        return AgentGraphNode(**raw)

    async def get_edges(
        self,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
    ) -> List[AgentGraphEdge]:
        graph = self._load()
        edges = []
        for u, v, _key, edge_data in graph.edges(keys=True, data=True):
            raw = edge_data.get("data", {})
            if not raw:
                continue
            if from_node_id and u != from_node_id:
                continue
            if to_node_id and v != to_node_id:
                continue
            edges.append(AgentGraphEdge(**raw))
        return edges

    async def load_graph(self) -> nx.MultiDiGraph:
        return self._load()

    async def flush(self) -> None:
        """
        Persist any pending mutations to disk.

        add_node and add_edge set a dirty flag instead of saving inline so that
        batch operations (register_flow, register_pipeline) can amortize the JSON
        serialization cost over multiple writes. Call flush() at the end of a
        logical batch to guarantee durability.
        """
        async with self._lock:
            if self._dirty:
                self._save(self._load())
                self._dirty = False

    async def delete_node(self, node_id: str) -> None:
        async with self._lock:
            graph = self._load()
            if node_id in graph:
                graph.remove_node(node_id)
                self._save(graph)

    async def update_node_properties(self, node_id: str, properties: dict) -> None:
        async with self._lock:
            graph = self._load()
            if node_id in graph:
                existing = graph.nodes[node_id].get("data", {})
                existing.setdefault("properties", {}).update(properties)
                existing["updated_at"] = datetime.now(timezone.utc).isoformat()
                graph.nodes[node_id]["data"] = existing
                self._save(graph)

    async def prune_stale(self, max_age_seconds: int) -> dict:
        """
        Remove nodes and edges whose last_seen is older than max_age_seconds.

        Call this at the end of a full scan run to evict ghost entries — tables,
        flows, or services that no longer exist in the source system but were never
        explicitly deleted.

        Nodes with no last_seen (registered before this feature was added) are
        left untouched so a single prune pass doesn't wipe legacy data.
        """
        async with self._lock:
            graph = self._load()
            cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds

            removed_nodes = []
            removed_edges = []
            removed_edge_ids: set = set()

            # Collect stale edges first (removing a node also removes its edges, so
            # we scan edges separately to build an accurate removed_edges list).
            # keys=True is required for MultiDiGraph to get the per-edge key.
            for u, v, key, edge_data in list(graph.edges(keys=True, data=True)):
                raw = edge_data.get("data", {})
                last_seen_raw = raw.get("timestamp")  # edges use timestamp, not last_seen
                if last_seen_raw is None:
                    continue
                try:
                    ts = datetime.fromisoformat(str(last_seen_raw).replace("Z", "+00:00"))
                    if ts.timestamp() < cutoff:
                        eid = raw.get("edge_id", f"{u}:{v}")
                        removed_edges.append(eid)
                        removed_edge_ids.add(eid)
                        graph.remove_edge(u, v, key)
                except (ValueError, TypeError):
                    continue

            # Collect stale nodes (skip stub nodes that have no last_seen).
            # Before removing each node, record any incident edges that weren't
            # already pruned in the edge pass above — NetworkX silently removes
            # them when the node is deleted, so we capture them here to keep the
            # returned summary accurate.
            for node_id in list(graph.nodes()):
                raw = graph.nodes[node_id].get("data", {})
                last_seen_raw = raw.get("last_seen")
                if last_seen_raw is None:
                    continue
                try:
                    ts = datetime.fromisoformat(str(last_seen_raw).replace("Z", "+00:00"))
                    if ts.timestamp() < cutoff:
                        for u, v, key, edata in list(graph.edges(node_id, keys=True, data=True)):
                            eid = edata.get("data", {}).get("edge_id", f"{u}:{v}")
                            if eid not in removed_edge_ids:
                                removed_edges.append(eid)
                                removed_edge_ids.add(eid)
                        for u, v, key, edata in list(graph.in_edges(node_id, keys=True, data=True)):
                            eid = edata.get("data", {}).get("edge_id", f"{u}:{v}")
                            if eid not in removed_edge_ids:
                                removed_edges.append(eid)
                                removed_edge_ids.add(eid)
                        removed_nodes.append(node_id)
                        graph.remove_node(node_id)
                except (ValueError, TypeError):
                    continue

            if removed_nodes or removed_edges:
                self._save(graph)

            summary = {"removed_nodes": removed_nodes, "removed_edges": removed_edges}
            logger.info(
                f"Graph prune: removed {len(removed_nodes)} nodes, "
                f"{len(removed_edges)} edges (cutoff {max_age_seconds}s)"
            )
            return summary
