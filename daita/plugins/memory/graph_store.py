"""
Memory graph storage adapter.

The memory graph has its own domain model, but it reuses the core graph backend
for persistence, locking, merge-on-flush behavior, and bounded subgraph loading.
This keeps memory semantics separate without creating a second graph storage
implementation.
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Iterable, Optional, Protocol

from ...core.graph.models import AgentGraphEdge, AgentGraphNode
from .graph_models import MemoryEdgeType, MemoryGraphEdge, MemoryGraphNode

if TYPE_CHECKING:
    import networkx as nx


class MemoryGraphStore(Protocol):
    async def add_node(self, node: MemoryGraphNode) -> None: ...

    async def add_edge(self, edge: MemoryGraphEdge) -> None: ...

    async def get_node(self, node_id: str) -> Optional[MemoryGraphNode]: ...

    async def subgraph(
        self,
        root: str,
        direction: str = "both",
        edge_types: Optional[Iterable[MemoryEdgeType | str]] = None,
        max_depth: int = 5,
    ) -> "nx.MultiDiGraph": ...

    async def flush(self) -> None: ...


class GraphBackendMemoryGraphStore:
    """Memory-domain adapter over the shared core GraphBackend."""

    def __init__(
        self,
        backend: object | None = None,
        storage_dir: str | PathLike[str] | None = None,
        graph_type: str = "memory",
    ):
        if backend is None:
            if storage_dir is None:
                from ...core.graph.backend import auto_select_backend

                backend = auto_select_backend(graph_type=graph_type)
            else:
                from ...core.graph.local_backend import LocalGraphBackend

                backend = LocalGraphBackend(
                    graph_type=graph_type,
                    storage_dir=storage_dir,
                )
        self.backend = backend

    async def add_node(self, node: MemoryGraphNode) -> None:
        await self.backend.add_node(AgentGraphNode(**node.model_dump()))

    async def add_edge(self, edge: MemoryGraphEdge) -> None:
        await self.backend.add_edge(AgentGraphEdge(**edge.model_dump()))

    async def get_node(self, node_id: str) -> Optional[MemoryGraphNode]:
        raw_node = await self.backend.get_node(node_id)
        if raw_node is None:
            return None
        raw = raw_node.model_dump() if hasattr(raw_node, "model_dump") else raw_node
        if not raw or not raw.get("node_type"):
            return None
        return MemoryGraphNode(**raw)

    async def subgraph(
        self,
        root: str,
        direction: str = "both",
        edge_types: Optional[Iterable[MemoryEdgeType | str]] = None,
        max_depth: int = 5,
    ) -> "nx.MultiDiGraph":
        return await self.backend.subgraph(
            root=root,
            direction=direction,
            edge_types=edge_types,
            max_depth=max_depth,
        )

    async def flush(self) -> None:
        if hasattr(self.backend, "flush"):
            await self.backend.flush()
