"""
Lightweight knowledge graph over agent memories.

Creates entity nodes from stored memories and connects them via relationships,
enabling traversal queries like "what are all the infrastructure constraints
for Project Orion?" that pure cosine similarity would miss.

Uses the existing GraphBackend infrastructure with graph_type='memory'
(separate from the lineage graph).
"""

import re
from collections import deque
from typing import Any, Dict, List, Optional

from ...core.graph.models import AgentGraphEdge, AgentGraphNode, EdgeType, NodeType


def _normalize_entity(name: str) -> str:
    """Normalize entity name for deterministic node IDs."""
    return name.strip().lower()


def _make_entity_id(name: str) -> str:
    return f"entity:{_normalize_entity(name)}"


def _make_memory_id(chunk_id: str) -> str:
    return f"memory:{chunk_id}"


# Patterns for heuristic entity extraction (no LLM required)
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_CAPITALIZED_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_TABLE_COLUMN_RE = re.compile(r"\b(\w+\.\w+)\b")
_QUOTED_RE = re.compile(r'"([^"]+)"')


class MemoryGraph:
    """Lightweight graph layer for memory relationships.

    Uses the existing GraphBackend with graph_type='memory' so it doesn't
    interfere with lineage graphs. Supports both LLM-extracted facts
    (from FactExtractor) and zero-LLM keyword heuristics.
    """

    def __init__(self, agent_id: Optional[str] = None):
        self._backend = None
        self._agent_id = agent_id

    @property
    def backend(self):
        """Lazy-init graph backend."""
        if self._backend is None:
            from ...core.graph.backend import auto_select_backend

            self._backend = auto_select_backend(graph_type="memory")
        return self._backend

    async def index_memory(
        self,
        chunk_id: str,
        content: str,
        facts: Optional[List[dict]] = None,
    ):
        """Create graph nodes and edges from a memory and its extracted facts.

        If facts are provided (from FactExtractor), use them directly.
        Otherwise, fall back to keyword heuristic extraction (no LLM cost).
        """
        memory_node_id = _make_memory_id(chunk_id)

        # Create MEMORY node
        memory_node = AgentGraphNode(
            node_id=memory_node_id,
            node_type=NodeType.MEMORY,
            name=chunk_id,
            created_by_agent=self._agent_id,
            properties={"content_preview": content[:200]},
        )
        await self.backend.add_node(memory_node)

        # Extract entities from facts or heuristics
        if facts:
            entity_pairs = self._entities_from_facts(facts)
        else:
            entity_pairs = self._extract_entities_heuristic(content)

        # Create entity nodes and edges
        for entity_info in entity_pairs:
            entity_name = entity_info["entity"]
            entity_id = _make_entity_id(entity_name)

            # Upsert ENTITY node
            entity_node = AgentGraphNode(
                node_id=entity_id,
                node_type=NodeType.ENTITY,
                name=entity_name,
                created_by_agent=self._agent_id,
            )
            await self.backend.add_node(entity_node)

            # MEMORY --MENTIONS--> ENTITY
            mention_edge = AgentGraphEdge(
                edge_id=AgentGraphEdge.make_id(
                    memory_node_id, EdgeType.MENTIONS, entity_id
                ),
                from_node_id=memory_node_id,
                to_node_id=entity_id,
                edge_type=EdgeType.MENTIONS,
                created_by_agent=self._agent_id,
            )
            await self.backend.add_edge(mention_edge)

            # If there's a related entity (value), create ENTITY --RELATED_TO--> ENTITY
            value = entity_info.get("value")
            if value and value != entity_name:
                value_id = _make_entity_id(value)
                value_node = AgentGraphNode(
                    node_id=value_id,
                    node_type=NodeType.ENTITY,
                    name=value,
                    created_by_agent=self._agent_id,
                )
                await self.backend.add_node(value_node)

                # MEMORY --MENTIONS--> VALUE_ENTITY
                value_mention = AgentGraphEdge(
                    edge_id=AgentGraphEdge.make_id(
                        memory_node_id, EdgeType.MENTIONS, value_id
                    ),
                    from_node_id=memory_node_id,
                    to_node_id=value_id,
                    edge_type=EdgeType.MENTIONS,
                    created_by_agent=self._agent_id,
                )
                await self.backend.add_edge(value_mention)

                # ENTITY --RELATED_TO--> VALUE_ENTITY
                relation = entity_info.get("relation", "related_to")
                rel_edge = AgentGraphEdge(
                    edge_id=AgentGraphEdge.make_id(
                        entity_id, EdgeType.RELATED_TO, value_id
                    ),
                    from_node_id=entity_id,
                    to_node_id=value_id,
                    edge_type=EdgeType.RELATED_TO,
                    created_by_agent=self._agent_id,
                    properties={"relation": relation},
                )
                await self.backend.add_edge(rel_edge)

    def _entities_from_facts(self, facts: List[dict]) -> List[dict]:
        """Convert FactExtractor output to entity pairs."""
        pairs = []
        for fact in facts:
            entity = fact.get("entity") or fact.get("subject")
            if not entity:
                continue
            value = fact.get("value") or fact.get("object")
            relation = fact.get("relation") or fact.get("predicate", "related_to")
            pairs.append({"entity": entity, "value": value, "relation": relation})
        return pairs

    def _extract_entities_heuristic(self, content: str) -> List[dict]:
        """Extract entities via regex patterns (no LLM cost).

        Finds: backtick-quoted identifiers, capitalized multi-word phrases,
        table.column patterns, and double-quoted strings.
        """
        entities = set()

        for match in _BACKTICK_RE.finditer(content):
            entities.add(match.group(1))

        for match in _CAPITALIZED_PHRASE_RE.finditer(content):
            entities.add(match.group(1))

        for match in _TABLE_COLUMN_RE.finditer(content):
            val = match.group(1)
            # Skip common false positives
            if val not in ("e.g", "i.e", "etc."):
                entities.add(val)

        for match in _QUOTED_RE.finditer(content):
            val = match.group(1)
            if len(val) > 2 and len(val) < 50:
                entities.add(val)

        # Return as entity-only (no value/relation for heuristic extraction)
        return [{"entity": e} for e in entities if len(e) > 1]

    async def get_connected_memories(
        self, chunk_id: str, max_depth: int = 2
    ) -> List[str]:
        """Return chunk_ids of memories connected via shared entities.

        Traversal: memory -> entity -> memory (2 hops = 1 shared entity).
        """
        memory_node_id = _make_memory_id(chunk_id)
        graph = await self.backend.load_graph()

        if memory_node_id not in graph:
            return []

        # BFS to find connected memory nodes

        visited = set()
        queue = deque([(memory_node_id, 0)])
        connected_memories = []

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)

            # Collect memory nodes (except the starting one)
            if node_id != memory_node_id and node_id.startswith("memory:"):
                connected_memories.append(node_id.removeprefix("memory:"))
                continue  # Don't traverse past memory nodes

            # Traverse both directions (undirected for memory graph)
            for neighbor in graph.successors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
            for neighbor in graph.predecessors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return connected_memories

    async def traverse_entity(
        self,
        entity_name: str,
        direction: str = "both",
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """Walk graph from an entity. Returns connected entities and their memories."""
        entity_id = _make_entity_id(entity_name)
        graph = await self.backend.load_graph()

        if entity_id not in graph:
            return {
                "entity": entity_name,
                "found": False,
                "connected_entities": [],
                "memories": [],
            }

        visited = set()
        queue = deque([(entity_id, 0)])
        entities = []
        memories = []

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)

            node_data = graph.nodes.get(node_id, {})

            if node_id != entity_id:
                if node_id.startswith("memory:"):
                    chunk_id = node_id.removeprefix("memory:")
                    memories.append(
                        {
                            "chunk_id": chunk_id,
                            "content_preview": node_data.get("properties", {}).get(
                                "content_preview", ""
                            ),
                        }
                    )
                    continue  # Don't traverse past memory nodes
                elif node_id.startswith("entity:"):
                    name = node_data.get("name", node_id.removeprefix("entity:"))
                    # Collect edge relation if available
                    entities.append({"name": name, "depth": depth})

            # Traverse both directions
            for neighbor in graph.successors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
            for neighbor in graph.predecessors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return {
            "entity": entity_name,
            "found": True,
            "connected_entities": entities,
            "memories": memories,
        }

    async def flush(self):
        """Persist pending graph mutations."""
        if hasattr(self.backend, "flush"):
            await self.backend.flush()
