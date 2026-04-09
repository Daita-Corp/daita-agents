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

# Entity quality filters
_TEMPORAL_RE = re.compile(
    r"^(as of|before|after|since|by|in|during|until)\s+\d", re.IGNORECASE
)
_CURRENCY_RE = re.compile(
    r"[\$€£]|(?:USD|EUR|GBP)\s*\d|^\d[\d,.]*\s*(million|billion|trillion|k|m|b)$",
    re.IGNORECASE,
)
_BARE_NUMBER_RE = re.compile(r"^[\d,.%/]+$")
_MAX_ENTITY_WORDS = 5

# Technical identifier patterns — these are code/data names, not English prose.
# They bypass the single-lowercase-word filter because "customers", "etl_orders_agg",
# and "orders.total" are high-value entities in data/engineering domains.
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")  # customer_segments
_DOT_NOTATION_RE = re.compile(r"^\w+\.\w+$")  # orders.total
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9_(),.]+$")  # DECIMAL(10,2), INTEGER


def _is_technical_identifier(name: str) -> bool:
    """Return True if name looks like a code/data identifier, not English prose."""
    stripped = name.strip()
    if _SNAKE_CASE_RE.match(stripped):
        return True
    if _DOT_NOTATION_RE.match(stripped):
        return True
    if _ALL_CAPS_RE.match(stripped):
        return True
    return False


_GENERIC_ENTITIES = frozenset(
    {
        "challenges",
        "challenges and limitations",
        "efforts",
        "r&d efforts",
        "companies",
        "companies and research institutions",
        "issues",
        "problems",
        "advantages",
        "benefits",
        "features",
        "results",
        "findings",
        "data",
        "information",
        "details",
        "aspects",
        "factors",
        "elements",
        "things",
    }
)


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
        """Convert FactExtractor output to entity pairs with quality filtering.

        Uses relaxed filtering (``from_facts=True``) because the LLM was
        explicitly asked to extract entities — single lowercase words like
        ``customers`` or ``orders`` are legitimate table/column names, not
        English prose noise.
        """
        pairs = []
        for fact in facts:
            entity = fact.get("entity") or fact.get("subject")
            if not entity:
                continue
            value = fact.get("value") or fact.get("object")
            relation = fact.get("relation") or fact.get("predicate", "related_to")
            pairs.append({"entity": entity, "value": value, "relation": relation})
        pairs = self._filter_entity_pairs(pairs, from_facts=True)
        return self._deduplicate_entities(pairs)

    @staticmethod
    def _is_low_quality_entity(name: str, from_facts: bool = False) -> bool:
        """Return True if the entity name is too generic, temporal, or numeric.

        Args:
            name: Entity name to check.
            from_facts: When True, skip the single-lowercase-word filter.
                LLM-extracted facts have already been curated for relevance,
                so single lowercase words like ``customers`` or ``orders``
                are legitimate domain entities, not English prose noise.
        """
        if not name or len(name) < 2:
            return True
        normalized = name.strip().lower()
        if normalized in _GENERIC_ENTITIES:
            return True
        if _TEMPORAL_RE.match(name):
            return True
        if _CURRENCY_RE.search(name):
            return True
        if _BARE_NUMBER_RE.match(normalized):
            return True
        if len(name.split()) > _MAX_ENTITY_WORDS:
            return True
        # Single words must be proper nouns (capitalized) or technical identifiers
        # to be useful entities. English words like "rapidly", "advancements",
        # "production" create low-value graph connections, but code/data names
        # like "customer_segments", "orders.total", "INTEGER" are high-value.
        # When from_facts=True, the LLM already filtered for relevance so we
        # trust single lowercase words as legitimate domain entities.
        if not from_facts and len(name.split()) == 1 and not name[0].isupper():
            if not _is_technical_identifier(name):
                return True
        return False

    @staticmethod
    def _filter_entity_pairs(pairs: List[dict], from_facts: bool = False) -> List[dict]:
        """Remove pairs whose entity or value fails quality checks.

        Args:
            pairs: Entity pairs to filter.
            from_facts: Pass True when pairs come from LLM-extracted facts
                to apply relaxed filtering (see ``_is_low_quality_entity``).
        """
        filtered = []
        for p in pairs:
            if MemoryGraph._is_low_quality_entity(p["entity"], from_facts):
                continue
            # Filter value too — if low quality, keep the entity but drop the value
            if p.get("value") and MemoryGraph._is_low_quality_entity(
                p["value"], from_facts
            ):
                p = {**p, "value": None}
            filtered.append(p)
        return filtered

    @staticmethod
    def _deduplicate_entities(pairs: List[dict]) -> List[dict]:
        """Absorb entities that are substrings of shorter canonical forms.

        If "solid-state batteries" and "solid-state battery technology" both
        appear, keep only "solid-state batteries" (the shorter one) and
        redirect the longer one's relations to the shorter entity.

        Uses word-set containment: if all words of the shorter entity appear
        in the longer entity, the longer one is absorbed. This handles
        inflection differences like "batteries" appearing in a name that
        contains "battery" by checking if the shorter word-stem (minus
        trailing 1-3 characters) matches a word in the longer name.
        """
        if len(pairs) <= 1:
            return pairs

        # Collect all unique entity names (normalized)
        entity_names = list({_normalize_entity(p["entity"]) for p in pairs})
        # Sort by length so shorter names come first
        entity_names.sort(key=len)

        def _words(name: str) -> set:
            return set(name.split())

        def _is_absorbed(shorter: str, longer: str) -> bool:
            """Check if shorter's words are contained in longer (with fuzzy stems).

            Single-token technical identifiers (no spaces) are never absorbed
            by other single-token identifiers — ``customers`` and
            ``customer_segments`` are distinct entities, not variations.
            """
            shorter_words = _words(shorter)
            longer_words = _words(longer)
            # Two single-token names are distinct unless exactly equal
            # (which is handled before this function is called).
            # This prevents "customers" from absorbing "customer_segments".
            if len(shorter_words) == 1 and len(longer_words) == 1:
                return False
            for sw in shorter_words:
                # Exact match
                if sw in longer_words:
                    continue
                # Stem match: check if any longer word shares a prefix (min 4 chars)
                stem = sw[: max(4, len(sw) - 3)]
                if any(lw.startswith(stem) for lw in longer_words):
                    continue
                return False
            return True

        # Build a mapping: long_name -> canonical short_name
        canonical = {}
        for i, name in enumerate(entity_names):
            for shorter in entity_names[:i]:
                if shorter != name and _is_absorbed(shorter, name):
                    canonical[name] = shorter
                    break

        if not canonical:
            return pairs

        # Apply the mapping — redirect long entity names to their canonical form
        deduped = []
        for p in pairs:
            norm = _normalize_entity(p["entity"])
            if norm in canonical:
                p = {**p, "entity": canonical[norm]}
            if p.get("value"):
                norm_val = _normalize_entity(p["value"])
                if norm_val in canonical:
                    p = {**p, "value": canonical[norm_val]}
            deduped.append(p)

        # Remove exact duplicate pairs that may have been created by canonicalization
        seen = set()
        unique = []
        for p in deduped:
            key = (
                _normalize_entity(p["entity"]),
                p.get("relation"),
                _normalize_entity(p.get("value") or ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

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
