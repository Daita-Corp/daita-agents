"""
Tests for memory knowledge graph.
"""

import pytest

from daita.core.graph.models import NodeType, EdgeType
from daita.plugins.memory.memory_graph import (
    MemoryGraph,
    _normalize_entity,
    _make_entity_id,
    _make_memory_id,
)

# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


class TestIdHelpers:
    def test_normalize_entity(self):
        assert _normalize_entity("PostgreSQL") == "postgresql"
        assert _normalize_entity("  Project Orion  ") == "project orion"

    def test_make_entity_id(self):
        assert _make_entity_id("PostgreSQL") == "entity:postgresql"

    def test_make_memory_id(self):
        assert _make_memory_id("abc123") == "memory:abc123"


# ---------------------------------------------------------------------------
# Enum extensions
# ---------------------------------------------------------------------------


class TestGraphModelExtensions:
    def test_memory_node_type(self):
        assert NodeType.MEMORY == "memory"
        assert NodeType.ENTITY == "entity"

    def test_memory_edge_types(self):
        assert EdgeType.MENTIONS == "mentions"
        assert EdgeType.RELATED_TO == "related_to"
        assert EdgeType.SUPERSEDES == "supersedes"


# ---------------------------------------------------------------------------
# Heuristic entity extraction
# ---------------------------------------------------------------------------


class TestHeuristicExtraction:
    def setup_method(self):
        self.mg = MemoryGraph(agent_id="test")

    def test_backtick_entities(self):
        entities = self.mg._extract_entities_heuristic(
            "The `users` table has a `created_at` column"
        )
        names = {e["entity"] for e in entities}
        assert "users" in names
        assert "created_at" in names

    def test_capitalized_phrases(self):
        entities = self.mg._extract_entities_heuristic(
            "Project Orion uses PostgreSQL 16 deployed on Amazon Web Services"
        )
        names = {e["entity"] for e in entities}
        assert "Project Orion" in names
        assert "Amazon Web Services" in names

    def test_table_column_pattern(self):
        entities = self.mg._extract_entities_heuristic(
            "The constraint references orders.customer_id"
        )
        names = {e["entity"] for e in entities}
        assert "orders.customer_id" in names

    def test_quoted_strings(self):
        entities = self.mg._extract_entities_heuristic(
            'The service name is "payment-gateway" in production'
        )
        names = {e["entity"] for e in entities}
        assert "payment-gateway" in names

    def test_skips_short_entities(self):
        entities = self.mg._extract_entities_heuristic("a b c")
        assert len(entities) == 0


# ---------------------------------------------------------------------------
# Facts-to-entities conversion
# ---------------------------------------------------------------------------


class TestFactsToEntities:
    def setup_method(self):
        self.mg = MemoryGraph(agent_id="test")

    def test_basic_fact(self):
        facts = [
            {
                "entity": "PostgreSQL",
                "relation": "has_limit",
                "value": "100 connections",
            },
        ]
        pairs = self.mg._entities_from_facts(facts)
        assert len(pairs) == 1
        assert pairs[0]["entity"] == "PostgreSQL"
        assert pairs[0]["value"] == "100 connections"
        assert pairs[0]["relation"] == "has_limit"

    def test_missing_entity_skipped(self):
        facts = [{"relation": "uses", "value": "something"}]
        pairs = self.mg._entities_from_facts(facts)
        assert len(pairs) == 0

    def test_alternate_keys(self):
        """FactExtractor may use subject/predicate/object instead of entity/relation/value."""
        facts = [
            {"subject": "Redis", "predicate": "stores", "object": "sessions"},
        ]
        pairs = self.mg._entities_from_facts(facts)
        assert len(pairs) == 1
        assert pairs[0]["entity"] == "Redis"
        assert pairs[0]["value"] == "sessions"


# ---------------------------------------------------------------------------
# Graph indexing and traversal (integration with LocalGraphBackend)
# ---------------------------------------------------------------------------


class TestMemoryGraphIndexing:
    async def test_index_memory_with_facts(self):
        mg = MemoryGraph(agent_id="test")

        facts = [
            {
                "entity": "PostgreSQL",
                "relation": "has_limit",
                "value": "100 connections",
            },
            {"entity": "Project Orion", "relation": "uses", "value": "PostgreSQL"},
        ]
        await mg.index_memory("chunk_1", "test content", facts)

        # Verify nodes were created
        graph = await mg.backend.load_graph()
        assert "memory:chunk_1" in graph
        assert "entity:postgresql" in graph
        assert "entity:project orion" in graph
        assert "entity:100 connections" in graph

    async def test_index_memory_with_heuristics(self):
        mg = MemoryGraph(agent_id="test")
        await mg.index_memory(
            "chunk_2",
            "Project Orion uses `PostgreSQL` 16 in production",
        )

        graph = await mg.backend.load_graph()
        assert "memory:chunk_2" in graph
        # Should find at least PostgreSQL via backtick and Project Orion via capitalization
        entity_nodes = [n for n in graph.nodes if n.startswith("entity:")]
        assert len(entity_nodes) >= 1

    async def test_get_connected_memories(self):
        mg = MemoryGraph(agent_id="test")

        # Store two memories that share an entity
        facts1 = [{"entity": "PostgreSQL", "relation": "has", "value": "pool limit"}]
        facts2 = [{"entity": "PostgreSQL", "relation": "version", "value": "16"}]
        await mg.index_memory("chunk_a", "PostgreSQL pool limit is 100", facts1)
        await mg.index_memory("chunk_b", "PostgreSQL version is 16", facts2)

        # chunk_a should find chunk_b via shared entity "PostgreSQL"
        connected = await mg.get_connected_memories("chunk_a", max_depth=2)
        assert "chunk_b" in connected

    async def test_get_connected_memories_no_match(self):
        mg = MemoryGraph(agent_id="test")
        connected = await mg.get_connected_memories("nonexistent", max_depth=2)
        assert connected == []

    async def test_traverse_entity(self):
        mg = MemoryGraph(agent_id="test")

        facts = [
            {
                "entity": "PostgreSQL",
                "relation": "has_limit",
                "value": "100 connections",
            },
            {"entity": "Project Orion", "relation": "uses", "value": "PostgreSQL"},
        ]
        await mg.index_memory("chunk_t1", "test content", facts)

        result = await mg.traverse_entity("PostgreSQL")
        assert result["found"] is True
        assert len(result["memories"]) >= 1

        # Should find connected entities
        entity_names = {e["name"] for e in result["connected_entities"]}
        # "100 connections" and/or "Project Orion" should be connected
        assert len(entity_names) >= 1

    async def test_traverse_entity_not_found(self):
        mg = MemoryGraph(agent_id="test")
        result = await mg.traverse_entity("Nonexistent Entity")
        assert result["found"] is False
        assert result["connected_entities"] == []
        assert result["memories"] == []
