"""
Tests for memory graph quality improvements:
- Merge-on-flush: multiple LocalGraphBackend instances writing to the same file
- Entity quality filtering: temporal, generic, phrase, and currency rejection
- Substring entity deduplication
"""

import json
from pathlib import Path

import pytest

from daita.core.graph.local_backend import LocalGraphBackend
from daita.core.graph.models import AgentGraphEdge, AgentGraphNode, EdgeType, NodeType
from daita.plugins.memory.memory_graph import MemoryGraph, _normalize_entity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(node_id: str, name: str, node_type=NodeType.ENTITY, **kwargs):
    return AgentGraphNode(
        node_id=node_id,
        node_type=node_type,
        name=name,
        **kwargs,
    )


def _make_edge(from_id: str, to_id: str, edge_type=EdgeType.MENTIONS, **kwargs):
    return AgentGraphEdge(
        edge_id=f"{from_id}:{edge_type.value}:{to_id}",
        from_node_id=from_id,
        to_node_id=to_id,
        edge_type=edge_type,
        **kwargs,
    )


def _read_graph_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fix 1: Merge-on-flush
# ---------------------------------------------------------------------------


class TestMergeOnFlush:
    async def test_sequential_flushes_accumulate(self, tmp_path):
        """Two backend instances flushing sequentially should merge, not overwrite."""
        graph_path = tmp_path / ".daita" / "graph" / "memory.json"

        # Simulate two separate backend instances (like two agents)
        backend_a = LocalGraphBackend(graph_type="memory")
        backend_a._graph_path = graph_path
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        backend_b = LocalGraphBackend(graph_type="memory")
        backend_b._graph_path = graph_path

        # Backend A adds a node and flushes
        node_a = _make_node("entity:toyota", "Toyota")
        await backend_a.add_node(node_a)
        await backend_a.flush()

        # Verify A's node is on disk
        data = _read_graph_json(graph_path)
        node_ids = [n["id"] for n in data["nodes"]]
        assert "entity:toyota" in node_ids

        # Backend B adds a different node and flushes
        node_b = _make_node("entity:quantumscape", "QuantumScape")
        await backend_b.add_node(node_b)
        await backend_b.flush()

        # Both nodes should be on disk
        data = _read_graph_json(graph_path)
        node_ids = [n["id"] for n in data["nodes"]]
        assert "entity:toyota" in node_ids, "Backend A's node was lost"
        assert "entity:quantumscape" in node_ids, "Backend B's node was lost"

    async def test_three_instances_accumulate(self, tmp_path):
        """Three backend instances flushing sequentially all merge correctly."""
        graph_path = tmp_path / ".daita" / "graph" / "memory.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        backends = []
        for i in range(3):
            b = LocalGraphBackend(graph_type="memory")
            b._graph_path = graph_path
            backends.append(b)

        # Each backend adds a unique node
        for i, b in enumerate(backends):
            await b.add_node(_make_node(f"entity:e{i}", f"Entity {i}"))

        # Flush all sequentially (simulates on_agent_stop loop)
        for b in backends:
            await b.flush()

        data = _read_graph_json(graph_path)
        node_ids = {n["id"] for n in data["nodes"]}
        assert node_ids == {"entity:e0", "entity:e1", "entity:e2"}

    async def test_edges_merge_across_instances(self, tmp_path):
        """Edges from different instances should both survive flush."""
        graph_path = tmp_path / ".daita" / "graph" / "memory.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        backend_a = LocalGraphBackend(graph_type="memory")
        backend_a._graph_path = graph_path
        backend_b = LocalGraphBackend(graph_type="memory")
        backend_b._graph_path = graph_path

        # A adds a node + edge
        await backend_a.add_node(_make_node("memory:m1", "m1", NodeType.MEMORY))
        await backend_a.add_node(_make_node("entity:toyota", "Toyota"))
        await backend_a.add_edge(_make_edge("memory:m1", "entity:toyota"))
        await backend_a.flush()

        # B adds a different node + edge to the same entity
        await backend_b.add_node(_make_node("memory:m2", "m2", NodeType.MEMORY))
        await backend_b.add_node(_make_node("entity:toyota", "Toyota"))
        await backend_b.add_edge(_make_edge("memory:m2", "entity:toyota"))
        await backend_b.flush()

        data = _read_graph_json(graph_path)
        node_ids = {n["id"] for n in data["nodes"]}
        edge_sources = {e["source"] for e in data["links"]}

        assert "memory:m1" in node_ids
        assert "memory:m2" in node_ids
        assert "entity:toyota" in node_ids
        assert "memory:m1" in edge_sources
        assert "memory:m2" in edge_sources

    async def test_merge_preserves_created_at(self, tmp_path):
        """When two instances write the same node, created_at from first writer wins."""
        graph_path = tmp_path / ".daita" / "graph" / "memory.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        backend_a = LocalGraphBackend(graph_type="memory")
        backend_a._graph_path = graph_path
        backend_b = LocalGraphBackend(graph_type="memory")
        backend_b._graph_path = graph_path

        # A writes first
        await backend_a.add_node(_make_node("entity:toyota", "Toyota"))
        await backend_a.flush()

        data = _read_graph_json(graph_path)
        original_created = next(
            n["data"]["created_at"] for n in data["nodes"] if n["id"] == "entity:toyota"
        )

        # B writes the same node later
        await backend_b.add_node(
            _make_node("entity:toyota", "Toyota", properties={"extra": "data"})
        )
        await backend_b.flush()

        data = _read_graph_json(graph_path)
        final_node = next(n for n in data["nodes"] if n["id"] == "entity:toyota")
        assert final_node["data"]["created_at"] == original_created
        assert final_node["data"]["properties"].get("extra") == "data"

    async def test_no_dirty_flush_is_noop(self, tmp_path):
        """Flushing a clean backend should not touch disk."""
        graph_path = tmp_path / ".daita" / "graph" / "memory.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        backend = LocalGraphBackend(graph_type="memory")
        backend._graph_path = graph_path

        # Flush without any mutations
        await backend.flush()
        assert not graph_path.exists()


# ---------------------------------------------------------------------------
# Fix 2: Entity quality filtering
# ---------------------------------------------------------------------------


class TestEntityQualityFiltering:
    def test_rejects_temporal_entities(self):
        assert MemoryGraph._is_low_quality_entity("as of 2023") is True
        assert MemoryGraph._is_low_quality_entity("before 2024") is True
        assert MemoryGraph._is_low_quality_entity("since 2020") is True

    def test_rejects_generic_entities(self):
        assert MemoryGraph._is_low_quality_entity("challenges") is True
        assert MemoryGraph._is_low_quality_entity("challenges and limitations") is True
        assert MemoryGraph._is_low_quality_entity("r&d efforts") is True
        assert (
            MemoryGraph._is_low_quality_entity("companies and research institutions")
            is True
        )

    def test_rejects_currency_values(self):
        assert MemoryGraph._is_low_quality_entity("$500 million") is True
        assert MemoryGraph._is_low_quality_entity("USD 1,359.18 million") is True

    def test_rejects_bare_numbers_and_percentages(self):
        assert MemoryGraph._is_low_quality_entity("43.11%") is True
        assert MemoryGraph._is_low_quality_entity("2023") is True
        assert MemoryGraph._is_low_quality_entity("99.9%") is True
        assert MemoryGraph._is_low_quality_entity("1,359.18") is True

    def test_rejects_single_lowercase_english_words(self):
        assert MemoryGraph._is_low_quality_entity("rapidly") is True
        assert MemoryGraph._is_low_quality_entity("advancements") is True
        assert MemoryGraph._is_low_quality_entity("production") is True

    def test_accepts_snake_case_identifiers(self):
        assert MemoryGraph._is_low_quality_entity("customer_segments") is False
        assert MemoryGraph._is_low_quality_entity("etl_orders_agg") is False
        assert MemoryGraph._is_low_quality_entity("revenue_daily") is False
        assert MemoryGraph._is_low_quality_entity("order_items") is False
        assert MemoryGraph._is_low_quality_entity("total_cents") is False

    def test_accepts_dot_notation_identifiers(self):
        assert MemoryGraph._is_low_quality_entity("orders.total") is False
        assert MemoryGraph._is_low_quality_entity("customers.tier") is False
        assert MemoryGraph._is_low_quality_entity("payments.amount") is False

    def test_accepts_all_caps_type_names(self):
        assert MemoryGraph._is_low_quality_entity("INTEGER") is False
        assert MemoryGraph._is_low_quality_entity("DECIMAL(10,2)") is False
        assert MemoryGraph._is_low_quality_entity("NOT_NULL") is False
        assert MemoryGraph._is_low_quality_entity("TEXT") is False

    def test_accepts_single_proper_nouns(self):
        assert MemoryGraph._is_low_quality_entity("Toyota") is False
        assert MemoryGraph._is_low_quality_entity("QuantumScape") is False
        assert MemoryGraph._is_low_quality_entity("Samsung") is False

    def test_rejects_long_phrases(self):
        assert (
            MemoryGraph._is_low_quality_entity(
                "technical hurdles like dendrite formation during charging"
            )
            is True
        )

    def test_accepts_good_entities(self):
        assert MemoryGraph._is_low_quality_entity("Toyota") is False
        assert MemoryGraph._is_low_quality_entity("solid-state batteries") is False
        assert MemoryGraph._is_low_quality_entity("QuantumScape") is False
        assert MemoryGraph._is_low_quality_entity("energy density") is False
        assert MemoryGraph._is_low_quality_entity("dendrite formation") is False

    def test_data_domain_facts_produce_entities(self):
        """LLM-extracted facts from data domains should produce entity pairs.

        Single lowercase words like 'customers' and 'orders' are legitimate
        table names when they come from fact extraction (from_facts=True).
        """
        facts = [
            {"entity": "customers", "relation": "has column", "value": "tier"},
            {"entity": "etl_orders_agg", "relation": "reads from", "value": "orders"},
            {"entity": "orders.total", "relation": "type", "value": "DECIMAL(10,2)"},
            {
                "entity": "customer_segments",
                "relation": "written by",
                "value": "etl_customer_segments",
            },
        ]
        graph = MemoryGraph()
        pairs = graph._entities_from_facts(facts)
        entities = {p["entity"] for p in pairs}
        values = {p.get("value") for p in pairs if p.get("value")}

        assert "customers" in entities
        assert "etl_orders_agg" in entities
        assert "orders.total" in entities
        assert "customer_segments" in entities
        assert "orders" in values
        assert "DECIMAL(10,2)" in values

    def test_heuristic_still_rejects_single_lowercase(self):
        """Heuristic path (from_facts=False) should still reject single lowercase words."""
        assert MemoryGraph._is_low_quality_entity("customers", from_facts=False) is True
        assert MemoryGraph._is_low_quality_entity("orders", from_facts=False) is True

    def test_facts_path_accepts_single_lowercase(self):
        """Facts path (from_facts=True) should accept single lowercase words."""
        assert MemoryGraph._is_low_quality_entity("customers", from_facts=True) is False
        assert MemoryGraph._is_low_quality_entity("orders", from_facts=True) is False
        assert MemoryGraph._is_low_quality_entity("tier", from_facts=True) is False

    def test_facts_path_still_rejects_generic_terms(self):
        """Even from_facts=True should reject known generic English words."""
        assert MemoryGraph._is_low_quality_entity("challenges", from_facts=True) is True
        assert MemoryGraph._is_low_quality_entity("data", from_facts=True) is True
        assert MemoryGraph._is_low_quality_entity("results", from_facts=True) is True

    def test_filter_removes_bad_entities(self):
        pairs = [
            {"entity": "Toyota", "value": "R&D leader", "relation": "is"},
            {"entity": "as of 2023", "value": "breakthrough", "relation": "saw"},
            {"entity": "challenges", "value": "scalability", "relation": "include"},
        ]
        filtered = MemoryGraph._filter_entity_pairs(pairs)
        entities = [p["entity"] for p in filtered]
        assert "Toyota" in entities
        assert "as of 2023" not in entities
        assert "challenges" not in entities

    def test_filter_drops_bad_value_but_keeps_entity(self):
        pairs = [
            {
                "entity": "market growth",
                "value": "$500 million",
                "relation": "projected",
            },
        ]
        filtered = MemoryGraph._filter_entity_pairs(pairs)
        assert len(filtered) == 1
        assert filtered[0]["entity"] == "market growth"
        assert filtered[0]["value"] is None


# ---------------------------------------------------------------------------
# Fix 2: Substring entity deduplication
# ---------------------------------------------------------------------------


class TestEntityDeduplication:
    def test_shorter_absorbs_longer(self):
        pairs = [
            {
                "entity": "solid-state batteries",
                "value": "high density",
                "relation": "have",
            },
            {
                "entity": "solid-state battery technology",
                "value": "advancing",
                "relation": "is",
            },
        ]
        deduped = MemoryGraph._deduplicate_entities(pairs)
        entities = [_normalize_entity(p["entity"]) for p in deduped]
        # The longer form should be redirected to the shorter canonical form
        assert "solid-state battery technology" not in entities
        assert "solid-state batteries" in entities

    def test_no_dedup_for_unrelated_entities(self):
        pairs = [
            {"entity": "Toyota", "value": None, "relation": "leads"},
            {"entity": "QuantumScape", "value": None, "relation": "develops"},
        ]
        deduped = MemoryGraph._deduplicate_entities(pairs)
        assert len(deduped) == 2

    def test_dedup_removes_duplicates_after_canonicalization(self):
        pairs = [
            {"entity": "batteries", "value": "improved", "relation": "are"},
            {"entity": "solid-state batteries", "value": "improved", "relation": "are"},
        ]
        deduped = MemoryGraph._deduplicate_entities(pairs)
        # "solid-state batteries" contains "batteries", so "solid-state batteries"
        # gets redirected to "batteries". Both now have the same key → deduplicated.
        assert len(deduped) == 1

    def test_single_pair_passthrough(self):
        pairs = [{"entity": "Toyota", "value": None, "relation": "leads"}]
        deduped = MemoryGraph._deduplicate_entities(pairs)
        assert deduped == pairs

    def test_empty_passthrough(self):
        assert MemoryGraph._deduplicate_entities([]) == []

    def test_value_also_deduplicated(self):
        """When a long-form entity also appears as a value, it gets redirected."""
        pairs = [
            {
                "entity": "Toyota",
                "value": "solid-state battery technology",
                "relation": "develops",
            },
            {
                "entity": "solid-state batteries",
                "value": None,
                "relation": "advancing",
            },
            {
                "entity": "solid-state battery technology",
                "value": "emerging",
                "relation": "is",
            },
        ]
        deduped = MemoryGraph._deduplicate_entities(pairs)
        # "solid-state battery technology" appears as both entity and value,
        # so it's in the canonical mapping and gets redirected everywhere
        for p in deduped:
            if p.get("value"):
                assert _normalize_entity(p["value"]) != "solid-state battery technology"
        entities = [_normalize_entity(p["entity"]) for p in deduped]
        assert "solid-state battery technology" not in entities
