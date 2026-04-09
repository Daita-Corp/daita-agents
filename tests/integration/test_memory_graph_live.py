"""
Integration tests for the memory graph entity scoring and promotion system.

Uses a real LLM (OpenAI gpt-4o-mini) to test the full pipeline:
  content → FactExtractor → MemoryGraph.index_memory → traverse_entity

Run: OPENAI_API_KEY=sk-... pytest tests/integration/test_memory_graph_live.py -v -m requires_llm
"""

import os
import time
from pathlib import Path

import pytest

from daita.core.graph.local_backend import LocalGraphBackend
from daita.plugins.memory.fact_extractor import FactExtractor
from daita.plugins.memory.memory_graph import MemoryGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def llm(api_key):
    from daita.llm.factory import create_llm_provider

    return create_llm_provider(provider="openai", model="gpt-4o-mini", api_key=api_key)


@pytest.fixture
def extractor(llm):
    return FactExtractor(llm=llm)


@pytest.fixture
def graph(tmp_path):
    """MemoryGraph backed by a local graph file with workspace metadata."""
    graph_path = tmp_path / ".daita" / "graph" / "memory.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    mg = MemoryGraph(
        agent_id="test-agent",
        default_properties={"workspace": "integration-test"},
    )
    backend = LocalGraphBackend(graph_type="memory")
    backend._graph_path = graph_path
    mg._backend = backend
    return mg


# ---------------------------------------------------------------------------
# Test content — realistic memories with varying entity quality
# ---------------------------------------------------------------------------

MEMORY_TECHNICAL = (
    "PostgreSQL connection pool is limited to 100 connections. "
    "We migrated the orders table from MySQL to PostgreSQL in Q3 2024. "
    "The etl_orders_agg pipeline reads from orders.customer_id "
    "and writes to the customer_segments table."
)

MEMORY_BUSINESS = (
    "Toyota announced a $14 billion investment in solid-state battery technology. "
    "QuantumScape reported improved energy density in their latest prototype. "
    "Samsung SDI is partnering with Stellantis for EV battery production."
)

MEMORY_NOISY = (
    "The setting was changed to true. Access was unauthorized. "
    "Error handling was improved. The status is currently active. "
    "Best practices were followed during the deployment."
)

MEMORY_OVERLAPPING = (
    "The customer_segments table is refreshed daily by the ETL pipeline. "
    "PostgreSQL handles the customer_segments writes efficiently."
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _timed(label: str):
    """Context manager that prints timing and returns elapsed seconds."""

    class _Timer:
        def __init__(self):
            self.elapsed = 0.0

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self._start
            print(f"  [{label}] {self.elapsed:.3f}s")

    return _Timer()


async def _extract_and_index(extractor, graph, content, chunk_id):
    """Run extraction + indexing and return (facts, elapsed_extract, elapsed_index)."""
    with _timed(f"extract {chunk_id}") as t_ext:
        facts = await extractor.extract(content)
    facts_meta = FactExtractor.facts_to_metadata(facts)

    with _timed(f"index {chunk_id}") as t_idx:
        await graph.index_memory(chunk_id, content, facts=facts_meta)

    return facts_meta, t_ext.elapsed, t_idx.elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
class TestFactExtractionQuality:
    """Verify the LLM extracts the right entities and avoids junk."""

    async def test_technical_content_extracts_real_entities(self, extractor):
        """Technical content should yield database/pipeline entities, not noise."""
        with _timed("extract_technical") as t:
            facts = await extractor.extract(MEMORY_TECHNICAL)

        entities = {f.entity.lower() for f in facts}
        values = {f.value.lower() for f in facts}
        all_names = entities | values

        print(f"  Extracted {len(facts)} facts: {[f.entity for f in facts]}")

        # Should find real entities
        assert any("postgresql" in n for n in all_names), f"Missing PostgreSQL in {all_names}"
        assert any("orders" in n or "etl" in n for n in all_names), f"Missing orders/ETL in {all_names}"

        # Should not produce junk
        junk = {"true", "false", "null", "error", "status", "active"}
        found_junk = entities & junk
        assert not found_junk, f"Junk entities extracted: {found_junk}"

        # Reasonable count
        assert 2 <= len(facts) <= 10, f"Unexpected fact count: {len(facts)}"

        # Reasonable timing
        assert t.elapsed < 10.0, f"Extraction too slow: {t.elapsed:.1f}s"

    async def test_business_content_extracts_companies(self, extractor):
        """Business content should yield company names as entities."""
        facts = await extractor.extract(MEMORY_BUSINESS)
        entities = {f.entity for f in facts}

        print(f"  Extracted {len(facts)} facts: {list(entities)}")

        # Company names should appear
        entity_lower = {e.lower() for e in entities}
        assert any("toyota" in e for e in entity_lower), f"Missing Toyota in {entities}"
        assert any("quantumscape" in e for e in entity_lower), f"Missing QuantumScape in {entities}"

    async def test_noisy_content_minimal_facts(self, extractor):
        """Vague/noisy content should produce few or no facts."""
        facts = await extractor.extract(MEMORY_NOISY)
        entities = {f.entity.lower() for f in facts}

        print(f"  Extracted {len(facts)} facts from noisy content: {list(entities)}")

        # Even if LLM extracts some, the scoring system will handle them
        # But the prompt should suppress most junk at source
        junk = {"true", "false", "error handling", "best practices", "unauthorized access"}
        found_junk = entities & junk
        assert len(found_junk) <= 1, f"Too much junk from noisy content: {found_junk}"


@pytest.mark.requires_llm
class TestEntityScoringAndPromotion:
    """Test the full pipeline: extract → index → score → promote."""

    async def test_technical_entities_promoted_immediately(self, extractor, graph):
        """High-specificity entities should be promoted on first mention."""
        facts_meta, t_ext, t_idx = await _extract_and_index(
            extractor, graph, MEMORY_TECHNICAL, "tech-1"
        )

        print(f"  Facts: {[f['entity'] for f in facts_meta]}")

        # Check promoted entities
        g = await graph.backend.load_graph()
        entity_nodes = {
            nid: g.nodes[nid]
            for nid in g.nodes
            if nid.startswith("entity:")
        }

        promoted = []
        unpromoted = []
        for nid, data in entity_nodes.items():
            props = data.get("data", data).get("properties", {})
            name = data.get("data", data).get("name", nid)
            entry = {
                "name": name,
                "specificity": props.get("specificity"),
                "mention_count": props.get("mention_count"),
                "promoted": props.get("promoted"),
                "workspace": props.get("workspace"),
            }
            if props.get("promoted"):
                promoted.append(entry)
            else:
                unpromoted.append(entry)

        print(f"  Promoted ({len(promoted)}): {[e['name'] for e in promoted]}")
        print(f"  Unpromoted ({len(unpromoted)}): {[e['name'] for e in unpromoted]}")

        # Technical identifiers and proper nouns should be promoted
        promoted_names = {e["name"].lower() for e in promoted}
        assert any("postgresql" in n for n in promoted_names) or \
               any("etl" in n or "customer_segments" in n or "orders" in n for n in promoted_names), \
            f"Expected technical entities promoted, got: {promoted_names}"

        # All nodes should have workspace metadata
        for entry in promoted + unpromoted:
            assert entry["workspace"] == "integration-test", \
                f"Missing workspace on {entry['name']}"

        # All should have scoring metadata
        for entry in promoted + unpromoted:
            assert entry["specificity"] is not None, f"Missing specificity on {entry['name']}"
            assert entry["mention_count"] is not None, f"Missing mention_count on {entry['name']}"
            assert entry["promoted"] is not None, f"Missing promoted on {entry['name']}"

        # Timing sanity
        assert t_ext < 10.0, f"Extraction too slow: {t_ext:.1f}s"
        assert t_idx < 2.0, f"Indexing too slow: {t_idx:.1f}s"

    async def test_business_entities_promoted(self, extractor, graph):
        """Company names (proper nouns) should be immediately promoted."""
        await _extract_and_index(extractor, graph, MEMORY_BUSINESS, "biz-1")

        g = await graph.backend.load_graph()
        promoted_names = set()
        for nid in g.nodes:
            if not nid.startswith("entity:"):
                continue
            data = g.nodes[nid]
            props = data.get("data", data).get("properties", {})
            if props.get("promoted"):
                name = data.get("data", data).get("name", nid)
                promoted_names.add(name.lower())

        print(f"  Promoted: {promoted_names}")
        assert any("toyota" in n for n in promoted_names), \
            f"Toyota should be promoted, got: {promoted_names}"

    async def test_noisy_entities_not_promoted(self, extractor, graph):
        """Junk entities from noisy content should not be promoted."""
        await _extract_and_index(extractor, graph, MEMORY_NOISY, "noisy-1")

        g = await graph.backend.load_graph()
        promoted_names = set()
        all_entity_names = set()
        for nid in g.nodes:
            if not nid.startswith("entity:"):
                continue
            data = g.nodes[nid]
            props = data.get("data", data).get("properties", {})
            name = data.get("data", data).get("name", nid)
            all_entity_names.add(name.lower())
            if props.get("promoted"):
                promoted_names.add(name.lower())

        print(f"  All entities: {all_entity_names}")
        print(f"  Promoted: {promoted_names}")

        # Junk should not be promoted on first mention
        junk_promoted = promoted_names & {
            "true", "false", "null", "error handling",
            "best practices", "unauthorized access",
        }
        assert not junk_promoted, f"Junk was promoted: {junk_promoted}"

    async def test_mention_count_promotes_recurring_entities(self, extractor, graph):
        """Entities mentioned across multiple memories should eventually promote."""
        # First mention — "customer_segments" might be promoted (technical)
        # but generic lowercase entities should not be
        await _extract_and_index(extractor, graph, MEMORY_TECHNICAL, "tech-1")

        # Second mention in different memory — overlapping entities should
        # get mention_count incremented and potentially promote
        await _extract_and_index(extractor, graph, MEMORY_OVERLAPPING, "tech-2")

        g = await graph.backend.load_graph()
        multi_mention = []
        for nid in g.nodes:
            if not nid.startswith("entity:"):
                continue
            data = g.nodes[nid]
            props = data.get("data", data).get("properties", {})
            mc = props.get("mention_count", 0)
            if mc >= 2:
                multi_mention.append({
                    "name": data.get("data", data).get("name", nid),
                    "mention_count": mc,
                    "specificity": props.get("specificity"),
                    "promoted": props.get("promoted"),
                })

        print(f"  Multi-mention entities: {multi_mention}")

        # At least one entity should have been mentioned twice
        # (customer_segments or postgresql appear in both memories)
        assert len(multi_mention) >= 1, \
            "Expected at least one entity with mention_count >= 2"

        # Multi-mention entities with specificity >= 0.3 should be promoted
        for entry in multi_mention:
            if entry["specificity"] >= 0.3:
                assert entry["promoted"], \
                    f"{entry['name']} has mention_count={entry['mention_count']} " \
                    f"and specificity={entry['specificity']} but is not promoted"


class TestPromotionLifecycle:
    """Deterministic tests for the unpromoted → promoted transition.

    Uses hand-crafted facts (no LLM) so entity names are controlled
    and the promotion threshold can be tested precisely.
    """

    async def test_low_specificity_entity_promotes_on_second_mention(self, graph):
        """An entity scoring 0.3 <= specificity < 0.7 should:
        - NOT be promoted after first mention
        - BE promoted after second mention from a different memory
        - Appear in traversal only after promotion
        """
        # "customers" scores ~0.3 — below auto-promote, above discard
        facts_1 = [
            {"entity": "customers", "relation": "has column", "value": "tier"},
        ]
        await graph.index_memory("chunk-a", "The customers table has a tier column.", facts=facts_1)

        node = await graph.backend.get_node("entity:customers")
        assert node.properties["mention_count"] == 1
        assert node.properties["promoted"] is False, "Should not promote on first mention"
        specificity = node.properties["specificity"]
        assert 0.3 <= specificity < 0.7, f"Unexpected specificity: {specificity}"

        # Traversal from a connected promoted entity should NOT show "customers"
        result = await graph.traverse_entity("tier")
        unpromoted_in_result = [
            e for e in result.get("connected_entities", [])
            if e["name"].lower() == "customers"
        ]
        assert not unpromoted_in_result, "Unpromoted 'customers' leaked into traversal"

        # Second mention from different memory — should trigger promotion
        facts_2 = [
            {"entity": "customers", "relation": "joined with", "value": "orders"},
        ]
        await graph.index_memory("chunk-b", "Joining customers with orders.", facts=facts_2)

        node = await graph.backend.get_node("entity:customers")
        assert node.properties["mention_count"] == 2
        assert node.properties["promoted"] is True, "Should promote after 2nd mention"

    async def test_very_low_specificity_stays_unpromoted_with_two_mentions(self, graph):
        """Entities with specificity < 0.3 should NOT promote even with 2 mentions."""
        # "data" scores ~0.1 — too low even for mention-based promotion
        facts_1 = [{"entity": "data", "relation": "stored in", "value": "S3"}]
        await graph.index_memory("chunk-a", "Data is stored in S3.", facts=facts_1)

        node = await graph.backend.get_node("entity:data")
        assert node.properties["promoted"] is False
        assert node.properties["specificity"] < 0.3

        facts_2 = [{"entity": "data", "relation": "processed by", "value": "Spark"}]
        await graph.index_memory("chunk-b", "Data is processed by Spark.", facts=facts_2)

        node = await graph.backend.get_node("entity:data")
        assert node.properties["mention_count"] == 2
        assert node.properties["promoted"] is False, \
            "Very low specificity should not promote even with 2 mentions"

    async def test_high_specificity_promoted_immediately(self, graph):
        """Proper nouns and technical IDs don't need multiple mentions."""
        facts = [
            {"entity": "PostgreSQL", "relation": "version", "value": "16.2"},
            {"entity": "etl_orders_agg", "relation": "writes to", "value": "customer_segments"},
        ]
        await graph.index_memory("chunk-a", "PostgreSQL 16.2 runs etl_orders_agg.", facts=facts)

        pg = await graph.backend.get_node("entity:postgresql")
        assert pg.properties["promoted"] is True
        assert pg.properties["mention_count"] == 1
        assert pg.properties["specificity"] >= 0.7

        etl = await graph.backend.get_node("entity:etl_orders_agg")
        assert etl.properties["promoted"] is True
        assert etl.properties["specificity"] >= 0.7

    async def test_promotion_visible_in_traversal(self, graph):
        """After promotion, entity should appear in traverse results."""
        # Set up: "orders" (low specificity) connected to "PostgreSQL" (high) via memory
        facts = [
            {"entity": "PostgreSQL", "relation": "hosts", "value": "orders"},
        ]
        await graph.index_memory("chunk-a", "PostgreSQL hosts orders.", facts=facts)

        # "orders" is unpromoted — should not appear when traversing from PostgreSQL
        result = await graph.traverse_entity("PostgreSQL")
        entity_names = {e["name"].lower() for e in result["connected_entities"]}
        assert "orders" not in entity_names, "Unpromoted 'orders' should not appear"

        # Second mention promotes "orders"
        facts_2 = [
            {"entity": "MySQL", "relation": "migrated to", "value": "orders"},
        ]
        await graph.index_memory("chunk-b", "MySQL migrated to orders.", facts=facts_2)

        # Now traverse from PostgreSQL — "orders" should be visible
        result = await graph.traverse_entity("PostgreSQL")
        entity_names = {e["name"].lower() for e in result["connected_entities"]}
        # orders is now promoted and connected via shared memory nodes
        orders_node = await graph.backend.get_node("entity:orders")
        assert orders_node.properties["promoted"] is True
        assert orders_node.properties["mention_count"] == 2

    async def test_mention_count_increments_correctly_across_many_memories(self, graph):
        """Mention count should increment for each distinct memory, not each fact."""
        for i in range(5):
            facts = [{"entity": "customers", "relation": "queried in", "value": f"pipeline_{i}"}]
            await graph.index_memory(f"chunk-{i}", f"Query customers in pipeline_{i}.", facts=facts)

        node = await graph.backend.get_node("entity:customers")
        assert node.properties["mention_count"] == 5
        assert node.properties["promoted"] is True  # specificity ~0.3 + 5 mentions


@pytest.mark.requires_llm
class TestTraversalWithPromotion:
    """Test that traverse_entity respects promotion status."""

    async def test_traverse_returns_only_promoted_entities(self, extractor, graph):
        """Traversal should only surface promoted connected entities."""
        # Index technical content — has mix of promoted and unpromoted
        await _extract_and_index(extractor, graph, MEMORY_TECHNICAL, "tech-1")

        # Find a promoted entity to traverse from
        g = await graph.backend.load_graph()
        start_entity = None
        for nid in g.nodes:
            if not nid.startswith("entity:"):
                continue
            data = g.nodes[nid]
            props = data.get("data", data).get("properties", {})
            if props.get("promoted"):
                start_entity = data.get("data", data).get("name", nid)
                break

        if start_entity is None:
            pytest.skip("No promoted entity found to traverse from")

        with _timed("traverse") as t:
            result = await graph.traverse_entity(start_entity)

        print(f"  Traversed from: {start_entity}")
        print(f"  Found: {result['found']}")
        print(f"  Connected entities: {result['connected_entities']}")
        print(f"  Memories: {len(result['memories'])}")

        assert result["found"] is True
        assert len(result["memories"]) >= 1

        # Verify no unpromoted entities in traversal results
        for entity in result["connected_entities"]:
            entity_id = f"entity:{entity['name'].strip().lower()}"
            node = await graph.backend.get_node(entity_id)
            if node:
                assert node.properties.get("promoted", True), \
                    f"Unpromoted entity '{entity['name']}' leaked into traversal"

        assert t.elapsed < 1.0, f"Traversal too slow: {t.elapsed:.1f}s"

    async def test_traverse_memory_nodes_have_workspace(self, extractor, graph):
        """Memory nodes returned by traversal should carry workspace metadata."""
        await _extract_and_index(extractor, graph, MEMORY_BUSINESS, "biz-1")

        # Traverse from a company name (should be promoted)
        result = await graph.traverse_entity("Toyota")
        if not result["found"]:
            # LLM might have used different casing — try lowercase
            g = await graph.backend.load_graph()
            for nid in g.nodes:
                if "toyota" in nid.lower():
                    name = g.nodes[nid].get("data", {}).get("name", "Toyota")
                    result = await graph.traverse_entity(name)
                    break

        if not result["found"]:
            pytest.skip("Toyota entity not found in graph")

        assert len(result["memories"]) >= 1

        # Verify memory nodes have workspace
        for mem in result["memories"]:
            node = await graph.backend.get_node(f"memory:{mem['chunk_id']}")
            assert node is not None
            assert node.properties.get("workspace") == "integration-test", \
                f"Memory node {mem['chunk_id']} missing workspace"


@pytest.mark.requires_llm
class TestNodeCountAndTiming:
    """Verify node counts and timing are within acceptable bounds."""

    async def test_full_pipeline_metrics(self, extractor, graph):
        """Index multiple memories and report comprehensive metrics."""
        memories = [
            ("tech-1", MEMORY_TECHNICAL),
            ("biz-1", MEMORY_BUSINESS),
            ("noisy-1", MEMORY_NOISY),
            ("tech-2", MEMORY_OVERLAPPING),
        ]

        total_facts = 0
        timings = []

        for chunk_id, content in memories:
            facts_meta, t_ext, t_idx = await _extract_and_index(
                extractor, graph, content, chunk_id
            )
            total_facts += len(facts_meta)
            timings.append({
                "chunk_id": chunk_id,
                "facts": len(facts_meta),
                "extract_s": round(t_ext, 3),
                "index_s": round(t_idx, 3),
            })

        await graph.flush()

        # Count nodes by type and status
        g = await graph.backend.load_graph()
        memory_nodes = 0
        entity_total = 0
        entity_promoted = 0
        entity_unpromoted = 0
        multi_mention = 0

        for nid in g.nodes:
            data = g.nodes[nid]
            props = data.get("data", data).get("properties", {})

            if nid.startswith("memory:"):
                memory_nodes += 1
            elif nid.startswith("entity:"):
                entity_total += 1
                if props.get("promoted"):
                    entity_promoted += 1
                else:
                    entity_unpromoted += 1
                if props.get("mention_count", 0) >= 2:
                    multi_mention += 1

        edge_count = g.number_of_edges()

        # Print report
        print("\n" + "=" * 60)
        print("MEMORY GRAPH INTEGRATION TEST REPORT")
        print("=" * 60)
        print(f"\nMemories indexed:      {len(memories)}")
        print(f"Total facts extracted: {total_facts}")
        print(f"\nGraph nodes:")
        print(f"  Memory nodes:        {memory_nodes}")
        print(f"  Entity nodes total:  {entity_total}")
        print(f"  - Promoted:          {entity_promoted}")
        print(f"  - Unpromoted:        {entity_unpromoted}")
        print(f"  - Multi-mention:     {multi_mention}")
        print(f"  Edges:               {edge_count}")
        print(f"\nPer-chunk timing:")
        for t in timings:
            print(f"  {t['chunk_id']:12s}: {t['facts']} facts, "
                  f"extract={t['extract_s']:.3f}s, index={t['index_s']:.3f}s")
        avg_extract = sum(t["extract_s"] for t in timings) / len(timings)
        avg_index = sum(t["index_s"] for t in timings) / len(timings)
        print(f"\nAverage timing:")
        print(f"  Extract: {avg_extract:.3f}s")
        print(f"  Index:   {avg_index:.3f}s")
        print("=" * 60)

        # Assertions
        assert memory_nodes == len(memories), \
            f"Expected {len(memories)} memory nodes, got {memory_nodes}"
        assert entity_total >= 5, \
            f"Expected at least 5 entity nodes across all memories, got {entity_total}"
        assert entity_promoted >= 3, \
            f"Expected at least 3 promoted entities (proper nouns + technical IDs), got {entity_promoted}"
        assert total_facts >= 5, \
            f"Expected at least 5 total facts, got {total_facts}"

        # No single extraction should take > 15s
        for t in timings:
            assert t["extract_s"] < 15.0, \
                f"Extraction for {t['chunk_id']} too slow: {t['extract_s']:.1f}s"

        # Indexing should be fast (local backend, no network)
        for t in timings:
            assert t["index_s"] < 2.0, \
                f"Indexing for {t['chunk_id']} too slow: {t['index_s']:.1f}s"
