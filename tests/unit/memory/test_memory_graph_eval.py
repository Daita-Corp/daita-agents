"""
Deterministic memory graph quality evals.

These tests use labeled fixture facts instead of live LLM extraction. That keeps
the signal focused on graph construction quality: which nodes are stored, which
nodes are promoted, whether relationships are preserved, and whether traversal
surfaces useful context without noisy concepts.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "networkx", reason="networkx required: pip install 'daita-agents[memory]'"
)

from daita.plugins.memory.graph_models import MemoryEdgeType
from daita.plugins.memory.memory_graph import (
    MemoryGraph,
    _make_entity_id,
    _normalize_entity,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "memory_graph_quality_cases.json"
)


def _load_cases() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


async def _index_fixture(tmp_path):
    fixture = _load_cases()
    graph = MemoryGraph(
        agent_id="quality-eval",
        default_properties={"workspace": "quality-eval"},
        storage_dir=tmp_path / "memory-graph",
    )
    for case in fixture["cases"]:
        await graph.index_memory(case["id"], case["content"], facts=case["facts"])
    return fixture, graph


def _payload(nx_graph, node_id: str) -> dict:
    return nx_graph.nodes[node_id].get("data", {})


def _value(value):
    return value.value if hasattr(value, "value") else value


def _expected_entity_ids(fixture: dict) -> set[str]:
    ids = set()
    for case in fixture["cases"]:
        for entity in case.get("expected_entities", []):
            ids.add(_make_entity_id(entity))
    return ids


def _absent_entity_ids(fixture: dict) -> set[str]:
    ids = set()
    for case in fixture["cases"]:
        for entity in case.get("expected_absent_entities", []):
            ids.add(_make_entity_id(entity))
    return ids


class TestMemoryGraphQualityEval:
    async def test_entities_match_fixture_without_structural_noise(self, tmp_path):
        fixture, graph = await _index_fixture(tmp_path)
        nx_graph = await graph.backend.load_graph()

        actual_entities = {
            node_id for node_id in nx_graph.nodes if node_id.startswith("entity:")
        }
        expected_entities = _expected_entity_ids(fixture)
        absent_entities = _absent_entity_ids(fixture)

        assert expected_entities <= actual_entities
        assert actual_entities.isdisjoint(absent_entities)
        assert actual_entities == expected_entities

    async def test_promoted_nodes_are_high_signal(self, tmp_path):
        fixture, graph = await _index_fixture(tmp_path)
        nx_graph = await graph.backend.load_graph()

        promoted_expected = set()
        unpromoted_expected = set()
        forbidden_promoted = set()
        for case in fixture["cases"]:
            promoted_expected.update(
                _make_entity_id(entity) for entity in case.get("expected_promoted", [])
            )
            unpromoted_expected.update(
                _make_entity_id(entity)
                for entity in case.get("expected_unpromoted", [])
            )
            forbidden_promoted.update(
                _make_entity_id(entity) for entity in case.get("forbidden_promoted", [])
            )

        for node_id in promoted_expected:
            assert _payload(nx_graph, node_id)["properties"]["promoted"] is True

        for node_id in unpromoted_expected:
            assert _payload(nx_graph, node_id)["properties"]["promoted"] is False

        promoted_forbidden = [
            node_id
            for node_id in forbidden_promoted
            if node_id in nx_graph
            and _payload(nx_graph, node_id)["properties"]["promoted"] is True
        ]
        assert (
            len(promoted_forbidden)
            <= fixture["quality_thresholds"]["promoted_forbidden_max"]
        )

    async def test_relationship_edges_preserve_expected_relations(self, tmp_path):
        fixture, graph = await _index_fixture(tmp_path)
        nx_graph = await graph.backend.load_graph()

        actual_relationships = set()
        for from_id, to_id, _key, edge_data in nx_graph.edges(keys=True, data=True):
            raw = edge_data.get("data", {})
            if _value(raw.get("edge_type")) != MemoryEdgeType.RELATED_TO.value:
                continue
            actual_relationships.add(
                (
                    from_id,
                    raw.get("properties", {}).get("relation"),
                    to_id,
                )
            )

        expected_relationships = set()
        for case in fixture["cases"]:
            for entity, relation, value in case.get("expected_relationships", []):
                expected_relationships.add(
                    (_make_entity_id(entity), relation, _make_entity_id(value))
                )

        assert expected_relationships <= actual_relationships

    async def test_multi_mention_nodes_have_evidence(self, tmp_path):
        fixture, graph = await _index_fixture(tmp_path)
        nx_graph = await graph.backend.load_graph()

        multi_mention = [
            node_id
            for node_id in nx_graph.nodes
            if node_id.startswith("entity:")
            and _payload(nx_graph, node_id)["properties"].get("mention_count", 0) >= 2
        ]

        assert len(multi_mention) >= fixture["quality_thresholds"]["multi_mention_min"]
        assert _make_entity_id("PostgreSQL") in multi_mention
        assert _make_entity_id("Project Orion") in multi_mention
        assert _make_entity_id("customer_segments") in multi_mention

    async def test_traversal_returns_relevant_context_without_noise(self, tmp_path):
        fixture, graph = await _index_fixture(tmp_path)

        for expectation in fixture["traversal_expectations"]:
            result = await graph.traverse_entity(
                expectation["entity"], max_depth=expectation["max_depth"]
            )
            memory_ids = {memory["chunk_id"] for memory in result["memories"]}
            entity_names = {
                _normalize_entity(entity["name"])
                for entity in result["connected_entities"]
            }

            assert set(expectation["expected_memory_ids"]) <= memory_ids
            assert {
                _normalize_entity(entity)
                for entity in expectation["expected_connected_entities"]
            } <= entity_names
            assert entity_names.isdisjoint(
                {
                    _normalize_entity(entity)
                    for entity in expectation["forbidden_connected_entities"]
                }
            )

    async def test_connected_memories_expand_through_shared_entities(self, tmp_path):
        _fixture, graph = await _index_fixture(tmp_path)

        connected = set(
            await graph.get_connected_memories(
                "orion_postgres_architecture", max_depth=2
            )
        )

        assert "postgres_operational_constraints" in connected
        assert "customer_segments_refresh" in connected
        assert "low_signal_deployment_noise" not in connected
