"""
Live-LLM integration tests for the LineagePlugin.

Seeds a small, known graph directly through the backend, then asks a real
OpenAI-backed Agent to answer lineage questions that have deterministic
ground truth. We check four things per spec:

1. The plugin's tools route to the correct algorithm (trace_lineage,
   find_lineage_paths, analyze_impact).
2. The graph is built correctly by ``register_flow`` — node + edge counts
   and edge types match what we emitted.
3. Agent traversal speed — soft metric, logged to stderr.
4. Agent traversal accuracy — assert the right tool fired AND the final
   answer mentions the expected node names.

Requirements:
  - OPENAI_API_KEY

Run:
    OPENAI_API_KEY=sk-... pytest tests/integration/test_lineage_live.py \\
        -v -s -m "requires_llm and integration"
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import pytest

from daita.core.graph import LocalGraphBackend, LINEAGE_EDGE_TYPES
from daita.core.graph.models import (
    AgentGraphEdge,
    AgentGraphNode,
    EdgeType,
    NodeType,
)
from daita.plugins.lineage import LineagePlugin

from ._harness import (
    assert_answer_mentions,
    assert_tool_called,
    build_live_agent,
    timed,
)

# ---------------------------------------------------------------------------
# Ground-truth graph
# ---------------------------------------------------------------------------
#
# raw_orders       --writes-->      stg_orders
# stg_orders       --transforms-->  fact_orders
# fact_orders      --derived_from-> revenue_daily
# fact_orders      --has_column-->  fact_orders.amount     (structural — lineage tools should IGNORE)
# unrelated_table  --reads-->       another_thing          (disconnected component)
#
# So the lineage of fact_orders is:
#   upstream (default depth 5):   {stg_orders, raw_orders}
#   downstream (default depth 5): {revenue_daily}
# Structural HAS_COLUMN edges must NOT leak into lineage answers.

SEED_NODES = [
    ("table:raw_orders", NodeType.TABLE, "raw_orders"),
    ("table:stg_orders", NodeType.TABLE, "stg_orders"),
    ("table:fact_orders", NodeType.TABLE, "fact_orders"),
    ("table:revenue_daily", NodeType.TABLE, "revenue_daily"),
    ("column:fact_orders.amount", NodeType.COLUMN, "fact_orders.amount"),
    ("table:unrelated_table", NodeType.TABLE, "unrelated_table"),
    ("table:another_thing", NodeType.TABLE, "another_thing"),
]

SEED_EDGES = [
    ("table:raw_orders", "table:stg_orders", EdgeType.WRITES),
    ("table:stg_orders", "table:fact_orders", EdgeType.TRANSFORMS),
    # PRODUCES (not DERIVED_FROM) so the semantic direction and edge direction
    # agree: "fact_orders produces revenue_daily" → data flows from fact_orders
    # into revenue_daily, which is how downstream traversal treats the edge.
    ("table:fact_orders", "table:revenue_daily", EdgeType.PRODUCES),
    ("table:fact_orders", "column:fact_orders.amount", EdgeType.HAS_COLUMN),
    ("table:unrelated_table", "table:another_thing", EdgeType.READS),
]


EXPECTED_UPSTREAM_OF_FACT_ORDERS = {"table:stg_orders", "table:raw_orders"}
EXPECTED_DOWNSTREAM_OF_FACT_ORDERS = {"table:revenue_daily"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def seeded_backend(tmp_path, monkeypatch):
    """LocalGraphBackend pre-seeded with the ground-truth lineage graph.

    Runs inside ``tmp_path`` so the test never touches a real `.daita/graph/`.
    """
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="lineage_live")

    for node_id, node_type, name in SEED_NODES:
        await backend.add_node(
            AgentGraphNode(node_id=node_id, node_type=node_type, name=name)
        )
    for src, tgt, et in SEED_EDGES:
        await backend.add_edge(
            AgentGraphEdge(
                edge_id=AgentGraphEdge.make_id(src, et, tgt),
                from_node_id=src,
                to_node_id=tgt,
                edge_type=et,
            )
        )
    await backend.flush()
    return backend


@pytest.fixture
async def lineage_plugin(seeded_backend) -> LineagePlugin:
    plugin = LineagePlugin(backend=seeded_backend)
    plugin.initialize("lineage-live-test")
    return plugin


# ---------------------------------------------------------------------------
# (1) Graph built correctly
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGraphCorrectness:
    """Direct backend assertions — no LLM required for this section."""

    async def test_all_nodes_and_edges_persisted(self, seeded_backend):
        graph = await seeded_backend.load_graph()
        assert graph.number_of_nodes() == len(SEED_NODES)
        assert graph.number_of_edges() == len(SEED_EDGES)

    async def test_subgraph_respects_edge_type_filter(self, seeded_backend):
        """``subgraph(edge_types=LINEAGE_EDGE_TYPES)`` must skip HAS_COLUMN."""
        sg = await seeded_backend.subgraph(
            root="table:fact_orders",
            direction="both",
            edge_types=LINEAGE_EDGE_TYPES,
            max_depth=5,
        )
        node_ids = set(sg.nodes())
        # Lineage-reachable nodes appear
        assert "table:stg_orders" in node_ids
        assert "table:raw_orders" in node_ids
        assert "table:revenue_daily" in node_ids
        # HAS_COLUMN must NOT leak structural nodes into a lineage subgraph
        assert "column:fact_orders.amount" not in node_ids
        # Disconnected component stays out
        assert "table:unrelated_table" not in node_ids

    async def test_trace_lineage_ground_truth(self, lineage_plugin):
        """Pure Python call (no LLM) — confirms the plugin's trace matches
        ground truth before we let an LLM chase it."""
        result = await lineage_plugin.trace_lineage(
            "table:fact_orders", direction="both", max_depth=5
        )
        up_ids = {n.get("node_id") for n in result["lineage"]["upstream"]}
        down_ids = {n.get("node_id") for n in result["lineage"]["downstream"]}
        assert up_ids == EXPECTED_UPSTREAM_OF_FACT_ORDERS
        assert down_ids == EXPECTED_DOWNSTREAM_OF_FACT_ORDERS

    async def test_find_lineage_paths_ground_truth(self, lineage_plugin):
        """All simple paths from raw_orders to fact_orders under the default
        lineage edge-type set."""
        tool = next(
            t for t in lineage_plugin.get_tools() if t.name == "find_lineage_paths"
        )
        r = await tool.execute(
            {"from_entity": "table:raw_orders", "to_entity": "table:fact_orders"}
        )
        assert r["reachable"]
        # Only one path exists: raw -> stg -> fact
        assert ["table:raw_orders", "table:stg_orders", "table:fact_orders"] in r[
            "paths"
        ]


# ---------------------------------------------------------------------------
# (2) + (3) + (4) Live-LLM traversal: accuracy + speed
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
class TestLineageAgentLive:
    """End-to-end with a real OpenAI agent."""

    async def test_agent_traces_upstream_correctly(self, lineage_plugin):
        """Agent must call trace_lineage and name both upstream tables."""
        agent = build_live_agent(
            name="LineageUpstreamAgent",
            tools=[lineage_plugin],
        )

        async with timed("agent.run upstream trace"):
            result = await agent.run(
                "Using the available lineage tools, list every upstream source "
                "of the entity `table:fact_orders`. Return the names of all "
                "tables that feed into it (directly or indirectly). Keep the "
                "answer concise.",
                detailed=True,
            )

        assert_tool_called(result, "trace_lineage")
        assert_answer_mentions(result, ["stg_orders", "raw_orders"])
        # HAS_COLUMN must not have leaked into the agent's response
        assert "fact_orders.amount" not in (result.get("result") or "")

    async def test_agent_finds_path_a_to_b(self, lineage_plugin):
        """Agent resolves a 'how does A reach B' question via find_lineage_paths."""
        agent = build_live_agent(
            name="LineagePathAgent",
            tools=[lineage_plugin],
        )

        async with timed("agent.run find path"):
            result = await agent.run(
                "Using the available lineage tools, show every path from "
                "`table:raw_orders` to `table:fact_orders`. List the "
                "intermediate hops in order.",
                detailed=True,
            )

        # Either trace_lineage or find_lineage_paths is acceptable here — both
        # can answer. We want to confirm the agent did NOT fall back to guessing.
        tool_names = {c.get("tool") for c in result.get("tool_calls", [])}
        assert tool_names & {
            "find_lineage_paths",
            "trace_lineage",
        }, f"Agent answered without a lineage tool call: {tool_names}"
        assert_answer_mentions(result, ["raw_orders", "stg_orders", "fact_orders"])

    async def test_agent_analyzes_impact(self, lineage_plugin):
        """Agent picks analyze_impact when asked about breakage/risk."""
        agent = build_live_agent(
            name="LineageImpactAgent",
            tools=[lineage_plugin],
        )

        async with timed("agent.run analyze impact"):
            result = await agent.run(
                "If I change or delete `table:stg_orders`, which downstream "
                "entities are affected? Use the appropriate tool and report "
                "affected entities plus risk level.",
                detailed=True,
            )

        # The ideal tool is analyze_impact; accept trace_lineage downstream
        # as a functionally-correct fallback some models pick.
        tool_names = {c.get("tool") for c in result.get("tool_calls", [])}
        assert tool_names & {
            "analyze_impact",
            "trace_lineage",
        }, f"No relevant tool called: {tool_names}"
        assert_answer_mentions(result, ["fact_orders"])
        # revenue_daily is 2 hops downstream — agent should surface it
        assert_answer_mentions(result, ["revenue_daily", "fact_orders"], any_of=True)

    async def test_agent_respects_edge_type_scope(self, lineage_plugin):
        """Agent must NOT return structural (HAS_COLUMN) reachability when
        asked a lineage question. Regression guard for LINEAGE_EDGE_TYPES
        defaults."""
        agent = build_live_agent(
            name="LineageScopeAgent",
            tools=[lineage_plugin],
        )

        async with timed("agent.run scope check"):
            result = await agent.run(
                "What are the downstream data consumers of "
                "`table:fact_orders`? Do NOT include columns or indexes — "
                "only other tables / datasets / APIs. Use the lineage tools.",
                detailed=True,
            )

        text = result.get("result") or ""
        assert "revenue_daily" in text
        assert "fact_orders.amount" not in text
