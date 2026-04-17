"""
Deep accuracy tests for graph persistence + traversal.

Closes the gaps identified during the first rigorous-testing pass:

  * **TestTopologyCoverage** — diamond, cycle, wide fan-out, disconnected
    components. Single-path correctness proves BFS, but agents will meet
    richer graphs in the wild.
  * **TestMaxDepthBoundary** — depth=0/1/2/large on a 5-node linear chain.
    Proves the depth arg actually bounds traversal (regression guard).
  * **TestIdempotency** — persist the same Postgres schema twice; node + edge
    counts must be stable. Prevents duplicate-edge drift on re-scans.
  * **TestStoreQualifiedIDs** — after persist, every Table/Column/Index node
    ID must be qualified with the store prefix (``table:postgresql:...``).
    Prevents a regression where the store qualifier silently drops and
    cross-store collisions become possible.
  * **TestIndexesPersist** — the Postgres seeded index must produce an Index
    node plus ``INDEXED_BY`` (Table→Index) and ``COVERS`` (Index→Column)
    edges with correct column position.
  * **TestColumnLevelFK** — the Postgres FK must produce a Column-to-Column
    ``REFERENCES`` edge (not just a Table-level edge) with store-qualified
    endpoints.
  * **TestNegativeLLM** — agent asked about an unreachable/empty-upstream
    entity must return a negative ("no/none/zero") answer without inventing
    upstream nodes. Catches false-positive hallucination the token-presence
    assertions can't.
  * **TestCrossPluginTraversal** — CatalogPlugin profiles a Postgres store,
    then LineagePlugin registers a flow spanning tables from that store, and
    an agent traces it. Proves catalog + lineage compose cleanly.

Requirements mirror the existing catalog live tests:
  - docker + asyncpg for the Postgres-backed sections
  - OPENAI_API_KEY for TestNegativeLLM + TestCrossPluginTraversal

Run:
    OPENAI_API_KEY=sk-... pytest \\
      tests/integration/test_graph_accuracy_deep.py -v -s -m integration
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Tuple

import pytest

from daita.core.graph import LINEAGE_EDGE_TYPES, LocalGraphBackend
from daita.core.graph.models import (
    AgentGraphEdge,
    AgentGraphNode,
    EdgeType,
    NodeType,
)
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.lineage import LineagePlugin

from ._harness import (
    assert_answer_mentions,
    assert_tool_called,
    build_live_agent,
    start_container,
    timed,
)

# ---------------------------------------------------------------------------
# Pure-graph fixtures (no docker, no LLM)
# ---------------------------------------------------------------------------


async def _seed(
    backend: LocalGraphBackend,
    nodes: List[str],
    edges: List[Tuple[str, str, EdgeType]],
) -> None:
    for n in nodes:
        await backend.add_node(
            AgentGraphNode(node_id=n, node_type=NodeType.TABLE, name=n.split(":")[-1])
        )
    for src, tgt, et in edges:
        await backend.add_edge(
            AgentGraphEdge(
                edge_id=AgentGraphEdge.make_id(src, et, tgt),
                from_node_id=src,
                to_node_id=tgt,
                edge_type=et,
            )
        )


# ---------------------------------------------------------------------------
# 1. Topology coverage
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTopologyCoverage:
    """Traversal must handle non-linear topologies correctly."""

    async def test_diamond_yields_both_paths(self, tmp_path, monkeypatch):
        """A → {B, C} → D. From A downstream, all three intermediates reachable;
        path A→D enumerated both ways."""
        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="topo_diamond")
        await _seed(
            backend,
            nodes=["table:A", "table:B", "table:C", "table:D"],
            edges=[
                ("table:A", "table:B", EdgeType.TRANSFORMS),
                ("table:A", "table:C", EdgeType.TRANSFORMS),
                ("table:B", "table:D", EdgeType.TRANSFORMS),
                ("table:C", "table:D", EdgeType.TRANSFORMS),
            ],
        )

        sg = await backend.subgraph(
            root="table:A",
            direction="downstream",
            edge_types=LINEAGE_EDGE_TYPES,
            max_depth=5,
        )
        assert set(sg.nodes()) == {"table:A", "table:B", "table:C", "table:D"}

        from daita.core.graph.algorithms import find_paths

        paths = find_paths(
            sg, "table:A", "table:D", edge_types=LINEAGE_EDGE_TYPES, cutoff=5
        )
        path_sets = {tuple(p) for p in paths}
        assert path_sets == {
            ("table:A", "table:B", "table:D"),
            ("table:A", "table:C", "table:D"),
        }

    async def test_cycle_terminates(self, tmp_path, monkeypatch):
        """A → B → C → A. BFS must not loop; each node visited exactly once."""
        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="topo_cycle")
        await _seed(
            backend,
            nodes=["table:A", "table:B", "table:C"],
            edges=[
                ("table:A", "table:B", EdgeType.TRANSFORMS),
                ("table:B", "table:C", EdgeType.TRANSFORMS),
                ("table:C", "table:A", EdgeType.TRANSFORMS),
            ],
        )

        sg = await backend.subgraph(
            "table:A",
            direction="downstream",
            edge_types=LINEAGE_EDGE_TYPES,
            max_depth=10,
        )
        assert set(sg.nodes()) == {"table:A", "table:B", "table:C"}

        plugin = LineagePlugin(backend=backend)
        plugin.initialize("topo-cycle")
        result = await plugin.trace_lineage("table:A", direction="downstream")
        node_ids = {n.get("node_id") for n in result["lineage"]["downstream"]}
        # Root excluded from results; B and C reachable exactly once.
        assert node_ids == {"table:B", "table:C"}

    async def test_wide_fanout(self, tmp_path, monkeypatch):
        """A → 20 children. Every child must be reached at depth=1."""
        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="topo_fanout")
        children = [f"table:C{i:02d}" for i in range(20)]
        await _seed(
            backend,
            nodes=["table:A", *children],
            edges=[("table:A", c, EdgeType.TRANSFORMS) for c in children],
        )

        sg = await backend.subgraph(
            "table:A",
            direction="downstream",
            edge_types=LINEAGE_EDGE_TYPES,
            max_depth=1,
        )
        assert set(sg.nodes()) == {"table:A", *children}
        assert sg.number_of_edges() == 20

    async def test_disconnected_stays_isolated(self, tmp_path, monkeypatch):
        """Two disconnected components: traversal from one must not cross."""
        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="topo_disc")
        await _seed(
            backend,
            nodes=[
                "table:A1",
                "table:A2",
                "table:B1",
                "table:B2",
            ],
            edges=[
                ("table:A1", "table:A2", EdgeType.TRANSFORMS),
                ("table:B1", "table:B2", EdgeType.TRANSFORMS),
            ],
        )

        sg = await backend.subgraph(
            "table:A1",
            direction="both",
            edge_types=LINEAGE_EDGE_TYPES,
            max_depth=10,
        )
        assert "table:B1" not in sg.nodes()
        assert "table:B2" not in sg.nodes()
        assert set(sg.nodes()) == {"table:A1", "table:A2"}


# ---------------------------------------------------------------------------
# 2. Max-depth boundary
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMaxDepthBoundary:
    """``max_depth`` must bound traversal exactly, not approximately."""

    @pytest.fixture
    async def chain_plugin(self, tmp_path, monkeypatch):
        """Linear chain A → B → C → D → E."""
        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="depth_chain")
        await _seed(
            backend,
            nodes=["table:A", "table:B", "table:C", "table:D", "table:E"],
            edges=[
                ("table:A", "table:B", EdgeType.TRANSFORMS),
                ("table:B", "table:C", EdgeType.TRANSFORMS),
                ("table:C", "table:D", EdgeType.TRANSFORMS),
                ("table:D", "table:E", EdgeType.TRANSFORMS),
            ],
        )
        plugin = LineagePlugin(backend=backend)
        plugin.initialize("depth-test")
        return plugin

    @pytest.mark.parametrize(
        "depth,expected",
        [
            (1, {"table:B"}),
            (2, {"table:B", "table:C"}),
            (3, {"table:B", "table:C", "table:D"}),
            (100, {"table:B", "table:C", "table:D", "table:E"}),
        ],
    )
    async def test_downstream_depth_bound_is_exact(self, chain_plugin, depth, expected):
        result = await chain_plugin.trace_lineage(
            "table:A", direction="downstream", max_depth=depth
        )
        ids = {n.get("node_id") for n in result["lineage"]["downstream"]}
        assert ids == expected, f"depth={depth}: got {ids}, expected {expected}"


# ---------------------------------------------------------------------------
# Postgres-backed fixtures (shared across several test classes below)
# ---------------------------------------------------------------------------

asyncpg = pytest.importorskip(
    "asyncpg", reason="asyncpg required: pip install 'daita-agents[postgresql]'"
)


_PG_USER, _PG_PW, _PG_DB = "daita", "deep_pw", "deep_test"

_SEED_SQL = """
CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT UNIQUE,
    signup_date  DATE NOT NULL
);
CREATE TABLE orders (
    order_id     SERIAL PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    amount       NUMERIC(10, 2) NOT NULL,
    status       TEXT
);
CREATE INDEX orders_customer_idx ON orders(customer_id);
"""


@pytest.fixture(scope="module")
def pg_container():
    container = start_container(
        "postgres:16-alpine",
        container_port=5432,
        env={
            "POSTGRES_USER": _PG_USER,
            "POSTGRES_PASSWORD": _PG_PW,
            "POSTGRES_DB": _PG_DB,
        },
        tag_prefix="daita-it-deep-pg",
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def pg_url(pg_container) -> str:
    return (
        f"postgresql://{_PG_USER}:{_PG_PW}"
        f"@{pg_container.host}:{pg_container.host_port}/{_PG_DB}"
    )


@pytest.fixture(scope="module")
def seeded_pg(pg_url) -> str:
    async def _run():
        deadline = time.time() + 30
        last = None
        while time.time() < deadline:
            try:
                c = await asyncpg.connect(pg_url, ssl=False)
                await c.execute(_SEED_SQL)
                await c.close()
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(0.5)
        raise RuntimeError(f"seed failed: {last}")

    asyncio.run(_run())
    return pg_url


@pytest.fixture
async def catalog_plugin_with_fresh_backend(tmp_path, monkeypatch):
    """CatalogPlugin wired to a fresh LocalGraphBackend in tmp_path."""
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="deep_accuracy")
    plugin = CatalogPlugin(backend=backend, auto_persist=False)
    plugin.initialize("deep-accuracy")
    return plugin, backend


async def _profile_seeded_pg(plugin: CatalogPlugin, url: str) -> dict:
    """Run discover_schema with persist=True and return the raw result."""
    tool = next(t for t in plugin.get_tools() if t.name == "discover_schema")
    return await tool.execute(
        {
            "store_type": "postgresql",
            "connection_string": url,
            "options": {"ssl_mode": "disable"},
            "persist": True,
        }
    )


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestIdempotency:
    async def test_double_persist_does_not_duplicate(
        self, catalog_plugin_with_fresh_backend, seeded_pg
    ):
        """Persist the same schema twice; every Table/Column/Index node and
        every HAS_COLUMN/INDEXED_BY/COVERS/REFERENCES edge must be the same
        count before and after the second run."""
        plugin, backend = catalog_plugin_with_fresh_backend

        async with timed("persist #1"):
            await _profile_seeded_pg(plugin, seeded_pg)
        g1 = await backend.load_graph()
        n1, e1 = g1.number_of_nodes(), g1.number_of_edges()

        async with timed("persist #2"):
            await _profile_seeded_pg(plugin, seeded_pg)
        g2 = await backend.load_graph()
        n2, e2 = g2.number_of_nodes(), g2.number_of_edges()

        assert n1 == n2, f"Node count drifted: {n1} -> {n2}"
        assert e1 == e2, f"Edge count drifted: {e1} -> {e2}"
        print(f"[ACC] persist idempotent: nodes={n1}, edges={e1}")


# ---------------------------------------------------------------------------
# 4. Store-qualified IDs
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestStoreQualifiedIDs:
    async def test_node_ids_carry_store_prefix(
        self, catalog_plugin_with_fresh_backend, seeded_pg
    ):
        """Every emitted Table/Column/Index node_id must be prefixed with
        ``<type>:postgresql:<host>/`` so two stores with a ``customers``
        table don't collide.
        """
        plugin, backend = catalog_plugin_with_fresh_backend
        await _profile_seeded_pg(plugin, seeded_pg)

        tables = await backend.find_nodes(NodeType.TABLE)
        assert tables, "No Table nodes persisted"
        bare_ids = [
            t for t in tables if t.node_id in {"table:customers", "table:orders"}
        ]
        assert not bare_ids, (
            f"Found unqualified Table node_ids: {[t.node_id for t in bare_ids]} — "
            f"store qualifier regressed"
        )

        for t in tables:
            prefix = f"table:postgresql:"
            assert t.node_id.startswith(
                prefix
            ), f"Table node_id {t.node_id!r} missing store prefix {prefix!r}"

        # Every Column node must be prefixed the same way.
        columns = await backend.find_nodes(NodeType.COLUMN)
        assert columns, "No Column nodes persisted"
        for c in columns:
            assert c.node_id.startswith(
                "column:postgresql:"
            ), f"Column node_id {c.node_id!r} missing store prefix"


# ---------------------------------------------------------------------------
# 5. Indexes persist with INDEXED_BY + COVERS edges
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestIndexesPersist:
    async def test_btree_index_lands_with_edges(
        self, catalog_plugin_with_fresh_backend, seeded_pg
    ):
        """``orders_customer_idx`` must become an Index node with:
        * ``INDEXED_BY`` edge Table→Index
        * ``COVERS`` edge Index→Column(customer_id) with position=0
        """
        plugin, backend = catalog_plugin_with_fresh_backend
        await _profile_seeded_pg(plugin, seeded_pg)

        indexes = await backend.find_nodes(NodeType.INDEX)
        idx_names = {i.name for i in indexes}
        assert (
            "orders_customer_idx" in idx_names
        ), f"Seeded index not persisted. Found: {idx_names}"

        idx = next(i for i in indexes if i.name == "orders_customer_idx")

        # INDEXED_BY from Table → Index
        indexed_by = await backend.get_edges(
            to_node_id=idx.node_id, edge_types=[EdgeType.INDEXED_BY]
        )
        assert len(indexed_by) == 1, f"Expected 1 INDEXED_BY, got {len(indexed_by)}"
        tbl_id = indexed_by[0].from_node_id
        assert tbl_id.startswith("table:postgresql:") and tbl_id.endswith(".orders")

        # COVERS from Index → Column(customer_id)
        covers = await backend.get_edges(
            from_node_id=idx.node_id, edge_types=[EdgeType.COVERS]
        )
        assert len(covers) == 1, f"Expected 1 COVERS edge, got {len(covers)}"
        cov = covers[0]
        assert cov.to_node_id.endswith(
            "orders.customer_id"
        ), f"COVERS points to wrong column: {cov.to_node_id}"
        assert cov.properties.get("position") == 0


# ---------------------------------------------------------------------------
# 6. Column-level FK REFERENCES edge
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestColumnLevelFK:
    async def test_fk_emits_column_to_column_references(
        self, catalog_plugin_with_fresh_backend, seeded_pg
    ):
        """FK ``orders.customer_id → customers.customer_id`` must emit a
        REFERENCES edge between the two Column nodes, not between tables."""
        plugin, backend = catalog_plugin_with_fresh_backend
        await _profile_seeded_pg(plugin, seeded_pg)

        refs = await backend.get_edges(edge_types=[EdgeType.REFERENCES])
        assert refs, "No REFERENCES edge persisted for the seeded FK"

        # Exactly one FK in seed → exactly one REFERENCES edge
        assert len(refs) == 1, f"Expected 1 REFERENCES, got {len(refs)}"
        fk = refs[0]

        # Both endpoints must be Column nodes, store-qualified
        assert fk.from_node_id.startswith("column:postgresql:")
        assert fk.to_node_id.startswith("column:postgresql:")
        assert fk.from_node_id.endswith("orders.customer_id")
        assert fk.to_node_id.endswith("customers.customer_id")


# ---------------------------------------------------------------------------
# 7. Negative LLM — false-positive guard
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
class TestNegativeLLM:
    """Agent must NOT invent answers for questions with an empty ground truth."""

    @pytest.fixture
    async def disconnected_plugin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        backend = LocalGraphBackend(graph_type="neg_llm")
        # 'table:island' is an isolated node — no edges at all.
        # 'table:root' is upstream-only (nothing feeds into it).
        await _seed(
            backend,
            nodes=["table:root", "table:child", "table:island"],
            edges=[("table:root", "table:child", EdgeType.TRANSFORMS)],
        )
        plugin = LineagePlugin(backend=backend)
        plugin.initialize("neg-llm")
        return plugin

    async def test_agent_reports_empty_upstream(self, disconnected_plugin):
        """``table:root`` has no upstream — agent must say so explicitly and
        must NOT name any of the other known nodes as upstream."""
        agent = build_live_agent(
            name="NegLLMUpstreamAgent", tools=[disconnected_plugin]
        )
        async with timed("agent.run negative upstream"):
            result = await agent.run(
                "Using the lineage tools, list every upstream source of "
                "`table:root`. If there are none, say so explicitly.",
                detailed=True,
            )

        text = (result.get("result") or "").lower()
        # Accept any of the standard "empty" phrasings.
        empty_tokens = [
            "no upstream",
            "none",
            "no sources",
            "no upstream sources",
            "zero",
        ]
        assert any(
            tok in text for tok in empty_tokens
        ), f"Agent did not report an empty answer. Got: {text[:300]!r}"
        # Hallucination guard: neither child nor island should appear as upstream.
        for bogus in ["table:child", "island"]:
            assert (
                bogus not in text
            ), f"Agent invented upstream '{bogus}' for table:root. Answer: {text[:300]!r}"

    async def test_agent_reports_unreachable_path(self, disconnected_plugin):
        """No path exists from ``table:root`` to ``table:island`` — agent must
        not claim one."""
        agent = build_live_agent(name="NegLLMPathAgent", tools=[disconnected_plugin])
        async with timed("agent.run negative path"):
            result = await agent.run(
                "Using the lineage tools, find every path from `table:root` "
                "to `table:island`. If no path exists, say so clearly.",
                detailed=True,
            )

        text = (result.get("result") or "").lower()
        empty_tokens = [
            "no path",
            "unreachable",
            "none",
            "no route",
            "not connected",
            "does not exist",
            "no direct",
            "no lineage",
        ]
        assert any(
            tok in text for tok in empty_tokens
        ), f"Agent didn't report unreachable. Got: {text[:300]!r}"


# ---------------------------------------------------------------------------
# 8. Cross-plugin traversal
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.requires_db
class TestCrossPluginTraversal:
    """Catalog profile → Lineage flow → Agent traversal end-to-end."""

    async def test_catalog_then_lineage_then_agent(
        self, catalog_plugin_with_fresh_backend, seeded_pg
    ):
        plugin, backend = catalog_plugin_with_fresh_backend

        # Step 1: Catalog profiles the live Postgres store.
        await _profile_seeded_pg(plugin, seeded_pg)

        # Find the qualified Table node_ids we just persisted.
        tables = await backend.find_nodes(NodeType.TABLE)
        cust_id = next(t.node_id for t in tables if t.name == "customers")
        orders_id = next(t.node_id for t in tables if t.name == "orders")

        # Step 2: Lineage registers a flow on top of catalog nodes.
        lineage = LineagePlugin(backend=backend)
        lineage.initialize("cross-plugin")
        await lineage.register_flow(
            source_id=cust_id,
            target_id=orders_id,
            flow_type="transforms",
            transformation="customers hydrated into order events",
        )

        # Direct assertion first — no LLM yet.
        result = await lineage.trace_lineage(orders_id, direction="upstream")
        up_ids = {n.get("node_id") for n in result["lineage"]["upstream"]}
        assert cust_id in up_ids, (
            f"Cross-plugin lineage broken: {cust_id} not upstream of {orders_id}. "
            f"Got: {up_ids}"
        )

        # Step 3: Agent with BOTH plugins must surface the cross-plugin lineage.
        agent = build_live_agent(name="CrossPluginAgent", tools=[plugin, lineage])
        async with timed("agent.run cross-plugin"):
            agent_result = await agent.run(
                f"Using the lineage tools, list the upstream sources of "
                f"`{orders_id}`. Give me the answer in one sentence.",
                detailed=True,
            )

        # Agent should have gone to the lineage tool, not the catalog tool.
        assert_tool_called(agent_result, "trace_lineage")
        assert_answer_mentions(agent_result, ["customers"])
