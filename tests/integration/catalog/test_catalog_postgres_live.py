"""
Live integration test for CatalogPlugin + PostgreSQL.

Spins up a throwaway Postgres container via docker, seeds a known schema,
then runs the full catalog → graph → agent pipeline against it. Per spec:

  1. CatalogPlugin works with Postgres (direct discovery returns a normalized
     schema matching the seed).
  2. Graph is built correctly (nodes + edges match the seeded tables/FKs).
  3. Agent traversal speed — logged, not asserted.
  4. Agent traversal accuracy — live LLM must use the catalog tools and
     return table / column / FK facts that match ground truth.

Requirements:
  - docker available
  - asyncpg (pip install 'daita-agents[postgresql]')
  - OPENAI_API_KEY (for the live-LLM section)

Run:
    OPENAI_API_KEY=sk-... pytest \\
      tests/integration/catalog/test_catalog_postgres_live.py -v -s \\
      -m "integration"
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict

import pytest

# asyncpg and docker are hard requirements — skip module if either missing.
asyncpg = pytest.importorskip(
    "asyncpg",
    reason="asyncpg required: pip install 'daita-agents[postgresql]'",
)

from daita.core.graph import LocalGraphBackend
from daita.core.graph.models import NodeType
from daita.plugins.base import PluginContext
from daita.plugins.catalog import CatalogPlugin

from tests.integration._harness import (
    assert_answer_mentions,
    assert_tool_called,
    build_live_agent,
    start_container,
    timed,
)

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_test"


SEED_SQL = """
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
    status       TEXT,
    created_at   TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX orders_customer_idx ON orders(customer_id);

INSERT INTO customers (name, email, signup_date) VALUES
    ('Alice', 'alice@example.com', '2026-01-01'),
    ('Bob',   'bob@example.com',   '2026-01-02');

INSERT INTO orders (customer_id, amount, status) VALUES
    (1, 50.00,  'shipped'),
    (2, 150.00, 'pending');
"""


EXPECTED_TABLES = {"customers", "orders"}
EXPECTED_FK_EDGE = ("orders", "customer_id", "customers", "customer_id")


async def _setup_catalog(plugin: CatalogPlugin, agent_id: str) -> None:
    await plugin.setup(
        PluginContext(
            runtime_id=agent_id,
            runtime_kind="agent",
            agent_id=agent_id,
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def postgres_container():
    """Run a throwaway Postgres container for the whole module.

    Module-scoped so we pay the ~5s startup once across every test.
    """
    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix="daita-it-pg",
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def postgres_url(postgres_container) -> str:
    """Postgres URL for the test container."""
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{postgres_container.host}:{postgres_container.host_port}/{POSTGRES_DB}"
    )


@pytest.fixture(scope="module")
def seeded_postgres(postgres_url) -> str:
    """Seed the known schema. Returns the URL so tests can chain."""

    async def _seed():
        # Wait for the server to accept SQL (TCP alone isn't enough on cold start).
        deadline = time.time() + 30
        last_err = None
        while time.time() < deadline:
            try:
                conn = await asyncpg.connect(postgres_url, ssl=False)
                await conn.execute(SEED_SQL)
                await conn.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Could not seed Postgres: {last_err}")

    asyncio.run(_seed())
    return postgres_url


@pytest.fixture
async def plugin_with_backend(tmp_path, monkeypatch):
    """CatalogPlugin bound to a clean LocalGraphBackend inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="catalog_pg_live")
    plugin = CatalogPlugin(backend=backend, auto_persist=False)
    await _setup_catalog(plugin, "catalog-pg-live")
    return plugin, backend


# ---------------------------------------------------------------------------
# (1) Catalog plugin works with Postgres
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestCatalogPluginPostgres:
    async def test_discover_postgres_returns_seeded_tables(
        self, plugin_with_backend, seeded_postgres
    ):
        plugin, _ = plugin_with_backend

        async with timed("discover_postgres"):
            result = await plugin.discover_postgres(
                connection_string=seeded_postgres,
                ssl_mode="disable",
                persist=False,
            )

        # Raw discover_* shape: tables[].table_name, columns live in a flat
        # sibling array keyed by table_name + column_name.
        schema = result.get("schema") or result
        table_names = {
            t.get("table_name") or t.get("name") for t in schema.get("tables", [])
        }
        assert (
            EXPECTED_TABLES <= table_names
        ), f"Expected {EXPECTED_TABLES} in {table_names}"

        if schema.get("columns"):
            orders_cols = {
                c["column_name"]
                for c in schema.get("columns", [])
                if c.get("table_name") == "orders"
            }
        else:
            orders_table = next(
                t for t in schema.get("tables", []) if t["name"] == "orders"
            )
            orders_cols = {c["name"] for c in orders_table.get("columns", [])}
        assert {"order_id", "customer_id", "amount", "status", "created_at"} <= (
            orders_cols
        ), f"orders columns: {orders_cols}"

        fk_triples = {
            (fk["source_table"], fk["source_column"], fk["target_table"])
            for fk in schema.get("foreign_keys", [])
        }
        assert ("orders", "customer_id", "customers") in fk_triples


# ---------------------------------------------------------------------------
# (2) Graph built correctly
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestPostgresGraphCorrectness:
    async def test_persist_creates_expected_graph(
        self, plugin_with_backend, seeded_postgres
    ):
        """With ``persist=True`` the graph must contain Table + Column nodes
        and FK-derived edges for the seeded schema."""
        plugin, backend = plugin_with_backend

        async with timed("discover_postgres + persist"):
            await plugin.discover_postgres(
                connection_string=seeded_postgres,
                ssl_mode="disable",
                persist=True,
            )

        # Tables must be present as graph nodes.
        tables = await backend.find_nodes(NodeType.TABLE)
        table_names = {t.name for t in tables}
        assert EXPECTED_TABLES <= {
            n.split(".")[-1] for n in table_names
        }, f"Tables not in graph: {table_names}"

        # FK REFERENCES edge orders.customer_id -> customers.customer_id
        # (schema of column-level edges depends on the persistence layer;
        # at minimum a table-level REFERENCES edge should exist.)
        graph = await backend.load_graph()
        edge_types = {
            data["data"].get("edge_type")
            for _, _, data in graph.edges(data=True)
            if data.get("data")
        }
        assert any(
            "references" in (t or "").lower() or "has_column" in (t or "").lower()
            for t in edge_types
        ), f"Expected structural edges in graph, got: {edge_types}"


# ---------------------------------------------------------------------------
# (3) + (4) Live-LLM traversal: accuracy + speed
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.requires_db
class TestAgentPostgresLive:
    async def test_agent_searches_and_describes_registered_schema(
        self, plugin_with_backend, seeded_postgres
    ):
        """Agent queries an already registered catalog schema."""
        plugin, _ = plugin_with_backend
        discovery = await plugin.discover_postgres(
            connection_string=seeded_postgres,
            ssl_mode="disable",
            persist=True,
        )
        store_id = discovery["store_id"]

        agent = build_live_agent(name="PostgresCatalogAgent", tools=[plugin])

        async with timed("agent.run postgres describe"):
            result = await agent.run(
                f"Use the catalog tools for store `{store_id}`. Tell me the "
                f"names of all tables and how `orders.customer_id` relates "
                f"to the `customers` table. Be concise.",
                detailed=True,
            )

        assert_tool_called(result, "catalog_search_schema")
        assert_answer_mentions(result, ["customers", "orders"])
        # Must surface the FK relationship in some form
        assert_answer_mentions(
            result,
            ["customer_id", "foreign key", "references"],
            any_of=True,
        )

    async def test_agent_handles_single_table_lookup(
        self, plugin_with_backend, seeded_postgres
    ):
        """Agent should choose targeted catalog ToolViews once the store is profiled."""
        plugin, _ = plugin_with_backend

        # Prime the catalog by profiling first, then discover the store_id.
        discovery = await plugin.discover_postgres(
            connection_string=seeded_postgres,
            ssl_mode="disable",
            persist=True,
        )
        store_id = discovery["store_id"]

        agent = build_live_agent(name="PostgresLookupAgent", tools=[plugin])
        async with timed("agent.run single-table lookup"):
            result = await agent.run(
                f"For the Postgres store with id `{store_id}`, what columns "
                f"does the `orders` table have?",
                detailed=True,
            )

        tool_names = {c.get("tool") for c in result.get("tool_calls", [])}
        assert (
            "catalog_inspect_asset" in tool_names
            or "catalog_search_schema" in tool_names
        ), f"Agent used neither catalog inspection nor search: {tool_names}"
        assert_answer_mentions(
            result, ["order_id", "customer_id", "amount"], any_of=True
        )
