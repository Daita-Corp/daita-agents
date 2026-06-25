"""
Live integration test for CatalogPlugin + MongoDB.

Throwaway MongoDB container seeded with two collections whose documents
exhibit a known field structure. The discoverer samples documents to
infer schema; the test verifies the inferred fields match ground truth.

Requirements:
  - docker
  - motor (pip install 'daita-agents[mongodb]')
  - OPENAI_API_KEY (for the live-LLM section)

Run:
    OPENAI_API_KEY=sk-... pytest \\
      tests/integration/catalog/test_catalog_mongodb_live.py -v -s -m "integration"
"""

from __future__ import annotations

import asyncio
import time

import pytest

motor = pytest.importorskip(
    "motor.motor_asyncio",
    reason="motor required: pip install 'daita-agents[mongodb]'",
)

from daita.core.graph import LocalGraphBackend
from daita.plugins.base import PluginContext
from daita.plugins.catalog import CatalogPlugin

from tests.integration._harness import (
    assert_answer_mentions,
    assert_tool_called,
    build_live_agent,
    start_container,
    timed,
)

MONGO_IMAGE = "mongo:7"
MONGO_DB = "daita_test"


SEED_DOCS = {
    "customers": [
        {
            "customer_id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "signup_date": "2026-01-01",
        },
        {
            "customer_id": 2,
            "name": "Bob",
            "email": "bob@example.com",
            "signup_date": "2026-01-02",
        },
    ],
    "orders": [
        {
            "order_id": 1,
            "customer_id": 1,
            "amount": 50.0,
            "status": "shipped",
            "items": 2,
        },
        {
            "order_id": 2,
            "customer_id": 2,
            "amount": 150.0,
            "status": "pending",
            "items": 5,
        },
    ],
}


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
def mongo_container():
    container = start_container(
        MONGO_IMAGE,
        container_port=27017,
        tag_prefix="daita-it-mongo",
        readiness_timeout=90.0,
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def mongo_url(mongo_container) -> str:
    return f"mongodb://{mongo_container.host}:{mongo_container.host_port}"


@pytest.fixture(scope="module")
def seeded_mongo(mongo_url, mongo_container) -> str:
    async def _seed():
        from motor.motor_asyncio import AsyncIOMotorClient

        deadline = time.time() + 60
        last_err = None
        while time.time() < deadline:
            try:
                client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=2000)
                # Force a connection attempt.
                await client.admin.command("ping")
                db = client[MONGO_DB]
                for coll, docs in SEED_DOCS.items():
                    await db[coll].delete_many({})
                    await db[coll].insert_many(docs)
                client.close()
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await asyncio.sleep(1.0)
        raise RuntimeError(f"Could not seed MongoDB: {last_err}")

    asyncio.run(_seed())
    return mongo_url


@pytest.fixture
async def plugin_with_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="catalog_mongo_live")
    plugin = CatalogPlugin(backend=backend, auto_persist=False)
    await _setup_catalog(plugin, "catalog-mongo-live")
    return plugin, backend


# ---------------------------------------------------------------------------
# (1) Catalog plugin works with MongoDB
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestCatalogPluginMongoDB:
    async def test_discover_mongodb_returns_collections(
        self, plugin_with_backend, seeded_mongo
    ):
        plugin, _ = plugin_with_backend

        async with timed("discover_mongodb"):
            result = await plugin.discover_mongodb(
                connection_string=seeded_mongo,
                database=MONGO_DB,
                sample_size=10,
                persist=False,
            )

        # Raw Mongo discovery shape: collections[].collection_name + fields[].field_name
        schema = result.get("schema") or result
        collections = schema.get("collections", [])
        collection_names = {c["collection_name"] for c in collections}
        assert {
            "customers",
            "orders",
        } <= collection_names, f"Expected collections in: {collection_names}"

        orders = next(c for c in collections if c["collection_name"] == "orders")
        orders_fields = {f["field_name"] for f in orders.get("fields", [])}
        for f in ["order_id", "customer_id", "amount", "status", "items"]:
            assert f in orders_fields, f"Missing inferred field {f} in {orders_fields}"


# ---------------------------------------------------------------------------
# (2) Graph built correctly
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
class TestMongoDBGraphCorrectness:
    async def test_persist_populates_graph(self, plugin_with_backend, seeded_mongo):
        from daita.core.graph.models import NodeType

        plugin, backend = plugin_with_backend

        async with timed("discover_mongodb + persist"):
            await plugin.discover_mongodb(
                connection_string=seeded_mongo,
                database=MONGO_DB,
                sample_size=10,
                persist=True,
            )

        tables = await backend.find_nodes(NodeType.TABLE)
        names = {t.name.split(".")[-1] for t in tables}
        assert {"customers", "orders"} <= names, f"Got: {names}"


# ---------------------------------------------------------------------------
# (3) + (4) Live-LLM traversal: accuracy + speed
# ---------------------------------------------------------------------------


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.requires_db
class TestAgentMongoDBLive:
    async def test_agent_describes_collections(self, plugin_with_backend, seeded_mongo):
        plugin, _ = plugin_with_backend
        discovery = await plugin.discover_mongodb(
            connection_string=seeded_mongo,
            database=MONGO_DB,
            sample_size=10,
            persist=True,
        )
        store_id = discovery["store_id"]

        agent = build_live_agent(name="MongoCatalogAgent", tools=[plugin])
        async with timed("agent.run mongo describe"):
            result = await agent.run(
                f"Use the catalog tools for store `{store_id}`. Tell me "
                f"which collections exist and what fields the `orders` "
                f"documents contain. Be concise.",
                detailed=True,
            )

        assert_tool_called(result, "catalog_search_schema")
        assert_answer_mentions(result, ["customers", "orders"])
        assert_answer_mentions(
            result,
            ["order_id", "customer_id", "amount", "status"],
            any_of=True,
        )
