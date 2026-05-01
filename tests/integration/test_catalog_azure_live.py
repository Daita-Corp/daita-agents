"""
Live integration test for CatalogPlugin + Azure — strictly read-only.

Uses the active Azure credentials (environment, managed identity, Azure CLI,
etc.) to run the live ``AzureDiscoverer`` against configured subscription(s).
The test issues only list/get/read management-plane calls.

Requirements:
  - Azure SDK packages (pip install 'daita-agents[azure]')
  - Azure credentials resolvable by DefaultAzureCredential
  - OPENAI_API_KEY (for the live-LLM section)
  - AZURE_SUBSCRIPTIONS (CSV) or access to list subscriptions
  - AZURE_LOCATIONS (optional CSV) to constrain scan scope

Run:
    OPENAI_API_KEY=sk-... \\
    AZURE_SUBSCRIPTIONS=00000000-0000-0000-0000-000000000000 \\
      AZURE_LOCATIONS=eastus,westus2 \\
      pytest tests/integration/test_catalog_azure_live.py -v -s -m "integration"
"""

import os

import pytest

pytest.importorskip(
    "azure.identity",
    reason="azure-identity required: pip install 'daita-agents[azure]'",
)

from daita.core.graph import LocalGraphBackend
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.azure import AzureDiscoverer

from ._harness import (
    assert_tool_called,
    build_live_agent,
    timed,
)

_AZURE_NATIVE_PROFILERS = {
    "azure_blob",
    "cosmosdb",
    "eventhub",
    "servicebus_queue",
    "servicebus_topic",
    "azure_apim",
}


def _subscriptions() -> list[str]:
    raw = os.environ.get("AZURE_SUBSCRIPTIONS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _add_azure_profilers(plugin: CatalogPlugin) -> None:
    """Register all Azure-native profilers on a CatalogPlugin."""
    from daita.plugins.catalog.profiler import (
        AzureAPIMProfiler,
        AzureBlobProfiler,
        AzureCosmosDBProfiler,
        AzureEventHubProfiler,
        AzureServiceBusQueueProfiler,
        AzureServiceBusTopicProfiler,
    )

    plugin.add_profiler(AzureBlobProfiler())
    plugin.add_profiler(AzureCosmosDBProfiler())
    plugin.add_profiler(AzureEventHubProfiler())
    plugin.add_profiler(AzureServiceBusQueueProfiler())
    plugin.add_profiler(AzureServiceBusTopicProfiler())
    plugin.add_profiler(AzureAPIMProfiler())


@pytest.fixture(scope="module")
def azure_guard():
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.subscription import SubscriptionClient

        credential = DefaultAzureCredential()
        if _subscriptions():
            return
        subs = list(SubscriptionClient(credential).subscriptions.list())
        if not subs:
            pytest.skip("No Azure subscriptions visible to current credentials")
    except Exception as exc:
        pytest.skip(f"Azure credentials not reachable: {exc}")


@pytest.fixture
def azure_discoverer(azure_guard) -> AzureDiscoverer:
    return AzureDiscoverer()


@pytest.fixture
async def plugin_with_azure(tmp_path, monkeypatch, azure_discoverer):
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="catalog_azure_live")
    plugin = CatalogPlugin(backend=backend, auto_persist=False)
    plugin.add_discoverer(azure_discoverer)
    plugin.initialize("catalog-azure-live")
    return plugin, backend


@pytest.fixture
async def profiled_azure_catalog_with_graph(tmp_path, monkeypatch, azure_discoverer):
    """Discover/profile live Azure resources and emit their schemas to graph."""
    monkeypatch.chdir(tmp_path)
    backend = LocalGraphBackend(graph_type="catalog_azure_graph_live")
    plugin = CatalogPlugin(backend=backend, auto_persist=True)
    plugin.add_discoverer(azure_discoverer)
    _add_azure_profilers(plugin)
    plugin.initialize("catalog-azure-graph-live")

    async with timed("catalog.discover_profile_graph azure"):
        await plugin.discover_and_profile()

    return plugin, backend


@pytest.mark.integration
class TestAzureDiscovererLive:
    """Direct discoverer calls against a real Azure subscription."""

    async def test_authenticate_resolves_subscriptions(self, azure_discoverer):
        await azure_discoverer.authenticate()
        assert azure_discoverer._subscriptions

    async def test_test_access_true(self, azure_discoverer):
        assert await azure_discoverer.test_access() is True

    async def test_enumerate_yields_or_empty(self, azure_discoverer):
        """enumerate() may return zero or more stores, but must not raise."""
        stores = []
        async with timed("azure enumerate()"):
            async for store in azure_discoverer.enumerate():
                stores.append(store)

        types = sorted({s.store_type for s in stores})
        subscriptions = sorted({s.metadata.get("subscription_id", "") for s in stores})
        print(
            f"\n[AZURE LIVE] discovered {len(stores)} store(s); "
            f"types={types or '-'}; subscriptions={subscriptions or '-'}"
        )

        for store in stores:
            assert store.id, "Every store must have a fingerprint id"
            assert store.store_type
            assert store.source.startswith("azure")
            assert store.metadata.get("subscription_id")


@pytest.mark.integration
class TestCatalogPluginAzure:
    """CatalogPlugin orchestration against live Azure."""

    async def test_discover_all(self, plugin_with_azure):
        plugin, _ = plugin_with_azure
        async with timed("catalog.discover_all azure"):
            result = await plugin.discover_all()

        assert not result.has_errors, f"Azure discoverer raised: {result.errors}"
        assert len(plugin.get_stores()) == result.store_count

    async def test_find_store_by_type(self, plugin_with_azure):
        """get_stores(store_type=...) filters correctly after a live scan."""
        plugin, _ = plugin_with_azure
        await plugin.discover_all()
        all_stores = plugin.get_stores()
        if not all_stores:
            pytest.skip("No stores in target Azure subscription(s) to filter on")

        first_type = all_stores[0].store_type
        filtered = plugin.get_stores(store_type=first_type)
        assert filtered, f"Expected at least one {first_type} store"
        assert all(store.store_type == first_type for store in filtered)


@pytest.mark.integration
class TestAzureProfilersLive:
    """Azure-native stores flow through discover -> normalize -> profile."""

    async def test_azure_native_stores_profile(self, azure_discoverer):
        plugin = CatalogPlugin()
        plugin.add_discoverer(azure_discoverer)
        _add_azure_profilers(plugin)

        async with timed("catalog.discover_and_profile azure"):
            await plugin.discover_and_profile()

        stores = [
            store
            for store in plugin.get_stores()
            if store.store_type in _AZURE_NATIVE_PROFILERS
        ]
        if not stores:
            pytest.skip("No Azure-native profileable stores discovered")

        profiled = []
        for store in stores:
            schema = plugin.get_schema(store.id)
            if schema is None:
                continue
            profiled.append((store, schema))
            assert schema.store_id == store.id
            assert schema.database_type == store.store_type
            assert schema.database_name
            assert schema.table_count == len(schema.tables)
            assert schema.metadata

        assert (
            profiled
        ), "Azure-native stores were discovered, but no profiler produced a schema"

        by_type = {store.store_type: schema for store, schema in profiled}
        if blob_schema := by_type.get("azure_blob"):
            assert blob_schema.database_name == "daita-events"
            assert blob_schema.table_count == 1
            assert blob_schema.tables[0].row_count == 1
            assert blob_schema.metadata.get("account") == "daitacat11ccd"
            assert blob_schema.metadata.get("prefixes", {}).get("events") == 1
            assert (
                blob_schema.metadata.get("content_types", {}).get("application/json")
                == 1
            )
            assert blob_schema.metadata.get("total_size_bytes") == 44

        if cosmos_schema := by_type.get("cosmosdb"):
            assert cosmos_schema.database_name == "daita-cosmos-11ccd"
            assert cosmos_schema.table_count == 1
            table = cosmos_schema.tables[0]
            assert table.name == "daita_catalog.orders"
            assert table.metadata.get("database") == "daita_catalog"
            assert table.metadata.get("container") == "orders"
            assert table.metadata.get("partition_key_paths") == ["/customer_id"]
            assert table.metadata.get("partition_key_kind") == "Hash"
            assert table.metadata.get("indexing_mode") == "consistent"
            columns = {column.name: column for column in table.columns}
            assert {"id", "_partition_key", "_ts", "_etag"} <= columns.keys()
            assert columns["id"].is_primary_key is True
            assert columns["_partition_key"].comment == "/customer_id"

        if eventhub_schema := by_type.get("eventhub"):
            assert eventhub_schema.database_name == "daita-events"
            assert eventhub_schema.metadata.get("namespace") == "daita-eh-11ccd"
            assert eventhub_schema.metadata.get("partition_count") == 2
            assert eventhub_schema.metadata.get("message_retention_days") == 1
            assert eventhub_schema.metadata.get("status") == "Active"
            event_columns = {
                column.name for column in eventhub_schema.tables[0].columns
            }
            assert "consumer_group:$Default" in event_columns

        if queue_schema := by_type.get("servicebus_queue"):
            assert queue_schema.database_name == "daita-orders"
            assert queue_schema.tables[0].row_count == 0
            assert queue_schema.metadata.get("namespace") == "daita-sb-11ccd"
            assert queue_schema.metadata.get("max_size_mb") == 1024
            assert (
                queue_schema.metadata.get("dead_lettering_on_message_expiration")
                is False
            )
            assert queue_schema.metadata.get("requires_duplicate_detection") is False

        print(
            "\n[AZURE PROFILE] "
            + ", ".join(
                f"{store.store_type}:{schema.database_name}/{schema.table_count}"
                for store, schema in profiled
            )
        )


@pytest.mark.integration
class TestAzureGraphEmissionLive:
    """Azure discover -> profile -> graph emission against live fixtures."""

    async def test_profiled_azure_schemas_emit_graph(
        self, profiled_azure_catalog_with_graph
    ):
        from daita.core.graph.models import EdgeType, NodeType

        plugin, backend = profiled_azure_catalog_with_graph
        profiled_stores = [
            store
            for store in plugin.get_stores()
            if plugin.get_schema(store.id) is not None
            and store.store_type in _AZURE_NATIVE_PROFILERS
        ]
        if not profiled_stores:
            pytest.skip("No Azure-native profiled stores available for graph emission")

        tables = await backend.find_nodes(NodeType.TABLE)
        columns = await backend.find_nodes(NodeType.COLUMN)
        buckets = await backend.find_nodes(NodeType.BUCKET)
        services = await backend.find_nodes(NodeType.SERVICE)
        has_column_edges = await backend.get_edges(edge_types=[EdgeType.HAS_COLUMN])

        graph_node_count = len(tables) + len(columns) + len(buckets) + len(services)
        assert graph_node_count > 0, "Profiled Azure schemas emitted no graph nodes"

        # Single-node Azure resources should become bucket/service nodes.
        if any(s.store_type == "azure_blob" for s in profiled_stores):
            assert buckets, "Azure Blob schema did not emit a BUCKET node"
            blob_nodes = [
                node
                for node in buckets
                if node.properties.get("database_type") == "azure_blob"
            ]
            assert blob_nodes, "No BUCKET node was tagged as azure_blob"
            blob_node = blob_nodes[0]
            metadata = blob_node.properties.get("metadata", {})
            assert metadata.get("account") == "daitacat11ccd"
            assert metadata.get("prefixes", {}).get("events") == 1
            assert metadata.get("content_types", {}).get("application/json") == 1

        if any(
            s.store_type in {"eventhub", "servicebus_queue"} for s in profiled_stores
        ):
            service_nodes = [
                node
                for node in services
                if node.properties.get("database_type")
                in {"eventhub", "servicebus_queue", "servicebus_topic"}
            ]
            assert service_nodes, "Azure messaging schemas did not emit SERVICE nodes"
            by_service_type = {
                node.properties.get("database_type"): node for node in service_nodes
            }
            if eventhub_node := by_service_type.get("eventhub"):
                metadata = eventhub_node.properties.get("metadata", {})
                assert metadata.get("namespace") == "daita-eh-11ccd"
                assert metadata.get("partition_count") == 2
            if queue_node := by_service_type.get("servicebus_queue"):
                metadata = queue_node.properties.get("metadata", {})
                assert metadata.get("namespace") == "daita-sb-11ccd"
                assert metadata.get("max_size_mb") == 1024

        # Fan-out Azure resources should emit table/column nodes and edges.
        cosmos_stores = [s for s in profiled_stores if s.store_type == "cosmosdb"]
        if cosmos_stores:
            cosmos_tables = [
                node
                for node in tables
                if node.properties.get("database_type") == "cosmosdb"
            ]
            assert cosmos_tables, "Cosmos DB schema did not emit TABLE nodes"
            assert columns, "Cosmos DB schema did not emit COLUMN nodes"
            assert has_column_edges, "Cosmos DB graph has no HAS_COLUMN edges"

            partition_columns = [
                node
                for node in columns
                if node.name == "_partition_key"
                and node.properties.get("store", "").startswith("cosmosdb:")
            ]
            assert partition_columns, "Cosmos DB partition key column missing"
            cosmos_table = cosmos_tables[0]
            assert cosmos_table.name == "daita_catalog.orders"
            assert cosmos_table.properties.get("partition_key_paths") == [
                "/customer_id"
            ]
            assert cosmos_table.properties.get("indexing_mode") == "consistent"
            assert partition_columns[0].properties.get("comment") == "/customer_id"

        print(
            "\n[AZURE GRAPH] "
            f"tables={len(tables)} columns={len(columns)} "
            f"buckets={len(buckets)} services={len(services)} "
            f"has_column_edges={len(has_column_edges)}"
        )


@pytest.mark.requires_llm
@pytest.mark.integration
class TestAgentAzureLive:
    """End-to-end: Agent + OpenAI + CatalogPlugin with live Azure."""

    async def test_agent_uses_discover_infrastructure(self, plugin_with_azure):
        """The agent must call discover_infrastructure and report the real count."""
        plugin, _ = plugin_with_azure

        agent = build_live_agent(name="AzureCatalogAgent", tools=[plugin])
        async with timed("agent.run azure enumerate"):
            result = await agent.run(
                "Use the discover_infrastructure tool to scan the configured "
                "Azure subscription. Then report: the total number of data "
                "stores found, and the unique store types (for example "
                "azure_blob, cosmosdb, eventhub, servicebus_queue). Be concise.",
                detailed=True,
            )

        assert_tool_called(result, "discover_infrastructure")

        text = (result.get("result") or "").lower()
        expected_count = len(plugin.get_stores())
        word_forms = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        forms = {str(expected_count), word_forms.get(expected_count, "")}
        assert any(f and f in text for f in forms), (
            f"Agent reported wrong count. Expected {expected_count}; "
            f"answer: {text[:300]!r}"
        )

    async def test_agent_finds_store_by_type(self, plugin_with_azure):
        """After discovery, the agent should use find_store for an Azure type."""
        plugin, _ = plugin_with_azure
        await plugin.discover_all()
        stores = plugin.get_stores()
        if not stores:
            pytest.skip("No Azure stores discovered — nothing for the agent to find")

        target_type = stores[0].store_type
        expected = len([s for s in stores if s.store_type == target_type])

        agent = build_live_agent(name="AzureCatalogAgent", tools=[plugin])
        async with timed("agent.run azure find_store"):
            result = await agent.run(
                f"Use the find_store tool with store_type='{target_type}' and "
                "report how many matches the catalog contains. Answer with "
                "just the number and the store type.",
                detailed=True,
            )

        assert_tool_called(result, "find_store")

        answer = result.get("result") or ""
        assert str(expected) in answer
        assert target_type in answer
