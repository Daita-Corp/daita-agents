"""Unit tests for Azure catalog discovery contracts."""

import pytest

from daita.plugins.catalog.azure import AzureDiscoverer
from daita.plugins.catalog.base_discoverer import DiscoveredStore
from daita.plugins.catalog.normalizer import (
    normalize_azure_apim,
    normalize_azure_blob,
    normalize_azure_cosmosdb,
    normalize_azure_eventhub,
    normalize_azure_servicebus_queue,
    normalize_azure_servicebus_topic,
    normalize_discovery,
)
from daita.plugins.catalog.profiler import (
    AzureAPIMProfiler,
    AzureBlobProfiler,
    AzureCosmosDBProfiler,
    AzureEventHubProfiler,
    AzureServiceBusQueueProfiler,
    AzureServiceBusTopicProfiler,
)


def test_azure_discoverer_env_fallbacks(monkeypatch):
    monkeypatch.setenv("AZURE_SUBSCRIPTIONS", "sub-a, sub-b")
    monkeypatch.setenv("AZURE_LOCATIONS", "eastus, westus2")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-1")

    discoverer = AzureDiscoverer()

    assert discoverer._subscriptions == ["sub-a", "sub-b"]
    assert discoverer._locations == ["eastus", "westus2"]
    assert discoverer._tenant_id == "tenant-1"


def test_azure_build_store_and_fingerprint_are_stable():
    discoverer = AzureDiscoverer(subscriptions=["sub-1"], tenant_id="tenant-1")
    resource_id = (
        "/subscriptions/sub-1/resourceGroups/rg/providers/Microsoft.Storage/"
        "storageAccounts/acct/blobServices/default/containers/events"
    )

    store = discoverer._build_store(
        store_type="azure_blob",
        display_name="acct/events",
        connection_hint={"account": "acct", "container": "events"},
        source="azure_blob",
        region="eastus",
        resource_id=resource_id,
        subscription_id="sub-1",
        metadata={},
    )

    assert store.id == discoverer.fingerprint(store)
    assert len(store.id) == 16
    assert store.connection_hint["subscription_id"] == "sub-1"
    assert store.connection_hint["tenant_id"] == "tenant-1"
    assert store.metadata["resource_group"] == "rg"


def test_azure_service_filter_ignores_unknown_services():
    discoverer = AzureDiscoverer(subscriptions=["sub-1"], services=["blob", "nope"])

    assert discoverer._SERVICE_METHODS["blob"] == "_enumerate_blob"
    assert "nope" not in discoverer._SERVICE_METHODS


def test_azure_normalizers_dispatch():
    blob = normalize_discovery(
        {
            "database_type": "azure_blob",
            "account": "acct",
            "container": "events",
            "object_count": 2,
        }
    )
    assert blob["database_type"] == "azure_blob"
    assert blob["database_name"] == "events"
    assert blob["tables"][0]["row_count"] == 2

    cosmos = normalize_discovery(
        {
            "database_type": "cosmosdb",
            "account": "acct",
            "databases": [
                {
                    "name": "db",
                    "containers": [
                        {
                            "name": "orders",
                            "partition_key_paths": ["/customer_id"],
                        }
                    ],
                }
            ],
        }
    )
    assert cosmos["table_count"] == 1
    assert cosmos["tables"][0]["name"] == "db.orders"


def test_azure_service_normalizers_shapes():
    eventhub = normalize_azure_eventhub(
        {
            "eventhub": "orders",
            "namespace": "ns",
            "consumer_groups": ["analytics"],
        }
    )
    assert eventhub["tables"][0]["columns"][-1]["name"] == "consumer_group:analytics"

    queue = normalize_azure_servicebus_queue(
        {"queue": "jobs", "namespace": "ns", "message_count": 4}
    )
    assert queue["tables"][0]["row_count"] == 4

    topic = normalize_azure_servicebus_topic(
        {"topic": "events", "namespace": "ns", "subscriptions": ["warehouse"]}
    )
    assert topic["tables"][0]["columns"][-1]["name"] == "sub:warehouse"

    apim = normalize_azure_apim(
        {
            "api_id": "orders-api",
            "service": "gateway",
            "operations": [{"display_name": "List orders", "method": "GET"}],
        }
    )
    assert apim["database_type"] == "azure_apim"
    assert apim["tables"][0]["columns"][0]["type"] == "GET"

    blob = normalize_azure_blob({"account": "acct", "container": "events"})
    assert blob["tables"][0]["columns"][0]["name"] == "name"

    cosmos = normalize_azure_cosmosdb({"account": "acct", "databases": []})
    assert cosmos["table_count"] == 0


@pytest.mark.parametrize(
    ("profiler", "store_type"),
    [
        (AzureBlobProfiler(), "azure_blob"),
        (AzureCosmosDBProfiler(), "cosmosdb"),
        (AzureEventHubProfiler(), "eventhub"),
        (AzureServiceBusQueueProfiler(), "servicebus_queue"),
        (AzureServiceBusTopicProfiler(), "servicebus_topic"),
        (AzureAPIMProfiler(), "azure_apim"),
    ],
)
def test_azure_profilers_support_expected_store_types(profiler, store_type):
    assert profiler.supports(store_type)
    assert not profiler.supports("postgresql")


def test_azure_discoverer_import_does_not_require_sdk():
    store = DiscoveredStore(
        id="s1",
        store_type="azure_blob",
        display_name="acct/events",
        connection_hint={},
        source="azure_blob",
        metadata={"resource_id": "rid", "subscription_id": "sub"},
    )
    assert AzureDiscoverer(subscriptions=["sub"]).fingerprint(store)
