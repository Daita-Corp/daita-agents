"""Unit tests for CatalogPlugin discovery helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from daita.plugins.catalog.aws import AWSDiscoverer
from daita.plugins.catalog.discovery._opensearch import discover_opensearch
from daita.plugins.catalog.discovery._postgres import discover_postgres
from daita.plugins.catalog.gcp import GCPDiscoverer


class _FakeAsyncpgConnection:
    def __init__(self):
        self.closed = False

    async def fetch(self, *args, **kwargs):
        return []

    async def close(self):
        self.closed = True


async def test_postgres_discovery_disables_asyncpg_statement_cache(monkeypatch):
    captured_kwargs = {}
    fake_connection = _FakeAsyncpgConnection()

    class FakeAsyncpg:
        async def connect(self, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_connection

    import sys

    monkeypatch.setitem(sys.modules, "asyncpg", FakeAsyncpg())

    await discover_postgres("postgresql://user:pass@db.example.com:6543/app")

    assert captured_kwargs["statement_cache_size"] == 0
    assert fake_connection.closed is True


async def test_opensearch_discovery_requires_aws_credentials(monkeypatch):
    class FakeSession:
        def get_credentials(self):
            return None

    fake_boto3 = SimpleNamespace(Session=lambda **_kwargs: FakeSession())
    fake_opensearch = SimpleNamespace(
        AWSV4SignerAuth=object,
        OpenSearch=object,
        RequestsHttpConnection=object,
    )

    import sys

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "opensearchpy", fake_opensearch)

    with pytest.raises(
        RuntimeError, match="AWS credentials are required for OpenSearch discovery"
    ):
        await discover_opensearch("search.example.com")


async def test_aws_s3_discovery_skips_bucket_without_name():
    discoverer = AWSDiscoverer(regions=["us-east-1"], services=["s3"])
    client = MagicMock()
    client.list_buckets.return_value = {
        "Buckets": [{}, {"Name": "reports", "CreationDate": "today"}]
    }
    client.get_bucket_location.return_value = {"LocationConstraint": None}
    session = MagicMock()
    session.client.return_value = client
    discoverer._session = session

    stores = [store async for store in discoverer._enumerate_s3("us-east-1")]

    assert len(stores) == 1
    assert stores[0].connection_hint["bucket"] == "reports"


async def test_gcp_cloudsql_discovery_handles_primary_mapping_without_ip(
    monkeypatch,
):
    from googleapiclient import discovery

    discoverer = GCPDiscoverer(projects=["project-1"], services=["cloudsql"])
    discoverer._credentials = object()
    service = MagicMock()
    service.instances.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "name": "orders",
                "databaseVersion": "POSTGRES_15",
                "region": "us-central1",
                "ipAddresses": [{"type": "PRIMARY"}],
            }
        ]
    }
    monkeypatch.setattr(discovery, "build", MagicMock(return_value=service))

    stores = [store async for store in discoverer._enumerate_cloudsql("project-1")]

    assert len(stores) == 1
    assert stores[0].connection_hint["host"] == ""
