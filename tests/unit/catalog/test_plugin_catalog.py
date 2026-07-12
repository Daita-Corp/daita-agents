"""Unit tests for CatalogPlugin discovery helpers."""

from types import SimpleNamespace

import pytest

from daita.plugins.catalog.discovery._opensearch import discover_opensearch
from daita.plugins.catalog.discovery._postgres import discover_postgres


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
