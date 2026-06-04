"""Unit tests for CatalogPlugin discovery helpers."""

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
