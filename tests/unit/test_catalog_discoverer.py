"""
Tests for BaseDiscoverer contract, discovery orchestration, and new tools.
"""

import pytest
from typing import AsyncIterator, List

from daita.plugins.catalog.base_discoverer import (
    BaseDiscoverer,
    DiscoveredStore,
    DiscoveryError,
    DiscoveryResult,
)
from daita.plugins.catalog.base_profiler import (
    BaseProfiler,
    NormalizedColumn,
    NormalizedForeignKey,
    NormalizedSchema,
    NormalizedTable,
)
from daita.plugins.catalog import CatalogPlugin

# ---------------------------------------------------------------------------
# Fake implementations for testing
# ---------------------------------------------------------------------------


class FakeDiscoverer(BaseDiscoverer):
    """A discoverer that yields pre-configured stores."""

    name = "fake"

    def __init__(self, stores: List[DiscoveredStore] = None, should_fail: bool = False):
        self._stores = stores or []
        self._should_fail = should_fail
        self.authenticated = False
        self.closed = False

    async def authenticate(self) -> None:
        if self._should_fail:
            raise RuntimeError("Auth failed")
        self.authenticated = True

    async def enumerate(self) -> AsyncIterator[DiscoveredStore]:
        for store in self._stores:
            yield store

    async def close(self) -> None:
        self.closed = True


class FakeProfiler(BaseProfiler):
    """A profiler that returns a fixed schema."""

    def __init__(self, supported_types: List[str] = None):
        self._supported_types = supported_types or ["postgresql"]

    def supports(self, store_type: str) -> bool:
        return store_type in self._supported_types

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        return NormalizedSchema(
            database_type=store.store_type,
            database_name="test_db",
            tables=[
                NormalizedTable(
                    name="users",
                    row_count=100,
                    columns=[
                        NormalizedColumn(
                            name="id",
                            type="integer",
                            nullable=False,
                            is_primary_key=True,
                        ),
                        NormalizedColumn(
                            name="email",
                            type="varchar",
                            nullable=True,
                            is_primary_key=False,
                        ),
                    ],
                )
            ],
            foreign_keys=[],
            table_count=1,
            store_id=store.id,
        )


def _make_store(
    id: str = "abc123",
    store_type: str = "postgresql",
    display_name: str = "test-db",
    source: str = "fake",
    **kwargs,
) -> DiscoveredStore:
    return DiscoveredStore(
        id=id,
        store_type=store_type,
        display_name=display_name,
        connection_hint=kwargs.get(
            "connection_hint", {"host": "localhost", "port": 5432}
        ),
        source=source,
        region=kwargs.get("region"),
        environment=kwargs.get("environment"),
        confidence=kwargs.get("confidence", 0.8),
        tags=kwargs.get("tags", []),
        metadata=kwargs.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# BaseDiscoverer contract tests
# ---------------------------------------------------------------------------


async def test_fake_discoverer_enumerate_yields_stores():
    stores = [_make_store(id="s1"), _make_store(id="s2")]
    d = FakeDiscoverer(stores=stores)
    await d.authenticate()
    assert d.authenticated

    result = [s async for s in d.enumerate()]
    assert len(result) == 2
    assert result[0].id == "s1"
    assert result[1].id == "s2"


async def test_fake_discoverer_close():
    d = FakeDiscoverer()
    await d.close()
    assert d.closed


async def test_base_discoverer_default_fingerprint():
    d = FakeDiscoverer()
    store = _make_store(connection_hint={"host": "db.example.com", "port": 5432})
    fp = d.fingerprint(store)
    assert isinstance(fp, str)
    assert len(fp) == 16

    # Same inputs produce same fingerprint
    fp2 = d.fingerprint(store)
    assert fp == fp2


async def test_base_discoverer_test_access_default():
    d = FakeDiscoverer()
    assert await d.test_access() is True


# ---------------------------------------------------------------------------
# BaseProfiler contract tests
# ---------------------------------------------------------------------------


async def test_fake_profiler_profile():
    profiler = FakeProfiler(supported_types=["postgresql"])
    assert profiler.supports("postgresql")
    assert not profiler.supports("mysql")

    store = _make_store()
    schema = await profiler.profile(store)
    assert isinstance(schema, NormalizedSchema)
    assert schema.database_type == "postgresql"
    assert schema.table_count == 1
    assert schema.store_id == store.id


async def test_normalized_schema_to_dict():
    schema = NormalizedSchema(
        database_type="postgresql",
        database_name="test",
        tables=[
            NormalizedTable(
                name="users",
                row_count=100,
                columns=[
                    NormalizedColumn(
                        name="id", type="integer", nullable=False, is_primary_key=True
                    ),
                    NormalizedColumn(
                        name="name",
                        type="varchar",
                        nullable=True,
                        is_primary_key=False,
                        comment="User name",
                    ),
                ],
            )
        ],
        foreign_keys=[
            NormalizedForeignKey(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        ],
        table_count=1,
    )
    d = schema.to_dict()

    assert d["database_type"] == "postgresql"
    assert d["database_name"] == "test"
    assert len(d["tables"]) == 1
    assert d["tables"][0]["name"] == "users"
    assert d["tables"][0]["row_count"] == 100
    assert len(d["tables"][0]["columns"]) == 2
    assert d["tables"][0]["columns"][1]["column_comment"] == "User name"
    assert d["tables"][0]["columns"][0].get("column_comment") is None
    assert len(d["foreign_keys"]) == 1
    assert d["foreign_keys"][0]["source_table"] == "orders"
    assert d["table_count"] == 1


# ---------------------------------------------------------------------------
# CatalogPlugin orchestration tests
# ---------------------------------------------------------------------------


async def test_discover_all_with_fake_discoverers():
    plugin = CatalogPlugin()
    stores = [_make_store(id="s1"), _make_store(id="s2")]
    plugin.add_discoverer(FakeDiscoverer(stores=stores))

    result = await plugin.discover_all()
    assert result.store_count == 2
    assert result.error_count == 0
    assert not result.has_errors


async def test_discover_all_merges_from_multiple_discoverers():
    plugin = CatalogPlugin()
    plugin.add_discoverer(FakeDiscoverer(stores=[_make_store(id="s1", source="aws")]))
    plugin.add_discoverer(
        FakeDiscoverer(stores=[_make_store(id="s2", source="github")])
    )

    result = await plugin.discover_all()
    assert result.store_count == 2


async def test_discover_all_partial_failure():
    plugin = CatalogPlugin()
    plugin.add_discoverer(FakeDiscoverer(stores=[_make_store(id="s1")]))
    plugin.add_discoverer(FakeDiscoverer(should_fail=True))

    result = await plugin.discover_all()
    # Stores from successful discoverer are still returned
    assert result.store_count == 1
    # Error from failed discoverer is captured
    assert result.error_count == 1
    assert result.has_errors


async def test_discover_all_dedup():
    plugin = CatalogPlugin()
    # Two discoverers finding the same store (same ID)
    store = _make_store(id="same_id", source="aws", confidence=0.9)
    store2 = _make_store(id="same_id", source="github", confidence=0.7)
    plugin.add_discoverer(FakeDiscoverer(stores=[store]))
    plugin.add_discoverer(FakeDiscoverer(stores=[store2]))

    result = await plugin.discover_all()
    assert result.store_count == 1
    # Higher confidence source should win
    assert result.stores[0].confidence == 0.9


async def test_discover_all_empty():
    plugin = CatalogPlugin()
    result = await plugin.discover_all()
    assert result.store_count == 0
    assert result.error_count == 0


async def test_public_accessor_api():
    plugin = CatalogPlugin()
    stores = [
        _make_store(id="s1", store_type="postgresql", environment="production"),
        _make_store(id="s2", store_type="mysql", environment="staging"),
        _make_store(id="s3", store_type="postgresql", environment="staging"),
    ]
    plugin.add_discoverer(FakeDiscoverer(stores=stores))
    await plugin.discover_all()

    # get_stores no filter
    assert len(plugin.get_stores()) == 3

    # Filter by type
    pg_stores = plugin.get_stores(store_type="postgresql")
    assert len(pg_stores) == 2

    # Filter by environment
    staging = plugin.get_stores(environment="staging")
    assert len(staging) == 2

    # Filter by both
    pg_staging = plugin.get_stores(store_type="postgresql", environment="staging")
    assert len(pg_staging) == 1

    # get_store by ID
    assert plugin.get_store("s1") is not None
    assert plugin.get_store("nonexistent") is None


async def test_get_tools_returns_expected_set():
    plugin = CatalogPlugin()
    tools = plugin.get_tools()
    names = {t.name for t in tools}
    assert names == {
        "discover_infrastructure",
        "discover_schema",
        "profile_store",
        "get_table_schema",
        "find_store",
        "compare_schemas",
        "export_diagram",
    }


async def test_lifecycle_on_before_run_empty():
    plugin = CatalogPlugin()
    result = await plugin.on_before_run("test prompt")
    assert result is None


async def test_lifecycle_on_before_run_with_stores():
    plugin = CatalogPlugin()
    plugin.add_discoverer(FakeDiscoverer(stores=[_make_store(id="s1")]))
    await plugin.discover_all()

    result = await plugin.on_before_run("test prompt")
    assert result is not None
    assert "1 data stores known" in result


async def test_lifecycle_on_agent_stop_closes_discoverers():
    d = FakeDiscoverer()
    plugin = CatalogPlugin()
    plugin.add_discoverer(d)
    await plugin.on_agent_stop()
    assert d.closed


async def test_find_store_tool_handler():
    from daita.plugins.catalog.tools import _handle_find_store

    plugin = CatalogPlugin()
    stores = [
        _make_store(
            id="s1",
            display_name="prod-orders",
            store_type="postgresql",
            environment="production",
            tags=["team:backend"],
        ),
        _make_store(
            id="s2",
            display_name="staging-users",
            store_type="mysql",
            environment="staging",
        ),
    ]
    plugin.add_discoverer(FakeDiscoverer(stores=stores))
    await plugin.discover_all()

    # Search by query
    result = await _handle_find_store(plugin, {"query": "orders"})
    assert result["total"] == 1
    assert result["stores"][0]["id"] == "s1"

    # Filter by type
    result = await _handle_find_store(plugin, {"store_type": "mysql"})
    assert result["total"] == 1

    # Filter by tag
    result = await _handle_find_store(plugin, {"tag": "team:backend"})
    assert result["total"] == 1

    # Pagination
    result = await _handle_find_store(plugin, {"offset": 0, "limit": 1})
    assert len(result["stores"]) == 1
    assert result["total"] == 2


async def test_discover_and_profile():
    plugin = CatalogPlugin()
    stores = [_make_store(id="s1", store_type="postgresql")]
    plugin.add_discoverer(FakeDiscoverer(stores=stores))
    plugin.add_profiler(FakeProfiler(supported_types=["postgresql"]))

    result = await plugin.discover_and_profile()
    assert result.store_count == 1

    schema = plugin.get_schema("s1")
    assert schema is not None
    assert schema.database_type == "postgresql"
    assert schema.store_id == "s1"
