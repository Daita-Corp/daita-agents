"""
Unit tests for CatalogPlugin.

Tests table_filter and max_tables parameters on discover_postgres
and discover_mysql tool handlers.
"""

import pytest
from daita.plugins.catalog import CatalogPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin():
    return CatalogPlugin()


def _make_tables(names):
    return [{"name": n, "row_count": 0} for n in names]


def _wrap_result(schema_dict):
    """Wrap a raw schema dict in the standard response shape."""
    return {"schema": schema_dict, "persisted": False, "persist_skipped": "catalog backend not configured"}


async def _fake_discover_postgres(
    connection_string, schema="public", persist=False, ssl_mode="verify-full"
):
    """Returns a fixed schema dict in the real wrapper shape."""
    return _wrap_result({
        "database_type": "postgresql",
        "schema": schema,
        "tables": _make_tables([f"table_{i}" for i in range(60)]),
        "columns": [],
    })


async def _fake_discover_mysql(connection_string, schema=None, persist=False):
    return _wrap_result({
        "database_type": "mysql",
        "schema": schema or "testdb",
        "tables": _make_tables([f"tbl_{i}" for i in range(60)]),
        "columns": [],
    })


# ---------------------------------------------------------------------------
# discover_postgres — tool params
# ---------------------------------------------------------------------------


def test_discover_postgres_tool_has_table_filter_param():
    plugin = make_plugin()
    tools = {t.name: t for t in plugin.get_tools()}
    props = tools["discover_postgres"].parameters["properties"]
    assert "table_filter" in props
    assert "max_tables" in props


def test_discover_mysql_tool_has_table_filter_param():
    plugin = make_plugin()
    tools = {t.name: t for t in plugin.get_tools()}
    props = tools["discover_mysql"].parameters["properties"]
    assert "table_filter" in props
    assert "max_tables" in props


# ---------------------------------------------------------------------------
# discover_postgres — max_tables
# ---------------------------------------------------------------------------


async def test_postgres_max_tables_default_is_50(monkeypatch):
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    tools = {t.name: t for t in plugin.get_tools()}
    result = await tools["discover_postgres"].handler(
        {"connection_string": "postgresql://localhost/db"}
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 50
    assert schema["total_tables"] == 60
    assert schema["truncated"] is True


async def test_postgres_max_tables_custom(monkeypatch):
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    tools = {t.name: t for t in plugin.get_tools()}
    result = await tools["discover_postgres"].handler(
        {
            "connection_string": "postgresql://localhost/db",
            "max_tables": 10,
        }
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 10
    assert schema["total_tables"] == 60


async def test_postgres_max_tables_larger_than_total_not_truncated(monkeypatch):
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    tools = {t.name: t for t in plugin.get_tools()}
    result = await tools["discover_postgres"].handler(
        {
            "connection_string": "postgresql://localhost/db",
            "max_tables": 100,
        }
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 60
    assert schema["truncated"] is False


# ---------------------------------------------------------------------------
# discover_postgres — table_filter
# ---------------------------------------------------------------------------


async def test_postgres_table_filter_glob(monkeypatch):
    plugin = make_plugin()

    async def fake_discover(
        connection_string, schema="public", persist=False, ssl_mode="verify-full"
    ):
        return _wrap_result({
            "tables": _make_tables(["orders", "order_items", "products", "users"]),
            "columns": [],
        })

    plugin.discover_postgres = fake_discover

    tools = {t.name: t for t in plugin.get_tools()}
    result = await tools["discover_postgres"].handler(
        {
            "connection_string": "postgresql://localhost/db",
            "table_filter": "order*",
        }
    )

    names = [t["name"] for t in result["schema"]["tables"]]
    assert "orders" in names
    assert "order_items" in names
    assert "products" not in names
    assert "users" not in names


async def test_postgres_table_filter_no_match_returns_empty(monkeypatch):
    plugin = make_plugin()

    async def fake_discover(
        connection_string, schema="public", persist=False, ssl_mode="verify-full"
    ):
        return _wrap_result({"tables": _make_tables(["users", "products"]), "columns": []})

    plugin.discover_postgres = fake_discover

    tools = {t.name: t for t in plugin.get_tools()}
    result = await tools["discover_postgres"].handler(
        {
            "connection_string": "postgresql://localhost/db",
            "table_filter": "xyz_*",
        }
    )

    assert result["schema"]["tables"] == []
    assert result["schema"]["total_tables"] == 0


# ---------------------------------------------------------------------------
# discover_mysql — max_tables and table_filter
# ---------------------------------------------------------------------------


async def test_mysql_max_tables_default_is_50(monkeypatch):
    plugin = make_plugin()
    plugin.discover_mysql = _fake_discover_mysql

    tools = {t.name: t for t in plugin.get_tools()}
    result = await tools["discover_mysql"].handler(
        {"connection_string": "mysql://localhost/db"}
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 50
    assert schema["total_tables"] == 60
    assert schema["truncated"] is True


async def test_mysql_table_filter_applied(monkeypatch):
    plugin = make_plugin()

    async def fake_discover(connection_string, schema=None, persist=False):
        return _wrap_result({
            "tables": _make_tables(["sales_2023", "sales_2024", "customers"]),
            "columns": [],
        })

    plugin.discover_mysql = fake_discover

    tools = {t.name: t for t in plugin.get_tools()}
    result = await tools["discover_mysql"].handler(
        {
            "connection_string": "mysql://localhost/db",
            "table_filter": "sales_*",
        }
    )

    names = [t["name"] for t in result["schema"]["tables"]]
    assert "sales_2023" in names
    assert "sales_2024" in names
    assert "customers" not in names
