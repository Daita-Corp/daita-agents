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
    return [{"table_name": n, "row_count": 0} for n in names]


async def _fake_discover_postgres(
    connection_string, schema="public", persist=False, ssl_mode="verify-full"
):
    """Returns a fixed schema dict for testing."""
    return {
        "database_type": "postgresql",
        "schema": schema,
        "tables": _make_tables([f"table_{i}" for i in range(60)]),
        "columns": [],
    }


async def _fake_discover_mysql(connection_string, schema=None, persist=False):
    return {
        "database_type": "mysql",
        "schema": schema or "testdb",
        "tables": _make_tables([f"tbl_{i}" for i in range(60)]),
        "columns": [],
    }


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

    result = await plugin._tool_discover_postgres(
        {"connection_string": "postgresql://localhost/db"}
    )

    assert len(result["tables"]) == 50
    assert result["total_tables"] == 60
    assert result["truncated"] is True


async def test_postgres_max_tables_custom(monkeypatch):
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    result = await plugin._tool_discover_postgres(
        {
            "connection_string": "postgresql://localhost/db",
            "max_tables": 10,
        }
    )

    assert len(result["tables"]) == 10
    assert result["total_tables"] == 60


async def test_postgres_max_tables_larger_than_total_not_truncated(monkeypatch):
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    result = await plugin._tool_discover_postgres(
        {
            "connection_string": "postgresql://localhost/db",
            "max_tables": 100,
        }
    )

    assert len(result["tables"]) == 60
    assert result["truncated"] is False


# ---------------------------------------------------------------------------
# discover_postgres — table_filter
# ---------------------------------------------------------------------------


async def test_postgres_table_filter_glob(monkeypatch):
    plugin = make_plugin()

    async def fake_discover(
        connection_string, schema="public", persist=False, ssl_mode="verify-full"
    ):
        return {
            "tables": _make_tables(["orders", "order_items", "products", "users"]),
            "columns": [],
        }

    plugin.discover_postgres = fake_discover

    result = await plugin._tool_discover_postgres(
        {
            "connection_string": "postgresql://localhost/db",
            "table_filter": "order*",
        }
    )

    names = [t["table_name"] for t in result["tables"]]
    assert "orders" in names
    assert "order_items" in names
    assert "products" not in names
    assert "users" not in names


async def test_postgres_table_filter_no_match_returns_empty(monkeypatch):
    plugin = make_plugin()

    async def fake_discover(
        connection_string, schema="public", persist=False, ssl_mode="verify-full"
    ):
        return {"tables": _make_tables(["users", "products"]), "columns": []}

    plugin.discover_postgres = fake_discover

    result = await plugin._tool_discover_postgres(
        {
            "connection_string": "postgresql://localhost/db",
            "table_filter": "xyz_*",
        }
    )

    assert result["tables"] == []
    assert result["total_tables"] == 0


# ---------------------------------------------------------------------------
# discover_mysql — max_tables and table_filter
# ---------------------------------------------------------------------------


async def test_mysql_max_tables_default_is_50(monkeypatch):
    plugin = make_plugin()
    plugin.discover_mysql = _fake_discover_mysql

    result = await plugin._tool_discover_mysql(
        {"connection_string": "mysql://localhost/db"}
    )

    assert len(result["tables"]) == 50
    assert result["total_tables"] == 60
    assert result["truncated"] is True


async def test_mysql_table_filter_applied(monkeypatch):
    plugin = make_plugin()

    async def fake_discover(connection_string, schema=None, persist=False):
        return {
            "tables": _make_tables(["sales_2023", "sales_2024", "customers"]),
            "columns": [],
        }

    plugin.discover_mysql = fake_discover

    result = await plugin._tool_discover_mysql(
        {
            "connection_string": "mysql://localhost/db",
            "table_filter": "sales_*",
        }
    )

    names = [t["table_name"] for t in result["tables"]]
    assert "sales_2023" in names
    assert "sales_2024" in names
    assert "customers" not in names
