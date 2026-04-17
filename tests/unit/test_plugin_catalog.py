"""
Unit tests for CatalogPlugin.

Exercises ``table_filter`` and ``max_tables`` options on the unified
``discover_schema`` tool, which dispatches by ``store_type`` to the
underlying ``discover_postgres`` / ``discover_mysql`` wrappers.
"""

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
    return {
        "schema": schema_dict,
        "persisted": False,
        "persist_skipped": "catalog backend not configured",
    }


async def _fake_discover_postgres(connection_string, schema="public", **kwargs):
    """Returns a fixed schema dict in the real wrapper shape."""
    return _wrap_result(
        {
            "database_type": "postgresql",
            "schema": schema,
            "tables": _make_tables([f"table_{i}" for i in range(60)]),
            "columns": [],
        }
    )


async def _fake_discover_mysql(connection_string, schema=None, **kwargs):
    return _wrap_result(
        {
            "database_type": "mysql",
            "schema": schema or "testdb",
            "tables": _make_tables([f"tbl_{i}" for i in range(60)]),
            "columns": [],
        }
    )


def _discover_schema_tool(plugin):
    return next(t for t in plugin.get_tools() if t.name == "discover_schema")


# ---------------------------------------------------------------------------
# discover_schema — tool surface
# ---------------------------------------------------------------------------


def test_discover_schema_tool_exists():
    plugin = make_plugin()
    tool = _discover_schema_tool(plugin)
    props = tool.parameters["properties"]
    assert "store_type" in props
    assert "connection_string" in props
    assert "options" in props
    # options is free-form; document that table_filter / max_tables are honored
    assert "table_filter" in props["options"]["description"]
    assert "max_tables" in props["options"]["description"]


# ---------------------------------------------------------------------------
# discover_schema (postgresql) — max_tables
# ---------------------------------------------------------------------------


async def test_postgres_max_tables_default_is_50():
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    result = await _discover_schema_tool(plugin).handler(
        {
            "store_type": "postgresql",
            "connection_string": "postgresql://localhost/db",
        }
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 50
    assert schema["total_tables"] == 60
    assert schema["truncated"] is True


async def test_postgres_max_tables_custom():
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    result = await _discover_schema_tool(plugin).handler(
        {
            "store_type": "postgresql",
            "connection_string": "postgresql://localhost/db",
            "options": {"max_tables": 10},
        }
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 10
    assert schema["total_tables"] == 60


async def test_postgres_max_tables_larger_than_total_not_truncated():
    plugin = make_plugin()
    plugin.discover_postgres = _fake_discover_postgres

    result = await _discover_schema_tool(plugin).handler(
        {
            "store_type": "postgresql",
            "connection_string": "postgresql://localhost/db",
            "options": {"max_tables": 100},
        }
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 60
    assert schema["truncated"] is False


# ---------------------------------------------------------------------------
# discover_schema (postgresql) — table_filter
# ---------------------------------------------------------------------------


async def test_postgres_table_filter_glob():
    plugin = make_plugin()

    async def fake_discover(connection_string, schema="public", **kwargs):
        return _wrap_result(
            {
                "tables": _make_tables(["orders", "order_items", "products", "users"]),
                "columns": [],
            }
        )

    plugin.discover_postgres = fake_discover

    result = await _discover_schema_tool(plugin).handler(
        {
            "store_type": "postgresql",
            "connection_string": "postgresql://localhost/db",
            "options": {"table_filter": "order*"},
        }
    )

    names = [t["name"] for t in result["schema"]["tables"]]
    assert "orders" in names
    assert "order_items" in names
    assert "products" not in names
    assert "users" not in names


async def test_postgres_table_filter_no_match_returns_empty():
    plugin = make_plugin()

    async def fake_discover(connection_string, schema="public", **kwargs):
        return _wrap_result(
            {"tables": _make_tables(["users", "products"]), "columns": []}
        )

    plugin.discover_postgres = fake_discover

    result = await _discover_schema_tool(plugin).handler(
        {
            "store_type": "postgresql",
            "connection_string": "postgresql://localhost/db",
            "options": {"table_filter": "xyz_*"},
        }
    )

    assert result["schema"]["tables"] == []
    assert result["schema"]["total_tables"] == 0


# ---------------------------------------------------------------------------
# discover_schema (mysql) — max_tables and table_filter
# ---------------------------------------------------------------------------


async def test_mysql_max_tables_default_is_50():
    plugin = make_plugin()
    plugin.discover_mysql = _fake_discover_mysql

    result = await _discover_schema_tool(plugin).handler(
        {
            "store_type": "mysql",
            "connection_string": "mysql://localhost/db",
        }
    )

    schema = result["schema"]
    assert len(schema["tables"]) == 50
    assert schema["total_tables"] == 60
    assert schema["truncated"] is True


async def test_mysql_table_filter_applied():
    plugin = make_plugin()

    async def fake_discover(connection_string, schema=None, **kwargs):
        return _wrap_result(
            {
                "tables": _make_tables(["sales_2023", "sales_2024", "customers"]),
                "columns": [],
            }
        )

    plugin.discover_mysql = fake_discover

    result = await _discover_schema_tool(plugin).handler(
        {
            "store_type": "mysql",
            "connection_string": "mysql://localhost/db",
            "options": {"table_filter": "sales_*"},
        }
    )

    names = [t["name"] for t in result["schema"]["tables"]]
    assert "sales_2023" in names
    assert "sales_2024" in names
    assert "customers" not in names
