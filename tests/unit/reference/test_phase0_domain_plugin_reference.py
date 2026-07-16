"""
Phase 0 current-behavior reference tests for domain plugin APIs.

These tests capture direct Python surfaces and the remaining explicit
compatibility adapters after capability-based runtimes took over.
"""

from unittest.mock import MagicMock

import pytest

from daita.core.exceptions import ValidationError
from daita.agents.agent import Agent
from daita.llm.mock import MockLLMProvider
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.data_quality import DataQualityPlugin
from daita.plugins.lineage import LineagePlugin
from daita.plugins.sqlite import SQLitePlugin
from tests.unit.plugins.projection_helpers import projected_tools, projected_tool_names


def _reference_schema():
    return {
        "database_type": "sqlite",
        "database_name": "shop",
        "tables": [
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "email", "data_type": "TEXT"},
                ],
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "customer_id", "data_type": "INTEGER"},
                    {"name": "total", "data_type": "REAL"},
                ],
            },
        ],
        "foreign_keys": [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
        ],
    }


async def test_current_behavior_sqlite_direct_python_api_still_works(tmp_path):
    db_path = tmp_path / "reference.sqlite"
    plugin = SQLitePlugin(path=str(db_path))

    await plugin.execute_script("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO customers (name) VALUES ('Ada'), ('Linus');
        """)

    try:
        assert await plugin.tables() == ["customers"]
        assert await plugin.count_rows("customers") == 2
        assert await plugin.query("SELECT name FROM customers ORDER BY id") == [
            {"name": "Ada"},
            {"name": "Linus"},
        ]
        columns = await plugin.describe("customers")
        assert [column["column_name"] for column in columns] == ["id", "name"]
    finally:
        await plugin.disconnect()


async def test_current_behavior_sqlite_read_only_tool_rejects_writes(tmp_path):
    db_path = tmp_path / "reference.sqlite"
    seed = SQLitePlugin(path=str(db_path))
    await seed.execute_script("CREATE TABLE customers (id INTEGER PRIMARY KEY)")
    await seed.disconnect()

    plugin = SQLitePlugin(path=str(db_path), read_only=True)
    try:
        tool_names = projected_tool_names(plugin)
        assert "sqlite_execute" not in tool_names

        with pytest.raises(ValidationError, match="read-only mode"):
            await plugin._tool_query({"sql": "INSERT INTO customers (id) VALUES (1)"})
    finally:
        await plugin.disconnect()


async def test_current_behavior_catalog_search_inspect_and_relationship_paths():
    catalog = CatalogPlugin(auto_persist=False)
    registered = await catalog.register_schema(
        _reference_schema(),
        store_type="sqlite",
        store_id="store:shop",
        persist=False,
    )

    search = catalog.catalog_search_schema("store:shop", "customer email")
    inspect = catalog.inspect_asset(
        "store:shop",
        "customers",
        blocked_fields=["email"],
    )
    paths = catalog.find_relationship_paths(
        "store:shop",
        ["orders"],
        ["customers"],
        relationship_types=["foreign_key"],
    )
    tool_view_names = {tool_view.name for tool_view in catalog.get_tool_views()}

    assert registered["store_id"] == "store:shop"
    assert search["total_matches"] >= 1
    assert search["tables"][0]["name"] == "customers"
    assert inspect["success"] is True
    assert inspect["relationships"][0]["source_asset"] == "orders"
    assert paths["reachable"] is True
    assert paths["path_count"] == 1
    assert {
        "search_schema",
        "inspect_asset",
        "find_relationships",
    } <= tool_view_names


async def test_current_behavior_data_quality_profile_tool_uses_configured_db():
    async def query(sql, params=None):
        sql_lower = sql.lower()
        if "pragma_table_info" in sql_lower or "information_schema" in sql_lower:
            return [{"column_name": "total"}]
        if "count(*) as total" in sql_lower:
            return [{"total": 4, "non_null": 3, "distinct_count": 2}]
        if "min(" in sql_lower:
            return [{"min_val": 10, "max_val": 20, "avg_val": 15}]
        return []

    db = MagicMock()
    db.sql_dialect = "postgresql"
    db.query = query
    plugin = DataQualityPlugin(db=db)
    agent = Agent(
        name="quality-reference",
        llm_provider=MockLLMProvider(delay=0),
        plugins=[plugin],
    )

    result = await agent.execute_capability(
        "quality.profile",
        {"table": "orders", "columns": ["total"]},
        owner="data_quality",
    )
    payload = result["evidence"][0]["payload"]

    assert payload["success"] is True
    assert payload["profile"]["total"]["total_rows"] == 4
    assert payload["profile"]["total"]["null_count"] == 1


async def test_current_behavior_lineage_register_trace_and_impact_fallback():
    lineage = LineagePlugin(risk_thresholds={"HIGH": 3, "MEDIUM": 1})

    assert lineage.manifest.id == "lineage"
    assert {capability.id for capability in lineage.declare_capabilities()} == {
        "lineage.trace",
        "lineage.impact.analyze",
        "lineage.flow.register",
        "lineage.path.find",
    }

    await lineage.register_flow(
        "table:raw_orders",
        "table:orders",
        transformation="clean",
    )
    await lineage.register_flow(
        "table:orders",
        "dashboard:revenue",
        transformation="aggregate",
    )

    trace = await lineage.trace_lineage("table:orders")
    impact = await lineage.analyze_impact("table:raw_orders")

    assert trace["upstream_count"] == 1
    assert trace["downstream_count"] == 1
    assert impact["total_affected_count"] == 2
    assert impact["risk_level"] == "MEDIUM"
