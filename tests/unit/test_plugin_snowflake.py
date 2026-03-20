"""
Unit tests for SnowflakePlugin and SnowflakeAdminPlugin.

Tests admin/core tool split, query_text truncation, and read_only enforcement
without a real Snowflake connection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from daita.plugins.snowflake import SnowflakePlugin, SnowflakeAdminPlugin


# ---------------------------------------------------------------------------
# Core vs Admin tool split
# ---------------------------------------------------------------------------


ADMIN_TOOL_NAMES = {
    "snowflake_list_warehouses",
    "snowflake_get_query_history",
    "snowflake_list_stages",
    "snowflake_load_from_stage",
    "snowflake_create_stage",
    "snowflake_switch_warehouse",
}

CORE_TOOL_NAMES = {
    "snowflake_query",
    "snowflake_inspect",
    "snowflake_list_schemas",
    "snowflake_count",
    "snowflake_sample",
}


def make_plugin(**kwargs):
    plugin = SnowflakePlugin(account="xy12345", user="u", password="p", warehouse="COMPUTE_WH", database="TESTDB", **kwargs)
    plugin._connection = MagicMock()
    return plugin


def make_admin_plugin(**kwargs):
    plugin = SnowflakeAdminPlugin(account="xy12345", user="u", password="p", warehouse="COMPUTE_WH", database="TESTDB", **kwargs)
    plugin._connection = MagicMock()
    return plugin


def test_base_plugin_does_not_expose_admin_tools():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    for admin_name in ADMIN_TOOL_NAMES:
        assert admin_name not in names, f"Admin tool {admin_name!r} leaked into base plugin"


def test_base_plugin_exposes_core_tools():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    for core_name in CORE_TOOL_NAMES:
        assert core_name in names, f"Core tool {core_name!r} missing from base plugin"


def test_admin_plugin_exposes_admin_tools():
    plugin = make_admin_plugin()
    names = {t.name for t in plugin.get_tools()}
    # Admin plugin exposes all admin tools (except switch_warehouse which requires not read_only)
    for admin_name in ADMIN_TOOL_NAMES - {"snowflake_switch_warehouse"}:
        assert admin_name in names, f"Admin tool {admin_name!r} missing from admin plugin"


def test_admin_plugin_also_exposes_core_tools():
    plugin = make_admin_plugin()
    names = {t.name for t in plugin.get_tools()}
    for core_name in CORE_TOOL_NAMES:
        assert core_name in names, f"Core tool {core_name!r} missing from admin plugin"


def test_switch_warehouse_absent_when_read_only():
    plugin = make_admin_plugin(read_only=True)
    names = {t.name for t in plugin.get_tools()}
    assert "snowflake_switch_warehouse" not in names


def test_switch_warehouse_present_when_not_read_only():
    plugin = make_admin_plugin(read_only=False)
    names = {t.name for t in plugin.get_tools()}
    assert "snowflake_switch_warehouse" in names


def test_execute_absent_in_base_when_read_only():
    plugin = make_plugin(read_only=True)
    names = {t.name for t in plugin.get_tools()}
    assert "snowflake_execute" not in names


def test_execute_present_in_base_when_not_read_only():
    plugin = make_plugin(read_only=False)
    names = {t.name for t in plugin.get_tools()}
    assert "snowflake_execute" in names


# ---------------------------------------------------------------------------
# query_text truncation in get_query_history
# ---------------------------------------------------------------------------


async def test_query_history_truncates_query_text():
    plugin = make_plugin()
    long_sql = "SELECT " + "x" * 500

    async def fake_query_history(limit=20):
        return [
            {"QUERY_TEXT": long_sql, "QUERY_TYPE": "SELECT", "EXECUTION_STATUS": "SUCCESS"},
            {"QUERY_TEXT": "SELECT 1", "QUERY_TYPE": "SELECT", "EXECUTION_STATUS": "SUCCESS"},
        ]

    plugin.query_history = fake_query_history

    result = await plugin._tool_get_query_history({"limit": 5})

    for row in result["queries"]:
        qt = row.get("QUERY_TEXT") or row.get("query_text", "")
        assert len(qt) <= 200, f"query_text not truncated: {len(qt)} chars"


async def test_query_history_short_query_text_unchanged():
    plugin = make_plugin()

    async def fake_query_history(limit=20):
        return [{"QUERY_TEXT": "SELECT 1", "QUERY_TYPE": "SELECT"}]

    plugin.query_history = fake_query_history

    result = await plugin._tool_get_query_history({})

    assert result["queries"][0]["QUERY_TEXT"] == "SELECT 1"


async def test_query_history_handles_lowercase_key():
    """Some Snowflake drivers return lowercase keys."""
    plugin = make_plugin()
    long_sql = "select " + "col, " * 100

    async def fake_query_history(limit=20):
        return [{"query_text": long_sql}]

    plugin.query_history = fake_query_history

    result = await plugin._tool_get_query_history({})

    assert len(result["queries"][0]["query_text"]) <= 200


# ---------------------------------------------------------------------------
# SnowflakeAdminPlugin inherits handler methods
# ---------------------------------------------------------------------------


async def test_admin_plugin_inherits_query_history_handler():
    plugin = make_admin_plugin()

    async def fake_query_history(limit=20):
        return [{"QUERY_TEXT": "SELECT 1"}]

    plugin.query_history = fake_query_history

    result = await plugin._tool_get_query_history({})
    assert "queries" in result
