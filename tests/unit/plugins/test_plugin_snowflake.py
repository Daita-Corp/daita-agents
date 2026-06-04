"""
Unit tests for SnowflakePlugin and SnowflakeAdminPlugin.

Tests admin/core tool split, query_text truncation, and read_only enforcement
without a real Snowflake connection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from daita.plugins import ExtensionRegistry
from daita.plugins.snowflake import SnowflakePlugin, SnowflakeAdminPlugin
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools, projected_tool_names

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
    plugin = SnowflakePlugin(
        account="xy12345",
        user="u",
        password="p",
        warehouse="COMPUTE_WH",
        database="TESTDB",
        **kwargs,
    )
    plugin._connection = MagicMock()
    return plugin


def make_admin_plugin(**kwargs):
    plugin = SnowflakeAdminPlugin(
        account="xy12345",
        user="u",
        password="p",
        warehouse="COMPUTE_WH",
        database="TESTDB",
        **kwargs,
    )
    plugin._connection = MagicMock()
    return plugin


def _operation_and_task(capability, input_payload):
    operation = Operation(
        id="op-1",
        operation_type=next(iter(capability.operation_types)),
        request=input_payload,
        required_evidence=capability.output_evidence,
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id=capability.id,
        executor_id=capability.executor,
        input=input_payload,
        required_evidence=capability.output_evidence,
    )
    return operation, task


# ---------------------------------------------------------------------------
# Extension declarations
# ---------------------------------------------------------------------------


def test_snowflake_plugin_declares_extension_first_contract():
    plugin = make_plugin(read_only=False)
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "snowflake"
    assert registry.plugin_ids == ("snowflake",)
    assert {capability.id for capability in registry.capabilities} == {
        "snowflake.sql.query",
        "snowflake.schema.inspect",
        "snowflake.table.count",
        "snowflake.table.sample",
        "snowflake.schema.list",
        "snowflake.sql.execute",
    }
    assert {view.name for view in registry.tool_views} == {
        "snowflake_query",
        "snowflake_inspect",
        "snowflake_count",
        "snowflake_sample",
        "snowflake_list_schemas",
        "snowflake_execute",
    }
    assert registry.get_tool_view_owner("snowflake_query") == "snowflake"
    assert registry.evidence_schemas[0].kind == "snowflake.operation.result"


def test_snowflake_admin_plugin_declares_admin_extension_contract():
    plugin = make_admin_plugin(read_only=False)
    registry = ExtensionRegistry()

    registry.register(plugin)

    capability_ids = {capability.id for capability in registry.capabilities}
    tool_view_names = {view.name for view in registry.tool_views}

    assert "snowflake.sql.query" in capability_ids
    assert "snowflake.warehouse.list" in capability_ids
    assert "snowflake.query_history.read" in capability_ids
    assert "snowflake.stage.load" in capability_ids
    assert "snowflake.stage.create" in capability_ids
    assert "snowflake.warehouse.switch" in capability_ids
    assert "snowflake_list_warehouses" in tool_view_names
    assert "snowflake_switch_warehouse" in tool_view_names


def test_snowflake_read_only_filters_write_capabilities_and_tool_views():
    plugin = make_admin_plugin(read_only=True)
    registry = ExtensionRegistry()

    registry.register(plugin)

    capability_ids = {capability.id for capability in registry.capabilities}
    tool_view_names = {view.name for view in registry.tool_views}

    assert "snowflake.sql.query" in capability_ids
    assert "snowflake.stage.load" in capability_ids
    assert "snowflake.sql.execute" not in capability_ids
    assert "snowflake.warehouse.switch" not in capability_ids
    assert "snowflake_query" in tool_view_names
    assert "snowflake_execute" not in tool_view_names
    assert "snowflake_switch_warehouse" not in tool_view_names


def test_snowflake_capabilities_carry_access_and_safety_metadata():
    plugin = make_admin_plugin(read_only=False)
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["snowflake.sql.query"].access is AccessMode.READ
    assert by_id["snowflake.sql.query"].risk is RiskLevel.MEDIUM
    assert by_id["snowflake.sql.query"].side_effecting is False
    assert by_id["snowflake.schema.inspect"].access is AccessMode.METADATA_READ
    assert by_id["snowflake.sql.execute"].access is AccessMode.WRITE
    assert by_id["snowflake.sql.execute"].risk is RiskLevel.HIGH
    assert by_id["snowflake.stage.create"].access is AccessMode.ADMIN
    assert by_id["snowflake.stage.create"].side_effecting is True


async def test_snowflake_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    calls = []

    async def fake_query(sql, params=None):
        calls.append((sql, params))
        return [{"NAME": "Ada"}]

    plugin.query = fake_query
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("snowflake.sql.query", owner="snowflake")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"sql": "SELECT NAME FROM USERS", "params": ["active"]},
    )

    evidence = await executor.execute(task, operation, {})

    assert calls == [("SELECT NAME FROM USERS LIMIT 50", ["active"])]
    assert evidence[0].kind == "snowflake.operation.result"
    assert evidence[0].owner == "snowflake"
    assert evidence[0].payload["operation"] == "data.query"
    assert evidence[0].payload["request"]["sql"] == "SELECT NAME FROM USERS"
    assert evidence[0].payload["result"]["rows"] == [{"NAME": "Ada"}]
    assert evidence[0].metadata["capability_id"] == "snowflake.sql.query"


async def test_snowflake_admin_executor_uses_existing_tool_handler():
    plugin = make_admin_plugin()

    async def fake_list_warehouses():
        return [{"name": "COMPUTE_WH"}]

    plugin.list_warehouses = fake_list_warehouses
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("snowflake.warehouse.list", owner="snowflake")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(capability, {})

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "warehouse.inspect"
    assert evidence[0].payload["result"] == {"warehouses": [{"name": "COMPUTE_WH"}]}


async def test_snowflake_registry_setup_and_teardown_use_connector_lifecycle():
    plugin = SnowflakePlugin(
        account="xy12345",
        user="u",
        password="p",
        warehouse="COMPUTE_WH",
        database="TESTDB",
    )
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._connection = object()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._connection = None

    plugin.connect = fake_connect
    plugin.disconnect = fake_disconnect
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("snowflake", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


async def test_connect_missing_connector_raises_import_error_with_extra_hint():
    plugin = SnowflakePlugin(
        account="xy12345",
        user="u",
        password="p",
        warehouse="COMPUTE_WH",
        database="TESTDB",
    )
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "snowflake.connector":
            raise ImportError("missing connector")
        return original_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="daita-agents\\[snowflake\\]"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("builtins.__import__", fake_import)
            await plugin.connect()


def test_key_pair_missing_crypto_raises_import_error_with_extra_hint():
    plugin = SnowflakePlugin(
        account="xy12345",
        user="u",
        private_key_path="/tmp/missing-key.p8",
        warehouse="COMPUTE_WH",
        database="TESTDB",
    )
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("missing crypto")
        return original_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="daita-agents\\[snowflake\\]"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("builtins.__import__", fake_import)
            plugin._load_private_key()


def test_base_plugin_does_not_expose_admin_tools():
    plugin = make_plugin()
    names = projected_tool_names(plugin)
    for admin_name in ADMIN_TOOL_NAMES:
        assert (
            admin_name not in names
        ), f"Admin tool {admin_name!r} leaked into base plugin"


def test_base_plugin_exposes_core_tools():
    plugin = make_plugin()
    names = projected_tool_names(plugin)
    for core_name in CORE_TOOL_NAMES:
        assert core_name in names, f"Core tool {core_name!r} missing from base plugin"


def test_admin_plugin_exposes_admin_tools():
    plugin = make_admin_plugin()
    names = projected_tool_names(plugin)
    # Admin plugin exposes all admin tools (except switch_warehouse which requires not read_only)
    for admin_name in ADMIN_TOOL_NAMES - {"snowflake_switch_warehouse"}:
        assert (
            admin_name in names
        ), f"Admin tool {admin_name!r} missing from admin plugin"


def test_admin_plugin_also_exposes_core_tools():
    plugin = make_admin_plugin()
    names = projected_tool_names(plugin)
    for core_name in CORE_TOOL_NAMES:
        assert core_name in names, f"Core tool {core_name!r} missing from admin plugin"


def test_switch_warehouse_absent_when_read_only():
    plugin = make_admin_plugin(read_only=True)
    names = projected_tool_names(plugin)
    assert "snowflake_switch_warehouse" not in names


def test_switch_warehouse_present_when_not_read_only():
    plugin = make_admin_plugin(read_only=False)
    names = projected_tool_names(plugin)
    assert "snowflake_switch_warehouse" in names


def test_execute_absent_in_base_when_read_only():
    plugin = make_plugin(read_only=True)
    names = projected_tool_names(plugin)
    assert "snowflake_execute" not in names


def test_execute_present_in_base_when_not_read_only():
    plugin = make_plugin(read_only=False)
    names = projected_tool_names(plugin)
    assert "snowflake_execute" in names


def test_base_projected_tools_carry_declared_capability_metadata():
    plugin = make_plugin(read_only=False)
    tools = projected_tools(plugin)

    assert tools["snowflake_query"].capability_ids == ("snowflake.sql.query",)
    assert tools["snowflake_inspect"].capability_ids == ("snowflake.schema.inspect",)
    assert tools["snowflake_count"].capability_ids == ("snowflake.table.count",)
    assert tools["snowflake_sample"].capability_ids == ("snowflake.table.sample",)
    assert tools["snowflake_list_schemas"].capability_ids == ("snowflake.schema.list",)
    assert tools["snowflake_execute"].capability_ids == ("snowflake.sql.execute",)
    assert tools["snowflake_query"].side_effecting is False
    assert tools["snowflake_execute"].side_effecting is True


def test_admin_projected_tools_carry_declared_capability_metadata():
    plugin = make_admin_plugin(read_only=False)
    tools = projected_tools(plugin)

    assert tools["snowflake_list_warehouses"].capability_ids == (
        "snowflake.warehouse.list",
    )
    assert tools["snowflake_get_query_history"].capability_ids == (
        "snowflake.query_history.read",
    )
    assert tools["snowflake_list_stages"].capability_ids == ("snowflake.stage.list",)
    assert tools["snowflake_load_from_stage"].capability_ids == (
        "snowflake.stage.load",
    )
    assert tools["snowflake_create_stage"].capability_ids == ("snowflake.stage.create",)
    assert tools["snowflake_switch_warehouse"].capability_ids == (
        "snowflake.warehouse.switch",
    )
    assert tools["snowflake_list_warehouses"].side_effecting is False
    assert tools["snowflake_create_stage"].side_effecting is True


# ---------------------------------------------------------------------------
# query_text truncation in get_query_history
# ---------------------------------------------------------------------------


async def test_query_history_truncates_query_text():
    plugin = make_plugin()
    long_sql = "SELECT " + "x" * 500

    async def fake_query_history(limit=20):
        return [
            {
                "QUERY_TEXT": long_sql,
                "QUERY_TYPE": "SELECT",
                "EXECUTION_STATUS": "SUCCESS",
            },
            {
                "QUERY_TEXT": "SELECT 1",
                "QUERY_TYPE": "SELECT",
                "EXECUTION_STATUS": "SUCCESS",
            },
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
