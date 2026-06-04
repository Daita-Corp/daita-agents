"""Unit tests for BigQueryPlugin extension declarations."""

from daita.plugins import ExtensionRegistry
from daita.plugins.bigquery import BigQueryPlugin
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin(read_only: bool = False):
    plugin = BigQueryPlugin(
        project="test-project", dataset="analytics", read_only=read_only
    )
    plugin._client = object()
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


def test_bigquery_plugin_declares_extension_first_contract():
    plugin = make_plugin(read_only=False)
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "bigquery"
    assert registry.plugin_ids == ("bigquery",)
    assert {capability.id for capability in registry.capabilities} == {
        "bigquery.sql.query",
        "bigquery.schema.inspect",
        "bigquery.table.count",
        "bigquery.table.sample",
        "bigquery.dataset.list",
        "bigquery.sql.execute",
    }
    assert {view.name for view in registry.tool_views} == {
        "bigquery_query",
        "bigquery_inspect",
        "bigquery_count",
        "bigquery_sample",
        "bigquery_list_datasets",
        "bigquery_execute",
    }
    assert registry.get_tool_view_owner("bigquery_query") == "bigquery"
    assert registry.evidence_schemas[0].kind == "bigquery.operation.result"


def test_bigquery_read_only_filters_write_capabilities_and_tool_views():
    plugin = make_plugin(read_only=True)
    registry = ExtensionRegistry()

    registry.register(plugin)

    capability_ids = {capability.id for capability in registry.capabilities}
    tool_view_names = {view.name for view in registry.tool_views}

    assert "bigquery.sql.query" in capability_ids
    assert "bigquery.dataset.list" in capability_ids
    assert "bigquery.sql.execute" not in capability_ids
    assert "bigquery_query" in tool_view_names
    assert "bigquery_execute" not in tool_view_names


def test_bigquery_capabilities_carry_access_and_safety_metadata():
    plugin = make_plugin(read_only=False)
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["bigquery.sql.query"].access is AccessMode.READ
    assert by_id["bigquery.sql.query"].risk is RiskLevel.MEDIUM
    assert by_id["bigquery.sql.query"].side_effecting is False
    assert by_id["bigquery.schema.inspect"].access is AccessMode.METADATA_READ
    assert by_id["bigquery.sql.execute"].access is AccessMode.WRITE
    assert by_id["bigquery.sql.execute"].risk is RiskLevel.HIGH
    assert by_id["bigquery.sql.execute"].side_effecting is True


async def test_bigquery_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    calls = []

    async def fake_query(sql, params=None):
        calls.append((sql, params))
        return [{"name": "Ada"}]

    plugin.query = fake_query
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("bigquery.sql.query", owner="bigquery")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"sql": "SELECT name FROM users", "params": ["active"]},
    )

    evidence = await executor.execute(task, operation, {})

    assert calls == [("SELECT name FROM users LIMIT 50", ["active"])]
    assert evidence[0].kind == "bigquery.operation.result"
    assert evidence[0].owner == "bigquery"
    assert evidence[0].payload["operation"] == "data.query"
    assert evidence[0].payload["request"]["sql"] == "SELECT name FROM users"
    assert evidence[0].payload["result"]["rows"] == [{"name": "Ada"}]
    assert evidence[0].metadata["capability_id"] == "bigquery.sql.query"


async def test_bigquery_write_executor_uses_existing_tool_handler():
    plugin = make_plugin(read_only=False)

    async def fake_execute(sql, params=None):
        assert sql == "UPDATE users SET active = true"
        assert params == [1]
        return 7

    plugin.execute = fake_execute
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("bigquery.sql.execute", owner="bigquery")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"sql": "UPDATE users SET active = true", "params": [1]},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "write.execute"
    assert evidence[0].payload["result"] == {"affected_rows": 7}


async def test_bigquery_registry_setup_and_teardown_use_connector_lifecycle():
    plugin = BigQueryPlugin(project="test-project", dataset="analytics")
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._client = object()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._client = None

    plugin.connect = fake_connect
    plugin.disconnect = fake_disconnect
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("bigquery", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


def test_bigquery_legacy_tools_carry_declared_capability_metadata():
    plugin = make_plugin(read_only=False)

    tools = projected_tools(plugin)

    assert tools["bigquery_query"].capability_ids == ("bigquery.sql.query",)
    assert tools["bigquery_inspect"].capability_ids == ("bigquery.schema.inspect",)
    assert tools["bigquery_count"].capability_ids == ("bigquery.table.count",)
    assert tools["bigquery_sample"].capability_ids == ("bigquery.table.sample",)
    assert tools["bigquery_list_datasets"].capability_ids == ("bigquery.dataset.list",)
    assert tools["bigquery_execute"].capability_ids == ("bigquery.sql.execute",)
    assert tools["bigquery_query"].side_effecting is False
    assert tools["bigquery_execute"].side_effecting is True
