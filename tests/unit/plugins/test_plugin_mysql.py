"""
Unit tests for MySQLPlugin extension declarations.

These tests avoid live MySQL dependencies by using a small subclass that
overrides the direct query/execute methods while preserving plugin contracts.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from daita.core.exceptions import ValidationError
from daita.plugins import ExtensionRegistry
from daita.plugins.mysql import MySQLPlugin
from daita.runtime import Operation, Task
from tests.unit.plugins.projection_helpers import projected_tools


class MockMySQLPlugin(MySQLPlugin):
    def __init__(self, **kwargs):
        super().__init__(
            host="localhost",
            database="testdb",
            username="user",
            password="pass",
            **kwargs,
        )
        self.connected = False
        self.queries = []
        self.executed = []

    async def connect(self):
        self.connected = True
        self._connection = object()

    async def disconnect(self):
        self.connected = False
        self._connection = None

    async def query(self, sql, params=None):
        self.queries.append((sql, params))
        if sql.startswith("EXPLAIN"):
            return [{"id": 1, "select_type": "SIMPLE"}]
        if "INFORMATION_SCHEMA.KEY_COLUMN_USAGE" in sql:
            return [
                {
                    "table_name": "orders",
                    "column_name": "user_id",
                    "referenced_table_name": "users",
                    "referenced_column_name": "id",
                    "constraint_name": "orders_user_id_fk",
                }
            ]
        return [{"id": 1, "name": "Ada"}]

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return 3

    async def tables(self):
        return ["users"]

    async def describe(self, table):
        return [
            {
                "column_name": "id",
                "data_type": "int",
                "is_nullable": "NO",
                "column_default": None,
                "is_primary_key": True,
            }
        ]


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


def test_mysql_pool_access_requires_connection():
    plugin = MySQLPlugin(host="localhost", database="testdb")

    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.pool


async def test_mysql_pool_access_tracks_connection_lifecycle(monkeypatch):
    class FakePool:
        def __init__(self):
            self.closed = False
            self.waited = False

        def close(self):
            self.closed = True

        async def wait_closed(self):
            self.waited = True

    pool = FakePool()

    async def fake_create_pool(**kwargs):
        return pool

    monkeypatch.setitem(
        sys.modules, "aiomysql", SimpleNamespace(create_pool=fake_create_pool)
    )
    plugin = MySQLPlugin(host="localhost", database="testdb")

    await plugin.connect()
    assert plugin.pool is pool

    await plugin.disconnect()
    assert pool.closed is True
    assert pool.waited is True
    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.pool


async def test_mysql_missing_sdk_raises_import_error_with_extra_hint(monkeypatch):
    real_import_module = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        if name == "aiomysql":
            raise ImportError("aiomysql not installed")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)
    plugin = MySQLPlugin(host="localhost", database="testdb")

    with pytest.raises(ImportError, match="pip install 'daita-agents\\[mysql\\]'"):
        await plugin.connect()


def test_mysql_plugin_declares_extension_first_contract():
    plugin = MockMySQLPlugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert registry.plugin_ids == ("mysql",)
    assert {capability.owner for capability in registry.capabilities} == {"mysql"}
    assert {capability.id for capability in registry.capabilities} == {
        "db.schema.inspect",
        "db.sql.validate",
        "db.sql.execute_read",
        "db.sql.execute_write",
        "db.sql.explain",
    }
    assert {view.name for view in registry.tool_views} == {
        "mysql_query",
        "mysql_inspect",
    }
    assert registry.get_tool_view_owner("mysql_query") == "mysql"


async def test_mysql_read_executor_returns_typed_query_evidence():
    plugin = MockMySQLPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("db.sql.execute_read", owner="mysql")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"sql": "SELECT id, name FROM users", "params": []},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "query.result"
    assert evidence[0].owner == "mysql"
    assert evidence[0].payload["rows"] == [{"id": 1, "name": "Ada"}]
    assert evidence[0].payload["sql"].endswith("LIMIT 50")


async def test_mysql_schema_executor_returns_profile_evidence():
    plugin = MockMySQLPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("db.schema.inspect", owner="mysql")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(capability, {"tables": ["users"]})

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "schema.asset_profile"
    assert evidence[0].owner == "mysql"
    assert evidence[0].payload["database_type"] == "mysql"
    assert evidence[0].payload["tables"][0]["name"] == "users"
    assert evidence[0].payload["foreign_keys"][0]["constraint_name"] == (
        "orders_user_id_fk"
    )


async def test_mysql_write_and_explain_executors_delegate_to_plugin_methods():
    plugin = MockMySQLPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)

    write_capability = registry.get_capability("db.sql.execute_write", owner="mysql")
    write_executor = registry.get_executor(write_capability.executor)
    operation, task = _operation_and_task(
        write_capability,
        {"sql": "UPDATE users SET name = %s WHERE id = %s", "params": ["Grace", 1]},
    )
    write_evidence = await write_executor.execute(task, operation, {})

    explain_capability = registry.get_capability("db.sql.explain", owner="mysql")
    explain_executor = registry.get_executor(explain_capability.executor)
    operation, task = _operation_and_task(
        explain_capability,
        {"sql": "SELECT id FROM users", "params": []},
    )
    explain_evidence = await explain_executor.execute(task, operation, {})

    assert write_evidence[0].payload == {
        "sql": "UPDATE users SET name = %s WHERE id = %s",
        "affected_rows": 3,
    }
    assert plugin.executed == [
        ("UPDATE users SET name = %s WHERE id = %s", ["Grace", 1])
    ]
    assert explain_evidence[0].kind == "sql.explain.plan"
    assert explain_evidence[0].payload["plan"] == [{"id": 1, "select_type": "SIMPLE"}]


def test_mysql_projected_tools_and_runtime_capabilities_carry_metadata():
    plugin = MockMySQLPlugin(read_only=False)

    tools = projected_tools(plugin)
    registry = ExtensionRegistry()
    registry.register(plugin)
    capabilities = {capability.id: capability for capability in registry.capabilities}

    assert tools["mysql_query"].capability_ids == ("db.sql.execute_read",)
    assert tools["mysql_inspect"].capability_ids == ("db.schema.inspect",)
    assert capabilities["db.sql.execute_write"].side_effecting is True
    assert capabilities["db.sql.explain"].side_effecting is False
    assert tools["mysql_query"].side_effecting is False
