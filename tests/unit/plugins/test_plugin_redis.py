"""Unit tests for RedisPlugin extension declarations."""

import builtins

import pytest

from daita.core.exceptions import ValidationError
from daita.plugins import ExtensionRegistry
from daita.plugins.redis import RedisPlugin
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools


class MockRedisPlugin(RedisPlugin):
    def __init__(self, **kwargs):
        super().__init__(url="redis://localhost:6379/9", **kwargs)
        self.connected = False
        self.values = {"user:1": "Ada"}
        self.deleted = []

    async def connect(self):
        self.connected = True
        self._connection = object()

    async def disconnect(self):
        self.connected = False
        self._connection = None

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ttl=None):
        self.values[key] = value
        return True

    async def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)

    async def exists(self, *keys):
        return sum(1 for key in keys if key in self.values)

    async def keys(self, pattern="*", count=1000):
        return list(self.values)[:count]

    async def ttl(self, key):
        return 60

    async def expire(self, key, seconds):
        return True

    async def hset(self, key, mapping):
        return len(mapping)

    async def hgetall(self, key):
        return {"theme": "dark"}

    async def lpush(self, key, *values):
        return len(values)

    async def rpush(self, key, *values):
        return len(values)

    async def lrange(self, key, start=0, stop=-1):
        return ["a", "b"]

    async def sadd(self, key, *members):
        return len(members)

    async def smembers(self, key):
        return ["blue", "green"]

    async def dbsize(self):
        return len(self.values)


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


def test_redis_client_access_requires_connection():
    plugin = RedisPlugin(url="redis://localhost:6379/9")

    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.client


async def test_redis_client_access_tracks_connection_lifecycle(module_stub):
    class FakeClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    client = FakeClient()
    redis_class = type(
        "Redis", (), {"from_url": staticmethod(lambda *args, **kwargs: client)}
    )
    module_stub("redis.asyncio", Redis=redis_class)
    plugin = RedisPlugin(url="redis://localhost:6379/9")

    await plugin.connect()
    assert plugin.client is client

    await plugin.disconnect()
    assert client.closed is True
    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.client


async def test_redis_missing_sdk_raises_import_error_with_extra_hint(monkeypatch):
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "redis.asyncio":
            raise ImportError("redis not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    plugin = RedisPlugin(url="redis://localhost:6379/9")

    with pytest.raises(ImportError, match="pip install 'daita-agents\\[redis\\]'"):
        await plugin.connect()


def test_redis_plugin_declares_extension_first_contract():
    plugin = MockRedisPlugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "redis"
    assert registry.plugin_ids == ("redis",)
    assert {capability.id for capability in registry.capabilities} == {
        "redis.key.get",
        "redis.key.scan",
        "redis.key.exists",
        "redis.key.ttl",
        "redis.hash.read",
        "redis.list.read",
        "redis.set.read",
        "redis.info.read",
        "redis.key.set",
        "redis.key.delete",
        "redis.hash.write",
        "redis.list.lpush",
        "redis.list.rpush",
        "redis.set.add",
        "redis.key.expire",
    }
    assert {view.name for view in registry.tool_views} >= {"redis_get", "redis_set"}
    assert registry.get_tool_view_owner("redis_get") == "redis"
    assert registry.evidence_schemas[0].kind == "redis.operation.result"


def test_redis_read_only_filters_write_capabilities_and_tool_views():
    plugin = MockRedisPlugin(read_only=True)
    registry = ExtensionRegistry()

    registry.register(plugin)

    capability_ids = {capability.id for capability in registry.capabilities}
    tool_view_names = {view.name for view in registry.tool_views}

    assert "redis.key.get" in capability_ids
    assert "redis.info.read" in capability_ids
    assert "redis.key.set" not in capability_ids
    assert "redis.key.delete" not in capability_ids
    assert "redis_get" in tool_view_names
    assert "redis_set" not in tool_view_names


def test_redis_capabilities_carry_access_and_safety_metadata():
    plugin = MockRedisPlugin()
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["redis.key.get"].access is AccessMode.READ
    assert by_id["redis.key.get"].risk is RiskLevel.LOW
    assert by_id["redis.key.get"].side_effecting is False
    assert by_id["redis.key.set"].access is AccessMode.WRITE
    assert by_id["redis.key.set"].risk is RiskLevel.MEDIUM
    assert by_id["redis.key.set"].side_effecting is True


async def test_redis_executor_returns_typed_operation_evidence():
    plugin = MockRedisPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("redis.key.get", owner="redis")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(capability, {"key": "user:1"})

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "redis.operation.result"
    assert evidence[0].owner == "redis"
    assert evidence[0].payload["operation"] == "redis.key.get"
    assert evidence[0].payload["request"] == {"key": "user:1"}
    assert evidence[0].payload["result"] == {"key": "user:1", "value": "Ada"}
    assert evidence[0].metadata["capability_id"] == "redis.key.get"


async def test_redis_write_executor_uses_existing_tool_handler():
    plugin = MockRedisPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("redis.key.set", owner="redis")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"key": "user:2", "value": "Grace", "ttl": 120},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "redis.key.set"
    assert evidence[0].payload["result"] == {"key": "user:2", "ok": True}
    assert plugin.values["user:2"] == "Grace"


async def test_redis_registry_setup_and_teardown_use_connector_lifecycle():
    plugin = MockRedisPlugin()
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("redis", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert plugin.is_connected is False


def test_redis_legacy_tools_carry_declared_capability_metadata():
    plugin = MockRedisPlugin()

    tools = projected_tools(plugin)

    assert tools["redis_get"].capability_ids == ("redis.key.get",)
    assert tools["redis_keys"].capability_ids == ("redis.key.scan",)
    assert tools["redis_info"].capability_ids == ("redis.info.read",)
    assert tools["redis_set"].capability_ids == ("redis.key.set",)
    assert tools["redis_delete"].capability_ids == ("redis.key.delete",)
    assert tools["redis_get"].side_effecting is False
    assert tools["redis_set"].side_effecting is True
