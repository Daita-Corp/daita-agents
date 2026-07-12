"""Unit tests for RedisMessagingPlugin extension declarations."""

import json
from unittest.mock import MagicMock

import pytest

from daita.plugins.manifest import PluginKind
from daita.plugins.redis_messaging import RedisEncodable, RedisMessagingPlugin
from daita.plugins.registry import ExtensionRegistry
from daita.runtime import AccessMode, Operation, RiskLevel, Task


def make_plugin() -> RedisMessagingPlugin:
    return RedisMessagingPlugin(url="redis://localhost:6379/9")


def test_redis_messaging_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "redis_messaging"
    assert plugin.manifest.kind is PluginKind.CONNECTOR
    assert registry.plugin_ids == ("redis_messaging",)
    assert {capability.id for capability in registry.capabilities} == {
        "redis_messaging.message.publish",
        "redis_messaging.message.latest",
        "redis_messaging.channel.clear",
        "redis_messaging.health.check",
        "redis_messaging.stats.read",
    }
    assert registry.tool_views == ()
    assert registry.evidence_schemas[0].kind == "redis_messaging.operation.result"


def test_redis_messaging_capabilities_are_runtime_only():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    by_id = {capability.id: capability for capability in registry.capabilities}
    publish = by_id["redis_messaging.message.publish"]
    latest = by_id["redis_messaging.message.latest"]
    clear = by_id["redis_messaging.channel.clear"]

    assert publish.runtime_only is True
    assert publish.model_visible is False
    assert publish.access is AccessMode.WRITE
    assert publish.risk is RiskLevel.MEDIUM
    assert publish.side_effecting is True
    assert publish.idempotent is False
    assert latest.side_effecting is False
    assert latest.idempotent is True
    assert clear.access is AccessMode.ADMIN
    assert clear.risk is RiskLevel.HIGH


async def test_redis_messaging_executor_returns_typed_operation_evidence(monkeypatch):
    plugin = make_plugin()

    async def fake_publish(channel, message, publisher=None):
        assert channel == "orders"
        assert message == {"order_id": 42}
        assert publisher == "agent-a"
        return "msg-123"

    monkeypatch.setattr(plugin, "publish", fake_publish)
    registry = ExtensionRegistry()
    registry.register(plugin)

    executor = registry.get_executor("redis_messaging.operations")
    operation = Operation(
        id="op-1",
        operation_type="redis_messaging.message.publish",
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="redis_messaging.message.publish",
        executor_id="redis_messaging.operations",
        input={
            "channel": "orders",
            "message": {"order_id": 42},
            "publisher": "agent-a",
        },
        required_evidence=frozenset({"redis_messaging.operation.result"}),
    )

    evidence = await executor.execute(task, operation, {})

    assert len(evidence) == 1
    assert evidence[0].kind == "redis_messaging.operation.result"
    assert evidence[0].owner == "redis_messaging"
    assert evidence[0].payload["operation"] == "redis_messaging.message.publish"
    assert evidence[0].payload["request"]["channel"] == "orders"
    assert evidence[0].payload["result"] == {"message_id": "msg-123"}
    assert evidence[0].metadata["capability_id"] == ("redis_messaging.message.publish")


async def test_redis_messaging_registry_setup_and_teardown_use_connector_lifecycle(
    monkeypatch,
):
    plugin = make_plugin()
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._running = True
        plugin._redis_pool = MagicMock()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._running = False
        plugin._redis_pool = None

    monkeypatch.setattr(plugin, "connect", fake_connect)
    monkeypatch.setattr(plugin, "disconnect", fake_disconnect)
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("redis_messaging", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


async def test_publish_encodes_stream_fields_for_redis(monkeypatch):
    plugin = make_plugin()
    plugin._running = True

    class FakeRedis:
        fields: dict[RedisEncodable, RedisEncodable] = {}

        async def xadd(self, stream_key, fields, **kwargs):
            self.fields = fields

        async def expire(self, stream_key, ttl):
            return True

        async def publish(self, channel, message):
            return 1

        async def close(self):
            return None

    redis_client = FakeRedis()

    async def fake_get_redis():
        return redis_client

    monkeypatch.setattr(plugin, "_get_redis", fake_get_redis)

    await plugin.publish("orders", {"order_id": 42})

    stored_data = redis_client.fields["data"]
    assert isinstance(stored_data, (str, bytes, bytearray))
    assert json.loads(stored_data) == {"order_id": 42}
    assert redis_client.fields["publisher"] == ""
    assert all(
        isinstance(value, (bytes, bytearray, memoryview, str, int, float))
        for value in redis_client.fields.values()
    )


@pytest.mark.parametrize(
    "fields",
    [
        {b"data": b'{"order_id": 42}'},
        {"data": '{"order_id": 42}'},
    ],
)
async def test_get_latest_decodes_bytes_and_text_payloads(monkeypatch, fields):
    plugin = make_plugin()
    plugin._running = True

    class FakeRedis:
        async def xrevrange(self, stream_key, count):
            return [(b"1-0", fields)]

        async def close(self):
            return None

    async def fake_get_redis():
        return FakeRedis()

    monkeypatch.setattr(plugin, "_get_redis", fake_get_redis)

    assert await plugin.get_latest("orders") == [{"order_id": 42}]
