"""Unit tests for RedisMessagingPlugin extension declarations."""

from daita.plugins.manifest import PluginKind
from daita.plugins.redis_messaging import RedisMessagingPlugin
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


async def test_redis_messaging_executor_returns_typed_operation_evidence():
    plugin = make_plugin()

    async def fake_publish(channel, message, publisher=None):
        assert channel == "orders"
        assert message == {"order_id": 42}
        assert publisher == "agent-a"
        return "msg-123"

    plugin.publish = fake_publish
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


async def test_redis_messaging_registry_setup_and_teardown_use_connector_lifecycle():
    plugin = make_plugin()
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._running = True
        plugin._redis_pool = object()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._running = False
        plugin._redis_pool = None

    plugin.connect = fake_connect
    plugin.disconnect = fake_disconnect
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("redis_messaging", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False
