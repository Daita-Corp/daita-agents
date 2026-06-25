"""Unit tests for SlackPlugin extension declarations."""

from unittest.mock import AsyncMock

from daita.plugins.manifest import PluginKind
from daita.plugins.registry import ExtensionRegistry
from daita.plugins.slack import SlackPlugin
from daita.runtime import Operation, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin() -> SlackPlugin:
    return SlackPlugin(token="xoxb-test-token", default_channel="#alerts")


def test_slack_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "slack"
    assert plugin.manifest.kind is PluginKind.CONNECTOR
    assert registry.plugin_ids == ("slack",)
    assert {capability.id for capability in registry.capabilities} == {
        "slack.message.send",
        "slack.summary.send",
        "slack.channel.list",
        "slack.message.read",
    }
    assert {view.name for view in registry.tool_views} == {
        "send_slack_message",
        "send_slack_summary",
        "list_slack_channels",
        "read_slack_messages",
    }
    assert registry.evidence_schemas[0].kind == "slack.operation.result"


def test_slack_projected_tools_carry_declared_capability_metadata():
    plugin = make_plugin()

    by_name = projected_tools(plugin)

    assert by_name["send_slack_message"].capability_ids == ("slack.message.send",)
    assert by_name["send_slack_message"].side_effecting is True
    assert by_name["send_slack_message"].idempotent is False
    assert by_name["list_slack_channels"].capability_ids == ("slack.channel.list",)
    assert by_name["list_slack_channels"].side_effecting is False
    assert by_name["list_slack_channels"].idempotent is True
    assert by_name["read_slack_messages"].capability_ids == ("slack.message.read",)
    assert by_name["read_slack_messages"].side_effecting is False


async def test_slack_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    plugin.send_message = AsyncMock(return_value={"channel": "C1", "ts": "123.45"})
    registry = ExtensionRegistry()
    registry.register(plugin)

    executor = registry.get_executor("slack.operations")
    operation = Operation(id="op-1", operation_type="slack.message.send")
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="slack.message.send",
        executor_id="slack.operations",
        input={"channel": "#alerts", "text": "Deploy finished"},
        required_evidence=frozenset({"slack.operation.result"}),
    )

    evidence = await executor.execute(
        task,
        operation,
        {"tool_view": {"name": "send_slack_message"}},
    )

    assert len(evidence) == 1
    assert evidence[0].kind == "slack.operation.result"
    assert evidence[0].owner == "slack"
    assert evidence[0].payload["operation"] == "send_slack_message"
    assert evidence[0].payload["request"]["text"] == "Deploy finished"
    assert evidence[0].payload["result"] == {
        "channel": "C1",
        "timestamp": "123.45",
        "message_sent": True,
    }
    assert evidence[0].metadata["capability_id"] == "slack.message.send"
    assert evidence[0].metadata["tool_view"] == "send_slack_message"
