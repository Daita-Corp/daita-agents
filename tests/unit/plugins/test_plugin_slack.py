"""Unit tests for SlackPlugin extension declarations and lifecycle."""

import builtins
from unittest.mock import AsyncMock, MagicMock

import pytest

from daita.core.exceptions import AuthenticationError, PluginError
from daita.plugins.manifest import PluginKind
from daita.plugins.registry import ExtensionRegistry
from daita.plugins.slack import SlackPlugin
from daita.runtime import Operation, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin() -> SlackPlugin:
    return SlackPlugin(token="xoxb-test-token", default_channel="#alerts")


class FakeSlackApiError(Exception):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


def _stub_slack_sdk(module_stub, constructor):
    module_stub("slack_sdk.errors", SlackApiError=FakeSlackApiError)
    module_stub("slack_sdk.web.async_client", AsyncWebClient=constructor)


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


async def test_slack_executor_returns_typed_operation_evidence(monkeypatch):
    plugin = make_plugin()
    monkeypatch.setattr(
        plugin,
        "send_message",
        AsyncMock(return_value={"channel": "C1", "ts": "123.45"}),
    )
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


def test_slack_client_requires_authenticated_connection():
    plugin = make_plugin()

    with pytest.raises(PluginError, match="not connected"):
        _ = plugin.client


async def test_slack_connect_publishes_client_and_identity_after_auth(module_stub):
    plugin = make_plugin()
    client = MagicMock()
    client.auth_test = AsyncMock(
        return_value={
            "user_id": "U1",
            "team_id": "T1",
            "team": "Daita",
            "user": "daita-bot",
            "bot_id": "B1",
        }
    )
    constructor = MagicMock(return_value=client)
    _stub_slack_sdk(module_stub, constructor)

    await plugin.connect()

    assert plugin.client is client
    assert plugin.is_connected is True
    assert plugin._user_info == {
        "user_id": "U1",
        "team_id": "T1",
        "team": "Daita",
        "user": "daita-bot",
        "bot_id": "B1",
    }

    await plugin.disconnect()

    assert plugin.is_connected is False
    assert plugin._user_info is None


async def test_slack_connect_does_not_publish_client_when_auth_fails(module_stub):
    plugin = make_plugin()
    client = MagicMock()
    client.auth_test = AsyncMock(side_effect=RuntimeError("unavailable"))
    _stub_slack_sdk(module_stub, MagicMock(return_value=client))

    with pytest.raises(PluginError, match="Failed to connect"):
        await plugin.connect()

    assert plugin.is_connected is False
    assert plugin._user_info is None


async def test_slack_connect_maps_invalid_auth(module_stub):
    plugin = make_plugin()
    response = MagicMock()
    response.get.return_value = "invalid_auth"
    client = MagicMock()
    client.auth_test = AsyncMock(
        side_effect=FakeSlackApiError("invalid token", response=response)
    )
    _stub_slack_sdk(module_stub, MagicMock(return_value=client))

    with pytest.raises(AuthenticationError, match="Invalid Slack token"):
        await plugin.connect()

    assert plugin.is_connected is False


async def test_slack_connect_reports_missing_sdk(monkeypatch):
    plugin = make_plugin()
    real_import = builtins.__import__

    def import_without_slack(name, *args, **kwargs):
        if name.startswith("slack_sdk"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_slack)

    with pytest.raises(ImportError, match=r"daita-agents\[slack\]"):
        await plugin.connect()

    assert plugin.is_connected is False


async def test_slack_empty_channel_and_history_results_are_normalized():
    plugin = make_plugin()
    client = MagicMock()
    client.conversations_list = AsyncMock(return_value={})
    client.conversations_history = AsyncMock(return_value={"messages": []})
    plugin._client = client

    assert await plugin.get_channels() == []
    assert await plugin.get_channel_history("C1") == []


async def test_slack_send_message_tool_requires_text():
    plugin = make_plugin()

    with pytest.raises(PluginError, match="non-empty text"):
        await plugin._tool_send_message({"channel": "C1"})


async def test_slack_send_summary_tool_requires_summary():
    plugin = make_plugin()

    with pytest.raises(PluginError, match="non-empty summary"):
        await plugin._tool_send_summary({"channel": "C1"})


async def test_slack_read_messages_tool_rejects_boolean_limit():
    plugin = make_plugin()

    with pytest.raises(PluginError, match="positive integer"):
        await plugin._tool_read_messages({"channel": "C1", "limit": True})
