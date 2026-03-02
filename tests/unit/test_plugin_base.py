"""
Unit tests for daita/plugins/base.py (BasePlugin contract).

Verifies the default behaviour of BasePlugin and that subclasses can
correctly extend it to expose tools.
"""

import pytest

from daita.core.tools import AgentTool
from daita.plugins.base import BasePlugin


# ===========================================================================
# Helpers — concrete subclasses
# ===========================================================================

class NoToolPlugin(BasePlugin):
    """Minimal plugin: does not override anything."""
    pass


class SingleToolPlugin(BasePlugin):
    """Plugin that exposes exactly one tool."""

    def get_tools(self):
        async def h(args):
            return "ok"

        return [
            AgentTool(
                name="do_thing",
                description="Does a thing",
                parameters={},
                handler=h,
            )
        ]


class MultiToolPlugin(BasePlugin):
    """Plugin that exposes multiple tools."""

    def get_tools(self):
        async def h(args):
            return "ok"

        return [
            AgentTool(name="tool_1", description="T1", parameters={}, handler=h),
            AgentTool(name="tool_2", description="T2", parameters={}, handler=h),
            AgentTool(name="tool_3", description="T3", parameters={}, handler=h),
        ]


class InitCapturingPlugin(BasePlugin):
    """Plugin that records the agent_id it receives during initialize()."""

    def __init__(self):
        self.received_agent_id = None

    def initialize(self, agent_id: str):
        self.received_agent_id = agent_id


# ===========================================================================
# BasePlugin defaults
# ===========================================================================

class TestBasePluginDefaults:
    def test_get_tools_returns_empty_list(self):
        plugin = NoToolPlugin()
        assert plugin.get_tools() == []

    def test_has_tools_false_when_no_tools(self):
        plugin = NoToolPlugin()
        assert plugin.has_tools is False

    def test_initialize_is_callable_without_error(self):
        plugin = NoToolPlugin()
        plugin.initialize(agent_id="agent-xyz")  # Should not raise

    def test_initialize_is_no_op_on_base(self):
        plugin = NoToolPlugin()
        result = plugin.initialize(agent_id="ignored")
        assert result is None  # No return value


# ===========================================================================
# Subclass with tools
# ===========================================================================

class TestPluginWithTools:
    def test_has_tools_true_when_get_tools_nonempty(self):
        plugin = SingleToolPlugin()
        assert plugin.has_tools is True

    def test_get_tools_returns_agent_tool_instances(self):
        plugin = SingleToolPlugin()
        tools = plugin.get_tools()
        assert all(isinstance(t, AgentTool) for t in tools)

    def test_get_tools_count_matches_exposed(self):
        plugin = MultiToolPlugin()
        assert len(plugin.get_tools()) == 3

    def test_tool_names_are_strings(self):
        plugin = SingleToolPlugin()
        tool = plugin.get_tools()[0]
        assert isinstance(tool.name, str)
        assert tool.name == "do_thing"


# ===========================================================================
# initialize() with agent_id
# ===========================================================================

class TestPluginInitialize:
    def test_initialize_receives_agent_id(self):
        plugin = InitCapturingPlugin()
        plugin.initialize(agent_id="test-agent-001")
        assert plugin.received_agent_id == "test-agent-001"

    def test_initialize_called_multiple_times_uses_last(self):
        plugin = InitCapturingPlugin()
        plugin.initialize(agent_id="first")
        plugin.initialize(agent_id="second")
        assert plugin.received_agent_id == "second"
