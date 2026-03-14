"""
Unit tests for daita/plugins/base.py (BasePlugin contract).

Verifies the default behaviour of BasePlugin and that subclasses can
correctly extend it to expose tools.
"""

import pytest

from daita.core.tools import AgentTool
from daita.plugins.base import BasePlugin, LifecyclePlugin

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


# ===========================================================================
# LifecyclePlugin
# ===========================================================================


class ConcreteLifecyclePlugin(LifecyclePlugin):
    """Minimal concrete LifecyclePlugin for testing default behaviour."""
    pass


class CustomLifecyclePlugin(LifecyclePlugin):
    """LifecyclePlugin that records hook calls."""

    def __init__(self):
        self.before_run_prompts = []
        self.stop_called = False

    async def on_before_run(self, prompt: str):
        self.before_run_prompts.append(prompt)
        return f"context for: {prompt}"

    async def on_agent_stop(self):
        self.stop_called = True


class TestLifecyclePlugin:
    def test_is_subclass_of_base_plugin(self):
        assert issubclass(LifecyclePlugin, BasePlugin)

    def test_instance_is_base_plugin(self):
        plugin = ConcreteLifecyclePlugin()
        assert isinstance(plugin, BasePlugin)

    def test_base_plugin_is_not_lifecycle_plugin(self):
        plugin = NoToolPlugin()
        assert not isinstance(plugin, LifecyclePlugin)

    async def test_on_before_run_default_returns_none(self):
        plugin = ConcreteLifecyclePlugin()
        result = await plugin.on_before_run("some prompt")
        assert result is None

    async def test_on_agent_stop_default_does_not_raise(self):
        plugin = ConcreteLifecyclePlugin()
        await plugin.on_agent_stop()  # Should not raise

    async def test_on_before_run_override_called_with_prompt(self):
        plugin = CustomLifecyclePlugin()
        await plugin.on_before_run("hello")
        assert plugin.before_run_prompts == ["hello"]

    async def test_on_before_run_override_returns_context(self):
        plugin = CustomLifecyclePlugin()
        result = await plugin.on_before_run("test prompt")
        assert result == "context for: test prompt"

    async def test_on_agent_stop_override_called(self):
        plugin = CustomLifecyclePlugin()
        await plugin.on_agent_stop()
        assert plugin.stop_called is True

    def test_lifecycle_plugin_still_exposes_tools(self):
        """LifecyclePlugin inherits get_tools from BasePlugin."""
        plugin = ConcreteLifecyclePlugin()
        assert plugin.get_tools() == []
        assert plugin.has_tools is False
