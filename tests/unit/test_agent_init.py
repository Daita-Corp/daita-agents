"""
Unit tests for Agent and BaseAgent initialisation (daita/agents/agent.py,
daita/agents/base.py).

Covers:
  - Agent ID and name assignment
  - Lazy LLM initialisation
  - configure_defaults()
  - add_plugin()
  - health property structure
  - get_token_usage() structure
"""

import re

import pytest

from daita.agents.agent import Agent
from daita.config.base import AgentConfig, AgentType
from daita.core.tools import AgentTool
from daita.llm.mock import MockLLMProvider


# ===========================================================================
# Helpers
# ===========================================================================

def make_tool(name="tool_x"):
    async def h(args):
        return "ok"
    return AgentTool(name=name, description="Desc", parameters={}, handler=h)


class MinimalPlugin:
    """Plugin with no tools and no initialize()."""
    def get_tools(self):
        return []


class InitializablePlugin:
    """Plugin that stores the agent_id passed to initialize()."""
    def __init__(self):
        self.received_agent_id = None

    def initialize(self, agent_id: str):
        self.received_agent_id = agent_id

    def get_tools(self):
        return []


class ToolProvidingPlugin:
    """Plugin that exposes two tools via get_tools()."""
    def get_tools(self):
        async def h(args):
            return "result"
        return [
            AgentTool(name="plugin_tool_1", description="T1", parameters={}, handler=h),
            AgentTool(name="plugin_tool_2", description="T2", parameters={}, handler=h),
        ]


# ===========================================================================
# ID and naming
# ===========================================================================

class TestAgentIdentity:
    def test_agent_id_generated_from_name(self, mock_llm):
        agent = Agent(name="My Agent", llm_provider=mock_llm)
        # slug_<8-hex> pattern
        assert re.match(r"^my_agent_[0-9a-f]{8}$", agent.agent_id)

    def test_agent_id_uses_explicit_value(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, agent_id="my-custom-id")
        assert agent.agent_id == "my-custom-id"

    def test_agent_name_stored(self, mock_llm):
        agent = Agent(name="NamedAgent", llm_provider=mock_llm)
        assert agent.name == "NamedAgent"

    def test_agent_default_name(self, mock_llm):
        agent = Agent(llm_provider=mock_llm)
        assert agent.name == "Agent"

    def test_agent_ids_are_unique(self, mock_llm):
        a1 = Agent(name="MyAgent", llm_provider=mock_llm)
        a2 = Agent(name="MyAgent", llm_provider=mock_llm)
        assert a1.agent_id != a2.agent_id


# ===========================================================================
# Lazy LLM initialisation
# ===========================================================================

class TestLazyLLM:
    def test_llm_is_none_without_api_key(self):
        # No API key, string provider — LLM should stay None
        agent = Agent(name="X", llm_provider="openai")
        assert agent._llm is None

    def test_llm_provider_name_stored_for_string_provider(self):
        # When a string provider name is given the provider name is stored
        # for lazy creation (no actual LLM object yet).
        agent = Agent(name="X", llm_provider="openai")
        assert agent._llm_provider_name == "openai"

    def test_llm_instance_sets_initialized_flag(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        assert agent._llm_initialized is True

    def test_llm_instance_is_accessible(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        assert agent.llm is mock_llm

    def test_llm_setter_replaces_provider(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        new_mock = MockLLMProvider(delay=0)
        agent.llm = new_mock
        assert agent.llm is new_mock

    def test_llm_setter_calls_set_agent_id(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        new_mock = MockLLMProvider(delay=0)
        agent.llm = new_mock
        assert new_mock.agent_id == agent.agent_id


# ===========================================================================
# configure_defaults()
# ===========================================================================

class TestConfigureDefaults:
    def setup_method(self):
        # Snapshot class-level defaults so we can restore after each test
        self._original_provider = Agent._default_llm_provider
        self._original_model = Agent._default_model

    def teardown_method(self):
        Agent._default_llm_provider = self._original_provider
        Agent._default_model = self._original_model

    def test_set_default_llm_provider(self):
        Agent.configure_defaults(llm_provider="anthropic")
        assert Agent._default_llm_provider == "anthropic"

    def test_set_default_model(self):
        Agent.configure_defaults(model="gpt-3.5-turbo")
        assert Agent._default_model == "gpt-3.5-turbo"

    def test_unknown_key_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown keys"):
            Agent.configure_defaults(temperature=0.5)

    def test_multiple_keys_set_together(self):
        Agent.configure_defaults(llm_provider="grok", model="grok-2")
        assert Agent._default_llm_provider == "grok"
        assert Agent._default_model == "grok-2"


# ===========================================================================
# Plugin and tool management
# ===========================================================================

class TestPluginAndToolManagement:
    def test_add_plugin_appends_to_sources(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = MinimalPlugin()
        agent.add_plugin(plugin)
        assert plugin in agent.tool_sources

    def test_add_plugin_calls_initialize(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = InitializablePlugin()
        agent.add_plugin(plugin)
        assert plugin.received_agent_id == agent.agent_id

    def test_add_plugin_without_initialize_no_error(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        plugin = MinimalPlugin()
        agent.add_plugin(plugin)  # Should not raise

    async def test_tool_increases_count(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[make_tool("my_tool")])
        await agent._setup_tools()
        assert agent.tool_registry.tool_count == 1

    async def test_multiple_tools_added(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[make_tool("t1"), make_tool("t2")])
        await agent._setup_tools()
        assert agent.tool_registry.tool_count == 2

    async def test_tool_names_reflects_registry(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[make_tool("alpha")])
        await agent._setup_tools()
        assert "alpha" in agent.tool_names

    async def test_available_tools_returns_list(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[make_tool("beta")])
        await agent._setup_tools()
        tools = agent.available_tools
        assert isinstance(tools, list)
        assert tools[0].name == "beta"


# ===========================================================================
# health property
# ===========================================================================

class TestHealthProperty:
    def test_health_has_required_keys(self, basic_agent):
        h = basic_agent.health
        for key in ("id", "name", "type", "running", "metrics", "tools", "relay", "llm"):
            assert key in h, f"Missing key: {key}"

    async def test_health_tools_count_matches_registry(self, mock_llm, simple_tool):
        agent = Agent(name="TestAgent", llm_provider=mock_llm, tools=[simple_tool])
        await agent._setup_tools()
        assert agent.health["tools"]["count"] == 1

    def test_health_relay_disabled_by_default(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        assert agent.health["relay"]["enabled"] is False

    def test_health_relay_enabled_when_set(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, relay="my_channel")
        h = agent.health
        assert h["relay"]["enabled"] is True
        assert h["relay"]["channel"] == "my_channel"

    def test_health_llm_available_with_mock(self, basic_agent):
        assert basic_agent.health["llm"]["available"] is True

    def test_health_id_matches_agent_id(self, basic_agent):
        assert basic_agent.health["id"] == basic_agent.agent_id


# ===========================================================================
# get_token_usage()
# ===========================================================================

class TestGetTokenUsage:
    def test_returns_dict_with_expected_keys(self, basic_agent):
        usage = basic_agent.get_token_usage()
        for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
            assert key in usage

    def test_no_llm_returns_zeros(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        # Before any LLM calls, all values should be zero
        usage = agent.get_token_usage()
        assert usage["total_tokens"] == 0
