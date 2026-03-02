"""
Unit tests for tool registration and resolution on Agent
(daita/agents/agent.py — _resolve_tools, call_tool, available_tools, tool_names,
and plugin tool setup).
"""

import pytest

from daita.agents.agent import Agent
from daita.core.tools import AgentTool
from daita.llm.mock import MockLLMProvider


# ===========================================================================
# Helpers
# ===========================================================================

def _tool(name: str):
    async def h(args):
        return f"result_from_{name}"
    return AgentTool(name=name, description=f"Tool {name}", parameters={}, handler=h)


class TwoToolPlugin:
    """Plugin exposing two tools via get_tools()."""
    def get_tools(self):
        return [_tool("plugin_alpha"), _tool("plugin_beta")]


# ===========================================================================
# _resolve_tools
# ===========================================================================

class TestResolveTools:
    def test_none_returns_all_registered(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        agent.register_tools([_tool("t1"), _tool("t2")])
        resolved = agent._resolve_tools(None)
        names = {t.name for t in resolved}
        assert names == {"t1", "t2"}

    def test_resolve_by_name(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        t = _tool("my_tool")
        agent.register_tool(t)
        resolved = agent._resolve_tools(["my_tool"])
        assert len(resolved) == 1
        assert resolved[0].name == "my_tool"

    def test_unknown_name_raises(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        with pytest.raises(ValueError, match="not found"):
            agent._resolve_tools(["nonexistent"])

    def test_agent_tool_instance_passes_through(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        t = _tool("direct")
        agent.register_tool(t)
        resolved = agent._resolve_tools([t])
        assert resolved[0] is t

    def test_empty_list_returns_empty(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        resolved = agent._resolve_tools([])
        assert resolved == []


# ===========================================================================
# call_tool
# ===========================================================================

class TestCallTool:
    async def test_call_tool_executes_handler(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)

        async def h(args):
            return args["x"] * 2

        t = AgentTool(
            name="double",
            description="Double",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "n"}},
                "required": ["x"],
            },
            handler=h,
        )
        agent.register_tool(t)
        result = await agent.call_tool("double", {"x": 5})
        assert result == 10

    async def test_call_tool_unknown_raises(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        with pytest.raises(RuntimeError, match="not found"):
            await agent.call_tool("ghost", {})


# ===========================================================================
# available_tools and tool_names
# ===========================================================================

class TestAvailableToolsAndNames:
    def test_available_tools_returns_agent_tool_list(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        agent.register_tool(_tool("ping"))
        tools = agent.available_tools
        assert isinstance(tools, list)
        assert all(isinstance(t, AgentTool) for t in tools)

    def test_available_tools_is_copy(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        agent.register_tool(_tool("ping"))
        copy1 = agent.available_tools
        copy2 = agent.available_tools
        assert copy1 is not copy2  # new list each time

    def test_tool_names_returns_strings(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        agent.register_tools([_tool("a"), _tool("b")])
        names = agent.tool_names
        assert isinstance(names, list)
        assert set(names) == {"a", "b"}


# ===========================================================================
# Plugin tool registration on first run
# ===========================================================================

class TestPluginToolSetup:
    async def test_plugin_tools_registered_on_setup(self, mock_llm):
        from tests.conftest import SequentialMockLLM

        agent = Agent(
            name="X",
            llm_provider=SequentialMockLLM(response_sequence=["Done."]),
        )
        agent.add_plugin(TwoToolPlugin())

        # Tools are registered lazily on first setup
        await agent._setup_tools()

        names = agent.tool_names
        assert "plugin_alpha" in names
        assert "plugin_beta" in names

    async def test_tools_setup_only_once(self, mock_llm):
        """Calling _setup_tools() twice should not double-register tools."""
        from tests.conftest import SequentialMockLLM

        agent = Agent(
            name="X",
            llm_provider=SequentialMockLLM(response_sequence=["Done."]),
        )
        agent.add_plugin(TwoToolPlugin())

        await agent._setup_tools()
        count_after_first = agent.tool_registry.tool_count

        await agent._setup_tools()
        count_after_second = agent.tool_registry.tool_count

        assert count_after_first == count_after_second
