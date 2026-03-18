"""
Unit tests for tool registration and resolution on Agent
(daita/agents/agent.py — _resolve_tools, call_tool, available_tools, tool_names,
FocusedTool, plugin tool setup, and _execute_tool_call).
"""

import asyncio
import pytest

from daita.agents.agent import Agent, FocusedTool, _execute_tool_call
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
    async def test_none_returns_all_registered(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[_tool("t1"), _tool("t2")])
        await agent._setup_tools()
        resolved = agent._resolve_tools(None)
        names = {t.name for t in resolved}
        assert names == {"t1", "t2"}

    async def test_resolve_by_name(self, mock_llm):
        t = _tool("my_tool")
        agent = Agent(name="X", llm_provider=mock_llm, tools=[t])
        await agent._setup_tools()
        resolved = agent._resolve_tools(["my_tool"])
        assert len(resolved) == 1
        assert resolved[0].name == "my_tool"

    def test_unknown_name_raises(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm)
        with pytest.raises(ValueError, match="not found"):
            agent._resolve_tools(["nonexistent"])

    async def test_agent_tool_instance_passes_through(self, mock_llm):
        t = _tool("direct")
        agent = Agent(name="X", llm_provider=mock_llm, tools=[t])
        await agent._setup_tools()
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
        agent = Agent(name="X", llm_provider=mock_llm, tools=[t])
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
    async def test_available_tools_returns_agent_tool_list(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[_tool("ping")])
        await agent._setup_tools()
        tools = agent.available_tools
        assert isinstance(tools, list)
        assert all(isinstance(t, AgentTool) for t in tools)

    async def test_available_tools_is_copy(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[_tool("ping")])
        await agent._setup_tools()
        copy1 = agent.available_tools
        copy2 = agent.available_tools
        assert copy1 is not copy2  # new list each time

    async def test_tool_names_returns_strings(self, mock_llm):
        agent = Agent(name="X", llm_provider=mock_llm, tools=[_tool("a"), _tool("b")])
        await agent._setup_tools()
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


# ===========================================================================
# _execute_tool_call
# ===========================================================================


def _make_returning_tool(name: str, return_value=None):
    async def h(args):
        return return_value

    return AgentTool(name=name, description=f"Tool {name}", parameters={}, handler=h)


def _make_slow_tool(name: str, sleep: float = 10.0):
    async def h(args):
        await asyncio.sleep(sleep)
        return "never"

    return AgentTool(
        name=name, description="Slow", parameters={}, handler=h, timeout_seconds=0.01
    )


class TestExecuteToolCall:
    async def test_finds_and_executes_tool(self):
        t = _make_returning_tool("calc", return_value=42)
        result = await _execute_tool_call({"name": "calc", "arguments": {}}, [t])
        assert result == 42

    async def test_unknown_tool_returns_error_dict(self):
        result = await _execute_tool_call({"name": "ghost", "arguments": {}}, [])
        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"]

    async def test_tool_timeout_returns_error_dict(self):
        t = _make_slow_tool("slow")
        result = await _execute_tool_call({"name": "slow", "arguments": {}}, [t])
        assert isinstance(result, dict)
        assert "error" in result
        assert "timed out" in result["error"]

    async def test_tool_exception_returns_error_dict(self):
        async def h(args):
            raise RuntimeError("unexpected failure")

        t = AgentTool(name="broken", description="Broken", parameters={}, handler=h)
        result = await _execute_tool_call({"name": "broken", "arguments": {}}, [t])
        assert isinstance(result, dict)
        assert "error" in result
        assert "failed" in result["error"]


# ===========================================================================
# FocusedTool — focus routing and result handling
# ===========================================================================


def _make_tool(name: str, returns, parameters: dict | None = None):
    """Create an AgentTool whose handler returns a fixed value."""

    async def h(args):
        return returns

    return AgentTool(
        name=name,
        description=f"Tool {name}",
        parameters=parameters or {"type": "object", "properties": {}, "required": []},
        handler=h,
    )


def _make_focused(tool: AgentTool, focus: str) -> FocusedTool:
    return FocusedTool(tool, focus)


class TestFocusedToolHandler:
    """
    Verify FocusedTool routes correctly across three cases:
      1. Tool schema has 'focus' property → inject DSL into args, skip Python apply_focus.
      2. Tool returns {"rows": [...]} wrapper → focus applied to rows only, wrapper preserved.
      3. Tool returns raw list/dict → existing Python apply_focus behaviour unchanged.
    """

    async def test_pushdown_route_injects_focus_into_args(self):
        """
        A tool that declares 'focus' in its schema (SQL plugin style) should
        receive the focus DSL as an arg and return its result directly —
        no Python apply_focus layer on top.
        """
        received_args = {}

        async def capturing_handler(args):
            received_args.update(args)
            # Simulate SQL pushdown: already returns focused data
            return {
                "success": True,
                "rows": [{"id": 1, "status": "active"}],
                "row_count": 1,
            }

        tool = AgentTool(
            name="postgres_query",
            description="SQL query",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["sql"],
            },
            handler=capturing_handler,
        )
        focused = FocusedTool(tool, "status == 'active' | SELECT id, status")

        result = await focused.handler({"sql": "SELECT * FROM t"})

        # focus was injected into the args the handler received
        assert received_args.get("focus") == "status == 'active' | SELECT id, status"
        # result passed through unmodified (pushdown already applied)
        assert result["rows"] == [{"id": 1, "status": "active"}]

    async def test_rows_wrapper_focus_applied_to_rows_not_wrapper(self):
        """
        A tool returning {"rows": [...], "row_count": N} without a 'focus'
        schema property should have focus applied to result["rows"], not the
        whole dict — fixing the silent data-destruction bug.
        """
        rows = [
            {"id": 1, "status": "active", "amount": 100},
            {"id": 2, "status": "inactive", "amount": 200},
            {"id": 3, "status": "active", "amount": 300},
        ]
        tool = _make_tool(
            "my_db_tool",
            returns={"success": True, "rows": rows, "row_count": 3},
        )
        focused = FocusedTool(tool, "status == 'active'")

        result = await focused.handler({})

        # Wrapper preserved
        assert result["success"] is True
        assert "row_count" in result
        # Focus applied to rows, not the wrapper dict
        assert result["row_count"] == 2
        assert all(r["status"] == "active" for r in result["rows"])
        assert {r["id"] for r in result["rows"]} == {1, 3}

    async def test_rows_wrapper_select_projection(self):
        """SELECT projection through the rows wrapper strips unwanted columns."""
        rows = [
            {"id": 1, "name": "Alice", "email": "a@example.com", "score": 90},
            {"id": 2, "name": "Bob", "email": "b@example.com", "score": 70},
        ]
        tool = _make_tool(
            "my_tool", returns={"success": True, "rows": rows, "row_count": 2}
        )
        focused = FocusedTool(tool, "SELECT id, name")

        result = await focused.handler({})

        assert result["row_count"] == 2
        for row in result["rows"]:
            assert set(row.keys()) == {"id", "name"}, f"Unexpected keys: {row.keys()}"

    async def test_rows_wrapper_row_count_updated(self):
        """row_count in the wrapper must reflect rows after focus, not before."""
        rows = [{"status": "active"}] * 3 + [{"status": "inactive"}] * 7
        tool = _make_tool("t", returns={"success": True, "rows": rows, "row_count": 10})
        focused = FocusedTool(tool, "status == 'active'")

        result = await focused.handler({})

        assert result["row_count"] == 3
        assert len(result["rows"]) == 3

    async def test_raw_list_unchanged_behaviour(self):
        """Tools returning a plain list still go through standard apply_focus."""
        rows = [
            {"x": 1, "keep": True},
            {"x": 2, "keep": False},
            {"x": 3, "keep": True},
        ]
        tool = _make_tool("raw_tool", returns=rows)
        focused = FocusedTool(tool, "keep == True")

        result = await focused.handler({})

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(r["keep"] is True for r in result)

    async def test_none_result_passthrough(self):
        """None returned by handler should pass through without error."""
        tool = _make_tool("null_tool", returns=None)
        focused = FocusedTool(tool, "x > 0")

        result = await focused.handler({})

        assert result is None

    async def test_filter_on_wrapper_dict_no_longer_destroys_data(self):
        """
        Regression: before the fix, apply_focus on {"success": True, "rows": [...]}
        would treat the whole dict as one row, fail the filter, and return {}.
        Verify that no longer happens.
        """
        rows = [{"status": "completed", "amount": 500}]
        tool = _make_tool("t", returns={"success": True, "rows": rows, "row_count": 1})
        focused = FocusedTool(tool, "status == 'completed'")

        result = await focused.handler({})

        # Must NOT return empty — the filter matches the row inside rows
        assert result != {}
        assert result["rows"] == [{"status": "completed", "amount": 500}]
