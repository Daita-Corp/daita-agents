"""
Tests for critical error handling paths that were previously untested.

Covers:
- _execute_tool_call: not found / exception / timeout error dicts
- _json_serializer: private attr filtering, to_dict, __dict__, TypeError
- LLMResult.from_response: unexpected type fallback
- Agent._resolve_tools: unregistered tool name
- Agent._build_initial_conversation: plugin on_before_run exception swallowed
- Loop detection: 3 consecutive identical tool errors -> AgentError
- Run timeout: timeout_seconds triggers AgentError
- BaseLLMProvider._validate_api_key: empty key raises ValueError
- BaseLLMProvider.generate(): string input normalization
"""

import asyncio
import json
import pytest

from daita.agents.agent import Agent, LLMResult, _execute_tool_call, _json_serializer
from daita.core.tools import AgentTool
from daita.core.exceptions import AgentError
from daita.llm.mock import MockLLMProvider
from daita.plugins.base import LifecyclePlugin

from tests.conftest import SequentialMockLLM


# ---------------------------------------------------------------------------
# _execute_tool_call — error dict shapes
# ---------------------------------------------------------------------------


class TestExecuteToolCall:
    def _make_tool(self, name, handler, timeout=None):
        return AgentTool(
            name=name,
            description="test",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=handler,
            timeout_seconds=timeout,
        )

    async def test_tool_not_found_returns_error_dict(self):
        result = await _execute_tool_call(
            {"name": "nonexistent", "arguments": {}}, tools=[]
        )
        assert isinstance(result, dict)
        assert "error" in result
        assert "nonexistent" in result["error"]

    async def test_handler_exception_returns_error_dict(self):
        async def broken_handler(args):
            raise ValueError("something went wrong")

        tool = self._make_tool("broken", broken_handler)
        result = await _execute_tool_call({"name": "broken", "arguments": {}}, [tool])

        assert isinstance(result, dict)
        assert "error" in result
        assert "broken" in result["error"]
        assert "something went wrong" in result["error"]

    async def test_handler_timeout_returns_error_dict(self):
        async def slow_handler(args):
            await asyncio.sleep(10)

        tool = self._make_tool("slow", slow_handler, timeout=0.001)
        result = await _execute_tool_call({"name": "slow", "arguments": {}}, [tool])

        assert isinstance(result, dict)
        assert "error" in result
        assert "timed out" in result["error"]

    async def test_successful_handler_returns_result_directly(self):
        async def good_handler(args):
            return {"value": 42}

        tool = self._make_tool("good", good_handler)
        result = await _execute_tool_call({"name": "good", "arguments": {}}, [tool])

        assert result == {"value": 42}

    async def test_tool_error_dict_does_not_raise(self):
        """Errors are returned as dicts, not raised as exceptions."""
        result = await _execute_tool_call(
            {"name": "missing", "arguments": {}}, tools=[]
        )
        assert isinstance(result, dict)  # Should not raise


# ---------------------------------------------------------------------------
# _json_serializer — edge cases
# ---------------------------------------------------------------------------


class TestJsonSerializer:
    def test_object_with_to_dict_method(self):
        class HasToDict:
            def to_dict(self):
                return {"key": "value"}

        result = _json_serializer(HasToDict())
        assert result == {"key": "value"}

    def test_object_with_dict_filters_private_attrs(self):
        class HasDict:
            def __init__(self):
                self.public = "visible"
                self._private = "hidden"
                self.__dunder = "also_hidden"

        obj = HasDict()
        result = _json_serializer(obj)
        assert "public" in result
        assert "_private" not in result

    def test_object_with_no_serialization_path_raises_type_error(self):
        """Objects with neither to_dict nor __dict__ (unlikely but possible) raise TypeError."""
        import ctypes

        # A ctypes instance has no __dict__ and no to_dict
        # Use a simpler approach: create a class that blocks __dict__
        class NoDict:
            __slots__ = ()

        with pytest.raises(TypeError):
            _json_serializer(NoDict())

    def test_json_serializer_used_in_json_dumps(self):
        """Confirm _json_serializer integrates correctly with json.dumps."""
        from datetime import datetime
        from decimal import Decimal
        from uuid import UUID

        data = {
            "ts": datetime(2024, 1, 1, 12, 0, 0),
            "amount": Decimal("3.14"),
            "id": UUID("12345678-1234-5678-1234-567812345678"),
            "raw": b"bytes",
        }
        # Should not raise
        serialized = json.dumps(data, default=_json_serializer)
        parsed = json.loads(serialized)
        assert parsed["amount"] == 3.14
        assert parsed["id"] == "12345678-1234-5678-1234-567812345678"


# ---------------------------------------------------------------------------
# LLMResult.from_response — unexpected type fallback
# ---------------------------------------------------------------------------


class TestLLMResultFromResponse:
    def test_string_response(self):
        result = LLMResult.from_response("hello")
        assert result.text == "hello"
        assert result.tool_calls == []

    def test_dict_response_with_content_and_tool_calls(self):
        tc = [{"id": "c1", "name": "foo", "arguments": {}}]
        result = LLMResult.from_response({"content": "thinking...", "tool_calls": tc})
        assert result.text == "thinking..."
        assert result.tool_calls == tc

    def test_unexpected_type_falls_back_to_str(self):
        result = LLMResult.from_response(42)
        assert result.text == "42"
        assert result.tool_calls == []

    def test_none_falls_back_to_str(self):
        result = LLMResult.from_response(None)
        assert result.text == "None"
        assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Agent._resolve_tools — unregistered tool name
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_unknown_tool_name_raises_value_error(self):
        agent = Agent(name="TestAgent", llm_provider=MockLLMProvider(delay=0))
        with pytest.raises(ValueError, match="'unknown_tool' not found"):
            agent._resolve_tools(["unknown_tool"])

    def test_none_returns_all_registered_tools(self):
        async def h(args):
            return {}

        tool = AgentTool(
            name="my_tool",
            description="test",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=h,
        )
        agent = Agent(name="TestAgent", llm_provider=MockLLMProvider(delay=0), tools=[tool])
        resolved = agent._resolve_tools(None)
        assert any(t.name == "my_tool" for t in resolved)

    def test_agentool_instance_passes_through(self):
        async def h(args):
            return {}

        tool = AgentTool(
            name="direct_tool",
            description="test",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=h,
        )
        agent = Agent(name="TestAgent", llm_provider=MockLLMProvider(delay=0))
        resolved = agent._resolve_tools([tool])
        assert resolved == [tool]


# ---------------------------------------------------------------------------
# Agent._build_initial_conversation — plugin on_before_run exception swallowed
# ---------------------------------------------------------------------------


class TestBuildInitialConversation:
    async def test_plugin_exception_in_on_before_run_is_swallowed(self):
        """If a plugin's on_before_run raises, the exception is silently caught
        and the conversation is still built normally."""

        class BrokenPlugin(LifecyclePlugin):
            def get_tools(self):
                return []

            async def on_before_run(self, prompt):
                raise RuntimeError("Plugin is broken")

        agent = Agent(
            name="TestAgent",
            llm_provider=MockLLMProvider(delay=0),
            prompt="System prompt",
        )
        agent.add_plugin(BrokenPlugin())

        # Should not raise
        conversation = await agent._build_initial_conversation("hello")
        assert conversation[-1]["role"] == "user"
        assert conversation[-1]["content"] == "hello"

    async def test_system_prompt_included_when_plugin_ok(self):
        agent = Agent(
            name="TestAgent",
            llm_provider=MockLLMProvider(delay=0),
            prompt="You are a test agent.",
        )
        conversation = await agent._build_initial_conversation("hi")
        system_msgs = [m for m in conversation if m["role"] == "system"]
        assert any("You are a test agent." in m["content"] for m in system_msgs)

    async def test_on_before_run_context_included_when_no_error(self):
        class GoodPlugin(LifecyclePlugin):
            def get_tools(self):
                return []

            async def on_before_run(self, prompt):
                return "Extra context for the agent."

        agent = Agent(name="TestAgent", llm_provider=MockLLMProvider(delay=0))
        agent.add_plugin(GoodPlugin())

        conversation = await agent._build_initial_conversation("hi")
        system_msgs = [m for m in conversation if m["role"] == "system"]
        assert any("Extra context" in m["content"] for m in system_msgs)


# ---------------------------------------------------------------------------
# Loop detection: 3 identical tool errors -> AgentError
# ---------------------------------------------------------------------------


class TestLoopDetection:
    async def test_three_consecutive_identical_tool_errors_raise_agent_error(self):
        """When the same tool with the same arguments returns an error dict
        3 times in a row, the agent detects a loop and raises AgentError."""

        # Tool that always errors
        async def always_fail(args):
            return {"error": "permanent failure"}

        fail_tool = AgentTool(
            name="fail_tool",
            description="always fails",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=always_fail,
        )

        # LLM keeps calling the same tool with same args
        same_call = {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "fail_tool", "arguments": {"x": 1}}],
        }
        llm = SequentialMockLLM(
            response_sequence=[same_call, same_call, same_call, same_call, "Final answer"]
        )

        agent = Agent(name="LoopAgent", llm_provider=llm, tools=[fail_tool])

        with pytest.raises(AgentError, match="Loop detected"):
            await agent.run("please call fail_tool")


# ---------------------------------------------------------------------------
# Run timeout
# ---------------------------------------------------------------------------


class TestRunTimeout:
    async def test_timeout_seconds_raises_agent_error(self):
        """When timeout_seconds is set and the run exceeds it, AgentError is raised."""

        async def slow_handler(args):
            await asyncio.sleep(10)
            return "done"

        slow_tool = AgentTool(
            name="slow_op",
            description="very slow",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=slow_handler,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "slow_op", "arguments": {}}],
                },
                "done",
            ]
        )
        agent = Agent(name="SlowAgent", llm_provider=llm, tools=[slow_tool])

        with pytest.raises(AgentError, match="timed out"):
            await agent.run("go slow", timeout_seconds=0.05)


# ---------------------------------------------------------------------------
# BaseLLMProvider._validate_api_key
# ---------------------------------------------------------------------------


class TestValidateApiKey:
    def test_falsy_api_key_raises_value_error(self):
        provider = MockLLMProvider(delay=0)
        provider.api_key = None
        with pytest.raises(ValueError):
            provider._validate_api_key()

    def test_empty_string_api_key_raises_value_error(self):
        provider = MockLLMProvider(delay=0)
        provider.api_key = ""
        with pytest.raises(ValueError):
            provider._validate_api_key()

    def test_valid_api_key_does_not_raise(self):
        provider = MockLLMProvider(delay=0)
        provider.api_key = "sk-test-key"
        provider._validate_api_key()  # should not raise


# ---------------------------------------------------------------------------
# BaseLLMProvider.generate() — string input normalization
# ---------------------------------------------------------------------------


class TestGenerateStringNormalization:
    async def test_string_input_does_not_raise(self):
        """generate() accepts a plain string and normalizes it to a message list."""
        llm = SequentialMockLLM(response_sequence=["Hello!"])
        # Should not raise even though "hello" is a string, not a list
        result = await llm.generate("hello")
        assert result == "Hello!"

    async def test_string_input_appears_in_call_history_as_list(self):
        """After normalization, messages in call_history should be a list."""
        llm = SequentialMockLLM(response_sequence=["ok"])
        await llm.generate("test string input")
        messages = llm.call_history[0]["messages"]
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test string input"
