"""
Tests for critical error handling paths that were previously untested.

Covers:
- json_serializer: private attr filtering, to_dict, __dict__, TypeError
- LLMResult.from_response: unexpected type fallback
- Agent._resolve_tools: unregistered tool name
- Loop detection: 3 consecutive identical tool errors -> AgentError
- Run timeout: timeout_seconds triggers AgentError
- BaseLLMProvider._validate_api_key: empty key raises ValueError
- BaseLLMProvider.generate(): string input normalization
"""

import json
import pytest

from daita.agents.agent import Agent
from daita.agents.chat.llm_turn import LLMResult
from daita.agents.chat.tools import json_serializer
from daita.core.tools import LocalTool
from daita.core.exceptions import AgentError
from daita.llm.mock import MockLLMProvider

from tests.conftest import SequentialMockLLM

# ---------------------------------------------------------------------------
# json_serializer — edge cases
# ---------------------------------------------------------------------------


class TestJsonSerializer:
    def test_object_with_to_dict_method(self):
        class HasToDict:
            def to_dict(self):
                return {"key": "value"}

        result = json_serializer(HasToDict())
        assert result == {"key": "value"}

    def test_object_with_dict_filters_private_attrs(self):
        class HasDict:
            def __init__(self):
                self.public = "visible"
                self._private = "hidden"
                self.__dunder = "also_hidden"

        obj = HasDict()
        result = json_serializer(obj)
        assert "public" in result
        assert "_private" not in result

    def test_object_with_no_serialization_path_raises_type_error(self):
        """Objects with neither to_dict nor __dict__ (unlikely but possible) raise TypeError."""

        # Use a class that blocks __dict__.
        class NoDict:
            __slots__ = ()

        with pytest.raises(TypeError):
            json_serializer(NoDict())

    def test_json_serializer_used_in_json_dumps(self):
        """Confirm json_serializer integrates correctly with json.dumps."""
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
        serialized = json.dumps(data, default=json_serializer)
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

        tool = LocalTool(
            name="my_tool",
            description="test",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=h,
        )
        agent = Agent(
            name="TestAgent", llm_provider=MockLLMProvider(delay=0), tools=[tool]
        )
        resolved = agent._resolve_tools(None)
        assert any(t.name == "my_tool" for t in resolved)

    def test_agentool_instance_passes_through(self):
        async def h(args):
            return {}

        tool = LocalTool(
            name="direct_tool",
            description="test",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=h,
        )
        agent = Agent(name="TestAgent", llm_provider=MockLLMProvider(delay=0))
        resolved = agent._resolve_tools([tool])
        assert resolved == [tool]


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

        fail_tool = LocalTool(
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
            response_sequence=[
                same_call,
                same_call,
                same_call,
                same_call,
                "Final answer",
            ]
        )

        agent = Agent(name="LoopAgent", llm_provider=llm, tools=[fail_tool])

        with pytest.raises(AgentError, match="Loop detected"):
            await agent.run("please call fail_tool")

    async def test_blocked_repair_results_trigger_loop_detection(self):
        async def blocked_repair(args):
            return {
                "blocked_repeat": True,
                "repair_required": True,
                "status": "repeated_invalid_sql_blocked",
                "message": "same SQL blocked",
            }

        validate_tool = LocalTool(
            name="validate",
            description="validate SQL",
            parameters={
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
            handler=blocked_repair,
        )
        same_call = {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "validate", "arguments": {"sql": "bad"}}
            ],
        }
        llm = SequentialMockLLM(
            response_sequence=[
                same_call,
                same_call,
                same_call,
                same_call,
                "Final answer",
            ]
        )
        agent = Agent(name="RepairLoopAgent", llm_provider=llm, tools=[validate_tool])

        with pytest.raises(AgentError, match="Loop detected"):
            await agent.run("please validate this SQL")


class TestFinalSynthesisWithoutTools:
    async def test_terminal_tool_can_disable_tools_for_final_turn(self):
        async def lookup(args):
            return {"answer": 42}

        lookup_tool = LocalTool(
            name="lookup",
            description="look up a value",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lookup,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {}}],
                },
                "The answer is 42.",
            ]
        )
        agent = Agent(name="SynthAgent", llm_provider=llm, tools=[lookup_tool])

        result = await agent.run(
            "look it up",
            tools=["lookup"],
            final_synthesis_without_tools=True,
            terminal_tools=("lookup",),
        )

        assert result == "The answer is 42."
        assert [tool["function"]["name"] for tool in llm.call_history[0]["tools"]] == [
            "lookup"
        ]
        assert llm.call_history[1]["tools"] in (None, [])


# ---------------------------------------------------------------------------
# Run timeout
# ---------------------------------------------------------------------------


class TestRunTimeout:
    async def test_timeout_seconds_raises_agent_error(self):
        """When timeout_seconds is set and the run exceeds it, AgentError is raised."""
        agent = Agent(
            name="SlowAgent",
            llm_provider=MockLLMProvider(delay=10),
        )

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
