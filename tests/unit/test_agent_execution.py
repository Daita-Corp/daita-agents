"""
Unit tests for Agent execution loop (daita/agents/agent.py).

Covers:
  - run() returns string, run_detailed() returns dict with expected keys
  - System prompt injection
  - Max iterations limit raises AgentError
  - Tool calling: tool is executed, result appears in run_detailed()
  - JSON serializer handles datetime / Decimal / UUID / bytes
  - on_event callback receives ITERATION and COMPLETE events
"""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List
from uuid import UUID, uuid4

import pytest

from daita.agents.agent import Agent
from daita.core.exceptions import AgentError
from daita.core.streaming import EventType
from daita.core.tools import AgentTool
from daita.llm.mock import MockLLMProvider

from tests.conftest import SequentialMockLLM


# ===========================================================================
# Helpers
# ===========================================================================

def _make_agent(responses: List[Any], tools=None) -> Agent:
    """Create an Agent with a SequentialMockLLM loaded with the given responses."""
    llm = SequentialMockLLM(response_sequence=responses)
    agent = Agent(name="ExecAgent", llm_provider=llm)
    if tools:
        for t in tools:
            agent.register_tool(t)
    return agent


def _add_tool():
    """Return an 'add' AgentTool used in tool-calling tests."""
    async def h(args):
        return args["a"] + args["b"]

    return AgentTool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First"},
                "b": {"type": "integer", "description": "Second"},
            },
            "required": ["a", "b"],
        },
        handler=h,
    )


# ===========================================================================
# Basic execution results
# ===========================================================================

class TestBasicExecution:
    async def test_run_returns_string(self):
        agent = _make_agent(["Hello from the agent."])
        result = await agent.run("Say hi")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_run_detailed_returns_dict(self):
        agent = _make_agent(["Hi there."])
        result = await agent.run_detailed("Say hi")
        assert isinstance(result, dict)

    async def test_run_detailed_has_result_key(self):
        agent = _make_agent(["Answer text."])
        result = await agent.run_detailed("prompt")
        assert "result" in result
        assert isinstance(result["result"], str)

    async def test_run_detailed_has_iterations_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run_detailed("prompt")
        assert "iterations" in result
        assert result["iterations"] >= 1

    async def test_run_detailed_has_tool_calls_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run_detailed("prompt")
        assert "tool_calls" in result
        assert isinstance(result["tool_calls"], list)

    async def test_run_detailed_has_tokens_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run_detailed("prompt")
        assert "tokens" in result

    async def test_run_detailed_has_cost_key(self):
        agent = _make_agent(["Done."])
        result = await agent.run_detailed("prompt")
        assert "cost" in result

    async def test_run_detailed_has_agent_id(self):
        agent = _make_agent(["Done."])
        result = await agent.run_detailed("prompt")
        assert result.get("agent_id") == agent.agent_id

    async def test_run_detailed_has_agent_name(self):
        agent = _make_agent(["Done."])
        result = await agent.run_detailed("prompt")
        assert result.get("agent_name") == agent.name

    async def test_run_no_llm_raises_agent_error(self):
        agent = Agent(name="NoLLM", llm_provider="openai")
        # No API key → llm property returns None
        with pytest.raises(AgentError, match="No API key"):
            await agent.run("hello")


# ===========================================================================
# System prompt injection
# ===========================================================================

class TestSystemPromptInjection:
    async def test_system_prompt_sent_in_conversation(self):
        llm = SequentialMockLLM(response_sequence=["Done."])
        agent = Agent(name="X", llm_provider=llm, prompt="You are a helpful assistant.")
        await agent.run("hi")
        # The first call's messages should contain a system message
        first_call = llm.call_history[0]
        messages = first_call["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles

    async def test_system_prompt_content_is_injected(self):
        llm = SequentialMockLLM(response_sequence=["Done."])
        agent = Agent(name="X", llm_provider=llm, prompt="Be concise.")
        await agent.run("hi")
        messages = llm.call_history[0]["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert any("Be concise." in m["content"] for m in system_msgs)

    async def test_no_system_prompt_when_none(self):
        llm = SequentialMockLLM(response_sequence=["Done."])
        agent = Agent(name="X", llm_provider=llm)  # no prompt
        await agent.run("hi")
        messages = llm.call_history[0]["messages"]
        roles = [m["role"] for m in messages]
        assert "system" not in roles


# ===========================================================================
# Max iterations
# ===========================================================================

class TestMaxIterations:
    async def test_max_iterations_raises_agent_error(self):
        # LLM always returns tool calls; no tool registered so the result
        # is an error message, but the loop still repeats.  After
        # max_iterations the agent should raise AgentError.
        always_tool_call = [
            {
                "content": "Calling tool...",
                "tool_calls": [
                    {"id": "tc", "name": "nonexistent", "arguments": {}}
                ],
            }
        ] * 10  # more than max_iterations

        agent = _make_agent(always_tool_call)
        with pytest.raises(AgentError, match="Max iterations"):
            await agent.run("go", max_iterations=2)

    async def test_single_iteration_when_no_tool_calls(self):
        agent = _make_agent(["Direct answer."])
        result = await agent.run_detailed("prompt", max_iterations=5)
        assert result["iterations"] == 1


# ===========================================================================
# Tool calling loop
# ===========================================================================

class TestToolCallingLoop:
    async def test_tool_handler_is_called(self):
        call_log = []

        async def h(args):
            call_log.append(args)
            return args["a"] + args["b"]

        tool = AgentTool(
            name="add",
            description="Add",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "a"},
                    "b": {"type": "integer", "description": "b"},
                },
                "required": ["a", "b"],
            },
            handler=h,
        )

        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Adding.",
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 2, "b": 3}}
                    ],
                },
                "The answer is 5.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm)
        agent.register_tool(tool)

        await agent.run("add 2 and 3")
        assert len(call_log) == 1
        assert call_log[0] == {"a": 2, "b": 3}

    async def test_tool_call_appears_in_run_detailed(self):
        tool = _add_tool()
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Adding.",
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 1}}
                    ],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm)
        agent.register_tool(tool)

        result = await agent.run_detailed("add 1 and 1")
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "add"

    async def test_iterations_incremented_per_llm_call(self):
        tool = _add_tool()
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Step 1.",
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 1}}
                    ],
                },
                "Final answer.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm)
        agent.register_tool(tool)
        result = await agent.run_detailed("go")
        # 1 iteration for tool call + 1 iteration for final answer = 2
        assert result["iterations"] == 2


# ===========================================================================
# JSON serialiser in _append_tool_messages
# ===========================================================================

class TestJsonSerialiser:
    """
    Tests for the custom json_serializer inside _append_tool_messages.
    We exercise it by making a tool return values that are not natively
    JSON-serialisable and verifying that run() completes without error.
    """

    async def _run_with_tool_returning(self, value):
        async def h(args):
            return value

        tool = AgentTool(
            name="special",
            description="Returns special type",
            parameters={},
            handler=h,
        )
        llm = SequentialMockLLM(
            response_sequence=[
                {
                    "content": "Calling.",
                    "tool_calls": [{"id": "c1", "name": "special", "arguments": {}}],
                },
                "Done.",
            ]
        )
        agent = Agent(name="T", llm_provider=llm)
        agent.register_tool(tool)
        # If serialisation fails this will raise; we just need it not to.
        await agent.run("go")

    async def test_datetime_serialised(self):
        await self._run_with_tool_returning(datetime(2024, 1, 15, 12, 0, 0))

    async def test_date_serialised(self):
        await self._run_with_tool_returning(date(2024, 1, 15))

    async def test_decimal_serialised(self):
        await self._run_with_tool_returning(Decimal("3.14"))

    async def test_uuid_serialised(self):
        await self._run_with_tool_returning(uuid4())

    async def test_bytes_serialised(self):
        await self._run_with_tool_returning(b"raw bytes")


# ===========================================================================
# on_event streaming callback
# ===========================================================================

class TestOnEventCallback:
    async def test_no_on_event_does_not_crash(self):
        agent = _make_agent(["Done."])
        # on_event=None is the default — should not raise
        result = await agent.run("hi", on_event=None)
        assert isinstance(result, str)

    async def test_on_event_receives_iteration_event(self):
        events = []

        def collect(event):
            events.append(event)

        llm = MockLLMProvider(delay=0)
        agent = Agent(name="X", llm_provider=llm)

        await agent.run("hi", on_event=collect)

        iteration_events = [e for e in events if e.type == EventType.ITERATION]
        assert len(iteration_events) >= 1

    async def test_on_event_receives_complete_event(self):
        events = []

        def collect(event):
            events.append(event)

        llm = MockLLMProvider(delay=0)
        agent = Agent(name="X", llm_provider=llm)

        await agent.run("hi", on_event=collect)

        complete_events = [e for e in events if e.type == EventType.COMPLETE]
        assert len(complete_events) == 1
