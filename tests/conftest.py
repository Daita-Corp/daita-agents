"""
Shared pytest fixtures for the Daita unit test suite.

All fixtures are self-contained — no API keys, running services, or
network access required.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from daita.agents.agent import Agent
from daita.core.tools import AgentTool
from daita.llm.mock import MockLLMProvider


# ---------------------------------------------------------------------------
# SequentialMockLLM
# ---------------------------------------------------------------------------

class SequentialMockLLM(MockLLMProvider):
    """
    MockLLMProvider that works through a fixed list of responses in order.

    Each call to _generate_impl returns the next item from the sequence.
    When the sequence is exhausted it returns a safe plain-text final answer.

    Useful for multi-turn tool-calling scenarios:
        response_sequence=[
            {"content": "...", "tool_calls": [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}]},
            "The answer is 3.",
        ]
    """

    def __init__(self, response_sequence: List[Any], **kwargs):
        kwargs.setdefault("delay", 0)
        super().__init__(**kwargs)
        self._response_sequence = list(response_sequence)
        self._call_index = 0

    async def _generate_impl(self, messages, tools=None, **kwargs):
        if self._call_index < len(self._response_sequence):
            response = self._response_sequence[self._call_index]
            self._call_index += 1
            self.call_history.append({
                "messages": messages,
                "tools": tools,
                "params": kwargs,
                "timestamp": 0,
            })
            return response
        # Sequence exhausted — plain final answer so the loop terminates.
        return "Final answer."


# ---------------------------------------------------------------------------
# LLM fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Zero-delay MockLLMProvider for fast unit tests."""
    return MockLLMProvider(delay=0)


@pytest.fixture
def mock_llm_with_tool_call():
    """
    SequentialMockLLM that simulates one tool call then a final answer.

    Iteration 1 → tool call for 'add' with a=3, b=4
    Iteration 2 → plain string final answer
    """
    return SequentialMockLLM(
        response_sequence=[
            {
                "content": "I will add the numbers.",
                "tool_calls": [
                    {
                        "id": "tc_001",
                        "name": "add",
                        "arguments": {"a": 3, "b": 4},
                    }
                ],
            },
            "The result of 3 + 4 is 7.",
        ]
    )


# ---------------------------------------------------------------------------
# Tool fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_tool():
    """AgentTool that adds two integers (async handler)."""

    async def _add(args: Dict[str, Any]) -> int:
        return args["a"] + args["b"]

    return AgentTool(
        name="add",
        description="Add two integers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
        handler=_add,
    )


@pytest.fixture
def async_tool():
    """AgentTool wrapping a trivially async handler."""

    async def _async_op(args: Dict[str, Any]) -> str:
        await asyncio.sleep(0)
        return f"async result: {args.get('value', 'none')}"

    return AgentTool(
        name="async_op",
        description="An async operation",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "Input value"},
            },
            "required": ["value"],
        },
        handler=_async_op,
    )


# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_agent(mock_llm):
    """Agent with mock LLM and no registered tools."""
    return Agent(name="TestAgent", llm_provider=mock_llm)


@pytest.fixture
def agent_with_tools(mock_llm, simple_tool):
    """Agent with mock LLM and the simple 'add' tool registered."""
    agent = Agent(name="ToolAgent", llm_provider=mock_llm)
    agent.register_tool(simple_tool)
    return agent
