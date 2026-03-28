"""
Tests for Agent.stream() — the async generator interface.

stream() uses _stream_impl (not _generate_impl), so these tests use
streaming-aware LLM mocks that yield LLMChunk objects.

Tests:
- Yields AgentEvent objects, terminates with COMPLETE
- ITERATION events emitted each loop cycle
- TOOL_CALL / TOOL_RESULT events emitted around tool execution
- Background task exception propagates from the generator
- Run timeout raises AgentError via the generator
- Early consumer break cleans up the background task
"""

import asyncio
import pytest

from daita.agents.agent import Agent
from daita.core.streaming import AgentEvent, EventType, LLMChunk
from daita.core.tools import AgentTool
from daita.core.exceptions import AgentError
from daita.llm.mock import MockLLMProvider

# ---------------------------------------------------------------------------
# Streaming-aware mock LLMs
# ---------------------------------------------------------------------------


class StreamingSequentialLLM(MockLLMProvider):
    """
    LLM mock for streaming tests. Works through a fixed response sequence where
    each item is either:
      - A plain string → yields text chunks
      - A dict with "tool_calls" → yields tool_call_complete chunks, then optional text
    """

    def __init__(self, response_sequence):
        super().__init__(delay=0)
        self._response_sequence = list(response_sequence)
        self._call_index = 0

    async def _stream_impl(self, messages, tools=None, **kwargs):
        if self._call_index < len(self._response_sequence):
            response = self._response_sequence[self._call_index]
            self._call_index += 1
        else:
            response = "Final answer."

        if isinstance(response, dict) and "tool_calls" in response:
            for tc in response.get("tool_calls") or []:
                yield LLMChunk(
                    type="tool_call_complete",
                    tool_name=tc["name"],
                    tool_args=tc["arguments"],
                    tool_call_id=tc.get("id", "tc_001"),
                )
            # Optionally yield text content
            text = response.get("content", "")
            if text:
                yield LLMChunk(type="text", content=text)
        else:
            text = response if isinstance(response, str) else str(response)
            yield LLMChunk(type="text", content=text)

    async def _generate_impl(self, messages, tools=None, **kwargs):
        # Used only in non-streaming path (not by stream())
        if self._call_index < len(self._response_sequence):
            response = self._response_sequence[self._call_index]
            self._call_index += 1
            return response
        return "Final answer."


class RaisingStreamLLM(MockLLMProvider):
    """LLM that raises from _stream_impl (the path used by Agent.stream())."""

    def __init__(self, exc):
        super().__init__(delay=0)
        self._exc = exc

    async def _stream_impl(self, messages, tools=None, **kwargs):
        raise self._exc
        # make this an async generator
        if False:
            yield  # noqa: unreachable — makes this an async generator

    async def _generate_impl(self, messages, tools=None, **kwargs):
        return "fallback"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool(name="add"):
    async def handler(args):
        return {"sum": args.get("a", 0) + args.get("b", 0)}

    return AgentTool(
        name=name,
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Basic streaming — yields AgentEvent, terminates with COMPLETE
# ---------------------------------------------------------------------------


class TestStreamBasic:
    async def test_stream_yields_agent_events(self):
        """stream() must yield AgentEvent instances, not raw dicts or strings."""
        llm = StreamingSequentialLLM(["The answer is 42."])
        agent = Agent(name="StreamAgent", llm_provider=llm)

        events = []
        async for event in agent.stream("What is 6 times 7?"):
            events.append(event)

        assert len(events) > 0
        for event in events:
            assert isinstance(
                event, AgentEvent
            ), f"Expected AgentEvent, got {type(event)}"

    async def test_stream_yields_complete_event(self):
        """COMPLETE event must be emitted at the end of a successful run."""
        llm = StreamingSequentialLLM(["Done."])
        agent = Agent(name="StreamAgent", llm_provider=llm)

        event_types = []
        async for event in agent.stream("hello"):
            event_types.append(event.type)

        assert EventType.COMPLETE in event_types

    async def test_stream_complete_event_has_final_result(self):
        """COMPLETE event should carry a non-empty final_result."""
        llm = StreamingSequentialLLM(["The result is 7."])
        agent = Agent(name="StreamAgent", llm_provider=llm)

        complete_events = []
        async for event in agent.stream("run"):
            if event.type == EventType.COMPLETE:
                complete_events.append(event)

        assert len(complete_events) == 1
        assert complete_events[0].final_result is not None

    async def test_stream_iteration_events_emitted(self):
        """ITERATION event must be emitted at the start of each loop cycle."""
        llm = StreamingSequentialLLM(["Answer."])
        agent = Agent(name="StreamAgent", llm_provider=llm)

        iteration_events = []
        async for event in agent.stream("go"):
            if event.type == EventType.ITERATION:
                iteration_events.append(event)

        assert len(iteration_events) >= 1
        assert iteration_events[0].iteration == 1

    async def test_stream_terminates_after_complete(self):
        """Generator must stop after COMPLETE — no infinite iteration."""
        llm = StreamingSequentialLLM(["Final answer."])
        agent = Agent(name="StreamAgent", llm_provider=llm)

        count = 0
        async for _event in agent.stream("prompt"):
            count += 1
            assert count < 100, "Stream did not terminate"

    async def test_stream_thinking_events_contain_text(self):
        """THINKING events carry the streamed text content."""
        llm = StreamingSequentialLLM(["Hello world"])
        agent = Agent(name="StreamAgent", llm_provider=llm)

        thinking_events = []
        async for event in agent.stream("hi"):
            if event.type == EventType.THINKING:
                thinking_events.append(event)

        assert len(thinking_events) > 0
        assert all(event.content is not None for event in thinking_events)


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------


class TestStreamToolEvents:
    async def test_tool_call_event_emitted(self):
        """TOOL_CALL event must be emitted before a tool is executed."""
        tool = make_tool("add")
        llm = StreamingSequentialLLM(
            [
                {
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 3, "b": 4}}
                    ]
                },
                "The sum is 7.",
            ]
        )
        agent = Agent(name="ToolStreamAgent", llm_provider=llm, tools=[tool])

        tool_call_events = []
        async for event in agent.stream("What is 3 + 4?"):
            if event.type == EventType.TOOL_CALL:
                tool_call_events.append(event)

        assert len(tool_call_events) == 1
        assert tool_call_events[0].tool_name == "add"

    async def test_tool_result_event_emitted(self):
        """TOOL_RESULT event must be emitted after a tool completes."""
        tool = make_tool("add")
        llm = StreamingSequentialLLM(
            [
                {
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 2, "b": 5}}
                    ]
                },
                "The answer is 7.",
            ]
        )
        agent = Agent(name="ToolStreamAgent", llm_provider=llm, tools=[tool])

        result_events = []
        async for event in agent.stream("add 2 and 5"):
            if event.type == EventType.TOOL_RESULT:
                result_events.append(event)

        assert len(result_events) == 1
        assert result_events[0].tool_name == "add"

    async def test_tool_call_before_tool_result(self):
        """TOOL_CALL must come before TOOL_RESULT for the same tool."""
        tool = make_tool("add")
        llm = StreamingSequentialLLM(
            [
                {
                    "tool_calls": [
                        {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 1}}
                    ]
                },
                "Done.",
            ]
        )
        agent = Agent(name="ToolStreamAgent", llm_provider=llm, tools=[tool])

        event_types = []
        async for event in agent.stream("add"):
            event_types.append(event.type)

        call_idx = next(
            (i for i, t in enumerate(event_types) if t == EventType.TOOL_CALL), None
        )
        result_idx = next(
            (i for i, t in enumerate(event_types) if t == EventType.TOOL_RESULT), None
        )
        assert call_idx is not None
        assert result_idx is not None
        assert call_idx < result_idx


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


class TestStreamExceptionPropagation:
    async def test_agent_error_propagates_from_stream(self):
        """AgentError from _stream_impl propagates out of the generator."""
        agent = Agent(
            name="ErrorAgent",
            llm_provider=RaisingStreamLLM(AgentError("LLM refused the request")),
        )

        with pytest.raises(AgentError, match="LLM refused"):
            async for _event in agent.stream("trigger error"):
                pass

    async def test_generic_exception_propagates_from_stream(self):
        """Generic exceptions from _stream_impl also propagate."""
        agent = Agent(
            name="CrashAgent",
            llm_provider=RaisingStreamLLM(RuntimeError("unexpected crash")),
        )

        with pytest.raises(RuntimeError, match="unexpected crash"):
            async for _event in agent.stream("crash"):
                pass

    async def test_timeout_raises_agent_error_in_stream(self):
        """When timeout_seconds is exceeded, AgentError is raised via stream()."""

        async def slow_handler(args):
            await asyncio.sleep(10)
            return "done"

        slow_tool = AgentTool(
            name="slow",
            description="slow",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=slow_handler,
        )
        llm = StreamingSequentialLLM(
            [
                {"tool_calls": [{"id": "c1", "name": "slow", "arguments": {}}]},
                "done",
            ]
        )
        agent = Agent(name="SlowStreamAgent", llm_provider=llm, tools=[slow_tool])

        with pytest.raises(AgentError, match="timed out"):
            async for _event in agent.stream("go slow", timeout_seconds=0.05):
                pass


# ---------------------------------------------------------------------------
# Early consumer break — task cancellation
# ---------------------------------------------------------------------------


class TestStreamEarlyBreak:
    async def test_early_break_does_not_hang(self):
        """Breaking out of stream() early must not hang — background task is cancelled."""
        llm = StreamingSequentialLLM(["Part 1", "Part 2", "Part 3"])
        agent = Agent(name="BreakAgent", llm_provider=llm)

        async for _event in agent.stream("go"):
            break  # immediate break

        # Give the event loop a moment to finalize cancellation
        await asyncio.sleep(0)
        # If we reach here without hanging, the test passes

    async def test_early_break_returns_first_event(self):
        """After breaking, the first event received is a valid AgentEvent."""
        llm = StreamingSequentialLLM(["Answer."])
        agent = Agent(name="BreakAgent", llm_provider=llm)

        first_event = None
        async for event in agent.stream("prompt"):
            first_event = event
            break

        assert first_event is not None
        assert isinstance(first_event, AgentEvent)
