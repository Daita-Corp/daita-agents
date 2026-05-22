"""Unit tests for runtime-scoped Agent retry behavior."""

import asyncio

import pytest

from daita.agents.agent import Agent
from daita.agents.runtime.retry import (
    mark_whole_run_retry_suppressed,
    run_model_turn_with_retry,
)
from daita.agents.runtime.state import RunState
from daita.agents.db.tools.query.facade import create_db_query_tools
from daita.config.base import AgentConfig, RetryPolicy, RetryStrategy
from daita.core.exceptions import LLMError, RateLimitError, TransientError
from daita.core.streaming import EventType, LLMChunk
from daita.core.tools import AgentTool
from daita.llm.mock import MockLLMProvider


def _retry_config(max_retries=1, max_delay=60.0):
    return AgentConfig(
        name="retry_agent",
        enable_retry=True,
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            strategy=RetryStrategy.FIXED,
            base_delay=0.1,
            max_delay=max_delay,
            jitter=False,
        ),
    )


@pytest.fixture(autouse=True)
def _skip_retry_sleeps(monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("daita.agents.runtime.retry.asyncio.sleep", no_sleep)


class FlakyModelLLM(MockLLMProvider):
    def __init__(self, failures, final_response="Recovered.", **kwargs):
        super().__init__(delay=0, **kwargs)
        self.failures = list(failures)
        self.final_response = final_response

    async def _generate_impl(self, messages, tools=None, **kwargs):
        self.call_history.append(
            {"messages": messages, "tools": tools, "params": kwargs}
        )
        if self.failures:
            raise self.failures.pop(0)
        return self.final_response


class ReplayProbeLLM(MockLLMProvider):
    async def _generate_impl(self, messages, tools=None, **kwargs):
        self.call_history.append(
            {"messages": messages, "tools": tools, "params": kwargs}
        )
        if any(message.get("role") == "tool" for message in messages):
            raise TransientError("provider stayed unavailable")
        return {
            "content": "Need the tool.",
            "tool_calls": [{"id": "tc1", "name": "side_effect", "arguments": {}}],
        }


class StreamingFlakyLLM(MockLLMProvider):
    def __init__(self, first_failure_after_event=False):
        super().__init__(delay=0)
        self.calls = 0
        self.first_failure_after_event = first_failure_after_event

    async def _stream_impl(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            if self.first_failure_after_event:
                yield LLMChunk(type="text", content="partial")
            raise TransientError("stream failed")
        yield LLMChunk(type="text", content="recovered")

    async def _generate_impl(self, messages, tools=None, **kwargs):
        return "fallback"


class ProviderStatusError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


class ProviderNamedError(Exception):
    pass


async def test_transient_model_failure_retries_same_turn_and_succeeds():
    llm = FlakyModelLLM([TransientError("temporary outage")], "Recovered.")
    agent = Agent(name="retry_agent", llm_provider=llm, config=_retry_config())

    result = await agent.run("hello", detailed=True)

    assert result["result"] == "Recovered."
    assert len(llm.call_history) == 2
    diagnostics = result["diagnostics"]
    assert diagnostics["retry_event_count"] == 2
    assert [event["decision"] for event in diagnostics["retry_events"]] == [
        "retry",
        "succeeded_after_retry",
    ]
    assert "timestamp" not in diagnostics["retry_events"][0]
    assert diagnostics["last_retry_decision"] == "succeeded_after_retry"


async def test_unknown_model_failure_does_not_retry_by_default():
    llm = FlakyModelLLM([ValueError("bug-shaped failure")])
    agent = Agent(name="retry_agent", llm_provider=llm, config=_retry_config())

    with pytest.raises(ValueError):
        await agent.run("hello", detailed=True)

    assert len(llm.call_history) == 1


async def test_retry_after_delay_is_recorded_and_capped(monkeypatch):
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("daita.agents.runtime.retry.asyncio.sleep", record_sleep)

    llm = FlakyModelLLM([RateLimitError(retry_after=12)], "Recovered.")
    agent = Agent(
        name="retry_agent",
        llm_provider=llm,
        config=_retry_config(max_retries=1, max_delay=5),
    )

    result = await agent.run("hello", detailed=True)

    assert result["result"] == "Recovered."
    assert delays == [5.0]
    retry_event = result["diagnostics"]["retry_events"][0]
    assert retry_event["retry_after_seconds"] == 12.0
    assert retry_event["delay_seconds"] == 5.0


async def test_cancelled_model_turn_is_not_retried():
    calls = 0
    run_state = RunState(agent_id="agent_cancel")
    policy = _retry_config(max_retries=2).retry_policy

    async def call():
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_model_turn_with_retry(call, policy=policy, run_state=run_state)

    assert calls == 1
    assert run_state.retry_events == []


async def test_whole_run_retry_is_suppressed_after_committed_tool_work():
    tool_calls = 0

    async def side_effect(_args):
        nonlocal tool_calls
        tool_calls += 1
        return {"ok": True}

    tool = AgentTool(
        name="side_effect",
        description="A tool whose work should not be replayed",
        parameters={"type": "object", "properties": {}},
        handler=side_effect,
    )
    llm = ReplayProbeLLM(delay=0)
    agent = Agent(
        name="retry_agent",
        llm_provider=llm,
        tools=[tool],
        config=_retry_config(max_retries=1),
    )

    with pytest.raises(TransientError) as exc_info:
        await agent.run("do the thing", detailed=True)

    assert tool_calls == 1
    diagnostics = getattr(exc_info.value, "_daita_run_diagnostics")
    assert diagnostics["whole_run_retry_suppressed"] is True
    assert any(
        event["scope"] == "whole_run" and event["decision"] == "suppressed"
        for event in diagnostics["retry_events"]
    )


def test_whole_run_retry_not_suppressed_when_all_committed_tools_are_replay_safe():
    run_state = RunState(agent_id="agent_replay_safe")
    run_state.record_tool_call(
        {
            "tool": "db_query",
            "result": {"rows": []},
            "replay_safe": True,
        }
    )
    error = TransientError("after read-only query")

    mark_whole_run_retry_suppressed(error, run_state)

    assert not getattr(error, "_daita_suppress_whole_run_retry", False)
    assert run_state.retry_events == []


def test_agent_tool_safety_defaults_are_conservative():
    async def handler(_args):
        return "ok"

    tool = AgentTool(
        name="custom", description="Custom", parameters={}, handler=handler
    )

    assert tool.retry_safe is False
    assert tool.replay_safe is False
    assert tool.idempotent is False
    assert tool.side_effecting is True


def test_db_query_tools_opt_into_read_only_replay_safety():
    async def handler(_args):
        return {}

    class Plugin:
        read_only = True
        sql_dialect = "postgresql"
        _tool_query = handler
        _tool_count = handler
        _tool_sample = handler

    tools = create_db_query_tools(Plugin(), {"database_type": "postgresql"})
    by_name = {tool.name: tool for tool in tools}

    assert by_name["db_query"].replay_safe is True
    assert by_name["db_query"].retry_safe is True
    assert by_name["db_query"].side_effecting is False
    assert "db_execute" not in by_name


def test_provider_error_helper_preserves_retry_context():
    llm = MockLLMProvider(delay=0)
    wrapped = llm._provider_error(
        "Mock generation failed", ProviderStatusError(429, retry_after=12)
    )

    assert isinstance(wrapped, LLMError)
    assert wrapped.retry_hint == "transient"
    assert wrapped.context["status_code"] == 429
    assert wrapped.context["retry_after"] == 12


@pytest.mark.parametrize(
    ("status_code", "retry_hint"),
    [
        (400, "permanent"),
        (401, "permanent"),
        (404, "permanent"),
        (408, "transient"),
        (429, "transient"),
        (500, "transient"),
        (503, "transient"),
    ],
)
def test_provider_error_helper_classifies_common_status_codes(status_code, retry_hint):
    llm = MockLLMProvider(delay=0)

    wrapped = llm._provider_error(
        "Mock generation failed", ProviderStatusError(status_code)
    )

    assert wrapped.retry_hint == retry_hint


def test_provider_error_helper_classifies_common_exception_names():
    llm = MockLLMProvider(delay=0)
    transient_type = type("APITimeoutError", (ProviderNamedError,), {})
    permanent_type = type("AuthenticationError", (ProviderNamedError,), {})

    transient = llm._provider_error("Mock generation failed", transient_type())
    permanent = llm._provider_error("Mock generation failed", permanent_type())

    assert transient.retry_hint == "transient"
    assert permanent.retry_hint == "permanent"


async def test_streaming_retries_before_first_emitted_event():
    llm = StreamingFlakyLLM(first_failure_after_event=False)
    agent = Agent(name="retry_agent", llm_provider=llm, config=_retry_config())

    events = []
    async for event in agent.stream("hello"):
        events.append(event)

    assert llm.calls == 2
    assert any(event.type == EventType.COMPLETE for event in events)
    assert any(
        event.type == EventType.THINKING and event.content == "recovered"
        for event in events
    )


async def test_streaming_does_not_retry_after_emitted_event():
    llm = StreamingFlakyLLM(first_failure_after_event=True)
    agent = Agent(name="retry_agent", llm_provider=llm, config=_retry_config())

    with pytest.raises(TransientError):
        async for _event in agent.stream("hello"):
            pass

    assert llm.calls == 1
