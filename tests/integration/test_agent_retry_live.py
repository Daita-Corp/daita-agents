"""
Live integration tests for Agent retry/runtime behavior across LLM providers.

These tests are opt-in and use real provider calls after deterministic injected
failures. They are intended to verify that retry/replay boundaries still work
when the successful attempt goes through the real OpenAI, Claude, Grok, or
Gemini adapter.

Run:
    DAITA_RUN_LIVE_LLM=1 pytest tests/integration/test_agent_retry_live.py \
        -m "requires_llm and integration" -v
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.config.base import AgentConfig, RetryPolicy, RetryStrategy
from daita.core.exceptions import AgentError, TransientError
from daita.core.streaming import EventType
from daita.core.tools import AgentTool
from daita.llm.factory import create_llm_provider

load_dotenv(Path.cwd() / ".env")


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    provider: str
    model_env: str
    default_model: str
    api_key_envs: tuple[str, ...]


PROVIDERS = [
    ProviderSpec(
        "openai", "openai", "OPENAI_TEST_MODEL", "gpt-5.4-mini", ("OPENAI_API_KEY",)
    ),
    ProviderSpec(
        "claude",
        "anthropic",
        "ANTHROPIC_TEST_MODEL",
        "claude-haiku-4-5",
        ("ANTHROPIC_API_KEY",),
    ),
    ProviderSpec(
        "grok", "grok", "GROK_TEST_MODEL", "grok-4.20", ("XAI_API_KEY", "GROK_API_KEY")
    ),
    ProviderSpec(
        "gemini",
        "gemini",
        "GEMINI_TEST_MODEL",
        "gemini-2.5-flash-lite",
        ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    ),
]


def _require_live_llm_enabled() -> None:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live retry integration tests")


def _require_env(*names: str) -> str:
    _require_live_llm_enabled()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    pytest.skip(f"Missing environment variable: one of {', '.join(names)}")


def _retry_config(max_retries: int = 1) -> AgentConfig:
    return AgentConfig(
        name="live_retry_agent",
        enable_retry=True,
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            strategy=RetryStrategy.FIXED,
            base_delay=0.1,
            jitter=False,
        ),
    )


def _live_provider(spec: ProviderSpec):
    return create_llm_provider(
        spec.provider,
        model=os.environ.get(spec.model_env, spec.default_model),
        api_key=_require_env(*spec.api_key_envs),
        temperature=0,
        max_tokens=32,
    )


class ScriptedLiveLLM:
    """Delegates to a live provider after consuming scripted failures/responses."""

    def __init__(self, live_llm, script: list[Any], *, delegate_tools: bool = True):
        self.live_llm = live_llm
        self.script = list(script)
        self.delegate_tools = delegate_tools
        self.generate_calls = 0
        self.live_calls = 0

    def set_agent_id(self, agent_id: str):
        return self.live_llm.set_agent_id(agent_id)

    def get_token_stats(self):
        return self.live_llm.get_token_stats()

    async def generate(self, *args, **kwargs):
        self.generate_calls += 1
        if self.script:
            next_item = self.script.pop(0)
            if isinstance(next_item, BaseException):
                raise next_item
            return next_item
        self.live_calls += 1
        if not self.delegate_tools:
            kwargs["tools"] = None
        return await self.live_llm.generate(*args, **kwargs)

    @property
    def info(self):
        return self.live_llm.info


def _replay_probe_tool(*, replay_safe: bool) -> AgentTool:
    async def handler(_args):
        return {"ok": True, "source": "replay_probe"}

    return AgentTool(
        name="replay_probe",
        description="Replay probe tool used by live retry integration tests.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
        replay_safe=replay_safe,
        retry_safe=replay_safe,
        side_effecting=not replay_safe,
    )


def _tool_call_response() -> dict[str, Any]:
    return {
        "content": "Calling replay probe.",
        "tool_calls": [
            {"id": "tc_live_retry", "name": "replay_probe", "arguments": {}}
        ],
    }


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.parametrize("spec", PROVIDERS, ids=[spec.id for spec in PROVIDERS])
async def test_live_provider_recovers_from_transient_model_turn_failure(spec):
    live_llm = _live_provider(spec)
    llm = ScriptedLiveLLM(live_llm, [TransientError("injected transient failure")])
    agent = Agent(name=f"LiveRetry_{spec.id}", llm_provider=llm, config=_retry_config())

    result = await agent.run(
        "Reply with exactly this token and nothing else: retry-ok",
        detailed=True,
        timeout_seconds=45,
    )

    assert result["result"].strip()
    assert llm.live_calls == 1
    events = result["diagnostics"]["retry_events"]
    assert [event["decision"] for event in events] == [
        "retry",
        "succeeded_after_retry",
    ]


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.parametrize("spec", PROVIDERS, ids=[spec.id for spec in PROVIDERS])
async def test_live_provider_whole_run_retries_after_replay_safe_tool_work(spec):
    live_llm = _live_provider(spec)
    llm = ScriptedLiveLLM(
        live_llm,
        [_tool_call_response()],
        delegate_tools=False,
    )
    agent = Agent(
        name=f"LiveReplaySafe_{spec.id}",
        llm_provider=llm,
        tools=[_replay_probe_tool(replay_safe=True)],
        config=_retry_config(max_retries=1),
        prompt="When asked for the final answer, do not call tools.",
    )

    result = await agent.run(
        "Reply with exactly this token and nothing else: replay-ok",
        detailed=True,
        max_iterations=1,
        timeout_seconds=45,
    )

    assert result["result"].strip()
    assert result.get("retry_attempt") == 2
    assert llm.live_calls == 1


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.parametrize("spec", PROVIDERS, ids=[spec.id for spec in PROVIDERS])
async def test_live_provider_whole_run_does_not_replay_unsafe_tool_work(spec):
    live_llm = _live_provider(spec)
    llm = ScriptedLiveLLM(live_llm, [_tool_call_response()])
    agent = Agent(
        name=f"LiveReplayUnsafe_{spec.id}",
        llm_provider=llm,
        tools=[_replay_probe_tool(replay_safe=False)],
        config=_retry_config(max_retries=1),
    )

    with pytest.raises(AgentError) as exc_info:
        await agent.run(
            "Use the replay probe once, then answer.",
            detailed=True,
            max_iterations=1,
            timeout_seconds=45,
        )

    assert llm.live_calls == 0
    diagnostics = getattr(exc_info.value, "_daita_run_diagnostics")
    assert diagnostics["whole_run_retry_suppressed"] is True
    assert diagnostics["retry_events"][-1]["decision"] == "suppressed"


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.parametrize("spec", PROVIDERS, ids=[spec.id for spec in PROVIDERS])
async def test_live_provider_stream_retries_before_first_emitted_event(spec):
    live_llm = _live_provider(spec)
    llm = ScriptedLiveLLM(live_llm, [TransientError("injected stream failure")])
    agent = Agent(
        name=f"LiveStreamRetry_{spec.id}", llm_provider=llm, config=_retry_config()
    )

    events = []
    async for event in agent.stream(
        "Reply with exactly this token and nothing else: stream-ok",
        timeout_seconds=45,
    ):
        events.append(event)

    assert llm.live_calls == 1
    assert any(event.type == EventType.COMPLETE for event in events)
