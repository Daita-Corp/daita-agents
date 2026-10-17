"""Public Agent composition with actual SDK, router and committed MCP effects."""

import asyncio
import json
from dataclasses import replace
from typing import Any, cast

import httpx
import openai
import pytest

from daita import Agent
from daita.config import AgentConfig
from daita.llm.models import ModelCallPolicy
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.loop.models import LoopExitKind
from daita.observation import AgentEvent, AgentEventKind
from daita.security import SecretReference
from tests.support.mcp_routine_harness import limits
from tests.support.retry_routes import (
    ConfiguredActionFixture,
    CountFaultTransport,
    record_configured_route,
)
from tests.support.workspace import workspace_for


@pytest.mark.parametrize("stream", [False, True])
async def test_configured_route_retry_preserves_committed_action(
    tmp_path, monkeypatch, stream
):
    monkeypatch.setenv("DAITA_TEST_RETRY_KEY", "offline-placeholder")
    profile = reviewed_model_profile("openai:gpt-5.6-terra")
    assert profile is not None
    fixture = ConfiguredActionFixture(
        tmp_path,
        replace(profile, supports_streaming=stream),
        limits(),
        SecretReference("env", "DAITA_TEST_RETRY_KEY"),
    )
    generations = []

    def respond(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        generations.append(json.loads(request.content))
        index = len(generations)
        if index < 3:
            name = "toolbox_load" if index == 1 else fixture.tool.local_name
            arguments = (
                {"tool_names": [fixture.tool.local_name]}
                if index == 1
                else {"destination": "fixed-room", "content": "Status"}
            )
            output: list[dict[str, object]] = [
                {
                    "type": "function_call",
                    "id": f"fc-{index}",
                    "call_id": f"call-{index}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                    "status": "completed",
                }
            ]
        else:
            assert index == 3, "completed generation or tool action was replayed"
            output = [
                {
                    "type": "message",
                    "id": "message-3",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Sent.", "annotations": []}
                    ],
                }
            ]
        response = {
            "id": f"resp-{index}",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-terra",
            "service_tier": "default",
            "output": output,
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 110,
            },
        }
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    "data: "
                    + json.dumps({"type": "response.completed", "response": response})
                    + "\n\n"
                ),
            )
        return httpx.Response(200, json=response)

    transport = CountFaultTransport(fixture, httpx.MockTransport(respond))
    recordings, clients = record_configured_route(monkeypatch, transport)
    await fixture.start()
    try:
        result = await fixture.agent.run("Send Status to fixed-room once.")
        assert result.kind is LoopExitKind.COMPLETED, (
            result.provider_failure.code if result.provider_failure else result.reason
        )
        assert result.steps == 3
        assert result.usage.total_tokens == 330
        assert len(fixture.server.calls) == len(await fixture.receipts()) == 1
        assert len(fixture.approvals) == 1
        recording = next(item for item in recordings if item.requests)
        assert len(recording.requests) == 4
        assert len(recording.responses) == 3
        assert all(item.deadline is not None for item in recording.requests)
        assert recording.requests[-1].deadline == recording.requests[-2].deadline
        assert replace(recording.requests[-1], attempt_deadline=None) == replace(
            recording.requests[-2], attempt_deadline=None
        )
        initial_token_allowance = fixture.config.limits.max_total_tokens
        assert [item.max_total_tokens for item in recording.requests] == [
            initial_token_allowance,
            initial_token_allowance - 110,
            initial_token_allowance - 220,
            initial_token_allowance - 220,
        ]
        assert [call["phase"] for call in transport.calls].count("count") == 4
        assert sum(call["injected"] for call in transport.calls) == 1
        assert len(clients) == 1 and clients[0].max_retries == 0
    finally:
        await fixture.agent.close()
    assert transport.closed and clients[0].is_closed()


class _SilentBody(httpx.AsyncByteStream):
    """An accepted stream that never sends an event."""

    async def __aiter__(self):
        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    async def aclose(self):
        return None


@pytest.mark.parametrize("stream", [False, True])
async def test_injected_provider_replays_transient_failures_before_output(
    tmp_path, stream
):
    """A provider passed to the agent gets the router's retry and diagnostics."""

    profile = reviewed_model_profile("openai:gpt-5.6-terra")
    assert profile is not None
    # Each run fails once before output, then answers. A stall needs streaming.
    faults = ["unavailable", "transport_drop", *(["stall"] if stream else [])]
    script = [item for fault in faults for item in (fault, "answer")]
    generations: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        step = script[len(generations)]
        generations.append(step)
        if step == "unavailable":
            return httpx.Response(503, json={"error": {"message": "offline outage"}})
        if step == "transport_drop":
            raise httpx.RemoteProtocolError("offline disconnect", request=request)
        if step == "stall":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_SilentBody()
            )
        response = {
            "id": f"resp-{len(generations)}",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-terra",
            "service_tier": "default",
            "output": [
                {
                    "type": "message",
                    "id": f"message-{len(generations)}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Ready.", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 110,
            },
        }
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="data: "
                + json.dumps({"type": "response.completed", "response": response})
                + "\n\n",
            )
        return httpx.Response(200, json=response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    sdk = openai.AsyncOpenAI(api_key="offline", http_client=client)
    provider = OpenAIResponsesProvider("gpt-5.6-terra", client=cast(Any, sdk))
    events: list[AgentEvent] = []
    agent = await Agent.create(
        "injected-retry",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=replace(profile, supports_streaming=stream),
        config=AgentConfig(
            model_call_policy=ModelCallPolicy(first_progress_timeout_seconds=0.5)
        ),
        observer=events.append,
    )
    try:
        for _ in faults:
            result = await agent.run("Is the release ready?")
            assert result.kind is LoopExitKind.COMPLETED, result.reason
            assert result.final_text == "Ready."
    finally:
        await agent.close()
        # The injected provider and its SDK client remain caller-owned.
        assert not client.is_closed
        await provider.close()
        await sdk.close()
        await client.aclose()

    assert generations == script
    steps: list[dict[str, Any]] = [
        event.data.to_dict()
        for event in events
        if event.kind is AgentEventKind.MODEL_COMPLETED
    ]
    outcomes = {
        "unavailable": "provider_unavailable",
        "transport_drop": "provider_unavailable",
        "stall": "stalled",
    }
    assert [[item["outcome"] for item in step["attempts"]] for step in steps] == [
        [outcomes[fault], "completed"] for fault in faults
    ]
    assert all(
        [item["retried"] for item in step["attempts"]] == [True, False]
        for step in steps
    )
    # An HTTP rejection proves zero usage; a drop or stall stays unreported.
    assert [step["usage_status"] for step in steps] == [
        "reported" if fault == "unavailable" else "partial" for fault in faults
    ]
    assert all(step["input_tokens"] == 100 for step in steps)
