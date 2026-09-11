"""Public Agent composition with actual SDK, router and committed MCP effects."""

import json
from dataclasses import replace

import httpx
import pytest

from daita.llm.profiles import reviewed_model_profile
from daita.loop.models import LoopExitKind
from daita.security import SecretReference
from tests.support.mcp_routine_harness import limits
from tests.support.retry_routes import (
    ConfiguredActionFixture,
    CountFaultTransport,
    record_configured_route,
)


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
        assert len({item.deadline for item in recording.requests}) == 1
        assert replace(recording.requests[-1], attempt_deadline=None) == replace(
            recording.requests[-2], attempt_deadline=None
        )
        assert [item.max_total_tokens for item in recording.requests] == [
            30000,
            29890,
            29780,
            29780,
        ]
        assert [call["phase"] for call in transport.calls].count("count") == 4
        assert sum(call["injected"] for call in transport.calls) == 1
        assert len(clients) == 1 and clients[0].max_retries == 0
    finally:
        await fixture.agent.close()
    assert transport.closed and clients[0].is_closed()
