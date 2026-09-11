"""Configured-route live recovery; generation is real and MCP effects are simulated."""

import json
import os

import httpx
import pytest

from daita.llm.profiles import reviewed_model_profile
from daita.loop.models import LoopExitKind, validate_completed_transcript
from daita.security import SecretReference
from daita.storage.sqlite_codecs.transcripts import encode_loop_exit, encode_message
from tests.support.mcp_routine_harness import (
    AUTHORIZATION,
    COST_ENV,
    limits,
    report_path,
)
from tests.support.retry_routes import (
    ConfiguredActionFixture,
    CountFaultTransport,
    record_configured_route,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(AUTHORIZATION) != "1",
        reason=(
            f"set {AUTHORIZATION}=1 only after authorizing up to twelve live "
            f"runs per model/repetition, each capped by {COST_ENV}; MCP is simulated"
        ),
    ),
]


@pytest.fixture
def evidence_path(request, record_property):
    path = report_path(request.node.nodeid).resolve()
    path = path.with_name(path.name.replace("phase-f-", "retry-route-", 1))
    record_property("retry_route_evidence", str(path))
    return path


async def test_live_configured_route_recovers_after_committed_action(
    tmp_path, monkeypatch, model_id, repetition, evidence_path, request
):
    if os.environ.get(AUTHORIZATION) != "1":
        raise ValueError("Explicit live-test authorization is required")
    if not model_id.startswith("openai:"):
        pytest.skip("Configured-route HTTP fault injection currently covers OpenAI")
    profile = reviewed_model_profile(model_id)
    assert profile is not None
    key_name = (
        "DAITA_PHASE_F_LIVE_OPENAI_API_KEY"
        if os.environ.get("DAITA_PHASE_F_LIVE_OPENAI_API_KEY")
        else "DAITA_PHASE_F_LIVE_LLM_API_KEY"
    )
    if not os.environ.get(key_name):
        raise ValueError("The authorized Phase F OpenAI credential is unavailable")
    fixture = ConfiguredActionFixture(
        tmp_path, profile, limits(), SecretReference("env", key_name)
    )
    transport = CountFaultTransport(fixture, httpx.AsyncHTTPTransport())
    recordings, clients = record_configured_route(monkeypatch, transport)
    result, transcript = None, None
    await fixture.start()
    try:
        result = await fixture.agent.run(
            'Send one notification to destination "fixed-room" with exactly '
            'the content "Status". Do not perform research. Then briefly report '
            "the invocation evidence."
        )
        transcript = await fixture.agent._embedded._store.load(result.run_id)
        assert result.kind is LoopExitKind.COMPLETED, result.reason
        validate_completed_transcript(transcript, result)
        assert fixture.server.calls == [
            ("notify", {"destination": "fixed-room", "content": "Status"})
        ]
        assert len(await fixture.receipts()) == len(fixture.approvals) == 1
        active = [item for item in recordings if item.requests]
        assert len(active) == 1
        recording = active[0]
        assert len(recording.requests) == len(recording.responses) + 1
        assert len(recording.responses) == result.steps
        assert len({item.deadline for item in recording.requests}) == 1
        assert sum(call["injected"] for call in transport.calls) == 1
        assert len(clients) == 1 and clients[0].max_retries == 0
        assert result.usage.total_tokens == sum(
            response.usage.total_tokens for response in recording.responses
        )
    finally:
        await fixture.agent.close()
        report = {
            "case_id": request.node.nodeid,
            "model_id": model_id,
            "composition": "AgentConfig.model_route",
            "fault": "one local synthetic count HTTP 503 after a committed simulated action",
            "result": encode_loop_exit(result) if result is not None else None,
            "transcript": (
                [encode_message(message) for message in transcript.messages]
                if transcript is not None
                else None
            ),
            "physical_attempts": [
                timing for item in recordings for timing in item.timings
            ],
            "http_calls": transport.calls,
            "mcp_calls": fixture.server.calls,
            "approvals": len(fixture.approvals),
            "transport_closed": transport.closed,
            "clients_closed": all(client.is_closed() for client in clients),
        }
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    assert transport.closed and all(client.is_closed() for client in clients)
