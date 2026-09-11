"""Configured-route acceptance fixtures; all MCP actions remain simulated."""

from dataclasses import replace
from time import perf_counter
from typing import Any

import httpx
import openai
from live.benchmarks._support import RecordingProvider
from test_mcp_actions import ActionFixture

from daita.config import AgentConfig
from daita.llm import factory
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy


class ConfiguredActionFixture(ActionFixture):
    def __init__(self, root, profile, limits, secret_reference):
        super().__init__(root)
        self.config = AgentConfig(
            model_route=ModelRoute(
                candidates=(
                    ModelRouteCandidate(
                        profile.id,
                        replace(profile, max_output_tokens=2048),
                        secret_reference=secret_reference,
                    ),
                ),
                retry_policy=RetryPolicy(
                    max_attempts_per_candidate=2, backoff_seconds=0
                ),
            ),
            limits=limits,
        )

    def kwargs(self) -> dict[str, Any]:
        arguments = super().kwargs()
        arguments.pop("model")
        arguments.pop("model_profile")
        arguments["config"] = self.config
        return arguments


class CountFaultTransport(httpx.AsyncBaseTransport):
    """Fail one admission call after a completed action, never a generation call."""

    def __init__(self, fixture, delegate):
        self.fixture = fixture
        self.delegate = delegate
        self.injected = False
        self.closed = False
        self.calls = []

    async def handle_async_request(self, request):
        started = perf_counter()
        count = request.url.path.endswith("/input_tokens")
        inject = bool(count and self.fixture.server.calls and not self.injected)
        if inject:
            self.injected = True
            response = httpx.Response(
                503,
                json={"error": {"message": "simulated admission outage"}},
            )
        else:
            response = await self.delegate.handle_async_request(request)
        self.calls.append(
            {
                "phase": "count" if count else "generation",
                "injected": inject,
                "status": response.status_code,
                "seconds_to_headers": perf_counter() - started,
            }
        )
        return response

    async def aclose(self):
        self.closed = True
        await self.delegate.aclose()


def record_configured_route(monkeypatch, transport):
    """Observe physical delegates inside the ordinary factory-owned router."""
    recordings, clients = [], []
    original_provider = factory.create_llm_provider
    original_sdk = openai.AsyncOpenAI

    def provider(*args, **kwargs):
        recording = RecordingProvider(original_provider(*args, **kwargs))
        recordings.append(recording)
        return recording

    def sdk(**kwargs):
        client = original_sdk(
            **kwargs, http_client=httpx.AsyncClient(transport=transport)
        )
        clients.append(client)
        return client

    monkeypatch.setattr(factory, "create_llm_provider", provider)
    monkeypatch.setattr(openai, "AsyncOpenAI", sdk)
    return recordings, clients
