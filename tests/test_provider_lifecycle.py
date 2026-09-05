from __future__ import annotations

import asyncio
import json
import importlib
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from dataclasses import replace
from decimal import Decimal
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest
from google import genai

import daita.hosting.embedded as embedded_module
import daita.llm.factory as factory_module
from daita.llm._lifecycle import closing_stream
from daita import Agent, AgentConfig
from daita.llm.models import FinishReason, ModelRequest, ModelResponse, ToolCall
from daita.llm.models import ModelStreamCompleted, ModelTextDelta, ModelStreamEvent
from daita.llm.errors import ModelProviderError
from daita.llm.models import CanonicalMessage, MessageRole, TextBlock
from daita.llm.factory import create_model_route_provider
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from daita.security import EmptySecretProvider, SecretReference


async def test_canonical_stream_cleanup_preserves_the_iteration_context():
    active = ContextVar("test_stream_context", default=False)

    async def stream():
        token = active.set(True)
        try:
            yield ModelTextDelta("hello")
        finally:
            active.reset(token)

    async with closing_stream(stream()) as events:
        await anext(events)
        assert active.get()
    assert not active.get()


@pytest.mark.parametrize(
    "module_name,test_name",
    (
        (
            "live.test_stage_a_kernel_live",
            "test_live_tool_round_trip_has_stable_context_and_durable_completion",
        ),
        (
            "live.test_stage0_sensitivity_live",
            "test_live_model_route_admits_internal_scope_and_blocks_public_only_route",
        ),
        (
            "test_learning_evaluation_live",
            "test_live_fixture_baseline_teaching_and_learned_report",
        ),
    ),
)
@pytest.mark.parametrize("fail_open", (True, False))
async def test_live_fixtures_release_provider_and_open_agent_on_failure(
    tmp_path, monkeypatch, module_name, test_name, fail_open
):
    module = importlib.import_module(module_name)
    provider = _ManagedProvider()
    monkeypatch.setenv(module._MODEL_ID, provider.provider_id)
    monkeypatch.setenv(module._MODEL_KEY, "offline-not-a-real-key")
    monkeypatch.setenv("DAITA_FIXTURE_POSTGRES_PASSWORD", "offline")
    if hasattr(module, "_live_provider"):
        profile = reviewed_model_profile(provider.provider_id)
        monkeypatch.setattr(module, "_live_provider", lambda **_: (profile, provider))
    else:
        monkeypatch.setattr(module, "create_llm_provider", lambda *_a, **_k: provider)

    class BrokenAgent:
        close_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            await self.close()

        async def close(self):
            self.close_calls += 1

        async def attach(self, *_a, **_k):
            raise RuntimeError("offline fixture failure")

        async def attach_postgresql(self, *_a, **_k):
            raise RuntimeError("offline fixture failure")

    agent = BrokenAgent()

    async def create(*_a, **_k):
        if fail_open:
            raise RuntimeError("offline fixture failure")
        return agent

    monkeypatch.setattr(module.Agent, "create", create)
    with pytest.raises(RuntimeError, match="offline fixture failure"):
        await getattr(module, test_name)(tmp_path)
    assert provider.close_calls == 1
    assert agent.close_calls == (0 if fail_open else 1)


class _ResponseBody(httpx.AsyncByteStream):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.closed = False
        self.reading = asyncio.Event()

    async def __aiter__(self):
        yield b'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
        if self.mode == "cancel":
            self.reading.set()
            await asyncio.Event().wait()
        if self.mode == "malformed":
            yield b'data: {"type":"response.output_text.delta","delta":123}\n\n'
            return
        response = {
            "id": "offline",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-terra",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        yield (
            "data: "
            + json.dumps({"type": "response.completed", "response": response})
            + "\n\n"
        ).encode()

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("mode", ("complete", "malformed", "cancel", "early_exit"))
@pytest.mark.parametrize("layer", ("adapter", "lazy", "router"))
async def test_real_sdk_response_stream_released_before_provider_shutdown(
    mode, layer, monkeypatch
):
    body = _ResponseBody(mode)
    sdk = openai.AsyncOpenAI(
        api_key="offline-not-a-real-key",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    request=request,
                    headers={"content-type": "text/event-stream"},
                    stream=body,
                )
            )
        ),
    )
    provider = OpenAIResponsesProvider("gpt-5.6-terra", client=cast(Any, sdk))
    model: Any = provider
    if layer != "adapter":
        monkeypatch.setattr(
            factory_module, "create_llm_provider", lambda *_a, **_k: provider
        )
        profile = reviewed_model_profile(provider.provider_id)
        assert profile is not None
        model = create_model_route_provider(
            ModelRoute(
                (ModelRouteCandidate(provider_id=profile.id, profile=profile),),
                retry_policy=RetryPolicy(
                    attempts=2 if layer == "router" else 1, backoff_seconds=0
                ),
            )
        )
    request = ModelRequest(
        messages=(
            CanonicalMessage(role=MessageRole.USER, content=(TextBlock("answer"),)),
        )
    )
    stream = cast(AsyncGenerator[ModelStreamEvent, None], model.stream(request))
    try:
        assert isinstance(await anext(stream), ModelTextDelta)
        if mode == "early_exit":
            await stream.aclose()
        elif mode == "cancel":
            pending = asyncio.ensure_future(anext(stream))
            await body.reading.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        elif mode == "malformed":
            with pytest.raises(ModelProviderError):
                await anext(stream)
        else:
            assert isinstance(await anext(stream), ModelStreamCompleted)
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        assert body.closed
        assert not sdk.is_closed()  # Request ownership is separate from the client.
    finally:
        await stream.aclose()
        await model.close()
        await sdk.close()


class _BlockingClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_calls = 0
        self.closed = False
        self.error: Exception | None = None

    async def close(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        self.closed = True

    async def aclose(self) -> None:
        await self.close()


async def _blocking_provider(monkeypatch, kind: str):
    client = _BlockingClient()
    provider: Any
    if kind == "anthropic":
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **_: client)
        provider = AnthropicMessagesProvider("test", api_key="offline")
    elif kind == "gemini":
        sdk = _GeminiClient()
        sdk.aio = cast(Any, client)
        monkeypatch.setattr(genai, "Client", lambda **_: sdk)
        provider = GeminiProvider("test", api_key="offline")
    elif kind == "compatible":
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_: client)
        provider = OpenAICompatibleProvider(
            "test", provider="custom", base_url="https://example.test/v1"
        )
    else:
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_: client)
        provider = OpenAIResponsesProvider("gpt-5.6-terra", api_key="offline")
    provider.client
    if kind in {"lazy", "router"}:
        delegate = provider

        async def generate(_request):
            return ModelResponse(finish_reason=FinishReason.STOP, text="done")

        monkeypatch.setattr(delegate, "generate", generate)
        monkeypatch.setattr(
            factory_module, "create_llm_provider", lambda *_a, **_k: delegate
        )
        route = _route()
        if kind == "router":
            route = replace(
                route, retry_policy=RetryPolicy(attempts=2, backoff_seconds=0)
            )
        provider = create_model_route_provider(route)
        await provider.generate(_request())
    return provider, client


@pytest.mark.parametrize(
    "kind", ("openai", "anthropic", "gemini", "compatible", "lazy", "router")
)
async def test_close_waits_for_cleanup_despite_repeated_cancellation(monkeypatch, kind):
    provider, client = await _blocking_provider(monkeypatch, kind)
    first = asyncio.create_task(provider.close())
    await client.started.wait()
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()
    second = asyncio.create_task(provider.close())
    await asyncio.sleep(0)
    try:
        assert not first.done()
        assert not second.done()
    finally:
        client.release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] is None
    await provider.close()
    assert client.closed
    assert client.close_calls == 1


@pytest.mark.parametrize(
    "kind", ("openai", "anthropic", "gemini", "compatible", "lazy", "router")
)
async def test_close_failure_is_not_reported_as_success_on_later_calls(
    monkeypatch, kind
):
    provider, client = await _blocking_provider(monkeypatch, kind)
    client.error = RuntimeError("offline cleanup failure")
    client.release.set()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="offline cleanup failure"):
            await provider.close()
    assert client.close_calls == 1


class _AsyncClient:
    def __init__(self) -> None:
        self.responses = self
        self.messages = self
        self.chat = self
        self.completions = self
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _GeminiAsyncClient:
    def __init__(self) -> None:
        self.models = self
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _GeminiClient:
    def __init__(self) -> None:
        self.aio = _GeminiAsyncClient()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    ("provider_factory", "sdk_symbol"),
    (
        (
            lambda: OpenAIResponsesProvider("gpt-test", api_key="key"),
            "openai",
        ),
        (
            lambda: AnthropicMessagesProvider("claude-test", api_key="key"),
            "anthropic",
        ),
        (
            lambda: OpenAICompatibleProvider(
                "model", provider="custom", base_url="https://example.test/v1"
            ),
            "openai",
        ),
    ),
    ids=("openai", "anthropic", "openai-compatible"),
)
async def test_async_provider_closes_only_the_client_it_created_once(
    monkeypatch, provider_factory, sdk_symbol: str
) -> None:
    owned = _AsyncClient()
    if sdk_symbol == "openai":
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: owned)
    else:
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **_kwargs: owned)
    provider = provider_factory()
    assert provider.client is owned

    await provider.close()
    await provider.close()

    assert owned.close_calls == 1


async def test_gemini_provider_closes_both_owned_sdk_surfaces_once(monkeypatch) -> None:
    owned = _GeminiClient()
    monkeypatch.setattr(genai, "Client", lambda **_kwargs: owned)
    provider = GeminiProvider("gemini-test", api_key="key")
    assert provider.client is owned

    await provider.close()
    await provider.close()

    assert owned.aio.close_calls == 1
    assert owned.close_calls == 1


@pytest.mark.parametrize(
    "provider",
    (
        OpenAIResponsesProvider("gpt-test", client=cast(Any, _AsyncClient())),
        AnthropicMessagesProvider("claude-test", client=cast(Any, _AsyncClient())),
        OpenAICompatibleProvider(
            "model",
            provider="custom",
            base_url="https://example.test/v1",
            client=cast(Any, _AsyncClient()),
        ),
        GeminiProvider("gemini-test", client=cast(Any, _GeminiClient())),
    ),
    ids=("openai", "anthropic", "openai-compatible", "gemini"),
)
async def test_provider_does_not_close_a_borrowed_client(provider) -> None:
    client = provider.client

    await provider.close()

    if isinstance(client, _GeminiClient):
        assert client.aio.close_calls == 0
    assert client.close_calls == 0


class _ManagedProvider:
    provider_id = "openai:gpt-5.6-terra"

    def __init__(self, response: ModelResponse | None = None) -> None:
        self.close_calls = 0
        self.response = response or ModelResponse(
            finish_reason=FinishReason.STOP,
            text="done",
        )

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return self.response

    async def close(self) -> None:
        self.close_calls += 1


def _route() -> ModelRoute:
    reviewed = reviewed_model_profile("openai:gpt-5.6-terra")
    assert reviewed is not None
    profile = replace(reviewed, supports_streaming=False)
    return ModelRoute(
        (ModelRouteCandidate(provider_id=profile.id, profile=profile),),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )


async def test_host_closes_its_configured_provider_after_run_drain(
    tmp_path, monkeypatch
) -> None:
    provider = _ManagedProvider()
    started = asyncio.Event()
    release = asyncio.Event()

    async def generate(_request):
        started.set()
        await release.wait()
        assert provider.close_calls == 0
        return provider.response

    monkeypatch.setattr(provider, "generate", generate)
    monkeypatch.setattr(
        embedded_module,
        "create_model_route_provider",
        lambda *_args, **_kwargs: provider,
    )
    agent = await Agent.create(
        "owns-configured-provider",
        root=tmp_path,
        hosted=True,
        config=AgentConfig(model_route=_route()),
    )
    running = asyncio.create_task(agent.run("answer"))
    await started.wait()
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    try:
        assert provider.close_calls == 0
        assert not closing.done()
    finally:
        release.set()
        await running
        with pytest.raises(asyncio.CancelledError):
            await closing
        await agent.close()

    assert provider.close_calls == 1


async def test_configured_openai_route_closes_the_real_sdk_http_client(
    tmp_path, monkeypatch
) -> None:
    real_client_type = openai.AsyncOpenAI
    sdk_clients: list[openai.AsyncOpenAI] = []

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "response-offline",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.6-terra",
                "output": (
                    {
                        "id": "message-offline",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": (
                            {
                                "type": "output_text",
                                "text": "Offline answer.",
                                "annotations": (),
                            },
                        ),
                    },
                ),
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    def create_client(**_kwargs: object) -> openai.AsyncOpenAI:
        client = real_client_type(
            api_key="offline-not-a-real-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        sdk_clients.append(client)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", create_client)
    agent = await Agent.create(
        "real-sdk-lifecycle",
        root=tmp_path,
        hosted=True,
        config=AgentConfig(model_route=_route()),
    )
    try:
        result = await agent.run("answer")
        assert result.final_text == "Offline answer."
    finally:
        await agent.close()

    assert len(sdk_clients) == 1
    assert sdk_clients[0].is_closed()


async def test_host_does_not_close_an_injected_shared_provider(tmp_path) -> None:
    provider = _ManagedProvider()
    profile = _route().model_profile
    agent = await Agent.create(
        "borrows-injected-provider",
        root=tmp_path,
        hosted=True,
        model=provider,
        model_profile=profile,
    )
    await agent.run("answer")

    await agent.close()

    assert provider.close_calls == 0


async def test_host_closes_separate_configured_primary_and_reviewer_routes(
    tmp_path, monkeypatch
) -> None:
    primary = _ManagedProvider()
    reviewer = _ManagedProvider()
    providers = iter((primary, reviewer))
    monkeypatch.setattr(
        embedded_module,
        "create_model_route_provider",
        lambda *_args, **_kwargs: next(providers),
    )
    agent = await Agent.create(
        "owns-configured-reviewer",
        root=tmp_path,
        hosted=True,
        config=AgentConfig(model_route=_route()),
        reviewer_max_estimated_cost_usd=Decimal("1"),
    )

    await agent.close()

    assert primary.close_calls == 1
    assert reviewer.close_calls == 1


def _request() -> ModelRequest:
    return ModelRequest(
        (CanonicalMessage(MessageRole.USER, content=(TextBlock("answer"),)),)
    )


async def test_unused_lazy_route_closes_without_constructing_a_provider(
    monkeypatch,
) -> None:
    created = []
    monkeypatch.setattr(
        factory_module,
        "create_llm_provider",
        lambda *_args, **_kwargs: created.append(_ManagedProvider()),
    )
    provider = create_model_route_provider(_route())

    await provider.close()

    assert created == []


@pytest.mark.parametrize("close_during_resolution", (False, True))
async def test_lazy_resolution_cannot_duplicate_or_reactivate_a_closed_owner(
    monkeypatch, close_during_resolution
):
    started = asyncio.Event()
    release = asyncio.Event()
    created: list[_ManagedProvider] = []

    class Secrets:
        async def resolve(self, _reference):
            started.set()
            await release.wait()
            return "offline"

    def create(*_a, **_k):
        delegate = _ManagedProvider()
        created.append(delegate)
        return delegate

    monkeypatch.setattr(factory_module, "create_llm_provider", create)
    route = _route()
    route = replace(
        route,
        candidates=(
            replace(
                route.candidates[0],
                secret_reference=SecretReference.environment("OFFLINE_KEY"),
            ),
        ),
    )
    provider = create_model_route_provider(route, secret_provider=Secrets())
    calls = [asyncio.create_task(provider.generate(_request())) for _ in range(2)]
    await started.wait()
    await asyncio.sleep(0)
    if close_during_resolution:
        await provider.close()
    release.set()
    outcomes = await asyncio.gather(*calls, return_exceptions=True)
    await provider.close()
    if close_during_resolution:
        assert all(isinstance(outcome, ModelProviderError) for outcome in outcomes)
        assert created == []
    else:
        assert all(isinstance(outcome, ModelResponse) for outcome in outcomes)
        assert len(created) == 1
        assert created[0].close_calls == 1


async def test_activated_lazy_route_closes_its_delegate_exactly_once(
    monkeypatch,
) -> None:
    created = []

    def create(*_args, **_kwargs):
        provider = _ManagedProvider()
        created.append(provider)
        return provider

    monkeypatch.setattr(factory_module, "create_llm_provider", create)
    provider = create_model_route_provider(_route())
    await provider.generate(_request())

    await provider.close()
    await provider.close()

    assert len(created) == 1
    assert created[0].close_calls == 1


async def test_temporary_route_validation_closes_its_owned_provider(
    monkeypatch,
) -> None:
    provider = _ManagedProvider(
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="validation-call",
                    name="daita_validate_tool_support",
                    arguments={},
                ),
            ),
            provider_id="openai:gpt-5.6-terra",
        )
    )
    monkeypatch.setattr(
        embedded_module,
        "create_model_route_provider",
        lambda *_args, **_kwargs: provider,
    )

    await embedded_module._validate_model_route(
        _route(),
        secret_provider=EmptySecretProvider(),
        injected_provider=None,
    )

    assert provider.close_calls == 1


async def test_route_validation_keeps_an_injected_provider_borrowed() -> None:
    provider = _ManagedProvider(
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="validation-call",
                    name="daita_validate_tool_support",
                    arguments={},
                ),
            ),
            provider_id="openai:gpt-5.6-terra",
        )
    )

    await embedded_module._validate_model_route(
        _route(),
        secret_provider=EmptySecretProvider(),
        injected_provider=provider,
    )

    assert provider.close_calls == 0
