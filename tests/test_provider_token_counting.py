"""Request admission counts real provider payloads, with all HTTP kept offline."""

from contextlib import asynccontextmanager
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from typing import Any, cast
import asyncio
import json

import anthropic
from google import genai
from google.genai import types
import httpx
import openai
import pytest
from live.benchmarks._support import RecordingProvider

from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    interrupted_model_usage,
)
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    TextBlock,
    ToolDefinition,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.openai import OpenAIResponsesProvider


@asynccontextmanager
async def provider_at(kind, transport, *, max_output=2048):
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    sdk: Any
    provider: OpenAIResponsesProvider | AnthropicMessagesProvider | GeminiProvider
    if kind == "openai":
        sdk = openai.AsyncOpenAI(api_key="offline", http_client=client)
        provider = OpenAIResponsesProvider(
            "gpt-5.6-terra", client=cast(Any, sdk), max_output_tokens=max_output
        )
    elif kind == "anthropic":
        sdk = anthropic.AsyncAnthropic(api_key="offline", http_client=client)
        provider = AnthropicMessagesProvider(
            "fixture-model", client=cast(Any, sdk), max_tokens=max_output
        )
    else:
        sdk = genai.Client(
            api_key="offline",
            http_options=types.HttpOptions(
                httpx_async_client=client,
                retry_options=types.HttpRetryOptions(attempts=4),
            ),
        )
        provider = GeminiProvider(
            "fixture-model", client=cast(Any, sdk), max_output_tokens=max_output
        )
    try:
        yield provider
        assert not client.is_closed
    finally:
        await provider.close()
        if kind == "gemini":
            await sdk.aio.aclose()
            sdk.close()
        else:
            await sdk.close()
        await client.aclose()


def input_request(*, remaining=22_000):
    # Reproduce the live failure's relation: input bytes exceed the remaining
    # allowance, although the provider can count an input that fits comfortably.
    return ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.SYSTEM,
                content=(TextBlock("Use the admitted tool and preserve evidence."),),
            ),
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("release readiness 日本語 🚀 " * 1000),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="notify_release",
                description="Publish the approved finding.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string"},
                        "schedule": {
                            "oneOf": [
                                {"type": "string", "enum": ["once"]},
                                {"type": "integer", "minimum": 1},
                            ]
                        },
                    },
                    "required": ["destination"],
                    "additionalProperties": False,
                },
            ),
        ),
        response_schema={
            "type": "object",
            "properties": {"invocation_status": {"type": "string"}},
            "required": ["invocation_status"],
            "additionalProperties": False,
        },
        max_total_tokens=remaining,
    )


def is_count(path):
    return path.endswith(("/input_tokens", "/count_tokens", ":countTokens"))


def count_response(kind, tokens):
    if kind == "gemini":
        return {"totalTokens": tokens}
    return {"input_tokens": tokens, "object": "response.input_tokens"}


async def invoke(provider, request, stream):
    if stream:
        async for _ in provider.stream(request):
            pass
    else:
        await provider.generate(request)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
async def test_counted_payload_fits_despite_large_serialized_bytes(kind, stream):
    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content)))
        if is_count(request.url.path):
            return httpx.Response(200, json=count_response(kind, 5000))
        return httpx.Response(503, json={"error": {"message": "offline"}})

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, input_request(), stream)
    assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert len(requests) == 2
    assert is_count(requests[0][0]) and not is_count(requests[1][0])
    counted, generated = requests[0][1], requests[1][1]
    if kind == "openai":
        assert counted == {
            key: value
            for key, value in generated.items()
            if key
            not in {"max_output_tokens", "service_tier", "store", "include", "stream"}
        }
        assert generated["max_output_tokens"] == 2048
    elif kind == "anthropic":
        assert counted == {
            key: value
            for key, value in generated.items()
            if key not in {"max_tokens", "stream"}
        }
        assert generated["max_tokens"] == 2048
    else:
        complete = counted["generateContentRequest"]
        assert complete.pop("model") == "models/fixture-model"
        expected = dict(generated)
        expected["generationConfig"] = dict(expected["generationConfig"])
        expected["generationConfig"].pop("maxOutputTokens")
        assert complete == expected
        assert generated["generationConfig"]["maxOutputTokens"] == 2048


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("tokens", [22_000, 30_000])
async def test_insufficient_request_allowance_stops_before_generation(kind, tokens):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=count_response(kind, tokens))

    async with provider_at(kind, respond) as provider:
        recorder = RecordingProvider(provider)
        with pytest.raises(ModelProviderError) as caught:
            await invoke(recorder, input_request(), True)
    assert caught.value.code.value == "token_budget_insufficient"
    assert len(paths) == 1 and is_count(paths[0])
    assert caught.value.usage.total_tokens == 0
    assert caught.value.usage.cost_estimate.amount_usd == Decimal(0)
    diagnostic = caught.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.input_tokens == tokens
    assert diagnostic.remaining_tokens == 22_000
    assert diagnostic.maximum_output_tokens == 2048
    saved = recorder.timings[0]
    admission_failure = cast(Mapping[str, object], saved["admission_failure"])
    assert admission_failure["input_tokens"] == tokens
    assert admission_failure["remaining_tokens"] == 22_000
    assert saved["input_tokens"] == 0
    assert saved["request_admission"] is None
    json.dumps(saved)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
async def test_success_preserves_counted_admission_separately_from_actual_usage(kind):
    def respond(request):
        if is_count(request.url.path):
            return httpx.Response(200, json=count_response(kind, 5000))
        if kind == "openai":
            payload = {
                "id": "resp_offline",
                "object": "response",
                "status": "completed",
                "model": "gpt-5.6-terra",
                "created_at": 0,
                "output": [
                    {
                        "id": "msg_offline",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Done.", "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 5010,
                    "output_tokens": 10,
                    "total_tokens": 5020,
                },
            }
        elif kind == "anthropic":
            payload = {
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": "fixture-model",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "Done."}],
                "usage": {"input_tokens": 5010, "output_tokens": 10},
            }
        else:
            payload = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "Done."}]},
                        "finishReason": "STOP",
                    }
                ],
                "modelVersion": "fixture-model",
                "usageMetadata": {
                    "promptTokenCount": 5010,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 5020,
                },
            }
        return httpx.Response(200, json=payload)

    async with provider_at(kind, respond) as provider:
        recorder = RecordingProvider(provider)
        response = await recorder.generate(
            replace(input_request(), response_schema=None)
        )
    assert response.usage.input_tokens == 5010
    assert response.usage.total_tokens == 5020
    assert dict(
        cast(Mapping[str, object], response.provider_metadata["request_admission"])
    ) == {
        "input_tokens": 5000,
        "remaining_tokens": 22000,
        "output_cap": 2048,
    }
    saved = recorder.timings[0]
    assert (
        cast(Mapping[str, object], saved["request_admission"])["input_tokens"] == 5000
    )
    assert saved["input_tokens"] == 5010
    assert saved["admission_failure"] is None
    json.dumps(saved)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
async def test_count_request_failure_is_not_a_billed_generation_attempt(kind):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(503, json={"error": {"message": "private vendor text"}})

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, input_request(), False)
    assert len(paths) == 1 and is_count(paths[0])
    assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert caught.value.diagnostic is not None
    assert caught.value.diagnostic.phase.value == "request_admission"
    assert "private vendor text" not in str(caught.value)
    assert caught.value.usage.cost_estimate.amount_usd == Decimal(0)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("count", [None, -1])
async def test_invalid_counts_never_admit_generation(kind, count):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=count_response(kind, count))

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, input_request(), False)
    assert caught.value.code is ProviderErrorCode.TOKEN_COUNT_UNAVAILABLE
    assert len(paths) == 1 and is_count(paths[0])


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
async def test_output_cap_reserves_counted_input(kind, stream):
    payloads = []

    def respond(request):
        if is_count(request.url.path):
            return httpx.Response(200, json=count_response(kind, 6000))
        payloads.append(json.loads(request.content))
        return httpx.Response(503, json={"error": {"message": "offline"}})

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, input_request(remaining=7000), stream)
    assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert len(payloads) == 1
    body = payloads[0]
    cap = (
        body["generationConfig"]["maxOutputTokens"]
        if kind == "gemini"
        else body["max_tokens" if kind == "anthropic" else "max_output_tokens"]
    )
    assert cap == 1000


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("deadline", [False, True])
async def test_count_cancellation_preserves_zero_usage_and_cancellation(kind, deadline):
    entered = asyncio.Event()
    released = asyncio.Event()
    paths = []

    async def respond(request):
        paths.append(request.url.path)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    async with provider_at(kind, respond) as provider:
        if deadline:
            with pytest.raises(TimeoutError) as caught:
                async with asyncio.timeout(0.1):
                    await invoke(provider, input_request(), True)
            error: BaseException = caught.value
        else:
            task = asyncio.create_task(invoke(provider, input_request(), True))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await task
            error = cancelled.value
            assert task.cancelled()
        assert released.is_set()
    assert len(paths) == 1 and is_count(paths[0])
    usage = interrupted_model_usage(error)
    assert usage.total_tokens == 0
    assert usage.cost_estimate.status.value == "complete"
    assert usage.cost_estimate.amount_usd == Decimal(0)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
async def test_unbounded_request_does_not_make_counting_call(kind):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(503, json={"error": {"message": "offline"}})

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError):
            await invoke(
                provider, replace(input_request(), max_total_tokens=None), False
            )
    assert len(paths) == 1 and not is_count(paths[0])


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("constraint", ["tokens", "cost"])
async def test_exhausted_allowance_does_not_even_count(kind, constraint):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        raise AssertionError("exhausted request performed I/O")

    request = replace(
        input_request(),
        max_total_tokens=0 if constraint == "tokens" else 30_000,
        max_estimated_cost_usd=Decimal(0) if constraint == "cost" else None,
    )
    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, request, False)
    assert caught.value.code.value == (
        "token_limit_reached" if constraint == "tokens" else "cost_limit_reached"
    )
    assert paths == []


async def test_gemini_count_preserves_signed_tool_exchange():
    import base64

    seen = []
    signature = base64.b64encode(b"synthetic fixture signature").decode("ascii")
    native_call = {"name": "notify_release", "args": {"destination": "release-room"}}
    assistant = CanonicalMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=(
            ToolCall(
                id="call-1",
                name="notify_release",
                arguments={"destination": "release-room"},
            ),
        ),
        provider_id="gemini:fixture-model",
        provider_metadata={
            "gemini_continuation": {
                "provider_id": "gemini:fixture-model",
                "content_parts": [
                    {"function_call": native_call, "thought_signature": signature}
                ],
            },
        },
    )
    result = CanonicalMessage(
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id="call-1", output={"status": "reported"}),),
    )
    request = input_request()
    request = replace(request, messages=(*request.messages, assistant, result))

    def respond(req):
        seen.append((req.url.path, json.loads(req.content)))
        if is_count(req.url.path):
            return httpx.Response(200, json={"totalTokens": 6000})
        return httpx.Response(503, json={"error": {"message": "offline"}})

    async with provider_at("gemini", respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, request, True)
    assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert len(seen) == 2
    counted = seen[0][1]["generateContentRequest"]["contents"]
    generated = seen[1][1]["contents"]
    assert counted == generated
    assert counted[-2]["parts"][0]["thoughtSignature"] == signature
    assert counted[-1]["parts"][0]["functionResponse"]["response"]["output"] == {
        "status": "reported"
    }


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
async def test_count_preserves_opaque_reasoning_and_tool_results(kind):
    seen = []
    model = "gpt-5.6-terra" if kind == "openai" else "fixture-model"
    provider_id = f"{kind}:{model}"
    opaque = "synthetic-opaque-block-" * 1500
    metadata = (
        {
            "openai_replay_items": [
                {
                    "type": "reasoning",
                    "id": "rs_offline",
                    "summary": [],
                    "encrypted_content": opaque,
                }
            ]
        }
        if kind == "openai"
        else {
            "anthropic_continuation": {
                "provider_id": provider_id,
                "content_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "synthetic reasoning",
                        "signature": opaque,
                    }
                ],
            }
        }
    )
    assistant = CanonicalMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=(
            ToolCall(
                id="call-1",
                provider_call_id="native-call-1",
                name="notify_release",
                arguments={"destination": "release-room"},
            ),
        ),
        provider_id=provider_id,
        provider_metadata=metadata,
    )
    result = CanonicalMessage(
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id="call-1", output={"status": "reported"}),),
    )
    request = input_request()
    request = replace(request, messages=(*request.messages, assistant, result))

    def respond(req):
        seen.append((req.url.path, json.loads(req.content)))
        if is_count(req.url.path):
            return httpx.Response(200, json=count_response(kind, 6000))
        return httpx.Response(503, json={"error": {"message": "offline"}})

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, request, True)
    assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert len(seen) == 2
    key = "input" if kind == "openai" else "messages"
    assert seen[0][1][key] == seen[1][1][key]
    assert opaque in json.dumps(seen[0][1][key])


@pytest.mark.parametrize("kind", ["compatible", "grok", "ollama", "codex"])
async def test_uncounted_routes_preserve_usage_based_progression(kind):
    from daita.llm.providers import (
        OpenAICompatibleProvider,
        GrokProvider,
        OllamaProvider,
        CodexSubscriptionProvider,
    )
    from test_subscription_providers import _credential

    seen = []

    def respond(req):
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(503, json={"error": {"message": "offline"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    sdk = openai.AsyncOpenAI(
        api_key="offline", base_url="https://fixture.test/v1", http_client=http_client
    )
    provider: OpenAICompatibleProvider | CodexSubscriptionProvider
    if kind == "compatible":
        provider = OpenAICompatibleProvider(
            "fixture",
            provider="custom",
            base_url="https://fixture.test/v1",
            client=cast(Any, sdk),
            max_tokens=8192,
        )
    elif kind == "grok":
        provider = GrokProvider("fixture", client=cast(Any, sdk), max_tokens=8192)
    elif kind == "ollama":
        provider = OllamaProvider("fixture", client=cast(Any, sdk), max_tokens=8192)
    else:
        provider = CodexSubscriptionProvider(
            "fixture",
            credential=_credential().to_secret(),
            client=cast(Any, sdk),
            max_output_tokens=8192,
        )
    try:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, input_request(remaining=7000), False)
        assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
        assert len(seen) == 1 and not is_count(seen[0][0])
        if kind == "codex":
            assert "max_output_tokens" not in seen[0][1]
        else:
            assert seen[0][1]["max_tokens"] == 7000
        seen.clear()
        with pytest.raises(ModelProviderError) as caught:
            await invoke(
                provider,
                replace(input_request(), max_estimated_cost_usd=Decimal("0.15")),
                False,
            )
        assert caught.value.code is ProviderErrorCode.TOKEN_COUNT_UNAVAILABLE
        assert seen == []
    finally:
        await provider.close()
        await sdk.close()


def test_subscription_cli_separates_request_bytes_and_token_allowance():
    from daita.llm.providers.subscription_cli import _request_document

    request = input_request(remaining=7000)
    document = _request_document(request, 8192)
    assert len(document.encode("utf-8")) > request.max_total_tokens
    assert json.loads(document)["maximum_output_tokens"] == 7000
    assert (
        json.loads(document)["messages"][1]["content"][0]["text"]
        == request.messages[1].content[0].text
    )
    with pytest.raises(ModelProviderError) as caught:
        _request_document(
            replace(request, max_estimated_cost_usd=Decimal("0.15")), 8192
        )
    assert caught.value.code is ProviderErrorCode.TOKEN_COUNT_UNAVAILABLE


@pytest.mark.parametrize("stream", [False, True])
async def test_loop_deadline_during_counting_retains_zero_charge(stream):
    from test_loop import NOW, ScriptedTools, TranscriptContext
    from daita.loop import AgentLoop, InMemoryTranscriptStore, LoopLimits, RunInput

    paths = []

    async def respond(request):
        paths.append(request.url.path)
        await asyncio.Event().wait()

    store = InMemoryTranscriptStore()
    async with provider_at("openai", respond) as provider:
        loop = AgentLoop(
            model=provider,
            context_builder=TranscriptContext(),
            tools=ScriptedTools({}),
            transcripts=store,
            limits=LoopLimits(max_wall_time_seconds=0.1),
            stream_model_calls=stream,
            clock=lambda: NOW,
        )
        result = await loop.run(RunInput("count-deadline", "agent-1", "question", NOW))
    assert paths and all(is_count(path) for path in paths)
    assert result.reason == "wall_time_exhausted"
    assert result.usage.total_tokens == 0
    assert result.usage.cost_estimate.amount_usd == Decimal(0)
    assert result.usage.cost_estimate.status.value == "complete"
    assert await store.result(result.run_id) == result


@pytest.mark.parametrize("stream", [False, True])
async def test_router_retains_prior_attempt_charges_when_counting_is_cancelled(stream):
    from daita.llm.errors import with_cancelled_model_usage
    from daita.llm.models import ModelUsage, ModelStreamCompleted
    from daita.llm.pricing import CostEstimate
    from daita.llm.providers.mock import MockStreamingModelProvider
    from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy

    entered = asyncio.Event()
    failed_usage = ModelUsage(
        input_tokens=30, cost_estimate=CostEstimate.complete(Decimal("0.01"))
    )

    class Provider(MockStreamingModelProvider):
        attempts = 0

        async def generate(self, request):
            self.attempts += 1
            if self.attempts == 1:
                raise ModelProviderError(
                    ProviderErrorCode.PROVIDER_UNAVAILABLE, usage=failed_usage
                )
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                raise with_cancelled_model_usage(
                    error, ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
                ) from None
            raise AssertionError("counting unexpectedly resumed")

        async def stream(self, request):
            yield ModelStreamCompleted(await self.generate(request))

    provider = Provider((), complete_pricing=True)
    router = ModelRouter(
        (ModelProviderRegistration(provider=provider, profile=provider.model_profile),),
        retry_policy=RetryPolicy(attempts=2, backoff_seconds=0),
    )
    try:
        task = asyncio.create_task(
            invoke(router, replace(input_request(), response_schema=None), stream)
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        usage = interrupted_model_usage(caught.value)
        assert usage.total_tokens == 30
        assert usage.cost_estimate.amount_usd == Decimal("0.01")
        assert usage.cost_estimate.status.value == "complete"
        assert provider.attempts == 2
    finally:
        await router.close()
