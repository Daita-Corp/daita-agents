"""Budget and retry boundaries exercised with actual SDK HTTP requests, offline."""

import json
from decimal import Decimal
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest
from google import genai
from google.genai import types
from test_provider_lifecycle import _ResponseBody

from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelUsage,
    TextBlock,
    ToolDefinition,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy


def request(**limits):
    return ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER, content=(TextBlock("Answer briefly."),)
            ),
        ),
        **limits,
    )


@pytest.mark.parametrize("kind", ["anthropic", "gemini", "compatible"])
@pytest.mark.parametrize("stream", [False, True])
async def test_actual_sdk_preserves_union_schema_and_single_budgeted_attempt(
    kind, stream
):
    payloads = []
    counts = []

    def unavailable(req):
        if req.url.path.endswith(("/count_tokens", ":countTokens")):
            counts.append(json.loads(req.content))
            return httpx.Response(
                200, json={"totalTokens" if kind == "gemini" else "input_tokens": 500}
            )
        payloads.append(json.loads(req.content))
        return httpx.Response(
            503,
            request=req,
            headers={"retry-after-ms": "1"},
            json={
                "error": {
                    "code": 503,
                    "message": "offline unavailable",
                    "status": "UNAVAILABLE",
                    "type": "server_error",
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    sdk: Any
    provider: AnthropicMessagesProvider | GeminiProvider | OpenAICompatibleProvider
    if kind == "anthropic":
        sdk = anthropic.AsyncAnthropic(api_key="offline", http_client=http_client)
        provider = AnthropicMessagesProvider(
            "test-model", client=cast(Any, sdk), max_tokens=8192
        )
    elif kind == "gemini":
        sdk = genai.Client(
            api_key="offline",
            http_options=types.HttpOptions(
                httpx_async_client=http_client,
                retry_options=types.HttpRetryOptions(attempts=4),
            ),
        )
        provider = GeminiProvider(
            "test-model", client=cast(Any, sdk), max_output_tokens=8192
        )
    else:
        sdk = openai.AsyncOpenAI(api_key="offline", http_client=http_client)
        provider = OpenAICompatibleProvider(
            "test-model",
            provider="custom",
            base_url="https://fixture.test/v1",
            client=cast(Any, sdk),
            max_tokens=8192,
        )
    schema = {
        "type": "object",
        "properties": {
            "schedule": {
                "type": "object",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"type": "string", "enum": [kind]}},
                        "required": ["kind"],
                        "additionalProperties": False,
                    }
                    for kind in ("once", "interval")
                ],
            }
        },
        "required": ["schedule"],
        "additionalProperties": False,
    }
    bounded = request(
        max_total_tokens=2000,
        tools=(
            ToolDefinition(
                name="fixture_schedule",
                description="Schedule fixture.",
                input_schema=schema,
            ),
        ),
    )
    try:
        with pytest.raises(ModelProviderError):
            if stream:
                async for _ in provider.stream(bounded):
                    pass
            else:
                await provider.generate(bounded)
        assert len(payloads) == 1
        payload = payloads[0]
        if kind == "anthropic":
            assert payload["tools"][0]["input_schema"] == schema
            cap = payload["max_tokens"]
        elif kind == "gemini":
            assert (
                payload["tools"][0]["functionDeclarations"][0]["parameters_json_schema"]
                == schema
            )
            cap = payload["generationConfig"]["maxOutputTokens"]
        else:
            assert payload["tools"][0]["function"]["parameters"] == schema
            cap = payload["max_tokens"]
        assert cap == (2000 if kind == "compatible" else 1500)
        assert len(counts) == (0 if kind == "compatible" else 1)
        assert not http_client.is_closed
    finally:
        await provider.close()
        if kind == "gemini":
            await sdk.aio.aclose()
            sdk.close()
        else:
            await sdk.close()
        await http_client.aclose()


@pytest.mark.parametrize("constraint", ("tokens", "cost", "cannot_fit"))
async def test_actual_sdk_applies_budget_to_wire_before_dispatch(constraint):
    import json

    payloads = []
    body = _ResponseBody("complete")

    def respond(req):
        if req.url.path.endswith("/input_tokens"):
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 500}
            )
        payloads.append(json.loads(req.content))
        return httpx.Response(
            200, request=req, headers={"content-type": "text/event-stream"}, stream=body
        )

    sdk = openai.AsyncOpenAI(
        api_key="offline-not-a-real-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    provider = OpenAIResponsesProvider(
        "gpt-5.6-terra", client=cast(Any, sdk), max_output_tokens=8192
    )
    bounded = request(
        max_total_tokens=(
            64
            if constraint == "cannot_fit"
            else (1500 if constraint == "tokens" else 100_000)
        ),
        max_estimated_cost_usd=Decimal("0.01") if constraint == "cost" else None,
    )
    try:
        if constraint == "cannot_fit":
            with pytest.raises(ModelProviderError) as caught:
                async for _ in provider.stream(bounded):
                    pass
            assert caught.value.code is ProviderErrorCode.TOKEN_BUDGET_INSUFFICIENT
            assert caught.value.usage.cost_estimate.amount_usd == 0
            assert payloads == []
        else:
            async for _ in provider.stream(bounded):
                pass
            assert len(payloads) == 1
            assert (
                0
                < payloads[0]["max_output_tokens"]
                < (1500 if constraint == "tokens" else 8192)
            )
            assert "max_total_tokens" not in payloads[0]
            assert "max_estimated_cost_usd" not in payloads[0]
            assert body.closed
        assert not sdk.is_closed()
    finally:
        await provider.close()
        await sdk.close()


@pytest.mark.parametrize("stream", (False, True))
async def test_actual_sdk_cannot_multiply_router_attempts(stream):
    attempts = []

    def unavailable(req):
        attempts.append(req.method)
        return httpx.Response(
            503,
            request=req,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "offline unavailable", "type": "server_error"}},
        )

    sdk = openai.AsyncOpenAI(
        api_key="offline-not-a-real-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(unavailable)),
    )
    provider = OpenAIResponsesProvider("gpt-5.6-terra", client=cast(Any, sdk))
    profile = reviewed_model_profile(provider.provider_id)
    assert profile is not None
    router = ModelRouter(
        (ModelProviderRegistration(provider=provider, profile=profile),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
    )
    try:
        with pytest.raises(ModelProviderError):
            if stream:
                async for _ in router.stream(request()):
                    pass
            else:
                await router.generate(request())
        assert len(attempts) == 2
        assert sdk.max_retries == 2  # The borrowed owner's settings were not mutated.
        assert not sdk.is_closed()
    finally:
        await router.close()
        await sdk.close()


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("unknown", (False, True))
async def test_router_accounts_failed_attempts_before_another_request(stream, unknown):
    consumed = ModelUsage(
        input_tokens=30,
        cost_estimate=(
            CostEstimate.unavailable()
            if unknown
            else CostEstimate.complete(Decimal("0.03"))
        ),
    )
    failure = ModelProviderError(ProviderErrorCode.TIMEOUT, usage=consumed)
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        text="done",
        usage=ModelUsage(
            input_tokens=10, cost_estimate=CostEstimate.complete(Decimal("0.01"))
        ),
    )
    provider = (
        MockStreamingModelProvider(
            (failure, (ModelStreamCompleted(response),)), complete_pricing=True
        )
        if stream
        else MockModelProvider((failure, response), complete_pricing=True)
    )
    router = ModelRouter(
        (ModelProviderRegistration(provider=provider, profile=provider.model_profile),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
    )
    bounded = request(max_total_tokens=100, max_estimated_cost_usd=Decimal("0.10"))

    async def run():
        if stream:
            result = None
            async for event in router.stream(bounded):
                if isinstance(event, ModelStreamCompleted):
                    result = event.response
            return result
        return await router.generate(bounded)

    try:
        if unknown:
            with pytest.raises(ModelProviderError):
                await run()
            assert len(provider.requests) == 1
        else:
            result = await run()
            assert result is not None
            assert result.usage.total_tokens == 40
            assert result.usage.cost_estimate.amount_usd == Decimal("0.04")
            assert len(provider.requests) == 2
            assert provider.requests[1].max_total_tokens == 70
            assert provider.requests[1].max_estimated_cost_usd == Decimal("0.07")
    finally:
        await router.close()
