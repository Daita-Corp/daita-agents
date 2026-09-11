"""Shared helpers extracted from ``test_token_counting.py``."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast

import anthropic
import httpx
import openai
from google import genai
from google.genai import types

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    TextBlock,
    ToolDefinition,
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
