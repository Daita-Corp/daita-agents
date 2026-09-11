"""Actual SDK and real loopback sockets: bytes, events, progress, and release."""

import asyncio
from dataclasses import replace
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest
import pytest_asyncio
from _stream_boundary_support import endpoint
from google import genai

from daita.llm import ModelCallPolicy, ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelStreamCompleted,
    TextBlock,
)
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider

POLICY = ModelCallPolicy(
    max_request_seconds=1.2,
    max_attempt_seconds=0.8,
    first_progress_timeout_seconds=0.5,
    progress_idle_timeout_seconds=0.3,
    read_timeout_seconds=0.7,
    cleanup_timeout_seconds=0.15,
)


def request(**kwargs):
    return ModelRequest(
        (CanonicalMessage(MessageRole.USER, content=(TextBlock("x"),)),),
        call_policy=POLICY,
        **kwargs,
    )


@pytest.mark.parametrize("family", ["openai", "anthropic", "gemini", "compatible"])
@pytest.mark.parametrize(
    "scenario",
    [
        "silent",
        "startup_only",
        "comments",
        "empty_events",
        "text_trickle",
        "complete",
        "eof_without_terminal",
    ],
)
async def test_actual_sdk_liveness_and_release(family, scenario):
    sdk: Any
    provider: Any
    async with endpoint(scenario, family=family) as (url, paths, _writes):
        http = httpx.AsyncClient(trust_env=False)
        if family == "openai":
            sdk = openai.AsyncOpenAI(
                api_key="offline", base_url=url, http_client=http, max_retries=7
            )
            provider = OpenAIResponsesProvider("test", client=cast(Any, sdk))
        elif family == "anthropic":
            sdk = anthropic.AsyncAnthropic(
                api_key="offline", base_url=url, http_client=http, max_retries=7
            )
            provider = AnthropicMessagesProvider("test", client=cast(Any, sdk))
        elif family == "gemini":
            sdk = genai.Client(
                api_key="offline",
                http_options=cast(Any, {"base_url": url, "httpx_async_client": http}),
            )
            provider = GeminiProvider("test", client=cast(Any, sdk))
        else:
            sdk = openai.AsyncOpenAI(
                api_key="offline", base_url=url, http_client=http, max_retries=7
            )
            provider = OpenAICompatibleProvider(
                "test", provider="custom", base_url=url, client=cast(Any, sdk)
            )
        events = []
        failure = None
        start = asyncio.get_running_loop().time()
        try:
            # A later emergency guard is never an accepted production outcome.
            async with asyncio.timeout(3):
                try:
                    async for event in provider.stream(request()):
                        events.append(event)
                except ModelProviderError as error:
                    failure = error
            elapsed = asyncio.get_running_loop().time() - start
            if scenario == "complete":
                assert failure is None
                assert isinstance(events[-1], ModelStreamCompleted)
                assert events[-1].response.usage.total_tokens == 11
            else:
                assert failure is not None
                assert not any(
                    isinstance(event, ModelStreamCompleted) for event in events
                )
                assert failure.diagnostic is not None
                assert failure.diagnostic.attempt is not None
                native_decode_failure = family == "gemini" and scenario == "comments"
                assert failure.code is (
                    ProviderErrorCode.MALFORMED_RESPONSE
                    if scenario == "eof_without_terminal" or native_decode_failure
                    else ProviderErrorCode.TIMEOUT
                )
                if native_decode_failure:
                    assert failure.diagnostic.code == "native_response_decode_failed"
                    assert (
                        failure.diagnostic.attempt["first_substantive_progress_seconds"]
                        is None
                    )
                if (
                    scenario not in {"eof_without_terminal"}
                    and not native_decode_failure
                ):
                    diagnostic = failure.diagnostic.attempt
                    assert diagnostic["progress_mode"] == "observable"
                    assert diagnostic["timeout_reason"] in (
                        None,
                        "attempt_deadline",
                        "first_progress",
                    )
                    assert elapsed < 1.15
                    if scenario == "text_trickle":
                        assert elapsed >= 0.75
                    else:
                        assert diagnostic["first_substantive_progress_seconds"] is None
                        assert diagnostic["timeout_reason"] == "first_progress"
                    # SDK cold imports can synchronously block the event loop;
                    # the supervising await must still return within the later guard.
            assert len(paths) == 1
            assert not provider._native_owner.poisoned
            assert not provider._native_owner.tasks
            await provider.close()
            assert not http.is_closed
            if family != "gemini":
                assert sdk.max_retries == 7
        finally:
            await provider.close()
            if family == "gemini":
                await sdk.aio.aclose()
                sdk.close()
            else:
                await sdk.close()
            await http.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_expired_direct_request_has_zero_dispatch(streaming):
    async with endpoint("complete") as (url, paths, _writes):
        sdk = openai.AsyncOpenAI(api_key="offline", base_url=url)
        provider = OpenAIResponsesProvider("test", client=cast(Any, sdk))
        try:
            with pytest.raises(ModelProviderError) as caught:
                if streaming:
                    async for _ in provider.stream(request(deadline=0)):
                        pass
                else:
                    await provider.generate(request(deadline=0))
            assert caught.value.usage.total_tokens == 0
            assert caught.value.usage.cost_estimate.status.value == "complete"
            assert paths == []
        finally:
            await sdk.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def warm_native_sdk_imports():
    """Load lazy SDK code through one bounded synthetic request before subsecond cases.

    Native synchronous imports cannot be preempted by an asyncio timer. This
    successful cold-start control is separate from the timed transport matrix.
    """
    async with (
        endpoint("complete") as (url, paths, _writes),
        openai.AsyncOpenAI(api_key="offline", base_url=url, max_retries=0) as sdk,
    ):
        provider = OpenAIResponsesProvider("test", client=cast(Any, sdk))
        policy = ModelCallPolicy(
            max_request_seconds=5,
            max_attempt_seconds=5,
            first_progress_timeout_seconds=4,
            progress_idle_timeout_seconds=3,
            cleanup_timeout_seconds=0.2,
        )
        async with asyncio.timeout(6):
            events = [
                event
                async for event in provider.stream(
                    replace(request(), call_policy=policy)
                )
            ]
        assert isinstance(events[-1], ModelStreamCompleted)
        assert len(paths) == 1
        await provider.close()
