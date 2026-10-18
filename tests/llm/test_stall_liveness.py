"""Provider liveness resets the stall window without counting as model output."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace

import httpx
import pytest

from daita.llm import ModelCallPolicy
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import ModelStreamCompleted
from daita.llm.routing import ModelRouter, RetryPolicy
from tests.llm._call_policy_support import request
from tests.llm._routing_support import registration
from tests.llm._streaming_support import _openai_text_response
from tests.llm._token_counting_support import count_response, is_count, provider_at

# Scaled down: 3s without output is past the 1.5s first-progress window. The
# window leaves room for cold SDK imports and validation on a loaded machine.
POLICY = ModelCallPolicy(
    max_request_seconds=20,
    max_attempt_seconds=10,
    first_progress_timeout_seconds=1.5,
    progress_idle_timeout_seconds=1.5,
    cleanup_timeout_seconds=0.1,
)
SHORT_ATTEMPT = replace(POLICY, max_request_seconds=8, max_attempt_seconds=4)


def sse(event: Mapping[str, object], name: str | None = None) -> bytes:
    prefix = "" if name is None else f"event: {name}\n"
    return (prefix + "data: " + json.dumps(event) + "\n\n").encode()


def openai_reasoning(completes: bool):
    reasoning = {"type": "reasoning", "id": "rs_offline", "summary": []}
    terminal = _openai_text_response("done")
    output = terminal["output"]
    assert isinstance(output, list)
    terminal["output"] = [reasoning, *output]
    terminal["usage"] = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    async def body():
        yield sse({"type": "response.created", "response": {"id": terminal["id"]}})
        yield sse(
            {"type": "response.output_item.added", "output_index": 0, "item": reasoning}
        )
        for _ in range(30 if completes else 1000):
            await asyncio.sleep(0.1)
            yield b": keepalive\n\n"  # An SSE comment the SDK never yields.
        yield sse(
            {"type": "response.output_item.done", "output_index": 0, "item": reasoning}
        )
        yield sse({"type": "response.completed", "response": terminal})

    return body


def anthropic_thinking():
    async def body():
        yield sse(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_offline",
                    "type": "message",
                    "role": "assistant",
                    "model": "fixture-model",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                },
            },
            "message_start",
        )
        for _ in range(30):
            await asyncio.sleep(0.1)
            yield sse({"type": "ping"}, "ping")  # Discarded inside the SDK.
        for name, event in (
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "done"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 2},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ):
            yield sse(event, name)

    return body


def silent():
    async def body():
        await asyncio.Event().wait()
        yield b""

    return body


async def run(kind: str, body, policy: ModelCallPolicy = POLICY):
    paths: list[str] = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for chunk in body():
                yield chunk

    async def respond(native_request):
        paths.append(native_request.url.path)
        if is_count(native_request.url.path):
            return httpx.Response(200, json=count_response(kind, 3))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Body()
        )

    async with provider_at(kind, respond) as adapter:
        router = ModelRouter(
            (registration(adapter, streaming=True),),
            retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
        )
        try:
            async with asyncio.timeout(
                30
            ):  # Harness guard, never the production bound.
                events = [
                    event
                    async for event in router.stream(
                        request(call_policy=policy, max_total_tokens=2000)
                    )
                ]
        finally:
            await router.close()
    return events, [path for path in paths if not is_count(path)]


@pytest.mark.parametrize(
    "kind,body",
    [("openai", openai_reasoning(True)), ("anthropic", anthropic_thinking())],
)
async def test_silent_reasoning_with_live_stream_is_not_a_stall(kind, body):
    events, generations = await run(kind, body)

    assert len(generations) == 1
    completed = events[-1]
    assert isinstance(completed, ModelStreamCompleted)
    assert completed.response.text == "done"
    assert completed.response.usage.total_tokens == 5


async def test_silent_stream_still_stalls_and_replays():
    with pytest.raises(ModelProviderError) as caught:
        await run("openai", silent())

    diagnostic = caught.value.diagnostic
    assert caught.value.code is ProviderErrorCode.TIMEOUT
    assert diagnostic is not None and diagnostic.attempt is not None
    assert diagnostic.code == "first_progress"
    assert diagnostic.attempt["total_attempt_number"] == 2


async def test_endless_keepalives_are_bounded_by_the_attempt_deadline():
    started = asyncio.get_running_loop().time()
    with pytest.raises(ModelProviderError) as caught:
        await run("openai", openai_reasoning(False), SHORT_ATTEMPT)

    diagnostic = caught.value.diagnostic
    assert caught.value.code is ProviderErrorCode.TIMEOUT
    assert diagnostic is not None and diagnostic.attempt is not None
    assert diagnostic.code == "attempt_deadline"
    # A deadline expiry may have generated output, so it is never replayed.
    assert diagnostic.attempt["total_attempt_number"] == 1
    assert asyncio.get_running_loop().time() - started < 8
