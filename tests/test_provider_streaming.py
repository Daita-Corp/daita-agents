from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing
from typing import Any, cast

import pytest
import httpx
import openai

from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from daita.llm.factory import create_llm_provider
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
    TextBlock,
    ToolDefinition,
)
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.grok import GrokProvider
from daita.llm.providers.ollama import OllamaProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider


class _NativeStream(AsyncIterator[object]):
    def __init__(self, events: Sequence[object]) -> None:
        self._events = iter(events)
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True

    def __aiter__(self) -> _NativeStream:
        return self

    async def __anext__(self) -> object:
        try:
            value = next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, asyncio.Event):
            value.set()
            await asyncio.Event().wait()
        return value


class _AnthropicStreamManager:
    def __init__(self, events: Sequence[object]) -> None:
        self._stream = _NativeStream(events)

    async def __aenter__(self) -> AsyncIterator[object]:
        return self._stream

    async def __aexit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> bool | None:
        await self._stream.close()
        return None


class _OpenAIResponsesResource:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = events
        self.arguments: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.arguments = kwargs
        return _NativeStream(self._events)


class _OpenAIClient:
    def __init__(self, events: Sequence[object]) -> None:
        self.responses = _OpenAIResponsesResource(events)


class _AnthropicMessagesResource:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = events
        self.arguments: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        raise AssertionError("streaming test called the atomic Anthropic API")

    def stream(self, **kwargs: object) -> _AnthropicStreamManager:
        self.arguments = kwargs
        return _AnthropicStreamManager(self._events)


class _AnthropicClient:
    def __init__(self, events: Sequence[object]) -> None:
        self.messages = _AnthropicMessagesResource(events)


class _GeminiModels:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = events
        self.arguments: dict[str, object] | None = None

    async def generate_content(self, **kwargs: object) -> object:
        raise AssertionError("streaming test called the atomic Gemini API")

    async def generate_content_stream(self, **kwargs: object) -> object:
        self.arguments = kwargs
        return _NativeStream(self._events)


class _GeminiAsyncClient:
    def __init__(self, events: Sequence[object]) -> None:
        self.models = _GeminiModels(events)


class _GeminiClient:
    def __init__(self, events: Sequence[object]) -> None:
        self.aio = _GeminiAsyncClient(events)


class _CompletionsResource:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = events
        self.arguments: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.arguments = kwargs
        return _NativeStream(self._events)


class _ChatResource:
    def __init__(self, events: Sequence[object]) -> None:
        self.completions = _CompletionsResource(events)


class _CompatibleClient:
    def __init__(self, events: Sequence[object]) -> None:
        self.chat = _ChatResource(events)


def _request(*, tools: bool = False) -> ModelRequest:
    definitions = (
        (
            ToolDefinition(
                name="catalog_schema",
                description="Read catalog schema",
                input_schema={"type": "object", "properties": {}},
            ),
        )
        if tools
        else ()
    )
    return ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Hello"),),
            ),
        ),
        tools=definitions,
    )


async def _events(provider: object, request: ModelRequest) -> list[ModelStreamEvent]:
    stream = getattr(provider, "stream")
    return [event async for event in stream(request)]


def _openai_text_response(text: str) -> dict[str, object]:
    return {
        "id": "resp-1",
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": ""},
                    {"type": "output_text", "text": text},
                ],
            }
        ],
        "usage": None,
    }


@pytest.mark.parametrize(
    "kind", ("openai", "anthropic", "gemini", "custom", "grok", "ollama")
)
@pytest.mark.parametrize("mode", ("complete", "failure", "cancel", "early_exit"))
async def test_native_streams_release_resources_on_every_exit(monkeypatch, kind, mode):
    opened: list[_NativeStream] = []
    original_init = _NativeStream.__init__

    def track(self, events):
        original_init(self, events)
        opened.append(self)

    monkeypatch.setattr(_NativeStream, "__init__", track)
    prefix: tuple[object, ...]
    tail: tuple[object, ...]
    make: Callable[[Sequence[object]], Any]
    if kind == "openai":
        prefix = ({"type": "response.output_text.delta", "delta": "Hello"},)
        tail = (
            {"type": "response.completed", "response": _openai_text_response("Hello")},
        )
        make = lambda events: OpenAIResponsesProvider(
            "test-model", client=cast(Any, _OpenAIClient(events))
        )
    elif kind == "anthropic":
        prefix = (
            _anthropic_message_start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        )
        tail = (
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        )
        make = lambda events: AnthropicMessagesProvider(
            "test-model", client=cast(Any, _AnthropicClient(events))
        )
    elif kind == "gemini":
        prefix = ({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},)
        tail = ({"candidates": [{"finish_reason": "STOP", "content": None}]},)
        make = lambda events: GeminiProvider(
            "test-model", client=cast(Any, _GeminiClient(events))
        )
    else:
        prefix = (
            {
                "id": "response-1",
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "Hello"}}],
            },
        )
        tail = ({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},)
        make = lambda events: _compatible_provider(kind, events)
    blocked = asyncio.Event()
    failure = ModelProviderError(
        ProviderErrorCode.PROVIDER_UNAVAILABLE, "offline stream failure"
    )
    if mode == "failure":
        tail = (failure,)
    elif mode == "cancel":
        tail = (blocked,)
    provider = make(prefix + tail)
    async with aclosing(cast(Any, provider.stream(_request()))) as stream:
        assert isinstance(await anext(stream), ModelTextDelta)
        if mode == "failure":
            with pytest.raises(ModelProviderError) as caught:
                await anext(stream)
            assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
        elif mode == "cancel":
            pending = asyncio.ensure_future(anext(stream))
            await blocked.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        elif mode == "complete":
            assert isinstance(await anext(stream), ModelStreamCompleted)
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
    assert len(opened) == 1
    assert opened[0].closed


async def test_openai_native_stream_ignores_empty_deltas_and_uses_terminal_response():
    client = _OpenAIClient(
        (
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": ""},
            {"type": "response.output_text.delta", "delta": "Hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {
                "type": "response.completed",
                "response": _openai_text_response("Hello"),
            },
        )
    )
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(Any, client),
    )

    events = await _events(provider, _request())

    assert [event.text for event in events if isinstance(event, ModelTextDelta)] == [
        "Hel",
        "lo",
    ]
    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.text == "Hello"
    assert client.responses.arguments is not None
    assert client.responses.arguments["stream"] is True


async def test_openai_native_tool_stream_ignores_empty_argument_delta():
    terminal_response = {
        "id": "resp-tool",
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "function_call",
                "call_id": "provider-call-1",
                "name": "catalog_schema",
                "arguments": "{}",
            }
        ],
        "usage": None,
    }
    client = _OpenAIClient(
        (
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "",
                    "name": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "",
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{}",
            },
            {"type": "response.function_call_arguments.done"},
            {"type": "response.completed", "response": terminal_response},
        )
    )
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(Any, client),
        id_factory=lambda _prefix: "call-1",
    )

    events = await _events(provider, _request(tools=True))

    tool_deltas = [event for event in events if isinstance(event, ModelToolCallDelta)]
    assert [event.arguments_delta for event in tool_deltas] == ["", "{}"]
    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.tool_calls[0].id == "call-1"
    assert completed.response.tool_calls[0].provider_call_id == "provider-call-1"
    assert dict(completed.response.tool_calls[0].arguments) == {}


async def test_openai_native_stream_still_rejects_non_text_delta():
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(
            Any,
            _OpenAIClient(({"type": "response.output_text.delta", "delta": None},)),
        ),
    )

    with pytest.raises(ModelProviderError) as caught:
        await _events(provider, _request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert caught.value.provider_id == "openai:test-model"
    assert caught.value.diagnostic == ProviderFailureDiagnostic(
        phase=ProviderFailurePhase.STREAM_EVENT,
        code="event_decode_failed",
        event_type="response.output_text.delta",
    )


async def test_openai_stream_reconstructs_official_completed_item_when_terminal_output_is_empty():
    response_id = "resp-official-stream"
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(
            Any,
            _OpenAIClient(
                (
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": "message-official-stream",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "completed text",
                                }
                            ],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "status": "completed",
                            "model": "test-model",
                            "output": [],
                            "usage": None,
                        },
                    },
                )
            ),
        ),
    )

    events = await _events(provider, _request())

    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.text == "completed text"
    assert completed.response.provider_response_id == response_id


async def test_openai_stream_prefers_official_completed_item_over_terminal_placeholder():
    response_id = "resp-terminal-placeholder"
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(
            Any,
            _OpenAIClient(
                (
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": "message-completed",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "completed text",
                                }
                            ],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "status": "completed",
                            "model": "test-model",
                            "output": [
                                {
                                    "id": "message-placeholder",
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                }
                            ],
                            "usage": None,
                        },
                    },
                )
            ),
        ),
    )

    events = await _events(provider, _request())

    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.text == "completed text"
    assert completed.response.provider_response_id == response_id


async def test_openai_malformed_terminal_retains_only_bounded_structure():
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(
            Any,
            _OpenAIClient(
                (
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp-reasoning-only",
                            "status": "completed",
                            "model": "test-model",
                            "output": [
                                {
                                    "id": "reasoning-1",
                                    "type": "reasoning",
                                    "summary": [],
                                }
                            ],
                            "usage": None,
                        },
                    },
                )
            ),
        ),
    )

    with pytest.raises(ModelProviderError) as caught:
        await _events(provider, _request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert caught.value.diagnostic == ProviderFailureDiagnostic(
        phase=ProviderFailurePhase.STREAM_TERMINAL,
        code="terminal_content_missing",
        event_type="response.completed",
        terminal_status="completed",
        output_item_types=("reasoning",),
        response_id_digest=(
            "sha256:9a6af422fc70773bc86cae3a1e4b86793d8a3c26656b2ded26c0ee7087e6c13a"
        ),
    )


async def test_openai_empty_completed_message_reports_missing_terminal_content():
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(
            Any,
            _OpenAIClient(
                (
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp-empty-message",
                            "status": "completed",
                            "model": "test-model",
                            "output": [
                                {
                                    "id": "message-empty",
                                    "type": "message",
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": [],
                                }
                            ],
                            "usage": None,
                        },
                    },
                )
            ),
        ),
    )

    with pytest.raises(ModelProviderError) as caught:
        await _events(provider, _request())

    diagnostic = caught.value.diagnostic
    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert diagnostic is not None
    assert diagnostic.phase is ProviderFailurePhase.STREAM_TERMINAL
    assert diagnostic.code == "terminal_content_missing"
    assert diagnostic.event_type == "response.completed"
    assert diagnostic.terminal_status == "completed"
    assert diagnostic.output_item_types == ("message",)
    assert diagnostic.response_id_digest is not None


@pytest.mark.parametrize(
    ("response_patch", "diagnostic_code"),
    (
        ({"model": None}, "response_metadata_invalid"),
        (
            {
                "usage": {
                    "input_tokens": None,
                    "output_tokens": 1,
                }
            },
            "usage_invalid",
        ),
    ),
)
async def test_openai_terminal_decode_reports_exact_bounded_checkpoint(
    response_patch: dict[str, object],
    diagnostic_code: str,
):
    response: dict[str, object] = {
        "id": "resp-invalid-checkpoint",
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "id": "message-valid",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
        "usage": None,
    }
    response.update(response_patch)
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(
            Any,
            _OpenAIClient(
                (
                    {
                        "type": "response.completed",
                        "response": response,
                    },
                )
            ),
        ),
    )

    with pytest.raises(ModelProviderError) as caught:
        await _events(provider, _request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert caught.value.diagnostic is not None
    assert caught.value.diagnostic.code == diagnostic_code
    assert caught.value.diagnostic.output_item_types == ("message",)


def _anthropic_message_start() -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            "id": "message-1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "test-model",
            "stop_reason": None,
            "usage": {"input_tokens": 1},
        },
    }


async def test_anthropic_native_stream_handles_forward_events_and_fallback_blocks():
    client = _AnthropicClient(
        (
            _anthropic_message_start(),
            {"type": "future_metadata", "value": "ignored"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "fallback"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        )
    )
    provider = AnthropicMessagesProvider(
        "test-model",
        client=cast(Any, client),
    )

    events = await _events(provider, _request())

    assert [event.text for event in events if isinstance(event, ModelTextDelta)] == [
        "Hello"
    ]
    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.text == "Hello"
    assert client.messages.arguments is not None


async def test_anthropic_native_no_argument_tool_call_finishes_as_empty_object():
    client = _AnthropicClient(
        (
            _anthropic_message_start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "provider-call-1",
                    "name": "catalog_schema",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ""},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        )
    )
    provider = AnthropicMessagesProvider(
        "test-model",
        client=cast(Any, client),
        id_factory=lambda _prefix: "call-1",
    )

    events = await _events(provider, _request(tools=True))

    tool_deltas = [event for event in events if isinstance(event, ModelToolCallDelta)]
    assert [event.arguments_delta for event in tool_deltas] == ["", ""]
    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.tool_calls[0].id == "call-1"
    assert dict(completed.response.tool_calls[0].arguments) == {}


async def test_gemini_native_stream_ignores_metadata_and_empty_text_chunks():
    client = _GeminiClient(
        (
            {"candidates": []},
            {
                "response_id": "response-1",
                "model_version": "test-model",
                "candidates": [{"content": {"parts": [{"text": ""}]}}],
            },
            {
                "response_id": "response-1",
                "model_version": "test-model",
                "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            },
            {
                "response_id": "response-1",
                "model_version": "test-model",
                "candidates": [{"finish_reason": "STOP", "content": None}],
            },
        )
    )
    provider = GeminiProvider("test-model", client=cast(Any, client))

    events = await _events(provider, _request())

    assert [event.text for event in events if isinstance(event, ModelTextDelta)] == [
        "Hello"
    ]
    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.text == "Hello"
    assert completed.response.provider_response_id == "response-1"
    assert client.aio.models.arguments is not None


def _compatible_provider(
    kind: str,
    events: Sequence[object],
) -> OpenAICompatibleProvider:
    client = cast(Any, _CompatibleClient(events))
    if kind == "custom":
        return OpenAICompatibleProvider(
            "test-model",
            provider="custom",
            base_url="https://models.example.test/v1",
            client=client,
        )
    if kind == "grok":
        return GrokProvider("test-model", client=client)
    if kind == "ollama":
        return OllamaProvider("test-model", client=client)
    raise AssertionError(f"unknown compatible provider kind: {kind}")


@pytest.mark.parametrize("kind", ("custom", "grok", "ollama"))
async def test_compatible_native_streams_ignore_empty_chunks_and_reconcile(kind: str):
    provider = _compatible_provider(
        kind,
        (
            {"id": "response-1", "model": "test-model"},
            {
                "id": "response-1",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "response-1",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "response-1",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            },
        ),
    )

    events = await _events(provider, _request())

    assert [event.text for event in events if isinstance(event, ModelTextDelta)] == [
        "Hello"
    ]
    completed = cast(ModelStreamCompleted, events[-1])
    assert completed.response.text == "Hello"
    assert completed.response.provider_id == f"{kind}:test-model"


async def test_compatible_native_no_argument_tool_call_finishes_as_empty_object():
    provider = _compatible_provider(
        "custom",
        (
            {
                "id": "response-tool",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "provider-call-1",
                                    "function": {
                                        "name": "catalog_schema",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "response-tool",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ),
    )

    events = await _events(provider, _request(tools=True))

    completed = cast(ModelStreamCompleted, events[-1])
    assert dict(completed.response.tool_calls[0].arguments) == {}


@pytest.mark.parametrize(
    ("provider_factory", "provider_id"),
    (
        (
            lambda: OpenAIResponsesProvider(
                "test-model",
                client=cast(Any, _OpenAIClient(())),
            ),
            "openai:test-model",
        ),
        (
            lambda: AnthropicMessagesProvider(
                "test-model",
                client=cast(Any, _AnthropicClient(())),
            ),
            "anthropic:test-model",
        ),
        (
            lambda: GeminiProvider(
                "test-model",
                client=cast(Any, _GeminiClient(())),
            ),
            "gemini:test-model",
        ),
        (
            lambda: OpenAICompatibleProvider(
                "test-model",
                provider="custom",
                base_url="https://models.example.test/v1",
                client=cast(Any, _CompatibleClient(())),
            ),
            "custom:test-model",
        ),
    ),
)
async def test_native_streams_require_canonical_terminal_completion(
    provider_factory: Callable[[], object],
    provider_id: str,
):
    with pytest.raises(ModelProviderError) as caught:
        await _events(provider_factory(), _request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert caught.value.provider_id == provider_id
    assert caught.value.diagnostic == ProviderFailureDiagnostic(
        phase=ProviderFailurePhase.STREAM_TERMINAL,
        code="terminal_completion_missing",
    )


@pytest.mark.parametrize(
    ("model_id", "base_url"),
    (
        ("openai:test-model", None),
        ("anthropic:test-model", None),
        ("gemini:test-model", None),
        ("grok:test-model", None),
        ("ollama:test-model", "http://127.0.0.1:11434/v1"),
    ),
)
def test_every_builtin_provider_constructs_a_lazy_streaming_adapter(
    model_id: str,
    base_url: str | None,
):
    provider = create_llm_provider(model_id, base_url=base_url)

    assert callable(getattr(provider, "stream", None))
