from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, cast

import pytest

from daita.llm.errors import ModelProviderError, ProviderErrorCode
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

    def __aiter__(self) -> _NativeStream:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


class _AnthropicStreamManager:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = events

    async def __aenter__(self) -> AsyncIterator[object]:
        return _NativeStream(self._events)

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None:
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
    "provider_factory",
    (
        lambda: OpenAIResponsesProvider(
            "test-model",
            client=cast(Any, _OpenAIClient(())),
        ),
        lambda: AnthropicMessagesProvider(
            "test-model",
            client=cast(Any, _AnthropicClient(())),
        ),
        lambda: GeminiProvider(
            "test-model",
            client=cast(Any, _GeminiClient(())),
        ),
        lambda: OpenAICompatibleProvider(
            "test-model",
            provider="custom",
            base_url="https://models.example.test/v1",
            client=cast(Any, _CompatibleClient(())),
        ),
    ),
)
async def test_native_streams_require_canonical_terminal_completion(
    provider_factory: Callable[[], object],
):
    with pytest.raises(ModelProviderError) as caught:
        await _events(provider_factory(), _request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE


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
