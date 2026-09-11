from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Any, cast

import pytest

from daita.llm.errors import (
    ModelProviderError,
    ProviderAttempt,
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
from tests.llm._streaming_support import _openai_text_response


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


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("fault", ["cleanup", "diagnostic", "both"])
async def test_attempt_completion_survives_cleanup_and_diagnostic_faults(
    monkeypatch, kind, fault
):
    from daita.llm._lifecycle import closing_stream
    from tests.support.job_benchmarks import RecordingProvider

    if fault in {"cleanup", "both"}:

        async def fail_close(self):
            self.closed = True
            raise RuntimeError("private cleanup details must not escape")

        monkeypatch.setattr(_NativeStream, "close", fail_close)
        monkeypatch.setattr(_NativeStream, "aclose", fail_close)
    if fault in {"diagnostic", "both"}:

        def fail_snapshot(self):
            raise ValueError("diagnostic extraction failed")

        monkeypatch.setattr(ProviderAttempt, "snapshot", fail_snapshot)
    provider: Any
    if kind == "openai":
        terminal = _openai_text_response("Done.")
        terminal["usage"] = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        provider = OpenAIResponsesProvider(
            "test-model",
            client=cast(
                Any,
                _OpenAIClient(({"type": "response.completed", "response": terminal},)),
            ),
        )
    elif kind == "anthropic":
        start = _anthropic_message_start()
        cast(dict[str, object], start["message"])["usage"] = {
            "input_tokens": 3,
            "output_tokens": 0,
        }
        provider = AnthropicMessagesProvider(
            "test-model",
            client=cast(
                Any,
                _AnthropicClient(
                    (
                        start,
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": "Done."},
                        },
                        {"type": "content_block_stop", "index": 0},
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn"},
                            "usage": {"output_tokens": 2},
                        },
                        {"type": "message_stop"},
                    )
                ),
            ),
        )
    else:
        provider = GeminiProvider(
            "test-model",
            client=cast(
                Any,
                _GeminiClient(
                    (
                        {
                            "candidates": [
                                {
                                    "finish_reason": "STOP",
                                    "content": {"parts": [{"text": "Done."}]},
                                }
                            ],
                            "usage_metadata": {
                                "prompt_token_count": 3,
                                "candidates_token_count": 2,
                                "total_token_count": 5,
                            },
                        },
                    )
                ),
            ),
        )
    recorder = RecordingProvider(provider)
    events = []
    try:
        async with closing_stream(recorder.stream(_request())) as stream:
            async for event in stream:
                events.append(event)
    except ModelProviderError:
        assert fault in {"cleanup", "both"}
    completed = [event for event in events if isinstance(event, ModelStreamCompleted)]
    assert len(completed) == (0 if fault in {"cleanup", "both"} else 1)
    if completed:
        assert completed[0].response.usage.total_tokens == 5
    assert len(recorder.timings) == 1
    assert recorder.timings[0]["usage_complete"] is True
    observation = cast(Mapping[str, object], recorder.timings[0]["attempt_diagnostic"])
    if fault == "cleanup":
        assert observation["terminal_observed"] is True
        assert observation["cleanup_failure"] == "cleanup_failed"
        assert observation["cleanup_finished_seconds"] is not None
    else:
        assert observation == {"measurement_availability": "unavailable"}


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
    assert caught.value.diagnostic is not None
    assert caught.value.diagnostic.attempt is not None
    assert replace(caught.value.diagnostic, attempt=None) == ProviderFailureDiagnostic(
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
    assert caught.value.diagnostic is not None
    assert caught.value.diagnostic.attempt is not None
    assert replace(caught.value.diagnostic, attempt=None) == ProviderFailureDiagnostic(
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
    assert caught.value.diagnostic is not None
    if provider_id.split(":")[0] in {"openai", "anthropic", "gemini"}:
        assert caught.value.diagnostic.attempt is not None
    assert replace(caught.value.diagnostic, attempt=None) == ProviderFailureDiagnostic(
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


def test_attempt_observation_bounds_and_unsafe_correlation_are_inert():
    from daita._json import canonical_json
    from daita.llm.errors import take_provider_attempt_diagnostic

    attempt = ProviderAttempt(
        ModelRequest(
            messages=(
                CanonicalMessage(
                    MessageRole.USER, content=(TextBlock("private prompt"),)
                ),
            )
        ),
        headers_supported=True,
    )
    attempt.headers("generation", 200, "a" * 257)
    assert attempt.snapshot()["generation_request_id_digest"] is None
    attempt.headers("generation", 200, "key=secret&credential=value")
    for index in range(64):
        attempt.native(f"event_{index}", "private response body / invalid id")
    attempt.native("event_0")
    attempt.finish(None)
    snapshot = take_provider_attempt_diagnostic()
    assert snapshot is not None
    counts = snapshot["native_event_counts"]
    assert isinstance(counts, Mapping) and len(counts) == 16
    assert counts["event_0"] == 2 and snapshot["native_event_overflow"] == 48
    encoded = canonical_json(snapshot)
    assert len(encoded.encode()) <= 8192
    assert "secret" not in encoded and "private" not in encoded
    assert take_provider_attempt_diagnostic() is None


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("stop", ["deadline", "cancel"])
@pytest.mark.parametrize("progress", ["silent", "arguments"])
async def test_actual_sdk_silent_stream_stops_without_tool_dispatch(
    routed, stop, progress
):
    """Contain the captured partial-argument symptom; this is not a causal repair."""
    import json
    from decimal import Decimal

    import httpx

    from daita.llm.routing import ModelRouter, RetryPolicy
    from daita.loop import AgentLoop, InMemoryTranscriptStore, LoopLimits, RunInput
    from tests.llm._routing_support import registration
    from tests.llm._token_counting_support import count_response, is_count, provider_at
    from tests.support.job_benchmarks import RecordingProvider
    from tests.support.loop import NOW, ScriptedTools, TranscriptContext

    paths = []
    released = []
    blocked = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if progress == "arguments":
                events = [
                    {
                        "type": "response.created",
                        "response": {"id": "resp_offline", "status": "in_progress"},
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "id": "fc_offline",
                            "call_id": "call_offline",
                            "name": "lookup",
                            "arguments": "",
                        },
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 0,
                        "item_id": "fc_offline",
                        "delta": "{",
                    },
                ]
                for event in events:
                    yield ("data: " + json.dumps(event) + "\n\n").encode()
            blocked.set()
            await asyncio.Event().wait()

        async def aclose(self):
            released.append(True)

    async def respond(request):
        paths.append(request.url.path)
        if is_count(request.url.path):
            await asyncio.sleep(0.05)  # Counting consumes the same absolute deadline.
            return httpx.Response(200, json=count_response("openai", 500))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Body()
        )

    async with provider_at("openai", respond) as adapter:
        recorder = RecordingProvider(adapter)
        entry = registration(recorder, streaming=True)
        router = ModelRouter(
            (replace(entry, profile=replace(entry.profile, supports_tools=True)),),
            retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
        )
        store = InMemoryTranscriptStore()
        runtime = ScriptedTools({})
        loop = AgentLoop(
            model=router if routed else recorder,
            context_builder=TranscriptContext(),
            tools=runtime,
            transcripts=store,
            clock=lambda: NOW,
            stream_model_calls=True,
            limits=LoopLimits(
                max_wall_time_seconds=1.5,
                max_total_tokens=2000,
                max_estimated_cost_usd=Decimal("0.05"),
            ),
        )
        run = RunInput(
            id="silent-sdk",
            agent_id="agent-1",
            message="Read the value.",
            created_at=NOW,
        )
        started = asyncio.get_running_loop().time()
        task = asyncio.create_task(loop.run(run))
        try:
            async with asyncio.timeout(3):
                await blocked.wait()
                if stop == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    await task
            result = await store.result(run.id)
            assert result is not None and result.kind.value == (
                "interrupted" if stop == "cancel" else "failed"
            )
            assert result.final_text is None
            assert result.usage.cost_estimate.status.value != "complete"
            assert not runtime.calls
            transcript = await store.load(run.id)
            assert (
                not transcript.tool_pairs
            )  # Partial arguments are never an executable call.
            assert len(recorder.requests) == 1 and not recorder.responses
            assert len(paths) == 2 and is_count(paths[0]) and not is_count(paths[1])
            request = recorder.requests[0]
            assert request.deadline is not None
            assert started <= request.deadline - 1.5 <= started + 0.1
            timing = recorder.timings[0]
            assert timing["usage_complete"] is False
            diagnostic = timing["attempt_diagnostic"]
            assert isinstance(diagnostic, Mapping)
            assert diagnostic["count_state"] == "succeeded"
            assert diagnostic["counted_input_tokens"] == 500
            assert diagnostic["generation_http_status"] == 200
            assert diagnostic["terminal_observed"] is False
            assert diagnostic["cleanup_finished_seconds"] is not None
            assert (diagnostic["first_native_event_seconds"] is not None) is (
                progress == "arguments"
            )
            assert diagnostic["transport_error_kind"] is None
            assert released == [True]
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await router.close()


async def test_gemini_repeated_cumulative_function_snapshots_emit_one_final_call(
    monkeypatch,
):
    from daita.llm._lifecycle import AttemptLifecycle

    progress = []
    original = AttemptLifecycle.progress

    def record(self, fragment):
        progress.append(fragment)
        original(self, fragment)

    monkeypatch.setattr(AttemptLifecycle, "progress", record)

    def snapshot(value):
        return {
            "response_id": "response-1",
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "id": "native-1",
                                    "name": "lookup",
                                    "args": {"value": value},
                                }
                            }
                        ]
                    }
                }
            ],
        }

    provider = GeminiProvider(
        "test-model",
        client=cast(
            Any,
            _GeminiClient(
                (
                    snapshot("a"),
                    snapshot("a"),
                    snapshot("ab"),
                    snapshot("ab"),
                    {
                        "response_id": "response-1",
                        "candidates": [{"finish_reason": "STOP", "content": None}],
                    },
                )
            ),
        ),
    )
    events = await _events(provider, _request())
    calls = [event for event in events if isinstance(event, ModelToolCallDelta)]
    assert len(calls) == 1
    assert calls[0].arguments_delta == '{"value":"ab"}'
    terminal = cast(ModelStreamCompleted, events[-1])
    assert dict(terminal.response.tool_calls[0].arguments) == {"value": "ab"}
    assert progress == ['{"value":"a"}', '{"value":"ab"}']


@pytest.mark.parametrize(
    "bad_event",
    [
        {"type": "response.output_text.delta", "sequence_number": 1, "delta": "late"},
        {"type": "response.in_progress", "response": {"id": "changed"}},
    ],
)
async def test_openai_validates_native_identity_and_sequence_before_progress(bad_event):
    provider = OpenAIResponsesProvider(
        "test-model",
        client=cast(
            Any,
            _OpenAIClient(
                (
                    {
                        "type": "response.created",
                        "sequence_number": 1,
                        "response": {"id": "original"},
                    },
                    bad_event,
                )
            ),
        ),
    )
    with pytest.raises(ModelProviderError) as caught:
        await _events(provider, _request())
    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert (
        caught.value.diagnostic is not None
        and caught.value.diagnostic.attempt is not None
    )
    assert caught.value.diagnostic.attempt["first_substantive_progress_seconds"] is None


@pytest.mark.parametrize("family", ["openai", "anthropic", "gemini", "custom"])
@pytest.mark.parametrize("substantive", [False, True])
async def test_reasoning_progress_requires_recognized_nonempty_native_content(
    family, substantive
):
    provider: Any
    marker = "observable-reasoning-fragment" if substantive else ""
    if family == "openai":
        provider = OpenAIResponsesProvider(
            "test-model",
            client=cast(
                Any,
                _OpenAIClient(
                    (
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "delta": marker,
                        },
                    )
                ),
            ),
        )
    elif family == "anthropic":
        provider = AnthropicMessagesProvider(
            "test-model",
            client=cast(
                Any,
                _AnthropicClient(
                    (
                        {
                            "type": "message_start",
                            "message": {
                                "id": "r",
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "stop_reason": None,
                            },
                        },
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "",
                            },
                        },
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "thinking_delta", "thinking": marker},
                        },
                    )
                ),
            ),
        )
    elif family == "gemini":
        provider = GeminiProvider(
            "test-model",
            client=cast(
                Any,
                _GeminiClient(
                    (
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [{"thought": True, "text": marker}]
                                    }
                                }
                            ]
                        },
                    )
                ),
            ),
        )
    else:
        provider = _compatible_provider(
            "custom",
            (
                {
                    "id": "r",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": marker},
                            "finish_reason": None,
                        }
                    ],
                },
            ),
        )
    with pytest.raises(ModelProviderError) as caught:
        await _events(provider, _request())
    assert (
        caught.value.diagnostic is not None
        and caught.value.diagnostic.attempt is not None
    )
    diagnostic = caught.value.diagnostic.attempt
    assert (diagnostic["first_substantive_progress_seconds"] is not None) == substantive
    assert "observable-reasoning-fragment" not in str(diagnostic.to_dict())
    assert diagnostic["canonical_emitted"] is False
