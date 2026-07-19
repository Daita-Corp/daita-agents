from __future__ import annotations

import asyncio
import builtins
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest

from daita._json import FrozenJsonObject
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelToolCallDelta,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.anthropic import AnthropicMessagesProvider


class FakeMessages:
    def __init__(
        self,
        *items: object,
        streams: Sequence[object] = (),
    ) -> None:
        self._items = list(items)
        self._streams = list(streams)
        self.calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def stream(self, **kwargs: object) -> FakeStreamContext:
        self.stream_calls.append(kwargs)
        return FakeStreamContext(self._streams.pop(0))


class FakeNativeStream:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> FakeNativeStream:
        return self

    async def __anext__(self) -> object:
        try:
            event = next(self._events)
        except StopIteration as error:
            raise StopAsyncIteration from error
        if isinstance(event, BaseException):
            raise event
        return event


class FakeStreamContext:
    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> FakeNativeStream:
        if isinstance(self._value, BaseException):
            raise self._value
        if not isinstance(self._value, Sequence) or isinstance(
            self._value,
            (str, bytes),
        ):
            raise TypeError("fake stream must be an event sequence")
        return FakeNativeStream(self._value)

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None


class FakeClient:
    def __init__(
        self,
        *items: object,
        streams: Sequence[object] = (),
    ) -> None:
        self.messages = FakeMessages(*items, streams=streams)


class FakeApiError(Exception):
    def __init__(self, status_code: int, error_type: str | None = None) -> None:
        self.status_code = status_code
        self.body = None if error_type is None else {"error": {"type": error_type}}
        super().__init__(f"status {status_code}")


def _plain_json_object(value: Mapping[str, object]) -> dict[str, object]:
    assert isinstance(value, FrozenJsonObject)
    return value.to_dict()


def _message(role: MessageRole, text: str) -> CanonicalMessage:
    return CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=role,
        content=(TextBlock(text),),
    )


def _request(
    *messages: CanonicalMessage,
    tools: tuple[ToolDefinition, ...] = (),
    response_schema: Mapping[str, object] | None = None,
) -> ModelRequest:
    return ModelRequest(
        operation_id="operation-1",
        turn_id="turn-1",
        messages=messages or (_message(MessageRole.USER, "hello"),),
        tools=tools,
        response_schema=response_schema,
    )


def _usage(
    *,
    input_tokens: int = 11,
    output_tokens: int = 7,
    cache_read_input_tokens: int = 3,
    cache_creation_input_tokens: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )


def _response(
    *content: object,
    stop_reason: str = "end_turn",
    response_id: str = "message-1",
    usage: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        type="message",
        role="assistant",
        content=content or (SimpleNamespace(type="text", text="done"),),
        stop_reason=stop_reason,
        usage=_usage() if usage is None else usage,
    )


def _message_start(*, response_id: str = "message-stream") -> SimpleNamespace:
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            id=response_id,
            type="message",
            role="assistant",
            content=(),
            stop_reason=None,
            usage=_usage(output_tokens=1),
        ),
    )


def _block_start(index: int, block: object) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=block,
    )


def _block_delta(index: int, delta: object) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=delta,
    )


def _block_stop(index: int) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=index)


def _message_delta(stop_reason: str, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(
            stop_reason=stop_reason,
            stop_sequence=None,
        ),
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


async def _stream_events(
    provider: AnthropicMessagesProvider, request: ModelRequest
) -> list[object]:
    return [event async for event in provider.stream(request)]


async def test_text_translation_lifts_system_and_normalizes_cache_usage() -> None:
    client = FakeClient(_response())
    provider = AnthropicMessagesProvider(
        "claude-test",
        max_tokens=512,
        client=client,
    )

    response = await provider.generate(
        _request(
            _message(MessageRole.SYSTEM, "Be exact."),
            _message(MessageRole.USER, "hello"),
        )
    )

    assert provider.provider_id == "anthropic:claude-test"
    assert client.messages.calls == [
        {
            "model": "claude-test",
            "max_tokens": 512,
            "system": "Be exact.",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }
    ]
    assert response.finish_reason is FinishReason.STOP
    assert response.text == "done"
    assert response.provider_response_id == "message-1"
    assert response.usage.input_tokens == 19
    assert response.usage.output_tokens == 7
    assert response.usage.cache_read_tokens == 3
    assert response.usage.cache_write_tokens == 5


async def test_tools_use_native_blocks_and_replay_only_same_origin_metadata() -> None:
    tool = ToolDefinition(
        name="fake.read",
        description="Read a deterministic value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    )
    thinking = {
        "type": "thinking",
        "thinking": "opaque internal state",
        "signature": "signed-state",
    }
    first_raw = _response(
        thinking,
        SimpleNamespace(
            type="tool_use",
            id="provider-a",
            name="fake.read",
            input={"key": "alpha"},
        ),
        SimpleNamespace(
            type="tool_use",
            id="provider-b",
            name="fake.read",
            input={"key": "beta"},
        ),
        stop_reason="tool_use",
    )
    client = FakeClient(first_raw, _response(SimpleNamespace(type="text", text="42")))
    ids = iter(("call-a", "call-b"))
    provider = AnthropicMessagesProvider(
        "claude-test",
        client=client,
        id_factory=lambda _prefix: next(ids),
    )

    first = await provider.generate(_request(tools=(tool,)))

    assert client.messages.calls[0]["tools"] == [
        {
            "name": "fake.read",
            "description": "Read a deterministic value.",
            "input_schema": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        }
    ]
    assert first.finish_reason is FinishReason.TOOL_CALLS
    assert [
        (call.id, call.provider_call_id, call.arguments["key"])
        for call in first.tool_calls
    ] == [
        ("call-a", "provider-a", "alpha"),
        ("call-b", "provider-b", "beta"),
    ]
    assert _plain_json_object(first.provider_metadata) == {
        "anthropic_continuation": {
            "provider_id": "anthropic:claude-test",
            "content_blocks": [thinking],
        }
    }

    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=first.tool_calls,
        provider_id=first.provider_id,
        provider_metadata=first.provider_metadata,
    )
    tool_result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=(
            ToolResultBlock(call_id="call-a", output={"value": 42}),
            ToolResultBlock(call_id="call-b", output={}, is_error=True),
        ),
    )

    second = await provider.generate(
        _request(_message(MessageRole.USER, "read both"), assistant, tool_result)
    )

    assert second.text == "42"
    assert client.messages.calls[1]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "read both"}],
        },
        {
            "role": "assistant",
            "content": [
                thinking,
                {
                    "type": "tool_use",
                    "id": "provider-a",
                    "name": "fake.read",
                    "input": {"key": "alpha"},
                },
                {
                    "type": "tool_use",
                    "id": "provider-b",
                    "name": "fake.read",
                    "input": {"key": "beta"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "provider-a",
                    "content": '{"value":42}',
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "provider-b",
                    "content": "{}",
                    "is_error": True,
                },
            ],
        },
    ]


async def test_foreign_anthropic_metadata_is_not_replayed() -> None:
    call = ToolCall(
        id="canonical-call",
        provider_call_id="foreign-call",
        name="fake.read",
        arguments={"key": "alpha"},
    )
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        content=(TextBlock("prior"),),
        tool_calls=(call,),
        provider_id="anthropic:other-model",
        provider_metadata={
            "anthropic_continuation": {
                "provider_id": "anthropic:other-model",
                "content_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "must not leak",
                        "signature": "foreign",
                    }
                ],
            }
        },
    )
    tool_result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call.id, {"value": 7}),),
    )
    client = FakeClient(_response())
    provider = AnthropicMessagesProvider("claude-test", client=client)

    await provider.generate(
        _request(
            _message(MessageRole.USER, "first"),
            assistant,
            tool_result,
        )
    )

    native_messages = client.messages.calls[0]["messages"]
    assert isinstance(native_messages, list)
    assert native_messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "prior"},
            {
                "type": "tool_use",
                "id": "canonical-call",
                "name": "fake.read",
                "input": {"key": "alpha"},
            },
        ],
    }
    assert native_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "canonical-call",
                "content": '{"value":7}',
                "is_error": False,
            }
        ],
    }


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            FakeApiError(401, "authentication_error"),
            ProviderErrorCode.AUTHENTICATION_ERROR,
        ),
        (FakeApiError(429, "rate_limit_error"), ProviderErrorCode.RATE_LIMIT_ERROR),
        (FakeApiError(404, "not_found_error"), ProviderErrorCode.MODEL_NOT_FOUND),
        (FakeApiError(413, "request_too_large"), ProviderErrorCode.CONTEXT_OVERFLOW),
        (FakeApiError(400, "invalid_request_error"), ProviderErrorCode.INVALID_REQUEST),
        (FakeApiError(400, "content_blocked"), ProviderErrorCode.CONTENT_BLOCKED),
        (FakeApiError(529, "overloaded_error"), ProviderErrorCode.PROVIDER_UNAVAILABLE),
        (TimeoutError(), ProviderErrorCode.TIMEOUT),
    ],
)
async def test_sdk_shaped_errors_are_normalized(
    error: Exception,
    code: ProviderErrorCode,
) -> None:
    provider = AnthropicMessagesProvider("claude-test", client=FakeClient(error))

    with pytest.raises(ModelProviderError) as captured:
        await provider.generate(_request())

    assert captured.value.code is code


async def test_cancellation_propagates_unchanged() -> None:
    provider = AnthropicMessagesProvider(
        "claude-test",
        client=FakeClient(asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.generate(_request())


async def test_refusal_is_a_terminal_content_policy_error() -> None:
    provider = AnthropicMessagesProvider(
        "claude-test",
        client=FakeClient(_response(stop_reason="refusal")),
    )

    with pytest.raises(ModelProviderError) as captured:
        await provider.generate(_request())

    assert captured.value.code is ProviderErrorCode.CONTENT_BLOCKED


async def test_stream_normalizes_text_and_has_one_terminal_completion() -> None:
    events = (
        _message_start(),
        _block_start(0, SimpleNamespace(type="text", text="")),
        _block_delta(0, SimpleNamespace(type="text_delta", text="Hello")),
        _block_delta(0, SimpleNamespace(type="text_delta", text=" ")),
        SimpleNamespace(type="ping"),
        _block_delta(0, SimpleNamespace(type="text_delta", text="world")),
        _block_stop(0),
        _message_delta("end_turn", 5),
        SimpleNamespace(type="message_stop"),
    )
    client = FakeClient(streams=(events,))
    provider = AnthropicMessagesProvider("claude-test", client=client)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    streamed = await _stream_events(
        provider,
        _request(response_schema=schema),
    )

    assert client.messages.stream_calls == [
        {
            "model": "claude-test",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        }
    ]
    assert streamed[:3] == [
        ModelTextDelta("Hello"),
        ModelTextDelta(" "),
        ModelTextDelta("world"),
    ]
    assert len(streamed) == 4
    completed = streamed[-1]
    assert isinstance(completed, ModelStreamCompleted)
    assert completed.response.provider_id == "anthropic:claude-test"
    assert completed.response.provider_response_id == "message-stream"
    assert completed.response.finish_reason is FinishReason.STOP
    assert completed.response.text == "Hello world"
    assert completed.response.usage.input_tokens == 19
    assert completed.response.usage.output_tokens == 5


async def test_stream_normalizes_tool_json_and_preserves_thinking_state() -> None:
    events = (
        _message_start(),
        _block_start(
            0,
            SimpleNamespace(type="thinking", thinking="", signature=""),
        ),
        _block_delta(
            0,
            SimpleNamespace(type="thinking_delta", thinking="opaque state"),
        ),
        _block_delta(
            0,
            SimpleNamespace(type="signature_delta", signature="signed"),
        ),
        _block_stop(0),
        _block_start(
            1,
            SimpleNamespace(
                type="tool_use",
                id="provider-call",
                name="fake.read",
                input={},
            ),
        ),
        _block_delta(
            1,
            SimpleNamespace(type="input_json_delta", partial_json='{"key":'),
        ),
        _block_delta(
            1,
            SimpleNamespace(type="input_json_delta", partial_json='"alpha"}'),
        ),
        _block_stop(1),
        _message_delta("tool_use", 9),
        SimpleNamespace(type="message_stop"),
    )
    client = FakeClient(streams=(events,))
    provider = AnthropicMessagesProvider(
        "claude-test",
        client=client,
        id_factory=lambda _prefix: "canonical-call",
    )

    streamed = await _stream_events(provider, _request())

    assert streamed[:3] == [
        ModelToolCallDelta(
            index=0,
            arguments_delta="",
            id="canonical-call",
            name="fake.read",
            provider_call_id="provider-call",
        ),
        ModelToolCallDelta(index=0, arguments_delta='{"key":'),
        ModelToolCallDelta(index=0, arguments_delta='"alpha"}'),
    ]
    assert len(streamed) == 4
    completed = streamed[-1]
    assert isinstance(completed, ModelStreamCompleted)
    assert completed.response.finish_reason is FinishReason.TOOL_CALLS
    assert completed.response.text is None
    assert [
        (
            call.id,
            call.provider_call_id,
            call.name,
            _plain_json_object(call.arguments),
        )
        for call in completed.response.tool_calls
    ] == [("canonical-call", "provider-call", "fake.read", {"key": "alpha"})]
    assert _plain_json_object(completed.response.provider_metadata) == {
        "anthropic_continuation": {
            "provider_id": "anthropic:claude-test",
            "content_blocks": [
                {
                    "type": "thinking",
                    "thinking": "opaque state",
                    "signature": "signed",
                }
            ],
        }
    }


@pytest.mark.parametrize(
    "events",
    [
        (),
        (
            _message_start(),
            _block_start(
                0,
                SimpleNamespace(
                    type="tool_use",
                    id="provider-call",
                    name="fake.read",
                    input={},
                ),
            ),
            _block_delta(
                0,
                SimpleNamespace(type="input_json_delta", partial_json="{"),
            ),
            _block_stop(0),
        ),
        (
            _message_start(),
            _block_start(0, SimpleNamespace(type="text", text="")),
            _block_delta(0, SimpleNamespace(type="text_delta", text="partial")),
            _block_stop(0),
            _message_delta("end_turn", 3),
        ),
        (
            _message_start(),
            SimpleNamespace(type="message_stop"),
        ),
    ],
)
async def test_malformed_or_missing_terminal_stream_fails_closed(
    events: Sequence[object],
) -> None:
    provider = AnthropicMessagesProvider(
        "claude-test",
        client=FakeClient(streams=(events,)),
    )

    with pytest.raises(ModelProviderError) as captured:
        await _stream_events(provider, _request())

    assert captured.value.code is ProviderErrorCode.MALFORMED_RESPONSE


async def test_stream_error_event_is_normalized() -> None:
    events = (
        _message_start(),
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(type="overloaded_error"),
        ),
    )
    provider = AnthropicMessagesProvider(
        "claude-test",
        client=FakeClient(streams=(events,)),
    )

    with pytest.raises(ModelProviderError) as captured:
        await _stream_events(provider, _request())

    assert captured.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE


async def test_stream_cancellation_propagates_unchanged() -> None:
    provider = AnthropicMessagesProvider(
        "claude-test",
        client=FakeClient(streams=((asyncio.CancelledError(),),)),
    )

    with pytest.raises(asyncio.CancelledError):
        await _stream_events(provider, _request())


@pytest.mark.parametrize(
    "response",
    [
        _response(
            SimpleNamespace(
                type="tool_use",
                id="provider-call",
                name="fake.read",
                input=[],
            ),
            stop_reason="tool_use",
        ),
        _response(SimpleNamespace(type="unknown")),
        _response(SimpleNamespace(type="text", text="done"), stop_reason="tool_use"),
        SimpleNamespace(
            id="message-empty",
            type="message",
            role="assistant",
            content=(),
            stop_reason="end_turn",
            usage=_usage(),
        ),
        _response(usage=SimpleNamespace(input_tokens=-1, output_tokens=1)),
    ],
)
async def test_malformed_responses_use_stable_error(response: object) -> None:
    provider = AnthropicMessagesProvider("claude-test", client=FakeClient(response))

    with pytest.raises(ModelProviderError) as captured:
        await provider.generate(_request())

    assert captured.value.code is ProviderErrorCode.MALFORMED_RESPONSE


def _block_anthropic_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_missing_sdk_uses_the_repository_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_anthropic_import(monkeypatch)
    provider = AnthropicMessagesProvider("claude-test")

    with pytest.raises(ImportError, match=r"pip install 'daita-agents\[anthropic\]'"):
        _ = provider.client


async def test_generate_preserves_the_missing_sdk_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_anthropic_import(monkeypatch)
    provider = AnthropicMessagesProvider("claude-test")

    with pytest.raises(ImportError, match=r"pip install 'daita-agents\[anthropic\]'"):
        await provider.generate(_request())
