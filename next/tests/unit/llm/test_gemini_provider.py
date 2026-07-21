from __future__ import annotations

import asyncio
import base64
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
from daita.llm.providers.gemini import GeminiProvider


class FakeModels:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []

    async def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def generate_content_stream(self, **kwargs: object) -> object:
        self.stream_calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, *items: object) -> None:
        self.aio = FakeAsyncClient(*items)


class FakeAsyncClient:
    def __init__(self, *items: object) -> None:
        self.models = FakeModels(*items)


class FakeAsyncStream:
    def __init__(self, *items: object) -> None:
        self.items = list(items)

    def __aiter__(self) -> FakeAsyncStream:
        return self

    async def __anext__(self) -> object:
        if not self.items:
            raise StopAsyncIteration
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class GeminiError(Exception):
    def __init__(self, code: int, status: str | None = None) -> None:
        self.code = code
        self.status = status
        super().__init__("sensitive vendor details")


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
    allow_parallel_tool_calls: bool | None = None,
) -> ModelRequest:
    return ModelRequest(
        operation_id="operation-1",
        turn_id="turn-1",
        messages=messages or (_message(MessageRole.USER, "hello"),),
        tools=tools,
        response_schema=response_schema,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
    )


@pytest.mark.parametrize("policy", (False, True))
async def test_explicit_tool_policy_is_rejected_before_gemini_io(
    policy: bool,
) -> None:
    client = FakeClient()
    provider = GeminiProvider("gemini-test", client=client)
    request = _request(allow_parallel_tool_calls=policy)

    assert provider.supports_request_policy(request) is False
    with pytest.raises(ModelProviderError) as generated:
        await provider.generate(request)
    with pytest.raises(ModelProviderError) as streamed:
        _ = [event async for event in provider.stream(request)]

    assert generated.value.code is ProviderErrorCode.INVALID_REQUEST
    assert streamed.value.code is ProviderErrorCode.INVALID_REQUEST
    assert client.aio.models.calls == []
    assert client.aio.models.stream_calls == []


def _response(*parts: object, finish_reason: str = "STOP") -> SimpleNamespace:
    return SimpleNamespace(
        response_id="gemini-response-1",
        candidates=(
            SimpleNamespace(
                finish_reason=finish_reason,
                content=SimpleNamespace(parts=parts),
            ),
        ),
        prompt_feedback=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=13,
            candidates_token_count=8,
            thoughts_token_count=3,
            cached_content_token_count=4,
        ),
    )


def _stream_chunk(
    *parts: object,
    finish_reason: str | None = None,
    usage: object | None = None,
    response_id: str = "gemini-stream-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        response_id=response_id,
        candidates=(
            SimpleNamespace(
                finish_reason=finish_reason,
                content=(None if not parts else SimpleNamespace(parts=parts)),
            ),
        ),
        prompt_feedback=None,
        usage_metadata=usage,
    )


async def test_gemini_native_text_shape_and_usage_are_normalized() -> None:
    client = FakeClient(_response(SimpleNamespace(text="done", thought=None)))
    provider = GeminiProvider("gemini-test", client=client)

    response = await provider.generate(
        _request(
            _message(MessageRole.SYSTEM, "system one"),
            _message(MessageRole.SYSTEM, "system two"),
            _message(MessageRole.USER, "hello"),
        )
    )

    assert provider.provider_id == "gemini:gemini-test"
    assert client.aio.models.calls == [
        {
            "config": {
                "max_output_tokens": 1_024,
                "system_instruction": "system one\nsystem two",
            },
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "model": "gemini-test",
        }
    ]
    assert response.finish_reason is FinishReason.STOP
    assert response.text == "done"
    assert response.usage.input_tokens == 13
    assert response.usage.output_tokens == 8
    assert response.usage.reasoning_tokens == 3
    assert response.usage.cache_read_tokens == 4
    assert response.provider_response_id == "gemini-response-1"


async def test_gemini_tools_and_thought_signature_are_origin_scoped() -> None:
    signature = b"signed-thought"
    client = FakeClient(
        _response(
            SimpleNamespace(
                thought_signature=signature,
                function_call=SimpleNamespace(
                    id="native-call",
                    name="lookup",
                    args={"key": "alpha"},
                ),
            ),
        ),
        _response(SimpleNamespace(text="finished")),
    )
    provider = GeminiProvider(
        "gemini-test",
        client=client,
        id_factory=lambda prefix: f"{prefix}-canonical",
    )
    tool = ToolDefinition(
        name="lookup",
        description="Look up one key",
        input_schema={"type": "object"},
    )

    first = await provider.generate(_request(tools=(tool,)))
    call = first.tool_calls[0]
    assert call.id == "call-canonical"
    assert call.provider_call_id == "native-call"
    metadata = FrozenJsonObject.from_mapping(first.provider_metadata).to_dict()
    continuation = metadata["gemini_continuation"]
    assert continuation == {
        "provider_id": "gemini:gemini-test",
        "content_parts": [
            {
                "thought_signature": base64.b64encode(signature).decode("ascii"),
                "function_call": {
                    "id": "native-call",
                    "name": "lookup",
                    "args": {"key": "alpha"},
                },
            }
        ],
    }

    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=(call,),
        provider_id=first.provider_id,
        provider_metadata=first.provider_metadata,
    )
    result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call.id, {"value": 7}),),
    )
    await provider.generate(_request(assistant, result, tools=(tool,)))

    contents = client.aio.models.calls[1]["contents"]
    assert isinstance(contents, list)
    assert contents[0]["parts"] == [
        {
            "thought_signature": signature,
            "function_call": {
                "id": "native-call",
                "name": "lookup",
                "args": {"key": "alpha"},
            },
        }
    ]
    assert contents[1]["parts"][0]["function_response"]["id"] == "native-call"


async def test_gemini_foreign_metadata_uses_canonical_tool_identity() -> None:
    client = FakeClient(_response(SimpleNamespace(text="finished")))
    provider = GeminiProvider("gemini-test", client=client)
    call = ToolCall(
        id="canonical-call",
        provider_call_id="foreign-call",
        name="lookup",
        arguments={"key": "alpha"},
    )
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=(call,),
        provider_id="gemini:other",
        provider_metadata={
            "gemini_continuation": {
                "provider_id": "gemini:other",
            }
        },
    )
    result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call.id, {"value": 7}),),
    )

    await provider.generate(_request(assistant, result))

    contents = client.aio.models.calls[0]["contents"]
    assert isinstance(contents, list)
    assert contents[0]["parts"][0]["function_call"]["id"] == "canonical-call"
    assert contents[0]["parts"][0]["thought_signature"] == (
        b"skip_thought_signature_validator"
    )
    assert contents[1]["parts"][0]["function_response"]["id"] == "canonical-call"


@pytest.mark.parametrize(
    "response",
    (
        SimpleNamespace(candidates=(), usage_metadata=None),
        _response(),
        _response(SimpleNamespace(function_call=SimpleNamespace(name="lookup"))),
        _response(SimpleNamespace(text="done"), finish_reason="UNKNOWN"),
    ),
)
async def test_gemini_malformed_responses_fail_closed(response: object) -> None:
    provider = GeminiProvider("gemini-test", client=FakeClient(response))

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE


async def test_gemini_safety_finish_is_content_blocked() -> None:
    provider = GeminiProvider(
        "gemini-test",
        client=FakeClient(_response(finish_reason="SAFETY")),
    )

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.CONTENT_BLOCKED


async def test_gemini_stream_normalizes_text_and_terminal_usage() -> None:
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=4,
        thoughts_token_count=None,
        cached_content_token_count=None,
    )
    client = FakeClient(
        FakeAsyncStream(
            _stream_chunk(SimpleNamespace(text="Hello", thought=None)),
            _stream_chunk(SimpleNamespace(text=" ")),
            _stream_chunk(
                SimpleNamespace(text="world"),
                finish_reason="STOP",
                usage=usage,
            ),
        )
    )
    provider = GeminiProvider("gemini-test", client=client)

    events = [event async for event in provider.stream(_request())]

    assert [event.text for event in events if isinstance(event, ModelTextDelta)] == [
        "Hello",
        " ",
        "world",
    ]
    completed = events[-1]
    assert isinstance(completed, ModelStreamCompleted)
    assert completed.response.text == "Hello world"
    assert completed.response.usage.input_tokens == 10
    assert completed.response.usage.output_tokens == 4
    assert completed.response.usage.reasoning_tokens == 0
    assert client.aio.models.stream_calls[0]["model"] == "gemini-test"


async def test_gemini_stream_preserves_signed_tool_part_and_call_identity() -> None:
    signature = b"stream-signature"
    client = FakeClient(
        FakeAsyncStream(
            _stream_chunk(
                SimpleNamespace(
                    thought_signature=signature,
                    function_call=SimpleNamespace(
                        id="native-call",
                        name="lookup",
                        args={"key": "alpha"},
                    ),
                ),
                finish_reason="STOP",
            )
        )
    )
    provider = GeminiProvider(
        "gemini-test",
        client=client,
        id_factory=lambda prefix: f"{prefix}-canonical",
    )

    events = [event async for event in provider.stream(_request())]

    delta = events[0]
    assert isinstance(delta, ModelToolCallDelta)
    assert delta.id == "call-canonical"
    assert delta.provider_call_id == "native-call"
    assert delta.arguments_delta == '{"key":"alpha"}'
    completed = events[-1]
    assert isinstance(completed, ModelStreamCompleted)
    assert completed.response.tool_calls[0].id == "call-canonical"
    metadata = FrozenJsonObject.from_mapping(
        completed.response.provider_metadata
    ).to_dict()
    continuation = metadata["gemini_continuation"]
    assert isinstance(continuation, dict)
    content_parts = continuation["content_parts"]
    assert isinstance(content_parts, list)
    first_part = content_parts[0]
    assert isinstance(first_part, dict)
    assert first_part["thought_signature"] == base64.b64encode(signature).decode(
        "ascii"
    )


async def test_gemini_stream_rejects_missing_terminal_finish() -> None:
    provider = GeminiProvider(
        "gemini-test",
        client=FakeClient(
            FakeAsyncStream(_stream_chunk(SimpleNamespace(text="unfinished")))
        ),
    )

    with pytest.raises(ModelProviderError) as caught:
        _ = [event async for event in provider.stream(_request())]

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE


async def test_gemini_stream_propagates_iteration_cancellation() -> None:
    provider = GeminiProvider(
        "gemini-test",
        client=FakeClient(FakeAsyncStream(asyncio.CancelledError())),
    )

    with pytest.raises(asyncio.CancelledError):
        _ = [event async for event in provider.stream(_request())]


async def test_gemini_structured_output_uses_native_json_schema() -> None:
    client = FakeClient(_response(SimpleNamespace(text='{"answer":7}')))
    provider = GeminiProvider("gemini-test", client=client)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }

    await provider.generate(_request(response_schema=schema))

    config = client.aio.models.calls[0]["config"]
    assert isinstance(config, dict)
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == schema


@pytest.mark.parametrize(
    ("error", "code"),
    (
        (GeminiError(401), ProviderErrorCode.AUTHENTICATION_ERROR),
        (GeminiError(429), ProviderErrorCode.RATE_LIMIT_ERROR),
        (GeminiError(404, "NOT_FOUND"), ProviderErrorCode.MODEL_NOT_FOUND),
        (
            GeminiError(400, "INVALID_ARGUMENT"),
            ProviderErrorCode.INVALID_REQUEST,
        ),
        (GeminiError(503), ProviderErrorCode.PROVIDER_UNAVAILABLE),
        (TimeoutError(), ProviderErrorCode.TIMEOUT),
    ),
)
async def test_gemini_errors_are_normalized(
    error: Exception,
    code: ProviderErrorCode,
) -> None:
    provider = GeminiProvider("gemini-test", client=FakeClient(error))

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is code
    assert "sensitive vendor details" not in str(caught.value)


async def test_gemini_cancellation_propagates_unchanged() -> None:
    provider = GeminiProvider(
        "gemini-test",
        client=FakeClient(asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.generate(_request())


async def test_missing_google_sdk_hint_survives_generate(
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
        if name == "google" or name.startswith("google."):
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    provider = GeminiProvider("gemini-test")

    with pytest.raises(ImportError, match=r"pip install 'daita-agents\[google\]'"):
        await provider.generate(_request())
