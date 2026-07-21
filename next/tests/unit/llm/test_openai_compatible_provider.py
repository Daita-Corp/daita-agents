from __future__ import annotations

import asyncio
import builtins
from collections.abc import Mapping, Sequence
import traceback
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
from daita.llm.providers.grok import GrokProvider
from daita.llm.providers.ollama import OllamaProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider


class FakeCompletions:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, *items: object) -> None:
        self.chat = FakeChat(*items)


class FakeChat:
    def __init__(self, *items: object) -> None:
        self.completions = FakeCompletions(*items)


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


class HttpError(Exception):
    def __init__(self, status_code: int, code: str | None = None) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__("vendor details must not escape")


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


def _completion(
    *,
    content: str | None = "done",
    finish_reason: str = "stop",
    tool_calls: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="response-1",
        choices=(
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    refusal=None,
                    tool_calls=() if tool_calls is None else tool_calls,
                ),
            ),
        ),
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )


def _stream_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: object = (),
    usage: object | None = None,
    response_id: str = "stream-response-1",
) -> SimpleNamespace:
    choices = (
        ()
        if content is None
        and finish_reason is None
        and not tool_calls
        and usage is not None
        else (
            SimpleNamespace(
                index=0,
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    content=content,
                    refusal=None,
                    tool_calls=tool_calls,
                ),
            ),
        )
    )
    return SimpleNamespace(id=response_id, choices=choices, usage=usage)


async def test_compatible_text_translation_and_usage_are_canonical() -> None:
    client = FakeClient(_completion())
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=client,
    )

    response = await provider.generate(
        _request(
            _message(MessageRole.SYSTEM, "system"),
            _message(MessageRole.USER, "hello"),
        )
    )

    assert provider.provider_id == "compatible:model-a"
    assert client.chat.completions.calls == [
        {
            "max_tokens": 1_024,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "model": "model-a",
        }
    ]
    assert response.finish_reason is FinishReason.STOP
    assert response.text == "done"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.cache_read_tokens == 3
    assert response.usage.reasoning_tokens == 2
    assert response.provider_response_id == "response-1"


async def test_compatible_explicit_sequential_policy_uses_native_option() -> None:
    client = FakeClient(_completion())
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=client,
    )
    request = _request(allow_parallel_tool_calls=False)

    assert provider.supports_request_policy(request) is True
    await provider.generate(request)

    assert client.chat.completions.calls[0]["parallel_tool_calls"] is False


async def test_compatible_tools_keep_canonical_and_native_ids_separate() -> None:
    native_call = SimpleNamespace(
        id="native-call-1",
        type="function",
        function=SimpleNamespace(name="lookup", arguments='{"key":"alpha"}'),
    )
    client = FakeClient(
        _completion(
            content="Checking.",
            finish_reason="tool_calls",
            tool_calls=(native_call,),
        ),
        _completion(content="finished"),
    )
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
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
    assert call.provider_call_id == "native-call-1"
    assert dict(call.arguments) == {"key": "alpha"}
    metadata = FrozenJsonObject.from_mapping(first.provider_metadata).to_dict()
    assert metadata["openai_compatible_continuation"] == {
        "provider_id": "compatible:model-a"
    }

    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        content=(TextBlock("Checking."),),
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
    await provider.generate(
        _request(_message(MessageRole.USER, "lookup"), assistant, result)
    )

    followup = client.chat.completions.calls[1]["messages"]
    assert isinstance(followup, list)
    assert followup[1]["tool_calls"][0]["id"] == "native-call-1"
    assert followup[2]["tool_call_id"] == "native-call-1"


async def test_foreign_continuation_uses_portable_canonical_call_id() -> None:
    client = FakeClient(_completion())
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=client,
    )
    call = ToolCall(
        id="canonical-call",
        provider_call_id="foreign-native-call",
        name="lookup",
        arguments={"key": "alpha"},
    )
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=(call,),
        provider_id="other:model",
        provider_metadata={
            "openai_compatible_continuation": {"provider_id": "other:model"}
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

    messages = client.chat.completions.calls[0]["messages"]
    assert isinstance(messages, list)
    assert messages[0]["tool_calls"][0]["id"] == "canonical-call"
    assert messages[1]["tool_call_id"] == "canonical-call"


@pytest.mark.parametrize(
    "response",
    (
        SimpleNamespace(id="response", choices=(), usage=None),
        _completion(content=None),
        _completion(finish_reason="unknown"),
        _completion(
            content=None,
            finish_reason="tool_calls",
            tool_calls=(
                SimpleNamespace(
                    id="call",
                    type="function",
                    function=SimpleNamespace(name="lookup", arguments="[]"),
                ),
            ),
        ),
    ),
)
async def test_compatible_malformed_responses_fail_closed(response: object) -> None:
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=FakeClient(response),
    )

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE


async def test_compatible_content_filter_is_a_terminal_provider_error() -> None:
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=FakeClient(_completion(content=None, finish_reason="content_filter")),
    )

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is ProviderErrorCode.CONTENT_BLOCKED


async def test_compatible_stream_normalizes_text_and_terminal_usage() -> None:
    usage = SimpleNamespace(
        prompt_tokens=9,
        completion_tokens=3,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    stream = FakeAsyncStream(
        _stream_chunk(content="Hello"),
        _stream_chunk(content=" "),
        _stream_chunk(content="world", finish_reason="stop"),
        _stream_chunk(usage=usage),
    )
    client = FakeClient(stream)
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=client,
    )

    events = [
        event
        async for event in provider.stream(_request(allow_parallel_tool_calls=False))
    ]

    assert [event.text for event in events if isinstance(event, ModelTextDelta)] == [
        "Hello",
        " ",
        "world",
    ]
    completed = events[-1]
    assert isinstance(completed, ModelStreamCompleted)
    assert completed.response.text == "Hello world"
    assert completed.response.usage.input_tokens == 9
    assert completed.response.usage.output_tokens == 3
    assert client.chat.completions.calls[0]["stream"] is True
    assert client.chat.completions.calls[0]["parallel_tool_calls"] is False
    assert client.chat.completions.calls[0]["stream_options"] == {"include_usage": True}


async def test_compatible_stream_normalizes_fragmented_tool_arguments() -> None:
    first_delta = SimpleNamespace(
        index=0,
        id="native-call",
        function=SimpleNamespace(name="lookup", arguments='{"key":'),
    )
    second_delta = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name=None, arguments='"alpha"}'),
    )
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=FakeClient(
            FakeAsyncStream(
                _stream_chunk(tool_calls=(first_delta,)),
                _stream_chunk(
                    tool_calls=(second_delta,),
                    finish_reason="tool_calls",
                ),
            )
        ),
        id_factory=lambda prefix: f"{prefix}-canonical",
    )

    events = [event async for event in provider.stream(_request())]

    deltas = [event for event in events if isinstance(event, ModelToolCallDelta)]
    assert [event.arguments_delta for event in deltas] == ['{"key":', '"alpha"}']
    assert deltas[0].id == "call-canonical"
    assert deltas[0].provider_call_id == "native-call"
    completed = events[-1]
    assert isinstance(completed, ModelStreamCompleted)
    assert completed.response.tool_calls[0].id == "call-canonical"
    assert dict(completed.response.tool_calls[0].arguments) == {"key": "alpha"}


@pytest.mark.parametrize(
    "stream",
    (
        FakeAsyncStream(_stream_chunk(content="unterminated")),
        FakeAsyncStream(
            _stream_chunk(
                tool_calls=(
                    SimpleNamespace(
                        index=0,
                        id="native-call",
                        function=SimpleNamespace(
                            name="lookup",
                            arguments="{",
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            )
        ),
    ),
)
async def test_compatible_stream_rejects_malformed_terminal_state(
    stream: FakeAsyncStream,
) -> None:
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=FakeClient(stream),
    )

    with pytest.raises(ModelProviderError) as caught:
        _ = [event async for event in provider.stream(_request())]

    assert caught.value.code is ProviderErrorCode.MALFORMED_RESPONSE


async def test_compatible_stream_propagates_iteration_cancellation() -> None:
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=FakeClient(FakeAsyncStream(asyncio.CancelledError())),
    )

    with pytest.raises(asyncio.CancelledError):
        _ = [event async for event in provider.stream(_request())]


async def test_compatible_structured_output_uses_chat_json_schema() -> None:
    client = FakeClient(_completion(content='{"answer":7}'))
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=client,
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    await provider.generate(_request(response_schema=schema))

    assert client.chat.completions.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "daita_response",
            "strict": True,
            "schema": schema,
        },
    }


@pytest.mark.parametrize(
    ("error", "code"),
    (
        (HttpError(401), ProviderErrorCode.AUTHENTICATION_ERROR),
        (HttpError(429), ProviderErrorCode.RATE_LIMIT_ERROR),
        (
            HttpError(400, "context_length_exceeded"),
            ProviderErrorCode.CONTEXT_OVERFLOW,
        ),
        (HttpError(404, "model_not_found"), ProviderErrorCode.MODEL_NOT_FOUND),
        (HttpError(503), ProviderErrorCode.PROVIDER_UNAVAILABLE),
        (TimeoutError(), ProviderErrorCode.TIMEOUT),
    ),
)
async def test_compatible_errors_are_normalized(
    error: Exception,
    code: ProviderErrorCode,
) -> None:
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=FakeClient(error),
    )

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(_request())

    assert caught.value.code is code
    assert "vendor details" not in str(caught.value)


async def test_compatible_cancellation_propagates_unchanged() -> None:
    provider = OpenAICompatibleProvider(
        "model-a",
        provider="compatible",
        base_url="https://models.example.test/v1",
        client=FakeClient(asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.generate(_request())


@pytest.mark.parametrize(
    "url",
    (
        "http://models.example.test/v1",
        "https://user:secret@models.example.test/v1",
        "https://models.example.test/v1?token=secret",
        "https://models.example.test/v1#fragment",
    ),
)
def test_compatible_endpoint_policy_rejects_unsafe_remote_urls(url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleProvider("model-a", provider="compatible", base_url=url)


def test_compatible_invalid_port_drops_parser_exception_diagnostics() -> None:
    secret = "base-url-secret"

    with pytest.raises(ValueError) as captured:
        OpenAICompatibleProvider(
            "model-a",
            provider="compatible",
            base_url=f"https://models.example.test:{secret}/v1",
        )

    assert str(captured.value) == "base_url must be a valid absolute URL"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert secret not in "".join(traceback.format_exception(captured.value))


def test_ollama_is_loopback_only_and_grok_endpoint_is_fixed() -> None:
    local = OllamaProvider("llama-test", client=FakeClient(_completion()))
    assert local.provider_id == "ollama:llama-test"
    assert local.base_url == "http://127.0.0.1:11434/v1"

    with pytest.raises(ValueError, match="loopback"):
        OllamaProvider(
            "llama-test",
            base_url="https://ollama.example.test/v1",
        )

    grok = GrokProvider("grok-test", client=FakeClient(_completion()))
    assert grok.provider_id == "grok:grok-test"
    assert grok.base_url == "https://api.x.ai/v1"


async def test_missing_openai_sdk_hint_survives_generate(
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
        if name == "openai" or name.startswith("openai."):
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    provider = OllamaProvider("llama-test")

    with pytest.raises(ImportError, match=r"pip install 'daita-agents\[openai\]'"):
        await provider.generate(_request())
