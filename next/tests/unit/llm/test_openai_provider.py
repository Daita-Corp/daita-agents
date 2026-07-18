from __future__ import annotations

import asyncio
import builtins
from types import SimpleNamespace

import pytest

from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.openai import OpenAIResponsesProvider


class FakeResponses:
    def __init__(self, *items: object) -> None:
        self._items = list(items)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, *items: object) -> None:
        self.responses = FakeResponses(*items)


class SimpleNamespaceError(Exception):
    def __init__(self, status_code: int, *, code: str | None = None) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(f"status {status_code}")


def _message(role: MessageRole, text: str) -> CanonicalMessage:
    return CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=role,
        content=(TextBlock(text),),
    )


def _request(
    *messages: CanonicalMessage, tools: tuple[ToolDefinition, ...] = ()
) -> ModelRequest:
    return ModelRequest(
        operation_id="operation-1",
        turn_id="turn-1",
        messages=messages or (_message(MessageRole.USER, "hello"),),
        tools=tools,
    )


def _text_response(text: str = "done") -> SimpleNamespace:
    return SimpleNamespace(
        id="response-1",
        status="completed",
        output=(
            SimpleNamespace(
                type="message",
                content=(SimpleNamespace(type="output_text", text=text),),
            ),
        ),
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )


async def test_text_translation_uses_responses_without_provider_state() -> None:
    client = FakeClient(_text_response())
    provider = OpenAIResponsesProvider("gpt-test", client=client)

    response = await provider.generate(_request())

    assert provider.provider_id == "openai:gpt-test"
    assert client.responses.calls == [
        {
            "model": "gpt-test",
            "input": [{"role": "user", "content": "hello"}],
            "include": ["reasoning.encrypted_content"],
            "store": False,
        }
    ]
    assert response.finish_reason is FinishReason.STOP
    assert response.text == "done"
    assert response.provider_response_id == "response-1"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.cache_read_tokens == 3
    assert response.usage.reasoning_tokens == 2


async def test_tools_and_multiple_calls_keep_canonical_and_provider_ids_separate() -> (
    None
):
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
    raw = SimpleNamespace(
        id="response-tools",
        status="completed",
        output=(
            SimpleNamespace(
                type="function_call",
                call_id="provider-a",
                name="fake.read",
                arguments='{"key":"alpha"}',
            ),
            SimpleNamespace(
                type="function_call",
                call_id="provider-b",
                name="fake.read",
                arguments='{"key":"beta"}',
            ),
        ),
        output_text="",
        usage=None,
    )
    ids = iter(("call-a", "call-b"))
    client = FakeClient(raw)
    provider = OpenAIResponsesProvider(
        "gpt-test", client=client, id_factory=lambda _prefix: next(ids)
    )

    response = await provider.generate(_request(tools=(tool,)))

    assert client.responses.calls[0]["tools"] == [
        {
            "type": "function",
            "name": "fake.read",
            "description": "Read a deterministic value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
            "strict": False,
        }
    ]
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert [
        (call.id, call.provider_call_id, call.arguments["key"])
        for call in response.tool_calls
    ] == [
        ("call-a", "provider-a", "alpha"),
        ("call-b", "provider-b", "beta"),
    ]


async def test_followup_rebuilds_function_call_and_output_from_canonical_messages() -> (
    None
):
    call = ToolCall(
        id="canonical-call",
        provider_call_id="provider-call",
        name="fake.read",
        arguments={"key": "alpha"},
    )
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=(call,),
    )
    tool_result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=(
            ToolResultBlock(
                call_id="canonical-call",
                output={"value": 42},
            ),
        ),
    )
    client = FakeClient(_text_response("42"))
    provider = OpenAIResponsesProvider("gpt-test", client=client)

    response = await provider.generate(
        _request(_message(MessageRole.USER, "read alpha"), assistant, tool_result)
    )

    assert response.text == "42"
    assert client.responses.calls[0]["input"] == [
        {"role": "user", "content": "read alpha"},
        {
            "type": "function_call",
            "call_id": "provider-call",
            "name": "fake.read",
            "arguments": '{"key":"alpha"}',
        },
        {
            "type": "function_call_output",
            "call_id": "provider-call",
            "output": '{"is_error":false,"output":{"value":42}}',
        },
    ]
    assert "previous_response_id" not in client.responses.calls[0]


async def test_reasoning_items_are_persistable_and_replayed_before_tool_outputs() -> (
    None
):
    reasoning_item = {
        "id": "reasoning-1",
        "type": "reasoning",
        "summary": [],
        "status": "completed",
        "encrypted_content": "encrypted-reasoning-state",
    }
    raw = SimpleNamespace(
        id="response-tools",
        status="completed",
        output=(
            reasoning_item,
            SimpleNamespace(
                type="function_call",
                call_id="provider-call",
                name="fake.read",
                arguments='{"key":"alpha"}',
            ),
        ),
        output_text="",
        usage=None,
    )
    client = FakeClient(raw, _text_response("42"))
    provider = OpenAIResponsesProvider(
        "gpt-test", client=client, id_factory=lambda _prefix: "canonical-call"
    )

    first = await provider.generate(_request())
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=first.tool_calls,
        provider_metadata=first.provider_metadata,
    )
    tool_result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id="canonical-call", output={"value": 42}),),
    )

    second = await provider.generate(
        _request(_message(MessageRole.USER, "read alpha"), assistant, tool_result)
    )

    assert second.text == "42"
    assert client.responses.calls[1]["input"] == [
        {"role": "user", "content": "read alpha"},
        reasoning_item,
        {
            "type": "function_call",
            "call_id": "provider-call",
            "name": "fake.read",
            "arguments": '{"key":"alpha"}',
        },
        {
            "type": "function_call_output",
            "call_id": "provider-call",
            "output": '{"is_error":false,"output":{"value":42}}',
        },
    ]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (SimpleNamespaceError(401), ProviderErrorCode.AUTHENTICATION_ERROR),
        (SimpleNamespaceError(429), ProviderErrorCode.RATE_LIMIT_ERROR),
        (SimpleNamespaceError(404), ProviderErrorCode.MODEL_NOT_FOUND),
        (
            SimpleNamespaceError(400, code="context_length_exceeded"),
            ProviderErrorCode.CONTEXT_OVERFLOW,
        ),
        (SimpleNamespaceError(400), ProviderErrorCode.INVALID_REQUEST),
        (SimpleNamespaceError(503), ProviderErrorCode.PROVIDER_UNAVAILABLE),
        (TimeoutError(), ProviderErrorCode.TIMEOUT),
    ],
)
async def test_sdk_shaped_errors_are_normalized(
    error: Exception, code: ProviderErrorCode
) -> None:
    provider = OpenAIResponsesProvider("gpt-test", client=FakeClient(error))

    with pytest.raises(ModelProviderError) as captured:
        await provider.generate(_request())

    assert captured.value.code is code


async def test_cancellation_propagates_unchanged() -> None:
    provider = OpenAIResponsesProvider(
        "gpt-test", client=FakeClient(asyncio.CancelledError())
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.generate(_request())


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            id="bad-arguments",
            status="completed",
            output=(
                SimpleNamespace(
                    type="function_call",
                    call_id="provider-call",
                    name="fake.read",
                    arguments="[]",
                ),
            ),
            output_text="",
            usage=None,
        ),
        SimpleNamespace(
            id="empty",
            status="completed",
            output=(),
            output_text="",
            usage=None,
        ),
    ],
)
async def test_malformed_responses_use_stable_error(response: object) -> None:
    provider = OpenAIResponsesProvider("gpt-test", client=FakeClient(response))

    with pytest.raises(ModelProviderError) as captured:
        await provider.generate(_request())

    assert captured.value.code is ProviderErrorCode.MALFORMED_RESPONSE


def test_missing_sdk_uses_the_repository_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai" or name.startswith("openai."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    provider = OpenAIResponsesProvider("gpt-test")

    with pytest.raises(ImportError, match=r"pip install 'daita-agents\[openai\]'"):
        _ = provider.client


async def test_generate_preserves_the_missing_sdk_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai" or name.startswith("openai."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    provider = OpenAIResponsesProvider("gpt-test")

    with pytest.raises(ImportError, match=r"pip install 'daita-agents\[openai\]'"):
        await provider.generate(_request())
