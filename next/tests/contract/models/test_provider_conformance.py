from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import traceback
from types import SimpleNamespace
from typing import cast

import pytest

from daita.llm import (
    AnthropicProvider,
    CanonicalMessage,
    FinishReason,
    GeminiProvider,
    GrokProvider,
    MessageRole,
    ModelProviderError,
    ModelRequest,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelToolCallDelta,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderErrorCode,
    StreamingModelProvider,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
)


class _FakeAsyncStream:
    def __init__(self, events: Sequence[object]) -> None:
        self._events = list(events)

    def __aiter__(self) -> _FakeAsyncStream:
        return self

    async def __anext__(self) -> object:
        if not self._events:
            raise StopAsyncIteration
        event = self._events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


class _AsyncResource:
    def __init__(self, items: Sequence[object]) -> None:
        self._items = list(items)
        self.calls: list[dict[str, object]] = []

    def _next(self) -> object:
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._next()

    async def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._next()

    async def generate_content_stream(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._next()


class _AnthropicStreamManager:
    def __init__(self, events: Sequence[object]) -> None:
        self._stream = _FakeAsyncStream(events)

    async def __aenter__(self) -> _FakeAsyncStream:
        return self._stream

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None


class _AnthropicResource:
    def __init__(
        self,
        items: Sequence[object],
        streams: Sequence[Sequence[object]],
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

    def stream(self, **kwargs: object) -> _AnthropicStreamManager:
        self.stream_calls.append(kwargs)
        return _AnthropicStreamManager(self._streams.pop(0))


class _FakeAnthropicClient:
    def __init__(self, messages: _AnthropicResource) -> None:
        self.messages = messages


class _FakeOpenAIClient:
    def __init__(self, responses: _AsyncResource) -> None:
        self.responses = responses


class _FakeGeminiAsyncClient:
    def __init__(self, models: _AsyncResource) -> None:
        self.models = models


class _FakeGeminiClient:
    def __init__(self, models: _AsyncResource) -> None:
        self.aio = _FakeGeminiAsyncClient(models)


class _FakeChat:
    def __init__(self, completions: _AsyncResource) -> None:
        self.completions = completions


class _FakeCompatibleClient:
    def __init__(self, completions: _AsyncResource) -> None:
        self.chat = _FakeChat(completions)


@dataclass(frozen=True, slots=True)
class _ProviderCase:
    name: str
    wire: str
    provider_name: str
    model: str
    sdk_module: str
    install_extra: str

    @property
    def provider_id(self) -> str:
        return f"{self.provider_name}:{self.model}"

    def make(
        self,
        *items: object,
        stream: Sequence[object] | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> tuple[StreamingModelProvider, _AsyncResource | _AnthropicResource]:
        if self.wire == "anthropic":
            anthropic_resource = _AnthropicResource(
                items,
                () if stream is None else (stream,),
            )
            anthropic_provider: object = AnthropicProvider(
                self.model,
                client=_FakeAnthropicClient(anthropic_resource),
                id_factory=id_factory,
            )
            return (
                cast(StreamingModelProvider, anthropic_provider),
                anthropic_resource,
            )

        resource = _AsyncResource(
            items if stream is None else (_FakeAsyncStream(stream),)
        )
        if self.wire == "openai":
            provider: object = OpenAIProvider(
                self.model,
                client=_FakeOpenAIClient(resource),
                id_factory=id_factory,
            )
        elif self.wire == "gemini":
            provider = GeminiProvider(
                self.model,
                client=_FakeGeminiClient(resource),
                id_factory=id_factory,
            )
        elif self.name == "ollama":
            provider = OllamaProvider(
                self.model,
                client=_FakeCompatibleClient(resource),
                id_factory=id_factory,
            )
        elif self.name == "grok":
            provider = GrokProvider(
                self.model,
                client=_FakeCompatibleClient(resource),
                id_factory=id_factory,
            )
        else:
            provider = OpenAICompatibleProvider(
                self.model,
                provider=self.provider_name,
                base_url="https://models.example.test/v1",
                client=_FakeCompatibleClient(resource),
                id_factory=id_factory,
            )
        return cast(StreamingModelProvider, provider), resource

    def make_without_client(self) -> StreamingModelProvider:
        if self.wire == "openai":
            provider: object = OpenAIProvider(self.model)
        elif self.wire == "anthropic":
            provider = AnthropicProvider(self.model)
        elif self.wire == "gemini":
            provider = GeminiProvider(self.model)
        elif self.name == "ollama":
            provider = OllamaProvider(self.model)
        elif self.name == "grok":
            provider = GrokProvider(self.model)
        else:
            provider = OpenAICompatibleProvider(
                self.model,
                provider=self.provider_name,
                base_url="https://models.example.test/v1",
            )
        return cast(StreamingModelProvider, provider)


CASES = (
    _ProviderCase("openai", "openai", "openai", "gpt-test", "openai", "openai"),
    _ProviderCase(
        "anthropic",
        "anthropic",
        "anthropic",
        "claude-test",
        "anthropic",
        "anthropic",
    ),
    _ProviderCase(
        "gemini",
        "gemini",
        "gemini",
        "gemini-test",
        "google",
        "google",
    ),
    _ProviderCase(
        "compatible",
        "compatible",
        "acme",
        "model-test",
        "openai",
        "openai",
    ),
    _ProviderCase(
        "ollama",
        "compatible",
        "ollama",
        "llama-test",
        "openai",
        "openai",
    ),
    _ProviderCase(
        "grok",
        "compatible",
        "grok",
        "grok-test",
        "openai",
        "openai",
    ),
)


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


TOOL = ToolDefinition(
    name="lookup",
    description="Look up one deterministic key",
    input_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _text_response(case: _ProviderCase, text: str = "done") -> object:
    if case.wire == "openai":
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
    if case.wire == "anthropic":
        return SimpleNamespace(
            id="response-1",
            type="message",
            role="assistant",
            content=(SimpleNamespace(type="text", text=text),),
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=8,
                output_tokens=7,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=0,
            ),
        )
    if case.wire == "gemini":
        return SimpleNamespace(
            response_id="response-1",
            candidates=(
                SimpleNamespace(
                    finish_reason="STOP",
                    content=SimpleNamespace(
                        parts=(SimpleNamespace(text=text),),
                    ),
                ),
            ),
            prompt_feedback=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=11,
                candidates_token_count=7,
                thoughts_token_count=2,
                cached_content_token_count=3,
            ),
        )
    return SimpleNamespace(
        id="response-1",
        choices=(
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=text,
                    refusal=None,
                    tool_calls=(),
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


def _tool_response(case: _ProviderCase) -> object:
    if case.wire == "openai":
        return SimpleNamespace(
            id="response-tools",
            status="completed",
            output=(
                {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": [],
                    "status": "completed",
                    "encrypted_content": "opaque",
                },
                SimpleNamespace(
                    type="function_call",
                    call_id="provider-a",
                    name="lookup",
                    arguments='{"key":"alpha"}',
                ),
                SimpleNamespace(
                    type="function_call",
                    call_id="provider-b",
                    name="lookup",
                    arguments='{"key":"beta"}',
                ),
            ),
            output_text="",
            usage=None,
        )
    if case.wire == "anthropic":
        return SimpleNamespace(
            id="response-tools",
            type="message",
            role="assistant",
            content=(
                {
                    "type": "thinking",
                    "thinking": "opaque",
                    "signature": "signed",
                },
                SimpleNamespace(
                    type="tool_use",
                    id="provider-a",
                    name="lookup",
                    input={"key": "alpha"},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="provider-b",
                    name="lookup",
                    input={"key": "beta"},
                ),
            ),
            stop_reason="tool_use",
            usage=None,
        )
    if case.wire == "gemini":
        return SimpleNamespace(
            response_id="response-tools",
            candidates=(
                SimpleNamespace(
                    finish_reason="STOP",
                    content=SimpleNamespace(
                        parts=(
                            SimpleNamespace(
                                thought_signature=b"signed",
                                function_call=SimpleNamespace(
                                    id="provider-a",
                                    name="lookup",
                                    args={"key": "alpha"},
                                ),
                            ),
                            SimpleNamespace(
                                function_call=SimpleNamespace(
                                    id="provider-b",
                                    name="lookup",
                                    args={"key": "beta"},
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            prompt_feedback=None,
            usage_metadata=None,
        )
    native_calls = tuple(
        SimpleNamespace(
            id=f"provider-{suffix}",
            type="function",
            function=SimpleNamespace(
                name="lookup",
                arguments=f'{{"key":"{key}"}}',
            ),
        )
        for suffix, key in (("a", "alpha"), ("b", "beta"))
    )
    return SimpleNamespace(
        id="response-tools",
        choices=(
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    refusal=None,
                    tool_calls=native_calls,
                ),
            ),
        ),
        usage=None,
    )


def _anthropic_message_start() -> object:
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            id="stream-1",
            type="message",
            role="assistant",
            content=(),
            stop_reason=None,
            usage=SimpleNamespace(
                input_tokens=8,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=0,
                output_tokens=0,
            ),
        ),
    )


def _text_stream(case: _ProviderCase, *, terminal: bool = True) -> Sequence[object]:
    if case.wire == "openai":
        events: list[object] = [
            SimpleNamespace(type="response.output_text.delta", delta="hel"),
            SimpleNamespace(type="response.output_text.delta", delta=" "),
            SimpleNamespace(type="response.output_text.delta", delta="lo"),
        ]
        if terminal:
            events.append(
                SimpleNamespace(
                    type="response.completed",
                    response=_text_response(case, "hel lo"),
                )
            )
        return events
    if case.wire == "anthropic":
        events = [
            _anthropic_message_start(),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text", text=""),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="hel"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text=" "),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="lo"),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=7),
            ),
        ]
        if terminal:
            events.append(SimpleNamespace(type="message_stop"))
        return events
    if case.wire == "gemini":
        gemini_events = [
            SimpleNamespace(
                response_id="stream-1",
                candidates=(
                    SimpleNamespace(
                        finish_reason=None,
                        content=SimpleNamespace(
                            parts=(SimpleNamespace(text=fragment),),
                        ),
                    ),
                ),
                prompt_feedback=None,
                usage_metadata=None,
            )
            for fragment in ("hel", " ", "lo")
        ]
        if terminal:
            gemini_events[-1].candidates[0].finish_reason = "STOP"
            gemini_events[-1].usage_metadata = SimpleNamespace(
                prompt_token_count=11,
                candidates_token_count=7,
                thoughts_token_count=0,
                cached_content_token_count=3,
            )
        return gemini_events
    compatible_events = [
        SimpleNamespace(
            id="stream-1",
            choices=(
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=fragment,
                        refusal=None,
                        tool_calls=(),
                    ),
                ),
            ),
            usage=None,
        )
        for fragment in ("hel", " ", "lo")
    ]
    if terminal:
        compatible_events[-1].choices[0].finish_reason = "stop"
        compatible_events.append(
            SimpleNamespace(
                id="stream-1",
                choices=(),
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
            )
        )
    return compatible_events


def _tool_stream(case: _ProviderCase) -> Sequence[object]:
    if case.wire == "openai":
        raw = _tool_response(case)
        events: list[object] = []
        for index, suffix in enumerate(("a", "b")):
            events.extend(
                (
                    SimpleNamespace(
                        type="response.output_item.added",
                        output_index=index,
                        item=SimpleNamespace(
                            type="function_call",
                            call_id=f"provider-{suffix}",
                            name="lookup",
                        ),
                    ),
                    SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        output_index=index,
                        delta=(
                            '{"key":"alpha"}' if suffix == "a" else '{"key":"beta"}'
                        ),
                    ),
                )
            )
        events.append(SimpleNamespace(type="response.completed", response=raw))
        return events
    if case.wire == "anthropic":
        events = [_anthropic_message_start()]
        for index, (suffix, key) in enumerate((("a", "alpha"), ("b", "beta"))):
            events.extend(
                (
                    SimpleNamespace(
                        type="content_block_start",
                        index=index,
                        content_block=SimpleNamespace(
                            type="tool_use",
                            id=f"provider-{suffix}",
                            name="lookup",
                            input={},
                        ),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        index=index,
                        delta=SimpleNamespace(
                            type="input_json_delta",
                            partial_json=f'{{"key":"{key}"}}',
                        ),
                    ),
                    SimpleNamespace(type="content_block_stop", index=index),
                )
            )
        events.extend(
            (
                SimpleNamespace(
                    type="message_delta",
                    delta=SimpleNamespace(stop_reason="tool_use"),
                    usage=SimpleNamespace(output_tokens=7),
                ),
                SimpleNamespace(type="message_stop"),
            )
        )
        return events
    if case.wire == "gemini":
        return (
            SimpleNamespace(
                response_id="stream-tools",
                candidates=(
                    SimpleNamespace(
                        finish_reason="STOP",
                        content=SimpleNamespace(
                            parts=(
                                SimpleNamespace(
                                    thought_signature=b"signed",
                                    function_call=SimpleNamespace(
                                        id="provider-a",
                                        name="lookup",
                                        args={"key": "alpha"},
                                    ),
                                ),
                                SimpleNamespace(
                                    function_call=SimpleNamespace(
                                        id="provider-b",
                                        name="lookup",
                                        args={"key": "beta"},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                prompt_feedback=None,
                usage_metadata=None,
            ),
        )
    events = []
    for index, (suffix, key) in enumerate((("a", "alpha"), ("b", "beta"))):
        events.append(
            SimpleNamespace(
                id="stream-tools",
                choices=(
                    SimpleNamespace(
                        index=0,
                        finish_reason=("tool_calls" if index == 1 else None),
                        delta=SimpleNamespace(
                            content=None,
                            refusal=None,
                            tool_calls=(
                                SimpleNamespace(
                                    index=index,
                                    id=f"provider-{suffix}",
                                    function=SimpleNamespace(
                                        name="lookup",
                                        arguments=f'{{"key":"{key}"}}',
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                usage=None,
            )
        )
    return events


def _structured_schema_from_call(
    case: _ProviderCase, call: Mapping[str, object]
) -> object:
    if case.wire == "openai":
        native_format = cast(Mapping[str, object], call["text"])["format"]
        return cast(Mapping[str, object], native_format)["schema"]
    if case.wire == "anthropic":
        native_format = cast(Mapping[str, object], call["output_config"])["format"]
        return cast(Mapping[str, object], native_format)["schema"]
    if case.wire == "gemini":
        return cast(Mapping[str, object], call["config"])["response_json_schema"]
    response_format = cast(Mapping[str, object], call["response_format"])
    return cast(Mapping[str, object], response_format["json_schema"])["schema"]


def _native_followup_ids(
    case: _ProviderCase,
    resource: _AsyncResource | _AnthropicResource,
    *,
    call_position: int = 1,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    call = resource.calls[call_position]
    if case.wire == "openai":
        items = cast(Sequence[Mapping[str, object]], call["input"])
        requested = tuple(
            cast(str, item["call_id"])
            for item in items
            if item.get("type") == "function_call"
        )
        results = tuple(
            cast(str, item["call_id"])
            for item in items
            if item.get("type") == "function_call_output"
        )
        return requested, results
    if case.wire == "anthropic":
        messages = cast(Sequence[Mapping[str, object]], call["messages"])
        assistant = cast(Sequence[Mapping[str, object]], messages[1]["content"])
        results_value = cast(Sequence[Mapping[str, object]], messages[2]["content"])
        return (
            tuple(
                cast(str, item["id"])
                for item in assistant
                if item["type"] == "tool_use"
            ),
            tuple(cast(str, item["tool_use_id"]) for item in results_value),
        )
    if case.wire == "gemini":
        contents = cast(Sequence[Mapping[str, object]], call["contents"])
        assistant = cast(Sequence[Mapping[str, object]], contents[1]["parts"])
        results_value = cast(Sequence[Mapping[str, object]], contents[2]["parts"])
        return (
            tuple(
                cast(
                    str,
                    cast(Mapping[str, object], item["function_call"])["id"],
                )
                for item in assistant
                if "function_call" in item
            ),
            tuple(
                cast(
                    str,
                    cast(Mapping[str, object], item["function_response"])["id"],
                )
                for item in results_value
            ),
        )
    messages = cast(Sequence[Mapping[str, object]], call["messages"])
    assistant_calls = cast(
        Sequence[Mapping[str, object]],
        messages[1]["tool_calls"],
    )
    return (
        tuple(cast(str, item["id"]) for item in assistant_calls),
        tuple(cast(str, message["tool_call_id"]) for message in messages[2:]),
    )


def _rate_limit_error(case: _ProviderCase) -> Exception:
    error = RuntimeError(
        "vendor detail must not escape: Authorization: Bearer sk-phase8-secret"
    )
    if case.wire == "gemini":
        error.code = 429  # type: ignore[attr-defined]
        error.status = "RESOURCE_EXHAUSTED"  # type: ignore[attr-defined]
    else:
        error.status_code = 429  # type: ignore[attr-defined]
        if case.wire == "anthropic":
            error.body = {  # type: ignore[attr-defined]
                "error": {"type": "rate_limit_error"}
            }
    return error


def _authentication_error(case: _ProviderCase) -> Exception:
    error = RuntimeError("vendor detail must not escape")
    if case.wire == "gemini":
        error.code = 401  # type: ignore[attr-defined]
        error.status = "PERMISSION_DENIED"  # type: ignore[attr-defined]
    else:
        error.status_code = 401  # type: ignore[attr-defined]
        if case.wire == "anthropic":
            error.body = {  # type: ignore[attr-defined]
                "error": {"type": "authentication_error"}
            }
    return error


def _context_overflow_error(case: _ProviderCase) -> Exception:
    error = RuntimeError("vendor detail must not escape")
    if case.wire == "gemini":
        error.code = 400  # type: ignore[attr-defined]
        error.status = "CONTEXT_LENGTH_EXCEEDED"  # type: ignore[attr-defined]
    else:
        error.status_code = 413 if case.wire == "anthropic" else 400  # type: ignore[attr-defined]
        error.code = "context_length_exceeded"  # type: ignore[attr-defined]
        if case.wire == "anthropic":
            error.body = {  # type: ignore[attr-defined]
                "error": {"type": "context_length_exceeded"}
            }
    return error


def _request_timeout_error(case: _ProviderCase) -> Exception:
    error = RuntimeError("vendor timeout detail must not escape")
    if case.wire == "gemini":
        error.code = 408  # type: ignore[attr-defined]
        error.status = "DEADLINE_EXCEEDED"  # type: ignore[attr-defined]
    else:
        error.status_code = 408  # type: ignore[attr-defined]
    return error


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_text_usage_and_structured_output(
    case: _ProviderCase,
) -> None:
    provider, resource = case.make(_text_response(case, '{"answer":7}'))

    response = await provider.generate(
        _request(response_schema=SCHEMA),
    )

    assert response.provider_id == case.provider_id
    assert response.text == '{"answer":7}'
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert _structured_schema_from_call(case, resource.calls[0]) == SCHEMA


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_parallel_tools_and_continuation(
    case: _ProviderCase,
) -> None:
    ids = iter(("canonical-a", "canonical-b"))
    provider, resource = case.make(
        _tool_response(case),
        _text_response(case, "finished"),
        _text_response(case, "portable"),
        id_factory=lambda _prefix: next(ids),
    )

    first = await provider.generate(_request(tools=(TOOL,)))

    assert first.finish_reason is FinishReason.TOOL_CALLS
    assert [call.id for call in first.tool_calls] == [
        "canonical-a",
        "canonical-b",
    ]
    assert [call.provider_call_id for call in first.tool_calls] == [
        "provider-a",
        "provider-b",
    ]
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=first.tool_calls,
        provider_id=first.provider_id,
        provider_metadata=first.provider_metadata,
    )
    results = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=tuple(
            ToolResultBlock(call.id, {"value": index})
            for index, call in enumerate(first.tool_calls)
        ),
    )

    second = await provider.generate(
        _request(_message(MessageRole.USER, "lookup"), assistant, results),
    )

    assert second.text == "finished"
    requested_ids, result_ids = _native_followup_ids(case, resource)
    assert requested_ids == ("provider-a", "provider-b")
    assert result_ids == ("provider-a", "provider-b")

    foreign_assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=first.tool_calls,
        provider_id="foreign:model",
        provider_metadata=first.provider_metadata,
    )
    portable = await provider.generate(
        _request(
            _message(MessageRole.USER, "portable lookup"),
            foreign_assistant,
            results,
        ),
    )

    assert portable.text == "portable"
    requested_ids, result_ids = _native_followup_ids(
        case,
        resource,
        call_position=2,
    )
    assert requested_ids == ("canonical-a", "canonical-b")
    assert result_ids == ("canonical-a", "canonical-b")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_streams_preserve_text_and_terminal_usage(
    case: _ProviderCase,
) -> None:
    provider, _resource = case.make(stream=_text_stream(case))

    events = [event async for event in provider.stream(_request())]

    assert [event.text for event in events if isinstance(event, ModelTextDelta)] == [
        "hel",
        " ",
        "lo",
    ]
    terminal = events[-1]
    assert isinstance(terminal, ModelStreamCompleted)
    assert terminal.response.provider_id == case.provider_id
    assert terminal.response.text == "hel lo"
    assert terminal.response.usage.input_tokens == 11
    assert terminal.response.usage.output_tokens == 7


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_streams_normalize_tool_arguments(
    case: _ProviderCase,
) -> None:
    ids = iter(("stream-a", "stream-b"))
    provider, _resource = case.make(
        stream=_tool_stream(case),
        id_factory=lambda _prefix: next(ids),
    )

    events = [event async for event in provider.stream(_request(tools=(TOOL,)))]

    deltas = [event for event in events if isinstance(event, ModelToolCallDelta)]
    assert {event.index for event in deltas} == {0, 1}
    terminal = events[-1]
    assert isinstance(terminal, ModelStreamCompleted)
    assert [call.id for call in terminal.response.tool_calls] == [
        "stream-a",
        "stream-b",
    ]
    assert [call.arguments["key"] for call in terminal.response.tool_calls] == [
        "alpha",
        "beta",
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_normalizes_errors_and_cancellation(
    case: _ProviderCase,
) -> None:
    failed, _resource = case.make(_rate_limit_error(case))
    with pytest.raises(ModelProviderError) as captured:
        await failed.generate(_request())
    assert captured.value.code is ProviderErrorCode.RATE_LIMIT_ERROR
    assert "vendor detail" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    formatted = "".join(traceback.format_exception(captured.value))
    assert "vendor detail" not in formatted
    assert "sk-phase8-secret" not in formatted

    cancelled, _resource = case.make(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await cancelled.generate(_request())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_stream_errors_drop_raw_exception_diagnostics(
    case: _ProviderCase,
) -> None:
    provider, _resource = case.make(stream=(_rate_limit_error(case),))

    with pytest.raises(ModelProviderError) as captured:
        _ = [event async for event in provider.stream(_request())]

    assert captured.value.code is ProviderErrorCode.RATE_LIMIT_ERROR
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    formatted = "".join(traceback.format_exception(captured.value))
    assert "vendor detail" not in formatted
    assert "sk-phase8-secret" not in formatted


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    ("factory", "expected"),
    (
        (_authentication_error, ProviderErrorCode.AUTHENTICATION_ERROR),
        (_context_overflow_error, ProviderErrorCode.CONTEXT_OVERFLOW),
    ),
    ids=("authentication", "context-overflow"),
)
async def test_advertised_provider_normalizes_terminal_request_errors(
    case: _ProviderCase,
    factory: Callable[[_ProviderCase], Exception],
    expected: ProviderErrorCode,
) -> None:
    provider, _resource = case.make(factory(case))

    with pytest.raises(ModelProviderError) as captured:
        await provider.generate(_request())

    assert captured.value.code is expected
    assert "vendor detail" not in str(captured.value)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_normalizes_http_408_as_retryable_timeout(
    case: _ProviderCase,
) -> None:
    provider, _resource = case.make(_request_timeout_error(case))

    with pytest.raises(ModelProviderError) as captured:
        await provider.generate(_request())

    assert captured.value.code is ProviderErrorCode.TIMEOUT
    assert "vendor timeout detail" not in str(captured.value)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_rejects_missing_stream_terminal(
    case: _ProviderCase,
) -> None:
    provider, _resource = case.make(
        stream=_text_stream(case, terminal=False),
    )

    with pytest.raises(ModelProviderError) as captured:
        _ = [event async for event in provider.stream(_request())]

    assert captured.value.code is ProviderErrorCode.MALFORMED_RESPONSE


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_advertised_provider_lazy_import_uses_exact_extra_hint(
    case: _ProviderCase,
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
        if name == case.sdk_module or name.startswith(f"{case.sdk_module}."):
            raise ImportError("blocked for conformance")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    provider = case.make_without_client()

    with pytest.raises(
        ImportError,
        match=rf"pip install 'daita-agents\[{case.install_extra}\]'",
    ):
        await provider.generate(_request())
