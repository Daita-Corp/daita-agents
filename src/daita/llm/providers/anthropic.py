"""Anthropic Messages adapter for the provider-neutral model boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
import json
from typing import Protocol, cast
from uuid import uuid4

from ..._json import FrozenJsonObject, canonical_json
from ..errors import ModelProviderError, ProviderErrorCode, detached_provider_error
from ..models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)

_CONTINUATION_KEY = "anthropic_continuation"
_OPAQUE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
_STREAM_MISSING = object()


class _MessagesResource(Protocol):
    async def create(self, **kwargs: object) -> object: ...

    def stream(self, **kwargs: object) -> _MessageStreamManager: ...


class _MessageStreamManager(Protocol):
    async def __aenter__(self) -> AsyncIterator[object]: ...

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None: ...


class _AnthropicClient(Protocol):
    @property
    def messages(self) -> _MessagesResource: ...


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class AnthropicMessagesProvider:
    """Translate canonical requests to Anthropic's Messages API only."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1_024,
        api_key: str | None = None,
        client: _AnthropicClient | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if id_factory is not None and not callable(id_factory):
            raise TypeError("id_factory must be callable")
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = client
        self._id_factory = _new_id if id_factory is None else id_factory

    @property
    def provider_id(self) -> str:
        return f"anthropic:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return request.allow_parallel_tool_calls is None

    @property
    def client(self) -> _AnthropicClient:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as error:
                raise ImportError(
                    "anthropic is required. Install with: "
                    "pip install 'daita-agents[anthropic]'"
                ) from error
            self._client = cast(
                _AnthropicClient,
                AsyncAnthropic(api_key=self._api_key),
            )
        return self._client

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        failure: ModelProviderError | None = None
        try:
            return await self._generate(request)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError as error:
            failure = error
        except Exception:
            failure = ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "Anthropic provider boundary failed",
            )
        if failure is None:
            raise AssertionError("Anthropic provider failed without an error")
        raise detached_provider_error(failure)

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._require_supported_request_policy(request)
        arguments = self._request_arguments(request)
        try:
            response = await self.client.messages.create(**arguments)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error) from error
        try:
            return self._decode_response(response)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "Anthropic returned a malformed response",
            ) from error

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        failure: ModelProviderError | None = None
        try:
            async for event in self._stream(request):
                yield event
            return
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError as error:
            failure = error
        except Exception:
            failure = ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "Anthropic provider boundary failed",
            )
        if failure is None:
            raise AssertionError("Anthropic provider failed without an error")
        raise detached_provider_error(failure)

    async def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._require_supported_request_policy(request)
        arguments = self._request_arguments(request)
        decoder = _AnthropicStreamDecoder(
            provider_id=self.provider_id,
            id_factory=self._id_factory,
        )
        try:
            manager = self.client.messages.stream(**arguments)
            async with manager as native_stream:
                async for native_event in native_stream:
                    try:
                        canonical_events = decoder.consume(native_event)
                    except ModelProviderError:
                        raise
                    except (KeyError, TypeError, ValueError) as error:
                        raise _malformed_stream(error) from error
                    for canonical_event in canonical_events:
                        yield canonical_event
            try:
                response = decoder.finish()
            except ModelProviderError:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise _malformed_stream(error) from error
            yield ModelStreamCompleted(response)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error) from error

    def _require_supported_request_policy(self, request: ModelRequest) -> None:
        if not self.supports_request_policy(request):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic cannot enforce the requested tool-call policy",
            )

    def _request_arguments(self, request: ModelRequest) -> dict[str, object]:
        system, messages = _message_input(request.messages, self.provider_id)
        arguments: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system is not None:
            arguments["system"] = system
        if request.tools:
            arguments["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": FrozenJsonObject.from_mapping(
                        tool.input_schema
                    ).to_dict(),
                }
                for tool in request.tools
            ]
        if request.response_schema is not None:
            arguments["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": FrozenJsonObject.from_mapping(
                        request.response_schema
                    ).to_dict(),
                }
            }
        return arguments

    def _decode_response(self, response: object) -> ModelResponse:
        response_type = _required_text(_field(response, "type"), "response type")
        if response_type != "message":
            raise ValueError("response type must be message")
        role = _required_text(_field(response, "role"), "response role")
        if role != "assistant":
            raise ValueError("response role must be assistant")
        response_id = _required_text(_field(response, "id"), "response id")
        stop_reason = _required_text(
            _field(response, "stop_reason"),
            "response stop_reason",
        )
        content = _field(response, "content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            raise ValueError("response content must be a sequence")

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        replay_blocks: list[dict[str, object]] = []
        canonical_ids: set[str] = set()
        for block in content:
            block_type = _required_text(_field(block, "type"), "content block type")
            if block_type == "text":
                text_parts.append(
                    _required_text(_field(block, "text"), "content block text")
                )
            elif block_type == "tool_use":
                provider_call_id = _required_text(
                    _field(block, "id"),
                    "tool-use id",
                )
                name = _required_text(_field(block, "name"), "tool-use name")
                arguments = _field(block, "input")
                if not isinstance(arguments, Mapping):
                    raise ValueError("tool-use input must be an object")
                canonical_id = self._id_factory("call")
                if canonical_id in canonical_ids:
                    raise ValueError("id_factory returned a duplicate call ID")
                canonical_ids.add(canonical_id)
                calls.append(
                    ToolCall(
                        id=canonical_id,
                        provider_call_id=provider_call_id,
                        name=name,
                        arguments=FrozenJsonObject.from_mapping(arguments),
                    )
                )
            elif block_type in _OPAQUE_BLOCK_TYPES:
                replay_blocks.append(_plain_opaque_block(block, block_type))
            else:
                raise ValueError("response contains an unsupported content block")

        text = "\n".join(text_parts).strip() or None
        if calls:
            if stop_reason != "tool_use":
                raise ValueError("tool-use content requires tool_use stop_reason")
            finish_reason = FinishReason.TOOL_CALLS
        else:
            if text is None:
                raise ValueError("response contains neither text nor tool calls")
            if stop_reason in {"end_turn", "stop_sequence"}:
                finish_reason = FinishReason.STOP
            elif stop_reason in {"max_tokens", "model_context_window_exceeded"}:
                finish_reason = FinishReason.LENGTH
            elif stop_reason == "refusal":
                raise ModelProviderError(
                    ProviderErrorCode.CONTENT_BLOCKED,
                    "Anthropic blocked the response",
                )
            else:
                raise ValueError("response contains an unsupported stop_reason")

        provider_metadata: dict[str, object] = {}
        if replay_blocks or calls:
            provider_metadata[_CONTINUATION_KEY] = {
                "provider_id": self.provider_id,
                "content_blocks": replay_blocks,
            }
        return ModelResponse(
            finish_reason=finish_reason,
            text=text,
            tool_calls=tuple(calls),
            usage=_decode_usage(_field(response, "usage", None)),
            provider_id=self.provider_id,
            provider_response_id=response_id,
            provider_metadata=provider_metadata,
        )


AnthropicProvider = AnthropicMessagesProvider


@dataclass(slots=True)
class _StreamBlockState:
    native_index: int
    block_type: str
    text_fragments: list[str] = field(default_factory=list)
    tool_index: int | None = None
    canonical_id: str | None = None
    provider_call_id: str | None = None
    name: str | None = None
    arguments_fragments: list[str] = field(default_factory=list)
    thinking_fragments: list[str] = field(default_factory=list)
    signature_fragments: list[str] = field(default_factory=list)
    opaque_block: dict[str, object] | None = None
    closed: bool = False


class _AnthropicStreamDecoder:
    """Strictly assemble one native Messages event stream."""

    def __init__(
        self,
        *,
        provider_id: str,
        id_factory: Callable[[str], str],
    ) -> None:
        self._provider_id = provider_id
        self._id_factory = id_factory
        self._started = False
        self._terminal = False
        self._message_delta_seen = False
        self._response_id: str | None = None
        self._stop_reason: str | None = None
        self._blocks: dict[int, _StreamBlockState] = {}
        self._calls: list[ToolCall] = []
        self._opaque_blocks: list[tuple[int, dict[str, object]]] = []
        self._canonical_ids: set[str] = set()
        self._provider_call_ids: set[str] = set()
        self._uncached_input_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._output_tokens = 0

    def consume(
        self,
        event: object,
    ) -> list[ModelTextDelta | ModelToolCallDelta]:
        event_type = _required_text(_field(event, "type"), "stream event type")
        if self._terminal:
            raise ValueError("stream emitted an event after message_stop")
        if event_type == "ping":
            return []
        if event_type == "error":
            native_error = _field(event, "error")
            error_type = _required_text(
                _field(native_error, "type"),
                "stream error type",
            )
            code = _code_from_error_type(error_type)
            raise ModelProviderError(
                code,
                f"Anthropic stream failed: {code.value}",
            )
        if event_type == "message_start":
            self._consume_message_start(event)
            return []
        if not self._started:
            raise ValueError("stream content preceded message_start")
        if event_type == "content_block_start":
            return self._consume_block_start(event)
        if event_type == "content_block_delta":
            return self._consume_block_delta(event)
        if event_type == "content_block_stop":
            self._consume_block_stop(event)
            return []
        if event_type == "message_delta":
            self._consume_message_delta(event)
            return []
        if event_type == "message_stop":
            self._consume_message_stop()
            return []
        raise ValueError("stream contains an unsupported event type")

    def finish(self) -> ModelResponse:
        if not self._started or not self._terminal:
            raise ValueError("stream ended without message_stop")
        if self._response_id is None or self._stop_reason is None:
            raise ValueError("terminal stream is missing message metadata")
        if any(not block.closed for block in self._blocks.values()):
            raise ValueError("terminal stream contains an open content block")

        text = (
            "\n".join(
                "".join(block.text_fragments)
                for block in self._blocks.values()
                if block.block_type == "text"
            ).strip()
            or None
        )
        if self._calls:
            if self._stop_reason != "tool_use":
                raise ValueError("streamed tool calls require tool_use stop_reason")
            finish_reason = FinishReason.TOOL_CALLS
        else:
            if text is None:
                raise ValueError("stream contains neither text nor tool calls")
            if self._stop_reason in {"end_turn", "stop_sequence"}:
                finish_reason = FinishReason.STOP
            elif self._stop_reason in {
                "max_tokens",
                "model_context_window_exceeded",
            }:
                finish_reason = FinishReason.LENGTH
            elif self._stop_reason == "refusal":
                raise ModelProviderError(
                    ProviderErrorCode.CONTENT_BLOCKED,
                    "Anthropic blocked the response",
                )
            else:
                raise ValueError("stream contains an unsupported stop_reason")

        replay_blocks = [
            block
            for _index, block in sorted(
                self._opaque_blocks,
                key=lambda item: item[0],
            )
        ]
        provider_metadata: dict[str, object] = {}
        if replay_blocks or self._calls:
            provider_metadata[_CONTINUATION_KEY] = {
                "provider_id": self._provider_id,
                "content_blocks": replay_blocks,
            }
        return ModelResponse(
            finish_reason=finish_reason,
            text=text,
            tool_calls=tuple(self._calls),
            usage=ModelUsage(
                input_tokens=(
                    self._uncached_input_tokens
                    + self._cache_read_tokens
                    + self._cache_write_tokens
                ),
                output_tokens=self._output_tokens,
                cache_read_tokens=self._cache_read_tokens,
                cache_write_tokens=self._cache_write_tokens,
                estimated_cost_usd=Decimal("0"),
            ),
            provider_id=self._provider_id,
            provider_response_id=self._response_id,
            provider_metadata=provider_metadata,
        )

    def _consume_message_start(self, event: object) -> None:
        if self._started:
            raise ValueError("stream contains duplicate message_start")
        message = _field(event, "message")
        if _required_text(_field(message, "type"), "stream message type") != (
            "message"
        ):
            raise ValueError("stream message type must be message")
        if _required_text(_field(message, "role"), "stream message role") != (
            "assistant"
        ):
            raise ValueError("stream message role must be assistant")
        content = _field(message, "content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            raise ValueError("stream message content must be a sequence")
        if content:
            raise ValueError("message_start content must be empty")
        if _field(message, "stop_reason", None) is not None:
            raise ValueError("message_start stop_reason must be null")
        self._response_id = _required_text(
            _field(message, "id"),
            "stream message id",
        )
        self._update_usage(_field(message, "usage", None))
        self._started = True

    def _consume_block_start(
        self,
        event: object,
    ) -> list[ModelTextDelta | ModelToolCallDelta]:
        if self._message_delta_seen:
            raise ValueError("content block started after message_delta")
        if any(not block.closed for block in self._blocks.values()):
            raise ValueError("content blocks must not overlap")
        index = _nonnegative_int(_field(event, "index"), "content block index")
        if index != len(self._blocks):
            raise ValueError("content block indices must be contiguous")
        content_block = _field(event, "content_block")
        block_type = _required_text(
            _field(content_block, "type"),
            "content block type",
        )
        state = _StreamBlockState(native_index=index, block_type=block_type)
        self._blocks[index] = state
        if block_type == "text":
            initial_text = _text_value(
                _field(content_block, "text"),
                "initial text",
            )
            return self._append_text(state, initial_text)
        if block_type == "tool_use":
            initial_input = _field(content_block, "input")
            if not isinstance(initial_input, Mapping) or initial_input:
                raise ValueError("streamed tool-use input must start empty")
            provider_call_id = _required_text(
                _field(content_block, "id"),
                "streamed tool-use id",
            )
            if provider_call_id in self._provider_call_ids:
                raise ValueError("stream contains duplicate provider tool-use IDs")
            canonical_id = self._id_factory("call")
            if canonical_id in self._canonical_ids:
                raise ValueError("id_factory returned a duplicate call ID")
            name = _required_text(
                _field(content_block, "name"),
                "streamed tool-use name",
            )
            self._provider_call_ids.add(provider_call_id)
            self._canonical_ids.add(canonical_id)
            state.tool_index = len(self._calls)
            state.canonical_id = canonical_id
            state.provider_call_id = provider_call_id
            state.name = name
            return [
                ModelToolCallDelta(
                    index=state.tool_index,
                    arguments_delta="",
                    id=canonical_id,
                    name=name,
                    provider_call_id=provider_call_id,
                )
            ]
        if block_type == "thinking":
            thinking = _text_value(
                _field(content_block, "thinking", ""),
                "initial thinking",
            )
            signature = _text_value(
                _field(content_block, "signature", ""),
                "initial thinking signature",
            )
            if thinking:
                state.thinking_fragments.append(thinking)
            if signature:
                state.signature_fragments.append(signature)
            return []
        if block_type == "redacted_thinking":
            state.opaque_block = _plain_opaque_block(
                content_block,
                block_type,
            )
            return []
        raise ValueError("stream contains an unsupported content block")

    def _consume_block_delta(
        self,
        event: object,
    ) -> list[ModelTextDelta | ModelToolCallDelta]:
        index = _nonnegative_int(_field(event, "index"), "content block index")
        state = self._open_block(index)
        delta = _field(event, "delta")
        delta_type = _required_text(_field(delta, "type"), "content delta type")
        if delta_type == "text_delta" and state.block_type == "text":
            return self._append_text(
                state,
                _text_value(_field(delta, "text"), "text delta"),
            )
        if delta_type == "input_json_delta" and state.block_type == "tool_use":
            partial_json = _text_value(
                _field(delta, "partial_json"),
                "tool input JSON delta",
            )
            state.arguments_fragments.append(partial_json)
            if state.tool_index is None:
                raise ValueError("streamed tool-use index is missing")
            return [
                ModelToolCallDelta(
                    index=state.tool_index,
                    arguments_delta=partial_json,
                )
            ]
        if delta_type == "thinking_delta" and state.block_type == "thinking":
            state.thinking_fragments.append(
                _text_value(_field(delta, "thinking"), "thinking delta")
            )
            return []
        if delta_type == "signature_delta" and state.block_type == "thinking":
            state.signature_fragments.append(
                _text_value(_field(delta, "signature"), "signature delta")
            )
            return []
        raise ValueError("content delta does not match its open block")

    def _consume_block_stop(self, event: object) -> None:
        index = _nonnegative_int(_field(event, "index"), "content block index")
        state = self._open_block(index)
        if state.block_type == "text":
            if not "".join(state.text_fragments).strip():
                raise ValueError("streamed text block is empty")
        elif state.block_type == "tool_use":
            encoded_arguments = "".join(state.arguments_fragments)
            if not encoded_arguments:
                raise ValueError("streamed tool-use arguments are missing")
            arguments = json.loads(encoded_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("streamed tool-use arguments must be an object")
            if (
                state.tool_index is None
                or state.canonical_id is None
                or state.provider_call_id is None
                or state.name is None
                or state.tool_index != len(self._calls)
            ):
                raise ValueError("streamed tool-use identity is incomplete")
            self._calls.append(
                ToolCall(
                    id=state.canonical_id,
                    provider_call_id=state.provider_call_id,
                    name=state.name,
                    arguments=arguments,
                )
            )
        elif state.block_type == "thinking":
            opaque: dict[str, object] = {
                "type": "thinking",
                "thinking": "".join(state.thinking_fragments),
                "signature": "".join(state.signature_fragments),
            }
            _validate_opaque_block(opaque, "thinking")
            self._opaque_blocks.append((index, opaque))
        elif state.block_type == "redacted_thinking":
            if state.opaque_block is None:
                raise ValueError("redacted thinking block is missing")
            self._opaque_blocks.append((index, state.opaque_block))
        else:
            raise ValueError("stream contains an unsupported content block")
        state.closed = True

    def _consume_message_delta(self, event: object) -> None:
        if not self._blocks or any(not block.closed for block in self._blocks.values()):
            raise ValueError("message_delta preceded complete content blocks")
        delta = _field(event, "delta")
        stop_value = _field(delta, "stop_reason", None)
        if stop_value is not None:
            stop_reason = _required_text(stop_value, "stream stop_reason")
            if self._stop_reason is not None and self._stop_reason != stop_reason:
                raise ValueError("stream stop_reason changed")
            self._stop_reason = stop_reason
        self._update_usage(_field(event, "usage", None))
        self._message_delta_seen = True

    def _consume_message_stop(self) -> None:
        if not self._message_delta_seen or self._stop_reason is None:
            raise ValueError("message_stop preceded terminal message_delta")
        if any(not block.closed for block in self._blocks.values()):
            raise ValueError("message_stop preceded content block completion")
        self._terminal = True

    def _open_block(self, index: int) -> _StreamBlockState:
        state = self._blocks.get(index)
        if state is None or state.closed:
            raise ValueError("stream delta references no open content block")
        return state

    def _append_text(
        self,
        state: _StreamBlockState,
        fragment: str,
    ) -> list[ModelTextDelta | ModelToolCallDelta]:
        state.text_fragments.append(fragment)
        if not fragment:
            return []
        return [ModelTextDelta(fragment)]

    def _update_usage(self, usage: object) -> None:
        for name, attribute, label in (
            ("input_tokens", "_uncached_input_tokens", "input tokens"),
            (
                "cache_read_input_tokens",
                "_cache_read_tokens",
                "cache read input tokens",
            ),
            (
                "cache_creation_input_tokens",
                "_cache_write_tokens",
                "cache creation input tokens",
            ),
            ("output_tokens", "_output_tokens", "output tokens"),
        ):
            value = _field(usage, name, _STREAM_MISSING)
            if value is not _STREAM_MISSING:
                setattr(self, attribute, _usage_int(value, label))


def _message_input(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> tuple[str | None, list[dict[str, object]]]:
    system_parts: list[str] = []
    provider_messages: list[dict[str, object]] = []
    provider_call_ids: dict[str, str] = {}
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            system_parts.extend(
                block.text for block in message.content if isinstance(block, TextBlock)
            )
            continue

        content: list[dict[str, object]] = []
        if message.role is MessageRole.ASSISTANT:
            content.extend(_continuation_blocks(message, provider_id))
        content.extend(
            {"type": "text", "text": block.text}
            for block in message.content
            if isinstance(block, TextBlock)
        )
        if message.role is MessageRole.ASSISTANT:
            same_origin = _same_origin(message, provider_id)
            for call in message.tool_calls:
                provider_call_id = (
                    call.provider_call_id
                    if same_origin and call.provider_call_id is not None
                    else call.id
                )
                provider_call_ids[call.id] = provider_call_id
                content.append(
                    {
                        "type": "tool_use",
                        "id": provider_call_id,
                        "name": call.name,
                        "input": FrozenJsonObject.from_mapping(
                            call.arguments
                        ).to_dict(),
                    }
                )
        elif message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": provider_call_ids.get(
                            block.call_id,
                            block.call_id,
                        ),
                        "content": canonical_json(block.output),
                        "is_error": block.is_error,
                    }
                )
        if not content:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "canonical message produced no Anthropic content blocks",
            )
        role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
        provider_messages.append({"role": role, "content": content})

    if not provider_messages:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no Anthropic messages",
        )
    system = "\n\n".join(part for part in system_parts if part.strip()).strip()
    return system or None, provider_messages


def _continuation_blocks(
    message: CanonicalMessage,
    provider_id: str,
) -> list[dict[str, object]]:
    if message.provider_id != provider_id:
        return []
    continuation_value = message.provider_metadata.get(_CONTINUATION_KEY)
    if continuation_value is None:
        return []
    if not isinstance(continuation_value, FrozenJsonObject):
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation metadata must be an object",
        )
    continuation = continuation_value.to_dict()
    origin = continuation.get("provider_id")
    if not isinstance(origin, str) or not origin.strip():
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation metadata requires a provider origin",
        )
    if origin != provider_id:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation origin does not match canonical provider",
        )
    blocks = continuation.get("content_blocks")
    if not isinstance(blocks, list):
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "Anthropic continuation metadata requires content blocks",
        )
    decoded: list[dict[str, object]] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic continuation content blocks must be objects",
            )
        block_type = block.get("type")
        if block_type not in _OPAQUE_BLOCK_TYPES:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic continuation contains an unsupported content block",
            )
        try:
            normalized = FrozenJsonObject.from_mapping(block).to_dict()
            _validate_opaque_block(normalized, cast(str, block_type))
        except (TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic continuation contains a malformed content block",
            ) from error
        decoded.append(normalized)
    return decoded


def _same_origin(message: CanonicalMessage, provider_id: str) -> bool:
    return message.provider_id == provider_id


def _plain_opaque_block(
    block: object,
    block_type: str,
) -> dict[str, object]:
    plain: dict[str, object]
    if isinstance(block, FrozenJsonObject):
        plain = block.to_dict()
    elif isinstance(block, Mapping):
        plain = FrozenJsonObject.from_mapping(block).to_dict()
    else:
        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json", exclude_none=True)
            if not isinstance(dumped, Mapping):
                raise ValueError("opaque content block dump must be an object")
            plain = FrozenJsonObject.from_mapping(dumped).to_dict()
        else:
            plain = {"type": block_type}
            field_names = (
                ("thinking", "signature") if block_type == "thinking" else ("data",)
            )
            for name in field_names:
                value = _field(block, name, None)
                if value is not None:
                    plain[name] = value
            plain = FrozenJsonObject.from_mapping(plain).to_dict()
    _validate_opaque_block(plain, block_type)
    return plain


def _validate_opaque_block(block: Mapping[str, object], block_type: str) -> None:
    if block.get("type") != block_type:
        raise ValueError("opaque content block type changed during normalization")
    if block_type == "thinking":
        _required_text(block.get("thinking"), "thinking content")
        _required_text(block.get("signature"), "thinking signature")
    elif block_type == "redacted_thinking":
        _required_text(block.get("data"), "redacted thinking data")
    else:
        raise ValueError("unsupported opaque content block")


def _decode_usage(value: object) -> ModelUsage:
    if value is None:
        return ModelUsage()
    uncached_input = _usage_int(
        _field(value, "input_tokens", 0),
        "input tokens",
    )
    cache_read = _usage_int(
        _field(value, "cache_read_input_tokens", 0),
        "cache read input tokens",
    )
    cache_write = _usage_int(
        _field(value, "cache_creation_input_tokens", 0),
        "cache creation input tokens",
    )
    return ModelUsage(
        input_tokens=uncached_input + cache_read + cache_write,
        output_tokens=_usage_int(
            _field(value, "output_tokens", 0),
            "output tokens",
        ),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        estimated_cost_usd=Decimal("0"),
    )


def _malformed_stream(error: Exception) -> ModelProviderError:
    return ModelProviderError(
        ProviderErrorCode.MALFORMED_RESPONSE,
        "Anthropic returned a malformed stream",
    )


def _code_from_error_type(error_type: str) -> ProviderErrorCode:
    if error_type in {"authentication_error", "permission_error"}:
        return ProviderErrorCode.AUTHENTICATION_ERROR
    if error_type == "rate_limit_error":
        return ProviderErrorCode.RATE_LIMIT_ERROR
    if error_type in {"not_found_error", "model_not_found"}:
        return ProviderErrorCode.MODEL_NOT_FOUND
    if error_type in {
        "request_too_large",
        "context_length_exceeded",
        "context_window_exceeded",
    }:
        return ProviderErrorCode.CONTEXT_OVERFLOW
    if error_type in {
        "content_blocked",
        "content_policy_violation",
        "safety_error",
    }:
        return ProviderErrorCode.CONTENT_BLOCKED
    if error_type == "invalid_request_error":
        return ProviderErrorCode.INVALID_REQUEST
    return ProviderErrorCode.PROVIDER_UNAVAILABLE


def _normalize_error(error: Exception) -> ModelProviderError:
    status_value = _field(error, "status_code", None)
    status = (
        status_value
        if isinstance(status_value, int) and not isinstance(status_value, bool)
        else None
    )
    error_type = _provider_error_type(error)
    name = type(error).__name__.lower()
    if (
        isinstance(error, (asyncio.TimeoutError, TimeoutError))
        or status == 408
        or "timeout" in name
    ):
        normalized = ProviderErrorCode.TIMEOUT
    elif (
        status in {401, 403}
        or error_type in {"authentication_error", "permission_error"}
        or "authentication" in name
        or "permission" in name
    ):
        normalized = ProviderErrorCode.AUTHENTICATION_ERROR
    elif (
        status == 429
        or error_type == "rate_limit_error"
        or "ratelimit" in name
        or "rate_limit" in name
    ):
        normalized = ProviderErrorCode.RATE_LIMIT_ERROR
    elif (
        status == 404
        or error_type in {"not_found_error", "model_not_found"}
        or "notfound" in name
        or "not_found" in name
    ):
        normalized = ProviderErrorCode.MODEL_NOT_FOUND
    elif status == 413 or error_type in {
        "request_too_large",
        "context_length_exceeded",
        "context_window_exceeded",
    }:
        normalized = ProviderErrorCode.CONTEXT_OVERFLOW
    elif error_type in {
        "content_blocked",
        "content_policy_violation",
        "safety_error",
    }:
        normalized = ProviderErrorCode.CONTENT_BLOCKED
    elif (
        status is not None and 400 <= status < 500
    ) or error_type == "invalid_request_error":
        normalized = ProviderErrorCode.INVALID_REQUEST
    elif (
        isinstance(error, ConnectionError)
        or status is not None
        and status >= 500
        or error_type in {"api_error", "overloaded_error"}
        or "connection" in name
    ):
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    else:
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    return ModelProviderError(
        normalized,
        f"Anthropic request failed: {normalized.value}",
    )


def _provider_error_type(error: Exception) -> str | None:
    for value in (
        _field(error, "code", None),
        _field(error, "type", None),
    ):
        if isinstance(value, str) and value.strip():
            return value
    body = _field(error, "body", None)
    nested = _field(body, "error", None)
    value = _field(nested, "type", None)
    if isinstance(value, str) and value.strip():
        return value
    return None


_MISSING = object()


def _field(value: object, name: str, default: object = _MISSING) -> object:
    if value is None:
        if default is not _MISSING:
            return default
        raise KeyError(name)
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise KeyError(name)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _text_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _usage_int(value: object, label: str) -> int:
    if value is None:
        return 0
    return _nonnegative_int(value, label)


__all__ = ["AnthropicMessagesProvider", "AnthropicProvider"]
