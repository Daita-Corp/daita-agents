"""Decode Anthropic Messages events without owning native I/O."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ...errors import ModelProviderError, ProviderErrorCode
from ...models import (
    FinishReason,
    ModelResponse,
    ModelTextDelta,
    ModelToolCallDelta,
    ModelUsage,
    ToolCall,
)
from .messages import (
    _CONTINUATION_KEY,
    _plain_opaque_block,
    _validate_opaque_block,
)
from .usage import (
    _anthropic_inference_geo,
    _anthropic_service_tier,
    _AnthropicBillingUsage,
)
from .._fields import (
    field as _field,
    nonnegative_int as _nonnegative_int,
    optional_text as _optional_text,
    required_text as _required_text,
    usage_int as _usage_int,
)
from .adapter import (
    _NONCONTENT_STREAM_BLOCK_TYPES,
    _STREAM_MISSING,
    _code_from_error_type,
    _text_value,
)


@dataclass(slots=True)
class _StreamBlockState:
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
        progress: Callable[[str], None],
    ) -> None:
        self._provider_id = provider_id
        self._id_factory = id_factory
        self._progress = progress
        self._started = False
        self._terminal = False
        self._message_delta_seen = False
        self._response_id: str | None = None
        self._response_model: str | None = None
        self._stop_reason: str | None = None
        self._blocks: dict[int, _StreamBlockState] = {}
        self._calls: list[ToolCall] = []
        self._opaque_blocks: list[tuple[int, dict[str, object]]] = []
        self._canonical_ids: set[str] = set()
        self._provider_call_ids: set[str] = set()
        self._uncached_input_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._cache_write_5m_tokens = 0
        self._cache_write_1h_tokens = 0
        self._cache_write_breakdown_seen = False
        self._output_tokens = 0
        self._reasoning_tokens = 0
        self._service_tier: str | None = None
        self._inference_geo: str | None = None
        self._usage_fields_seen: set[str] = set()

    @property
    def response_model(self) -> str | None:
        return self._response_model

    def billing_usage(self) -> _AnthropicBillingUsage:
        breakdown_complete = (
            self._cache_write_tokens == 0 and not self._cache_write_breakdown_seen
        ) or (
            self._cache_write_breakdown_seen
            and self._cache_write_5m_tokens + self._cache_write_1h_tokens
            == self._cache_write_tokens
        )
        return _AnthropicBillingUsage(
            usage=ModelUsage(
                input_tokens=(
                    self._uncached_input_tokens
                    + self._cache_read_tokens
                    + self._cache_write_tokens
                ),
                output_tokens=self._output_tokens,
                reasoning_tokens=self._reasoning_tokens,
                cache_read_tokens=self._cache_read_tokens,
                cache_write_tokens=self._cache_write_tokens,
            ),
            token_counts_complete={
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "output_tokens",
            }
            <= self._usage_fields_seen,
            cache_write_5m_tokens=self._cache_write_5m_tokens,
            cache_write_1h_tokens=self._cache_write_1h_tokens,
            cache_write_breakdown_complete=breakdown_complete,
            service_tier=self._service_tier,
            inference_geo=self._inference_geo,
        )

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
        if (
            event_type
            in {
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            }
            and not self._started
        ):
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
        # Anthropic may add event types without a version change. Unknown
        # lifecycle or metadata events are not canonical model output.
        return []

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
                if self._stop_reason == "max_tokens":
                    raise ModelProviderError(
                        ProviderErrorCode.OUTPUT_LIMIT,
                        "Anthropic exhausted the output token limit",
                    )
                if self._stop_reason == "model_context_window_exceeded":
                    raise ModelProviderError(
                        ProviderErrorCode.CONTEXT_OVERFLOW,
                        "Anthropic exhausted the context window",
                    )
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
                reasoning_tokens=self._reasoning_tokens,
                cache_read_tokens=self._cache_read_tokens,
                cache_write_tokens=self._cache_write_tokens,
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
        self._response_model = _optional_text(
            _field(message, "model", None),
            "stream response model",
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
        state = _StreamBlockState(block_type=block_type)
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
        if block_type in _NONCONTENT_STREAM_BLOCK_TYPES:
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
            fragment = _text_value(_field(delta, "text"), "text delta")
            self._progress(fragment)
            return self._append_text(state, fragment)
        if delta_type == "input_json_delta" and state.block_type == "tool_use":
            partial_json = _text_value(
                _field(delta, "partial_json"),
                "tool input JSON delta",
            )
            self._progress(partial_json)
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
            fragment = _text_value(_field(delta, "thinking"), "thinking delta")
            self._progress(fragment)
            state.thinking_fragments.append(fragment)
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
        if state.block_type == "tool_use":
            encoded_arguments = "".join(state.arguments_fragments)
            arguments = {} if not encoded_arguments else json.loads(encoded_arguments)
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
        elif (
            state.block_type != "text"
            and state.block_type not in _NONCONTENT_STREAM_BLOCK_TYPES
        ):
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
            # SDK delta models expose omitted optional counters as None. They
            # carry no new measurement and must not erase earlier usage.
            if value is not _STREAM_MISSING and value is not None:
                setattr(self, attribute, _usage_int(value, label))
                self._usage_fields_seen.add(name)
        cache_creation = _field(usage, "cache_creation", _STREAM_MISSING)
        if cache_creation is not _STREAM_MISSING and cache_creation is not None:
            self._cache_write_5m_tokens = _usage_int(
                _field(cache_creation, "ephemeral_5m_input_tokens"),
                "5-minute cache creation input tokens",
            )
            self._cache_write_1h_tokens = _usage_int(
                _field(cache_creation, "ephemeral_1h_input_tokens"),
                "1-hour cache creation input tokens",
            )
            self._cache_write_breakdown_seen = True
        output_details = _field(usage, "output_tokens_details", _STREAM_MISSING)
        if output_details is not _STREAM_MISSING and output_details is not None:
            self._reasoning_tokens = _usage_int(
                _field(output_details, "thinking_tokens", 0),
                "thinking tokens",
            )
        service_tier = _field(usage, "service_tier", _STREAM_MISSING)
        if service_tier is not _STREAM_MISSING and service_tier is not None:
            self._service_tier = _anthropic_service_tier(service_tier)
        inference_geo = _field(usage, "inference_geo", _STREAM_MISSING)
        if inference_geo is not _STREAM_MISSING and inference_geo is not None:
            self._inference_geo = _anthropic_inference_geo(inference_geo)
