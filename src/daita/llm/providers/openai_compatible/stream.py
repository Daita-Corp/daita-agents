"""Decode Chat Completions events without constructing native streams."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from ..._lifecycle import AttemptLifecycle
from ...errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from ...models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
    ToolCall,
)
from .._fields import (
    field as _field,
    nonnegative_int as _nonnegative_int,
    optional_text as _optional_text,
    required_text as _required_text,
    safe_structural_token as _safe_structural_token,
)
from .messages import _CONTINUATION_KEY


async def decode_compatible_stream(
    self: Any,
    source: AsyncIterator[object],
    request: ModelRequest,
    attempt: AttemptLifecycle,
    *,
    requested_at: datetime,
) -> AsyncIterator[ModelStreamEvent]:
    from .adapter import (
        _finish_reason,
        _normalize_error,
        _sequence,
        _StreamedToolCall,
    )

    text_fragments: list[str] = []
    tool_states: dict[int, _StreamedToolCall] = {}
    finish_reason: str | None = None
    usage_value: object | None = None
    response_id: str | None = None
    response_model: str | None = None
    service_tier: str | None = None
    terminal_event: ModelStreamCompleted | None = None
    async with attempt.stream(source) as iterator:
        while True:
            try:
                chunk = await anext(iterator)
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                raise
            except ModelProviderError:
                raise
            except Exception as error:
                raise _normalize_error(error, self.provider) from error
            try:
                chunk_id = _optional_text(_field(chunk, "id", None), "chunk id")
                if chunk_id is not None:
                    if response_id is not None and response_id != chunk_id:
                        raise ValueError("stream response ID changed")
                    response_id = chunk_id
                chunk_model = _optional_text(
                    _field(chunk, "model", None),
                    "stream response model",
                )
                if chunk_model is not None:
                    if response_model is not None and response_model != chunk_model:
                        raise ValueError("stream response model changed")
                    response_model = chunk_model
                chunk_service_tier = _optional_text(
                    _field(chunk, "service_tier", None),
                    "stream service tier",
                )
                if chunk_service_tier is not None:
                    if service_tier is not None and service_tier != chunk_service_tier:
                        raise ValueError("stream service tier changed")
                    service_tier = chunk_service_tier
                chunk_usage = _field(chunk, "usage", None)
                if chunk_usage is not None:
                    usage_value = chunk_usage
                choices = _sequence(
                    _field(chunk, "choices", ()),
                    "stream choices",
                )
                if len(choices) > 1:
                    raise ValueError("stream must contain at most one choice")
                if not choices:
                    # OpenAI-compatible servers may emit metadata-only or
                    # heartbeat chunks. Terminal validity is checked below.
                    continue
                choice = choices[0]
                choice_index = _field(choice, "index", 0)
                if choice_index != 0:
                    raise ValueError("stream choice index must be zero")
                delta = _field(choice, "delta")
                refusal = _optional_text(
                    _field(delta, "refusal", None),
                    "stream refusal",
                )
                if refusal is not None:
                    raise ModelProviderError(
                        ProviderErrorCode.CONTENT_BLOCKED,
                        f"{self.provider} blocked the response",
                    )
                content = _field(delta, "content", None)
                if content is not None:
                    if not isinstance(content, str):
                        raise ValueError("stream content must be text")
                    text_fragments.append(content)
                    if content:
                        attempt.progress(content)
                        yield ModelTextDelta(content)
                for field in ("reasoning", "reasoning_content"):
                    reasoning = _field(delta, field, None)
                    if reasoning is not None:
                        if not isinstance(reasoning, str):
                            raise ValueError("reasoning delta must be text")
                        attempt.progress(reasoning)
                raw_calls = _field(delta, "tool_calls", ())
                if raw_calls is None:
                    raw_calls = ()
                for item in _sequence(raw_calls, "stream tool calls"):
                    index = _nonnegative_int(
                        _field(item, "index"),
                        "stream tool index",
                    )
                    state = tool_states.get(index)
                    is_first = state is None
                    if state is None:
                        state = _StreamedToolCall(self._id_factory("call"))
                        tool_states[index] = state
                    native_id = _optional_text(
                        _field(item, "id", None),
                        "stream provider call ID",
                    )
                    if native_id is not None:
                        if (
                            state.provider_call_id is not None
                            and state.provider_call_id != native_id
                        ):
                            raise ValueError("stream provider call ID changed")
                        state.provider_call_id = native_id
                    function = _field(item, "function", None)
                    name: str | None = None
                    argument_delta = ""
                    if function is not None:
                        name = _optional_text(
                            _field(function, "name", None),
                            "stream tool name",
                        )
                        if name is not None:
                            if state.name is not None and state.name != name:
                                raise ValueError("stream tool name changed")
                            state.name = name
                        raw_arguments = _field(function, "arguments", None)
                        if raw_arguments is not None:
                            if not isinstance(raw_arguments, str):
                                raise ValueError("stream tool arguments must be text")
                            argument_delta = raw_arguments
                    state.argument_fragments.append(argument_delta)
                    attempt.progress(argument_delta or "")
                    yield ModelToolCallDelta(
                        index=index,
                        arguments_delta=argument_delta,
                        id=state.canonical_id if is_first else None,
                        name=name,
                        provider_call_id=native_id,
                    )
                native_finish = _field(choice, "finish_reason", None)
                if native_finish is not None:
                    decoded_finish = _required_text(
                        native_finish,
                        "stream finish reason",
                    )
                    if finish_reason is not None and finish_reason != decoded_finish:
                        raise ValueError("stream finish reason changed")
                    finish_reason = decoded_finish
            except asyncio.CancelledError:
                raise
            except ModelProviderError:
                raise
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ModelProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    f"{self.provider} returned a malformed stream",
                    provider_id=self.provider_id,
                    diagnostic=ProviderFailureDiagnostic(
                        phase=ProviderFailurePhase.STREAM_EVENT,
                        code="event_decode_failed",
                        terminal_status=_safe_structural_token(finish_reason),
                    ),
                ) from error

        if finish_reason is None:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                f"{self.provider} stream ended without a finish reason",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.STREAM_TERMINAL,
                    code="terminal_completion_missing",
                ),
            )
        try:
            if finish_reason == "content_filter":
                raise ModelProviderError(
                    ProviderErrorCode.CONTENT_BLOCKED,
                    f"{self.provider} blocked the response",
                )
            canonical_finish = _finish_reason(finish_reason)
            if sorted(tool_states) != list(range(len(tool_states))):
                raise ValueError("stream tool indexes must be contiguous")
            calls: list[ToolCall] = []
            for index in sorted(tool_states):
                state = tool_states[index]
                if state.provider_call_id is None or state.name is None:
                    raise ValueError("stream tool call is missing identity")
                encoded_arguments = "".join(state.argument_fragments)
                arguments_value = (
                    {} if not encoded_arguments else json.loads(encoded_arguments)
                )
                if not isinstance(arguments_value, dict):
                    raise ValueError("stream tool arguments must decode to an object")
                calls.append(
                    ToolCall(
                        id=state.canonical_id,
                        provider_call_id=state.provider_call_id,
                        name=state.name,
                        arguments=arguments_value,
                    )
                )
            if calls and canonical_finish is not FinishReason.TOOL_CALLS:
                raise ValueError("stream tool calls require tool_calls finish")
            text = "".join(text_fragments).strip() or None
            if not calls and text is None and canonical_finish is FinishReason.LENGTH:
                raise ModelProviderError(
                    ProviderErrorCode.OUTPUT_LIMIT,
                    f"{self.provider} exhausted the output token limit",
                )
            response = ModelResponse(
                finish_reason=canonical_finish,
                text=text,
                tool_calls=tuple(calls),
                usage=self._decode_priced_usage(
                    usage_value,
                    response_model=response_model,
                    service_tier=service_tier,
                    requested_at=requested_at,
                ),
                provider_id=self.provider_id,
                provider_response_id=response_id,
                provider_metadata={
                    _CONTINUATION_KEY: {"provider_id": self.provider_id},
                    "pricing_dimensions": {
                        "response_model": response_model,
                        "service_tier": service_tier,
                    },
                },
            )
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                f"{self.provider} returned a malformed stream",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.STREAM_TERMINAL,
                    code="terminal_response_decode_failed",
                    terminal_status=_safe_structural_token(finish_reason),
                ),
            ) from error
        attempt.response(response)
        terminal_event = ModelStreamCompleted(response)
    if terminal_event is not None:
        yield terminal_event
