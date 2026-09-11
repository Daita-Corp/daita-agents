"""Decode OpenAI Responses events without constructing native streams."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Any, cast

from ..._lifecycle import AttemptLifecycle
from ...errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from ...models import (
    ModelRequest,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
)
from ...pricing import with_request_admission
from .._fields import (
    field as _field,
    nonnegative_int as _nonnegative_int,
    optional_text as _optional_text,
    required_text as _required_text,
    safe_structural_token as _safe_structural_token,
)


async def decode_openai_stream(
    self: Any,
    source: AsyncIterator[object],
    request: ModelRequest,
    attempt: AttemptLifecycle,
    *,
    arguments: dict[str, object],
    requested_at: datetime,
    counted_input_tokens: int | None,
) -> AsyncIterator[ModelStreamEvent]:
    from .adapter import (
        _code_from_provider_value,
        _normalize_error,
        _openai_failure_diagnostic,
        _OpenAIResponseDecodeFailure,
        _optional_stream_identity,
        _ResponseOutputOverride,
        _safe_field,
        _stream_fragment,
    )

    canonical_ids_by_index: dict[int, str] = {}
    canonical_ids_by_provider_call_id: dict[str, str] = {}
    provider_call_ids_by_index: dict[int, str] = {}
    names_by_index: dict[int, str] = {}
    completed_items_by_index: dict[int, object] = {}
    allocated_ids: set[str] = set()
    completed = False
    current_event_type: str | None = None
    terminal_diagnostic: ProviderFailureDiagnostic | None = None
    sequence_number: int | None = None
    response_identity: str | None = None
    terminal_event: ModelStreamCompleted | None = None
    async with attempt.stream(source) as stream:
        try:
            async for event in cast(AsyncIterator[object], stream):
                attempt.native(
                    _safe_field(event, "type"),
                    _safe_field(_safe_field(event, "response"), "id"),
                )
                current_event_type = None
                event_type = _required_text(_field(event, "type"), "stream event type")
                current_event_type = _safe_structural_token(event_type)
                sequence = _field(event, "sequence_number", None)
                if sequence is not None:
                    sequence = _nonnegative_int(sequence, "stream sequence number")
                    if sequence_number is not None and sequence <= sequence_number:
                        raise ValueError("stream sequence did not advance")
                    sequence_number = sequence
                native_response = _field(event, "response", None)
                native_identity = _optional_stream_identity(
                    _field(native_response, "id", None), "stream response id"
                )
                if native_identity is not None:
                    if (
                        response_identity is not None
                        and native_identity != response_identity
                    ):
                        raise ValueError("stream response identity changed")
                    response_identity = native_identity
                if event_type == "response.output_text.delta":
                    delta = _stream_fragment(_field(event, "delta"), "text delta")
                    if delta:
                        attempt.progress(delta)
                        yield ModelTextDelta(delta)
                elif event_type == "response.output_item.added":
                    item = _field(event, "item")
                    if _field(item, "type", None) != "function_call":
                        continue
                    index = _nonnegative_int(
                        _field(event, "output_index"), "output index"
                    )
                    provider_call_id = _optional_stream_identity(
                        _field(item, "call_id", None), "provider call_id"
                    )
                    name = _optional_stream_identity(
                        _field(item, "name", None), "function name"
                    )
                    canonical_id = canonical_ids_by_index.get(index)
                    if canonical_id is None:
                        canonical_id = self._id_factory("call")
                        if canonical_id in allocated_ids:
                            raise ValueError("id_factory returned a duplicate call ID")
                        allocated_ids.add(canonical_id)
                        canonical_ids_by_index[index] = canonical_id
                    if provider_call_id is not None:
                        canonical_ids_by_provider_call_id[provider_call_id] = (
                            canonical_id
                        )
                        provider_call_ids_by_index[index] = provider_call_id
                    if name is not None:
                        names_by_index[index] = name
                    yield ModelToolCallDelta(
                        index=index,
                        arguments_delta="",
                        id=canonical_id,
                        name=name,
                        provider_call_id=provider_call_id,
                    )
                elif event_type == "response.function_call_arguments.delta":
                    index = _nonnegative_int(
                        _field(event, "output_index"), "output index"
                    )
                    arguments_delta = _stream_fragment(
                        _field(event, "delta"), "function arguments delta"
                    )
                    if not arguments_delta:
                        continue
                    canonical_id = canonical_ids_by_index.get(index)
                    if canonical_id is None:
                        canonical_id = self._id_factory("call")
                        if canonical_id in allocated_ids:
                            raise ValueError("id_factory returned a duplicate call ID")
                        allocated_ids.add(canonical_id)
                        canonical_ids_by_index[index] = canonical_id
                    attempt.progress(arguments_delta)
                    yield ModelToolCallDelta(
                        index=index,
                        arguments_delta=arguments_delta,
                        id=canonical_id,
                        name=names_by_index.get(index),
                        provider_call_id=provider_call_ids_by_index.get(index),
                    )
                elif event_type == "response.output_item.done":
                    index = _nonnegative_int(
                        _field(event, "output_index"), "output index"
                    )
                    completed_items_by_index[index] = _field(event, "item")
                elif event_type in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    native_response = _field(event, "response")
                    terminal_diagnostic = _openai_failure_diagnostic(
                        native_response,
                        phase=ProviderFailurePhase.STREAM_TERMINAL,
                        code="terminal_response_decode_failed",
                        event_type=event_type,
                    )
                    native_output = _field(native_response, "output", ())
                    if (
                        isinstance(native_output, Sequence)
                        and not isinstance(native_output, (str, bytes))
                        and completed_items_by_index
                    ):
                        output_count = max(
                            len(native_output),
                            max(completed_items_by_index) + 1,
                        )
                        completed_output: list[object] = []
                        for output_index in range(output_count):
                            completed_item = completed_items_by_index.get(output_index)
                            if completed_item is not None:
                                completed_output.append(completed_item)
                            elif output_index < len(native_output):
                                completed_output.append(native_output[output_index])
                            else:
                                terminal_diagnostic = _openai_failure_diagnostic(
                                    native_response,
                                    phase=ProviderFailurePhase.STREAM_TERMINAL,
                                    code="terminal_output_reconstruction_invalid",
                                    event_type=event_type,
                                )
                                raise ValueError(
                                    "completed stream output contains an index gap"
                                )
                        native_response = _ResponseOutputOverride(
                            native_response,
                            tuple(completed_output),
                        )
                    try:
                        response = self._decode_response(
                            native_response,
                            requested_at=requested_at,
                            canonical_ids_by_index=canonical_ids_by_index,
                            canonical_ids_by_provider_call_id=(
                                canonical_ids_by_provider_call_id
                            ),
                        )
                    except _OpenAIResponseDecodeFailure as error:
                        terminal_diagnostic = _openai_failure_diagnostic(
                            native_response,
                            phase=ProviderFailurePhase.STREAM_TERMINAL,
                            code=error.diagnostic_code,
                            event_type=event_type,
                        )
                        raise
                    terminal_event = ModelStreamCompleted(
                        with_request_admission(
                            response,
                            request,
                            input_tokens=counted_input_tokens,
                            output_cap=cast(
                                int | None, arguments.get("max_output_tokens")
                            ),
                        )
                    )
                    attempt.response(terminal_event.response)
                    completed = True
                    break
                elif event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }:
                    attempt.progress(
                        _stream_fragment(_field(event, "delta"), "reasoning delta")
                    )
                elif event_type == "error":
                    code = _optional_text(
                        _field(event, "code", None), "stream error code"
                    )
                    raise ModelProviderError(
                        _code_from_provider_value(code),
                        "OpenAI stream failed",
                    )
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OpenAI returned a malformed stream",
                provider_id=self.provider_id,
                diagnostic=(
                    terminal_diagnostic
                    or ProviderFailureDiagnostic(
                        phase=ProviderFailurePhase.STREAM_EVENT,
                        code="event_decode_failed",
                        event_type=current_event_type,
                    )
                ),
            ) from error
        except Exception as error:
            self._observe_headers(
                attempt, "generation", _safe_field(error, "response"), arrived=False
            )
            attempt.transport_failure(error, phase="generation")
            raise _normalize_error(error) from error
    if terminal_event is not None:
        yield terminal_event
        return
    if not completed:
        raise ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            "OpenAI stream ended without a terminal response",
            provider_id=self.provider_id,
            diagnostic=ProviderFailureDiagnostic(
                phase=ProviderFailurePhase.STREAM_TERMINAL,
                code="terminal_completion_missing",
            ),
        )
