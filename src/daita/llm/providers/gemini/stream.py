"""Decode Gemini generate-content events without constructing native streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Any, cast

from ...._json import FrozenJsonObject, canonical_json
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
    optional_text as _optional_text,
    required_text as _required_text,
    safe_structural_token as _safe_structural_token,
)


def _argument_snapshot_grew(previous: object, current: object) -> bool:
    """Validate monotone cumulative argument snapshots without counting repeats."""
    if previous == current:
        return False
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        if not previous.keys() <= current.keys():
            raise ValueError("function argument snapshot lost fields")
        for key, value in previous.items():
            _argument_snapshot_grew(value, current[key])
        return True
    if (
        isinstance(previous, str)
        and isinstance(current, str)
        and current.startswith(previous)
    ):
        return True
    raise ValueError("function argument snapshot changed existing values")


async def decode_gemini_stream(
    self: Any,
    source: AsyncIterator[object],
    request: ModelRequest,
    attempt: AttemptLifecycle,
    *,
    arguments: dict[str, object],
    requested_at: datetime,
    counted_input_tokens: int | None,
) -> AsyncIterator[ModelStreamEvent]:
    from .adapter import _enum_value, _normalize_error, _sequence

    text_fragments: list[str] = []
    provider_parts: list[dict[str, object]] = []
    canonical_call_ids: list[str] = []
    calls_by_native_id: dict[str, dict[str, object]] = {}
    finish_reason: str | None = None
    usage_value: object | None = None
    response_id: str | None = None
    model_version: str | None = None
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
                attempt.transport_failure(error, phase="generation")
                raise _normalize_error(error) from error
            try:
                native_id = _field(chunk, "response_id", None)
            except Exception:
                native_id = None
            attempt.native("generate_content_chunk", native_id)
            try:
                chunk_model = _optional_text(
                    _field(chunk, "model_version", None),
                    "stream model version",
                )
                if chunk_model is not None:
                    if model_version is not None and model_version != chunk_model:
                        raise ValueError("stream model version changed")
                    model_version = chunk_model
                chunk_id = _optional_text(
                    _field(chunk, "response_id", None),
                    "stream response id",
                )
                if chunk_id is not None:
                    if response_id is not None and response_id != chunk_id:
                        raise ValueError("stream response ID changed")
                    response_id = chunk_id
                chunk_usage = _field(chunk, "usage_metadata", None)
                if chunk_usage is not None:
                    usage_value = chunk_usage
                feedback = _field(chunk, "prompt_feedback", None)
                block_reason = _enum_value(_field(feedback, "block_reason", None))
                if block_reason not in {None, "BLOCK_REASON_UNSPECIFIED"}:
                    raise ModelProviderError(
                        ProviderErrorCode.CONTENT_BLOCKED,
                        "Gemini blocked the response",
                    )
                candidates_value = _field(chunk, "candidates", ())
                if candidates_value is None:
                    candidates_value = ()
                candidates = _sequence(candidates_value, "stream candidates")
                if len(candidates) > 1:
                    raise ValueError("stream must contain at most one candidate")
                if not candidates:
                    # Native streams may contain metadata-only or heartbeat
                    # responses. They carry no canonical model output.
                    continue
                candidate = candidates[0]
                native_finish = _field(candidate, "finish_reason", None)
                if native_finish is not None:
                    decoded_finish = _required_text(
                        _enum_value(native_finish),
                        "stream finish reason",
                    )
                    if decoded_finish in {
                        "SAFETY",
                        "RECITATION",
                        "BLOCKLIST",
                        "PROHIBITED_CONTENT",
                        "SPII",
                    }:
                        raise ModelProviderError(
                            ProviderErrorCode.CONTENT_BLOCKED,
                            "Gemini blocked the response",
                        )
                    if finish_reason is not None and finish_reason != decoded_finish:
                        raise ValueError("stream finish reason changed")
                    finish_reason = decoded_finish
                content = _field(candidate, "content", None)
                if content is None:
                    continue
                parts_value = _field(content, "parts", ())
                if parts_value is None:
                    parts_value = ()
                for part in _sequence(parts_value, "stream candidate parts"):
                    provider_part: dict[str, object] = {}
                    signature = _field(part, "thought_signature", None)
                    if signature is not None:
                        provider_part["thought_signature"] = signature
                    thought = _field(part, "thought", False)
                    if thought is None:
                        thought = False
                    if thought is not False and thought is not True:
                        raise ValueError("stream thought flag must be boolean")
                    if thought:
                        provider_part["thought"] = True
                    part_text = _field(part, "text", None)
                    if part_text is not None:
                        if not isinstance(part_text, str):
                            raise ValueError("stream part text must be text")
                        attempt.progress(part_text)
                        provider_part["text"] = part_text
                        if not thought:
                            text_fragments.append(part_text)
                            if part_text:
                                yield ModelTextDelta(part_text)
                    function_call = _field(part, "function_call", None)
                    if function_call is not None:
                        # Vertex partial-argument messages are not supported
                        # by this Gemini API adapter. Never treat a partial
                        # JSON object as an executable complete function call.
                        if (
                            _field(function_call, "partial_args", None) is not None
                            or _field(function_call, "will_continue", None) is True
                        ):
                            raise ValueError("partial function protocol is unsupported")
                        if part_text is not None:
                            raise ValueError(
                                "stream part cannot contain text and a function call"
                            )
                        arguments_value = _field(function_call, "args")
                        if not isinstance(arguments_value, Mapping):
                            raise ValueError(
                                "stream function arguments must be an object"
                            )
                        name = _required_text(
                            _field(function_call, "name"),
                            "stream function name",
                        )
                        native_id = _optional_text(
                            _field(function_call, "id", None),
                            "stream function id",
                        )
                        native_call: dict[str, object] = {
                            "name": name,
                            "args": FrozenJsonObject.from_mapping(
                                arguments_value
                            ).to_dict(),
                        }
                        if native_id is not None:
                            native_call["id"] = native_id
                            previous_call = calls_by_native_id.get(native_id)
                            if previous_call is not None:
                                if previous_call["name"] != name:
                                    raise ValueError("stream function identity changed")
                                if _argument_snapshot_grew(
                                    previous_call["args"], native_call["args"]
                                ):
                                    previous_call["args"] = native_call["args"]
                                    attempt.progress(canonical_json(arguments_value))
                                continue
                            calls_by_native_id[native_id] = native_call
                        provider_part["function_call"] = native_call
                        canonical_id = self._id_factory("call")
                        if canonical_id in canonical_call_ids:
                            raise ValueError("id_factory returned a duplicate call ID")
                        canonical_call_ids.append(canonical_id)
                        attempt.progress(
                            canonical_json(arguments_value) if arguments_value else ""
                        )
                    if not provider_part:
                        raise ValueError("stream contains an empty part")
                    if (
                        set(provider_part).issubset({"text", "thought_signature"})
                        and "text" in provider_part
                        and not provider_part.get("thought", False)
                        and provider_parts
                        and set(provider_parts[-1]).issubset(
                            {"text", "thought_signature"}
                        )
                        and "text" in provider_parts[-1]
                    ):
                        previous = provider_parts[-1]
                        if (
                            "thought_signature" in previous
                            and "thought_signature" in provider_part
                        ):
                            raise ValueError(
                                "stream text contains multiple thought signatures"
                            )
                        previous["text"] = cast(str, previous["text"]) + cast(
                            str,
                            provider_part["text"],
                        )
                        if "thought_signature" in provider_part:
                            previous["thought_signature"] = provider_part[
                                "thought_signature"
                            ]
                    else:
                        provider_parts.append(provider_part)
            except asyncio.CancelledError:
                raise
            except ModelProviderError:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise ModelProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "Gemini returned a malformed stream",
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
                "Gemini stream ended without a finish reason",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.STREAM_TERMINAL,
                    code="terminal_completion_missing",
                ),
            )
        try:
            response = self._decode_response(
                {
                    "response_id": response_id,
                    "candidates": [
                        {
                            "finish_reason": finish_reason,
                            "content": {"parts": provider_parts},
                        }
                    ],
                    "prompt_feedback": None,
                    "usage_metadata": usage_value,
                    "model_version": model_version,
                },
                canonical_call_ids=canonical_call_ids,
                requested_at=requested_at,
            )
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "Gemini returned a malformed stream",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.STREAM_TERMINAL,
                    code="terminal_response_decode_failed",
                    terminal_status=_safe_structural_token(finish_reason),
                ),
            ) from error
        # Whole-object native snapshots are emitted once, after their final
        # response has validated; replayed snapshots cannot renew progress
        # or produce duplicate/partially executable canonical calls.
        attempt.response(response)
        for index, call in enumerate(response.tool_calls):
            yield ModelToolCallDelta(
                index=index,
                arguments_delta=canonical_json(call.arguments),
                id=call.id,
                name=call.name,
                provider_call_id=call.provider_call_id,
            )
        terminal_event = ModelStreamCompleted(
            with_request_admission(
                response,
                request,
                input_tokens=counted_input_tokens,
                output_cap=cast(
                    int | None,
                    cast(dict[str, object], arguments["config"]).get(
                        "max_output_tokens"
                    ),
                ),
            )
        )
    if terminal_event is not None:
        yield terminal_event
