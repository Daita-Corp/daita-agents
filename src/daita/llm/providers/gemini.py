"""Translate canonical requests and streaming responses for Google Gemini."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol, cast
from uuid import uuid4

from ..._installation import repair_guidance
from ..._json import FrozenJsonObject, canonical_json
from ..errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
    detached_provider_error,
)
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
from ..pricing import (
    BillableQuantity,
    CostEstimate,
    PricingSchedule,
    calculate_cost_estimate,
    has_complete_pricing_coverage,
    load_bundled_pricing_schedules,
    validate_pricing_schedules,
)

_CONTINUATION_KEY = "gemini_continuation"
_PORTABLE_FUNCTION_CALL_SIGNATURE = b"skip_thought_signature_validator"


class _GeminiModels(Protocol):
    async def generate_content(self, **kwargs: object) -> object: ...

    async def generate_content_stream(self, **kwargs: object) -> object: ...


class _GeminiAsyncClient(Protocol):
    @property
    def models(self) -> _GeminiModels: ...


class _GeminiClient(Protocol):
    @property
    def aio(self) -> _GeminiAsyncClient: ...


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class GeminiProvider:
    """Translate canonical requests to native Gemini generate-content calls."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_output_tokens: int = 1_024,
        client: _GeminiClient | None = None,
        id_factory: Callable[[str], str] | None = None,
        pricing_schedules: Iterable[PricingSchedule] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if id_factory is not None and not callable(id_factory):
            raise TypeError("id_factory must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.model = model.strip()
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens
        self._client = client
        self._id_factory = _new_id if id_factory is None else id_factory
        self._pricing_schedules = (
            load_bundled_pricing_schedules()
            if pricing_schedules is None
            else validate_pricing_schedules(pricing_schedules)
        )
        self._clock = clock

    @property
    def provider_id(self) -> str:
        return f"gemini:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return request.allow_parallel_tool_calls is None

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return has_complete_pricing_coverage(
            self._pricing_schedules,
            provider="gemini",
            model=self.model,
            endpoint="generate_content",
            requested_at=self._clock(),
            qualifiers={"service_tier": "standard"},
            required_metrics=(
                "input_uncached_tokens",
                "input_cache_read_tokens",
                "output_tokens",
            ),
            usage_range_metric="request_input_tokens",
        )

    @property
    def client(self) -> _GeminiClient:
        if self._client is None:
            try:
                from google import genai
            except ImportError as error:
                raise ImportError(
                    "Daita's Gemini runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from error
            self._client = cast(_GeminiClient, genai.Client(api_key=self._api_key))
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
                "Gemini provider boundary failed",
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.PROVIDER_BOUNDARY,
                    code="unexpected_provider_boundary_failure",
                ),
            )
        if failure is None:
            raise AssertionError("Gemini provider failed without an error")
        raise detached_provider_error(failure, provider_id=self.provider_id)

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._require_supported_request_policy(request)
        arguments = self._request_arguments(request)
        requested_at = self._clock()
        try:
            response = await self.client.aio.models.generate_content(**arguments)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error) from error
        try:
            return self._decode_response(response, requested_at=requested_at)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "Gemini returned a malformed response",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.RESPONSE_DECODE,
                    code="response_decode_failed",
                ),
            ) from error

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
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
                "Gemini provider boundary failed",
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.PROVIDER_BOUNDARY,
                    code="unexpected_provider_boundary_failure",
                ),
            )
        if failure is None:
            raise AssertionError("Gemini provider failed without an error")
        raise detached_provider_error(failure, provider_id=self.provider_id)

    async def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._require_supported_request_policy(request)
        arguments = self._request_arguments(request)
        requested_at = self._clock()
        try:
            raw_stream = await self.client.aio.models.generate_content_stream(
                **arguments
            )
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error) from error
        iterator_method = getattr(raw_stream, "__aiter__", None)
        if not callable(iterator_method):
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "Gemini returned a malformed stream",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.STREAM_TERMINAL,
                    code="stream_not_iterable",
                ),
            )

        text_fragments: list[str] = []
        provider_parts: list[dict[str, object]] = []
        canonical_call_ids: list[str] = []
        finish_reason: str | None = None
        usage_value: object | None = None
        response_id: str | None = None
        model_version: str | None = None
        iterator = cast(AsyncIterator[object], iterator_method())
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
                raise _normalize_error(error) from error
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
                        provider_part["text"] = part_text
                        if not thought:
                            text_fragments.append(part_text)
                            if part_text:
                                yield ModelTextDelta(part_text)
                    function_call = _field(part, "function_call", None)
                    if function_call is not None:
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
                        provider_part["function_call"] = native_call
                        canonical_id = self._id_factory("call")
                        if canonical_id in canonical_call_ids:
                            raise ValueError("id_factory returned a duplicate call ID")
                        canonical_call_ids.append(canonical_id)
                        yield ModelToolCallDelta(
                            index=len(canonical_call_ids) - 1,
                            arguments_delta=canonical_json(arguments_value),
                            id=canonical_id,
                            name=name,
                            provider_call_id=native_id,
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
        yield ModelStreamCompleted(response)

    def _require_supported_request_policy(self, request: ModelRequest) -> None:
        if not self.supports_request_policy(request):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Gemini cannot enforce the requested tool-call policy",
            )

    def _request_arguments(self, request: ModelRequest) -> dict[str, object]:
        try:
            contents, system_instruction = _gemini_contents(
                request.messages,
                self.provider_id,
            )
            config: dict[str, object] = {
                "max_output_tokens": self._max_output_tokens,
            }
            if system_instruction is not None:
                config["system_instruction"] = system_instruction
            if request.tools:
                config["tools"] = [
                    {
                        "function_declarations": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": FrozenJsonObject.from_mapping(
                                    tool.input_schema
                                ).to_dict(),
                            }
                            for tool in request.tools
                        ]
                    }
                ]
            if request.response_schema is not None:
                config["response_mime_type"] = "application/json"
                config["response_json_schema"] = FrozenJsonObject.from_mapping(
                    request.response_schema
                ).to_dict()
            return {
                "model": self.model,
                "contents": contents,
                "config": config,
            }
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "canonical request cannot be translated for Gemini",
            ) from error

    def _decode_response(
        self,
        response: object,
        *,
        canonical_call_ids: Sequence[str] | None = None,
        requested_at: datetime | None = None,
    ) -> ModelResponse:
        feedback = _field(response, "prompt_feedback", None)
        block_reason = _enum_value(_field(feedback, "block_reason", None))
        if block_reason not in {None, "BLOCK_REASON_UNSPECIFIED"}:
            raise ModelProviderError(
                ProviderErrorCode.CONTENT_BLOCKED,
                "Gemini blocked the response",
            )
        candidates = _sequence(_field(response, "candidates"), "candidates")
        if len(candidates) != 1:
            raise ValueError("response must contain exactly one candidate")
        candidate = candidates[0]
        native_finish = _required_text(
            _enum_value(_field(candidate, "finish_reason")),
            "finish reason",
        )
        if native_finish in {
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
        content = _field(candidate, "content")
        parts = _sequence(_field(content, "parts"), "candidate parts")
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        provider_parts: list[dict[str, object]] = []
        has_signature = False
        canonical_ids: set[str] = set()
        call_position = 0
        for part in parts:
            recognized = False
            provider_part: dict[str, object] = {}
            signature = _field(part, "thought_signature", None)
            if signature is not None:
                provider_part["thought_signature"] = _encode_signature(signature)
                has_signature = True
                recognized = True
            thought = _field(part, "thought", False)
            if thought is None:
                thought = False
            if thought is not False and thought is not True:
                raise ValueError("part thought flag must be a boolean")
            if thought:
                provider_part["thought"] = True
            text = _field(part, "text", None)
            if text is not None:
                if not isinstance(text, str):
                    raise ValueError("part text must be text")
                provider_part["text"] = text
                if not thought and text.strip():
                    decoded_text = _required_text(text, "part text")
                    text_parts.append(decoded_text)
                recognized = True
            function_call = _field(part, "function_call", None)
            if function_call is not None:
                if text is not None:
                    raise ValueError("part cannot contain text and a function call")
                arguments = _field(function_call, "args")
                if not isinstance(arguments, Mapping):
                    raise ValueError("function-call arguments must be an object")
                if canonical_call_ids is None:
                    canonical_id = self._id_factory("call")
                else:
                    if call_position >= len(canonical_call_ids):
                        raise ValueError(
                            "stream canonical call IDs do not match response"
                        )
                    canonical_id = canonical_call_ids[call_position]
                call_position += 1
                if canonical_id in canonical_ids:
                    raise ValueError("id_factory returned a duplicate call ID")
                canonical_ids.add(canonical_id)
                native_id = _optional_text(
                    _field(function_call, "id", None),
                    "function-call id",
                )
                native_call: dict[str, object] = {
                    "name": _required_text(
                        _field(function_call, "name"),
                        "function-call name",
                    ),
                    "args": FrozenJsonObject.from_mapping(arguments).to_dict(),
                }
                if native_id is not None:
                    native_call["id"] = native_id
                provider_part["function_call"] = native_call
                calls.append(
                    ToolCall(
                        id=canonical_id,
                        provider_call_id=native_id,
                        name=cast(str, native_call["name"]),
                        arguments=dict(arguments),
                    )
                )
                recognized = True
            if not recognized:
                raise ValueError("candidate contains an unsupported empty part")
            provider_parts.append(provider_part)
        if canonical_call_ids is not None and call_position != len(canonical_call_ids):
            raise ValueError("stream canonical call IDs do not match response")
        text = "\n".join(text_parts).strip() or None
        mapped_finish = _finish_reason(native_finish)
        finish_reason = FinishReason.TOOL_CALLS if calls else mapped_finish
        if not calls and text is None and mapped_finish is FinishReason.LENGTH:
            raise ModelProviderError(
                ProviderErrorCode.OUTPUT_LIMIT,
                "Gemini exhausted the output token limit",
            )
        metadata: dict[str, object] = {}
        model_version = _optional_text(
            _field(response, "model_version", None),
            "model version",
        )
        usage_value = _field(response, "usage_metadata", None)
        service_tier = _gemini_service_tier(_field(usage_value, "service_tier", None))
        metadata["pricing_dimensions"] = {
            "requested_model": self.model,
            "response_model": model_version,
            "service_tier": service_tier,
        }
        if has_signature or calls:
            continuation: dict[str, object] = {
                "provider_id": self.provider_id,
            }
            if has_signature:
                continuation["content_parts"] = provider_parts
            metadata[_CONTINUATION_KEY] = continuation
        return ModelResponse(
            finish_reason=finish_reason,
            text=text,
            tool_calls=tuple(calls),
            usage=self._decode_priced_usage(
                usage_value,
                requested_at=requested_at or self._clock(),
                service_tier=service_tier,
            ),
            provider_id=self.provider_id,
            provider_response_id=_optional_text(
                _field(response, "response_id", None),
                "response id",
            ),
            provider_metadata=metadata,
        )

    def _decode_priced_usage(
        self,
        value: object,
        *,
        requested_at: datetime,
        service_tier: str,
    ) -> ModelUsage:
        usage = _decode_usage(value)
        if value is None:
            return usage
        if not _has_complete_gemini_billing_dimensions(value):
            return replace(
                usage,
                cost_estimate=CostEstimate.unavailable("billing_dimensions_incomplete"),
            )
        return replace(
            usage,
            cost_estimate=calculate_cost_estimate(
                self._pricing_schedules,
                provider="gemini",
                model=self.model,
                endpoint="generate_content",
                requested_at=requested_at,
                qualifiers={"service_tier": service_tier},
                usage_values={
                    "request_input_tokens": Decimal(usage.input_tokens),
                },
                quantities=_gemini_billable_quantities(usage),
            ),
        )


def _gemini_contents(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> tuple[list[dict[str, object]], str | None]:
    contents: list[dict[str, object]] = []
    system_parts: list[str] = []
    call_ids: dict[str, tuple[str | None, str]] = {}
    for message in messages:
        text = "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
        if message.role is MessageRole.SYSTEM:
            if not text:
                raise ValueError("system message produced no text")
            system_parts.append(text)
            continue
        if message.role is MessageRole.ASSISTANT:
            continuation = _same_origin_continuation(
                message,
                provider_id,
            )
            replay_value = (
                None if continuation is None else continuation.get("content_parts")
            )
            if replay_value is not None:
                parts = _replay_content_parts(replay_value, message)
            else:
                parts = []
                if text:
                    parts.append({"text": text})
            for call_index, call in enumerate(message.tool_calls):
                native_id = (
                    call.provider_call_id
                    if continuation is not None and call.provider_call_id is not None
                    else (None if continuation is not None else call.id)
                )
                call_ids[call.id] = (native_id, call.name)
                if replay_value is None:
                    native_call: dict[str, object] = {
                        "name": call.name,
                        "args": FrozenJsonObject.from_mapping(call.arguments).to_dict(),
                    }
                    if native_id is not None:
                        native_call["id"] = native_id
                    native_part: dict[str, object] = {"function_call": native_call}
                    if continuation is None and call_index == 0:
                        native_part["thought_signature"] = (
                            _PORTABLE_FUNCTION_CALL_SIGNATURE
                        )
                    parts.append(native_part)
            if not parts:
                raise ValueError("assistant message produced no Gemini parts")
            contents.append({"role": "model", "parts": parts})
            continue
        if message.role is MessageRole.TOOL:
            parts = []
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise ValueError("tool message contains a non-tool result")
                try:
                    native_id, name = call_ids[block.call_id]
                except KeyError as error:
                    raise ValueError(
                        "tool result has no preceding Gemini function call"
                    ) from error
                function_response: dict[str, object] = {
                    "name": name,
                    "response": {
                        "is_error": block.is_error,
                        "output": FrozenJsonObject.from_mapping(block.output).to_dict(),
                    },
                }
                if native_id is not None:
                    function_response["id"] = native_id
                parts.append(
                    {
                        "function_response": function_response,
                    }
                )
            contents.append({"role": "user", "parts": parts})
            continue
        if not text:
            raise ValueError("user message produced no text")
        contents.append({"role": "user", "parts": [{"text": text}]})
    if not contents:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no Gemini contents",
        )
    return contents, "\n".join(system_parts) or None


def _same_origin_continuation(
    message: CanonicalMessage,
    provider_id: str,
) -> Mapping[str, object] | None:
    if message.provider_id != provider_id:
        return None
    value = message.provider_metadata.get(_CONTINUATION_KEY)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Gemini continuation metadata must be an object")
    origin = _required_text(
        value.get("provider_id"),
        "Gemini continuation provider origin",
    )
    if origin != provider_id:
        raise ValueError("Gemini continuation origin does not match canonical provider")
    return value


def _replay_content_parts(
    value: object,
    message: CanonicalMessage,
) -> list[dict[str, object]]:
    raw_parts = _sequence(value, "Gemini continuation content parts")
    replay_parts: list[dict[str, object]] = []
    replay_text: list[str] = []
    replay_calls: list[tuple[str | None, str, FrozenJsonObject]] = []
    saw_signature = False
    allowed_keys = frozenset({"function_call", "text", "thought", "thought_signature"})
    for value_part in raw_parts:
        if not isinstance(value_part, Mapping):
            raise ValueError("Gemini continuation parts must be objects")
        if set(value_part) - allowed_keys:
            raise ValueError("Gemini continuation part has unsupported fields")
        replay_part: dict[str, object] = {}
        signature = value_part.get("thought_signature")
        if signature is not None:
            replay_part["thought_signature"] = _decode_signature(signature)
            saw_signature = True
        thought = value_part.get("thought", False)
        if thought is not False and thought is not True:
            raise ValueError("Gemini continuation thought flag must be boolean")
        if thought:
            replay_part["thought"] = True
        part_text = value_part.get("text")
        if part_text is not None:
            if not isinstance(part_text, str):
                raise ValueError("Gemini continuation text must be text")
            replay_part["text"] = part_text
            if not thought and part_text.strip():
                replay_text.append(part_text)
        function_call = value_part.get("function_call")
        if function_call is not None:
            if part_text is not None or not isinstance(function_call, Mapping):
                raise ValueError("Gemini continuation function call is malformed")
            name = _required_text(
                function_call.get("name"),
                "Gemini continuation function name",
            )
            arguments = function_call.get("args")
            if not isinstance(arguments, Mapping):
                raise ValueError("Gemini continuation arguments must be an object")
            native_id = _optional_text(
                function_call.get("id"),
                "Gemini continuation function id",
            )
            native_call: dict[str, object] = {
                "name": name,
                "args": FrozenJsonObject.from_mapping(arguments).to_dict(),
            }
            if native_id is not None:
                native_call["id"] = native_id
            replay_part["function_call"] = native_call
            replay_calls.append(
                (native_id, name, FrozenJsonObject.from_mapping(arguments))
            )
        if not replay_part:
            raise ValueError("Gemini continuation contains an empty part")
        replay_parts.append(replay_part)
    if not replay_parts or not saw_signature:
        raise ValueError("Gemini continuation requires signed content parts")

    canonical_text = "\n".join(
        block.text for block in message.content if isinstance(block, TextBlock)
    ).strip()
    if "\n".join(replay_text).strip() != canonical_text:
        raise ValueError("Gemini continuation text does not match canonical content")
    if len(replay_calls) != len(message.tool_calls):
        raise ValueError("Gemini continuation calls do not match canonical calls")
    for replay_call, canonical_call in zip(
        replay_calls,
        message.tool_calls,
        strict=True,
    ):
        native_id, name, arguments = replay_call
        if (
            native_id != canonical_call.provider_call_id
            or name != canonical_call.name
            or arguments != FrozenJsonObject.from_mapping(canonical_call.arguments)
        ):
            raise ValueError(
                "Gemini continuation call does not match canonical content"
            )
    return replay_parts


def _encode_signature(value: object) -> str:
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    else:
        raise ValueError("thought signature must be bytes or text")
    if not raw:
        raise ValueError("thought signature must not be empty")
    return base64.b64encode(raw).decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("encoded thought signature must be text")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError("encoded thought signature is malformed") from error


def _decode_usage(value: object) -> ModelUsage:
    if value is None:
        return ModelUsage()
    prompt_tokens = _usage_int(
        _field(value, "prompt_token_count", 0),
        "prompt tokens",
    )
    candidate_tokens = _usage_int(
        _field(value, "candidates_token_count", 0),
        "candidate tokens",
    )
    thought_tokens = _usage_int(
        _field(value, "thoughts_token_count", 0),
        "thought tokens",
    )
    cached_tokens = _usage_int(
        _field(value, "cached_content_token_count", 0),
        "cached tokens",
    )
    tool_use_tokens = _usage_int(
        _field(value, "tool_use_prompt_token_count", 0),
        "tool-use prompt tokens",
    )
    if cached_tokens > prompt_tokens:
        raise ValueError("Gemini cached tokens exceed total prompt tokens")
    total_tokens = prompt_tokens + tool_use_tokens + candidate_tokens + thought_tokens
    missing = object()
    reported_total = _field(value, "total_token_count", missing)
    if reported_total is not missing and reported_total is not None:
        if _usage_int(reported_total, "total tokens") != total_tokens:
            raise ValueError("Gemini token counters do not match total tokens")
    return ModelUsage(
        input_tokens=prompt_tokens + tool_use_tokens,
        output_tokens=candidate_tokens + thought_tokens,
        reasoning_tokens=thought_tokens,
        cache_read_tokens=cached_tokens,
    )


def _has_complete_gemini_billing_dimensions(value: object) -> bool:
    missing = object()
    return all(
        _field(value, name, missing) is not missing
        for name in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
        )
    )


def _gemini_service_tier(value: object) -> str:
    native = _enum_value(value)
    if native is None:
        return "standard"
    if not isinstance(native, str):
        raise ValueError("Gemini service tier must be text")
    normalized = native.strip().casefold()
    if normalized.startswith("service_tier_"):
        normalized = normalized.removeprefix("service_tier_")
    if normalized in {"", "unspecified"}:
        return "standard"
    if normalized not in {"standard", "flex", "priority"}:
        raise ValueError("Gemini service tier is unknown")
    return normalized


def _gemini_billable_quantities(
    usage: ModelUsage,
) -> tuple[BillableQuantity, ...]:
    uncached = usage.input_tokens - usage.cache_read_tokens
    if uncached < 0 or usage.reasoning_tokens > usage.output_tokens:
        raise ValueError("Gemini usage counters are internally inconsistent")
    return (
        BillableQuantity(
            "input_uncached_tokens",
            Decimal(uncached),
            "token",
        ),
        BillableQuantity(
            "input_cache_read_tokens",
            Decimal(usage.cache_read_tokens),
            "token",
        ),
        BillableQuantity(
            "output_tokens",
            Decimal(usage.output_tokens),
            "token",
        ),
    )


def _finish_reason(value: str) -> FinishReason:
    try:
        return {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
        }[value]
    except KeyError as error:
        raise ValueError("unknown finish reason") from error


def _normalize_error(error: Exception) -> ModelProviderError:
    code_value = _lenient_field(error, "code")
    status = (
        code_value
        if isinstance(code_value, int) and not isinstance(code_value, bool)
        else None
    )
    status_value = _lenient_field(error, "status")
    status_name = status_value if isinstance(status_value, str) else ""
    name = type(error).__name__.lower()
    if (
        isinstance(error, (asyncio.TimeoutError, TimeoutError))
        or status == 408
        or "timeout" in name
    ):
        normalized = ProviderErrorCode.TIMEOUT
    elif status in {401, 403} or "permission" in status_name.casefold():
        normalized = ProviderErrorCode.AUTHENTICATION_ERROR
    elif status == 429 or status_name == "RESOURCE_EXHAUSTED":
        normalized = ProviderErrorCode.RATE_LIMIT_ERROR
    elif status == 404 or status_name == "NOT_FOUND":
        normalized = ProviderErrorCode.MODEL_NOT_FOUND
    elif "context" in status_name.casefold():
        normalized = ProviderErrorCode.CONTEXT_OVERFLOW
    elif status is not None and status >= 500:
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    elif status is not None and 400 <= status < 500:
        normalized = ProviderErrorCode.INVALID_REQUEST
    elif isinstance(error, ConnectionError):
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    else:
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    return ModelProviderError(
        normalized,
        f"Gemini request failed: {normalized.value}",
    )


def _safe_structural_token(value: object) -> str | None:
    if not isinstance(value, str) or not 1 <= len(value) <= 96:
        return None
    if (
        not value[0].isascii()
        or not value[0].isalnum()
        or any(
            not character.isascii() or not (character.isalnum() or character in "._:-")
            for character in value
        )
    ):
        return None
    return value


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


def _lenient_field(value: object, name: str) -> object | None:
    try:
        return _field(value, name, None)
    except Exception:
        return None


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _usage_int(value: object, label: str) -> int:
    if value is None:
        return 0
    return _nonnegative_int(value, label)


__all__ = ["GeminiProvider"]
