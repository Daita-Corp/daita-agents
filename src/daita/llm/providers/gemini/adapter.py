"""Translate canonical requests and streaming responses for Google Gemini."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol, cast
from uuid import uuid4

from ...._installation import repair_guidance
from ...._json import FrozenJsonObject
from ..._lifecycle import (
    AttemptLifecycle,
    CloseCoordinator,
    NativeOwner,
    closing_stream,
    execute_generate_attempt,
    execute_stream_attempt,
    native_events,
)
from ...errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
    before_generation,
    retry_after_from_headers,
    token_count_error,
    with_cancelled_model_usage,
)
from ...models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ToolCall,
)
from ...pricing import (
    BillableQuantity,
    CostEstimate,
    PricingSchedule,
    bound_request_output,
    calculate_cost_estimate,
    has_complete_pricing_coverage,
    load_bundled_pricing_schedules,
    validate_pricing_schedules,
    with_request_admission,
)
from ...provider_definitions import supports_builtin_request_policy
from .._fields import (
    field as _field,
    optional_text as _optional_text,
    required_text as _required_text,
    usage_int as _usage_int,
)
from .messages import (
    _CONTINUATION_KEY,
    _count_request,
    _encode_signature,
    _gemini_contents,
)


class _GeminiModels(Protocol):
    async def count_tokens(self, **kwargs: object) -> object: ...

    async def generate_content(self, **kwargs: object) -> object: ...

    async def generate_content_stream(self, **kwargs: object) -> object: ...


class _GeminiAsyncClient(Protocol):
    @property
    def models(self) -> _GeminiModels: ...

    async def aclose(self) -> None: ...


class _GeminiClient(Protocol):
    @property
    def aio(self) -> _GeminiAsyncClient: ...

    def close(self) -> None: ...


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
        self._owns_client = client is None
        self._native_owner = NativeOwner()
        self._close = CloseCoordinator(self._native_owner)
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
        return supports_builtin_request_policy("gemini", request)

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
        if self._close.started:
            raise RuntimeError("Gemini provider is closed")
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

    async def close(self, *, deadline: float | None = None) -> None:
        """Join the once-only cleanup of this provider's owned SDK client."""

        await self._close.close(self._finish_close, deadline=deadline)

    async def _finish_close(self) -> None:
        client = self._client
        if self._owns_client and client is not None:
            first_error: BaseException | None = None
            try:
                await client.aio.aclose()
            except BaseException as error:
                first_error = error
            try:
                client.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            if first_error is not None:
                raise first_error
        self._client = None

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await execute_generate_attempt(
            self._native_owner,
            request,
            provider_id=self.provider_id,
            boundary_name="Gemini",
            operation=self._generate,
            headers_supported=False,
        )

    async def _generate(
        self, request: ModelRequest, attempt: AttemptLifecycle
    ) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._require_supported_request_policy(request)
        arguments = self._request_arguments(request)
        cast(dict, arguments["config"])["http_options"]["timeout"] = max(
            1,
            int(
                min(
                    request.call_policy.read_timeout_seconds,
                    cast(float, request.attempt_deadline)
                    - asyncio.get_running_loop().time(),
                )
                * 1000
            ),
        )
        requested_at = self._clock()
        counted_input_tokens = await self._admit_request(
            request, arguments, attempt, requested_at=requested_at
        )
        try:
            attempt.values["output_cap"] = cast(
                dict[str, object], arguments["config"]
            ).get("max_output_tokens")
            attempt.dispatch()
            response = await attempt.run_native(
                self.client.aio.models.generate_content(**arguments)
            )
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            attempt.transport_failure(error, phase="generation")
            raise _normalize_error(error) from error
        try:
            return with_request_admission(
                self._decode_response(response, requested_at=requested_at),
                request,
                input_tokens=counted_input_tokens,
                output_cap=cast(
                    int | None,
                    cast(dict[str, object], arguments["config"]).get(
                        "max_output_tokens"
                    ),
                ),
            )
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

    def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        return execute_stream_attempt(
            self._native_owner,
            request,
            provider_id=self.provider_id,
            boundary_name="Gemini",
            operation=self._stream,
            headers_supported=False,
        )

    async def _stream(
        self,
        request: ModelRequest,
        attempt: AttemptLifecycle,
    ) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._require_supported_request_policy(request)
        arguments = self._request_arguments(request)
        cast(dict, arguments["config"])["http_options"]["timeout"] = max(
            1,
            int(
                min(
                    request.call_policy.read_timeout_seconds,
                    cast(float, request.attempt_deadline)
                    - asyncio.get_running_loop().time(),
                )
                * 1000
            ),
        )
        requested_at = self._clock()
        counted_input_tokens = await self._admit_request(
            request, arguments, attempt, requested_at=requested_at
        )
        try:
            attempt.values["output_cap"] = cast(
                dict[str, object], arguments["config"]
            ).get("max_output_tokens")
            attempt.dispatch()
            source = native_events(
                lambda: self.client.aio.models.generate_content_stream(**arguments)
            )
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            attempt.transport_failure(error, phase="generation")
            raise _normalize_error(error) from error
        from .stream import decode_gemini_stream

        async with closing_stream(
            decode_gemini_stream(
                self,
                source,
                request,
                attempt,
                arguments=arguments,
                requested_at=requested_at,
                counted_input_tokens=counted_input_tokens,
            )
        ) as events:
            async for event in events:
                yield event

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
                "http_options": {"retry_options": {"attempts": 1}},
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
                                "parameters_json_schema": FrozenJsonObject.from_mapping(
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
            arguments: dict[str, object] = {
                "model": self.model,
                "contents": contents,
                "config": config,
            }
            return arguments
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "canonical request cannot be translated for Gemini",
            ) from error

    async def _admit_request(
        self,
        request: ModelRequest,
        arguments: dict[str, object],
        attempt: AttemptLifecycle,
        *,
        requested_at: datetime,
    ) -> int | None:
        request.remaining_after(
            ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
        )
        if request.max_total_tokens is None and request.max_estimated_cost_usd is None:
            return None

        def output_limit(input_tokens: int, *, counted: bool = True) -> int:
            return bound_request_output(
                request,
                input_tokens=input_tokens,
                input_tokens_counted=counted,
                maximum_output_tokens=self._max_output_tokens,
                schedules=self._pricing_schedules,
                provider="gemini",
                model=self.model,
                endpoint="generate_content",
                requested_at=requested_at,
                qualifiers={"service_tier": "standard"},
            )

        output_limit(0, counted=False)
        attempt.start_count()
        try:
            # google-genai's Developer API count config rejects tools/system
            # fields. Its public extra_body option supports the documented full
            # generateContentRequest instead. Use the already prepared content.
            counted = await attempt.run_native(
                self.client.aio.models.count_tokens(
                    model=self.model,
                    contents=None,
                    config={
                        "http_options": {
                            "retry_options": {"attempts": 1},
                            "timeout": max(
                                1,
                                int(
                                    (
                                        attempt.execution_deadline
                                        - asyncio.get_running_loop().time()
                                    )
                                    * 1000
                                ),
                            ),
                            "extra_body": {
                                "generateContentRequest": _count_request(arguments),
                            },
                        },
                    },
                )
            )
        except asyncio.CancelledError as error:
            raise with_cancelled_model_usage(
                error, ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
            ) from None
        except ImportError:
            raise
        except (TypeError, ValueError):
            raise token_count_error(invalid=True) from None
        except Exception as error:
            attempt.transport_failure(error, phase="count")
            raise before_generation(
                _normalize_error(error),
                code="input_token_count_failed",
            ) from None
        tokens = _field(counted, "total_tokens", None)
        if type(tokens) is not int or tokens < 0:
            raise token_count_error(invalid=True)
        attempt.counted(tokens)
        cast(dict[str, object], arguments["config"])["max_output_tokens"] = (
            output_limit(tokens)
        )
        attempt.values["output_cap"] = cast(dict[str, object], arguments["config"])[
            "max_output_tokens"
        ]
        return tokens

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
    if isinstance(error, ModelProviderError):
        return error
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return ModelProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            "Gemini returned an undecodable native response",
            diagnostic=ProviderFailureDiagnostic(
                phase=ProviderFailurePhase.STREAM_EVENT,
                code="native_response_decode_failed",
            ),
        )
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
        retry_after_seconds=(
            retry_after_from_headers(
                _lenient_field(_lenient_field(error, "response"), "headers")
            )
            if normalized
            in {
                ProviderErrorCode.RATE_LIMIT_ERROR,
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
            }
            else None
        ),
    )


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


__all__ = ["GeminiProvider"]
