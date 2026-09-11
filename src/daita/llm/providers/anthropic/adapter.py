"""Translate canonical requests and streaming responses for Anthropic Messages."""

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
    transport_timeout,
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
    ModelStreamCompleted,
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
    load_bundled_pricing_schedules,
    validate_pricing_schedules,
    with_request_admission,
)
from ...provider_definitions import supports_builtin_request_policy
from .messages import (
    _CONTINUATION_KEY,
    _OPAQUE_BLOCK_TYPES,
    _message_input,
    _plain_opaque_block,
)
from .usage import (
    _AnthropicBillingUsage,
    _decode_anthropic_billing_usage,
)
from .._fields import (
    field as _field,
    optional_text as _optional_text,
    required_text as _required_text,
    safe_structural_token as _safe_structural_token,
)

_NONCONTENT_STREAM_BLOCK_TYPES = frozenset({"fallback"})
_STREAM_MISSING = object()


class _MessagesResource(Protocol):
    async def count_tokens(self, **kwargs: object) -> object: ...

    async def create(self, **kwargs: object) -> object: ...

    def stream(self, **kwargs: object) -> _MessageStreamManager: ...


class _MessageStreamManager(Protocol):
    async def __aenter__(self) -> AsyncIterator[object]: ...

    async def __aexit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> bool | None: ...


class _AnthropicClient(Protocol):
    @property
    def messages(self) -> _MessagesResource: ...

    async def close(self) -> None: ...


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


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
        pricing_schedules: Iterable[PricingSchedule] | None = None,
        clock: Callable[[], datetime] = _utc_now,
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
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.model = model.strip()
        self.max_tokens = max_tokens
        self._api_key = api_key
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
        return f"anthropic:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return supports_builtin_request_policy("anthropic", request)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        # The current request surface does not pin the actual service tier or
        # workspace inference geography. Price the returned dimensions, but do
        # not claim a complete preflight estimate before Anthropic responds.
        return False

    @property
    def client(self) -> _AnthropicClient:
        if self._close.started:
            raise RuntimeError("Anthropic provider is closed")
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as error:
                raise ImportError(
                    "Daita's Anthropic runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from error
            self._client = cast(
                _AnthropicClient,
                AsyncAnthropic(api_key=self._api_key, max_retries=0),
            )
        if not self._owns_client:
            # Use a request view; do not change or close the caller's SDK client.
            with_options = getattr(self._client, "with_options", None)
            if callable(with_options):
                return cast(_AnthropicClient, with_options(max_retries=0))
        return self._client

    async def close(self, *, deadline: float | None = None) -> None:
        """Join the once-only cleanup of this provider's owned SDK client."""

        await self._close.close(self._finish_close, deadline=deadline)

    async def _finish_close(self) -> None:
        client = self._client
        if self._owns_client and client is not None:
            await client.close()
        self._client = None

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await execute_generate_attempt(
            self._native_owner,
            request,
            provider_id=self.provider_id,
            boundary_name="Anthropic",
            operation=self._generate,
            headers_supported=True,
        )

    async def _generate(
        self, request: ModelRequest, attempt: AttemptLifecycle
    ) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        self._require_supported_request_policy(request)
        arguments = self._request_arguments(request)
        requested_at = self._clock()
        counted_input_tokens = await self._admit_request(request, arguments, attempt)
        try:
            attempt.values["output_cap"] = arguments.get("max_tokens")
            attempt.dispatch()
            response = await self._sdk_call(
                self.client.messages, "create", arguments, attempt, "generation"
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
                output_cap=cast(int | None, arguments.get("max_tokens")),
            )
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "Anthropic returned a malformed response",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.RESPONSE_DECODE,
                    code="response_decode_failed",
                    terminal_status=_safe_structural_token(
                        _safe_field(response, "stop_reason")
                    ),
                ),
            ) from error

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        return execute_stream_attempt(
            self._native_owner,
            request,
            provider_id=self.provider_id,
            boundary_name="Anthropic",
            operation=self._stream,
            headers_supported=True,
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
        requested_at = self._clock()
        counted_input_tokens = await self._admit_request(request, arguments, attempt)
        from .stream import _AnthropicStreamDecoder

        decoder = _AnthropicStreamDecoder(
            provider_id=self.provider_id,
            id_factory=self._id_factory,
            progress=attempt.progress,
        )
        try:
            attempt.values["output_cap"] = arguments.get("max_tokens")
            attempt.dispatch()
            source = native_events(
                lambda: self.client.messages.stream(
                    **arguments, timeout=transport_timeout(request)
                ),
                manager=True,
                observe=lambda stream: self._observe_headers(
                    attempt, "generation", _safe_field(stream, "response")
                ),
            )
            async with attempt.stream(source) as native_stream:
                async for native_event in native_stream:
                    attempt.native(
                        _safe_field(native_event, "type"),
                        _safe_field(_safe_field(native_event, "message"), "id"),
                    )
                    event_type = _safe_structural_token(
                        _safe_field(native_event, "type")
                    )
                    try:
                        canonical_events = decoder.consume(native_event)
                    except ModelProviderError:
                        raise
                    except (KeyError, TypeError, ValueError) as error:
                        raise _malformed_stream(
                            phase=ProviderFailurePhase.STREAM_EVENT,
                            code="event_decode_failed",
                            provider_id=self.provider_id,
                            event_type=event_type,
                        ) from error
                    for canonical_event in canonical_events:
                        yield canonical_event
                try:
                    response = decoder.finish()
                    response = self._with_priced_usage(
                        response,
                        billing=decoder.billing_usage(),
                        response_model=decoder.response_model,
                        requested_at=requested_at,
                    )
                except ModelProviderError:
                    raise
                except (KeyError, TypeError, ValueError) as error:
                    raise _malformed_stream(
                        phase=ProviderFailurePhase.STREAM_TERMINAL,
                        code="terminal_completion_missing",
                        provider_id=self.provider_id,
                    ) from error
                yield ModelStreamCompleted(
                    with_request_admission(
                        response,
                        request,
                        input_tokens=counted_input_tokens,
                        output_cap=cast(int | None, arguments.get("max_tokens")),
                    )
                )
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            error_response = _safe_field(error, "response")
            if error_response is not None:
                self._observe_headers(
                    attempt, "generation", error_response, arrived=False
                )
            attempt.transport_failure(error, phase="generation")
            raise _normalize_error(error) from error

    def _require_supported_request_policy(self, request: ModelRequest) -> None:
        if not self.supports_request_policy(request):
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Anthropic cannot enforce the requested tool-call policy",
            )

    @staticmethod
    def _observe_headers(
        attempt: AttemptLifecycle, phase: str, response: object, *, arrived: bool = True
    ) -> None:
        try:
            headers = getattr(response, "headers", None)
            request_id = (
                headers.get("request-id") if isinstance(headers, Mapping) else None
            )
            attempt.headers(
                phase,
                getattr(response, "status_code", None),
                request_id,
                arrived=arrived,
            )
        except Exception:
            pass  # Optional SDK metadata cannot change request behavior.

    async def _sdk_call(
        self,
        resource: object,
        method: str,
        arguments: dict[str, object],
        attempt: AttemptLifecycle,
        phase: str,
    ) -> object:
        arguments = {
            **arguments,
            "timeout": transport_timeout(
                attempt.request, deadline=attempt._phase_deadline
            ),
        }

        async def scope():
            view = getattr(resource, "with_streaming_response", None)
            if view is None:
                attempt.values[f"{phase}_headers_availability"] = "unsupported"
                yield await getattr(resource, method)(**arguments)
                return
            try:
                async with getattr(view, method)(**arguments) as raw:
                    attempt.track_response_release(_safe_field(raw, "http_response"))
                    self._observe_headers(attempt, phase, raw)
                    yield await raw.parse()
            except Exception as error:
                self._observe_headers(
                    attempt, phase, _safe_field(error, "response"), arrived=False
                )
                raise

        if phase == "count":

            async def count():
                async with closing_stream(scope()) as results:
                    return await anext(results)

            return await attempt.run_native(count())
        async with attempt.stream(scope()) as results:
            response = await anext(results)
            attempt.response(
                self._decode_response(response, requested_at=self._clock())
            )
            return response

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

    async def _admit_request(
        self,
        request: ModelRequest,
        arguments: dict[str, object],
        attempt: AttemptLifecycle,
    ) -> int | None:
        request.remaining_after(
            ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
        )
        if request.max_total_tokens is None and request.max_estimated_cost_usd is None:
            return None
        # This adapter cannot pin advance billing dimensions. The common check
        # rejects cost-limited requests before counting or generation.
        bound_request_output(
            request,
            input_tokens=0,
            input_tokens_counted=False,
            maximum_output_tokens=self.max_tokens,
        )
        count_arguments = {
            key: value for key, value in arguments.items() if key != "max_tokens"
        }
        attempt.start_count()
        try:
            counted = await self._sdk_call(
                self.client.messages,
                "count_tokens",
                count_arguments,
                attempt,
                "count",
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
        tokens = _field(counted, "input_tokens", None)
        if type(tokens) is not int or tokens < 0:
            raise token_count_error(invalid=True)
        attempt.counted(tokens)
        arguments["max_tokens"] = bound_request_output(
            request, input_tokens=tokens, maximum_output_tokens=self.max_tokens
        )
        attempt.values["output_cap"] = arguments["max_tokens"]
        return tokens

    def _decode_response(
        self,
        response: object,
        *,
        requested_at: datetime | None = None,
    ) -> ModelResponse:
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
                if stop_reason == "max_tokens":
                    raise ModelProviderError(
                        ProviderErrorCode.OUTPUT_LIMIT,
                        "Anthropic exhausted the output token limit",
                    )
                if stop_reason == "model_context_window_exceeded":
                    raise ModelProviderError(
                        ProviderErrorCode.CONTEXT_OVERFLOW,
                        "Anthropic exhausted the context window",
                    )
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
        response_model = _optional_text(
            _field(response, "model", None),
            "response model",
        )
        billing = _decode_anthropic_billing_usage(_field(response, "usage", None))
        provider_metadata["pricing_dimensions"] = {
            "response_model": response_model,
            "service_tier": billing.service_tier,
            "inference_geo": billing.inference_geo,
        }
        if replay_blocks or calls:
            provider_metadata[_CONTINUATION_KEY] = {
                "provider_id": self.provider_id,
                "content_blocks": replay_blocks,
            }
        return ModelResponse(
            finish_reason=finish_reason,
            text=text,
            tool_calls=tuple(calls),
            usage=self._priced_usage(
                billing,
                response_model=response_model,
                requested_at=requested_at or self._clock(),
            ),
            provider_id=self.provider_id,
            provider_response_id=response_id,
            provider_metadata=provider_metadata,
        )

    def _with_priced_usage(
        self,
        response: ModelResponse,
        *,
        billing: _AnthropicBillingUsage,
        response_model: str | None,
        requested_at: datetime,
    ) -> ModelResponse:
        metadata = dict(response.provider_metadata)
        metadata["pricing_dimensions"] = {
            "response_model": response_model,
            "service_tier": billing.service_tier,
            "inference_geo": billing.inference_geo,
        }
        return replace(
            response,
            usage=self._priced_usage(
                billing,
                response_model=response_model,
                requested_at=requested_at,
            ),
            provider_metadata=metadata,
        )

    def _priced_usage(
        self,
        billing: _AnthropicBillingUsage,
        *,
        response_model: str | None,
        requested_at: datetime,
    ) -> ModelUsage:
        usage = billing.usage
        if (
            response_model is None
            or billing.service_tier is None
            or not billing.token_counts_complete
            or not billing.cache_write_breakdown_complete
        ):
            return replace(
                usage,
                cost_estimate=CostEstimate.unavailable("billing_dimensions_incomplete"),
            )
        qualifiers = {"service_tier": billing.service_tier}
        if response_model in {"claude-opus-4-8", "claude-sonnet-5"}:
            if billing.inference_geo is None:
                return replace(
                    usage,
                    cost_estimate=CostEstimate.unavailable(
                        "billing_dimensions_incomplete"
                    ),
                )
            qualifiers["inference_geo"] = billing.inference_geo
        return replace(
            usage,
            cost_estimate=calculate_cost_estimate(
                self._pricing_schedules,
                provider="anthropic",
                model=response_model,
                endpoint="messages",
                requested_at=requested_at,
                qualifiers=qualifiers,
                usage_values={
                    "request_input_tokens": Decimal(usage.input_tokens),
                },
                quantities=(
                    BillableQuantity(
                        "input_uncached_tokens",
                        Decimal(
                            usage.input_tokens
                            - usage.cache_read_tokens
                            - usage.cache_write_tokens
                        ),
                        "token",
                    ),
                    BillableQuantity(
                        "input_cache_read_tokens",
                        Decimal(usage.cache_read_tokens),
                        "token",
                    ),
                    BillableQuantity(
                        "input_cache_write_5m_tokens",
                        Decimal(billing.cache_write_5m_tokens),
                        "token",
                    ),
                    BillableQuantity(
                        "input_cache_write_1h_tokens",
                        Decimal(billing.cache_write_1h_tokens),
                        "token",
                    ),
                    BillableQuantity(
                        "output_tokens",
                        Decimal(usage.output_tokens),
                        "token",
                    ),
                ),
            ),
        )


AnthropicProvider = AnthropicMessagesProvider


def _malformed_stream(
    *,
    phase: ProviderFailurePhase,
    code: str,
    provider_id: str,
    event_type: str | None = None,
) -> ModelProviderError:
    return ModelProviderError(
        ProviderErrorCode.MALFORMED_RESPONSE,
        "Anthropic returned a malformed stream",
        provider_id=provider_id,
        diagnostic=ProviderFailureDiagnostic(
            phase=phase,
            code=code,
            event_type=event_type,
        ),
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
    if isinstance(error, ModelProviderError):
        return error
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
        retry_after_seconds=(
            retry_after_from_headers(
                _field(_field(error, "response", None), "headers", None)
            )
            if normalized
            in {
                ProviderErrorCode.RATE_LIMIT_ERROR,
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
            }
            else None
        ),
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


def _safe_field(value: object, name: str) -> object | None:
    try:
        return _field(value, name, None)
    except Exception:
        return None


def _text_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    return value


__all__ = ["AnthropicMessagesProvider", "AnthropicProvider"]
