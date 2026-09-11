"""Translate canonical requests for scoped OpenAI-compatible chat endpoints."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol, cast
from urllib.parse import urlsplit
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
    retry_after_from_headers,
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
    CostBasis,
    PricingQualifier,
    PricingSchedule,
    bound_request_output,
    calculate_cost_estimate,
    has_complete_pricing_coverage,
    validate_pricing_schedules,
)
from .._fields import (
    field as _field,
    optional_text as _optional_text,
    required_text as _required_text,
    usage_int as _usage_int,
)
from .messages import _CONTINUATION_KEY, _chat_messages

_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class _CompletionsResource(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _ChatResource(Protocol):
    @property
    def completions(self) -> _CompletionsResource: ...


class _OpenAICompatibleClient(Protocol):
    @property
    def chat(self) -> _ChatResource: ...

    async def close(self) -> None: ...


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class _StreamedToolCall:
    canonical_id: str
    provider_call_id: str | None = None
    name: str | None = None
    argument_fragments: list[str] = field(default_factory=list)


class OpenAICompatibleProvider:
    """Translate canonical requests to one explicitly configured chat endpoint."""

    def __init__(
        self,
        model: str,
        *,
        provider: str,
        base_url: str,
        api_key: str | None = None,
        max_tokens: int = 1_024,
        client: _OpenAICompatibleClient | None = None,
        id_factory: Callable[[str], str] | None = None,
        pricing_schedules: Iterable[PricingSchedule] = (),
        pricing_qualifiers: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(provider, str) or not _PROVIDER_NAME.fullmatch(provider):
            raise ValueError("provider must be a canonical provider name")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        if id_factory is not None and not callable(id_factory):
            raise TypeError("id_factory must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if pricing_qualifiers is not None and not isinstance(
            pricing_qualifiers, Mapping
        ):
            raise TypeError("pricing_qualifiers must be a mapping or None")
        if pricing_qualifiers is not None and len(pricing_qualifiers) > 16:
            raise ValueError("pricing_qualifiers exceed their bound")
        admitted_schedules = validate_pricing_schedules(pricing_schedules)
        if any(
            schedule.basis is not CostBasis.CONFIGURED_CONTRACT
            for schedule in admitted_schedules
        ):
            raise ValueError(
                "compatible endpoint schedules must use configured_contract"
            )
        qualifier_values = {} if pricing_qualifiers is None else pricing_qualifiers
        admitted_qualifiers = tuple(
            PricingQualifier(name, value) for name, value in qualifier_values.items()
        )
        self.model = model.strip()
        self.provider = provider
        self.base_url = _validate_base_url(base_url)
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = client
        self._owns_client = client is None
        self._native_owner = NativeOwner()
        self._close = CloseCoordinator(self._native_owner)
        self._id_factory = _new_id if id_factory is None else id_factory
        self._pricing_schedules = admitted_schedules
        self._pricing_qualifiers = admitted_qualifiers
        self._clock = clock

    @property
    def provider_id(self) -> str:
        return f"{self.provider}:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return has_complete_pricing_coverage(
            self._pricing_schedules,
            provider=self.provider,
            model=self.model,
            endpoint="chat_completions",
            requested_at=self._clock(),
            qualifiers=self._pricing_qualifiers,
            required_metrics=(
                "input_uncached_tokens",
                "input_cache_read_tokens",
                "input_cache_write_tokens",
                "output_tokens",
            ),
            usage_range_metric="request_input_tokens",
        )

    @property
    def client(self) -> _OpenAICompatibleClient:
        if self._close.started:
            raise RuntimeError(f"{self.provider} provider is closed")
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise ImportError(
                    "Daita's OpenAI-compatible runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from error
            self._client = cast(
                _OpenAICompatibleClient,
                AsyncOpenAI(
                    api_key=self._api_key, base_url=self.base_url, max_retries=0
                ),
            )
        if not self._owns_client:
            # Use a request view; do not change or close the caller's SDK client.
            with_options = getattr(self._client, "with_options", None)
            if callable(with_options):
                return cast(_OpenAICompatibleClient, with_options(max_retries=0))
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
            boundary_name="Compatible",
            operation=self._generate,
            headers_supported=False,
        )

    async def _generate(
        self, request: ModelRequest, attempt: AttemptLifecycle
    ) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        arguments = self._request_arguments(request)
        requested_at = self._clock()
        try:
            attempt.dispatch()
            response = await attempt.run_native(
                self.client.chat.completions.create(
                    **arguments, timeout=transport_timeout(request)
                )
            )
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error, self.provider) from error
        try:
            return self._decode_response(response, requested_at=requested_at)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                f"{self.provider} returned a malformed response",
                provider_id=self.provider_id,
                diagnostic=ProviderFailureDiagnostic(
                    phase=ProviderFailurePhase.RESPONSE_DECODE,
                    code="response_decode_failed",
                ),
            ) from error

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Translate ordered Responses API events into canonical stream events."""
        return execute_stream_attempt(
            self._native_owner,
            request,
            provider_id=self.provider_id,
            boundary_name="Compatible",
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
        arguments = self._request_arguments(request)
        arguments["stream"] = True
        arguments["stream_options"] = {"include_usage": True}
        requested_at = self._clock()
        try:
            attempt.dispatch()
            source = native_events(
                lambda: self.client.chat.completions.create(
                    **arguments, timeout=transport_timeout(request)
                )
            )
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error, self.provider) from error

        from .stream import decode_compatible_stream

        async with closing_stream(
            decode_compatible_stream(
                self,
                source,
                request,
                attempt,
                requested_at=requested_at,
            )
        ) as events:
            async for event in events:
                yield event

    def _request_arguments(self, request: ModelRequest) -> dict[str, object]:
        try:
            arguments: dict[str, object] = {
                "max_tokens": self._max_tokens,
                "messages": _chat_messages(request.messages, self.provider_id),
                "model": self.model,
            }
            if request.allow_parallel_tool_calls is not None:
                arguments["parallel_tool_calls"] = request.allow_parallel_tool_calls
            if request.tools:
                arguments["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": FrozenJsonObject.from_mapping(
                                tool.input_schema
                            ).to_dict(),
                        },
                    }
                    for tool in request.tools
                ]
            if request.response_schema is not None:
                arguments["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "daita_response",
                        "strict": True,
                        "schema": FrozenJsonObject.from_mapping(
                            request.response_schema
                        ).to_dict(),
                    },
                }
            arguments["max_tokens"] = bound_request_output(
                request,
                # Chat compatibility defines no complete request-count API.
                # Bound output here and reconcile input usage in the loop.
                input_tokens=None,
                maximum_output_tokens=self._max_tokens,
                schedules=self._pricing_schedules,
                provider=self.provider,
                model=self.model,
                endpoint="chat_completions",
                requested_at=self._clock(),
                qualifiers=self._pricing_qualifiers,
            )
            return arguments
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "canonical request cannot be translated for compatible chat",
            ) from error

    def _decode_response(
        self,
        response: object,
        *,
        requested_at: datetime | None = None,
    ) -> ModelResponse:
        choices = _sequence(_field(response, "choices"), "response choices")
        if len(choices) != 1:
            raise ValueError("response must contain exactly one choice")
        choice = choices[0]
        message = _field(choice, "message")
        refusal = _optional_text(_field(message, "refusal", None), "refusal")
        if refusal is not None:
            raise ModelProviderError(
                ProviderErrorCode.CONTENT_BLOCKED,
                f"{self.provider} blocked the response",
            )
        content = _optional_text(_field(message, "content", None), "content")
        raw_calls = _field(message, "tool_calls", ())
        if raw_calls is None:
            raw_calls = ()
        tool_calls = _sequence(raw_calls, "message tool calls")
        calls: list[ToolCall] = []
        canonical_ids: set[str] = set()
        for item in tool_calls:
            if _required_text(_field(item, "type", "function"), "tool type") != (
                "function"
            ):
                raise ValueError("only function tool calls are supported")
            function = _field(item, "function")
            encoded_arguments = _required_text(
                _field(function, "arguments"),
                "tool arguments",
            )
            decoded_arguments = json.loads(encoded_arguments)
            if not isinstance(decoded_arguments, dict):
                raise ValueError("tool arguments must decode to an object")
            canonical_id = self._id_factory("call")
            if canonical_id in canonical_ids:
                raise ValueError("id_factory returned a duplicate call ID")
            canonical_ids.add(canonical_id)
            calls.append(
                ToolCall(
                    id=canonical_id,
                    provider_call_id=_required_text(
                        _field(item, "id"),
                        "provider tool-call id",
                    ),
                    name=_required_text(_field(function, "name"), "tool name"),
                    arguments=decoded_arguments,
                )
            )
        native_finish = _required_text(
            _field(choice, "finish_reason"),
            "finish reason",
        )
        if native_finish == "content_filter":
            raise ModelProviderError(
                ProviderErrorCode.CONTENT_BLOCKED,
                f"{self.provider} blocked the response",
            )
        finish_reason = _finish_reason(native_finish)
        if not calls and content is None and finish_reason is FinishReason.LENGTH:
            raise ModelProviderError(
                ProviderErrorCode.OUTPUT_LIMIT,
                f"{self.provider} exhausted the output token limit",
            )
        response_id = _optional_text(_field(response, "id", None), "response id")
        response_model = _optional_text(
            _field(response, "model", None),
            "response model",
        )
        service_tier = _optional_text(
            _field(response, "service_tier", None),
            "response service tier",
        )
        return ModelResponse(
            finish_reason=finish_reason,
            text=content,
            tool_calls=tuple(calls),
            usage=self._decode_priced_usage(
                _field(response, "usage", None),
                response_model=response_model,
                service_tier=service_tier,
                requested_at=requested_at or self._clock(),
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

    def _decode_usage(self, value: object) -> ModelUsage:
        """Decode compatible usage; fixed providers may refine native semantics."""

        return _decode_usage(value)

    def _decode_priced_usage(
        self,
        value: object,
        *,
        response_model: str | None,
        service_tier: str | None,
        requested_at: datetime,
    ) -> ModelUsage:
        usage = self._decode_usage(value)
        if value is None or not self._pricing_schedules:
            return usage
        qualifiers = {item.name: item.value for item in self._pricing_qualifiers}
        if service_tier is not None and "service_tier" in qualifiers:
            qualifiers["service_tier"] = service_tier
        return replace(
            usage,
            cost_estimate=calculate_cost_estimate(
                self._pricing_schedules,
                provider=self.provider,
                model=response_model or self.model,
                endpoint="chat_completions",
                requested_at=requested_at,
                qualifiers=qualifiers,
                usage_values={"request_input_tokens": Decimal(usage.input_tokens)},
                quantities=_billable_quantities(usage),
            ),
        )


def _decode_usage(value: object) -> ModelUsage:
    if value is None:
        return ModelUsage()
    prompt_details = _field(value, "prompt_tokens_details", None)
    completion_details = _field(value, "completion_tokens_details", None)
    input_tokens = _usage_int(
        _field(value, "prompt_tokens"),
        "prompt tokens",
    )
    output_tokens = _usage_int(
        _field(value, "completion_tokens"),
        "completion tokens",
    )
    reasoning_tokens = _usage_int(
        _field(completion_details, "reasoning_tokens", 0),
        "reasoning tokens",
    )
    cache_read_tokens = _usage_int(
        _field(prompt_details, "cached_tokens", 0),
        "cached tokens",
    )
    cache_write_tokens = _usage_int(
        _field(prompt_details, "cache_write_tokens", 0),
        "cache write tokens",
    )
    if cache_read_tokens + cache_write_tokens > input_tokens:
        raise ValueError("compatible cache token subsets exceed total prompt tokens")
    if reasoning_tokens > output_tokens:
        raise ValueError("compatible reasoning tokens exceed completion tokens")
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _billable_quantities(usage: ModelUsage) -> tuple[BillableQuantity, ...]:
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0 or usage.reasoning_tokens > usage.output_tokens:
        raise ValueError("compatible usage counters are internally inconsistent")
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
            "input_cache_write_tokens",
            Decimal(usage.cache_write_tokens),
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
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALLS,
            "length": FinishReason.LENGTH,
        }[value]
    except KeyError as error:
        raise ValueError("unknown finish reason") from error


def _normalize_error(error: Exception, provider: str) -> ModelProviderError:
    status_value = _lenient_field(error, "status_code")
    status = (
        status_value
        if isinstance(status_value, int) and not isinstance(status_value, bool)
        else None
    )
    code_value = _lenient_field(error, "code")
    code = code_value if isinstance(code_value, str) else None
    name = type(error).__name__.lower()
    if (
        isinstance(error, (asyncio.TimeoutError, TimeoutError))
        or status == 408
        or "timeout" in name
    ):
        normalized = ProviderErrorCode.TIMEOUT
    elif status in {401, 403} or "authentication" in name or "permission" in name:
        normalized = ProviderErrorCode.AUTHENTICATION_ERROR
    elif status == 429 or "ratelimit" in name or "rate_limit" in name:
        normalized = ProviderErrorCode.RATE_LIMIT_ERROR
    elif status == 404 or code in {"model_not_found", "unknown_model"}:
        normalized = ProviderErrorCode.MODEL_NOT_FOUND
    elif code in {"context_length_exceeded", "context_window_exceeded"}:
        normalized = ProviderErrorCode.CONTEXT_OVERFLOW
    elif code in {"content_policy_violation", "content_blocked"}:
        normalized = ProviderErrorCode.CONTENT_BLOCKED
    elif isinstance(error, ConnectionError) or status is not None and status >= 500:
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    elif status is not None and 400 <= status < 500:
        normalized = ProviderErrorCode.INVALID_REQUEST
    else:
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    return ModelProviderError(
        normalized,
        f"{provider} request failed: {normalized.value}",
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


def _validate_base_url(value: str, *, loopback_only: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("base_url must be a normalized absolute URL")
    parse_failed = False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        parse_failed = True
    if parse_failed:
        raise ValueError("base_url must be a valid absolute URL")
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an uncredentialed HTTP(S) endpoint")
    loopback = _is_loopback_host(parsed.hostname)
    if parsed.scheme == "http" and not loopback:
        raise ValueError("base_url permits HTTP only for a loopback endpoint")
    if loopback_only and not loopback:
        raise ValueError("base_url must use a loopback endpoint")
    return value.rstrip("/")


def _is_loopback_host(value: str) -> bool:
    if value.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _lenient_field(value: object, name: str) -> object | None:
    try:
        return _field(value, name, None)
    except Exception:
        return None


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


__all__ = ["OpenAICompatibleProvider"]
