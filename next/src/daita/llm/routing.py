"""Explicit provider registry and the sole normalized retry/fallback router."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import math
from time import monotonic_ns
from typing import cast

from .._json import canonical_json
from .errors import ModelProviderError, ProviderErrorCode, detached_provider_error
from .models import (
    CanonicalMessage,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelRouteAttempt,
    ModelRouteAttemptOutcome,
    ModelRoutingTrace,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
    ModelUsage,
    ToolCall,
)
from .protocols import ModelProvider

_DEFAULT_RETRYABLE_CODES = frozenset(
    {
        ProviderErrorCode.RATE_LIMIT_ERROR,
        ProviderErrorCode.PROVIDER_UNAVAILABLE,
        ProviderErrorCode.TIMEOUT,
    }
)


class RetryStrategy(str, Enum):
    """Deterministic delay strategy for same-provider retries."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Versioned bounded retry policy applied only by ``ModelRouter``."""

    schema_version: int = 1
    max_attempts_per_provider: int = 1
    strategy: RetryStrategy = RetryStrategy.FIXED
    base_delay_seconds: float = 0.0
    max_delay_seconds: float = 0.0
    retryable_codes: frozenset[ProviderErrorCode] = _DEFAULT_RETRYABLE_CODES

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("retry policy schema_version must be 1")
        if (
            not isinstance(self.max_attempts_per_provider, int)
            or isinstance(self.max_attempts_per_provider, bool)
            or not 1 <= self.max_attempts_per_provider <= 4
        ):
            raise ValueError("max_attempts_per_provider must be from one through four")
        if not isinstance(self.strategy, RetryStrategy):
            raise TypeError("retry strategy must be a RetryStrategy")
        for value, field_name in (
            (self.base_delay_seconds, "base_delay_seconds"),
            (self.max_delay_seconds, "max_delay_seconds"),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, float(value))
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be less than base_delay_seconds")
        if not isinstance(self.retryable_codes, frozenset) or any(
            not isinstance(item, ProviderErrorCode) for item in self.retryable_codes
        ):
            raise TypeError("retryable_codes must be a frozenset of ProviderErrorCode")
        if not self.retryable_codes <= _DEFAULT_RETRYABLE_CODES:
            raise ValueError("retry policy cannot make terminal failures retryable")

    def delay_after(self, failed_attempt: int) -> float:
        """Return the deterministic delay after a numbered failed attempt."""

        if (
            not isinstance(failed_attempt, int)
            or isinstance(failed_attempt, bool)
            or failed_attempt < 1
        ):
            raise ValueError("failed_attempt must be a positive integer")
        if self.strategy is RetryStrategy.FIXED:
            delay = self.base_delay_seconds
        elif self.strategy is RetryStrategy.LINEAR:
            delay = self.base_delay_seconds * failed_attempt
        else:
            delay = self.base_delay_seconds * (2 ** (failed_attempt - 1))
        return min(delay, self.max_delay_seconds)


@dataclass(frozen=True, slots=True)
class ModelProviderRegistration:
    """One explicit provider/profile/policy binding; never a global registration."""

    provider: ModelProvider = field(repr=False, compare=False)
    profile: ModelProfile
    allowed_sensitivities: frozenset[ModelSensitivity]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ModelProfile):
            raise TypeError("provider registration profile must be ModelProfile")
        provider_id = getattr(self.provider, "provider_id", None)
        if provider_id != self.profile.id:
            raise ValueError("provider registration profile must match provider_id")
        if isinstance(self.allowed_sensitivities, (str, bytes)):
            raise TypeError("allowed_sensitivities must be a set of sensitivities")
        allowed = frozenset(self.allowed_sensitivities)
        if not allowed:
            raise ValueError(
                "provider registration requires an explicit sensitivity grant"
            )
        if any(not isinstance(item, ModelSensitivity) for item in allowed):
            raise TypeError(
                "allowed_sensitivities must contain ModelSensitivity values"
            )
        object.__setattr__(self, "allowed_sensitivities", allowed)


class ModelRegistry:
    """Immutable, explicitly injected provider lookup without package scanning."""

    def __init__(self, registrations: Iterable[ModelProviderRegistration]) -> None:
        if isinstance(registrations, (str, bytes)):
            raise TypeError("registrations must be an iterable of provider bindings")
        items = tuple(registrations)
        if not items:
            raise ValueError("model registry requires at least one provider")
        if any(not isinstance(item, ModelProviderRegistration) for item in items):
            raise TypeError("registrations must contain ModelProviderRegistration")
        ids = tuple(item.profile.id for item in items)
        if len(ids) != len(set(ids)):
            raise ValueError("model registry provider IDs must be unique")
        self._registrations = items
        self._by_id = dict(zip(ids, items, strict=True))

    @property
    def registrations(self) -> tuple[ModelProviderRegistration, ...]:
        return self._registrations

    def get(self, provider_id: str) -> ModelProviderRegistration:
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        try:
            return self._by_id[provider_id]
        except KeyError as error:
            raise KeyError(f"unknown model provider: {provider_id}") from error


class ModelRouter:
    """Select compatible approved providers and own all retry/fallback decisions."""

    def __init__(
        self,
        primary: ModelProviderRegistration,
        fallbacks: Iterable[ModelProviderRegistration] = (),
        *,
        retry_policy: RetryPolicy | None = None,
        max_attempts_per_provider: int | None = None,
        retryable_codes: frozenset[ProviderErrorCode] | None = None,
        monotonic_clock: Callable[[], int] = monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(primary, ModelProviderRegistration):
            raise TypeError("primary must be a ModelProviderRegistration")
        if isinstance(fallbacks, (str, bytes)):
            raise TypeError("fallbacks must be provider registrations")
        candidates = (primary, *tuple(fallbacks))
        if any(not isinstance(item, ModelProviderRegistration) for item in candidates):
            raise TypeError("fallbacks must contain ModelProviderRegistration")
        if len(candidates) > 8:
            raise ValueError("model route supports at most eight candidates")
        ids = tuple(item.profile.id for item in candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("model route candidates must have unique provider IDs")
        if retry_policy is not None and (
            max_attempts_per_provider is not None or retryable_codes is not None
        ):
            raise ValueError(
                "retry_policy cannot be combined with legacy retry arguments"
            )
        if retry_policy is None:
            retry_policy = RetryPolicy(
                max_attempts_per_provider=(
                    1
                    if max_attempts_per_provider is None
                    else max_attempts_per_provider
                ),
                strategy=RetryStrategy.FIXED,
                base_delay_seconds=0.0,
                max_delay_seconds=0.0,
                retryable_codes=(
                    _DEFAULT_RETRYABLE_CODES
                    if retryable_codes is None
                    else retryable_codes
                ),
            )
        elif not isinstance(retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy or None")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        if not callable(sleep):
            raise TypeError("sleep must be callable")

        self._candidates = candidates
        self._retry_policy = retry_policy
        self._max_attempts = retry_policy.max_attempts_per_provider
        self._retryable_codes = retry_policy.retryable_codes
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        fingerprint = _route_fingerprint(
            candidates,
            retry_policy=retry_policy,
        )
        self._provider_id = f"router:{fingerprint}"
        self._profile = replace(
            primary.profile,
            id=self._provider_id,
            input_cost_per_million_usd=None,
            output_cost_per_million_usd=None,
            data_routing_classification="explicit_route_policy",
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def candidates(self) -> tuple[ModelProviderRegistration, ...]:
        return self._candidates

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_policy

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
        if failure is None:
            raise AssertionError("model route failed without an error")
        raise detached_provider_error(failure)

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        attempts: list[ModelRouteAttempt] = []
        last_error: ModelProviderError | None = None
        eligible = False
        for registration in self._candidates:
            if not _candidate_is_eligible(registration, request):
                continue
            eligible = True
            portable_request = _request_for_provider(request, registration.profile.id)
            for attempt_number in range(1, self._max_attempts + 1):
                started = self._read_clock()
                try:
                    response = await registration.provider.generate(portable_request)
                except asyncio.CancelledError:
                    raise
                except ImportError:
                    raise
                except ModelProviderError as error:
                    latency = self._elapsed_ms(started)
                    attempts.append(
                        ModelRouteAttempt(
                            provider_id=registration.profile.id,
                            attempt=attempt_number,
                            outcome=ModelRouteAttemptOutcome.FAILED,
                            latency_ms=latency,
                            error_code=error.code.value,
                        )
                    )
                    last_error = error
                    if error.code not in self._retryable_codes:
                        raise self._terminal_error(error.code, attempts) from error
                    break_or_retry = attempt_number >= self._max_attempts
                    if break_or_retry:
                        break
                    await self._wait_before_retry(attempt_number)
                    continue
                except Exception as error:
                    latency = self._elapsed_ms(started)
                    attempts.append(
                        ModelRouteAttempt(
                            provider_id=registration.profile.id,
                            attempt=attempt_number,
                            outcome=ModelRouteAttemptOutcome.FAILED,
                            latency_ms=latency,
                            error_code=ProviderErrorCode.MALFORMED_RESPONSE.value,
                        )
                    )
                    raise self._terminal_error(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        attempts,
                    ) from error
                latency = self._elapsed_ms(started)
                if not isinstance(response, ModelResponse):
                    attempts.append(
                        ModelRouteAttempt(
                            provider_id=registration.profile.id,
                            attempt=attempt_number,
                            outcome=ModelRouteAttemptOutcome.FAILED,
                            latency_ms=latency,
                            error_code=ProviderErrorCode.MALFORMED_RESPONSE.value,
                        )
                    )
                    raise self._terminal_error(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        attempts,
                    )
                if response.provider_id not in {None, registration.profile.id}:
                    attempts.append(
                        ModelRouteAttempt(
                            provider_id=registration.profile.id,
                            attempt=attempt_number,
                            outcome=ModelRouteAttemptOutcome.FAILED,
                            latency_ms=latency,
                            error_code=ProviderErrorCode.MALFORMED_RESPONSE.value,
                        )
                    )
                    raise self._terminal_error(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        attempts,
                    )
                attempts.append(
                    ModelRouteAttempt(
                        provider_id=registration.profile.id,
                        attempt=attempt_number,
                        outcome=ModelRouteAttemptOutcome.SUCCEEDED,
                        latency_ms=latency,
                    )
                )
                trace = ModelRoutingTrace(
                    route_id=self._provider_id,
                    primary_provider_id=self._candidates[0].profile.id,
                    selected_provider_id=registration.profile.id,
                    attempts=tuple(attempts),
                )
                return replace(
                    response,
                    provider_id=registration.profile.id,
                    usage=_usage_with_profile_cost(
                        response.usage, registration.profile
                    ),
                    routing=trace,
                )

        code = (
            last_error.code
            if last_error is not None
            else ProviderErrorCode.INVALID_REQUEST
        )
        if not eligible:
            return_error = self._terminal_error(code, attempts)
            raise return_error
        raise self._terminal_error(code, attempts) from last_error

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Route a canonical stream, falling back only before any delta escapes."""

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
        if failure is None:
            raise AssertionError("model route failed without an error")
        raise detached_provider_error(failure)

    async def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        attempts: list[ModelRouteAttempt] = []
        last_error: ModelProviderError | None = None
        eligible = False
        for registration in self._candidates:
            if not _candidate_is_eligible(
                registration,
                request,
                require_streaming=True,
            ):
                continue
            eligible = True
            portable_request = _request_for_provider(request, registration.profile.id)
            stream_value = getattr(registration.provider, "stream", None)
            if not callable(stream_value):
                raise self._terminal_error(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    attempts,
                )
            stream_method = cast(
                Callable[[ModelRequest], AsyncIterator[ModelStreamEvent]],
                stream_value,
            )
            for attempt_number in range(1, self._max_attempts + 1):
                started = self._read_clock()
                emitted = False
                try:
                    provider_stream = stream_method(portable_request)
                    async for event in provider_stream:
                        if isinstance(event, (ModelTextDelta, ModelToolCallDelta)):
                            emitted = True
                            yield event
                            continue
                        if not isinstance(event, ModelStreamCompleted):
                            raise ModelProviderError(
                                ProviderErrorCode.MALFORMED_RESPONSE,
                                "provider returned an unknown stream event",
                            )
                        response = event.response
                        if response.provider_id not in {
                            None,
                            registration.profile.id,
                        }:
                            raise ModelProviderError(
                                ProviderErrorCode.MALFORMED_RESPONSE,
                                "provider stream returned another provider identity",
                            )
                        attempts.append(
                            ModelRouteAttempt(
                                provider_id=registration.profile.id,
                                attempt=attempt_number,
                                outcome=ModelRouteAttemptOutcome.SUCCEEDED,
                                latency_ms=self._elapsed_ms(started),
                            )
                        )
                        trace = ModelRoutingTrace(
                            route_id=self._provider_id,
                            primary_provider_id=self._candidates[0].profile.id,
                            selected_provider_id=registration.profile.id,
                            attempts=tuple(attempts),
                        )
                        yield ModelStreamCompleted(
                            replace(
                                response,
                                provider_id=registration.profile.id,
                                usage=_usage_with_profile_cost(
                                    response.usage,
                                    registration.profile,
                                ),
                                routing=trace,
                            )
                        )
                        return
                    raise ModelProviderError(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        "provider stream ended without a terminal response",
                    )
                except asyncio.CancelledError:
                    raise
                except ImportError:
                    raise
                except ModelProviderError as error:
                    attempts.append(
                        ModelRouteAttempt(
                            provider_id=registration.profile.id,
                            attempt=attempt_number,
                            outcome=ModelRouteAttemptOutcome.FAILED,
                            latency_ms=self._elapsed_ms(started),
                            error_code=error.code.value,
                        )
                    )
                    last_error = error
                    if emitted or error.code not in self._retryable_codes:
                        raise self._terminal_error(error.code, attempts) from error
                    if attempt_number < self._max_attempts:
                        await self._wait_before_retry(attempt_number)
                        continue
                    break
                except Exception as error:
                    attempts.append(
                        ModelRouteAttempt(
                            provider_id=registration.profile.id,
                            attempt=attempt_number,
                            outcome=ModelRouteAttemptOutcome.FAILED,
                            latency_ms=self._elapsed_ms(started),
                            error_code=ProviderErrorCode.MALFORMED_RESPONSE.value,
                        )
                    )
                    raise self._terminal_error(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        attempts,
                    ) from error

        code = (
            last_error.code
            if last_error is not None
            else ProviderErrorCode.INVALID_REQUEST
        )
        if not eligible:
            raise self._terminal_error(code, attempts)
        raise self._terminal_error(code, attempts) from last_error

    def _terminal_error(
        self,
        code: ProviderErrorCode,
        attempts: list[ModelRouteAttempt],
    ) -> ModelProviderError:
        trace = ModelRoutingTrace(
            route_id=self._provider_id,
            primary_provider_id=self._candidates[0].profile.id,
            attempts=tuple(attempts),
            terminal_error_code=code.value,
        )
        return ModelProviderError(
            code,
            f"model route failed: {code.value}",
            routing=trace,
        )

    def _read_clock(self) -> int:
        value = self._monotonic_clock()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(
                "monotonic model-routing clock returned an invalid value"
            )
        return value

    def _elapsed_ms(self, started: int) -> int:
        finished = self._read_clock()
        return max(0, finished - started) // 1_000_000

    async def _wait_before_retry(self, failed_attempt: int) -> None:
        delay = self._retry_policy.delay_after(failed_attempt)
        if delay > 0:
            await self._sleep(delay)


def _candidate_is_eligible(
    registration: ModelProviderRegistration,
    request: ModelRequest,
    *,
    require_streaming: bool = False,
) -> bool:
    profile = registration.profile
    if request.sensitivity not in registration.allowed_sensitivities:
        return False
    if not profile.available or not profile.healthy:
        return False
    if request.tools and not profile.supports_tools:
        return False
    if request.response_schema is not None and not profile.supports_structured_output:
        return False
    if require_streaming and not profile.supports_streaming:
        return False
    estimated_input = request.context_selection.get("estimated_input_tokens")
    # The context builder owns token accounting; routing must never guess or
    # silently bypass that safety fact. Direct provider use remains possible
    # without a context-selection record, but a routed request fails closed.
    if (
        not isinstance(estimated_input, int)
        or isinstance(estimated_input, bool)
        or estimated_input < 0
        or estimated_input > profile.maximum_input_tokens
    ):
        return False
    return True


def _request_for_provider(request: ModelRequest, provider_id: str) -> ModelRequest:
    messages = tuple(
        _message_for_provider(message, provider_id) for message in request.messages
    )
    return (
        request if messages == request.messages else replace(request, messages=messages)
    )


def _message_for_provider(
    message: CanonicalMessage,
    provider_id: str,
) -> CanonicalMessage:
    if message.role is not MessageRole.ASSISTANT or message.provider_id == provider_id:
        return message
    portable_calls = tuple(
        ToolCall(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
        )
        for call in message.tool_calls
    )
    return replace(
        message,
        tool_calls=portable_calls,
        provider_id=None,
        provider_metadata={},
    )


def _usage_with_profile_cost(
    usage: ModelUsage,
    profile: ModelProfile,
) -> ModelUsage:
    if (
        profile.input_cost_per_million_usd is None
        or profile.output_cost_per_million_usd is None
    ):
        return usage
    million = Decimal(1_000_000)
    estimated = (
        Decimal(usage.input_tokens) * profile.input_cost_per_million_usd
        + Decimal(usage.output_tokens) * profile.output_cost_per_million_usd
    ) / million
    return replace(usage, estimated_cost_usd=estimated)


def _route_fingerprint(
    candidates: tuple[ModelProviderRegistration, ...],
    *,
    retry_policy: RetryPolicy,
) -> str:
    encoded = canonical_json(
        {
            "candidates": [
                {
                    "allowed_sensitivities": sorted(
                        item.value for item in registration.allowed_sensitivities
                    ),
                    "profile": _profile_data(registration.profile),
                }
                for registration in candidates
            ],
            "retry_policy": {
                "base_delay_seconds": retry_policy.base_delay_seconds,
                "max_attempts_per_provider": (retry_policy.max_attempts_per_provider),
                "max_delay_seconds": retry_policy.max_delay_seconds,
                "retryable_codes": sorted(
                    item.value for item in retry_policy.retryable_codes
                ),
                "schema_version": retry_policy.schema_version,
                "strategy": retry_policy.strategy.value,
            },
            "schema_version": 1,
        }
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _profile_data(profile: ModelProfile) -> dict[str, object]:
    return {
        "available": profile.available,
        "context_window_tokens": profile.context_window_tokens,
        "data_routing_classification": profile.data_routing_classification,
        "healthy": profile.healthy,
        "id": profile.id,
        "input_cost_per_million_usd": (
            None
            if profile.input_cost_per_million_usd is None
            else str(profile.input_cost_per_million_usd)
        ),
        "max_output_tokens": profile.max_output_tokens,
        "output_cost_per_million_usd": (
            None
            if profile.output_cost_per_million_usd is None
            else str(profile.output_cost_per_million_usd)
        ),
        "supports_documents": profile.supports_documents,
        "supports_native_continuation": profile.supports_native_continuation,
        "supports_parallel_tools": profile.supports_parallel_tools,
        "supports_prompt_caching": profile.supports_prompt_caching,
        "supports_reasoning": profile.supports_reasoning,
        "supports_streaming": profile.supports_streaming,
        "supports_structured_output": profile.supports_structured_output,
        "supports_tools": profile.supports_tools,
        "supports_vision": profile.supports_vision,
    }


__all__ = [
    "ModelProviderRegistration",
    "ModelRegistry",
    "ModelRouter",
    "RetryPolicy",
    "RetryStrategy",
]
