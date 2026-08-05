"""Small ordered provider fallback with transport-only retries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, replace

from ..security import SecretReference
from .errors import ModelProviderError, ProviderErrorCode
from .models import (
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelUsage,
)
from .pricing import aggregate_cost_estimates
from .protocols import ModelProvider, provider_has_complete_pricing

_TRANSIENT = frozenset(
    {
        ProviderErrorCode.RATE_LIMIT_ERROR,
        ProviderErrorCode.PROVIDER_UNAVAILABLE,
        ProviderErrorCode.TIMEOUT,
    }
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 2
    backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempts, int)
            or isinstance(self.attempts, bool)
            or not 1 <= self.attempts <= 5
        ):
            raise ValueError("retry attempts must be from 1 through 5")
        if (
            not isinstance(self.backoff_seconds, (int, float))
            or isinstance(self.backoff_seconds, bool)
            or not 0 <= self.backoff_seconds <= 30
        ):
            raise ValueError("retry backoff_seconds must be from 0 through 30")
        object.__setattr__(self, "backoff_seconds", float(self.backoff_seconds))


@dataclass(frozen=True, slots=True)
class ModelProviderRegistration:
    provider: ModelProvider
    profile: ModelProfile
    allowed_sensitivities: frozenset[ModelSensitivity] = frozenset(
        {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
    )

    def __post_init__(self) -> None:
        if self.provider.provider_id != self.profile.id:
            raise ValueError("provider and profile identities differ")
        allowed = frozenset(self.allowed_sensitivities)
        if not allowed or any(
            not isinstance(item, ModelSensitivity) for item in allowed
        ):
            raise ValueError("allowed_sensitivities must contain model sensitivities")
        object.__setattr__(self, "allowed_sensitivities", allowed)


@dataclass(frozen=True, slots=True)
class ModelRouteCandidate:
    provider_id: str
    profile: ModelProfile
    base_url: str | None = None
    secret_reference: SecretReference | None = None
    allowed_sensitivities: frozenset[ModelSensitivity] = frozenset(
        {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
    )

    def __post_init__(self) -> None:
        if self.provider_id != self.profile.id:
            raise ValueError("route candidate provider_id must match its profile")
        if self.base_url is not None and (
            not isinstance(self.base_url, str) or not self.base_url.strip()
        ):
            raise ValueError("base_url must be non-empty when provided")
        if self.secret_reference is not None and not isinstance(
            self.secret_reference, SecretReference
        ):
            raise TypeError("secret_reference must be SecretReference or None")
        allowed = frozenset(self.allowed_sensitivities)
        if not allowed:
            raise ValueError("route candidate requires an allowed sensitivity")
        object.__setattr__(self, "allowed_sensitivities", allowed)


@dataclass(frozen=True, slots=True)
class ModelRoute:
    candidates: tuple[ModelRouteCandidate, ...]
    retry_policy: RetryPolicy = RetryPolicy()

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates or any(
            not isinstance(item, ModelRouteCandidate) for item in candidates
        ):
            raise ValueError("model route requires candidates")
        ids = tuple(item.provider_id for item in candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("model route candidates cannot repeat")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be RetryPolicy")
        object.__setattr__(self, "candidates", candidates)

    @property
    def model_profile(self) -> ModelProfile:
        if len(self.candidates) == 1:
            return self.candidates[0].profile
        profiles = tuple(item.profile for item in self.candidates)
        return ModelProfile(
            id="router:configured",
            context_window_tokens=min(item.context_window_tokens for item in profiles),
            max_output_tokens=min(item.max_output_tokens for item in profiles),
            supports_tools=all(item.supports_tools for item in profiles),
            supports_parallel_tools=all(
                item.supports_parallel_tools for item in profiles
            ),
            supports_structured_output=all(
                item.supports_structured_output for item in profiles
            ),
            supports_streaming=all(item.supports_streaming for item in profiles),
            supports_reasoning=all(item.supports_reasoning for item in profiles),
            supports_vision=all(item.supports_vision for item in profiles),
            supports_documents=all(item.supports_documents for item in profiles),
        )


class ModelRouter:
    """Try eligible providers in order; retry only normalized transient failures."""

    def __init__(
        self,
        candidates: Iterable[ModelProviderRegistration],
        *,
        retry_policy: RetryPolicy = RetryPolicy(),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        registrations = tuple(candidates)
        if not registrations or any(
            not isinstance(item, ModelProviderRegistration) for item in registrations
        ):
            raise ValueError("router requires provider registrations")
        if len({item.provider.provider_id for item in registrations}) != len(
            registrations
        ):
            raise ValueError("router providers cannot repeat")
        self._candidates = registrations
        self._retry_policy = retry_policy
        self._sleep = sleep

    @property
    def provider_id(self) -> str:
        return self.model_profile.id

    @property
    def candidates(self) -> tuple[ModelProviderRegistration, ...]:
        return self._candidates

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_policy

    @property
    def model_profile(self) -> ModelProfile:
        return ModelRoute(
            tuple(
                ModelRouteCandidate(
                    provider_id=item.profile.id,
                    profile=item.profile,
                    allowed_sensitivities=item.allowed_sensitivities,
                )
                for item in self._candidates
            ),
            self._retry_policy,
        ).model_profile

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return any(_eligible(item, request) for item in self._candidates)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        eligible = tuple(item for item in self._candidates if _eligible(item, request))
        return bool(eligible) and all(
            provider_has_complete_pricing(item.provider, request) for item in eligible
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        last_error: ModelProviderError | None = None
        attempt_usage: list[ModelUsage] = []
        for registration in self._candidates:
            if not _eligible(registration, request):
                continue
            for attempt in range(self._retry_policy.attempts):
                try:
                    response = await registration.provider.generate(request)
                except asyncio.CancelledError:
                    raise
                except ModelProviderError as error:
                    last_error = error
                    attempt_usage.append(error.usage)
                    if error.code not in _TRANSIENT:
                        break
                    if attempt + 1 < self._retry_policy.attempts:
                        delay = (
                            error.retry_after_seconds
                            if error.retry_after_seconds is not None
                            else self._retry_policy.backoff_seconds * (2**attempt)
                        )
                        if delay:
                            await self._sleep(delay)
                else:
                    attempt_usage.append(response.usage)
                    return replace(
                        response,
                        usage=_aggregate_usage(attempt_usage),
                    )
        if last_error is not None:
            raise ModelProviderError(
                last_error.code,
                str(last_error),
                provider_id=last_error.provider_id,
                retry_after_seconds=last_error.retry_after_seconds,
                usage=_aggregate_usage(attempt_usage),
            )
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "no configured provider can handle this request",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Route one canonical stream without retrying after visible progress."""

        last_error: ModelProviderError | None = None
        attempt_usage: list[ModelUsage] = []
        for registration in self._candidates:
            if not _eligible(registration, request):
                continue
            stream = getattr(registration.provider, "stream", None)
            if not registration.profile.supports_streaming or not callable(stream):
                continue
            for attempt in range(self._retry_policy.attempts):
                emitted = False
                try:
                    completed = False
                    async for event in stream(request):
                        if completed:
                            raise ModelProviderError(
                                ProviderErrorCode.MALFORMED_RESPONSE,
                                "provider stream continued after completion",
                                provider_id=registration.provider.provider_id,
                            )
                        if isinstance(event, ModelStreamCompleted):
                            completed = True
                            attempt_usage.append(event.response.usage)
                            yield ModelStreamCompleted(
                                replace(
                                    event.response,
                                    usage=_aggregate_usage(attempt_usage),
                                )
                            )
                            return
                        else:
                            emitted = True
                            yield event
                    if not completed:
                        raise ModelProviderError(
                            ProviderErrorCode.MALFORMED_RESPONSE,
                            "provider stream ended without a canonical completion",
                            provider_id=registration.provider.provider_id,
                        )
                    return
                except asyncio.CancelledError:
                    raise
                except ModelProviderError as error:
                    last_error = error
                    attempt_usage.append(error.usage)
                    if emitted:
                        raise ModelProviderError(
                            error.code,
                            str(error),
                            provider_id=error.provider_id,
                            retry_after_seconds=error.retry_after_seconds,
                            usage=_aggregate_usage(attempt_usage),
                        ) from None
                    if error.code not in _TRANSIENT:
                        break
                    if attempt + 1 < self._retry_policy.attempts:
                        delay = (
                            error.retry_after_seconds
                            if error.retry_after_seconds is not None
                            else self._retry_policy.backoff_seconds * (2**attempt)
                        )
                        if delay:
                            await self._sleep(delay)
        if last_error is not None:
            raise ModelProviderError(
                last_error.code,
                str(last_error),
                provider_id=last_error.provider_id,
                retry_after_seconds=last_error.retry_after_seconds,
                usage=_aggregate_usage(attempt_usage),
            )
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "no configured provider can stream this request",
        )


def _eligible(registration: ModelProviderRegistration, request: ModelRequest) -> bool:
    profile = registration.profile
    if request.sensitivity not in registration.allowed_sensitivities:
        return False
    if not profile.available or not profile.healthy:
        return False
    if request.tools and not profile.supports_tools:
        return False
    if request.response_schema is not None and not profile.supports_structured_output:
        return False
    return registration.provider.supports_request_policy(request) is True


def _aggregate_usage(items: Iterable[ModelUsage]) -> ModelUsage:
    usage = tuple(items)
    return ModelUsage(
        input_tokens=sum(item.input_tokens for item in usage),
        output_tokens=sum(item.output_tokens for item in usage),
        reasoning_tokens=sum(item.reasoning_tokens for item in usage),
        cache_read_tokens=sum(item.cache_read_tokens for item in usage),
        cache_write_tokens=sum(item.cache_write_tokens for item in usage),
        cost_estimate=aggregate_cost_estimates(item.cost_estimate for item in usage),
    )


__all__ = [
    "ModelProviderRegistration",
    "ModelRoute",
    "ModelRouteCandidate",
    "ModelRouter",
    "RetryPolicy",
]
