"""Select ordered model providers and apply transport-only retries and fallbacks."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal

from ..errors import ErrorRetryability
from ..security import SecretReference
from ._lifecycle import await_cleanup, closing_stream
from .errors import (
    ModelProviderError,
    with_cancelled_model_usage,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
    interrupted_model_usage,
    before_generation,
)
from .models import (
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelUsage,
)
from .pricing import CostEstimateStatus, aggregate_cost_estimates
from .protocols import (
    ManagedModelProvider,
    ModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
    provider_supports_request_policy,
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
    close_with_router: bool = False

    def __post_init__(self) -> None:
        if self.provider.provider_id != self.profile.id:
            raise ValueError("provider and profile identities differ")
        if not isinstance(self.close_with_router, bool):
            raise TypeError("close_with_router must be a boolean")
        if self.close_with_router and not isinstance(
            self.provider, ManagedModelProvider
        ):
            raise TypeError("a router-owned provider must support managed close")
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
        if not allowed or any(
            not isinstance(item, ModelSensitivity) for item in allowed
        ):
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


@dataclass(slots=True)
class RunRoute:
    """One explicit provider-selection handle owned by a single run."""

    candidate_provider_ids: tuple[str, ...]
    initial_sensitivity: ModelSensitivity
    selected_provider_id: str | None = None

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_provider_ids)
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("run route requires distinct candidate provider IDs")
        if any(not isinstance(item, str) or not item for item in candidates):
            raise ValueError("run route candidate provider IDs must be non-empty")
        if not isinstance(self.initial_sensitivity, ModelSensitivity):
            raise TypeError("run route initial sensitivity is invalid")
        if self.selected_provider_id is not None and (
            self.selected_provider_id not in candidates
        ):
            raise ValueError("run route selected provider is not a candidate")
        self.candidate_provider_ids = candidates


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
        self._close_task: asyncio.Task[None] | None = None

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

    async def close(self) -> None:
        """Join once-only cleanup without activating unused delegates."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        await await_cleanup(self._close_task)

    async def _finish_close(self) -> None:
        first_error: BaseException | None = None
        for registration in reversed(self._candidates):
            if not registration.close_with_router:
                continue
            provider = registration.provider
            assert isinstance(provider, ManagedModelProvider)
            try:
                await provider.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _require_open(self) -> None:
        if self._close_task is not None:
            raise before_generation(
                ModelProviderError(
                    ProviderErrorCode.PROVIDER_UNAVAILABLE,
                    "model router is closed",
                ),
                code="router_closed",
            )

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return any(_eligible(item, request) for item in self._candidates)

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        eligible = tuple(item for item in self._candidates if _eligible(item, request))
        return bool(eligible) and all(
            provider_has_complete_pricing(item.provider, request) for item in eligible
        )

    def begin_run(self, sensitivity: ModelSensitivity) -> RunRoute:
        """Freeze the eligible initial candidate order for one run."""

        self._require_open()
        if not isinstance(sensitivity, ModelSensitivity):
            raise TypeError("run route sensitivity must be ModelSensitivity")
        candidates = tuple(
            item.provider.provider_id
            for item in self._candidates
            if sensitivity in item.allowed_sensitivities
            and item.profile.available
            and item.profile.healthy
        )
        if not candidates:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "no configured provider is eligible for the run sensitivity",
            )
        return RunRoute(candidates, sensitivity)

    def supports_run_request(self, route: RunRoute, request: ModelRequest) -> bool:
        return any(_eligible(item, request) for item in self._run_candidates(route))

    def has_complete_run_pricing(
        self,
        route: RunRoute,
        request: ModelRequest,
    ) -> bool:
        eligible = tuple(
            item for item in self._run_candidates(route) if _eligible(item, request)
        )
        return bool(eligible) and all(
            provider_has_complete_pricing(item.provider, request) for item in eligible
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await self._generate(request, None)

    async def generate_for_run(
        self,
        route: RunRoute,
        request: ModelRequest,
    ) -> ModelResponse:
        return await self._generate(request, route)

    async def _generate(
        self,
        request: ModelRequest,
        route: RunRoute | None,
    ) -> ModelResponse:
        self._require_open()
        last_error: ModelProviderError | None = None
        attempt_usage: list[ModelUsage] = []
        in_flight = False
        candidates = self._candidates if route is None else self._run_candidates(route)
        try:
            async with asyncio.timeout_at(request.deadline):
                for registration in candidates:
                    if not _eligible(registration, request):
                        continue
                    for attempt in range(self._retry_policy.attempts):
                        attempt_request = request.remaining_after(
                            _aggregate_usage(attempt_usage)
                        )
                        in_flight = True
                        try:
                            response = await registration.provider.generate(
                                attempt_request
                            )
                        except ModelProviderError as error:
                            in_flight = False
                            last_error = error
                            attempt_usage.append(error.usage)
                            _require_recoverable_usage(request, error, attempt_usage)
                            if not await self._wait_for_retry(
                                request, error, attempt, attempt_usage
                            ):
                                break
                        else:
                            in_flight = False
                            attempt_usage.append(response.usage)
                            if route is not None:
                                route.selected_provider_id = (
                                    registration.provider.provider_id
                                )
                            return replace(
                                response, usage=_aggregate_usage(attempt_usage)
                            )
        except asyncio.CancelledError as error:
            raise with_cancelled_model_usage(
                error,
                _interruption_usage(error, attempt_usage, in_flight),
            ) from None
        except TimeoutError as error:
            raise ModelProviderError(
                ProviderErrorCode.TIMEOUT,
                "The model request deadline expired.",
                usage=_interruption_usage(error, attempt_usage, in_flight),
            ) from None
        except ModelProviderError:
            raise
        except Exception as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "The model provider boundary failed.",
                usage=_interruption_usage(error, attempt_usage, in_flight),
            ) from None
        if last_error is not None:
            raise _aggregate_failure(last_error, attempt_usage)
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "no configured provider can handle this request",
            usage=_aggregate_usage(attempt_usage),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Route one canonical stream without retrying after visible progress."""

        async with closing_stream(self._stream(request, None)) as events:
            async for event in events:
                yield event

    async def stream_for_run(
        self,
        route: RunRoute,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        async with closing_stream(self._stream(request, route)) as events:
            async for event in events:
                yield event

    async def _stream(
        self,
        request: ModelRequest,
        route: RunRoute | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Keep completion and all visible progress terminal for retry purposes."""
        self._require_open()
        last_error: ModelProviderError | None = None
        attempt_usage: list[ModelUsage] = []
        in_flight = False
        candidates = self._candidates if route is None else self._run_candidates(route)
        try:
            async with asyncio.timeout_at(request.deadline):
                for registration in candidates:
                    if not _eligible(registration, request):
                        continue
                    if not registration.profile.supports_streaming or not isinstance(
                        registration.provider, StreamingModelProvider
                    ):
                        continue
                    for attempt in range(self._retry_policy.attempts):
                        attempt_request = request.remaining_after(
                            _aggregate_usage(attempt_usage)
                        )
                        emitted = False
                        completed = False
                        in_flight = True
                        try:
                            async with closing_stream(
                                registration.provider.stream(attempt_request)
                            ) as events:
                                async for event in events:
                                    # A completion is visible progress too. Close the
                                    # request on its first terminal event, without
                                    # reading or retrying any subsequent generation.
                                    emitted = True
                                    if isinstance(event, ModelStreamCompleted):
                                        completed = True
                                        in_flight = False
                                        attempt_usage.append(event.response.usage)
                                        if route is not None:
                                            route.selected_provider_id = (
                                                registration.provider.provider_id
                                            )
                                        yield ModelStreamCompleted(
                                            replace(
                                                event.response,
                                                usage=_aggregate_usage(attempt_usage),
                                            )
                                        )
                                        return
                                    yield event
                            raise ModelProviderError(
                                ProviderErrorCode.MALFORMED_RESPONSE,
                                "provider stream ended without a canonical completion",
                                provider_id=registration.provider.provider_id,
                                diagnostic=ProviderFailureDiagnostic(
                                    phase=ProviderFailurePhase.STREAM_TERMINAL,
                                    code="canonical_completion_missing",
                                ),
                            )
                        except ModelProviderError as error:
                            in_flight = False
                            last_error = error
                            # Cleanup after completion is not another model attempt.
                            if not completed:
                                attempt_usage.append(error.usage)
                            _require_recoverable_usage(
                                request, error, attempt_usage, terminal=emitted
                            )
                            if not await self._wait_for_retry(
                                request, error, attempt, attempt_usage
                            ):
                                break
        except asyncio.CancelledError as error:
            raise with_cancelled_model_usage(
                error,
                _interruption_usage(error, attempt_usage, in_flight),
            ) from None
        except TimeoutError as error:
            raise ModelProviderError(
                ProviderErrorCode.TIMEOUT,
                "The model request deadline expired.",
                usage=_interruption_usage(error, attempt_usage, in_flight),
            ) from None
        except ModelProviderError:
            raise
        except Exception as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "The model provider stream failed.",
                usage=_interruption_usage(error, attempt_usage, in_flight),
            ) from None
        if last_error is not None:
            raise _aggregate_failure(last_error, attempt_usage)
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "no configured provider can stream this request",
            usage=_aggregate_usage(attempt_usage),
        )

    async def _wait_for_retry(
        self,
        request: ModelRequest,
        error: ModelProviderError,
        attempt: int,
        usage: list[ModelUsage],
    ) -> bool:
        """One retry/backoff policy for both delivery modes, within existing limits."""
        if (
            error.retryability is not ErrorRetryability.TRANSIENT
            or attempt + 1 >= self._retry_policy.attempts
        ):
            return False
        request.remaining_after(_aggregate_usage(usage))
        delay = error.retry_after_seconds
        if delay is None:
            delay = min(
                30.0, self._retry_policy.backoff_seconds * (2**attempt)
            ) * random.uniform(0.75, 1.0)
        # Never shorten a server's minimum wait. A request cannot spend unlimited
        # time waiting or begin a retry with an already exhausted deadline.
        if delay > 60 or (
            request.deadline is not None
            and asyncio.get_running_loop().time() + delay >= request.deadline
        ):
            raise _aggregate_failure(error, usage)
        if delay:
            await self._sleep(delay)
        request.remaining_after(_aggregate_usage(usage))
        return True

    def _run_candidates(
        self,
        route: RunRoute,
    ) -> tuple[ModelProviderRegistration, ...]:
        if not isinstance(route, RunRoute):
            raise TypeError("route must be RunRoute")
        configured = {item.provider.provider_id: item for item in self._candidates}
        if any(item not in configured for item in route.candidate_provider_ids):
            raise ValueError("run route does not belong to this router")
        provider_ids = (
            (route.selected_provider_id,)
            if route.selected_provider_id is not None
            else route.candidate_provider_ids
        )
        return tuple(configured[item] for item in provider_ids)


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
    return provider_supports_request_policy(registration.provider, request)


def autonomous_request_is_admissible(
    provider: object,
    request: ModelRequest,
    *,
    max_estimated_cost_usd: Decimal | None,
) -> bool:
    """Admit a future unattended call only with a bounded, priced route."""

    return (
        isinstance(max_estimated_cost_usd, Decimal)
        and max_estimated_cost_usd.is_finite()
        and max_estimated_cost_usd >= 0
        and provider_supports_request_policy(provider, request)
        and provider_has_complete_pricing(provider, request)
    )


def _require_recoverable_usage(
    request: ModelRequest,
    error: ModelProviderError,
    usage: list[ModelUsage],
    *,
    terminal: bool = False,
) -> None:
    if terminal or (
        (
            request.max_total_tokens is not None
            or request.max_estimated_cost_usd is not None
        )
        and error.usage.cost_estimate.status is not CostEstimateStatus.COMPLETE
    ):
        raise _aggregate_failure(error, usage) from None


def _interruption_usage(
    error: BaseException,
    usage: list[ModelUsage],
    in_flight: bool,
) -> ModelUsage:
    return _aggregate_usage(
        [*usage, interrupted_model_usage(error)] if in_flight else usage
    )


def _aggregate_failure(
    error: ModelProviderError, usage: Iterable[ModelUsage]
) -> ModelProviderError:
    return ModelProviderError(
        error.code,
        str(error),
        provider_id=error.provider_id,
        retry_after_seconds=error.retry_after_seconds,
        usage=_aggregate_usage(usage),
        diagnostic=error.diagnostic,
    )


def _aggregate_usage(items: Iterable[ModelUsage]) -> ModelUsage:
    usage = tuple(items)
    if not usage:
        from .pricing import CostEstimate

        return ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
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
    "RunRoute",
    "autonomous_request_is_admissible",
]
