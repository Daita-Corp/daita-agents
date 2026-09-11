from dataclasses import replace
from decimal import Decimal

import pytest

from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from daita.llm.factory import create_model_route_provider
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import (
    CostBasis,
    CostEstimate,
    CostEstimateStatus,
)
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.routing import (
    ModelProviderRegistration,
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
    RetryPolicy,
    autonomous_request_is_admissible,
)


def request():
    return ModelRequest(
        messages=(
            CanonicalMessage(role=MessageRole.USER, content=(TextBlock("hello"),)),
        )
    )


def registration(provider, *, streaming=False, allowed_sensitivities=None):
    return ModelProviderRegistration(
        provider=provider,
        profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=10_000,
            max_output_tokens=1_000,
            supports_streaming=streaming,
        ),
        allowed_sensitivities=(
            frozenset({ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL})
            if allowed_sensitivities is None
            else allowed_sensitivities
        ),
    )


async def test_router_close_attempts_all_owned_delegates_and_retains_failure():
    attempts = []

    class Provider(MockModelProvider):
        async def close(self, *, deadline: float | None = None):
            attempts.append(self.provider_id)
            if self.provider_id == "mock:last":
                raise RuntimeError("offline cleanup failure")

    first = Provider((), provider_id="mock:first")
    borrowed = Provider((), provider_id="mock:borrowed")
    last = Provider((), provider_id="mock:last")
    router = ModelRouter(
        (
            replace(registration(first), close_with_router=True),
            registration(borrowed),
            replace(registration(last), close_with_router=True),
        )
    )
    for _ in range(2):
        with pytest.raises(ModelProviderError, match="cleanup_failed"):
            await router.close()
    assert attempts == ["mock:last", "mock:first"]


async def test_closed_router_cannot_call_a_borrowed_provider():
    delegate = MockStreamingModelProvider(
        (
            (
                ModelStreamCompleted(
                    ModelResponse(finish_reason=FinishReason.STOP, text="unused")
                ),
            ),
        )
    )
    router = ModelRouter((registration(delegate, streaming=True),))
    await router.close()
    with pytest.raises(ModelProviderError):
        await router.generate(request())
    with pytest.raises(ModelProviderError):
        await anext(router.stream(request()))
    assert delegate.requests == ()


async def test_route_admission_uses_request_sensitivity_before_provider_io():
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="public"),)
    )
    router = ModelRouter((registration(provider),))

    public = ModelRequest(
        messages=request().messages,
        sensitivity=ModelSensitivity.PUBLIC,
    )
    assert (await router.generate(public)).text == "public"

    confidential = ModelRequest(
        messages=public.messages,
        sensitivity=ModelSensitivity.CONFIDENTIAL,
    )
    try:
        await router.generate(confidential)
    except ModelProviderError as error:
        assert error.code is ProviderErrorCode.INVALID_REQUEST
    else:
        raise AssertionError("an ineligible sensitive route must fail closed")
    assert len(provider.requests) == 1


def test_single_lazy_route_enforces_its_configured_sensitivity_set():
    profile = ModelProfile(
        id="ollama:stage0",
        context_window_tokens=10_000,
        max_output_tokens=1_000,
    )
    provider = create_model_route_provider(
        ModelRoute(
            (
                ModelRouteCandidate(
                    provider_id=profile.id,
                    profile=profile,
                    allowed_sensitivities=frozenset({ModelSensitivity.PUBLIC}),
                ),
            ),
            retry_policy=RetryPolicy(
                max_attempts_per_candidate=1, max_total_attempts=1, backoff_seconds=0
            ),
        )
    )

    assert isinstance(provider, ModelRouter)
    assert provider.retry_policy.max_total_attempts == 1
    assert provider.supports_request_policy(
        ModelRequest(
            messages=request().messages,
            sensitivity=ModelSensitivity.PUBLIC,
        )
    )
    assert not provider.supports_request_policy(
        ModelRequest(
            messages=request().messages,
            sensitivity=ModelSensitivity.CONFIDENTIAL,
        )
    )


def test_autonomous_admission_requires_complete_pricing_and_cost_bound():
    complete = MockModelProvider((), complete_pricing=True)
    unavailable = MockModelProvider((), provider_id="mock:unavailable")
    partially_priced_route = ModelRouter(
        (registration(complete), registration(unavailable))
    )

    assert autonomous_request_is_admissible(
        complete,
        request(),
        max_estimated_cost_usd=Decimal("0.05"),
    )
    assert not autonomous_request_is_admissible(
        complete,
        request(),
        max_estimated_cost_usd=None,
    )
    assert not autonomous_request_is_admissible(
        unavailable,
        request(),
        max_estimated_cost_usd=Decimal("0.05"),
    )
    assert not autonomous_request_is_admissible(
        partially_priced_route,
        request(),
        max_estimated_cost_usd=Decimal("0.05"),
    )

    partial = CostEstimate.partial(
        Decimal("0.01"),
        code="pricing_schedule_incomplete",
    )
    unavailable_estimate = CostEstimate.unavailable()
    assert partial.status is CostEstimateStatus.PARTIAL
    assert partial.amount_usd == Decimal("0.01")
    assert unavailable_estimate.status is CostEstimateStatus.UNAVAILABLE
    assert unavailable_estimate.amount_usd is None


async def test_router_retries_transient_failure_then_returns():
    provider = MockModelProvider(
        (
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            ModelResponse(finish_reason=FinishReason.STOP, text="ok"),
        )
    )
    delays = []

    async def sleep(delay):
        delays.append(delay)

    router = ModelRouter(
        (registration(provider),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0.1),
        sleep=sleep,
    )

    assert (await router.generate(request())).text == "ok"
    assert len(provider.requests) == 2
    assert len(delays) == 1 and 0.075 <= delays[0] <= 0.1


async def test_router_does_not_retry_permanent_failure_and_uses_fallback():
    first = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.INVALID_REQUEST),),
        provider_id="mock:first",
    )
    second = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="fallback"),),
        provider_id="mock:second",
    )
    router = ModelRouter(
        (registration(first), registration(second)),
        retry_policy=RetryPolicy(max_attempts_per_candidate=3, backoff_seconds=0),
    )

    assert (await router.generate(request())).text == "fallback"
    assert len(first.requests) == 1
    assert len(second.requests) == 1


def priced_usage(amount, *, input_tokens=1, output_tokens=1):
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_estimate=CostEstimate.complete(
            Decimal(amount),
            basis=CostBasis.PUBLIC_LIST,
            rate_schedule_id="public:test",
        ),
    )


async def test_router_aggregates_every_complete_attempt():
    provider = MockModelProvider(
        (
            ModelProviderError(
                ProviderErrorCode.TIMEOUT,
                usage=priced_usage("0.01", input_tokens=2, output_tokens=0),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="ok",
                usage=priced_usage("0.02", input_tokens=3, output_tokens=1),
            ),
        )
    )
    router = ModelRouter(
        (registration(provider),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
    )

    response = await router.generate(request())

    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 1
    assert response.request_input_tokens == 3
    assert response.usage.cost_estimate.status is CostEstimateStatus.COMPLETE
    assert response.usage.cost_estimate.amount_usd == Decimal("0.03")


async def test_router_marks_known_success_partial_after_unpriced_failed_attempt():
    first = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.INVALID_REQUEST),),
        provider_id="mock:first",
    )
    second = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="fallback",
                usage=priced_usage("0.02"),
            ),
        ),
        provider_id="mock:second",
    )
    router = ModelRouter(
        (registration(first), registration(second)),
        retry_policy=RetryPolicy(
            max_attempts_per_candidate=1, max_total_attempts=2, backoff_seconds=0
        ),
    )

    response = await router.generate(request())

    assert response.usage.cost_estimate.status is CostEstimateStatus.PARTIAL
    assert response.usage.cost_estimate.amount_usd == Decimal("0.02")
    assert response.usage.cost_estimate.code == "pricing_schedule_unavailable"


async def test_router_preserves_unavailable_estimate_when_all_attempts_fail():
    provider = MockModelProvider(
        (
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            ModelProviderError(ProviderErrorCode.TIMEOUT),
        )
    )
    router = ModelRouter(
        (registration(provider),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
    )

    try:
        await router.generate(request())
    except ModelProviderError as error:
        assert error.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE
        assert error.usage.cost_estimate.amount_usd is None
    else:
        raise AssertionError("router should preserve the final provider failure")


async def test_router_preserves_final_bounded_provider_diagnostic():
    diagnostic = ProviderFailureDiagnostic(
        phase=ProviderFailurePhase.RESPONSE_DECODE,
        code="response_decode_failed",
        terminal_status="completed",
    )
    provider = MockModelProvider(
        (
            ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                provider_id="mock:diagnostic",
                diagnostic=diagnostic,
            ),
        ),
        provider_id="mock:diagnostic",
    )
    router = ModelRouter(
        (registration(provider),),
        retry_policy=RetryPolicy(
            max_attempts_per_candidate=1, max_total_attempts=1, backoff_seconds=0
        ),
    )

    with pytest.raises(ModelProviderError) as caught:
        await router.generate(request())

    assert caught.value.provider_id == "mock:diagnostic"
    assert caught.value.diagnostic is not None
    assert caught.value.diagnostic.attempt is not None
    assert replace(caught.value.diagnostic, attempt=None) == diagnostic
    assert caught.value.diagnostic.attempt["total_attempt_number"] == 1


async def test_router_retries_stream_before_progress_and_aggregates_completion():
    provider = MockStreamingModelProvider(
        (
            ModelProviderError(
                ProviderErrorCode.TIMEOUT,
                usage=priced_usage("0.01", input_tokens=2, output_tokens=0),
            ),
            (
                ModelTextDelta("ok"),
                ModelStreamCompleted(
                    ModelResponse(
                        finish_reason=FinishReason.STOP,
                        text="ok",
                        usage=priced_usage(
                            "0.02",
                            input_tokens=3,
                            output_tokens=1,
                        ),
                    )
                ),
            ),
        )
    )
    delays = []

    async def sleep(delay):
        delays.append(delay)

    router = ModelRouter(
        (registration(provider, streaming=True),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0.1),
        sleep=sleep,
    )

    events = [event async for event in router.stream(request())]

    assert len(delays) == 1 and 0.075 <= delays[0] <= 0.1
    assert events[0] == ModelTextDelta("ok")
    assert isinstance(events[1], ModelStreamCompleted)
    assert events[1].response.text == "ok"
    assert events[1].response.usage.input_tokens == 5
    assert events[1].response.usage.cost_estimate.amount_usd == Decimal("0.03")
    assert router.model_profile.supports_streaming is True


async def test_router_never_retries_or_falls_back_after_stream_progress():
    first = MockStreamingModelProvider(
        (
            (
                ModelTextDelta("visible"),
                ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE),
            ),
        ),
        provider_id="mock:first-stream",
    )
    second = MockStreamingModelProvider(
        (
            (
                ModelTextDelta("must not appear"),
                ModelStreamCompleted(
                    ModelResponse(finish_reason=FinishReason.STOP, text="fallback")
                ),
            ),
        ),
        provider_id="mock:second-stream",
    )
    router = ModelRouter(
        (
            registration(first, streaming=True),
            registration(second, streaming=True),
        ),
        retry_policy=RetryPolicy(max_attempts_per_candidate=3, backoff_seconds=0),
    )
    events = []

    try:
        async for event in router.stream(request()):
            events.append(event)
    except ModelProviderError as error:
        assert error.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    else:
        raise AssertionError("visible stream failure must remain authoritative")

    assert events == [ModelTextDelta("visible")]
    assert len(first.requests) == 1
    assert second.requests == ()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "per_candidate,total,expected",
    [(2, 3, [2, 1, 0]), (5, 1, [1, 0, 0]), (1, 3, [1, 1, 1])],
)
async def test_global_attempt_ceiling_spans_candidates_without_renewing_logical_deadline(
    streaming, per_candidate, total, expected
):
    from daita.llm._lifecycle import closing_stream

    providers = []
    for index in range(3):
        failures = tuple(
            ModelProviderError(
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
            )
            for _ in range(5)
        )
        kind = MockStreamingModelProvider if streaming else MockModelProvider
        providers.append(kind(failures, provider_id=f"mock:limit-{index}"))
    router = ModelRouter(
        tuple(
            ModelProviderRegistration(item, item.model_profile) for item in providers
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_candidate=per_candidate,
            max_total_attempts=total,
            backoff_seconds=0,
        ),
    )
    with pytest.raises(ModelProviderError) as caught:
        if streaming:
            async with closing_stream(router.stream(request())) as events:
                async for _ in events:
                    pass
        else:
            await router.generate(request())
    assert [len(item.requests) for item in providers] == expected
    requests = [req for item in providers for req in item.requests]
    assert len({req.deadline for req in requests}) == 1
    assert all(
        req.attempt_deadline is not None
        and req.deadline is not None
        and req.attempt_deadline <= req.deadline
        for req in requests
    )
    assert (
        caught.value.diagnostic is not None
        and caught.value.diagnostic.attempt is not None
    )
    assert caught.value.diagnostic.attempt["total_attempt_number"] == total
    await router.close()


async def test_unresolved_lazy_secret_blocks_fallback_and_late_provider_construction(
    monkeypatch,
):
    import asyncio

    from daita.llm import ModelCallPolicy
    from daita.security import SecretReference

    release = asyncio.Event()
    calls = []
    made = []

    class Secrets:
        async def resolve(self, reference):
            calls.append(reference)
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return "offline"

    monkeypatch.setattr(
        "daita.llm.factory.create_llm_provider", lambda *a, **kw: made.append(a)
    )
    profiles = [
        ModelProfile(
            id=f"openai:fixture-{i}", context_window_tokens=1000, max_output_tokens=100
        )
        for i in range(2)
    ]
    router = create_model_route_provider(
        ModelRoute(
            tuple(
                ModelRouteCandidate(
                    p.id, p, secret_reference=SecretReference.environment("OFFLINE_KEY")
                )
                for p in profiles
            )
        ),
        secret_provider=Secrets(),
    )
    policy = ModelCallPolicy(
        max_request_seconds=0.15,
        max_attempt_seconds=0.05,
        first_progress_timeout_seconds=0.03,
        progress_idle_timeout_seconds=0.03,
        cleanup_timeout_seconds=0.03,
    )
    try:
        with pytest.raises(ModelProviderError) as caught:
            await router.generate(replace(request(), call_policy=policy))
        assert caught.value.cleanup_unresolved
        assert caught.value.usage.cost_estimate.status is CostEstimateStatus.COMPLETE
        assert len(calls) == 1 and made == []
        assert isinstance(router, ModelRouter)
        lazy = router.candidates[0].provider
        from typing import Any, cast

        lazy = cast(Any, lazy)
        pending = tuple(lazy._native_owner.tasks)
        release.set()
        await asyncio.gather(*pending)
        assert made == []
        with pytest.raises(ModelProviderError):
            await router.generate(replace(request(), call_policy=policy))
        assert len(calls) == 1
    finally:
        release.set()
        await router.close()
