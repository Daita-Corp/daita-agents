from decimal import Decimal

from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import (
    CostBasis,
    CostEstimate,
    CostEstimateStatus,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy


def request():
    return ModelRequest(
        messages=(
            CanonicalMessage(role=MessageRole.USER, content=(TextBlock("hello"),)),
        )
    )


def registration(provider):
    return ModelProviderRegistration(
        provider=provider,
        profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=10_000,
            max_output_tokens=1_000,
        ),
    )


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
        retry_policy=RetryPolicy(attempts=2, backoff_seconds=0.1),
        sleep=sleep,
    )

    assert (await router.generate(request())).text == "ok"
    assert len(provider.requests) == 2
    assert delays == [0.1]


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
        retry_policy=RetryPolicy(attempts=3, backoff_seconds=0),
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
        retry_policy=RetryPolicy(attempts=2, backoff_seconds=0),
    )

    response = await router.generate(request())

    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 1
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
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
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
        retry_policy=RetryPolicy(attempts=2, backoff_seconds=0),
    )

    try:
        await router.generate(request())
    except ModelProviderError as error:
        assert error.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE
        assert error.usage.cost_estimate.amount_usd is None
    else:
        raise AssertionError("router should preserve the final provider failure")
