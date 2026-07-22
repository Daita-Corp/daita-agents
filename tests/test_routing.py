from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
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
