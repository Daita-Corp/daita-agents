"""Retry accounting and failure recovery with actual SDKs and offline transports."""

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from daita._json import FrozenJsonObject
from daita.llm._lifecycle import closing_stream
from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
    interrupted_model_usage,
    retry_after_from_headers,
)
from daita.llm.factory import create_model_route_provider
from daita.llm.models import (
    FinishReason,
    ModelCallPolicy,
    ModelProfile,
    ModelResponse,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
)
from daita.llm.pricing import STALL_USAGE_UNREPORTED, CostEstimate
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.routing import (
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
    RetryPolicy,
)
from daita.security import SecretReference, SecretResolutionError
from tests.llm._routing_support import (
    lookup_response,
    measured_usage,
    registration,
    request,
    routed_loop,
    scripted_attempts,
)
from tests.llm._token_counting_support import (
    count_response,
    input_request,
    is_count,
    provider_at,
)


def usage(tokens=30, cost="0.01"):
    return ModelUsage(
        input_tokens=tokens, cost_estimate=CostEstimate.complete(Decimal(cost))
    )


async def invoke(provider, request, stream) -> ModelResponse:
    if not stream:
        return await provider.generate(request)
    completed = None
    async for event in provider.stream(request):
        if isinstance(event, ModelStreamCompleted):
            completed = event.response
    assert completed is not None
    return completed


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("deadline", [False, True])
async def test_backoff_interruption_retains_prior_usage(stream, deadline):
    entered = asyncio.Event()

    async def sleep(delay):
        entered.set()
        await asyncio.Event().wait()

    failed = ModelProviderError(ProviderErrorCode.TIMEOUT, usage=usage())
    provider = (
        MockStreamingModelProvider((failed,))
        if stream
        else MockModelProvider((failed,))
    )
    router = ModelRouter(
        (registration(provider, streaming=stream),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2),
        sleep=sleep,
    )
    bounded = replace(request(), max_total_tokens=100)
    error: BaseException
    try:
        if deadline:
            with pytest.raises(TimeoutError) as caught:
                async with asyncio.timeout(0.05):
                    await invoke(router, bounded, stream)
            error = caught.value
        else:
            task = asyncio.create_task(invoke(router, bounded, stream))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await task
            error = cancelled.value
        assert entered.is_set()
        assert len(provider.requests) == 1
        observed = interrupted_model_usage(error)
        assert observed.total_tokens == 30
        assert observed.cost_estimate.amount_usd == Decimal("0.01")
        assert observed.cost_estimate.status.value == "complete"
    finally:
        await router.close()


def stalled(code: str = "first_progress") -> ModelProviderError:
    # A progress deadline before any canonical output: the provider reported
    # no usage, exactly as a silent stream ends.
    return ModelProviderError(
        ProviderErrorCode.TIMEOUT,
        usage=ModelUsage(cost_estimate=CostEstimate.unavailable("usage_unavailable")),
        diagnostic=ProviderFailureDiagnostic(
            phase=ProviderFailurePhase.PROVIDER_BOUNDARY, code=code
        ),
    )


def stall_then(stream: bool, failure: ModelProviderError):
    answer = ModelResponse(
        finish_reason=FinishReason.STOP,
        text="recovered",
        usage=ModelUsage(
            input_tokens=10,
            output_tokens=2,
            cost_estimate=CostEstimate.unavailable("subscription_billing"),
        ),
    )
    if stream:
        return MockStreamingModelProvider(
            (failure, (ModelTextDelta("recovered"), ModelStreamCompleted(answer)))
        )
    return MockModelProvider((failure, answer))


async def no_wait(_delay: float) -> None:
    return None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("code", ["first_progress", "idle_progress"])
async def test_silent_stall_replays_under_a_token_limit(stream, code):
    provider = stall_then(stream, stalled(code))
    router = ModelRouter((registration(provider, streaming=stream),), sleep=no_wait)
    try:
        response = await invoke(
            router, replace(request(), max_total_tokens=1_000), stream
        )
        assert response.text == "recovered"
        assert len(provider.requests) == 2
        assert response.usage.total_tokens == 12
        # The stalled attempt's unreported consumption stays named.
        assert response.usage.cost_estimate.code == STALL_USAGE_UNREPORTED
    finally:
        await router.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_silent_stall_never_replays_under_a_dollar_cap(stream):
    provider = stall_then(stream, stalled())
    router = ModelRouter((registration(provider, streaming=stream),), sleep=no_wait)
    capped = replace(
        request(), max_total_tokens=1_000, max_estimated_cost_usd=Decimal("1")
    )
    try:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(router, capped, stream)
        assert caught.value.code is ProviderErrorCode.TIMEOUT
        assert len(provider.requests) == 1
    finally:
        await router.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_unmeasured_deadline_timeout_is_not_a_replayable_stall(stream):
    provider = stall_then(stream, stalled("attempt_deadline"))
    router = ModelRouter((registration(provider, streaming=stream),), sleep=no_wait)
    try:
        with pytest.raises(ModelProviderError):
            await invoke(router, replace(request(), max_total_tokens=1_000), stream)
        assert len(provider.requests) == 1
    finally:
        await router.close()


def unreported(code: ProviderErrorCode, transport: str | None = None):
    # A transient failure before any output whose consumption is unknown.
    return ModelProviderError(
        code,
        usage=ModelUsage(cost_estimate=CostEstimate.unavailable("usage_unavailable")),
        diagnostic=(
            None
            if transport is None
            else ProviderFailureDiagnostic(
                phase=ProviderFailurePhase.PROVIDER_BOUNDARY,
                code=code.value,
                attempt=FrozenJsonObject.from_mapping(
                    {"transport_error_kind": transport}
                ),
            )
        ),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "code, transport",
    [
        (ProviderErrorCode.RATE_LIMIT_ERROR, None),
        (ProviderErrorCode.PROVIDER_UNAVAILABLE, None),
        (ProviderErrorCode.PROVIDER_UNAVAILABLE, "remote_protocol_error"),
        (ProviderErrorCode.TIMEOUT, "connect_timeout"),
    ],
    ids=["rate_limited", "unavailable", "transport_drop", "connect_timeout"],
)
async def test_failure_before_output_replays_under_a_token_limit(
    stream, code, transport
):
    failure = unreported(code, transport)
    provider = stall_then(stream, failure)
    router = ModelRouter((registration(provider, streaming=stream),), sleep=no_wait)
    try:
        response = await invoke(
            router, replace(request(), max_total_tokens=1_000), stream
        )
        assert response.text == "recovered"
        assert len(provider.requests) == 2
        assert response.attempts is not None
        assert [item.outcome for item in response.attempts] == [
            failure.code.value,
            "completed",
        ]
        assert [item.retried for item in response.attempts] == [True, False]
    finally:
        await router.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("proven_zero", [False, True])
async def test_dollar_cap_replays_only_a_failure_with_proven_zero_usage(
    stream, proven_zero
):
    failure = unreported(ProviderErrorCode.PROVIDER_UNAVAILABLE)
    if proven_zero:
        failure.usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
    provider = stall_then(stream, failure)
    router = ModelRouter((registration(provider, streaming=stream),), sleep=no_wait)
    capped = replace(
        request(), max_total_tokens=1_000, max_estimated_cost_usd=Decimal("1")
    )
    try:
        if proven_zero:
            assert (await invoke(router, capped, stream)).text == "recovered"
            assert len(provider.requests) == 2
        else:
            with pytest.raises(ModelProviderError) as caught:
                await invoke(router, capped, stream)
            assert caught.value.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
            assert caught.value.usage.cost_estimate.status.value == "unavailable"
            assert caught.value.attempts is not None
            assert [item.outcome for item in caught.value.attempts] == [
                "provider_unavailable"
            ]
            assert len(provider.requests) == 1
    finally:
        await router.close()


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize(
    "status, expected",
    [
        (503, ProviderErrorCode.PROVIDER_UNAVAILABLE),
        (429, ProviderErrorCode.RATE_LIMIT_ERROR),
        (401, ProviderErrorCode.AUTHENTICATION_ERROR),
    ],
)
async def test_count_failure_preserves_actionable_classification(
    kind, status, expected
):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        return httpx.Response(
            status,
            headers={"retry-after": "7"},
            json={"error": {"message": "offline count failure"}},
        )

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, input_request(), False)
    assert len(paths) == 1
    assert caught.value.usage.total_tokens == 0
    assert caught.value.usage.cost_estimate.status.value == "complete"
    assert caught.value.code is expected


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("status", [429, 503])
async def test_provider_retry_after_reaches_router(kind, status):
    paths, delays = [], []

    def respond(req):
        paths.append(req.url.path)
        return httpx.Response(
            status,
            headers={"retry-after": "7"},
            json={"error": {"message": "offline rate limit"}},
        )

    async def sleep(delay):
        delays.append(delay)

    async with provider_at(kind, respond) as provider:
        router = ModelRouter((registration(provider),), sleep=sleep)
        try:
            with pytest.raises(ModelProviderError):
                await router.generate(
                    request()
                )  # Unbounded isolates header forwarding.
        finally:
            await router.close()
    assert len(paths) == 2
    assert delays == [7.0]


async def test_completion_cleanup_failure_cannot_restart_generation():
    class Provider(MockStreamingModelProvider):
        attempts = 0

        async def stream(self, request):
            self.attempts += 1
            try:
                yield ModelStreamCompleted(
                    ModelResponse(
                        finish_reason=FinishReason.STOP,
                        text="done",
                        usage=usage(),
                    )
                )
            finally:
                if self.attempts == 1:
                    raise ModelProviderError(
                        ProviderErrorCode.TIMEOUT, usage=usage(0, "0")
                    )

    provider = Provider(())
    router = ModelRouter(
        (registration(provider, streaming=True),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
    )
    events = []
    try:
        try:
            async for event in router.stream(replace(request(), max_total_tokens=100)):
                events.append(event)
        except ModelProviderError:
            pass  # Cleanup may fail; it cannot generate another response.
        assert provider.attempts == 1
        assert events == []
    finally:
        await router.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_known_local_secret_failure_can_retry_before_generation(
    monkeypatch, stream
):
    class Secrets:
        calls = 0

        async def resolve(self, reference):
            self.calls += 1
            if self.calls == 1:
                raise SecretResolutionError(
                    "secret_provider_unavailable", "Offline temporary failure"
                )
            return "offline-placeholder"

    response = ModelResponse(
        finish_reason=FinishReason.STOP, text="done", usage=usage(5, "0.001")
    )
    delegate: MockModelProvider | MockStreamingModelProvider = (
        MockStreamingModelProvider(
            ((ModelStreamCompleted(response),),), provider_id="openai:fixture"
        )
    )
    if not stream:
        delegate = MockModelProvider((response,), provider_id="openai:fixture")
    monkeypatch.setattr(
        "daita.llm.factory.create_llm_provider", lambda *args, **kwargs: delegate
    )
    secrets = Secrets()
    profile = ModelProfile(
        id=delegate.provider_id,
        context_window_tokens=10000,
        max_output_tokens=1000,
        supports_streaming=stream,
    )
    router = create_model_route_provider(
        ModelRoute(
            (
                ModelRouteCandidate(
                    provider_id=profile.id,
                    profile=profile,
                    secret_reference=SecretReference.environment("OFFLINE_KEY"),
                ),
            ),
            retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
        ),
        secret_provider=secrets,
    )
    try:
        response = await invoke(
            router, replace(request(), max_total_tokens=100), stream
        )
        assert response.text == "done"
        assert secrets.calls == 2
        assert len(delegate.requests) == 1
    finally:
        await router.close()


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
async def test_temporary_count_outage_retries_only_admission_before_generation(kind):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        if len(paths) == 1:
            return httpx.Response(
                503, json={"error": {"message": "offline count outage"}}
            )
        if is_count(req.url.path):
            return httpx.Response(200, json=count_response(kind, 6000))
        raise AssertionError("The remaining 5000 tokens cannot admit generation.")

    async with provider_at(kind, respond) as provider:
        router = ModelRouter(
            (
                replace(
                    registration(provider),
                    profile=replace(
                        registration(provider).profile, supports_tools=True
                    ),
                ),
            ),
            retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
        )
        try:
            with pytest.raises(ModelProviderError) as caught:
                await router.generate(
                    replace(input_request(remaining=5000), response_schema=None)
                )
            assert len(paths) == 2
            assert all(is_count(path) for path in paths)
            assert caught.value.code is ProviderErrorCode.TOKEN_BUDGET_INSUFFICIENT
            assert caught.value.usage.total_tokens == 0
        finally:
            await router.close()


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
async def test_count_response_close_failure_blocks_retry(kind):
    paths = []
    consumed = []
    closes = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(count_response(kind, 5000)).encode()
            consumed.append(1)

        async def aclose(self):
            closes.append(1)
            raise RuntimeError("synthetic native close failure")

    def respond(req):
        paths.append(req.url.path)
        return httpx.Response(
            200, stream=Body(), headers={"content-type": "application/json"}
        )

    async with provider_at(kind, respond) as provider:
        entry = registration(provider)
        router = ModelRouter(
            (replace(entry, profile=replace(entry.profile, supports_tools=True)),),
            retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
        )
        try:
            with pytest.raises(ModelProviderError) as caught:
                await router.generate(replace(input_request(), response_schema=None))
            assert caught.value.code is ProviderErrorCode.CLEANUP_FAILED
            assert caught.value.cleanup_unresolved
            assert caught.value.usage.total_tokens == 0
            assert caught.value.usage.cost_estimate.status.value == "complete"
            assert provider._native_owner.poisoned
            assert len(paths) == 1 and is_count(paths[0])
            assert consumed == [1]
            assert closes == [1]
        finally:
            await router.close()


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
async def test_count_timeout_releases_request_and_retries_inside_same_deadline(
    kind, stream
):
    paths = []
    released = asyncio.Event()

    async def respond(req):
        paths.append(req.url.path)
        if len(paths) == 1:
            try:
                await asyncio.Event().wait()
            finally:
                released.set()
        assert released.is_set()
        assert is_count(req.url.path)
        return httpx.Response(200, json=count_response(kind, 6000))

    async with provider_at(kind, respond) as provider:
        entry = registration(provider, streaming=stream)
        router = ModelRouter(
            (replace(entry, profile=replace(entry.profile, supports_tools=True)),),
            retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
        )
        bounded = replace(
            input_request(remaining=5000),
            response_schema=None,
            deadline=asyncio.get_running_loop().time() + 1,
            call_policy=ModelCallPolicy(input_count_timeout_seconds=0.02),
        )
        try:
            with pytest.raises(ModelProviderError) as caught:
                await invoke(router, bounded, stream)
            assert caught.value.code is ProviderErrorCode.TOKEN_BUDGET_INSUFFICIENT
            assert caught.value.usage.total_tokens == 0
            assert caught.value.usage.cost_estimate.status.value == "complete"
            assert len(paths) == 2 and all(is_count(path) for path in paths)
        finally:
            await router.close()


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
async def test_expired_direct_request_never_dispatches(kind, stream):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        raise AssertionError("Expired request reached HTTP.")

    async with provider_at(kind, respond) as provider:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(provider, replace(input_request(), deadline=0), stream)
    assert caught.value.code is ProviderErrorCode.TIMEOUT
    assert caught.value.usage.cost_estimate.status.value == "complete"
    assert paths == []


@pytest.mark.parametrize("stream", [False, True])
async def test_request_deadline_during_backoff_preserves_usage(stream):
    entered = asyncio.Event()

    async def sleep(delay):
        entered.set()
        await asyncio.Event().wait()

    failure = ModelProviderError(ProviderErrorCode.TIMEOUT, usage=usage())
    provider = (
        MockStreamingModelProvider((failure,))
        if stream
        else MockModelProvider((failure,))
    )
    router = ModelRouter(
        (registration(provider, streaming=stream),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0.001),
        sleep=sleep,
    )
    try:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(
                router,
                replace(
                    request(),
                    max_total_tokens=100,
                    deadline=asyncio.get_running_loop().time() + 0.05,
                ),
                stream,
            )
        assert entered.is_set()
        assert caught.value.code is ProviderErrorCode.TIMEOUT
        assert caught.value.usage.total_tokens == 30
        assert caught.value.usage.cost_estimate.amount_usd == Decimal("0.01")
        assert caught.value.usage.cost_estimate.status.value == "complete"
        assert len(provider.requests) == 1
    finally:
        await router.close()


@pytest.mark.parametrize("completed", [False, True])
async def test_consumer_cancellation_between_stream_events_retains_attempt_usage(
    completed,
):
    entered = asyncio.Event()
    terminal = ModelStreamCompleted(
        ModelResponse(
            finish_reason=FinishReason.STOP, text="done", usage=usage(5, "0.001")
        )
    )
    first_event = terminal if completed else ModelTextDelta("partial")
    provider = MockStreamingModelProvider(
        (
            ModelProviderError(ProviderErrorCode.TIMEOUT, usage=usage()),
            (first_event,),
        )
    )
    router = ModelRouter(
        (registration(provider, streaming=True),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
    )

    async def consume():
        async with closing_stream(
            router.stream(replace(request(), max_total_tokens=100))
        ) as events:
            async for event in events:
                entered.set()
                await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        measured = interrupted_model_usage(caught.value)
        assert measured.total_tokens == (35 if completed else 30)
        assert measured.cost_estimate.amount_usd == Decimal(
            "0.011" if completed else "0.01"
        )
        assert measured.cost_estimate.status.value == (
            "complete" if completed else "partial"
        )
        assert len(provider.requests) == 2
    finally:
        await router.close()


@pytest.mark.parametrize("delay, deadline", [(7, True), (61, False)])
async def test_retry_wait_that_cannot_fit_is_never_shortened(delay, deadline):
    provider = MockModelProvider(
        (
            ModelProviderError(
                ProviderErrorCode.RATE_LIMIT_ERROR,
                retry_after_seconds=delay,
                usage=usage(0, "0"),
            ),
        )
    )
    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    router = ModelRouter((registration(provider),), sleep=sleep)
    try:
        with pytest.raises(ModelProviderError) as caught:
            await router.generate(
                replace(
                    request(),
                    max_total_tokens=100,
                    deadline=(
                        asyncio.get_running_loop().time() + 0.05 if deadline else None
                    ),
                )
            )
        assert caught.value.code is ProviderErrorCode.RATE_LIMIT_ERROR
        assert len(provider.requests) == 1
        assert delays == []
    finally:
        await router.close()


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"retry-after": "7"}, 7),
        ({"retry-after-ms": "250"}, 0.25),
        ({"retry-after": "nan"}, None),
        ({"retry-after": "-1"}, None),
        ({"retry-after": "x" * 129}, None),
        ({"retry-after-ms": "bad", "retry-after": "2"}, 2),
    ],
)
def test_retry_metadata_is_bounded(headers, expected):
    assert retry_after_from_headers(headers) == expected


@pytest.mark.parametrize("stream", [False, True])
async def test_single_attempt_lazy_resolution_uses_request_deadline(stream):
    class Secrets:
        calls = 0

        async def resolve(self, reference: SecretReference) -> str:
            del reference
            self.calls += 1
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    secrets = Secrets()
    profile = ModelProfile(
        id="openai:fixture",
        context_window_tokens=10000,
        max_output_tokens=100,
        supports_streaming=stream,
    )
    route = ModelRoute(
        (
            ModelRouteCandidate(
                profile.id,
                profile,
                secret_reference=SecretReference("env", "OFFLINE_KEY"),
            ),
        ),
        retry_policy=RetryPolicy(max_attempts_per_candidate=1, max_total_attempts=1),
    )
    provider = create_model_route_provider(route, secret_provider=secrets)
    try:
        with pytest.raises(ModelProviderError) as caught:
            await invoke(
                provider,
                replace(
                    request(),
                    max_total_tokens=100,
                    deadline=asyncio.get_running_loop().time() + 0.02,
                ),
                stream,
            )
        assert caught.value.code is ProviderErrorCode.TIMEOUT
        assert caught.value.usage.total_tokens == 0
        assert caught.value.usage.cost_estimate.status.value == "complete"
        assert secrets.calls == 1
    finally:
        await provider.close()


@pytest.mark.parametrize("mode", ["cancel", "cleanup_failure"])
async def test_loop_retains_completion_usage_through_stream_shutdown(mode):
    from daita.loop import AgentLoop, InMemoryTranscriptStore, RunInput
    from tests.support.loop import NOW, ScriptedTools, TranscriptContext

    class Provider(MockStreamingModelProvider):
        async def stream(self, request):
            yield ModelStreamCompleted(
                ModelResponse(
                    finish_reason=FinishReason.STOP, text="done", usage=usage()
                )
            )
            if mode == "cancel":
                current = asyncio.current_task()
                assert current is not None
                current.cancel()
                await asyncio.sleep(0)
            else:
                raise ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE)

    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=Provider(()),
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        transcripts=store,
        clock=lambda: NOW,
        stream_model_calls=True,
    )
    run = RunInput(
        id="completion-shutdown", agent_id="agent-1", message="Question", created_at=NOW
    )
    task = asyncio.create_task(loop.run(run))
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    result = await store.result(run.id)
    assert result is not None
    assert result.usage.total_tokens == 30
    assert result.usage.cost_estimate.amount_usd == Decimal(".01")
    assert result.usage.cost_estimate.status.value == "complete"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("outcome", ["recovered", "exhausted"])
async def test_loop_usage_is_the_sum_of_measured_retry_attempts(stream, outcome):
    from daita.loop import LoopExitKind, RunInput
    from tests.support.loop import NOW

    first = measured_usage(30, 0, "0.010")
    second = measured_usage(40, 5, "0.020")
    third = measured_usage(50, 0, "0.030")
    fourth = measured_usage(60, 7, "0.040")
    attempts: tuple[ModelResponse | ModelProviderError, ...]
    before_second_step: tuple[ModelUsage, ...]
    if outcome == "recovered":
        # Step 1 fails once and recovers; step 2 answers on its first attempt.
        attempts = (
            ModelProviderError(ProviderErrorCode.TIMEOUT, usage=first),
            lookup_response(second),
            ModelResponse(finish_reason=FinishReason.STOP, text="done", usage=third),
        )
        before_second_step = (first, second)
    else:
        # Step 1 answers; step 2 fails twice and the router gives up.
        attempts = (
            lookup_response(second),
            ModelProviderError(ProviderErrorCode.TIMEOUT, usage=first),
            ModelProviderError(ProviderErrorCode.TIMEOUT, usage=fourth),
        )
        before_second_step = (second,)
    measured = tuple(item.usage for item in attempts)
    provider = scripted_attempts(stream, attempts)
    router, store, loop = routed_loop(provider, stream)
    run = RunInput(
        id=f"retry-accounting-{outcome}-{stream}",
        agent_id="agent-1",
        message="Question",
        created_at=NOW,
    )
    try:
        result = await loop.run(run)
    finally:
        await router.close()

    assert len(provider.requests) == len(attempts)
    assert result.kind is (
        LoopExitKind.COMPLETED if outcome == "recovered" else LoopExitKind.FAILED
    )
    assert result.usage.input_tokens == sum(item.input_tokens for item in measured)
    assert result.usage.output_tokens == sum(item.output_tokens for item in measured)
    assert result.usage.cost_estimate.status.value == "complete"
    assert result.usage.cost_estimate.amount_usd == sum(
        (item.cost_estimate.amount_usd or Decimal(0) for item in measured),
        Decimal(0),
    )
    assert await store.result(run.id) == result
    # The second logical step is admitted against exactly the usage of the first.
    initial_allowance = provider.requests[0].max_total_tokens
    assert initial_allowance is not None
    assert provider.requests[len(before_second_step)].max_total_tokens == (
        initial_allowance - sum(item.total_tokens for item in before_second_step)
    )


class _InterruptedAfterMeasurement(MockStreamingModelProvider):
    """Fail one attempt, then measure a second attempt before interruption."""

    def __init__(self) -> None:
        super().__init__((), complete_pricing=True)
        self.entered = asyncio.Event()

    async def _attempt(self, request):
        from daita.llm.errors import with_cancelled_model_usage

        self._requests.append(request)
        if len(self._requests) == 1:
            raise ModelProviderError(
                ProviderErrorCode.TIMEOUT, usage=measured_usage(30, 0, "0.010")
            )
        try:
            self.entered.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            raise with_cancelled_model_usage(error, measured_usage(7, 0, "0.002"))
        raise AssertionError("the in-flight attempt must be interrupted")

    async def generate(self, request):
        await self._attempt(request)
        raise AssertionError("unreachable")

    async def stream(self, request):
        await self._attempt(request)
        yield  # pragma: no cover


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
async def test_interrupted_in_flight_attempt_is_counted_once(stream, interruption):
    from daita.loop import AgentLoop, InMemoryTranscriptStore, LoopLimits, RunInput
    from tests.support.loop import NOW, ScriptedTools, TranscriptContext

    provider = _InterruptedAfterMeasurement()
    router = ModelRouter(
        (registration(provider, streaming=stream, tools=True),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, backoff_seconds=0),
    )
    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=router,
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        transcripts=store,
        clock=lambda: NOW,
        stream_model_calls=stream,
        limits=LoopLimits(
            max_wall_time_seconds=0.2 if interruption == "deadline" else 300.0
        ),
    )
    run = RunInput(
        id=f"interrupted-{interruption}-{stream}",
        agent_id="agent-1",
        message="Question",
        created_at=NOW,
    )
    task = asyncio.create_task(loop.run(run))
    try:
        await asyncio.wait_for(provider.entered.wait(), 1)
        if interruption == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, 2)
    finally:
        await router.close()

    result = await store.result(run.id)
    assert result is not None
    assert len(provider.requests) == 2
    # 30 tokens from the failed attempt plus 7 measured before interruption,
    # counted by the router once and adopted by the loop without re-adding.
    assert result.usage.total_tokens == 37
    assert result.usage.cost_estimate.amount_usd == Decimal("0.012")
    assert result.usage.cost_estimate.status.value == "complete"


def test_retry_after_http_date():
    from datetime import UTC, datetime

    now = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
    assert (
        retry_after_from_headers(
            httpx.Headers({"Retry-After": "Tue, 08 Sep 2026 12:00:07 GMT"}), now=now
        )
        == 7
    )
    assert (
        retry_after_from_headers(
            {"retry-after": "Tue, 08 Sep 2026 11:59:00 GMT"}, now=now
        )
        == 0
    )


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("content", [b"{invalid", b"[]"])
async def test_malformed_native_count_response_is_terminal(kind, content):
    paths = []

    def respond(req):
        paths.append(req.url.path)
        return httpx.Response(
            200, content=content, headers={"content-type": "application/json"}
        )

    async with provider_at(kind, respond) as provider:
        entry = registration(provider)
        router = ModelRouter(
            (replace(entry, profile=replace(entry.profile, supports_tools=True)),),
            retry_policy=RetryPolicy(max_attempts_per_candidate=2),
        )
        try:
            with pytest.raises(ModelProviderError) as caught:
                await router.generate(replace(input_request(), response_schema=None))
            assert caught.value.code is ProviderErrorCode.TOKEN_COUNT_UNAVAILABLE
            assert caught.value.usage.cost_estimate.status.value == "complete"
            assert len(paths) == 1 and is_count(paths[0])
            assert not provider._native_owner.poisoned
        finally:
            await router.close()
