"""Router attempt records and the loop's step events reconcile with run usage."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from daita.evaluation import measure_observer_events
from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
)
from daita.llm.pricing import STALL_USAGE_UNREPORTED, CostEstimate
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.loop import LoopExitKind, RunInput
from daita.observation import AgentEvent, AgentEventKind
from tests.llm._routing_support import lookup_response, routed_loop, scripted_attempts
from tests.support.loop import NOW


def _measured(tokens: int, cost: str = "0.01", *, cached: int = 0) -> ModelUsage:
    return ModelUsage(
        input_tokens=tokens,
        output_tokens=2,
        cache_read_tokens=cached,
        cost_estimate=CostEstimate.complete(Decimal(cost)),
    )


def _answer(usage: ModelUsage) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text="done", usage=usage)


def _timeout(tokens: int = 30) -> ModelProviderError:
    return ModelProviderError(ProviderErrorCode.TIMEOUT, usage=_measured(tokens))


def _stalled() -> ModelProviderError:
    return ModelProviderError(
        ProviderErrorCode.TIMEOUT,
        usage=ModelUsage(cost_estimate=CostEstimate.unavailable("usage_unavailable")),
        diagnostic=ProviderFailureDiagnostic(
            phase=ProviderFailurePhase.PROVIDER_BOUNDARY, code="first_progress"
        ),
    )


def _run(name: str) -> RunInput:
    return RunInput(id=name, agent_id="agent-1", message="Question", created_at=NOW)


def _steps(events: list[AgentEvent]) -> list[dict[str, Any]]:
    return [
        {"kind": event.kind, **event.data.to_dict()}
        for event in events
        if event.kind in {AgentEventKind.MODEL_COMPLETED, AgentEventKind.MODEL_FAILED}
    ]


def _attempts(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for step in steps:
        items = step["attempts"]
        assert isinstance(items, list)
        attempts.extend(items)
    return attempts


def _reconcile(steps: list[dict[str, Any]], usage: ModelUsage) -> None:
    """Every counted token comes from exactly one physical attempt record."""

    attempts = _attempts(steps)
    for field in ("input_tokens", "output_tokens", "cache_read_tokens"):
        known = sum(item[field] for item in attempts if isinstance(item[field], int))
        assert known == getattr(usage, field)
        counted = sum(step[field] for step in steps if isinstance(step[field], int))
        assert counted == getattr(usage, field)
    for item in attempts:
        if item["usage_status"] == "unavailable":
            # Unknown is never presented as a measured zero.
            assert item["input_tokens"] is None
            assert item["output_tokens"] is None
            assert item["total_tokens"] is None
            assert isinstance(item["usage_unavailable_reason"], str)
        # Cache tokens are a subset of the attempt's input.
        if isinstance(item["cache_read_tokens"], int):
            assert item["cache_read_tokens"] <= item["input_tokens"]


async def _routed(script, stream: bool, name: str):
    provider = scripted_attempts(stream, script)
    events: list[AgentEvent] = []
    router, store, loop = routed_loop(provider, stream, observer=events.append)
    try:
        result = await loop.run(_run(name))
    finally:
        await router.close()
    return provider, events, result, store


@pytest.mark.parametrize("stream", [False, True])
async def test_one_successful_attempt_is_one_completed_record(stream):
    provider, events, result, _ = await _routed(
        (_answer(_measured(40, cached=25)),), stream, f"success-{stream}"
    )
    steps = _steps(events)
    assert result.kind is LoopExitKind.COMPLETED
    assert [step["kind"] for step in steps] == [AgentEventKind.MODEL_COMPLETED]
    (attempt,) = steps[0]["attempts"]
    assert attempt["outcome"] == "completed"
    assert attempt["retried"] is False
    assert attempt["candidate_attempt_number"] == attempt["total_attempt_number"] == 1
    assert attempt["provider_id"] == provider.provider_id
    assert isinstance(attempt["duration_ms"], int) and attempt["duration_ms"] >= 0
    assert steps[0]["usage_status"] == "reported"
    assert steps[0]["cache_read_tokens"] == 25
    assert steps[0]["cost_status"] == "complete"
    assert steps[0]["cost_amount_usd"] == "0.01"
    assert steps[0]["remaining_tokens_before"] == provider.requests[0].max_total_tokens
    _reconcile(steps, result.usage)


@pytest.mark.parametrize("stream", [False, True])
async def test_retry_then_success_records_both_attempts_in_one_step(stream):
    provider, events, result, _ = await _routed(
        (_timeout(30), lookup_response(_measured(50)), _answer(_measured(70))),
        stream,
        f"retry-success-{stream}",
    )
    steps = _steps(events)
    assert result.kind is LoopExitKind.COMPLETED
    assert len(provider.requests) == 3
    assert [step["kind"] for step in steps] == [AgentEventKind.MODEL_COMPLETED] * 2
    first, second = steps
    assert first["attempt_count"] == 2
    assert [item["outcome"] for item in first["attempts"]] == ["timeout", "completed"]
    assert [item["retried"] for item in first["attempts"]] == [True, False]
    assert [item["total_attempt_number"] for item in first["attempts"]] == [1, 2]
    assert first["input_tokens"] == 80
    assert second["attempt_count"] == 1
    assert result.usage.input_tokens == 150
    _reconcile(steps, result.usage)


@pytest.mark.parametrize("stream", [False, True])
async def test_retry_then_give_up_is_one_failed_step_that_is_not_a_model_call(stream):
    provider, events, result, _ = await _routed(
        (_timeout(30), _timeout(20)), stream, f"give-up-{stream}"
    )
    steps = _steps(events)
    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "timeout"
    assert len(provider.requests) == 2
    assert [step["kind"] for step in steps] == [AgentEventKind.MODEL_FAILED]
    (failed,) = steps
    assert failed["error_code"] == "timeout"
    assert "has_text" not in failed
    assert [item["outcome"] for item in failed["attempts"]] == ["timeout", "timeout"]
    assert [item["retried"] for item in failed["attempts"]] == [True, False]
    assert failed["input_tokens"] == result.usage.input_tokens == 50
    _reconcile(steps, result.usage)
    # Only completed logical steps are model calls.
    assert measure_observer_events(events).model_calls == 0


@pytest.mark.parametrize("stream", [False, True])
async def test_replayed_stall_names_its_unreported_attempt(stream):
    answer = _answer(
        ModelUsage(
            input_tokens=10,
            output_tokens=2,
            cost_estimate=CostEstimate.unavailable("subscription_billing"),
        )
    )
    provider: MockStreamingModelProvider | MockModelProvider = (
        MockStreamingModelProvider(
            (_stalled(), (ModelTextDelta("done"), ModelStreamCompleted(answer)))
        )
        if stream
        else MockModelProvider((_stalled(), answer))
    )
    events: list[AgentEvent] = []
    router, _, loop = routed_loop(provider, stream, observer=events.append)
    try:
        result = await loop.run(_run(f"stall-{stream}"))
    finally:
        await router.close()

    steps = _steps(events)
    assert result.kind is LoopExitKind.COMPLETED
    assert result.usage.cost_estimate.code == STALL_USAGE_UNREPORTED
    (step,) = steps
    stalled, completed = step["attempts"]
    assert stalled["outcome"] == "stalled"
    assert stalled["retried"] is True
    assert stalled["usage_status"] == "unavailable"
    assert stalled["usage_unavailable_reason"] == STALL_USAGE_UNREPORTED
    assert completed["usage_status"] == "reported"
    assert completed["input_tokens"] == 10
    # Subscription routes have tokens but never a price, and never a zero price.
    assert completed["cost_status"] == "unavailable"
    assert completed["cost_amount_usd"] is None
    assert completed["cost_code"] == "subscription_billing"
    assert step["usage_status"] == "partial"
    assert step["usage_unavailable_reason"] == STALL_USAGE_UNREPORTED
    assert step["input_tokens"] == 10
    assert step["cost_amount_usd"] is None
    _reconcile(steps, result.usage)


class _FailThenHang:
    """First attempt fails transiently; the retry hangs until cancelled."""

    provider_id = "mock:hang"

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.entered = asyncio.Event()

    def supports_request_policy(self, request) -> bool:
        del request
        return True

    def has_complete_pricing(self, request) -> bool:
        del request
        return True

    async def _attempt(self, request) -> None:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise _timeout(30)
        self.entered.set()
        await asyncio.Event().wait()

    async def generate(self, request) -> ModelResponse:
        await self._attempt(request)
        raise AssertionError("unreachable")

    async def stream(self, request):
        await self._attempt(request)
        yield ModelTextDelta("unreachable")


@pytest.mark.parametrize("stream", [False, True])
async def test_cancellation_records_the_interrupted_attempt_as_unavailable(stream):
    provider = _FailThenHang()
    events: list[AgentEvent] = []
    router, store, loop = routed_loop(provider, stream, observer=events.append)
    run = _run(f"cancel-{stream}")
    try:
        task = asyncio.create_task(loop.run(run))
        await asyncio.wait_for(provider.entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await router.close()

    persisted = await store.result(run.id)
    assert persisted is not None and persisted.kind is LoopExitKind.INTERRUPTED
    (step,) = _steps(events)
    assert step["kind"] is AgentEventKind.MODEL_FAILED
    assert step["error_code"] == "cancelled"
    first, interrupted = step["attempts"]
    assert first["outcome"] == "timeout" and first["retried"] is True
    assert interrupted["outcome"] == "cancelled"
    assert interrupted["usage_status"] == "unavailable"
    assert interrupted["usage_unavailable_reason"] == "model_attempt_interrupted"
    assert step["usage_status"] == "partial"
    _reconcile([step], persisted.usage)
    assert persisted.usage.input_tokens == 30


async def test_records_are_copies_that_never_change_accounting():
    """The same script with and without an observer settles identically."""

    def script():
        return (_timeout(30), lookup_response(_measured(50)), _answer(_measured(70)))

    _, _, observed, observed_store = await _routed(script(), False, "same")
    provider = scripted_attempts(False, script())
    router, store, loop = routed_loop(provider, False)
    try:
        silent = await loop.run(_run("same"))
    finally:
        await router.close()
    assert observed == silent
    assert (await observed_store.load("same")).messages == (
        await store.load("same")
    ).messages
