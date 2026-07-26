from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import (
    CostBasis,
    CostComponent,
    CostEstimate,
    CostEstimateStatus,
    aggregate_cost_estimates,
    format_cost_estimate,
)
from daita.llm.providers.anthropic import (
    AnthropicProvider,
    _decode_usage as decode_anthropic_usage,
)
from daita.llm.providers.gemini import (
    GeminiProvider,
    _decode_usage as decode_gemini_usage,
)
from daita.llm.providers.grok import GrokProvider
from daita.llm.providers.mock import MockModelProvider
from daita.llm.providers.ollama import OllamaProvider
from daita.llm.providers.openai import (
    OpenAIProvider,
    _decode_usage as decode_openai_usage,
)
from daita.llm.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _decode_usage as decode_compatible_usage,
)
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.storage.sqlite import SQLiteStateStore

NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def _complete(
    amount: str,
    *,
    schedule: str = "public:test:2026-07",
) -> CostEstimate:
    decimal = Decimal(amount)
    return CostEstimate.complete(
        decimal,
        basis=CostBasis.PUBLIC_LIST,
        rate_schedule_id=schedule,
        components=(CostComponent("model", decimal),),
    )


def _request() -> ModelRequest:
    return ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        )
    )


def test_cost_records_are_immutable_bounded_and_status_consistent():
    estimate = _complete("0.1250")
    assert estimate.amount_usd == Decimal("0.1250")
    assert estimate.status is CostEstimateStatus.COMPLETE
    assert estimate.basis is CostBasis.PUBLIC_LIST
    assert estimate.rate_schedule_id == "public:test:2026-07"
    assert estimate.components == (CostComponent("model", Decimal("0.1250")),)
    assert estimate.code is None
    with pytest.raises(FrozenInstanceError):
        estimate.amount_usd = Decimal("10")  # type: ignore[misc]

    invalid = (
        lambda: CostEstimate(
            amount_usd=None,
            status=CostEstimateStatus.COMPLETE,
        ),
        lambda: CostEstimate(
            amount_usd=Decimal("1"),
            status=CostEstimateStatus.UNAVAILABLE,
            code="unknown",
        ),
        lambda: CostEstimate.partial(Decimal("1"), code=""),
        lambda: CostEstimate.unavailable("x" * 129),
        lambda: CostEstimate.complete(
            Decimal("1"),
            components=tuple(
                CostComponent(str(index), Decimal("0")) for index in range(65)
            ),
        ),
        lambda: CostEstimate.complete(Decimal("-0.01")),
        lambda: CostEstimate.complete(Decimal("NaN")),
    )
    for construct in invalid:
        with pytest.raises((TypeError, ValueError)):
            construct()


def test_cost_aggregation_preserves_complete_partial_unavailable_and_zero():
    complete = aggregate_cost_estimates((_complete("0.10"), _complete("0.20")))
    assert complete.status is CostEstimateStatus.COMPLETE
    assert complete.amount_usd == Decimal("0.30")
    assert complete.rate_schedule_id == "public:test:2026-07"
    assert len(complete.components) == 2

    mixed_schedules = aggregate_cost_estimates(
        (
            _complete("0.10", schedule="public:first"),
            _complete("0.20", schedule="public:second"),
        )
    )
    assert mixed_schedules.status is CostEstimateStatus.COMPLETE
    assert mixed_schedules.rate_schedule_id is None
    assert {component.rate_schedule_id for component in mixed_schedules.components} == {
        "public:first",
        "public:second",
    }

    partial = aggregate_cost_estimates(
        (_complete("0.10"), CostEstimate.unavailable("unpriced_attempt"))
    )
    assert partial.status is CostEstimateStatus.PARTIAL
    assert partial.amount_usd == Decimal("0.10")
    assert partial.code == "unpriced_attempt"
    assert partial.rate_schedule_id == "public:test:2026-07"

    unavailable = aggregate_cost_estimates(
        (
            CostEstimate.unavailable("unpriced_attempt"),
            CostEstimate.unavailable("unpriced_attempt"),
        )
    )
    assert unavailable == CostEstimate.unavailable("unpriced_attempt")

    zero = aggregate_cost_estimates((_complete("0"), _complete("0")))
    assert zero.status is CostEstimateStatus.COMPLETE
    assert zero.amount_usd == Decimal("0")
    assert format_cost_estimate(zero) == ("$0 explicit estimate at public list rates")


def test_cost_rendering_distinguishes_every_semantic_state():
    assert format_cost_estimate(_complete("0.1250")) == (
        "$0.125 estimated at public list rates"
    )
    assert (
        format_cost_estimate(
            CostEstimate.partial(
                Decimal("0.12"),
                code="unpriced_attempt",
                basis=CostBasis.PUBLIC_LIST,
                rate_schedule_id="public:test:2026-07",
            )
        )
        == "≥$0.12 estimated; some attempts were unpriced"
    )
    assert format_cost_estimate(CostEstimate.unavailable()) == "cost unavailable"
    assert format_cost_estimate(_complete("0")) == (
        "$0 explicit estimate at public list rates"
    )


def test_current_providers_have_no_admitted_complete_pricing():
    request = _request()
    providers = (
        OpenAIProvider("test-model"),
        AnthropicProvider("test-model"),
        GeminiProvider("test-model"),
        OpenAICompatibleProvider(
            "test-model",
            provider="custom",
            base_url="https://example.com/v1",
        ),
        GrokProvider("test-model"),
        OllamaProvider("test-model"),
        MockModelProvider(()),
    )
    assert all(
        provider.has_complete_pricing(request) is False for provider in providers
    )

    decoded = (
        decode_openai_usage({"input_tokens": 1, "output_tokens": 2}),
        decode_anthropic_usage({"input_tokens": 1, "output_tokens": 2}),
        decode_gemini_usage({"prompt_token_count": 1, "candidates_token_count": 2}),
        decode_compatible_usage({"prompt_tokens": 1, "completion_tokens": 2}),
    )
    assert all(
        usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE
        and usage.cost_estimate.amount_usd is None
        for usage in decoded
    )


async def test_sqlite_persists_the_exact_estimate_without_repricing(tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.sqlite3")
    run = RunInput(
        id="run-priced",
        agent_id="agent-priced",
        message="question",
        created_at=NOW,
        conversation_id="conversation-priced",
    )
    original = CostEstimate.partial(
        Decimal("0.1200"),
        code="unpriced_retry",
        basis=CostBasis.CONFIGURED_CONTRACT,
        rate_schedule_id="contract:2026-q3",
        components=(CostComponent("known-attempt", Decimal("0.1200")),),
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id="conversation-priced",
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=NOW,
        final_text="answer",
        steps=1,
        usage=ModelUsage(
            input_tokens=3,
            output_tokens=2,
            cost_estimate=original,
        ),
    )

    await store.start(run)
    await store.finish(result)
    replacement_schedule = replace(
        _complete("9.99", schedule="public:new"),
        components=(CostComponent("new", Decimal("9.99")),),
    )
    restored = await store.result(run.id)
    await store.close()

    assert replacement_schedule != original
    assert restored is not None
    assert restored.usage.cost_estimate == original
    assert restored == result
