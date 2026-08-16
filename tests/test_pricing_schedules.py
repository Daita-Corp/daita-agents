import copy
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal
from importlib import resources

import pytest

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import (
    BillableQuantity,
    CostBasis,
    CostEstimateStatus,
    PricingModifier,
    PricingQualifier,
    PricingRate,
    PricingSchedule,
    PricingSource,
    PricingUsageRange,
    calculate_cost_estimate,
    has_complete_pricing_coverage,
    load_bundled_pricing_schedules,
    parse_pricing_schedules,
    provider_reported_cost_estimate,
    select_pricing_schedule,
    validate_pricing_schedules,
)
from daita.llm.providers.anthropic import AnthropicProvider
from daita.llm.providers.gemini import (
    GeminiProvider,
    _decode_usage as decode_gemini_usage,
)
from daita.llm.providers.grok import GrokProvider
from daita.llm.providers.openai import (
    OpenAIProvider,
    _billable_quantities as openai_billable_quantities,
    _decode_usage as decode_openai_usage,
)
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.storage.sqlite import SQLiteStateStore

NOW = datetime(2026, 7, 26, 12, tzinfo=UTC)
SOURCE = PricingSource(
    purpose="rates",
    url="https://example.com/pricing",
    reviewed_on=date(2026, 7, 26),
)
TOKEN_RATES = (
    PricingRate(
        metric="input_uncached_tokens",
        unit="token",
        unit_size=1_000_000,
        price_usd=Decimal("1"),
    ),
    PricingRate(
        metric="input_cache_read_tokens",
        unit="token",
        unit_size=1_000_000,
        price_usd=Decimal("0.1"),
    ),
    PricingRate(
        metric="input_cache_write_tokens",
        unit="token",
        unit_size=1_000_000,
        price_usd=Decimal("1.25"),
    ),
    PricingRate(
        metric="output_tokens",
        unit="token",
        unit_size=1_000_000,
        price_usd=Decimal("2"),
    ),
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


def _schedule(
    schedule_id: str,
    *,
    basis: CostBasis = CostBasis.PUBLIC_LIST,
    provider: str = "provider",
    model: str = "exact-model",
    endpoint: str = "responses",
    effective_from: datetime = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    ),
    effective_until: datetime | None = None,
    qualifiers: tuple[PricingQualifier, ...] = (
        PricingQualifier("region", "global"),
        PricingQualifier("service_tier", "default"),
    ),
    usage_range: PricingUsageRange | None = None,
    rates: tuple[PricingRate, ...] = TOKEN_RATES,
    modifiers: tuple[PricingModifier, ...] = (),
) -> PricingSchedule:
    return PricingSchedule(
        schedule_id=schedule_id,
        basis=basis,
        provider=provider,
        model=model,
        endpoint=endpoint,
        effective_from=effective_from,
        effective_until=effective_until,
        qualifiers=qualifiers,
        usage_range=usage_range,
        rates=rates,
        modifiers=modifiers,
        sources=(SOURCE,),
    )


def _calculate_openai(
    *,
    model: str = "gpt-5.6-sol",
    input_tokens: int,
    uncached: int,
    cache_read: int,
    cache_write: int,
    output: int,
):
    assert uncached + cache_read + cache_write == input_tokens
    return calculate_cost_estimate(
        load_bundled_pricing_schedules(),
        provider="openai",
        model=model,
        endpoint="responses",
        requested_at=NOW,
        qualifiers={"service_tier": "default", "region": "global"},
        usage_values={"request_input_tokens": Decimal(input_tokens)},
        quantities=(
            BillableQuantity(
                "input_uncached_tokens",
                Decimal(uncached),
                "token",
            ),
            BillableQuantity(
                "input_cache_read_tokens",
                Decimal(cache_read),
                "token",
            ),
            BillableQuantity(
                "input_cache_write_tokens",
                Decimal(cache_write),
                "token",
            ),
            BillableQuantity("output_tokens", Decimal(output), "token"),
        ),
    )


def _bundled_document() -> dict[str, object]:
    data = (
        resources.files("daita.llm")
        .joinpath("pricing_schedules.json")
        .read_text(encoding="utf-8")
    )
    decoded = json.loads(data)
    assert isinstance(decoded, dict)
    return decoded


def test_bundled_schedule_file_is_bounded_packaged_and_official():
    schedule_file = resources.files("daita.llm").joinpath("pricing_schedules.json")
    data = schedule_file.read_bytes()
    schedules = load_bundled_pricing_schedules()

    assert schedule_file.is_file()
    assert 0 < len(data) <= 256 * 1_024
    assert len(schedules) == 16
    assert {item.provider for item in schedules} == {
        "anthropic",
        "gemini",
        "openai",
    }
    assert {item.model for item in schedules} == {
        "claude-haiku-4-5-20251001",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }
    assert {item.endpoint for item in schedules} == {
        "generate_content",
        "messages",
        "responses",
    }
    assert {
        (item.usage_range.minimum_inclusive, item.usage_range.maximum_inclusive)
        for item in schedules
        if item.usage_range is not None
    } == {(0, 272_000), (272_001, None)}
    assert all(
        source.url.startswith(
            (
                "https://ai.google.dev/",
                "https://developers.openai.com/",
                "https://platform.claude.com/",
            )
        )
        for schedule in schedules
        for source in schedule.sources
    )
    lowered = data.lower()
    assert b"api_key" not in lowered
    assert b"billing_account" not in lowered
    assert b"credential" not in lowered
    assert b"secret" not in lowered


def test_schedule_records_are_immutable_and_decimal_native():
    schedule = load_bundled_pricing_schedules()[0]
    assert all(isinstance(rate.price_usd, Decimal) for rate in schedule.rates)
    with pytest.raises(FrozenInstanceError):
        schedule.model = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda schedule: schedule.update(effective_from="2026-07-26"),
        lambda schedule: schedule["rates"][0].update(price_usd="-1"),
        lambda schedule: schedule["rates"][0].update(price_usd=1.25),
        lambda schedule: schedule["rates"][0].update(unit_size=0),
        lambda schedule: schedule["rates"][0].update(metric="unknown_meter"),
        lambda schedule: schedule["rates"][0].update(currency="EUR"),
        lambda schedule: schedule.update(basis="provider_reported"),
        lambda schedule: schedule["sources"][0].update(
            url="https://user:password@example.com/pricing"
        ),
        lambda schedule: schedule["qualifiers"].update(billing_account="customer-123"),
    ),
)
def test_schedule_parser_rejects_malformed_or_sensitive_records(mutate):
    document = _bundled_document()
    schedules = document["schedules"]
    assert isinstance(schedules, list)
    schedule = schedules[0]
    assert isinstance(schedule, dict)
    mutate(schedule)
    with pytest.raises((TypeError, ValueError)):
        parse_pricing_schedules(json.dumps(document))


def test_schedule_parser_rejects_size_duplicates_and_ambiguity():
    with pytest.raises(ValueError, match="size bound"):
        parse_pricing_schedules(b" " * (256 * 1_024 + 1))
    with pytest.raises(ValueError, match="keys cannot repeat"):
        parse_pricing_schedules('{"schedules":[],"schedules":[]}')

    document = _bundled_document()
    schedules = document["schedules"]
    assert isinstance(schedules, list)
    duplicate = copy.deepcopy(schedules[0])
    schedules.append(duplicate)
    with pytest.raises(ValueError, match="IDs cannot repeat"):
        parse_pricing_schedules(json.dumps(document))

    duplicate["schedule_id"] = "different-id"
    with pytest.raises(ValueError, match="overlap ambiguously"):
        parse_pricing_schedules(json.dumps(document))


def test_schedule_parser_rejects_every_record_and_array_bound():
    mutations = (
        lambda schedule: schedule.update(schedule_id="x" * 257),
        lambda schedule: schedule.update(
            qualifiers={f"dimension_{index}": "value" for index in range(17)}
        ),
        lambda schedule: schedule.update(rates=schedule["rates"] * 9),
        lambda schedule: schedule.update(
            modifiers=[
                {"name": f"modifier_{index}", "multiplier": "1"} for index in range(17)
            ]
        ),
        lambda schedule: schedule.update(sources=schedule["sources"] * 4),
        lambda schedule: schedule["sources"][0].update(
            url="https://example.com/" + "x" * 2_048
        ),
    )
    for mutate in mutations:
        document = _bundled_document()
        schedules = document["schedules"]
        assert isinstance(schedules, list)
        schedule = schedules[0]
        assert isinstance(schedule, dict)
        mutate(schedule)
        with pytest.raises((TypeError, ValueError)):
            parse_pricing_schedules(json.dumps(document))

    document = _bundled_document()
    schedules = document["schedules"]
    assert isinstance(schedules, list)
    first = schedules[0]
    assert isinstance(first, dict)
    document["schedules"] = [
        {**copy.deepcopy(first), "schedule_id": f"schedule-{index}"}
        for index in range(129)
    ]
    with pytest.raises(ValueError, match="bounded array"):
        parse_pricing_schedules(json.dumps(document))


def test_effective_periods_are_start_inclusive_end_exclusive_and_future_safe():
    first = _schedule(
        "closed",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        effective_until=datetime(2026, 2, 1, tzinfo=UTC),
    )
    future = _schedule(
        "future",
        effective_from=datetime(2026, 2, 1, tzinfo=UTC),
    )
    schedules = validate_pricing_schedules((first, future))

    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=datetime(2026, 1, 1, tzinfo=UTC),
            qualifiers={"service_tier": "default", "region": "global"},
        )
        == first
    )
    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=datetime(
                2026,
                1,
                31,
                23,
                59,
                59,
                tzinfo=UTC,
            ),
            qualifiers={"service_tier": "default", "region": "global"},
        )
        == first
    )
    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=datetime(2026, 2, 1, tzinfo=UTC),
            qualifiers={"service_tier": "default", "region": "global"},
        )
        == future
    )
    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=datetime(2025, 12, 31, tzinfo=UTC),
            qualifiers={"service_tier": "default", "region": "global"},
        )
        is None
    )


def test_selection_requires_exact_model_endpoint_tier_region_and_range():
    short = _schedule(
        "short",
        usage_range=PricingUsageRange(
            "request_input_tokens",
            minimum_inclusive=0,
            maximum_inclusive=100,
        ),
    )
    long = _schedule(
        "long",
        usage_range=PricingUsageRange(
            "request_input_tokens",
            minimum_inclusive=101,
            maximum_inclusive=None,
        ),
    )
    schedules = (short, long)

    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=NOW,
            qualifiers={"service_tier": "default", "region": "global"},
            usage_values={"request_input_tokens": Decimal("100")},
        )
        == short
    )
    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=NOW,
            qualifiers={"service_tier": "default", "region": "global"},
            usage_values={"request_input_tokens": Decimal("101")},
        )
        == long
    )
    for model, endpoint, qualifiers in (
        (
            "model-alias",
            "responses",
            {"service_tier": "default", "region": "global"},
        ),
        (
            "exact-model",
            "chat_completions",
            {"service_tier": "default", "region": "global"},
        ),
        (
            "exact-model",
            "responses",
            {"service_tier": "priority", "region": "global"},
        ),
        ("exact-model", "responses", {"service_tier": "default"}),
    ):
        assert (
            select_pricing_schedule(
                schedules,
                provider="provider",
                model=model,
                endpoint=endpoint,
                requested_at=NOW,
                qualifiers=qualifiers,
                usage_values={"request_input_tokens": Decimal("1")},
            )
            is None
        )


def test_openai_uncached_cache_and_output_arithmetic_is_auditable():
    estimate = _calculate_openai(
        input_tokens=272_000,
        uncached=100_000,
        cache_read=100_000,
        cache_write=72_000,
        output=10_000,
    )

    assert estimate.status is CostEstimateStatus.COMPLETE
    assert estimate.amount_usd == Decimal("1.3000")
    assert estimate.basis is CostBasis.PUBLIC_LIST
    assert estimate.rate_schedule_id is not None
    assert estimate.rate_schedule_id.endswith(".short.2026-07-26")
    assert len(estimate.components) == 4
    assert {
        component.metric: (
            component.quantity,
            component.unit,
            component.unit_size,
            component.rate_usd,
            component.amount_usd,
        )
        for component in estimate.components
    } == {
        "input_uncached_tokens": (
            Decimal("100000"),
            "token",
            1_000_000,
            Decimal("5.00"),
            Decimal("0.5000"),
        ),
        "input_cache_read_tokens": (
            Decimal("100000"),
            "token",
            1_000_000,
            Decimal("0.50"),
            Decimal("0.0500"),
        ),
        "input_cache_write_tokens": (
            Decimal("72000"),
            "token",
            1_000_000,
            Decimal("6.25"),
            Decimal("0.45000"),
        ),
        "output_tokens": (
            Decimal("10000"),
            "token",
            1_000_000,
            Decimal("30.00"),
            Decimal("0.3000"),
        ),
    }


def test_openai_reasoning_is_an_output_subset_and_is_not_double_counted():
    usage = decode_openai_usage(
        {
            "input_tokens": 100,
            "output_tokens": 40,
            "input_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 10,
            },
            "output_tokens_details": {"reasoning_tokens": 30},
        }
    )
    quantities = openai_billable_quantities(usage)

    assert usage.reasoning_tokens == 30
    assert usage.output_tokens == 40
    assert {item.metric: item.quantity for item in quantities} == {
        "input_uncached_tokens": Decimal("70"),
        "input_cache_read_tokens": Decimal("20"),
        "input_cache_write_tokens": Decimal("10"),
        "output_tokens": Decimal("40"),
    }


@pytest.mark.parametrize(
    "usage",
    (
        {
            "input_tokens": 10,
            "output_tokens": 1,
            "input_tokens_details": {
                "cached_tokens": 8,
                "cache_write_tokens": 3,
            },
        },
        {
            "input_tokens": 1,
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    ),
)
def test_openai_malformed_usage_fails_closed(usage):
    with pytest.raises(ValueError):
        decode_openai_usage(usage)


def test_long_context_threshold_prices_the_whole_request():
    short = _calculate_openai(
        input_tokens=272_000,
        uncached=272_000,
        cache_read=0,
        cache_write=0,
        output=100_000,
    )
    long = _calculate_openai(
        input_tokens=272_001,
        uncached=272_001,
        cache_read=0,
        cache_write=0,
        output=100_000,
    )

    assert short.amount_usd == Decimal("4.36000")
    assert long.amount_usd == Decimal("7.220010")
    assert short.rate_schedule_id is not None
    assert long.rate_schedule_id is not None
    assert ".short." in short.rate_schedule_id
    assert ".long." in long.rate_schedule_id
    assert {component.rate_usd for component in long.components} == {
        Decimal("10.00"),
        Decimal("1.00"),
        Decimal("12.50"),
        Decimal("45.00"),
    }


def test_tier_and_endpoint_variants_are_separate_exact_schedules():
    standard = _schedule("standard")
    flex = _schedule(
        "flex",
        qualifiers=(
            PricingQualifier("region", "global"),
            PricingQualifier("service_tier", "flex"),
        ),
    )
    priority = _schedule(
        "priority",
        qualifiers=(
            PricingQualifier("region", "global"),
            PricingQualifier("service_tier", "priority"),
        ),
    )
    batch = _schedule("batch", endpoint="batch")
    schedules = (standard, flex, priority, batch)

    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=NOW,
            qualifiers={"region": "global", "service_tier": "default"},
        )
        == standard
    )
    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=NOW,
            qualifiers={"region": "global", "service_tier": "flex"},
        )
        == flex
    )
    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="responses",
            requested_at=NOW,
            qualifiers={"region": "global", "service_tier": "priority"},
        )
        == priority
    )
    assert (
        select_pricing_schedule(
            schedules,
            provider="provider",
            model="exact-model",
            endpoint="batch",
            requested_at=NOW,
            qualifiers={"region": "global", "service_tier": "default"},
        )
        == batch
    )


def test_regional_modifier_is_named_flat_and_retained():
    regional = _schedule(
        "regional",
        qualifiers=(
            PricingQualifier("region", "eu"),
            PricingQualifier("service_tier", "default"),
        ),
        rates=(
            PricingRate(
                "server_side_tool_calls",
                "call",
                1,
                Decimal("1"),
            ),
        ),
        modifiers=(PricingModifier("data_residency_uplift", Decimal("1.10")),),
    )
    estimate = calculate_cost_estimate(
        (regional,),
        provider="provider",
        model="exact-model",
        endpoint="responses",
        requested_at=NOW,
        qualifiers={"region": "eu", "service_tier": "default"},
        quantities=(
            BillableQuantity(
                "server_side_tool_calls",
                Decimal("1"),
                "call",
            ),
        ),
    )

    assert estimate.amount_usd == Decimal("1.10")
    assert estimate.components[0].modifiers == (
        PricingModifier("data_residency_uplift", Decimal("1.10")),
    )


def test_generic_non_token_meter_uses_the_same_arithmetic_contract():
    schedule = _schedule(
        "tools",
        rates=(
            PricingRate(
                "server_side_tool_calls",
                "call",
                1,
                Decimal("0.01"),
            ),
        ),
    )
    estimate = calculate_cost_estimate(
        (schedule,),
        provider="provider",
        model="exact-model",
        endpoint="responses",
        requested_at=NOW,
        qualifiers={"region": "global", "service_tier": "default"},
        quantities=(
            BillableQuantity(
                "server_side_tool_calls",
                Decimal("2"),
                "call",
            ),
        ),
    )

    assert estimate.status is CostEstimateStatus.COMPLETE
    assert estimate.amount_usd == Decimal("0.02")
    component = estimate.components[0]
    assert component.metric == "server_side_tool_calls"
    assert component.quantity == Decimal("2")
    assert component.unit == "call"
    assert component.unit_size == 1
    assert component.rate_usd == Decimal("0.01")


def test_provider_reported_total_is_authoritative_exact_and_schedule_free():
    estimate = provider_reported_cost_estimate(
        Decimal("0.0000158500"),
        currency="USD",
        unit="request",
    )

    assert estimate.status is CostEstimateStatus.COMPLETE
    assert estimate.basis is CostBasis.PROVIDER_REPORTED
    assert estimate.rate_schedule_id is None
    assert estimate.amount_usd == Decimal("0.0000158500")
    assert len(estimate.components) == 1
    component = estimate.components[0]
    assert component.name == "provider_reported_total"
    assert component.metric == "provider_reported_total"
    assert component.rate_schedule_id is None
    assert component.amount_usd == estimate.amount_usd
    with pytest.raises(ValueError):
        provider_reported_cost_estimate(
            Decimal("1"),
            currency="EUR",
            unit="request",
        )


def test_xai_provider_reported_ticks_take_precedence_over_token_rates():
    provider = GrokProvider("grok-4.5")
    usage = provider._decode_usage(  # noqa: SLF001 - focused adapter contract
        {
            "prompt_tokens": 199,
            "completion_tokens": 1,
            "cost_in_usd_ticks": 158_500,
        }
    )

    assert usage.input_tokens == 199
    assert usage.output_tokens == 1
    assert usage.cost_estimate.amount_usd == Decimal("0.00001585")
    assert usage.cost_estimate.basis is CostBasis.PROVIDER_REPORTED
    assert usage.cost_estimate.rate_schedule_id is None
    assert provider.has_complete_pricing(_request()) is True
    with pytest.raises(ValueError):
        provider._decode_usage(  # noqa: SLF001 - focused adapter contract
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_in_usd_ticks": "158500",
            }
        )


def test_configured_contract_schedule_uses_the_validated_public_shape():
    document = _bundled_document()
    schedules = document["schedules"]
    assert isinstance(schedules, list)
    configured = copy.deepcopy(schedules[0])
    assert isinstance(configured, dict)
    configured["schedule_id"] = "customer.responses.exact-model.2026-q3"
    configured["basis"] = "configured_contract"
    configured["provider"] = "custom"
    configured["model"] = "exact-model"
    configured["usage_range"] = None
    document["schedules"] = [configured]
    parsed = parse_pricing_schedules(json.dumps(document))
    estimate = calculate_cost_estimate(
        parsed,
        provider="custom",
        model="exact-model",
        endpoint="responses",
        requested_at=NOW,
        qualifiers={"service_tier": "default", "region": "global"},
        quantities=(
            BillableQuantity(
                "input_uncached_tokens",
                Decimal("1000000"),
                "token",
            ),
            BillableQuantity(
                "input_cache_read_tokens",
                Decimal("0"),
                "token",
            ),
            BillableQuantity(
                "input_cache_write_tokens",
                Decimal("0"),
                "token",
            ),
            BillableQuantity("output_tokens", Decimal("0"), "token"),
        ),
    )

    assert estimate.status is CostEstimateStatus.COMPLETE
    assert estimate.basis is CostBasis.CONFIGURED_CONTRACT
    assert estimate.rate_schedule_id == configured["schedule_id"]
    assert estimate.amount_usd == Decimal("5.00")


def test_exact_configured_contract_enables_only_its_compatible_endpoint():
    configured = _schedule(
        "custom.chat.exact-model.2026-q3",
        basis=CostBasis.CONFIGURED_CONTRACT,
        provider="custom",
        model="exact-model",
        endpoint="chat_completions",
        qualifiers=(),
        usage_range=None,
    )
    provider = OpenAICompatibleProvider(
        "exact-model",
        provider="custom",
        base_url="https://example.com/v1",
        pricing_schedules=(configured,),
        clock=lambda: NOW,
    )
    unpriced = OpenAICompatibleProvider(
        "exact-model",
        provider="custom",
        base_url="https://example.com/v1",
        clock=lambda: NOW,
    )
    response = {
        "id": "completion-1",
        "model": "exact-model",
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": "ok",
                    "refusal": None,
                    "tool_calls": [],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "completion_tokens_details": {"reasoning_tokens": 500_000},
        },
    }

    decoded = provider._decode_response(  # noqa: SLF001 - adapter contract
        response,
        requested_at=NOW,
    )

    assert provider.has_complete_pricing(_request()) is True
    assert unpriced.has_complete_pricing(_request()) is False
    assert decoded.usage.reasoning_tokens == 500_000
    assert decoded.usage.cost_estimate.status is CostEstimateStatus.COMPLETE
    assert decoded.usage.cost_estimate.basis is CostBasis.CONFIGURED_CONTRACT
    assert decoded.usage.cost_estimate.amount_usd == Decimal("3")
    assert decoded.usage.cost_estimate.rate_schedule_id == configured.schedule_id


def test_unknown_identity_missing_dimensions_and_missing_rates_fail_closed():
    schedules = load_bundled_pricing_schedules()
    quantities = (
        BillableQuantity(
            "input_uncached_tokens",
            Decimal("1"),
            "token",
        ),
    )
    unknown = calculate_cost_estimate(
        schedules,
        provider="openai",
        model="unknown-model",
        endpoint="responses",
        requested_at=NOW,
        usage_values={"request_input_tokens": Decimal("1")},
        quantities=quantities,
        qualifiers={"service_tier": "default", "region": "global"},
    )
    missing_qualifier = calculate_cost_estimate(
        schedules,
        provider="openai",
        model="gpt-5.6-sol",
        endpoint="responses",
        requested_at=NOW,
        usage_values={"request_input_tokens": Decimal("1")},
        quantities=quantities,
        qualifiers={"service_tier": "default"},
    )
    missing_rate = calculate_cost_estimate(
        (
            _schedule(
                "only-input",
                rates=(TOKEN_RATES[0],),
            ),
        ),
        provider="provider",
        model="exact-model",
        endpoint="responses",
        requested_at=NOW,
        qualifiers={"service_tier": "default", "region": "global"},
        quantities=(
            BillableQuantity(
                "input_uncached_tokens",
                Decimal("1"),
                "token",
            ),
            BillableQuantity("output_tokens", Decimal("1"), "token"),
        ),
    )

    assert unknown.status is CostEstimateStatus.UNAVAILABLE
    assert missing_qualifier.status is CostEstimateStatus.UNAVAILABLE
    assert missing_rate.status is CostEstimateStatus.PARTIAL
    assert missing_rate.code == "pricing_rate_unavailable"


def test_complete_pricing_coverage_requires_gap_free_effective_ranges():
    bundled = load_bundled_pricing_schedules()
    assert has_complete_pricing_coverage(
        bundled,
        provider="openai",
        model="gpt-5.6-sol",
        endpoint="responses",
        requested_at=NOW,
        qualifiers={"service_tier": "default", "region": "global"},
        required_metrics=(
            "input_uncached_tokens",
            "input_cache_read_tokens",
            "input_cache_write_tokens",
            "output_tokens",
        ),
        usage_range_metric="request_input_tokens",
    )
    assert not has_complete_pricing_coverage(
        bundled,
        provider="openai",
        model="gpt-5.6",
        endpoint="responses",
        requested_at=NOW,
        qualifiers={"service_tier": "default", "region": "global"},
        required_metrics=(
            "input_uncached_tokens",
            "input_cache_read_tokens",
            "input_cache_write_tokens",
            "output_tokens",
        ),
        usage_range_metric="request_input_tokens",
    )


def _anthropic_response(
    *,
    model: str = "claude-opus-4-8",
    service_tier: str = "standard",
    inference_geo: str | None = "global",
    input_tokens: int = 1_000_000,
    cache_read_tokens: int = 100_000,
    cache_write_tokens: int = 200_000,
    cache_write_5m_tokens: int = 100_000,
    cache_write_1h_tokens: int = 100_000,
    output_tokens: int = 100_000,
) -> dict[str, object]:
    usage: dict[str, object] = {
        "input_tokens": input_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_write_tokens,
        "cache_creation": {
            "ephemeral_5m_input_tokens": cache_write_5m_tokens,
            "ephemeral_1h_input_tokens": cache_write_1h_tokens,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {"thinking_tokens": output_tokens // 2},
        "service_tier": service_tier,
    }
    if inference_geo is not None:
        usage["inference_geo"] = inference_geo
    return {
        "type": "message",
        "role": "assistant",
        "id": "message-1",
        "model": model,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
        "usage": usage,
    }


def test_anthropic_exact_response_dimensions_price_both_cache_ttls_and_geo():
    provider = AnthropicProvider("claude-opus-4-8", clock=lambda: NOW)
    decoded = provider._decode_response(  # noqa: SLF001 - adapter contract
        _anthropic_response(),
        requested_at=NOW,
    )

    assert decoded.usage.input_tokens == 1_300_000
    assert decoded.usage.output_tokens == 100_000
    assert decoded.usage.reasoning_tokens == 50_000
    assert decoded.usage.cache_read_tokens == 100_000
    assert decoded.usage.cache_write_tokens == 200_000
    assert decoded.usage.cost_estimate.status is CostEstimateStatus.COMPLETE
    assert decoded.usage.cost_estimate.amount_usd == Decimal("9.175")
    assert {
        component.metric for component in decoded.usage.cost_estimate.components
    } == {
        "input_uncached_tokens",
        "input_cache_read_tokens",
        "input_cache_write_5m_tokens",
        "input_cache_write_1h_tokens",
        "output_tokens",
    }
    assert provider.has_complete_pricing(_request()) is False

    us_response = _anthropic_response(inference_geo="us")
    us = provider._decode_response(us_response, requested_at=NOW)  # noqa: SLF001
    assert us.usage.cost_estimate.status is CostEstimateStatus.COMPLETE
    assert us.usage.cost_estimate.amount_usd == Decimal("10.09250")
    assert all(
        component.modifiers == (PricingModifier("data_residency", Decimal("1.10")),)
        for component in us.usage.cost_estimate.components
    )


def test_anthropic_sonnet_transition_haiku_and_unknown_dimensions_fail_closed():
    sonnet = AnthropicProvider("claude-sonnet-5", clock=lambda: NOW)
    response = _anthropic_response(
        model="claude-sonnet-5",
        input_tokens=1_000_000,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cache_write_5m_tokens=0,
        cache_write_1h_tokens=0,
        output_tokens=1_000_000,
    )
    introductory = sonnet._decode_response(  # noqa: SLF001
        response,
        requested_at=datetime(2026, 8, 31, 23, 59, 59, tzinfo=UTC),
    )
    standard = sonnet._decode_response(  # noqa: SLF001
        response,
        requested_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert introductory.usage.cost_estimate.amount_usd == Decimal("12")
    assert standard.usage.cost_estimate.amount_usd == Decimal("18")

    haiku = AnthropicProvider("claude-haiku-4-5-20251001", clock=lambda: NOW)
    haiku_response = _anthropic_response(
        model="claude-haiku-4-5-20251001",
        inference_geo=None,
        input_tokens=1_000_000,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cache_write_5m_tokens=0,
        cache_write_1h_tokens=0,
        output_tokens=1_000_000,
    )
    haiku_decoded = haiku._decode_response(  # noqa: SLF001
        haiku_response,
        requested_at=NOW,
    )
    assert haiku_decoded.usage.cost_estimate.amount_usd == Decimal("6")

    priority = _anthropic_response(service_tier="priority")
    assert (
        sonnet._decode_response(  # noqa: SLF001
            priority,
            requested_at=NOW,
        ).usage.cost_estimate.status
        is CostEstimateStatus.UNAVAILABLE
    )
    incomplete = _anthropic_response()
    incomplete_usage = incomplete["usage"]
    assert isinstance(incomplete_usage, dict)
    incomplete_usage.pop("cache_creation")
    decoded_incomplete = sonnet._decode_response(  # noqa: SLF001
        incomplete,
        requested_at=NOW,
    )
    assert (
        decoded_incomplete.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE
    )
    assert decoded_incomplete.usage.cost_estimate.code == (
        "billing_dimensions_incomplete"
    )


def _gemini_response(
    *,
    service_tier: str = "STANDARD",
    total_tokens: int = 1_160_000,
) -> dict[str, object]:
    return {
        "response_id": "response-1",
        "model_version": "gemini-3.6-flash",
        "prompt_feedback": None,
        "candidates": [
            {
                "finish_reason": "STOP",
                "content": {"parts": [{"text": "ok"}]},
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 1_000_000,
            "cached_content_token_count": 100_000,
            "tool_use_prompt_token_count": 10_000,
            "candidates_token_count": 100_000,
            "thoughts_token_count": 50_000,
            "total_token_count": total_tokens,
            "service_tier": service_tier,
        },
    }


def test_gemini_standard_paid_list_estimate_includes_thinking_and_tool_use():
    provider = GeminiProvider("gemini-3.6-flash", clock=lambda: NOW)
    decoded = provider._decode_response(  # noqa: SLF001 - adapter contract
        _gemini_response(),
        requested_at=NOW,
    )

    assert decoded.usage.input_tokens == 1_010_000
    assert decoded.usage.output_tokens == 150_000
    assert decoded.usage.reasoning_tokens == 50_000
    assert decoded.usage.cache_read_tokens == 100_000
    assert decoded.usage.total_tokens == 1_160_000
    assert decoded.usage.cost_estimate.status is CostEstimateStatus.COMPLETE
    assert decoded.usage.cost_estimate.amount_usd == Decimal("2.505")
    assert provider.has_complete_pricing(_request()) is True
    pricing_dimensions = decoded.provider_metadata["pricing_dimensions"]
    assert isinstance(pricing_dimensions, Mapping)
    assert pricing_dimensions["response_model"] == "gemini-3.6-flash"
    assert pricing_dimensions["service_tier"] == "standard"


def test_gemini_exact_models_are_covered_but_aliases_and_other_tiers_are_not():
    request = _request()
    assert all(
        GeminiProvider(model, clock=lambda: NOW).has_complete_pricing(request)
        for model in (
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        )
    )
    assert (
        GeminiProvider(
            "gemini-flash-latest",
            clock=lambda: NOW,
        ).has_complete_pricing(request)
        is False
    )

    priority = GeminiProvider(
        "gemini-3.6-flash",
        clock=lambda: NOW,
    )._decode_response(  # noqa: SLF001
        _gemini_response(service_tier="PRIORITY"),
        requested_at=NOW,
    )
    assert priority.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE

    incomplete = _gemini_response()
    usage = incomplete["usage_metadata"]
    assert isinstance(usage, dict)
    usage.pop("total_token_count")
    decoded_incomplete = GeminiProvider(
        "gemini-3.6-flash",
        clock=lambda: NOW,
    )._decode_response(  # noqa: SLF001
        incomplete,
        requested_at=NOW,
    )
    assert (
        decoded_incomplete.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE
    )
    assert decoded_incomplete.usage.cost_estimate.code == (
        "billing_dimensions_incomplete"
    )

    with pytest.raises(ValueError, match="do not match"):
        decode_gemini_usage(_gemini_response(total_tokens=1)["usage_metadata"])
    with pytest.raises(ValueError, match="cached tokens exceed"):
        decode_gemini_usage(
            {
                "prompt_token_count": 1,
                "cached_content_token_count": 2,
                "candidates_token_count": 0,
            }
        )


def test_openai_concrete_response_model_overrides_request_alias():
    provider = OpenAIProvider("gpt-5.6", clock=lambda: NOW)
    response = {
        "id": "response-1",
        "model": "gpt-5.6-sol",
        "service_tier": "default",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens_details": {"reasoning_tokens": 50_000},
        },
    }

    decoded = provider._decode_response(  # noqa: SLF001 - adapter contract
        response,
        requested_at=NOW,
    )
    assert decoded.usage.cost_estimate.status is CostEstimateStatus.COMPLETE
    assert decoded.usage.cost_estimate.amount_usd == Decimal("14.500")
    pricing_dimensions = decoded.provider_metadata["pricing_dimensions"]
    assert isinstance(pricing_dimensions, Mapping)
    assert pricing_dimensions["response_model"] == "gpt-5.6-sol"
    assert provider.has_complete_pricing(_request()) is False

    response["model"] = "gpt-5.6"
    alias = provider._decode_response(  # noqa: SLF001 - adapter contract
        response,
        requested_at=NOW,
    )
    assert alias.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE

    incomplete = copy.deepcopy(response)
    incomplete["model"] = "gpt-5.6-sol"
    incomplete_usage = incomplete["usage"]
    assert isinstance(incomplete_usage, dict)
    incomplete_usage.pop("input_tokens_details")
    decoded_incomplete = provider._decode_response(  # noqa: SLF001
        incomplete,
        requested_at=NOW,
    )
    assert (
        decoded_incomplete.usage.cost_estimate.status is CostEstimateStatus.UNAVAILABLE
    )
    assert (
        decoded_incomplete.usage.cost_estimate.code == "billing_dimensions_incomplete"
    )
    with pytest.raises(KeyError):
        decode_openai_usage({"input_tokens": 1})


def test_schedule_calculated_zero_is_explicit_and_complete():
    estimate = _calculate_openai(
        input_tokens=0,
        uncached=0,
        cache_read=0,
        cache_write=0,
        output=0,
    )
    assert estimate.status is CostEstimateStatus.COMPLETE
    assert estimate.amount_usd == Decimal("0")
    assert len(estimate.components) == 4


async def test_expanded_component_arithmetic_is_persisted_without_repricing(tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.sqlite3")
    estimate = _calculate_openai(
        input_tokens=10_000,
        uncached=7_000,
        cache_read=2_000,
        cache_write=1_000,
        output=500,
    )
    run = RunInput(
        id="run-priced-schedule",
        agent_id="agent-priced",
        message="question",
        created_at=NOW,
        conversation_id="conversation-priced",
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
            input_tokens=10_000,
            output_tokens=500,
            cache_read_tokens=2_000,
            cache_write_tokens=1_000,
            cost_estimate=estimate,
        ),
    )

    await store.start(run)
    await store.finish(result)
    restored = await store.result(run.id)
    await store.close()

    assert restored == result
    assert restored is not None
    assert restored.usage.cost_estimate == estimate
    assert restored.usage.cost_estimate.components[0].rate_usd == Decimal("5.00")
