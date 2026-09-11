"""Decode Anthropic token and billing usage dimensions."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import ModelUsage
from .._fields import field as _field, usage_int as _usage_int


@dataclass(frozen=True, slots=True)
class _AnthropicBillingUsage:
    usage: ModelUsage
    token_counts_complete: bool
    cache_write_5m_tokens: int
    cache_write_1h_tokens: int
    cache_write_breakdown_complete: bool
    service_tier: str | None
    inference_geo: str | None


def _decode_usage(value: object) -> ModelUsage:
    return _decode_anthropic_billing_usage(value).usage


def _decode_anthropic_billing_usage(
    value: object,
) -> _AnthropicBillingUsage:
    if value is None:
        return _AnthropicBillingUsage(
            usage=ModelUsage(),
            token_counts_complete=False,
            cache_write_5m_tokens=0,
            cache_write_1h_tokens=0,
            cache_write_breakdown_complete=True,
            service_tier=None,
            inference_geo=None,
        )
    uncached_input = _usage_int(
        _field(value, "input_tokens", 0),
        "input tokens",
    )
    cache_read = _usage_int(
        _field(value, "cache_read_input_tokens", 0),
        "cache read input tokens",
    )
    cache_write = _usage_int(
        _field(value, "cache_creation_input_tokens", 0),
        "cache creation input tokens",
    )
    output_tokens = _usage_int(
        _field(value, "output_tokens", 0),
        "output tokens",
    )
    output_details = _field(value, "output_tokens_details", None)
    reasoning_tokens = _usage_int(
        _field(output_details, "thinking_tokens", 0),
        "thinking tokens",
    )
    if reasoning_tokens > output_tokens:
        raise ValueError("Anthropic thinking tokens exceed total output tokens")
    cache_creation = _field(value, "cache_creation", None)
    cache_write_5m = (
        0
        if cache_creation is None
        else _usage_int(
            _field(cache_creation, "ephemeral_5m_input_tokens"),
            "5-minute cache creation input tokens",
        )
    )
    cache_write_1h = (
        0
        if cache_creation is None
        else _usage_int(
            _field(cache_creation, "ephemeral_1h_input_tokens"),
            "1-hour cache creation input tokens",
        )
    )
    cache_breakdown_complete = (cache_write == 0 and cache_creation is None) or (
        cache_creation is not None and cache_write_5m + cache_write_1h == cache_write
    )
    usage = ModelUsage(
        input_tokens=uncached_input + cache_read + cache_write,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
    missing = object()
    token_counts_complete = all(
        _field(value, name, missing) is not missing
        for name in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
        )
    )
    service_tier_value = _field(value, "service_tier", None)
    inference_geo_value = _field(value, "inference_geo", None)
    return _AnthropicBillingUsage(
        usage=usage,
        token_counts_complete=token_counts_complete,
        cache_write_5m_tokens=cache_write_5m,
        cache_write_1h_tokens=cache_write_1h,
        cache_write_breakdown_complete=cache_breakdown_complete,
        service_tier=(
            None
            if service_tier_value is None
            else _anthropic_service_tier(service_tier_value)
        ),
        inference_geo=(
            None
            if inference_geo_value is None
            else _anthropic_inference_geo(inference_geo_value)
        ),
    )


def _anthropic_service_tier(value: object) -> str:
    if not isinstance(value, str):
        native = getattr(value, "value", value)
        if not isinstance(native, str):
            raise ValueError("Anthropic service tier must be text")
        value = native
    normalized = value.strip().casefold()
    if normalized not in {"standard", "priority", "batch"}:
        raise ValueError("Anthropic service tier is unknown")
    return normalized


def _anthropic_inference_geo(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Anthropic inference geo must be text")
    normalized = value.strip().casefold()
    if normalized not in {"global", "us"}:
        raise ValueError("Anthropic inference geo is unknown")
    return normalized
