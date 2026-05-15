"""Resolve model prices and estimate LLM call cost."""

from __future__ import annotations

from decimal import Decimal
from fnmatch import fnmatchcase
from typing import Any, Mapping, Optional

from .catalog import FALLBACK_PRICING, PRICE_CATALOG
from .models import CostEstimate, ModelPricing, TokenUsage


def normalize_provider(provider: Optional[str]) -> str:
    provider = (provider or "unknown").lower().strip()
    aliases = {
        "google": "gemini",
        "xai": "grok",
        "openai-compatible": "openai",
    }
    return aliases.get(provider, provider)


def normalize_model(model: Optional[str]) -> str:
    return (model or "unknown").lower().strip()


class PricingRegistry:
    """Lookup and calculate model pricing from a data catalog."""

    def __init__(
        self,
        catalog: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
        fallback: Optional[Mapping[str, Any]] = None,
    ):
        self.catalog = catalog or PRICE_CATALOG
        self.fallback = fallback or FALLBACK_PRICING

    def resolve(self, provider: str, model: str) -> tuple[ModelPricing, str]:
        provider_key = normalize_provider(provider)
        model_key = normalize_model(model)
        provider_prices = self.catalog.get(provider_key, {})

        if model_key in provider_prices:
            return (
                self._build_pricing(
                    provider_key, model_key, provider_prices[model_key]
                ),
                "high",
            )

        for pattern, data in provider_prices.items():
            if fnmatchcase(model_key, pattern):
                return (
                    self._build_pricing(provider_key, pattern, data),
                    "medium" if "*" in pattern else "high",
                )

        return (
            self._build_pricing(provider_key, "fallback", self.fallback),
            "low",
        )

    def estimate(
        self,
        provider: str,
        model: str,
        usage: TokenUsage,
    ) -> CostEstimate:
        provider_key = normalize_provider(provider)
        model_key = normalize_model(model)
        if usage.total_tokens <= 0:
            return CostEstimate(
                usd=None,
                provider=provider_key,
                model=model_key,
                pricing_model=None,
                pricing_source="none",
                pricing_confidence="none",
                warning="No token usage was reported.",
            )

        pricing, confidence = self.resolve(provider_key, model_key)
        warning = None
        if confidence == "low":
            warning = (
                f"No pricing entry for {provider_key}:{model_key}; "
                "used explicit generic fallback pricing."
            )

        return CostEstimate(
            usd=pricing.calculate(usage),
            provider=provider_key,
            model=model_key,
            pricing_model=pricing.model,
            pricing_source=pricing.source or "catalog",
            pricing_confidence=confidence,
            warning=warning,
        )

    @staticmethod
    def _build_pricing(
        provider: str, model: str, data: Mapping[str, Any]
    ) -> ModelPricing:
        def dec(key: str) -> Optional[Decimal]:
            value = data.get(key)
            return Decimal(str(value)) if value is not None else None

        return ModelPricing(
            provider=provider,
            model=model,
            input_per_1m=dec("input_per_1m") or Decimal("0"),
            output_per_1m=dec("output_per_1m") or Decimal("0"),
            cached_input_per_1m=dec("cached_input_per_1m"),
            reasoning_per_1m=dec("reasoning_per_1m"),
            source=data.get("source"),
        )


DEFAULT_PRICING_REGISTRY = PricingRegistry()


def estimate_llm_cost(
    provider: str,
    model: str,
    usage: TokenUsage,
) -> CostEstimate:
    return DEFAULT_PRICING_REGISTRY.estimate(provider, model, usage)


def estimate_llm_cost_from_counts(
    provider: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> CostEstimate:
    usage = TokenUsage.from_counts(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    return estimate_llm_cost(provider, model, usage)
