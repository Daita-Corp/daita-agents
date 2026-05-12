"""Shared LLM token and pricing models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class TokenUsage:
    """Provider-normalized token usage for one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_counts(
        cls,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> "TokenUsage":
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        total_tokens = int(total_tokens or input_tokens + output_tokens)
        if total_tokens and not input_tokens and not output_tokens:
            input_tokens = total_tokens
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=int(cached_input_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
        )

    def to_legacy_dict(self) -> dict[str, int]:
        """Return the historical prompt/completion token keys."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class ModelPricing:
    """Per-model token prices in USD per one million tokens."""

    provider: str
    model: str
    input_per_1m: Decimal
    output_per_1m: Decimal
    cached_input_per_1m: Optional[Decimal] = None
    reasoning_per_1m: Optional[Decimal] = None
    source: Optional[str] = None

    def calculate(self, usage: TokenUsage) -> Decimal:
        billable_input = usage.input_tokens
        cached_cost = Decimal("0")
        if self.cached_input_per_1m is not None and usage.cached_input_tokens:
            cached = min(usage.cached_input_tokens, usage.input_tokens)
            billable_input -= cached
            cached_cost = (Decimal(cached) * self.cached_input_per_1m) / Decimal(
                1_000_000
            )

        input_cost = (Decimal(billable_input) * self.input_per_1m) / Decimal(1_000_000)
        output_cost = (Decimal(usage.output_tokens) * self.output_per_1m) / Decimal(
            1_000_000
        )
        reasoning_cost = Decimal("0")
        if self.reasoning_per_1m is not None and usage.reasoning_tokens:
            reasoning_cost = (
                Decimal(usage.reasoning_tokens) * self.reasoning_per_1m
            ) / Decimal(1_000_000)

        return input_cost + cached_cost + output_cost + reasoning_cost


@dataclass(frozen=True)
class CostEstimate:
    """Result of pricing a normalized token usage payload."""

    usd: Optional[Decimal]
    provider: str
    model: str
    pricing_model: Optional[str]
    pricing_source: str
    pricing_confidence: str
    warning: Optional[str] = None

    def as_float(self) -> Optional[float]:
        return float(self.usd) if self.usd is not None else None
