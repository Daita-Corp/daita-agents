"""Fixed-endpoint xAI Grok specialization of the compatible chat adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal

from ..models import ModelRequest, ModelUsage
from ..pricing import provider_reported_cost_estimate
from .openai_compatible import (
    OpenAICompatibleProvider,
    _OpenAICompatibleClient,
    _decode_usage,
    _field,
)

_XAI_BASE_URL = "https://api.x.ai/v1"
_USD_TICKS_PER_DOLLAR = Decimal("10000000000")


class GrokProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 1_024,
        client: _OpenAICompatibleClient | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(
            model,
            provider="grok",
            base_url=_XAI_BASE_URL,
            api_key=api_key,
            max_tokens=max_tokens,
            client=client,
            id_factory=id_factory,
        )

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return True

    def _decode_usage(self, value: object) -> ModelUsage:
        usage = _decode_usage(value)
        if value is None:
            return usage
        ticks = _field(value, "cost_in_usd_ticks", None)
        if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 0:
            raise ValueError("xAI usage is missing a valid per-request USD charge")
        return replace(
            usage,
            cost_estimate=provider_reported_cost_estimate(
                Decimal(ticks) / _USD_TICKS_PER_DOLLAR,
                currency="USD",
                unit="request",
            ),
        )


__all__ = ["GrokProvider"]
