"""Fixed-endpoint xAI Grok specialization of the compatible chat adapter."""

from __future__ import annotations

from collections.abc import Callable

from .openai_compatible import OpenAICompatibleProvider, _OpenAICompatibleClient

_XAI_BASE_URL = "https://api.x.ai/v1"


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


__all__ = ["GrokProvider"]
