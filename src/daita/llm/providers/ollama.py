"""Loopback-only Ollama specialization of the compatible chat adapter."""

from __future__ import annotations

from collections.abc import Callable

from .openai_compatible import (
    OpenAICompatibleProvider,
    _OpenAICompatibleClient,
    _validate_base_url,
)


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str = "ollama",
        max_tokens: int = 1_024,
        client: _OpenAICompatibleClient | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(
            model,
            provider="ollama",
            base_url=_validate_base_url(base_url, loopback_only=True),
            api_key=api_key,
            max_tokens=max_tokens,
            client=client,
            id_factory=id_factory,
        )


__all__ = ["OllamaProvider"]
