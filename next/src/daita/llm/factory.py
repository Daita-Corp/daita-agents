"""Explicit construction of retained built-in model-provider adapters."""

from __future__ import annotations

import re

from .protocols import ModelProvider
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    GrokProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)

_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


def create_llm_provider(
    model_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_output_tokens: int = 1_024,
) -> ModelProvider:
    """Create one explicit retained adapter from a canonical provider:model ID.

    Unknown provider names are accepted only with an explicit compatible API
    endpoint. The resulting adapter remains a normal registry/router input;
    this function does not mutate global process state.
    """

    if (
        not isinstance(model_id, str)
        or model_id != model_id.strip()
        or len(model_id) > 256
        or any(character.isspace() for character in model_id)
    ):
        raise ValueError("model_id must be canonical provider:model text")
    provider_name, separator, model = model_id.partition(":")
    if (
        not separator
        or not _PROVIDER_NAME.fullmatch(provider_name)
        or not model
        or model != model.strip()
    ):
        raise ValueError("model_id must use canonical provider:model form")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be a non-empty string when provided")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")

    if provider_name == "openai":
        _reject_base_url(provider_name, base_url)
        return OpenAIProvider(
            model,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "anthropic":
        _reject_base_url(provider_name, base_url)
        return AnthropicProvider(
            model,
            api_key=api_key,
            max_tokens=max_output_tokens,
        )
    if provider_name == "gemini":
        _reject_base_url(provider_name, base_url)
        return GeminiProvider(
            model,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
    if provider_name == "grok":
        _reject_base_url(provider_name, base_url)
        return GrokProvider(
            model,
            api_key=api_key,
            max_tokens=max_output_tokens,
        )
    if provider_name == "ollama":
        return OllamaProvider(
            model,
            base_url=("http://127.0.0.1:11434/v1" if base_url is None else base_url),
            api_key="ollama" if api_key is None else api_key,
            max_tokens=max_output_tokens,
        )
    if base_url is None:
        raise ValueError(
            "an explicit base_url is required for an OpenAI-compatible provider"
        )
    return OpenAICompatibleProvider(
        model,
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_output_tokens,
    )


def _reject_base_url(provider_name: str, base_url: str | None) -> None:
    if base_url is not None:
        raise ValueError(f"{provider_name} uses its fixed provider endpoint")


__all__ = ["create_llm_provider"]
