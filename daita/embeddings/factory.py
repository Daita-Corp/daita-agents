"""
Factory for creating embedding provider instances.
"""

import logging
from typing import Optional

from ..core.exceptions import ConfigError
from .base import BaseEmbeddingProvider
from .openai import OpenAIEmbeddingProvider
from .voyage import VoyageEmbeddingProvider
from .gemini import GeminiEmbeddingProvider
from .sentence_transformers import SentenceTransformersEmbeddingProvider
from .mock import MockEmbeddingProvider

logger = logging.getLogger(__name__)

PROVIDER_REGISTRY = {
    "openai": OpenAIEmbeddingProvider,
    "voyage": VoyageEmbeddingProvider,
    "gemini": GeminiEmbeddingProvider,
    "sentence-transformers": SentenceTransformersEmbeddingProvider,
    "mock": MockEmbeddingProvider,
}


def create_embedding_provider(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseEmbeddingProvider:
    """
    Create an embedding provider instance.

    Args:
        provider: Provider name ('openai', 'mock', or a registered custom name)
        model: Model identifier (provider-specific default if None)
        api_key: API key for authentication
        **kwargs: Additional provider-specific parameters

    Returns:
        Embedding provider instance

    Raises:
        ConfigError: If provider is not supported
    """
    provider_name = provider.lower()

    if provider_name not in PROVIDER_REGISTRY:
        available = list(PROVIDER_REGISTRY.keys())
        raise ConfigError(
            f"Unsupported embedding provider: {provider}. "
            f"Available providers: {available}"
        )

    provider_class = PROVIDER_REGISTRY[provider_name]

    build_kwargs = {**kwargs}
    if api_key is not None:
        build_kwargs["api_key"] = api_key
    if model is not None:
        build_kwargs["model"] = model

    try:
        return provider_class(**build_kwargs)
    except Exception as e:
        logger.error(f"Failed to create {provider} embedding provider: {e}")
        raise ConfigError(
            f"Failed to create {provider} embedding provider: {e}"
        )


def register_embedding_provider(name: str, provider_class) -> None:
    """
    Register a custom embedding provider.

    Args:
        name: Provider name
        provider_class: Class that extends BaseEmbeddingProvider
    """
    PROVIDER_REGISTRY[name.lower()] = provider_class
    logger.info(f"Registered custom embedding provider: {name}")
