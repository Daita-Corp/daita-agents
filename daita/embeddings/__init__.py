"""
Embedding providers for Daita Agents.

Mirrors the daita.llm module pattern: BaseEmbeddingProvider + factory + registry.
"""

from .base import BaseEmbeddingProvider
from .factory import create_embedding_provider, register_embedding_provider

__all__ = [
    "BaseEmbeddingProvider",
    "create_embedding_provider",
    "register_embedding_provider",
]
