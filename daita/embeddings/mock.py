"""
Mock embedding provider for testing.

Returns deterministic, hash-based vectors — no API calls or external dependencies.
"""

import hashlib
from typing import List

from .base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider that returns deterministic vectors for testing."""

    def __init__(
        self,
        model: str = "mock-embedding",
        dim: int = 1536,
        **kwargs,
    ):
        kwargs.pop("api_key", None)
        super().__init__(model=model, api_key="mock-key", cache_size=0, **kwargs)
        self._dim = dim

    @property
    def dimensions(self) -> int:
        return self._dim

    async def _embed_text_impl(self, text: str) -> List[float]:
        return self._deterministic_vector(text)

    async def _embed_texts_impl(self, texts: List[str]) -> List[List[float]]:
        return [self._deterministic_vector(t) for t in texts]

    def _deterministic_vector(self, text: str) -> List[float]:
        """Generate a repeatable unit-ish vector from text via SHA-256."""
        digest = hashlib.sha256(text.encode()).digest()
        # Stretch the 32-byte digest to fill `dim` floats
        floats: List[float] = []
        for i in range(self._dim):
            # Cycle through the digest bytes
            byte_idx = i % len(digest)
            # Combine position and byte value for variety
            seed = digest[byte_idx] ^ (i & 0xFF)
            floats.append((seed / 255.0) * 2 - 1)  # range [-1, 1]
        # Normalize to unit length
        norm = sum(f * f for f in floats) ** 0.5
        if norm > 0:
            floats = [f / norm for f in floats]
        return floats
