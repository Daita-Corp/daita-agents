"""
Voyage AI embedding provider.

Supports voyage-3 (1024-dim), voyage-3-large (1536-dim),
voyage-3-lite (512-dim), and voyage-code-3 (1024-dim).
"""

import os
import logging
from typing import List, Optional

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

# Known model -> dimension mapping
_VOYAGE_DIMENSIONS = {
    "voyage-3": 1024,
    "voyage-3-large": 1536,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
}


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Voyage AI embedding provider with lazy client initialization."""

    def __init__(
        self,
        model: str = "voyage-3-large",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        api_key = (
            api_key or os.getenv("VOYAGE_API_KEY") or os.getenv("VOYAGEAI_API_KEY")
        )
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "VOYAGE_API_KEY environment variable required for Voyage embeddings"
                )
            try:
                import voyageai
            except ImportError:
                raise ImportError(
                    "voyageai is required for Voyage embeddings. "
                    "Install with: pip install 'daita-agents[voyage]'"
                )
            self._client = voyageai.AsyncClient(api_key=self.api_key)
        return self._client

    @property
    def dimensions(self) -> int:
        if self.model in _VOYAGE_DIMENSIONS:
            return _VOYAGE_DIMENSIONS[self.model]
        logger.warning(
            f"Unknown Voyage embedding model '{self.model}', assuming 1024 dimensions"
        )
        return 1024

    async def _embed_text_impl(self, text: str) -> List[float]:
        result = await self.client.embed(texts=[text], model=self.model)
        return result.embeddings[0]

    async def _embed_texts_impl(self, texts: List[str]) -> List[List[float]]:
        result = await self.client.embed(texts=texts, model=self.model)
        return result.embeddings
