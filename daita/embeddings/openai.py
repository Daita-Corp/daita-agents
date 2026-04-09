"""
OpenAI embedding provider.

Supports text-embedding-3-small (1536), text-embedding-3-large (3072),
and text-embedding-ada-002 (1536).
"""

import os
import logging
from typing import List, Optional

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

# Known model → dimension mapping
_OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider with lazy client initialization."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable required for OpenAI embeddings"
                )
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI embeddings. "
                    "Install with: pip install 'daita-agents[memory]'"
                )
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    @property
    def dimensions(self) -> int:
        if self.model in _OPENAI_DIMENSIONS:
            return _OPENAI_DIMENSIONS[self.model]
        # Default for unknown models
        logger.warning(
            f"Unknown OpenAI embedding model '{self.model}', assuming 1536 dimensions"
        )
        return 1536

    async def _embed_text_impl(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    async def _embed_texts_impl(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
