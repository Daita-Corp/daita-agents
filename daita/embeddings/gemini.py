"""
Google Gemini embedding provider.

Supports gemini-embedding-001 with configurable output dimensions
(3072 default, 1536 and 768 recommended).
Uses the google-genai SDK.
"""

from __future__ import annotations

import asyncio
import os
import logging
from typing import List, Optional, TYPE_CHECKING

from .base import BaseEmbeddingProvider

if TYPE_CHECKING:
    from google.genai import Client
    from google.genai.types import ContentUnion, EmbedContentConfig

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider with configurable output dimensions."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        output_dimensionality: int = 1536,
        **kwargs,
    ):
        """
        Args:
            model: Gemini embedding model name.
            api_key: Google AI API key (or set GOOGLE_API_KEY / GEMINI_API_KEY).
            output_dimensionality: Output vector dimensions (default 1536).
                Recommended values: 3072, 1536, 768.
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._output_dimensionality = output_dimensionality
        self._client: Client | None = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable required for Gemini embeddings"
                )
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required for Gemini embeddings. "
                    "Install with: pip install 'daita-agents[google]'"
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def dimensions(self) -> int:
        return self._output_dimensionality

    @property
    def _embedding_config(self) -> EmbedContentConfig:
        try:
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini embeddings. "
                "Install with: pip install 'daita-agents[google]'"
            ) from exc
        return types.EmbedContentConfig(
            output_dimensionality=self._output_dimensionality
        )

    async def _embed_text_impl(self, text: str) -> List[float]:
        result = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.model,
            contents=text,
            config=self._embedding_config,
        )
        embeddings = result.embeddings
        if not embeddings or embeddings[0].values is None:
            raise ValueError("Gemini embedding response did not contain values")
        return list(embeddings[0].values)

    async def _embed_texts_impl(self, texts: List[str]) -> List[List[float]]:
        contents: List[ContentUnion] = [text for text in texts]
        result = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.model,
            contents=contents,
            config=self._embedding_config,
        )
        embeddings = result.embeddings or []
        if any(embedding.values is None for embedding in embeddings):
            raise ValueError("Gemini embedding response did not contain values")
        return [list(embedding.values or []) for embedding in embeddings]
