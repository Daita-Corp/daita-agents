"""
Sentence Transformers embedding provider.

Runs any HuggingFace sentence-transformers model locally — no API key required.
Not supported in Daita Cloud (torch dependency, no GPU in Lambda).

Usage:
    MemoryPlugin(embedding_provider="sentence-transformers", embedding_model="all-MiniLM-L6-v2")
    MemoryPlugin(embedding_provider="sentence-transformers", embedding_model="BAAI/bge-large-en-v1.5")
"""

import asyncio
import os
import logging
from typing import List

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using any HuggingFace sentence-transformers model."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ):
        if os.getenv("DAITA_RUNTIME") == "lambda":
            raise ValueError(
                "sentence-transformers is a local-only embedding provider and cannot "
                "run in Daita Cloud. Use a cloud provider instead (openai, voyage, gemini)."
            )
        kwargs.pop("api_key", None)
        super().__init__(model=model, api_key=None, **kwargs)
        self._model = None
        self._dimensions = None

    @property
    def _st_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install 'daita-agents[sentence-transformers]'"
                )
            self._model = SentenceTransformer(self.model)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.debug(f"Loaded {self.model} ({self._dimensions} dimensions)")
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Trigger model load to discover dimensions
            _ = self._st_model
        return self._dimensions

    async def _embed_text_impl(self, text: str) -> List[float]:
        embedding = await asyncio.to_thread(
            self._st_model.encode, text, normalize_embeddings=True
        )
        return embedding.tolist()

    async def _embed_texts_impl(self, texts: List[str]) -> List[List[float]]:
        embeddings = await asyncio.to_thread(
            self._st_model.encode, texts, normalize_embeddings=True
        )
        return embeddings.tolist()
