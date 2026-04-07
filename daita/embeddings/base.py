"""
Base class for all embedding providers.

Embedding calls are automatically traced (latency, token count) without any
configuration required. Subclass this to add a new embedding provider.
"""

import logging
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ..core.interfaces import EmbeddingProvider
from ..core.tracing import TraceType, get_trace_manager

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(EmbeddingProvider):
    """
    Base class for embedding providers with caching and tracing.

    Subclasses implement:
    - ``dimensions`` property: return the vector dimensionality
    - ``_embed_text_impl(text)``: embed a single text via the provider API
    - ``_embed_texts_impl(texts)``: batch-embed via the provider API

    Public ``embed_text`` / ``embed_texts`` handle LRU caching, tracing, and
    delegate to the ``_impl`` methods for the actual API call.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        cache_size: int = 2048,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self._cache_size = cache_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()

        self.trace_manager = get_trace_manager()
        self.provider_name = self.__class__.__name__.replace("EmbeddingProvider", "").lower()
        self.agent_id: Optional[str] = None

        # Accumulated metrics
        self._total_calls = 0

        logger.debug(
            f"Initialized {self.__class__.__name__} with model {model} "
            f"(cache_size={cache_size})"
        )

    def set_agent_id(self, agent_id: str) -> None:
        """Set agent ID for tracing context."""
        self.agent_id = agent_id

    # ------------------------------------------------------------------
    # Public API (cache-aware, traced)
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text with caching and tracing."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        async with self.trace_manager.span(
            operation_name="embedding",
            trace_type=TraceType.TOOL_EXECUTION,
            agent_id=self.agent_id,
            embedding_provider=self.provider_name,
            embedding_model=self.model,
            input_count=1,
        ):
            result = await self._embed_text_impl(text)

        self._cache_put(text, result)
        self._total_calls += 1
        return result

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with caching — only calls the API for uncached texts."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            async with self.trace_manager.span(
                operation_name="embedding_batch",
                trace_type=TraceType.TOOL_EXECUTION,
                agent_id=self.agent_id,
                embedding_provider=self.provider_name,
                embedding_model=self.model,
                input_count=len(uncached_texts),
            ):
                embeddings = await self._embed_texts_impl(uncached_texts)

            for j, emb in enumerate(embeddings):
                idx = uncached_indices[j]
                results[idx] = emb
                self._cache_put(uncached_texts[j], emb)

            self._total_calls += 1

        return results  # type: ignore[return-value]

    async def __call__(self, text: str) -> List[float]:
        """Make the provider callable for compatibility with ``embedding_fn`` parameters."""
        return await self.embed_text(text)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    async def _embed_text_impl(self, text: str) -> List[float]:
        """Provider-specific single-text embedding. No caching needed."""
        ...

    @abstractmethod
    async def _embed_texts_impl(self, texts: List[str]) -> List[List[float]]:
        """Provider-specific batch embedding. No caching needed."""
        ...

    # ------------------------------------------------------------------
    # LRU cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[List[float]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: List[float]) -> None:
        if self._cache_size <= 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "dimensions": self.dimensions,
            "cache_size": self._cache_size,
            "cache_entries": len(self._cache),
            "total_api_calls": self._total_calls,
        }
