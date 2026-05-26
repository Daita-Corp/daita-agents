"""
Tests for the daita.embeddings module.

Covers:
- BaseEmbeddingProvider LRU cache behaviour
- MockEmbeddingProvider determinism and dimensions
- Factory: known providers, unknown providers, custom registration
- Dimension validation in SQLiteVectorSearch
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from daita.embeddings.base import BaseEmbeddingProvider
from daita.embeddings.factory import (
    PROVIDER_REGISTRY,
    create_embedding_provider,
    register_embedding_provider,
)
from daita.embeddings.mock import MockEmbeddingProvider

# ---------------------------------------------------------------------------
# MockEmbeddingProvider
# ---------------------------------------------------------------------------


class TestMockEmbeddingProvider:
    async def test_deterministic(self):
        provider = MockEmbeddingProvider(dim=8)
        v1 = await provider.embed_text("hello")
        v2 = await provider.embed_text("hello")
        assert v1 == v2

    async def test_different_texts_differ(self):
        provider = MockEmbeddingProvider(dim=8)
        v1 = await provider.embed_text("hello")
        v2 = await provider.embed_text("world")
        assert v1 != v2

    async def test_correct_dimensions(self):
        for dim in (8, 1536, 3072):
            provider = MockEmbeddingProvider(dim=dim)
            vec = await provider.embed_text("test")
            assert len(vec) == dim

    async def test_batch(self):
        provider = MockEmbeddingProvider(dim=8)
        results = await provider.embed_texts(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 8 for v in results)

    async def test_callable(self):
        provider = MockEmbeddingProvider(dim=8)
        vec = await provider("hello")
        assert len(vec) == 8

    def test_dimensions_property(self):
        provider = MockEmbeddingProvider(dim=256)
        assert provider.dimensions == 256


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------


class TestBaseEmbeddingProviderCache:
    def _cached_provider(self, dim=4, cache_size=2048):
        """Create a MockEmbeddingProvider with caching enabled."""
        p = MockEmbeddingProvider(dim=dim)
        p._cache_size = cache_size  # override the 0 default from Mock
        return p

    async def test_cache_hit_avoids_impl(self):
        provider = self._cached_provider(dim=4)
        provider._cache_put("cached_text", [1.0, 2.0, 3.0, 4.0])

        result = await provider.embed_text("cached_text")
        assert result == [1.0, 2.0, 3.0, 4.0]

    async def test_lru_eviction(self):
        provider = self._cached_provider(dim=4, cache_size=2)

        await provider.embed_text("a")
        await provider.embed_text("b")
        await provider.embed_text("c")  # should evict "a"

        assert provider._cache_get("a") is None
        assert provider._cache_get("b") is not None
        assert provider._cache_get("c") is not None

    async def test_cache_access_refreshes(self):
        provider = self._cached_provider(dim=4, cache_size=2)

        await provider.embed_text("a")
        await provider.embed_text("b")
        await provider.embed_text("a")  # refresh "a"
        await provider.embed_text("c")  # should evict "b", not "a"

        assert provider._cache_get("a") is not None
        assert provider._cache_get("b") is None
        assert provider._cache_get("c") is not None

    async def test_batch_uses_cache(self):
        provider = self._cached_provider(dim=4)
        first = await provider.embed_text("hello")

        results = await provider.embed_texts(["hello", "world"])
        assert results[0] == first  # from cache
        assert len(results[1]) == 4


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestEmbeddingFactory:
    def test_create_mock(self):
        provider = create_embedding_provider("mock", model="test-model")
        assert isinstance(provider, MockEmbeddingProvider)

    def test_unknown_provider_raises(self):
        from daita.core.exceptions import ConfigError

        with pytest.raises(ConfigError, match="Unsupported embedding provider"):
            create_embedding_provider("nonexistent")

    def test_register_custom(self):
        class CustomProvider(BaseEmbeddingProvider):
            @property
            def dimensions(self):
                return 128

            async def _embed_text_impl(self, text):
                return [0.0] * 128

            async def _embed_texts_impl(self, texts):
                return [[0.0] * 128 for _ in texts]

        register_embedding_provider("custom_test", CustomProvider)
        assert "custom_test" in PROVIDER_REGISTRY

        provider = create_embedding_provider("custom_test", model="v1")
        assert provider.dimensions == 128

        # Clean up
        del PROVIDER_REGISTRY["custom_test"]

    def test_case_insensitive(self):
        provider = create_embedding_provider("Mock", model="test")
        assert isinstance(provider, MockEmbeddingProvider)


# ---------------------------------------------------------------------------
# Dimension validation in SQLiteVectorSearch
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    def test_first_run_stores_dimension(self, tmp_path):
        import sqlite3

        from daita.plugins.memory.search import SQLiteVectorSearch

        embedder = MockEmbeddingProvider(dim=256)
        search = SQLiteVectorSearch(tmp_path / "test.db", embedder)

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM embedding_meta WHERE key = 'embedding_dim'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert int(row[0]) == 256

    def test_matching_dimension_succeeds(self, tmp_path):
        from daita.plugins.memory.search import SQLiteVectorSearch

        db_path = tmp_path / "test.db"
        embedder = MockEmbeddingProvider(dim=256)

        # First init
        SQLiteVectorSearch(db_path, embedder)
        # Second init with same dimensions — should not raise
        SQLiteVectorSearch(db_path, embedder)

    def test_mismatched_dimension_raises(self, tmp_path):
        from daita.plugins.memory.search import SQLiteVectorSearch

        db_path = tmp_path / "test.db"

        # First init with 256 dims
        SQLiteVectorSearch(db_path, MockEmbeddingProvider(dim=256))

        # Second init with different dims — should raise
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            SQLiteVectorSearch(db_path, MockEmbeddingProvider(dim=1536))


# ---------------------------------------------------------------------------
# Info / introspection
# ---------------------------------------------------------------------------


class TestProviderInfo:
    def test_info_fields(self):
        provider = MockEmbeddingProvider(dim=1536)
        info = provider.info
        assert info["provider"] == "mock"
        assert info["dimensions"] == 1536
        assert "cache_size" in info
        assert "total_api_calls" in info
