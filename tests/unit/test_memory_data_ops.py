"""
Tests for memory plugin data operation improvements:
- query_facts (structured fact querying)
- remember batch mode
- list_memories with include_stats
- recall with since/before temporal filtering
- pinned memories always-inject in on_before_run
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.memory_plugin import MemoryPlugin, _parse_time_param
from daita.plugins.memory.metadata import MemoryMetadata
from daita.plugins.memory.search import SQLiteVectorSearch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared mock embedder for tests — small dimension for speed
_test_embedder = MockEmbeddingProvider(dim=8)


def _make_backend(tmp_path: Path) -> LocalMemoryBackend:
    """Create a LocalMemoryBackend with mock embeddings (no OpenAI calls)."""
    return LocalMemoryBackend(
        workspace="test",
        agent_id="test_agent",
        scope="project",
        base_dir=tmp_path,
        embedder=_test_embedder,
    )


def _make_search(db_path: Path) -> SQLiteVectorSearch:
    """Create a SQLiteVectorSearch with mock embedding calls."""
    return SQLiteVectorSearch(db_path, _test_embedder)




# ---------------------------------------------------------------------------
# _parse_time_param
# ---------------------------------------------------------------------------


class TestParseTimeParam:
    def test_none(self):
        assert _parse_time_param(None) is None

    def test_iso_passthrough(self):
        iso = "2026-04-01T00:00:00"
        assert _parse_time_param(iso) == iso

    def test_hours_relative(self):
        result = _parse_time_param("24h")
        parsed = datetime.fromisoformat(result)
        assert (datetime.now() - parsed).total_seconds() < 86400 + 5

    def test_days_relative(self):
        result = _parse_time_param("7d")
        parsed = datetime.fromisoformat(result)
        diff = (datetime.now() - parsed).days
        assert diff == 7 or diff == 6  # allow rounding


# ---------------------------------------------------------------------------
# Batch remember
# ---------------------------------------------------------------------------


class TestRememberBatch:
    async def test_batch_stores_multiple(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [
            {"content": "fact one", "importance": 0.8, "category": "test"},
            {"content": "fact two", "importance": 0.6, "category": "test"},
            {"content": "fact three"},
        ]
        result = await backend.remember_batch(items)
        assert result["status"] == "success"
        assert result["stored"] == 3
        assert result["skipped"] == 0
        assert len(result["items"]) == 3

    async def test_batch_skips_existing(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [{"content": "duplicate fact"}]
        await backend.remember_batch(items)
        result = await backend.remember_batch(items)
        assert result["stored"] == 0
        assert result["skipped"] == 1

    async def test_batch_with_extra_metadata(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [{"content": "fact with extras", "importance": 0.7}]
        extra = [{"extracted_facts": [{"entity": "x", "relation": "y", "value": "z"}]}]
        result = await backend.remember_batch(items, extra_metadata_list=extra)
        assert result["stored"] == 1

        # Verify metadata was stored
        conn = sqlite3.connect(str(backend.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM chunks")
        row = cursor.fetchone()
        conn.close()
        meta = json.loads(row[0])
        assert "extracted_facts" in meta


# ---------------------------------------------------------------------------
# query_facts
# ---------------------------------------------------------------------------


class TestQueryFacts:
    async def test_query_by_entity(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [
            {"content": "users table has email column", "importance": 0.7},
            {"content": "orders table has total column", "importance": 0.7},
        ]
        extra = [
            {
                "extracted_facts": [
                    {
                        "entity": "users",
                        "relation": "has column",
                        "value": "email",
                        "temporal_context": None,
                    },
                ]
            },
            {
                "extracted_facts": [
                    {
                        "entity": "orders",
                        "relation": "has column",
                        "value": "total",
                        "temporal_context": None,
                    },
                ]
            },
        ]
        await backend.remember_batch(items, extra_metadata_list=extra)

        results = await backend.query_facts(entity="users")
        assert len(results) == 1
        assert results[0]["entity"] == "users"
        assert results[0]["value"] == "email"

    async def test_query_by_relation(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [{"content": "FK: orders.user_id -> users.id", "importance": 0.8}]
        extra = [
            {
                "extracted_facts": [
                    {
                        "entity": "orders.user_id",
                        "relation": "FK references",
                        "value": "users.id",
                        "temporal_context": None,
                    },
                ]
            }
        ]
        await backend.remember_batch(items, extra_metadata_list=extra)

        results = await backend.query_facts(relation="FK")
        assert len(results) == 1
        assert results[0]["relation"] == "FK references"

    async def test_query_no_facts(self, tmp_path):
        backend = _make_backend(tmp_path)
        results = await backend.query_facts(entity="nonexistent")
        assert results == []

    async def test_query_combined_filters(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [{"content": "schema info", "importance": 0.7}]
        extra = [
            {
                "extracted_facts": [
                    {
                        "entity": "users",
                        "relation": "has column",
                        "value": "email",
                        "temporal_context": None,
                    },
                    {
                        "entity": "users",
                        "relation": "has constraint",
                        "value": "UNIQUE on email",
                        "temporal_context": None,
                    },
                ]
            }
        ]
        await backend.remember_batch(items, extra_metadata_list=extra)

        results = await backend.query_facts(entity="users", relation="has constraint")
        assert len(results) == 1
        assert results[0]["value"] == "UNIQUE on email"


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    async def test_stats_empty(self, tmp_path):
        backend = _make_backend(tmp_path)
        stats = await backend.get_stats()
        assert stats["total_memories"] == 0
        assert stats["categories"] == {}
        assert stats["pinned_count"] == 0

    async def test_stats_with_data(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [
            {"content": "fact a", "importance": 0.8, "category": "schema"},
            {"content": "fact b", "importance": 0.6, "category": "schema"},
            {"content": "fact c", "importance": 0.5, "category": "rules"},
        ]
        await backend.remember_batch(items)

        stats = await backend.get_stats()
        assert stats["total_memories"] == 3
        assert "schema" in stats["categories"]
        assert stats["categories"]["schema"]["count"] == 2
        assert "rules" in stats["categories"]
        assert stats["categories"]["rules"]["count"] == 1
        assert stats["oldest"] is not None
        assert stats["newest"] is not None


# ---------------------------------------------------------------------------
# get_pinned_memories
# ---------------------------------------------------------------------------


class TestGetPinnedMemories:
    async def test_no_pinned(self, tmp_path):
        backend = _make_backend(tmp_path)
        items = [{"content": "not pinned", "importance": 0.5}]
        await backend.remember_batch(items)
        pinned = await backend.get_pinned_memories()
        assert pinned == []

    async def test_pinned_returned(self, tmp_path):
        backend = _make_backend(tmp_path)
        # Store a memory and then pin it via metadata update
        metadata = MemoryMetadata(
            content="all tables must have created_at",
            importance=1.0,
            source="user_explicit",
            pinned=True,
            category="rules",
        )
        await backend.search.store_chunk("all tables must have created_at", metadata)

        pinned = await backend.get_pinned_memories()
        assert len(pinned) == 1
        assert "created_at" in pinned[0]["content"]


# ---------------------------------------------------------------------------
# Temporal filtering in recall
# ---------------------------------------------------------------------------


class TestTemporalFiltering:
    async def test_since_filters_old(self, tmp_path):
        backend = _make_backend(tmp_path)

        # Store a memory with an old timestamp
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        metadata_old = MemoryMetadata(
            content="old fact",
            importance=0.8,
            created_at=datetime.now() - timedelta(days=10),
        )
        await backend.search.store_chunk("old fact", metadata_old, chunk_id="old_chunk")

        # Store a recent memory
        metadata_new = MemoryMetadata(
            content="new fact",
            importance=0.8,
        )
        await backend.search.store_chunk("new fact", metadata_new, chunk_id="new_chunk")

        # Recall with since=2 days ago should only return the new fact
        since = (datetime.now() - timedelta(days=2)).isoformat()
        results = await backend.recall(
            "fact", limit=10, score_threshold=0.0, since=since
        )
        contents = [r["content"] for r in results]
        assert "new fact" in contents
        assert "old fact" not in contents

    async def test_before_filters_new(self, tmp_path):
        backend = _make_backend(tmp_path)

        # Store a memory with an old timestamp
        metadata_old = MemoryMetadata(
            content="old fact",
            importance=0.8,
            created_at=datetime.now() - timedelta(days=10),
        )
        await backend.search.store_chunk(
            "old fact", metadata_old, chunk_id="old_chunk2"
        )

        # Store a recent memory
        metadata_new = MemoryMetadata(
            content="new fact",
            importance=0.8,
        )
        await backend.search.store_chunk(
            "new fact", metadata_new, chunk_id="new_chunk2"
        )

        # Recall with before=5 days ago should only return the old fact
        before = (datetime.now() - timedelta(days=5)).isoformat()
        results = await backend.recall(
            "fact", limit=10, score_threshold=0.0, before=before
        )
        contents = [r["content"] for r in results]
        assert "old fact" in contents
        assert "new fact" not in contents


# ---------------------------------------------------------------------------
# embed_texts batch
# ---------------------------------------------------------------------------


class TestEmbedTextsBatch:
    async def test_batch_delegates_to_provider(self, tmp_path):
        search = _make_search(tmp_path / "test_batch.db")

        results = await search.embed_texts(["hello", "world"])
        assert len(results) == 2
        assert all(len(v) == 8 for v in results)

    async def test_batch_deterministic(self, tmp_path):
        search = _make_search(tmp_path / "test_deterministic.db")

        r1 = await search.embed_texts(["hello", "world"])
        r2 = await search.embed_texts(["hello", "world"])
        assert r1 == r2


# ---------------------------------------------------------------------------
# Auto-classification
# ---------------------------------------------------------------------------


class TestAutoClassify:
    def test_infer_category_rule(self):
        from daita.plugins.memory.auto_classify import infer_category

        assert infer_category("All databases must have encryption") == "rule"
        assert infer_category("Never expose credentials") == "rule"
        assert infer_category("Backups are mandatory for compliance") == "rule"

    def test_infer_category_schema(self):
        from daita.plugins.memory.auto_classify import infer_category

        assert infer_category("FK: orders.user_id -> users.id") == "schema"
        assert infer_category("Added foreign key constraint") == "schema"
        assert infer_category("Table users has 5 columns") == "schema"

    def test_infer_category_incident(self):
        from daita.plugins.memory.auto_classify import infer_category

        assert infer_category("Production outage at 3pm") == "incident"
        assert infer_category("Bug in payment processing") == "incident"

    def test_infer_category_none(self):
        from daita.plugins.memory.auto_classify import infer_category

        assert infer_category("The weather is nice today") is None
        assert infer_category("General observation about the system") is None

    def test_infer_importance_production(self):
        from daita.plugins.memory.auto_classify import infer_importance

        assert infer_importance("RDS instance in production") == 0.8
        assert infer_importance("prod database discovered") == 0.8

    def test_infer_importance_critical(self):
        from daita.plugins.memory.auto_classify import infer_importance

        assert infer_importance("Critical security vulnerability found") == 0.9

    def test_infer_importance_dev(self):
        from daita.plugins.memory.auto_classify import infer_importance

        assert infer_importance("dev environment database") == 0.4
        assert infer_importance("local test server") == 0.4

    def test_infer_importance_staging(self):
        from daita.plugins.memory.auto_classify import infer_importance

        assert infer_importance("staging environment RDS") == 0.6

    def test_infer_importance_default(self):
        from daita.plugins.memory.auto_classify import infer_importance

        assert infer_importance("General note about something") == 0.5

    def test_infer_importance_rule_bump(self):
        from daita.plugins.memory.auto_classify import infer_importance

        result = infer_importance("All services must use TLS")
        assert result == 0.6  # 0.5 + 0.1 from rule keyword


# ---------------------------------------------------------------------------
# TTL and pruning
# ---------------------------------------------------------------------------


class TestTTL:
    def test_should_prune_with_ttl_expired(self):
        meta = MemoryMetadata(
            content="old fact",
            importance=0.9,
            ttl_days=7,
            created_at=datetime.now() - timedelta(days=10),
        )
        assert meta.should_prune() is True

    def test_should_prune_with_ttl_not_expired(self):
        meta = MemoryMetadata(
            content="recent fact",
            importance=0.9,
            ttl_days=30,
            created_at=datetime.now() - timedelta(days=5),
        )
        assert meta.should_prune() is False

    def test_should_prune_pinned_ignores_ttl(self):
        meta = MemoryMetadata(
            content="pinned rule",
            importance=1.0,
            pinned=True,
            ttl_days=1,
            created_at=datetime.now() - timedelta(days=100),
        )
        assert meta.should_prune() is False

    def test_should_prune_no_ttl_uses_age_rules(self):
        meta = MemoryMetadata(
            content="old low-importance",
            importance=0.2,
            ttl_days=None,
            created_at=datetime.now() - timedelta(days=100),
        )
        assert meta.should_prune() is True

    def test_ttl_days_in_serialization(self):
        meta = MemoryMetadata(content="test", ttl_days=30)
        d = meta.to_dict()
        assert d["ttl_days"] == 30

        restored = MemoryMetadata.from_dict(d)
        assert restored.ttl_days == 30


class TestPruning:
    async def test_prune_expired_deletes_ttl_chunks(self, tmp_path):
        backend = _make_backend(tmp_path)

        # Store a memory with expired TTL
        expired_meta = MemoryMetadata(
            content="ephemeral data",
            importance=0.5,
            ttl_days=1,
            created_at=datetime.now() - timedelta(days=5),
        )
        await backend.search.store_chunk(
            "ephemeral data", expired_meta, chunk_id="expired_chunk"
        )

        # Store a memory without TTL
        permanent_meta = MemoryMetadata(
            content="permanent fact",
            importance=0.8,
        )
        await backend.search.store_chunk(
            "permanent fact", permanent_meta, chunk_id="permanent_chunk"
        )

        deleted = await backend._prune_expired()
        assert deleted == 1

        # Verify only permanent remains
        stats = await backend.get_stats()
        assert stats["total_memories"] == 1

    async def test_enforce_size_limit(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.max_chunks = 3  # Low limit for testing

        # Store 5 memories
        items = [
            {"content": f"memory {i}", "importance": i * 0.2, "category": "test"}
            for i in range(5)
        ]
        await backend.remember_batch(items)

        await backend._enforce_size_limit()

        stats = await backend.get_stats()
        assert stats["total_memories"] == 3

    async def test_prune_combines_both(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.max_chunks = 10

        # Store an expired memory
        expired_meta = MemoryMetadata(
            content="expired",
            importance=0.5,
            ttl_days=1,
            created_at=datetime.now() - timedelta(days=5),
        )
        await backend.search.store_chunk("expired", expired_meta, chunk_id="exp")

        # Store a normal memory
        await backend.remember_batch(
            [
                {"content": "normal fact", "importance": 0.7},
            ]
        )

        await backend.prune()

        stats = await backend.get_stats()
        assert stats["total_memories"] == 1


# ---------------------------------------------------------------------------
# Vectorized search (regression)
# ---------------------------------------------------------------------------


class TestVectorizedSearch:
    async def test_search_returns_results(self, tmp_path):
        """Vectorized search should produce valid results."""
        backend = _make_backend(tmp_path)

        await backend.remember_batch(
            [
                {"content": "PostgreSQL database in production", "importance": 0.8},
                {"content": "S3 bucket for staging data", "importance": 0.6},
                {"content": "DynamoDB table for sessions", "importance": 0.7},
            ]
        )

        results = await backend.recall("database", limit=10, score_threshold=0.0)
        assert len(results) > 0
        assert all("score" in r for r in results)
        # Results should be sorted by score descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_norm_column_stored(self, tmp_path):
        """store_chunk should persist pre-computed norms."""
        backend = _make_backend(tmp_path)

        meta = MemoryMetadata(content="test norm", importance=0.5)
        await backend.search.store_chunk("test norm", meta, chunk_id="norm_test")

        conn = sqlite3.connect(str(backend.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT norm FROM embeddings WHERE chunk_id = 'norm_test'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] is not None
        assert row[0] > 0


# ---------------------------------------------------------------------------
# Lazy fact extraction
# ---------------------------------------------------------------------------


class TestLazyFactExtraction:
    async def test_get_unextracted_chunks(self, tmp_path):
        backend = _make_backend(tmp_path)

        # Store with _facts_extracted=False flag
        await backend.remember_batch(
            [{"content": "some fact", "importance": 0.5}],
            extra_metadata_list=[{"_facts_extracted": False}],
        )

        unextracted = await backend.get_unextracted_chunks()
        assert len(unextracted) == 1
        assert unextracted[0][1] == "some fact"

    async def test_already_extracted_not_returned(self, tmp_path):
        backend = _make_backend(tmp_path)

        # Store with facts already extracted
        await backend.remember_batch(
            [{"content": "extracted fact", "importance": 0.5}],
            extra_metadata_list=[
                {
                    "_facts_extracted": True,
                    "extracted_facts": [{"entity": "x", "relation": "y", "value": "z"}],
                }
            ],
        )

        unextracted = await backend.get_unextracted_chunks()
        assert len(unextracted) == 0
