"""
Tests for outcome-based learning (reinforcement).
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.metadata import MemoryMetadata
from daita.plugins.memory.search import SQLiteVectorSearch

_test_embedder = MockEmbeddingProvider(dim=8)


def _make_backend(tmp_path: Path) -> LocalMemoryBackend:
    return LocalMemoryBackend(
        workspace="test",
        agent_id="test_agent",
        scope="project",
        base_dir=tmp_path,
        embedder=_test_embedder,
    )


# ---------------------------------------------------------------------------
# MemoryMetadata reinforcement fields
# ---------------------------------------------------------------------------


class TestMetadataReinforcement:
    def test_default_reinforcements_is_none(self):
        meta = MemoryMetadata(content="test")
        assert meta.reinforcements is None
        assert meta.flagged_for_review is False

    def test_reinforcements_serialization(self):
        meta = MemoryMetadata(
            content="test",
            reinforcements=[
                {"outcome": "positive", "signal": 0.7, "timestamp": "2026-04-01"},
            ],
            flagged_for_review=True,
        )
        d = meta.to_dict()
        assert len(d["reinforcements"]) == 1
        assert d["flagged_for_review"] is True

        # Round-trip
        meta2 = MemoryMetadata.from_dict(d)
        assert len(meta2.reinforcements) == 1
        assert meta2.flagged_for_review is True

    def test_should_prune_negative_dominant(self):
        """Memories with >=70% negative reinforcements prune at half age."""
        meta = MemoryMetadata(
            content="test",
            importance=0.5,
            created_at=datetime.now() - timedelta(days=50),
            reinforcements=[
                {"outcome": "negative"},
                {"outcome": "negative"},
                {"outcome": "negative"},
                {"outcome": "positive"},
            ],
        )
        # 3/4 = 75% negative, age 50 > 45 (half of 90), importance 0.5 < 0.6
        assert meta.should_prune(max_age_days=90) is True

    def test_should_not_prune_positive_dominant(self):
        """Memories with mostly positive reinforcements should not prune early."""
        meta = MemoryMetadata(
            content="test",
            importance=0.5,
            created_at=datetime.now() - timedelta(days=50),
            reinforcements=[
                {"outcome": "positive"},
                {"outcome": "positive"},
                {"outcome": "positive"},
                {"outcome": "negative"},
            ],
        )
        assert meta.should_prune(max_age_days=90) is False

    def test_should_not_prune_few_reinforcements(self):
        """Need at least 3 reinforcements to trigger early pruning."""
        meta = MemoryMetadata(
            content="test",
            importance=0.5,
            created_at=datetime.now() - timedelta(days=50),
            reinforcements=[
                {"outcome": "negative"},
                {"outcome": "negative"},
            ],
        )
        assert meta.should_prune(max_age_days=90) is False

    def test_pinned_overrides_reinforcement(self):
        """Pinned memories should never prune regardless of reinforcement."""
        meta = MemoryMetadata(
            content="test",
            importance=0.5,
            pinned=True,
            created_at=datetime.now() - timedelta(days=100),
            reinforcements=[
                {"outcome": "negative"},
                {"outcome": "negative"},
                {"outcome": "negative"},
            ],
        )
        assert meta.should_prune(max_age_days=90) is False


# ---------------------------------------------------------------------------
# LocalMemoryBackend.append_reinforcement()
# ---------------------------------------------------------------------------


class TestBackendAppendReinforcement:
    async def test_append_reinforcement(self, tmp_path):
        backend = _make_backend(tmp_path)
        result = await backend.remember("PostgreSQL pool limit is 100")
        chunk_id = result["chunk_id"]

        reinforcement = {
            "outcome": "positive",
            "signal": 0.7,
            "timestamp": datetime.now().isoformat(),
            "context": "query succeeded",
        }
        await backend.append_reinforcement(chunk_id, reinforcement, importance_delta=0.07)

        # Verify metadata was updated
        conn = sqlite3.connect(str(backend.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()

        meta = json.loads(row[0])
        assert len(meta["reinforcements"]) == 1
        assert meta["reinforcements"][0]["outcome"] == "positive"
        assert meta["importance"] == pytest.approx(0.57, abs=0.01)

    async def test_append_multiple_reinforcements(self, tmp_path):
        backend = _make_backend(tmp_path)
        result = await backend.remember("test memory")
        chunk_id = result["chunk_id"]

        for outcome in ["positive", "negative", "positive"]:
            await backend.append_reinforcement(
                chunk_id,
                {"outcome": outcome, "signal": 0.5, "timestamp": "now"},
            )

        conn = sqlite3.connect(str(backend.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()

        meta = json.loads(row[0])
        assert len(meta["reinforcements"]) == 3

    async def test_importance_capped_at_1(self, tmp_path):
        backend = _make_backend(tmp_path)
        meta = MemoryMetadata(content="test", importance=0.95)
        result = await backend.remember("test", metadata=meta)
        chunk_id = result["chunk_id"]

        await backend.append_reinforcement(
            chunk_id,
            {"outcome": "positive", "signal": 1.0, "timestamp": "now"},
            importance_delta=0.1,
        )

        conn = sqlite3.connect(str(backend.vector_db))
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        conn.close()

        meta = json.loads(row[0])
        assert meta["importance"] <= 1.0


# ---------------------------------------------------------------------------
# LocalMemoryBackend.get_chunk()
# ---------------------------------------------------------------------------


class TestBackendGetChunk:
    async def test_get_existing_chunk(self, tmp_path):
        backend = _make_backend(tmp_path)
        result = await backend.remember("hello world")
        chunk_id = result["chunk_id"]

        chunk = await backend.get_chunk(chunk_id)
        assert chunk is not None
        assert chunk["chunk_id"] == chunk_id
        assert chunk["content"] == "hello world"

    async def test_get_missing_chunk(self, tmp_path):
        backend = _make_backend(tmp_path)
        chunk = await backend.get_chunk("nonexistent_id")
        assert chunk is None


# ---------------------------------------------------------------------------
# Score adjustments with reinforcement
# ---------------------------------------------------------------------------


class TestScoreAdjustmentsReinforcement:
    def test_positive_reinforcement_boosts_score(self):
        search = SQLiteVectorSearch(Path(tempfile.mktemp(suffix=".db")), _test_embedder)
        meta = {
            "importance": 0.5,
            "reinforcements": [
                {"outcome": "positive"},
                {"outcome": "positive"},
            ],
        }
        score_with = search._apply_score_adjustments(0.5, meta)
        score_without = search._apply_score_adjustments(0.5, {"importance": 0.5})
        assert score_with > score_without

    def test_negative_reinforcement_penalizes_score(self):
        search = SQLiteVectorSearch(Path(tempfile.mktemp(suffix=".db")), _test_embedder)
        meta = {
            "importance": 0.5,
            "reinforcements": [
                {"outcome": "negative"},
                {"outcome": "negative"},
            ],
        }
        score_with = search._apply_score_adjustments(0.5, meta)
        score_without = search._apply_score_adjustments(0.5, {"importance": 0.5})
        assert score_with < score_without
