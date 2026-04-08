"""
Tests for memory plugin performance optimizations:
- Parallel fact extraction in query_facts()
- Deferred contradiction checking (queued at remember, processed at stop)
- Background eager fact extraction during remember()
- Graceful degradation when LLM calls fail
"""

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.memory.contradiction import ConflictResult
from daita.plugins.memory.fact_extractor import ExtractedFact, FactExtractor
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.memory_tools import handle_query_facts, handle_remember
from daita.plugins.memory.metadata import MemoryMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_test_embedder = MockEmbeddingProvider(dim=8)


def _make_backend(tmp_path: Path) -> LocalMemoryBackend:
    return LocalMemoryBackend(
        workspace="test",
        agent_id="test_agent",
        scope="project",
        base_dir=tmp_path,
        embedder=_test_embedder,
    )


def _make_plugin(tmp_path: Path, **overrides):
    """Build a minimal MemoryPlugin-like object for tool handler tests."""
    backend = _make_backend(tmp_path)

    plugin = MagicMock()
    plugin.backend = backend
    plugin._fact_extractor = overrides.get("fact_extractor", None)
    plugin._working_memory = overrides.get("working_memory", None)
    plugin._memory_graph = overrides.get("memory_graph", None)
    plugin._checker = overrides.get("checker", None)
    plugin.enable_fact_extraction = overrides.get("enable_fact_extraction", False)
    plugin.default_ttl_days = overrides.get("default_ttl_days", None)
    plugin._pending_contradiction_checks = []
    plugin._background_tasks = []
    return plugin


def _make_mock_fact_extractor(facts_per_call: Optional[List[ExtractedFact]] = None):
    """Create a mock FactExtractor that returns canned facts."""
    extractor = MagicMock(spec=FactExtractor)

    async def _extract(content):
        await asyncio.sleep(0.01)  # Simulate small latency
        if facts_per_call is not None:
            return facts_per_call
        return [
            ExtractedFact(
                entity="test_entity",
                relation="has",
                value="test_value",
            )
        ]

    extractor.extract = _extract
    extractor.facts_to_metadata = FactExtractor.facts_to_metadata
    return extractor


def _make_mock_checker(status="no_conflict", reason="test reason"):
    """Create a mock ContradictionChecker that returns a fixed result."""
    checker = MagicMock()

    async def _check(content, importance):
        await asyncio.sleep(0.01)
        return ConflictResult(
            status=status,
            conflicting_chunk_id="old_chunk_123" if status != "no_conflict" else None,
            conflicting_content="old content" if status != "no_conflict" else None,
            conflict_reason=reason if status != "no_conflict" else None,
        )

    checker.check = _check
    return checker


def _read_chunk_metadata(backend, chunk_id):
    """Read raw metadata JSON for a chunk from SQLite."""
    conn = sqlite3.connect(str(backend.vector_db))
    cursor = conn.cursor()
    cursor.execute("SELECT metadata FROM chunks WHERE chunk_id = ?", (chunk_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row[0])


# ---------------------------------------------------------------------------
# Phase 1: Parallel fact extraction
# ---------------------------------------------------------------------------


class TestParallelFactExtraction:
    async def test_extracts_all_chunks_concurrently(self, tmp_path):
        """Verify all unextracted chunks get processed."""
        backend = _make_backend(tmp_path)

        # Store 5 chunks with _facts_extracted=False
        chunk_ids = []
        for i in range(5):
            meta = MemoryMetadata(
                content=f"fact number {i}", importance=0.7, source="test"
            )
            result = await backend.remember(
                f"fact number {i}",
                category="test",
                metadata=meta,
                extra_metadata={"_facts_extracted": False},
            )
            chunk_ids.append(result["chunk_id"])

        extractor = _make_mock_fact_extractor()
        plugin = _make_plugin(tmp_path, fact_extractor=extractor)
        plugin.backend = backend

        await handle_query_facts(plugin, entity="test_entity")

        # All chunks should now be marked as extracted
        for cid in chunk_ids:
            meta = _read_chunk_metadata(backend, cid)
            assert meta["_facts_extracted"] is True
            assert "extracted_facts" in meta

    async def test_failed_extraction_does_not_block_others(self, tmp_path):
        """One failing extraction should not prevent others from completing."""
        backend = _make_backend(tmp_path)

        chunk_ids = []
        for i in range(3):
            meta = MemoryMetadata(
                content=f"fact {i}", importance=0.7, source="test"
            )
            result = await backend.remember(
                f"fact {i}",
                category="test",
                metadata=meta,
                extra_metadata={"_facts_extracted": False},
            )
            chunk_ids.append(result["chunk_id"])

        call_count = 0

        async def _flaky_extract(content):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated LLM failure")
            return [ExtractedFact(entity="e", relation="r", value="v")]

        extractor = MagicMock(spec=FactExtractor)
        extractor.extract = _flaky_extract
        extractor.facts_to_metadata = FactExtractor.facts_to_metadata

        plugin = _make_plugin(tmp_path, fact_extractor=extractor)
        plugin.backend = backend

        # Should not raise
        await handle_query_facts(plugin, entity="e")

        # At least 2 of 3 chunks should be extracted
        extracted_count = sum(
            1
            for cid in chunk_ids
            if _read_chunk_metadata(backend, cid).get("_facts_extracted") is True
        )
        assert extracted_count >= 2


# ---------------------------------------------------------------------------
# Phase 2: Deferred contradiction checking
# ---------------------------------------------------------------------------


class TestDeferredContradictionChecking:
    async def test_remember_queues_check_without_blocking(self, tmp_path):
        """High-importance remember should store immediately and queue check."""
        checker = _make_mock_checker(status="no_conflict")
        plugin = _make_plugin(tmp_path, checker=checker)

        result = await handle_remember(
            plugin, content="important finding", importance=0.9, category="test"
        )

        assert result["status"] == "success"
        assert len(plugin._pending_contradiction_checks) == 1
        cid, content, imp = plugin._pending_contradiction_checks[0]
        assert cid == result["chunk_id"]
        assert content == "important finding"
        assert imp == 0.9

    async def test_low_importance_skips_check(self, tmp_path):
        """Low-importance memories should not queue contradiction checks."""
        checker = _make_mock_checker()
        plugin = _make_plugin(tmp_path, checker=checker)

        result = await handle_remember(
            plugin, content="minor note", importance=0.3, category="test"
        )

        assert result["status"] == "success"
        assert len(plugin._pending_contradiction_checks) == 0

    async def test_no_checker_skips_queue(self, tmp_path):
        """Without a checker, no checks should be queued."""
        plugin = _make_plugin(tmp_path, checker=None)

        result = await handle_remember(
            plugin, content="important but no checker", importance=0.9
        )

        assert result["status"] == "success"
        assert len(plugin._pending_contradiction_checks) == 0

    async def test_process_deferred_flags_contradiction(self, tmp_path):
        """Deferred processing should flag contradicted chunks."""
        from daita.plugins.memory.memory_plugin import MemoryPlugin

        checker = _make_mock_checker(
            status="contradiction", reason="directly conflicts"
        )
        plugin = _make_plugin(tmp_path, checker=checker)
        backend = plugin.backend

        # Store a memory and queue the check
        result = await handle_remember(
            plugin, content="new claim", importance=0.9, category="test"
        )
        chunk_id = result["chunk_id"]
        assert len(plugin._pending_contradiction_checks) == 1

        # Simulate the deferred processing from on_agent_stop
        real_plugin = MagicMock(spec=MemoryPlugin)
        real_plugin.backend = backend
        real_plugin._checker = checker
        real_plugin._pending_contradiction_checks = (
            plugin._pending_contradiction_checks
        )
        await MemoryPlugin._process_deferred_contradictions(real_plugin)

        # Chunk should be flagged, not deleted
        meta = _read_chunk_metadata(backend, chunk_id)
        assert meta is not None, "Chunk should NOT be deleted"
        assert meta.get("flagged_contradiction") is True
        assert meta.get("conflict_reason") == "directly conflicts"
        assert meta.get("_contradiction_checked") is True

    async def test_process_deferred_handles_evolution(self, tmp_path):
        """Evolution should trigger update_memory during deferred processing."""
        from daita.plugins.memory.memory_plugin import MemoryPlugin

        checker = _make_mock_checker(status="evolution", reason="state changed")
        plugin = _make_plugin(tmp_path, checker=checker)
        backend = plugin.backend

        # Store the "old" memory that will be the evolution target
        old_meta = MemoryMetadata(
            content="old content", importance=0.8, source="test"
        )
        await backend.remember(
            "old content", category="test", metadata=old_meta
        )

        # Store the new memory
        result = await handle_remember(
            plugin, content="updated claim", importance=0.9, category="test"
        )
        chunk_id = result["chunk_id"]

        # Process deferred checks
        real_plugin = MagicMock(spec=MemoryPlugin)
        real_plugin.backend = backend
        real_plugin._checker = checker
        real_plugin._pending_contradiction_checks = (
            plugin._pending_contradiction_checks
        )
        await MemoryPlugin._process_deferred_contradictions(real_plugin)

        # The new chunk should be marked as checked
        meta = _read_chunk_metadata(backend, chunk_id)
        assert meta.get("_contradiction_checked") is True
        assert meta.get("flagged_contradiction") is not True

    async def test_deferred_check_failure_does_not_raise(self, tmp_path):
        """Checker failure during deferred processing should not crash."""
        from daita.plugins.memory.memory_plugin import MemoryPlugin

        checker = MagicMock()

        async def _failing_check(content, importance):
            raise RuntimeError("LLM API down")

        checker.check = _failing_check
        plugin = _make_plugin(tmp_path, checker=checker)

        result = await handle_remember(
            plugin, content="will fail check", importance=0.9, category="test"
        )

        real_plugin = MagicMock(spec=MemoryPlugin)
        real_plugin.backend = plugin.backend
        real_plugin._checker = checker
        real_plugin._pending_contradiction_checks = (
            plugin._pending_contradiction_checks
        )

        # Should not raise
        await MemoryPlugin._process_deferred_contradictions(real_plugin)


# ---------------------------------------------------------------------------
# Phase 3: Background eager fact extraction
# ---------------------------------------------------------------------------


class TestBackgroundEagerExtraction:
    async def test_remember_creates_background_task(self, tmp_path):
        """remember() with fact extractor should fire background extraction."""
        extractor = _make_mock_fact_extractor()
        plugin = _make_plugin(tmp_path, fact_extractor=extractor)

        result = await handle_remember(
            plugin, content="extractable content", importance=0.6, category="test"
        )

        assert result["status"] == "success"
        assert len(plugin._background_tasks) == 1

        # Wait for the background task
        await asyncio.gather(*plugin._background_tasks, return_exceptions=True)

        # Chunk should now have facts extracted
        meta = _read_chunk_metadata(plugin.backend, result["chunk_id"])
        assert meta["_facts_extracted"] is True
        assert len(meta["extracted_facts"]) == 1
        assert meta["extracted_facts"][0]["entity"] == "test_entity"

    async def test_background_extraction_failure_leaves_flag_false(self, tmp_path):
        """Failed background extraction should leave _facts_extracted=False."""
        extractor = MagicMock(spec=FactExtractor)

        async def _failing_extract(content):
            raise RuntimeError("extraction boom")

        extractor.extract = _failing_extract
        extractor.facts_to_metadata = FactExtractor.facts_to_metadata

        plugin = _make_plugin(tmp_path, fact_extractor=extractor)

        result = await handle_remember(
            plugin, content="will fail extraction", importance=0.5, category="test"
        )

        # Wait for background task
        await asyncio.gather(*plugin._background_tasks, return_exceptions=True)

        # Chunk should still have _facts_extracted=False (for lazy retry)
        meta = _read_chunk_metadata(plugin.backend, result["chunk_id"])
        assert meta["_facts_extracted"] is False

    async def test_no_fact_extractor_no_background_task(self, tmp_path):
        """Without fact extractor, no background tasks should be created."""
        plugin = _make_plugin(tmp_path, fact_extractor=None)

        result = await handle_remember(
            plugin, content="no extraction", importance=0.5, category="test"
        )

        assert result["status"] == "success"
        assert len(plugin._background_tasks) == 0
