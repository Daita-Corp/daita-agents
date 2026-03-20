"""
Unit tests for QdrantPlugin.

Tests that only _original_id is stripped from payloads (not all _ prefixed keys),
without a real Qdrant connection.
"""

import pytest
from unittest.mock import MagicMock
from daita.plugins.qdrant import QdrantPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin():
    plugin = QdrantPlugin(collection="test_col", url="http://localhost:6333")
    plugin._client = MagicMock()
    return plugin


class FakeScoredPoint:
    """Minimal stand-in for qdrant_client ScoredPoint."""

    def __init__(self, id, score, payload, vector=None):
        self.id = id
        self.score = score
        self.payload = payload
        self.vector = vector


def _mock_search(plugin, points):
    result = MagicMock()
    result.points = points
    plugin._client.query_points = MagicMock(return_value=result)


# ---------------------------------------------------------------------------
# Payload filtering — _original_id only
# ---------------------------------------------------------------------------


async def test_original_id_stripped_from_payload():
    plugin = make_plugin()
    _mock_search(plugin, [
        FakeScoredPoint(
            id="uuid-1",
            score=0.9,
            payload={"_original_id": "my-id", "name": "Alice", "age": 30},
        )
    ])

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert "_original_id" not in results[0]["payload"]
    assert results[0]["payload"]["name"] == "Alice"
    assert results[0]["payload"]["age"] == 30


async def test_other_underscore_keys_preserved():
    """Keys starting with _ other than _original_id must NOT be stripped."""
    plugin = make_plugin()
    _mock_search(plugin, [
        FakeScoredPoint(
            id="uuid-2",
            score=0.8,
            payload={
                "_original_id": "ext-99",
                "_source": "import_job",
                "_version": 3,
                "title": "Engineer",
            },
        )
    ])

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    payload = results[0]["payload"]
    assert "_original_id" not in payload
    assert payload["_source"] == "import_job"
    assert payload["_version"] == 3
    assert payload["title"] == "Engineer"


async def test_original_id_used_as_result_id():
    """_original_id in payload should become the id in the result."""
    plugin = make_plugin()
    _mock_search(plugin, [
        FakeScoredPoint(
            id="internal-uuid",
            score=0.95,
            payload={"_original_id": "human-readable-id", "data": "x"},
        )
    ])

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert results[0]["id"] == "human-readable-id"


async def test_no_original_id_uses_internal_id():
    plugin = make_plugin()
    _mock_search(plugin, [
        FakeScoredPoint(
            id="internal-uuid",
            score=0.7,
            payload={"name": "Bob"},
        )
    ])

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert results[0]["id"] == "internal-uuid"


async def test_payload_without_underscore_keys_unchanged():
    plugin = make_plugin()
    _mock_search(plugin, [
        FakeScoredPoint(
            id="abc",
            score=0.6,
            payload={"city": "NYC", "score": 99},
        )
    ])

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert results[0]["payload"] == {"city": "NYC", "score": 99}


async def test_empty_payload_handled():
    """Empty dict payload is falsy — plugin omits the payload key entirely."""
    plugin = make_plugin()
    _mock_search(plugin, [
        FakeScoredPoint(id="x", score=0.5, payload={})
    ])

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    # Empty payload {} is falsy, so the plugin skips adding it — no KeyError
    assert results[0]["id"] == "x"
    assert "payload" not in results[0]
