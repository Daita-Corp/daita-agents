"""
Unit tests for MongoDBPlugin.

Tests $limit injection in aggregate, default limit on find,
and tool naming — without a real MongoDB connection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from daita.plugins.mongodb import MongoDBPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin():
    plugin = MongoDBPlugin(host="localhost", database="testdb")
    # Inject fake client/db so connect() is never called
    plugin._client = MagicMock()
    plugin._db = MagicMock()
    return plugin


def _mock_aggregate(plugin, results):
    """Wire _db[collection].aggregate(...).to_list() to return results."""
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=results)
    plugin._db.__getitem__ = MagicMock(
        return_value=MagicMock(aggregate=MagicMock(return_value=cursor))
    )


def _mock_find(plugin, results):
    """Wire _db[collection].find().to_list() to return results."""
    cursor = MagicMock()
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=results)
    plugin._db.__getitem__ = MagicMock(
        return_value=MagicMock(find=MagicMock(return_value=cursor))
    )


# ---------------------------------------------------------------------------
# Tool names
# ---------------------------------------------------------------------------


def test_tool_names_have_mongodb_prefix():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "mongodb_find" in names
    assert "mongodb_insert" in names
    assert "mongodb_update" in names
    assert "mongodb_delete" in names
    assert "mongodb_list_collections" in names
    assert "mongodb_count" in names


def test_no_unprefixed_tool_names():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "find_documents" not in names
    assert "insert_document" not in names


# ---------------------------------------------------------------------------
# Aggregate — $limit injection
# ---------------------------------------------------------------------------


async def test_aggregate_injects_limit_when_missing():
    plugin = make_plugin()
    captured = []

    async def fake_aggregate(collection, pipeline):
        captured.append(pipeline)
        return []

    plugin.aggregate = fake_aggregate

    await plugin._tool_aggregate(
        {"collection": "orders", "pipeline": [{"$match": {"status": "active"}}]}
    )

    assert len(captured) == 1
    pipeline_used = captured[0]
    limit_stages = [s for s in pipeline_used if "$limit" in s]
    assert len(limit_stages) == 1
    assert limit_stages[0]["$limit"] == 200


async def test_aggregate_does_not_double_inject_limit():
    plugin = make_plugin()
    captured = []

    async def fake_aggregate(collection, pipeline):
        captured.append(pipeline)
        return []

    plugin.aggregate = fake_aggregate

    await plugin._tool_aggregate(
        {
            "collection": "orders",
            "pipeline": [{"$match": {}}, {"$limit": 50}],
        }
    )

    pipeline_used = captured[0]
    limit_stages = [s for s in pipeline_used if "$limit" in s]
    assert len(limit_stages) == 1
    assert limit_stages[0]["$limit"] == 50  # original value preserved


async def test_aggregate_empty_pipeline_gets_limit():
    plugin = make_plugin()
    captured = []

    async def fake_aggregate(collection, pipeline):
        captured.append(pipeline)
        return []

    plugin.aggregate = fake_aggregate

    await plugin._tool_aggregate({"collection": "logs", "pipeline": []})

    assert {"$limit": 200} in captured[0]


# ---------------------------------------------------------------------------
# Find — default limit applied via cursor.limit()
# ---------------------------------------------------------------------------


async def test_find_default_limit_is_50():
    """_tool_find sets limit=50 when not provided and calls cursor.limit(50)."""
    plugin = make_plugin()
    captured_limits = []

    cursor = MagicMock()

    # cursor.limit() returns itself (chainable) and records the call
    def limit_side_effect(n):
        captured_limits.append(n)
        return cursor

    cursor.limit = MagicMock(side_effect=limit_side_effect)
    cursor.to_list = AsyncMock(return_value=[])

    collection_mock = MagicMock()
    collection_mock.find = MagicMock(return_value=cursor)
    plugin._db.__getitem__ = MagicMock(return_value=collection_mock)

    await plugin._tool_find({"collection": "users"})

    assert captured_limits == [50]


async def test_find_respects_explicit_limit():
    plugin = make_plugin()
    captured_limits = []

    cursor = MagicMock()

    def limit_side_effect(n):
        captured_limits.append(n)
        return cursor

    cursor.limit = MagicMock(side_effect=limit_side_effect)
    cursor.to_list = AsyncMock(return_value=[])

    collection_mock = MagicMock()
    collection_mock.find = MagicMock(return_value=cursor)
    plugin._db.__getitem__ = MagicMock(return_value=collection_mock)

    await plugin._tool_find({"collection": "users", "limit": 10})

    assert captured_limits == [10]


# ---------------------------------------------------------------------------
# Aggregate result passthrough
# ---------------------------------------------------------------------------


async def test_aggregate_returns_results():
    plugin = make_plugin()
    docs = [{"_id": "1", "total": 100}, {"_id": "2", "total": 200}]

    async def fake_aggregate(collection, pipeline):
        return docs

    plugin.aggregate = fake_aggregate

    result = await plugin._tool_aggregate(
        {
            "collection": "sales",
            "pipeline": [{"$group": {"_id": "$region", "total": {"$sum": "$amount"}}}],
        }
    )

    assert result["results"] == docs
