"""
Unit tests for MongoDBPlugin tool handlers.

Tests real behavior: aggregate $limit injection, find filter/projection/limit
forwarding, insert/update/delete result shapes, ObjectId serialization —
without a real MongoDB connection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from daita.plugins.mongodb import MongoDBPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin(**kwargs):
    plugin = MongoDBPlugin(host="localhost", database="testdb", **kwargs)
    # Inject fake client/db so connect() is never called
    plugin._client = MagicMock()
    plugin._db = MagicMock()
    return plugin


def _make_find_cursor(results):
    """Return a mock cursor chain for collection.find()."""
    cursor = MagicMock()
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.sort = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=results)
    return cursor


def _wire_collection(plugin, cursor):
    """Wire plugin._db[collection] to return a collection mock using cursor."""
    collection_mock = MagicMock()
    collection_mock.find = MagicMock(return_value=cursor)
    plugin._db.__getitem__ = MagicMock(return_value=collection_mock)
    return collection_mock


# ---------------------------------------------------------------------------
# _tool_aggregate — $limit injection (real pipeline mutation logic)
# ---------------------------------------------------------------------------


class TestAggregateLimit:
    async def test_injects_limit_when_missing(self):
        plugin = make_plugin()
        captured = []

        async def fake_aggregate(collection, pipeline):
            captured.append(pipeline)
            return []

        plugin.aggregate = fake_aggregate
        await plugin._tool_aggregate(
            {"collection": "orders", "pipeline": [{"$match": {"status": "active"}}]}
        )

        limit_stages = [s for s in captured[0] if "$limit" in s]
        assert len(limit_stages) == 1
        assert limit_stages[0]["$limit"] == 200

    async def test_does_not_double_inject_limit(self):
        plugin = make_plugin()
        captured = []

        async def fake_aggregate(collection, pipeline):
            captured.append(pipeline)
            return []

        plugin.aggregate = fake_aggregate
        await plugin._tool_aggregate(
            {"collection": "orders", "pipeline": [{"$match": {}}, {"$limit": 50}]}
        )

        limit_stages = [s for s in captured[0] if "$limit" in s]
        assert len(limit_stages) == 1
        assert (
            limit_stages[0]["$limit"] == 50
        )  # original value preserved, not overwritten

    async def test_empty_pipeline_gets_limit(self):
        plugin = make_plugin()
        captured = []

        async def fake_aggregate(collection, pipeline):
            captured.append(pipeline)
            return []

        plugin.aggregate = fake_aggregate
        await plugin._tool_aggregate({"collection": "logs", "pipeline": []})
        assert {"$limit": 200} in captured[0]

    async def test_result_has_results_key(self):
        plugin = make_plugin()
        docs = [{"_id": "1", "total": 100}, {"_id": "2", "total": 200}]

        async def fake_aggregate(collection, pipeline):
            return docs

        plugin.aggregate = fake_aggregate
        result = await plugin._tool_aggregate(
            {
                "collection": "sales",
                "pipeline": [
                    {"$group": {"_id": "$region", "total": {"$sum": "$amount"}}}
                ],
            }
        )
        assert result["results"] == docs

    async def test_collection_name_forwarded(self):
        plugin = make_plugin()
        captured_collection = []

        async def fake_aggregate(collection, pipeline):
            captured_collection.append(collection)
            return []

        plugin.aggregate = fake_aggregate
        await plugin._tool_aggregate({"collection": "events", "pipeline": []})
        assert captured_collection[0] == "events"


# ---------------------------------------------------------------------------
# _tool_find — filter, projection, and limit forwarding
# ---------------------------------------------------------------------------


class TestFindForwarding:
    async def test_default_limit_is_50(self):
        plugin = make_plugin()
        cursor = _make_find_cursor([])
        collection_mock = _wire_collection(plugin, cursor)

        await plugin._tool_find({"collection": "users"})

        cursor.limit.assert_called_once_with(50)

    async def test_explicit_limit_respected(self):
        plugin = make_plugin()
        cursor = _make_find_cursor([])
        collection_mock = _wire_collection(plugin, cursor)

        await plugin._tool_find({"collection": "users", "limit": 10})

        cursor.limit.assert_called_once_with(10)

    async def test_filter_forwarded_to_find(self):
        plugin = make_plugin()
        cursor = _make_find_cursor([])
        collection_mock = _wire_collection(plugin, cursor)
        filter_doc = {"status": "active", "age": {"$gt": 18}}

        await plugin._tool_find({"collection": "users", "filter": filter_doc})

        call_args = collection_mock.find.call_args
        assert call_args[0][0] == filter_doc  # first positional arg is filter

    async def test_empty_filter_when_not_provided(self):
        plugin = make_plugin()
        cursor = _make_find_cursor([])
        collection_mock = _wire_collection(plugin, cursor)

        await plugin._tool_find({"collection": "users"})

        call_args = collection_mock.find.call_args
        # When no filter is given, an empty dict should be passed
        assert call_args[0][0] == {} or call_args[0][0] is None

    async def test_projection_forwarded_to_find(self):
        plugin = make_plugin()
        cursor = _make_find_cursor([])
        collection_mock = _wire_collection(plugin, cursor)
        projection = {"name": 1, "email": 1, "_id": 0}

        await plugin._tool_find({"collection": "users", "projection": projection})

        call_args = collection_mock.find.call_args
        # projection is second arg to find()
        assert call_args[0][1] == projection

    async def test_result_has_documents_key(self):
        plugin = make_plugin()
        docs = [{"name": "Alice"}, {"name": "Bob"}]
        cursor = _make_find_cursor(docs)
        _wire_collection(plugin, cursor)

        result = await plugin._tool_find({"collection": "users"})

        assert "documents" in result

    async def test_result_documents_contain_returned_rows(self):
        plugin = make_plugin()
        docs = [{"name": "Alice", "_id": "507f1f77bcf86cd799439011"}]
        cursor = _make_find_cursor(docs)
        _wire_collection(plugin, cursor)

        result = await plugin._tool_find({"collection": "users"})

        assert len(result["documents"]) == 1
        assert result["documents"][0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# _tool_find — ObjectId serialization
# ---------------------------------------------------------------------------


class TestFindObjectIdSerialization:
    async def test_objectid_converted_to_string(self):
        """_tool_find must convert ObjectId _id values to strings for JSON safety."""
        plugin = make_plugin()

        # Simulate a motor ObjectId (has __str__ but is not a plain string)
        class FakeObjectId:
            def __str__(self):
                return "507f1f77bcf86cd799439011"

        docs = [{"_id": FakeObjectId(), "name": "Alice"}]
        cursor = _make_find_cursor(docs)
        _wire_collection(plugin, cursor)

        result = await plugin._tool_find({"collection": "users"})

        # _id should be a plain string in the result
        doc = result["documents"][0]
        assert isinstance(doc["_id"], str)
        assert doc["_id"] == "507f1f77bcf86cd799439011"


# ---------------------------------------------------------------------------
# _tool_insert — result shape
# ---------------------------------------------------------------------------


class TestInsertHandler:
    async def test_returns_inserted_id_and_collection(self):
        plugin = make_plugin()

        async def fake_insert(collection, doc):
            return "507f1f77bcf86cd799439011"

        plugin.insert = fake_insert
        result = await plugin._tool_insert(
            {"collection": "users", "document": {"name": "Alice"}}
        )

        assert "inserted_id" in result
        assert result["inserted_id"] == "507f1f77bcf86cd799439011"
        assert result["collection"] == "users"

    async def test_document_forwarded_to_insert(self):
        plugin = make_plugin()
        captured_doc = []

        async def fake_insert(collection, doc):
            captured_doc.append(doc)
            return "abc123"

        plugin.insert = fake_insert
        doc = {"name": "Bob", "role": "admin"}
        await plugin._tool_insert({"collection": "users", "document": doc})

        assert captured_doc[0] == doc


# ---------------------------------------------------------------------------
# _tool_update — result shape
# ---------------------------------------------------------------------------


class TestUpdateHandler:
    async def test_returns_matched_and_modified_counts(self):
        plugin = make_plugin()

        async def fake_update(collection, filter_doc, update_doc, upsert=False):
            return {"matched_count": 3, "modified_count": 2, "upserted_id": None}

        plugin.update = fake_update
        result = await plugin._tool_update(
            {
                "collection": "users",
                "filter": {"role": "guest"},
                "update": {"$set": {"role": "member"}},
            }
        )

        assert result["matched_count"] == 3
        assert result["modified_count"] == 2
        assert result["collection"] == "users"

    async def test_filter_and_update_forwarded(self):
        plugin = make_plugin()
        captured = []

        async def fake_update(collection, filter_doc, update_doc, upsert=False):
            captured.append((filter_doc, update_doc))
            return {"matched_count": 1, "modified_count": 1, "upserted_id": None}

        plugin.update = fake_update
        f = {"_id": "abc"}
        u = {"$set": {"active": True}}
        await plugin._tool_update({"collection": "users", "filter": f, "update": u})

        assert captured[0] == (f, u)


# ---------------------------------------------------------------------------
# _tool_delete — result shape
# ---------------------------------------------------------------------------


class TestDeleteHandler:
    async def test_returns_deleted_count_and_collection(self):
        plugin = make_plugin()

        async def fake_delete(collection, filter_doc):
            return 5

        plugin.delete = fake_delete
        result = await plugin._tool_delete(
            {"collection": "sessions", "filter": {"expired": True}}
        )

        assert result["deleted_count"] == 5
        assert result["collection"] == "sessions"

    async def test_filter_forwarded_to_delete(self):
        plugin = make_plugin()
        captured_filter = []

        async def fake_delete(collection, filter_doc):
            captured_filter.append(filter_doc)
            return 1

        plugin.delete = fake_delete
        f = {"status": "inactive"}
        await plugin._tool_delete({"collection": "users", "filter": f})

        assert captured_filter[0] == f


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_default_tools_present(self):
        plugin = make_plugin()
        names = {t.name for t in plugin.get_tools()}
        assert "mongodb_find" in names
        assert "mongodb_insert" in names
        assert "mongodb_update" in names
        assert "mongodb_delete" in names
        assert "mongodb_list_collections" in names
        assert "mongodb_count" in names

    def test_write_tools_absent_when_read_only(self):
        plugin = make_plugin(read_only=True)
        names = {t.name for t in plugin.get_tools()}
        assert "mongodb_insert" not in names
        assert "mongodb_update" not in names
        assert "mongodb_delete" not in names

    def test_read_tools_present_when_read_only(self):
        plugin = make_plugin(read_only=True)
        names = {t.name for t in plugin.get_tools()}
        assert "mongodb_find" in names
        assert "mongodb_count" in names

    def test_no_unprefixed_tool_names(self):
        plugin = make_plugin()
        names = {t.name for t in plugin.get_tools()}
        assert "find_documents" not in names
        assert "insert_document" not in names
