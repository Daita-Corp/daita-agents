"""Unit tests for ChromaPlugin extension declarations."""

import pytest
from unittest.mock import MagicMock

from daita.core.exceptions import ValidationError
from daita.plugins import ExtensionRegistry
from daita.plugins.chroma import ChromaPlugin
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin():
    plugin = ChromaPlugin(collection="test_col")
    plugin._client = MagicMock()
    plugin._collection = MagicMock()
    return plugin


def _operation_and_task(capability, input_payload):
    operation = Operation(
        id="op-1",
        operation_type=next(iter(capability.operation_types)),
        request=input_payload,
        required_evidence=capability.output_evidence,
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id=capability.id,
        executor_id=capability.executor,
        input=input_payload,
        required_evidence=capability.output_evidence,
    )
    return operation, task


def _mock_query(plugin, result):
    plugin._collection.query = MagicMock(return_value=result)


def test_chroma_access_requires_connection():
    plugin = ChromaPlugin(collection="test_col")

    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.client
    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.collection


async def test_chroma_access_tracks_connection_lifecycle(monkeypatch):
    import chromadb

    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    monkeypatch.setattr(chromadb, "Client", lambda: client)
    plugin = ChromaPlugin(collection="test_col")

    await plugin.connect()
    assert plugin.client is client
    assert plugin.collection is collection

    await plugin.disconnect()
    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.collection


def test_chroma_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "chroma"
    assert registry.plugin_ids == ("chroma",)
    assert {capability.id for capability in registry.capabilities} == {
        "chroma.vector.search",
        "chroma.vector.upsert",
        "chroma.vector.delete",
        "chroma.collection.list",
    }
    assert {view.name for view in registry.tool_views} == {
        "chroma_search",
        "chroma_upsert",
        "chroma_delete",
        "chroma_collections",
    }
    assert registry.get_tool_view_owner("chroma_search") == "chroma"
    assert registry.evidence_schemas[0].kind == "chroma.operation.result"


def test_chroma_capabilities_carry_access_and_safety_metadata():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["chroma.vector.search"].access is AccessMode.READ
    assert by_id["chroma.vector.search"].risk is RiskLevel.LOW
    assert by_id["chroma.vector.search"].side_effecting is False
    assert by_id["chroma.vector.upsert"].access is AccessMode.WRITE
    assert by_id["chroma.vector.delete"].risk is RiskLevel.HIGH
    assert by_id["chroma.collection.list"].access is AccessMode.METADATA_READ
    assert by_id["chroma.collection.list"].side_effecting is False


async def test_chroma_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    _mock_query(
        plugin,
        {
            "ids": [["doc-1"]],
            "distances": [[0.25]],
            "metadatas": [[{"title": "Ada"}]],
            "documents": [["hello"]],
        },
    )
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("chroma.vector.search", owner="chroma")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"vector": [0.1] * 4, "top_k": 1},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "chroma.operation.result"
    assert evidence[0].owner == "chroma"
    assert evidence[0].payload["operation"] == "vector.search"
    assert evidence[0].payload["request"]["top_k"] == 1
    assert evidence[0].payload["result"]["success"] is True
    assert evidence[0].payload["result"]["matches"][0]["id"] == "doc-1"
    assert evidence[0].metadata["capability_id"] == "chroma.vector.search"


async def test_chroma_write_executor_uses_existing_tool_handler():
    plugin = make_plugin()

    async def fake_upsert(ids, vectors, metadata=None, documents=None):
        return {"success": True, "upserted_count": len(ids), "collection": "test_col"}

    plugin.upsert = fake_upsert
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("chroma.vector.upsert", owner="chroma")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {
            "ids": ["doc-1"],
            "vectors": [[0.1, 0.2]],
            "metadata": [{"title": "Ada"}],
            "documents": ["hello"],
        },
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "vector.write"
    assert evidence[0].payload["result"] == {
        "success": True,
        "upserted_count": 1,
        "collection": "test_col",
    }


async def test_chroma_registry_setup_and_teardown_use_connector_lifecycle(monkeypatch):
    plugin = ChromaPlugin(collection="test_col")
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._client = MagicMock()
        plugin._collection = MagicMock()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._client = None
        plugin._collection = None

    monkeypatch.setattr(plugin, "connect", fake_connect)
    monkeypatch.setattr(plugin, "disconnect", fake_disconnect)
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("chroma", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


def test_chroma_legacy_tools_carry_declared_capability_metadata():
    plugin = make_plugin()
    tools = projected_tools(plugin)

    assert tools["chroma_search"].capability_ids == ("chroma.vector.search",)
    assert tools["chroma_upsert"].capability_ids == ("chroma.vector.upsert",)
    assert tools["chroma_delete"].capability_ids == ("chroma.vector.delete",)
    assert tools["chroma_collections"].capability_ids == ("chroma.collection.list",)
    assert tools["chroma_search"].side_effecting is False
    assert tools["chroma_upsert"].side_effecting is True


async def test_connect_missing_client_raises_import_error_with_extra_hint():
    plugin = ChromaPlugin(collection="test_col")
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "chromadb":
            raise ImportError("missing chromadb")
        return original_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="daita-agents\\[chromadb\\]"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("builtins.__import__", fake_import)
            await plugin.connect()


async def test_query_includes_document_and_metadata():
    plugin = make_plugin()
    _mock_query(
        plugin,
        {
            "ids": [["doc-1"]],
            "distances": [[0.5]],
            "metadatas": [[{"category": "notes"}]],
            "documents": [["hello"]],
        },
    )

    results = await plugin.query([0.1] * 4, top_k=1)

    assert results == [
        {
            "id": "doc-1",
            "distance": 0.5,
            "score": 1 / 1.5,
            "metadata": {"category": "notes"},
            "document": "hello",
        }
    ]


async def test_query_normalizes_absent_optional_result_fields():
    plugin = make_plugin()
    _mock_query(plugin, {"ids": [[]]})

    assert await plugin.query([0.1] * 4) == []


async def test_fetch_normalizes_absent_optional_result_fields():
    plugin = make_plugin()
    plugin._collection.get = MagicMock(return_value={"ids": ["doc-1"]})

    assert await plugin.fetch(["doc-1"]) == [{"id": "doc-1"}]
