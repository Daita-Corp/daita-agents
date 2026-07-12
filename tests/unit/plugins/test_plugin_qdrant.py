"""
Unit tests for QdrantPlugin.

Tests that only _original_id is stripped from payloads (not all _ prefixed keys),
without a real Qdrant connection.
"""

import pytest
from unittest.mock import MagicMock
from daita.core.exceptions import ValidationError
from daita.plugins import ExtensionRegistry
from daita.plugins.qdrant import QdrantPlugin
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools

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


def test_qdrant_client_access_requires_connection():
    plugin = QdrantPlugin(collection="test_col", url="http://localhost:6333")

    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.client


async def test_qdrant_client_access_tracks_connection_lifecycle(monkeypatch):
    import qdrant_client

    class FakeClient:
        def __init__(self):
            self.closed = False

        def get_collection(self, name):
            return MagicMock()

        def close(self):
            self.closed = True

    client = FakeClient()
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: client)
    plugin = QdrantPlugin(collection="test_col", url="http://localhost:6333")

    await plugin.connect()
    assert plugin.client is client

    await plugin.disconnect()
    assert client.closed is True
    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.client


# ---------------------------------------------------------------------------
# Extension declarations
# ---------------------------------------------------------------------------


def test_qdrant_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "qdrant"
    assert registry.plugin_ids == ("qdrant",)
    assert {capability.id for capability in registry.capabilities} == {
        "qdrant.vector.search",
        "qdrant.vector.upsert",
        "qdrant.vector.delete",
        "qdrant.collection.create",
    }
    assert {view.name for view in registry.tool_views} == {
        "qdrant_search",
        "qdrant_upsert",
        "qdrant_delete",
        "qdrant_create_collection",
    }
    assert registry.get_tool_view_owner("qdrant_search") == "qdrant"
    assert registry.evidence_schemas[0].kind == "qdrant.operation.result"


def test_qdrant_capabilities_carry_access_and_safety_metadata():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["qdrant.vector.search"].access is AccessMode.READ
    assert by_id["qdrant.vector.search"].risk is RiskLevel.LOW
    assert by_id["qdrant.vector.search"].side_effecting is False
    assert by_id["qdrant.vector.upsert"].access is AccessMode.WRITE
    assert by_id["qdrant.vector.delete"].risk is RiskLevel.HIGH
    assert by_id["qdrant.collection.create"].access is AccessMode.ADMIN
    assert by_id["qdrant.collection.create"].side_effecting is True


async def test_qdrant_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    _mock_search(
        plugin,
        [
            FakeScoredPoint(
                id="uuid-1",
                score=0.9,
                payload={"_original_id": "doc-1", "title": "Ada"},
            )
        ],
    )
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("qdrant.vector.search", owner="qdrant")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"vector": [0.1] * 4, "top_k": 1},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "qdrant.operation.result"
    assert evidence[0].owner == "qdrant"
    assert evidence[0].payload["operation"] == "vector.search"
    assert evidence[0].payload["request"]["top_k"] == 1
    assert evidence[0].payload["result"]["count"] == 1
    assert evidence[0].payload["result"]["matches"][0]["id"] == "doc-1"
    assert evidence[0].metadata["capability_id"] == "qdrant.vector.search"


async def test_qdrant_write_executor_uses_existing_tool_handler():
    plugin = make_plugin()

    async def fake_upsert(ids, vectors, metadata=None):
        return {"upserted_count": len(ids), "collection": plugin.collection_name}

    plugin.upsert = fake_upsert
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("qdrant.vector.upsert", owner="qdrant")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"ids": ["doc-1"], "vectors": [[0.1, 0.2]], "metadata": [{"title": "Ada"}]},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "vector.write"
    assert evidence[0].payload["result"] == {
        "upserted_count": 1,
        "collection": "test_col",
    }


async def test_qdrant_registry_setup_and_teardown_use_connector_lifecycle(monkeypatch):
    plugin = QdrantPlugin(collection="test_col", url="http://localhost:6333")
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._client = MagicMock()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._client = None

    monkeypatch.setattr(plugin, "connect", fake_connect)
    monkeypatch.setattr(plugin, "disconnect", fake_disconnect)
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("qdrant", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


def test_qdrant_legacy_tools_carry_declared_capability_metadata():
    plugin = make_plugin()
    tools = projected_tools(plugin)

    assert tools["qdrant_search"].capability_ids == ("qdrant.vector.search",)
    assert tools["qdrant_upsert"].capability_ids == ("qdrant.vector.upsert",)
    assert tools["qdrant_delete"].capability_ids == ("qdrant.vector.delete",)
    assert tools["qdrant_create_collection"].capability_ids == (
        "qdrant.collection.create",
    )
    assert tools["qdrant_search"].side_effecting is False
    assert tools["qdrant_upsert"].side_effecting is True


async def test_connect_missing_client_raises_import_error_with_extra_hint():
    plugin = QdrantPlugin(collection="test_col", url="http://localhost:6333")
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "qdrant_client":
            raise ImportError("missing qdrant")
        return original_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="daita-agents\\[qdrant\\]"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("builtins.__import__", fake_import)
            await plugin.connect()


# ---------------------------------------------------------------------------
# Payload filtering — _original_id only
# ---------------------------------------------------------------------------


async def test_original_id_stripped_from_payload():
    plugin = make_plugin()
    _mock_search(
        plugin,
        [
            FakeScoredPoint(
                id="uuid-1",
                score=0.9,
                payload={"_original_id": "my-id", "name": "Alice", "age": 30},
            )
        ],
    )

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert "_original_id" not in results[0]["payload"]
    assert results[0]["payload"]["name"] == "Alice"
    assert results[0]["payload"]["age"] == 30


async def test_other_underscore_keys_preserved():
    """Keys starting with _ other than _original_id must NOT be stripped."""
    plugin = make_plugin()
    _mock_search(
        plugin,
        [
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
        ],
    )

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    payload = results[0]["payload"]
    assert "_original_id" not in payload
    assert payload["_source"] == "import_job"
    assert payload["_version"] == 3
    assert payload["title"] == "Engineer"


async def test_original_id_used_as_result_id():
    """_original_id in payload should become the id in the result."""
    plugin = make_plugin()
    _mock_search(
        plugin,
        [
            FakeScoredPoint(
                id="internal-uuid",
                score=0.95,
                payload={"_original_id": "human-readable-id", "data": "x"},
            )
        ],
    )

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert results[0]["id"] == "human-readable-id"


async def test_no_original_id_uses_internal_id():
    plugin = make_plugin()
    _mock_search(
        plugin,
        [
            FakeScoredPoint(
                id="internal-uuid",
                score=0.7,
                payload={"name": "Bob"},
            )
        ],
    )

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert results[0]["id"] == "internal-uuid"


async def test_payload_without_underscore_keys_unchanged():
    plugin = make_plugin()
    _mock_search(
        plugin,
        [
            FakeScoredPoint(
                id="abc",
                score=0.6,
                payload={"city": "NYC", "score": 99},
            )
        ],
    )

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    assert results[0]["payload"] == {"city": "NYC", "score": 99}


async def test_empty_payload_handled():
    """Empty dict payload is falsy — plugin omits the payload key entirely."""
    plugin = make_plugin()
    _mock_search(plugin, [FakeScoredPoint(id="x", score=0.5, payload={})])

    results = await plugin.query([0.1] * 128, top_k=1, with_payload=True)

    # Empty payload {} is falsy, so the plugin skips adding it — no KeyError
    assert results[0]["id"] == "x"
    assert "payload" not in results[0]


async def test_upsert_preserves_absent_operation_id():
    plugin = make_plugin()
    plugin._client.upsert = MagicMock(return_value=MagicMock(operation_id=None))

    result = await plugin.upsert(["doc-1"], [[0.1, 0.2]])

    assert result["upserted_count"] == 1
    assert result["operation_id"] is None
