"""Unit tests for PineconePlugin extension declarations."""

import pytest
from unittest.mock import MagicMock

from daita.core.exceptions import ValidationError
from daita.plugins import ExtensionRegistry
from daita.plugins.pinecone import PineconePlugin
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin():
    plugin = PineconePlugin(api_key="test-key", index="test-index", namespace="ns")
    plugin._client = MagicMock()
    plugin._index = MagicMock()
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


def _mock_query(plugin, matches):
    plugin._index.query = MagicMock(return_value={"matches": matches})


def test_pinecone_access_requires_connection():
    plugin = PineconePlugin(api_key="test-key", index="test-index")

    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.client
    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.pinecone_index


async def test_pinecone_access_tracks_connection_lifecycle(module_stub):
    index = MagicMock()
    client = MagicMock()
    client.Index.return_value = index
    module_stub("pinecone", Pinecone=lambda **kwargs: client)
    plugin = PineconePlugin(api_key="test-key", index="test-index")

    await plugin.connect()
    assert plugin.client is client
    assert plugin.pinecone_index is index

    await plugin.disconnect()
    with pytest.raises(ValidationError, match="not connected"):
        _ = plugin.pinecone_index


def test_pinecone_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "pinecone"
    assert registry.plugin_ids == ("pinecone",)
    assert {capability.id for capability in registry.capabilities} == {
        "pinecone.vector.search",
        "pinecone.vector.upsert",
        "pinecone.vector.delete",
        "pinecone.index.stats",
    }
    assert {view.name for view in registry.tool_views} == {
        "pinecone_search",
        "pinecone_upsert",
        "pinecone_delete",
        "pinecone_stats",
    }
    assert registry.get_tool_view_owner("pinecone_search") == "pinecone"
    assert registry.evidence_schemas[0].kind == "pinecone.operation.result"


def test_pinecone_capabilities_carry_access_and_safety_metadata():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["pinecone.vector.search"].access is AccessMode.READ
    assert by_id["pinecone.vector.search"].risk is RiskLevel.LOW
    assert by_id["pinecone.vector.search"].side_effecting is False
    assert by_id["pinecone.vector.upsert"].access is AccessMode.WRITE
    assert by_id["pinecone.vector.delete"].risk is RiskLevel.HIGH
    assert by_id["pinecone.index.stats"].access is AccessMode.METADATA_READ
    assert by_id["pinecone.index.stats"].side_effecting is False


async def test_pinecone_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    _mock_query(
        plugin,
        [
            {
                "id": "doc-1",
                "score": 0.9,
                "metadata": {"title": "Ada"},
                "values": [0.1, 0.2],
            }
        ],
    )
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("pinecone.vector.search", owner="pinecone")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"vector": [0.1] * 4, "top_k": 1, "namespace": "ns"},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "pinecone.operation.result"
    assert evidence[0].owner == "pinecone"
    assert evidence[0].payload["operation"] == "vector.search"
    assert evidence[0].payload["request"]["top_k"] == 1
    assert evidence[0].payload["result"]["success"] is True
    assert evidence[0].payload["result"]["matches"][0]["id"] == "doc-1"
    assert evidence[0].metadata["capability_id"] == "pinecone.vector.search"


async def test_pinecone_write_executor_uses_existing_tool_handler():
    plugin = make_plugin()

    async def fake_upsert(ids, vectors, metadata=None, namespace=None):
        return {"upserted_count": len(ids), "namespace": namespace}

    plugin.upsert = fake_upsert
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability("pinecone.vector.upsert", owner="pinecone")
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {
            "ids": ["doc-1"],
            "vectors": [[0.1, 0.2]],
            "metadata": [{"title": "Ada"}],
            "namespace": "custom",
        },
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "vector.write"
    assert evidence[0].payload["result"] == {
        "success": True,
        "upserted_count": 1,
        "namespace": "custom",
    }


async def test_pinecone_registry_setup_and_teardown_use_connector_lifecycle(
    monkeypatch,
):
    plugin = PineconePlugin(api_key="test-key", index="test-index")
    calls = []

    async def fake_connect():
        calls.append("connect")
        plugin._client = MagicMock()
        plugin._index = MagicMock()

    async def fake_disconnect():
        calls.append("disconnect")
        plugin._client = None
        plugin._index = None

    monkeypatch.setattr(plugin, "connect", fake_connect)
    monkeypatch.setattr(plugin, "disconnect", fake_disconnect)
    registry = ExtensionRegistry()
    registry.register(plugin)

    await registry.setup_plugin("pinecone", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


def test_pinecone_legacy_tools_carry_declared_capability_metadata():
    plugin = make_plugin()
    tools = projected_tools(plugin)

    assert tools["pinecone_search"].capability_ids == ("pinecone.vector.search",)
    assert tools["pinecone_upsert"].capability_ids == ("pinecone.vector.upsert",)
    assert tools["pinecone_delete"].capability_ids == ("pinecone.vector.delete",)
    assert tools["pinecone_stats"].capability_ids == ("pinecone.index.stats",)
    assert tools["pinecone_search"].side_effecting is False
    assert tools["pinecone_upsert"].side_effecting is True


async def test_connect_missing_client_raises_import_error_with_extra_hint():
    plugin = PineconePlugin(api_key="test-key", index="test-index")
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pinecone":
            raise ImportError("missing pinecone")
        return original_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="daita-agents\\[pinecone\\]"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("builtins.__import__", fake_import)
            await plugin.connect()


async def test_query_includes_metadata_and_values_when_requested():
    plugin = make_plugin()
    _mock_query(
        plugin,
        [
            {
                "id": "doc-1",
                "score": 0.5,
                "metadata": {"category": "notes"},
                "values": [0.1, 0.2],
            }
        ],
    )

    results = await plugin.query([0.1] * 4, top_k=1, include_values=True)

    assert results == [
        {
            "id": "doc-1",
            "score": 0.5,
            "metadata": {"category": "notes"},
            "values": [0.1, 0.2],
        }
    ]


async def test_query_normalizes_empty_matches():
    plugin = make_plugin()
    _mock_query(plugin, [])

    assert await plugin.query([0.1] * 4) == []


async def test_upsert_defaults_absent_count_to_input_size():
    plugin = make_plugin()
    plugin._index.upsert = MagicMock(return_value={})

    result = await plugin.upsert(["doc-1"], [[0.1, 0.2]])

    assert result["upserted_count"] == 1
