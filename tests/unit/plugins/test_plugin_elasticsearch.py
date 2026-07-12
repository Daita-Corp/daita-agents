"""Unit tests for ElasticsearchPlugin extension declarations."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from daita.core.exceptions import PluginError
from daita.plugins import ExtensionRegistry
from daita.plugins.elasticsearch import ElasticsearchPlugin
from daita.runtime import AccessMode, Operation, RiskLevel, Task
from tests.unit.plugins.projection_helpers import projected_tools


def make_plugin(read_only=False):
    plugin = ElasticsearchPlugin(hosts="localhost:9200", read_only=read_only)
    plugin._client = MagicMock()
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


def _mock_search(plugin, documents):
    plugin._client.search = AsyncMock(
        return_value={
            "hits": {
                "total": {"value": len(documents)},
                "max_score": 1.0,
                "hits": [{"_source": document} for document in documents],
            },
            "took": 7,
            "timed_out": False,
        }
    )


def test_elasticsearch_client_access_requires_connection():
    plugin = ElasticsearchPlugin(hosts="localhost:9200")

    with pytest.raises(PluginError, match="not connected"):
        _ = plugin.client


async def test_elasticsearch_client_access_tracks_connection_lifecycle(monkeypatch):
    import elasticsearch

    class FakeClient:
        def __init__(self):
            self.closed = False

        async def info(self):
            return {
                "cluster_name": "test-cluster",
                "version": {"number": "9.0", "lucene_version": "10.0"},
                "tagline": "test",
            }

        async def close(self):
            self.closed = True

    client = FakeClient()
    monkeypatch.setattr(elasticsearch, "AsyncElasticsearch", lambda **kwargs: client)
    plugin = ElasticsearchPlugin(hosts="localhost:9200")

    await plugin.connect()
    assert plugin.client is client

    await plugin.disconnect()
    assert client.closed is True
    with pytest.raises(PluginError, match="not connected"):
        _ = plugin.client


def test_elasticsearch_plugin_declares_extension_first_contract():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert plugin.manifest.id == "elasticsearch"
    assert registry.plugin_ids == ("elasticsearch",)
    assert {capability.id for capability in registry.capabilities} == {
        "elasticsearch.search.query",
        "elasticsearch.index.mapping.read",
        "elasticsearch.cluster.health.read",
        "elasticsearch.document.index",
        "elasticsearch.document.bulk_index",
        "elasticsearch.document.delete",
        "elasticsearch.index.create",
    }
    assert {view.name for view in registry.tool_views} == {
        "es_search",
        "es_get_mapping",
        "es_cluster_health",
        "es_index_document",
        "es_bulk_index",
        "es_delete_document",
        "es_create_index",
    }
    assert registry.get_tool_view_owner("es_search") == "elasticsearch"
    assert registry.evidence_schemas[0].kind == "elasticsearch.operation.result"


def test_elasticsearch_read_only_filters_write_capabilities_and_tool_views():
    plugin = make_plugin(read_only=True)
    registry = ExtensionRegistry()

    registry.register(plugin)

    assert {capability.id for capability in registry.capabilities} == {
        "elasticsearch.search.query",
        "elasticsearch.index.mapping.read",
        "elasticsearch.cluster.health.read",
    }
    assert {view.name for view in registry.tool_views} == {
        "es_search",
        "es_get_mapping",
        "es_cluster_health",
    }


def test_elasticsearch_capabilities_carry_access_and_safety_metadata():
    plugin = make_plugin()
    registry = ExtensionRegistry()

    registry.register(plugin)
    by_id = {capability.id: capability for capability in registry.capabilities}

    assert by_id["elasticsearch.search.query"].access is AccessMode.READ
    assert by_id["elasticsearch.search.query"].risk is RiskLevel.LOW
    assert by_id["elasticsearch.search.query"].side_effecting is False
    assert by_id["elasticsearch.document.index"].access is AccessMode.WRITE
    assert by_id["elasticsearch.document.index"].side_effecting is True
    assert by_id["elasticsearch.document.delete"].risk is RiskLevel.HIGH
    assert by_id["elasticsearch.index.create"].access is AccessMode.ADMIN


async def test_elasticsearch_executor_returns_typed_operation_evidence():
    plugin = make_plugin()
    _mock_search(plugin, [{"message": "hello"}])
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability(
        "elasticsearch.search.query", owner="elasticsearch"
    )
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"index": "logs", "query": {"match_all": {}}, "size": 1},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].kind == "elasticsearch.operation.result"
    assert evidence[0].owner == "elasticsearch"
    assert evidence[0].payload["operation"] == "elasticsearch.search.query"
    assert evidence[0].payload["request"]["index"] == "logs"
    assert evidence[0].payload["result"] == {
        "documents": [{"message": "hello"}],
        "total": 1,
        "took_ms": 7,
    }
    assert evidence[0].metadata["capability_id"] == "elasticsearch.search.query"


async def test_elasticsearch_write_executor_uses_existing_tool_handler():
    plugin = make_plugin()
    plugin._client.index = AsyncMock(
        return_value={
            "_id": "doc-1",
            "_index": "logs",
            "_version": 1,
            "result": "created",
        }
    )
    registry = ExtensionRegistry()
    registry.register(plugin)
    capability = registry.get_capability(
        "elasticsearch.document.index", owner="elasticsearch"
    )
    executor = registry.get_executor(capability.executor)
    operation, task = _operation_and_task(
        capability,
        {"index": "logs", "document": {"message": "hello"}, "doc_id": "doc-1"},
    )

    evidence = await executor.execute(task, operation, {})

    assert evidence[0].payload["operation"] == "elasticsearch.document.index"
    assert evidence[0].payload["result"] == {
        "id": "doc-1",
        "index": "logs",
        "created": True,
    }


async def test_elasticsearch_registry_setup_and_teardown_use_connector_lifecycle(
    monkeypatch,
):
    plugin = ElasticsearchPlugin(hosts="localhost:9200")
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

    await registry.setup_plugin("elasticsearch", context=None)
    assert plugin.is_connected is True

    await registry.teardown_all()

    assert calls == ["connect", "disconnect"]
    assert plugin.is_connected is False


def test_elasticsearch_legacy_tools_carry_declared_capability_metadata():
    plugin = make_plugin()
    tools = projected_tools(plugin)

    assert tools["es_search"].capability_ids == ("elasticsearch.search.query",)
    assert tools["es_get_mapping"].capability_ids == (
        "elasticsearch.index.mapping.read",
    )
    assert tools["es_cluster_health"].capability_ids == (
        "elasticsearch.cluster.health.read",
    )
    assert tools["es_index_document"].capability_ids == (
        "elasticsearch.document.index",
    )
    assert tools["es_bulk_index"].capability_ids == (
        "elasticsearch.document.bulk_index",
    )
    assert tools["es_delete_document"].capability_ids == (
        "elasticsearch.document.delete",
    )
    assert tools["es_create_index"].capability_ids == ("elasticsearch.index.create",)
    assert tools["es_search"].side_effecting is False
    assert tools["es_index_document"].side_effecting is True


async def test_connect_missing_client_raises_import_error_with_extra_hint():
    plugin = ElasticsearchPlugin(hosts="localhost:9200")
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "elasticsearch":
            raise ImportError("missing elasticsearch")
        return original_import(name, *args, **kwargs)

    with pytest.raises(ImportError, match="daita-agents\\[elasticsearch\\]"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("builtins.__import__", fake_import)
            await plugin.connect()


async def test_search_shapes_documents_and_totals():
    plugin = make_plugin()
    _mock_search(plugin, [{"message": "hello"}, {"message": "bye"}])

    result = await plugin.search("logs", {"match_all": {}}, size=2)

    assert result["hits"]["documents"] == [
        {"message": "hello"},
        {"message": "bye"},
    ]
    assert result["hits"]["total"] == {"value": 2}
    assert result["took"] == 7


async def test_search_normalizes_empty_hits():
    plugin = make_plugin()
    _mock_search(plugin, [])

    result = await plugin.search("logs")

    assert result["hits"]["documents"] == []
    assert result["hits"]["total"] == {"value": 0}


async def test_index_document_reports_non_created_result():
    plugin = make_plugin()
    plugin._client.index = AsyncMock(
        return_value={
            "_id": "doc-1",
            "_index": "logs",
            "_version": 2,
            "result": "updated",
        }
    )

    result = await plugin.index_document("logs", {"message": "updated"})

    assert result["result"] == "updated"
    assert result["created"] is False
