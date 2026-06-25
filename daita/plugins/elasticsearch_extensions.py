"""
Extension declarations for ElasticsearchPlugin.
"""

from __future__ import annotations

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    ToolView,
)

from .manifest import PluginKind, PluginManifest

ELASTICSEARCH_MANIFEST = PluginManifest(
    id="elasticsearch",
    display_name="Elasticsearch",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"elasticsearch", "search", "document_store"}),
    provides=frozenset({"search", "indexing", "analytics"}),
    optional_dependencies=frozenset({"elasticsearch"}),
)


ELASTICSEARCH_OPERATION_DEFINITIONS = (
    {
        "tool_name": "es_search",
        "capability_id": "elasticsearch.search.query",
        "operation_type": "elasticsearch.search.query",
        "description": "Search an Elasticsearch index using query DSL.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": "Index name or pattern to search in",
                },
                "query": {
                    "type": "object",
                    "description": "Elasticsearch query DSL object",
                },
                "size": {
                    "type": "integer",
                    "description": "Number of results to return",
                },
                "from_": {
                    "type": "integer",
                    "description": "Starting offset for pagination",
                },
                "sort": {
                    "type": "array",
                    "description": "Sort configuration",
                    "items": {"type": "object"},
                },
            },
            "required": ["index", "query"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 60,
        "handler_name": "_tool_search",
    },
    {
        "tool_name": "es_get_mapping",
        "capability_id": "elasticsearch.index.mapping.read",
        "operation_type": "elasticsearch.index.mapping.read",
        "description": "Get the field mapping for an Elasticsearch index.",
        "parameters": {
            "type": "object",
            "properties": {"index": {"type": "string", "description": "Index name"}},
            "required": ["index"],
        },
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_get_mapping",
    },
    {
        "tool_name": "es_cluster_health",
        "capability_id": "elasticsearch.cluster.health.read",
        "operation_type": "elasticsearch.cluster.health.read",
        "description": "Get Elasticsearch cluster health status and shard metrics.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_cluster_health",
    },
    {
        "tool_name": "es_index_document",
        "capability_id": "elasticsearch.document.index",
        "operation_type": "elasticsearch.document.index",
        "description": "Index a single document into an Elasticsearch index.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "string", "description": "Index name"},
                "document": {
                    "type": "object",
                    "description": "Document data as JSON object",
                },
                "doc_id": {
                    "type": "string",
                    "description": "Optional document ID",
                },
            },
            "required": ["index", "document"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_index",
    },
    {
        "tool_name": "es_bulk_index",
        "capability_id": "elasticsearch.document.bulk_index",
        "operation_type": "elasticsearch.document.bulk_index",
        "description": "Bulk index multiple documents into Elasticsearch.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "string", "description": "Index name"},
                "documents": {
                    "type": "array",
                    "description": "List of document objects to index",
                    "items": {"type": "object"},
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Documents per batch",
                },
            },
            "required": ["index", "documents"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 120,
        "handler_name": "_tool_bulk_index",
    },
    {
        "tool_name": "es_delete_document",
        "capability_id": "elasticsearch.document.delete",
        "operation_type": "elasticsearch.document.delete",
        "description": "Delete a document from Elasticsearch by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "string", "description": "Index name"},
                "doc_id": {"type": "string", "description": "Document ID to delete"},
            },
            "required": ["index", "doc_id"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "read_only": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_delete_document",
    },
    {
        "tool_name": "es_create_index",
        "capability_id": "elasticsearch.index.create",
        "operation_type": "elasticsearch.index.create",
        "description": "Create an Elasticsearch index with optional mapping and settings.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "string", "description": "Index name to create"},
                "mapping": {"type": "object", "description": "Index mapping"},
                "settings": {"type": "object", "description": "Index settings"},
            },
            "required": ["index"],
        },
        "access": AccessMode.ADMIN,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_create_index",
    },
)


def elasticsearch_operation_definitions(read_only: bool = False) -> tuple[dict, ...]:
    if not read_only:
        return ELASTICSEARCH_OPERATION_DEFINITIONS
    return tuple(
        definition
        for definition in ELASTICSEARCH_OPERATION_DEFINITIONS
        if definition["read_only"]
    )


def elasticsearch_capabilities(read_only: bool = False) -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="elasticsearch",
            description=definition["description"],
            domains=frozenset({"elasticsearch", "search", "document_store"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"elasticsearch.operation.result"}),
            executor="elasticsearch.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in elasticsearch_operation_definitions(read_only)
    )


def elasticsearch_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="elasticsearch.operation.result",
            owner="elasticsearch",
            json_schema={"type": "object"},
            description="Elasticsearch operation result.",
        ),
    )


def elasticsearch_tool_views(read_only: bool = False) -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in elasticsearch_operation_definitions(read_only)
    )


class ElasticsearchExecutor:
    """Execute Elasticsearch runtime capabilities and return typed evidence."""

    id = "elasticsearch.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in elasticsearch_operation_definitions(
                self._plugin.read_only
            )
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="elasticsearch.operation.result",
                owner="elasticsearch",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "operation": definition["operation_type"],
                    "request": dict(task.input or {}),
                    "result": result,
                },
                metadata={"capability_id": task.capability_id},
            )
        ]
