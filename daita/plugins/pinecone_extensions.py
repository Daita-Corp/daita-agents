"""
Extension declarations for PineconePlugin.
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

PINECONE_MANIFEST = PluginManifest(
    id="pinecone",
    display_name="Pinecone",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"pinecone", "vector", "vector_db"}),
    provides=frozenset({"vector_search", "vector_storage"}),
    optional_dependencies=frozenset({"pinecone"}),
)


PINECONE_OPERATION_DEFINITIONS = (
    {
        "tool_name": "pinecone_search",
        "capability_id": "pinecone.vector.search",
        "operation_type": "vector.search",
        "description": "Search Pinecone for similar vectors.",
        "parameters": {
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "description": "Query vector as array of floats",
                    "items": {"type": "number"},
                },
                "text": {
                    "type": "string",
                    "description": "Query text when an embedding function is configured",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                },
                "filter": {
                    "type": "object",
                    "description": "Pinecone filter dictionary",
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace to search within",
                },
            },
            "required": [],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 60,
        "handler_name": "_tool_search",
    },
    {
        "tool_name": "pinecone_upsert",
        "capability_id": "pinecone.vector.upsert",
        "operation_type": "vector.write",
        "description": "Insert or update vectors in Pinecone.",
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "description": "List of unique vector IDs",
                    "items": {"type": "string"},
                },
                "vectors": {
                    "type": "array",
                    "description": "List of vectors",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "metadata": {
                    "type": "array",
                    "description": "Optional list of metadata objects",
                    "items": {"type": "object"},
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace",
                },
            },
            "required": ["ids", "vectors"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 60,
        "handler_name": "_tool_upsert",
    },
    {
        "tool_name": "pinecone_delete",
        "capability_id": "pinecone.vector.delete",
        "operation_type": "vector.delete",
        "description": "Delete vectors from Pinecone by ID, filter, or namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "description": "List of vector IDs to delete",
                    "items": {"type": "string"},
                },
                "filter": {
                    "type": "object",
                    "description": "Pinecone filter for deletion",
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace",
                },
                "delete_all": {
                    "type": "boolean",
                    "description": "If true, delete all vectors in namespace",
                },
            },
            "required": [],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 60,
        "handler_name": "_tool_delete",
    },
    {
        "tool_name": "pinecone_stats",
        "capability_id": "pinecone.index.stats",
        "operation_type": "vector.index.stats",
        "description": "Get Pinecone index statistics.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_stats",
    },
)


def pinecone_operation_definitions() -> tuple[dict, ...]:
    return PINECONE_OPERATION_DEFINITIONS


def pinecone_capabilities() -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="pinecone",
            description=definition["description"],
            domains=frozenset({"pinecone", "vector", "vector_db"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"pinecone.operation.result"}),
            executor="pinecone.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in pinecone_operation_definitions()
    )


def pinecone_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="pinecone.operation.result",
            owner="pinecone",
            json_schema={"type": "object"},
            description="Pinecone vector operation result.",
        ),
    )


def pinecone_tool_views() -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in pinecone_operation_definitions()
    )


class PineconeExecutor:
    """Execute Pinecone runtime capabilities and return typed evidence."""

    id = "pinecone.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in pinecone_operation_definitions()
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="pinecone.operation.result",
                owner="pinecone",
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
