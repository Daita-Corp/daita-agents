"""
Extension declarations for QdrantPlugin.
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

QDRANT_MANIFEST = PluginManifest(
    id="qdrant",
    display_name="Qdrant",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"qdrant", "vector", "vector_db"}),
    provides=frozenset({"vector_search", "vector_storage"}),
    optional_dependencies=frozenset({"qdrant-client"}),
)


QDRANT_OPERATION_DEFINITIONS = (
    {
        "tool_name": "qdrant_search",
        "capability_id": "qdrant.vector.search",
        "operation_type": "vector.search",
        "description": "Search Qdrant for similar vectors.",
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
                    "description": "Simple filter dictionary",
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
        "tool_name": "qdrant_upsert",
        "capability_id": "qdrant.vector.upsert",
        "operation_type": "vector.write",
        "description": "Insert or update vectors in Qdrant.",
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
                    "description": "Optional list of payload objects",
                    "items": {"type": "object"},
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
        "tool_name": "qdrant_delete",
        "capability_id": "qdrant.vector.delete",
        "operation_type": "vector.delete",
        "description": "Delete vectors from Qdrant by ID or filter.",
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
                    "description": "Simple filter dictionary for deletion",
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
        "tool_name": "qdrant_create_collection",
        "capability_id": "qdrant.collection.create",
        "operation_type": "vector.collection.create",
        "description": "Create a Qdrant collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Collection name"},
                "vector_size": {
                    "type": "integer",
                    "description": "Dimension of vectors",
                },
                "distance": {
                    "type": "string",
                    "description": "Distance metric",
                },
            },
            "required": ["name", "vector_size"],
        },
        "access": AccessMode.ADMIN,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_create_collection",
    },
)


def qdrant_operation_definitions() -> tuple[dict, ...]:
    return QDRANT_OPERATION_DEFINITIONS


def qdrant_capabilities() -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="qdrant",
            description=definition["description"],
            domains=frozenset({"qdrant", "vector", "vector_db"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"qdrant.operation.result"}),
            executor="qdrant.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in qdrant_operation_definitions()
    )


def qdrant_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="qdrant.operation.result",
            owner="qdrant",
            json_schema={"type": "object"},
            description="Qdrant vector operation result.",
        ),
    )


def qdrant_tool_views() -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in qdrant_operation_definitions()
    )


class QdrantExecutor:
    """Execute Qdrant runtime capabilities and return typed evidence."""

    id = "qdrant.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"] for definition in qdrant_operation_definitions()
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="qdrant.operation.result",
                owner="qdrant",
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
