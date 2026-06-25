"""
Extension declarations for ChromaPlugin.
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

CHROMA_MANIFEST = PluginManifest(
    id="chroma",
    display_name="ChromaDB",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"chroma", "vector", "vector_db"}),
    provides=frozenset({"vector_search", "vector_storage"}),
    optional_dependencies=frozenset({"chromadb"}),
)


CHROMA_OPERATION_DEFINITIONS = (
    {
        "tool_name": "chroma_search",
        "capability_id": "chroma.vector.search",
        "operation_type": "vector.search",
        "description": "Search ChromaDB for similar vectors.",
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
                    "description": "Chroma where filter",
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
        "tool_name": "chroma_upsert",
        "capability_id": "chroma.vector.upsert",
        "operation_type": "vector.write",
        "description": "Insert or update vectors in ChromaDB.",
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
                "documents": {
                    "type": "array",
                    "description": "Optional list of raw document texts",
                    "items": {"type": "string"},
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
        "tool_name": "chroma_delete",
        "capability_id": "chroma.vector.delete",
        "operation_type": "vector.delete",
        "description": "Delete vectors from ChromaDB by ID or filter.",
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
                    "description": "Chroma where filter for deletion",
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
        "tool_name": "chroma_collections",
        "capability_id": "chroma.collection.list",
        "operation_type": "vector.collection.list",
        "description": "List ChromaDB collections.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_collections",
    },
)


def chroma_operation_definitions() -> tuple[dict, ...]:
    return CHROMA_OPERATION_DEFINITIONS


def chroma_capabilities() -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="chroma",
            description=definition["description"],
            domains=frozenset({"chroma", "vector", "vector_db"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"chroma.operation.result"}),
            executor="chroma.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in chroma_operation_definitions()
    )


def chroma_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="chroma.operation.result",
            owner="chroma",
            json_schema={"type": "object"},
            description="ChromaDB vector operation result.",
        ),
    )


def chroma_tool_views() -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in chroma_operation_definitions()
    )


class ChromaExecutor:
    """Execute Chroma runtime capabilities and return typed evidence."""

    id = "chroma.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"] for definition in chroma_operation_definitions()
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="chroma.operation.result",
                owner="chroma",
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
