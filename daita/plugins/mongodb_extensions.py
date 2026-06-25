"""
Extension declarations for MongoDBPlugin.
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

MONGODB_MANIFEST = PluginManifest(
    id="mongodb",
    display_name="MongoDB",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"mongodb", "document_store", "db"}),
    provides=frozenset({"document_store", "collections"}),
    optional_dependencies=frozenset({"motor"}),
)


MONGODB_OPERATION_DEFINITIONS = (
    {
        "tool_name": "mongodb_find",
        "capability_id": "mongodb.document.find",
        "operation_type": "mongodb.document.find",
        "description": "Find documents in a MongoDB collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "filter": {"type": "object"},
                "limit": {"type": "integer"},
                "projection": {"type": "object"},
            },
            "required": ["collection"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_find",
    },
    {
        "tool_name": "mongodb_list_collections",
        "capability_id": "mongodb.collection.list",
        "operation_type": "mongodb.collection.list",
        "description": "List MongoDB collections.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_list_collections",
    },
    {
        "tool_name": "mongodb_aggregate",
        "capability_id": "mongodb.pipeline.aggregate",
        "operation_type": "mongodb.pipeline.aggregate",
        "description": "Run a MongoDB aggregation pipeline.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "pipeline": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["collection", "pipeline"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.MEDIUM,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_aggregate",
    },
    {
        "tool_name": "mongodb_count",
        "capability_id": "mongodb.document.count",
        "operation_type": "mongodb.document.count",
        "description": "Count documents in a MongoDB collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "filter": {"type": "object"},
            },
            "required": ["collection"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_count",
    },
    {
        "tool_name": "mongodb_insert",
        "capability_id": "mongodb.document.insert",
        "operation_type": "mongodb.document.insert",
        "description": "Insert a document into a MongoDB collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "document": {"type": "object"},
            },
            "required": ["collection", "document"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_insert",
    },
    {
        "tool_name": "mongodb_update",
        "capability_id": "mongodb.document.update",
        "operation_type": "mongodb.document.update",
        "description": "Update MongoDB documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "filter": {"type": "object"},
                "update": {"type": "object"},
            },
            "required": ["collection", "filter", "update"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_update",
    },
    {
        "tool_name": "mongodb_delete",
        "capability_id": "mongodb.document.delete",
        "operation_type": "mongodb.document.delete",
        "description": "Delete MongoDB documents by filter.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "filter": {"type": "object"},
            },
            "required": ["collection", "filter"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "read_only": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "handler_name": "_tool_delete",
    },
)


def mongodb_operation_definitions(read_only: bool = False) -> tuple[dict, ...]:
    if not read_only:
        return MONGODB_OPERATION_DEFINITIONS
    return tuple(
        definition
        for definition in MONGODB_OPERATION_DEFINITIONS
        if definition["read_only"]
    )


def mongodb_capabilities(read_only: bool = False) -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="mongodb",
            description=definition["description"],
            domains=frozenset({"mongodb", "document_store", "db"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"mongodb.operation.result"}),
            executor="mongodb.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
        )
        for definition in mongodb_operation_definitions(read_only)
    )


def mongodb_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="mongodb.operation.result",
            owner="mongodb",
            json_schema={"type": "object"},
            description="MongoDB document-store operation result.",
        ),
    )


def mongodb_tool_views(read_only: bool = False) -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in mongodb_operation_definitions(read_only)
    )


class MongoDBExecutor:
    """Execute MongoDB runtime capabilities and return typed evidence."""

    id = "mongodb.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in mongodb_operation_definitions(self._plugin.read_only)
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="mongodb.operation.result",
                owner="mongodb",
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
