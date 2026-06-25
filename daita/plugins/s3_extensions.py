"""
Extension declarations for S3Plugin.
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

S3_MANIFEST = PluginManifest(
    id="s3",
    display_name="AWS S3",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"s3", "object_storage", "storage"}),
    provides=frozenset({"object_storage", "files"}),
    optional_dependencies=frozenset({"boto3"}),
)


S3_OPERATION_DEFINITIONS = (
    {
        "tool_name": "read_s3_file",
        "capability_id": "s3.object.read",
        "operation_type": "s3.object.read",
        "description": (
            "Read and parse an object from S3 with automatic format detection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "S3 object key (file path within the bucket)",
                },
                "format": {
                    "type": "string",
                    "description": (
                        "Format hint: 'auto', 'csv', 'json', 'pandas', 'text'."
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": "Focus DSL to filter/project parsed results.",
                },
            },
            "required": ["key"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 120,
        "handler_name": "_tool_read_file",
    },
    {
        "tool_name": "write_s3_file",
        "capability_id": "s3.object.write",
        "operation_type": "s3.object.write",
        "description": "Write data to an S3 object.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "S3 object key (file path within the bucket)",
                },
                "data": {
                    "description": (
                        "Data to write (dict or list for JSON, string for text, "
                        "or bytes for binary)"
                    ),
                },
            },
            "required": ["key", "data"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 120,
        "handler_name": "_tool_write_file",
    },
    {
        "tool_name": "list_s3_objects",
        "capability_id": "s3.object.list",
        "operation_type": "s3.object.list",
        "description": "List objects in an S3 bucket with an optional prefix filter.",
        "parameters": {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Filter objects by prefix.",
                },
                "max_keys": {
                    "type": "integer",
                    "description": "Maximum number of objects to return.",
                },
                "focus": {
                    "type": "string",
                    "description": "Focus DSL to filter/project listed objects.",
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
        "handler_name": "_tool_list_objects",
    },
    {
        "tool_name": "delete_s3_file",
        "capability_id": "s3.object.delete",
        "operation_type": "s3.object.delete",
        "description": "Delete an object from S3.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "S3 object key to delete"}
            },
            "required": ["key"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_delete_file",
    },
    {
        "tool_name": "head_s3_object",
        "capability_id": "s3.object.head",
        "operation_type": "s3.object.head",
        "description": "Read S3 object metadata without downloading the object.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "S3 object key"}},
            "required": ["key"],
        },
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 15,
        "handler_name": "_tool_head_object",
    },
)


def s3_operation_definitions() -> tuple[dict, ...]:
    return S3_OPERATION_DEFINITIONS


def s3_capabilities() -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="s3",
            description=definition["description"],
            domains=frozenset({"s3", "object_storage", "storage"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"s3.operation.result"}),
            executor="s3.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in s3_operation_definitions()
    )


def s3_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="s3.operation.result",
            owner="s3",
            json_schema={"type": "object"},
            description="S3 object-storage operation result.",
        ),
    )


def s3_tool_views() -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in s3_operation_definitions()
    )


class S3Executor:
    """Execute S3 runtime capabilities and return typed evidence."""

    id = "s3.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"] for definition in s3_operation_definitions()
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="s3.operation.result",
                owner="s3",
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
