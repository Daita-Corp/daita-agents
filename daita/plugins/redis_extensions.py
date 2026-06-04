"""
Extension declarations for RedisPlugin.
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

REDIS_MANIFEST = PluginManifest(
    id="redis",
    display_name="Redis",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"redis", "cache", "kv"}),
    provides=frozenset({"key_value_store", "cache"}),
    optional_dependencies=frozenset({"redis"}),
)


REDIS_OPERATION_DEFINITIONS = (
    {
        "tool_name": "redis_get",
        "capability_id": "redis.key.get",
        "operation_type": "redis.key.get",
        "description": "Get the value of a Redis key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key name"}},
            "required": ["key"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_get",
    },
    {
        "tool_name": "redis_keys",
        "capability_id": "redis.key.scan",
        "operation_type": "redis.key.scan",
        "description": "Scan Redis keys matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (default: '*')",
                },
                "count": {
                    "type": "integer",
                    "description": "Max keys to return (default: 1000)",
                },
            },
            "required": [],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_keys",
    },
    {
        "tool_name": "redis_exists",
        "capability_id": "redis.key.exists",
        "operation_type": "redis.key.exists",
        "description": "Check whether one or more Redis keys exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to check",
                }
            },
            "required": ["keys"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_exists",
    },
    {
        "tool_name": "redis_ttl",
        "capability_id": "redis.key.ttl",
        "operation_type": "redis.key.ttl",
        "description": "Get the remaining TTL of a Redis key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key name"}},
            "required": ["key"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_ttl",
    },
    {
        "tool_name": "redis_hgetall",
        "capability_id": "redis.hash.read",
        "operation_type": "redis.hash.read",
        "description": "Read all fields from a Redis hash.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Hash key name"}},
            "required": ["key"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_hgetall",
    },
    {
        "tool_name": "redis_lrange",
        "capability_id": "redis.list.read",
        "operation_type": "redis.list.read",
        "description": "Read elements from a Redis list by index range.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "List key name"},
                "start": {"type": "integer", "description": "Start index"},
                "stop": {"type": "integer", "description": "Stop index"},
            },
            "required": ["key"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_lrange",
    },
    {
        "tool_name": "redis_smembers",
        "capability_id": "redis.set.read",
        "operation_type": "redis.set.read",
        "description": "Read all members of a Redis set.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Set key name"}},
            "required": ["key"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_smembers",
    },
    {
        "tool_name": "redis_info",
        "capability_id": "redis.info.read",
        "operation_type": "redis.info.read",
        "description": "Read Redis database size and connection metadata.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "handler_name": "_tool_info",
    },
    {
        "tool_name": "redis_set",
        "capability_id": "redis.key.set",
        "operation_type": "redis.key.set",
        "description": "Set a Redis key-value pair with optional TTL.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name"},
                "value": {"type": "string", "description": "Value to store"},
                "ttl": {"type": "integer", "description": "Optional TTL in seconds"},
            },
            "required": ["key", "value"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": True,
        "side_effecting": True,
        "handler_name": "_tool_set",
    },
    {
        "tool_name": "redis_delete",
        "capability_id": "redis.key.delete",
        "operation_type": "redis.key.delete",
        "description": "Delete one or more Redis keys.",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to delete",
                }
            },
            "required": ["keys"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "handler_name": "_tool_delete",
    },
    {
        "tool_name": "redis_hset",
        "capability_id": "redis.hash.write",
        "operation_type": "redis.hash.write",
        "description": "Set fields on a Redis hash.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Hash key name"},
                "mapping": {"type": "object", "description": "Field-value pairs"},
            },
            "required": ["key", "mapping"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": True,
        "side_effecting": True,
        "handler_name": "_tool_hset",
    },
    {
        "tool_name": "redis_lpush",
        "capability_id": "redis.list.lpush",
        "operation_type": "redis.list.lpush",
        "description": "Push values to the left of a Redis list.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "List key name"},
                "values": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key", "values"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_lpush",
    },
    {
        "tool_name": "redis_rpush",
        "capability_id": "redis.list.rpush",
        "operation_type": "redis.list.rpush",
        "description": "Push values to the right of a Redis list.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "List key name"},
                "values": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key", "values"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "handler_name": "_tool_rpush",
    },
    {
        "tool_name": "redis_sadd",
        "capability_id": "redis.set.add",
        "operation_type": "redis.set.add",
        "description": "Add members to a Redis set.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Set key name"},
                "members": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key", "members"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": True,
        "side_effecting": True,
        "handler_name": "_tool_sadd",
    },
    {
        "tool_name": "redis_expire",
        "capability_id": "redis.key.expire",
        "operation_type": "redis.key.expire",
        "description": "Set a TTL on a Redis key.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name"},
                "seconds": {"type": "integer", "description": "TTL in seconds"},
            },
            "required": ["key", "seconds"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "handler_name": "_tool_expire",
    },
)


def redis_operation_definitions(read_only: bool = False) -> tuple[dict, ...]:
    if not read_only:
        return REDIS_OPERATION_DEFINITIONS
    return tuple(
        definition
        for definition in REDIS_OPERATION_DEFINITIONS
        if definition["read_only"]
    )


def redis_capabilities(read_only: bool = False) -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="redis",
            description=definition["description"],
            domains=frozenset({"redis", "cache", "kv"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"redis.operation.result"}),
            executor="redis.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
        )
        for definition in redis_operation_definitions(read_only)
    )


def redis_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="redis.operation.result",
            owner="redis",
            json_schema={"type": "object"},
            description="Redis data-store operation result.",
        ),
    )


def redis_tool_views(read_only: bool = False) -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in redis_operation_definitions(read_only)
    )


class RedisExecutor:
    """Execute Redis runtime capabilities and return typed evidence."""

    id = "redis.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in redis_operation_definitions(self._plugin.read_only)
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="redis.operation.result",
                owner="redis",
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
