"""
Extension declarations for SnowflakePlugin.
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

SNOWFLAKE_MANIFEST = PluginManifest(
    id="snowflake",
    display_name="Snowflake",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"snowflake", "db", "warehouse"}),
    provides=frozenset({"sql", "schema", "warehouse"}),
    optional_dependencies=frozenset({"snowflake-connector-python"}),
)


SNOWFLAKE_CORE_OPERATION_DEFINITIONS = (
    {
        "tool_name": "snowflake_query",
        "capability_id": "snowflake.sql.query",
        "operation_type": "data.query",
        "description": "Run a guarded Snowflake SELECT query.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query with placeholders",
                },
                "params": {
                    "type": "array",
                    "description": "Optional parameter values",
                    "items": {},
                },
                "focus": {
                    "type": "string",
                    "description": "Focus DSL to filter/project results",
                },
            },
            "required": ["sql"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.MEDIUM,
        "read_only": True,
        "admin": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 60,
        "handler_name": "_tool_query",
    },
    {
        "tool_name": "snowflake_inspect",
        "capability_id": "snowflake.schema.inspect",
        "operation_type": "schema.query",
        "description": "Inspect Snowflake table schemas.",
        "parameters": {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific tables",
                },
                "schema": {
                    "type": "string",
                    "description": "Optional schema name",
                },
            },
            "required": [],
        },
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "admin": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_inspect",
    },
    {
        "tool_name": "snowflake_count",
        "capability_id": "snowflake.table.count",
        "operation_type": "metric.query",
        "description": "Count rows in a Snowflake table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name"},
                "filter": {
                    "type": "string",
                    "description": "Optional WHERE clause without WHERE keyword",
                },
            },
            "required": ["table"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "admin": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_count",
    },
    {
        "tool_name": "snowflake_sample",
        "capability_id": "snowflake.table.sample",
        "operation_type": "data.sample",
        "description": "Return a random sample of rows from a Snowflake table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name"},
                "n": {
                    "type": "integer",
                    "description": "Number of rows to sample",
                },
            },
            "required": ["table"],
        },
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "admin": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_sample",
    },
    {
        "tool_name": "snowflake_list_schemas",
        "capability_id": "snowflake.schema.list",
        "operation_type": "schema.query",
        "description": "List schemas in the Snowflake database.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "admin": False,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_list_schemas",
    },
    {
        "tool_name": "snowflake_execute",
        "capability_id": "snowflake.sql.execute",
        "operation_type": "write.execute",
        "description": "Execute Snowflake DML.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL DML statement",
                },
                "params": {
                    "type": "array",
                    "description": "Optional parameter values",
                    "items": {},
                },
            },
            "required": ["sql"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "read_only": False,
        "admin": False,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 60,
        "handler_name": "_tool_execute",
    },
)


SNOWFLAKE_ADMIN_OPERATION_DEFINITIONS = (
    {
        "tool_name": "snowflake_list_warehouses",
        "capability_id": "snowflake.warehouse.list",
        "operation_type": "warehouse.inspect",
        "description": "List Snowflake compute warehouses.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "admin": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_list_warehouses",
    },
    {
        "tool_name": "snowflake_get_query_history",
        "capability_id": "snowflake.query_history.read",
        "operation_type": "warehouse.inspect",
        "description": "Read recent Snowflake query history.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of queries to return",
                }
            },
            "required": [],
        },
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "admin": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 45,
        "handler_name": "_tool_get_query_history",
    },
    {
        "tool_name": "snowflake_list_stages",
        "capability_id": "snowflake.stage.list",
        "operation_type": "stage.inspect",
        "description": "List Snowflake stages.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "admin": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_list_stages",
    },
    {
        "tool_name": "snowflake_load_from_stage",
        "capability_id": "snowflake.stage.load",
        "operation_type": "warehouse.load",
        "description": "Load data from a Snowflake stage into a table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Target table name"},
                "stage": {"type": "string", "description": "Stage location"},
                "file_format": {
                    "type": "string",
                    "description": "File format type",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional file pattern",
                },
            },
            "required": ["table", "stage"],
        },
        "access": AccessMode.WRITE,
        "risk": RiskLevel.HIGH,
        "read_only": True,
        "admin": True,
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 180,
        "handler_name": "_tool_load_from_stage",
    },
    {
        "tool_name": "snowflake_create_stage",
        "capability_id": "snowflake.stage.create",
        "operation_type": "warehouse.configure",
        "description": "Create a Snowflake stage.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Stage name"},
                "url": {"type": "string", "description": "External stage URL"},
                "storage_integration": {
                    "type": "string",
                    "description": "Storage integration name",
                },
            },
            "required": ["name"],
        },
        "access": AccessMode.ADMIN,
        "risk": RiskLevel.HIGH,
        "read_only": True,
        "admin": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_create_stage",
    },
    {
        "tool_name": "snowflake_switch_warehouse",
        "capability_id": "snowflake.warehouse.switch",
        "operation_type": "warehouse.configure",
        "description": "Switch the active Snowflake compute warehouse.",
        "parameters": {
            "type": "object",
            "properties": {
                "warehouse": {
                    "type": "string",
                    "description": "Warehouse name",
                }
            },
            "required": ["warehouse"],
        },
        "access": AccessMode.ADMIN,
        "risk": RiskLevel.MEDIUM,
        "read_only": False,
        "admin": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": True,
        "timeout_seconds": 30,
        "handler_name": "_tool_switch_warehouse",
    },
)


def snowflake_operation_definitions(
    read_only: bool = False,
    include_admin: bool = False,
) -> tuple[dict, ...]:
    definitions = SNOWFLAKE_CORE_OPERATION_DEFINITIONS
    if include_admin:
        definitions = definitions + SNOWFLAKE_ADMIN_OPERATION_DEFINITIONS
    if not read_only:
        return definitions
    return tuple(definition for definition in definitions if definition["read_only"])


def snowflake_capabilities(
    read_only: bool = False,
    include_admin: bool = False,
) -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="snowflake",
            description=definition["description"],
            domains=frozenset({"snowflake", "db", "warehouse"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"snowflake.operation.result"}),
            executor="snowflake.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in snowflake_operation_definitions(read_only, include_admin)
    )


def snowflake_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="snowflake.operation.result",
            owner="snowflake",
            json_schema={"type": "object"},
            description="Snowflake warehouse operation result.",
        ),
    )


def snowflake_tool_views(
    read_only: bool = False,
    include_admin: bool = False,
) -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in snowflake_operation_definitions(read_only, include_admin)
    )


class SnowflakeExecutor:
    """Execute Snowflake runtime capabilities and return typed evidence."""

    id = "snowflake.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in snowflake_operation_definitions(
                self._plugin.read_only,
                self._plugin._include_admin_extensions(),
            )
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="snowflake.operation.result",
                owner="snowflake",
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
