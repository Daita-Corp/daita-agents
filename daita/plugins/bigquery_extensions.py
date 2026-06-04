"""
Extension declarations for BigQueryPlugin.
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

BIGQUERY_MANIFEST = PluginManifest(
    id="bigquery",
    display_name="BigQuery",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"bigquery", "db", "warehouse"}),
    provides=frozenset({"sql", "schema", "warehouse"}),
    optional_dependencies=frozenset({"google-cloud-bigquery"}),
)


BIGQUERY_OPERATION_DEFINITIONS = (
    {
        "tool_name": "bigquery_query",
        "capability_id": "bigquery.sql.query",
        "operation_type": "data.query",
        "description": "Run a guarded BigQuery SELECT query.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query with %s placeholders",
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
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 120,
        "handler_name": "_tool_query",
    },
    {
        "tool_name": "bigquery_inspect",
        "capability_id": "bigquery.schema.inspect",
        "operation_type": "schema.query",
        "description": "Inspect BigQuery dataset table schemas.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "description": "Dataset name",
                },
                "tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific tables",
                },
            },
            "required": [],
        },
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 60,
        "handler_name": "_tool_inspect",
    },
    {
        "tool_name": "bigquery_count",
        "capability_id": "bigquery.table.count",
        "operation_type": "metric.query",
        "description": "Count rows in a BigQuery table.",
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
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 60,
        "handler_name": "_tool_count",
    },
    {
        "tool_name": "bigquery_sample",
        "capability_id": "bigquery.table.sample",
        "operation_type": "data.sample",
        "description": "Return a random sample of rows from a BigQuery table.",
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
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 60,
        "handler_name": "_tool_sample",
    },
    {
        "tool_name": "bigquery_list_datasets",
        "capability_id": "bigquery.dataset.list",
        "operation_type": "schema.query",
        "description": "List BigQuery datasets in a project.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "access": AccessMode.METADATA_READ,
        "risk": RiskLevel.LOW,
        "read_only": True,
        "retry_safe": True,
        "replay_safe": True,
        "idempotent": True,
        "side_effecting": False,
        "timeout_seconds": 30,
        "handler_name": "_tool_list_datasets",
    },
    {
        "tool_name": "bigquery_execute",
        "capability_id": "bigquery.sql.execute",
        "operation_type": "write.execute",
        "description": "Execute BigQuery DML or DDL.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL DML/DDL statement",
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
        "retry_safe": False,
        "replay_safe": False,
        "idempotent": False,
        "side_effecting": True,
        "timeout_seconds": 120,
        "handler_name": "_tool_execute",
    },
)


def bigquery_operation_definitions(read_only: bool = False) -> tuple[dict, ...]:
    if not read_only:
        return BIGQUERY_OPERATION_DEFINITIONS
    return tuple(
        definition
        for definition in BIGQUERY_OPERATION_DEFINITIONS
        if definition["read_only"]
    )


def bigquery_capabilities(read_only: bool = False) -> tuple[Capability, ...]:
    return tuple(
        Capability(
            id=definition["capability_id"],
            owner="bigquery",
            description=definition["description"],
            domains=frozenset({"bigquery", "db", "warehouse"}),
            operation_types=frozenset({definition["operation_type"]}),
            access=definition["access"],
            risk=definition["risk"],
            input_schema=definition["parameters"],
            output_evidence=frozenset({"bigquery.operation.result"}),
            executor="bigquery.operations",
            runtime_only=False,
            model_visible=True,
            retry_safe=definition["retry_safe"],
            replay_safe=definition["replay_safe"],
            idempotent=definition["idempotent"],
            side_effecting=definition["side_effecting"],
            timeout_seconds=definition["timeout_seconds"],
        )
        for definition in bigquery_operation_definitions(read_only)
    )


def bigquery_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    return (
        EvidenceSchema(
            kind="bigquery.operation.result",
            owner="bigquery",
            json_schema={"type": "object"},
            description="BigQuery warehouse operation result.",
        ),
    )


def bigquery_tool_views(read_only: bool = False) -> tuple[ToolView, ...]:
    return tuple(
        ToolView(
            name=definition["tool_name"],
            capability_id=definition["capability_id"],
            description=definition["description"],
            parameters=definition["parameters"],
        )
        for definition in bigquery_operation_definitions(read_only)
    )


class BigQueryExecutor:
    """Execute BigQuery runtime capabilities and return typed evidence."""

    id = "bigquery.operations"

    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            definition["capability_id"]
            for definition in bigquery_operation_definitions(self._plugin.read_only)
        )

    async def execute(self, task, operation, context):
        definition = self._plugin._definition_for_capability(task.capability_id)
        handler = getattr(self._plugin, definition["handler_name"])
        result = await handler(dict(task.input or {}))
        return [
            Evidence(
                kind="bigquery.operation.result",
                owner="bigquery",
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
