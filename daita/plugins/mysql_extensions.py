"""
Extension declarations for MySQLPlugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    RiskLevel,
    Task,
    ToolView,
)

from .manifest import PluginKind, PluginManifest

MYSQL_MANIFEST = PluginManifest(
    id="mysql",
    display_name="MySQL",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"db"}),
    provides=frozenset({"sql", "schema"}),
    optional_dependencies=frozenset({"aiomysql", "SQLAlchemy"}),
)


def mysql_capabilities() -> tuple[Capability, ...]:
    common_schema = {"type": "object"}
    return (
        Capability(
            id="db.schema.inspect",
            owner="mysql",
            description="Inspect MySQL schema metadata.",
            domains=frozenset({"db"}),
            operation_types=frozenset(
                {"schema.query", "schema.relationships", "source.profile"}
            ),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.asset_profile"}),
            executor="mysql.schema.inspect",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.validate",
            owner="mysql",
            description="Validate MySQL SQL against connector guardrails.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "write.propose"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"sql.validation"}),
            executor="mysql.sql.validate",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.execute_read",
            owner="mysql",
            description="Execute a guarded read-only MySQL query.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "metric.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"query.result"}),
            executor="mysql.sql.execute_read",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.execute_write",
            owner="mysql",
            description="Execute a guarded MySQL write statement.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"write.execute"}),
            access=AccessMode.WRITE,
            risk=RiskLevel.HIGH,
            input_schema=common_schema,
            output_evidence=frozenset({"write.execution"}),
            executor="mysql.sql.execute_write",
            runtime_only=True,
            side_effecting=True,
        ),
        Capability(
            id="db.sql.explain",
            owner="mysql",
            description="Explain a MySQL query plan.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "query.plan"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"sql.explain.plan"}),
            executor="mysql.sql.explain",
            runtime_only=True,
            side_effecting=False,
        ),
    )


def mysql_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    object_schema = {"type": "object"}
    return (
        EvidenceSchema(
            kind="schema.asset_profile",
            owner="mysql",
            json_schema=object_schema,
            description="MySQL schema profile.",
        ),
        EvidenceSchema(
            kind="sql.validation",
            owner="mysql",
            json_schema=object_schema,
            description="MySQL SQL validation result.",
        ),
        EvidenceSchema(
            kind="query.result",
            owner="mysql",
            json_schema=object_schema,
            description="MySQL read query result.",
        ),
        EvidenceSchema(
            kind="write.execution",
            owner="mysql",
            json_schema=object_schema,
            description="MySQL write execution result.",
        ),
        EvidenceSchema(
            kind="sql.explain.plan",
            owner="mysql",
            json_schema=object_schema,
            description="MySQL connector explain plan.",
        ),
    )


def mysql_tool_views() -> tuple[ToolView, ...]:
    parameters = {"type": "object"}
    return (
        ToolView(
            name="mysql_query",
            capability_id="db.sql.execute_read",
            description="Run a guarded MySQL read query.",
            parameters=parameters,
        ),
        ToolView(
            name="mysql_inspect",
            capability_id="db.schema.inspect",
            description="Inspect MySQL schema metadata.",
            parameters=parameters,
        ),
    )


@dataclass(frozen=True)
class MySQLExecutor:
    """Executor that delegates one task to a MySQLPlugin method."""

    id: str
    capability_ids: frozenset[str]
    evidence_kind: str
    handler: Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        payload = await self.handler(task.input)
        return [
            Evidence(
                kind=self.evidence_kind,
                owner="mysql",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]
