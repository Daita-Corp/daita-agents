"""
Extension declarations for PostgreSQLPlugin.
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

POSTGRESQL_MANIFEST = PluginManifest(
    id="postgresql",
    display_name="PostgreSQL",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"db"}),
    provides=frozenset({"sql", "schema"}),
    optional_dependencies=frozenset({"asyncpg", "psycopg2-binary"}),
)


def postgresql_capabilities() -> tuple[Capability, ...]:
    common_schema = {"type": "object"}
    return (
        Capability(
            id="db.schema.inspect",
            owner="postgresql",
            description="Inspect PostgreSQL schema metadata.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.query", "source.profile"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.asset_profile"}),
            executor="postgresql.schema.inspect",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.validate",
            owner="postgresql",
            description="Validate PostgreSQL SQL against connector guardrails.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "write.propose"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"sql.validation"}),
            executor="postgresql.sql.validate",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.execute_read",
            owner="postgresql",
            description="Execute a guarded read-only PostgreSQL query.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "metric.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"query.result"}),
            executor="postgresql.sql.execute_read",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.execute_write",
            owner="postgresql",
            description="Execute a guarded PostgreSQL write statement.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"write.execute"}),
            access=AccessMode.WRITE,
            risk=RiskLevel.HIGH,
            input_schema=common_schema,
            output_evidence=frozenset({"write.execution"}),
            executor="postgresql.sql.execute_write",
            runtime_only=True,
            side_effecting=True,
        ),
        Capability(
            id="db.sql.explain",
            owner="postgresql",
            description="Explain a PostgreSQL query plan.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "query.plan"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"sql.explain.plan"}),
            executor="postgresql.sql.explain",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="db.column_values.profile",
            owner="postgresql",
            description="Profile bounded observed values for one PostgreSQL column.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"source.profile", "data.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"column_values.profile"}),
            executor="postgresql.column_values.profile",
            runtime_only=True,
            side_effecting=False,
            metadata={
                "profile_policy": {
                    "bounded_aggregate": True,
                    "max_values_limit": 100,
                    "default_max_distinct_count": 100,
                    "default_max_profile_rows": 1_000_000,
                    "default_timeout_seconds": 5,
                    "fingerprint_only_supported": True,
                    "redacts_sensitive_columns": True,
                    "profile_only_readable_tables": True,
                    "skip_reasons": [
                        "blocked_table",
                        "sensitive_or_blocked_column",
                        "profile_timeout",
                        "row_count_exceeds_profile_limit",
                        "high_distinct_count",
                        "value_too_long",
                    ],
                }
            },
        ),
    )


def postgresql_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    object_schema = {"type": "object"}
    return (
        EvidenceSchema(
            kind="schema.asset_profile",
            owner="postgresql",
            json_schema=object_schema,
            description="PostgreSQL schema profile.",
        ),
        EvidenceSchema(
            kind="sql.validation",
            owner="postgresql",
            json_schema=object_schema,
            description="PostgreSQL SQL validation result.",
        ),
        EvidenceSchema(
            kind="query.result",
            owner="postgresql",
            json_schema=object_schema,
            description="PostgreSQL read query result.",
        ),
        EvidenceSchema(
            kind="write.execution",
            owner="postgresql",
            json_schema=object_schema,
            description="PostgreSQL write execution result.",
        ),
        EvidenceSchema(
            kind="sql.explain.plan",
            owner="postgresql",
            json_schema=object_schema,
            description="PostgreSQL connector explain plan.",
        ),
        EvidenceSchema(
            kind="column_values.profile",
            owner="postgresql",
            json_schema=object_schema,
            description="PostgreSQL bounded column value profile.",
        ),
    )


def postgresql_tool_views() -> tuple[ToolView, ...]:
    parameters = {"type": "object"}
    return (
        ToolView(
            name="postgres_query",
            capability_id="db.sql.execute_read",
            description="Run a guarded PostgreSQL read query.",
            parameters=parameters,
        ),
        ToolView(
            name="postgres_inspect",
            capability_id="db.schema.inspect",
            description="Inspect PostgreSQL schema metadata.",
            parameters=parameters,
        ),
    )


@dataclass(frozen=True)
class PostgreSQLExecutor:
    """Executor that delegates one task to a PostgreSQLPlugin method."""

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
                owner="postgresql",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]
