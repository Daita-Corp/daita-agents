"""
Extension declarations for SQLitePlugin.
"""

from __future__ import annotations

from daita.runtime import (
    AccessMode,
    Capability,
    EvidenceSchema,
    RiskLevel,
    ToolView,
)

from .manifest import PluginKind, PluginManifest

SQLITE_MANIFEST = PluginManifest(
    id="sqlite",
    display_name="SQLite",
    version="2.0.0",
    kind=PluginKind.CONNECTOR,
    domains=frozenset({"db"}),
    provides=frozenset({"sql", "schema"}),
    optional_dependencies=frozenset({"aiosqlite"}),
)


def sqlite_capabilities() -> tuple[Capability, ...]:
    common_schema = {"type": "object"}
    return (
        Capability(
            id="db.schema.inspect",
            owner="sqlite",
            description="Inspect SQLite schema metadata.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"schema.query", "source.profile"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"schema.asset_profile"}),
            executor="sqlite.schema.inspect",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.validate",
            owner="sqlite",
            description="Validate SQLite SQL against connector guardrails.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "write.propose"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"sql.validation"}),
            executor="sqlite.sql.validate",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.execute_read",
            owner="sqlite",
            description="Execute a guarded read-only SQLite query.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "metric.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"query.result"}),
            executor="sqlite.sql.execute_read",
            model_visible=True,
            side_effecting=False,
        ),
        Capability(
            id="db.sql.execute_write",
            owner="sqlite",
            description="Execute a guarded SQLite write statement.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"write.execute"}),
            access=AccessMode.WRITE,
            risk=RiskLevel.HIGH,
            input_schema=common_schema,
            output_evidence=frozenset({"write.execution"}),
            executor="sqlite.sql.execute_write",
            runtime_only=True,
            side_effecting=True,
        ),
        Capability(
            id="db.sql.explain",
            owner="sqlite",
            description="Explain a SQLite query plan.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query", "query.plan"}),
            access=AccessMode.METADATA_READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"sql.explain.plan"}),
            executor="sqlite.sql.explain",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="db.column_values.profile",
            owner="sqlite",
            description="Profile bounded observed values for one SQLite column.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"source.profile", "data.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"column_values.profile"}),
            executor="sqlite.column_values.profile",
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


def sqlite_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    object_schema = {"type": "object"}
    return (
        EvidenceSchema(
            kind="schema.asset_profile",
            owner="sqlite",
            json_schema=object_schema,
            description="SQLite schema profile.",
        ),
        EvidenceSchema(
            kind="sql.validation",
            owner="sqlite",
            json_schema=object_schema,
            description="SQLite SQL validation result.",
        ),
        EvidenceSchema(
            kind="query.result",
            owner="sqlite",
            json_schema=object_schema,
            description="SQLite read query result.",
        ),
        EvidenceSchema(
            kind="write.execution",
            owner="sqlite",
            json_schema=object_schema,
            description="SQLite write execution result.",
        ),
        EvidenceSchema(
            kind="sql.explain.plan",
            owner="sqlite",
            json_schema=object_schema,
            description="SQLite connector explain plan.",
        ),
        EvidenceSchema(
            kind="column_values.profile",
            owner="sqlite",
            json_schema=object_schema,
            description="SQLite bounded column value profile.",
        ),
    )


def sqlite_tool_views() -> tuple[ToolView, ...]:
    parameters = {"type": "object"}
    return (
        ToolView(
            name="sqlite_query",
            capability_id="db.sql.execute_read",
            description="Run a guarded SQLite read query.",
            parameters=parameters,
        ),
        ToolView(
            name="sqlite_inspect",
            capability_id="db.schema.inspect",
            description="Inspect SQLite schema metadata.",
            parameters=parameters,
        ),
    )
