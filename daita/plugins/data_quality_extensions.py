"""
Extension declarations for DataQualityPlugin.
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
)

from .manifest import PluginKind, PluginManifest

DATA_QUALITY_MANIFEST = PluginManifest(
    id="data_quality",
    display_name="Data Quality",
    version="2.0.0",
    kind=PluginKind.DOMAIN_SERVICE,
    domains=frozenset({"db"}),
    provides=frozenset({"quality", "profiling", "freshness"}),
)


def data_quality_capabilities() -> tuple[Capability, ...]:
    common_schema = {"type": "object"}
    return (
        Capability(
            id="quality.profile",
            owner="data_quality",
            description="Profile table data quality and completeness.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"quality.check", "data.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"quality.profile"}),
            executor="data_quality.profile",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="quality.anomaly.detect",
            owner="data_quality",
            description="Detect statistical anomalies in a numeric column.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"quality.check", "anomaly.investigate"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"quality.anomaly"}),
            executor="data_quality.anomaly.detect",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="quality.freshness.check",
            owner="data_quality",
            description="Check freshness of timestamped table data.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"quality.check"}),
            access=AccessMode.READ,
            risk=RiskLevel.LOW,
            input_schema=common_schema,
            output_evidence=frozenset({"quality.freshness"}),
            executor="data_quality.freshness.check",
            runtime_only=True,
            side_effecting=False,
        ),
        Capability(
            id="quality.report.generate",
            owner="data_quality",
            description="Generate a consolidated data quality report.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"quality.check", "report.generate"}),
            access=AccessMode.READ,
            risk=RiskLevel.MEDIUM,
            input_schema=common_schema,
            output_evidence=frozenset({"quality.report"}),
            executor="data_quality.report.generate",
            runtime_only=True,
            side_effecting=False,
        ),
    )


def data_quality_evidence_schemas() -> tuple[EvidenceSchema, ...]:
    object_schema = {"type": "object"}
    return (
        EvidenceSchema(
            kind="quality.profile",
            owner="data_quality",
            json_schema=object_schema,
            description="Data quality profile result.",
        ),
        EvidenceSchema(
            kind="quality.anomaly",
            owner="data_quality",
            json_schema=object_schema,
            description="Data quality anomaly detection result.",
        ),
        EvidenceSchema(
            kind="quality.freshness",
            owner="data_quality",
            json_schema=object_schema,
            description="Data freshness check result.",
        ),
        EvidenceSchema(
            kind="quality.report",
            owner="data_quality",
            json_schema=object_schema,
            description="Consolidated data quality report.",
        ),
    )


@dataclass(frozen=True)
class DataQualityExecutor:
    """Executor that delegates one task to a DataQualityPlugin method."""

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
                owner="data_quality",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]
