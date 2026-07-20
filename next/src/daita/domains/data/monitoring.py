"""Data-domain-owned deterministic projection of monitor operation evidence."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import math

from ..._json import canonical_json
from ...monitors.models import (
    MonitorCheckpoint,
    MonitorConditionKind,
    MonitorDefinition,
    MonitorFindingSeverity,
)
from ...monitors.scheduler import MonitorOutcomeProjection
from ...operations.checkpoints import OperationSnapshot
from ...operations.models import Evidence, TriggerKind
from .comparison import TABULAR_COMPARE_EVIDENCE_KIND
from .controller import (
    POSTGRESQL_QUERY_EVIDENCE_KIND,
    SQLITE_QUERY_EVIDENCE_KIND,
)
from .file_capabilities import LOCAL_FILE_READ_EVIDENCE_KIND

_READ_EVIDENCE_KINDS = frozenset(
    {
        LOCAL_FILE_READ_EVIDENCE_KIND,
        POSTGRESQL_QUERY_EVIDENCE_KIND,
        SQLITE_QUERY_EVIDENCE_KIND,
        TABULAR_COMPARE_EVIDENCE_KIND,
    }
)


class DataMonitorOutcomeProjector:
    """Evaluate confirmed typed conditions over accepted current-run evidence."""

    async def project(
        self,
        *,
        definition: MonitorDefinition,
        operation: OperationSnapshot,
        checkpoint: MonitorCheckpoint | None,
    ) -> MonitorOutcomeProjection:
        del checkpoint
        if not isinstance(definition, MonitorDefinition):
            raise TypeError("definition must be a MonitorDefinition")
        if not isinstance(operation, OperationSnapshot):
            raise TypeError("operation must be an OperationSnapshot")
        scope = _bound_monitor_scope(definition, operation)
        eligible = tuple(
            evidence
            for evidence in operation.evidence
            if _eligible_evidence(evidence, operation, scope=scope)
        )
        condition = definition.condition
        if condition.kind is MonitorConditionKind.ALWAYS:
            if not eligible:
                return MonitorOutcomeProjection(matched=False)
            evidence = eligible[-1]
            return _matched_projection(
                definition,
                operation,
                evidence,
                details={"condition_kind": "always"},
                severity=MonitorFindingSeverity.INFO,
            )
        if condition.kind is not MonitorConditionKind.THRESHOLD:
            raise ValueError("unsupported monitor condition reached projection")
        evidence_kind = condition.configuration.get("evidence_kind")
        candidates = tuple(
            evidence
            for evidence in eligible
            if evidence_kind is None or evidence.kind == evidence_kind
        )
        if not candidates:
            return MonitorOutcomeProjection(matched=False)
        evidence = candidates[-1]
        if evidence.payload.get("truncated") is True:
            return MonitorOutcomeProjection(matched=False)
        actual = _resolve_path(evidence.payload, condition.expression or "")
        expected = condition.configuration.get("value")
        operator = condition.configuration.get("operator")
        if not _numeric(actual) or not _numeric(expected):
            return MonitorOutcomeProjection(matched=False)
        if not _compare(actual, expected, operator):
            return MonitorOutcomeProjection(matched=False)
        return _matched_projection(
            definition,
            operation,
            evidence,
            details={
                "actual": actual,
                "condition_kind": "threshold",
                "operator": operator,
                "path": condition.expression,
                "threshold": expected,
            },
            severity=MonitorFindingSeverity.WARNING,
        )


def _bound_monitor_scope(
    definition: MonitorDefinition,
    operation: OperationSnapshot,
) -> tuple[frozenset[str], frozenset[str]]:
    if operation.trigger.kind is not TriggerKind.MONITOR:
        raise ValueError("monitor projector requires a monitor operation")
    payload = operation.trigger.payload
    if payload.get("monitor_definition_hash") != definition.content_hash:
        raise ValueError("monitor operation definition binding does not match")
    scope = payload.get("monitor_scope")
    if not isinstance(scope, Mapping) or set(scope) != {"resource_ids", "source_ids"}:
        raise ValueError("monitor operation has no scope binding")
    source_ids = scope.get("source_ids")
    resource_ids = scope.get("resource_ids")
    if (
        not isinstance(source_ids, tuple)
        or not isinstance(resource_ids, tuple)
        or source_ids != definition.scope.source_ids
        or resource_ids != definition.scope.resource_ids
    ):
        raise ValueError("monitor operation scope differs from its definition")
    return frozenset(source_ids), frozenset(resource_ids)


def _eligible_evidence(
    evidence: Evidence,
    operation: OperationSnapshot,
    *,
    scope: tuple[frozenset[str], frozenset[str]],
) -> bool:
    source_scope, resource_scope = scope
    facts = evidence.validation_facts
    return bool(
        evidence.operation_id == operation.operation.id
        and evidence.kind in _READ_EVIDENCE_KINDS
        and evidence.accepted
        and evidence.applicable
        and evidence.metadata_schema_version == 1
        and evidence.applicability_reason == "current_operation"
        and facts.schema_version == 1
        and facts.validation_passed
        and facts.in_scope
        and facts.freshness_state == "current"
        and facts.source_ids
        and set(facts.source_ids) <= source_scope
        and (not resource_scope or set(facts.resource_ids) <= resource_scope)
    )


def _resolve_path(payload: Mapping[str, object], path: str) -> object:
    value: object = payload
    for component in path.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                return None
            value = value[component]
        elif isinstance(value, tuple) and component.isdigit():
            index = int(component)
            if index >= len(value):
                return None
            value = value[index]
        else:
            return None
    return value


def _numeric(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _compare(actual: object, expected: object, operator: object) -> bool:
    if not isinstance(operator, str):
        return False
    try:
        left = Decimal(str(actual))
        right = Decimal(str(expected))
    except (InvalidOperation, ValueError):
        return False
    return {
        "eq": left == right,
        "gt": left > right,
        "gte": left >= right,
        "lt": left < right,
        "lte": left <= right,
        "ne": left != right,
    }.get(operator, False)


def _matched_projection(
    definition: MonitorDefinition,
    operation: OperationSnapshot,
    evidence: Evidence,
    *,
    details: Mapping[str, object],
    severity: MonitorFindingSeverity,
) -> MonitorOutcomeProjection:
    payload = operation.trigger.payload
    material = {
        "definition_hash": definition.content_hash,
        "evidence_id": evidence.id,
        "monitor_id": payload.get("monitor_id"),
        "occurrence_id": payload.get("monitor_occurrence_id"),
    }
    dedupe_key = (
        "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()
    )
    return MonitorOutcomeProjection(
        matched=True,
        evidence_id=evidence.id,
        severity=severity,
        summary=f"{definition.name} matched its confirmed condition.",
        details={
            **details,
            "definition_hash": definition.content_hash,
            "evidence_kind": evidence.kind,
        },
        dedupe_key=dedupe_key,
    )


__all__ = ["DataMonitorOutcomeProjector"]
