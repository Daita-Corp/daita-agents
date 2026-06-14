"""Shared helpers for consuming SQL validation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from daita.runtime import Evidence


@dataclass(frozen=True)
class SqlValidationFacts:
    """Normalized facts produced by accepted SQL validation evidence."""

    valid: bool | None
    is_read: bool | None
    target_resources: tuple[str, ...]
    sql_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "is_read": self.is_read,
            "target_resources": list(self.target_resources),
            "sql_fingerprint": self.sql_fingerprint,
        }


def sql_validation_facts_from_evidence(evidence: Evidence) -> SqlValidationFacts:
    """Return normalized guardrail facts from one sql.validation evidence item."""

    statement_facts = dict(evidence.payload.get("statement_facts") or {})
    target_resources = (
        statement_facts.get("target_resources")
        or evidence.payload.get("target_resources")
        or evidence.payload.get("tables")
        or evidence.payload.get("referenced_tables")
        or ()
    )
    return SqlValidationFacts(
        valid=evidence.payload.get("valid", evidence.payload.get("ok")),
        is_read=evidence.payload.get("is_read", statement_facts.get("is_read")),
        target_resources=tuple(str(item) for item in target_resources),
        sql_fingerprint=(
            evidence.payload.get("sql_fingerprint")
            or statement_facts.get("sql_fingerprint")
            or evidence.metadata.get("payload_fingerprint")
            or evidence.id
        ),
    )


def effective_source_scope(
    default_scope: tuple[str, ...] | list[str] | str,
    plan: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return step-level source scope, falling back to the default scope."""

    raw = plan.get("source_scope")
    values = raw if raw is not None and raw != [] else default_scope
    if isinstance(values, str):
        return (values,)
    return tuple(str(item) for item in values or ())


def blocked_scope_resources(
    resources: tuple[str, ...] | list[str],
    allowed: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Return resources not covered by an allowed source scope."""

    if not allowed or "*" in allowed:
        return ()
    blocked = []
    for resource in resources:
        if not resource_allowed(str(resource), [str(item) for item in allowed]):
            blocked.append(str(resource))
    return tuple(blocked)


def resource_allowed(resource: str, allowed: list[str]) -> bool:
    normalized = resource.lower()
    for candidate in allowed:
        scoped = candidate.lower()
        if normalized == scoped or normalized.endswith(f".{scoped}"):
            return True
        if scoped.endswith(f".{normalized}"):
            return True
    return False
