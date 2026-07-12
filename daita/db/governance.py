"""Default governance policies for the DB runtime."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .models import DbIntentKind

from daita.runtime import (
    AccessMode,
    Operation,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
)

_WRITE_OPERATION_TYPES = frozenset(
    {
        DbIntentKind.WRITE_PROPOSE.value,
        DbIntentKind.WRITE_EXECUTE.value,
        DbIntentKind.ADMIN.value,
    }
)
_WRITE_MODE_ALIASES = frozenset({*_WRITE_OPERATION_TYPES, "write", "write_execute"})
_DESTRUCTIVE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bdrop\s+(table|database|schema|view|index)\b",
        r"\btruncate\s+(table\s+)?\w+",
        r"\balter\s+(table|database|schema|view|index)\b",
        r"\bdelete\s+from\b",
        r"\bdelete\s+all\b",
        r"\bdelete\b",
        r"\bwipe\b",
        r"\bpurge\b",
        r"\bdestroy\b",
    )
)


class DbWriteApprovalPolicy:
    """Require human approval before write/admin DB operations execute."""

    id = "approval_required_for_writes"
    owner = "runtime"
    version = "1"

    def applies_to(self, request: Any, operation_type: str) -> bool:
        return (
            operation_type in _WRITE_OPERATION_TYPES
            or _request_access(request) in {AccessMode.WRITE, AccessMode.ADMIN}
            or _capability_requires_write_approval(request)
        )

    def modify_contract(self, contract: Any) -> Any:
        return contract

    def evaluate_operation(self, operation: Operation) -> PolicyDecision | None:
        if not self.applies_to(operation.request, operation.operation_type):
            return None
        if _destructive_decision_facts(
            operation.request,
            operation.operation_type,
            operation.metadata,
        ):
            return None
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.REQUIRE_APPROVAL,
            reason="Write and admin DB operations require approval before execution.",
            severity=RiskLevel.HIGH,
            operation_id=operation.id,
            required_approvals=("human",),
            metadata={
                "operation_type": operation.operation_type,
                "access": operation.metadata.get("access"),
                "capability": _request_capability(operation.request),
            },
        )


class DbDestructiveOperationPolicy:
    """Deny destructive DB operations by default."""

    id = "deny_destructive_operations"
    owner = "runtime"
    version = "2"

    def applies_to(self, request: Any, operation_type: str) -> bool:
        return bool(_destructive_decision_facts(request, operation_type))

    def modify_contract(self, contract: Any) -> Any:
        return contract

    def evaluate_operation(self, operation: Operation) -> PolicyDecision | None:
        facts = _destructive_decision_facts(
            operation.request,
            operation.operation_type,
            operation.metadata,
        )
        if not facts:
            return None
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.DENY,
            reason="Destructive DB operations are denied by default.",
            severity=RiskLevel.CRITICAL,
            operation_id=operation.id,
            metadata={
                "operation_type": operation.operation_type,
                **facts,
            },
        )


def default_db_policies() -> tuple[Any, ...]:
    """Return the default DB runtime governance policies."""
    return (DbDestructiveOperationPolicy(), DbWriteApprovalPolicy())


def _request_prompt(request: Any) -> str:
    if isinstance(request, Mapping):
        return str(request.get("prompt") or "").lower()
    return str(request or "").lower()


def _request_access(request: Any) -> AccessMode | None:
    if not isinstance(request, Mapping):
        return None
    value = request.get("access")
    if value is None and isinstance(request.get("metadata"), Mapping):
        value = request["metadata"].get("access")
    if value is None:
        return None
    try:
        return AccessMode(value)
    except ValueError:
        return None


def _request_capability(request: Any) -> dict[str, Any] | None:
    if not isinstance(request, Mapping):
        return None
    capability = request.get("capability")
    if isinstance(capability, Mapping):
        return dict(capability)
    return None


def _request_governance_facts(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        return {}
    facts = request.get("governance_facts")
    return dict(facts) if isinstance(facts, Mapping) else {}


def _destructive_decision_facts(
    request: Any,
    operation_type: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    governance_facts = _request_governance_facts(request)
    authoritative = governance_facts.get("authoritative")
    authoritative = authoritative if isinstance(authoritative, Mapping) else {}
    validation = authoritative.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    destructive_classes = _safe_strings(validation.get("destructive_statement_classes"))
    admin_classes = _safe_strings(validation.get("admin_statement_classes"))
    if destructive_classes or admin_classes:
        return {
            "decision_source": "validated_statement_facts",
            "fact_source": "sql_validation",
            "matched_statement_classes": tuple(
                sorted({*destructive_classes, *admin_classes})
            ),
            "destructive_statement_classes": destructive_classes,
            "admin_statement_classes": admin_classes,
            "statement_types": _safe_strings(validation.get("statement_types")),
            "target_resources": _safe_strings(validation.get("target_resources")),
            "guardrail_results": _safe_strings(validation.get("guardrail_results")),
            "sql_fingerprints": _safe_strings(validation.get("sql_fingerprints")),
            "validation_evidence_ids": _safe_strings(validation.get("evidence_ids")),
            "validation_task_ids": _safe_strings(validation.get("task_ids")),
        }

    planned = authoritative.get("operation") or governance_facts.get("operation")
    planned = planned if isinstance(planned, Mapping) else {}
    if bool(planned.get("destructive")) or bool(planned.get("admin_destructive")):
        return {
            "decision_source": "planned_operation_facts",
            "fact_source": planned.get("source") or "planning",
            "planned_destructive": bool(planned.get("destructive")),
            "planned_admin_destructive": bool(planned.get("admin_destructive")),
            "planned_admin": bool(planned.get("admin")),
            "operation_type": planned.get("operation_type") or operation_type,
            "access": planned.get("access"),
            "capability_ids": _safe_strings(planned.get("capability_ids")),
        }

    prompt_matches = _destructive_matches(_request_prompt(request))
    if not prompt_matches:
        return None
    if not _narrow_prompt_fallback_allowed(
        request,
        operation_type,
        metadata=metadata,
        planned=planned,
    ):
        return None
    return {
        "decision_source": "narrow_prompt_fallback",
        "fact_source": "prompt_fallback",
        "matched_terms": prompt_matches,
        "operation_type": operation_type,
        "access": _metadata_access(request, metadata),
    }


def _narrow_prompt_fallback_allowed(
    request: Any,
    operation_type: str,
    *,
    metadata: Mapping[str, Any] | None,
    planned: Mapping[str, Any],
) -> bool:
    if bool(planned.get("write_or_admin_context")):
        return True
    if _explicit_write_mode(request, metadata):
        return True
    access = _request_access(request) or _metadata_access_mode(metadata)
    if access in {AccessMode.WRITE, AccessMode.ADMIN}:
        return True
    capability = _request_capability(request)
    if capability is not None and (
        _capability_access(capability) in {AccessMode.WRITE, AccessMode.ADMIN}
        or _capability_side_effecting(capability)
    ):
        return True
    return operation_type in _WRITE_OPERATION_TYPES


def _metadata_access(request: Any, metadata: Mapping[str, Any] | None) -> str | None:
    access = _request_access(request) or _metadata_access_mode(metadata)
    if access not in {AccessMode.WRITE, AccessMode.ADMIN} and _explicit_write_mode(
        request,
        metadata,
    ):
        access = AccessMode.WRITE
    return access.value if access is not None else None


def _explicit_write_mode(
    request: Any,
    metadata: Mapping[str, Any] | None,
) -> bool:
    modes: list[str] = []
    if isinstance(request, Mapping):
        modes.append(str(request.get("mode") or "").lower())
        request_metadata = request.get("metadata")
        if isinstance(request_metadata, Mapping):
            modes.append(str(request_metadata.get("mode") or "").lower())
        safety_frame = request.get("safety_frame")
        if isinstance(safety_frame, Mapping):
            modes.append(str(safety_frame.get("explicit_mode") or "").lower())
    if isinstance(metadata, Mapping):
        modes.append(str(metadata.get("mode") or "").lower())
        safety_frame = metadata.get("safety_frame")
        if isinstance(safety_frame, Mapping):
            modes.append(str(safety_frame.get("explicit_mode") or "").lower())
    return any(_normalized_mode(mode) in _WRITE_MODE_ALIASES for mode in modes)


def _normalized_mode(mode: str) -> str:
    return str(mode or "").strip().lower().replace("-", "_")


def _metadata_access_mode(metadata: Mapping[str, Any] | None) -> AccessMode | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("access")
    if value is None:
        return None
    try:
        return AccessMode(value)
    except ValueError:
        return None


def _safe_strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, (list, tuple, set, frozenset)):
        return tuple(str(value) for value in values if value is not None)
    return (str(values),)


def _capability_requires_write_approval(request: Any) -> bool:
    capability = _request_capability(request)
    if capability is None:
        return False
    access = _capability_access(capability)
    if access is AccessMode.ADMIN:
        return True
    risk = _capability_risk(capability)
    if (
        access is AccessMode.WRITE
        and _capability_side_effecting(capability)
        and risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    ):
        return True
    return False


def _capability_access(capability: Mapping[str, Any]) -> AccessMode | None:
    value = capability.get("access")
    if value is None:
        return None
    try:
        return AccessMode(value)
    except ValueError:
        return None


def _capability_risk(capability: Mapping[str, Any]) -> RiskLevel | None:
    value = capability.get("risk")
    if value is None:
        return None
    try:
        return RiskLevel(value)
    except ValueError:
        return None


def _capability_side_effecting(capability: Mapping[str, Any]) -> bool:
    return bool(capability.get("side_effecting"))


def _destructive_matches(prompt: str) -> tuple[str, ...]:
    return tuple(
        match.group(0)
        for pattern in _DESTRUCTIVE_PATTERNS
        for match in pattern.finditer(prompt)
    )
