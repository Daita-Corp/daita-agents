"""Default governance policies for the DB runtime."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from daita.runtime import (
    AccessMode,
    Operation,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
)

from .authorization import (
    authorization_concrete_facts,
    authorization_from_governance_request,
    match_preauthorization_grant,
)

_WRITE_OPERATION_TYPES = frozenset({"write.execute", "admin"})
_DESTRUCTIVE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bdrop\s+(table|database|schema|view|index)\b",
        r"\btruncate\s+(table\s+)?\w+",
        r"\balter\s+(table|database|schema|view|index)\b",
        r"\bdelete\s+from\b",
        r"\bdelete\s+all\b",
        r"\bwipe\b",
        r"\bpurge\b",
        r"\bdestroy\b",
    )
)


class DbLaneContractPolicy:
    """Deny concrete tasks that exceed their lane-built operation contract."""

    id = "deny_lane_contract_violations"
    owner = "runtime"
    version = "1"

    def applies_to(self, request: Any, operation_type: str) -> bool:
        return bool(_lane_contract_facts(request))

    def modify_contract(self, contract: Any) -> Any:
        return contract

    def evaluate_operation(self, operation: Operation) -> PolicyDecision | None:
        facts = _lane_contract_facts(operation.request)
        if not facts:
            return None
        violation = _lane_contract_violation(facts)
        if violation is None:
            return None
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.DENY,
            reason=violation["reason"],
            severity=RiskLevel.HIGH,
            operation_id=operation.id,
            metadata=violation,
        )


class DbAuthorizationPolicy:
    """Enforce deny/preauthorized modes for concrete DB work."""

    id = "enforce_authorization_modes"
    owner = "runtime"
    version = "1"

    def applies_to(self, request: Any, operation_type: str) -> bool:
        authorization = authorization_from_governance_request(request)
        mode = authorization.get("mode", "interactive")
        if mode == "interactive":
            return False
        if mode == "deny":
            return _authorization_has_db_surface(request, operation_type)
        return _authorization_has_concrete_target(request)

    def modify_contract(self, contract: Any) -> Any:
        return contract

    def evaluate_operation(self, operation: Operation) -> PolicyDecision | None:
        authorization = authorization_from_governance_request(operation.request)
        mode = authorization.get("mode", "interactive")
        if mode == "interactive":
            return None
        destructive = _destructive_decision_facts(
            operation.request,
            operation.operation_type,
            operation.metadata,
        )
        require_destructive = bool(
            destructive
            and (
                destructive.get("destructive_statement_classes")
                or destructive.get("planned_destructive")
            )
        )
        require_admin = bool(
            destructive
            and (
                destructive.get("admin_statement_classes")
                or destructive.get("planned_admin")
                or destructive.get("planned_admin_destructive")
            )
        )
        concrete = authorization_concrete_facts(
            operation.request,
            require_destructive=require_destructive,
            require_admin=require_admin,
        )
        if mode == "deny":
            return PolicyDecision(
                policy_id=self.id,
                owner=self.owner,
                effect=PolicyEffect.DENY,
                reason="DB work is not authorized for this caller.",
                severity=RiskLevel.HIGH,
                operation_id=operation.id,
                metadata={
                    "authorization_mode": mode,
                    **concrete,
                },
            )
        if mode != "preauthorized" or not _authorization_has_concrete_target(
            operation.request
        ):
            return None

        match = match_preauthorization_grant(
            operation.request,
            require_destructive=require_destructive,
            require_admin=require_admin,
        )
        if match["matched"]:
            return PolicyDecision(
                policy_id=self.id,
                owner=self.owner,
                effect=PolicyEffect.ALLOW,
                reason="DB work matched an explicit preauthorization grant.",
                severity=RiskLevel.LOW,
                operation_id=operation.id,
                metadata={
                    "authorization_mode": mode,
                    "authorization_grant_id": match.get("grant_id"),
                    **match["facts"],
                },
            )
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.DENY,
            reason="DB work requires an explicit matching preauthorization grant.",
            severity=RiskLevel.HIGH,
            operation_id=operation.id,
            metadata={
                "authorization_mode": mode,
                "authorization_denial": match.get("reason"),
                **match["facts"],
            },
        )


class DbWriteApprovalPolicy:
    """Require human approval before interactive DB side effects."""

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
        authorization = authorization_from_governance_request(operation.request)
        mode = authorization.get("mode", "interactive")
        if mode != "interactive":
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
                "authorization_mode": mode,
                "operation_type": operation.operation_type,
                "access": operation.metadata.get("access"),
                "capability": _request_capability(operation.request),
            },
        )


class DbDestructiveOperationPolicy:
    """Deny destructive/admin SQL for interactive work by default."""

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
        authorization = authorization_from_governance_request(operation.request)
        mode = authorization.get("mode", "interactive")
        if mode != "interactive":
            return None
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.DENY,
            reason="Destructive DB operations are denied by default.",
            severity=RiskLevel.CRITICAL,
            operation_id=operation.id,
            metadata={
                "authorization_mode": mode,
                "operation_type": operation.operation_type,
                **facts,
            },
        )


def default_db_policies() -> tuple[Any, ...]:
    """Return the default DB runtime governance policies."""
    return (
        DbLaneContractPolicy(),
        DbAuthorizationPolicy(),
        DbDestructiveOperationPolicy(),
        DbWriteApprovalPolicy(),
    )


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


def _lane_contract_facts(request: Any) -> dict[str, Any]:
    governance_facts = _request_governance_facts(request)
    contract = governance_facts.get("contract")
    operation = governance_facts.get("operation")
    task = governance_facts.get("task")
    capability = governance_facts.get("capability")
    if not isinstance(contract, Mapping):
        contract = {}
    if not isinstance(operation, Mapping):
        operation = {}
    if not isinstance(task, Mapping):
        task = {}
    if not isinstance(capability, Mapping):
        capability = {}
    granted_lanes = _safe_strings(
        contract.get("granted_lanes") or operation.get("granted_lanes")
    )
    forbidden_capabilities = _safe_strings(
        contract.get("forbidden_capabilities")
        or operation.get("forbidden_capabilities")
    )
    if not granted_lanes and not forbidden_capabilities:
        return {}
    return {
        "contract": dict(contract),
        "operation": dict(operation),
        "task": dict(task),
        "capability": dict(capability),
        "validation": (
            governance_facts.get("validation")
            if isinstance(governance_facts.get("validation"), Mapping)
            else {}
        ),
        "granted_lanes": granted_lanes,
        "forbidden_capabilities": forbidden_capabilities,
    }


def _lane_contract_violation(facts: Mapping[str, Any]) -> dict[str, Any] | None:
    contract = facts["contract"]
    task = facts["task"]
    capability = facts["capability"]
    granted_lanes = frozenset(facts["granted_lanes"])
    forbidden_capabilities = frozenset(facts["forbidden_capabilities"])
    capability_id = str(task.get("capability_id") or capability.get("id") or "").strip()
    if not capability_id:
        return None
    if capability_id in forbidden_capabilities:
        return _lane_violation(
            "Task capability is forbidden by the safety lane contract.",
            "forbidden_capability",
            capability_id,
            granted_lanes,
            contract,
            task,
            capability,
        )

    allowed_ids = frozenset(
        {
            *_safe_strings(contract.get("required_capabilities")),
            *_safe_strings(contract.get("selected_capability_ids")),
            "db.answer.synthesize",
            *(
                str(item.get("id"))
                for item in contract.get("selected_capabilities") or ()
                if isinstance(item, Mapping) and item.get("id")
            ),
        }
    )
    if capability_id not in allowed_ids:
        return _lane_violation(
            "Task capability is outside the operation contract.",
            "capability_outside_contract",
            capability_id,
            granted_lanes,
            contract,
            task,
            capability,
        )

    access = _capability_access(capability)
    if capability_id == "db.sql.execute_read" and "read" not in granted_lanes:
        return _lane_violation(
            "SQL read execution exceeds the granted lanes.",
            "sql_read_lane_required",
            capability_id,
            granted_lanes,
            contract,
            task,
            capability,
        )
    if capability_id == "db.sql.execute_write" and "write_execute" not in granted_lanes:
        return _lane_violation(
            "SQL write execution exceeds the granted lanes.",
            "sql_write_execute_lane_required",
            capability_id,
            granted_lanes,
            contract,
            task,
            capability,
        )
    if access is AccessMode.ADMIN and "admin" not in granted_lanes:
        return _lane_violation(
            "Admin capability access exceeds the granted lanes.",
            "admin_lane_required",
            capability_id,
            granted_lanes,
            contract,
            task,
            capability,
        )
    if access is AccessMode.WRITE and not _write_access_allowed(
        capability_id, granted_lanes
    ):
        return _lane_violation(
            "Write capability access exceeds the granted lanes.",
            "write_lane_required",
            capability_id,
            granted_lanes,
            contract,
            task,
            capability,
        )
    if (
        access is AccessMode.READ
        and "read" not in granted_lanes
        and not _selected_declared_operation_capability(
            capability_id,
            contract,
            capability,
        )
    ):
        return _lane_violation(
            "Read capability access exceeds the granted lanes.",
            "read_lane_required",
            capability_id,
            granted_lanes,
            contract,
            task,
            capability,
        )
    return None


def _write_access_allowed(capability_id: str, granted_lanes: frozenset[str]) -> bool:
    if capability_id.startswith("db.monitor."):
        return bool({"monitor_write", "monitor_execute"} & granted_lanes)
    if capability_id.startswith("db.memory.") or capability_id.startswith("memory."):
        return "memory_write" in granted_lanes
    return bool({"write_execute", "admin"} & granted_lanes)


def _selected_declared_operation_capability(
    capability_id: str,
    contract: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> bool:
    selected = frozenset(
        {
            *_safe_strings(contract.get("selected_capability_ids")),
            *(
                str(item.get("id"))
                for item in contract.get("selected_capabilities") or ()
                if isinstance(item, Mapping) and item.get("id")
            ),
        }
    )
    if capability_id not in selected:
        return False
    operation_type = str(contract.get("operation_type") or "")
    return operation_type in _safe_strings(capability.get("operation_types"))


def _lane_violation(
    reason: str,
    code: str,
    capability_id: str,
    granted_lanes: frozenset[str],
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "decision_source": "lane_contract_facts",
        "violation": code,
        "reason": reason,
        "capability_id": capability_id,
        "executor_id": task.get("executor_id"),
        "owner": task.get("owner") or capability.get("owner"),
        "requested_lane": task.get("requested_lane"),
        "required_lane": task.get("required_lane"),
        "granted_lanes": tuple(sorted(granted_lanes)),
        "forbidden_capabilities": _safe_strings(contract.get("forbidden_capabilities")),
        "required_capabilities": _safe_strings(contract.get("required_capabilities")),
        "selected_capability_ids": _safe_strings(
            contract.get("selected_capability_ids")
        ),
        "contract_operation_type": contract.get("operation_type"),
        "contract_access": contract.get("access"),
        "capability_access": capability.get("access"),
    }


def _authorization_has_db_surface(request: Any, operation_type: str) -> bool:
    governance_facts = _request_governance_facts(request)
    authoritative = governance_facts.get("authoritative")
    authoritative = authoritative if isinstance(authoritative, Mapping) else {}
    operation = authoritative.get("operation") or governance_facts.get("operation")
    operation = operation if isinstance(operation, Mapping) else {}
    contract = authoritative.get("contract") or governance_facts.get("contract")
    contract = contract if isinstance(contract, Mapping) else {}
    return bool(
        _authorization_has_concrete_target(request)
        or operation_type in _WRITE_OPERATION_TYPES
        or operation.get("access")
        or contract.get("access")
        or operation.get("capability_ids")
        or contract.get("required_capabilities")
    )


def _authorization_has_concrete_target(request: Any) -> bool:
    governance_facts = _request_governance_facts(request)
    authoritative = governance_facts.get("authoritative")
    authoritative = authoritative if isinstance(authoritative, Mapping) else {}
    task = authoritative.get("task") or governance_facts.get("task")
    capability = authoritative.get("capability") or governance_facts.get("capability")
    return bool(
        (isinstance(task, Mapping) and task.get("id"))
        or (isinstance(capability, Mapping) and capability.get("id"))
    )


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
    return access.value if access is not None else None


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
