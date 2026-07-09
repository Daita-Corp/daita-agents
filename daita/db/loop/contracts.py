"""Contract and intent helpers for the DB agent loop."""

from __future__ import annotations

from typing import Any

from daita.runtime import AccessMode, GovernanceResult, Operation

from ..models import DbIntent, DbIntentKind, DbLimits, DbOperationContract
from ..planner_protocol import DbLoopState
from .utils import _string_tuple


def _fallback_contract_for_operation(
    operation: Operation,
    *,
    limits: DbLimits,
) -> DbOperationContract:
    return DbOperationContract(
        operation_type=operation.operation_type,
        required_evidence=tuple(sorted(operation.required_evidence)),
        access=AccessMode.NONE,
        limits=limits,
        policy_ids=(),
        metadata={"source": "operation_state"},
    )


def _contract_from_latest_snapshot(
    operation: Operation,
    fallback: DbOperationContract,
) -> DbOperationContract:
    context = operation.metadata.get("resume_context")
    context = context if isinstance(context, dict) else {}
    snapshot = (
        operation.metadata.get("latest_compiled_contract_snapshot")
        or context.get("latest_compiled_contract_snapshot")
        or context.get("contract")
    )
    if not isinstance(snapshot, dict):
        return fallback
    limits = snapshot.get("limits")
    limits = limits if isinstance(limits, dict) else {}
    metadata = snapshot.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    try:
        access = AccessMode(str(snapshot.get("access") or fallback.access.value))
    except ValueError:
        access = fallback.access
    try:
        db_limits = DbLimits(**limits) if limits else fallback.limits
    except (TypeError, ValueError):
        db_limits = fallback.limits
    return DbOperationContract(
        operation_type=str(snapshot.get("operation_type") or fallback.operation_type),
        required_evidence=_string_tuple(snapshot.get("required_evidence")),
        access=access,
        limits=db_limits,
        policy_ids=_string_tuple(snapshot.get("policy_ids")),
        metadata=metadata,
    )


def _fallback_intent_for_operation(
    operation: Operation,
    contract: DbOperationContract,
) -> DbIntent:
    context = operation.metadata.get("resume_context")
    context = context if isinstance(context, dict) else {}
    intent_context = context.get("intent")
    intent_context = intent_context if isinstance(intent_context, dict) else {}
    kind_value = str(intent_context.get("kind") or operation.operation_type)
    try:
        kind = DbIntentKind(kind_value)
    except ValueError:
        kind = DbIntentKind.CONVERSATIONAL
    return DbIntent(
        kind=kind,
        confidence=1.0,
        access=contract.access,
        evidence_mode="planner_loop",
        diagnostics={
            "source": "operation_state",
            "operation_type": operation.operation_type,
        },
    )


def _intent_from_contract(
    contract: DbOperationContract,
    fallback: DbIntent,
) -> DbIntent:
    intent_metadata = contract.metadata.get("planner_intent")
    intent_metadata = intent_metadata if isinstance(intent_metadata, dict) else {}
    operation_type = str(
        intent_metadata.get("operation_type") or contract.operation_type
    )
    try:
        kind = DbIntentKind(operation_type)
    except ValueError:
        kind = fallback.kind
    return DbIntent(
        kind=kind,
        confidence=1.0,
        access=contract.access,
        evidence_mode="planner_loop",
        diagnostics={
            "source": "planner_compiled_contract",
            "operation_type": operation_type,
            "planner_intent": intent_metadata,
        },
    )


def _intent_summary(intent: DbIntent) -> dict[str, Any]:
    return {
        "kind": intent.kind.value,
        "access": intent.access.value,
        "evidence_mode": intent.evidence_mode,
        "diagnostics": intent.diagnostics,
    }


def _contract_snapshot(contract: DbOperationContract) -> dict[str, Any]:
    return {
        "operation_type": contract.operation_type,
        "required_capabilities": list(contract.required_capabilities),
        "required_evidence": list(contract.required_evidence),
        "access": contract.access.value,
        "limits": contract.limits.to_dict(),
        "policy_ids": list(contract.policy_ids),
        "metadata": contract.metadata,
    }


def _governance_summary(governance: GovernanceResult | None) -> dict[str, Any]:
    if governance is None:
        return {}
    return governance.to_dict()


def _access_rank(value: str) -> int:
    return _ACCESS_ORDER.get(value, _ACCESS_ORDER[AccessMode.ADMIN.value])


def _state_allows_read_profile(state: DbLoopState) -> bool:
    safety_frame = state.safety_frame or {}
    max_access = str(
        safety_frame.get("max_access")
        or safety_frame.get("max_allowed_access")
        or AccessMode.ADMIN.value
    )
    return _access_rank(max_access) >= _access_rank(AccessMode.READ.value)


_ACCESS_ORDER = {
    AccessMode.NONE.value: 0,
    AccessMode.METADATA_READ.value: 1,
    AccessMode.READ.value: 2,
    AccessMode.WRITE.value: 3,
    AccessMode.ADMIN.value: 4,
}
