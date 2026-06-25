"""Policy evaluation boundary for runtime governance decisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from .primitives import (
    ApprovalRequest,
    GovernanceResult,
    Operation,
    Policy,
    PolicyDecision,
    PolicyEffect,
)


class PolicyEvaluator:
    """Evaluate registered policies without owning planning or execution."""

    def __init__(self, policies: Iterable[Policy] = ()) -> None:
        self._policies = tuple(policies)

    @property
    def policies(self) -> tuple[Policy, ...]:
        """Policies evaluated in deterministic registration order."""
        return self._policies

    def evaluate_operation(
        self,
        operation: Operation,
        *,
        contract: Mapping[str, Any] | None = None,
    ) -> GovernanceResult:
        """Evaluate applicable policies for one operation."""
        modified_contract = dict(contract or {})
        decisions: list[PolicyDecision] = []
        applicable_policy_count = 0

        for policy in self._policies:
            if not policy.applies_to(operation.request, operation.operation_type):
                continue
            applicable_policy_count += 1

            shaped_contract = policy.modify_contract(modified_contract)
            if shaped_contract is not None:
                modified_contract = _contract_dict(shaped_contract, policy.id)

            decision = policy.evaluate_operation(operation)
            if decision is None:
                continue
            if not isinstance(decision, PolicyDecision):
                raise TypeError(
                    f"policy {policy.id!r} must return PolicyDecision or None"
                )
            decision = _decision_with_policy_identity(policy, decision, operation)
            decisions.append(decision)
            if decision.effect is PolicyEffect.MODIFY and decision.modifications:
                modified_contract.update(decision.modifications)

        approval_requests = tuple(
            _approval_requests_for_decision(operation, decision, modified_contract)
            for decision in decisions
            if decision.effect is PolicyEffect.REQUIRE_APPROVAL
        )
        flattened_requests = tuple(
            request for group in approval_requests for request in group
        )
        blocked = any(decision.effect is PolicyEffect.DENY for decision in decisions)
        pending_approval = bool(flattened_requests)

        return GovernanceResult(
            allowed=not blocked and not pending_approval,
            blocked=blocked,
            pending_approval=pending_approval,
            decisions=tuple(decisions),
            approval_requests=flattened_requests,
            modified_contract=modified_contract,
            metadata={
                "policy_count": len(self._policies),
                "applicable_policy_count": applicable_policy_count,
            },
        )


def evaluate_policies(
    policies: Iterable[Policy],
    operation: Operation,
    *,
    contract: Mapping[str, Any] | None = None,
) -> GovernanceResult:
    """Evaluate policies for one operation with the default evaluator."""
    return PolicyEvaluator(policies).evaluate_operation(operation, contract=contract)


def _contract_dict(value: Any, policy_id: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"policy {policy_id!r} modify_contract() must return a mapping or None"
    )


def _decision_with_policy_identity(
    policy: Policy,
    decision: PolicyDecision,
    operation: Operation,
) -> PolicyDecision:
    version = str(
        getattr(policy, "policy_version", None)
        or getattr(policy, "version", None)
        or decision.policy_version
        or "1"
    )
    owner = str(getattr(policy, "owner", None) or decision.owner)
    policy_id = str(getattr(policy, "id", None) or decision.policy_id)
    identity = f"{owner}:{policy_id}@{version}"
    metadata = {
        **decision.metadata,
        "policy_identity": identity,
        "policy_version": version,
    }
    return replace(
        decision,
        policy_id=policy_id,
        owner=owner,
        policy_version=version,
        policy_identity=identity,
        operation_id=decision.operation_id or operation.id,
        metadata=metadata,
    )


def _approval_requests_for_decision(
    operation: Operation,
    decision: PolicyDecision,
    modified_contract: Mapping[str, Any],
) -> tuple[ApprovalRequest, ...]:
    approval_names = decision.required_approvals or (decision.policy_id,)
    evidence_ids = tuple(
        evidence.id for evidence in decision.evidence if evidence.id is not None
    )

    return tuple(
        ApprovalRequest(
            approval_id=f"{operation.id}:{decision.policy_id}:{approval_name}",
            operation_id=operation.id,
            reason=decision.reason,
            proposed_action={
                "operation_type": operation.operation_type,
                "request": operation.request,
                "contract": dict(modified_contract),
                "approval": approval_name,
            },
            risk=decision.severity,
            evidence_ids=evidence_ids,
            requested_by_policy_id=decision.policy_id,
            owner=decision.owner,
            metadata={
                "decision": decision.to_dict(),
                "policy_identity": decision.policy_identity,
                "policy_version": decision.policy_version,
            },
        )
        for approval_name in approval_names
    )
