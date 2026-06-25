from dataclasses import dataclass, field

import pytest

from daita.plugins import ExtensionRegistry, PluginKind, PluginManifest
from daita.runtime import (
    Evidence,
    Operation,
    PolicyDecision,
    PolicyEffect,
    PolicyEvaluator,
    RiskLevel,
    evaluate_policies,
)


@dataclass
class RecordingPolicy:
    id: str
    owner: str = "governance"
    version: str = "2026-06-03"
    operation_type: str = "data.query"
    decision: PolicyDecision | None = None
    contract_update: dict | None = None
    calls: list[str] = field(default_factory=list)

    def applies_to(self, request, operation_type: str) -> bool:
        self.calls.append(f"applies:{operation_type}")
        return operation_type == self.operation_type

    def modify_contract(self, contract):
        self.calls.append("modify")
        if self.contract_update is None:
            return contract
        return {**contract, **self.contract_update}

    def evaluate_operation(self, operation: Operation):
        self.calls.append(f"evaluate:{operation.id}")
        return self.decision


class EvidencePolicy(RecordingPolicy):
    def evaluate_operation(self, operation: Operation):
        return Evidence(
            kind="governance.decision",
            owner=self.owner,
            payload={"legacy": True},
        )


class GovernancePlugin:
    manifest = PluginManifest(
        id="governance",
        display_name="Governance",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self, policies):
        self._policies = tuple(policies)

    def declare_policies(self):
        return self._policies


def _operation(operation_type: str = "data.query") -> Operation:
    return Operation(
        id="op-1",
        operation_type=operation_type,
        request={"prompt": "show orders"},
    )


def _decision(
    policy_id: str,
    effect: PolicyEffect | str,
    *,
    severity: RiskLevel = RiskLevel.LOW,
    required_approvals: tuple[str, ...] = (),
    modifications: dict | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        policy_id=policy_id,
        owner="governance",
        effect=effect,
        reason=f"{policy_id} decided {effect}",
        severity=severity,
        required_approvals=required_approvals,
        modifications=modifications or {},
    )


def test_policy_evaluator_filters_applicable_policies_and_shapes_contract():
    matching = RecordingPolicy(
        id="governance.limit",
        decision=_decision(
            "governance.limit",
            PolicyEffect.MODIFY,
            modifications={"limit": 100},
        ),
        contract_update={"profile": "restricted"},
    )
    skipped = RecordingPolicy(
        id="governance.write_only",
        operation_type="data.write",
        decision=_decision("governance.write_only", PolicyEffect.DENY),
    )

    result = PolicyEvaluator((matching, skipped)).evaluate_operation(
        _operation(),
        contract={"limit": 1000},
    )

    assert result.allowed is True
    assert result.blocked is False
    assert result.pending_approval is False
    assert [decision.policy_id for decision in result.decisions] == ["governance.limit"]
    assert result.modified_contract == {"limit": 100, "profile": "restricted"}
    assert result.metadata == {"policy_count": 2, "applicable_policy_count": 1}
    assert matching.calls == ["applies:data.query", "modify", "evaluate:op-1"]
    assert skipped.calls == ["applies:data.query"]


def test_policy_evaluator_composes_deny_and_approval_deterministically():
    allow = RecordingPolicy(
        id="governance.allow",
        decision=_decision("governance.allow", PolicyEffect.ALLOW),
    )
    approval = RecordingPolicy(
        id="governance.require_approval",
        decision=_decision(
            "governance.require_approval",
            PolicyEffect.REQUIRE_APPROVAL,
            severity=RiskLevel.HIGH,
            required_approvals=("data_owner", "security"),
        ),
    )
    deny = RecordingPolicy(
        id="governance.deny",
        decision=_decision("governance.deny", PolicyEffect.DENY),
    )

    result = evaluate_policies((allow, approval, deny), _operation())

    assert result.allowed is False
    assert result.blocked is True
    assert result.pending_approval is True
    assert [decision.policy_id for decision in result.decisions] == [
        "governance.allow",
        "governance.require_approval",
        "governance.deny",
    ]
    assert [request.approval_id for request in result.approval_requests] == [
        "op-1:governance.require_approval:data_owner",
        "op-1:governance.require_approval:security",
    ]
    assert result.approval_requests[0].proposed_action["operation_type"] == (
        "data.query"
    )


def test_policy_evaluator_reads_policies_registered_by_extension_registry():
    policy = RecordingPolicy(
        id="governance.warn",
        decision=_decision("governance.warn", PolicyEffect.WARN),
    )
    registry = ExtensionRegistry()
    registry.register(GovernancePlugin([policy]))

    result = PolicyEvaluator(registry.policies).evaluate_operation(_operation())

    assert result.allowed is True
    assert result.decisions[0].policy_id == policy.decision.policy_id
    assert result.decisions[0].policy_version == "2026-06-03"
    assert result.decisions[0].policy_identity == (
        "governance:governance.warn@2026-06-03"
    )


def test_policy_evaluator_adds_stable_policy_identity_and_version_to_decisions():
    policy = RecordingPolicy(
        id="governance.require_approval",
        owner="acme",
        version="v7",
        decision=PolicyDecision(
            policy_id="ignored.local_id",
            owner="ignored",
            effect=PolicyEffect.REQUIRE_APPROVAL,
            reason="Approval required.",
            severity=RiskLevel.HIGH,
            required_approvals=("owner",),
        ),
    )

    result = PolicyEvaluator((policy,)).evaluate_operation(_operation())
    decision = result.decisions[0]
    approval = result.approval_requests[0]

    assert decision.policy_id == "governance.require_approval"
    assert decision.owner == "acme"
    assert decision.policy_version == "v7"
    assert decision.policy_identity == "acme:governance.require_approval@v7"
    assert decision.metadata["policy_identity"] == decision.policy_identity
    assert approval.metadata["policy_version"] == "v7"
    assert approval.metadata["policy_identity"] == decision.policy_identity


def test_policy_evaluator_rejects_legacy_evidence_policy_results():
    policy = EvidencePolicy(id="governance.legacy")

    with pytest.raises(TypeError, match="PolicyDecision"):
        PolicyEvaluator((policy,)).evaluate_operation(_operation())


def test_policy_evaluator_rejects_non_mapping_contract_modifications():
    class BadContractPolicy(RecordingPolicy):
        def modify_contract(self, contract):
            return ["not", "a", "mapping"]

    policy = BadContractPolicy(
        id="governance.bad_contract",
        decision=_decision("governance.bad_contract", PolicyEffect.ALLOW),
    )

    with pytest.raises(TypeError, match="mapping"):
        PolicyEvaluator((policy,)).evaluate_operation(_operation())
