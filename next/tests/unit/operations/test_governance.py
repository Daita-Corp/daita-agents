from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import hashlib

import pytest

from daita._json import canonical_json
from daita.capabilities import AccessMode, RiskLevel
from daita.operations.governance import (
    ApprovalRequest,
    ApprovalStatus,
    DefaultPolicyEvaluator,
    DefaultPolicyProfile,
    GovernanceDecision,
    GovernanceFacts,
    PolicyEffect,
)

NOW = datetime(2026, 7, 18, 15, 0, tzinfo=timezone.utc)
CAPABILITY_HASH = "sha256:" + ("a" * 64)
ARGUMENTS_HASH = "sha256:" + ("b" * 64)


def _facts(**changes: object) -> GovernanceFacts:
    values: dict[str, object] = {
        "operation_id": "operation-governed",
        "task_id": "task-governed",
        "capability_id": "fake.read",
        "executor_id": "fake.read.executor",
        "capability_fingerprint": CAPABILITY_HASH,
        "arguments_hash": ARGUMENTS_HASH,
        "access_mode": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "side_effecting": False,
        "idempotent": True,
        "replay_safe": True,
        "idempotency_key": None,
        "validation_passed": True,
        "in_scope": True,
        "destructive": False,
        "sensitivity_class": "internal",
        "actor_id": "actor-local",
    }
    values.update(changes)
    return GovernanceFacts(**values)  # type: ignore[arg-type]


def test_policy_and_approval_vocabularies_are_small_and_exact() -> None:
    assert tuple(effect.value for effect in PolicyEffect) == (
        "allow",
        "deny",
        "require_approval",
    )
    assert tuple(status.value for status in ApprovalStatus) == (
        "pending",
        "approved",
        "denied",
        "cancelled",
    )


def test_governance_facts_bind_every_authoritative_input_in_one_fingerprint() -> None:
    facts = _facts()
    expected_material = {
        "access_mode": "read",
        "actor_id": "actor-local",
        "arguments_hash": ARGUMENTS_HASH,
        "capability_fingerprint": CAPABILITY_HASH,
        "capability_id": "fake.read",
        "destructive": False,
        "executor_id": "fake.read.executor",
        "idempotency_key": None,
        "idempotent": True,
        "in_scope": True,
        "operation_id": "operation-governed",
        "replay_safe": True,
        "risk": "low",
        "sensitivity_class": "internal",
        "side_effecting": False,
        "task_id": "task-governed",
        "validation_passed": True,
    }
    expected = (
        "sha256:"
        + hashlib.sha256(canonical_json(expected_material).encode("utf-8")).hexdigest()
    )

    assert facts.task_fingerprint == expected
    assert _facts().task_fingerprint == facts.task_fingerprint
    assert replace(facts, actor_id="actor-other").task_fingerprint != expected
    with pytest.raises(FrozenInstanceError):
        facts.actor_id = "actor-other"  # type: ignore[misc]


def test_explicit_validation_fingerprint_extends_without_changing_legacy_hash() -> None:
    legacy = _facts()
    explicit = replace(
        legacy,
        validation_fingerprint="sha256:" + ("c" * 64),
    )

    assert legacy.validation_fingerprint is None
    assert explicit.task_fingerprint != legacy.task_fingerprint
    assert (
        replace(
            explicit, validation_fingerprint="sha256:" + ("d" * 64)
        ).task_fingerprint
        != explicit.task_fingerprint
    )
    with pytest.raises(ValueError, match="validation_fingerprint"):
        replace(legacy, validation_fingerprint="sha256:bad")


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"operation_id": ""}, "operation_id"),
        ({"capability_fingerprint": "sha256:ABC"}, "canonical lowercase"),
        ({"arguments_hash": "b" * 64}, "canonical lowercase"),
        ({"validation_passed": 1}, "validation_passed"),
        ({"access_mode": "read"}, "AccessMode"),
        ({"risk": "low"}, "RiskLevel"),
        ({"replay_safe": True, "idempotent": False}, "replay_safe"),
        (
            {
                "access_mode": AccessMode.WRITE,
                "side_effecting": True,
                "idempotent": False,
                "replay_safe": False,
                "idempotency_key": "stable-key",
            },
            "idempotency_key",
        ),
        ({"destructive": True}, "destructive"),
    ),
)
def test_governance_facts_reject_malformed_or_incoherent_inputs(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _facts(**changes)


def test_default_policy_profile_has_a_canonical_versioned_fingerprint() -> None:
    profile = DefaultPolicyProfile()
    expected = (
        "sha256:"
        + hashlib.sha256(
            canonical_json(
                {
                    "allow_destructive": False,
                    "id": "daita.default",
                    "version": "1",
                }
            ).encode("utf-8")
        ).hexdigest()
    )

    assert profile.id == "daita.default"
    assert profile.version == "1"
    assert not profile.allow_destructive
    assert profile.fingerprint == expected
    assert replace(profile, allow_destructive=True).fingerprint != expected
    with pytest.raises(ValueError, match="policy id"):
        DefaultPolicyProfile(id="")
    with pytest.raises(TypeError, match="allow_destructive"):
        DefaultPolicyProfile(allow_destructive=1)  # type: ignore[arg-type]


def test_governance_decision_is_immutable_and_requires_bound_fingerprints() -> None:
    facts = _facts()
    profile = DefaultPolicyProfile()
    decision = GovernanceDecision(
        effect=PolicyEffect.ALLOW,
        code="bounded_read_allowed",
        reason="The validated bounded read is in scope.",
        task_fingerprint=facts.task_fingerprint,
        policy_fingerprint=profile.fingerprint,
        evaluated_at=NOW,
    )

    assert decision.effect is PolicyEffect.ALLOW
    with pytest.raises(FrozenInstanceError):
        decision.code = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(decision, evaluated_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="task_fingerprint"):
        replace(decision, task_fingerprint="sha256:bad")


@pytest.mark.parametrize(
    "status",
    (ApprovalStatus.APPROVED, ApprovalStatus.DENIED, ApprovalStatus.CANCELLED),
)
def test_terminal_approval_requires_complete_decision_metadata(
    status: ApprovalStatus,
) -> None:
    facts = _facts()
    profile = DefaultPolicyProfile()
    approval = ApprovalRequest(
        id="approval-governed",
        operation_id=facts.operation_id,
        task_id=facts.task_id,
        task_fingerprint=facts.task_fingerprint,
        policy_fingerprint=profile.fingerprint,
        status=status,
        requested_at=NOW,
        decided_at=NOW,
        decided_by="actor-reviewer",
        decision_reason="Explicit decision.",
    )

    assert approval.status is status
    assert approval.decided_at == NOW
    with pytest.raises(ValueError, match="terminal approval"):
        replace(approval, decided_by=None)


def test_pending_approval_rejects_terminal_decision_metadata() -> None:
    facts = _facts()
    profile = DefaultPolicyProfile()
    pending = ApprovalRequest(
        id="approval-governed",
        operation_id=facts.operation_id,
        task_id=facts.task_id,
        task_fingerprint=facts.task_fingerprint,
        policy_fingerprint=profile.fingerprint,
        requested_at=NOW,
    )

    assert pending.status is ApprovalStatus.PENDING
    assert pending.decided_at is None
    with pytest.raises(ValueError, match="pending approval"):
        replace(
            pending,
            decided_at=NOW,
            decided_by="actor-reviewer",
            decision_reason="Premature metadata.",
        )
    with pytest.raises(ValueError, match="requested_at"):
        replace(
            pending,
            status=ApprovalStatus.APPROVED,
            decided_at=NOW.replace(year=2025),
            decided_by="actor-reviewer",
            decision_reason="Invalid chronology.",
        )


@pytest.mark.parametrize(
    ("facts", "effect", "code"),
    (
        (
            _facts(
                validation_passed=False,
                in_scope=False,
                access_mode=AccessMode.WRITE,
                risk=RiskLevel.HIGH,
                side_effecting=True,
                idempotent=False,
                replay_safe=False,
                destructive=True,
            ),
            PolicyEffect.DENY,
            "validation_failed",
        ),
        (
            _facts(in_scope=False, access_mode=AccessMode.WRITE),
            PolicyEffect.DENY,
            "out_of_scope",
        ),
        (
            _facts(
                access_mode=AccessMode.WRITE,
                risk=RiskLevel.HIGH,
                side_effecting=True,
                idempotent=False,
                replay_safe=False,
                destructive=True,
            ),
            PolicyEffect.DENY,
            "destructive_denied",
        ),
        (
            _facts(
                access_mode=AccessMode.WRITE,
                risk=RiskLevel.MEDIUM,
                side_effecting=True,
                idempotent=False,
                replay_safe=False,
            ),
            PolicyEffect.REQUIRE_APPROVAL,
            "approval_required",
        ),
        (
            _facts(),
            PolicyEffect.ALLOW,
            "bounded_read_allowed",
        ),
    ),
)
def test_default_policy_enforces_fail_closed_precedence(
    facts: GovernanceFacts,
    effect: PolicyEffect,
    code: str,
) -> None:
    evaluator = DefaultPolicyEvaluator()

    decision = evaluator.evaluate(facts, evaluated_at=NOW)

    assert decision.effect is effect
    assert decision.code == code
    assert decision.task_fingerprint == facts.task_fingerprint
    assert decision.policy_fingerprint == evaluator.profile.fingerprint
    assert decision.evaluated_at == NOW


def test_destructive_test_profile_still_requires_write_approval() -> None:
    facts = _facts(
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
        destructive=True,
    )
    evaluator = DefaultPolicyEvaluator(DefaultPolicyProfile(allow_destructive=True))

    decision = evaluator.evaluate(facts, evaluated_at=NOW)

    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL
    assert decision.code == "approval_required"


def test_default_policy_rejects_wrong_facts_and_naive_evaluation_time() -> None:
    evaluator = DefaultPolicyEvaluator()
    with pytest.raises(TypeError, match="GovernanceFacts"):
        evaluator.evaluate(object(), evaluated_at=NOW)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluator.evaluate(_facts(), evaluated_at=NOW.replace(tzinfo=None))
