"""Portable governance facts, decisions, approvals, and default policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import re

from .._json import canonical_json
from ..capabilities import AccessMode, RiskLevel

_CANONICAL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


def _canonical_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _CANONICAL_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical lowercase sha256 hash")


def _fingerprint(material: dict[str, object]) -> str:
    encoded = canonical_json(material).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class GovernanceFacts:
    """Immutable facts bound to one exact materialized task."""

    operation_id: str
    task_id: str
    capability_id: str
    executor_id: str
    capability_fingerprint: str
    arguments_hash: str
    access_mode: AccessMode
    risk: RiskLevel
    side_effecting: bool
    idempotent: bool
    replay_safe: bool
    idempotency_key: str | None
    validation_passed: bool
    in_scope: bool
    destructive: bool
    sensitivity_class: str
    actor_id: str
    validation_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for field_name, text_value in (
            ("operation_id", self.operation_id),
            ("task_id", self.task_id),
            ("capability_id", self.capability_id),
            ("executor_id", self.executor_id),
            ("sensitivity_class", self.sensitivity_class),
            ("actor_id", self.actor_id),
        ):
            _required_text(text_value, field_name)
        _canonical_sha256(
            self.capability_fingerprint,
            "capability_fingerprint",
        )
        _canonical_sha256(self.arguments_hash, "arguments_hash")
        if self.validation_fingerprint is not None:
            _canonical_sha256(
                self.validation_fingerprint,
                "validation_fingerprint",
            )
        if not isinstance(self.access_mode, AccessMode):
            raise TypeError("access_mode must be an AccessMode")
        if not isinstance(self.risk, RiskLevel):
            raise TypeError("risk must be a RiskLevel")
        for field_name, flag_value in (
            ("side_effecting", self.side_effecting),
            ("idempotent", self.idempotent),
            ("replay_safe", self.replay_safe),
            ("validation_passed", self.validation_passed),
            ("in_scope", self.in_scope),
            ("destructive", self.destructive),
        ):
            if not isinstance(flag_value, bool):
                raise TypeError(f"{field_name} must be a boolean")

        if self.access_mode is AccessMode.READ and self.side_effecting:
            raise ValueError("read governance facts cannot declare a side effect")
        if self.replay_safe and not self.idempotent:
            raise ValueError("replay_safe governance facts must be idempotent")
        if self.idempotency_key is not None:
            _required_text(self.idempotency_key, "idempotency_key")
            if not self.side_effecting or not self.idempotent:
                raise ValueError("idempotency_key requires an idempotent side effect")
        if self.side_effecting and self.replay_safe and self.idempotency_key is None:
            raise ValueError("replay-safe side effects require an idempotency_key")
        if self.destructive and (
            self.access_mode is not AccessMode.WRITE or not self.side_effecting
        ):
            raise ValueError(
                "destructive governance facts require a side-effecting write"
            )

    @property
    def task_fingerprint(self) -> str:
        material: dict[str, object] = {
            "access_mode": self.access_mode.value,
            "actor_id": self.actor_id,
            "arguments_hash": self.arguments_hash,
            "capability_fingerprint": self.capability_fingerprint,
            "capability_id": self.capability_id,
            "destructive": self.destructive,
            "executor_id": self.executor_id,
            "idempotency_key": self.idempotency_key,
            "idempotent": self.idempotent,
            "in_scope": self.in_scope,
            "operation_id": self.operation_id,
            "replay_safe": self.replay_safe,
            "risk": self.risk.value,
            "sensitivity_class": self.sensitivity_class,
            "side_effecting": self.side_effecting,
            "task_id": self.task_id,
            "validation_passed": self.validation_passed,
        }
        # Schema-0 validation facts reproduce the exact Phase-2 fingerprint so
        # approvals already pending in a v12 database remain resumable.
        if self.validation_fingerprint is not None:
            material["validation_fingerprint"] = self.validation_fingerprint
        return _fingerprint(material)


@dataclass(frozen=True, slots=True)
class DefaultPolicyProfile:
    id: str = "daita.default"
    version: str = "1"
    allow_destructive: bool = False

    def __post_init__(self) -> None:
        _required_text(self.id, "policy id")
        _required_text(self.version, "policy version")
        if not isinstance(self.allow_destructive, bool):
            raise TypeError("allow_destructive must be a boolean")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "allow_destructive": self.allow_destructive,
                "id": self.id,
                "version": self.version,
            }
        )


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    effect: PolicyEffect
    code: str
    reason: str
    task_fingerprint: str
    policy_fingerprint: str
    evaluated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.effect, PolicyEffect):
            raise TypeError("effect must be a PolicyEffect")
        _required_text(self.code, "decision code")
        _required_text(self.reason, "decision reason")
        _canonical_sha256(self.task_fingerprint, "task_fingerprint")
        _canonical_sha256(self.policy_fingerprint, "policy_fingerprint")
        _aware(self.evaluated_at, "evaluated_at")


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    id: str
    operation_id: str
    task_id: str
    task_fingerprint: str
    policy_fingerprint: str
    requested_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("approval id", self.id),
            ("operation_id", self.operation_id),
            ("task_id", self.task_id),
        ):
            _required_text(value, field_name)
        _canonical_sha256(self.task_fingerprint, "task_fingerprint")
        _canonical_sha256(self.policy_fingerprint, "policy_fingerprint")
        _aware(self.requested_at, "requested_at")
        if not isinstance(self.status, ApprovalStatus):
            raise TypeError("status must be an ApprovalStatus")

        terminal_metadata = (
            self.decided_at,
            self.decided_by,
            self.decision_reason,
        )
        if self.status is ApprovalStatus.PENDING:
            if any(value is not None for value in terminal_metadata):
                raise ValueError(
                    "pending approval cannot contain terminal decision metadata"
                )
            return

        if any(value is None for value in terminal_metadata):
            raise ValueError(
                "terminal approval requires decided_at, decided_by, and "
                "decision_reason"
            )
        assert self.decided_at is not None
        assert self.decided_by is not None
        assert self.decision_reason is not None
        _aware(self.decided_at, "decided_at")
        _required_text(self.decided_by, "decided_by")
        _required_text(self.decision_reason, "decision_reason")
        if self.decided_at < self.requested_at:
            raise ValueError("decided_at cannot precede requested_at")


class DefaultPolicyEvaluator:
    """Evaluate the small fail-closed default policy from explicit facts."""

    def __init__(
        self,
        profile: DefaultPolicyProfile = DefaultPolicyProfile(),
    ) -> None:
        if not isinstance(profile, DefaultPolicyProfile):
            raise TypeError("profile must be a DefaultPolicyProfile")
        self._profile = profile

    @property
    def profile(self) -> DefaultPolicyProfile:
        return self._profile

    def evaluate(
        self,
        facts: GovernanceFacts,
        *,
        evaluated_at: datetime,
    ) -> GovernanceDecision:
        if not isinstance(facts, GovernanceFacts):
            raise TypeError("facts must be GovernanceFacts")
        _aware(evaluated_at, "evaluated_at")

        if not facts.validation_passed:
            effect = PolicyEffect.DENY
            code = "validation_failed"
            reason = "The task has no passing validation facts."
        elif not facts.in_scope:
            effect = PolicyEffect.DENY
            code = "out_of_scope"
            reason = "The task is outside its configured resource scope."
        elif facts.destructive and not self._profile.allow_destructive:
            effect = PolicyEffect.DENY
            code = "destructive_denied"
            reason = "The active policy denies destructive operations."
        elif facts.access_mode is AccessMode.WRITE or facts.side_effecting:
            effect = PolicyEffect.REQUIRE_APPROVAL
            code = "approval_required"
            reason = "The task writes or declares a side effect."
        else:
            effect = PolicyEffect.ALLOW
            code = "bounded_read_allowed"
            reason = "The validated bounded read is within scope."

        return GovernanceDecision(
            effect=effect,
            code=code,
            reason=reason,
            task_fingerprint=facts.task_fingerprint,
            policy_fingerprint=self._profile.fingerprint,
            evaluated_at=evaluated_at,
        )


__all__ = [
    "ApprovalRequest",
    "ApprovalStatus",
    "DefaultPolicyEvaluator",
    "DefaultPolicyProfile",
    "GovernanceDecision",
    "GovernanceFacts",
    "PolicyEffect",
]
