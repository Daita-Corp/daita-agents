"""Portable, redaction-safe learning proposal contracts.

Learning is a durable proposal lifecycle, not an execution path.  This module
owns candidate safety and state-transition legality while leaving persistence
behind a narrow protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256
import re
from typing import Protocol, runtime_checkable
import unicodedata

from ._json import FrozenJsonObject, FrozenJsonValue, canonical_json

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_IDENTITY_CHARACTERS = 512
_MAX_CANDIDATE_CHARACTERS = 16_000
_MAX_STRING_CHARACTERS = 4_096
_MAX_CONTAINER_ITEMS = 64
_MAX_DEPTH = 8


def _required_text(
    value: str,
    field_name: str,
    *,
    maximum: int = _MAX_IDENTITY_CHARACTERS,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")


def _optional_text(value: str | None, field_name: str) -> None:
    if value is not None:
        _required_text(value, field_name)


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


def _positive_int(value: int | None, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must use canonical lowercase sha256")


class LearningProposalKind(str, Enum):
    MEMORY = "memory"
    SKILL = "skill"


class LearningProposalState(str, Enum):
    PROPOSED = "proposed"
    COMMITTED = "committed"
    REJECTED = "rejected"


class LearningCandidateCategory(str, Enum):
    EXPLICIT_CORRECTION = "explicit_correction"
    EVIDENCE_BACKED_FACT = "evidence_backed_fact"
    SKILL_CHANGE = "skill_change"


class LearningSourceOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class LearningRejectionCategory(str, Enum):
    RAW_ROW_OR_PII = "raw_row_or_pii"
    SECRET_OR_CREDENTIAL = "secret_or_credential"
    POLICY_OR_SECURITY_MUTATION = "policy_or_security_mutation"
    EXECUTABLE_OR_RUNTIME_EFFECT = "executable_or_runtime_effect"
    OUT_OF_BOUNDS = "out_of_bounds"
    INELIGIBLE_SOURCE = "ineligible_source"


@dataclass(frozen=True, slots=True)
class LearningProvenance:
    agent_id: str
    operation_id: str
    trigger_id: str
    source_outcome: LearningSourceOutcome
    source_hash: str
    evidence_id: str | None = None
    evidence_accepted: bool = False

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "learning provenance agent_id")
        _required_text(self.operation_id, "learning provenance operation_id")
        _required_text(self.trigger_id, "learning provenance trigger_id")
        if not isinstance(self.source_outcome, LearningSourceOutcome):
            raise TypeError(
                "learning provenance source_outcome must be a LearningSourceOutcome"
            )
        _require_hash(self.source_hash, "learning provenance source_hash")
        _optional_text(self.evidence_id, "learning provenance evidence_id")
        if not isinstance(self.evidence_accepted, bool):
            raise TypeError("learning provenance evidence_accepted must be a boolean")
        if (self.evidence_id is None) != (not self.evidence_accepted):
            raise ValueError(
                "learning evidence provenance requires an accepted evidence_id"
            )


@dataclass(frozen=True, slots=True)
class LearningSafetyDecision:
    candidate_hash: str
    allowed: bool
    candidate_payload: Mapping[str, object] | None = None
    rejection_category: LearningRejectionCategory | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        _require_hash(self.candidate_hash, "learning safety candidate_hash")
        if not isinstance(self.allowed, bool):
            raise TypeError("learning safety allowed must be a boolean")
        _optional_text(self.rejection_reason, "learning safety rejection_reason")
        if self.allowed:
            if self.candidate_payload is None:
                raise ValueError("allowed learning candidate requires a payload")
            if self.rejection_category is not None or self.rejection_reason is not None:
                raise ValueError(
                    "allowed learning candidate cannot contain a rejection"
                )
            payload = FrozenJsonObject.from_mapping(self.candidate_payload)
            if not payload:
                raise ValueError("allowed learning candidate cannot be empty")
            if learning_candidate_hash(payload) != self.candidate_hash:
                raise ValueError("learning candidate payload does not match its hash")
            encoded = canonical_json(payload)
            if (
                len(encoded) > _MAX_CANDIDATE_CHARACTERS
                or _candidate_violation(
                    payload,
                    path="$",
                    depth=0,
                )
                is not None
            ):
                raise ValueError("allowed learning candidate failed safety validation")
            object.__setattr__(self, "candidate_payload", payload)
        else:
            if self.candidate_payload is not None:
                raise ValueError(
                    "rejected learning candidate cannot retain raw payload"
                )
            if not isinstance(self.rejection_category, LearningRejectionCategory):
                raise TypeError(
                    "rejected learning candidate requires a rejection category"
                )
            if self.rejection_reason is None:
                raise ValueError("rejected learning candidate requires a reason")


@dataclass(frozen=True, slots=True)
class LearningDecision:
    proposal_id: str
    idempotency_key: str
    candidate_hash: str
    state: LearningProposalState
    decided_at: datetime
    result_memory_id: str | None = None
    result_memory_version: int | None = None
    result_skill_id: str | None = None
    result_skill_version: int | None = None
    rejection_category: LearningRejectionCategory | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.proposal_id, "learning decision proposal_id")
        _require_hash(self.idempotency_key, "learning decision idempotency_key")
        _require_hash(self.candidate_hash, "learning decision candidate_hash")
        if self.state is LearningProposalState.PROPOSED:
            raise ValueError("learning decision must be committed or rejected")
        if not isinstance(self.state, LearningProposalState):
            raise TypeError("learning decision state must be a LearningProposalState")
        _aware(self.decided_at, "learning decision decided_at")
        _optional_text(self.result_memory_id, "learning decision result_memory_id")
        _optional_text(self.result_skill_id, "learning decision result_skill_id")
        _optional_text(self.rejection_reason, "learning decision rejection_reason")
        if self.result_memory_version is not None:
            _positive_int(
                self.result_memory_version,
                "learning decision result_memory_version",
            )
        if self.result_skill_version is not None:
            _positive_int(
                self.result_skill_version,
                "learning decision result_skill_version",
            )

        memory_ref = (
            self.result_memory_id is not None and self.result_memory_version is not None
        )
        skill_ref = (
            self.result_skill_id is not None and self.result_skill_version is not None
        )
        partial_ref = (self.result_memory_id is None) != (
            self.result_memory_version is None
        ) or (self.result_skill_id is None) != (self.result_skill_version is None)
        if partial_ref:
            raise ValueError("learning decision result references must be complete")
        if self.state is LearningProposalState.COMMITTED:
            if memory_ref == skill_ref:
                raise ValueError(
                    "committed learning decision requires exactly one result reference"
                )
            if self.rejection_category is not None or self.rejection_reason is not None:
                raise ValueError(
                    "committed learning decision cannot contain a rejection"
                )
        else:
            if memory_ref or skill_ref:
                raise ValueError("rejected learning decision cannot contain a result")
            if not isinstance(self.rejection_category, LearningRejectionCategory):
                raise TypeError("rejected learning decision requires a category")
            if self.rejection_reason is None:
                raise ValueError("rejected learning decision requires a reason")

    @property
    def fingerprint(self) -> str:
        """Hash the semantic decision; replay timestamps do not change identity."""

        return _hash_json(
            {
                "candidate_hash": self.candidate_hash,
                "idempotency_key": self.idempotency_key,
                "proposal_id": self.proposal_id,
                "rejection_category": (
                    None
                    if self.rejection_category is None
                    else self.rejection_category.value
                ),
                "rejection_reason": self.rejection_reason,
                "result_memory_id": self.result_memory_id,
                "result_memory_version": self.result_memory_version,
                "result_skill_id": self.result_skill_id,
                "result_skill_version": self.result_skill_version,
                "state": self.state.value,
            }
        )


@dataclass(frozen=True, slots=True)
class LearningProposal:
    id: str
    kind: LearningProposalKind
    category: LearningCandidateCategory
    state: LearningProposalState
    provenance: LearningProvenance
    candidate_hash: str
    idempotency_key: str
    created_at: datetime
    candidate_payload: Mapping[str, object] | None = None
    resolved_at: datetime | None = None
    decision_hash: str | None = None
    result_memory_id: str | None = None
    result_memory_version: int | None = None
    result_skill_id: str | None = None
    result_skill_version: int | None = None
    rejection_category: LearningRejectionCategory | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "learning proposal id")
        if not isinstance(self.kind, LearningProposalKind):
            raise TypeError("learning proposal kind must be a LearningProposalKind")
        if not isinstance(self.category, LearningCandidateCategory):
            raise TypeError(
                "learning proposal category must be a LearningCandidateCategory"
            )
        if (
            self.kind is LearningProposalKind.SKILL
            and self.category is not LearningCandidateCategory.SKILL_CHANGE
        ) or (
            self.kind is LearningProposalKind.MEMORY
            and self.category is LearningCandidateCategory.SKILL_CHANGE
        ):
            raise ValueError("learning proposal kind and category do not match")
        if not isinstance(self.state, LearningProposalState):
            raise TypeError("learning proposal state must be a LearningProposalState")
        if not isinstance(self.provenance, LearningProvenance):
            raise TypeError("learning proposal provenance must be LearningProvenance")
        _require_hash(self.candidate_hash, "learning proposal candidate_hash")
        _require_hash(self.idempotency_key, "learning proposal idempotency_key")
        expected_key = learning_idempotency_key(
            self.kind,
            self.category,
            self.provenance,
            self.candidate_hash,
        )
        if self.idempotency_key != expected_key:
            raise ValueError("learning proposal idempotency_key is not canonical")
        _aware(self.created_at, "learning proposal created_at")
        if self.resolved_at is not None:
            _aware(self.resolved_at, "learning proposal resolved_at")
            if self.resolved_at < self.created_at:
                raise ValueError(
                    "learning proposal resolved_at cannot precede creation"
                )
        _optional_text(self.result_memory_id, "learning proposal result_memory_id")
        _optional_text(self.result_skill_id, "learning proposal result_skill_id")
        _optional_text(self.rejection_reason, "learning proposal rejection_reason")
        if self.result_memory_version is not None:
            _positive_int(
                self.result_memory_version,
                "learning proposal result_memory_version",
            )
        if self.result_skill_version is not None:
            _positive_int(
                self.result_skill_version,
                "learning proposal result_skill_version",
            )

        memory_ref = (
            self.result_memory_id is not None and self.result_memory_version is not None
        )
        skill_ref = (
            self.result_skill_id is not None and self.result_skill_version is not None
        )
        partial_ref = (self.result_memory_id is None) != (
            self.result_memory_version is None
        ) or (self.result_skill_id is None) != (self.result_skill_version is None)
        if partial_ref:
            raise ValueError("learning proposal result references must be complete")

        if self.state is LearningProposalState.PROPOSED:
            if self.provenance.source_outcome is not LearningSourceOutcome.SUCCEEDED:
                raise ValueError("only a succeeded source may retain a proposal")
            if self.candidate_payload is None:
                raise ValueError("proposed learning requires a candidate payload")
            payload = FrozenJsonObject.from_mapping(self.candidate_payload)
            if not payload:
                raise ValueError("proposed learning candidate cannot be empty")
            if learning_candidate_hash(payload) != self.candidate_hash:
                raise ValueError(
                    "learning proposal payload does not match candidate_hash"
                )
            if not validate_learning_candidate(payload).allowed:
                raise ValueError("proposed learning payload failed safety validation")
            if any(
                value is not None
                for value in (
                    self.resolved_at,
                    self.decision_hash,
                    self.result_memory_id,
                    self.result_memory_version,
                    self.result_skill_id,
                    self.result_skill_version,
                    self.rejection_category,
                    self.rejection_reason,
                )
            ):
                raise ValueError("proposed learning cannot contain terminal fields")
            object.__setattr__(self, "candidate_payload", payload)
            return

        if self.resolved_at is None or self.decision_hash is None:
            raise ValueError("terminal learning proposal requires resolution metadata")
        _require_hash(self.decision_hash, "learning proposal decision_hash")
        if self.state is LearningProposalState.COMMITTED:
            if self.provenance.source_outcome is not LearningSourceOutcome.SUCCEEDED:
                raise ValueError("failed or blocked operations cannot commit learning")
            if self.candidate_payload is None:
                raise ValueError("committed learning must retain its sanitized payload")
            payload = FrozenJsonObject.from_mapping(self.candidate_payload)
            if learning_candidate_hash(payload) != self.candidate_hash:
                raise ValueError(
                    "learning proposal payload does not match candidate_hash"
                )
            if not validate_learning_candidate(payload).allowed:
                raise ValueError("committed learning payload failed safety validation")
            if self.kind is LearningProposalKind.MEMORY and not memory_ref:
                raise ValueError(
                    "committed memory learning requires a memory reference"
                )
            if self.kind is LearningProposalKind.SKILL and not skill_ref:
                raise ValueError("committed skill learning requires a skill reference")
            if memory_ref == skill_ref:
                raise ValueError(
                    "committed learning requires exactly one result reference"
                )
            if self.rejection_category is not None or self.rejection_reason is not None:
                raise ValueError("committed learning cannot contain a rejection")
            object.__setattr__(self, "candidate_payload", payload)
        else:
            if self.candidate_payload is not None:
                raise ValueError(
                    "rejected learning cannot retain raw candidate payload"
                )
            if memory_ref or skill_ref:
                raise ValueError("rejected learning cannot contain a result reference")
            if not isinstance(self.rejection_category, LearningRejectionCategory):
                raise TypeError("rejected learning requires a rejection category")
            if self.rejection_reason is None:
                raise ValueError("rejected learning requires a rejection reason")

        assert self.resolved_at is not None
        terminal_decision = LearningDecision(
            proposal_id=self.id,
            idempotency_key=self.idempotency_key,
            candidate_hash=self.candidate_hash,
            state=self.state,
            decided_at=self.resolved_at,
            result_memory_id=self.result_memory_id,
            result_memory_version=self.result_memory_version,
            result_skill_id=self.result_skill_id,
            result_skill_version=self.result_skill_version,
            rejection_category=self.rejection_category,
            rejection_reason=self.rejection_reason,
        )
        if self.decision_hash != terminal_decision.fingerprint:
            raise ValueError("learning proposal decision_hash is not canonical")

    @classmethod
    def create(
        cls,
        *,
        proposal_id: str,
        kind: LearningProposalKind,
        category: LearningCandidateCategory,
        provenance: LearningProvenance,
        candidate: Mapping[str, object],
        created_at: datetime,
    ) -> LearningProposal:
        """Build a proposed record or a terminal redacted rejection."""

        if not isinstance(kind, LearningProposalKind):
            raise TypeError("kind must be a LearningProposalKind")
        if not isinstance(category, LearningCandidateCategory):
            raise TypeError("category must be a LearningCandidateCategory")
        if not isinstance(provenance, LearningProvenance):
            raise TypeError("provenance must be LearningProvenance")
        safety = validate_learning_candidate(candidate)
        idempotency_key = learning_idempotency_key(
            kind,
            category,
            provenance,
            safety.candidate_hash,
        )
        if provenance.source_outcome is not LearningSourceOutcome.SUCCEEDED:
            return _rejected_proposal(
                proposal_id=proposal_id,
                kind=kind,
                category=category,
                provenance=provenance,
                candidate_hash=safety.candidate_hash,
                idempotency_key=idempotency_key,
                created_at=created_at,
                rejection_category=LearningRejectionCategory.INELIGIBLE_SOURCE,
                rejection_reason=(
                    f"source_operation_{provenance.source_outcome.value}"
                ),
            )
        if not safety.allowed:
            assert safety.rejection_category is not None
            assert safety.rejection_reason is not None
            return _rejected_proposal(
                proposal_id=proposal_id,
                kind=kind,
                category=category,
                provenance=provenance,
                candidate_hash=safety.candidate_hash,
                idempotency_key=idempotency_key,
                created_at=created_at,
                rejection_category=safety.rejection_category,
                rejection_reason=safety.rejection_reason,
            )
        assert safety.candidate_payload is not None
        return cls(
            id=proposal_id,
            kind=kind,
            category=category,
            state=LearningProposalState.PROPOSED,
            provenance=provenance,
            candidate_hash=safety.candidate_hash,
            idempotency_key=idempotency_key,
            candidate_payload=safety.candidate_payload,
            created_at=created_at,
        )

    @classmethod
    def reject(
        cls,
        *,
        proposal_id: str,
        kind: LearningProposalKind,
        category: LearningCandidateCategory,
        provenance: LearningProvenance,
        candidate: Mapping[str, object],
        created_at: datetime,
        rejection_category: LearningRejectionCategory,
        rejection_reason: str,
    ) -> LearningProposal:
        """Build one redacted rejection for a contextually ineligible candidate."""

        if not isinstance(kind, LearningProposalKind):
            raise TypeError("kind must be a LearningProposalKind")
        if not isinstance(category, LearningCandidateCategory):
            raise TypeError("category must be a LearningCandidateCategory")
        if not isinstance(provenance, LearningProvenance):
            raise TypeError("provenance must be LearningProvenance")
        if not isinstance(rejection_category, LearningRejectionCategory):
            raise TypeError("rejection_category must be LearningRejectionCategory")
        _required_text(rejection_reason, "learning rejection_reason")
        safety = validate_learning_candidate(candidate)
        resolved_category = (
            rejection_category if safety.allowed else safety.rejection_category
        )
        resolved_reason = (
            rejection_reason if safety.allowed else safety.rejection_reason
        )
        assert resolved_category is not None
        assert resolved_reason is not None
        return _rejected_proposal(
            proposal_id=proposal_id,
            kind=kind,
            category=category,
            provenance=provenance,
            candidate_hash=safety.candidate_hash,
            idempotency_key=learning_idempotency_key(
                kind,
                category,
                provenance,
                safety.candidate_hash,
            ),
            created_at=created_at,
            rejection_category=resolved_category,
            rejection_reason=resolved_reason,
        )


class LearningTransitionError(RuntimeError):
    """Raised when a resolution conflicts with durable proposal state."""


def resolve_learning_proposal(
    proposal: LearningProposal,
    decision: LearningDecision,
) -> LearningProposal:
    """Apply one idempotent terminal decision without persisting it."""

    if not isinstance(proposal, LearningProposal):
        raise TypeError("proposal must be a LearningProposal")
    if not isinstance(decision, LearningDecision):
        raise TypeError("decision must be a LearningDecision")
    if (
        decision.proposal_id != proposal.id
        or decision.idempotency_key != proposal.idempotency_key
        or decision.candidate_hash != proposal.candidate_hash
    ):
        raise LearningTransitionError(
            "learning decision does not match proposal identity"
        )
    if proposal.state is not LearningProposalState.PROPOSED:
        if proposal.decision_hash == decision.fingerprint:
            return proposal
        raise LearningTransitionError("learning proposal already has another decision")
    if decision.decided_at < proposal.created_at:
        raise LearningTransitionError("learning decision cannot precede its proposal")
    if (
        decision.state is LearningProposalState.COMMITTED
        and proposal.provenance.source_outcome is not LearningSourceOutcome.SUCCEEDED
    ):
        raise LearningTransitionError(
            "failed or blocked operations cannot commit learning"
        )
    memory_ref = decision.result_memory_id is not None
    skill_ref = decision.result_skill_id is not None
    if decision.state is LearningProposalState.COMMITTED:
        if proposal.kind is LearningProposalKind.MEMORY and not memory_ref:
            raise LearningTransitionError(
                "memory proposal decision requires a memory result"
            )
        if proposal.kind is LearningProposalKind.SKILL and not skill_ref:
            raise LearningTransitionError(
                "skill proposal decision requires a skill result"
            )
    return replace(
        proposal,
        state=decision.state,
        candidate_payload=(
            proposal.candidate_payload
            if decision.state is LearningProposalState.COMMITTED
            else None
        ),
        resolved_at=decision.decided_at,
        decision_hash=decision.fingerprint,
        result_memory_id=decision.result_memory_id,
        result_memory_version=decision.result_memory_version,
        result_skill_id=decision.result_skill_id,
        result_skill_version=decision.result_skill_version,
        rejection_category=decision.rejection_category,
        rejection_reason=decision.rejection_reason,
    )


def validate_learning_candidate(
    candidate: Mapping[str, object],
) -> LearningSafetyDecision:
    """Return a canonical payload or a redacted deterministic rejection."""

    if not isinstance(candidate, Mapping):
        raise TypeError("learning candidate must be a mapping")
    frozen = FrozenJsonObject.from_mapping(candidate)
    encoded = canonical_json(frozen)
    candidate_hash = _hash_bytes(encoded.encode("utf-8"))
    if not frozen:
        return _unsafe(
            candidate_hash,
            LearningRejectionCategory.OUT_OF_BOUNDS,
            "candidate_empty",
        )
    if len(encoded) > _MAX_CANDIDATE_CHARACTERS:
        return _unsafe(
            candidate_hash,
            LearningRejectionCategory.OUT_OF_BOUNDS,
            "candidate_too_large",
        )
    violation = _candidate_violation(frozen, path="$", depth=0)
    if violation is not None:
        category, reason = violation
        return _unsafe(candidate_hash, category, reason)
    return LearningSafetyDecision(
        candidate_hash=candidate_hash,
        allowed=True,
        candidate_payload=frozen,
    )


def learning_candidate_hash(candidate: Mapping[str, object]) -> str:
    if not isinstance(candidate, Mapping):
        raise TypeError("learning candidate must be a mapping")
    return _hash_bytes(canonical_json(candidate).encode("utf-8"))


def learning_idempotency_key(
    kind: LearningProposalKind,
    category: LearningCandidateCategory,
    provenance: LearningProvenance,
    candidate_hash: str,
) -> str:
    if not isinstance(kind, LearningProposalKind):
        raise TypeError("learning kind must be a LearningProposalKind")
    if not isinstance(category, LearningCandidateCategory):
        raise TypeError("learning category must be a LearningCandidateCategory")
    if not isinstance(provenance, LearningProvenance):
        raise TypeError("learning provenance must be LearningProvenance")
    _require_hash(candidate_hash, "learning candidate_hash")
    return _hash_json(
        {
            "agent_id": provenance.agent_id,
            "candidate_hash": candidate_hash,
            "category": category.value,
            "evidence_id": provenance.evidence_id,
            "kind": kind.value,
            "operation_id": provenance.operation_id,
            "source_hash": provenance.source_hash,
            "source_outcome": provenance.source_outcome.value,
            "trigger_id": provenance.trigger_id,
        }
    )


class LearningStoreError(RuntimeError):
    """Base class for portable learning-store failures."""


class LearningStoreConflictError(LearningStoreError):
    """Raised when a proposal transition loses its state/version guard."""


@runtime_checkable
class LearningStore(Protocol):
    async def create_proposal(
        self,
        proposal: LearningProposal,
    ) -> LearningProposal: ...

    async def load_proposal(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> LearningProposal | None: ...

    async def list_proposals(
        self,
        agent_id: str,
        *,
        operation_id: str | None,
        states: tuple[LearningProposalState, ...],
        limit: int,
    ) -> tuple[LearningProposal, ...]: ...

    async def resolve_proposal(
        self,
        decision: LearningDecision,
        *,
        expected_state: LearningProposalState,
    ) -> LearningProposal: ...


_RAW_ROW_KEYS = frozenset(
    {
        "raw_row",
        "raw_rows",
        "row",
        "rows",
        "record",
        "records",
        "result_rows",
        "query_results",
        "sample_rows",
        "observed_values",
    }
)
_PII_KEYS = frozenset(
    {
        "address",
        "birth_date",
        "card_number",
        "credit_card",
        "cvv",
        "date_of_birth",
        "dob",
        "email",
        "national_id",
        "passport",
        "phone",
        "postal_code",
        "social_security",
        "ssn",
        "street_address",
    }
)
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "api_secret",
        "auth_token",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "token",
    }
)
_POLICY_KEYS = frozenset(
    {
        "access_control",
        "approval_policy",
        "governance",
        "permission",
        "permissions",
        "policy",
        "policies",
        "security",
        "security_policy",
    }
)
_EXECUTABLE_KEYS = frozenset(
    {
        "code",
        "command",
        "entrypoint",
        "exec",
        "executor",
        "executor_id",
        "import",
        "imports",
        "javascript",
        "module",
        "python",
        "runtime_effect",
        "runtime_effects",
        "script",
        "shell",
        "source_code",
        "subprocess",
        "tool_call",
        "tool_calls",
        "worker",
    }
)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(
    r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)" r"\d{3}[\s.-]?\d{4}(?!\w)"
)
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SECRET_TEXT = re.compile(
    r"(?i)(?:\bsk-[a-z0-9_-]{12,}\b|\bAKIA[A-Z0-9]{16}\b|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"\b(?:api[_ -]?key|credential|password|secret|token)\s*[:=]\s*\S+)"
)
_POLICY_MUTATION_TEXT = re.compile(
    r"(?i)(?:"
    r"\b(?:allow|bypass|change|disable[ds]?|grant|modify|override[ds]?|remove|set|weaken)\b"
    r".{0,64}\b(?:approval|governance|permission|policy|security)\b|"
    r"\b(?:approval|governance|permission|policy|security)\b.{0,64}"
    r"\b(?:allow|bypass|change|disable[ds]?|grant|modify|override[ds]?|remove|set|weaken)\b"
    r")"
)
_EXECUTABLE_TEXT = re.compile(
    r"(?im)(?:^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+\s*\()|"
    r"\b(?:eval|exec|subprocess)\s*\(|<script\b|^#!\s*/)"
)


def _candidate_violation(
    value: FrozenJsonValue,
    *,
    path: str,
    depth: int,
) -> tuple[LearningRejectionCategory, str] | None:
    if depth > _MAX_DEPTH:
        return LearningRejectionCategory.OUT_OF_BOUNDS, "candidate_nesting_too_deep"
    if isinstance(value, FrozenJsonObject):
        if len(value) > _MAX_CONTAINER_ITEMS:
            return LearningRejectionCategory.OUT_OF_BOUNDS, "candidate_object_too_large"
        for key, item in value.items():
            if len(key) > 128:
                return LearningRejectionCategory.OUT_OF_BOUNDS, "candidate_key_too_long"
            normalized_key = _normalized_key(key)
            if normalized_key in _RAW_ROW_KEYS:
                return LearningRejectionCategory.RAW_ROW_OR_PII, "raw_row_field"
            if normalized_key in _PII_KEYS:
                return LearningRejectionCategory.RAW_ROW_OR_PII, "pii_field"
            if normalized_key in _SECRET_KEYS or (
                normalized_key.endswith("_token") and normalized_key != "token_budget"
            ):
                return (
                    LearningRejectionCategory.SECRET_OR_CREDENTIAL,
                    "credential_field",
                )
            if normalized_key in _POLICY_KEYS:
                return (
                    LearningRejectionCategory.POLICY_OR_SECURITY_MUTATION,
                    "policy_or_security_field",
                )
            if normalized_key in _EXECUTABLE_KEYS:
                return (
                    LearningRejectionCategory.EXECUTABLE_OR_RUNTIME_EFFECT,
                    "executable_or_runtime_field",
                )
            if (
                normalized_key in {"data", "items", "output", "results", "values"}
                and isinstance(item, tuple)
                and any(isinstance(element, FrozenJsonObject) for element in item)
            ):
                return LearningRejectionCategory.RAW_ROW_OR_PII, "raw_row_shape"
            nested = _candidate_violation(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
            if nested is not None:
                return nested
        return None
    if isinstance(value, tuple):
        if len(value) > _MAX_CONTAINER_ITEMS:
            return LearningRejectionCategory.OUT_OF_BOUNDS, "candidate_array_too_large"
        for index, item in enumerate(value):
            nested = _candidate_violation(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )
            if nested is not None:
                return nested
        return None
    if isinstance(value, str):
        if len(value) > _MAX_STRING_CHARACTERS:
            return LearningRejectionCategory.OUT_OF_BOUNDS, "candidate_string_too_long"
        if _contains_pii(value):
            return LearningRejectionCategory.RAW_ROW_OR_PII, "pii_value"
        if _SECRET_TEXT.search(value):
            return LearningRejectionCategory.SECRET_OR_CREDENTIAL, "credential_value"
        if _POLICY_MUTATION_TEXT.search(value):
            return (
                LearningRejectionCategory.POLICY_OR_SECURITY_MUTATION,
                "policy_or_security_mutation",
            )
        if _EXECUTABLE_TEXT.search(value):
            return (
                LearningRejectionCategory.EXECUTABLE_OR_RUNTIME_EFFECT,
                "executable_content",
            )
    return None


def _contains_pii(value: str) -> bool:
    if _EMAIL.search(value) or _SSN.search(value) or _PHONE.search(value):
        return True
    for match in _CARD.finditer(value):
        if _luhn_valid(match.group(0)):
            return True
    return False


def _luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _normalized_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "_".join(token for token in re.split(r"[^a-z0-9]+", normalized) if token)


def _unsafe(
    candidate_hash: str,
    category: LearningRejectionCategory,
    reason: str,
) -> LearningSafetyDecision:
    return LearningSafetyDecision(
        candidate_hash=candidate_hash,
        allowed=False,
        rejection_category=category,
        rejection_reason=reason,
    )


def _rejected_proposal(
    *,
    proposal_id: str,
    kind: LearningProposalKind,
    category: LearningCandidateCategory,
    provenance: LearningProvenance,
    candidate_hash: str,
    idempotency_key: str,
    created_at: datetime,
    rejection_category: LearningRejectionCategory,
    rejection_reason: str,
) -> LearningProposal:
    decision = LearningDecision(
        proposal_id=proposal_id,
        idempotency_key=idempotency_key,
        candidate_hash=candidate_hash,
        state=LearningProposalState.REJECTED,
        decided_at=created_at,
        rejection_category=rejection_category,
        rejection_reason=rejection_reason,
    )
    return LearningProposal(
        id=proposal_id,
        kind=kind,
        category=category,
        state=LearningProposalState.REJECTED,
        provenance=provenance,
        candidate_hash=candidate_hash,
        idempotency_key=idempotency_key,
        created_at=created_at,
        resolved_at=created_at,
        decision_hash=decision.fingerprint,
        rejection_category=rejection_category,
        rejection_reason=rejection_reason,
    )


def _hash_json(value: Mapping[str, object]) -> str:
    return _hash_bytes(canonical_json(value).encode("utf-8"))


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()


__all__ = [
    "LearningCandidateCategory",
    "LearningDecision",
    "LearningProposal",
    "LearningProposalKind",
    "LearningProposalState",
    "LearningProvenance",
    "LearningRejectionCategory",
    "LearningSafetyDecision",
    "LearningSourceOutcome",
    "LearningStore",
    "LearningStoreConflictError",
    "LearningStoreError",
    "LearningTransitionError",
    "learning_candidate_hash",
    "learning_idempotency_key",
    "resolve_learning_proposal",
    "validate_learning_candidate",
]
