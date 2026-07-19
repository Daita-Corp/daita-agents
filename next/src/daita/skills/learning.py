"""Explicit, inert skill-change proposals over the portable skill lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from .._json import FrozenJsonObject
from ..learning import (
    LearningCandidateCategory,
    LearningDecision,
    LearningProposal,
    LearningProposalKind,
    LearningProposalState,
    LearningProvenance,
    LearningStore,
    learning_candidate_hash,
)
from .models import (
    Skill,
    SkillActivation,
    SkillActivationMode,
    SkillIndex,
    SkillInspection,
    SkillSource,
    SkillVersion,
)
from .service import (
    SkillActivationConflictError,
    SkillNotFoundError,
    SkillService,
    SkillStore,
)

_CANDIDATE_KEYS = frozenset(
    {
        "activation_mode",
        "description",
        "domains",
        "instructions",
        "policy_notes",
        "required_capability_ids",
        "resource_kinds",
        "sensitivity_notes",
        "stable_name",
        "version",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class SkillChangeLearningError(RuntimeError):
    """Base class for portable skill-change learning failures."""


class SkillChangeFormatError(SkillChangeLearningError):
    """Raised when a skill-change payload cannot form portable skill records."""


class SkillChangeConflictError(SkillChangeLearningError):
    """Raised when proposal, version, or active-pointer state changed."""


@dataclass(frozen=True, slots=True)
class SkillChangeCandidate:
    """One bounded procedure version proposed for explicit acceptance."""

    stable_name: str
    version: str
    description: str
    instructions: str
    activation_mode: SkillActivationMode = SkillActivationMode.EXPLICIT
    domains: tuple[str, ...] = ()
    resource_kinds: tuple[str, ...] = ()
    required_capability_ids: tuple[str, ...] = ()
    sensitivity_notes: str | None = None
    policy_notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.activation_mode, SkillActivationMode):
            raise TypeError("skill-change activation_mode must be SkillActivationMode")
        for field_name in (
            "domains",
            "resource_kinds",
            "required_capability_ids",
        ):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)):
                raise TypeError(f"skill-change {field_name} must be a sequence")
            object.__setattr__(self, field_name, tuple(value))

    @property
    def payload(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "activation_mode": self.activation_mode.value,
                "description": self.description,
                "domains": self.domains,
                "instructions": self.instructions,
                "policy_notes": self.policy_notes,
                "required_capability_ids": self.required_capability_ids,
                "resource_kinds": self.resource_kinds,
                "sensitivity_notes": self.sensitivity_notes,
                "stable_name": self.stable_name,
                "version": self.version,
            }
        )


@dataclass(frozen=True, slots=True)
class SkillChangeProposalResult:
    """Durable proposal plus a non-persisted inert version preview when safe."""

    proposal: LearningProposal
    proposed_version: SkillVersion | None

    def __post_init__(self) -> None:
        _require_skill_proposal(self.proposal)
        if self.proposal.state is LearningProposalState.REJECTED:
            if self.proposed_version is not None:
                raise ValueError("rejected skill change cannot expose raw version data")
            return
        if not isinstance(self.proposed_version, SkillVersion):
            raise TypeError("safe skill change requires a SkillVersion preview")
        if (
            self.proposed_version.agent_id != self.proposal.provenance.agent_id
            or self.proposed_version.source is not SkillSource.LEARNED_PROPOSAL
            or self.proposed_version.content_hash != self.proposal.candidate_hash
        ):
            raise ValueError("skill-change preview does not match its proposal")


@dataclass(frozen=True, slots=True)
class SkillChangeCommit:
    """Complete input for one atomic proposal acceptance and activation."""

    proposal: LearningProposal
    skill: Skill
    version: SkillVersion
    staged_index: SkillIndex
    activation: SkillActivation
    decision: LearningDecision
    expected_active_version_id: str | None
    expected_skill_version_count: int

    def __post_init__(self) -> None:
        _require_skill_proposal(self.proposal)
        if self.proposal.state is not LearningProposalState.PROPOSED:
            raise ValueError("skill-change commit requires a proposed proposal")
        for value, expected_type, field_name in (
            (self.skill, Skill, "skill"),
            (self.version, SkillVersion, "version"),
            (self.staged_index, SkillIndex, "staged_index"),
            (self.activation, SkillActivation, "activation"),
            (self.decision, LearningDecision, "decision"),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"skill-change {field_name} has an invalid type")
        if (
            not isinstance(self.expected_skill_version_count, int)
            or isinstance(self.expected_skill_version_count, bool)
            or self.expected_skill_version_count < 0
        ):
            raise ValueError("expected_skill_version_count must be non-negative")
        if (
            self.expected_skill_version_count == 0
            and self.expected_active_version_id is not None
        ):
            raise ValueError("new skill change cannot expect an active version")
        if (
            self.skill.agent_id != self.proposal.provenance.agent_id
            or self.version.agent_id != self.skill.agent_id
            or self.staged_index.agent_id != self.skill.agent_id
            or self.activation.agent_id != self.skill.agent_id
            or self.version.skill_id != self.skill.id
            or self.staged_index.skill_id != self.skill.id
            or self.activation.skill_id != self.skill.id
            or self.version.stable_name != self.skill.stable_name
            or self.staged_index.stable_name != self.skill.stable_name
            or self.skill.source is not SkillSource.LEARNED_PROPOSAL
            or self.version.source is not self.skill.source
            or self.staged_index.source is not self.skill.source
        ):
            raise ValueError("skill-change commit records do not share one identity")
        if not self.staged_index.matches(self.version):
            raise ValueError("staged skill index does not match its immutable version")
        if self.staged_index.active_version_id != self.expected_active_version_id:
            raise ValueError("staged skill index changed the active pointer")
        if (
            self.version.content_hash != self.proposal.candidate_hash
            or self.activation.version_id != self.version.id
            or self.activation.previous_version_id != self.expected_active_version_id
        ):
            raise ValueError("skill-change activation does not match the proposal")
        if (
            self.decision.proposal_id != self.proposal.id
            or self.decision.idempotency_key != self.proposal.idempotency_key
            or self.decision.candidate_hash != self.proposal.candidate_hash
            or self.decision.state is not LearningProposalState.COMMITTED
            or self.decision.result_skill_id != self.skill.id
            or self.decision.result_skill_version
            != self.expected_skill_version_count + 1
        ):
            raise ValueError("skill-change decision does not match its guarded commit")


@dataclass(frozen=True, slots=True)
class SkillChangeAcceptanceResult:
    """Terminal proposal and complete skill audit returned by the atomic store."""

    proposal: LearningProposal
    inspection: SkillInspection
    replayed: bool = False

    def __post_init__(self) -> None:
        _require_skill_proposal(self.proposal)
        if self.proposal.state is not LearningProposalState.COMMITTED:
            raise ValueError("accepted skill change requires a committed proposal")
        if not isinstance(self.inspection, SkillInspection):
            raise TypeError("accepted skill change requires SkillInspection")
        if not isinstance(self.replayed, bool):
            raise TypeError("skill-change replayed must be a boolean")
        if self.proposal.result_skill_id != self.inspection.skill.id:
            raise ValueError("accepted proposal references another skill")
        ordinal = self.proposal.result_skill_version
        assert ordinal is not None
        if ordinal > len(self.inspection.versions):
            raise ValueError("accepted proposal references an absent skill version")


class SkillChangeStore(LearningStore, SkillStore, Protocol):
    """Atomic acceptance seam implemented by the shared durable store.

    ``commit_skill_change`` must run in one transaction.  It compare-and-swaps
    the proposal from PROPOSED, the skill's active version pointer, and the
    immutable version count against ``SkillChangeCommit``.  It then inserts or
    verifies the learned ``Skill`` and ``SkillVersion``, updates the compact
    index, appends the ``SkillActivation``, persists the ``LearningDecision``,
    and resolves the proposal.  Any conflict leaves all six durable projections
    unchanged.  An exact already-committed retry returns ``replayed=True``.
    """

    async def commit_skill_change(
        self,
        request: SkillChangeCommit,
    ) -> SkillChangeAcceptanceResult: ...


class SkillChangeLearningService:
    """Stage safe procedure changes inertly and explicitly accept/activate them."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: SkillChangeStore,
        skills: SkillService,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        _required_text(agent_id, "skill-change agent_id", maximum=256)
        for method_name in (
            "create_proposal",
            "load_proposal",
            "list_proposals",
            "resolve_proposal",
            "record_discovery",
            "list_skill_index",
            "load_skill_index",
            "load_skill_version",
            "inspect_skill",
            "activate_skill",
            "commit_skill_change",
        ):
            if not callable(getattr(store, method_name, None)):
                raise TypeError(f"skill-change store must provide {method_name}")
        if not isinstance(skills, SkillService):
            raise TypeError("skills must be a SkillService")
        if not callable(clock):
            raise TypeError("skill-change clock must be callable")
        if not callable(id_factory):
            raise TypeError("skill-change id_factory must be callable")
        self._agent_id = agent_id
        self._store = store
        self._skills = skills
        self._clock = clock
        self._id_factory = id_factory

    async def propose(
        self,
        candidate: SkillChangeCandidate,
        provenance: LearningProvenance,
    ) -> SkillChangeProposalResult:
        """Persist a safe proposal without mutating skill or activation state."""

        if not isinstance(candidate, SkillChangeCandidate):
            raise TypeError("candidate must be a SkillChangeCandidate")
        if not isinstance(provenance, LearningProvenance):
            raise TypeError("provenance must be LearningProvenance")
        if provenance.agent_id != self._agent_id:
            raise ValueError("skill-change provenance belongs to another agent")
        created_at = self._now()
        _version_from_candidate(
            self._agent_id,
            candidate,
            content_hash=learning_candidate_hash(candidate.payload),
            created_at=created_at,
        )
        proposal = LearningProposal.create(
            proposal_id=self._id_factory("learning-proposal"),
            kind=LearningProposalKind.SKILL,
            category=LearningCandidateCategory.SKILL_CHANGE,
            provenance=provenance,
            candidate=candidate.payload,
            created_at=created_at,
        )
        stored = await self._store.create_proposal(proposal)
        _validate_stored_proposal(proposal, stored)
        if stored.state is LearningProposalState.REJECTED:
            return SkillChangeProposalResult(stored, None)
        assert stored.candidate_payload is not None
        stored_candidate = _candidate_from_payload(stored.candidate_payload)
        stored_preview = _version_from_candidate(
            self._agent_id,
            stored_candidate,
            content_hash=stored.candidate_hash,
            created_at=stored.created_at,
        )
        return SkillChangeProposalResult(stored, stored_preview)

    async def accept(
        self,
        proposal_id: str,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillChangeAcceptanceResult:
        """Explicitly commit and activate one exact proposed procedure version."""

        _required_text(proposal_id, "skill-change proposal_id", maximum=256)
        proposal = await self._store.load_proposal(self._agent_id, proposal_id)
        if proposal is None:
            raise SkillChangeConflictError(
                f"unknown skill-change proposal: {proposal_id}"
            )
        _require_skill_proposal(proposal)
        if proposal.provenance.agent_id != self._agent_id:
            raise ValueError("skill-change store returned another agent's proposal")
        if proposal.state is LearningProposalState.REJECTED:
            raise SkillChangeConflictError("skill-change proposal was rejected")
        if proposal.candidate_payload is None:
            raise SkillChangeConflictError("skill-change proposal lost its payload")
        candidate = _candidate_from_payload(proposal.candidate_payload)
        proposed_version = _version_from_candidate(
            self._agent_id,
            candidate,
            content_hash=proposal.candidate_hash,
            created_at=proposal.created_at,
        )
        if proposal.state is LearningProposalState.COMMITTED:
            inspection = await self._skills.inspect(proposed_version.skill_id)
            _require_version_in_history(proposed_version, inspection)
            return SkillChangeAcceptanceResult(
                proposal=proposal,
                inspection=inspection,
                replayed=True,
            )

        existing = await self._load_existing(proposed_version.skill_id)
        if existing is None:
            skill = Skill(
                id=proposed_version.skill_id,
                agent_id=self._agent_id,
                stable_name=proposed_version.stable_name,
                source=SkillSource.LEARNED_PROPOSAL,
                created_at=proposal.created_at,
            )
            existing_versions: tuple[SkillVersion, ...] = ()
        else:
            skill = existing.skill
            existing_versions = existing.versions
            if (
                skill.source is not SkillSource.LEARNED_PROPOSAL
                or skill.stable_name != proposed_version.stable_name
            ):
                raise SkillChangeConflictError(
                    "skill-change proposal cannot replace another skill source"
                )
        if any(
            item.id == proposed_version.id or item.version == proposed_version.version
            for item in existing_versions
        ):
            raise SkillChangeConflictError(
                "skill-change semantic version is already durable"
            )
        staged_index = SkillIndex.from_version(
            proposed_version,
            active_version_id=expected_active_version_id,
        )
        try:
            activation = await self._skills.prepare_change_activation(
                skill,
                proposed_version,
                expected_active_version_id=expected_active_version_id,
                actor_id=actor_id,
                reason=reason,
            )
        except SkillActivationConflictError as error:
            raise SkillChangeConflictError(str(error)) from error
        decided_at = activation.activated_at
        if decided_at < proposal.created_at:
            raise ValueError("skill-change acceptance cannot precede its proposal")
        expected_count = len(existing_versions)
        decision = LearningDecision(
            proposal_id=proposal.id,
            idempotency_key=proposal.idempotency_key,
            candidate_hash=proposal.candidate_hash,
            state=LearningProposalState.COMMITTED,
            decided_at=decided_at,
            result_skill_id=skill.id,
            result_skill_version=expected_count + 1,
        )
        request = SkillChangeCommit(
            proposal=proposal,
            skill=skill,
            version=proposed_version,
            staged_index=staged_index,
            activation=activation,
            decision=decision,
            expected_active_version_id=expected_active_version_id,
            expected_skill_version_count=expected_count,
        )
        result = await self._store.commit_skill_change(request)
        _validate_acceptance_result(request, result)
        return result

    async def _load_existing(self, skill_id: str) -> SkillInspection | None:
        try:
            return await self._skills.inspect(skill_id)
        except SkillNotFoundError:
            return None

    def _now(self) -> datetime:
        value = self._clock()
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError("skill-change clock must return a timezone-aware datetime")
        return value


def _version_from_candidate(
    agent_id: str,
    candidate: SkillChangeCandidate,
    *,
    content_hash: str,
    created_at: datetime,
) -> SkillVersion:
    skill_id = f"skill:{candidate.stable_name}"
    try:
        return SkillVersion(
            id=f"skill-version:{content_hash.removeprefix('sha256:')}",
            agent_id=agent_id,
            skill_id=skill_id,
            stable_name=candidate.stable_name,
            version=candidate.version,
            description=candidate.description,
            domains=tuple(sorted(candidate.domains)),
            resource_kinds=tuple(sorted(candidate.resource_kinds)),
            required_capability_ids=tuple(sorted(candidate.required_capability_ids)),
            activation_mode=candidate.activation_mode,
            sensitivity_notes=candidate.sensitivity_notes,
            policy_notes=candidate.policy_notes,
            source=SkillSource.LEARNED_PROPOSAL,
            content_hash=content_hash,
            instructions=candidate.instructions,
            source_path=None,
            created_at=created_at,
        )
    except (TypeError, ValueError) as error:
        raise SkillChangeFormatError(
            "skill-change candidate violates the portable skill contract"
        ) from error


def _candidate_from_payload(payload: Mapping[str, object]) -> SkillChangeCandidate:
    if frozenset(payload) != _CANDIDATE_KEYS:
        raise SkillChangeFormatError(
            "skill-change payload must contain exactly the documented fields"
        )
    try:
        domains = _string_tuple(payload["domains"], "domains")
        resource_kinds = _string_tuple(payload["resource_kinds"], "resource_kinds")
        capabilities = _string_tuple(
            payload["required_capability_ids"],
            "required_capability_ids",
        )
        return SkillChangeCandidate(
            stable_name=_payload_text(payload["stable_name"], "stable_name"),
            version=_payload_text(payload["version"], "version"),
            description=_payload_text(payload["description"], "description"),
            instructions=_payload_text(payload["instructions"], "instructions"),
            activation_mode=SkillActivationMode(
                _payload_text(payload["activation_mode"], "activation_mode")
            ),
            domains=domains,
            resource_kinds=resource_kinds,
            required_capability_ids=capabilities,
            sensitivity_notes=_optional_payload_text(
                payload["sensitivity_notes"],
                "sensitivity_notes",
            ),
            policy_notes=_optional_payload_text(
                payload["policy_notes"],
                "policy_notes",
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SkillChangeFormatError("skill-change payload is invalid") from error


def _validate_stored_proposal(
    requested: LearningProposal,
    stored: object,
) -> None:
    if not isinstance(stored, LearningProposal):
        raise TypeError("skill-change store must return LearningProposal")
    _require_skill_proposal(stored)
    if (
        stored.provenance != requested.provenance
        or stored.candidate_hash != requested.candidate_hash
        or stored.idempotency_key != requested.idempotency_key
    ):
        raise ValueError("skill-change store returned another proposal")


def _validate_acceptance_result(
    request: SkillChangeCommit,
    result: object,
) -> None:
    if not isinstance(result, SkillChangeAcceptanceResult):
        raise TypeError("skill-change store returned an invalid acceptance result")
    if (
        result.proposal.id != request.proposal.id
        or result.proposal.decision_hash != request.decision.fingerprint
        or result.proposal.result_skill_id != request.skill.id
        or result.proposal.result_skill_version
        != request.expected_skill_version_count + 1
    ):
        raise ValueError("skill-change store committed another proposal decision")
    _require_version_in_history(request.version, result.inspection)
    if not any(
        item.id == request.activation.id for item in result.inspection.activations
    ):
        raise ValueError("skill-change store omitted the activation audit")
    if not result.replayed and (
        result.inspection.index.active_version_id != request.version.id
        or result.inspection.activations[-1] != request.activation
    ):
        raise ValueError("skill-change store did not activate the accepted version")


def _require_version_in_history(
    expected: SkillVersion,
    inspection: SkillInspection,
) -> None:
    stored = next(
        (item for item in inspection.versions if item.id == expected.id), None
    )
    if stored != expected:
        raise ValueError("accepted skill version is absent from immutable history")


def _require_skill_proposal(proposal: LearningProposal) -> None:
    if not isinstance(proposal, LearningProposal):
        raise TypeError("skill change requires LearningProposal")
    if (
        proposal.kind is not LearningProposalKind.SKILL
        or proposal.category is not LearningCandidateCategory.SKILL_CHANGE
    ):
        raise ValueError("learning proposal is not a skill change")


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"skill-change {field_name} must be a string tuple")
    return value


def _payload_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"skill-change {field_name} must be a string")
    return value


def _optional_payload_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _payload_text(value, field_name)


def _required_text(value: object, field_name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise ValueError(f"{field_name} must be a bounded normalized string")


__all__ = [
    "SkillChangeAcceptanceResult",
    "SkillChangeCandidate",
    "SkillChangeCommit",
    "SkillChangeConflictError",
    "SkillChangeFormatError",
    "SkillChangeLearningError",
    "SkillChangeLearningService",
    "SkillChangeProposalResult",
    "SkillChangeStore",
]
