"""Strict transport-neutral outcome, target, and logical-delivery records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import TypeAlias

from .._json import canonical_json
from ..artifacts.models import (
    ArtifactAuthorship,
    ArtifactRef,
    artifact_provenance_to_mapping,
)
from ..llm.models import ModelSensitivity

MAX_OUTCOME_ARTIFACT_REQUIREMENTS = 4
MAX_DISTRIBUTION_TARGETS = 4
MAX_OUTCOME_ARTIFACT_REFERENCES = 8
MAX_OUTCOME_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_OUTCOME_TOTAL_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES = 48 * 1024
MAX_DELIVERIES_PER_AGENT = 256
MAX_DELIVERY_LIST_PAGE_SIZE = 50
MAX_DISTRIBUTION_IDENTITY_CHARACTERS = 1_024
MAX_DISTRIBUTION_LABEL_CHARACTERS = 128
CONVERSATION_INBOX_DESTINATION_REVISION = 1

_ARTIFACT_ID = re.compile(r"artifact-[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FAILURE_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_MEDIA_TYPE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z")


def _text(
    value: str,
    name: str,
    *,
    maximum: int = MAX_DISTRIBUTION_IDENTITY_CHARACTERS,
) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character in "\r\n\x00" for character in value)
    ):
        raise ValueError(f"{name} must be bounded non-empty single-line text")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 digest")


def _utc(value: datetime, name: str) -> None:
    offset = value.utcoffset() if isinstance(value, datetime) else None
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or offset is None
        or offset.total_seconds() != 0
    ):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _optional_utc(value: datetime | None, name: str) -> None:
    if value is not None:
        _utc(value, name)


def _sha256_mapping(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def conclusion_preview_projection(value: str) -> tuple[str, str, bool]:
    """Return exact text identity and one UTF-8-safe bounded preview."""

    if not isinstance(value, str):
        raise TypeError("conclusion preview source must be text")
    encoded = value.encode("utf-8")
    truncated = len(encoded) > MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES
    preview = (
        encoded[:MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES].decode(
            "utf-8",
            errors="ignore",
        )
        if truncated
        else value
    )
    return "sha256:" + sha256(encoded).hexdigest(), preview, truncated


def _media_types(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(values) > 16:
        raise ValueError(f"{name} exceeds its bound")
    for value in values:
        if not isinstance(value, str) or _MEDIA_TYPE.fullmatch(value) is None:
            raise ValueError(f"{name} contains an invalid canonical media type")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(values))


def _identities(
    values: tuple[str, ...],
    name: str,
    *,
    maximum_items: int = 64,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(values) > maximum_items:
        raise ValueError(f"{name} exceeds its bound")
    for value in values:
        _text(value, name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(values))


def _sensitivity_mapping(value: ModelSensitivity) -> str:
    if not isinstance(value, ModelSensitivity):
        raise TypeError("sensitivity must be ModelSensitivity")
    return value.value


@dataclass(frozen=True, slots=True)
class ArtifactRequirement:
    """Declarative validation for one bounded group of outcome artifacts."""

    required: bool
    minimum_count: int
    maximum_count: int
    allowed_media_types: tuple[str, ...]
    allowed_authorships: tuple[ArtifactAuthorship, ...]
    allowed_producer_capability_ids: tuple[str, ...]
    maximum_artifact_bytes: int
    maximum_total_bytes: int
    maximum_sensitivity: ModelSensitivity

    def __post_init__(self) -> None:
        if not isinstance(self.required, bool):
            raise TypeError("artifact requirement required must be a boolean")
        for value, name in (
            (self.minimum_count, "artifact requirement minimum_count"),
            (self.maximum_count, "artifact requirement maximum_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.minimum_count <= self.maximum_count:
            raise ValueError("artifact requirement count range is invalid")
        if not 1 <= self.maximum_count <= MAX_OUTCOME_ARTIFACT_REFERENCES:
            raise ValueError("artifact requirement maximum_count exceeds its bound")
        if self.required != (self.minimum_count > 0):
            raise ValueError("artifact requirement required and count disagree")
        media_types = _media_types(
            self.allowed_media_types,
            "artifact requirement allowed_media_types",
        )
        authorships = tuple(self.allowed_authorships)
        if len(authorships) > len(ArtifactAuthorship) or any(
            not isinstance(value, ArtifactAuthorship) for value in authorships
        ):
            raise ValueError("artifact requirement authorships are invalid")
        if len(authorships) != len(set(authorships)):
            raise ValueError("artifact requirement authorships cannot duplicate")
        authorships = tuple(sorted(authorships, key=lambda item: item.value))
        capability_ids = _identities(
            self.allowed_producer_capability_ids,
            "artifact requirement producer capability_ids",
        )
        for value, name, maximum in (
            (
                self.maximum_artifact_bytes,
                "artifact requirement maximum_artifact_bytes",
                MAX_OUTCOME_ARTIFACT_BYTES,
            ),
            (
                self.maximum_total_bytes,
                "artifact requirement maximum_total_bytes",
                MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{name} exceeds its inclusive bound")
        if self.maximum_total_bytes < self.maximum_artifact_bytes:
            raise ValueError("artifact requirement total bytes are too small")
        _sensitivity_mapping(self.maximum_sensitivity)
        object.__setattr__(self, "allowed_media_types", media_types)
        object.__setattr__(self, "allowed_authorships", authorships)
        object.__setattr__(
            self,
            "allowed_producer_capability_ids",
            capability_ids,
        )

    @property
    def digest(self) -> str:
        return _sha256_mapping(_artifact_requirement_mapping(self))


@dataclass(frozen=True, slots=True)
class OutcomeContract:
    """One exact ceiling for validating a completed unattended conclusion."""

    require_terminal_conclusion: bool
    artifact_requirements: tuple[ArtifactRequirement, ...]
    maximum_total_artifact_bytes: int
    maximum_effective_sensitivity: ModelSensitivity
    require_current_run_provenance: bool
    require_exact_source_bindings: bool

    def __post_init__(self) -> None:
        if self.require_terminal_conclusion is not True:
            raise ValueError("outcome contract requires a terminal conclusion")
        if not isinstance(self.artifact_requirements, tuple):
            raise TypeError("outcome artifact requirements must be a tuple")
        requirements = tuple(self.artifact_requirements)
        if len(requirements) > MAX_OUTCOME_ARTIFACT_REQUIREMENTS or any(
            not isinstance(value, ArtifactRequirement) for value in requirements
        ):
            raise ValueError("outcome artifact requirements exceed their bound")
        if len({value.digest for value in requirements}) != len(requirements):
            raise ValueError("outcome artifact requirements cannot duplicate")
        requirements = tuple(sorted(requirements, key=lambda value: value.digest))
        if (
            not isinstance(self.maximum_total_artifact_bytes, int)
            or isinstance(self.maximum_total_artifact_bytes, bool)
            or not 0
            <= self.maximum_total_artifact_bytes
            <= MAX_OUTCOME_TOTAL_ARTIFACT_BYTES
        ):
            raise ValueError("outcome maximum artifact bytes exceed their bound")
        if requirements and self.maximum_total_artifact_bytes == 0:
            raise ValueError("artifact requirements need a positive total byte bound")
        if any(
            value.maximum_total_bytes > self.maximum_total_artifact_bytes
            for value in requirements
        ):
            raise ValueError("artifact requirement exceeds the outcome byte ceiling")
        _sensitivity_mapping(self.maximum_effective_sensitivity)
        if self.require_current_run_provenance is not True:
            raise ValueError("outcome contract requires current-run provenance")
        if not isinstance(self.require_exact_source_bindings, bool):
            raise TypeError("exact-source-binding requirement must be a boolean")
        object.__setattr__(self, "artifact_requirements", requirements)

    @property
    def digest(self) -> str:
        return outcome_contract_digest(self)


def outcome_contract_digest(value: OutcomeContract) -> str:
    if not isinstance(value, OutcomeContract):
        raise TypeError("outcome contract digest requires OutcomeContract")
    return _sha256_mapping(_outcome_contract_mapping(value))


def conversation_inbox_destination_id(conversation_id: str) -> str:
    _text(conversation_id, "conversation inbox conversation_id")
    return f"conversation_inbox:{conversation_id}"


def target_fingerprint(
    *,
    conversation_id: str,
    destination_id: str,
    destination_revision: int,
    sensitivity_ceiling: ModelSensitivity,
) -> str:
    _text(conversation_id, "distribution target conversation_id")
    _text(destination_id, "distribution target destination_id")
    if (
        not isinstance(destination_revision, int)
        or isinstance(destination_revision, bool)
        or destination_revision < 1
    ):
        raise ValueError("distribution target revision must be positive")
    return _sha256_mapping(
        {
            "kind": "conversation_inbox",
            "conversation_id": conversation_id,
            "destination_id": destination_id,
            "destination_revision": destination_revision,
            "sensitivity_ceiling": _sensitivity_mapping(sensitivity_ceiling),
        }
    )


@dataclass(frozen=True, slots=True)
class ConversationInboxTarget:
    """The only supported distribution destination."""

    conversation_id: str
    destination_id: str
    destination_revision: int
    sensitivity_ceiling: ModelSensitivity
    target_fingerprint: str

    def __post_init__(self) -> None:
        _text(self.conversation_id, "conversation inbox target conversation_id")
        _text(self.destination_id, "conversation inbox target destination_id")
        if self.destination_id != conversation_inbox_destination_id(
            self.conversation_id
        ):
            raise ValueError("conversation inbox destination identity is invalid")
        if self.destination_revision != CONVERSATION_INBOX_DESTINATION_REVISION:
            raise ValueError("conversation inbox destination revision is unsupported")
        expected = target_fingerprint(
            conversation_id=self.conversation_id,
            destination_id=self.destination_id,
            destination_revision=self.destination_revision,
            sensitivity_ceiling=self.sensitivity_ceiling,
        )
        if self.target_fingerprint != expected:
            raise ValueError("conversation inbox target fingerprint does not match")


DistributionTargetBinding: TypeAlias = ConversationInboxTarget


@dataclass(frozen=True, slots=True)
class DistributionPlan:
    """One immutable ordered target ceiling containing one inbox."""

    targets: tuple[DistributionTargetBinding, ...]
    required_target_count: int
    plan_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.targets, tuple):
            raise TypeError("distribution targets must be a tuple")
        targets = tuple(self.targets)
        if not 1 <= len(targets) <= MAX_DISTRIBUTION_TARGETS or any(
            not isinstance(value, ConversationInboxTarget) for value in targets
        ):
            raise ValueError("distribution targets are invalid or outside their bound")
        if len(targets) != 1:
            raise ValueError(
                "distribution requires exactly one conversation inbox target"
            )
        targets = tuple(sorted(targets, key=lambda value: value.target_fingerprint))
        if len({value.target_fingerprint for value in targets}) != len(targets):
            raise ValueError("distribution targets cannot duplicate")
        if self.required_target_count != len(targets):
            raise ValueError("distribution required target count must be exact")
        expected = distribution_plan_digest(
            targets=targets,
            required_target_count=self.required_target_count,
        )
        if self.plan_digest != expected:
            raise ValueError("distribution plan digest does not match")
        object.__setattr__(self, "targets", targets)


def distribution_plan_digest(
    *,
    targets: tuple[DistributionTargetBinding, ...],
    required_target_count: int,
) -> str:
    return _sha256_mapping(
        {
            "targets": [_target_mapping(value) for value in targets],
            "required_target_count": required_target_count,
        }
    )


class OutcomeConclusionKind(str, Enum):
    TERMINAL_RUN = "terminal_run"
    NO_MODEL_OCCURRENCE = "no_model_occurrence"


class OutcomeState(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED_NO_CHANGE = "skipped_no_change"


@dataclass(frozen=True, slots=True)
class OutcomeArtifactReference:
    """Immutable artifact identity and safe validation facts for one outcome."""

    artifact_id: str
    producing_run_id: str
    producing_call_id: str
    producer_capability_id: str
    sha256: str
    media_type: str
    byte_size: int
    sensitivity: ModelSensitivity
    provenance_digest: str
    authorship: ArtifactAuthorship

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_id, str)
            or _ARTIFACT_ID.fullmatch(self.artifact_id) is None
        ):
            raise ValueError("outcome artifact_id is invalid")
        for value, name in (
            (self.producing_run_id, "outcome artifact producing_run_id"),
            (self.producing_call_id, "outcome artifact producing_call_id"),
            (self.producer_capability_id, "outcome artifact producer_capability_id"),
        ):
            _text(value, name)
        _digest(self.sha256, "outcome artifact sha256")
        if (
            not isinstance(self.media_type, str)
            or _MEDIA_TYPE.fullmatch(self.media_type) is None
        ):
            raise ValueError("outcome artifact media_type is invalid")
        if (
            not isinstance(self.byte_size, int)
            or isinstance(self.byte_size, bool)
            or not 0 <= self.byte_size <= MAX_OUTCOME_ARTIFACT_BYTES
        ):
            raise ValueError("outcome artifact byte_size exceeds its bound")
        _sensitivity_mapping(self.sensitivity)
        _digest(self.provenance_digest, "outcome artifact provenance_digest")
        if not isinstance(self.authorship, ArtifactAuthorship):
            raise TypeError("outcome artifact authorship is invalid")

    @property
    def digest(self) -> str:
        return _sha256_mapping(_outcome_artifact_reference_mapping(self))


def outcome_artifact_reference(value: ArtifactRef) -> OutcomeArtifactReference:
    """Project one committed artifact ref into its immutable outcome identity."""

    if not isinstance(value, ArtifactRef):
        raise TypeError("outcome artifact projection requires ArtifactRef")
    return OutcomeArtifactReference(
        artifact_id=value.artifact_id,
        producing_run_id=value.run_id,
        producing_call_id=value.call_id,
        producer_capability_id=value.capability_id,
        sha256=value.sha256,
        media_type=value.media_type,
        byte_size=value.byte_size,
        sensitivity=ModelSensitivity(value.sensitivity.value),
        provenance_digest=_sha256_mapping(
            artifact_provenance_to_mapping(value.provenance)
        ),
        authorship=value.provenance.authorship,
    )


def validate_outcome_artifact_references(
    references: tuple[OutcomeArtifactReference, ...],
    *,
    contract: OutcomeContract,
    resulting_run_id: str,
) -> tuple[OutcomeArtifactReference, ...]:
    """Validate one exact committed artifact set against its frozen contract."""

    if not isinstance(contract, OutcomeContract):
        raise TypeError("outcome artifact validation requires OutcomeContract")
    _text(resulting_run_id, "outcome artifact resulting_run_id")
    material = tuple(sorted(references, key=lambda item: item.artifact_id))
    if len(material) > MAX_OUTCOME_ARTIFACT_REFERENCES or any(
        not isinstance(item, OutcomeArtifactReference) for item in material
    ):
        raise ValueError("outcome artifact references exceed their bound")
    if any(item.producing_run_id != resulting_run_id for item in material):
        raise ValueError("outcome artifact provenance belongs to another run")
    if sum(item.byte_size for item in material) > contract.maximum_total_artifact_bytes:
        raise ValueError("outcome artifacts exceed the contract byte ceiling")
    if any(
        item.sensitivity.routing_rank
        > contract.maximum_effective_sensitivity.routing_rank
        for item in material
    ):
        raise ValueError("outcome artifact sensitivity exceeds its contract")
    requirements = contract.artifact_requirements
    if not requirements and material:
        raise ValueError("outcome contract permits no artifacts")

    def matches(
        item: OutcomeArtifactReference,
        requirement: ArtifactRequirement,
    ) -> bool:
        return (
            (
                not requirement.allowed_media_types
                or item.media_type in requirement.allowed_media_types
            )
            and (
                not requirement.allowed_authorships
                or item.authorship in requirement.allowed_authorships
            )
            and (
                not requirement.allowed_producer_capability_ids
                or item.producer_capability_id
                in requirement.allowed_producer_capability_ids
            )
            and item.byte_size <= requirement.maximum_artifact_bytes
            and item.sensitivity.routing_rank
            <= requirement.maximum_sensitivity.routing_rank
        )

    matched_ids: set[str] = set()
    for requirement in requirements:
        matched = tuple(item for item in material if matches(item, requirement))
        if not requirement.minimum_count <= len(matched) <= requirement.maximum_count:
            raise ValueError("outcome artifact requirement count is not satisfied")
        if sum(item.byte_size for item in matched) > requirement.maximum_total_bytes:
            raise ValueError("outcome artifact requirement byte ceiling is exceeded")
        matched_ids.update(item.artifact_id for item in matched)
    if matched_ids != {item.artifact_id for item in material}:
        raise ValueError("an outcome artifact matches no frozen requirement")
    if contract.require_exact_source_bindings and any(
        item.authorship is not ArtifactAuthorship.EXACT_SOURCE_DATA for item in material
    ):
        raise ValueError("outcome artifacts lack required exact source bindings")
    return material


@dataclass(frozen=True, slots=True)
class OutcomeReference:
    """One immutable bounded conclusion and its committed artifact references."""

    conclusion_kind: OutcomeConclusionKind
    conclusion_state: OutcomeState
    conclusion_id: str
    conclusion_digest: str
    conclusion_preview: str
    conclusion_preview_truncated: bool
    resulting_run_id: str | None
    artifact_references: tuple[OutcomeArtifactReference, ...]
    effective_sensitivity: ModelSensitivity
    provenance_digest: str
    failure_code: str | None
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.conclusion_kind, OutcomeConclusionKind):
            raise TypeError("outcome conclusion kind is invalid")
        if not isinstance(self.conclusion_state, OutcomeState):
            raise TypeError("outcome conclusion state is invalid")
        _text(self.conclusion_id, "outcome conclusion_id")
        _digest(self.conclusion_digest, "outcome conclusion_digest")
        if (
            not isinstance(self.conclusion_preview, str)
            or "\x00" in self.conclusion_preview
        ):
            raise ValueError("outcome conclusion preview is invalid")
        if (
            len(self.conclusion_preview.encode("utf-8"))
            > MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES
        ):
            raise ValueError("outcome conclusion preview exceeds its byte bound")
        if not isinstance(self.conclusion_preview_truncated, bool):
            raise TypeError("outcome preview truncated flag must be a boolean")
        if self.resulting_run_id is not None:
            _text(self.resulting_run_id, "outcome resulting_run_id")
        if self.conclusion_kind is OutcomeConclusionKind.TERMINAL_RUN:
            if (
                self.resulting_run_id is None
                or self.conclusion_id != self.resulting_run_id
            ):
                raise ValueError("terminal-run outcome requires its exact run identity")
        elif self.resulting_run_id is not None:
            raise ValueError("no-model outcome cannot claim a resulting run")
        if not isinstance(self.artifact_references, tuple):
            raise TypeError("outcome artifact references must be a tuple")
        references = tuple(self.artifact_references)
        if len(references) > MAX_OUTCOME_ARTIFACT_REFERENCES or any(
            not isinstance(value, OutcomeArtifactReference) for value in references
        ):
            raise ValueError("outcome artifact references exceed their bound")
        references = tuple(sorted(references, key=lambda value: value.artifact_id))
        if len({value.artifact_id for value in references}) != len(references):
            raise ValueError("outcome artifact references cannot duplicate")
        if (
            sum(value.byte_size for value in references)
            > MAX_OUTCOME_TOTAL_ARTIFACT_BYTES
        ):
            raise ValueError("outcome artifact references exceed the total byte bound")
        _sensitivity_mapping(self.effective_sensitivity)
        if any(
            value.sensitivity.routing_rank > self.effective_sensitivity.routing_rank
            for value in references
        ):
            raise ValueError("outcome sensitivity cannot downgrade an artifact")
        _digest(self.provenance_digest, "outcome provenance_digest")
        if self.conclusion_state is OutcomeState.FAILED:
            if (
                self.failure_code is None
                or _FAILURE_CODE.fullmatch(self.failure_code) is None
            ):
                raise ValueError("failed outcome requires a safe failure code")
        elif self.failure_code is not None:
            raise ValueError("non-failed outcome cannot contain a failure code")
        _utc(self.observed_at, "outcome observed_at")
        object.__setattr__(self, "artifact_references", references)


class DeliverySubjectKind(str, Enum):
    AUTONOMOUS_FOLLOWUP = "autonomous_followup"
    ROUTINE_OCCURRENCE = "routine_occurrence"


class DeliveryState(str, Enum):
    AVAILABLE = "available"
    ACKNOWLEDGED = "acknowledged"
    BLOCKED = "blocked"


def logical_delivery_key(
    *,
    agent_id: str,
    subject_kind: DeliverySubjectKind,
    subject_id: str,
    target_fingerprint: str,
) -> str:
    _text(agent_id, "delivery key agent_id")
    if not isinstance(subject_kind, DeliverySubjectKind):
        raise TypeError("delivery key subject kind is invalid")
    _text(subject_id, "delivery key subject_id")
    _digest(target_fingerprint, "delivery key target_fingerprint")
    return "delivery:" + _sha256_mapping(
        {
            "agent_id": agent_id,
            "subject_kind": subject_kind.value,
            "subject_id": subject_id,
            "target_fingerprint": target_fingerprint,
        }
    )


@dataclass(frozen=True, slots=True)
class Delivery:
    """One independently meaningful logical result for one exact target."""

    delivery_id: str
    agent_id: str
    conversation_id: str
    subject_kind: DeliverySubjectKind
    subject_id: str
    logical_key: str
    target: ConversationInboxTarget
    outcome: OutcomeReference
    visibility_state: DeliveryState
    acknowledged_at: datetime | None
    blocked_reason_code: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.delivery_id, "delivery_id"),
            (self.agent_id, "delivery agent_id"),
            (self.conversation_id, "delivery conversation_id"),
            (self.subject_id, "delivery subject_id"),
        ):
            _text(value, name)
        if not isinstance(self.subject_kind, DeliverySubjectKind):
            raise TypeError("delivery subject kind is invalid")
        _text(self.logical_key, "delivery logical_key")
        if not isinstance(self.target, ConversationInboxTarget):
            raise TypeError("delivery target is invalid")
        if self.target.conversation_id != self.conversation_id:
            raise ValueError("delivery target conversation identity differs")
        expected_key = logical_delivery_key(
            agent_id=self.agent_id,
            subject_kind=self.subject_kind,
            subject_id=self.subject_id,
            target_fingerprint=self.target.target_fingerprint,
        )
        if self.logical_key != expected_key:
            raise ValueError("delivery logical key does not match its identities")
        if not isinstance(self.outcome, OutcomeReference):
            raise TypeError("delivery outcome is invalid")
        if (
            self.subject_kind is DeliverySubjectKind.AUTONOMOUS_FOLLOWUP
            and self.outcome.conclusion_kind is not OutcomeConclusionKind.TERMINAL_RUN
        ):
            raise ValueError("autonomous follow-up delivery requires a terminal run")
        if not isinstance(self.visibility_state, DeliveryState):
            raise TypeError("delivery visibility state is invalid")
        _optional_utc(self.acknowledged_at, "delivery acknowledged_at")
        if (self.visibility_state is DeliveryState.ACKNOWLEDGED) != (
            self.acknowledged_at is not None
        ):
            raise ValueError("delivery acknowledgment state is inconsistent")
        if self.visibility_state is DeliveryState.BLOCKED:
            if (
                self.blocked_reason_code is None
                or _FAILURE_CODE.fullmatch(self.blocked_reason_code) is None
            ):
                raise ValueError("blocked delivery requires a safe reason code")
        elif self.blocked_reason_code is not None and (
            self.visibility_state is not DeliveryState.ACKNOWLEDGED
            or _FAILURE_CODE.fullmatch(self.blocked_reason_code) is None
        ):
            raise ValueError("delivery blocked reason is invalid for its state")
        if (
            self.visibility_state is DeliveryState.AVAILABLE
            and self.outcome.effective_sensitivity.routing_rank
            > self.target.sensitivity_ceiling.routing_rank
        ):
            raise ValueError("visible delivery exceeds its destination sensitivity")
        _utc(self.created_at, "delivery created_at")
        _utc(self.updated_at, "delivery updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("delivery updated_at precedes created_at")


@dataclass(frozen=True, slots=True)
class InboxView:
    """Bounded user-facing projection over one logical delivery."""

    delivery_id: str
    conversation_id: str
    subject_kind: DeliverySubjectKind
    subject_id: str
    conclusion_kind: OutcomeConclusionKind
    conclusion_state: OutcomeState
    conclusion_digest: str
    conclusion_preview: str
    conclusion_preview_truncated: bool
    resulting_run_id: str | None
    artifact_references: tuple[OutcomeArtifactReference, ...]
    effective_sensitivity: ModelSensitivity
    provenance_digest: str
    destination_id: str
    state: DeliveryState
    created_at: datetime
    updated_at: datetime
    acknowledged_at: datetime | None
    blocked_reason_code: str | None
    failure_code: str | None

    def __post_init__(self) -> None:
        for value, name in (
            (self.delivery_id, "inbox view delivery_id"),
            (self.conversation_id, "inbox view conversation_id"),
            (self.subject_id, "inbox view subject_id"),
            (self.destination_id, "inbox view destination_id"),
        ):
            _text(value, name)
        if not isinstance(self.subject_kind, DeliverySubjectKind):
            raise TypeError("inbox view subject kind is invalid")
        if not isinstance(self.conclusion_kind, OutcomeConclusionKind):
            raise TypeError("inbox view conclusion kind is invalid")
        if not isinstance(self.conclusion_state, OutcomeState):
            raise TypeError("inbox view conclusion state is invalid")
        _digest(self.conclusion_digest, "inbox view conclusion_digest")
        if (
            not isinstance(self.conclusion_preview, str)
            or len(self.conclusion_preview.encode("utf-8"))
            > MAX_OUTCOME_CONCLUSION_PREVIEW_BYTES
        ):
            raise ValueError("inbox view conclusion preview is invalid")
        if not isinstance(self.conclusion_preview_truncated, bool):
            raise TypeError("inbox view truncated flag must be a boolean")
        if self.resulting_run_id is not None:
            _text(self.resulting_run_id, "inbox view resulting_run_id")
        references = tuple(self.artifact_references)
        if len(references) > MAX_OUTCOME_ARTIFACT_REFERENCES or any(
            not isinstance(value, OutcomeArtifactReference) for value in references
        ):
            raise ValueError("inbox view artifact references exceed their bound")
        if not isinstance(self.effective_sensitivity, ModelSensitivity):
            raise TypeError("inbox view sensitivity is invalid")
        _digest(self.provenance_digest, "inbox view provenance_digest")
        if not isinstance(self.state, DeliveryState):
            raise TypeError("inbox view state is invalid")
        _utc(self.created_at, "inbox view created_at")
        _utc(self.updated_at, "inbox view updated_at")
        _optional_utc(self.acknowledged_at, "inbox view acknowledged_at")
        if (
            self.blocked_reason_code is not None
            and _FAILURE_CODE.fullmatch(self.blocked_reason_code) is None
        ):
            raise ValueError("inbox view blocked reason code is invalid")
        if (
            self.failure_code is not None
            and _FAILURE_CODE.fullmatch(self.failure_code) is None
        ):
            raise ValueError("inbox view failure code is invalid")
        object.__setattr__(self, "artifact_references", references)


@dataclass(frozen=True, slots=True)
class DeliveryInspection:
    """Safe exact inspection of one agent-owned logical delivery."""

    delivery: Delivery

    def __post_init__(self) -> None:
        if not isinstance(self.delivery, Delivery):
            raise TypeError("delivery inspection requires a Delivery")


class DistributionDestinationState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DistributionDestination:
    """Bounded safe destination discovery metadata."""

    destination_id: str
    kind: str
    label: str
    state: DistributionDestinationState
    revision: int
    sensitivity_ceiling: ModelSensitivity
    supported_outcome_media: tuple[str, ...]
    selectable: bool

    def __post_init__(self) -> None:
        _text(self.destination_id, "distribution destination_id")
        _text(self.kind, "distribution destination kind", maximum=64)
        if self.kind != "conversation_inbox":
            raise ValueError("only conversation inbox destinations are supported")
        _text(
            self.label,
            "distribution destination label",
            maximum=MAX_DISTRIBUTION_LABEL_CHARACTERS,
        )
        if not isinstance(self.state, DistributionDestinationState):
            raise TypeError("distribution destination state is invalid")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("distribution destination revision must be positive")
        _sensitivity_mapping(self.sensitivity_ceiling)
        media = _media_types(
            self.supported_outcome_media,
            "distribution destination supported media",
        )
        if not isinstance(self.selectable, bool):
            raise TypeError("distribution destination selectable must be a boolean")
        if self.selectable != (self.state is DistributionDestinationState.AVAILABLE):
            raise ValueError("distribution destination selectability is inconsistent")
        object.__setattr__(self, "supported_outcome_media", media)


def _artifact_requirement_mapping(value: ArtifactRequirement) -> dict[str, object]:
    return {
        "required": value.required,
        "minimum_count": value.minimum_count,
        "maximum_count": value.maximum_count,
        "allowed_media_types": list(value.allowed_media_types),
        "allowed_authorships": [item.value for item in value.allowed_authorships],
        "allowed_producer_capability_ids": list(value.allowed_producer_capability_ids),
        "maximum_artifact_bytes": value.maximum_artifact_bytes,
        "maximum_total_bytes": value.maximum_total_bytes,
        "maximum_sensitivity": value.maximum_sensitivity.value,
    }


def _outcome_contract_mapping(value: OutcomeContract) -> dict[str, object]:
    return {
        "require_terminal_conclusion": value.require_terminal_conclusion,
        "artifact_requirements": [
            _artifact_requirement_mapping(item) for item in value.artifact_requirements
        ],
        "maximum_total_artifact_bytes": value.maximum_total_artifact_bytes,
        "maximum_effective_sensitivity": value.maximum_effective_sensitivity.value,
        "require_current_run_provenance": value.require_current_run_provenance,
        "require_exact_source_bindings": value.require_exact_source_bindings,
    }


def outcome_contract_projection(value: OutcomeContract) -> dict[str, object]:
    """Return the bounded safe current projection of an outcome contract."""

    if not isinstance(value, OutcomeContract):
        raise TypeError("outcome contract projection requires OutcomeContract")
    return _outcome_contract_mapping(value)


def _target_mapping(value: DistributionTargetBinding) -> dict[str, object]:
    if not isinstance(value, ConversationInboxTarget):
        raise TypeError("distribution target must be ConversationInboxTarget")
    return {
        "kind": "conversation_inbox",
        "conversation_id": value.conversation_id,
        "destination_id": value.destination_id,
        "destination_revision": value.destination_revision,
        "sensitivity_ceiling": value.sensitivity_ceiling.value,
        "target_fingerprint": value.target_fingerprint,
    }


def distribution_plan_projection(value: DistributionPlan) -> dict[str, object]:
    """Return the bounded safe current projection of a distribution plan."""

    if not isinstance(value, DistributionPlan):
        raise TypeError("distribution plan projection requires DistributionPlan")
    return {
        "targets": [_target_mapping(target) for target in value.targets],
        "required_target_count": value.required_target_count,
        "plan_digest": value.plan_digest,
    }


def distribution_destination_projection(
    value: DistributionDestination,
) -> dict[str, object]:
    """Return bounded safe discovery metadata for one destination."""

    if not isinstance(value, DistributionDestination):
        raise TypeError("destination projection requires DistributionDestination")
    return {
        "destination_id": value.destination_id,
        "kind": value.kind,
        "label": value.label,
        "state": value.state.value,
        "revision": value.revision,
        "sensitivity_ceiling": value.sensitivity_ceiling.value,
        "supported_outcome_media": list(value.supported_outcome_media),
        "selectable": value.selectable,
    }


def _outcome_artifact_reference_mapping(
    value: OutcomeArtifactReference,
) -> dict[str, object]:
    return {
        "artifact_id": value.artifact_id,
        "producing_run_id": value.producing_run_id,
        "producing_call_id": value.producing_call_id,
        "producer_capability_id": value.producer_capability_id,
        "sha256": value.sha256,
        "media_type": value.media_type,
        "byte_size": value.byte_size,
        "sensitivity": value.sensitivity.value,
        "provenance_digest": value.provenance_digest,
        "authorship": value.authorship.value,
    }


def outcome_artifact_reference_projection(
    value: OutcomeArtifactReference,
) -> dict[str, object]:
    """Return immutable safe identity and provenance facts for one artifact."""

    if not isinstance(value, OutcomeArtifactReference):
        raise TypeError("artifact projection requires OutcomeArtifactReference")
    return _outcome_artifact_reference_mapping(value)


def inbox_view_projection(value: InboxView) -> dict[str, object]:
    """Return the bounded public/model projection of one logical delivery."""

    if not isinstance(value, InboxView):
        raise TypeError("inbox projection requires InboxView")
    return {
        "delivery_id": value.delivery_id,
        "conversation_id": value.conversation_id,
        "subject_kind": value.subject_kind.value,
        "subject_id": value.subject_id,
        "conclusion_kind": value.conclusion_kind.value,
        "conclusion_state": value.conclusion_state.value,
        "conclusion_digest": value.conclusion_digest,
        "conclusion_preview": value.conclusion_preview,
        "conclusion_preview_truncated": value.conclusion_preview_truncated,
        "resulting_run_id": value.resulting_run_id,
        "artifact_references": [
            _outcome_artifact_reference_mapping(item)
            for item in value.artifact_references
        ],
        "effective_sensitivity": value.effective_sensitivity.value,
        "provenance_digest": value.provenance_digest,
        "destination_id": value.destination_id,
        "state": value.state.value,
        "created_at": value.created_at.isoformat(),
        "updated_at": value.updated_at.isoformat(),
        "acknowledged_at": (
            None if value.acknowledged_at is None else value.acknowledged_at.isoformat()
        ),
        "blocked_reason_code": value.blocked_reason_code,
        "failure_code": value.failure_code,
    }


def delivery_inspection_projection(value: DeliveryInspection) -> dict[str, object]:
    """Return exact safe target and outcome facts without bytes or diagnostics."""

    if not isinstance(value, DeliveryInspection):
        raise TypeError("delivery inspection projection requires DeliveryInspection")
    delivery = value.delivery
    outcome = delivery.outcome
    return {
        "delivery_id": delivery.delivery_id,
        "conversation_id": delivery.conversation_id,
        "subject_kind": delivery.subject_kind.value,
        "subject_id": delivery.subject_id,
        "logical_key": delivery.logical_key,
        "target": _target_mapping(delivery.target),
        "outcome": {
            "conclusion_kind": outcome.conclusion_kind.value,
            "conclusion_state": outcome.conclusion_state.value,
            "conclusion_id": outcome.conclusion_id,
            "conclusion_digest": outcome.conclusion_digest,
            "conclusion_preview": outcome.conclusion_preview,
            "conclusion_preview_truncated": outcome.conclusion_preview_truncated,
            "resulting_run_id": outcome.resulting_run_id,
            "artifact_references": [
                _outcome_artifact_reference_mapping(item)
                for item in outcome.artifact_references
            ],
            "effective_sensitivity": outcome.effective_sensitivity.value,
            "provenance_digest": outcome.provenance_digest,
            "failure_code": outcome.failure_code,
            "observed_at": outcome.observed_at.isoformat(),
        },
        "visibility_state": delivery.visibility_state.value,
        "acknowledged_at": (
            None
            if delivery.acknowledged_at is None
            else delivery.acknowledged_at.isoformat()
        ),
        "blocked_reason_code": delivery.blocked_reason_code,
        "created_at": delivery.created_at.isoformat(),
        "updated_at": delivery.updated_at.isoformat(),
    }
