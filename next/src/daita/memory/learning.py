"""Explicit user learning into versioned memory and inert fact proposals.

The precise canonical resource-alias form remains supported::

    Remember resource alias correction: <canonical JSON record>

Ordinary ``Remember that ...`` and correction messages use the same proposal,
safety, provenance, revision, and atomic-memory owners. Evidence-backed facts
remain visible proposals rather than silently becoming durable memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Protocol, cast, runtime_checkable

from .._json import FrozenJsonObject, canonical_json
from ..capabilities import AccessMode
from ..catalog.models import CatalogResource, Sensitivity
from ..learning import (
    LearningCandidateCategory,
    LearningDecision,
    LearningProposal,
    LearningProposalKind,
    LearningProposalState,
    LearningProvenance,
    LearningRejectionCategory,
    LearningSourceOutcome,
    resolve_learning_proposal,
)
from ..loop.models import LoopPhase
from ..operations.checkpoints import OperationSnapshot
from ..operations.models import Evidence, OperationStatus, TaskStatus, TriggerKind
from .models import (
    MemoryCreator,
    MemoryHistory,
    MemoryKind,
    MemoryProvenance,
    MemoryProvenanceKind,
    MemoryRecord,
    MemoryScope,
    MemorySensitivity,
    MemorySnapshot,
    MemoryState,
    MemoryVersion,
    normalize_memory_logical_key,
)

RESOURCE_ALIAS_CORRECTION_PREFIX = "Remember resource alias correction: "
NATURAL_REMEMBER_PREFIXES = (
    "remember that ",
    "please remember that ",
    "correction: ",
    "actually, ",
)

_CORRECTION_KEYS = frozenset(
    {
        "business_term",
        "field",
        "resource_id",
        "resource_revision",
        "source_id",
        "stored_value",
    }
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_TRIGGER_CHARACTERS = 4_096
_NATURAL_ALIAS = re.compile(
    r"(?is)^(?P<business>.+?)\s+(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"is\s+stored\s+as\s+(?P<stored>.+)$"
)
_NATURAL_ALIAS_MEANS = re.compile(
    r"(?is)^(?P<business>.+?)\s+means\s+"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=|is)\s*(?P<stored>.+)$"
)
_EVIDENCE_FACT = re.compile(
    r"(?is)^(?:please\s+)?(?:propose|learn)\s+(?:a\s+)?fact\s+from\s+"
    r"(?:the\s+|this\s+)?evidence(?:\s+that)?\s*[: ]\s*(?P<content>.+)$"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExplicitCorrectionLearningError(RuntimeError):
    """Base class for explicit-correction learning failures."""


class ExplicitCorrectionNotEligibleError(ExplicitCorrectionLearningError):
    """Raised when an operation or catalog binding cannot be learned."""


class ExplicitCorrectionFormatError(ExplicitCorrectionLearningError):
    """Raised when the user trigger is not the one supported correction form."""


class ExplicitCorrectionStoreContractError(ExplicitCorrectionLearningError):
    """Raised when the atomic store returns an incoherent result."""


class ExplicitCorrectionStoreConflictError(ExplicitCorrectionLearningError):
    """Raised when the alias-version compare-and-swap loses its race."""


@dataclass(frozen=True, slots=True)
class ResourceAliasCorrection:
    """The one typed correction record accepted from an original user trigger."""

    source_id: str
    resource_id: str
    resource_revision: str
    field: str
    business_term: str
    stored_value: str

    def __post_init__(self) -> None:
        _required_text(self.source_id, "correction source_id", maximum=512)
        _required_text(self.resource_id, "correction resource_id", maximum=512)
        _required_text(
            self.resource_revision,
            "correction resource_revision",
            maximum=512,
        )
        if _SHA256.fullmatch(self.resource_revision) is None:
            raise ValueError("correction resource_revision must be a sha256 hash")
        _required_text(self.field, "correction field", maximum=256)
        _required_text(
            self.business_term,
            "correction business_term",
            maximum=256,
        )
        _required_text(
            self.stored_value,
            "correction stored_value",
            maximum=1_024,
        )

    @property
    def candidate(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "business_term": self.business_term,
                "field": self.field,
                "resource_id": self.resource_id,
                "resource_revision": self.resource_revision,
                "source_id": self.source_id,
                "stored_value": self.stored_value,
            }
        )

    def to_trigger_message(self) -> str:
        """Render the exact bounded pattern consumed by the learning service."""

        message = RESOURCE_ALIAS_CORRECTION_PREFIX + canonical_json(self.candidate)
        if len(message) > _MAX_TRIGGER_CHARACTERS:
            raise ValueError("resource-alias correction trigger is too large")
        return message


@dataclass(frozen=True, slots=True)
class ExplicitCorrectionCommit:
    """One atomic proposal resolution and optional memory-version write.

    ``expected_memory_version=None`` means that the logical alias must not yet
    exist. Otherwise the store must compare-and-swap the exact active head.
    """

    proposal: LearningProposal
    expected_memory_version: int | None = None
    intended_memory: MemorySnapshot | None = None
    decision: LearningDecision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, LearningProposal):
            raise TypeError("explicit correction proposal must be LearningProposal")
        if self.proposal.kind is not LearningProposalKind.MEMORY or (
            self.proposal.category is not LearningCandidateCategory.EXPLICIT_CORRECTION
        ):
            raise ValueError("explicit correction requires a memory proposal")
        if self.proposal.state is LearningProposalState.REJECTED:
            if any(
                value is not None
                for value in (
                    self.expected_memory_version,
                    self.intended_memory,
                    self.decision,
                )
            ):
                raise ValueError("rejected correction cannot write memory")
            return
        if self.proposal.state is not LearningProposalState.PROPOSED:
            raise ValueError("correction commit requires proposed or rejected state")
        if not isinstance(self.intended_memory, MemorySnapshot):
            raise TypeError("safe correction requires an intended MemorySnapshot")
        if not isinstance(self.decision, LearningDecision):
            raise TypeError("safe correction requires a LearningDecision")
        if self.decision.state is not LearningProposalState.COMMITTED:
            raise ValueError("safe correction decision must commit memory")
        resolve_learning_proposal(self.proposal, self.decision)

        snapshot = self.intended_memory
        if snapshot.record.kind not in {
            MemoryKind.RESOURCE_ALIAS,
            MemoryKind.SEMANTIC_FACT,
        }:
            raise ValueError("explicit learning requires an allowed memory kind")
        if snapshot.record.state is not MemoryState.ACTIVE:
            raise ValueError("correction memory must remain active")
        if snapshot.version.creator is not MemoryCreator.LEARNING_SERVICE:
            raise ValueError("correction version must be learning-service owned")
        if snapshot.version.provenance.kind is not MemoryProvenanceKind.USER_STATEMENT:
            raise ValueError("correction memory requires USER_STATEMENT provenance")
        if (
            self.decision.result_memory_id != snapshot.record.id
            or self.decision.result_memory_version != snapshot.version.version
        ):
            raise ValueError("correction decision does not match intended memory")
        if self.expected_memory_version is None:
            if snapshot.version.version != 1 or (
                snapshot.version.supersedes_version is not None
            ):
                raise ValueError("new correction memory must begin at version one")
        else:
            _positive_integer(
                self.expected_memory_version,
                "expected_memory_version",
            )
            if snapshot.version.version != self.expected_memory_version + 1 or (
                snapshot.version.supersedes_version != self.expected_memory_version
            ):
                raise ValueError("correction memory does not follow its CAS version")


@dataclass(frozen=True, slots=True)
class ExplicitCorrectionResult:
    """Terminal atomic outcome; ``replayed`` means no new state was written."""

    proposal: LearningProposal
    memory: MemoryHistory | None
    replayed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, LearningProposal):
            raise TypeError("explicit correction result requires LearningProposal")
        if self.proposal.state is LearningProposalState.PROPOSED:
            raise ValueError("explicit correction result must be terminal")
        if not isinstance(self.replayed, bool):
            raise TypeError("explicit correction replayed must be a boolean")
        if self.proposal.state is LearningProposalState.REJECTED:
            if self.memory is not None:
                raise ValueError("rejected correction cannot return memory")
            return
        if not isinstance(self.memory, MemoryHistory):
            raise TypeError("committed correction requires MemoryHistory")
        if self.memory.record.kind not in {
            MemoryKind.RESOURCE_ALIAS,
            MemoryKind.SEMANTIC_FACT,
        }:
            raise ValueError("committed learning returned an unsupported memory kind")
        if self.memory.record.state is not MemoryState.ACTIVE:
            raise ValueError("committed correction memory must remain active")
        if self.proposal.result_memory_id != self.memory.record.id:
            raise ValueError("terminal proposal does not match returned memory")
        result_version = self.proposal.result_memory_version
        assert result_version is not None
        if self.replayed:
            if result_version not in {
                version.version for version in self.memory.versions
            }:
                raise ValueError(
                    "replayed proposal result version is absent from memory history"
                )
        elif result_version != self.memory.record.current_version:
            raise ValueError("fresh proposal does not match the current memory head")


@runtime_checkable
class ExplicitCorrectionCatalogReader(Protocol):
    """Catalog-owned current resource lookup needed for correction binding."""

    async def load_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> CatalogResource | None: ...


@runtime_checkable
class ExplicitCorrectionStore(Protocol):
    """Small atomic seam for proposals and resource-alias memory.

    Implementations must enforce a unique proposal ``idempotency_key``. If a
    terminal result already exists for the same key, they return that exact
    result with ``replayed=True`` before evaluating the memory CAS and perform
    no write. Otherwise they atomically persist the proposal, append/create the
    intended immutable memory version when present, advance its guarded head,
    and persist the terminal decision. A missing or stale expected memory head
    raises :class:`ExplicitCorrectionStoreConflictError` with no partial write.
    """

    async def load_resource_alias(
        self,
        scope: MemoryScope,
        logical_key: str,
    ) -> MemoryHistory | None: ...

    async def load_learning_memory(
        self,
        scope: MemoryScope,
        kind: MemoryKind,
        logical_key: str,
    ) -> MemoryHistory | None: ...

    async def load_proposal(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> LearningProposal | None: ...

    async def commit_explicit_correction(
        self,
        request: ExplicitCorrectionCommit,
    ) -> ExplicitCorrectionResult: ...

    async def create_proposal(
        self,
        proposal: LearningProposal,
    ) -> LearningProposal: ...


class ExplicitCorrectionLearningService:
    """Learn one exact, safe resource alias after a succeeded operation."""

    def __init__(
        self,
        *,
        catalog: ExplicitCorrectionCatalogReader,
        store: ExplicitCorrectionStore,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(catalog, ExplicitCorrectionCatalogReader):
            raise TypeError("catalog must implement ExplicitCorrectionCatalogReader")
        if not isinstance(store, ExplicitCorrectionStore):
            raise TypeError("store must implement ExplicitCorrectionStore")
        if not callable(clock):
            raise TypeError("explicit correction clock must be callable")
        self._catalog = catalog
        self._store = store
        self._clock = clock

    async def learn(
        self,
        snapshot: OperationSnapshot,
    ) -> ExplicitCorrectionResult:
        """Commit one safe canonical or natural explicit user statement."""

        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("explicit correction requires an OperationSnapshot")
        _validate_completed_user_operation(snapshot)
        message = snapshot.trigger.payload.get("message")
        if not isinstance(message, str):
            raise ExplicitCorrectionFormatError(
                "user trigger must contain a string message"
            )
        candidate = _parse_trigger_candidate(message, snapshot=snapshot)

        now = self._now()
        source_hash = _hash_text(message)
        provenance = LearningProvenance(
            agent_id=snapshot.operation.agent_id,
            operation_id=snapshot.operation.id,
            trigger_id=snapshot.trigger.id,
            source_outcome=LearningSourceOutcome.SUCCEEDED,
            source_hash=source_hash,
        )
        proposal = LearningProposal.create(
            proposal_id=_proposal_id(snapshot, source_hash),
            kind=LearningProposalKind.MEMORY,
            category=LearningCandidateCategory.EXPLICIT_CORRECTION,
            provenance=provenance,
            candidate=candidate,
            created_at=now,
        )
        replay = await _load_replayed_correction(
            self._store,
            proposal=proposal,
            candidate=candidate,
        )
        if replay is not None:
            return replay

        if proposal.state is LearningProposalState.REJECTED:
            request = ExplicitCorrectionCommit(proposal=proposal)
            result = await self._store.commit_explicit_correction(request)
            return _validate_store_result(request, result)

        if frozenset(candidate) == _CORRECTION_KEYS:
            correction = _safe_correction(candidate)
            source_id = correction.source_id
            resource_id = correction.resource_id
            resource_revision = correction.resource_revision
            resource = await self._catalog.load_resource(
                snapshot.operation.agent_id,
                resource_id,
            )
            _validate_catalog_binding(
                resource,
                agent_id=snapshot.operation.agent_id,
                source_id=source_id,
                resource_id=resource_id,
                resource_revision=resource_revision,
            )
            assert resource is not None
            scope = MemoryScope(
                agent_id=snapshot.operation.agent_id,
                source_id=resource.source_id,
                resource_id=resource.id,
            )
            logical_key = normalize_memory_logical_key(
                f"{correction.field}:{correction.business_term}"
            )
            existing = await self._store.load_learning_memory(
                scope,
                MemoryKind.RESOURCE_ALIAS,
                logical_key,
            )
            _validate_existing_memory(
                existing,
                scope=scope,
                kind=MemoryKind.RESOURCE_ALIAS,
                logical_key=logical_key,
            )
            intended, expected_version = _intended_memory(
                correction,
                resource=resource,
                scope=scope,
                logical_key=logical_key,
                existing=existing,
                provenance=_user_statement_provenance(
                    snapshot,
                    source_hash=source_hash,
                ),
                created_at=now,
            )
        else:
            content = _safe_semantic_fact(candidate)
            scope = MemoryScope(agent_id=snapshot.operation.agent_id)
            logical_key = normalize_memory_logical_key(content)
            existing = await self._store.load_learning_memory(
                scope,
                MemoryKind.SEMANTIC_FACT,
                logical_key,
            )
            _validate_existing_memory(
                existing,
                scope=scope,
                kind=MemoryKind.SEMANTIC_FACT,
                logical_key=logical_key,
            )
            intended, expected_version = _intended_semantic_memory(
                content,
                scope=scope,
                logical_key=logical_key,
                existing=existing,
                provenance=_user_statement_provenance(
                    snapshot,
                    source_hash=source_hash,
                ),
                created_at=now,
            )

        decision = LearningDecision(
            proposal_id=proposal.id,
            idempotency_key=proposal.idempotency_key,
            candidate_hash=proposal.candidate_hash,
            state=LearningProposalState.COMMITTED,
            decided_at=now,
            result_memory_id=intended.record.id,
            result_memory_version=intended.version.version,
        )
        request = ExplicitCorrectionCommit(
            proposal=proposal,
            expected_memory_version=expected_version,
            intended_memory=intended,
            decision=decision,
        )
        result = await self._store.commit_explicit_correction(request)
        return _validate_store_result(request, result)

    async def propose_evidence_fact(
        self,
        snapshot: OperationSnapshot,
    ) -> LearningProposal | None:
        """Persist a visible fact proposal only from accepted governed evidence."""

        if not isinstance(snapshot, OperationSnapshot):
            raise TypeError("evidence fact requires an OperationSnapshot")
        _validate_completed_user_operation(snapshot)
        message = snapshot.trigger.payload.get("message")
        if not isinstance(message, str):
            return None
        match = _EVIDENCE_FACT.fullmatch(message.strip())
        if match is None:
            return None
        content = _normalized_statement(match.group("content"))
        evidence = _eligible_fact_evidence(snapshot)
        candidate: dict[str, object] = {
            "content": content,
            "kind": MemoryKind.SEMANTIC_FACT.value,
        }
        if evidence is not None:
            facts = evidence.validation_facts
            candidate.update(
                {
                    "resource_ids": facts.resource_ids,
                    "resource_revisions": tuple(
                        {"resource_id": resource_id, "revision": revision}
                        for resource_id, revision in facts.resource_revisions
                    ),
                    "source_ids": facts.source_ids,
                    "source_revisions": tuple(
                        {"revision": revision, "source_id": source_id}
                        for source_id, revision in facts.source_revisions
                    ),
                }
            )
        source_hash = _hash_text(message) if evidence is None else evidence.content_hash
        provenance = LearningProvenance(
            agent_id=snapshot.operation.agent_id,
            operation_id=snapshot.operation.id,
            trigger_id=snapshot.trigger.id,
            source_outcome=LearningSourceOutcome.SUCCEEDED,
            source_hash=source_hash,
            evidence_id=None if evidence is None else evidence.id,
            evidence_accepted=evidence is not None,
        )
        proposal_id = _proposal_id(
            snapshot,
            _hash_text(f"evidence-fact:{message}"),
        )
        if evidence is None:
            proposal = LearningProposal.reject(
                proposal_id=proposal_id,
                kind=LearningProposalKind.MEMORY,
                category=LearningCandidateCategory.EVIDENCE_BACKED_FACT,
                provenance=provenance,
                candidate=candidate,
                created_at=self._now(),
                rejection_category=LearningRejectionCategory.INELIGIBLE_SOURCE,
                rejection_reason="accepted_current_read_evidence_required",
            )
        else:
            proposal = LearningProposal.create(
                proposal_id=proposal_id,
                kind=LearningProposalKind.MEMORY,
                category=LearningCandidateCategory.EVIDENCE_BACKED_FACT,
                provenance=provenance,
                candidate=candidate,
                created_at=self._now(),
            )
        stored = await self._store.create_proposal(proposal)
        if not isinstance(stored, LearningProposal) or (
            stored.idempotency_key != proposal.idempotency_key
            or stored.candidate_hash != proposal.candidate_hash
            or stored.provenance != proposal.provenance
        ):
            raise ExplicitCorrectionStoreContractError(
                "store returned another evidence-backed fact proposal"
            )
        return stored

    def _now(self) -> datetime:
        value = self._clock()
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "explicit correction clock must return a timezone-aware datetime"
            )
        return value


async def _load_replayed_correction(
    store: ExplicitCorrectionStore,
    *,
    proposal: LearningProposal,
    candidate: Mapping[str, object],
) -> ExplicitCorrectionResult | None:
    """Return an exact terminal replay before consulting mutable catalog state."""

    existing = await store.load_proposal(
        proposal.provenance.agent_id,
        proposal.id,
    )
    if existing is None:
        return None
    if not isinstance(existing, LearningProposal) or any(
        getattr(existing, field_name) != getattr(proposal, field_name)
        for field_name in (
            "id",
            "kind",
            "category",
            "provenance",
            "candidate_hash",
            "idempotency_key",
        )
    ):
        raise ExplicitCorrectionStoreContractError(
            "store returned another correction proposal"
        )
    if existing.state is LearningProposalState.PROPOSED:
        raise ExplicitCorrectionStoreContractError(
            "atomic correction storage exposed a nonterminal proposal"
        )
    if existing.state is LearningProposalState.REJECTED:
        if (
            proposal.state is not LearningProposalState.REJECTED
            or existing.rejection_category != proposal.rejection_category
            or existing.rejection_reason != proposal.rejection_reason
            or existing.decision_hash != proposal.decision_hash
        ):
            raise ExplicitCorrectionStoreContractError(
                "stored correction rejection differs from its safety result"
            )
        return ExplicitCorrectionResult(
            proposal=existing,
            memory=None,
            replayed=True,
        )
    if proposal.state is not LearningProposalState.PROPOSED:
        raise ExplicitCorrectionStoreContractError(
            "stored committed correction conflicts with current safety validation"
        )
    if existing.candidate_payload != proposal.candidate_payload:
        raise ExplicitCorrectionStoreContractError(
            "stored committed correction payload differs from its safety result"
        )

    scope, kind, logical_key = _candidate_memory_identity(
        candidate,
        agent_id=proposal.provenance.agent_id,
    )
    history = await store.load_learning_memory(scope, kind, logical_key)
    _validate_existing_memory(
        history,
        scope=scope,
        kind=kind,
        logical_key=logical_key,
    )
    if history is None:
        raise ExplicitCorrectionStoreContractError(
            "committed correction is missing its memory history"
        )
    result = ExplicitCorrectionResult(
        proposal=existing,
        memory=history,
        replayed=True,
    )
    _validate_loaded_replay(candidate, result)
    return result


def _validate_loaded_replay(
    candidate: Mapping[str, object],
    result: ExplicitCorrectionResult,
) -> None:
    assert result.memory is not None
    result_version = result.proposal.result_memory_version
    assert result_version is not None
    version = next(
        item for item in result.memory.versions if item.version == result_version
    )
    if frozenset(candidate) == _CORRECTION_KEYS:
        correction = _safe_correction(candidate)
        expected_attributes = FrozenJsonObject.from_mapping(
            {
                "business_term": correction.business_term,
                "field": correction.field,
                "stored_value": correction.stored_value,
            }
        )
        expected_content = "Resource alias " + canonical_json(expected_attributes)
        expected_kind = MemoryKind.RESOURCE_ALIAS
        expected_revision: str | None = correction.resource_revision
    else:
        content = _safe_semantic_fact(candidate)
        expected_attributes = FrozenJsonObject.from_mapping({"statement": content})
        expected_content = content
        expected_kind = MemoryKind.SEMANTIC_FACT
        expected_revision = None
    provenance = version.provenance
    proposal_provenance = result.proposal.provenance
    if (
        result.memory.record.kind is not expected_kind
        or version.content != expected_content
        or version.attributes != expected_attributes
        or version.creator is not MemoryCreator.LEARNING_SERVICE
        or version.confidence != 1.0
        or version.resource_revision != expected_revision
        or version.expires_at is not None
        or provenance.kind is not MemoryProvenanceKind.USER_STATEMENT
        or provenance.content_hash != proposal_provenance.source_hash
        or provenance.operation_id != proposal_provenance.operation_id
        or provenance.trigger_id != proposal_provenance.trigger_id
    ):
        raise ExplicitCorrectionStoreContractError(
            "stored correction replay does not match its originating proposal"
        )


def _validate_completed_user_operation(snapshot: OperationSnapshot) -> None:
    if snapshot.operation.status is not OperationStatus.SUCCEEDED or (
        snapshot.loop_state.phase is not LoopPhase.TERMINAL
    ):
        raise ExplicitCorrectionNotEligibleError(
            "learning requires a completed SUCCEEDED operation"
        )
    if snapshot.trigger.kind is not TriggerKind.USER:
        raise ExplicitCorrectionNotEligibleError(
            "learning requires the original USER trigger"
        )
    if (
        snapshot.operation.trigger_id != snapshot.trigger.id
        or snapshot.operation.agent_id != snapshot.trigger.agent_id
    ):
        raise ExplicitCorrectionNotEligibleError(
            "operation is not bound to the supplied original trigger"
        )


def is_explicit_learning_message(message: object) -> bool:
    """Whether a user message requests direct, governed memory learning."""

    if not isinstance(message, str):
        return False
    lowered = message.casefold()
    return message.startswith(RESOURCE_ALIAS_CORRECTION_PREFIX) or any(
        lowered.startswith(prefix) for prefix in NATURAL_REMEMBER_PREFIXES
    )


def _parse_trigger_candidate(
    message: str,
    *,
    snapshot: OperationSnapshot,
) -> dict[str, object]:
    _required_text(
        message,
        "explicit correction trigger message",
        maximum=_MAX_TRIGGER_CHARACTERS,
    )
    if not message.startswith(RESOURCE_ALIAS_CORRECTION_PREFIX):
        content = _natural_statement(message)
        alias = _natural_alias_parts(content)
        binding = _natural_resource_binding(snapshot)
        if alias is not None:
            if binding is None:
                raise ExplicitCorrectionNotEligibleError(
                    "natural resource alias requires one accepted current read binding"
                )
            business_term, field, stored_value = alias
            source_id, resource_id, resource_revision = binding
            return ResourceAliasCorrection(
                source_id=source_id,
                resource_id=resource_id,
                resource_revision=resource_revision,
                field=field,
                business_term=business_term,
                stored_value=stored_value,
            ).candidate.to_dict()
        normalized = _normalized_statement(content)
        return {
            "content": normalized,
            "kind": MemoryKind.SEMANTIC_FACT.value,
        }
    encoded = message.removeprefix(RESOURCE_ALIAS_CORRECTION_PREFIX)
    try:
        parsed = json.loads(
            encoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ExplicitCorrectionFormatError(
            "resource-alias correction must contain valid canonical JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise ExplicitCorrectionFormatError(
            "resource-alias correction JSON must be one object"
        )
    try:
        canonical = canonical_json(parsed)
    except (TypeError, ValueError) as error:
        raise ExplicitCorrectionFormatError(
            "resource-alias correction contains unsupported JSON"
        ) from error
    if encoded != canonical:
        raise ExplicitCorrectionFormatError(
            "resource-alias correction JSON must use canonical encoding"
        )
    return cast(dict[str, object], parsed)


def _natural_statement(message: str) -> str:
    lowered = message.casefold()
    prefix = next(
        (
            candidate
            for candidate in NATURAL_REMEMBER_PREFIXES
            if lowered.startswith(candidate)
        ),
        None,
    )
    if prefix is None:
        raise ExplicitCorrectionFormatError(
            "trigger does not use a supported explicit learning phrase"
        )
    content = message[len(prefix) :].strip()
    _required_text(content, "natural learning statement", maximum=3_900)
    return content


def _normalized_statement(value: str) -> str:
    normalized = " ".join(value.strip().split())
    normalized = normalized.rstrip(".! ")
    _required_text(normalized, "natural learning statement", maximum=3_900)
    return normalized[:1].upper() + normalized[1:] + "."


def _natural_alias_parts(value: str) -> tuple[str, str, str] | None:
    normalized = " ".join(value.strip().rstrip(".").split())
    match = _NATURAL_ALIAS.fullmatch(normalized) or _NATURAL_ALIAS_MEANS.fullmatch(
        normalized
    )
    if match is None:
        return None
    values = tuple(
        _unquoted(match.group(name)) for name in ("business", "field", "stored")
    )
    if any(not value for value in values):
        return None
    return cast(tuple[str, str, str], values)


def _unquoted(value: str) -> str:
    normalized = value.strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0]
        in {
            '"',
            "'",
        }
    ):
        normalized = normalized[1:-1].strip()
    return normalized


def _natural_resource_binding(
    snapshot: OperationSnapshot,
) -> tuple[str, str, str] | None:
    bindings = {
        (
            evidence.validation_facts.source_ids[0],
            evidence.validation_facts.resource_ids[0],
            evidence.validation_facts.resource_revisions[0][1],
        )
        for evidence in _governed_read_evidence(snapshot)
        if len(evidence.validation_facts.source_ids) == 1
        and len(evidence.validation_facts.resource_ids) == 1
        and len(evidence.validation_facts.resource_revisions) == 1
        and evidence.validation_facts.resource_revisions[0][0]
        == evidence.validation_facts.resource_ids[0]
    }
    if len(bindings) != 1:
        return None
    return next(iter(bindings))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate correction key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"unsupported JSON constant: {value}")


def _candidate_text(
    candidate: Mapping[str, object],
    key: str,
    *,
    maximum: int,
) -> str:
    value = candidate.get(key)
    try:
        _required_text(value, f"correction {key}", maximum=maximum)
    except (TypeError, ValueError) as error:
        raise ExplicitCorrectionFormatError(
            f"resource-alias correction requires bounded {key}"
        ) from error
    assert isinstance(value, str)
    return value


def _safe_correction(candidate: Mapping[str, object]) -> ResourceAliasCorrection:
    if frozenset(candidate) != _CORRECTION_KEYS:
        raise ExplicitCorrectionFormatError(
            "safe resource-alias correction must contain exactly the documented keys"
        )
    try:
        return ResourceAliasCorrection(
            source_id=_candidate_text(candidate, "source_id", maximum=512),
            resource_id=_candidate_text(candidate, "resource_id", maximum=512),
            resource_revision=_candidate_text(
                candidate,
                "resource_revision",
                maximum=512,
            ),
            field=_candidate_text(candidate, "field", maximum=256),
            business_term=_candidate_text(
                candidate,
                "business_term",
                maximum=256,
            ),
            stored_value=_candidate_text(
                candidate,
                "stored_value",
                maximum=1_024,
            ),
        )
    except ValueError as error:
        raise ExplicitCorrectionFormatError(
            "resource-alias correction record is invalid"
        ) from error


def _safe_semantic_fact(candidate: Mapping[str, object]) -> str:
    if (
        frozenset(candidate) != {"content", "kind"}
        or candidate.get("kind") != MemoryKind.SEMANTIC_FACT.value
    ):
        raise ExplicitCorrectionFormatError(
            "natural memory must contain one semantic fact"
        )
    content = candidate.get("content")
    try:
        _required_text(content, "semantic fact content", maximum=3_900)
    except (TypeError, ValueError) as error:
        raise ExplicitCorrectionFormatError(
            "semantic fact content must be bounded normalized text"
        ) from error
    assert isinstance(content, str)
    return content


def _candidate_memory_identity(
    candidate: Mapping[str, object],
    *,
    agent_id: str,
) -> tuple[MemoryScope, MemoryKind, str]:
    if frozenset(candidate) == _CORRECTION_KEYS:
        correction = _safe_correction(candidate)
        scope = MemoryScope(
            agent_id=agent_id,
            source_id=correction.source_id,
            resource_id=correction.resource_id,
        )
        return (
            scope,
            MemoryKind.RESOURCE_ALIAS,
            normalize_memory_logical_key(
                f"{correction.field}:{correction.business_term}"
            ),
        )
    content = _safe_semantic_fact(candidate)
    return (
        MemoryScope(agent_id=agent_id),
        MemoryKind.SEMANTIC_FACT,
        normalize_memory_logical_key(content),
    )


def _user_statement_provenance(
    snapshot: OperationSnapshot,
    *,
    source_hash: str,
) -> MemoryProvenance:
    return MemoryProvenance(
        kind=MemoryProvenanceKind.USER_STATEMENT,
        content_hash=source_hash,
        operation_id=snapshot.operation.id,
        trigger_id=snapshot.trigger.id,
        session_id=snapshot.operation.session_id,
    )


def _governed_read_evidence(snapshot: OperationSnapshot) -> tuple[Evidence, ...]:
    tasks = {task.id: task for task in snapshot.tasks}
    return tuple(
        evidence
        for evidence in snapshot.evidence
        if (
            evidence.operation_id == snapshot.operation.id
            and evidence.accepted
            and evidence.applicable
            and evidence.metadata_schema_version == 1
            and evidence.applicability_reason == "current_operation"
            and evidence.validation_facts.schema_version == 1
            and evidence.validation_facts.validation_passed
            and evidence.validation_facts.in_scope
            and not evidence.validation_facts.destructive
            and evidence.validation_facts.freshness_state == "current"
            and evidence.task_id in tasks
            and tasks[evidence.task_id].status is TaskStatus.SUCCEEDED
            and tasks[evidence.task_id].execution_facts.access_mode is AccessMode.READ
            and not tasks[evidence.task_id].execution_facts.side_effecting
        )
    )


def _eligible_fact_evidence(snapshot: OperationSnapshot) -> Evidence | None:
    eligible = _governed_read_evidence(snapshot)
    return None if not eligible else eligible[-1]


def _validate_catalog_binding(
    resource: CatalogResource | None,
    *,
    agent_id: str,
    source_id: str,
    resource_id: str,
    resource_revision: str,
) -> None:
    if resource is None:
        raise ExplicitCorrectionNotEligibleError(
            "correction resource is absent from the current catalog"
        )
    if not isinstance(resource, CatalogResource):
        raise ExplicitCorrectionStoreContractError(
            "catalog returned an unsupported resource record"
        )
    if (
        resource.agent_id != agent_id
        or resource.id != resource_id
        or resource.source_id != source_id
    ):
        raise ExplicitCorrectionNotEligibleError(
            "correction source/resource is outside the operation agent scope"
        )
    if resource.current_revision != resource_revision:
        raise ExplicitCorrectionNotEligibleError(
            "correction resource revision is not current"
        )


def _validate_existing_alias(
    existing: MemoryHistory | None,
    *,
    scope: MemoryScope,
    logical_key: str,
) -> None:
    _validate_existing_memory(
        existing,
        scope=scope,
        kind=MemoryKind.RESOURCE_ALIAS,
        logical_key=logical_key,
    )


def _validate_existing_memory(
    existing: MemoryHistory | None,
    *,
    scope: MemoryScope,
    kind: MemoryKind,
    logical_key: str,
) -> None:
    if existing is None:
        return
    if not isinstance(existing, MemoryHistory):
        raise ExplicitCorrectionStoreContractError(
            "store returned an unsupported memory history"
        )
    if (
        existing.record.scope != scope
        or existing.record.kind is not kind
        or existing.record.logical_key != logical_key
    ):
        raise ExplicitCorrectionStoreContractError(
            "store returned another logical memory identity"
        )
    if existing.record.state is not MemoryState.ACTIVE:
        raise ExplicitCorrectionNotEligibleError(
            "only active memory can receive explicit learning"
        )


def _intended_memory(
    correction: ResourceAliasCorrection,
    *,
    resource: CatalogResource,
    scope: MemoryScope,
    logical_key: str,
    existing: MemoryHistory | None,
    provenance: MemoryProvenance,
    created_at: datetime,
) -> tuple[MemorySnapshot, int | None]:
    if existing is None:
        memory_id = _memory_id(scope, logical_key)
        version_number = 1
        expected_version = None
        record = MemoryRecord(
            id=memory_id,
            scope=scope,
            kind=MemoryKind.RESOURCE_ALIAS,
            logical_key=logical_key,
            current_version=version_number,
            state=MemoryState.ACTIVE,
            created_at=created_at,
            updated_at=created_at,
        )
    else:
        expected_version = existing.record.current_version
        version_number = expected_version + 1
        if created_at < existing.record.updated_at:
            raise ValueError("explicit correction clock precedes memory history")
        record = replace(
            existing.record,
            current_version=version_number,
            updated_at=created_at,
        )
    version = MemoryVersion(
        memory_id=record.id,
        version=version_number,
        content="Resource alias "
        + canonical_json(
            {
                "business_term": correction.business_term,
                "field": correction.field,
                "stored_value": correction.stored_value,
            }
        ),
        creator=MemoryCreator.LEARNING_SERVICE,
        confidence=1.0,
        sensitivity=_memory_sensitivity(resource.sensitivity),
        provenance=provenance,
        attributes={
            "business_term": correction.business_term,
            "field": correction.field,
            "stored_value": correction.stored_value,
        },
        resource_revision=resource.current_revision,
        supersedes_version=expected_version,
        created_at=created_at,
    )
    return MemorySnapshot(record, version), expected_version


def _intended_semantic_memory(
    content: str,
    *,
    scope: MemoryScope,
    logical_key: str,
    existing: MemoryHistory | None,
    provenance: MemoryProvenance,
    created_at: datetime,
) -> tuple[MemorySnapshot, int | None]:
    if existing is None:
        memory_id = _memory_id(
            scope,
            logical_key,
            kind=MemoryKind.SEMANTIC_FACT,
        )
        version_number = 1
        expected_version = None
        record = MemoryRecord(
            id=memory_id,
            scope=scope,
            kind=MemoryKind.SEMANTIC_FACT,
            logical_key=logical_key,
            current_version=version_number,
            state=MemoryState.ACTIVE,
            created_at=created_at,
            updated_at=created_at,
        )
    else:
        expected_version = existing.record.current_version
        version_number = expected_version + 1
        if created_at < existing.record.updated_at:
            raise ValueError("explicit learning clock precedes memory history")
        record = replace(
            existing.record,
            current_version=version_number,
            updated_at=created_at,
        )
    version = MemoryVersion(
        memory_id=record.id,
        version=version_number,
        content=content,
        creator=MemoryCreator.LEARNING_SERVICE,
        confidence=1.0,
        sensitivity=MemorySensitivity.INTERNAL,
        provenance=provenance,
        attributes={"statement": content},
        supersedes_version=expected_version,
        created_at=created_at,
    )
    return MemorySnapshot(record, version), expected_version


def _validate_store_result(
    request: ExplicitCorrectionCommit,
    result: object,
) -> ExplicitCorrectionResult:
    if not isinstance(result, ExplicitCorrectionResult):
        raise ExplicitCorrectionStoreContractError(
            "store must return ExplicitCorrectionResult"
        )
    proposal = result.proposal
    if (
        proposal.id != request.proposal.id
        or proposal.idempotency_key != request.proposal.idempotency_key
        or proposal.candidate_hash != request.proposal.candidate_hash
        or proposal.provenance != request.proposal.provenance
    ):
        raise ExplicitCorrectionStoreContractError(
            "store returned another correction proposal"
        )
    if request.proposal.state is LearningProposalState.REJECTED:
        if not result.replayed and proposal != request.proposal:
            raise ExplicitCorrectionStoreContractError(
                "store changed a terminal redacted rejection"
            )
        return result

    assert request.decision is not None
    assert request.intended_memory is not None
    if result.replayed:
        assert result.memory is not None
        _validate_replayed_memory(request.intended_memory, result)
        return result
    expected = resolve_learning_proposal(request.proposal, request.decision)
    if proposal != expected or result.memory is None:
        raise ExplicitCorrectionStoreContractError(
            "store did not atomically apply the correction decision"
        )
    if (
        result.memory.record != request.intended_memory.record
        or result.memory.current != request.intended_memory.version
    ):
        raise ExplicitCorrectionStoreContractError(
            "store returned a different memory head"
        )
    return result


def _validate_replayed_memory(
    intended: MemorySnapshot,
    result: ExplicitCorrectionResult,
) -> None:
    assert result.memory is not None
    result_version = result.proposal.result_memory_version
    assert result_version is not None
    historical = next(
        version
        for version in result.memory.versions
        if version.version == result_version
    )
    expected = intended.version
    if (
        result.memory.record.logical_identity != intended.record.logical_identity
        or any(
            getattr(historical, field_name) != getattr(expected, field_name)
            for field_name in (
                "content",
                "creator",
                "confidence",
                "sensitivity",
                "provenance",
                "attributes",
                "expires_at",
                "resource_revision",
            )
        )
    ):
        raise ExplicitCorrectionStoreContractError(
            "idempotent replay returned a different semantic memory result"
        )


def _memory_sensitivity(value: Sensitivity) -> MemorySensitivity:
    if value is Sensitivity.PUBLIC:
        return MemorySensitivity.PUBLIC
    if value is Sensitivity.CONFIDENTIAL:
        return MemorySensitivity.CONFIDENTIAL
    if value is Sensitivity.RESTRICTED:
        return MemorySensitivity.RESTRICTED
    return MemorySensitivity.INTERNAL


def _proposal_id(snapshot: OperationSnapshot, source_hash: str) -> str:
    digest = sha256(
        canonical_json(
            {
                "agent_id": snapshot.operation.agent_id,
                "operation_id": snapshot.operation.id,
                "source_hash": source_hash,
                "trigger_id": snapshot.trigger.id,
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"learning-proposal:{digest}"


def _memory_id(
    scope: MemoryScope,
    logical_key: str,
    *,
    kind: MemoryKind = MemoryKind.RESOURCE_ALIAS,
) -> str:
    digest = sha256(
        canonical_json(
            {
                "kind": kind.value,
                "logical_key": logical_key,
                "scope": scope.fingerprint,
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"memory-{kind.value.replace('_', '-')}:{digest}"


def _hash_text(value: str) -> str:
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def _required_text(value: object, field_name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field_name} must be a bounded normalized string")


def _positive_integer(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "NATURAL_REMEMBER_PREFIXES",
    "RESOURCE_ALIAS_CORRECTION_PREFIX",
    "ExplicitCorrectionCatalogReader",
    "ExplicitCorrectionCommit",
    "ExplicitCorrectionFormatError",
    "ExplicitCorrectionLearningError",
    "ExplicitCorrectionLearningService",
    "ExplicitCorrectionNotEligibleError",
    "ExplicitCorrectionResult",
    "ExplicitCorrectionStore",
    "ExplicitCorrectionStoreConflictError",
    "ExplicitCorrectionStoreContractError",
    "ResourceAliasCorrection",
    "is_explicit_learning_message",
]
