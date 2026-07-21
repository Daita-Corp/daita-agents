"""Portable session-history projection and compression checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
import re
from typing import Protocol

from .._json import canonical_json
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelProfile,
    ModelSensitivity,
    TextBlock,
    ToolResultBlock,
)
from ..operations.governance import ApprovalStatus
from ..operations.models import OperationStatus
from ..sessions import SessionCompressionCheckpoint, SessionTranscript
from .budgeting import RequiredContextOverflow, estimate_context_block_tokens
from .models import (
    ContextBlock,
    ContextKind,
    ContextMessageGroup,
    ContextProvenance,
    ContextTrust,
)

_CORRECTION = re.compile(
    r"\b(?:actually|correction|corrected|instead|means|remember|should be)\b",
    re.IGNORECASE,
)
_MAX_REFERENCES = 1_024
_TERMINAL_OPERATION_STATUS_VALUES = frozenset(
    {
        OperationStatus.SUCCEEDED.value,
        OperationStatus.FAILED.value,
        OperationStatus.CANCELLED.value,
        OperationStatus.INTERRUPTED.value,
    }
)


def _session_sensitivity(
    facts: Mapping[str, SessionOperationFacts],
    historical_operation_ids: tuple[str, ...],
) -> ModelSensitivity:
    return max(
        (facts[operation_id].sensitivity for operation_id in historical_operation_ids),
        default=ModelSensitivity.INTERNAL,
        key=lambda sensitivity: sensitivity.routing_rank,
    )


def _required_text(value: str, field_name: str, *, maximum: int = 512) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} characters")


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    normalized = tuple(values)
    if len(normalized) > _MAX_REFERENCES:
        raise ValueError(f"{field_name} exceeds its reference bound")
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class SessionApprovalStateFact:
    """One approval identity and its canonical durable state."""

    approval_id: str
    status: ApprovalStatus

    def __post_init__(self) -> None:
        _required_text(self.approval_id, "session approval fact approval_id")
        if not isinstance(self.status, ApprovalStatus):
            raise TypeError("session approval fact status must be an ApprovalStatus")


@dataclass(frozen=True, slots=True)
class SessionResourceScopeFact:
    """One exact source/resource revision scope used by an operation."""

    source_id: str
    resource_id: str
    source_revision: str
    resource_revision: str

    def __post_init__(self) -> None:
        for value, field_name, maximum in (
            (self.source_id, "session resource scope source_id", 512),
            (self.resource_id, "session resource scope resource_id", 512),
            (
                self.source_revision,
                "session resource scope source_revision",
                1_024,
            ),
            (
                self.resource_revision,
                "session resource scope resource_revision",
                1_024,
            ),
        ):
            _required_text(value, field_name, maximum=maximum)


@dataclass(frozen=True, slots=True)
class SessionOperationFacts:
    """Bounded operation-owned facts needed by session compression."""

    operation_id: str
    agent_id: str
    session_id: str
    revision: str
    status: str
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL
    evidence_ids: tuple[str, ...] = ()
    approval_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()
    final_text: str | None = None
    objective: str | None = None
    terminal_reason: str | None = None
    approval_state_facts: tuple[SessionApprovalStateFact, ...] = ()
    resource_scope_facts: tuple[SessionResourceScopeFact, ...] = ()

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.operation_id, "session operation id"),
            (self.agent_id, "session operation agent_id"),
            (self.session_id, "session operation session_id"),
            (self.revision, "session operation revision"),
            (self.status, "session operation status"),
        ):
            _required_text(value, field_name)
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("session operation sensitivity must be a ModelSensitivity")
        object.__setattr__(
            self,
            "evidence_ids",
            _string_tuple(self.evidence_ids, "session operation evidence_ids"),
        )
        object.__setattr__(
            self,
            "approval_ids",
            _string_tuple(self.approval_ids, "session operation approval_ids"),
        )
        object.__setattr__(
            self,
            "resource_ids",
            _string_tuple(self.resource_ids, "session operation resource_ids"),
        )
        if self.final_text is not None:
            _required_text(
                self.final_text,
                "session operation final_text",
                maximum=32_768,
            )
        if self.objective is not None:
            _required_text(
                self.objective,
                "session operation objective",
                maximum=2_048,
            )
        if self.terminal_reason is not None:
            _required_text(
                self.terminal_reason,
                "session operation terminal_reason",
                maximum=512,
            )

        approval_facts = tuple(self.approval_state_facts)
        if len(approval_facts) > _MAX_REFERENCES:
            raise ValueError(
                "session operation approval_state_facts exceed their bound"
            )
        if any(
            not isinstance(fact, SessionApprovalStateFact) for fact in approval_facts
        ):
            raise TypeError(
                "session operation approval_state_facts must contain "
                "SessionApprovalStateFact records"
            )
        approval_fact_ids = tuple(fact.approval_id for fact in approval_facts)
        if len(approval_fact_ids) != len(set(approval_fact_ids)):
            raise ValueError(
                "session operation approval_state_facts cannot repeat approvals"
            )
        if not set(approval_fact_ids) <= set(self.approval_ids):
            raise ValueError(
                "session operation approval state facts must reference approval_ids"
            )

        resource_facts = tuple(self.resource_scope_facts)
        if len(resource_facts) > _MAX_REFERENCES:
            raise ValueError(
                "session operation resource_scope_facts exceed their bound"
            )
        if any(
            not isinstance(fact, SessionResourceScopeFact) for fact in resource_facts
        ):
            raise TypeError(
                "session operation resource_scope_facts must contain "
                "SessionResourceScopeFact records"
            )
        resource_fact_ids = tuple(fact.resource_id for fact in resource_facts)
        if len(resource_fact_ids) != len(set(resource_fact_ids)):
            raise ValueError(
                "session operation resource_scope_facts cannot repeat resources"
            )
        if not set(resource_fact_ids) <= set(self.resource_ids):
            raise ValueError(
                "session operation resource scope facts must reference resource_ids"
            )
        object.__setattr__(self, "approval_state_facts", approval_facts)
        object.__setattr__(self, "resource_scope_facts", resource_facts)


@dataclass(frozen=True, slots=True)
class SessionCompressionPolicy:
    """Hard deterministic bounds for one session projection."""

    schema_version: int = 1
    compression_threshold_tokens: int | None = None
    retain_latest_operations: int = 4
    max_summary_characters: int = 16_384
    max_excerpt_characters: int = 512
    max_corrections: int = 32

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("session compression policy schema_version must be 1")
        if self.compression_threshold_tokens is not None and (
            not isinstance(self.compression_threshold_tokens, int)
            or isinstance(self.compression_threshold_tokens, bool)
            or self.compression_threshold_tokens < 1
        ):
            raise ValueError(
                "compression_threshold_tokens must be a positive integer or None"
            )
        for value, field_name in (
            (self.retain_latest_operations, "retain_latest_operations"),
            (self.max_summary_characters, "max_summary_characters"),
            (self.max_excerpt_characters, "max_excerpt_characters"),
            (self.max_corrections, "max_corrections"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_summary_characters > 32_768:
            raise ValueError("max_summary_characters cannot exceed 32768")
        if self.max_excerpt_characters > self.max_summary_characters:
            raise ValueError("excerpt bound cannot exceed the summary bound")


@dataclass(frozen=True, slots=True)
class SessionContextProjection:
    """A session-owned summary plus recent blocks for one current operation."""

    agent_id: str
    session_id: str
    current_operation_id: str
    historical_operation_ids: tuple[str, ...]
    blocks: tuple[ContextBlock, ...]
    checkpoint: SessionCompressionCheckpoint | None
    compressed_now: bool
    threshold_tokens: int
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.agent_id, "session projection agent_id"),
            (self.session_id, "session projection session_id"),
            (self.current_operation_id, "session projection current_operation_id"),
        ):
            _required_text(value, field_name)
        historical = _string_tuple(
            self.historical_operation_ids,
            "session projection historical_operation_ids",
        )
        if self.current_operation_id in historical:
            raise ValueError("current operation cannot appear in historical projection")
        blocks = tuple(self.blocks)
        if any(not isinstance(block, ContextBlock) for block in blocks):
            raise TypeError("session projection blocks must be ContextBlock records")
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, SessionCompressionCheckpoint):
                raise TypeError(
                    "session projection checkpoint must be a compression checkpoint"
                )
            if (
                self.checkpoint.agent_id != self.agent_id
                or self.checkpoint.session_id != self.session_id
            ):
                raise ValueError("session projection checkpoint scope does not match")
        if not isinstance(self.compressed_now, bool):
            raise TypeError("session projection compressed_now must be a boolean")
        if (
            not isinstance(self.threshold_tokens, int)
            or isinstance(self.threshold_tokens, bool)
            or self.threshold_tokens < 1
        ):
            raise ValueError("session projection threshold must be positive")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("session projection sensitivity must be ModelSensitivity")
        object.__setattr__(self, "historical_operation_ids", historical)
        object.__setattr__(self, "blocks", blocks)

    @property
    def estimated_tokens(self) -> int:
        return sum(estimate_context_block_tokens(block) for block in self.blocks)


class SessionCompressionError(RuntimeError):
    """Base portable session-compression failure."""


class SessionCompressionScopeError(SessionCompressionError):
    """Raised when session, operation, or checkpoint ownership disagrees."""


class SessionCompressionIntegrityError(SessionCompressionError):
    """Raised when a checkpoint no longer matches its immutable source prefix."""


class _IncompleteToolExchangeError(SessionCompressionIntegrityError):
    """Internal signal for a terminal transcript that needs factual fallback."""


class SessionTranscriptReader(Protocol):
    async def load_session(
        self,
        agent_id: str,
        session_id: str,
    ) -> SessionTranscript | None: ...


class SessionCompressionCheckpointReader(Protocol):
    async def load_session_compression(
        self,
        agent_id: str,
        session_id: str,
    ) -> SessionCompressionCheckpoint | None: ...


class SessionOperationFactsReader(Protocol):
    async def load_session_operation(
        self,
        operation_id: str,
    ) -> SessionOperationFacts | None: ...


class SessionCompressionCheckpointCommitter(Protocol):
    async def commit_session_compression(
        self,
        checkpoint: SessionCompressionCheckpoint,
        *,
        expected_version: int,
    ) -> SessionCompressionCheckpoint: ...


class SessionCompressionService:
    """Create bounded model projections without deleting canonical history."""

    def __init__(
        self,
        *,
        transcripts: SessionTranscriptReader,
        checkpoints: SessionCompressionCheckpointReader,
        operations: SessionOperationFactsReader,
        committer: SessionCompressionCheckpointCommitter,
        policy: SessionCompressionPolicy,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        for value, method_name, field_name in (
            (transcripts, "load_session", "transcripts"),
            (checkpoints, "load_session_compression", "checkpoints"),
            (operations, "load_session_operation", "operations"),
            (committer, "commit_session_compression", "committer"),
        ):
            if not callable(getattr(value, method_name, None)):
                raise TypeError(f"{field_name} must provide {method_name}()")
        if not isinstance(policy, SessionCompressionPolicy):
            raise TypeError("policy must be a SessionCompressionPolicy")
        if not callable(clock) or not callable(id_factory):
            raise TypeError("clock and id_factory must be callable")
        self._transcripts = transcripts
        self._checkpoints = checkpoints
        self._operations = operations
        self._committer = committer
        self._policy = policy
        self._clock = clock
        self._id_factory = id_factory

    async def project(
        self,
        *,
        agent_id: str,
        session_id: str,
        current_operation_id: str,
        profile: ModelProfile,
        maximum_projection_tokens: int,
    ) -> SessionContextProjection:
        for value, field_name in (
            (agent_id, "session projection agent_id"),
            (session_id, "session projection session_id"),
            (current_operation_id, "session projection current_operation_id"),
        ):
            _required_text(value, field_name)
        if not isinstance(profile, ModelProfile):
            raise TypeError("session projection profile must be a ModelProfile")
        if (
            not isinstance(maximum_projection_tokens, int)
            or isinstance(maximum_projection_tokens, bool)
            or maximum_projection_tokens < 0
        ):
            raise ValueError(
                "session maximum_projection_tokens must be a non-negative integer"
            )

        transcript = await self._transcripts.load_session(agent_id, session_id)
        if transcript is None:
            raise SessionCompressionScopeError("session transcript does not exist")
        if (
            transcript.session.agent_id != agent_id
            or transcript.session.id != session_id
        ):
            raise SessionCompressionScopeError(
                "session transcript scope does not match"
            )
        try:
            current_position = transcript.operation_ids.index(current_operation_id)
        except ValueError as error:
            raise SessionCompressionScopeError(
                "current operation is not linked to the session"
            ) from error
        if current_position != len(transcript.operation_ids) - 1:
            raise SessionCompressionScopeError(
                "current operation must be the session history frontier"
            )

        historical_ids = transcript.operation_ids[:current_position]
        facts = await self._load_facts(
            (*historical_ids, current_operation_id),
            agent_id=agent_id,
            session_id=session_id,
        )
        messages_by_operation = _messages_by_operation(transcript, historical_ids)
        checkpoint = await self._checkpoints.load_session_compression(
            agent_id,
            session_id,
        )
        if checkpoint is not None:
            self._validate_checkpoint(
                checkpoint,
                transcript=transcript,
                historical_ids=historical_ids,
                messages_by_operation=messages_by_operation,
                facts=facts,
            )

        configured_threshold = self._policy.compression_threshold_tokens
        threshold = min(
            (
                max(1, profile.maximum_input_tokens * 3 // 4)
                if configured_threshold is None
                else configured_threshold
            ),
            profile.maximum_input_tokens,
        )
        blocks = _projection_blocks(
            agent_id=agent_id,
            session_id=session_id,
            current_operation_id=current_operation_id,
            historical_ids=historical_ids,
            messages_by_operation=messages_by_operation,
            facts=facts,
            checkpoint=checkpoint,
            policy=self._policy,
        )
        projected_tokens = sum(estimate_context_block_tokens(block) for block in blocks)
        current_prefix_count = (
            0 if checkpoint is None else checkpoint.through_position + 1
        )
        compressed_now = False
        threshold_exceeded = projected_tokens > threshold
        residual_exceeded = projected_tokens > maximum_projection_tokens
        threshold_requires_advance = threshold_exceeded and current_prefix_count < len(
            historical_ids
        )
        if threshold_requires_advance or residual_exceeded:
            available_recent = len(historical_ids) - current_prefix_count
            maximum_recent = min(
                self._policy.retain_latest_operations,
                available_recent,
            )
            first_candidate_prefix = len(historical_ids) - maximum_recent
            winning_checkpoint: SessionCompressionCheckpoint | None = None
            winning_blocks: tuple[ContextBlock, ...] | None = None
            residual_checkpoint: SessionCompressionCheckpoint | None = None
            residual_blocks: tuple[ContextBlock, ...] | None = None
            minimum_tokens = projected_tokens
            first_eligible_prefix = max(
                current_prefix_count,
                first_candidate_prefix,
            )
            target_tokens = (
                min(threshold, maximum_projection_tokens)
                if threshold_exceeded
                else maximum_projection_tokens
            )
            for prefix_count in range(
                first_eligible_prefix,
                len(historical_ids) + 1,
            ):
                if prefix_count == current_prefix_count:
                    candidate_checkpoint = checkpoint
                    candidate_blocks = blocks
                    candidate_tokens = projected_tokens
                else:
                    candidate_checkpoint = self._new_checkpoint(
                        agent_id=agent_id,
                        session_id=session_id,
                        prefix_ids=historical_ids[:prefix_count],
                        messages_by_operation=messages_by_operation,
                        facts=facts,
                        previous=checkpoint,
                    )
                    candidate_blocks = _projection_blocks(
                        agent_id=agent_id,
                        session_id=session_id,
                        current_operation_id=current_operation_id,
                        historical_ids=historical_ids,
                        messages_by_operation=messages_by_operation,
                        facts=facts,
                        checkpoint=candidate_checkpoint,
                        policy=self._policy,
                    )
                    candidate_tokens = sum(
                        estimate_context_block_tokens(block)
                        for block in candidate_blocks
                    )
                minimum_tokens = candidate_tokens
                if candidate_tokens <= maximum_projection_tokens:
                    residual_checkpoint = candidate_checkpoint
                    residual_blocks = candidate_blocks
                    if candidate_tokens <= target_tokens:
                        winning_checkpoint = candidate_checkpoint
                        winning_blocks = candidate_blocks
                        break

            if winning_blocks is None and residual_blocks is not None:
                winning_checkpoint = residual_checkpoint
                winning_blocks = residual_blocks

            if winning_blocks is None:
                raise RequiredContextOverflow(
                    profile_id=profile.id,
                    required_tokens=minimum_tokens,
                    available_tokens=maximum_projection_tokens,
                    tool_tokens=0,
                    output_reserve_tokens=profile.max_output_tokens,
                    input_limit_tokens=maximum_projection_tokens,
                    minimum_session_tokens=minimum_tokens,
                    projected_session_tokens=projected_tokens,
                )

            if winning_checkpoint is not None and winning_checkpoint != checkpoint:
                committed = await self._committer.commit_session_compression(
                    winning_checkpoint,
                    expected_version=(0 if checkpoint is None else checkpoint.version),
                )
                if not isinstance(committed, SessionCompressionCheckpoint):
                    raise SessionCompressionIntegrityError(
                        "checkpoint committer returned an invalid record"
                    )
                if committed != winning_checkpoint:
                    raise SessionCompressionIntegrityError(
                        "checkpoint commit did not return the proposed CAS winner"
                    )
                checkpoint = committed
                compressed_now = True
            blocks = winning_blocks

        return SessionContextProjection(
            agent_id=agent_id,
            session_id=session_id,
            current_operation_id=current_operation_id,
            historical_operation_ids=historical_ids,
            blocks=blocks,
            checkpoint=checkpoint,
            compressed_now=compressed_now,
            threshold_tokens=threshold,
            sensitivity=_session_sensitivity(facts, historical_ids),
        )

    async def _load_facts(
        self,
        operation_ids: tuple[str, ...],
        *,
        agent_id: str,
        session_id: str,
    ) -> dict[str, SessionOperationFacts]:
        facts: dict[str, SessionOperationFacts] = {}
        for operation_id in operation_ids:
            record = await self._operations.load_session_operation(operation_id)
            if record is None:
                raise SessionCompressionScopeError(
                    f"session operation facts do not exist: {operation_id}"
                )
            if not isinstance(record, SessionOperationFacts):
                raise SessionCompressionIntegrityError(
                    "operation reader returned an invalid facts record"
                )
            if (
                record.operation_id != operation_id
                or record.agent_id != agent_id
                or record.session_id != session_id
            ):
                raise SessionCompressionScopeError(
                    f"session operation facts have the wrong scope: {operation_id}"
                )
            facts[operation_id] = record
        return facts

    def _validate_checkpoint(
        self,
        checkpoint: SessionCompressionCheckpoint,
        *,
        transcript: SessionTranscript,
        historical_ids: tuple[str, ...],
        messages_by_operation: Mapping[str, tuple[CanonicalMessage, ...]],
        facts: Mapping[str, SessionOperationFacts],
    ) -> None:
        if (
            checkpoint.agent_id != transcript.session.agent_id
            or checkpoint.session_id != transcript.session.id
        ):
            raise SessionCompressionScopeError("compression checkpoint scope mismatch")
        frontier = checkpoint.through_position + 1
        if frontier > len(historical_ids):
            raise SessionCompressionScopeError(
                "compression checkpoint crosses the current-operation frontier"
            )
        expected_prefix = historical_ids[:frontier]
        if (
            checkpoint.operation_ids != expected_prefix
            or checkpoint.through_operation_id != expected_prefix[-1]
        ):
            raise SessionCompressionScopeError(
                "compression checkpoint operation frontier mismatch"
            )
        expected_fingerprint = _source_fingerprint(
            expected_prefix,
            messages_by_operation,
            facts,
        )
        if checkpoint.source_fingerprint != expected_fingerprint:
            raise SessionCompressionIntegrityError(
                "compression checkpoint source fingerprint mismatch"
            )
        evidence_ids, approval_ids, resource_ids = _prefix_references(
            expected_prefix,
            facts,
        )
        if (
            checkpoint.evidence_ids != evidence_ids
            or checkpoint.approval_ids != approval_ids
            or checkpoint.resource_ids != resource_ids
        ):
            raise SessionCompressionIntegrityError(
                "compression checkpoint references no longer match its source prefix"
            )
        expected_summary = _build_summary(
            expected_prefix,
            messages_by_operation,
            facts,
            evidence_ids=evidence_ids,
            approval_ids=approval_ids,
            resource_ids=resource_ids,
            policy=self._policy,
        )
        _validate_summary(checkpoint, expected_summary=expected_summary)

    def _new_checkpoint(
        self,
        *,
        agent_id: str,
        session_id: str,
        prefix_ids: tuple[str, ...],
        messages_by_operation: Mapping[str, tuple[CanonicalMessage, ...]],
        facts: Mapping[str, SessionOperationFacts],
        previous: SessionCompressionCheckpoint | None,
    ) -> SessionCompressionCheckpoint:
        evidence_ids, approval_ids, resource_ids = _prefix_references(
            prefix_ids,
            facts,
        )
        summary = _build_summary(
            prefix_ids,
            messages_by_operation,
            facts,
            evidence_ids=evidence_ids,
            approval_ids=approval_ids,
            resource_ids=resource_ids,
            policy=self._policy,
        )
        checkpoint = SessionCompressionCheckpoint(
            id=self._id_factory("session-compression"),
            agent_id=agent_id,
            session_id=session_id,
            version=1 if previous is None else previous.version + 1,
            through_position=len(prefix_ids) - 1,
            through_operation_id=prefix_ids[-1],
            source_fingerprint=_source_fingerprint(
                prefix_ids,
                messages_by_operation,
                facts,
            ),
            summary=summary,
            operation_ids=prefix_ids,
            evidence_ids=evidence_ids,
            approval_ids=approval_ids,
            resource_ids=resource_ids,
            created_at=self._clock(),
        )
        return checkpoint


def _messages_by_operation(
    transcript: SessionTranscript,
    operation_ids: tuple[str, ...],
) -> dict[str, tuple[CanonicalMessage, ...]]:
    allowed = set(operation_ids)
    grouped: dict[str, list[CanonicalMessage]] = {
        operation_id: [] for operation_id in operation_ids
    }
    for message in transcript.messages:
        if message.operation_id in allowed:
            grouped[message.operation_id].append(message)
    return {key: tuple(values) for key, values in grouped.items()}


def _projection_blocks(
    *,
    agent_id: str,
    session_id: str,
    current_operation_id: str,
    historical_ids: tuple[str, ...],
    messages_by_operation: Mapping[str, tuple[CanonicalMessage, ...]],
    facts: Mapping[str, SessionOperationFacts],
    checkpoint: SessionCompressionCheckpoint | None,
    policy: SessionCompressionPolicy,
) -> tuple[ContextBlock, ...]:
    blocks: list[ContextBlock] = []
    first_recent_position = 0
    if checkpoint is not None:
        first_recent_position = checkpoint.through_position + 1
        summary_message = CanonicalMessage(
            agent_id=agent_id,
            operation_id=current_operation_id,
            session_id=session_id,
            role=MessageRole.USER,
            content=(TextBlock(f"UNTRUSTED_SESSION_SUMMARY={checkpoint.summary}"),),
        )
        blocks.append(
            ContextBlock(
                id="session.summary",
                owner="sessions",
                kind=ContextKind.SESSION_SUMMARY,
                trust=ContextTrust.UNTRUSTED_EXTERNAL,
                provenance=(
                    ContextProvenance(
                        kind="session.compression",
                        reference_id=checkpoint.id,
                        revision=checkpoint.source_fingerprint,
                    ),
                ),
                groups=(
                    ContextMessageGroup(
                        id="session.summary.group",
                        messages=(summary_message,),
                    ),
                ),
                priority=100,
                required=True,
            )
        )
    for position, operation_id in enumerate(
        historical_ids[first_recent_position:],
        start=first_recent_position,
    ):
        messages = messages_by_operation[operation_id]
        if not messages:
            continue
        try:
            groups = _rebound_groups(
                messages,
                agent_id=agent_id,
                session_id=session_id,
                current_operation_id=current_operation_id,
                operation_position=position,
            )
        except _IncompleteToolExchangeError:
            operation_facts = facts[operation_id]
            if operation_facts.status not in _TERMINAL_OPERATION_STATUS_VALUES:
                raise SessionCompressionIntegrityError(
                    "nonterminal session operation has an incomplete tool exchange"
                )
            groups = (
                _terminal_factual_group(
                    operation_id=operation_id,
                    operation_facts=operation_facts,
                    messages=messages,
                    agent_id=agent_id,
                    session_id=session_id,
                    current_operation_id=current_operation_id,
                    operation_position=position,
                    policy=policy,
                ),
            )
        blocks.append(
            ContextBlock(
                id=f"session.recent.{position}",
                owner="sessions",
                kind=ContextKind.SESSION_RECENT,
                trust=ContextTrust.UNTRUSTED_EXTERNAL,
                provenance=(
                    ContextProvenance(
                        kind="session.operation",
                        reference_id=operation_id,
                        revision=facts[operation_id].revision,
                    ),
                ),
                groups=groups,
                priority=1_000 + position,
                required=True,
            )
        )
    return tuple(blocks)


def _terminal_factual_group(
    *,
    operation_id: str,
    operation_facts: SessionOperationFacts,
    messages: tuple[CanonicalMessage, ...],
    agent_id: str,
    session_id: str,
    current_operation_id: str,
    operation_position: int,
    policy: SessionCompressionPolicy,
) -> ContextMessageGroup:
    operation_ids = (operation_id,)
    facts = {operation_id: operation_facts}
    messages_by_operation = {operation_id: messages}
    payload = {
        "approval_ids": operation_facts.approval_ids,
        "approvals": _approval_entries(operation_ids, facts),
        "evidence": _evidence_entries(operation_ids, facts),
        "evidence_ids": operation_facts.evidence_ids,
        "final_text": (
            None
            if operation_facts.final_text is None
            else _clip(operation_facts.final_text, policy.max_excerpt_characters)
        ),
        "kind": "terminal_session_operation_facts",
        "objective": operation_facts.objective,
        "operation_id": operation_id,
        "recent_intent": _recent_intent(
            operation_ids,
            messages_by_operation,
            maximum=policy.max_excerpt_characters,
        ),
        "resource_ids": operation_facts.resource_ids,
        "resource_scope": _resource_scope_entries(operation_ids, facts),
        "schema_version": 2,
        "status": operation_facts.status,
        "terminal_reason": operation_facts.terminal_reason,
        "trust": ContextTrust.UNTRUSTED_EXTERNAL.value,
    }
    encoded = canonical_json(payload)
    if len(encoded) > policy.max_summary_characters:
        raise SessionCompressionIntegrityError(
            "terminal session operation facts exceed the configured summary bound"
        )
    return ContextMessageGroup(
        id=f"session.group.{operation_position}.facts",
        messages=(
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=current_operation_id,
                session_id=session_id,
                role=MessageRole.USER,
                content=(TextBlock(f"UNTRUSTED_SESSION_OPERATION_FACTS={encoded}"),),
            ),
        ),
    )


def _rebound_groups(
    messages: tuple[CanonicalMessage, ...],
    *,
    agent_id: str,
    session_id: str,
    current_operation_id: str,
    operation_position: int,
) -> tuple[ContextMessageGroup, ...]:
    rebound = tuple(
        replace(
            message,
            agent_id=agent_id,
            operation_id=current_operation_id,
            session_id=session_id,
            turn_id=None,
            provider_id=None,
            provider_metadata={},
            tool_calls=tuple(
                replace(call, provider_call_id=None) for call in message.tool_calls
            ),
        )
        for message in messages
    )
    groups: list[ContextMessageGroup] = []
    index = 0
    while index < len(rebound):
        message = rebound[index]
        grouped_messages = [message]
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            expected = {call.id for call in message.tool_calls}
            observed: set[str] = set()
            index += 1
            while index < len(rebound) and rebound[index].role is MessageRole.TOOL:
                tool_message = rebound[index]
                grouped_messages.append(tool_message)
                observed.update(
                    block.call_id
                    for block in tool_message.content
                    if isinstance(block, ToolResultBlock)
                )
                index += 1
                if observed == expected:
                    break
            if observed != expected:
                raise _IncompleteToolExchangeError(
                    "session transcript contains an incomplete tool exchange"
                )
        else:
            index += 1
        try:
            groups.append(
                ContextMessageGroup(
                    id=f"session.group.{operation_position}.{len(groups)}",
                    messages=tuple(grouped_messages),
                )
            )
        except (TypeError, ValueError) as error:
            raise SessionCompressionIntegrityError(
                "session transcript cannot be grouped without splitting tool I/O"
            ) from error
    return tuple(groups)


def _source_fingerprint(
    operation_ids: tuple[str, ...],
    messages_by_operation: Mapping[str, tuple[CanonicalMessage, ...]],
    facts: Mapping[str, SessionOperationFacts],
) -> str:
    payload = {
        "operations": [
            {
                "facts": _facts_data(facts[operation_id]),
                "messages": [
                    _message_data(message)
                    for message in messages_by_operation[operation_id]
                ],
                "operation_id": operation_id,
            }
            for operation_id in operation_ids
        ],
        "schema_version": 2,
    }
    digest = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _prefix_references(
    operation_ids: tuple[str, ...],
    facts: Mapping[str, SessionOperationFacts],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        _ordered_unique(
            value
            for operation_id in operation_ids
            for value in facts[operation_id].evidence_ids
        ),
        _ordered_unique(
            value
            for operation_id in operation_ids
            for value in facts[operation_id].approval_ids
        ),
        _ordered_unique(
            value
            for operation_id in operation_ids
            for value in facts[operation_id].resource_ids
        ),
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _active_objective(
    operation_ids: tuple[str, ...],
    facts: Mapping[str, SessionOperationFacts],
) -> dict[str, str] | None:
    for operation_id in reversed(operation_ids):
        objective = facts[operation_id].objective
        if objective is not None:
            return {"operation_id": operation_id, "text": objective}
    return None


def _operation_state_entries(
    operation_ids: tuple[str, ...],
    facts: Mapping[str, SessionOperationFacts],
) -> list[dict[str, object]]:
    return [
        {
            "operation_id": operation_id,
            "status": facts[operation_id].status,
            "terminal_reason": facts[operation_id].terminal_reason,
        }
        for operation_id in operation_ids
    ]


def _approval_entries(
    operation_ids: tuple[str, ...],
    facts: Mapping[str, SessionOperationFacts],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for operation_id in operation_ids:
        operation_facts = facts[operation_id]
        states = {
            fact.approval_id: fact.status.value
            for fact in operation_facts.approval_state_facts
        }
        entries.extend(
            {
                "approval_id": approval_id,
                "operation_id": operation_id,
                "state": states.get(approval_id),
            }
            for approval_id in operation_facts.approval_ids
        )
    return entries


def _evidence_entries(
    operation_ids: tuple[str, ...],
    facts: Mapping[str, SessionOperationFacts],
) -> list[dict[str, str]]:
    return [
        {"evidence_id": evidence_id, "operation_id": operation_id}
        for operation_id in operation_ids
        for evidence_id in facts[operation_id].evidence_ids
    ]


def _resource_scope_entries(
    operation_ids: tuple[str, ...],
    facts: Mapping[str, SessionOperationFacts],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for operation_id in operation_ids:
        operation_facts = facts[operation_id]
        scopes = {
            fact.resource_id: fact for fact in operation_facts.resource_scope_facts
        }
        for resource_id in operation_facts.resource_ids:
            scope = scopes.get(resource_id)
            entries.append(
                {
                    "operation_id": operation_id,
                    "resource_id": resource_id,
                    "resource_revision": (
                        None if scope is None else scope.resource_revision
                    ),
                    "source_id": None if scope is None else scope.source_id,
                    "source_revision": None if scope is None else scope.source_revision,
                }
            )
    return entries


def _recent_intent(
    operation_ids: tuple[str, ...],
    messages_by_operation: Mapping[str, tuple[CanonicalMessage, ...]],
    *,
    maximum: int,
) -> dict[str, str] | None:
    for operation_id in reversed(operation_ids):
        for message in reversed(messages_by_operation[operation_id]):
            if message.role is not MessageRole.USER:
                continue
            texts = tuple(
                block.text for block in message.content if isinstance(block, TextBlock)
            )
            if texts:
                return {
                    "operation_id": operation_id,
                    "text": _clip("\n".join(texts), maximum),
                }
    return None


def _build_summary(
    operation_ids: tuple[str, ...],
    messages_by_operation: Mapping[str, tuple[CanonicalMessage, ...]],
    facts: Mapping[str, SessionOperationFacts],
    *,
    evidence_ids: tuple[str, ...],
    approval_ids: tuple[str, ...],
    resource_ids: tuple[str, ...],
    policy: SessionCompressionPolicy,
) -> str:
    corrections: list[dict[str, str]] = []
    for operation_id in operation_ids:
        for message in messages_by_operation[operation_id]:
            if message.role is not MessageRole.USER:
                continue
            for block in message.content:
                if isinstance(block, TextBlock) and _CORRECTION.search(block.text):
                    corrections.append(
                        {
                            "operation_id": operation_id,
                            "text": _clip(block.text, policy.max_excerpt_characters),
                        }
                    )
                    if len(corrections) > policy.max_corrections:
                        raise SessionCompressionIntegrityError(
                            "session corrections exceed the configured summary bound"
                        )

    summary: dict[str, object] = {
        "active_objective": _active_objective(operation_ids, facts),
        "approval_ids": approval_ids,
        "approvals": _approval_entries(operation_ids, facts),
        "evidence": _evidence_entries(operation_ids, facts),
        "evidence_ids": evidence_ids,
        "kind": "extractive_session_history",
        "operation_states": _operation_state_entries(operation_ids, facts),
        "operation_ids": operation_ids,
        "recent_intent": _recent_intent(
            operation_ids,
            messages_by_operation,
            maximum=policy.max_excerpt_characters,
        ),
        "resource_ids": resource_ids,
        "resource_scope": _resource_scope_entries(operation_ids, facts),
        "schema_version": 2,
        "trust": ContextTrust.UNTRUSTED_EXTERNAL.value,
        "user_corrections": corrections,
    }
    encoded = canonical_json(summary)
    if len(encoded) > policy.max_summary_characters:
        raise SessionCompressionIntegrityError(
            "required summary references exceed the configured character bound"
        )

    excerpts: list[dict[str, object]] = []
    for operation_id in operation_ids:
        operation_excerpts: list[dict[str, str]] = []
        for message in messages_by_operation[operation_id]:
            if message.role not in {MessageRole.USER, MessageRole.ASSISTANT}:
                continue
            for block in message.content:
                if isinstance(block, TextBlock):
                    operation_excerpts.append(
                        {
                            "role": message.role.value,
                            "text": _clip(block.text, policy.max_excerpt_characters),
                        }
                    )
                    break
            if len(operation_excerpts) >= 2:
                break
        final_text = facts[operation_id].final_text
        candidate: dict[str, object] = {
            "excerpts": operation_excerpts,
            "final_text": (
                None
                if final_text is None
                else _clip(
                    final_text,
                    policy.max_excerpt_characters,
                )
            ),
            "operation_id": operation_id,
            "status": facts[operation_id].status,
        }
        candidate_excerpts: list[dict[str, object]] = [*excerpts, candidate]
        candidate_summary = {**summary, "operation_excerpts": candidate_excerpts}
        candidate_encoded = canonical_json(candidate_summary)
        if len(candidate_encoded) > policy.max_summary_characters:
            break
        excerpts = candidate_excerpts
        encoded = candidate_encoded
    return encoded


def _validate_summary(
    checkpoint: SessionCompressionCheckpoint,
    *,
    expected_summary: str,
) -> None:
    try:
        decoded = json.loads(checkpoint.summary)
    except json.JSONDecodeError as error:
        raise SessionCompressionIntegrityError(
            "compression checkpoint summary is not JSON"
        ) from error
    if not isinstance(decoded, dict) or canonical_json(decoded) != checkpoint.summary:
        raise SessionCompressionIntegrityError(
            "compression checkpoint summary is not canonical JSON"
        )
    if checkpoint.summary != expected_summary:
        raise SessionCompressionIntegrityError(
            "compression checkpoint summary drifted from its fingerprinted source"
        )
    expected = {
        "operation_ids": list(checkpoint.operation_ids),
        "evidence_ids": list(checkpoint.evidence_ids),
        "approval_ids": list(checkpoint.approval_ids),
        "resource_ids": list(checkpoint.resource_ids),
    }
    if any(decoded.get(key) != value for key, value in expected.items()):
        raise SessionCompressionIntegrityError(
            "compression checkpoint summary omits required references"
        )
    if (
        decoded.get("schema_version") != 2
        or decoded.get("trust") != ContextTrust.UNTRUSTED_EXTERNAL.value
        or decoded.get("kind") != "extractive_session_history"
    ):
        raise SessionCompressionIntegrityError(
            "compression checkpoint summary metadata is invalid"
        )


def _facts_data(facts: SessionOperationFacts) -> dict[str, object]:
    return {
        "agent_id": facts.agent_id,
        "approval_ids": facts.approval_ids,
        "approval_state_facts": [
            {
                "approval_id": fact.approval_id,
                "status": fact.status.value,
            }
            for fact in facts.approval_state_facts
        ],
        "evidence_ids": facts.evidence_ids,
        "final_text": facts.final_text,
        "objective": facts.objective,
        "operation_id": facts.operation_id,
        "resource_ids": facts.resource_ids,
        "resource_scope_facts": [
            {
                "resource_id": fact.resource_id,
                "resource_revision": fact.resource_revision,
                "source_id": fact.source_id,
                "source_revision": fact.source_revision,
            }
            for fact in facts.resource_scope_facts
        ],
        "revision": facts.revision,
        "session_id": facts.session_id,
        "status": facts.status,
        "terminal_reason": facts.terminal_reason,
    }


def _message_data(message: CanonicalMessage) -> dict[str, object]:
    content: list[dict[str, object]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append({"text": block.text, "type": "text"})
        elif isinstance(block, ToolResultBlock):
            content.append(
                {
                    "call_id": block.call_id,
                    "is_error": block.is_error,
                    "output": block.output,
                    "type": "tool_result",
                }
            )
    return {
        "agent_id": message.agent_id,
        "content": content,
        "operation_id": message.operation_id,
        "provider_id": message.provider_id,
        "provider_metadata": message.provider_metadata,
        "role": message.role.value,
        "session_id": message.session_id,
        "tool_calls": [
            {
                "arguments": call.arguments,
                "id": call.id,
                "name": call.name,
                "provider_call_id": call.provider_call_id,
            }
            for call in message.tool_calls
        ],
        "turn_id": message.turn_id,
    }


def _clip(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    marker = "…[truncated]"
    if maximum <= len(marker):
        return marker[:maximum]
    return value[: maximum - len(marker)] + marker


__all__ = [
    "SessionCompressionCheckpointCommitter",
    "SessionCompressionCheckpointReader",
    "SessionCompressionError",
    "SessionCompressionIntegrityError",
    "SessionCompressionPolicy",
    "SessionCompressionScopeError",
    "SessionCompressionService",
    "SessionContextProjection",
    "SessionApprovalStateFact",
    "SessionOperationFacts",
    "SessionOperationFactsReader",
    "SessionResourceScopeFact",
    "SessionTranscriptReader",
]
