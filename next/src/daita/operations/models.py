"""Canonical operation records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import re

from .._json import FrozenJsonObject, canonical_json
from ..capabilities import AccessMode, RiskLevel

_CANONICAL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_VALIDATION_ITEMS = 256
_MAX_IMPACT_CHARACTERS = 16_384


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class OperationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_INPUT = "waiting_for_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TriggerKind(str, Enum):
    USER = "user"
    SCHEDULE = "schedule"
    MONITOR = "monitor"
    EVENT = "event"
    INTERNAL = "internal"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    CLAIMED = "claimed"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"


_TERMINAL_STATUSES = {
    OperationStatus.SUCCEEDED,
    OperationStatus.FAILED,
    OperationStatus.CANCELLED,
    OperationStatus.INTERRUPTED,
}


@dataclass(frozen=True, slots=True)
class AgentTrigger:
    id: str
    agent_id: str
    kind: TriggerKind
    source_id: str
    payload: Mapping[str, object]
    created_at: datetime
    session_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "trigger id")
        _required_text(self.agent_id, "trigger agent_id")
        _required_text(self.source_id, "trigger source_id")
        if not isinstance(self.kind, TriggerKind):
            raise TypeError("trigger kind must be a TriggerKind")
        if self.session_id is not None:
            _required_text(self.session_id, "trigger session_id")
        _aware(self.created_at, "trigger created_at")
        object.__setattr__(self, "payload", FrozenJsonObject.from_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class Operation:
    id: str
    agent_id: str
    trigger_id: str
    status: OperationStatus
    created_at: datetime
    updated_at: datetime
    session_id: str | None = None
    final_text: str | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "operation id")
        _required_text(self.agent_id, "operation agent_id")
        _required_text(self.trigger_id, "operation trigger_id")
        if not isinstance(self.status, OperationStatus):
            raise TypeError("operation status must be an OperationStatus")
        _aware(self.created_at, "operation created_at")
        _aware(self.updated_at, "operation updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("operation updated_at cannot precede created_at")
        if self.session_id is not None:
            _required_text(self.session_id, "operation session_id")
        if self.final_text is not None:
            _required_text(self.final_text, "operation final text")
        if self.terminal_reason is not None:
            _required_text(self.terminal_reason, "operation terminal reason")

        is_terminal = self.status in _TERMINAL_STATUSES
        if is_terminal and self.terminal_reason is None:
            raise ValueError("terminal operation requires a terminal reason")
        if not is_terminal and self.terminal_reason is not None:
            raise ValueError("nonterminal operation cannot have a terminal reason")
        if self.status is OperationStatus.SUCCEEDED and self.final_text is None:
            raise ValueError("succeeded operation requires final text")
        if not is_terminal and self.final_text is not None:
            raise ValueError("only a terminal operation may store final text")


@dataclass(frozen=True, slots=True)
class ActionValidationFacts:
    """Validator-owned authority bound to one proposed and materialized action."""

    schema_version: int = 0
    validation_passed: bool = True
    in_scope: bool = True
    destructive: bool = False
    sensitivity_class: str = "internal"
    source_id: str | None = None
    resource_ids: tuple[str, ...] = ()
    resource_revisions: tuple[tuple[str, str], ...] = ()
    source_revision: str | None = None
    impact: Mapping[str, object] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version not in {0, 1}
        ):
            raise ValueError("validation schema_version must be 0 or 1")
        for field_name, value in (
            ("validation_passed", self.validation_passed),
            ("in_scope", self.in_scope),
            ("destructive", self.destructive),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"validation {field_name} must be a boolean")

        _required_text(self.sensitivity_class, "validation sensitivity_class")
        if (
            self.sensitivity_class != self.sensitivity_class.strip()
            or len(self.sensitivity_class) > 128
        ):
            raise ValueError(
                "validation sensitivity_class must be bounded text without "
                "surrounding whitespace"
            )
        if self.source_id is not None:
            _required_text(self.source_id, "validation source_id")
            if self.source_id != self.source_id.strip() or len(self.source_id) > 512:
                raise ValueError(
                    "validation source_id must be bounded text without surrounding "
                    "whitespace"
                )
        if self.source_revision is not None:
            _required_text(self.source_revision, "validation source_revision")
            if (
                self.source_revision != self.source_revision.strip()
                or len(self.source_revision) > 1_024
            ):
                raise ValueError(
                    "validation source_revision must be bounded text without "
                    "surrounding whitespace"
                )

        resource_ids = self._text_tuple(
            self.resource_ids,
            "resource_ids",
            maximum=_MAX_VALIDATION_ITEMS,
        )
        evidence_ids = self._text_tuple(
            self.evidence_ids,
            "evidence_ids",
            maximum=_MAX_VALIDATION_ITEMS,
        )
        if isinstance(self.resource_revisions, (str, bytes)):
            raise TypeError("validation resource_revisions must be a sequence")
        resource_revisions = tuple(tuple(item) for item in self.resource_revisions)
        if len(resource_revisions) > _MAX_VALIDATION_ITEMS:
            raise ValueError("validation resource_revisions exceed the item limit")
        for item in resource_revisions:
            if len(item) != 2:
                raise ValueError(
                    "validation resource_revisions must contain identifier/revision "
                    "pairs"
                )
            resource_id, revision = item
            _required_text(resource_id, "validation resource revision id")
            if resource_id not in resource_ids:
                raise ValueError(
                    "validation resource revision references an unscoped resource"
                )
            if (
                not isinstance(revision, str)
                or _CANONICAL_SHA256.fullmatch(revision) is None
            ):
                raise ValueError(
                    "validation resource revision must be a canonical lowercase "
                    "sha256 hash"
                )
        resource_revisions = tuple(sorted(resource_revisions))
        revision_ids = tuple(item[0] for item in resource_revisions)
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("validation resource revisions must be unique")
        if resource_revisions and set(revision_ids) != set(resource_ids):
            raise ValueError(
                "validation resource revisions must cover every scoped resource"
            )

        impact = FrozenJsonObject.from_mapping(self.impact)
        if len(canonical_json(impact)) > _MAX_IMPACT_CHARACTERS:
            raise ValueError("validation impact facts must be bounded")

        object.__setattr__(self, "resource_ids", resource_ids)
        object.__setattr__(self, "resource_revisions", resource_revisions)
        object.__setattr__(self, "impact", impact)
        object.__setattr__(self, "evidence_ids", evidence_ids)

        if self.schema_version == 0:
            if (
                not self.validation_passed
                or not self.in_scope
                or self.destructive
                or self.sensitivity_class != "internal"
                or self.source_id is not None
                or resource_ids
                or resource_revisions
                or self.source_revision is not None
                or impact
                or evidence_ids
            ):
                raise ValueError(
                    "legacy validation facts cannot contain explicit authority"
                )
        elif self.source_id is None:
            raise ValueError("explicit validation facts require source_id")

    @staticmethod
    def _text_tuple(
        values: tuple[str, ...],
        field_name: str,
        *,
        maximum: int,
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"validation {field_name} must be a sequence of strings")
        normalized = tuple(values)
        if len(normalized) > maximum:
            raise ValueError(f"validation {field_name} exceed the item limit")
        for value in normalized:
            _required_text(value, f"validation {field_name} item")
            if value != value.strip() or len(value) > 512:
                raise ValueError(
                    f"validation {field_name} must contain bounded text without "
                    "surrounding whitespace"
                )
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"validation {field_name} must be unique")
        return normalized

    @property
    def fingerprint(self) -> str | None:
        """Return no extra hash for legacy facts, preserving v12 approvals."""

        if self.schema_version == 0:
            return None
        material = {
            "destructive": self.destructive,
            "evidence_ids": self.evidence_ids,
            "impact": self.impact,
            "in_scope": self.in_scope,
            "resource_ids": self.resource_ids,
            "resource_revisions": self.resource_revisions,
            "schema_version": self.schema_version,
            "sensitivity_class": self.sensitivity_class,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "validation_passed": self.validation_passed,
        }
        encoded = canonical_json(material).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def audit_projection(self) -> dict[str, object]:
        """Return a bounded argument-free projection for audit and inspection."""

        return {
            "schema_version": self.schema_version,
            "validation_passed": self.validation_passed,
            "in_scope": self.in_scope,
            "destructive": self.destructive,
            "sensitivity_class": self.sensitivity_class,
            "source_id": self.source_id,
            "resource_ids": self.resource_ids,
            "resource_revisions": self.resource_revisions,
            "source_revision": self.source_revision,
            "impact": self.impact,
            "evidence_ids": self.evidence_ids,
            "validation_fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ActionProposal:
    operation_id: str
    turn_id: str
    call_id: str
    capability_id: str
    proposed_at: datetime
    arguments: Mapping[str, object] = field(default_factory=dict)
    validation_facts: ActionValidationFacts = field(
        default_factory=ActionValidationFacts
    )

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "proposal operation_id")
        _required_text(self.turn_id, "proposal turn_id")
        _required_text(self.call_id, "proposal call_id")
        _required_text(self.capability_id, "proposal capability_id")
        _aware(self.proposed_at, "proposal proposed_at")
        if not isinstance(self.validation_facts, ActionValidationFacts):
            raise TypeError(
                "proposal validation_facts must be an ActionValidationFacts record"
            )
        object.__setattr__(
            self, "arguments", FrozenJsonObject.from_mapping(self.arguments)
        )


@dataclass(frozen=True, slots=True)
class ActionRejection:
    """A bounded, provider-neutral semantic rejection returned to the loop."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.code, "action rejection code")
        _required_text(self.message, "action rejection message")
        if len(self.code) > 128 or len(self.message) > 512:
            raise ValueError("action rejection text must be bounded")
        details = FrozenJsonObject.from_mapping(self.details)
        if len(canonical_json(details)) > 2048:
            raise ValueError("action rejection details must be bounded")
        object.__setattr__(self, "details", details)


@dataclass(frozen=True, slots=True)
class Observation:
    operation_id: str
    turn_id: str
    code: str
    message: str
    payload: Mapping[str, object]
    success: bool
    created_at: datetime
    call_id: str | None = None
    task_id: str | None = None
    evidence_id: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "observation operation_id")
        _required_text(self.turn_id, "observation turn_id")
        _required_text(self.code, "observation code")
        _required_text(self.message, "observation message")
        if not isinstance(self.success, bool):
            raise TypeError("observation success must be a boolean")
        if not isinstance(self.truncated, bool):
            raise TypeError("observation truncated must be a boolean")
        _aware(self.created_at, "observation created_at")
        if self.call_id is not None:
            _required_text(self.call_id, "observation call_id")
        if self.task_id is not None:
            _required_text(self.task_id, "observation task_id")
        if self.evidence_id is not None:
            _required_text(self.evidence_id, "observation evidence_id")
        if self.evidence_id is not None and self.task_id is None:
            raise ValueError("observation evidence_id requires task_id")
        if self.success and self.task_id is not None and self.evidence_id is None:
            raise ValueError("successful task observation requires evidence_id")
        object.__setattr__(self, "payload", FrozenJsonObject.from_mapping(self.payload))


_LEGACY_ZERO_HASH = "sha256:" + ("0" * 64)


@dataclass(frozen=True, slots=True)
class TaskExecutionFacts:
    """Immutable safety facts captured when a task is materialized."""

    capability_fingerprint: str
    arguments_hash: str
    access_mode: AccessMode
    risk: RiskLevel
    side_effecting: bool
    idempotent: bool
    replay_safe: bool
    idempotency_key: str | None = None
    validation_facts: ActionValidationFacts = field(
        default_factory=ActionValidationFacts
    )

    def __post_init__(self) -> None:
        for hash_name, hash_value in (
            ("capability_fingerprint", self.capability_fingerprint),
            ("arguments_hash", self.arguments_hash),
        ):
            if (
                not isinstance(hash_value, str)
                or _CANONICAL_SHA256.fullmatch(hash_value) is None
            ):
                raise ValueError(
                    f"{hash_name} must be a canonical lowercase sha256 hash"
                )
        if not isinstance(self.access_mode, AccessMode):
            raise TypeError("access_mode must be an AccessMode")
        if not isinstance(self.risk, RiskLevel):
            raise TypeError("risk must be a RiskLevel")
        if not isinstance(self.validation_facts, ActionValidationFacts):
            raise TypeError("validation_facts must be an ActionValidationFacts record")
        for flag_name, flag_value in (
            ("side_effecting", self.side_effecting),
            ("idempotent", self.idempotent),
            ("replay_safe", self.replay_safe),
        ):
            if not isinstance(flag_value, bool):
                raise TypeError(f"{flag_name} must be a boolean")

        if self.access_mode is AccessMode.READ and self.side_effecting:
            raise ValueError("read execution facts cannot declare a side effect")
        if self.replay_safe and not self.idempotent:
            raise ValueError("replay_safe execution facts must be idempotent")

        if self.idempotency_key is not None:
            if (
                not isinstance(self.idempotency_key, str)
                or not self.idempotency_key.strip()
            ):
                raise ValueError("idempotency_key must be a non-empty string")
            if not self.side_effecting or not self.idempotent:
                raise ValueError(
                    "idempotency_key is only valid for idempotent side effects"
                )
        if self.side_effecting and self.replay_safe and self.idempotency_key is None:
            raise ValueError("replay-safe side effects require an idempotency_key")
        if self.validation_facts.destructive and (
            self.access_mode is not AccessMode.WRITE or not self.side_effecting
        ):
            raise ValueError(
                "destructive validation facts require a side-effecting write"
            )


def _legacy_execution_facts() -> TaskExecutionFacts:
    """Fail-closed defaults for records written before execution facts existed."""

    return TaskExecutionFacts(
        capability_fingerprint=_LEGACY_ZERO_HASH,
        arguments_hash=_LEGACY_ZERO_HASH,
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
    )


@dataclass(frozen=True, slots=True)
class TaskDependency:
    operation_id: str
    task_id: str
    prerequisite_task_id: str

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "dependency operation_id")
        _required_text(self.task_id, "dependency task_id")
        _required_text(
            self.prerequisite_task_id,
            "dependency prerequisite_task_id",
        )
        if self.task_id == self.prerequisite_task_id:
            raise ValueError("task cannot depend on itself")


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    operation_id: str
    turn_id: str
    call_id: str
    capability_id: str
    executor_id: str
    status: TaskStatus
    attempt: int
    arguments: Mapping[str, object]
    created_at: datetime
    updated_at: datetime
    execution_facts: TaskExecutionFacts = field(default_factory=_legacy_execution_facts)
    evidence_ids: tuple[str, ...] = ()
    error_code: str | None = None
    cancellation_requested: bool = False
    manual_recovery_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "task id")
        _required_text(self.operation_id, "task operation_id")
        _required_text(self.turn_id, "task turn_id")
        _required_text(self.call_id, "task call_id")
        _required_text(self.capability_id, "task capability_id")
        _required_text(self.executor_id, "task executor_id")
        if not isinstance(self.status, TaskStatus):
            raise TypeError("task status must be a TaskStatus")
        if not isinstance(self.execution_facts, TaskExecutionFacts):
            raise TypeError("task execution_facts must be a TaskExecutionFacts record")
        arguments = FrozenJsonObject.from_mapping(self.arguments)
        if self.execution_facts.arguments_hash != _LEGACY_ZERO_HASH:
            actual_arguments_hash = (
                "sha256:"
                + hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest()
            )
            if self.execution_facts.arguments_hash != actual_arguments_hash:
                raise ValueError(
                    "task execution_facts arguments_hash does not match arguments"
                )
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 1
        ):
            raise ValueError("task attempt must be a positive integer")
        _aware(self.created_at, "task created_at")
        _aware(self.updated_at, "task updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("task updated_at cannot precede created_at")
        if not isinstance(self.cancellation_requested, bool):
            raise TypeError("task cancellation_requested must be a boolean")
        if self.error_code is not None:
            _required_text(self.error_code, "task error_code")
        if self.status is TaskStatus.FAILED and self.error_code is None:
            raise ValueError("failed task requires error_code")
        if self.status is not TaskStatus.FAILED and self.error_code is not None:
            raise ValueError("only failed task may contain error_code")
        if self.status is TaskStatus.MANUAL_RECOVERY_REQUIRED:
            if (
                not isinstance(self.manual_recovery_reason, str)
                or not self.manual_recovery_reason.strip()
            ):
                raise ValueError(
                    "manual recovery task requires a manual recovery reason"
                )
        elif self.manual_recovery_reason is not None:
            raise ValueError(
                "only manual recovery tasks may contain a manual recovery reason"
            )
        evidence_ids = tuple(self.evidence_ids)
        if any(
            not isinstance(evidence_id, str) or not evidence_id.strip()
            for evidence_id in evidence_ids
        ):
            raise ValueError("task evidence_ids must contain non-empty strings")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("task evidence_ids must be unique")
        if self.status is TaskStatus.SUCCEEDED and not evidence_ids:
            raise ValueError("succeeded task requires accepted evidence")
        if self.status is not TaskStatus.SUCCEEDED and evidence_ids:
            raise ValueError("only succeeded tasks may link accepted evidence")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(
            self,
            "arguments",
            arguments,
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    operation_id: str
    task_id: str
    turn_id: str
    capability_id: str
    executor_id: str
    kind: str
    schema_version: int
    attempt: int
    accepted: bool
    payload: Mapping[str, object]
    content_hash: str
    created_at: datetime
    blob_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "evidence id")
        _required_text(self.operation_id, "evidence operation_id")
        _required_text(self.task_id, "evidence task_id")
        _required_text(self.turn_id, "evidence turn_id")
        _required_text(self.capability_id, "evidence capability_id")
        _required_text(self.executor_id, "evidence executor_id")
        _required_text(self.kind, "evidence kind")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version < 1
        ):
            raise ValueError("evidence schema_version must be a positive integer")
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 1
        ):
            raise ValueError("evidence attempt must be a positive integer")
        if not isinstance(self.accepted, bool):
            raise TypeError("evidence accepted must be a boolean")
        _required_text(self.content_hash, "evidence content_hash")
        if not self.content_hash.startswith("sha256:"):
            raise ValueError("evidence content_hash must use sha256")
        if self.blob_id is not None:
            _required_text(self.blob_id, "evidence blob_id")
        _aware(self.created_at, "evidence created_at")
        object.__setattr__(self, "payload", FrozenJsonObject.from_mapping(self.payload))
