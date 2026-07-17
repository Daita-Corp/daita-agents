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
class ActionProposal:
    operation_id: str
    turn_id: str
    call_id: str
    capability_id: str
    proposed_at: datetime
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "proposal operation_id")
        _required_text(self.turn_id, "proposal turn_id")
        _required_text(self.call_id, "proposal call_id")
        _required_text(self.capability_id, "proposal capability_id")
        _aware(self.proposed_at, "proposal proposed_at")
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


_CANONICAL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
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
        _aware(self.created_at, "evidence created_at")
        object.__setattr__(self, "payload", FrozenJsonObject.from_mapping(self.payload))
