"""Canonical operation records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .._json import FrozenJsonObject


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
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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
class Observation:
    operation_id: str
    turn_id: str
    code: str
    message: str
    payload: Mapping[str, object]
    success: bool
    created_at: datetime
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
        if self.task_id is not None:
            _required_text(self.task_id, "observation task_id")
        if self.evidence_id is not None:
            _required_text(self.evidence_id, "observation evidence_id")
        if self.evidence_id is not None and self.task_id is None:
            raise ValueError("observation evidence_id requires task_id")
        if self.success and self.task_id is not None and self.evidence_id is None:
            raise ValueError("successful task observation requires evidence_id")
        object.__setattr__(self, "payload", FrozenJsonObject.from_mapping(self.payload))


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
    evidence_ids: tuple[str, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "task id")
        _required_text(self.operation_id, "task operation_id")
        _required_text(self.turn_id, "task turn_id")
        _required_text(self.call_id, "task call_id")
        _required_text(self.capability_id, "task capability_id")
        _required_text(self.executor_id, "task executor_id")
        if not isinstance(self.status, TaskStatus):
            raise TypeError("task status must be a TaskStatus")
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
        if self.error_code is not None:
            _required_text(self.error_code, "task error_code")
        if self.status is TaskStatus.FAILED and self.error_code is None:
            raise ValueError("failed task requires error_code")
        if self.status is not TaskStatus.FAILED and self.error_code is not None:
            raise ValueError("only failed task may contain error_code")
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
            FrozenJsonObject.from_mapping(self.arguments),
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
