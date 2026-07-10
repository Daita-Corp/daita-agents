"""
Framework-level runtime primitives.

These records are intentionally domain-neutral. They describe capabilities,
tasks, evidence, context, workers, operations, and events in a form that can be
shared by the generic chat runtime, the upcoming DB runtime, and future
focused runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
import time
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

_DOTTED_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AccessMode(str, Enum):
    """The level of access a capability or operation requires."""

    NONE = "none"
    METADATA_READ = "metadata_read"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class RiskLevel(str, Enum):
    """The expected blast radius of using a capability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContextAudience(str, Enum):
    """The intended consumer for a rendered context block."""

    PRIMARY_MODEL = "primary_model"
    FINAL_SYNTHESIZER = "final_synthesizer"
    HUMAN_REVIEWER = "human_reviewer"
    OPERATION_INSPECTOR = "operation_inspector"


class TaskStatus(str, Enum):
    """Lifecycle status for a runtime task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class OperationStatus(str, Enum):
    """Lifecycle status for a runtime operation."""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class RuntimeEventType(str, Enum):
    """Coarse event categories emitted by runtimes."""

    OPERATION_CREATED = "operation.created"
    OPERATION_UPDATED = "operation.updated"
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    EVIDENCE_ACCEPTED = "evidence.accepted"
    POLICY_DECISION = "policy.decision"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_UPDATED = "approval.updated"
    LLM_REQUESTED = "llm.requested"
    LLM_COMPLETED = "llm.completed"
    EXECUTOR_STARTED = "executor.started"
    EXECUTOR_COMPLETED = "executor.completed"
    EXECUTOR_FAILED = "executor.failed"
    MONITOR_TICKED = "monitor.ticked"
    MONITOR_TRIGGERED = "monitor.triggered"
    MONITOR_SKIPPED = "monitor.skipped"
    WORKER_HANDOFF = "worker.handoff"
    WORKER_DELEGATED = "worker.delegated"
    WORKER_LEASE_CLAIMED = "worker.lease_claimed"
    WORKER_HEARTBEAT = "worker.heartbeat"
    WORKER_COMPLETED = "worker.completed"
    WORKER_FAILED = "worker.failed"
    WORKER_TIMEOUT = "worker.timeout"
    WORKER_CANCELLED = "worker.cancelled"
    OPERATION_RESUMED = "operation.resumed"
    TASK_SKIPPED = "task.skipped"
    DIAGNOSTIC = "diagnostic"
    ERROR = "error"


class PolicyEffect(str, Enum):
    """Possible outcomes from evaluating one governance policy."""

    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    REQUIRE_APPROVAL = "require_approval"
    WARN = "warn"


class ApprovalStatus(str, Enum):
    """Lifecycle state for an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskDependencyKind(str, Enum):
    """Kinds of predicates that can make a persisted task runnable."""

    EVIDENCE = "evidence"
    APPROVAL = "approval"


def _validate_dotted_id(value: str, field_name: str) -> None:
    if not _DOTTED_ID_RE.match(value):
        raise ValueError(
            f"{field_name} must be a lowercase dotted identifier "
            "(for example 'db.sql.execute_read')"
        )


def _validate_tool_name(value: str, field_name: str) -> None:
    if not _NAME_RE.match(value):
        raise ValueError(
            f"{field_name} must be a valid tool-style name "
            "(letters, numbers, and underscores)"
        )


def _frozen_strings(
    values: frozenset[str] | set[str] | tuple[str, ...] | list[str],
) -> frozenset[str]:
    frozen = frozenset(values)
    for value in frozen:
        if not isinstance(value, str):
            raise TypeError("collection values must be strings")
    return frozen


def _dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError("runtime primitive mappings must be JSON serializable") from exc
    return copied


@dataclass(frozen=True)
class Capability:
    """Runtime-plannable behavior declared by a plugin or runtime extension."""

    id: str
    owner: str
    description: str
    domains: frozenset[str]
    operation_types: frozenset[str]
    access: AccessMode
    risk: RiskLevel
    input_schema: dict[str, Any]
    output_evidence: frozenset[str]
    executor: str
    model_visible: bool = False
    runtime_only: bool = False
    retry_safe: bool = False
    replay_safe: bool = False
    idempotent: bool = False
    side_effecting: bool = True
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.id, "capability id")
        _validate_dotted_id(self.owner, "capability owner")
        _validate_dotted_id(self.executor, "capability executor")
        object.__setattr__(self, "domains", _frozen_strings(self.domains))
        object.__setattr__(
            self, "operation_types", _frozen_strings(self.operation_types)
        )
        object.__setattr__(
            self, "output_evidence", _frozen_strings(self.output_evidence)
        )
        object.__setattr__(self, "input_schema", _dict(self.input_schema))
        object.__setattr__(self, "metadata", _dict(self.metadata))
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner": self.owner,
            "description": self.description,
            "domains": sorted(self.domains),
            "operation_types": sorted(self.operation_types),
            "access": self.access.value,
            "risk": self.risk.value,
            "input_schema": self.input_schema,
            "output_evidence": sorted(self.output_evidence),
            "executor": self.executor,
            "model_visible": self.model_visible,
            "runtime_only": self.runtime_only,
            "retry_safe": self.retry_safe,
            "replay_safe": self.replay_safe,
            "idempotent": self.idempotent,
            "side_effecting": self.side_effecting,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Capability":
        values = dict(data)
        values["access"] = AccessMode(values["access"])
        values["risk"] = RiskLevel(values["risk"])
        values["domains"] = frozenset(values.get("domains", ()))
        values["operation_types"] = frozenset(values.get("operation_types", ()))
        values["output_evidence"] = frozenset(values.get("output_evidence", ()))
        return cls(**values)


@dataclass(frozen=True)
class EvidenceSchema:
    """Declared schema for evidence that executors can produce."""

    kind: str
    owner: str
    json_schema: dict[str, Any]
    description: str = ""

    def __post_init__(self) -> None:
        _validate_dotted_id(self.kind, "evidence schema kind")
        _validate_dotted_id(self.owner, "evidence schema owner")
        object.__setattr__(self, "json_schema", _dict(self.json_schema))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "owner": self.owner,
            "json_schema": self.json_schema,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceSchema":
        return cls(**dict(data))


@dataclass(frozen=True)
class Evidence:
    """Typed runtime output accepted from an executor, worker, or policy."""

    kind: str
    payload: dict[str, Any]
    id: str | None = None
    owner: str | None = None
    operation_id: str | None = None
    task_id: str | None = None
    schema_version: str | None = None
    accepted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.kind, "evidence kind")
        if self.owner is not None:
            _validate_dotted_id(self.owner, "evidence owner")
        object.__setattr__(self, "payload", _dict(self.payload))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "id": self.id,
            "owner": self.owner,
            "operation_id": self.operation_id,
            "task_id": self.task_id,
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evidence":
        return cls(**dict(data))


@dataclass(frozen=True)
class ContextBlock:
    """Bounded context rendered for a specific runtime audience."""

    id: str
    owner: str
    audience: ContextAudience
    content: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.id, "context block id")
        _validate_dotted_id(self.owner, "context block owner")
        object.__setattr__(self, "audience", ContextAudience(self.audience))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner": self.owner,
            "audience": self.audience.value,
            "content": self.content,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextBlock":
        values = dict(data)
        values["audience"] = ContextAudience(values["audience"])
        return cls(**values)


@dataclass(frozen=True)
class ToolView:
    """Model-visible presentation of a capability."""

    name: str
    capability_id: str
    description: str
    parameters: dict[str, Any]
    model_visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_tool_name(self.name, "tool view name")
        _validate_dotted_id(self.capability_id, "tool view capability_id")
        object.__setattr__(self, "parameters", _dict(self.parameters))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capability_id": self.capability_id,
            "description": self.description,
            "parameters": self.parameters,
            "model_visible": self.model_visible,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolView":
        return cls(**dict(data))


@dataclass(frozen=True)
class Worker:
    """Specialist or background worker declaration."""

    id: str
    owner: str
    role: str
    capability_ids: frozenset[str]
    input_schema: dict[str, Any]
    output_evidence: frozenset[str]
    max_concurrency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.id, "worker id")
        _validate_dotted_id(self.owner, "worker owner")
        object.__setattr__(self, "capability_ids", _frozen_strings(self.capability_ids))
        for capability_id in self.capability_ids:
            _validate_dotted_id(capability_id, "worker capability_id")
        object.__setattr__(
            self, "output_evidence", _frozen_strings(self.output_evidence)
        )
        object.__setattr__(self, "input_schema", _dict(self.input_schema))
        object.__setattr__(self, "metadata", _dict(self.metadata))
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner": self.owner,
            "role": self.role,
            "capability_ids": sorted(self.capability_ids),
            "input_schema": self.input_schema,
            "output_evidence": sorted(self.output_evidence),
            "max_concurrency": self.max_concurrency,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Worker":
        values = dict(data)
        values["capability_ids"] = frozenset(values.get("capability_ids", ()))
        values["output_evidence"] = frozenset(values.get("output_evidence", ()))
        return cls(**values)


@dataclass(frozen=True)
class Operation:
    """Runtime operation state shared by specialized runtimes."""

    id: str
    operation_type: str
    status: OperationStatus = OperationStatus.PENDING
    request: dict[str, Any] = field(default_factory=dict)
    required_evidence: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.operation_type, "operation_type")
        object.__setattr__(self, "status", OperationStatus(self.status))
        object.__setattr__(
            self, "required_evidence", _frozen_strings(self.required_evidence)
        )
        object.__setattr__(self, "request", _dict(self.request))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation_type": self.operation_type,
            "status": self.status.value,
            "request": self.request,
            "required_evidence": sorted(self.required_evidence),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Operation":
        values = dict(data)
        values["status"] = OperationStatus(
            values.get("status", OperationStatus.PENDING)
        )
        values["required_evidence"] = frozenset(values.get("required_evidence", ()))
        return cls(**values)


@dataclass(frozen=True)
class TaskDependency:
    """Predicate that must be satisfied before a task is runnable."""

    kind: TaskDependencyKind | Literal["evidence", "approval"]
    evidence_kind: str | None = None
    evidence_id: str | None = None
    evidence_owner: str | None = None
    producer_task_id: str | None = None
    producer_capability_id: str | None = None
    producer_executor_id: str | None = None
    evidence_payload: dict[str, Any] = field(default_factory=dict)
    evidence_accepted: bool = True
    input_hash: str | None = None
    payload_fingerprint: str | None = None
    approval_id: str | None = None
    approval_policy_id: str | None = None
    approval_name: str | None = None
    approval_version: int | None = None
    approval_status: (
        ApprovalStatus
        | Literal["pending", "approved", "rejected", "cancelled", "expired"]
        | None
    ) = None
    operation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TaskDependencyKind(self.kind))
        if self.evidence_kind is not None:
            _validate_dotted_id(self.evidence_kind, "dependency evidence_kind")
        if self.evidence_owner is not None:
            _validate_dotted_id(self.evidence_owner, "dependency evidence_owner")
        if self.producer_capability_id is not None:
            _validate_dotted_id(
                self.producer_capability_id, "dependency producer_capability_id"
            )
        if self.producer_executor_id is not None:
            _validate_dotted_id(
                self.producer_executor_id, "dependency producer_executor_id"
            )
        if self.approval_policy_id is not None:
            _validate_dotted_id(
                self.approval_policy_id, "dependency approval_policy_id"
            )
        if self.approval_status is not None:
            object.__setattr__(
                self, "approval_status", ApprovalStatus(self.approval_status)
            )
        object.__setattr__(self, "evidence_payload", _dict(self.evidence_payload))
        object.__setattr__(self, "metadata", _dict(self.metadata))
        if self.kind is TaskDependencyKind.EVIDENCE and self.evidence_kind is None:
            raise ValueError("evidence dependencies require evidence_kind")
        if self.kind is TaskDependencyKind.APPROVAL and self.approval_status is None:
            raise ValueError("approval dependencies require approval_status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": (
                self.kind.value
                if isinstance(self.kind, TaskDependencyKind)
                else self.kind
            ),
            "evidence_kind": self.evidence_kind,
            "evidence_id": self.evidence_id,
            "evidence_owner": self.evidence_owner,
            "producer_task_id": self.producer_task_id,
            "producer_capability_id": self.producer_capability_id,
            "producer_executor_id": self.producer_executor_id,
            "evidence_payload": self.evidence_payload,
            "evidence_accepted": self.evidence_accepted,
            "input_hash": self.input_hash,
            "payload_fingerprint": self.payload_fingerprint,
            "approval_id": self.approval_id,
            "approval_policy_id": self.approval_policy_id,
            "approval_name": self.approval_name,
            "approval_version": self.approval_version,
            "approval_status": (
                self.approval_status.value
                if isinstance(self.approval_status, ApprovalStatus)
                else self.approval_status
            ),
            "operation_id": self.operation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskDependency":
        values = dict(data)
        values["kind"] = TaskDependencyKind(values["kind"])
        if values.get("approval_status") is not None:
            values["approval_status"] = ApprovalStatus(values["approval_status"])
        values["evidence_payload"] = dict(values.get("evidence_payload") or {})
        return cls(**values)


@dataclass(frozen=True)
class Task:
    """A single unit of work assigned to an executor or worker."""

    id: str
    operation_id: str
    capability_id: str
    executor_id: str
    input: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    required_evidence: frozenset[str] = field(default_factory=frozenset)
    dependencies: tuple[TaskDependency, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.capability_id, "task capability_id")
        _validate_dotted_id(self.executor_id, "task executor_id")
        object.__setattr__(self, "status", TaskStatus(self.status))
        object.__setattr__(
            self, "required_evidence", _frozen_strings(self.required_evidence)
        )
        dependencies = tuple(
            (
                dependency
                if isinstance(dependency, TaskDependency)
                else TaskDependency.from_dict(dependency)
            )
            for dependency in self.dependencies
        )
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "input", _dict(self.input))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation_id": self.operation_id,
            "capability_id": self.capability_id,
            "executor_id": self.executor_id,
            "input": self.input,
            "status": self.status.value,
            "required_evidence": sorted(self.required_evidence),
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        values = dict(data)
        values["status"] = TaskStatus(values.get("status", TaskStatus.PENDING))
        values["required_evidence"] = frozenset(values.get("required_evidence", ()))
        values["dependencies"] = tuple(
            TaskDependency.from_dict(item) for item in values.get("dependencies", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class RuntimeEvent:
    """Serializable event emitted during runtime execution."""

    type: RuntimeEventType
    operation_id: str
    message: str
    id: str | None = None
    runtime_id: str | None = None
    runtime_kind: str | None = None
    task_id: str | None = None
    capability_id: str | None = None
    executor_id: str | None = None
    plugin_id: str | None = None
    policy_id: str | None = None
    approval_id: str | None = None
    evidence_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", RuntimeEventType(self.type))
        object.__setattr__(self, "payload", _dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "operation_id": self.operation_id,
            "message": self.message,
            "id": self.id,
            "runtime_id": self.runtime_id,
            "runtime_kind": self.runtime_kind,
            "task_id": self.task_id,
            "capability_id": self.capability_id,
            "executor_id": self.executor_id,
            "plugin_id": self.plugin_id,
            "policy_id": self.policy_id,
            "approval_id": self.approval_id,
            "evidence_id": self.evidence_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeEvent":
        values = dict(data)
        values["type"] = RuntimeEventType(values["type"])
        return cls(**values)


@dataclass(frozen=True)
class PolicyDecision:
    """A first-class runtime decision produced by one policy."""

    policy_id: str
    owner: str
    effect: (
        PolicyEffect | Literal["allow", "deny", "modify", "require_approval", "warn"]
    )
    reason: str
    severity: RiskLevel
    operation_id: str | None = None
    policy_version: str = "1"
    policy_identity: str | None = None
    modifications: dict[str, Any] = field(default_factory=dict)
    required_approvals: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.policy_id, "policy_id")
        _validate_dotted_id(self.owner, "policy owner")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("policy_version must be a non-empty string")
        if self.policy_identity is None:
            object.__setattr__(
                self,
                "policy_identity",
                f"{self.owner}:{self.policy_id}@{self.policy_version}",
            )
        elif not isinstance(self.policy_identity, str) or not self.policy_identity:
            raise ValueError("policy_identity must be a non-empty string")
        object.__setattr__(self, "effect", PolicyEffect(self.effect))
        object.__setattr__(self, "severity", RiskLevel(self.severity))
        object.__setattr__(self, "modifications", _dict(self.modifications))
        object.__setattr__(self, "required_approvals", tuple(self.required_approvals))
        for approval in self.required_approvals:
            if not isinstance(approval, str):
                raise TypeError("required approvals must be strings")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        for item in self.evidence:
            if not isinstance(item, Evidence):
                raise TypeError("policy decision evidence must be Evidence")
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "owner": self.owner,
            "policy_version": self.policy_version,
            "policy_identity": self.policy_identity,
            "effect": (
                self.effect.value
                if isinstance(self.effect, PolicyEffect)
                else self.effect
            ),
            "reason": self.reason,
            "severity": self.severity.value,
            "operation_id": self.operation_id,
            "modifications": self.modifications,
            "required_approvals": list(self.required_approvals),
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyDecision":
        values = dict(data)
        values["policy_version"] = str(values.get("policy_version") or "1")
        values["policy_identity"] = values.get("policy_identity")
        values["effect"] = PolicyEffect(values["effect"])
        values["severity"] = RiskLevel(values["severity"])
        values["required_approvals"] = tuple(values.get("required_approvals", ()))
        values["evidence"] = tuple(
            Evidence.from_dict(item) for item in values.get("evidence", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class PolicyDecisionTrace:
    """Inspectable trace explaining one policy decision from runtime facts."""

    trace_id: str
    operation_id: str
    policy_id: str
    owner: str
    policy_version: str
    policy_identity: str
    effect: (
        PolicyEffect | Literal["allow", "deny", "modify", "require_approval", "warn"]
    )
    reason: str
    stage: str
    task_id: str | None = None
    capability_id: str | None = None
    approval_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    actor: dict[str, Any] = field(default_factory=dict)
    tenant: dict[str, Any] = field(default_factory=dict)
    source_scope: tuple[str, ...] = ()
    resource: dict[str, Any] = field(default_factory=dict)
    runtime_facts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dotted_id(self.policy_id, "trace policy_id")
        _validate_dotted_id(self.owner, "trace policy owner")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("trace policy_version must be a non-empty string")
        if not isinstance(self.policy_identity, str) or not self.policy_identity:
            raise ValueError("trace policy_identity must be a non-empty string")
        object.__setattr__(self, "effect", PolicyEffect(self.effect))
        object.__setattr__(self, "approval_ids", tuple(self.approval_ids))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "source_scope", tuple(self.source_scope))
        for value in (*self.approval_ids, *self.evidence_ids, *self.source_scope):
            if not isinstance(value, str):
                raise TypeError("trace id and source scope values must be strings")
        object.__setattr__(self, "actor", _dict(self.actor))
        object.__setattr__(self, "tenant", _dict(self.tenant))
        object.__setattr__(self, "resource", _dict(self.resource))
        object.__setattr__(self, "runtime_facts", _dict(self.runtime_facts))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "operation_id": self.operation_id,
            "policy_id": self.policy_id,
            "owner": self.owner,
            "policy_version": self.policy_version,
            "policy_identity": self.policy_identity,
            "effect": (
                self.effect.value
                if isinstance(self.effect, PolicyEffect)
                else self.effect
            ),
            "reason": self.reason,
            "stage": self.stage,
            "task_id": self.task_id,
            "capability_id": self.capability_id,
            "approval_ids": list(self.approval_ids),
            "evidence_ids": list(self.evidence_ids),
            "actor": self.actor,
            "tenant": self.tenant,
            "source_scope": list(self.source_scope),
            "resource": self.resource,
            "runtime_facts": self.runtime_facts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyDecisionTrace":
        values = dict(data)
        values["effect"] = PolicyEffect(values["effect"])
        values["approval_ids"] = tuple(values.get("approval_ids", ()))
        values["evidence_ids"] = tuple(values.get("evidence_ids", ()))
        values["source_scope"] = tuple(values.get("source_scope", ()))
        return cls(**values)


@dataclass(frozen=True)
class GovernanceAuditRecord:
    """Immutable audit record for one governance evaluation."""

    audit_id: str
    operation_id: str
    stage: str
    allowed: bool
    blocked: bool
    pending_approval: bool
    policy_decisions: tuple[PolicyDecision, ...] = ()
    traces: tuple[PolicyDecisionTrace, ...] = ()
    task_id: str | None = None
    capability_id: str | None = None
    actor: dict[str, Any] = field(default_factory=dict)
    tenant: dict[str, Any] = field(default_factory=dict)
    source_scope: tuple[str, ...] = ()
    resource: dict[str, Any] = field(default_factory=dict)
    operation_context: dict[str, Any] = field(default_factory=dict)
    task_context: dict[str, Any] = field(default_factory=dict)
    capability_context: dict[str, Any] = field(default_factory=dict)
    approval_context: dict[str, Any] = field(default_factory=dict)
    evidence_context: dict[str, Any] = field(default_factory=dict)
    runtime_facts: dict[str, Any] = field(default_factory=dict)
    evaluation_trace: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_decisions", tuple(self.policy_decisions))
        for decision in self.policy_decisions:
            if not isinstance(decision, PolicyDecision):
                raise TypeError("audit policy_decisions must be PolicyDecision")
        object.__setattr__(self, "traces", tuple(self.traces))
        for trace in self.traces:
            if not isinstance(trace, PolicyDecisionTrace):
                raise TypeError("audit traces must be PolicyDecisionTrace")
        object.__setattr__(self, "source_scope", tuple(self.source_scope))
        for value in self.source_scope:
            if not isinstance(value, str):
                raise TypeError("audit source_scope values must be strings")
        object.__setattr__(self, "actor", _dict(self.actor))
        object.__setattr__(self, "tenant", _dict(self.tenant))
        object.__setattr__(self, "resource", _dict(self.resource))
        object.__setattr__(self, "operation_context", _dict(self.operation_context))
        object.__setattr__(self, "task_context", _dict(self.task_context))
        object.__setattr__(self, "capability_context", _dict(self.capability_context))
        object.__setattr__(self, "approval_context", _dict(self.approval_context))
        object.__setattr__(self, "evidence_context", _dict(self.evidence_context))
        object.__setattr__(self, "runtime_facts", _dict(self.runtime_facts))
        object.__setattr__(self, "evaluation_trace", _dict(self.evaluation_trace))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "operation_id": self.operation_id,
            "stage": self.stage,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "pending_approval": self.pending_approval,
            "policy_decisions": [
                decision.to_dict() for decision in self.policy_decisions
            ],
            "traces": [trace.to_dict() for trace in self.traces],
            "task_id": self.task_id,
            "capability_id": self.capability_id,
            "actor": self.actor,
            "tenant": self.tenant,
            "source_scope": list(self.source_scope),
            "resource": self.resource,
            "operation_context": self.operation_context,
            "task_context": self.task_context,
            "capability_context": self.capability_context,
            "approval_context": self.approval_context,
            "evidence_context": self.evidence_context,
            "runtime_facts": self.runtime_facts,
            "evaluation_trace": self.evaluation_trace,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernanceAuditRecord":
        values = dict(data)
        values["policy_decisions"] = tuple(
            PolicyDecision.from_dict(item)
            for item in values.get("policy_decisions", ())
        )
        values["traces"] = tuple(
            PolicyDecisionTrace.from_dict(item) for item in values.get("traces", ())
        )
        values["source_scope"] = tuple(values.get("source_scope", ()))
        return cls(**values)


@dataclass(frozen=True)
class ApprovalRequest:
    """Runtime approval state requested by a policy decision."""

    approval_id: str
    operation_id: str
    reason: str
    proposed_action: dict[str, Any]
    risk: RiskLevel
    evidence_ids: tuple[str, ...] = ()
    status: (
        ApprovalStatus
        | Literal["pending", "approved", "rejected", "cancelled", "expired"]
    ) = ApprovalStatus.PENDING
    requested_by_policy_id: str | None = None
    owner: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk", RiskLevel(self.risk))
        object.__setattr__(self, "status", ApprovalStatus(self.status))
        object.__setattr__(self, "proposed_action", _dict(self.proposed_action))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        for evidence_id in self.evidence_ids:
            if not isinstance(evidence_id, str):
                raise TypeError("evidence_ids must be strings")
        if self.requested_by_policy_id is not None:
            _validate_dotted_id(self.requested_by_policy_id, "requested_by_policy_id")
        if self.owner is not None:
            _validate_dotted_id(self.owner, "approval owner")
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "operation_id": self.operation_id,
            "reason": self.reason,
            "proposed_action": self.proposed_action,
            "risk": self.risk.value,
            "evidence_ids": list(self.evidence_ids),
            "status": (
                self.status.value
                if isinstance(self.status, ApprovalStatus)
                else self.status
            ),
            "requested_by_policy_id": self.requested_by_policy_id,
            "owner": self.owner,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApprovalRequest":
        values = dict(data)
        values["risk"] = RiskLevel(values["risk"])
        values["status"] = ApprovalStatus(values.get("status", ApprovalStatus.PENDING))
        values["evidence_ids"] = tuple(values.get("evidence_ids", ()))
        return cls(**values)


@dataclass(frozen=True)
class GovernanceResult:
    """Composed policy outcome for one runtime operation."""

    allowed: bool
    blocked: bool
    pending_approval: bool
    decisions: tuple[PolicyDecision, ...] = ()
    approval_requests: tuple[ApprovalRequest, ...] = ()
    modified_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))
        for decision in self.decisions:
            if not isinstance(decision, PolicyDecision):
                raise TypeError("governance decisions must be PolicyDecision")
        object.__setattr__(self, "approval_requests", tuple(self.approval_requests))
        for request in self.approval_requests:
            if not isinstance(request, ApprovalRequest):
                raise TypeError("approval_requests must be ApprovalRequest")
        object.__setattr__(self, "modified_contract", _dict(self.modified_contract))
        object.__setattr__(self, "metadata", _dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blocked": self.blocked,
            "pending_approval": self.pending_approval,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "approval_requests": [
                request.to_dict() for request in self.approval_requests
            ],
            "modified_contract": self.modified_contract,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernanceResult":
        values = dict(data)
        values["decisions"] = tuple(
            PolicyDecision.from_dict(item) for item in values.get("decisions", ())
        )
        values["approval_requests"] = tuple(
            ApprovalRequest.from_dict(item)
            for item in values.get("approval_requests", ())
        )
        return cls(**values)


@runtime_checkable
class Executor(Protocol):
    """Behavioral contract for task executors."""

    id: str
    capability_ids: frozenset[str]

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        """Execute a task and return typed evidence."""


@runtime_checkable
class Policy(Protocol):
    """Behavioral contract for contract-shaping and governance policies."""

    id: str
    owner: str

    def applies_to(self, request: Any, operation_type: str) -> bool:
        """Return whether this policy applies to the operation type."""

    def modify_contract(self, contract: Any) -> Any:
        """Return a modified operation contract."""

    def evaluate_operation(self, operation: Operation) -> PolicyDecision | None:
        """Return a runtime policy decision when applicable."""


@runtime_checkable
class ContextProvider(Protocol):
    """Behavioral contract for audience-specific context rendering."""

    id: str
    owner: str
    audiences: frozenset[ContextAudience]

    async def render(
        self,
        context: Mapping[str, Any],
        audience: ContextAudience,
        token_budget: int,
    ) -> ContextBlock | None:
        """Render bounded context for one audience."""


@runtime_checkable
class RuntimeStore(Protocol):
    """Persistence contract for runtime operations, tasks, events, and evidence."""

    async def save_operation(self, operation: Operation) -> None:
        """Persist an operation snapshot."""

    async def load_operation(self, operation_id: str) -> Operation | None:
        """Load one operation snapshot by ID."""

    async def list_operations(self) -> list[Operation]:
        """Return persisted operation snapshots in creation order."""

    async def save_task(self, task: Task) -> None:
        """Persist a task snapshot."""

    async def load_task(self, task_id: str) -> Task | None:
        """Load one task snapshot by ID."""

    async def list_tasks(self, operation_id: str | None = None) -> list[Task]:
        """Return persisted task snapshots, optionally for one operation."""

    async def claim_task(
        self,
        task_id: str,
        *,
        lease_id: str | None = None,
        lease_owner: str,
        lease_expires_at: float | None = None,
        worker_id: str | None = None,
        worker_owner: str | None = None,
    ) -> Task | None:
        """Atomically claim pending or blocked work before executor invocation."""

    async def heartbeat_task(
        self,
        task_id: str,
        *,
        lease_id: str,
        lease_expires_at: float,
    ) -> Task | None:
        """Extend the current task lease only when the fencing token matches."""

    async def commit_task_blocked(
        self,
        *,
        operation: Operation | None,
        task: Task,
        events: tuple[RuntimeEvent, ...],
        lease_id: str | None = None,
    ) -> bool:
        """Atomically persist a blocked task transition and related events."""

    async def commit_task_started(self, task: Task, event: RuntimeEvent) -> None:
        """Atomically persist a claimed/running task and start event."""

    async def commit_task_succeeded(
        self,
        task: Task,
        evidence: tuple[Evidence, ...],
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        """Atomically persist task success, output evidence, and success event."""

    async def commit_task_failed(
        self,
        task: Task,
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        """Atomically persist task failure and failure event."""

    async def commit_approval_update(
        self,
        request: ApprovalRequest,
        event: RuntimeEvent,
    ) -> None:
        """Atomically persist an approval update and corresponding event."""

    async def save_evidence(self, evidence: Evidence) -> None:
        """Persist accepted or rejected evidence."""

    async def append_event(self, event: RuntimeEvent) -> None:
        """Persist a runtime event."""

    async def save_policy_decision(self, decision: PolicyDecision) -> None:
        """Persist a policy decision snapshot."""

    async def save_governance_audit_record(self, record: GovernanceAuditRecord) -> None:
        """Persist an immutable governance audit record."""

    async def commit_governance_evaluation(
        self,
        *,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        """Atomically persist a governance evaluation and related events."""

    async def commit_governance_blocked(
        self,
        *,
        operation: Operation,
        task: Task | None,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        """Atomically persist governance audit and blocked lifecycle state."""

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        """Persist an approval request snapshot."""

    async def list_evidence(self, operation_id: str) -> list[Evidence]:
        """Return evidence associated with an operation."""

    async def list_events(self, operation_id: str | None = None) -> list[RuntimeEvent]:
        """Return persisted runtime events, optionally for one operation."""

    async def list_policy_decisions(
        self, operation_id: str | None = None
    ) -> list[PolicyDecision]:
        """Return persisted policy decisions, optionally for one operation."""

    async def list_governance_audit_records(
        self, operation_id: str | None = None
    ) -> list[GovernanceAuditRecord]:
        """Return immutable governance audit records, optionally for one operation."""

    async def list_approval_requests(
        self, operation_id: str | None = None
    ) -> list[ApprovalRequest]:
        """Return persisted approval requests, optionally for one operation."""
