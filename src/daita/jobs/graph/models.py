"""Immutable records for the unregistered revision-2 graph persistence draft."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256

from ..._json import FrozenJsonObject, canonical_json
from ...llm.models import ModelSensitivity

MAX_GRAPH_TASKS = 64
MAX_GRAPH_EDGES = 192
MAX_GRAPH_DEPTH = 12
MAX_DIRECT_PARENTS = 16
MAX_MUTATION_FAN_OUT = 16
MAX_TASK_ATTEMPTS = 3
MAX_GRAPH_PARALLELISM = 4
MAX_TASK_WALL_TIME_SECONDS = 300
MAX_CHECKPOINTS_PER_ATTEMPT = 8
MAX_COMMENTS_PER_TASK = 32
MAX_PLANNER_TASKS = 8
MAX_GRAPH_MUTATIONS = 256
MAX_GRAPH_EVENTS = 4_096
MAX_CONTROLS_PER_TASK = 16
MAX_MODEL_REQUESTS_PER_ATTEMPT = 12
MAX_TASK_RESULT_BYTES = 64 * 1024
MAX_CHECKPOINT_BYTES = 16 * 1024
MAX_COMMENT_BYTES = 8 * 1024
MAX_GRAPH_INLINE_BYTES = 8 * 1024 * 1024
MAX_GRAPH_ARTIFACTS = 64
MAX_GRAPH_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_GRAPH_TOTAL_ARTIFACT_BYTES = 64 * 1024 * 1024
MIN_GRAPH_DEADLINE_SECONDS = 60
DEFAULT_GRAPH_DEADLINE_SECONDS = 3_600
MAX_GRAPH_DEADLINE_SECONDS = 24 * 3_600
MAX_GRAPH_INSPECTION_EVENTS = 100
MAX_GRAPH_TEXT = 8_192
MAX_GRAPH_JSON_DEPTH = 16

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded identifier")


def _text(value: str, name: str, *, maximum: int = MAX_GRAPH_TEXT) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{name} must be bounded non-empty text")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 digest")


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _optional_utc(value: datetime | None, name: str) -> None:
    if value is not None:
        _utc(value, name)


def _json_depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (tuple, list)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _bounded_json(
    value: Mapping[str, object],
    name: str,
    *,
    maximum_bytes: int,
    maximum_depth: int = MAX_GRAPH_JSON_DEPTH,
) -> FrozenJsonObject:
    frozen = FrozenJsonObject.from_mapping(value)
    if len(canonical_json(frozen).encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte bound")
    if _json_depth(frozen) > maximum_depth:
        raise ValueError(f"{name} exceeds its depth bound")
    return frozen


def _sorted_identifiers(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    material = tuple(values)
    for value in material:
        _identifier(value, name)
    if material != tuple(sorted(set(material))):
        raise ValueError(f"{name} values must be unique and sorted")
    return material


def _positive(value: int, name: str, *, maximum: int | None = None) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be a bounded positive integer")


def _nonnegative(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def canonical_digest(value: Mapping[str, object]) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def reserved_artifact_id(attempt_id: str) -> str:
    """Derive the one restart-stable artifact reservation for an internal attempt."""

    _identifier(attempt_id, "attempt artifact reservation")
    return "artifact-" + sha256(attempt_id.encode("utf-8")).hexdigest()[:32]


class GraphState(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    BLOCKED = "blocked"
    NEEDS_ATTENTION = "needs_attention"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_GRAPH_STATES = frozenset(
    {GraphState.SUCCEEDED, GraphState.FAILED, GraphState.CANCELLED}
)


class GraphDesiredState(str, Enum):
    RUN = "run"
    CANCEL = "cancel"


class TaskState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    REVIEW = "review"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    SUPERSEDED = "superseded"


TERMINAL_TASK_STATES = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.SKIPPED,
        TaskState.SUPERSEDED,
    }
)


class TaskRole(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    REVIEWER = "reviewer"
    FINALIZER = "finalizer"
    INTERNAL = "internal"


CONTROL_BUDGET_ROLES = frozenset(
    {TaskRole.PLANNER, TaskRole.REVIEWER, TaskRole.FINALIZER}
)


class TaskExecutionKind(str, Enum):
    MODEL = "model"
    INTERNAL_CAPABILITY = "internal_capability"


class EdgeKind(str, Enum):
    REQUIRES_ACCEPTED_SUCCESS = "requires_accepted_success"


class AttemptState(str, Enum):
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    REVIEW_REQUESTED = "review_requested"
    TIMED_OUT = "timed_out"
    PROTOCOL_VIOLATION = "protocol_violation"
    FENCED = "fenced"


ACTIVE_ATTEMPT_STATES = frozenset({AttemptState.CLAIMED, AttemptState.RUNNING})
TERMINAL_ATTEMPT_STATES = frozenset(set(AttemptState) - set(ACTIVE_ATTEMPT_STATES))


class ControlState(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ControlKind(str, Enum):
    NEEDS_INPUT = "needs_input"
    NEEDS_AUTHORIZATION = "needs_authorization"
    NEEDS_REPLAN = "needs_replan"
    REVIEW_REQUESTED = "review_requested"
    CHANGES_REQUESTED = "changes_requested"
    EFFECT_UNCERTAIN = "effect_uncertain"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    SOURCE_OR_CONTRACT_DRIFT = "source_or_contract_drift"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RETRY_CIRCUIT_OPEN = "retry_circuit_open"


class MutationDecision(str, Enum):
    COMMITTED = "committed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class GraphLimits:
    max_tasks: int = MAX_GRAPH_TASKS
    max_edges: int = MAX_GRAPH_EDGES
    max_depth: int = MAX_GRAPH_DEPTH
    max_direct_parents: int = MAX_DIRECT_PARENTS
    max_fan_out: int = MAX_MUTATION_FAN_OUT
    max_attempts_per_task: int = MAX_TASK_ATTEMPTS
    max_parallelism: int = MAX_GRAPH_PARALLELISM
    max_mutations: int = MAX_GRAPH_MUTATIONS
    max_events: int = MAX_GRAPH_EVENTS
    max_controls_per_task: int = MAX_CONTROLS_PER_TASK
    max_checkpoints_per_attempt: int = MAX_CHECKPOINTS_PER_ATTEMPT
    max_comments_per_task: int = MAX_COMMENTS_PER_TASK
    max_artifacts: int = MAX_GRAPH_ARTIFACTS
    max_artifact_bytes: int = MAX_GRAPH_ARTIFACT_BYTES
    max_total_artifact_bytes: int = MAX_GRAPH_TOTAL_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        ceilings = (
            (self.max_tasks, "graph task limit", MAX_GRAPH_TASKS),
            (self.max_edges, "graph edge limit", MAX_GRAPH_EDGES),
            (self.max_depth, "graph depth limit", MAX_GRAPH_DEPTH),
            (
                self.max_direct_parents,
                "graph direct-parent limit",
                MAX_DIRECT_PARENTS,
            ),
            (self.max_fan_out, "graph fan-out limit", MAX_MUTATION_FAN_OUT),
            (
                self.max_attempts_per_task,
                "graph attempt limit",
                MAX_TASK_ATTEMPTS,
            ),
            (
                self.max_parallelism,
                "graph parallelism limit",
                MAX_GRAPH_PARALLELISM,
            ),
            (self.max_mutations, "graph mutation limit", MAX_GRAPH_MUTATIONS),
            (self.max_events, "graph event limit", MAX_GRAPH_EVENTS),
            (
                self.max_controls_per_task,
                "graph control limit",
                MAX_CONTROLS_PER_TASK,
            ),
            (
                self.max_checkpoints_per_attempt,
                "graph checkpoint limit",
                MAX_CHECKPOINTS_PER_ATTEMPT,
            ),
            (
                self.max_comments_per_task,
                "graph comment limit",
                MAX_COMMENTS_PER_TASK,
            ),
            (self.max_artifacts, "graph artifact limit", MAX_GRAPH_ARTIFACTS),
            (
                self.max_artifact_bytes,
                "graph artifact byte limit",
                MAX_GRAPH_ARTIFACT_BYTES,
            ),
            (
                self.max_total_artifact_bytes,
                "graph total artifact byte limit",
                MAX_GRAPH_TOTAL_ARTIFACT_BYTES,
            ),
        )
        for value, name, maximum in ceilings:
            _positive(value, name, maximum=maximum)


@dataclass(frozen=True, slots=True, order=True)
class BudgetAmount:
    dimension: str
    amount: int

    def __post_init__(self) -> None:
        _identifier(self.dimension, "budget dimension")
        _nonnegative(self.amount, "budget amount")


@dataclass(frozen=True, slots=True, order=True)
class BudgetLimit:
    dimension: str
    ceiling: int
    control_reserved: int = 0

    def __post_init__(self) -> None:
        _identifier(self.dimension, "budget dimension")
        _nonnegative(self.ceiling, "budget ceiling")
        _nonnegative(self.control_reserved, "budget control reserve")
        if self.control_reserved > self.ceiling:
            raise ValueError("budget control reserve exceeds its ceiling")


@dataclass(frozen=True, slots=True)
class GraphAuthority:
    source_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()
    connector_ids: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    access_modes: tuple[str, ...] = ()
    operational_effects: tuple[str, ...] = ("none",)
    model_route_ids: tuple[str, ...] = ()
    sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED
    contract_bindings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "source_ids",
            "resource_ids",
            "connector_ids",
            "capability_ids",
            "access_modes",
            "operational_effects",
            "model_route_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _sorted_identifiers(tuple(getattr(self, field_name)), field_name),
            )
        if self.operational_effects not in ((), ("none",)):
            raise ValueError("graph V1 authority must be effect-free")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("graph authority sensitivity must be ModelSensitivity")
        bindings = _bounded_json(
            self.contract_bindings,
            "graph authority contract bindings",
            maximum_bytes=MAX_TASK_RESULT_BYTES,
        )
        object.__setattr__(self, "contract_bindings", bindings)

    @property
    def digest(self) -> str:
        return canonical_digest(self.digest_material())

    def digest_material(self) -> dict[str, object]:
        return {
            "source_ids": self.source_ids,
            "resource_ids": self.resource_ids,
            "connector_ids": self.connector_ids,
            "capability_ids": self.capability_ids,
            "access_modes": self.access_modes,
            "operational_effects": self.operational_effects,
            "model_route_ids": self.model_route_ids,
            "sensitivity": self.sensitivity.value,
            "contract_bindings": self.contract_bindings,
        }


@dataclass(frozen=True, slots=True)
class GraphJobSpecification:
    principal_id: str
    objective: str
    outcome_contract: Mapping[str, object]
    authority: GraphAuthority
    distribution_plan_digest: str
    budgets: tuple[BudgetLimit, ...]
    deadline_at: datetime
    limits: GraphLimits = GraphLimits()
    retry_policy: Mapping[str, object] = field(default_factory=dict)
    cancellation_policy: Mapping[str, object] = field(default_factory=dict)
    effect_mode: str = "disabled"
    planner_task_template: Mapping[str, object] | None = None
    finalizer_task_template: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.principal_id, "graph principal_id")
        _text(self.objective, "graph objective", maximum=MAX_TASK_RESULT_BYTES)
        if not isinstance(self.authority, GraphAuthority):
            raise TypeError("graph authority must be GraphAuthority")
        _digest(self.distribution_plan_digest, "distribution plan digest")
        _utc(self.deadline_at, "graph deadline")
        if not isinstance(self.limits, GraphLimits):
            raise TypeError("graph limits must be GraphLimits")
        if self.effect_mode != "disabled":
            raise ValueError("graph V1 effect mode must be disabled")
        budgets = tuple(self.budgets)
        if not budgets or any(not isinstance(item, BudgetLimit) for item in budgets):
            raise ValueError("graph budgets must contain BudgetLimit records")
        if budgets != tuple(sorted(budgets)) or len(
            {item.dimension for item in budgets}
        ) != len(budgets):
            raise ValueError("graph budgets must be unique and sorted")
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(
            self,
            "outcome_contract",
            _bounded_json(
                self.outcome_contract,
                "graph outcome contract",
                maximum_bytes=MAX_TASK_RESULT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "retry_policy",
            _bounded_json(
                self.retry_policy,
                "graph retry policy",
                maximum_bytes=MAX_CHECKPOINT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "cancellation_policy",
            _bounded_json(
                self.cancellation_policy,
                "graph cancellation policy",
                maximum_bytes=MAX_CHECKPOINT_BYTES,
            ),
        )
        if self.planner_task_template is not None:
            object.__setattr__(
                self,
                "planner_task_template",
                _bounded_json(
                    self.planner_task_template,
                    "planner task template",
                    maximum_bytes=MAX_CHECKPOINT_BYTES,
                ),
            )
        object.__setattr__(
            self,
            "finalizer_task_template",
            _bounded_json(
                self.finalizer_task_template,
                "finalizer task template",
                maximum_bytes=MAX_CHECKPOINT_BYTES,
            ),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.digest_material())

    def digest_material(self) -> dict[str, object]:
        return {
            "principal_id": self.principal_id,
            "objective": self.objective,
            "outcome_contract": self.outcome_contract,
            "authority": self.authority.digest_material(),
            "distribution_plan_digest": self.distribution_plan_digest,
            "budgets": [
                {
                    "dimension": item.dimension,
                    "ceiling": item.ceiling,
                    "control_reserved": item.control_reserved,
                }
                for item in self.budgets
            ],
            "deadline_at": self.deadline_at.isoformat(),
            "limits": {
                field_name: getattr(self.limits, field_name)
                for field_name in self.limits.__dataclass_fields__
            },
            "retry_policy": self.retry_policy,
            "cancellation_policy": self.cancellation_policy,
            "effect_mode": self.effect_mode,
            "planner_task_template": self.planner_task_template,
            "finalizer_task_template": self.finalizer_task_template,
        }


@dataclass(frozen=True, slots=True)
class GraphTaskSpecification:
    title: str
    description: str
    expected_result_contract: Mapping[str, object]
    authority: GraphAuthority
    budgets: tuple[BudgetAmount, ...]
    max_steps: int
    max_wall_time_seconds: int
    created_by: str

    def __post_init__(self) -> None:
        _text(self.title, "task title", maximum=512)
        _text(self.description, "task description", maximum=MAX_TASK_RESULT_BYTES)
        if not isinstance(self.authority, GraphAuthority):
            raise TypeError("task authority must be GraphAuthority")
        budgets = tuple(self.budgets)
        if any(not isinstance(item, BudgetAmount) for item in budgets):
            raise TypeError("task budgets must contain BudgetAmount records")
        if budgets != tuple(sorted(budgets)) or len(
            {item.dimension for item in budgets}
        ) != len(budgets):
            raise ValueError("task budgets must be unique and sorted")
        _positive(
            self.max_steps,
            "task model request limit",
            maximum=MAX_MODEL_REQUESTS_PER_ATTEMPT,
        )
        _positive(
            self.max_wall_time_seconds,
            "task wall time",
            maximum=MAX_TASK_WALL_TIME_SECONDS,
        )
        _identifier(self.created_by, "task creator")
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(
            self,
            "expected_result_contract",
            _bounded_json(
                self.expected_result_contract,
                "task result contract",
                maximum_bytes=MAX_TASK_RESULT_BYTES,
            ),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.digest_material())

    def digest_material(self) -> dict[str, object]:
        return {
            "title": self.title,
            "description": self.description,
            "expected_result_contract": self.expected_result_contract,
            "authority": self.authority.digest_material(),
            "budgets": [
                {"dimension": item.dimension, "amount": item.amount}
                for item in self.budgets
            ],
            "max_steps": self.max_steps,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "created_by": self.created_by,
        }


@dataclass(frozen=True, slots=True)
class GraphJob:
    agent_id: str
    job_id: str
    conversation_id: str
    origin_run_id: str
    origin_call_id: str
    state: GraphState
    desired_state: GraphDesiredState
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime
    specification: GraphJobSpecification
    specification_digest: str
    finalizer_task_id: str
    terminal_at: datetime | None = None
    terminal_result_id: str | None = None
    failure_code: str | None = None
    migration_provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for identifier, name in (
            (self.agent_id, "graph agent_id"),
            (self.job_id, "graph job_id"),
            (self.conversation_id, "graph conversation_id"),
            (self.origin_run_id, "graph origin_run_id"),
            (self.origin_call_id, "graph origin_call_id"),
            (self.finalizer_task_id, "graph finalizer_task_id"),
        ):
            _identifier(identifier, name)
        if not isinstance(self.state, GraphState):
            raise TypeError("graph state must be GraphState")
        if not isinstance(self.desired_state, GraphDesiredState):
            raise TypeError("graph desired state must be GraphDesiredState")
        for timestamp, name in (
            (self.created_at, "graph created_at"),
            (self.updated_at, "graph updated_at"),
            (self.deadline_at, "graph deadline_at"),
        ):
            _utc(timestamp, name)
        _optional_utc(self.terminal_at, "graph terminal_at")
        if not isinstance(self.specification, GraphJobSpecification):
            raise TypeError("graph specification is invalid")
        if self.updated_at < self.created_at or self.deadline_at <= self.created_at:
            raise ValueError("graph timestamps are invalid")
        if self.deadline_at != self.specification.deadline_at:
            raise ValueError("graph deadline must match its specification")
        _digest(self.specification_digest, "graph specification digest")
        if self.specification_digest != self.specification.digest:
            raise ValueError("graph specification digest does not match")
        if (self.state in TERMINAL_GRAPH_STATES) != (self.terminal_at is not None):
            raise ValueError("graph terminal state must agree with terminal_at")
        if self.state is GraphState.SUCCEEDED:
            if self.terminal_result_id is None:
                raise ValueError("successful graph requires a finalizer result")
        elif self.terminal_result_id is not None:
            raise ValueError("only a successful graph may expose a terminal result")
        if self.failure_code is not None:
            _identifier(self.failure_code, "graph failure code")
        object.__setattr__(
            self,
            "migration_provenance",
            _bounded_json(
                self.migration_provenance,
                "graph migration provenance",
                maximum_bytes=MAX_GRAPH_INLINE_BYTES,
            ),
        )

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_GRAPH_STATES


@dataclass(frozen=True, slots=True)
class JobGraph:
    agent_id: str
    job_id: str
    revision: int
    task_count: int
    edge_count: int
    mutation_count: int
    active_attempt_count: int
    next_ready_at: datetime | None
    finalization_attempt_id: str | None
    finalization_started_revision: int | None
    created_at: datetime
    updated_at: datetime
    topology_digest: str

    def __post_init__(self) -> None:
        _identifier(self.agent_id, "job graph agent_id")
        _identifier(self.job_id, "job graph job_id")
        _nonnegative(self.revision, "job graph revision")
        for value, name, maximum in (
            (self.task_count, "job graph task count", MAX_GRAPH_TASKS),
            (self.edge_count, "job graph edge count", MAX_GRAPH_EDGES),
            (self.mutation_count, "job graph mutation count", MAX_GRAPH_MUTATIONS),
            (
                self.active_attempt_count,
                "job graph active attempt count",
                MAX_GRAPH_PARALLELISM,
            ),
        ):
            _nonnegative(value, name)
            if value > maximum:
                raise ValueError(f"{name} exceeds its bound")
        _optional_utc(self.next_ready_at, "job graph next_ready_at")
        if (self.finalization_attempt_id is None) != (
            self.finalization_started_revision is None
        ):
            raise ValueError("job graph finalization seal is incomplete")
        if self.finalization_attempt_id is not None:
            _identifier(self.finalization_attempt_id, "finalization attempt_id")
            assert self.finalization_started_revision is not None
            _nonnegative(
                self.finalization_started_revision,
                "finalization started revision",
            )
        _utc(self.created_at, "job graph created_at")
        _utc(self.updated_at, "job graph updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("job graph updated_at precedes created_at")
        _digest(self.topology_digest, "job graph topology digest")


@dataclass(frozen=True, slots=True)
class GraphTask:
    agent_id: str
    job_id: str
    task_id: str
    state: TaskState
    role: TaskRole
    execution_kind: TaskExecutionKind
    priority: int
    not_before: datetime | None
    current_attempt_id: str | None
    task_revision: int
    specification: GraphTaskSpecification
    task_spec_digest: str
    task_scope_digest: str
    attempt_count: int
    failure_streak: int
    fencing_epoch: int
    created_at: datetime
    updated_at: datetime
    terminal_at: datetime | None = None
    supersedes_task_id: str | None = None
    superseded_by_task_id: str | None = None
    latest_result_id: str | None = None
    latest_control_id: str | None = None
    latest_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        for identifier, name in (
            (self.agent_id, "task agent_id"),
            (self.job_id, "task job_id"),
            (self.task_id, "task_id"),
        ):
            _identifier(identifier, name)
        if not isinstance(self.state, TaskState):
            raise TypeError("task state must be TaskState")
        if not isinstance(self.role, TaskRole):
            raise TypeError("task role must be TaskRole")
        if not isinstance(self.execution_kind, TaskExecutionKind):
            raise TypeError("task execution kind is invalid")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError("task priority must be an integer")
        if not -1_000 <= self.priority <= 1_000:
            raise ValueError("task priority exceeds its bound")
        _optional_utc(self.not_before, "task not_before")
        if self.current_attempt_id is not None:
            _identifier(self.current_attempt_id, "task current attempt_id")
        _positive(self.task_revision, "task revision")
        if not isinstance(self.specification, GraphTaskSpecification):
            raise TypeError("task specification is invalid")
        _digest(self.task_spec_digest, "task specification digest")
        _digest(self.task_scope_digest, "task scope digest")
        if self.task_spec_digest != self.specification.digest:
            raise ValueError("task specification digest does not match")
        if self.task_scope_digest != self.specification.authority.digest:
            raise ValueError("task scope digest does not match")
        _nonnegative(self.attempt_count, "task attempt count")
        _nonnegative(self.failure_streak, "task failure streak")
        _nonnegative(self.fencing_epoch, "task fencing epoch")
        if self.attempt_count > MAX_TASK_ATTEMPTS:
            raise ValueError("task attempt count exceeds its bound")
        _utc(self.created_at, "task created_at")
        _utc(self.updated_at, "task updated_at")
        _optional_utc(self.terminal_at, "task terminal_at")
        if self.updated_at < self.created_at:
            raise ValueError("task updated_at precedes created_at")
        if (self.state in TERMINAL_TASK_STATES) != (self.terminal_at is not None):
            raise ValueError("task terminal state must agree with terminal_at")
        if (self.state is TaskState.RUNNING) != (self.current_attempt_id is not None):
            raise ValueError("task running state must agree with current attempt")
        if self.state is TaskState.SUCCEEDED and self.latest_result_id is None:
            raise ValueError("successful task requires an accepted result")
        if self.state is not TaskState.SUCCEEDED and self.latest_result_id is not None:
            raise ValueError("only a successful task may expose an accepted result")
        for reference, name in (
            (self.supersedes_task_id, "task supersedes reference"),
            (self.superseded_by_task_id, "task superseded-by reference"),
            (self.latest_result_id, "task result reference"),
            (self.latest_control_id, "task control reference"),
            (self.latest_checkpoint_id, "task checkpoint reference"),
        ):
            if reference is not None:
                _identifier(reference, name)
        if self.state is TaskState.SUPERSEDED and self.superseded_by_task_id is None:
            raise ValueError("superseded task requires its replacement")
        if (
            self.supersedes_task_id == self.task_id
            or self.superseded_by_task_id == self.task_id
        ):
            raise ValueError("task cannot supersede itself")


@dataclass(frozen=True, slots=True)
class TaskDependency:
    agent_id: str
    job_id: str
    upstream_task_id: str
    downstream_task_id: str
    edge_kind: EdgeKind
    created_at: datetime
    creator_key: str
    mutation_id: str | None = None

    def __post_init__(self) -> None:
        for identifier, name in (
            (self.agent_id, "edge agent_id"),
            (self.job_id, "edge job_id"),
            (self.upstream_task_id, "edge upstream task_id"),
            (self.downstream_task_id, "edge downstream task_id"),
            (self.creator_key, "edge creator key"),
        ):
            _identifier(identifier, name)
        if self.upstream_task_id == self.downstream_task_id:
            raise ValueError("task dependency cannot be a self-edge")
        if not isinstance(self.edge_kind, EdgeKind):
            raise TypeError("task dependency kind is invalid")
        _utc(self.created_at, "edge created_at")
        if self.mutation_id is not None:
            _identifier(self.mutation_id, "edge mutation_id")


@dataclass(frozen=True, slots=True)
class TaskAttempt:
    agent_id: str
    job_id: str
    task_id: str
    attempt_id: str
    ordinal: int
    fencing_epoch: int
    state: AttemptState
    claim_token: str
    run_id: str
    lease_expires_at: datetime | None
    absolute_deadline_at: datetime
    started_at: datetime | None
    heartbeat_at: datetime | None
    ended_at: datetime | None
    execution_scope_digest: str
    executor_id: str
    error_code: str | None = None
    diagnostic: str | None = None
    reserved_budgets: tuple[BudgetAmount, ...] = ()
    measured_usage: tuple[BudgetAmount, ...] = ()
    loop_exit_id: str | None = None
    checkpoint_ids: tuple[str, ...] = ()
    result_id: str | None = None
    control_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    effect_receipt_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for identifier, name in (
            (self.agent_id, "attempt agent_id"),
            (self.job_id, "attempt job_id"),
            (self.task_id, "attempt task_id"),
            (self.attempt_id, "attempt_id"),
            (self.claim_token, "attempt claim_token"),
            (self.run_id, "attempt run_id"),
            (self.executor_id, "attempt executor_id"),
        ):
            _identifier(identifier, name)
        _positive(self.ordinal, "attempt ordinal", maximum=MAX_TASK_ATTEMPTS)
        _positive(self.fencing_epoch, "attempt fencing epoch")
        if not isinstance(self.state, AttemptState):
            raise TypeError("attempt state is invalid")
        _optional_utc(self.lease_expires_at, "attempt lease expiry")
        _utc(self.absolute_deadline_at, "attempt absolute deadline")
        _optional_utc(self.started_at, "attempt started_at")
        _optional_utc(self.heartbeat_at, "attempt heartbeat_at")
        _optional_utc(self.ended_at, "attempt ended_at")
        _digest(self.execution_scope_digest, "attempt execution scope digest")
        if self.state in ACTIVE_ATTEMPT_STATES:
            if self.lease_expires_at is None or self.ended_at is not None:
                raise ValueError("active attempt lease/end state is invalid")
        elif self.ended_at is None or self.lease_expires_at is not None:
            raise ValueError("settled attempt lease/end state is invalid")
        if self.state is AttemptState.CLAIMED and self.started_at is not None:
            raise ValueError("claimed attempt cannot have started_at")
        if self.state is AttemptState.RUNNING and self.started_at is None:
            raise ValueError("running attempt requires started_at")
        if self.error_code is not None:
            _identifier(self.error_code, "attempt error code")
        if self.diagnostic is not None:
            _text(self.diagnostic, "attempt diagnostic", maximum=MAX_CHECKPOINT_BYTES)
        for field_name in ("reserved_budgets", "measured_usage"):
            budgets = tuple(getattr(self, field_name))
            if any(not isinstance(item, BudgetAmount) for item in budgets):
                raise TypeError(
                    f"attempt {field_name} must contain BudgetAmount records"
                )
            if budgets != tuple(sorted(budgets)) or len(
                {item.dimension for item in budgets}
            ) != len(budgets):
                raise ValueError(f"attempt {field_name} must be unique and sorted")
            object.__setattr__(self, field_name, budgets)
        for reference, name in (
            (self.loop_exit_id, "attempt LoopExit reference"),
            (self.result_id, "attempt result reference"),
        ):
            if reference is not None:
                _identifier(reference, name)
        for field_name, maximum in (
            ("checkpoint_ids", MAX_CHECKPOINTS_PER_ATTEMPT),
            ("control_ids", MAX_CONTROLS_PER_TASK),
            ("artifact_ids", MAX_GRAPH_ARTIFACTS),
            ("effect_receipt_ids", 0),
        ):
            references = _sorted_identifiers(
                tuple(getattr(self, field_name)), f"attempt {field_name}"
            )
            if len(references) > maximum:
                raise ValueError(f"attempt {field_name} exceeds its bound")
            object.__setattr__(self, field_name, references)


@dataclass(frozen=True, slots=True)
class TaskResult:
    agent_id: str
    job_id: str
    task_id: str
    result_id: str
    attempt_id: str
    run_id: str
    result_kind: str
    schema_digest: str
    payload: Mapping[str, object]
    summary: str
    sensitivity: ModelSensitivity
    provenance: Mapping[str, object]
    artifact_ids: tuple[str, ...]
    effect_receipt_ids: tuple[str, ...]
    verification: Mapping[str, object]
    residual_risk: str | None
    downstream_constraints: Mapping[str, object]
    completed_at: datetime
    result_digest: str

    def __post_init__(self) -> None:
        for identifier, name in (
            (self.agent_id, "result agent_id"),
            (self.job_id, "result job_id"),
            (self.task_id, "result task_id"),
            (self.result_id, "result_id"),
            (self.attempt_id, "result attempt_id"),
            (self.run_id, "result run_id"),
            (self.result_kind, "result kind"),
        ):
            _identifier(identifier, name)
        _digest(self.schema_digest, "result schema digest")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("result sensitivity is invalid")
        _text(self.summary, "result summary", maximum=MAX_CHECKPOINT_BYTES)
        if self.residual_risk is not None:
            _text(
                self.residual_risk, "result residual risk", maximum=MAX_CHECKPOINT_BYTES
            )
        artifacts = _sorted_identifiers(tuple(self.artifact_ids), "artifact reference")
        effects = _sorted_identifiers(
            tuple(self.effect_receipt_ids), "effect receipt reference"
        )
        if len(artifacts) > MAX_GRAPH_ARTIFACTS:
            raise ValueError("result artifact references exceed their bound")
        if effects:
            raise ValueError("graph V1 result cannot reference effects")
        object.__setattr__(self, "artifact_ids", artifacts)
        object.__setattr__(self, "effect_receipt_ids", effects)
        for field_name, maximum in (
            ("payload", MAX_TASK_RESULT_BYTES),
            ("provenance", MAX_TASK_RESULT_BYTES),
            ("verification", MAX_CHECKPOINT_BYTES),
            ("downstream_constraints", MAX_CHECKPOINT_BYTES),
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_json(
                    getattr(self, field_name),
                    f"result {field_name}",
                    maximum_bytes=maximum,
                ),
            )
        _utc(self.completed_at, "result completed_at")
        _digest(self.result_digest, "result digest")
        if self.result_digest != self.computed_digest:
            raise ValueError("result digest does not match its content")

    @property
    def computed_digest(self) -> str:
        return canonical_digest(self.digest_material())

    def digest_material(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "result_id": self.result_id,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "result_kind": self.result_kind,
            "schema_digest": self.schema_digest,
            "payload": self.payload,
            "summary": self.summary,
            "sensitivity": self.sensitivity.value,
            "provenance": self.provenance,
            "artifact_ids": self.artifact_ids,
            "effect_receipt_ids": self.effect_receipt_ids,
            "verification": self.verification,
            "residual_risk": self.residual_risk,
            "downstream_constraints": self.downstream_constraints,
            "completed_at": self.completed_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TaskCheckpoint:
    agent_id: str
    job_id: str
    task_id: str
    attempt_id: str
    checkpoint_id: str
    fencing_epoch: int
    ordinal: int
    milestone: str
    payload: Mapping[str, object]
    created_at: datetime
    payload_digest: str

    def __post_init__(self) -> None:
        for identifier, name in (
            (self.agent_id, "checkpoint agent_id"),
            (self.job_id, "checkpoint job_id"),
            (self.task_id, "checkpoint task_id"),
            (self.attempt_id, "checkpoint attempt_id"),
            (self.checkpoint_id, "checkpoint_id"),
        ):
            _identifier(identifier, name)
        _positive(self.fencing_epoch, "checkpoint fencing epoch")
        _positive(
            self.ordinal,
            "checkpoint ordinal",
            maximum=MAX_CHECKPOINTS_PER_ATTEMPT,
        )
        _text(self.milestone, "checkpoint milestone", maximum=512)
        payload = _bounded_json(
            self.payload,
            "checkpoint payload",
            maximum_bytes=MAX_CHECKPOINT_BYTES,
        )
        object.__setattr__(self, "payload", payload)
        _utc(self.created_at, "checkpoint created_at")
        _digest(self.payload_digest, "checkpoint payload digest")
        if self.payload_digest != canonical_digest(payload):
            raise ValueError("checkpoint digest does not match")


@dataclass(frozen=True, slots=True)
class TaskComment:
    agent_id: str
    job_id: str
    task_id: str
    comment_id: str
    author_kind: str
    author_id: str
    sensitivity: ModelSensitivity
    body: str
    created_at: datetime
    body_digest: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.agent_id, "comment agent_id"),
            (self.job_id, "comment job_id"),
            (self.task_id, "comment task_id"),
            (self.comment_id, "comment_id"),
            (self.author_kind, "comment author kind"),
            (self.author_id, "comment author_id"),
        ):
            _identifier(value, name)
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("comment sensitivity is invalid")
        _text(self.body, "comment body", maximum=MAX_COMMENT_BYTES)
        _utc(self.created_at, "comment created_at")
        _digest(self.body_digest, "comment body digest")
        if self.body_digest != canonical_digest({"body": self.body}):
            raise ValueError("comment digest does not match")


@dataclass(frozen=True, slots=True)
class TaskControl:
    agent_id: str
    job_id: str
    task_id: str
    control_id: str
    kind: ControlKind
    state: ControlState
    requesting_attempt_id: str | None
    payload: Mapping[str, object]
    created_at: datetime
    payload_digest: str
    resolved_at: datetime | None = None
    resolved_by_kind: str | None = None
    resolved_by_id: str | None = None
    resolution: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.agent_id, "control agent_id"),
            (self.job_id, "control job_id"),
            (self.task_id, "control task_id"),
            (self.control_id, "control_id"),
        ):
            _identifier(value, name)
        if not isinstance(self.kind, ControlKind) or not isinstance(
            self.state, ControlState
        ):
            raise TypeError("control enum is invalid")
        if self.requesting_attempt_id is not None:
            _identifier(self.requesting_attempt_id, "control requesting attempt")
        payload = _bounded_json(
            self.payload,
            "control payload",
            maximum_bytes=MAX_CHECKPOINT_BYTES,
        )
        object.__setattr__(self, "payload", payload)
        _utc(self.created_at, "control created_at")
        _digest(self.payload_digest, "control payload digest")
        if self.payload_digest != canonical_digest(payload):
            raise ValueError("control digest does not match")
        _optional_utc(self.resolved_at, "control resolved_at")
        if self.state is ControlState.OPEN:
            if any(
                item is not None
                for item in (
                    self.resolved_at,
                    self.resolved_by_kind,
                    self.resolved_by_id,
                    self.resolution,
                )
            ):
                raise ValueError("open control cannot have resolution fields")
        else:
            if (
                self.resolved_at is None
                or self.resolved_by_kind is None
                or self.resolved_by_id is None
                or self.resolution is None
            ):
                raise ValueError("settled control requires complete resolution")
            _identifier(self.resolved_by_kind, "control resolver kind")
            _identifier(self.resolved_by_id, "control resolver id")
            object.__setattr__(
                self,
                "resolution",
                _bounded_json(
                    self.resolution,
                    "control resolution",
                    maximum_bytes=MAX_CHECKPOINT_BYTES,
                ),
            )


@dataclass(frozen=True, slots=True)
class GraphMutationRequest:
    agent_id: str
    job_id: str
    mutation_id: str
    actor_kind: str
    actor_key: str
    idempotency_key: str
    expected_revision: int
    created_at: datetime
    tasks: tuple[GraphTask, ...] = ()
    dependencies: tuple[TaskDependency, ...] = ()
    supersessions: tuple[tuple[str, str], ...] = ()
    creator_task_id: str | None = None
    creator_attempt_id: str | None = None
    claim_token: str | None = None
    fencing_epoch: int | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.agent_id, "mutation agent_id"),
            (self.job_id, "mutation job_id"),
            (self.mutation_id, "mutation_id"),
            (self.actor_kind, "mutation actor kind"),
            (self.actor_key, "mutation actor key"),
            (self.idempotency_key, "mutation idempotency key"),
        ):
            _identifier(value, name)
        _nonnegative(self.expected_revision, "mutation expected revision")
        _utc(self.created_at, "mutation created_at")
        tasks = tuple(self.tasks)
        dependencies = tuple(self.dependencies)
        supersessions = tuple(tuple(item) for item in self.supersessions)
        if len(tasks) > MAX_MUTATION_FAN_OUT:
            raise ValueError("mutation task fan-out exceeds its bound")
        if any(not isinstance(item, GraphTask) for item in tasks):
            raise TypeError("mutation tasks must be GraphTask records")
        if any(not isinstance(item, TaskDependency) for item in dependencies):
            raise TypeError("mutation dependencies must be TaskDependency records")
        if any(len(item) != 2 for item in supersessions):
            raise ValueError("mutation supersession pairs are invalid")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "supersessions", supersessions)
        caller_fields = (
            self.creator_task_id,
            self.creator_attempt_id,
            self.claim_token,
            self.fencing_epoch,
        )
        if any(item is not None for item in caller_fields):
            if any(item is None for item in caller_fields):
                raise ValueError("mutation attempt binding must be complete")
            assert self.creator_task_id is not None
            assert self.creator_attempt_id is not None
            assert self.claim_token is not None
            assert self.fencing_epoch is not None
            _identifier(self.creator_task_id, "mutation creator task")
            _identifier(self.creator_attempt_id, "mutation creator attempt")
            _identifier(self.claim_token, "mutation claim token")
            _positive(self.fencing_epoch, "mutation fencing epoch")

    @property
    def payload_digest(self) -> str:
        return canonical_digest(self.payload_material())

    def payload_material(self) -> dict[str, object]:
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "role": task.role.value,
                    "execution_kind": task.execution_kind.value,
                    "priority": task.priority,
                    "not_before": (
                        None if task.not_before is None else task.not_before.isoformat()
                    ),
                    "specification_digest": task.task_spec_digest,
                    "scope_digest": task.task_scope_digest,
                }
                for task in self.tasks
            ],
            "dependencies": [
                {
                    "upstream": edge.upstream_task_id,
                    "downstream": edge.downstream_task_id,
                    "kind": edge.edge_kind.value,
                }
                for edge in self.dependencies
            ],
            "supersessions": self.supersessions,
        }


@dataclass(frozen=True, slots=True)
class GraphMutation:
    agent_id: str
    job_id: str
    mutation_id: str
    actor_kind: str
    actor_key: str
    idempotency_key: str
    payload_digest: str
    expected_revision: int
    committed_revision: int
    decision: MutationDecision
    created_at: datetime
    resulting_task_ids: tuple[str, ...] = ()
    resulting_edges: tuple[tuple[str, str], ...] = ()
    failure_code: str | None = None
    creator_task_id: str | None = None
    creator_attempt_id: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.agent_id, "mutation agent_id"),
            (self.job_id, "mutation job_id"),
            (self.mutation_id, "mutation_id"),
            (self.actor_kind, "mutation actor kind"),
            (self.actor_key, "mutation actor key"),
            (self.idempotency_key, "mutation idempotency key"),
        ):
            _identifier(value, name)
        _digest(self.payload_digest, "mutation payload digest")
        _nonnegative(self.expected_revision, "mutation expected revision")
        _nonnegative(self.committed_revision, "mutation committed revision")
        if not isinstance(self.decision, MutationDecision):
            raise TypeError("mutation decision is invalid")
        _utc(self.created_at, "mutation created_at")
        tasks = _sorted_identifiers(
            tuple(self.resulting_task_ids), "mutation resulting task"
        )
        edges = tuple(tuple(item) for item in self.resulting_edges)
        if edges != tuple(sorted(set(edges))) or any(len(item) != 2 for item in edges):
            raise ValueError("mutation resulting edges must be unique and sorted")
        for upstream, downstream in edges:
            _identifier(upstream, "mutation edge upstream")
            _identifier(downstream, "mutation edge downstream")
        if self.decision is MutationDecision.COMMITTED:
            if self.failure_code is not None:
                raise ValueError("committed mutation cannot have failure code")
        elif self.failure_code is None:
            raise ValueError("rejected mutation requires failure code")
        if self.failure_code is not None:
            _identifier(self.failure_code, "mutation failure code")
        for reference, name in (
            (self.creator_task_id, "mutation creator task"),
            (self.creator_attempt_id, "mutation creator attempt"),
        ):
            if reference is not None:
                _identifier(reference, name)
        object.__setattr__(self, "resulting_task_ids", tasks)
        object.__setattr__(self, "resulting_edges", edges)


@dataclass(frozen=True, slots=True)
class GraphEvent:
    event_id: int
    agent_id: str
    job_id: str
    kind: str
    created_at: datetime
    payload: Mapping[str, object]
    task_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        _positive(self.event_id, "graph event_id")
        for identifier, name in (
            (self.agent_id, "event agent_id"),
            (self.job_id, "event job_id"),
            (self.kind, "event kind"),
        ):
            _identifier(identifier, name)
        for reference, name in (
            (self.task_id, "event task_id"),
            (self.attempt_id, "event attempt_id"),
        ):
            if reference is not None:
                _identifier(reference, name)
        _utc(self.created_at, "event created_at")
        object.__setattr__(
            self,
            "payload",
            _bounded_json(
                self.payload,
                "event payload",
                maximum_bytes=MAX_CHECKPOINT_BYTES,
            ),
        )


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    agent_id: str
    job_id: str
    dimension: str
    ceiling: int
    settled: int
    reserved: int
    control_reserved: int
    updated_at: datetime
    task_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.agent_id, "ledger agent_id")
        _identifier(self.job_id, "ledger job_id")
        _identifier(self.dimension, "ledger dimension")
        if self.task_id is not None:
            _identifier(self.task_id, "ledger task_id")
        for value, name in (
            (self.ceiling, "ledger ceiling"),
            (self.settled, "ledger settled"),
            (self.reserved, "ledger reserved"),
            (self.control_reserved, "ledger control reserve"),
        ):
            _nonnegative(value, name)
        if self.settled + self.reserved > self.ceiling:
            raise ValueError("ledger consumption exceeds its ceiling")
        if self.control_reserved > self.ceiling:
            raise ValueError("ledger control reserve exceeds its ceiling")
        if self.task_id is not None and self.control_reserved != 0:
            raise ValueError("task ledger cannot contain a control reserve")
        _utc(self.updated_at, "ledger updated_at")


@dataclass(frozen=True, slots=True)
class AttemptBudgetReservation:
    agent_id: str
    job_id: str
    task_id: str
    attempt_id: str
    dimension: str
    reserved: int
    settled: int | None
    updated_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.agent_id, "reservation agent_id"),
            (self.job_id, "reservation job_id"),
            (self.task_id, "reservation task_id"),
            (self.attempt_id, "reservation attempt_id"),
            (self.dimension, "reservation dimension"),
        ):
            _identifier(value, name)
        _nonnegative(self.reserved, "attempt reservation")
        if self.settled is not None:
            _nonnegative(self.settled, "attempt settled usage")
        _utc(self.updated_at, "attempt reservation updated_at")


@dataclass(frozen=True, slots=True)
class GraphAdmission:
    job: GraphJob
    graph: JobGraph
    tasks: tuple[GraphTask, ...]
    dependencies: tuple[TaskDependency, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.job, GraphJob) or not isinstance(self.graph, JobGraph):
            raise TypeError("graph admission requires job and graph records")
        tasks = tuple(self.tasks)
        edges = tuple(self.dependencies)
        if any(not isinstance(item, GraphTask) for item in tasks):
            raise TypeError("graph admission tasks are invalid")
        if any(not isinstance(item, TaskDependency) for item in edges):
            raise TypeError("graph admission dependencies are invalid")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "dependencies", edges)


@dataclass(frozen=True, slots=True)
class GraphInspection:
    job: GraphJob
    graph: JobGraph
    tasks: tuple[GraphTask, ...]
    dependencies: tuple[TaskDependency, ...]
    attempts: tuple[TaskAttempt, ...]
    results: tuple[TaskResult, ...]
    controls: tuple[TaskControl, ...]
    checkpoints: tuple[TaskCheckpoint, ...] = ()
    comments: tuple[TaskComment, ...] = ()
    budget_ledgers: tuple[BudgetLedger, ...] = ()
    events: tuple[GraphEvent, ...] = ()
    delivery_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphEventPage:
    events: tuple[GraphEvent, ...]
    next_cursor: int | None


def topology_digest(
    tasks: tuple[GraphTask, ...], dependencies: tuple[TaskDependency, ...]
) -> str:
    return canonical_digest(
        {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "specification_digest": task.task_spec_digest,
                    "scope_digest": task.task_scope_digest,
                    "role": task.role.value,
                    "execution_kind": task.execution_kind.value,
                }
                for task in sorted(tasks, key=lambda item: item.task_id)
            ],
            "dependencies": [
                {
                    "upstream": edge.upstream_task_id,
                    "downstream": edge.downstream_task_id,
                    "kind": edge.edge_kind.value,
                }
                for edge in sorted(
                    dependencies,
                    key=lambda item: (item.upstream_task_id, item.downstream_task_id),
                )
            ],
        }
    )


__all__ = [
    "ACTIVE_ATTEMPT_STATES",
    "CONTROL_BUDGET_ROLES",
    "DEFAULT_GRAPH_DEADLINE_SECONDS",
    "MAX_CHECKPOINT_BYTES",
    "MAX_COMMENT_BYTES",
    "MAX_GRAPH_ARTIFACTS",
    "MAX_GRAPH_ARTIFACT_BYTES",
    "MAX_GRAPH_DEPTH",
    "MAX_GRAPH_EDGES",
    "MAX_GRAPH_EVENTS",
    "MAX_GRAPH_INLINE_BYTES",
    "MAX_GRAPH_INSPECTION_EVENTS",
    "MAX_GRAPH_MUTATIONS",
    "MAX_GRAPH_PARALLELISM",
    "MAX_GRAPH_TASKS",
    "MAX_GRAPH_TOTAL_ARTIFACT_BYTES",
    "MAX_TASK_ATTEMPTS",
    "MAX_TASK_RESULT_BYTES",
    "TERMINAL_ATTEMPT_STATES",
    "TERMINAL_GRAPH_STATES",
    "TERMINAL_TASK_STATES",
    "AttemptBudgetReservation",
    "AttemptState",
    "BudgetAmount",
    "BudgetLedger",
    "BudgetLimit",
    "ControlKind",
    "ControlState",
    "EdgeKind",
    "GraphAdmission",
    "GraphAuthority",
    "GraphDesiredState",
    "GraphEvent",
    "GraphEventPage",
    "GraphInspection",
    "GraphJob",
    "GraphJobSpecification",
    "GraphLimits",
    "GraphMutation",
    "GraphMutationRequest",
    "GraphState",
    "GraphTask",
    "GraphTaskSpecification",
    "JobGraph",
    "MutationDecision",
    "TaskAttempt",
    "TaskCheckpoint",
    "TaskComment",
    "TaskControl",
    "TaskDependency",
    "TaskExecutionKind",
    "TaskResult",
    "TaskRole",
    "TaskState",
    "canonical_digest",
    "reserved_artifact_id",
    "topology_digest",
]
