"""Immutable records persisted at operation-runtime checkpoints.

These records describe structural state only. The operation runtime remains the
owner of transition legality, execution, governance, evidence acceptance, and
recovery decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..events.models import RuntimeEvent
from ..llm.models import ModelRequest, ModelResponse
from ..loop.models import LoopBudgets, LoopPhase, LoopState, Readiness, Turn
from .leases import TaskLease
from .governance import ApprovalRequest, ApprovalStatus
from .models import (
    AgentTrigger,
    Evidence,
    Observation,
    Operation,
    OperationStatus,
    Task,
    TaskDependency,
    TaskStatus,
)


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class ModelCallStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelCall:
    """A normalized model-I/O checkpoint with no provider wire payload."""

    id: str
    operation_id: str
    turn_id: str
    provider_id: str
    request: ModelRequest
    status: ModelCallStatus
    created_at: datetime
    updated_at: datetime
    response: ModelResponse | None = None
    error_code: str | None = None
    cancellation_requested: bool = False

    def __post_init__(self) -> None:
        _required_text(self.id, "model call id")
        _required_text(self.operation_id, "model call operation_id")
        _required_text(self.turn_id, "model call turn_id")
        _required_text(self.provider_id, "model call provider_id")
        if not isinstance(self.request, ModelRequest):
            raise TypeError("model call request must be a ModelRequest")
        if not isinstance(self.status, ModelCallStatus):
            raise TypeError("model call status must be a ModelCallStatus")
        if self.response is not None and not isinstance(self.response, ModelResponse):
            raise TypeError("model call response must be a ModelResponse or None")
        if not isinstance(self.cancellation_requested, bool):
            raise TypeError("model call cancellation_requested must be a boolean")
        _aware(self.created_at, "model call created_at")
        _aware(self.updated_at, "model call updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("model call updated_at cannot precede created_at")
        if self.request.operation_id != self.operation_id:
            raise ValueError("model request operation does not match model call")
        if self.request.turn_id != self.turn_id:
            raise ValueError("model request turn does not match model call")
        if self.error_code is not None:
            _required_text(self.error_code, "model call error_code")

        if self.status is ModelCallStatus.STARTED:
            if self.response is not None:
                raise ValueError("started model call cannot contain a response")
            if self.error_code is not None:
                raise ValueError("started model call cannot contain an error")
        elif self.status is ModelCallStatus.COMPLETED:
            if self.response is None:
                raise ValueError("completed model call requires a response")
            if self.error_code is not None:
                raise ValueError("completed model call cannot contain an error")
        else:
            if self.response is not None:
                raise ValueError("failed model call cannot contain a response")
            if self.error_code is None:
                raise ValueError("failed model call requires an error")


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    """One structurally self-contained, immutable operation checkpoint."""

    trigger: AgentTrigger
    operation: Operation
    loop_state: LoopState
    budgets: LoopBudgets
    turns: tuple[Turn, ...]
    model_calls: tuple[ModelCall, ...]
    readiness: tuple[Readiness, ...]
    tasks: tuple[Task, ...]
    evidence: tuple[Evidence, ...]
    observations: tuple[Observation, ...]
    events: tuple[RuntimeEvent, ...]
    task_dependencies: tuple[TaskDependency, ...] = ()
    task_leases: tuple[TaskLease, ...] = ()
    approvals: tuple[ApprovalRequest, ...] = ()

    def __post_init__(self) -> None:
        self._validate_root_records()
        self._normalize_collections()
        self._validate_collection_types()
        self._validate_root_linkage()
        self._validate_children()
        self._validate_events()

    def _validate_root_records(self) -> None:
        for value, expected, field_name in (
            (self.trigger, AgentTrigger, "trigger"),
            (self.operation, Operation, "operation"),
            (self.loop_state, LoopState, "loop_state"),
            (self.budgets, LoopBudgets, "budgets"),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"snapshot {field_name} must be a {expected.__name__}")

    def _normalize_collections(self) -> None:
        for field_name in (
            "turns",
            "model_calls",
            "readiness",
            "tasks",
            "task_dependencies",
            "task_leases",
            "approvals",
            "evidence",
            "observations",
            "events",
        ):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)):
                raise TypeError(f"snapshot {field_name} must be a sequence of records")
            try:
                normalized = tuple(value)
            except TypeError as error:
                raise TypeError(
                    f"snapshot {field_name} must be a sequence of records"
                ) from error
            object.__setattr__(self, field_name, normalized)

    def _validate_collection_types(self) -> None:
        for values, expected, field_name in (
            (self.turns, Turn, "turns"),
            (self.model_calls, ModelCall, "model_calls"),
            (self.readiness, Readiness, "readiness"),
            (self.tasks, Task, "tasks"),
            (self.task_dependencies, TaskDependency, "task_dependencies"),
            (self.task_leases, TaskLease, "task_leases"),
            (self.approvals, ApprovalRequest, "approvals"),
            (self.evidence, Evidence, "evidence"),
            (self.observations, Observation, "observations"),
            (self.events, RuntimeEvent, "events"),
        ):
            if any(not isinstance(value, expected) for value in values):
                raise TypeError(
                    f"snapshot {field_name} must contain {expected.__name__} records"
                )

    def _validate_root_linkage(self) -> None:
        if self.trigger.agent_id != self.operation.agent_id:
            raise ValueError("trigger agent does not match operation agent")
        if self.trigger.id != self.operation.trigger_id:
            raise ValueError("trigger identity does not match operation trigger")
        if self.trigger.session_id != self.operation.session_id:
            raise ValueError("trigger session does not match operation session")

    @staticmethod
    def _require_unique_ids(values: tuple[object, ...], label: str) -> None:
        identifiers = [getattr(value, "id") for value in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate {label} id in operation snapshot")

    def _validate_children(self) -> None:
        self._require_unique_ids(self.turns, "turn")
        self._require_unique_ids(self.model_calls, "model call")
        self._require_unique_ids(self.tasks, "task")
        self._require_unique_ids(self.approvals, "approval")
        self._require_unique_ids(self.evidence, "evidence")

        operation_id = self.operation.id
        turn_by_id = {turn.id: turn for turn in self.turns}
        model_call_by_id = {
            model_call.id: model_call for model_call in self.model_calls
        }
        task_by_id = {task.id: task for task in self.tasks}
        approval_by_id = {approval.id: approval for approval in self.approvals}
        evidence_by_id = {item.id: item for item in self.evidence}
        dependency_edges = {
            (dependency.task_id, dependency.prerequisite_task_id)
            for dependency in self.task_dependencies
        }
        tool_call_ids_by_turn: dict[str, set[str]] = {}
        for model_call_record in self.model_calls:
            if model_call_record.response is None:
                continue
            tool_call_ids_by_turn.setdefault(model_call_record.turn_id, set()).update(
                call.id for call in model_call_record.response.tool_calls
            )

        for turn in self.turns:
            if turn.operation_id != operation_id:
                raise ValueError("turn operation does not match snapshot operation")
            for model_call_id in (turn.model_request_id, turn.model_response_id):
                if model_call_id is None:
                    continue
                linked_model_call = model_call_by_id.get(model_call_id)
                if linked_model_call is None:
                    raise ValueError("turn model call does not exist in snapshot")
                if linked_model_call.turn_id != turn.id:
                    raise ValueError("turn model call belongs to a different turn")

        for model_call in self.model_calls:
            if model_call.operation_id != operation_id:
                raise ValueError(
                    "model call operation does not match snapshot operation"
                )
            linked_turn = turn_by_id.get(model_call.turn_id)
            if linked_turn is None:
                raise ValueError("model call turn does not exist in snapshot")
            if linked_turn.model_request_id != model_call.id:
                raise ValueError("model call is not owned by its turn request pointer")
            if (model_call.status is ModelCallStatus.COMPLETED) != (
                linked_turn.model_response_id == model_call.id
            ):
                raise ValueError(
                    "turn response pointer requires its completed model call"
                )
            if any(
                message.agent_id != self.operation.agent_id
                for message in model_call.request.messages
            ):
                raise ValueError(
                    "model call message agent does not match operation agent"
                )

        for task_record in self.tasks:
            if task_record.operation_id != operation_id:
                raise ValueError("task operation does not match snapshot operation")
            if task_record.turn_id not in turn_by_id:
                raise ValueError("task turn does not exist in snapshot")
            if task_record.call_id not in tool_call_ids_by_turn.get(
                task_record.turn_id,
                set(),
            ):
                raise ValueError("task tool call does not exist in its turn response")
            for evidence_id in task_record.evidence_ids:
                linked_evidence = evidence_by_id.get(evidence_id)
                if linked_evidence is None:
                    raise ValueError("task evidence does not exist in snapshot")
                if (
                    linked_evidence.task_id != task_record.id
                    or linked_evidence.turn_id != task_record.turn_id
                ):
                    raise ValueError("task evidence belongs to a different task")
            for (
                evidence_id
            ) in task_record.execution_facts.validation_facts.evidence_ids:
                validation_evidence = evidence_by_id.get(evidence_id)
                if validation_evidence is None or not validation_evidence.accepted:
                    raise ValueError(
                        "task validation evidence is not accepted in the snapshot"
                    )
                if (
                    task_record.id,
                    validation_evidence.task_id,
                ) not in dependency_edges:
                    raise ValueError(
                        "task validation evidence requires a prerequisite edge"
                    )

        self._validate_task_dependencies(task_by_id)
        self._validate_task_leases(task_by_id)
        approval_task_ids: set[str] = set()
        for approval in self.approvals:
            if approval.operation_id != operation_id:
                raise ValueError("approval operation does not match snapshot operation")
            if approval.task_id not in task_by_id:
                raise ValueError("approval task does not exist in snapshot")
            if approval.task_id in approval_task_ids:
                raise ValueError("task may have at most one approval request")
            approval_task_ids.add(approval.task_id)

        waiting_approval_id = self.loop_state.waiting_approval_id
        operation_is_waiting = (
            self.operation.status is OperationStatus.WAITING_FOR_APPROVAL
        )
        loop_is_waiting = self.loop_state.phase is LoopPhase.AWAITING_APPROVAL
        if operation_is_waiting != loop_is_waiting:
            raise ValueError("operation and loop approval-wait state must agree")
        if operation_is_waiting != (waiting_approval_id is not None):
            raise ValueError("approval-waiting state requires one approval identity")
        if waiting_approval_id is not None:
            waiting_approval = approval_by_id.get(waiting_approval_id)
            if waiting_approval is None:
                raise ValueError("loop waiting approval does not exist in snapshot")
            if (
                task_by_id[waiting_approval.task_id].status
                is not TaskStatus.WAITING_FOR_APPROVAL
            ):
                raise ValueError("loop waiting approval task is not waiting")
            if waiting_approval.status not in {
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
                ApprovalStatus.DENIED,
            }:
                raise ValueError("cancelled approval cannot remain an active wait")
        waiting_tasks = tuple(
            task
            for task in self.tasks
            if task.status is TaskStatus.WAITING_FOR_APPROVAL
        )
        if operation_is_waiting:
            assert waiting_approval_id is not None
            if len(waiting_tasks) != 1:
                raise ValueError("approval wait requires exactly one waiting task")
        elif waiting_tasks:
            raise ValueError("waiting task requires an approval-waiting operation")

        for evidence_record in self.evidence:
            if evidence_record.operation_id != operation_id:
                raise ValueError("evidence operation does not match snapshot operation")
            linked_task = task_by_id.get(evidence_record.task_id)
            if linked_task is None:
                raise ValueError("evidence task does not exist in snapshot")
            if evidence_record.turn_id != linked_task.turn_id:
                raise ValueError("evidence turn does not match its task")
            if (
                evidence_record.capability_id != linked_task.capability_id
                or evidence_record.executor_id != linked_task.executor_id
                or evidence_record.attempt != linked_task.attempt
            ):
                raise ValueError("evidence execution identity does not match its task")
            if evidence_record.accepted != (
                evidence_record.id in linked_task.evidence_ids
            ):
                raise ValueError("task evidence acceptance linkage is not symmetric")
            if (
                evidence_record.metadata_schema_version >= 1
                and evidence_record.validation_facts
                != linked_task.execution_facts.validation_facts
            ):
                raise ValueError("evidence validator authority does not match its task")

        for observation in self.observations:
            if observation.operation_id != operation_id:
                raise ValueError(
                    "observation operation does not match snapshot operation"
                )
            if observation.turn_id not in turn_by_id:
                raise ValueError("observation turn does not exist in snapshot")
            linked_task = (
                None
                if observation.task_id is None
                else task_by_id.get(observation.task_id)
            )
            if observation.task_id is not None and linked_task is None:
                raise ValueError("observation task does not exist in snapshot")
            if observation.call_id is not None and observation.call_id not in (
                tool_call_ids_by_turn.get(observation.turn_id, set())
            ):
                raise ValueError(
                    "observation tool call does not exist in its turn response"
                )
            if linked_task is not None and (
                linked_task.turn_id != observation.turn_id
                or (
                    observation.call_id is not None
                    and observation.call_id != linked_task.call_id
                )
            ):
                raise ValueError("observation linkage does not match its task")
            if observation.evidence_id is not None:
                linked_evidence = evidence_by_id.get(observation.evidence_id)
                if linked_evidence is None:
                    raise ValueError("observation evidence does not exist in snapshot")
                if linked_task is None or linked_evidence.task_id != linked_task.id:
                    raise ValueError("observation evidence does not match its task")
                if linked_evidence.id not in linked_task.evidence_ids:
                    raise ValueError("observation evidence is not accepted by its task")

    def _validate_task_dependencies(self, task_by_id: dict[str, Task]) -> None:
        operation_id = self.operation.id
        edges: set[tuple[str, str]] = set()
        prerequisites_by_task: dict[str, set[str]] = {}
        for dependency in self.task_dependencies:
            if dependency.operation_id != operation_id:
                raise ValueError(
                    "task dependency operation does not match snapshot operation"
                )
            if dependency.task_id not in task_by_id:
                raise ValueError("dependency task does not exist in snapshot")
            if dependency.prerequisite_task_id not in task_by_id:
                raise ValueError(
                    "dependency prerequisite task does not exist in snapshot"
                )
            if dependency.task_id == dependency.prerequisite_task_id:
                raise ValueError("task dependency cannot be a self-edge")

            edge = (dependency.task_id, dependency.prerequisite_task_id)
            if edge in edges:
                raise ValueError("duplicate task dependency in operation snapshot")
            edges.add(edge)
            prerequisites_by_task.setdefault(dependency.task_id, set()).add(
                dependency.prerequisite_task_id
            )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError("task dependency graph contains a cycle")
            visiting.add(task_id)
            for prerequisite_task_id in prerequisites_by_task.get(task_id, ()):
                visit(prerequisite_task_id)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in prerequisites_by_task:
            visit(task_id)

    def _validate_task_leases(self, task_by_id: dict[str, Task]) -> None:
        operation_id = self.operation.id
        attempts_by_task: dict[str, set[int]] = {}
        fences_by_task: dict[str, set[int]] = {}
        last_attempt_by_task: dict[str, int] = {}
        last_fence_by_task: dict[str, int] = {}
        previous_lease_by_task: dict[str, TaskLease] = {}
        unreleased_tasks: set[str] = set()

        for lease in self.task_leases:
            if lease.operation_id != operation_id:
                raise ValueError(
                    "task lease operation does not match snapshot operation"
                )
            if lease.task_id not in task_by_id:
                raise ValueError("lease task does not exist in snapshot")

            attempts = attempts_by_task.setdefault(lease.task_id, set())
            if lease.attempt in attempts:
                raise ValueError("task lease attempt must be unique per task")
            attempts.add(lease.attempt)

            fences = fences_by_task.setdefault(lease.task_id, set())
            if lease.fencing_token in fences:
                raise ValueError("task lease fencing token must be unique per task")
            fences.add(lease.fencing_token)

            previous_attempt = last_attempt_by_task.get(lease.task_id)
            if previous_attempt is not None and lease.attempt <= previous_attempt:
                raise ValueError("task lease attempt must strictly increase per task")
            previous_fence = last_fence_by_task.get(lease.task_id)
            if previous_fence is not None and lease.fencing_token <= previous_fence:
                raise ValueError(
                    "task lease fencing token must strictly increase per task"
                )
            last_attempt_by_task[lease.task_id] = lease.attempt
            last_fence_by_task[lease.task_id] = lease.fencing_token

            previous_lease = previous_lease_by_task.get(lease.task_id)
            if previous_lease is not None:
                if previous_lease.released_at is None:
                    raise ValueError(
                        "a later task lease requires the prior lease to be released"
                    )
                if lease.acquired_at < previous_lease.released_at:
                    raise ValueError(
                        "task lease acquisition cannot overlap the prior attempt"
                    )
            previous_lease_by_task[lease.task_id] = lease

            if lease.released_at is None:
                if lease.task_id in unreleased_tasks:
                    raise ValueError("task may have at most one unreleased lease")
                unreleased_tasks.add(lease.task_id)

    def _validate_events(self) -> None:
        self._require_unique_ids(self.events, "event")
        turn_by_id = {turn.id: turn for turn in self.turns}
        model_call_by_id = {
            model_call.id: model_call for model_call in self.model_calls
        }
        task_by_id = {task.id: task for task in self.tasks}
        evidence_by_id = {item.id: item for item in self.evidence}
        approval_by_id = {approval.id: approval for approval in self.approvals}
        tool_call_ids_by_model_call: dict[str, set[str]] = {}
        for model_call_record in self.model_calls:
            tool_call_ids_by_model_call[model_call_record.id] = (
                set()
                if model_call_record.response is None
                else {call.id for call in model_call_record.response.tool_calls}
            )

        for event in self.events:
            if event.agent_id != self.operation.agent_id:
                raise ValueError("event agent does not match operation agent")
            if event.operation_id != self.operation.id:
                raise ValueError("event operation does not match snapshot operation")
            if event.session_id != self.operation.session_id:
                raise ValueError("event session does not match operation session")
            if event.turn_id is not None and event.turn_id not in turn_by_id:
                raise ValueError("event turn does not exist in snapshot")
            linked_model_call = (
                None
                if event.model_call_id is None
                else model_call_by_id.get(event.model_call_id)
            )
            if event.model_call_id is not None and linked_model_call is None:
                raise ValueError("event model call does not exist in snapshot")
            if event.model_call_id is not None and event.turn_id is None:
                raise ValueError("event model call requires an explicit turn")
            if (
                event.turn_id is not None
                and event.type != "turn.created"
                and any(
                    candidate.turn_id == event.turn_id for candidate in self.model_calls
                )
                and event.model_call_id is None
            ):
                raise ValueError("event is missing its model call correlation")
            if (
                linked_model_call is not None
                and event.turn_id is not None
                and linked_model_call.turn_id != event.turn_id
            ):
                raise ValueError("event model call does not match its turn")
            if event.call_id is not None:
                if event.turn_id is None or linked_model_call is None:
                    raise ValueError(
                        "event tool call requires turn and model call correlation"
                    )
                if event.call_id not in tool_call_ids_by_model_call.get(
                    linked_model_call.id,
                    set(),
                ):
                    raise ValueError(
                        "event tool call does not exist in its model response"
                    )

            task = None if event.task_id is None else task_by_id.get(event.task_id)
            if event.task_id is not None and task is None:
                raise ValueError("event task does not exist in snapshot")
            approval = (
                None
                if event.approval_id is None
                else approval_by_id.get(event.approval_id)
            )
            if event.approval_id is not None and approval is None:
                raise ValueError("event approval does not exist in snapshot")
            if approval is not None:
                if task is None or approval.task_id != task.id:
                    raise ValueError("event approval does not match its task")
            if task is not None:
                if (
                    event.turn_id is None
                    or linked_model_call is None
                    or event.call_id is None
                ):
                    raise ValueError(
                        "event task requires turn, model call, and tool call correlation"
                    )
                if task.turn_id != event.turn_id:
                    raise ValueError("event task does not match its turn")
                if task.call_id != event.call_id:
                    raise ValueError("event tool call does not match its task")
                if (
                    event.capability_id is not None
                    and event.capability_id != task.capability_id
                ):
                    raise ValueError("event capability does not match its task")
                if (
                    event.executor_id is not None
                    and event.executor_id != task.executor_id
                ):
                    raise ValueError("event executor does not match its task")

            item = (
                None
                if event.evidence_id is None
                else evidence_by_id.get(event.evidence_id)
            )
            if event.evidence_id is not None and item is None:
                raise ValueError("event evidence does not exist in snapshot")
            if item is not None:
                if task is None:
                    raise ValueError("event evidence requires task correlation")
                if item.task_id != task.id:
                    raise ValueError("event evidence does not match its task")
                if item.turn_id != event.turn_id:
                    raise ValueError("event evidence does not match its turn")
                if (
                    event.capability_id is not None
                    and item.capability_id != event.capability_id
                ):
                    raise ValueError("event evidence capability does not match")
                if (
                    event.executor_id is not None
                    and item.executor_id != event.executor_id
                ):
                    raise ValueError("event evidence executor does not match")
