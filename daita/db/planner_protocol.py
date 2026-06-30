"""Typed protocol records for the DB agent planner loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping, Protocol


class DbPlannerDecisionStatus(str, Enum):
    """Stable terminal and continuation statuses returned by a DB planner."""

    CONTINUE = "continue"
    FINISH = "finish"
    CLARIFY = "clarify"
    BLOCKED = "blocked"
    FAILED = "failed"


class DbPlannerActionKind(str, Enum):
    """Stable action vocabulary proposed by a DB planner."""

    INSPECT_SCHEMA = "inspect_schema"
    REGISTER_CATALOG_SOURCE = "register_catalog_source"
    SEARCH_SCHEMA = "search_schema"
    INSPECT_ASSET = "inspect_asset"
    FIND_RELATIONSHIP_PATHS = "find_relationship_paths"
    SEARCH_COLUMN_VALUES = "search_column_values"
    BUILD_PLANNING_CONTEXT = "build_planning_context"
    PROPOSE_SQL_READ = "propose_sql_read"
    REPAIR_QUERY_PLAN = "repair_query_plan"
    EXECUTE_VALIDATED_READ = "execute_validated_read"
    PROPOSE_SQL_WRITE = "propose_sql_write"
    EXECUTE_VALIDATED_WRITE = "execute_validated_write"
    RECALL_MEMORY = "recall_memory"
    PLAN_MEMORY_UPDATE = "plan_memory_update"
    COMMIT_MEMORY_UPDATE = "commit_memory_update"
    PLAN_ANALYSIS = "plan_analysis"
    EXECUTE_ANALYSIS_STEP = "execute_analysis_step"
    SUMMARIZE_ANALYSIS = "summarize_analysis"
    PLAN_MONITOR_CREATE = "plan_monitor_create"
    COMMIT_MONITOR_CREATE = "commit_monitor_create"
    PLAN_MONITOR_LIFECYCLE = "plan_monitor_lifecycle"
    COMMIT_MONITOR_LIFECYCLE = "commit_monitor_lifecycle"
    READ_MONITOR_STATE = "read_monitor_state"
    RESOLVE_MONITOR_APPROVAL = "resolve_monitor_approval"
    SYNTHESIZE = "synthesize"
    CLARIFY = "clarify"
    FINISH = "finish"


@dataclass(frozen=True)
class DbPlannerAction:
    """One model-proposed action in a typed DAG patch."""

    action_id: str
    kind: DbPlannerActionKind
    input: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        object.__setattr__(self, "kind", DbPlannerActionKind(self.kind))
        object.__setattr__(self, "input", _json_dict(self.input))
        object.__setattr__(self, "depends_on", _tuple_strings(self.depends_on))
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "input": self.input,
            "depends_on": list(self.depends_on),
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbPlannerAction":
        values = dict(data)
        values["kind"] = DbPlannerActionKind(values["kind"])
        values["depends_on"] = tuple(values.get("depends_on") or ())
        return cls(**values)


@dataclass(frozen=True)
class DbPlannerDecision:
    """One planner response containing intent, actions, and stop conditions."""

    status: DbPlannerDecisionStatus
    intent: dict[str, Any] = field(default_factory=dict)
    actions: tuple[DbPlannerAction, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    clarification_question: str | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", DbPlannerDecisionStatus(self.status))
        object.__setattr__(self, "intent", _json_dict(self.intent))
        object.__setattr__(
            self,
            "actions",
            tuple(
                (
                    action
                    if isinstance(action, DbPlannerAction)
                    else DbPlannerAction.from_dict(action)
                )
                for action in self.actions
            ),
        )
        object.__setattr__(
            self, "stop_conditions", _tuple_strings(self.stop_conditions)
        )
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "intent": self.intent,
            "actions": [action.to_dict() for action in self.actions],
            "stop_conditions": list(self.stop_conditions),
            "clarification_question": self.clarification_question,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbPlannerDecision":
        values = dict(data)
        values["status"] = DbPlannerDecisionStatus(values["status"])
        values["actions"] = tuple(
            DbPlannerAction.from_dict(item) for item in values.get("actions", ())
        )
        values["stop_conditions"] = tuple(values.get("stop_conditions") or ())
        return cls(**values)


@dataclass(frozen=True)
class DbPlannerObservation:
    """Compact runtime facts returned to the planner."""

    accepted_evidence_summaries: tuple[dict[str, Any], ...] = ()
    rejected_evidence_summaries: tuple[dict[str, Any], ...] = ()
    task_statuses: tuple[dict[str, Any], ...] = ()
    governance_status: dict[str, Any] = field(default_factory=dict)
    approval_state: dict[str, Any] = field(default_factory=dict)
    validation_errors: tuple[dict[str, Any], ...] = ()
    execution_errors: tuple[dict[str, Any], ...] = ()
    zero_row_facts: tuple[dict[str, Any], ...] = ()
    truncation_facts: tuple[dict[str, Any], ...] = ()
    retry_facts: tuple[dict[str, Any], ...] = ()
    no_progress_facts: tuple[dict[str, Any], ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "accepted_evidence_summaries",
            "rejected_evidence_summaries",
            "task_statuses",
            "validation_errors",
            "execution_errors",
            "zero_row_facts",
            "truncation_facts",
            "retry_facts",
            "no_progress_facts",
        ):
            object.__setattr__(
                self, field_name, _tuple_json_dicts(getattr(self, field_name))
            )
        object.__setattr__(
            self, "governance_status", _json_dict(self.governance_status)
        )
        object.__setattr__(self, "approval_state", _json_dict(self.approval_state))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_evidence_summaries": list(self.accepted_evidence_summaries),
            "rejected_evidence_summaries": list(self.rejected_evidence_summaries),
            "task_statuses": list(self.task_statuses),
            "governance_status": self.governance_status,
            "approval_state": self.approval_state,
            "validation_errors": list(self.validation_errors),
            "execution_errors": list(self.execution_errors),
            "zero_row_facts": list(self.zero_row_facts),
            "truncation_facts": list(self.truncation_facts),
            "retry_facts": list(self.retry_facts),
            "no_progress_facts": list(self.no_progress_facts),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbPlannerObservation":
        return cls(**dict(data))


@dataclass(frozen=True)
class DbLoopState:
    """Planner input assembled from persisted runtime state."""

    operation_id: str
    normalized_user_request: dict[str, Any]
    source_scope: tuple[str, ...] = ()
    explicit_mode: str | None = None
    explicit_requested_capabilities: tuple[str, ...] = ()
    safety_frame: dict[str, Any] = field(default_factory=dict)
    latest_compiled_contract_snapshot: dict[str, Any] | None = None
    available_action_kinds: tuple[DbPlannerActionKind, ...] = ()
    capability_summaries: tuple[dict[str, Any], ...] = ()
    catalog_context: dict[str, Any] = field(default_factory=dict)
    memory_context: dict[str, Any] = field(default_factory=dict)
    task_summaries: tuple[dict[str, Any], ...] = ()
    accepted_evidence_summaries: tuple[dict[str, Any], ...] = ()
    rejected_evidence_summaries: tuple[dict[str, Any], ...] = ()
    approval_state: dict[str, Any] = field(default_factory=dict)
    validation_summaries: tuple[dict[str, Any], ...] = ()
    execution_error_summaries: tuple[dict[str, Any], ...] = ()
    planner_observations: tuple[DbPlannerObservation, ...] = ()
    runtime_limits: dict[str, Any] = field(default_factory=dict)
    remaining_budget: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id is required")
        object.__setattr__(
            self, "normalized_user_request", _json_dict(self.normalized_user_request)
        )
        object.__setattr__(self, "source_scope", _tuple_strings(self.source_scope))
        object.__setattr__(
            self,
            "explicit_requested_capabilities",
            _tuple_strings(self.explicit_requested_capabilities),
        )
        object.__setattr__(self, "safety_frame", _json_dict(self.safety_frame))
        if self.latest_compiled_contract_snapshot is not None:
            object.__setattr__(
                self,
                "latest_compiled_contract_snapshot",
                _json_dict(self.latest_compiled_contract_snapshot),
            )
        object.__setattr__(
            self,
            "available_action_kinds",
            tuple(DbPlannerActionKind(kind) for kind in self.available_action_kinds),
        )
        for field_name in (
            "capability_summaries",
            "task_summaries",
            "accepted_evidence_summaries",
            "rejected_evidence_summaries",
            "validation_summaries",
            "execution_error_summaries",
        ):
            object.__setattr__(
                self, field_name, _tuple_json_dicts(getattr(self, field_name))
            )
        object.__setattr__(self, "catalog_context", _json_dict(self.catalog_context))
        object.__setattr__(self, "memory_context", _json_dict(self.memory_context))
        object.__setattr__(self, "approval_state", _json_dict(self.approval_state))
        object.__setattr__(
            self,
            "planner_observations",
            tuple(
                (
                    observation
                    if isinstance(observation, DbPlannerObservation)
                    else DbPlannerObservation.from_dict(observation)
                )
                for observation in self.planner_observations
            ),
        )
        object.__setattr__(self, "runtime_limits", _json_dict(self.runtime_limits))
        object.__setattr__(self, "remaining_budget", _json_dict(self.remaining_budget))
        object.__setattr__(self, "diagnostics", _json_dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "normalized_user_request": self.normalized_user_request,
            "source_scope": list(self.source_scope),
            "explicit_mode": self.explicit_mode,
            "explicit_requested_capabilities": list(
                self.explicit_requested_capabilities
            ),
            "safety_frame": self.safety_frame,
            "latest_compiled_contract_snapshot": self.latest_compiled_contract_snapshot,
            "available_action_kinds": [
                kind.value for kind in self.available_action_kinds
            ],
            "capability_summaries": list(self.capability_summaries),
            "catalog_context": self.catalog_context,
            "memory_context": self.memory_context,
            "task_summaries": list(self.task_summaries),
            "accepted_evidence_summaries": list(self.accepted_evidence_summaries),
            "rejected_evidence_summaries": list(self.rejected_evidence_summaries),
            "approval_state": self.approval_state,
            "validation_summaries": list(self.validation_summaries),
            "execution_error_summaries": list(self.execution_error_summaries),
            "planner_observations": [
                observation.to_dict() for observation in self.planner_observations
            ],
            "runtime_limits": self.runtime_limits,
            "remaining_budget": self.remaining_budget,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbLoopState":
        values = dict(data)
        values["source_scope"] = tuple(values.get("source_scope") or ())
        values["explicit_requested_capabilities"] = tuple(
            values.get("explicit_requested_capabilities") or ()
        )
        values["available_action_kinds"] = tuple(
            DbPlannerActionKind(kind)
            for kind in values.get("available_action_kinds", ())
        )
        values["planner_observations"] = tuple(
            DbPlannerObservation.from_dict(item)
            for item in values.get("planner_observations", ())
        )
        return cls(**values)


class DbAgentPlanner(Protocol):
    """Protocol implemented by DB agent planners."""

    async def plan(self, state: DbLoopState) -> DbPlannerDecision:
        """Return the next planner decision for a persisted DB loop state."""


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError(
            "DB planner protocol mappings must be JSON serializable"
        ) from exc
    return copied


def _tuple_json_dicts(
    values: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(_json_dict(value) for value in values)


def _tuple_strings(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    items = tuple(values)
    for value in items:
        if not isinstance(value, str):
            raise TypeError("values must be strings")
    return items
