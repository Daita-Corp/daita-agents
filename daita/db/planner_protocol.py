"""Typed protocol for DB model-planner actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

ALLOWED_DB_PLANNER_ACTION_KINDS: tuple[str, ...] = (
    "clarify",
    "inspect_schema",
    "register_catalog_source",
    "find_relationship_paths",
    "build_planning_context",
    "search_column_values",
    "profile_column_values",
    "register_column_values",
    "resolve_column_value_hints",
    "propose_sql_read",
    "repair_query_plan",
    "propose_sql_write",
    "execute_validated_read",
    "execute_validated_write",
    "recall_memory",
    "write_memory",
    "inspect_monitor",
    "update_monitor",
    "execute_monitor",
    "synthesize",
    "finish",
)

CONTROL_DB_PLANNER_ACTION_KINDS: frozenset[str] = frozenset({"clarify", "finish"})


@dataclass(frozen=True)
class DbPlannerAction:
    """One concrete action proposed by a DB planner."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    action_id: str | None = None
    rationale: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_action_kind(self.kind)
        object.__setattr__(self, "payload", _json_dict(self.payload))
        object.__setattr__(self, "notes", _tuple_strings(self.notes))

    @property
    def executable(self) -> bool:
        return self.kind not in CONTROL_DB_PLANNER_ACTION_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "action_id": self.action_id,
            "rationale": self.rationale,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbPlannerAction":
        values = dict(data)
        return cls(
            kind=str(values["kind"]),
            payload=dict(values.get("payload") or {}),
            action_id=(
                str(values["action_id"])
                if values.get("action_id") is not None
                else None
            ),
            rationale=(
                str(values["rationale"])
                if values.get("rationale") is not None
                else None
            ),
            notes=tuple(str(item) for item in values.get("notes", ())),
        )


@dataclass(frozen=True)
class DbPlannerObservation:
    """Durable observation returned to the planner loop."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    action_id: str | None = None
    task_ids: tuple[str, ...] = ()
    blocked: bool = False
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("observation kind is required")
        object.__setattr__(self, "payload", _json_dict(self.payload))
        object.__setattr__(self, "task_ids", _tuple_strings(self.task_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "action_id": self.action_id,
            "task_ids": list(self.task_ids),
            "blocked": self.blocked,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbPlannerObservation":
        values = dict(data)
        return cls(
            kind=str(values["kind"]),
            payload=dict(values.get("payload") or {}),
            action_id=(
                str(values["action_id"])
                if values.get("action_id") is not None
                else None
            ),
            task_ids=tuple(str(item) for item in values.get("task_ids", ())),
            blocked=bool(values.get("blocked", False)),
            message=(
                str(values["message"]) if values.get("message") is not None else None
            ),
        )


@dataclass(frozen=True)
class DbPlannerDecision:
    """Planner response for one loop turn."""

    actions: tuple[DbPlannerAction, ...]
    decision_id: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        actions = tuple(
            (
                action
                if isinstance(action, DbPlannerAction)
                else DbPlannerAction.from_dict(action)
            )
            for action in self.actions
        )
        if not actions:
            raise ValueError("planner decision requires at least one action")
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "decision_id": self.decision_id,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DbPlannerDecision":
        values = dict(data)
        return cls(
            actions=tuple(
                DbPlannerAction.from_dict(item) for item in values.get("actions", ())
            ),
            decision_id=(
                str(values["decision_id"])
                if values.get("decision_id") is not None
                else None
            ),
            rationale=(
                str(values["rationale"])
                if values.get("rationale") is not None
                else None
            ),
        )


def db_planner_action_from_dict(data: Mapping[str, Any]) -> DbPlannerAction:
    return DbPlannerAction.from_dict(data)


def db_planner_observation_from_dict(
    data: Mapping[str, Any],
) -> DbPlannerObservation:
    return DbPlannerObservation.from_dict(data)


def db_planner_decision_from_dict(data: Mapping[str, Any]) -> DbPlannerDecision:
    return DbPlannerDecision.from_dict(data)


def _validate_action_kind(kind: str) -> None:
    if kind not in ALLOWED_DB_PLANNER_ACTION_KINDS:
        allowed = ", ".join(ALLOWED_DB_PLANNER_ACTION_KINDS)
        raise ValueError(
            f"unsupported DB planner action kind {kind!r}; allowed: {allowed}"
        )


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied)
    except TypeError as exc:
        raise TypeError("planner protocol mappings must be JSON serializable") from exc
    return copied


def _tuple_strings(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    items = tuple(values)
    for item in items:
        if not isinstance(item, str):
            raise TypeError("planner protocol string collections must contain strings")
    return items
