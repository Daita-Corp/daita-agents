"""Typed monitor create intent records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MonitorTargetIntent:
    target_type: str
    name: str | None
    source_scope: tuple[str, ...] = ()
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_scope", tuple(self.source_scope))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        _validate_confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "name": self.name,
            "source_scope": list(self.source_scope),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class MonitorConditionIntent:
    kind: str
    expression: str | None = None
    operator: str | None = None
    value: Any = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expression": self.expression,
            "operator": self.operator,
            "value": self.value,
            "path": self.path,
        }


@dataclass(frozen=True)
class MonitorScheduleIntent:
    kind: str
    interval_seconds: int | None = None
    expression: str | None = None
    timezone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "interval_seconds": self.interval_seconds,
            "expression": self.expression,
            "timezone": self.timezone,
        }

    def to_schedule_dict(self) -> dict[str, Any]:
        if self.kind == "interval" and self.interval_seconds is not None:
            return {"interval_seconds": self.interval_seconds}
        payload: dict[str, Any] = {}
        if self.expression:
            payload["expression"] = self.expression
        if self.timezone:
            payload["timezone"] = self.timezone
        return payload


@dataclass(frozen=True)
class MonitorDeliveryRequest:
    delivery_kind: str
    target: dict[str, Any] = field(default_factory=dict)
    explicit: bool = False
    payload_source: dict[str, Any] = field(
        default_factory=lambda: {"type": "monitor.report"}
    )
    template: str | None = None
    include_observed_rows: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", dict(self.target))
        object.__setattr__(self, "payload_source", dict(self.payload_source))

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivery_kind": self.delivery_kind,
            "target": dict(self.target),
            "explicit": self.explicit,
            "payload_source": dict(self.payload_source),
            "template": self.template,
            "include_observed_rows": self.include_observed_rows,
        }

    def to_action_plan_intent(self) -> dict[str, Any]:
        payload = {
            "delivery_kind": self.delivery_kind,
            "target": dict(self.target),
            "payload_source": dict(self.payload_source),
            "include_observed_rows": self.include_observed_rows,
        }
        if self.template is not None:
            payload["template"] = self.template
        return payload


@dataclass(frozen=True)
class MonitorActionIntent:
    actions: tuple[str, ...] = ()
    steps: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "steps", tuple(dict(step) for step in self.steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "steps": [dict(step) for step in self.steps],
        }


@dataclass(frozen=True)
class MonitorDisplayIntent:
    explicit_name: str | None = None
    suggested_name: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "explicit_name": self.explicit_name,
            "suggested_name": self.suggested_name,
            "description": self.description,
        }


@dataclass(frozen=True)
class MonitorPolicyIntent:
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class MonitorBudgetIntent:
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class MonitorCreateIntent:
    target: MonitorTargetIntent
    condition: MonitorConditionIntent
    schedule: MonitorScheduleIntent | None
    delivery: MonitorDeliveryRequest | None
    action: MonitorActionIntent = field(default_factory=MonitorActionIntent)
    display: MonitorDisplayIntent = field(default_factory=MonitorDisplayIntent)
    policy: MonitorPolicyIntent = field(default_factory=MonitorPolicyIntent)
    budget: MonitorBudgetIntent = field(default_factory=MonitorBudgetIntent)
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "condition": self.condition.to_dict(),
            "schedule": None if self.schedule is None else self.schedule.to_dict(),
            "delivery": None if self.delivery is None else self.delivery.to_dict(),
            "action": self.action.to_dict(),
            "display": self.display.to_dict(),
            "policy": self.policy.to_dict(),
            "budget": self.budget.to_dict(),
            "confidence": self.confidence,
            "diagnostics": dict(self.diagnostics),
        }


def _validate_confidence(value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError("confidence must be between 0 and 1")
