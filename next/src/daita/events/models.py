"""Canonical provider-neutral runtime-event records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from .._json import FrozenJsonObject


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """One immutable state-correlated event, before durable cursor assignment."""

    id: str
    type: str
    agent_id: str
    operation_id: str | None
    created_at: datetime
    session_id: str | None = None
    turn_id: str | None = None
    model_call_id: str | None = None
    call_id: str | None = None
    task_id: str | None = None
    evidence_id: str | None = None
    capability_id: str | None = None
    executor_id: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.id, "event id")
        _required_text(self.type, "event type")
        _required_text(self.agent_id, "event agent_id")
        _aware(self.created_at, "event created_at")
        for field_name, value in (
            ("operation_id", self.operation_id),
            ("session_id", self.session_id),
            ("turn_id", self.turn_id),
            ("model_call_id", self.model_call_id),
            ("call_id", self.call_id),
            ("task_id", self.task_id),
            ("evidence_id", self.evidence_id),
            ("capability_id", self.capability_id),
            ("executor_id", self.executor_id),
        ):
            if value is not None:
                _required_text(value, f"event {field_name}")
        object.__setattr__(self, "payload", FrozenJsonObject.from_mapping(self.payload))
