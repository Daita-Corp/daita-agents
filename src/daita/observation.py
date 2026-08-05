"""Best-effort observation records for direct agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from ._json import FrozenJsonObject, FrozenJsonValue

_MAX_IDENTIFIER_CHARACTERS = 256
_MAX_DATA_KEY_CHARACTERS = 128
_MAX_DATA_STRING_CHARACTERS = 1_024


class AgentEventKind(str, Enum):
    RUN_STARTED = "run.started"
    MODEL_TEXT_DELTA = "model.text_delta"
    MODEL_COMPLETED = "model.completed"
    TOOL_STARTED = "tool.started"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    TOOL_COMPLETED = "tool.completed"
    RUN_COMPLETED = "run.completed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """One bounded, immutable, non-durable observation of execution activity."""

    kind: AgentEventKind
    occurred_at: datetime
    run_id: str
    conversation_id: str
    data: FrozenJsonObject

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AgentEventKind):
            raise TypeError("event kind must be AgentEventKind")
        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("event occurred_at must be timezone-aware")
        _bounded_identifier(self.run_id, "event run_id")
        _bounded_identifier(self.conversation_id, "event conversation_id")
        data = FrozenJsonObject.from_mapping(self.data)
        _validate_data(data)
        object.__setattr__(self, "data", data)


class AgentObserver(Protocol):
    """Synchronous observer that must return promptly."""

    def __call__(self, event: AgentEvent, /) -> None: ...


def _emit_safely(observer: AgentObserver | None, event: AgentEvent) -> None:
    if observer is None:
        return
    try:
        observer(event)
    except Exception:
        pass


def _bounded_identifier(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_IDENTIFIER_CHARACTERS
    ):
        raise ValueError(f"{field_name} must be a bounded non-empty string")


def _validate_data(value: FrozenJsonValue, *, key: str | None = None) -> None:
    if (
        key is not None
        and key.endswith("_id")
        and not (key == "cost_rate_schedule_id" and value is None)
    ):
        _bounded_identifier(value, f"event data {key}")
    if key == "duration_ms" and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError("event duration_ms must be a non-negative integer")
    if key == "model_call_index" and (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 1_000_000
    ):
        raise ValueError("event model_call_index must be a bounded positive integer")
    if isinstance(value, FrozenJsonObject):
        for item_key, item in value.items():
            if not item_key or len(item_key) > _MAX_DATA_KEY_CHARACTERS:
                raise ValueError("event data keys must be bounded non-empty strings")
            _validate_data(item, key=item_key)
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_data(item)
        return
    if isinstance(value, str) and len(value) > _MAX_DATA_STRING_CHARACTERS:
        raise ValueError("event data strings must be bounded")


__all__ = ["AgentEvent", "AgentEventKind", "AgentObserver"]
