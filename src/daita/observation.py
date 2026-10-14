"""Define bounded execution events and deliver them to observers best-effort."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from ._json import FrozenJsonObject, FrozenJsonValue, canonical_json

_MAX_IDENTIFIER_CHARACTERS = 256
_MAX_DATA_KEY_CHARACTERS = 128
_MAX_DATA_STRING_CHARACTERS = 1_024


@dataclass(frozen=True, slots=True)
class SupervisorStatus:
    """Payload-free health of one background supervisor."""

    name: str
    state: Literal["running", "retrying", "failed", "stopped"]
    consecutive_failures: int = 0
    failure_code: str | None = None


class AgentEventKind(str, Enum):
    ADMISSION_UPDATED = "admission.updated"
    RUN_STARTED = "run.started"
    MODEL_TEXT_DELTA = "model.text_delta"
    MODEL_COMPLETED = "model.completed"
    TOOL_STARTED = "tool.started"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    TOOL_COMPLETED = "tool.completed"
    ANALYSIS_UPDATED = "analysis.updated"
    RUN_COMPLETED = "run.completed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """One bounded, immutable, non-durable observation of execution activity."""

    kind: AgentEventKind
    occurred_at: datetime
    run_id: str
    conversation_id: str
    data: FrozenJsonObject
    run_origin: str = "user"

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
        _bounded_identifier(self.run_origin, "event run_origin")
        data = FrozenJsonObject.from_mapping(self.data)
        _validate_data(data)
        object.__setattr__(self, "data", data)


class AgentObserver(Protocol):
    """Synchronous observer that must return promptly."""

    def __call__(self, event: AgentEvent, /) -> None: ...


class PrivateDiagnosticSink:
    """Opt-in payload-free JSONL outside the source repository; no home storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("diagnostic path must be absolute")
        repository = Path(__file__).resolve().parents[2]
        if (
            repository / "pyproject.toml"
        ).exists() and self.path.resolve().is_relative_to(repository):
            raise ValueError("diagnostics must remain outside the repository")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)

    def __call__(self, event: AgentEvent, /) -> None:
        if event.kind not in {
            AgentEventKind.RUN_STARTED,
            AgentEventKind.MODEL_COMPLETED,
            AgentEventKind.TOOL_COMPLETED,
            AgentEventKind.RUN_COMPLETED,
        }:
            return
        record = {
            "kind": event.kind.value,
            "occurred_at": event.occurred_at.isoformat(),
            "run_id": event.run_id,
            "conversation_id": event.conversation_id,
            "run_origin": event.run_origin,
            "data": event.data,
        }
        descriptor = os.open(
            self.path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.write(descriptor, (canonical_json(record) + "\n").encode())
        finally:
            os.close(descriptor)


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


__all__ = ["AgentEvent", "AgentEventKind", "AgentObserver", "PrivateDiagnosticSink"]
