"""Runtime-native streaming event envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .primitives import RuntimeEvent, RuntimeEventType


@dataclass(frozen=True)
class RuntimeStreamEvent:
    """Delivery envelope for runtime progress subscribers."""

    type: RuntimeEventType
    operation_id: str
    message: str
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
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_runtime_event(cls, event: RuntimeEvent) -> "RuntimeStreamEvent":
        """Project a durable runtime event into a user-delivery envelope."""
        timestamp = (
            datetime.fromtimestamp(event.timestamp)
            if event.timestamp is not None
            else datetime.now()
        )
        return cls(
            type=event.type,
            operation_id=event.operation_id,
            message=event.message,
            runtime_id=event.runtime_id,
            runtime_kind=event.runtime_kind,
            task_id=event.task_id,
            capability_id=event.capability_id,
            executor_id=event.executor_id,
            plugin_id=event.plugin_id,
            policy_id=event.policy_id,
            approval_id=event.approval_id,
            evidence_id=event.evidence_id,
            trace_id=event.trace_id,
            span_id=event.span_id,
            payload=dict(event.payload),
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly delivery shape."""
        return {
            "type": self.type.value,
            "operation_id": self.operation_id,
            "message": self.message,
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
            "payload": dict(self.payload),
            "timestamp": self.timestamp.isoformat(),
        }
