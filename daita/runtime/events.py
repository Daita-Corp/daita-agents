"""Runtime-native streaming event envelopes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .primitives import RuntimeEvent, RuntimeEventType

_DEFAULT_QUEUE_SIZE = 256
_MIN_QUEUE_SIZE = 1
_MAX_QUEUE_SIZE = 10_000
_SUBSCRIPTION_FINISHED = object()


class RuntimeEventSubscription:
    """One bounded, operation-scoped in-process event subscription."""

    def __init__(
        self,
        *,
        broker: "RuntimeEventBroker",
        operation_id: str,
        queue_size: int,
    ) -> None:
        self.operation_id = operation_id
        self._broker = broker
        self._queue: asyncio.Queue[RuntimeEvent | object] = asyncio.Queue(
            maxsize=queue_size
        )
        self._dropped_count = 0
        self._closed = False

    @property
    def queue_size(self) -> int:
        """Return the configured queue bound."""
        return self._queue.maxsize

    @property
    def pending_count(self) -> int:
        """Return the current bounded queue depth."""
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        """Return the number of events dropped since the last diagnostic read."""
        return self._dropped_count

    @property
    def closed(self) -> bool:
        return self._closed

    async def get(self) -> RuntimeEvent | None:
        """Wait for the next delivered event."""
        event = await self._queue.get()
        return event if isinstance(event, RuntimeEvent) else None

    def get_nowait(self) -> RuntimeEvent | None:
        """Return the next queued event without waiting."""
        event = self._queue.get_nowait()
        return event if isinstance(event, RuntimeEvent) else None

    def take_dropped_count(self) -> int:
        """Consume the current dropped-event count for one diagnostic."""
        count = self._dropped_count
        self._dropped_count = 0
        return count

    def close(self) -> None:
        """Remove this subscription from its broker."""
        if self._closed:
            return
        self._closed = True
        self._broker.unsubscribe(self)

    def finish(self) -> None:
        """Wake one waiting consumer after its producer has finished."""
        try:
            self._queue.put_nowait(_SUBSCRIPTION_FINISHED)
        except asyncio.QueueFull:
            # A waiting get already owns an event when the bounded queue is full.
            pass

    def _deliver(self, event: RuntimeEvent) -> None:
        if self._closed or event.operation_id != self.operation_id:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped_count += 1


class RuntimeEventBroker:
    """Bounded in-process delivery for durably persisted runtime events.

    The broker is deliberately non-authoritative and non-distributed. Runtime
    stores remain the source of truth; subscribers only receive events that the
    kernel publishes after persistence succeeds.
    """

    def __init__(self, *, default_queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._validate_queue_size(default_queue_size)
        self.default_queue_size = default_queue_size
        self._subscriptions: dict[str, set[RuntimeEventSubscription]] = {}

    @property
    def subscriber_count(self) -> int:
        """Return the number of active in-process subscriptions."""
        return sum(len(items) for items in self._subscriptions.values())

    def subscribe(
        self,
        operation_id: str,
        *,
        queue_size: int | None = None,
    ) -> RuntimeEventSubscription:
        """Subscribe to events for exactly one operation ID."""
        selected_size = self.default_queue_size if queue_size is None else queue_size
        self._validate_queue_size(selected_size)
        subscription = RuntimeEventSubscription(
            broker=self,
            operation_id=operation_id,
            queue_size=selected_size,
        )
        self._subscriptions.setdefault(operation_id, set()).add(subscription)
        return subscription

    def unsubscribe(self, subscription: RuntimeEventSubscription) -> None:
        """Remove one subscription without affecting other operation IDs."""
        subscriptions = self._subscriptions.get(subscription.operation_id)
        if subscriptions is None:
            return
        subscriptions.discard(subscription)
        if not subscriptions:
            self._subscriptions.pop(subscription.operation_id, None)

    async def publish(self, event: RuntimeEvent) -> None:
        """Deliver without blocking runtime execution on slow consumers."""
        for subscription in tuple(self._subscriptions.get(event.operation_id, ())):
            subscription._deliver(event)

    @staticmethod
    def _validate_queue_size(queue_size: int) -> None:
        if (
            not isinstance(queue_size, int)
            or isinstance(queue_size, bool)
            or not _MIN_QUEUE_SIZE <= queue_size <= _MAX_QUEUE_SIZE
        ):
            raise ValueError("queue_size must be an integer from 1 through 10,000")


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
