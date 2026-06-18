"""Runtime-native monitor definitions and operation creation."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping
from uuid import uuid4

from .kernel import RuntimeKernel, RuntimeKernelExecutionError, TaskExecutionResult
from .primitives import (
    Capability,
    ContextAudience,
    Operation,
    RuntimeEvent,
    RuntimeEventType,
    Task,
)
from .status import reconcile_operation_status


@dataclass(frozen=True)
class MonitorSpec:
    """Runtime-owned monitor declaration."""

    id: str
    name: str
    source_capability_id: str | None = None
    source_context_provider_id: str | None = None
    schedule: dict[str, Any] | None = None
    stream: dict[str, Any] | None = None
    trigger: dict[str, Any] = field(default_factory=dict)
    action_capability_id: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    cooldown_seconds: float | None = None
    cursor: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("monitor id is required")
        if not self.name:
            raise ValueError("monitor name is required")
        object.__setattr__(
            self, "schedule", None if self.schedule is None else dict(self.schedule)
        )
        object.__setattr__(
            self, "stream", None if self.stream is None else dict(self.stream)
        )
        object.__setattr__(self, "trigger", dict(self.trigger))
        object.__setattr__(self, "action_input", dict(self.action_input))
        object.__setattr__(self, "cursor", dict(self.cursor))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.cooldown_seconds is not None and self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")


@dataclass(frozen=True)
class MonitorTickResult:
    """Result from one monitor tick."""

    monitor_id: str
    triggered: bool
    operation_id: str | None = None
    task_ids: tuple[str, ...] = ()
    events: tuple[RuntimeEvent, ...] = ()


@dataclass
class MonitorState:
    """Lightweight in-memory monitor cursor state."""

    monitor_id: str
    status: str = "created"
    cursor: dict[str, Any] = field(default_factory=dict)
    last_tick_at: float | None = None
    last_triggered_at: float | None = None
    last_value_summary: Any = None
    cooldown_until: float | None = None
    consecutive_failures: int = 0
    last_error: str | None = None


class MonitorRuntime:
    """Turns monitor triggers into persisted runtime operations and tasks."""

    def __init__(self, *, kernel: RuntimeKernel) -> None:
        self.kernel = kernel
        self.store = kernel.store
        self.registry = kernel.extension_registry
        self._states: dict[str, MonitorState] = {}

    def state_for(self, monitor_id: str) -> MonitorState | None:
        """Return the current in-memory monitor state."""
        return self._states.get(monitor_id)

    async def tick(
        self,
        spec: MonitorSpec,
        *,
        value: Any = None,
        action_input: Mapping[str, Any] | None = None,
        execute_actions: bool = True,
        raise_action_errors: bool = False,
        context: Mapping[str, Any] | None = None,
    ) -> MonitorTickResult:
        """Evaluate one monitor tick and create runtime work when triggered."""
        state = self._states.setdefault(
            spec.id,
            MonitorState(monitor_id=spec.id, cursor=dict(spec.cursor)),
        )
        now = time.time()
        state.status = "running"
        state.last_tick_at = now
        events: list[RuntimeEvent] = [
            self._event(
                RuntimeEventType.MONITOR_TICKED,
                spec=spec,
                operation_id=f"monitor-{spec.id}",
                message=f"Monitor {spec.id} ticked.",
                payload={"monitor": _monitor_payload(spec)},
            )
        ]
        await self.store.append_event(events[-1])

        try:
            observed_value = await self._observe_value(
                spec,
                value=value,
                context=context,
            )
            state.last_value_summary = summarize_monitor_value(observed_value)
            if state.cooldown_until is not None and state.cooldown_until > now:
                skipped = self._event(
                    RuntimeEventType.MONITOR_SKIPPED,
                    spec=spec,
                    operation_id=f"monitor-{spec.id}",
                    message=f"Monitor {spec.id} skipped due to cooldown.",
                    payload={
                        "cooldown_until": state.cooldown_until,
                        "value_summary": state.last_value_summary,
                    },
                )
                await self.store.append_event(skipped)
                events.append(skipped)
                return MonitorTickResult(spec.id, False, events=tuple(events))

            if not monitor_trigger_matches(observed_value, spec.trigger):
                skipped = self._event(
                    RuntimeEventType.MONITOR_SKIPPED,
                    spec=spec,
                    operation_id=f"monitor-{spec.id}",
                    message=f"Monitor {spec.id} did not trigger.",
                    payload={"value_summary": state.last_value_summary},
                )
                await self.store.append_event(skipped)
                events.append(skipped)
                return MonitorTickResult(spec.id, False, events=tuple(events))

            operation, tasks, execution_events = await self._create_action_work(
                spec,
                value=observed_value,
                action_input=dict(action_input or {}),
                execute_actions=execute_actions,
                raise_action_errors=raise_action_errors,
                context=context,
            )
            state.last_triggered_at = now
            if spec.cooldown_seconds:
                state.cooldown_until = now + spec.cooldown_seconds
            triggered = self._event(
                RuntimeEventType.MONITOR_TRIGGERED,
                spec=spec,
                operation_id=operation.id,
                message=f"Monitor {spec.id} triggered operation {operation.id}.",
                payload={
                    "monitor": _monitor_payload(spec),
                    "value_summary": state.last_value_summary,
                    "task_ids": [task.id for task in tasks],
                },
            )
            await self.store.append_event(triggered)
            events.append(triggered)
            events.extend(execution_events)
            state.consecutive_failures = 0
            state.last_error = None
            return MonitorTickResult(
                monitor_id=spec.id,
                triggered=True,
                operation_id=operation.id,
                task_ids=tuple(task.id for task in tasks),
                events=tuple(events),
            )
        except Exception as exc:
            state.status = "error"
            state.consecutive_failures += 1
            state.last_error = f"{type(exc).__name__}: {exc}"
            error_event = self._event(
                RuntimeEventType.ERROR,
                spec=spec,
                operation_id=f"monitor-{spec.id}",
                message=f"Monitor {spec.id} failed.",
                payload={"error": {"type": type(exc).__name__, "message": str(exc)}},
            )
            await self.store.append_event(error_event)
            events.append(error_event)
            raise

    async def _observe_value(
        self,
        spec: MonitorSpec,
        *,
        value: Any,
        context: Mapping[str, Any] | None,
    ) -> Any:
        if spec.source_capability_id is not None:
            result = await self.kernel.execute_capability(
                spec.source_capability_id,
                input=dict(spec.cursor),
                operation_type="monitor.source",
                task_metadata={"monitor_id": spec.id, "monitor_role": "source"},
                context={"monitor_id": spec.id, **dict(context or {})},
            )
            return [item.payload for item in result.evidence]
        if spec.source_context_provider_id is not None:
            provider = self.registry.get_context_provider(
                spec.source_context_provider_id
            )
            block = await provider.render(
                {
                    "monitor_id": spec.id,
                    "monitor_name": spec.name,
                    "cursor": dict(spec.cursor),
                },
                ContextAudience.OPERATION_INSPECTOR,
                2000,
            )
            return None if block is None else block.to_dict()
        return value

    async def _create_action_work(
        self,
        spec: MonitorSpec,
        *,
        value: Any,
        action_input: dict[str, Any],
        execute_actions: bool,
        raise_action_errors: bool,
        context: Mapping[str, Any] | None,
    ) -> tuple[Operation, tuple[Task, ...], tuple[RuntimeEvent, ...]]:
        capability = (
            self.registry.get_capability(spec.action_capability_id)
            if spec.action_capability_id is not None
            else None
        )
        operation_type = (
            sorted(capability.operation_types)[0]
            if capability is not None and capability.operation_types
            else "monitor.triggered"
        )
        operation = await self.kernel.create_operation(
            operation_id=f"monitor-op-{uuid4()}",
            operation_type=operation_type,
            request={
                "monitor_id": spec.id,
                "monitor_name": spec.name,
                "value": value,
                "action_input": {
                    **dict(spec.action_input),
                    **action_input,
                },
            },
            required_evidence=(
                capability.output_evidence if capability is not None else ()
            ),
            metadata={
                "monitor_id": spec.id,
                "monitor_name": spec.name,
                "runtime_id": self.kernel.runtime_id,
                "runtime_kind": self.kernel.runtime_kind,
                **spec.metadata,
            },
        )
        if capability is None:
            return operation, (), ()
        task = await self.kernel.plan_task(
            operation_id=operation.id,
            capability_id=capability.id,
            owner=capability.owner,
            input={
                "monitor_id": spec.id,
                "monitor_name": spec.name,
                "value": value,
                **dict(spec.action_input),
                **action_input,
            },
            metadata={
                "monitor_id": spec.id,
                "reason": "monitor_action",
                **spec.metadata,
            },
        )
        if not execute_actions:
            return operation, (task,), ()
        try:
            execution = await self.kernel.execute_task(
                task.id,
                context={"monitor_id": spec.id, **dict(context or {})},
            )
            await reconcile_operation_status(self.kernel, operation.id)
            return operation, (task,), execution.events
        except RuntimeKernelExecutionError as exc:
            await reconcile_operation_status(self.kernel, operation.id)
            if raise_action_errors:
                raise
            return operation, (task,), exc.result.events if exc.result else ()

    def _event(
        self,
        type: RuntimeEventType,
        *,
        spec: MonitorSpec,
        operation_id: str,
        message: str,
        payload: Mapping[str, Any] | None = None,
    ) -> RuntimeEvent:
        return RuntimeEvent(
            type=type,
            runtime_id=self.kernel.runtime_id,
            runtime_kind=self.kernel.runtime_kind,
            operation_id=operation_id,
            message=message,
            payload=dict(payload or {}),
        )


def monitor_trigger_matches(value: Any, trigger: Mapping[str, Any]) -> bool:
    """Evaluate the generic monitor predicate language."""
    if not trigger:
        return True
    candidate = (
        _extract_path(value, str(trigger.get("path") or ""))
        if trigger.get("path")
        else value
    )
    operator = trigger.get("operator")
    if operator == "count_gt":
        try:
            threshold = int(trigger.get("value", 0))
        except (TypeError, ValueError):
            return False
        if not (_candidate_count(candidate) > threshold):
            return False
    elif operator == "count_gte":
        try:
            threshold = int(trigger.get("value", 0))
        except (TypeError, ValueError):
            return False
        if not (_candidate_count(candidate) >= threshold):
            return False
    if "equals" in trigger and candidate != trigger["equals"]:
        return False
    if "not_equals" in trigger and candidate == trigger["not_equals"]:
        return False
    if trigger.get("truthy") and not candidate:
        return False
    if "gt" in trigger and not (candidate > trigger["gt"]):
        return False
    if "gte" in trigger and not (candidate >= trigger["gte"]):
        return False
    if "lt" in trigger and not (candidate < trigger["lt"]):
        return False
    if "lte" in trigger and not (candidate <= trigger["lte"]):
        return False
    return True


def _candidate_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        if isinstance(value.get("count"), int):
            return int(value["count"])
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 1


def _extract_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            return None
    return current


def summarize_monitor_value(value: Any) -> Any:
    """Return a compact, JSON-friendly summary of an observed monitor value."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {"type": "object", "keys": sorted(str(key) for key in value.keys())[:20]}
    if isinstance(value, (list, tuple)):
        return {"type": "array", "count": len(value)}
    return {"type": type(value).__name__}


def _monitor_payload(spec: MonitorSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "source_capability_id": spec.source_capability_id,
        "source_context_provider_id": spec.source_context_provider_id,
        "action_capability_id": spec.action_capability_id,
        "schedule": spec.schedule,
        "stream": spec.stream,
        "trigger": spec.trigger,
        "cooldown_seconds": spec.cooldown_seconds,
        "metadata": spec.metadata,
    }
