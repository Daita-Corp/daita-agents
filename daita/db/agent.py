"""
User-facing facade for the new database runtime.
"""

from __future__ import annotations

import re
from typing import Any

from .monitor_commands import DbMonitorCommandService
from .models import DbOperationResult, DbRequest, DbRuntimeInspection
from .monitors import DbMonitor, DbMonitorInspection
from .runtime import DbRuntime


class DbAgent:
    """Facade returned by the future `Agent.from_db()` implementation."""

    def __init__(self, *, runtime: DbRuntime, name: str | None = None) -> None:
        self.runtime = runtime
        self.name = name
        self._monitor_commands = DbMonitorCommandService(runtime)

    @property
    def operations(self) -> tuple[DbOperationResult, ...]:
        """Typed operation results retained by the runtime."""
        return self.runtime.operation_results

    @property
    def audit_log(self) -> tuple[dict[str, Any], ...]:
        """Redacted operation audit summaries retained by the runtime."""
        return self.runtime.audit_log

    async def run(self, prompt: str, **kwargs) -> str:
        """Run a DB request and return the synthesized answer string."""
        result = await self.run_detailed(prompt, **kwargs)
        return result.answer or ""

    async def run_detailed(self, prompt: str, **kwargs) -> DbOperationResult:
        """Run a DB request and return the typed operation result."""
        request = _request_from_kwargs(prompt, kwargs)
        monitor_result = await self._monitor_commands.run(request)
        if monitor_result is not None:
            return monitor_result
        return await self.runtime.run(request)

    async def describe(self) -> DbRuntimeInspection:
        """Return runtime diagnostics for inspection."""
        return await self.runtime.inspect()

    async def monitor(
        self,
        *,
        name: str,
        schedule: str | dict[str, Any] | None = None,
        watch: str | tuple[str, ...] = "",
        trigger: str | dict[str, Any] | None = None,
        then: str | tuple[str, ...] | list[str] = (),
        monitor_id: str | None = None,
        description: str = "",
        source_scope: tuple[str, ...] = (),
        stream: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        budgets: dict[str, Any] | None = None,
        owner: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DbMonitor:
        """Create a durable DB monitor from typed Phase 1 fields."""
        monitor = DbMonitor(
            id=monitor_id or _monitor_id_from_name(name),
            name=name,
            description=description or _description_from_watch(watch),
            status="active",
            source_scope=source_scope,
            schedule=_schedule_dict(schedule),
            stream=stream,
            trigger=_trigger_dict(trigger, schedule=schedule),
            observation_plan=_observation_plan(watch),
            action_plan=_action_plan(then),
            policy=dict(policy or {}),
            budgets=dict(budgets or {}),
            owner=dict(owner or {}),
            metadata=dict(metadata or {}),
        )
        return await self.runtime.create_monitor(monitor)

    async def create_monitor(self, monitor: DbMonitor) -> DbMonitor:
        """Persist a prebuilt DB monitor definition."""
        return await self.runtime.create_monitor(monitor)

    async def list_monitors(
        self, *, status: str | None = None
    ) -> tuple[DbMonitor, ...]:
        """List durable DB monitor definitions."""
        return await self.runtime.list_monitors(status=status)

    async def inspect_monitor(self, monitor_id: str) -> DbMonitorInspection | None:
        """Return a monitor definition with state and run summaries."""
        return await self.runtime.inspect_monitor(monitor_id)

    async def update_monitor(
        self,
        monitor_id: str,
        patch: dict[str, Any],
    ) -> DbMonitor:
        """Patch a durable DB monitor definition."""
        return await self.runtime.update_monitor(monitor_id, patch)

    async def pause_monitor(
        self,
        monitor_id: str,
        *,
        paused_until: str | None = None,
    ) -> DbMonitor:
        """Pause a durable DB monitor."""
        return await self.runtime.pause_monitor(
            monitor_id,
            paused_until=paused_until,
        )

    async def resume_monitor(self, monitor_id: str) -> DbMonitor:
        """Resume a durable DB monitor."""
        return await self.runtime.resume_monitor(monitor_id)

    async def delete_monitor(self, monitor_id: str) -> DbMonitor:
        """Delete a durable DB monitor definition."""
        return await self.runtime.delete_monitor(monitor_id)

    async def list_monitor_approvals(
        self,
        *,
        monitor_id: str | None = None,
        monitor_run_id: str | None = None,
        pending_only: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        """List pending monitor approval requests."""
        return await self.runtime.list_monitor_approvals(
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            pending_only=pending_only,
        )

    async def approve_monitor_approval(self, approval_id: str):
        """Approve a monitor approval through the runtime approval channel."""
        return await self.runtime.approve_monitor_approval(approval_id)

    async def reject_monitor_approval(self, approval_id: str):
        """Reject a monitor approval through the runtime approval channel."""
        return await self.runtime.reject_monitor_approval(approval_id)

    async def cancel_monitor_approval(self, approval_id: str):
        """Cancel a monitor approval through the runtime approval channel."""
        return await self.runtime.cancel_monitor_approval(approval_id)

    async def stop(self) -> None:
        """Release runtime resources."""
        await self.runtime.teardown()

    async def teardown(self) -> None:
        """Alias for framework code that manages runtimes directly."""
        await self.stop()

    async def stream(self, prompt: str, **kwargs):
        """Streaming will be implemented once DB synthesis exists."""
        result = await self.run_detailed(prompt, **kwargs)
        yield result


def _request_from_kwargs(prompt: str, kwargs: dict[str, Any]) -> DbRequest:
    values = dict(kwargs)
    metadata = dict(values.pop("metadata", {}) or {})
    metadata.update(values)
    return DbRequest(
        prompt=prompt,
        user_id=metadata.pop("user_id", None),
        session_id=metadata.pop("session_id", None),
        source_scope=tuple(metadata.pop("source_scope", ()) or ()),
        mode=metadata.pop("mode", None),
        requested_capabilities=tuple(metadata.pop("requested_capabilities", ()) or ()),
        constraints=dict(metadata.pop("constraints", {}) or {}),
        metadata=metadata,
    )


def _monitor_id_from_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return normalized or "db_monitor"


def _description_from_watch(watch: str | tuple[str, ...]) -> str:
    if isinstance(watch, str):
        return watch
    return ", ".join(watch)


def _schedule_dict(schedule: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, str):
        return {"expression": schedule}
    return dict(schedule)


def _trigger_dict(
    trigger: str | dict[str, Any] | None,
    *,
    schedule: str | dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(trigger, dict):
        return dict(trigger)
    if isinstance(trigger, str):
        return {"type": "condition", "expression": trigger}
    if schedule is not None:
        return {"type": "schedule", "expression": "always on schedule"}
    return {"type": "manual", "expression": "manual tick"}


def _observation_plan(watch: str | tuple[str, ...]) -> dict[str, Any]:
    if isinstance(watch, str):
        watches = [watch] if watch else []
    else:
        watches = list(watch)
    return {"watch": watches}


def _action_plan(then: str | tuple[str, ...] | list[str]) -> dict[str, Any]:
    if isinstance(then, str):
        steps = [then] if then else []
    else:
        steps = list(then)
    return {"steps": [{"kind": "instruction", "instruction": step} for step in steps]}
