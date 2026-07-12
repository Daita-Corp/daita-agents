"""
User-facing facade for the new database runtime.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
import re
from dataclasses import replace
from typing import Any
from uuid import uuid4

from daita.agents.conversation import ConversationHistory
from daita.core.tracing import TraceType, get_trace_manager
from daita.runtime import OperationStatus, RuntimeEvent, RuntimeStreamEvent

from .context_projection import (
    ProjectionContext,
    ProjectionMode,
    project_runtime_stream_drop_diagnostic,
    project_runtime_stream_event,
    project_runtime_stream_terminal,
)
from .models import DbOperationResult, DbRequest, DbRuntimeInspection
from .monitors import DbMonitor, DbMonitorInspection
from .runtime import DbRuntime
from .session_context import DbSessionContextBuilder


class DbAgent:
    """Facade returned by the future `Agent.from_db()` implementation."""

    def __init__(
        self,
        *,
        runtime: DbRuntime,
        name: str | None = None,
        default_history: ConversationHistory | None = None,
    ) -> None:
        self.runtime = runtime
        self.name = name
        self._default_history = default_history

    @property
    def operations(self) -> tuple[DbOperationResult, ...]:
        """Typed operation results retained by the runtime."""
        return self.runtime.operation_results

    @property
    def audit_log(self) -> tuple[dict[str, Any], ...]:
        """Redacted operation audit summaries retained by the runtime."""
        return self.runtime.audit_log

    async def run(
        self,
        prompt: str,
        *,
        history: ConversationHistory | None = None,
        **kwargs,
    ) -> str:
        """Run a DB request and return the synthesized answer string."""
        result = await self.run_detailed(prompt, history=history, **kwargs)
        return result.answer or ""

    async def run_detailed(
        self,
        prompt: str,
        *,
        history: ConversationHistory | None = None,
        **kwargs,
    ) -> DbOperationResult:
        """Run a DB request and return the typed operation result."""
        return await self._run_detailed(
            prompt,
            history=history,
            kwargs=kwargs,
        )

    async def _run_detailed(
        self,
        prompt: str,
        *,
        history: ConversationHistory | None,
        kwargs: dict[str, Any],
        operation_id: str | None = None,
    ) -> DbOperationResult:
        active_history = self._resolve_history(history, kwargs)
        if active_history is not None:
            active_history._set_workspace(self._history_workspace())
        request = _request_from_kwargs(prompt, kwargs)
        if active_history is not None and request.session_id is None:
            request = replace(request, session_id=active_history.session_id)
        request = await self._with_session_context(request, active_history)
        trace_manager = get_trace_manager()
        async with trace_manager.span(
            "from_db_operation",
            TraceType.AGENT_EXECUTION,
            agent_id=self.name,
            runtime_id=self.runtime.runtime_id,
            runtime_kind=self.runtime.runtime_kind,
            agent_name=self.name,
            prompt=request.prompt,
            user_id=request.user_id,
            session_id=request.session_id,
            mode=request.mode,
            stateful_history=active_history is not None,
        ):
            if operation_id is None:
                result = await self.runtime.run(request)
            else:
                result = await self.runtime.run(
                    request,
                    operation_id=operation_id,
                )
            if active_history is not None:
                await active_history.add_turn(prompt, result.answer or "")
            return result

    async def describe(self) -> DbRuntimeInspection:
        """Return runtime diagnostics for inspection."""
        return await self.runtime.inspect()

    async def monitor(
        self,
        *,
        name: str,
        schedule: str | dict[str, Any] | None = None,
        watch: str | tuple[str, ...] = "",
        observation_plan: dict[str, Any] | None = None,
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
        """Create a durable DB monitor from typed executable fields."""
        if observation_plan is None:
            raise ValueError(
                "DbAgent.monitor() requires an executable observation_plan. "
                "Use run_detailed() for prompt-planned monitors."
            )
        monitor = DbMonitor(
            id=monitor_id or _monitor_id_from_name(name),
            name=name,
            description=description or _description_from_watch(watch),
            status="active",
            source_scope=source_scope,
            schedule=_schedule_dict(schedule),
            stream=stream,
            trigger=_trigger_dict(trigger, schedule=schedule),
            observation_plan=dict(observation_plan),
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

    async def stream(
        self,
        prompt: str,
        *,
        history: ConversationHistory | None = None,
        **kwargs,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        """Stream bounded, projected runtime progress for one DB operation."""
        operation_id = f"db-op-{uuid4()}"
        subscription = self.runtime.kernel.event_broker.subscribe(operation_id)
        projection = ProjectionContext(mode=ProjectionMode.PUBLIC_RESULT)
        run_task = asyncio.create_task(
            self._run_detailed(
                prompt,
                history=history,
                kwargs=dict(kwargs),
                operation_id=operation_id,
            ),
            name=f"db-stream-run:{operation_id}",
        )
        event_waiter: asyncio.Task[Any] | None = None
        cancelled_by_consumer = False
        terminal_delivered = False

        def project_event(event: RuntimeEvent) -> RuntimeStreamEvent:
            nonlocal terminal_delivered
            projected = project_runtime_stream_event(event, projection)
            terminal_delivered = (
                terminal_delivered or projected.payload.get("terminal") is True
            )
            return projected

        try:
            while not run_task.done():
                event_waiter = asyncio.create_task(subscription.get())
                done, _ = await asyncio.wait(
                    (run_task, event_waiter),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if event_waiter in done:
                    event = event_waiter.result()
                    event_waiter = None
                    if event is None:
                        break
                    yield project_event(event)
                    if subscription.pending_count == 0:
                        dropped_count = subscription.take_dropped_count()
                        if dropped_count:
                            yield project_runtime_stream_drop_diagnostic(
                                operation_id=operation_id,
                                runtime_id=self.runtime.runtime_id,
                                runtime_kind=self.runtime.runtime_kind,
                                dropped_count=dropped_count,
                            )
                    continue
                subscription.finish()
                event = await event_waiter
                event_waiter = None
                if event is not None:
                    yield project_event(event)
                    if subscription.pending_count == 0:
                        dropped_count = subscription.take_dropped_count()
                        if dropped_count:
                            yield project_runtime_stream_drop_diagnostic(
                                operation_id=operation_id,
                                runtime_id=self.runtime.runtime_id,
                                runtime_kind=self.runtime.runtime_kind,
                                dropped_count=dropped_count,
                            )

            while subscription.pending_count:
                event = subscription.get_nowait()
                if event is not None:
                    yield project_event(event)
            dropped_count = subscription.take_dropped_count()
            if dropped_count:
                yield project_runtime_stream_drop_diagnostic(
                    operation_id=operation_id,
                    runtime_id=self.runtime.runtime_id,
                    runtime_kind=self.runtime.runtime_kind,
                    dropped_count=dropped_count,
                )

            result = await run_task
            if not terminal_delivered:
                yield project_runtime_stream_terminal(
                    result,
                    runtime_id=self.runtime.runtime_id,
                    runtime_kind=self.runtime.runtime_kind,
                )
        except (asyncio.CancelledError, GeneratorExit):
            cancelled_by_consumer = True
            raise
        finally:
            if event_waiter is not None and not event_waiter.done():
                event_waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await event_waiter
            if not run_task.done():
                run_task.cancel()
            with suppress(BaseException):
                await run_task
            if cancelled_by_consumer:
                with suppress(Exception):
                    await self._preserve_stream_operation_for_resume(operation_id)
            subscription.close()

    async def _preserve_stream_operation_for_resume(self, operation_id: str) -> None:
        operation = await self.runtime.store.load_operation(operation_id)
        if operation is None or operation.status in {
            OperationStatus.BLOCKED,
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
        }:
            return
        await self.runtime.kernel.block_operation(
            operation_id,
            message="Operation interrupted and preserved for resume.",
            payload={"status": OperationStatus.BLOCKED.value},
        )

    def _resolve_history(
        self,
        explicit_history: ConversationHistory | None,
        kwargs: dict[str, Any],
    ) -> ConversationHistory | None:
        if explicit_history is not None:
            return explicit_history
        if "session_id" in kwargs and kwargs.get("session_id") is not None:
            return None
        return self._default_history

    def _history_workspace(self) -> str:
        return self.name or self.runtime.runtime_id

    async def _with_session_context(
        self,
        request: DbRequest,
        history: ConversationHistory | None,
    ) -> DbRequest:
        messages = history.messages if history is not None else ()
        context = await DbSessionContextBuilder(self.runtime).build(
            request,
            conversation_messages=messages,
        )
        return replace(request, session_context=context.to_request_dict())


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


def _action_plan(then: str | tuple[str, ...] | list[str]) -> dict[str, Any]:
    if isinstance(then, str):
        steps = [then] if then else []
    else:
        steps = list(then)
    return {"steps": [{"kind": "instruction", "instruction": step} for step in steps]}
