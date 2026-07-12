"""Durable DB monitor scheduling and tick control plane."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping
from uuid import uuid4

from daita.runtime import (
    Evidence,
    Operation,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
)
from daita.runtime.monitors import MonitorRuntime

from ..monitors import (
    DbMonitor,
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
    DbMonitorStore,
)
from .actions import DbMonitorActionRunner
from .delivery import DbMonitorDeliveryRunner
from .observation import DbMonitorObservationRunner, _observation_plan_kind
from .state import (
    _blocked_reason,
    _coerce_datetime,
    _error_backoff_seconds,
    _is_due,
    _iso,
    _parse_iso,
    _state_for_blocked_decision,
    _state_for_scheduler_decision,
    _state_value_summary,
    _validation_payload,
)
from .triggers import (
    _compile_monitor_spec,
    _cooldown_seconds,
    _required_consecutive_matches,
    _runtime_spec_for_decision,
    _trigger_decision,
)
from .types import (
    DbMonitorObservationBlocked,
    DbMonitorObservationResult,
    DbMonitorSchedulerResult,
)


class DbMonitorScheduler:
    """Find due DB monitors, claim tick leases, and hand ticks to the runner."""

    def __init__(
        self,
        *,
        runtime: Any,
        monitor_store: DbMonitorStore | None = None,
        runner: "DbMonitorRunner | None" = None,
        scheduler_id: str | None = None,
        lease_seconds: float = 60.0,
    ) -> None:
        self.runtime = runtime
        self.monitor_store = monitor_store or runtime.monitor_store
        self.runner = runner or DbMonitorRunner(
            runtime=runtime,
            monitor_store=self.monitor_store,
        )
        self.scheduler_id = scheduler_id or f"db-monitor-scheduler-{uuid4()}"
        self.lease_seconds = lease_seconds

    async def run_once(
        self,
        *,
        now: datetime | str | None = None,
    ) -> tuple[DbMonitorSchedulerResult, ...]:
        """Evaluate one scheduler pass for persisted monitors."""

        tick_time = _coerce_datetime(now)
        results: list[DbMonitorSchedulerResult] = []
        for monitor in await self.monitor_store.list_monitors():
            state = await self.monitor_store.load_monitor_state(monitor.id)
            due, reason = _is_due(monitor, state, tick_time)
            if not due:
                run = await self.runner.record_scheduler_decision(
                    monitor.id,
                    reason=reason,
                    now=tick_time,
                )
                results.append(DbMonitorSchedulerResult(monitor.id, run))
                continue

            lease_id = f"{self.scheduler_id}-{uuid4()}"
            claimed = await self.monitor_store.claim_monitor_tick_lease(
                monitor.id,
                lease_id=lease_id,
                now=_iso(tick_time),
                expires_at=_iso(tick_time + timedelta(seconds=self.lease_seconds)),
            )
            if not claimed:
                run = await self.runner.record_scheduler_decision(
                    monitor.id,
                    reason="lease_lost",
                    now=tick_time,
                )
                results.append(DbMonitorSchedulerResult(monitor.id, run))
                continue

            try:
                run = await self.runner.tick_monitor(
                    monitor.id,
                    now=tick_time,
                    lease_id=lease_id,
                )
                results.append(DbMonitorSchedulerResult(monitor.id, run, claimed=True))
            finally:
                await self.monitor_store.release_monitor_tick_lease(
                    monitor.id,
                    lease_id=lease_id,
                )
        return tuple(results)


class DbMonitorRunner:
    """DB-specific owner for durable monitor tick records."""

    def __init__(
        self,
        *,
        runtime: Any,
        monitor_store: DbMonitorStore | None = None,
        monitor_runtime: MonitorRuntime | None = None,
    ) -> None:
        self.runtime = runtime
        self.monitor_store = monitor_store or runtime.monitor_store
        self.monitor_runtime = monitor_runtime or MonitorRuntime(kernel=runtime.kernel)
        self.observation_runner = DbMonitorObservationRunner(runtime=runtime)
        self.action_runner = DbMonitorActionRunner(
            runtime=runtime,
            monitor_store=self.monitor_store,
        )
        self.delivery_runner = DbMonitorDeliveryRunner(
            runtime=runtime,
            monitor_store=self.monitor_store,
        )

    async def tick_monitor(
        self,
        monitor_id: str,
        *,
        now: datetime | str | None = None,
        lease_id: str | None = None,
    ) -> DbMonitorRun:
        """Load a persisted monitor by ID and execute one durable tick."""

        tick_time = _coerce_datetime(now)
        tick_operation_id = f"monitor-tick-{uuid4()}"
        run_id = f"monitor-run-{uuid4()}"
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)

        gate = _blocked_reason(monitor, state, tick_time)
        if gate is not None:
            return await self._commit_run(
                monitor,
                state,
                operation_id=tick_operation_id,
                run_id=run_id,
                now=tick_time,
                reason=gate,
                status="blocked" if gate.startswith("validation") else "skipped",
                triggered=False,
                state_after=_state_for_blocked_decision(
                    state,
                    gate,
                    tick_time,
                    tick_operation_id,
                ),
                accepted=not gate.startswith("validation"),
                lease_id=lease_id,
            )

        await self._begin_tick_operation(
            monitor,
            state,
            operation_id=tick_operation_id,
            run_id=run_id,
            now=tick_time,
            lease_id=lease_id,
        )
        operation = await self.runtime.store.load_operation(tick_operation_id)
        if operation is None:
            raise RuntimeError(f"monitor tick operation {tick_operation_id} missing")
        observation = None
        try:
            observation = await self.observation_runner.observe(
                monitor,
                state,
                operation,
                run_id=run_id,
                now=tick_time,
            )
        except DbMonitorObservationBlocked as exc:
            return await self._commit_observation_unsuccessful_run(
                monitor,
                state,
                tick_time,
                exc,
                operation_id=tick_operation_id,
                run_id=run_id,
                lease_id=lease_id,
            )
        except Exception as exc:
            return await self._commit_failed_run(
                monitor,
                state,
                tick_time,
                exc,
                operation_id=tick_operation_id,
                run_id=run_id,
                lease_id=lease_id,
                observation=observation,
            )
        spec = _compile_monitor_spec(
            monitor,
            state,
            tick_operation_id=tick_operation_id,
            run_id=run_id,
        )
        value = observation.value
        matched, value_summary, trigger_reason = _trigger_decision(
            value,
            monitor.trigger,
        )
        next_matches = state.consecutive_matches + 1 if matched else 0
        required_matches = _required_consecutive_matches(monitor)
        trigger_ready = matched and next_matches >= required_matches

        try:
            runtime_result = await self.monitor_runtime.tick(
                _runtime_spec_for_decision(spec, trigger_ready),
                value=(
                    {"triggered": True, "value_summary": value_summary}
                    if trigger_ready
                    else False
                ),
                execute_actions=False,
                context={
                    "monitor_id": monitor.id,
                    "db_monitor_phase": 3,
                    "trigger_ready": trigger_ready,
                },
            )
        except Exception as exc:
            return await self._commit_failed_run(
                monitor,
                state,
                tick_time,
                exc,
                operation_id=tick_operation_id,
                run_id=run_id,
                lease_id=lease_id,
            )

        triggered_operation_id = runtime_result.operation_id if trigger_ready else None
        status = "triggered" if trigger_ready else "succeeded"
        cooldown_until = (
            _iso(tick_time + timedelta(seconds=_cooldown_seconds(monitor)))
            if trigger_ready and _cooldown_seconds(monitor) > 0
            else state.cooldown_until
        )
        state_after = DbMonitorState.from_dict(
            {
                **state.to_dict(),
                "cursor": {
                    **state.cursor,
                    "last_scheduler_tick": _iso(tick_time),
                    **dict(observation.evidence.payload.get("cursor_updates") or {}),
                },
                "last_tick_at": _iso(tick_time),
                "last_triggered_at": (
                    _iso(tick_time) if trigger_ready else state.last_triggered_at
                ),
                "last_operation_id": tick_operation_id,
                "last_tick_operation_id": tick_operation_id,
                "last_triggered_operation_id": (
                    triggered_operation_id
                    if trigger_ready
                    else state.last_triggered_operation_id
                ),
                "last_value_summary": _state_value_summary(value_summary),
                "consecutive_matches": next_matches,
                "consecutive_failures": 0,
                "cooldown_until": cooldown_until,
                "paused_until": (
                    None
                    if state.paused_until is not None
                    and _parse_iso(state.paused_until) <= tick_time
                    else state.paused_until
                ),
                "error": None,
            }
        )
        run = await self._commit_run(
            monitor,
            state,
            operation_id=tick_operation_id,
            run_id=run_id,
            now=tick_time,
            reason=trigger_reason,
            status=status,
            triggered=trigger_ready,
            state_after=state_after,
            lease_id=lease_id,
            operation_precreated=True,
            summary={
                "matched": matched,
                "required_consecutive_matches": required_matches,
                "consecutive_matches": next_matches,
                "trigger_ready": trigger_ready,
                "triggered_operation_id": triggered_operation_id,
                "task_ids": list(runtime_result.task_ids),
                "observation_task_ids": list(observation.task_ids),
                "observation_evidence_id": observation.evidence.id,
                "observation_source_evidence_ids": list(observation.evidence_ids),
                "runtime_event_count": len(runtime_result.events),
                "value_summary": value_summary,
            },
            observation=observation,
        )
        if trigger_ready and triggered_operation_id:
            action_run = await self.action_runner.execute_action(
                monitor,
                state_after,
                run,
                child_operation_id=triggered_operation_id,
                tick_operation_id=tick_operation_id,
                lease_id=lease_id,
            )
            action_state = await self.monitor_store.load_monitor_state(monitor.id)
            action_state = action_state or state_after
            return await self.delivery_runner.execute_delivery(
                monitor,
                action_state,
                action_run,
                child_operation_id=triggered_operation_id,
                tick_operation_id=tick_operation_id,
                lease_id=lease_id,
            )
        return run

    async def record_scheduler_decision(
        self,
        monitor_id: str,
        *,
        reason: str,
        now: datetime | str | None = None,
    ) -> DbMonitorRun:
        """Persist a scheduler decision that did not become a monitor tick."""

        tick_time = _coerce_datetime(now)
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        return await self._commit_run(
            monitor,
            state,
            operation_id=f"monitor-tick-{uuid4()}",
            run_id=f"monitor-run-{uuid4()}",
            now=tick_time,
            reason=reason,
            status="skipped",
            triggered=False,
            state_after=_state_for_scheduler_decision(state, reason, tick_time),
        )

    async def _commit_observation_unsuccessful_run(
        self,
        monitor: DbMonitor,
        state: DbMonitorState,
        now: datetime,
        exc: DbMonitorObservationBlocked,
        *,
        operation_id: str,
        run_id: str,
        lease_id: str | None = None,
    ) -> DbMonitorRun:
        state_after = DbMonitorState.from_dict(
            {
                **state.to_dict(),
                "last_tick_at": _iso(now),
                "last_operation_id": operation_id,
                "last_tick_operation_id": operation_id,
                "consecutive_failures": state.consecutive_failures + 1,
                "error": {
                    "reason": exc.reason,
                    "checked_at": _iso(now),
                    "details": exc.details,
                },
            }
        )
        observation = DbMonitorObservationResult(
            value=None,
            evidence=Evidence(
                id=f"monitor-observation-{uuid4()}",
                kind="monitor.observation",
                owner="db.monitor",
                operation_id=operation_id,
                accepted=False,
                payload={
                    "monitor_id": monitor.id,
                    "monitor_name": monitor.name,
                    "run_id": run_id,
                    "status": exc.status,
                    "reason": exc.reason,
                    "details": exc.details,
                    "task_ids": list(exc.task_ids),
                    "evidence_ids": list(exc.evidence_ids),
                    "observation_plan_kind": _observation_plan_kind(
                        monitor.observation_plan
                    ),
                },
            ),
            task_ids=exc.task_ids,
            evidence_ids=exc.evidence_ids,
            summary={
                "status": exc.status,
                "reason": exc.reason,
                "details": exc.details,
            },
        )
        return await self._commit_run(
            monitor,
            state,
            operation_id=operation_id,
            run_id=run_id,
            now=now,
            reason=exc.reason,
            status=exc.status,
            triggered=False,
            state_after=state_after,
            accepted=False,
            lease_id=lease_id,
            operation_precreated=True,
            summary={
                "observation_evidence_id": observation.evidence.id,
                "observation_task_ids": list(observation.task_ids),
                "observation_source_evidence_ids": list(observation.evidence_ids),
                "error": exc.details,
            },
            observation=observation,
        )

    async def _commit_failed_run(
        self,
        monitor: DbMonitor,
        state: DbMonitorState,
        now: datetime,
        exc: Exception,
        *,
        operation_id: str,
        run_id: str,
        lease_id: str | None = None,
        observation: DbMonitorObservationResult | None = None,
    ) -> DbMonitorRun:
        failures = state.consecutive_failures + 1
        backoff_seconds = _error_backoff_seconds(monitor, failures)
        state_after = DbMonitorState.from_dict(
            {
                **state.to_dict(),
                "last_tick_at": _iso(now),
                "last_operation_id": operation_id,
                "last_tick_operation_id": operation_id,
                "consecutive_failures": failures,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "backoff_until": _iso(now + timedelta(seconds=backoff_seconds)),
                },
            }
        )
        return await self._commit_run(
            monitor,
            state,
            operation_id=operation_id,
            run_id=run_id,
            now=now,
            reason="failed",
            status="failed",
            triggered=False,
            state_after=state_after,
            accepted=False,
            lease_id=lease_id,
            operation_precreated=True,
            summary={
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "backoff_seconds": backoff_seconds,
                },
                **(
                    {
                        "observation_evidence_id": observation.evidence.id,
                        "observation_task_ids": list(observation.task_ids),
                        "observation_source_evidence_ids": list(
                            observation.evidence_ids
                        ),
                    }
                    if observation is not None
                    else {}
                ),
            },
            observation=observation,
        )

    async def _begin_tick_operation(
        self,
        monitor: DbMonitor,
        state: DbMonitorState,
        *,
        operation_id: str,
        run_id: str,
        now: datetime,
        lease_id: str | None,
    ) -> None:
        operation = Operation(
            id=operation_id,
            operation_type="monitor.tick",
            status=OperationStatus.RUNNING,
            request={
                "kind": "monitor.tick",
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "run_id": run_id,
            },
            required_evidence=frozenset(
                {"monitor.observation", "monitor.trigger_decision"}
            ),
            metadata={
                "runtime_id": self.runtime.runtime_id,
                "runtime_kind": self.runtime.runtime_kind,
                "control_plane": "db.monitor",
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "run_id": run_id,
                "tick_started_at": _iso(now),
            },
        )
        event = RuntimeEvent(
            id=f"monitor-tick-event-{uuid4()}",
            type=RuntimeEventType.OPERATION_CREATED,
            operation_id=operation_id,
            runtime_id=self.runtime.runtime_id,
            runtime_kind=self.runtime.runtime_kind,
            message=f"Monitor tick operation {operation_id} created.",
            payload={"operation_type": "monitor.tick", "monitor_id": monitor.id},
        )
        await self.runtime.commit_monitor_mutation(
            DbMonitorMutation(
                action="tick",
                operation=operation,
                events=(event,),
                monitor_before=monitor,
                state_before=state,
                lease_id=lease_id,
            )
        )

    async def _commit_run(
        self,
        monitor: DbMonitor,
        state: DbMonitorState,
        *,
        operation_id: str,
        run_id: str,
        now: datetime,
        reason: str,
        status: str,
        triggered: bool,
        state_after: DbMonitorState | None = None,
        accepted: bool = True,
        lease_id: str | None = None,
        operation_precreated: bool = False,
        summary: Mapping[str, Any] | None = None,
        observation: DbMonitorObservationResult | None = None,
    ) -> DbMonitorRun:
        evidence_id = f"monitor-trigger-decision-{uuid4()}"
        tick_finished_at = _iso(now)
        operation_status = {
            "blocked": OperationStatus.BLOCKED,
            "failed": OperationStatus.FAILED,
        }.get(status, OperationStatus.SUCCEEDED)
        required_evidence = (
            frozenset({"monitor.observation", "monitor.trigger_decision"})
            if observation is not None or operation_precreated
            else frozenset({"monitor.trigger_decision"})
        )
        operation = Operation(
            id=operation_id,
            operation_type="monitor.tick",
            status=operation_status,
            request={
                "kind": "monitor.tick",
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
            },
            required_evidence=required_evidence,
            metadata={
                "runtime_id": self.runtime.runtime_id,
                "runtime_kind": self.runtime.runtime_kind,
                "control_plane": "db.monitor",
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "decision_reason": reason,
            },
        )
        evidence = Evidence(
            id=evidence_id,
            kind="monitor.trigger_decision",
            owner="db.monitor",
            operation_id=operation_id,
            accepted=accepted,
            payload={
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "reason": reason,
                "triggered": triggered,
                "status": status,
                "summary": dict(summary or {}),
                "observation_evidence_id": (
                    observation.evidence.id if observation is not None else None
                ),
                "observation_task_ids": (
                    list(observation.task_ids) if observation is not None else []
                ),
                "observation_source_evidence_ids": (
                    list(observation.evidence_ids) if observation is not None else []
                ),
                "state_before": state.to_dict(),
                "state_after": None if state_after is None else state_after.to_dict(),
                "validation": _validation_payload(monitor),
            },
        )
        run = DbMonitorRun(
            id=run_id,
            monitor_id=monitor.id,
            operation_id=operation_id,
            tick_started_at=_iso(now),
            tick_finished_at=tick_finished_at,
            triggered=triggered,
            trigger_decision_evidence_id=evidence_id,
            status=status,  # type: ignore[arg-type]
            summary={
                "reason": reason,
                **dict(summary or {}),
            },
        )
        created_event = RuntimeEvent(
            id=f"monitor-tick-event-{uuid4()}",
            type=RuntimeEventType.OPERATION_CREATED,
            operation_id=operation_id,
            runtime_id=self.runtime.runtime_id,
            runtime_kind=self.runtime.runtime_kind,
            message=f"Monitor tick operation {operation_id} created.",
            payload={"operation_type": "monitor.tick", "monitor_id": monitor.id},
        )
        ticked_event = RuntimeEvent(
            id=f"monitor-tick-event-{uuid4()}",
            type=RuntimeEventType.MONITOR_TICKED,
            operation_id=operation_id,
            runtime_id=self.runtime.runtime_id,
            runtime_kind=self.runtime.runtime_kind,
            evidence_id=evidence_id,
            message=f"Monitor {monitor.id} scheduler decision: {reason}.",
            payload={
                "monitor_id": monitor.id,
                "status": status,
                "triggered": triggered,
                "reason": reason,
            },
        )
        final_event = RuntimeEvent(
            id=f"monitor-tick-event-{uuid4()}",
            type=(
                RuntimeEventType.MONITOR_TRIGGERED
                if triggered
                else RuntimeEventType.MONITOR_SKIPPED
            ),
            operation_id=operation_id,
            runtime_id=self.runtime.runtime_id,
            runtime_kind=self.runtime.runtime_kind,
            evidence_id=evidence_id,
            message=f"Monitor {monitor.id} tick finished with {status}.",
            payload={"monitor_id": monitor.id, "status": status, "reason": reason},
        )
        completed_event = RuntimeEvent(
            id=f"monitor-tick-event-{uuid4()}",
            type=RuntimeEventType.OPERATION_UPDATED,
            operation_id=operation_id,
            runtime_id=self.runtime.runtime_id,
            runtime_kind=self.runtime.runtime_kind,
            evidence_id=evidence_id,
            message=f"Monitor tick {operation_id} finished.",
            payload={
                "status": operation_status.value,
                "monitor_id": monitor.id,
                "reason": reason,
            },
        )
        events = (
            (ticked_event, final_event, completed_event)
            if operation_precreated
            else (created_event, ticked_event, final_event, completed_event)
        )
        await self.runtime.commit_monitor_mutation(
            DbMonitorMutation(
                action="run",
                operation=operation,
                evidence=(
                    ((observation.evidence,) if observation is not None else ())
                    + (evidence,)
                ),
                events=events,
                monitor_before=monitor,
                state_before=state,
                state_after=state_after,
                run_after=run,
                lease_id=lease_id,
            )
        )
        return run
