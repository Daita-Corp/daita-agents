"""Durable DB monitor scheduling and tick control plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Mapping
from uuid import uuid4

from daita.runtime import (
    Evidence,
    Operation,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
)
from daita.runtime.monitors import (
    MonitorRuntime,
    MonitorSpec,
    monitor_trigger_matches,
    summarize_monitor_value,
)

from .monitor_commands import DbMonitorValidation
from .monitors import (
    DbMonitor,
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
    DbMonitorStore,
)
from .sql_evidence import (
    blocked_scope_resources,
    effective_source_scope,
    sql_validation_facts_from_evidence,
)


@dataclass(frozen=True)
class DbMonitorSchedulerResult:
    """One scheduler decision for a durable DB monitor."""

    monitor_id: str
    run: DbMonitorRun
    claimed: bool = False


@dataclass(frozen=True)
class DbMonitorObservationResult:
    """Compact observed facts and runtime provenance for one monitor tick."""

    value: Any
    evidence: Evidence
    task_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class DbMonitorActionResult:
    """Compact action execution provenance for a triggered monitor child op."""

    status: str
    evidence_id: str | None = None
    report_evidence_id: str | None = None
    task_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class DbMonitorDeliveryResult:
    """Compact delivery provenance for a triggered monitor child op."""

    status: str
    evidence_id: str | None = None
    task_ids: tuple[str, ...] = ()
    plugin_evidence_refs: tuple[dict[str, Any], ...] = ()
    summary: dict[str, Any] | None = None


class DbMonitorObservationBlocked(RuntimeError):
    """Raised when a persisted observation plan cannot safely execute."""

    status = "blocked"

    def __init__(
        self,
        reason: str,
        *,
        details: Mapping[str, Any] | None = None,
        task_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details or {})
        self.task_ids = tuple(task_ids)
        self.evidence_ids = tuple(evidence_ids)


class DbMonitorObservationFailed(DbMonitorObservationBlocked):
    """Raised when observation execution fails after a tick has started."""

    status = "failed"


class DbMonitorObservationRunner:
    """Compile persisted monitor observation plans into read-only DB tasks."""

    def __init__(self, *, runtime: Any) -> None:
        self.runtime = runtime

    async def observe(
        self,
        monitor: DbMonitor,
        state: DbMonitorState,
        operation: Operation,
        *,
        run_id: str,
        now: datetime,
    ) -> DbMonitorObservationResult:
        plan = dict(monitor.observation_plan or {})
        kind = _observation_plan_kind(plan)
        if kind == "plugin_source":
            result = await self.runtime.execute_monitor_source_observation(
                operation,
                monitor_id=monitor.id,
                monitor_run_id=run_id,
                tick_operation_id=operation.id,
                source_step=plan,
                cursor=state.cursor,
            )
            status = str(result.get("status") or "blocked")
            if status != "succeeded":
                error_type = (
                    DbMonitorObservationFailed
                    if status == "failed"
                    else DbMonitorObservationBlocked
                )
                raise error_type(
                    str(result.get("block_reason") or "plugin_source_blocked"),
                    details=dict(result.get("details") or {}),
                    task_ids=tuple(str(item) for item in result.get("task_ids") or ()),
                    evidence_ids=tuple(
                        str(item.get("id"))
                        for item in result.get("plugin_evidence_refs") or ()
                        if isinstance(item, Mapping) and item.get("id")
                    ),
                )
            value = result.get("value")
            task_ids = tuple(str(item) for item in result.get("task_ids") or ())
            plugin_refs = tuple(
                dict(item)
                for item in result.get("plugin_evidence_refs") or ()
                if isinstance(item, Mapping)
            )
            evidence_ids = tuple(
                str(item.get("id")) for item in plugin_refs if item.get("id")
            )
            summary = {
                "observation_plan_kind": kind,
                "source_kind": plan.get("source_kind"),
                "capability_id": result.get("capability_id"),
                "capability_owner": result.get("capability_owner"),
                "value": _compact_observed_value(value),
                "value_summary": summarize_monitor_value(value),
                "task_ids": list(task_ids),
                "evidence_ids": list(evidence_ids),
                "plugin_evidence_refs": [dict(item) for item in plugin_refs],
            }
            observation_evidence = Evidence(
                id=f"monitor-observation-{uuid4()}",
                kind="monitor.observation",
                owner="db.monitor",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "monitor_id": monitor.id,
                    "monitor_name": monitor.name,
                    "run_id": run_id,
                    "status": "succeeded",
                    "summary": summary,
                    "observed_value": _compact_observed_value(value),
                    "cursor_before": state.cursor,
                    "cursor_updates": {
                        "last_observation_at": _iso(now),
                        "last_plugin_source_evidence_ids": list(evidence_ids),
                    },
                    "task_ids": list(task_ids),
                    "evidence_ids": list(evidence_ids),
                    "plugin_evidence_refs": [dict(item) for item in plugin_refs],
                },
            )
            return DbMonitorObservationResult(
                value=value,
                evidence=observation_evidence,
                task_ids=task_ids,
                evidence_ids=evidence_ids,
                summary=summary,
            )
        if kind not in {"metric_sql", "freshness_sql", "planned_read"}:
            raise DbMonitorObservationBlocked(
                "missing_executable_observation_plan",
                details={
                    "observation_plan_kind": kind,
                    "source_scope": list(monitor.source_scope),
                },
            )

        sql = _observation_sql(plan)
        if not sql:
            raise DbMonitorObservationBlocked(
                "missing_observation_sql",
                details={"observation_plan_kind": kind},
            )

        owner = _observation_capability_owner(plan)
        try:
            validation_task, read_task = self.runtime.plan_validated_read_tasks(
                operation,
                sql=sql,
                params=list(plan.get("parameters") or plan.get("params") or ()),
                owner=owner,
                reason="monitor_observation_read",
                sequence=1,
                focus=plan.get("metric") or plan.get("purpose") or monitor.name,
                metadata={
                    "monitor_id": monitor.id,
                    "monitor_run_id": run_id,
                    "tick_operation_id": operation.id,
                    "observation_plan_kind": kind,
                },
            )
        except (KeyError, ValueError) as exc:
            raise DbMonitorObservationBlocked(
                "missing_observation_capability",
                details={
                    "capability_id": "db.sql.execute_read",
                    "owner": owner,
                    "error": str(exc),
                },
            ) from exc

        try:
            validation_evidence = await self.runtime.execute_task(
                validation_task,
                operation,
                context={
                    "monitor_id": monitor.id,
                    "monitor_run_id": run_id,
                    "tick_operation_id": operation.id,
                    "db_monitor_phase": 4,
                    "monitor_observation_role": "validation",
                },
            )
        except Exception as exc:
            raise DbMonitorObservationFailed(
                "observation_validation_failed",
                details={"type": type(exc).__name__, "message": str(exc)},
                task_ids=(validation_task.id,),
            ) from exc
        validation = _validation_evidence(validation_evidence)
        validation_ids = tuple(item.id for item in validation_evidence if item.id)
        task_ids = (validation_task.id, read_task.id)
        if validation is None or not validation.accepted:
            raise DbMonitorObservationBlocked(
                "observation_sql_validation_failed",
                details={
                    "validation_evidence_id": (
                        validation.id if validation is not None else None
                    )
                },
                task_ids=(validation_task.id,),
                evidence_ids=validation_ids,
            )
        validation_facts = sql_validation_facts_from_evidence(validation)
        if validation_facts.is_read is False:
            raise DbMonitorObservationBlocked(
                "unsafe_observation_sql",
                details={"validation": validation.payload},
                task_ids=(validation_task.id,),
                evidence_ids=validation_ids,
            )
        if validation_facts.valid is False:
            raise DbMonitorObservationBlocked(
                "observation_sql_validation_failed",
                details={"validation": validation.payload},
                task_ids=(validation_task.id,),
                evidence_ids=validation_ids,
            )
        source_scope = effective_source_scope(monitor.source_scope, plan)
        blocked_scope = blocked_scope_resources(
            validation_facts.target_resources,
            source_scope,
        )
        if blocked_scope:
            raise DbMonitorObservationBlocked(
                "observation_source_scope_blocked",
                details={
                    "source_scope": list(source_scope),
                    "target_resources": list(validation_facts.target_resources),
                    "blocked_resources": list(blocked_scope),
                },
                task_ids=(validation_task.id,),
                evidence_ids=validation_ids,
            )

        try:
            read_evidence = await self.runtime.execute_task(
                read_task,
                operation,
                context={
                    "monitor_id": monitor.id,
                    "monitor_run_id": run_id,
                    "tick_operation_id": operation.id,
                    "db_monitor_phase": 4,
                    "monitor_observation_role": "read",
                },
            )
        except Exception as exc:
            raise DbMonitorObservationFailed(
                "observation_execution_failed",
                details={"type": type(exc).__name__, "message": str(exc)},
                task_ids=task_ids,
                evidence_ids=validation_ids,
            ) from exc
        query_result = next(
            (item for item in read_evidence if item.kind == "query.result"),
            None,
        )
        if query_result is None or not query_result.accepted:
            raise DbMonitorObservationBlocked(
                "observation_query_result_missing",
                details={"read_task_id": read_task.id},
                task_ids=tuple(
                    task.id for task in (validation_task, read_task) if task is not None
                ),
                evidence_ids=tuple(
                    item.id
                    for item in (*validation_evidence, *read_evidence)
                    if item.id
                ),
            )

        value = _observed_value(plan, query_result.payload, now=now)
        evidence_ids = tuple(
            item.id for item in (*validation_evidence, *read_evidence) if item.id
        )
        row_count = _query_row_count(query_result.payload)
        summary = {
            "observation_plan_kind": kind,
            "metric": plan.get("metric"),
            "purpose": plan.get("purpose"),
            "value": _compact_observed_value(value),
            "value_summary": summarize_monitor_value(value),
            "row_count": row_count,
            "source_scope": list(source_scope),
            "sql_fingerprint": validation_facts.sql_fingerprint,
            "task_ids": list(task_ids),
            "evidence_ids": list(evidence_ids),
        }
        observation_evidence = Evidence(
            id=f"monitor-observation-{uuid4()}",
            kind="monitor.observation",
            owner="db.monitor",
            operation_id=operation.id,
            accepted=True,
            payload={
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "run_id": run_id,
                "status": "succeeded",
                "summary": summary,
                "observed_value": _compact_observed_value(value),
                "cursor_before": state.cursor,
                "cursor_updates": {
                    "last_observation_at": _iso(now),
                    "last_observation_fingerprint": validation_facts.sql_fingerprint,
                },
                "task_ids": list(task_ids),
                "evidence_ids": list(evidence_ids),
            },
        )
        return DbMonitorObservationResult(
            value=value,
            evidence=observation_evidence,
            task_ids=task_ids,
            evidence_ids=evidence_ids,
            summary=summary,
        )


class DbMonitorActionRunner:
    """Hand persisted monitor action plans to the DB runtime execution owner."""

    def __init__(self, *, runtime: Any, monitor_store: DbMonitorStore) -> None:
        self.runtime = runtime
        self.monitor_store = monitor_store

    async def execute_action(
        self,
        monitor: DbMonitor,
        state: DbMonitorState,
        run: DbMonitorRun,
        *,
        child_operation_id: str,
        tick_operation_id: str,
        lease_id: str | None,
    ) -> DbMonitorRun:
        tick_evidence = tuple(
            item
            for item in await self.runtime.store.list_evidence(tick_operation_id)
            if item.kind in {"monitor.observation", "monitor.trigger_decision"}
        )
        result = await self.runtime.execute_monitor_action(
            child_operation_id,
            monitor_id=monitor.id,
            monitor_name=monitor.name,
            monitor_run_id=run.id,
            tick_operation_id=tick_operation_id,
            action_plan=dict(monitor.action_plan or {}),
            tick_evidence_refs=tuple(
                _monitor_evidence_ref(item) for item in tick_evidence
            ),
            source_scope=tuple(monitor.source_scope),
        )
        child_operation = await self.runtime.store.load_operation(child_operation_id)
        if child_operation is None:
            raise RuntimeError(f"monitor action operation {child_operation_id} missing")
        action_status = str(result.get("status") or "unknown")
        produced_refs = [
            dict(item)
            for item in result.get("produced_evidence_refs") or ()
            if isinstance(item, Mapping)
        ]
        action_evidence_id = result.get("action_result_evidence_id") or _latest_ref_id(
            produced_refs, "monitor.action_result"
        )
        report_evidence_id = _latest_ref_id(produced_refs, "monitor.report")
        updated_run = DbMonitorRun.from_dict(
            {
                **run.to_dict(),
                "summary": {
                    **run.summary,
                    "action_status": action_status,
                    "action_kind": result.get("action_kind"),
                    "action_plan_fingerprint": result.get("action_plan_fingerprint"),
                    "action_evidence_id": action_evidence_id,
                    "report_evidence_id": report_evidence_id,
                    "action_task_ids": list(result.get("task_ids") or ()),
                    "action_produced_evidence_refs": produced_refs,
                    "action_block_reason": result.get("block_reason"),
                    "action_budget_usage": dict(result.get("budget_usage") or {}),
                },
            }
        )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="run",
                operation=child_operation,
                events=(
                    RuntimeEvent(
                        id=f"monitor-action-event-{uuid4()}",
                        type=RuntimeEventType.OPERATION_UPDATED,
                        operation_id=child_operation_id,
                        runtime_id=self.runtime.runtime_id,
                        runtime_kind=self.runtime.runtime_kind,
                        message=(
                            f"Monitor {monitor.id} action finished with "
                            f"{action_status}."
                        ),
                        payload={
                            "monitor_id": monitor.id,
                            "monitor_run_id": run.id,
                            "tick_operation_id": tick_operation_id,
                            "status": action_status,
                            "action_evidence_id": action_evidence_id,
                        },
                    ),
                ),
                monitor_before=monitor,
                state_before=state,
                state_after=state,
                run_after=updated_run,
                lease_id=lease_id,
            )
        )
        return updated_run


class DbMonitorDeliveryRunner:
    """Hand persisted monitor delivery intents to the DB runtime execution owner."""

    def __init__(self, *, runtime: Any, monitor_store: DbMonitorStore) -> None:
        self.runtime = runtime
        self.monitor_store = monitor_store

    async def execute_delivery(
        self,
        monitor: DbMonitor,
        state: DbMonitorState,
        run: DbMonitorRun,
        *,
        child_operation_id: str,
        tick_operation_id: str,
        lease_id: str | None,
    ) -> DbMonitorRun:
        report_evidence_id = run.summary.get("report_evidence_id")
        if not report_evidence_id:
            return run
        result = await self.runtime.execute_monitor_delivery(
            child_operation_id,
            monitor_id=monitor.id,
            monitor_name=monitor.name,
            monitor_run_id=run.id,
            tick_operation_id=tick_operation_id,
            report_evidence_id=str(report_evidence_id),
            governed=_monitor_governed_delivery_enabled(monitor),
        )
        delivery_status = str(result.get("status") or "unknown")
        if delivery_status == "skipped":
            return run
        child_operation = await self.runtime.store.load_operation(child_operation_id)
        if child_operation is None:
            raise RuntimeError(
                f"monitor delivery operation {child_operation_id} missing"
            )
        plugin_refs = [
            dict(item)
            for item in result.get("plugin_result_evidence_refs") or ()
            if isinstance(item, Mapping)
        ]
        updated_run = DbMonitorRun.from_dict(
            {
                **run.to_dict(),
                "summary": {
                    **run.summary,
                    "delivery_status": delivery_status,
                    "delivery_kind": result.get("delivery_kind"),
                    "delivery_operation_id": result.get("delivery_operation_id"),
                    "delivery_result_evidence_id": result.get(
                        "delivery_result_evidence_id"
                    ),
                    "delivery_task_ids": list(result.get("task_ids") or ()),
                    "delivery_plugin_result_evidence_refs": plugin_refs,
                    "delivery_block_reason": result.get("block_reason"),
                    "delivery_idempotency_key": result.get("idempotency_key"),
                },
            }
        )
        state_after = DbMonitorState.from_dict(
            {
                **state.to_dict(),
                "cursor": {
                    **state.cursor,
                    "last_delivery_status": delivery_status,
                    "last_delivery_result_evidence_id": result.get(
                        "delivery_result_evidence_id"
                    ),
                    "last_delivery_idempotency_key": result.get("idempotency_key"),
                },
            }
        )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="run",
                operation=child_operation,
                events=(
                    RuntimeEvent(
                        id=f"monitor-delivery-event-{uuid4()}",
                        type=RuntimeEventType.OPERATION_UPDATED,
                        operation_id=child_operation_id,
                        runtime_id=self.runtime.runtime_id,
                        runtime_kind=self.runtime.runtime_kind,
                        evidence_id=result.get("delivery_result_evidence_id"),
                        message=(
                            f"Monitor {monitor.id} delivery finished with "
                            f"{delivery_status}."
                        ),
                        payload={
                            "monitor_id": monitor.id,
                            "monitor_run_id": run.id,
                            "tick_operation_id": tick_operation_id,
                            "status": delivery_status,
                        },
                    ),
                ),
                monitor_before=monitor,
                state_before=state,
                state_after=state_after,
                run_after=updated_run,
                lease_id=lease_id,
            )
        )
        return updated_run


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
        await self.monitor_store.commit_monitor_mutation(
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
        await self.monitor_store.commit_monitor_mutation(
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


def _observation_plan_kind(plan: Mapping[str, Any]) -> str:
    return str(plan.get("kind") or plan.get("type") or "")


def _observation_sql(plan: Mapping[str, Any]) -> str:
    if plan.get("sql") is not None:
        return str(plan.get("sql") or "")
    query_plan = plan.get("query_plan")
    if isinstance(query_plan, Mapping):
        return str(query_plan.get("sql") or "")
    return ""


def _observation_capability_owner(plan: Mapping[str, Any]) -> str | None:
    owner = (
        plan.get("capability_owner") or plan.get("source_owner") or plan.get("owner")
    )
    return str(owner) if owner else None


def _validation_evidence(evidence: tuple[Evidence, ...]) -> Evidence | None:
    return next((item for item in evidence if item.kind == "sql.validation"), None)


def _observed_value(
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    now: datetime,
) -> Any:
    rows = payload.get("rows") or []
    value_path = str(plan.get("value_path") or "")
    if value_path:
        value = _extract_observation_path({"rows": rows, **dict(payload)}, value_path)
    elif rows and isinstance(rows, list) and isinstance(rows[0], Mapping):
        first = dict(rows[0])
        value = next(iter(first.values())) if len(first) == 1 else first
    else:
        value = rows

    if _observation_plan_kind(plan) == "freshness_sql":
        return _freshness_value(plan, value, now=now)
    metric = plan.get("metric")
    if metric and not isinstance(value, Mapping):
        return {str(metric): value}
    return value


def _freshness_value(
    plan: Mapping[str, Any],
    value: Any,
    *,
    now: datetime,
) -> dict[str, Any]:
    latest = str(value) if value is not None else None
    age_seconds = None
    fresh = None
    if latest:
        try:
            age_seconds = max(0.0, (now - _parse_iso(latest)).total_seconds())
        except ValueError:
            age_seconds = None
    if age_seconds is not None and plan.get("freshness_sla_seconds") is not None:
        try:
            fresh = age_seconds <= float(plan["freshness_sla_seconds"])
        except (TypeError, ValueError):
            fresh = None
    return {
        str(plan.get("metric") or "freshness"): latest,
        "latest_timestamp": latest,
        "age_seconds": age_seconds,
        "fresh": fresh,
    }


def _extract_observation_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _query_row_count(payload: Mapping[str, Any]) -> int | None:
    rows = payload.get("rows")
    return len(rows) if isinstance(rows, list) else None


def _compact_observed_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _compact_observed_value(item)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "array",
            "count": len(value),
            "sample": [_compact_observed_value(item) for item in list(value)[:3]],
        }
    return str(value)


def _is_due(
    monitor: DbMonitor,
    state: DbMonitorState | None,
    now: datetime,
) -> tuple[bool, str]:
    if monitor.status == "paused":
        return False, "paused"
    if monitor.status == "disabled":
        return False, "disabled"
    if monitor.status == "error":
        return False, "error"
    state = state or DbMonitorState(monitor_id=monitor.id)
    blocked = _time_gate_reason(state, now)
    if blocked is not None:
        return False, blocked
    interval = _schedule_interval_seconds(monitor.schedule)
    if interval is None:
        return True, "due"
    if state.last_tick_at is None:
        return True, "due"
    elapsed = (now - _parse_iso(state.last_tick_at)).total_seconds()
    return (elapsed >= interval, "due" if elapsed >= interval else "not_due")


def _blocked_reason(
    monitor: DbMonitor,
    state: DbMonitorState,
    now: datetime,
) -> str | None:
    if monitor.status == "paused":
        return "paused"
    if monitor.status in {"disabled", "error"}:
        return monitor.status
    time_gate = _time_gate_reason(state, now)
    if time_gate is not None:
        return time_gate
    validation = _validation_for_monitor(monitor)
    if validation is None:
        return None
    if (
        not validation.accepted
        or validation.errors
        or validation.missing_capabilities
        or validation.unsupported_actions
    ):
        return "validation_blocked"
    return None


def _time_gate_reason(state: DbMonitorState, now: datetime) -> str | None:
    if state.paused_until is not None and _parse_iso(state.paused_until) > now:
        return "paused"
    if state.cooldown_until is not None and _parse_iso(state.cooldown_until) > now:
        return "cooldown"
    backoff_until = (state.error or {}).get("backoff_until")
    if backoff_until and _parse_iso(str(backoff_until)) > now:
        return "backoff"
    return None


def _state_for_scheduler_decision(
    state: DbMonitorState,
    reason: str,
    now: datetime,
) -> DbMonitorState | None:
    if reason == "lease_lost":
        return None
    if reason == "paused" and state.paused_until is not None:
        return state
    if reason in {"cooldown", "backoff", "not_due"}:
        return state
    if reason in {"disabled", "error"}:
        return DbMonitorState.from_dict(
            {**state.to_dict(), "error": {"reason": reason, "checked_at": _iso(now)}}
        )
    return state


def _state_for_blocked_decision(
    state: DbMonitorState,
    reason: str,
    now: datetime,
    operation_id: str,
) -> DbMonitorState:
    if reason == "validation_blocked":
        return DbMonitorState.from_dict(
            {
                **state.to_dict(),
                "last_tick_at": _iso(now),
                "last_operation_id": operation_id,
                "last_tick_operation_id": operation_id,
                "consecutive_failures": state.consecutive_failures + 1,
                "error": {"reason": reason, "checked_at": _iso(now)},
            }
        )
    return state


def _compile_monitor_spec(
    monitor: DbMonitor,
    state: DbMonitorState,
    *,
    tick_operation_id: str,
    run_id: str,
) -> MonitorSpec:
    return MonitorSpec(
        id=monitor.id,
        name=monitor.name,
        schedule=monitor.schedule,
        stream=monitor.stream,
        trigger={},
        action_input={"action_plan": monitor.action_plan},
        cooldown_seconds=_cooldown_seconds(monitor) or None,
        cursor=state.cursor,
        metadata={
            "parent_operation_id": tick_operation_id,
            "tick_operation_id": tick_operation_id,
            "monitor_run_id": run_id,
            "db_monitor": {
                "id": monitor.id,
                "tick_operation_id": tick_operation_id,
                "run_id": run_id,
                "source_scope": list(monitor.source_scope),
                "observation_plan": monitor.observation_plan,
                "policy": monitor.policy,
                "budgets": monitor.budgets,
                "owner": monitor.owner,
                "validation": _validation_payload(monitor),
            },
        },
    )


def _runtime_spec_for_decision(spec: MonitorSpec, trigger_ready: bool) -> MonitorSpec:
    return MonitorSpec(
        id=spec.id,
        name=spec.name,
        schedule=spec.schedule,
        stream=spec.stream,
        trigger={} if trigger_ready else {"truthy": True},
        action_input=spec.action_input,
        cooldown_seconds=spec.cooldown_seconds,
        cursor=spec.cursor,
        metadata=spec.metadata,
    )


def _monitor_tick_value(monitor: DbMonitor) -> Any:
    if "tick_value" in monitor.metadata:
        return monitor.metadata["tick_value"]
    if "observed_value" in monitor.metadata:
        return monitor.metadata["observed_value"]
    if "value" in monitor.observation_plan:
        return monitor.observation_plan["value"]
    if monitor.trigger.get("type") in {"schedule", "scheduled"}:
        return True
    return None


def _trigger_decision(
    value: Any,
    trigger: Mapping[str, Any],
) -> tuple[bool, Any, str]:
    if not trigger:
        return True, summarize_monitor_value(value), "trigger_matched"
    if trigger.get("type") in {"schedule", "scheduled"}:
        return True, summarize_monitor_value(value), "schedule"
    if "force_trigger" in trigger:
        return bool(trigger["force_trigger"]), summarize_monitor_value(value), "forced"
    if value is None:
        return False, None, "observation_unavailable"
    try:
        matched = monitor_trigger_matches(value, trigger)
    except TypeError:
        return False, summarize_monitor_value(value), "trigger_type_mismatch"
    return (
        matched,
        summarize_monitor_value(value),
        "trigger_matched" if matched else "no_match",
    )


def _required_consecutive_matches(monitor: DbMonitor) -> int:
    raw = (
        monitor.trigger.get("consecutive_matches")
        or monitor.budgets.get("consecutive_matches")
        or monitor.policy.get("consecutive_matches")
        or 1
    )
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _cooldown_seconds(monitor: DbMonitor) -> float:
    raw = (
        monitor.budgets.get("cooldown_seconds")
        or monitor.budgets.get("cooldown")
        or monitor.policy.get("cooldown_seconds")
        or 0
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _error_backoff_seconds(monitor: DbMonitor, failures: int) -> float:
    raw = monitor.budgets.get("error_backoff_seconds") or monitor.budgets.get(
        "backoff_seconds"
    )
    try:
        base = float(raw if raw is not None else 60.0)
    except (TypeError, ValueError):
        base = 60.0
    return max(1.0, base * max(1, failures))


def _schedule_interval_seconds(schedule: Mapping[str, Any] | None) -> float | None:
    if not schedule:
        return None
    for key in ("interval_seconds", "every_seconds"):
        if key in schedule:
            try:
                return max(0.0, float(schedule[key]))
            except (TypeError, ValueError):
                return None
    expression = str(schedule.get("expression") or "")
    every = re.search(r"every\s+(\d+)\s+(second|minute|hour)s?", expression, re.I)
    if every:
        amount = int(every.group(1))
        unit = every.group(2).lower()
        multiplier = {"second": 1, "minute": 60, "hour": 3600}[unit]
        return float(amount * multiplier)
    cron_every = re.match(r"^\*/(\d+)\s+\*\s+\*\s+\*\s+\*$", expression)
    if cron_every:
        return float(int(cron_every.group(1)) * 60)
    return None


def _validation_for_monitor(monitor: DbMonitor) -> DbMonitorValidation | None:
    data = monitor.metadata.get("validation")
    if not isinstance(data, Mapping):
        return None
    return DbMonitorValidation.from_dict(dict(data))


def _validation_payload(monitor: DbMonitor) -> dict[str, Any] | None:
    validation = _validation_for_monitor(monitor)
    return None if validation is None else validation.to_dict()


def _state_value_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _coerce_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        return _parse_iso(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _monitor_evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "operation_id": evidence.operation_id,
        "task_id": evidence.task_id,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint"),
    }


def _latest_ref_id(refs: list[dict[str, Any]], kind: str) -> str | None:
    matches = [
        str(item["id"]) for item in refs if item.get("kind") == kind and item.get("id")
    ]
    return matches[-1] if matches else None


def _monitor_governed_delivery_enabled(monitor: DbMonitor) -> bool:
    policy = monitor.policy if isinstance(monitor.policy, Mapping) else {}
    action_plan = (
        monitor.action_plan if isinstance(monitor.action_plan, Mapping) else {}
    )
    delivery_intent = action_plan.get("delivery_intent")
    delivery_intent = delivery_intent if isinstance(delivery_intent, Mapping) else {}
    return bool(
        policy.get("governed_delivery")
        or policy.get("approval_gated_delivery")
        or delivery_intent.get("governed")
        or delivery_intent.get("requires_approval")
        or delivery_intent.get("approval_required")
    )
