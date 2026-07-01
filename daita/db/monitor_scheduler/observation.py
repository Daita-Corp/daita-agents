"""Observation execution for durable DB monitor ticks."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping
from uuid import uuid4

from daita.runtime import Evidence, Operation
from daita.runtime.monitors import summarize_monitor_value

from ..monitors import DbMonitor, DbMonitorState
from ..sql_evidence import (
    blocked_scope_resources,
    effective_source_scope,
    sql_validation_facts_from_evidence,
)
from ..runtime.tasks import DbTaskSpec
from .params import resolve_observation_params, resolve_monitor_state_ref
from .state import _iso, _parse_iso
from .types import (
    DbMonitorObservationBlocked,
    DbMonitorObservationFailed,
    DbMonitorObservationResult,
)


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
        params, param_specs = resolve_observation_params(plan, state)
        try:
            read_capability = self.runtime.registry.get_capability(
                "db.sql.execute_read",
                owner=owner,
            )
            validation_capability = self.runtime.registry.get_capability(
                "db.sql.validate",
                owner=read_capability.owner,
            )
            execute_input: dict[str, Any] = {
                "sql_ref": "sql.validation",
                "params": list(params),
            }
            if param_specs:
                execute_input["param_specs"] = list(param_specs)
            task_plan = await self.runtime.plan_task_specs(
                operation,
                (
                    DbTaskSpec(
                        capability_id=validation_capability.id,
                        owner=validation_capability.owner,
                        input={"sql": sql, "operation": "query"},
                        reason="monitor_observation_read_validation",
                        sequence=1,
                        metadata={
                            "monitor_id": monitor.id,
                            "monitor_run_id": run_id,
                            "tick_operation_id": operation.id,
                            "observation_plan_kind": kind,
                        },
                    ),
                    DbTaskSpec(
                        capability_id=read_capability.id,
                        owner=read_capability.owner,
                        input=execute_input,
                        reason="monitor_observation_read",
                        sequence=2,
                        metadata={
                            "monitor_id": monitor.id,
                            "monitor_run_id": run_id,
                            "tick_operation_id": operation.id,
                            "observation_plan_kind": kind,
                        },
                    ),
                ),
            )
            validation_task, read_task = task_plan.tasks
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
        cursor_updates = {
            "last_observation_at": _iso(now),
            "last_observation_fingerprint": validation_facts.sql_fingerprint,
            **_cursor_updates_from_plan(plan, query_result.payload),
        }
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
                "cursor_updates": cursor_updates,
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
        plan.get("execution_owner")
        or plan.get("validation_owner")
        or plan.get("capability_owner")
        or plan.get("source_owner")
        or plan.get("owner")
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


def _observation_params(
    plan: Mapping[str, Any],
    state: DbMonitorState,
) -> list[Any]:
    params, _param_specs = resolve_observation_params(plan, state)
    return params


def _resolve_observation_param(value: Any, state: DbMonitorState) -> Any:
    if isinstance(value, str):
        return resolve_monitor_state_ref(value, state)
    return value


def _cursor_updates_from_plan(
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return updates
    cursor_update = plan.get("cursor_update")
    if not isinstance(cursor_update, Mapping):
        return updates
    for cursor_key, rule in cursor_update.items():
        if not isinstance(cursor_key, str) or not isinstance(rule, str):
            continue
        match = re.fullmatch(r"max\(rows\.([A-Za-z_][A-Za-z0-9_]*)\)", rule)
        if not match:
            continue
        field = match.group(1)
        values = [
            row.get(field)
            for row in rows
            if isinstance(row, Mapping) and row.get(field) is not None
        ]
        if values:
            updates[cursor_key] = max(values)
    return updates


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
