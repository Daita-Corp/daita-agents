"""Runtime-owned DB monitor lifecycle extension executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from daita.runtime import Evidence, Operation, Task

from ...analysis import stable_fingerprint
from ...evidence import load_evidence
from ...monitor_commands.types import DbMonitorValidation
from ...monitors import (
    DbMonitor,
    DbMonitorMutation,
    DbMonitorState,
    monitor_with_updates,
)


@dataclass(frozen=True)
class DbMonitorPlanLifecycleExecutor:
    """Executor that persists accepted or blocked monitor lifecycle proposals."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.plan_lifecycle"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.plan_lifecycle"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        action = _monitor_lifecycle_action(
            str(task.input.get("action") or operation.operation_type)
        )
        monitor_id = str(
            task.input.get("monitor_id") or operation.metadata.get("monitor_id") or ""
        )
        if not monitor_id:
            raise RuntimeError("monitor lifecycle proposal requires monitor_id")

        monitor = await runtime.monitor_store.load_monitor(monitor_id)
        errors: list[str] = []
        before = monitor.to_dict() if monitor is not None else None
        after = None
        patch = dict(task.input.get("patch") or {})
        paused_until = task.input.get("paused_until")
        if monitor is None:
            errors.append("monitor.lifecycle:monitor_not_found")
        else:
            try:
                updated = _monitor_after_lifecycle_action(
                    monitor,
                    action=action,
                    patch=patch,
                    paused_until=paused_until,
                )
                after = None if updated is None else updated.to_dict()
                reason = (
                    None
                    if updated is None
                    else _non_executable_active_monitor_reason(
                        updated.observation_plan,
                    )
                )
                if updated is not None and updated.status == "active" and reason:
                    errors.append(f"monitor.lifecycle:{reason}")
            except ValueError as exc:
                errors.append(f"monitor.lifecycle:{str(exc)}")

        validation = DbMonitorValidation(
            accepted=not errors,
            errors=tuple(errors),
            diagnostics={
                "action": action,
                "operation_type": operation.operation_type,
            },
        )
        proposal = {
            "kind": "monitor.proposal",
            "operation_type": operation.operation_type,
            "action": action,
            "monitor_id": monitor_id,
            "before": before,
            "after": after,
            "patch": patch,
            "paused_until": paused_until,
            "validation": validation.to_dict(),
        }
        fingerprint = stable_fingerprint(proposal)
        proposal["proposal_fingerprint"] = fingerprint
        return [
            Evidence(
                kind="monitor.proposal",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=validation.accepted,
                payload=proposal,
                metadata={
                    "payload_fingerprint": fingerprint,
                    "monitor_id": monitor_id,
                    "action": action,
                    "validation_accepted": validation.accepted,
                },
            )
        ]


@dataclass(frozen=True)
class DbMonitorCommitLifecycleExecutor:
    """Executor that idempotently commits a monitor lifecycle proposal."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.commit_lifecycle"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.commit_lifecycle"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        proposal_evidence = await _load_monitor_proposal_evidence(
            runtime,
            operation,
            task,
            task.input.get("proposal_evidence_id"),
        )
        if proposal_evidence is None:
            raise RuntimeError("monitor lifecycle proposal evidence is required")
        if not proposal_evidence.accepted:
            raise RuntimeError("monitor lifecycle proposal evidence was not accepted")
        proposal = dict(proposal_evidence.payload)
        expected_fingerprint = task.input.get("proposal_fingerprint")
        actual_fingerprint = proposal.get("proposal_fingerprint") or stable_fingerprint(
            proposal
        )
        if expected_fingerprint and expected_fingerprint != actual_fingerprint:
            raise RuntimeError("monitor lifecycle proposal fingerprint mismatch")

        action = _monitor_lifecycle_action(str(proposal.get("action") or "update"))
        monitor_id = str(proposal.get("monitor_id") or "")
        before_payload = proposal.get("before")
        after_payload = proposal.get("after")
        before = (
            DbMonitor.from_dict(before_payload)
            if isinstance(before_payload, Mapping)
            else None
        )
        after = (
            DbMonitor.from_dict(after_payload)
            if isinstance(after_payload, Mapping)
            else None
        )
        state = await runtime.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        existing = await runtime.monitor_store.load_monitor(monitor_id)
        idempotent_existing = _monitor_lifecycle_already_committed(
            existing,
            after=after,
            action=action,
        )
        if not idempotent_existing:
            await runtime.monitor_store.commit_monitor_mutation(
                DbMonitorMutation(
                    action=_monitor_lifecycle_mutation_action(action),
                    operation=operation,
                    monitor_before=before,
                    monitor_after=after,
                    state_after=(
                        None
                        if action == "delete"
                        else DbMonitorState.from_dict(
                            {
                                **state.to_dict(),
                                "last_operation_id": operation.id,
                                "last_management_operation_id": operation.id,
                                "paused_until": (
                                    proposal.get("paused_until")
                                    if action == "pause"
                                    else (
                                        None
                                        if action == "resume"
                                        else state.paused_until
                                    )
                                ),
                            }
                        )
                    ),
                )
            )

        evidence_kind = _monitor_lifecycle_commit_evidence_kind(action)
        payload = {
            "monitor_id": monitor_id,
            "action": action,
            "before": before_payload,
            "after": after_payload,
            "patch": dict(proposal.get("patch") or {}),
            "proposal_evidence_id": proposal_evidence.id,
            "proposal_fingerprint": actual_fingerprint,
            "idempotent_existing": idempotent_existing,
        }
        if action == "delete":
            payload["monitor"] = before_payload
        elif after_payload is not None:
            payload["monitor"] = after_payload
        return [
            Evidence(
                kind=evidence_kind,
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload,
                metadata={
                    "payload_fingerprint": actual_fingerprint,
                    "proposal_evidence_id": proposal_evidence.id,
                    "monitor_id": monitor_id,
                    "action": action,
                    "idempotent_existing": idempotent_existing,
                },
            )
        ]


@dataclass(frozen=True)
class DbMonitorLocalDeliveryExecutor:
    """Executor that records local monitor notification delivery evidence."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.delivery.local"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"monitor.delivery.local"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        payload_source = dict(task.input.get("payload_source") or {})
        report = await load_evidence(
            runtime,
            operation.id,
            payload_source.get("report_evidence_id"),
        )
        if report is None:
            raise RuntimeError("monitor report evidence is required for local delivery")
        delivery_kind = str(task.input.get("delivery_kind") or "")
        if delivery_kind != "local":
            raise RuntimeError("local delivery executor only supports local delivery")
        target = dict(task.input.get("target") or {})
        target_type = str(target.get("type") or target.get("channel") or "")
        if target_type not in {"runtime_console", "terminal", "stdout", "callback"}:
            raise RuntimeError("unsupported_local_delivery_target")
        payload = {
            "monitor_id": task.metadata.get("monitor_id"),
            "monitor_run_id": task.metadata.get("monitor_run_id"),
            "tick_operation_id": task.metadata.get("tick_operation_id"),
            "delivery_operation_id": operation.id,
            "delivery_kind": delivery_kind,
            "target": target,
            "target_channel": target.get("channel") or target_type,
            "format": task.input.get("format"),
            "subject": task.input.get("subject"),
            "status": "delivered",
            "idempotency_key": (
                task.input.get("idempotency_key")
                or task.metadata.get("idempotency_key")
            ),
            "report_evidence_id": report.id,
            "report_fingerprint": payload_source.get("report_fingerprint"),
            "action_plan_fingerprint": payload_source.get("action_plan_fingerprint"),
            "source_evidence_refs": list(
                payload_source.get("source_evidence_refs") or ()
            ),
        }
        return [
            Evidence(
                kind="local.notification.delivery",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload,
                metadata={
                    "monitor_id": payload["monitor_id"],
                    "monitor_run_id": payload["monitor_run_id"],
                    "tick_operation_id": payload["tick_operation_id"],
                    "monitor_delivery_kind": delivery_kind,
                    "monitor_report_fingerprint": payload["report_fingerprint"],
                    "monitor_action_fingerprint": payload["action_plan_fingerprint"],
                    "idempotency_key": payload["idempotency_key"],
                    "payload_fingerprint": stable_fingerprint(payload),
                },
            )
        ]


async def _load_monitor_proposal_evidence(
    runtime: Any,
    operation: Operation,
    task: Task,
    evidence_id: Any,
) -> Evidence | None:
    explicit = await load_evidence(runtime, operation.id, evidence_id)
    if explicit is not None:
        return explicit
    evidence = await runtime.store.list_evidence(operation.id)
    for dependency in task.dependencies:
        if dependency.kind.value != "evidence":
            continue
        if dependency.evidence_kind != "monitor.proposal":
            continue
        for item in reversed(evidence):
            if _evidence_matches_dependency(item, dependency):
                return item
    for item in reversed(evidence):
        if item.kind == "monitor.proposal" and item.accepted:
            return item
    return None


def _evidence_matches_dependency(evidence: Evidence, dependency: Any) -> bool:
    if evidence.kind != dependency.evidence_kind:
        return False
    if dependency.evidence_id is not None and evidence.id != dependency.evidence_id:
        return False
    if (
        dependency.evidence_owner is not None
        and evidence.owner != dependency.evidence_owner
    ):
        return False
    if (
        dependency.producer_task_id is not None
        and evidence.task_id != dependency.producer_task_id
    ):
        return False
    if evidence.accepted is not dependency.evidence_accepted:
        return False
    for key, value in dependency.evidence_payload.items():
        if evidence.payload.get(key) != value:
            return False
    return True


def _monitor_lifecycle_action(value: str) -> str:
    normalized = value.removeprefix("monitor.").replace("_", ".").lower()
    if normalized in {"update", "pause", "resume", "delete", "disable"}:
        return normalized
    if normalized == "disabled":
        return "disable"
    raise ValueError(f"unsupported monitor lifecycle action: {value!r}")


def _monitor_after_lifecycle_action(
    monitor: DbMonitor,
    *,
    action: str,
    patch: dict[str, Any],
    paused_until: Any = None,
) -> DbMonitor | None:
    if action == "delete":
        return None
    if action == "update":
        return monitor_with_updates(monitor, patch)
    if action == "pause":
        return monitor_with_updates(monitor, {"status": "paused", **patch})
    if action == "resume":
        return monitor_with_updates(monitor, {"status": "active", **patch})
    if action == "disable":
        return monitor_with_updates(monitor, {"status": "disabled", **patch})
    raise ValueError(f"unsupported monitor lifecycle action: {action!r}")


def _monitor_lifecycle_commit_evidence_kind(action: str) -> str:
    if action == "delete":
        return "monitor.deleted"
    if action == "disable":
        return "monitor.disabled"
    if action == "pause":
        return "monitor.paused"
    if action == "resume":
        return "monitor.resumed"
    return "monitor.state_update"


def _monitor_lifecycle_mutation_action(action: str) -> str:
    if action == "disable":
        return "update"
    return action


def _monitor_lifecycle_already_committed(
    existing: DbMonitor | None,
    *,
    after: DbMonitor | None,
    action: str,
) -> bool:
    if action == "delete":
        return existing is None
    return existing == after


def _non_executable_active_monitor_reason(plan: dict[str, Any]) -> str | None:
    kind = (plan or {}).get("kind")
    if kind not in {"planned_read", "metric_sql", "freshness_sql", "plugin_source"}:
        return "missing_executable_kind"
    if kind in {"planned_read", "metric_sql", "freshness_sql"}:
        if not isinstance(plan.get("sql"), str) or not plan["sql"].strip():
            return "missing_observation_sql"
        if not plan.get("value_path"):
            return "missing_value_path"
    if kind == "planned_read":
        if not plan.get("cursor") or not plan.get("cursor_update"):
            return "missing_cursor_strategy"
    if kind == "plugin_source":
        if not plan.get("capability_id") and not plan.get("source_kind"):
            return "missing_plugin_source_capability"
    return None
