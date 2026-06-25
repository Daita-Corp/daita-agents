"""Monitor action dispatch for durable DB monitor ticks."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

from daita.runtime import Evidence, RuntimeEvent, RuntimeEventType

from ..monitors import (
    DbMonitor,
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
    DbMonitorStore,
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
