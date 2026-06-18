"""Monitor delivery dispatch for durable DB monitor ticks."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

from daita.runtime import RuntimeEvent, RuntimeEventType

from ..monitors import (
    DbMonitor,
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
    DbMonitorStore,
)


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
                    "delivery_target": dict(result.get("delivery_target") or {}),
                    "delivery_channel": result.get("delivery_channel"),
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
