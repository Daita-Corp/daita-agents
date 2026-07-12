"""Monitor action resume finalization and run summary repair."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from uuid import uuid4

from daita.runtime import (
    Operation,
    OperationSnapshot,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
    TaskStatus,
)

from ...monitors import DbMonitorMutation, DbMonitorRun
from ..monitor_helpers import (
    _monitor_action_status_from_operation,
    _terminal_monitor_approval_reason,
)
from ..resume import _monitor_action_context

if TYPE_CHECKING:
    from daita.runtime import Evidence, RuntimeStore

    from ...monitors import DbMonitorStore
    from ..tasks.runtime import DbTaskRuntime


class DbRuntimeMonitorActionResumeMixin:
    if TYPE_CHECKING:
        tasks: DbTaskRuntime
        store: RuntimeStore
        monitor_store: DbMonitorStore
        runtime_id: str
        runtime_kind: str

        async def _latest_monitor_action_result(
            self,
            operation_id: str,
            *,
            action_plan_fingerprint: str,
        ) -> Evidence | None: ...

        def _has_pending_approvals(self, snapshot: OperationSnapshot) -> bool: ...

        async def _persist_monitor_action_plan_evidence(
            self,
            operation: Operation,
            *,
            monitor_id: str,
            monitor_run_id: str,
            tick_operation_id: str,
            action_plan: dict[str, Any],
            action_plan_fingerprint: str,
            tick_evidence_refs: tuple[dict[str, Any], ...],
        ) -> Evidence: ...

        async def _finalize_resumed_monitor_write_action(
            self,
            snapshot: OperationSnapshot,
            *,
            monitor_id: str,
            monitor_run_id: str,
            tick_operation_id: str,
            action_plan: dict[str, Any],
            action_plan_fingerprint: str,
            tick_evidence_refs: tuple[dict[str, Any], ...],
            plan_evidence: Evidence,
        ) -> dict[str, Any]: ...

        async def _persist_monitor_report_evidence(
            self,
            operation: Operation,
            *,
            monitor_id: str,
            monitor_run_id: str,
            tick_operation_id: str,
            action_plan: dict[str, Any],
            action_plan_fingerprint: str,
            tick_evidence_refs: tuple[dict[str, Any], ...],
            produced_evidence: tuple[Evidence, ...],
        ) -> Evidence: ...

        async def _persist_monitor_action_result(
            self,
            operation: Operation,
            *,
            monitor_id: str,
            monitor_run_id: str,
            tick_operation_id: str,
            action_kind: str,
            action_plan_fingerprint: str,
            tick_evidence_refs: tuple[dict[str, Any], ...],
            plan_evidence: Evidence,
            status: str,
            block_reason: str | None = None,
            extra_produced_evidence: tuple[Evidence, ...] = (),
            supersede_approval_block: bool = False,
        ) -> dict[str, Any]: ...

        async def commit_monitor_mutation(
            self,
            mutation: DbMonitorMutation,
        ) -> None: ...

    async def _finalize_resumed_monitor_action(
        self,
        snapshot: OperationSnapshot,
    ) -> None:
        context = _monitor_action_context(snapshot.operation)
        if not context:
            return
        fingerprint = str(context.get("action_plan_fingerprint") or "")
        if not fingerprint:
            return
        existing = await self._latest_monitor_action_result(
            snapshot.operation.id,
            action_plan_fingerprint=fingerprint,
        )
        if existing is not None:
            is_resumable_write = context.get("action_kind") == "write_proposal" and (
                _terminal_monitor_approval_reason(snapshot.approval_requests)
                or (
                    snapshot.operation.status is OperationStatus.BLOCKED
                    and not self._has_pending_approvals(snapshot)
                )
                or any(
                    task.metadata.get("monitor_action_role") == "write_execution"
                    and task.status is TaskStatus.SUCCEEDED
                    for task in snapshot.tasks
                )
            )
            if not is_resumable_write:
                await self._refresh_monitor_action_run_summary(
                    snapshot.operation,
                    result_payload=dict(existing.payload),
                )
                return

        action_plan = dict(context.get("normalized_action_plan") or {})
        monitor_id = str(context.get("monitor_id") or "")
        monitor_run_id = str(context.get("monitor_run_id") or "")
        tick_operation_id = str(context.get("tick_operation_id") or "")
        tick_evidence_refs = tuple(
            dict(item)
            for item in context.get("cited_tick_evidence_refs") or ()
            if isinstance(item, dict)
        )
        plan_evidence = await self.tasks.latest_evidence(
            snapshot.operation.id,
            "monitor.action_plan",
            payload={"action_plan_fingerprint": fingerprint},
        )
        if plan_evidence is None:
            plan_evidence = await self._persist_monitor_action_plan_evidence(
                snapshot.operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=fingerprint,
                tick_evidence_refs=tick_evidence_refs,
            )

        if action_plan.get("kind") == "write_proposal":
            result_payload = await self._finalize_resumed_monitor_write_action(
                snapshot,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
            )
            await self._refresh_monitor_action_run_summary(
                snapshot.operation,
                result_payload=result_payload,
            )
            return

        status = _monitor_action_status_from_operation(snapshot.operation)
        if action_plan.get("kind") == "scheduled_report":
            report = await self.tasks.latest_evidence(
                snapshot.operation.id,
                "monitor.report",
                payload={"action_plan_fingerprint": fingerprint},
            )
            if report is None and status == "succeeded":
                report = await self._persist_monitor_report_evidence(
                    snapshot.operation,
                    monitor_id=monitor_id,
                    monitor_run_id=monitor_run_id,
                    tick_operation_id=tick_operation_id,
                    action_plan=action_plan,
                    action_plan_fingerprint=fingerprint,
                    tick_evidence_refs=tick_evidence_refs,
                    produced_evidence=tuple(
                        item
                        for item in await self.store.list_evidence(
                            snapshot.operation.id
                        )
                        if item.accepted
                        and item.kind
                        in {
                            "analysis.synthesis",
                            "analysis.checkpoint",
                            "query.result",
                            "quality.profile",
                            "quality.report",
                            "schema.search_result",
                            "schema.asset_profile",
                            "schema.relationship_path",
                            "lineage.trace",
                            "memory.semantic.recall",
                            "memory.fact.query",
                        }
                    ),
                )
            extra = (report,) if report is not None else ()
        else:
            extra = ()

        result_payload = await self._persist_monitor_action_result(
            snapshot.operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind=str(action_plan.get("kind") or context.get("action_kind")),
            action_plan_fingerprint=fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            status=status,
            block_reason=(
                snapshot.operation.metadata.get("block_reason")
                if status in {"blocked", "failed"}
                else None
            ),
            extra_produced_evidence=extra,
        )
        await self._refresh_monitor_action_run_summary(
            snapshot.operation,
            result_payload=result_payload,
        )

    async def _refresh_monitor_action_run_summary(
        self,
        operation: Operation,
        *,
        result_payload: dict[str, Any],
    ) -> None:
        monitor_id = str(result_payload.get("monitor_id") or "")
        monitor_run_id = str(result_payload.get("monitor_run_id") or "")
        if not monitor_id or not monitor_run_id:
            return
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            return
        runs = await self.monitor_store.list_monitor_runs(monitor_id)
        run = next((item for item in runs if item.id == monitor_run_id), None)
        if run is None:
            return
        produced_refs = [
            dict(item)
            for item in result_payload.get("produced_evidence_refs") or ()
            if isinstance(item, dict)
        ]
        report_evidence_id = next(
            (
                str(item.get("id"))
                for item in produced_refs
                if item.get("kind") == "monitor.report" and item.get("id")
            ),
            None,
        )
        summary = {
            **run.summary,
            "action_status": result_payload.get("status"),
            "action_kind": result_payload.get("action_kind"),
            "action_plan_fingerprint": result_payload.get("action_plan_fingerprint"),
            "action_evidence_id": result_payload.get("action_result_evidence_id"),
            "report_evidence_id": report_evidence_id,
            "action_task_ids": list(result_payload.get("task_ids") or ()),
            "action_produced_evidence_refs": produced_refs,
            "action_block_reason": result_payload.get("block_reason"),
            "action_budget_usage": dict(result_payload.get("budget_usage") or {}),
        }
        if summary == run.summary:
            return
        updated_run = DbMonitorRun.from_dict({**run.to_dict(), "summary": summary})
        state = await self.monitor_store.load_monitor_state(monitor_id)
        await self.commit_monitor_mutation(
            DbMonitorMutation(
                action="run",
                operation=operation,
                events=(
                    RuntimeEvent(
                        id=f"monitor-action-resume-event-{uuid4()}",
                        type=RuntimeEventType.OPERATION_UPDATED,
                        operation_id=operation.id,
                        runtime_id=self.runtime_id,
                        runtime_kind=self.runtime_kind,
                        evidence_id=result_payload.get("action_result_evidence_id"),
                        message=(
                            f"Monitor {monitor_id} action resume summary refreshed."
                        ),
                        payload={
                            "monitor_id": monitor_id,
                            "monitor_run_id": monitor_run_id,
                            "tick_operation_id": result_payload.get(
                                "tick_operation_id"
                            ),
                            "status": result_payload.get("status"),
                            "action_evidence_id": result_payload.get(
                                "action_result_evidence_id"
                            ),
                        },
                    ),
                ),
                monitor_before=monitor,
                state_before=state,
                state_after=state,
                run_after=updated_run,
            )
        )
