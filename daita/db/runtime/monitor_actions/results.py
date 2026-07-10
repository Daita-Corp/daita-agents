"""Monitor action evidence and result materialization."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from daita.runtime import Evidence, Operation

from ...analysis import (
    DbAnalysisPlan,
    analysis_metadata,
    evidence_ref,
)
from ...fingerprints import persisted_fingerprint
from ..monitor_helpers import _monitor_action_budget_usage


class DbRuntimeMonitorActionResultsMixin:
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
    ) -> Evidence:
        existing = await self.tasks.latest_evidence(
            operation.id,
            "monitor.action_plan",
            payload={"action_plan_fingerprint": action_plan_fingerprint},
        )
        if existing is not None:
            return existing
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "action_kind": action_plan.get("kind"),
            "action_plan_fingerprint": action_plan_fingerprint,
            "normalized_action_plan": action_plan,
            "cited_tick_evidence_refs": [dict(item) for item in tick_evidence_refs],
        }
        evidence = Evidence(
            id=f"monitor-action-plan-{uuid4()}",
            kind="monitor.action_plan",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=action_plan.get("valid") is not False,
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": action_plan.get("kind"),
                "monitor_action_fingerprint": action_plan_fingerprint,
                "payload_fingerprint": persisted_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

    async def _seed_monitor_analysis_plan(
        self,
        operation: Operation,
        *,
        analysis_plan: DbAnalysisPlan,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_plan_fingerprint: str,
        tick_evidence_refs: tuple[dict[str, Any], ...],
    ) -> Evidence:
        fingerprint = persisted_fingerprint(analysis_plan.to_dict())
        existing = await self.tasks.latest_accepted_evidence(
            operation.id,
            "analysis.plan",
            payload={"analysis_id": analysis_plan.analysis_id},
        )
        if (
            existing is not None
            and existing.payload.get("plan_fingerprint") == fingerprint
        ):
            return existing
        payload = {
            **analysis_plan.to_dict(),
            "plan_fingerprint": fingerprint,
        }
        evidence = Evidence(
            id=f"monitor-analysis-plan-{uuid4()}",
            kind="analysis.plan",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload=payload,
            metadata={
                **analysis_metadata(
                    analysis_id=analysis_plan.analysis_id,
                    step_id="monitor_action_plan",
                    phase="plan",
                ),
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_fingerprint": action_plan_fingerprint,
                "cited_tick_evidence_refs": [dict(item) for item in tick_evidence_refs],
                "payload_fingerprint": persisted_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

    async def _block_monitor_action(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_plan: dict[str, Any],
        action_plan_fingerprint: str,
        tick_evidence_refs: tuple[dict[str, Any], ...],
        plan_evidence: Evidence,
        reason: str,
    ) -> dict[str, Any]:
        checkpoint = await self._persist_monitor_action_checkpoint(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind=str(action_plan.get("kind") or "invalid"),
            action_plan_fingerprint=action_plan_fingerprint,
            reason=reason,
            plan_evidence=plan_evidence,
        )
        await self.kernel.block_operation(
            operation.id,
            message=f"Monitor action blocked: {reason}.",
        )
        return await self._persist_monitor_action_result(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind=str(action_plan.get("kind") or "invalid"),
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            status="blocked",
            block_reason=reason,
            extra_produced_evidence=(checkpoint,),
        )

    async def _persist_monitor_action_checkpoint(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_kind: str,
        action_plan_fingerprint: str,
        reason: str,
        plan_evidence: Evidence,
    ) -> Evidence:
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "action_kind": action_kind,
            "action_plan_fingerprint": action_plan_fingerprint,
            "pause_reason": reason,
            "plan_evidence_id": plan_evidence.id,
        }
        evidence = Evidence(
            id=f"monitor-action-checkpoint-{uuid4()}",
            kind="analysis.checkpoint",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload=payload,
            metadata={
                "analysis_id": f"monitor-action-{operation.id}",
                "analysis_step_id": "monitor_action_blocked",
                "analysis_step_kind": "checkpoint",
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": action_kind,
                "monitor_action_fingerprint": action_plan_fingerprint,
                "payload_fingerprint": persisted_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

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
    ) -> Evidence:
        existing = await self.tasks.latest_evidence(
            operation.id,
            "monitor.report",
            payload={"action_plan_fingerprint": action_plan_fingerprint},
        )
        if existing is not None:
            return existing
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "action_kind": action_plan.get("kind") or "scheduled_report",
            "action_plan_fingerprint": action_plan_fingerprint,
            "title": action_plan.get("title"),
            "format": dict(action_plan.get("output") or {}).get("format"),
            "template": action_plan.get("template"),
            "delivery_status": "deferred",
            "delivery_phase": action_plan.get("delivery_phase") or 6,
            "delivery_intent": dict(action_plan.get("delivery_intent") or {}),
            "cited_tick_evidence_refs": [dict(item) for item in tick_evidence_refs],
            "produced_evidence_refs": [
                evidence_ref(item) for item in produced_evidence if item.id
            ],
        }
        evidence = Evidence(
            id=f"monitor-report-{uuid4()}",
            kind="monitor.report",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": action_plan.get("kind") or "scheduled_report",
                "monitor_action_fingerprint": action_plan_fingerprint,
                "payload_fingerprint": persisted_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

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
    ) -> dict[str, Any]:
        existing = await self._latest_monitor_action_result(
            operation.id,
            action_plan_fingerprint=action_plan_fingerprint,
        )
        if existing is not None and not (
            supersede_approval_block
            and existing.payload.get("block_reason")
            in {"governance_approval_required", "approval_required"}
        ):
            return dict(existing.payload)
        tasks = tuple(await self.store.list_tasks(operation.id))
        evidence_items = tuple(await self.store.list_evidence(operation.id))
        produced_refs = [
            evidence_ref(item)
            for item in (*evidence_items, *extra_produced_evidence)
            if item.id
            and item.kind
            in {
                "analysis.plan",
                "analysis.plan.validation",
                "analysis.checkpoint",
                "analysis.synthesis",
                "query.result",
                "quality.report",
                "quality.profile",
                "monitor.report",
                "monitor.write_proposal",
                "monitor.write_execution",
                "write.execution",
                "sql.execution",
            }
        ]
        budget_usage = _monitor_action_budget_usage(evidence_items)
        evidence_id = f"monitor-action-result-{uuid4()}"
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "action_kind": action_kind,
            "action_plan_fingerprint": action_plan_fingerprint,
            "status": status,
            "block_reason": block_reason,
            "action_result_evidence_id": evidence_id,
            "cited_tick_evidence_refs": [dict(item) for item in tick_evidence_refs],
            "plan_evidence_id": plan_evidence.id,
            "task_ids": [task.id for task in tasks],
            "produced_evidence_refs": produced_refs,
            "budget_usage": budget_usage,
        }
        evidence = Evidence(
            id=evidence_id,
            kind="monitor.action_result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=status == "succeeded",
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": action_kind,
                "monitor_action_fingerprint": action_plan_fingerprint,
                "payload_fingerprint": persisted_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return payload
