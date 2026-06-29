"""Monitor action orchestration for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from daita.runtime import AccessMode, Operation, OperationStatus

from ...analysis import stable_fingerprint
from ...models import DbOperationContract, DbRequest
from ..monitor_helpers import _normalize_monitor_action_plan
from ..resume import (
    _db_contract_context,
    _db_request_context,
)
from .reports import DbRuntimeMonitorActionReportsMixin
from .results import DbRuntimeMonitorActionResultsMixin
from .resume import DbRuntimeMonitorActionResumeMixin
from .writes import DbRuntimeMonitorActionWritesMixin


class DbRuntimeMonitorActionsMixin(
    DbRuntimeMonitorActionResumeMixin,
    DbRuntimeMonitorActionReportsMixin,
    DbRuntimeMonitorActionWritesMixin,
    DbRuntimeMonitorActionResultsMixin,
):
    async def execute_monitor_action(
        self,
        operation_id: str,
        *,
        monitor_id: str,
        monitor_name: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_plan: dict[str, Any],
        tick_evidence_refs: tuple[dict[str, Any], ...],
        source_scope: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Execute a persisted DB monitor action inside its child operation."""

        if not self._is_setup:
            await self.setup()
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        normalized = _normalize_monitor_action_plan(
            action_plan,
            operation_id=operation_id,
        )
        fingerprint = stable_fingerprint(normalized)
        operation = await self._prepare_monitor_action_operation(
            operation,
            monitor_id=monitor_id,
            monitor_name=monitor_name,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan=normalized,
            action_kind=str(normalized.get("kind") or "invalid"),
            action_fingerprint=fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            source_scope=source_scope,
        )
        existing_result = await self._latest_monitor_action_result(
            operation.id,
            action_plan_fingerprint=fingerprint,
        )
        if existing_result is not None:
            return dict(existing_result.payload)

        plan_evidence = await self._persist_monitor_action_plan_evidence(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan=normalized,
            action_plan_fingerprint=fingerprint,
            tick_evidence_refs=tick_evidence_refs,
        )
        if normalized.get("valid") is False:
            return await self._block_monitor_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=normalized,
                action_plan_fingerprint=fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                reason=str(normalized.get("block_reason") or "invalid_action_plan"),
            )

        kind = str(normalized.get("kind") or "")
        if kind == "notification":
            return await self._execute_monitor_notification_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=normalized,
                action_plan_fingerprint=fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
            )
        if kind == "investigation":
            return await self._execute_monitor_investigation_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=normalized,
                action_plan_fingerprint=fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
            )
        if kind == "scheduled_report":
            return await self._execute_monitor_scheduled_report_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=normalized,
                action_plan_fingerprint=fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                source_scope=source_scope,
            )
        if kind == "write_proposal":
            return await self._execute_monitor_write_proposal_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=normalized,
                action_plan_fingerprint=fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                source_scope=source_scope,
            )
        return await self._block_monitor_action(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan=normalized,
            action_plan_fingerprint=fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            reason="unsupported_action_kind",
        )

    async def _prepare_monitor_action_operation(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_name: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_plan: dict[str, Any],
        action_kind: str,
        action_fingerprint: str,
        tick_evidence_refs: tuple[dict[str, Any], ...],
        source_scope: tuple[str, ...],
    ) -> Operation:
        request = DbRequest(
            prompt=f"Monitor action {action_kind} for {monitor_name}",
            source_scope=source_scope,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": action_kind,
                "monitor_action_fingerprint": action_fingerprint,
            },
        )
        contract = DbOperationContract(
            operation_type=operation.operation_type,
            required_capabilities=(
                "db.analysis.plan.validate",
                "db.analysis.checkpoint",
                "db.analysis.summarize",
            ),
            required_evidence=(
                "monitor.action_plan",
                "analysis.plan",
                "analysis.plan.validation",
                "monitor.action_result",
            ),
            access=(
                AccessMode.WRITE if action_kind == "write_proposal" else AccessMode.READ
            ),
            limits=self.config.limits,
            metadata={
                "monitor_id": monitor_id,
                "monitor_action_kind": action_kind,
                "monitor_action_fingerprint": action_fingerprint,
            },
        )
        metadata = {
            **operation.metadata,
            "monitor_id": monitor_id,
            "monitor_name": monitor_name,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "monitor_action_kind": action_kind,
            "monitor_action_fingerprint": action_fingerprint,
            "monitor_action_context": {
                "monitor_id": monitor_id,
                "monitor_name": monitor_name,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "action_kind": action_kind,
                "action_plan_fingerprint": action_fingerprint,
                "normalized_action_plan": action_plan,
                "source_scope": list(source_scope),
                "cited_tick_evidence_refs": [dict(item) for item in tick_evidence_refs],
            },
            "resume_context": {
                "request": _db_request_context(request),
                "contract": _db_contract_context(contract),
            },
        }
        updated = replace(
            operation,
            status=OperationStatus.RUNNING,
            required_evidence=frozenset(
                {
                    *operation.required_evidence,
                    "monitor.action_plan",
                    "monitor.action_result",
                }
            ),
            metadata=metadata,
        )
        await self.store.save_operation(updated)
        return updated
