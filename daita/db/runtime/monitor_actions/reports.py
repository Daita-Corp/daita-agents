"""Monitor notification, investigation, and report action execution."""

from __future__ import annotations

from typing import Any

from daita.runtime import Evidence, Operation, OperationStatus, Task

from ...analysis import DbAnalysisPlan
from ...evidence import DbEvidenceStore
from ..monitor_helpers import _monitor_report_has_analysis_work
from ..resume import (
    _db_contract_from_context,
    _db_request_from_context,
)
from ..types import DbRuntimeGovernanceBlocked


class DbRuntimeMonitorActionReportsMixin:
    async def _execute_monitor_notification_action(
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
    ) -> dict[str, Any]:
        report = await self._persist_monitor_report_evidence(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan=action_plan,
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            produced_evidence=(),
        )
        await self.kernel.complete_operation(
            operation.id,
            status=OperationStatus.SUCCEEDED,
            message=f"Monitor notification action {operation.id} succeeded.",
        )
        return await self._persist_monitor_action_result(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind="notification",
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            status="succeeded",
            extra_produced_evidence=(report,),
        )

    async def _execute_monitor_investigation_action(
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
    ) -> dict[str, Any]:
        analysis_plan = DbAnalysisPlan.from_mapping(action_plan["analysis_plan"])
        seeded = await self._seed_monitor_analysis_plan(
            operation,
            analysis_plan=analysis_plan,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
        )
        request = _db_request_from_context(operation)
        contract = _db_contract_from_context(operation)
        try:
            result = await self._run_multi_step_analysis(
                request,
                contract,
                operation,
                base_diagnostics={
                    "runtime_id": self.runtime_id,
                    "monitor_action": {
                        "monitor_id": monitor_id,
                        "monitor_run_id": monitor_run_id,
                        "tick_operation_id": tick_operation_id,
                        "action_plan_fingerprint": action_plan_fingerprint,
                        "seeded_analysis_plan_evidence_id": seeded.id,
                    },
                },
                reuse_existing_plan=True,
            )
        except Exception as exc:
            await self.kernel.fail_operation_if_active(operation.id, exc)
            return await self._persist_monitor_action_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_kind="investigation",
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                status="failed",
                block_reason=str(exc),
            )
        return await self._persist_monitor_action_result(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind="investigation",
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            status=result.status.value,
        )

    async def _execute_monitor_scheduled_report_action(
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
        source_scope: tuple[str, ...],
    ) -> dict[str, Any]:
        tasks: list[Task] = []
        evidence_store = DbEvidenceStore()
        produced: list[Evidence] = []
        try:
            for sequence, step in enumerate(action_plan.get("steps") or (), start=1):
                if step["kind"] in {"metric_sql", "freshness_sql", "planned_read"}:
                    produced.extend(
                        await self._execute_monitor_report_read_step(
                            operation,
                            monitor_id=monitor_id,
                            monitor_run_id=monitor_run_id,
                            tick_operation_id=tick_operation_id,
                            action_plan_fingerprint=action_plan_fingerprint,
                            source_scope=source_scope,
                            step=step,
                            sequence=sequence * 10,
                            tasks=tasks,
                        )
                    )
            analysis_plan = DbAnalysisPlan.from_mapping(action_plan["analysis_plan"])
            analysis_plan_evidence = await self._seed_monitor_analysis_plan(
                operation,
                analysis_plan=analysis_plan,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
            )
            if _monitor_report_has_analysis_work(analysis_plan):
                result = await self._run_multi_step_analysis(
                    _db_request_from_context(operation),
                    _db_contract_from_context(operation),
                    operation,
                    base_diagnostics={
                        "runtime_id": self.runtime_id,
                        "monitor_action": {
                            "monitor_id": monitor_id,
                            "monitor_run_id": monitor_run_id,
                            "tick_operation_id": tick_operation_id,
                            "action_plan_fingerprint": action_plan_fingerprint,
                        },
                    },
                    reuse_existing_plan=True,
                )
                if result.status is not OperationStatus.SUCCEEDED:
                    return await self._persist_monitor_action_result(
                        operation,
                        monitor_id=monitor_id,
                        monitor_run_id=monitor_run_id,
                        tick_operation_id=tick_operation_id,
                        action_kind="scheduled_report",
                        action_plan_fingerprint=action_plan_fingerprint,
                        tick_evidence_refs=tick_evidence_refs,
                        plan_evidence=plan_evidence,
                        status=result.status.value,
                        block_reason=(
                            "analysis_blocked"
                            if result.status is OperationStatus.BLOCKED
                            else None
                        ),
                    )
                produced = [
                    item
                    for item in await self.store.list_evidence(operation.id)
                    if item.accepted
                    and item.kind
                    in {
                        "query.result",
                        "quality.profile",
                        "quality.report",
                        "schema.search_result",
                        "schema.asset_profile",
                        "schema.relationship_path",
                        "lineage.trace",
                        "memory.semantic.recall",
                        "memory.fact.query",
                        "analysis.checkpoint",
                        "analysis.synthesis",
                    }
                ]
                report = await self._persist_monitor_report_evidence(
                    operation,
                    monitor_id=monitor_id,
                    monitor_run_id=monitor_run_id,
                    tick_operation_id=tick_operation_id,
                    action_plan=action_plan,
                    action_plan_fingerprint=action_plan_fingerprint,
                    tick_evidence_refs=tick_evidence_refs,
                    produced_evidence=tuple(produced),
                )
                return await self._persist_monitor_action_result(
                    operation,
                    monitor_id=monitor_id,
                    monitor_run_id=monitor_run_id,
                    tick_operation_id=tick_operation_id,
                    action_kind="scheduled_report",
                    action_plan_fingerprint=action_plan_fingerprint,
                    tick_evidence_refs=tick_evidence_refs,
                    plan_evidence=plan_evidence,
                    status="succeeded",
                    extra_produced_evidence=(report,),
                )
            validation_evidence = await self._execute_analysis_validation_task(
                operation,
                tasks,
                evidence_store,
                plan_evidence=analysis_plan_evidence,
            )
            if not validation_evidence.accepted:
                return await self._block_monitor_action(
                    operation,
                    monitor_id=monitor_id,
                    monitor_run_id=monitor_run_id,
                    tick_operation_id=tick_operation_id,
                    action_plan=action_plan,
                    action_plan_fingerprint=action_plan_fingerprint,
                    tick_evidence_refs=tick_evidence_refs,
                    plan_evidence=plan_evidence,
                    reason="analysis_plan_invalid",
                )
            synthesis = await self._execute_analysis_synthesis_task(
                operation,
                tasks,
                evidence_store,
                analysis_id=analysis_plan.analysis_id,
                step_id="report_summary",
                plan_evidence=analysis_plan_evidence,
                cited_evidence=tuple(
                    item
                    for item in await self.store.list_evidence(operation.id)
                    if item.accepted
                    and item.kind
                    in {
                        "query.result",
                        "quality.profile",
                        "analysis.checkpoint",
                    }
                ),
            )
            produced.append(synthesis)
            report = await self._persist_monitor_report_evidence(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                produced_evidence=tuple(produced),
            )
            await self.kernel.complete_operation(
                operation.id,
                status=OperationStatus.SUCCEEDED,
                message=f"Monitor report action {operation.id} succeeded.",
            )
            return await self._persist_monitor_action_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_kind="scheduled_report",
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                status="succeeded",
                extra_produced_evidence=(report,),
            )
        except DbRuntimeGovernanceBlocked as exc:
            blocked_evidence = tuple(await self.store.list_evidence(operation.id))
            await self._checkpoint_blocked_analysis_state(
                operation,
                tasks,
                evidence_store,
                governance=exc.governance,
                evidence=blocked_evidence,
            )
            return await self._block_monitor_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                reason="governance_blocked",
            )
        except Exception as exc:
            await self.kernel.fail_operation_if_active(operation.id, exc)
            return await self._persist_monitor_action_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_kind="scheduled_report",
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                status="failed",
                block_reason=str(exc),
            )

    async def _execute_monitor_report_read_step(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_plan_fingerprint: str,
        source_scope: tuple[str, ...],
        step: dict[str, Any],
        sequence: int,
        tasks: list[Task],
    ) -> tuple[Evidence, ...]:
        result = await self.execute_monitor_validated_read(
            operation,
            sql=str(step.get("sql") or ""),
            params=list(step.get("parameters") or step.get("params") or ()),
            owner=(
                str(step.get("capability_owner"))
                if step.get("capability_owner")
                else None
            ),
            reason="monitor_report_read",
            sequence=sequence,
            source_scope=source_scope,
            source_plan=step,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_role": "report",
                "monitor_action_kind": "scheduled_report",
                "monitor_action_fingerprint": action_plan_fingerprint,
                "monitor_report_step_id": step.get("id"),
                "monitor_report_step_kind": step.get("kind"),
            },
            validation_context={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "db_monitor_phase": 5,
                "monitor_action_role": "report_validation",
            },
            read_context={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "db_monitor_phase": 5,
                "monitor_action_role": "report_read",
            },
            invalid_reason="report_sql_validation_failed",
            unsafe_reason="unsafe_report_sql",
            scope_reason="report_source_scope_blocked",
            missing_result_reason="report_query_result_missing",
        )
        tasks.extend([result.validation_task, result.read_task])
        if result.status != "succeeded":
            raise RuntimeError(result.block_reason or "report_read_blocked")
        return result.evidence
