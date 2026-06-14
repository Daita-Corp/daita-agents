"""Monitor runtime behavior for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from daita.runtime import (
    AccessMode,
    ApprovalStatus,
    Evidence,
    Operation,
    OperationSnapshot,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
    Task,
    TaskDependency,
    TaskStatus,
)

from ..analysis import (
    DbAnalysisPlan,
    analysis_metadata,
    evidence_ref,
    stable_fingerprint,
)
from ..evidence import DbEvidenceStore
from ..models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbRequest,
)
from ..monitors import (
    DbMonitorMutation,
    DbMonitorRun,
)
from ..sql_evidence import (
    blocked_scope_resources,
    effective_source_scope,
    sql_validation_facts_from_evidence,
)
from .analysis import (
    _payload_fingerprint,
    _stable_hash,
)
from .governance import (
    _governance_policy_block_reason,
    _sql_validation_governance_facts,
)
from .resume import (
    _db_contract_context,
    _db_contract_from_context,
    _db_intent_context,
    _db_intent_from_context,
    _db_request_context,
    _db_request_from_context,
    _monitor_action_context,
)
from .types import DbRuntimeGovernanceBlocked


from .monitor_helpers import (
    _monitor_action_budget_usage,
    _monitor_action_status_from_operation,
    _monitor_report_has_analysis_work,
    _normalize_monitor_action_plan,
    _terminal_monitor_approval_reason,
)


class DbRuntimeMonitorActionsMixin:
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
        intent = DbIntent(
            kind=(
                DbIntentKind.REPORT_GENERATE
                if action_kind == "scheduled_report"
                else DbIntentKind.ANOMALY_INVESTIGATE
            ),
            access=(
                AccessMode.WRITE if action_kind == "write_proposal" else AccessMode.READ
            ),
            evidence_mode="analysis",
            requested_outputs=("analysis.synthesis", "monitor.action_result"),
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
                "intent": _db_intent_context(intent),
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
        existing = await self._latest_evidence(
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
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

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
        intent = _db_intent_from_context(operation)
        contract = _db_contract_from_context(operation)
        try:
            result = await self._run_multi_step_analysis(
                request,
                intent,
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
                    _db_intent_from_context(operation),
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

    async def _execute_monitor_write_proposal_action(
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
        sql = str(action_plan.get("sql") or "").strip()
        if not sql:
            return await self._block_monitor_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                reason="missing_write_sql",
            )
        owner = action_plan.get("capability_owner")
        try:
            write_capability = self.registry.get_capability(
                "db.sql.execute_write",
                owner=str(owner) if owner else None,
            )
            validation_capability = self._validation_capability_for_sql_execute(
                write_capability
            )
            if validation_capability is None:
                raise KeyError("db.sql.validate")
        except (KeyError, ValueError) as exc:
            return await self._block_monitor_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                reason=(
                    "ambiguous_write_capability"
                    if isinstance(exc, ValueError)
                    else "missing_write_capability"
                ),
            )
        validation_task = self._task_for_capability(
            operation,
            validation_capability,
            input={"sql": sql, "operation": "write.execute"},
            reason="monitor_write_validation",
            sequence=500,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_role": "write_validation",
            },
        )
        validation_evidence_items = await self.execute_task(
            validation_task,
            operation,
            context={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "db_monitor_phase": 7,
                "monitor_action_role": "write_validation",
            },
        )
        validation_evidence = next(
            (
                item
                for item in validation_evidence_items
                if item.kind == "sql.validation"
            ),
            None,
        )
        if validation_evidence is None:
            return await self._block_monitor_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                reason="write_validation_missing",
            )
        validation_facts = sql_validation_facts_from_evidence(validation_evidence)
        sql_fingerprint = validation_facts.sql_fingerprint or _stable_hash({"sql": sql})
        proposal_fingerprint = _stable_hash(
            {
                "action_plan_fingerprint": action_plan_fingerprint,
                "sql_fingerprint": sql_fingerprint,
                "validation_evidence_id": validation_evidence.id,
                "source_evidence_refs": tick_evidence_refs,
            }
        )
        validation_payload_fingerprint = validation_evidence.metadata.get(
            "payload_fingerprint"
        ) or _payload_fingerprint(validation_evidence.payload)
        proposal = await self._persist_monitor_write_proposal(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan_fingerprint=action_plan_fingerprint,
            proposal_fingerprint=proposal_fingerprint,
            sql_fingerprint=sql_fingerprint,
            validation_evidence=validation_evidence,
            source_evidence_refs=tick_evidence_refs,
            status="validating",
            approval_ids=(),
        )
        write_task = self._task_for_capability(
            operation,
            write_capability,
            input={
                "sql_ref": "sql.validation",
                "params": list(action_plan.get("params") or ()),
                "proposal_fingerprint": proposal_fingerprint,
                "validation_evidence_id": validation_evidence.id,
                "validation_payload_fingerprint": validation_payload_fingerprint,
            },
            reason="monitor_write_execution",
            sequence=510,
            validation_task=validation_task,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_role": "write_execution",
                "proposal_fingerprint": proposal_fingerprint,
                "sql_fingerprint": sql_fingerprint,
                "validation_evidence_id": validation_evidence.id,
                "validation_payload_fingerprint": validation_payload_fingerprint,
                "source_scope": list(effective_source_scope(source_scope, action_plan)),
                "proposal_evidence_id": proposal.id,
            },
        )
        authoritative = _sql_validation_governance_facts((validation_evidence,))
        operation_override = {
            "operation_type": "write.execute",
            "access": AccessMode.WRITE.value,
        }
        if authoritative.get("destructive_statement_classes") or authoritative.get(
            "admin_statement_classes"
        ):
            governance_decision = await self.evaluate_monitor_effect_governance(
                operation,
                capability=write_capability,
                task=write_task,
                intent={
                    "kind": "monitor.write_execution",
                    "monitor_id": monitor_id,
                    "monitor_run_id": monitor_run_id,
                    "tick_operation_id": tick_operation_id,
                    "proposal_fingerprint": proposal_fingerprint,
                    "sql_fingerprint": sql_fingerprint,
                },
                phase="write_execution",
                mutate_approvals=True,
                operation_override=operation_override,
            )
            return await self._block_monitor_write_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                proposal=proposal,
                reason=(
                    _governance_policy_block_reason(governance_decision.result)
                    or governance_decision.reason
                    or "governance_blocked"
                ),
            )
        if validation_facts.valid is not True:
            return await self._block_monitor_write_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                proposal=proposal,
                reason="write_sql_validation_failed",
            )
        blocked_resources = blocked_scope_resources(
            validation_facts.target_resources,
            effective_source_scope(source_scope, action_plan),
        )
        if blocked_resources:
            return await self._block_monitor_write_action(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan=action_plan,
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                proposal=proposal,
                reason="write_source_scope_blocked",
            )
        governance_decision = await self.evaluate_monitor_effect_governance(
            operation,
            capability=write_capability,
            task=write_task,
            intent={
                "kind": "monitor.write_execution",
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "proposal_fingerprint": proposal_fingerprint,
                "sql_fingerprint": sql_fingerprint,
                "target_resources": list(validation_facts.target_resources),
                "source_evidence_refs": [dict(item) for item in tick_evidence_refs],
            },
            phase="write_execution",
            mutate_approvals=True,
            operation_override=operation_override,
        )
        approval_requests = governance_decision.result.approval_requests
        if approval_requests:
            approval_dependencies = tuple(
                dependency
                for dependency in write_task.dependencies
                if not (
                    dependency.kind.value == "approval"
                    and dependency.approval_id is None
                    and dependency.approval_policy_id == "approval_required_for_writes"
                )
            )
            write_task = replace(
                write_task,
                dependencies=(
                    *approval_dependencies,
                    *(
                        TaskDependency(
                            kind="approval",
                            approval_status=ApprovalStatus.APPROVED,
                            approval_id=request.approval_id,
                            approval_policy_id=request.requested_by_policy_id,
                            approval_name=str(
                                request.proposed_action.get("approval") or ""
                            ),
                            operation_id=operation.id,
                        )
                        for request in approval_requests
                    ),
                ),
            )
        await self._plan_kernel_task(write_task)
        status = (
            "approval_required"
            if governance_decision.result.pending_approval or approval_requests
            else "blocked"
        )
        block_reason = (
            governance_decision.reason
            if not governance_decision.allowed
            else "write_approval_required"
        )
        proposal = await self._persist_monitor_write_proposal(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan_fingerprint=action_plan_fingerprint,
            proposal_fingerprint=proposal_fingerprint,
            sql_fingerprint=sql_fingerprint,
            validation_evidence=validation_evidence,
            source_evidence_refs=tick_evidence_refs,
            status=status,
            approval_ids=tuple(request.approval_id for request in approval_requests),
            block_reason=block_reason,
            supersede=True,
        )
        stored_write_task = await self.store.load_task(write_task.id)
        if stored_write_task is not None:
            await self.store.save_task(
                replace(
                    stored_write_task,
                    metadata={
                        **stored_write_task.metadata,
                        "proposal_evidence_id": proposal.id,
                    },
                )
            )
        await self.kernel.block_operation(
            operation.id,
            message=(
                "Monitor write execution requires approval."
                if status == "approval_required"
                else "Monitor write execution blocked by governance."
            ),
        )
        return await self._persist_monitor_action_result(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind="write_proposal",
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            status=status,
            block_reason=block_reason,
            extra_produced_evidence=(proposal,),
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
        validation_task, read_task = self.plan_validated_read_tasks(
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
            focus=step.get("metric") or step.get("purpose") or step.get("id"),
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": "scheduled_report",
                "monitor_action_fingerprint": action_plan_fingerprint,
                "monitor_report_step_id": step.get("id"),
                "monitor_report_step_kind": step.get("kind"),
            },
        )
        validation_task = await self._plan_kernel_task(validation_task)
        read_task = await self._plan_kernel_task(read_task)
        tasks.extend([validation_task, read_task])
        validation_evidence = await self.execute_task(
            validation_task,
            operation,
            context={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "db_monitor_phase": 5,
                "monitor_action_role": "report_validation",
            },
        )
        validation = next(
            (item for item in validation_evidence if item.kind == "sql.validation"),
            None,
        )
        if validation is None or not validation.accepted:
            raise RuntimeError("report_sql_validation_failed")
        facts = sql_validation_facts_from_evidence(validation)
        if facts.is_read is False or facts.valid is False:
            raise RuntimeError("unsafe_report_sql")
        blocked = blocked_scope_resources(
            facts.target_resources,
            effective_source_scope(source_scope, step),
        )
        if blocked:
            raise RuntimeError("report_source_scope_blocked")
        read_evidence = await self.execute_task(
            read_task,
            operation,
            context={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "db_monitor_phase": 5,
                "monitor_action_role": "report_read",
            },
        )
        return (*validation_evidence, *read_evidence)

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
        fingerprint = stable_fingerprint(analysis_plan.to_dict())
        existing = await self._latest_accepted_evidence(
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
                "payload_fingerprint": _payload_fingerprint(payload),
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
                "payload_fingerprint": _payload_fingerprint(payload),
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
        existing = await self._latest_evidence(
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
            "action_kind": "scheduled_report",
            "action_plan_fingerprint": action_plan_fingerprint,
            "title": action_plan.get("title"),
            "format": dict(action_plan.get("output") or {}).get("format"),
            "delivery_status": "deferred",
            "delivery_phase": 6,
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
                "monitor_action_kind": "scheduled_report",
                "monitor_action_fingerprint": action_plan_fingerprint,
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

    async def _persist_monitor_write_proposal(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_plan_fingerprint: str,
        proposal_fingerprint: str,
        sql_fingerprint: str,
        validation_evidence: Evidence,
        source_evidence_refs: tuple[dict[str, Any], ...],
        status: str,
        approval_ids: tuple[str, ...],
        block_reason: str | None = None,
        supersede: bool = False,
    ) -> Evidence:
        existing = await self._latest_evidence(
            operation.id,
            "monitor.write_proposal",
            payload={"proposal_fingerprint": proposal_fingerprint},
        )
        if existing is not None and not supersede:
            return existing
        evidence_id = f"monitor-write-proposal-{uuid4()}"
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "action_operation_id": operation.id,
            "action_plan_fingerprint": action_plan_fingerprint,
            "proposal_fingerprint": proposal_fingerprint,
            "sql_fingerprint": sql_fingerprint,
            "validation_evidence_id": validation_evidence.id,
            "validation_payload_fingerprint": (
                validation_evidence.metadata.get("payload_fingerprint")
                or _payload_fingerprint(validation_evidence.payload)
            ),
            "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            "status": status,
            "approval_ids": list(approval_ids),
            "block_reason": block_reason,
        }
        evidence = Evidence(
            id=evidence_id,
            kind="monitor.write_proposal",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=status
            in {"validating", "approval_required", "approved", "executed"},
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": "write_proposal",
                "monitor_action_fingerprint": action_plan_fingerprint,
                "proposal_fingerprint": proposal_fingerprint,
                "sql_fingerprint": sql_fingerprint,
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

    async def _block_monitor_write_action(
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
        proposal: Evidence,
        reason: str,
    ) -> dict[str, Any]:
        blocked_proposal = await self._persist_monitor_write_proposal(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_plan_fingerprint=action_plan_fingerprint,
            proposal_fingerprint=str(proposal.payload.get("proposal_fingerprint")),
            sql_fingerprint=str(proposal.payload.get("sql_fingerprint")),
            validation_evidence=next(
                item
                for item in await self.store.list_evidence(operation.id)
                if item.id == proposal.payload.get("validation_evidence_id")
            ),
            source_evidence_refs=tick_evidence_refs,
            status="blocked",
            approval_ids=tuple(proposal.payload.get("approval_ids") or ()),
            block_reason=reason,
            supersede=True,
        )
        await self.kernel.block_operation(
            operation.id,
            message=f"Monitor write action blocked: {reason}.",
        )
        return await self._persist_monitor_action_result(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind=str(action_plan.get("kind") or "write_proposal"),
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            status="blocked",
            block_reason=reason,
            extra_produced_evidence=(blocked_proposal,),
        )

    async def _persist_monitor_write_execution_result(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        action_plan_fingerprint: str,
        proposal: Evidence,
        write_task: Task,
        write_evidence: tuple[Evidence, ...],
        status: str,
        block_reason: str | None = None,
    ) -> Evidence:
        existing = await self._latest_evidence(
            operation.id,
            "monitor.write_execution",
            payload={
                "proposal_fingerprint": str(
                    proposal.payload.get("proposal_fingerprint") or ""
                )
            },
        )
        if existing is not None:
            return existing
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "action_operation_id": operation.id,
            "action_plan_fingerprint": action_plan_fingerprint,
            "proposal_evidence_id": proposal.id,
            "proposal_fingerprint": proposal.payload.get("proposal_fingerprint"),
            "sql_fingerprint": proposal.payload.get("sql_fingerprint"),
            "validation_evidence_id": proposal.payload.get("validation_evidence_id"),
            "task_id": write_task.id,
            "write_evidence_refs": [
                evidence_ref(item) for item in write_evidence if item.id
            ],
            "status": status,
            "block_reason": block_reason,
        }
        evidence = Evidence(
            id=f"monitor-write-execution-{uuid4()}",
            kind="monitor.write_execution",
            owner="db_runtime",
            operation_id=operation.id,
            task_id=write_task.id,
            accepted=status == "executed",
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_action_kind": "write_proposal",
                "monitor_action_fingerprint": action_plan_fingerprint,
                "proposal_fingerprint": proposal.payload.get("proposal_fingerprint"),
                "payload_fingerprint": _payload_fingerprint(payload),
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
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return payload

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
        plan_evidence = await self._latest_evidence(
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
            report = await self._latest_evidence(
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
    ) -> dict[str, Any]:
        write_task = next(
            (
                task
                for task in snapshot.tasks
                if task.metadata.get("monitor_action_role") == "write_execution"
            ),
            None,
        )
        proposal_fingerprint = (
            str(write_task.metadata.get("proposal_fingerprint") or "")
            if write_task is not None
            else ""
        )
        proposal = (
            await self._latest_evidence(
                snapshot.operation.id,
                "monitor.write_proposal",
                payload={"proposal_fingerprint": proposal_fingerprint},
            )
            if proposal_fingerprint
            else None
        )
        if write_task is None or proposal is None:
            return await self._persist_monitor_action_result(
                snapshot.operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_kind="write_proposal",
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                status="blocked",
                block_reason="missing_write_execution_task",
                supersede_approval_block=True,
            )
        if write_task.status is TaskStatus.SUCCEEDED:
            write_evidence = tuple(
                item for item in snapshot.evidence if item.task_id == write_task.id
            )
            execution = await self._persist_monitor_write_execution_result(
                snapshot.operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan_fingerprint=action_plan_fingerprint,
                proposal=proposal,
                write_task=write_task,
                write_evidence=write_evidence,
                status="executed",
            )
            executed_proposal = await self._persist_monitor_write_proposal(
                snapshot.operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_plan_fingerprint=action_plan_fingerprint,
                proposal_fingerprint=str(proposal.payload.get("proposal_fingerprint")),
                sql_fingerprint=str(proposal.payload.get("sql_fingerprint")),
                validation_evidence=next(
                    item
                    for item in snapshot.evidence
                    if item.id == proposal.payload.get("validation_evidence_id")
                ),
                source_evidence_refs=tick_evidence_refs,
                status="executed",
                approval_ids=tuple(proposal.payload.get("approval_ids") or ()),
                supersede=True,
            )
            return await self._persist_monitor_action_result(
                snapshot.operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                action_kind="write_proposal",
                action_plan_fingerprint=action_plan_fingerprint,
                tick_evidence_refs=tick_evidence_refs,
                plan_evidence=plan_evidence,
                status="succeeded",
                extra_produced_evidence=(executed_proposal, execution, *write_evidence),
                supersede_approval_block=True,
            )
        terminal_reason = _terminal_monitor_approval_reason(snapshot.approval_requests)
        return await self._persist_monitor_action_result(
            snapshot.operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            action_kind=str(action_plan.get("kind") or "write_proposal"),
            action_plan_fingerprint=action_plan_fingerprint,
            tick_evidence_refs=tick_evidence_refs,
            plan_evidence=plan_evidence,
            status="blocked",
            block_reason=terminal_reason or "write_execution_not_completed",
            extra_produced_evidence=(proposal,),
            supersede_approval_block=True,
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
        await self.monitor_store.commit_monitor_mutation(
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
