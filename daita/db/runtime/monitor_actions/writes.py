"""Monitor governed write action execution and resume helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from daita.runtime import (
    AccessMode,
    ApprovalStatus,
    Evidence,
    Operation,
    OperationSnapshot,
    Task,
    TaskDependency,
    TaskDependencyKind,
    TaskStatus,
)

from ...analysis import evidence_ref
from ...fingerprints import persisted_fingerprint
from ...sql_evidence import (
    blocked_scope_resources,
    effective_source_scope,
    sql_validation_facts_from_evidence,
)
from ..governance import (
    _governance_policy_block_reason,
    _sql_validation_governance_facts,
)
from ..monitor_helpers import _terminal_monitor_approval_reason
from ..tasks.models import DbTaskSpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from daita.plugins import ExtensionRegistry
    from daita.runtime import Capability, RuntimeKernel, RuntimeStore

    from ...models import DbOperationContract
    from ..governance import _MonitorEffectGovernanceDecision
    from ..tasks.models import DbTaskPlan
    from ..tasks.runtime import DbTaskRuntime


class DbRuntimeMonitorActionWritesMixin:
    if TYPE_CHECKING:
        registry: ExtensionRegistry
        tasks: DbTaskRuntime
        store: RuntimeStore
        kernel: RuntimeKernel

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
        ) -> dict[str, Any]: ...

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

        async def plan_task_specs(
            self,
            operation: Operation,
            specs: Iterable[DbTaskSpec],
            *,
            contract: DbOperationContract | Mapping[str, Any] | None = None,
        ) -> DbTaskPlan: ...

        async def execute_task(
            self,
            task: Task,
            operation: Operation,
            context: dict[str, Any] | None = None,
        ) -> tuple[Evidence, ...]: ...

        async def evaluate_monitor_effect_governance(
            self,
            operation: Operation,
            *,
            capability: Capability,
            task: Task | None = None,
            intent: dict[str, Any],
            phase: str,
            mutate_approvals: bool = False,
            operation_override: dict[str, Any] | None = None,
        ) -> _MonitorEffectGovernanceDecision: ...

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
            validation_capability = self.tasks.validation_capability_for_sql_execute(
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
        validation_spec = DbTaskSpec(
            capability_id=validation_capability.id,
            owner=validation_capability.owner,
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
        validation_plan = await self.plan_task_specs(
            operation,
            (validation_spec,),
        )
        validation_task = validation_plan.tasks[0]
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
        sql_fingerprint = validation_facts.sql_fingerprint or persisted_fingerprint(
            {"sql": sql}
        )
        proposal_fingerprint = persisted_fingerprint(
            {
                "action_plan_fingerprint": action_plan_fingerprint,
                "sql_fingerprint": sql_fingerprint,
                "validation_evidence_id": validation_evidence.id,
                "source_evidence_refs": tick_evidence_refs,
            }
        )
        validation_payload_fingerprint = validation_evidence.metadata.get(
            "payload_fingerprint"
        ) or persisted_fingerprint(validation_evidence.payload)
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
        write_spec = DbTaskSpec(
            capability_id=write_capability.id,
            owner=write_capability.owner,
            input={
                "sql_ref": "sql.validation",
                "params": list(action_plan.get("params") or ()),
                "proposal_fingerprint": proposal_fingerprint,
                "validation_evidence_id": validation_evidence.id,
                "validation_payload_fingerprint": validation_payload_fingerprint,
            },
            reason="monitor_write_execution",
            sequence=510,
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
        write_plan = await self.plan_task_specs(
            operation,
            (validation_spec, write_spec),
        )
        write_task = write_plan.tasks[-1]
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
                    dependency.kind == TaskDependencyKind.APPROVAL
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
        write_plan = await self.plan_task_specs(
            operation,
            (
                validation_spec,
                replace(
                    write_spec,
                    dependencies=write_task.dependencies,
                ),
            ),
        )
        planned_write_task = write_plan.tasks[-1]
        if planned_write_task.dependencies != write_task.dependencies:
            planned_write_task = replace(
                planned_write_task,
                dependencies=write_task.dependencies,
            )
            await self.store.save_task(planned_write_task)
        write_task = planned_write_task
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
        existing = await self.tasks.latest_evidence(
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
                or persisted_fingerprint(validation_evidence.payload)
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
                "payload_fingerprint": persisted_fingerprint(payload),
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
        existing = await self.tasks.latest_evidence(
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
                "payload_fingerprint": persisted_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

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
            await self.tasks.latest_evidence(
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
