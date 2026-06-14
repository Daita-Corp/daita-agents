"""
Skeleton database runtime built on the extension registry.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import time
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from daita.plugins import ExtensionRegistry, PluginContext, PluginKind, ServiceRegistry
from daita.runtime import (
    AccessMode,
    Capability,
    ContextAudience,
    ContextBlock,
    Evidence,
    ApprovalRequest,
    ApprovalStatus,
    GovernanceAuditRecord,
    GovernanceResult,
    InMemoryApprovalChannel,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    OperationSnapshot,
    PolicyEvaluator,
    PolicyEffect,
    PolicyDecisionTrace,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeKernel,
    RuntimeKernelExecutorFailed,
    RuntimeKernelGovernanceBlocked,
    RuntimeKernelLeaseLost,
    RuntimeKernelTaskAlreadyTerminal,
    RuntimeKernelTaskNotRunnable,
    RuntimeStore,
    SQLiteRuntimeStore,
    Task,
    TaskDependency,
    TaskStatus,
)
from daita.skills import SkillResolution, SkillResolver

from .analysis import (
    DbAnalysisPlan,
    analysis_metadata,
    capability_contract_for_step_kind,
    evidence_ref,
    stable_fingerprint,
    validate_analysis_plan_payload,
    with_analysis_evidence_trace,
)
from .evidence import DbEvidenceStore
from .execution import DbOperationExecutor
from .governance import default_db_policies
from .llm_service import DbLLMService, db_llm_service_from_metadata
from .models import (
    DbIntent,
    DbIntentKind,
    DbLimits,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
    DbRuntimeConfig,
    DbRuntimeInspection,
)
from .monitor_plugin_planning import (
    MonitorPluginPlanner,
    MonitorPluginPlanningBlocked,
    monitor_delivery_source_refs,
    monitor_report_fingerprint,
    monitor_source_observed_value,
)
from .monitors import (
    DbMonitor,
    DbMonitorInspection,
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
    DbMonitorStore,
    InMemoryDbMonitorStore,
    SQLiteDbMonitorStore,
    monitor_with_updates,
)
from .monitor_scheduler import DbMonitorScheduler
from .planning import DbContractBuilder, DbIntentClassifier
from .runtime_extensions import DbRuntimePlanningPlugin
from .synthesis import DbSynthesizer
from .sql_evidence import (
    blocked_scope_resources,
    effective_source_scope,
    sql_validation_facts_from_evidence,
)
from .verification import DbVerifier

_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }
)
_DEFAULT_TASK_LEASE_SECONDS = 300.0


@dataclass(frozen=True)
class _GovernancePersistence:
    result: GovernanceResult
    audit_record: GovernanceAuditRecord
    approvals_to_request: tuple[ApprovalRequest, ...]
    events: tuple[RuntimeEvent, ...]


@dataclass(frozen=True)
class _MonitorEffectGovernanceDecision:
    status: str
    reason: str | None
    result: GovernanceResult
    audit_record: GovernanceAuditRecord

    @property
    def allowed(self) -> bool:
        return self.status == "allowed"


@dataclass(frozen=True)
class _AnalysisPlanState:
    plan: DbAnalysisPlan
    plan_evidence: Evidence
    validation_evidence: Evidence
    revision_evidence: Evidence | None = None

    @property
    def selected_plan_evidence(self) -> Evidence:
        return self.revision_evidence or self.plan_evidence


@dataclass(frozen=True)
class _SourcePreparationSnapshot:
    evidence: Evidence
    store_id: str
    schema_fingerprint: str
    cached_at: float


class DbRuntime:
    """Operation-centric database runtime.

    This class is the new architecture entry point for DB operations. It owns
    plugin setup, contract building, task execution, and typed operation
    results without delegating to the legacy generic Agent tool loop.
    """

    runtime_kind = "db"

    def __init__(
        self,
        *,
        source: Any = None,
        config: DbRuntimeConfig | None = None,
        registry: ExtensionRegistry | None = None,
        plugins: tuple[Any, ...] | list[Any] = (),
        store: RuntimeStore | None = None,
        monitor_store: DbMonitorStore | None = None,
        approval_channel: InMemoryApprovalChannel | None = None,
        runtime_id: str | None = None,
        db_llm_service: DbLLMService | None = None,
    ) -> None:
        self.source = source
        self.config = config or DbRuntimeConfig()
        self.registry = registry or ExtensionRegistry()
        self.store = store or InMemoryRuntimeStore()
        self.monitor_store = monitor_store or _default_monitor_store(self.store)
        self.approval_channel = approval_channel or InMemoryApprovalChannel(self.store)
        self.runtime_id = runtime_id or f"db-runtime-{uuid4()}"
        self.intent_classifier = DbIntentClassifier()
        self.verifier = DbVerifier()
        self.synthesizer = DbSynthesizer()
        self.db_llm_service = db_llm_service or db_llm_service_from_metadata(
            self.config.metadata
        )
        self._setup_context: PluginContext | None = None
        self._is_setup = False
        self._schema_profile_cache: dict[str, Any] | None = None
        self._catalog_source_cache: _SourcePreparationSnapshot | None = None
        self._operation_results: list[DbOperationResult] = []
        self._audit_log: list[dict[str, Any]] = []
        self.kernel = RuntimeKernel(
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            extension_registry=self.registry,
            runtime_store=self.store,
            approval_channel=self.approval_channel,
            fact_provider=self,
        )

        self.registry.register(
            DbRuntimePlanningPlugin(llm_capable=self.db_llm_service.available)
        )
        self.registry.register_many(self.config.plugins)
        self.registry.register_many(tuple(plugins))

    @property
    def is_setup(self) -> bool:
        """Whether plugin setup has run for this runtime."""
        return self._is_setup

    @property
    def setup_context(self) -> PluginContext | None:
        """The last setup context, exposed for diagnostics and tests."""
        return self._setup_context

    @property
    def operation_results(self) -> tuple[DbOperationResult, ...]:
        """Typed operation results retained by this in-memory runtime."""
        return tuple(self._operation_results)

    @property
    def audit_log(self) -> tuple[dict[str, Any], ...]:
        """Redacted operation audit summaries retained by this runtime."""
        return tuple(dict(entry) for entry in self._audit_log)

    async def create_monitor(self, monitor: DbMonitor) -> DbMonitor:
        """Persist a DB monitor definition and audit the create operation."""
        operation, evidence, events = self._monitor_management_artifacts(
            "create",
            monitor=monitor,
            evidence_kind="monitor.definition",
            payload={"monitor": monitor.to_dict()},
        )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="create",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_after=monitor,
                state_after=DbMonitorState(
                    monitor_id=monitor.id,
                    last_operation_id=operation.id,
                    last_management_operation_id=operation.id,
                ),
            )
        )
        return monitor

    async def list_monitors(
        self, *, status: str | None = None
    ) -> tuple[DbMonitor, ...]:
        """List durable DB monitor definitions."""
        return await self.monitor_store.list_monitors(status=status)

    async def inspect_monitor(self, monitor_id: str) -> DbMonitorInspection | None:
        """Return a monitor definition with durable state and run summaries."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            return None
        state = await self.monitor_store.load_monitor_state(monitor_id)
        runs = await self.monitor_store.list_monitor_runs(monitor_id)
        return DbMonitorInspection(monitor=monitor, state=state, runs=runs)

    async def update_monitor(
        self,
        monitor_id: str,
        patch: dict[str, Any],
    ) -> DbMonitor:
        """Patch a durable monitor definition and audit the update."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        updated = monitor_with_updates(monitor, patch)
        operation, evidence, events = self._monitor_management_artifacts(
            "update",
            monitor=updated,
            evidence_kind="monitor.state_update",
            payload={
                "monitor_id": monitor_id,
                "before": monitor.to_dict(),
                "after": updated.to_dict(),
                "patch": dict(patch),
            },
        )
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="update",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
                monitor_after=updated,
                state_after=DbMonitorState.from_dict(
                    {
                        **state.to_dict(),
                        "last_operation_id": operation.id,
                        "last_management_operation_id": operation.id,
                    }
                ),
            )
        )
        return updated

    async def pause_monitor(
        self,
        monitor_id: str,
        *,
        paused_until: str | None = None,
    ) -> DbMonitor:
        """Mark a monitor paused and persist its pause state."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        updated = monitor_with_updates(monitor, {"status": "paused"})
        operation, evidence, events = self._monitor_management_artifacts(
            "pause",
            monitor=updated,
            evidence_kind="monitor.state_update",
            payload={
                "monitor_id": monitor_id,
                "before": monitor.to_dict(),
                "after": updated.to_dict(),
                "paused_until": paused_until,
            },
        )
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="pause",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
                monitor_after=updated,
                state_after=DbMonitorState.from_dict(
                    {
                        **state.to_dict(),
                        "last_operation_id": operation.id,
                        "last_management_operation_id": operation.id,
                        "paused_until": paused_until,
                    }
                ),
            )
        )
        return updated

    async def resume_monitor(self, monitor_id: str) -> DbMonitor:
        """Resume a paused monitor."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        updated = monitor_with_updates(monitor, {"status": "active"})
        operation, evidence, events = self._monitor_management_artifacts(
            "resume",
            monitor=updated,
            evidence_kind="monitor.state_update",
            payload={
                "monitor_id": monitor_id,
                "before": monitor.to_dict(),
                "after": updated.to_dict(),
            },
        )
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="resume",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
                monitor_after=updated,
                state_after=DbMonitorState.from_dict(
                    {
                        **state.to_dict(),
                        "last_operation_id": operation.id,
                        "last_management_operation_id": operation.id,
                        "paused_until": None,
                    }
                ),
            )
        )
        return updated

    async def delete_monitor(self, monitor_id: str) -> DbMonitor:
        """Delete a monitor control-plane record and audit the removal."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        operation, evidence, events = self._monitor_management_artifacts(
            "delete",
            monitor=monitor,
            evidence_kind="monitor.definition",
            payload={"deleted_monitor": monitor.to_dict()},
        )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="delete",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
            )
        )
        return monitor

    async def list_monitor_approvals(
        self,
        *,
        monitor_id: str | None = None,
        monitor_run_id: str | None = None,
        pending_only: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        """List monitor-related approvals without mutating approval state."""

        approvals: list[dict[str, Any]] = []
        for approval in await self.store.list_approval_requests():
            if pending_only and approval.status is not ApprovalStatus.PENDING:
                continue
            context = await self._monitor_approval_context(approval)
            if not context:
                continue
            if monitor_id is not None and context.get("monitor_id") != monitor_id:
                continue
            if (
                monitor_run_id is not None
                and context.get("monitor_run_id") != monitor_run_id
            ):
                continue
            approvals.append(
                {
                    "approval_id": approval.approval_id,
                    "operation_id": approval.operation_id,
                    "task_id": getattr(approval, "task_id", None),
                    "status": approval.status.value,
                    "reason": approval.reason,
                    "risk": approval.risk.value,
                    "requested_by_policy_id": approval.requested_by_policy_id,
                    "owner": approval.owner,
                    "context": context,
                }
            )
        return tuple(approvals)

    async def approve_monitor_approval(self, approval_id: str) -> ApprovalRequest:
        """Approve a monitor approval through the configured approval channel."""

        return await self.approval_channel.approve(approval_id)

    async def reject_monitor_approval(self, approval_id: str) -> ApprovalRequest:
        """Reject a monitor approval through the configured approval channel."""

        return await self.approval_channel.reject(approval_id)

    async def cancel_monitor_approval(self, approval_id: str) -> ApprovalRequest:
        """Cancel a monitor approval through the configured approval channel."""

        return await self.approval_channel.cancel(approval_id)

    async def tick_monitors(
        self, *, now: datetime | str | None = None
    ) -> tuple[DbMonitorRun, ...]:
        """Run one durable DB monitor scheduler pass."""
        if not self._is_setup:
            await self.setup()
        scheduler = DbMonitorScheduler(runtime=self)
        results = await scheduler.run_once(now=now)
        return tuple(result.run for result in results)

    async def record_monitor_command_result(
        self,
        *,
        request: DbRequest,
        kind: str,
        command: dict[str, Any],
        status: OperationStatus,
        answer: str,
        operation_id: str | None = None,
        evidence: tuple[Evidence, ...] = (),
        evidence_kind: str | None = None,
        payload: dict[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
        diagnostics: dict[str, Any] | None = None,
        persist_operation: bool = True,
    ) -> DbOperationResult:
        """Record a prompt-level monitor command result through runtime audit.

        This is a control-plane audit path only. It persists operation/evidence/
        event records for monitor management commands without planning SQL,
        creating tasks, invoking executors, or evaluating governance.
        """
        command_payload = dict(command)
        operation_id = operation_id or f"monitor-command-{kind}-{uuid4()}"
        persisted_evidence = tuple(evidence)
        if persist_operation:
            operation = Operation(
                id=operation_id,
                operation_type=f"monitor.{kind}",
                status=status,
                request={
                    "kind": f"monitor.{kind}",
                    "prompt": request.prompt,
                    "command": command_payload,
                },
                required_evidence=(
                    frozenset({evidence_kind})
                    if evidence_kind is not None
                    else frozenset()
                ),
                metadata={
                    "runtime_id": self.runtime_id,
                    "runtime_kind": self.runtime_kind,
                    "control_plane": "db.monitor",
                    "monitor_id": command_payload.get("monitor_id"),
                    "command_kind": kind,
                },
            )
            if evidence_kind is not None:
                persisted_evidence = (
                    Evidence(
                        id=f"monitor-command-evidence-{uuid4()}",
                        kind=evidence_kind,
                        owner="db.monitor",
                        operation_id=operation_id,
                        payload=dict(payload or {}),
                        accepted=status is OperationStatus.SUCCEEDED,
                    ),
                )
            created_event = RuntimeEvent(
                id=f"monitor-command-event-{uuid4()}",
                type=RuntimeEventType.OPERATION_CREATED,
                operation_id=operation_id,
                runtime_id=self.runtime_id,
                runtime_kind=self.runtime_kind,
                message=f"Monitor command operation {operation_id} created.",
                payload={"operation_type": operation.operation_type},
            )
            completed_event = RuntimeEvent(
                id=f"monitor-command-event-{uuid4()}",
                type=RuntimeEventType.OPERATION_UPDATED,
                operation_id=operation_id,
                runtime_id=self.runtime_id,
                runtime_kind=self.runtime_kind,
                evidence_id=(persisted_evidence[0].id if persisted_evidence else None),
                message=f"Monitor command {kind} finished with {status.value}.",
                payload={
                    "status": status.value,
                    "command_kind": kind,
                    "monitor_id": command_payload.get("monitor_id"),
                    "warnings": list(warnings),
                },
            )
            await self.store.save_operation(operation)
            for item in persisted_evidence:
                await self.store.save_evidence(item)
            await self.store.append_event(created_event)
            await self.store.append_event(completed_event)

        result = DbOperationResult(
            operation_id=operation_id,
            request=request,
            intent=DbIntent(
                kind=DbIntentKind.ADMIN,
                confidence=float(command_payload.get("confidence") or 0.0),
                access=AccessMode.NONE,
                diagnostics={
                    "command_kind": kind,
                    **dict(command_payload.get("diagnostics") or {}),
                },
            ),
            contract=DbOperationContract(
                operation_type=f"monitor.{kind}",
                required_evidence=tuple(item.kind for item in persisted_evidence),
                access=AccessMode.NONE,
                metadata={
                    "control_plane": "db.monitor",
                    "command": command_payload,
                },
            ),
            status=status,
            answer=answer,
            evidence=persisted_evidence,
            warnings=warnings,
            diagnostics={
                "command": command_payload,
                **dict(diagnostics or {}),
            },
        )
        return await self._record_operation_result(result)

    def _monitor_management_artifacts(
        self,
        action: str,
        *,
        monitor: DbMonitor,
        evidence_kind: str,
        payload: dict[str, Any],
    ) -> tuple[Operation, tuple[Evidence, ...], tuple[RuntimeEvent, ...]]:
        operation_id = f"monitor-{action}-{uuid4()}"
        operation = Operation(
            id=operation_id,
            operation_type=f"monitor.{action}",
            status=OperationStatus.SUCCEEDED,
            request={
                "kind": f"monitor.{action}",
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
            },
            required_evidence=frozenset({evidence_kind}),
            metadata={
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "monitor_status": monitor.status,
                "control_plane": "db.monitor",
            },
        )
        evidence = Evidence(
            id=f"monitor-evidence-{uuid4()}",
            kind=evidence_kind,
            owner="db.monitor",
            operation_id=operation_id,
            payload=payload,
        )
        created_event = RuntimeEvent(
            id=f"monitor-event-{uuid4()}",
            type=RuntimeEventType.OPERATION_CREATED,
            operation_id=operation_id,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            message=f"Monitor operation {operation_id} created.",
            payload={"operation_type": operation.operation_type},
        )
        completed_event = RuntimeEvent(
            id=f"monitor-event-{uuid4()}",
            type=RuntimeEventType.OPERATION_UPDATED,
            operation_id=operation_id,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            evidence_id=evidence.id,
            message=f"Monitor {monitor.id} {action} committed.",
            payload={
                "status": OperationStatus.SUCCEEDED.value,
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "action": action,
            },
        )
        return operation, (evidence,), (created_event, completed_event)

    def register_plugin(self, plugin: Any) -> None:
        """Register a plugin before runtime setup.

        Runtime plugin declarations are collected before setup so future phases
        can build contracts from a stable registry snapshot.
        """
        if self._is_setup:
            raise RuntimeError("cannot register plugins after DbRuntime.setup()")
        self.registry.register(plugin)

    async def setup(self, *, agent_id: str | None = None) -> None:
        """Set up registered plugins with a typed runtime context."""
        if self._is_setup:
            return
        services = ServiceRegistry(
            {
                "db_runtime": self,
                "extension_registry": self.registry,
                "runtime_store": self.store,
                "db_llm_service": self.db_llm_service,
            }
        )
        context = PluginContext(
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            agent_id=agent_id,
            services=services,
            config=MappingProxyType(
                {
                    "profile": self.config.profile,
                    "limits": self.config.limits.to_dict(),
                    "metadata": self.config.metadata,
                    "db_llm": self.db_llm_service.safe_metadata,
                }
            ),
        )
        await self.registry.setup_all(context)
        self._setup_context = context
        self._is_setup = True

    async def teardown(self) -> None:
        """Tear down registered plugins in reverse registration order."""
        if not self._is_setup:
            return
        await self.registry.teardown_all()
        self._setup_context = None
        self._is_setup = False

    async def inspect(self) -> DbRuntimeInspection:
        """Return a diagnostics snapshot of registry-backed runtime state."""
        diagnostics = tuple(
            {
                "plugin_id": diagnostic.plugin_id,
                "declaration_type": diagnostic.declaration_type,
                "declaration_id": diagnostic.declaration_id,
                "message": diagnostic.message,
            }
            for diagnostic in self.registry.diagnostics
        )
        return DbRuntimeInspection(
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            source_type=(
                type(self.source).__name__ if self.source is not None else "none"
            ),
            source_repr=_safe_source_repr(self.source),
            profile=self.config.profile,
            plugin_ids=tuple(
                plugin_id
                for plugin_id in self.registry.plugin_ids
                if plugin_id != "db_runtime"
            ),
            capability_count=len(self.registry.capabilities),
            executor_count=len(self.registry.executors),
            evidence_schema_count=len(self.registry.evidence_schemas),
            policy_count=len(self.registry.policies),
            context_provider_count=len(self.registry.context_providers),
            tool_view_count=len(self.registry.tool_views),
            worker_count=len(self.registry.workers),
            capability_ids=tuple(
                f"{capability.owner}:{capability.id}"
                for capability in self.registry.capabilities
            ),
            executor_ids=tuple(executor.id for executor in self.registry.executors),
            evidence_schema_kinds=tuple(
                f"{schema.owner}:{schema.kind}"
                for schema in self.registry.evidence_schemas
            ),
            policy_ids=tuple(
                f"{policy.owner}:{policy.id}" for policy in self.registry.policies
            ),
            context_provider_ids=tuple(
                f"{provider.owner}:{provider.id}"
                for provider in self.registry.context_providers
            ),
            tool_view_names=tuple(
                tool_view.name for tool_view in self.registry.tool_views
            ),
            worker_ids=tuple(
                f"{worker.owner}:{worker.id}" for worker in self.registry.workers
            ),
            diagnostics=diagnostics,
            operation_count=await self._stored_operation_count(),
            last_operation_id=await self._last_stored_operation_id(),
            limits=self.config.limits.to_dict(),
            metadata=self.config.metadata,
        )

    async def render_context(
        self,
        *,
        prompt: str,
        audience: ContextAudience = ContextAudience.PRIMARY_MODEL,
        token_budget: int = 2000,
    ) -> tuple[ContextBlock, ...]:
        """Render context blocks from registered context providers."""
        blocks: list[ContextBlock] = []
        for provider in self.registry.context_providers:
            block = await provider.render(
                {"prompt": prompt, "runtime_id": self.runtime_id},
                ContextAudience(audience),
                token_budget,
            )
            if block is not None:
                blocks.append(block)
        return tuple(sorted(blocks, key=lambda item: item.priority, reverse=True))

    async def execute_task(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        """Execute one runtime task through the shared runtime kernel."""
        capability = self._capability_for_task(task)
        if capability.executor != task.executor_id:
            raise ValueError(
                f"task executor {task.executor_id!r} does not match capability "
                f"{task.capability_id!r} executor {capability.executor!r}"
            )
        stored_task = await self.store.load_task(task.id)
        if stored_task is None:
            task = replace(
                task,
                dependencies=task.dependencies
                or _task_dependencies_for_capability(operation, capability),
            )
            task = await self._plan_kernel_task(task)
        elif (
            stored_task.status is TaskStatus.PENDING and stored_task.input != task.input
        ):
            task = replace(
                stored_task,
                input=task.input,
                dependencies=task.dependencies or stored_task.dependencies,
                metadata={
                    **stored_task.metadata,
                    **{
                        key: value
                        for key, value in task.metadata.items()
                        if key in {"owner", "reason"}
                    },
                },
            )
            await self.store.save_task(task)
        else:
            task = stored_task
        default_dependencies = _task_dependencies_for_capability(operation, capability)
        if not task.dependencies and default_dependencies:
            task = replace(task, dependencies=default_dependencies)
            await self.store.save_task(task)
        try:
            result = await self.kernel.execute_task(
                task.id,
                context={
                    "capability_owner": capability.owner,
                    **(context or {}),
                },
                lease_owner=self.runtime_id,
                lease_seconds=_DEFAULT_TASK_LEASE_SECONDS,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            result = exc.result
            raise DbRuntimeGovernanceBlocked(
                operation=result.operation if result is not None else operation,
                task=result.task if result is not None else task,
                governance=(
                    result.governance
                    if result is not None and result.governance is not None
                    else GovernanceResult(False, True, False)
                ),
            ) from exc
        except RuntimeKernelTaskAlreadyTerminal as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                f"Task {blocked_task.id} is already {blocked_task.status.value}; "
                "completed tasks are not replayed without explicit invalidation.",
            ) from exc
        except RuntimeKernelLeaseLost as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                f"Task {blocked_task.id} lease was lost before commit.",
            ) from exc
        except RuntimeKernelTaskNotRunnable as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            readiness = (
                result.events[-1].payload.get("readiness", {})
                if result is not None and result.events
                else {}
            )
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                str(exc),
                readiness=readiness,
            ) from exc
        except RuntimeKernelExecutorFailed as exc:
            raise (exc.__cause__ or exc) from exc
        return result.evidence

    async def execute_capability(
        self,
        capability_id: str,
        *,
        owner: str | None = None,
        operation_type: str,
        input: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> tuple[Evidence, ...]:
        """Create and execute a single task for one registered capability."""
        if not self._is_setup:
            await self.setup()
        capability = self.registry.get_capability(capability_id, owner=owner)
        output_evidence = capability.output_evidence
        validation_capability = self._validation_capability_for_sql_execute(capability)
        if (
            capability.id
            in {
                "db.sql.execute_read",
                "db.sql.execute_write",
            }
            and validation_capability is not None
        ):
            output_evidence = frozenset(
                (
                    *sorted(validation_capability.output_evidence),
                    *sorted(output_evidence),
                )
            )
        try:
            operation = await self.kernel.create_operation(
                operation_id=operation_id,
                operation_type=operation_type,
                request={
                    "prompt": _prompt_from_direct_input(input or {}),
                    "input": input or {},
                    "capability_id": capability.id,
                    "capability_owner": capability.owner,
                },
                required_evidence=output_evidence,
                metadata={
                    "direct_capability_id": capability.id,
                    "direct_capability_owner": capability.owner,
                    "access": capability.access.value,
                },
                evaluate_governance=False,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            raise DbRuntimeGovernanceBlocked(
                operation=exc.operation
                or await self.store.load_operation(operation_id or ""),
                task=None,
                governance=exc.governance or GovernanceResult(False, True, False),
            ) from exc
        task_plans = self._direct_capability_tasks(
            operation,
            capability,
            input or {},
            validation_capability=validation_capability,
        )
        tasks = []
        for task in task_plans:
            tasks.append(await self._plan_kernel_task(task))
        primary_task = tasks[-1]
        if (
            capability.id == "db.sql.execute_write"
            and validation_capability is None
            and not (input or {}).get("validated_evidence_id")
        ):
            blocked_task = await self.kernel.block_task(
                primary_task.id,
                message=(
                    "Direct write execution requires db.sql.validate "
                    "or a validated_evidence_id."
                ),
            )
            await self.kernel.block_operation(
                operation.id,
                message=(
                    "Direct write execution requires db.sql.validate "
                    "or a validated_evidence_id."
                ),
            )
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                "Direct write execution requires db.sql.validate or validated_evidence_id.",
            )
        try:
            await self.kernel.evaluate_operation_governance(
                operation.id,
                capability=capability,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            await self.kernel.block_task(
                primary_task.id,
                message=f"Task {primary_task.id} blocked by operation governance.",
                payload={
                    "governance": (
                        exc.governance.to_dict() if exc.governance is not None else {}
                    )
                },
            )
            raise DbRuntimeGovernanceBlocked(
                operation=exc.operation or operation,
                task=replace(primary_task, status=TaskStatus.BLOCKED),
                governance=exc.governance or GovernanceResult(False, True, False),
            ) from exc
        try:
            collected: list[Evidence] = []
            for task in tasks:
                collected.extend(await self.execute_task(task, operation))
            evidence = tuple(collected)
        except DbRuntimeGovernanceBlocked:
            raise
        except Exception as exc:
            await self.kernel.fail_operation_if_active(operation.id, exc)
            raise
        await self.kernel.complete_operation(operation.id)
        return evidence

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

    async def execute_monitor_source_observation(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        source_step: dict[str, Any],
        cursor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one deterministic read-only plugin source step for a tick."""

        if not self._is_setup:
            await self.setup()
        step = dict(source_step or {})
        try:
            plan = MonitorPluginPlanner(tuple(self.registry.capabilities)).plan_source(
                step,
                cursor=cursor or {},
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
            )
        except MonitorPluginPlanningBlocked as exc:
            return {
                "status": "blocked",
                "block_reason": exc.reason,
                "details": exc.details,
                "task_ids": [],
                "plugin_evidence_refs": [],
            }

        task = self._monitor_plugin_task_for_capability(
            operation,
            plan.capability,
            input_payload=plan.input_payload,
            input_hash=plan.input_hash,
            idempotency_key=plan.idempotency_key,
            reason=plan.reason,
            sequence=plan.sequence,
            metadata=plan.metadata,
        )
        try:
            evidence = await self._execute_or_reuse_monitor_plugin_task(
                task,
                operation,
                context={
                    "monitor_id": monitor_id,
                    "monitor_run_id": monitor_run_id,
                    "tick_operation_id": tick_operation_id,
                    "db_monitor_phase": 6,
                    "monitor_observation_role": "plugin_source",
                },
            )
        except DbRuntimeGovernanceBlocked:
            return {
                "status": "blocked",
                "block_reason": "governance_blocked",
                "capability_id": plan.capability.id,
                "capability_owner": plan.capability.owner,
                "task_ids": [task.id],
                "plugin_evidence_refs": [],
            }
        except Exception as exc:
            return {
                "status": "failed",
                "block_reason": "plugin_source_failed",
                "details": {"type": type(exc).__name__, "message": str(exc)},
                "capability_id": plan.capability.id,
                "capability_owner": plan.capability.owner,
                "task_ids": [task.id],
                "plugin_evidence_refs": [],
            }
        if not evidence:
            return {
                "status": "blocked",
                "block_reason": "plugin_source_evidence_missing",
                "capability_id": plan.capability.id,
                "capability_owner": plan.capability.owner,
                "task_ids": [task.id],
                "plugin_evidence_refs": [],
            }
        return {
            "status": "succeeded",
            "source_kind": plan.intent_payload.get("source_kind"),
            "capability_id": plan.capability.id,
            "capability_owner": plan.capability.owner,
            "task_ids": [task.id],
            "plugin_evidence_refs": [evidence_ref(item) for item in evidence],
            "value": monitor_source_observed_value(step, evidence),
        }

    async def execute_monitor_delivery(
        self,
        operation_id: str,
        *,
        monitor_id: str,
        monitor_name: str,
        monitor_run_id: str,
        tick_operation_id: str,
        report_evidence_id: str | None = None,
        governed: bool = False,
    ) -> dict[str, Any]:
        """Deliver Phase 5 monitor report evidence through a registered capability."""

        if not self._is_setup:
            await self.setup()
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        report = await self._monitor_report_for_delivery(
            operation.id,
            report_evidence_id=report_evidence_id,
        )
        if report is None:
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=None,
                capability=None,
                action_plan_fingerprint="",
                report_fingerprint="",
                source_evidence_refs=(),
                task_ids=(),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key="",
                block_reason="missing_monitor_report",
            )

        intent = dict(report.payload.get("delivery_intent") or {})
        action_fingerprint = str(report.payload.get("action_plan_fingerprint") or "")
        report_fingerprint = monitor_report_fingerprint(report)
        source_refs = monitor_delivery_source_refs(
            report,
            tuple(await self.store.list_evidence(operation.id)),
        )
        if not intent:
            return {
                "status": "skipped",
                "block_reason": "missing_delivery_intent",
                "report_evidence_id": report.id,
                "source_evidence_refs": [dict(item) for item in source_refs],
            }

        operation = await self._prepare_monitor_delivery_operation(
            operation,
            monitor_id=monitor_id,
            monitor_name=monitor_name,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            report=report,
            source_evidence_refs=source_refs,
        )
        existing = await self._latest_evidence(
            operation.id,
            "monitor.delivery_result",
            payload={"report_fingerprint": report_fingerprint},
        )
        if existing is not None and not await self._monitor_delivery_can_resume(
            operation.id, existing
        ):
            return dict(existing.payload)

        try:
            plan = MonitorPluginPlanner(
                tuple(self.registry.capabilities)
            ).plan_delivery(
                intent,
                report=report,
                source_evidence_refs=source_refs,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
            )
        except MonitorPluginPlanningBlocked as exc:
            await self._persist_monitor_delivery_plan(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_intent=intent,
                report=report,
                source_evidence_refs=source_refs,
                accepted=False,
                block_reason=exc.reason,
                details=exc.details,
            )
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(
                    intent.get("delivery_kind") or intent.get("mode") or ""
                ),
                capability=None,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=(),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key="",
                block_reason=exc.reason,
            )

        capability = plan.capability
        idempotency_key = plan.idempotency_key
        existing = await self._latest_evidence(
            operation.id,
            "monitor.delivery_result",
            payload={"idempotency_key": idempotency_key},
        )
        if existing is not None and not await self._monitor_delivery_can_resume(
            operation.id, existing
        ):
            return dict(existing.payload)

        plan_evidence = await self._persist_monitor_delivery_plan(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            delivery_intent=intent,
            report=report,
            source_evidence_refs=source_refs,
            capability=capability,
            idempotency_key=idempotency_key,
            accepted=True,
        )
        governance_decision = await self.evaluate_monitor_effect_governance(
            operation,
            capability=capability,
            intent=plan.intent_payload,
            phase="delivery",
            mutate_approvals=governed,
        )
        if not governance_decision.allowed:
            task_ids: tuple[str, ...] = ()
            if (
                governed
                and governance_decision.result.pending_approval
                and governance_decision.result.approval_requests
            ):
                task = self._monitor_plugin_task_for_capability(
                    operation,
                    capability,
                    input_payload=plan.input_payload,
                    input_hash=plan.input_hash,
                    idempotency_key=idempotency_key,
                    reason=plan.reason,
                    sequence=plan.sequence,
                    metadata=plan.metadata,
                    approval_requests=governance_decision.result.approval_requests,
                )
                await self._plan_kernel_task(task)
                task_ids = (task.id,)
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(plan.intent_payload.get("delivery_kind") or ""),
                capability=capability,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=task_ids,
                plugin_result_evidence=(),
                status=governance_decision.status,
                idempotency_key=idempotency_key,
                block_reason=governance_decision.reason,
                plan_evidence=plan_evidence,
            )

        task = self._monitor_plugin_task_for_capability(
            operation,
            capability,
            input_payload=plan.input_payload,
            input_hash=plan.input_hash,
            idempotency_key=idempotency_key,
            reason=plan.reason,
            sequence=plan.sequence,
            metadata=plan.metadata,
        )
        try:
            plugin_evidence = await self._execute_or_reuse_monitor_plugin_task(
                task,
                operation,
                context={
                    "monitor_id": monitor_id,
                    "monitor_run_id": monitor_run_id,
                    "tick_operation_id": tick_operation_id,
                    "db_monitor_phase": 6,
                    "monitor_action_role": "delivery",
                },
            )
        except DbRuntimeGovernanceBlocked:
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(
                    intent.get("delivery_kind") or intent.get("mode") or ""
                ),
                capability=capability,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=(task.id,),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key=idempotency_key,
                block_reason="governance_blocked",
                plan_evidence=plan_evidence,
            )
        except Exception as exc:
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(
                    intent.get("delivery_kind") or intent.get("mode") or ""
                ),
                capability=capability,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=(task.id,),
                plugin_result_evidence=(),
                status="failed",
                idempotency_key=idempotency_key,
                block_reason=str(exc),
                plan_evidence=plan_evidence,
            )
        return await self._persist_monitor_delivery_result(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            delivery_kind=str(intent.get("delivery_kind") or intent.get("mode") or ""),
            capability=capability,
            action_plan_fingerprint=action_fingerprint,
            report_fingerprint=report_fingerprint,
            source_evidence_refs=source_refs,
            task_ids=(task.id,),
            plugin_result_evidence=plugin_evidence,
            status="succeeded",
            idempotency_key=idempotency_key,
            plan_evidence=plan_evidence,
            supersede_approval_block=True,
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

    async def _monitor_report_for_delivery(
        self,
        operation_id: str,
        *,
        report_evidence_id: str | None,
    ) -> Evidence | None:
        reports = [
            item
            for item in await self.store.list_evidence(operation_id)
            if item.kind == "monitor.report" and item.accepted
        ]
        if report_evidence_id is not None:
            return next(
                (item for item in reports if item.id == report_evidence_id), None
            )
        return reports[-1] if reports else None

    async def _prepare_monitor_delivery_operation(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_name: str,
        monitor_run_id: str,
        tick_operation_id: str,
        report: Evidence,
        source_evidence_refs: tuple[dict[str, Any], ...],
    ) -> Operation:
        metadata = {
            **operation.metadata,
            "monitor_delivery_context": {
                "monitor_id": monitor_id,
                "monitor_name": monitor_name,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "report_evidence_id": report.id,
                "report_fingerprint": (
                    report.metadata.get("payload_fingerprint")
                    or _payload_fingerprint(report.payload)
                ),
                "action_plan_fingerprint": report.payload.get(
                    "action_plan_fingerprint"
                ),
                "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            },
        }
        updated = replace(
            operation,
            status=OperationStatus.RUNNING,
            required_evidence=frozenset(
                {
                    *operation.required_evidence,
                    "monitor.delivery_plan",
                    "monitor.delivery_result",
                }
            ),
            metadata=metadata,
        )
        await self.store.save_operation(updated)
        return updated

    async def _persist_monitor_delivery_plan(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        delivery_intent: dict[str, Any],
        report: Evidence,
        source_evidence_refs: tuple[dict[str, Any], ...],
        capability: Capability | None = None,
        idempotency_key: str | None = None,
        accepted: bool,
        block_reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Evidence:
        report_fingerprint = str(
            report.metadata.get("payload_fingerprint") or ""
        ) or _payload_fingerprint(report.payload)
        existing = await self._latest_evidence(
            operation.id,
            "monitor.delivery_plan",
            payload={"report_fingerprint": report_fingerprint},
            accepted=accepted,
        )
        if existing is not None:
            return existing
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "delivery_kind": delivery_intent.get("delivery_kind")
            or delivery_intent.get("mode"),
            "capability_id": capability.id if capability is not None else None,
            "capability_owner": capability.owner if capability is not None else None,
            "report_evidence_id": report.id,
            "report_fingerprint": report_fingerprint,
            "action_plan_fingerprint": report.payload.get("action_plan_fingerprint"),
            "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            "delivery_intent": dict(delivery_intent),
            "idempotency_key": idempotency_key,
            "status": "planned" if accepted else "blocked",
            "block_reason": block_reason,
            "details": dict(details or {}),
        }
        evidence = Evidence(
            id=f"monitor-delivery-plan-{uuid4()}",
            kind="monitor.delivery_plan",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=accepted,
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_delivery_kind": payload["delivery_kind"],
                "monitor_report_fingerprint": report_fingerprint,
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

    async def _persist_monitor_delivery_result(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        delivery_kind: str | None,
        capability: Capability | None,
        action_plan_fingerprint: str,
        report_fingerprint: str,
        source_evidence_refs: tuple[dict[str, Any], ...],
        task_ids: tuple[str, ...],
        plugin_result_evidence: tuple[Evidence, ...],
        status: str,
        idempotency_key: str,
        block_reason: str | None = None,
        plan_evidence: Evidence | None = None,
        supersede_approval_block: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            existing = await self._latest_evidence(
                operation.id,
                "monitor.delivery_result",
                payload={"idempotency_key": idempotency_key},
            )
            if existing is not None and not (
                supersede_approval_block
                and existing.payload.get("block_reason")
                == "governance_approval_required"
            ):
                return dict(existing.payload)
        evidence_id = f"monitor-delivery-result-{uuid4()}"
        plugin_refs = [evidence_ref(item) for item in plugin_result_evidence if item.id]
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "delivery_operation_id": operation.id,
            "delivery_kind": delivery_kind,
            "capability_id": capability.id if capability is not None else None,
            "capability_owner": capability.owner if capability is not None else None,
            "action_plan_fingerprint": action_plan_fingerprint,
            "report_fingerprint": report_fingerprint,
            "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            "task_ids": list(task_ids),
            "plugin_result_evidence_refs": plugin_refs,
            "status": status,
            "idempotency_key": idempotency_key,
            "block_reason": block_reason,
            "plan_evidence_id": plan_evidence.id if plan_evidence is not None else None,
            "delivery_result_evidence_id": evidence_id,
            "report_delivery_status": (
                "delivered"
                if status == "succeeded"
                else "blocked" if status == "blocked" else "failed"
            ),
        }
        evidence = Evidence(
            id=evidence_id,
            kind="monitor.delivery_result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=status == "succeeded",
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_delivery_kind": delivery_kind,
                "monitor_report_fingerprint": report_fingerprint,
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        if status == "succeeded":
            await self.kernel.complete_operation(
                operation.id,
                status=OperationStatus.SUCCEEDED,
                message=f"Monitor delivery {operation.id} succeeded.",
            )
        elif status == "blocked":
            await self.kernel.block_operation(
                operation.id,
                message=f"Monitor delivery blocked: {block_reason}.",
            )
        else:
            await self.kernel.fail_operation_if_active(
                operation.id, RuntimeError(block_reason or status)
            )
        return payload

    def _monitor_plugin_task_for_capability(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input_payload: dict[str, Any],
        input_hash: str,
        idempotency_key: str,
        reason: str,
        sequence: int,
        metadata: dict[str, Any],
        approval_requests: tuple[ApprovalRequest, ...] = (),
    ) -> Task:
        task_key = _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "capability_owner": capability.owner,
                "idempotency_key": idempotency_key,
            }
        )
        task = Task(
            id=f"monitor-plugin-task-{task_key[:32]}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**input_payload, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            metadata={
                **dict(metadata),
                "owner": capability.owner,
                "reason": reason,
                "sequence": sequence,
                "input_hash": input_hash,
                "idempotency_key": idempotency_key,
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        approval_dependencies = tuple(
            TaskDependency(
                kind="approval",
                approval_status=ApprovalStatus.APPROVED,
                approval_id=request.approval_id,
                approval_policy_id=request.requested_by_policy_id,
                approval_name=str(request.proposed_action.get("approval") or ""),
                operation_id=operation.id,
            )
            for request in approval_requests
        )
        if approval_dependencies:
            task = replace(
                task,
                dependencies=(*task.dependencies, *approval_dependencies),
            )
        return task

    async def _execute_or_reuse_monitor_plugin_task(
        self,
        task: Task,
        operation: Operation,
        *,
        context: dict[str, Any],
    ) -> tuple[Evidence, ...]:
        stored = await self.store.load_task(task.id)
        if stored is None:
            stored = await self._plan_kernel_task(task)
        elif stored.status in _TERMINAL_TASK_STATUSES:
            return tuple(
                item
                for item in await self.store.list_evidence(operation.id)
                if item.task_id == stored.id
            )
        elif stored.input != task.input and stored.status is TaskStatus.PENDING:
            stored = replace(
                stored,
                input=task.input,
                metadata={**stored.metadata, **task.metadata},
            )
            await self.store.save_task(stored)
        return await self.execute_task(stored, operation, context=context)

    async def _monitor_delivery_can_resume(
        self,
        operation_id: str,
        result: Evidence,
    ) -> bool:
        if result.payload.get("block_reason") != "governance_approval_required":
            return False
        approvals = await self.store.list_approval_requests(operation_id)
        if not approvals:
            return False
        if any(
            approval.status
            in {
                ApprovalStatus.REJECTED,
                ApprovalStatus.CANCELLED,
                ApprovalStatus.EXPIRED,
            }
            for approval in approvals
        ):
            return False
        return all(approval.status is ApprovalStatus.APPROVED for approval in approvals)

    async def _monitor_approval_context(
        self,
        approval: ApprovalRequest,
    ) -> dict[str, Any]:
        operation = await self.store.load_operation(approval.operation_id)
        metadata = operation.metadata if operation is not None else {}
        context: dict[str, Any] = {}
        action_context = metadata.get("monitor_action_context")
        if isinstance(action_context, dict):
            context.update(
                {
                    "kind": f"monitor.{action_context.get('action_kind')}",
                    "monitor_id": action_context.get("monitor_id"),
                    "monitor_run_id": action_context.get("monitor_run_id"),
                    "tick_operation_id": action_context.get("tick_operation_id"),
                    "operation_id": approval.operation_id,
                    "action_plan_fingerprint": action_context.get(
                        "action_plan_fingerprint"
                    ),
                    "source_evidence_refs": list(
                        action_context.get("cited_tick_evidence_refs") or ()
                    ),
                }
            )
        delivery_context = metadata.get("monitor_delivery_context")
        if isinstance(delivery_context, dict):
            context.update(
                {
                    "kind": "monitor.delivery",
                    "monitor_id": delivery_context.get("monitor_id"),
                    "monitor_run_id": delivery_context.get("monitor_run_id"),
                    "tick_operation_id": delivery_context.get("tick_operation_id"),
                    "operation_id": approval.operation_id,
                    "report_fingerprint": delivery_context.get("report_fingerprint"),
                    "action_plan_fingerprint": delivery_context.get(
                        "action_plan_fingerprint"
                    ),
                    "source_evidence_refs": list(
                        delivery_context.get("source_evidence_refs") or ()
                    ),
                }
            )
        governance_facts = (
            approval.proposed_action.get("request", {}).get("governance_facts", {})
            if isinstance(approval.proposed_action.get("request"), dict)
            else {}
        )
        monitor_effect = (
            governance_facts.get("monitor_effect")
            if isinstance(governance_facts, dict)
            else {}
        )
        monitor_effect = monitor_effect if isinstance(monitor_effect, dict) else {}
        intent = monitor_effect.get("intent")
        intent = intent if isinstance(intent, dict) else {}
        context.update(
            {
                key: value
                for key, value in {
                    "capability_id": (
                        approval.proposed_action.get("request", {})
                        .get("capability", {})
                        .get("id")
                        if isinstance(approval.proposed_action.get("request"), dict)
                        and isinstance(
                            approval.proposed_action["request"].get("capability"),
                            dict,
                        )
                        else None
                    ),
                    "capability_owner": (
                        approval.proposed_action.get("request", {})
                        .get("capability", {})
                        .get("owner")
                        if isinstance(approval.proposed_action.get("request"), dict)
                        and isinstance(
                            approval.proposed_action["request"].get("capability"),
                            dict,
                        )
                        else None
                    ),
                    "target": intent.get("target"),
                    "risk": approval.risk.value,
                    "reason": approval.reason,
                }.items()
                if value is not None
            }
        )
        return context if context.get("monitor_id") else {}

    async def _finalize_resumed_monitor_delivery(
        self,
        snapshot: OperationSnapshot,
    ) -> None:
        context = _monitor_delivery_context(snapshot.operation)
        if not context:
            return
        report_id = context.get("report_evidence_id")
        result = await self._latest_evidence(
            snapshot.operation.id,
            "monitor.delivery_result",
            payload={
                "report_fingerprint": str(context.get("report_fingerprint") or "")
            },
        )
        terminal_reason = _terminal_monitor_approval_reason(snapshot.approval_requests)
        if result is not None and terminal_reason:
            payload = await self._persist_monitor_delivery_result(
                snapshot.operation,
                monitor_id=str(context.get("monitor_id") or ""),
                monitor_run_id=str(context.get("monitor_run_id") or ""),
                tick_operation_id=str(context.get("tick_operation_id") or ""),
                delivery_kind=result.payload.get("delivery_kind"),
                capability=(
                    self.registry.get_capability(
                        str(result.payload.get("capability_id")),
                        owner=str(result.payload.get("capability_owner")),
                    )
                    if result.payload.get("capability_id")
                    and result.payload.get("capability_owner")
                    else None
                ),
                action_plan_fingerprint=str(
                    result.payload.get("action_plan_fingerprint") or ""
                ),
                report_fingerprint=str(result.payload.get("report_fingerprint") or ""),
                source_evidence_refs=tuple(
                    dict(item)
                    for item in result.payload.get("source_evidence_refs") or ()
                    if isinstance(item, dict)
                ),
                task_ids=tuple(
                    str(item) for item in result.payload.get("task_ids") or ()
                ),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key=str(result.payload.get("idempotency_key") or ""),
                block_reason=terminal_reason,
                supersede_approval_block=True,
            )
            await self._refresh_monitor_delivery_run_summary(
                snapshot.operation,
                result_payload=payload,
            )
            return
        if result is not None and not await self._monitor_delivery_can_resume(
            snapshot.operation.id, result
        ):
            await self._refresh_monitor_delivery_run_summary(
                snapshot.operation,
                result_payload=dict(result.payload),
            )
            return
        payload = await self.execute_monitor_delivery(
            snapshot.operation.id,
            monitor_id=str(context.get("monitor_id") or ""),
            monitor_name=str(context.get("monitor_name") or ""),
            monitor_run_id=str(context.get("monitor_run_id") or ""),
            tick_operation_id=str(context.get("tick_operation_id") or ""),
            report_evidence_id=str(report_id) if report_id else None,
            governed=True,
        )
        await self._refresh_monitor_delivery_run_summary(
            snapshot.operation,
            result_payload=payload,
        )

    async def _refresh_monitor_delivery_run_summary(
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
        summary = {
            **run.summary,
            "delivery_status": result_payload.get("status"),
            "delivery_kind": result_payload.get("delivery_kind"),
            "delivery_operation_id": result_payload.get("delivery_operation_id"),
            "delivery_result_evidence_id": result_payload.get(
                "delivery_result_evidence_id"
            ),
            "delivery_plugin_result_evidence_refs": list(
                result_payload.get("plugin_result_evidence_refs") or ()
            ),
            "delivery_task_ids": list(result_payload.get("task_ids") or ()),
            "delivery_block_reason": result_payload.get("block_reason"),
            "delivery_idempotency_key": result_payload.get("idempotency_key"),
        }
        if summary == run.summary:
            return
        updated_run = DbMonitorRun.from_dict({**run.to_dict(), "summary": summary})
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state_after = state
        if state is not None:
            state_after = DbMonitorState.from_dict(
                {
                    **state.to_dict(),
                    "cursor": {
                        **state.cursor,
                        "last_delivery_status": result_payload.get("status"),
                        "last_delivery_result_evidence_id": result_payload.get(
                            "delivery_result_evidence_id"
                        ),
                    },
                }
            )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="run",
                operation=operation,
                events=(
                    RuntimeEvent(
                        id=f"monitor-delivery-resume-event-{uuid4()}",
                        type=RuntimeEventType.OPERATION_UPDATED,
                        operation_id=operation.id,
                        runtime_id=self.runtime_id,
                        runtime_kind=self.runtime_kind,
                        evidence_id=result_payload.get("delivery_result_evidence_id"),
                        message=(
                            f"Monitor {monitor_id} delivery resume summary refreshed."
                        ),
                        payload={
                            "monitor_id": monitor_id,
                            "monitor_run_id": monitor_run_id,
                            "status": result_payload.get("status"),
                        },
                    ),
                ),
                monitor_before=monitor,
                state_before=state,
                state_after=state_after,
                run_after=updated_run,
            )
        )

    async def inspect_operation(self, operation_id: str) -> OperationSnapshot | None:
        """Inspect persisted state for one operation."""
        inspect = getattr(self.store, "inspect_operation", None)
        if inspect is not None:
            return await inspect(operation_id)
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            return None
        tasks = tuple(await self.store.list_tasks(operation_id))
        completed_task_ids = tuple(
            task.id
            for task in tasks
            if task.status
            in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.SKIPPED,
            }
        )
        resumable_task_ids = tuple(
            task.id
            for task in tasks
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
        )
        return OperationSnapshot(
            operation=operation,
            tasks=tasks,
            evidence=tuple(await self.store.list_evidence(operation_id)),
            events=tuple(await self.store.list_events(operation_id)),
            policy_decisions=tuple(
                await self.store.list_policy_decisions(operation_id)
            ),
            governance_audit_records=tuple(
                await self.store.list_governance_audit_records(operation_id)
            ),
            approval_requests=tuple(
                await self.store.list_approval_requests(operation_id)
            ),
            resumable_task_ids=resumable_task_ids,
            completed_task_ids=completed_task_ids,
        )

    async def inspect_analysis_operation(self, operation_id: str) -> dict[str, Any]:
        """Inspect persisted multi-step analysis progress for one operation."""
        snapshot = await self.inspect_operation(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        return self._analysis_progress_payload(snapshot)

    async def resume_operation(self, operation_id: str) -> OperationSnapshot:
        """Resume persisted operation state without re-running completed tasks."""
        if not self._is_setup:
            await self.setup()
        snapshot = await self.inspect_operation(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        await self.kernel.append_event(
            RuntimeEventType.OPERATION_RESUMED,
            operation_id=operation_id,
            message=f"Operation {operation_id} resume requested.",
            payload={
                "completed_task_ids": list(snapshot.completed_task_ids),
                "resumable_task_ids": list(snapshot.resumable_task_ids),
            },
        )
        for task in snapshot.tasks:
            if task.id in snapshot.completed_task_ids:
                await self.kernel.append_event(
                    RuntimeEventType.TASK_SKIPPED,
                    operation_id=operation_id,
                    task=task,
                    capability=self._capability_for_task(task),
                    message=f"Task {task.id} already completed; not re-running.",
                )

        terminal_operation = await self.kernel.apply_terminal_approval_state(
            operation_id
        )
        if terminal_operation is not None:
            terminal_snapshot = await self.inspect_operation(operation_id)
            if terminal_snapshot is None:
                raise KeyError(operation_id)
            if _monitor_action_context(terminal_snapshot.operation):
                await self._finalize_resumed_monitor_action(terminal_snapshot)
                terminal_snapshot = await self.inspect_operation(operation_id)
                if terminal_snapshot is None:
                    raise KeyError(operation_id)
            if _monitor_delivery_context(terminal_snapshot.operation):
                await self._finalize_resumed_monitor_delivery(terminal_snapshot)
                terminal_snapshot = await self.inspect_operation(operation_id)
                if terminal_snapshot is None:
                    raise KeyError(operation_id)
            return terminal_snapshot

        refreshed = await self.inspect_operation(operation_id)
        if refreshed is None:
            raise KeyError(operation_id)
        snapshot = refreshed

        if self._has_pending_approvals(snapshot):
            await self.kernel.block_operation(operation_id)
            resumed = await self.inspect_operation(operation_id)
            if resumed is None:
                raise KeyError(operation_id)
            return resumed

        recovered_tasks = await self.kernel.recover_expired_task_claims(operation_id)
        if recovered_tasks:
            recovered = await self.inspect_operation(operation_id)
            if recovered is None:
                raise KeyError(operation_id)
            snapshot = recovered

        resumable_tasks = tuple(
            task
            for task in _tasks_in_resume_order(snapshot.tasks)
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
            and not task.metadata.get("manual_recovery_required")
        )
        operation = snapshot.operation
        if resumable_tasks:
            operation = await self.kernel.update_operation(
                operation_id,
                OperationStatus.RUNNING,
                message=f"Operation {operation_id} resumed execution.",
            )
        for task in resumable_tasks:
            try:
                await self.execute_task(task, operation)
            except DbRuntimeGovernanceBlocked:
                resumed = await self.inspect_operation(operation_id)
                if resumed is None:
                    raise KeyError(operation_id)
                return resumed
            except DbRuntimeTaskNotRunnable:
                resumed = await self.inspect_operation(operation_id)
                if resumed is None:
                    raise KeyError(operation_id)
                return resumed
            except Exception as exc:
                if _monitor_action_context(operation) and str(exc).startswith(
                    "monitor_write_"
                ):
                    await self.kernel.block_operation(
                        operation.id,
                        message=f"Monitor write execution blocked: {exc}.",
                    )
                    resumed = await self.inspect_operation(operation_id)
                    if resumed is None:
                        raise KeyError(operation_id)
                    await self._finalize_resumed_monitor_action(resumed)
                    finalized = await self.inspect_operation(operation_id)
                    if finalized is None:
                        raise KeyError(operation_id)
                    return finalized
                await self.kernel.fail_operation_if_active(operation.id, exc)
                raise

        completed = await self.inspect_operation(operation_id)
        if completed is None:
            raise KeyError(operation_id)
        if _has_running_tasks(completed.tasks):
            return completed
        if completed.resumable_task_ids:
            return completed
        if _monitor_action_context(completed.operation):
            await self._finalize_resumed_monitor_action(completed)
            action_finalized = await self.inspect_operation(operation_id)
            if action_finalized is None:
                raise KeyError(operation_id)
            if _monitor_delivery_context(action_finalized.operation):
                await self._finalize_resumed_monitor_delivery(action_finalized)
                delivery_finalized = await self.inspect_operation(operation_id)
                if delivery_finalized is None:
                    raise KeyError(operation_id)
                return delivery_finalized
            return action_finalized
        if _monitor_delivery_context(completed.operation):
            await self._finalize_resumed_monitor_delivery(completed)
            finalized = await self.inspect_operation(operation_id)
            if finalized is None:
                raise KeyError(operation_id)
            return finalized
        if completed.operation.status is OperationStatus.SUCCEEDED:
            return completed

        if _snapshot_has_incomplete_analysis(completed):
            request = _db_request_from_context(completed.operation)
            intent = _db_intent_from_context(completed.operation)
            contract = _db_contract_from_context(completed.operation)
            operation = await self.kernel.update_operation(
                operation_id,
                OperationStatus.RUNNING,
                message=f"Operation {operation_id} resumed analysis materialization.",
            )
            await self._run_multi_step_analysis(
                request,
                intent,
                contract,
                operation,
                base_diagnostics={
                    "runtime_id": self.runtime_id,
                    "resume": {
                        "operation_id": operation_id,
                        "completed_task_ids": list(completed.completed_task_ids),
                    },
                },
                reuse_existing_plan=True,
            )
            resumed_analysis = await self.inspect_operation(operation_id)
            if resumed_analysis is None:
                raise KeyError(operation_id)
            if _monitor_action_context(resumed_analysis.operation):
                await self._finalize_resumed_monitor_action(resumed_analysis)
                finalized = await self.inspect_operation(operation_id)
                if finalized is None:
                    raise KeyError(operation_id)
                if _monitor_delivery_context(finalized.operation):
                    await self._finalize_resumed_monitor_delivery(finalized)
                    delivery_finalized = await self.inspect_operation(operation_id)
                    if delivery_finalized is None:
                        raise KeyError(operation_id)
                    return delivery_finalized
                return finalized
            return resumed_analysis

        if _operation_has_run_context(completed.operation):
            await self._complete_resumed_run_operation(completed)
        elif (
            completed.tasks
            and completed.operation.status is not OperationStatus.SUCCEEDED
        ):
            await self.kernel.complete_operation(
                operation_id,
                message=f"Operation {operation_id} succeeded after resume.",
            )
        resumed = await self.inspect_operation(operation_id)
        if resumed is None:
            raise KeyError(operation_id)
        return resumed

    def cached_schema_evidence(self, *, operation_id: str) -> Evidence | None:
        """Return cached schema profile evidence when the runtime cache is fresh."""
        cached = self._schema_profile_cache
        if not cached:
            return None
        ttl = _schema_cache_ttl(self.config.metadata)
        if ttl is not None:
            if ttl <= 0:
                return None
            if time.monotonic() - float(cached["cached_at"]) > ttl:
                return None
        return Evidence(
            kind=str(cached["kind"]),
            owner=str(cached["owner"]) if cached.get("owner") else None,
            operation_id=operation_id,
            payload=dict(cached["payload"]),
            metadata={
                **dict(cached.get("metadata") or {}),
                "schema_cache": "hit",
            },
        )

    def cached_catalog_source_evidence(
        self,
        *,
        operation_id: str,
        schema: dict[str, Any],
        store_id: str,
    ) -> Evidence | None:
        """Return fresh catalog source-registration evidence for this schema."""
        cached = self._catalog_source_cache
        if cached is None:
            return None
        ttl = _schema_cache_ttl(self.config.metadata)
        if ttl is not None:
            if ttl <= 0:
                return None
            if time.monotonic() - cached.cached_at > ttl:
                return None
        if cached.store_id != store_id:
            return None
        if cached.schema_fingerprint != _source_schema_fingerprint(schema):
            return None
        return replace(
            cached.evidence,
            id=None,
            operation_id=operation_id,
            task_id=None,
            metadata={
                **dict(cached.evidence.metadata),
                "catalog_source_cache": "hit",
                "reused_evidence_id": cached.evidence.id,
                "schema_fingerprint": cached.schema_fingerprint,
            },
        )

    def persisted_schema_evidence(self, *, operation_id: str) -> Evidence | None:
        """Return persisted catalog schema profile evidence when fresh enough."""
        from daita.plugins.catalog.persistence import load_schema_snapshot

        options = _from_db_options(self.config.metadata)
        profile_key = options.get("catalog_profile_key")
        if not profile_key:
            return None
        loaded = load_schema_snapshot(
            str(profile_key),
            catalog_keys=[
                str(item) for item in (options.get("catalog_keys") or []) if item
            ],
            ttl=_schema_cache_ttl(self.config.metadata),
        )
        if loaded is None:
            return None
        payload, is_expired = loaded
        if is_expired:
            return None
        evidence = Evidence(
            kind="schema.asset_profile",
            owner=str(payload.get("database_type") or "catalog"),
            operation_id=operation_id,
            payload=dict(payload),
            metadata={"schema_cache": "persistent_hit"},
        )
        self.remember_schema_evidence(evidence)
        return evidence

    def stale_persisted_schema_evidence(
        self,
        *,
        operation_id: str,
        error: Exception,
    ) -> Evidence | None:
        """Return an expired persisted schema profile after refresh failure."""
        from daita.plugins.catalog.persistence import load_schema_snapshot

        options = _from_db_options(self.config.metadata)
        profile_key = options.get("catalog_profile_key")
        if not profile_key:
            return None
        loaded = load_schema_snapshot(
            str(profile_key),
            catalog_keys=[
                str(item) for item in (options.get("catalog_keys") or []) if item
            ],
            ttl=_schema_cache_ttl(self.config.metadata),
        )
        if loaded is None:
            return None
        payload, is_expired = loaded
        if not is_expired:
            return None
        evidence = Evidence(
            kind="schema.asset_profile",
            owner=str(payload.get("database_type") or "catalog"),
            operation_id=operation_id,
            payload=dict(payload),
            metadata={
                "schema_cache": "persistent_stale_fallback",
                "refresh_error_type": type(error).__name__,
                "refresh_error": str(error),
            },
        )
        self.remember_schema_evidence(evidence)
        return evidence

    def remember_schema_evidence(self, evidence: Evidence) -> None:
        """Store schema profile payload for subsequent operations on this runtime."""
        if evidence.kind != "schema.asset_profile":
            return
        self._schema_profile_cache = {
            "kind": evidence.kind,
            "owner": evidence.owner,
            "payload": dict(evidence.payload),
            "metadata": {"schema_cache": "stored"},
            "cached_at": time.monotonic(),
        }

    def remember_catalog_source_evidence(
        self,
        evidence: Evidence,
        *,
        schema: dict[str, Any],
        store_id: str,
    ) -> None:
        """Remember an accepted catalog source-registration evidence reference."""
        if evidence.kind != "catalog.source_registered" or not evidence.accepted:
            return
        self._catalog_source_cache = _SourcePreparationSnapshot(
            evidence=evidence,
            store_id=store_id,
            schema_fingerprint=_source_schema_fingerprint(schema),
            cached_at=time.monotonic(),
        )

    def classify_request(self, request: DbRequest | str) -> DbIntent:
        """Classify a prompt into a DB intent."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        return self.intent_classifier.classify(db_request)

    def build_contract(
        self,
        request: DbRequest | str,
        intent: DbIntent | None = None,
        *,
        skill_resolution: SkillResolution | None = None,
        include_skills: bool = True,
    ) -> DbOperationContract:
        """Build a structured operation contract from registry capabilities."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        resolved_intent = intent or self.classify_request(db_request)
        resolved_skills = skill_resolution
        if resolved_skills is None and include_skills:
            resolved_skills = self._resolve_skills(db_request)
        return DbContractBuilder(self.registry, self.config).build(
            db_request,
            resolved_intent,
            skill_effects=resolved_skills.effects if resolved_skills else (),
        )

    def _db_request_from_operation(self, operation: Operation) -> DbRequest:
        return _db_request_from_context(operation)

    def _db_intent_from_operation(self, operation: Operation) -> DbIntent:
        return _db_intent_from_context(operation)

    def _db_contract_from_context(self, operation: Operation) -> DbOperationContract:
        return _db_contract_from_context(operation)

    async def run(self, request: DbRequest | str) -> DbOperationResult:
        """Plan and execute a DB operation through typed runtime capabilities."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        if not self._is_setup:
            await self.setup()
        intent = self.classify_request(db_request)
        skill_resolution = self._resolve_skills(db_request)
        contract = self.build_contract(
            db_request,
            intent,
            skill_resolution=skill_resolution,
        )
        operation = await self.kernel.create_operation(
            operation_type=contract.operation_type,
            request={
                "prompt": db_request.prompt,
                "user_id": db_request.user_id,
                "session_id": db_request.session_id,
                "source_scope": list(db_request.source_scope),
                "mode": db_request.mode,
                "requested_capabilities": list(db_request.requested_capabilities),
                "constraints": db_request.constraints,
                "metadata": db_request.metadata,
            },
            required_evidence=frozenset(contract.required_evidence),
            metadata={
                "intent_kind": intent.kind.value,
                "access": contract.access.value,
                "skills": skill_resolution.to_metadata(),
                "resume_context": {
                    "request": _db_request_context(db_request),
                    "intent": _db_intent_context(intent),
                    "contract": _db_contract_context(contract),
                    "skills": skill_resolution.to_metadata(),
                },
            },
            evaluate_governance=False,
        )
        operation_id = operation.id
        await self._record_skill_resolution(operation_id, skill_resolution, contract)

        base_diagnostics = {
            "runtime_id": self.runtime_id,
            "registered_plugins": list(self.registry.plugin_ids),
            "contract": contract.metadata,
            "skills": skill_resolution.to_metadata(),
        }
        if contract.metadata.get("missing_capabilities"):
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer="Required DB capabilities are not registered.",
                    warnings=("db_runtime_missing_capabilities",),
                    diagnostics=base_diagnostics,
                ),
                operation=operation,
            )

        analysis_route = self._should_route_multi_step_analysis(db_request, intent)
        if not analysis_route:
            await self._persist_contract_tasks(operation, contract)
        try:
            governance = await self.kernel.evaluate_operation_governance(operation.id)
        except RuntimeKernelGovernanceBlocked as exc:
            governance = exc.governance or GovernanceResult(False, True, False)
            base_diagnostics = {
                **base_diagnostics,
                "governance": governance.to_dict(),
            }
            if governance.blocked:
                return await self._record_operation_result(
                    DbOperationResult(
                        operation_id=operation_id,
                        request=db_request,
                        intent=intent,
                        contract=contract,
                        status=OperationStatus.BLOCKED,
                        answer="This operation was denied by governance policy.",
                        warnings=("db_runtime_governance_denied",),
                        diagnostics=base_diagnostics,
                    ),
                    operation=exc.operation or operation,
                )
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer="This operation requires approval before execution.",
                    warnings=("db_runtime_approval_required",),
                    diagnostics=base_diagnostics,
                ),
                operation=exc.operation or operation,
            )
        base_diagnostics = {
            **base_diagnostics,
            "governance": governance.to_dict(),
        }

        if analysis_route:
            return await self._run_multi_step_analysis(
                db_request,
                intent,
                contract,
                operation,
                base_diagnostics=base_diagnostics,
            )

        try:
            outcome = await DbOperationExecutor(self).execute(
                db_request, intent, contract, operation
            )
        except DbRuntimeGovernanceBlocked as exc:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer=_governance_blocked_answer(exc.governance),
                    warnings=(_governance_blocked_warning(exc.governance),),
                    diagnostics={
                        **base_diagnostics,
                        "governance": exc.governance.to_dict(),
                    },
                ),
                operation=operation,
            )
        except Exception as exc:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.FAILED,
                    answer=f"DB operation failed: {exc}",
                    warnings=("db_runtime_execution_failed",),
                    diagnostics={
                        **base_diagnostics,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                ),
                operation=operation,
            )

        verification = self.verifier.verify(
            contract, intent, outcome.evidence, outcome.tasks
        )
        if not verification.passed:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.FAILED,
                    answer="DB operation could not be verified against required evidence.",
                    evidence=outcome.evidence,
                    warnings=tuple((*outcome.warnings, *verification.warnings)),
                    diagnostics={
                        **base_diagnostics,
                        "execution": {
                            **outcome.diagnostics,
                            "task_count": len(outcome.tasks),
                            "tasks": [task.to_dict() for task in outcome.tasks],
                        },
                        "verification": verification.to_dict(),
                    },
                ),
                operation=operation,
            )

        verification_evidence = await self._persist_verification_result_evidence(
            operation,
            verification,
            outcome.evidence,
        )
        synthesis_evidence, synthesis_task = await self._execute_answer_synthesis(
            operation=operation,
            intent=intent,
            outcome_evidence=(*outcome.evidence, verification_evidence),
        )
        final_evidence = (*outcome.evidence, verification_evidence, synthesis_evidence)
        final_tasks = (*outcome.tasks, synthesis_task)
        return await self._record_operation_result(
            DbOperationResult(
                operation_id=operation_id,
                request=db_request,
                intent=intent,
                contract=contract,
                status=OperationStatus.SUCCEEDED,
                answer=_answer_from_synthesis_evidence(synthesis_evidence),
                evidence=final_evidence,
                warnings=tuple(
                    (
                        *outcome.warnings,
                        *(
                            synthesis_evidence.payload.get("warnings")
                            if isinstance(
                                synthesis_evidence.payload.get("warnings"), list
                            )
                            else ()
                        ),
                    )
                ),
                diagnostics={
                    **base_diagnostics,
                    "execution": {
                        **outcome.diagnostics,
                        "task_count": len(final_tasks),
                        "tasks": [task.to_dict() for task in final_tasks],
                    },
                    "verification": verification.to_dict(),
                    "synthesis": synthesis_evidence.payload,
                },
            ),
            operation=operation,
        )

    def _should_route_multi_step_analysis(
        self,
        request: DbRequest,
        intent: DbIntent,
    ) -> bool:
        if intent.kind not in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
            return False
        if request.metadata.get("analysis_mode") == "multi_step":
            return True
        prompt = request.prompt.lower()
        explicit = (
            "multi-step",
            "multiple queries",
            "long-running",
            "deep analysis",
            "investigate",
            "explain why",
            "why did",
            "root cause",
            "break down and compare",
            "compare drivers",
            "then drill",
            "step by step analysis",
        )
        return self.db_llm_service.available and any(
            term in prompt for term in explicit
        )

    async def _run_multi_step_analysis(
        self,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        operation: Operation,
        *,
        base_diagnostics: dict[str, Any],
        reuse_existing_plan: bool = False,
    ) -> DbOperationResult:
        executor = DbOperationExecutor(self)
        evidence_store = DbEvidenceStore()
        tasks: list[Task] = []
        warnings: list[str] = []
        started_at = time.monotonic()
        diagnostics: dict[str, Any] = {
            "planner_strategy": "analysis",
            "phase": "multi_step_analysis",
            "evidence_refs": [],
            "evidence_kinds": [],
        }
        try:
            schema_evidence = await executor._inspect_schema_if_available(
                operation, tasks, evidence_store
            )
            schema = schema_evidence.payload if schema_evidence is not None else {}
            planning_context = await executor._build_planning_context(
                request,
                operation,
                tasks,
                evidence_store,
                schema_evidence=schema_evidence,
                catalog_evidence=tuple(
                    item
                    for item in evidence_store.list()
                    if item.kind.startswith("catalog.")
                    or item.kind in {"schema.search_result", "catalog.source"}
                ),
                relationship_evidence=(),
                analysis_metadata={
                    "analysis_id": f"analysis-{operation.id}",
                    "analysis_step_id": "analysis_context",
                    "analysis_phase": "context",
                },
            )
            plan_evidence = (
                await self._latest_accepted_evidence(operation.id, "analysis.plan")
                if reuse_existing_plan
                else None
            )
            validation_evidence = (
                await self._latest_accepted_evidence(
                    operation.id, "analysis.plan.validation", payload={"valid": True}
                )
                if plan_evidence is not None
                else None
            )
            if plan_evidence is None:
                plan_evidence = await self._execute_analysis_plan_task(
                    operation,
                    tasks,
                    evidence_store,
                    planning_context=planning_context,
                )
            if validation_evidence is None:
                validation_evidence = await self._execute_analysis_validation_task(
                    operation,
                    tasks,
                    evidence_store,
                    plan_evidence=plan_evidence,
                )
            if not plan_evidence.accepted or not validation_evidence.accepted:
                warnings.append("analysis_plan_unavailable_or_invalid")
                checkpoint = await self._execute_analysis_checkpoint_task(
                    operation,
                    tasks,
                    evidence_store,
                    analysis_id=str(
                        plan_evidence.payload.get("analysis_id")
                        or plan_evidence.metadata.get("analysis_id")
                        or f"analysis-{operation.id}"
                    ),
                    step_id="analysis_blocked",
                    plan_evidence=plan_evidence,
                    cited_evidence=(plan_evidence, validation_evidence),
                    remaining_step_ids=(),
                    diagnostics={"blocked_reason": "analysis_plan_invalid"},
                )
                all_evidence = (*evidence_store.list(), checkpoint)
                return await self._record_operation_result(
                    DbOperationResult(
                        operation_id=operation.id,
                        request=request,
                        intent=intent,
                        contract=contract,
                        status=OperationStatus.BLOCKED,
                        answer=str(
                            plan_evidence.payload.get("clarification_question")
                            or "The multi-step analysis plan could not be validated."
                        ),
                        evidence=all_evidence,
                        warnings=tuple(warnings),
                        diagnostics={
                            **base_diagnostics,
                            "execution": {
                                **diagnostics,
                                "tasks": [task.to_dict() for task in tasks],
                            },
                            "analysis_plan_validation": validation_evidence.payload,
                        },
                    ),
                    operation=operation,
                )

            plan_state = await self._analysis_plan_state(
                operation.id,
                plan_evidence=plan_evidence,
                validation_evidence=validation_evidence,
            )
            plan = plan_state.plan
            selected_plan_evidence = plan_state.selected_plan_evidence
            analysis_id = plan.analysis_id
            completed_by_step = self._accepted_analysis_step_evidence_map(
                await self.store.list_evidence(operation.id),
                analysis_id=analysis_id,
            )
            while True:
                budget_failure = self._analysis_budget_failure(
                    plan,
                    tuple(await self.store.list_evidence(operation.id)),
                    started_at=started_at,
                )
                if budget_failure is not None:
                    cited_evidence = tuple(
                        evidence
                        for values in completed_by_step.values()
                        for evidence in values
                        if evidence.accepted and evidence.id
                    )
                    remaining_step_ids = tuple(
                        item.id
                        for item in plan.steps
                        if item.id not in completed_by_step
                    )
                    checkpoint = await self._execute_analysis_checkpoint_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id="budget_checkpoint",
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=cited_evidence,
                        remaining_step_ids=remaining_step_ids,
                        diagnostics=budget_failure,
                    )
                    partial_synthesis = await self._execute_analysis_synthesis_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id="budget_partial_synthesis",
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=(*cited_evidence, checkpoint),
                        partial=True,
                        pause_reason="budget_exhausted",
                        remaining_step_ids=remaining_step_ids,
                    )
                    if _analysis_replan_enabled(self.config.metadata):
                        await self._execute_analysis_replan_task(
                            operation,
                            tasks,
                            evidence_store,
                            analysis_id=analysis_id,
                            plan_evidence=selected_plan_evidence,
                            trigger_evidence=(checkpoint,),
                            failed_step_ids=remaining_step_ids,
                            budget_usage=dict(budget_failure.get("budget_usage") or {}),
                            retry_rationale="budget_exhausted",
                        )
                    all_evidence = tuple(await self.store.list_evidence(operation.id))
                    all_tasks = tuple(await self.store.list_tasks(operation.id))
                    return await self._record_operation_result(
                        DbOperationResult(
                            operation_id=operation.id,
                            request=request,
                            intent=intent,
                            contract=contract,
                            status=OperationStatus.BLOCKED,
                            answer=_answer_from_analysis_synthesis_evidence(
                                partial_synthesis
                            ),
                            evidence=all_evidence,
                            warnings=tuple((*warnings, "analysis_budget_exhausted")),
                            diagnostics={
                                **base_diagnostics,
                                "execution": {
                                    **diagnostics,
                                    "evidence_kinds": [
                                        item.kind for item in all_evidence
                                    ],
                                    "evidence_refs": [item.id for item in all_evidence],
                                    "task_count": len(all_tasks),
                                    "tasks": [task.to_dict() for task in all_tasks],
                                },
                                "analysis": partial_synthesis.payload,
                                "budget": budget_failure,
                            },
                        ),
                        operation=operation,
                    )
                ready_steps = self._select_ready_analysis_steps(
                    plan,
                    completed_by_step,
                    serial=not _analysis_parallel_enabled(self.config.metadata),
                )
                if not ready_steps:
                    break
                parallel_query_steps = tuple(
                    step for step in ready_steps if step.kind == "query"
                )
                if (
                    _analysis_parallel_enabled(self.config.metadata)
                    and len(parallel_query_steps) > 1
                    and len(parallel_query_steps) == len(ready_steps)
                ):
                    parallel_results = (
                        await self._execute_parallel_analysis_query_steps(
                            executor,
                            request,
                            intent,
                            contract,
                            operation,
                            schema=schema,
                            schema_evidence=schema_evidence,
                            plan=plan,
                            plan_evidence=selected_plan_evidence,
                            steps=parallel_query_steps,
                        )
                    )
                    for step, produced, step_tasks in parallel_results:
                        tasks.extend(step_tasks)
                        evidence_store.add_many(produced)
                        completed_by_step[step.id] = produced
                        step_budget_failure = self._analysis_step_budget_failure(
                            step,
                            produced,
                        )
                        checkpoint = await self._execute_analysis_checkpoint_task(
                            operation,
                            tasks,
                            evidence_store,
                            analysis_id=analysis_id,
                            step_id=f"{step.id}_checkpoint",
                            plan_evidence=selected_plan_evidence,
                            cited_evidence=produced,
                            remaining_step_ids=tuple(
                                item.id
                                for item in plan.steps
                                if item.id not in completed_by_step
                            ),
                            diagnostics=step_budget_failure or {"parallel_batch": True},
                        )
                        completed_by_step.setdefault(
                            f"{step.id}_checkpoint",
                            (checkpoint,),
                        )
                    continue
                step = ready_steps[0]
                dependencies = tuple(
                    evidence
                    for dependency_id in step.depends_on
                    for evidence in completed_by_step.get(dependency_id, ())
                    if evidence.accepted and evidence.id
                )
                step_meta = analysis_metadata(
                    analysis_id=analysis_id,
                    step_id=step.id,
                    step_kind=step.kind,
                    plan_evidence_id=selected_plan_evidence.id,
                )
                if step.kind == "query":
                    produced = await self._execute_analysis_query_step(
                        executor,
                        request,
                        intent,
                        contract,
                        operation,
                        tasks,
                        evidence_store,
                        schema=schema,
                        schema_evidence=schema_evidence,
                        step_prompt=f"{request.prompt}\n\nAnalysis step {step.id}: {step.purpose}",
                        step_metadata=step_meta,
                        context_dependencies=await self._analysis_context_dependencies(
                            operation.id,
                            step.context_evidence_refs,
                        ),
                    )
                    verification_evidence = (
                        await self._persist_analysis_verification_result_evidence(
                            operation,
                            contract,
                            intent,
                            produced,
                            tuple(tasks),
                            step_metadata=step_meta,
                        )
                    )
                    evidence_store.add(verification_evidence)
                    produced = (*produced, verification_evidence)
                    completed_by_step[step.id] = produced
                    step_budget_failure = self._analysis_step_budget_failure(
                        step,
                        produced,
                    )
                    checkpoint = await self._execute_analysis_checkpoint_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id=f"{step.id}_checkpoint",
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=produced,
                        remaining_step_ids=tuple(
                            item.id
                            for item in plan.steps
                            if item.id not in completed_by_step
                        ),
                        diagnostics=step_budget_failure or {},
                    )
                    completed_by_step.setdefault(f"{step.id}_checkpoint", (checkpoint,))
                    if step_budget_failure is not None:
                        break
                elif step.kind == "checkpoint":
                    checkpoint = await self._execute_analysis_checkpoint_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id=step.id,
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=dependencies,
                        remaining_step_ids=tuple(
                            item.id
                            for item in plan.steps
                            if item.id not in completed_by_step
                        ),
                    )
                    completed_by_step[step.id] = (checkpoint,)
                elif step.kind == "synthesis":
                    synthesis = await self._execute_analysis_synthesis_task(
                        operation,
                        tasks,
                        evidence_store,
                        analysis_id=analysis_id,
                        step_id=step.id,
                        plan_evidence=selected_plan_evidence,
                        cited_evidence=dependencies
                        or tuple(
                            evidence
                            for values in completed_by_step.values()
                            for evidence in values
                            if evidence.accepted and evidence.id
                        ),
                    )
                    completed_by_step[step.id] = (synthesis,)
                elif capability_contract_for_step_kind(step.kind) is not None:
                    produced = await self._execute_analysis_capability_step(
                        operation,
                        tasks,
                        evidence_store,
                        step=step,
                        step_metadata=step_meta,
                        dependency_evidence=dependencies,
                    )
                    completed_by_step[step.id] = produced

            synthesis = await self._latest_final_analysis_synthesis(operation.id)
            if synthesis is None:
                synthesis = await self._execute_analysis_synthesis_task(
                    operation,
                    tasks,
                    evidence_store,
                    analysis_id=analysis_id,
                    step_id="final_synthesis",
                    plan_evidence=selected_plan_evidence,
                    cited_evidence=tuple(
                        evidence
                        for values in completed_by_step.values()
                        for evidence in values
                        if evidence.accepted and evidence.id
                    ),
                )
            all_evidence = tuple(await self.store.list_evidence(operation.id))
            all_tasks = tuple(await self.store.list_tasks(operation.id))
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation.id,
                    request=request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.SUCCEEDED,
                    answer=_answer_from_analysis_synthesis_evidence(synthesis),
                    evidence=all_evidence,
                    warnings=tuple(warnings),
                    diagnostics={
                        **base_diagnostics,
                        "execution": {
                            **diagnostics,
                            "evidence_kinds": [item.kind for item in all_evidence],
                            "evidence_refs": [item.id for item in all_evidence],
                            "task_count": len(all_tasks),
                            "tasks": [task.to_dict() for task in all_tasks],
                        },
                        "analysis": synthesis.payload,
                    },
                ),
                operation=operation,
            )
        except DbRuntimeGovernanceBlocked as exc:
            blocked_evidence = tuple(await self.store.list_evidence(operation.id))
            blocked_checkpoint, blocked_synthesis = (
                await self._checkpoint_blocked_analysis_state(
                    operation,
                    tasks,
                    evidence_store,
                    governance=exc.governance,
                    evidence=blocked_evidence,
                )
            )
            final_blocked_evidence = tuple(await self.store.list_evidence(operation.id))
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation.id,
                    request=request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer=(
                        _answer_from_analysis_synthesis_evidence(blocked_synthesis)
                        if blocked_synthesis is not None
                        else _governance_blocked_answer(exc.governance)
                    ),
                    evidence=final_blocked_evidence,
                    warnings=(_governance_blocked_warning(exc.governance),),
                    diagnostics={
                        **base_diagnostics,
                        "governance": exc.governance.to_dict(),
                        "analysis_checkpoint_id": (
                            blocked_checkpoint.id if blocked_checkpoint else None
                        ),
                        "analysis_partial_synthesis_id": (
                            blocked_synthesis.id if blocked_synthesis else None
                        ),
                    },
                ),
                operation=operation,
            )
        except Exception as exc:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation.id,
                    request=request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.FAILED,
                    answer=f"DB analysis failed: {exc}",
                    evidence=tuple(await self.store.list_evidence(operation.id)),
                    warnings=("db_runtime_analysis_failed",),
                    diagnostics={
                        **base_diagnostics,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                ),
                operation=operation,
            )

    async def _execute_analysis_plan_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence,
    ) -> Evidence:
        analysis_id = f"analysis-{operation.id}"
        capability = self.registry.get_capability(
            "db.analysis.plan", owner="db_runtime"
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "planning_context_evidence_id": planning_context.id,
            },
            metadata=analysis_metadata(
                analysis_id=analysis_id,
                step_id="analysis_plan",
                phase="plan",
            ),
            dependencies=(_dependency_for_evidence(planning_context),),
            sequence=100,
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        plan = next((item for item in evidence if item.kind == "analysis.plan"), None)
        if plan is None:
            raise RuntimeError("analysis.plan evidence was not produced")
        return plan

    async def _execute_analysis_validation_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        plan_evidence: Evidence,
    ) -> Evidence:
        analysis_id = str(
            plan_evidence.payload.get("analysis_id")
            or plan_evidence.metadata.get("analysis_id")
            or f"analysis-{operation.id}"
        )
        capability = self.registry.get_capability(
            "db.analysis.plan.validate", owner="db_runtime"
        )
        task = self._analysis_task(
            operation,
            capability,
            input={"analysis_plan_evidence_id": plan_evidence.id},
            metadata=analysis_metadata(
                analysis_id=analysis_id,
                step_id="analysis_plan_validation",
                plan_evidence_id=plan_evidence.id,
                phase="validation",
            ),
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=plan_evidence.kind,
                    evidence_id=plan_evidence.id,
                    evidence_owner=plan_evidence.owner,
                    producer_task_id=plan_evidence.task_id,
                    evidence_accepted=plan_evidence.accepted,
                    operation_id=plan_evidence.operation_id,
                    payload_fingerprint=plan_evidence.metadata.get(
                        "payload_fingerprint"
                    )
                    or _payload_fingerprint(plan_evidence.payload),
                ),
            ),
            sequence=101,
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        validation = next(
            (item for item in evidence if item.kind == "analysis.plan.validation"),
            None,
        )
        if validation is None:
            raise RuntimeError("analysis.plan.validation evidence was not produced")
        return validation

    async def _execute_analysis_query_step(
        self,
        executor: DbOperationExecutor,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        schema: dict[str, Any],
        schema_evidence: Evidence | None,
        step_prompt: str,
        step_metadata: dict[str, Any],
        context_dependencies: tuple[TaskDependency, ...] = (),
    ) -> tuple[Evidence, ...]:
        from dataclasses import replace as dataclass_replace

        before_ids = {item.id for item in await self.store.list_evidence(operation.id)}
        step_request = dataclass_replace(request, prompt=step_prompt)
        planning_context = await executor._build_planning_context(
            step_request,
            operation,
            tasks,
            evidence_store,
            schema_evidence=schema_evidence,
            catalog_evidence=(),
            relationship_evidence=(),
            analysis_metadata=step_metadata,
            extra_dependencies=context_dependencies,
        )
        plan_evidence, strategy_warnings, _ = await executor._plan_query(
            step_request,
            intent,
            operation,
            schema,
            None,
            planning_context,
            tasks,
            evidence_store,
            analysis_metadata=step_metadata,
            extra_dependencies=context_dependencies,
        )
        validation = await executor._validate_query_plan(
            operation,
            tasks,
            evidence_store,
            plan_evidence=plan_evidence,
            planning_context=planning_context,
            analysis_metadata=step_metadata,
            extra_dependencies=context_dependencies,
        )
        sql = validation.payload.get("accepted_sql")
        if sql:
            sql_validation = await executor._execute_sql_validation(
                contract,
                operation,
                tasks,
                evidence_store,
                sql,
                plan_validation=validation,
                analysis_metadata=step_metadata,
                extra_dependencies=context_dependencies,
            )
            await executor._execute_validated_read(
                contract,
                operation,
                tasks,
                evidence_store,
                sql_validation,
                analysis_metadata=step_metadata,
                extra_dependencies=context_dependencies,
            )
        produced = tuple(
            item
            for item in await self.store.list_evidence(operation.id)
            if item.id not in before_ids
            and item.metadata.get("analysis_step_id")
            == step_metadata.get("analysis_step_id")
        )
        if strategy_warnings:
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation.id,
                message="Analysis query step produced planner warnings.",
                payload={
                    "analysis_step_id": step_metadata.get("analysis_step_id"),
                    "warnings": list(strategy_warnings),
                },
            )
        return produced

    async def _execute_parallel_analysis_query_steps(
        self,
        executor: DbOperationExecutor,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        operation: Operation,
        *,
        schema: dict[str, Any],
        schema_evidence: Evidence | None,
        plan: DbAnalysisPlan,
        plan_evidence: Evidence,
        steps: tuple[Any, ...],
    ) -> tuple[tuple[Any, tuple[Evidence, ...], tuple[Task, ...]], ...]:
        semaphore = asyncio.Semaphore(_analysis_max_concurrency(self.config.metadata))

        async def run_step(
            step: Any,
        ) -> tuple[Any, tuple[Evidence, ...], tuple[Task, ...]]:
            async with semaphore:
                step_tasks: list[Task] = []
                step_evidence_store = DbEvidenceStore()
                step_meta = analysis_metadata(
                    analysis_id=plan.analysis_id,
                    step_id=step.id,
                    step_kind=step.kind,
                    plan_evidence_id=plan_evidence.id,
                )
                produced = await self._execute_analysis_query_step(
                    executor,
                    request,
                    intent,
                    contract,
                    operation,
                    step_tasks,
                    step_evidence_store,
                    schema=schema,
                    schema_evidence=schema_evidence,
                    step_prompt=(
                        f"{request.prompt}\n\nAnalysis step {step.id}: {step.purpose}"
                    ),
                    step_metadata=step_meta,
                    context_dependencies=await self._analysis_context_dependencies(
                        operation.id,
                        step.context_evidence_refs,
                    ),
                )
                verification_evidence = (
                    await self._persist_analysis_verification_result_evidence(
                        operation,
                        contract,
                        intent,
                        produced,
                        tuple(step_tasks),
                        step_metadata=step_meta,
                    )
                )
                produced = (*produced, verification_evidence)
                return step, produced, tuple(step_tasks)

        results = await asyncio.gather(*(run_step(step) for step in steps))
        return tuple(sorted(results, key=lambda item: item[0].id))

    async def _persist_analysis_verification_result_evidence(
        self,
        operation: Operation,
        contract: DbOperationContract,
        intent: DbIntent,
        evidence: tuple[Evidence, ...],
        tasks: tuple[Task, ...],
        *,
        step_metadata: dict[str, Any],
    ) -> Evidence:
        verification = self.verifier.verify(contract, intent, evidence, tasks)
        evidence_details = [
            evidence_ref(item) for item in evidence if item.accepted and item.id
        ]
        payload = {
            "passed": bool(verification.passed),
            "evidence_refs": [item["id"] for item in evidence_details],
            "evidence_details": evidence_details,
            "warnings": list(verification.warnings),
            "missing_evidence": list(verification.missing_evidence),
            "diagnostics": verification.diagnostics,
            "input_fingerprint": stable_fingerprint(evidence_details),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        verification_evidence = Evidence(
            id=f"evidence-{uuid4()}",
            kind="verification.result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=verification.passed,
            payload=payload,
            metadata={
                **step_metadata,
                "payload_fingerprint": stable_fingerprint(payload),
                "input_fingerprint": payload["input_fingerprint"],
            },
        )
        await self.store.save_evidence(verification_evidence)
        return verification_evidence

    async def _execute_analysis_checkpoint_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        analysis_id: str,
        step_id: str,
        plan_evidence: Evidence,
        cited_evidence: tuple[Evidence, ...],
        remaining_step_ids: tuple[str, ...],
        diagnostics: dict[str, Any] | None = None,
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.analysis.checkpoint", owner="db_runtime"
        )
        dependencies = tuple(
            TaskDependency(
                kind="evidence",
                evidence_kind=item.kind,
                evidence_id=item.id,
                evidence_owner=item.owner,
                producer_task_id=item.task_id,
                evidence_accepted=item.accepted,
                operation_id=item.operation_id,
                payload_fingerprint=item.metadata.get("payload_fingerprint")
                or _payload_fingerprint(item.payload),
            )
            for item in cited_evidence
            if item.id
        )
        metadata = analysis_metadata(
            analysis_id=analysis_id,
            step_id=step_id,
            step_kind="checkpoint",
            plan_evidence_id=plan_evidence.id,
        )
        progress = self._analysis_progress_payload(
            await self.inspect_operation(operation.id),
            plan_evidence=plan_evidence,
        )
        checkpoint_diagnostics = {
            "checkpoint_reason": _checkpoint_reason(diagnostics or {}),
            "operation_status": operation.status.value,
            "progress": progress,
            **dict(diagnostics or {}),
        }
        task = self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "analysis_step_id": step_id,
                "remaining_step_ids": list(remaining_step_ids),
                "diagnostics": checkpoint_diagnostics,
            },
            metadata=metadata,
            dependencies=dependencies,
            sequence=8000 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        checkpoint = next(
            (item for item in evidence if item.kind == "analysis.checkpoint"),
            None,
        )
        if checkpoint is None:
            raise RuntimeError("analysis.checkpoint evidence was not produced")
        return checkpoint

    async def _execute_analysis_synthesis_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        analysis_id: str,
        step_id: str,
        plan_evidence: Evidence,
        cited_evidence: tuple[Evidence, ...],
        partial: bool = False,
        pause_reason: str | None = None,
        remaining_step_ids: tuple[str, ...] = (),
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.analysis.summarize", owner="db_runtime"
        )
        dependencies = tuple(
            _dependency_for_evidence(item)
            for item in cited_evidence
            if item.id and item.accepted and item.operation_id == operation.id
        )
        metadata = analysis_metadata(
            analysis_id=analysis_id,
            step_id=step_id,
            step_kind="synthesis",
            plan_evidence_id=plan_evidence.id,
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "analysis_step_id": step_id,
                "partial": partial,
                "pause_reason": pause_reason,
                "remaining_step_ids": list(remaining_step_ids),
            },
            metadata=metadata,
            dependencies=dependencies,
            sequence=9000 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        synthesis = next(
            (item for item in evidence if item.kind == "analysis.synthesis"),
            None,
        )
        if synthesis is None:
            raise RuntimeError("analysis.synthesis evidence was not produced")
        return synthesis

    async def _execute_analysis_replan_task(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        analysis_id: str,
        plan_evidence: Evidence,
        trigger_evidence: tuple[Evidence, ...],
        failed_step_ids: tuple[str, ...],
        budget_usage: dict[str, Any],
        retry_rationale: str,
        replacement_steps: tuple[dict[str, Any], ...] = (),
    ) -> Evidence:
        capability = self.registry.get_capability(
            "db.analysis.replan", owner="db_runtime"
        )
        dependencies = (
            _dependency_for_evidence(plan_evidence),
            *tuple(
                _dependency_for_evidence(item)
                for item in trigger_evidence
                if item.id and item.accepted and item.operation_id == operation.id
            ),
        )
        metadata = analysis_metadata(
            analysis_id=analysis_id,
            step_id="analysis_replan",
            phase="replan",
            plan_evidence_id=plan_evidence.id,
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                "analysis_id": analysis_id,
                "parent_plan_evidence_id": plan_evidence.id,
                "trigger_evidence_ids": [
                    item.id for item in trigger_evidence if item.id
                ],
                "failed_step_ids": list(failed_step_ids),
                "replacement_steps": [dict(item) for item in replacement_steps],
                "budget_usage": dict(budget_usage),
                "retry_rationale": retry_rationale,
            },
            metadata=metadata,
            dependencies=dependencies,
            sequence=8500 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        revision = next(
            (item for item in evidence if item.kind == "analysis.plan.revision"),
            None,
        )
        if revision is None:
            raise RuntimeError("analysis.plan.revision evidence was not produced")
        return revision

    async def _checkpoint_blocked_analysis_state(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        governance: GovernanceResult,
        evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence | None, Evidence | None]:
        plan_evidence = next(
            (
                item
                for item in reversed(evidence)
                if item.kind == "analysis.plan" and item.accepted
            ),
            None,
        )
        validation_evidence = next(
            (
                item
                for item in reversed(evidence)
                if item.kind == "analysis.plan.validation"
                and item.accepted
                and item.payload.get("valid") is True
            ),
            None,
        )
        if plan_evidence is None or validation_evidence is None:
            return None, None
        try:
            plan_state = await self._analysis_plan_state(
                operation.id,
                plan_evidence=plan_evidence,
                validation_evidence=validation_evidence,
            )
        except Exception:
            return None, None
        plan = plan_state.plan
        analysis_id = plan.analysis_id
        completed_by_step = self._accepted_analysis_step_evidence_map(
            evidence,
            analysis_id=analysis_id,
        )
        cited_evidence = tuple(
            item
            for values in completed_by_step.values()
            for item in values
            if item.accepted and item.id
        )
        remaining_step_ids = tuple(
            step.id for step in plan.steps if step.id not in completed_by_step
        )
        checkpoint = await self._execute_analysis_checkpoint_task(
            operation,
            tasks,
            evidence_store,
            analysis_id=analysis_id,
            step_id="analysis_blocked_checkpoint",
            plan_evidence=plan_state.selected_plan_evidence,
            cited_evidence=cited_evidence,
            remaining_step_ids=remaining_step_ids,
            diagnostics={
                "blocked_reason": "governance",
                "pending_approval": governance.pending_approval,
                "governance": governance.to_dict(),
            },
        )
        synthesis = None
        if cited_evidence:
            synthesis = await self._execute_analysis_synthesis_task(
                operation,
                tasks,
                evidence_store,
                analysis_id=analysis_id,
                step_id="analysis_blocked_partial_synthesis",
                plan_evidence=plan_state.selected_plan_evidence,
                cited_evidence=(*cited_evidence, checkpoint),
                partial=True,
                pause_reason=(
                    "approval_required"
                    if governance.pending_approval
                    else "governance_blocked"
                ),
                remaining_step_ids=remaining_step_ids,
            )
        return checkpoint, synthesis

    async def _execute_analysis_capability_step(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        step: Any,
        step_metadata: dict[str, Any],
        dependency_evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence, ...]:
        if not step.capability_id or not step.capability_owner:
            raise RuntimeError(
                f"analysis capability step {step.id} is missing capability"
            )
        capability = self.registry.get_capability(
            step.capability_id,
            owner=step.capability_owner,
        )
        contract = capability_contract_for_step_kind(step.kind)
        if contract is None or capability.id not in contract["capabilities"]:
            raise RuntimeError(
                f"analysis step {step.id} uses unsupported capability {capability.id}"
            )
        expected = set(step.expected_evidence)
        if expected and not expected <= set(capability.output_evidence):
            raise RuntimeError(
                f"analysis step {step.id} expects evidence not produced by {capability.id}"
            )
        if capability.side_effecting:
            raise RuntimeError(
                f"analysis step {step.id} cannot execute side-effecting capability"
            )
        context_dependencies = await self._analysis_context_dependencies(
            operation.id,
            step.context_evidence_refs,
        )
        dependencies = (
            *tuple(
                _dependency_for_evidence(item)
                for item in dependency_evidence
                if item.id and item.accepted and item.operation_id == operation.id
            ),
            *context_dependencies,
        )
        task = self._analysis_task(
            operation,
            capability,
            input={
                **dict(step.input),
                "dependency_evidence_refs": [
                    evidence_ref(item)
                    for item in dependency_evidence
                    if item.id and item.accepted
                ],
            },
            metadata=step_metadata,
            dependencies=dependencies,
            sequence=4000 + len(tasks),
        )
        tasks.append(task)
        evidence = await self.execute_task(task, operation)
        evidence_store.add_many(evidence)
        return tuple(
            item
            for item in evidence
            if item.accepted and (not expected or item.kind in expected)
        )

    async def _analysis_context_dependencies(
        self,
        operation_id: str,
        refs: tuple[dict[str, Any], ...],
    ) -> tuple[TaskDependency, ...]:
        dependencies: list[TaskDependency] = []
        evidence = await self.store.list_evidence(operation_id)
        by_id = {item.id: item for item in evidence if item.id}
        for ref in refs:
            evidence_id = ref.get("id")
            if not evidence_id:
                continue
            item = by_id.get(str(evidence_id))
            if item is None or not item.accepted or item.operation_id != operation_id:
                raise RuntimeError(
                    f"analysis context evidence not accepted: {evidence_id}"
                )
            fingerprint = ref.get("payload_fingerprint")
            actual = item.metadata.get("payload_fingerprint") or _payload_fingerprint(
                item.payload
            )
            if fingerprint is not None and str(fingerprint) != actual:
                raise RuntimeError(
                    f"analysis context evidence fingerprint mismatch: {evidence_id}"
                )
            dependencies.append(_dependency_for_evidence(item))
        return tuple(dependencies)

    def _analysis_task(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input: dict[str, Any],
        metadata: dict[str, Any],
        dependencies: tuple[TaskDependency, ...],
        sequence: int,
    ) -> Task:
        task_input = {**input}
        input_hash = _stable_hash(task_input)
        return Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**task_input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            dependencies=dependencies,
            metadata=with_analysis_evidence_trace(
                {
                    "owner": capability.owner,
                    "reason": "analysis",
                    "sequence": sequence,
                    "input_hash": input_hash,
                    "idempotency_key": _stable_hash(
                        {
                            "operation_id": operation.id,
                            "capability_id": capability.id,
                            "input": task_input,
                            **metadata,
                        }
                    ),
                    "idempotent": capability.idempotent,
                    "replay_safe": capability.replay_safe,
                    "side_effecting": capability.side_effecting,
                    **metadata,
                }
            ),
        )

    def _accepted_analysis_step_evidence_map(
        self,
        evidence: tuple[Evidence, ...],
        *,
        analysis_id: str,
    ) -> dict[str, tuple[Evidence, ...]]:
        grouped: dict[str, list[Evidence]] = {}
        for item in evidence:
            if not item.accepted:
                continue
            if item.metadata.get("analysis_id") != analysis_id:
                continue
            step_id = item.metadata.get("analysis_step_id")
            if not isinstance(step_id, str) or not step_id:
                continue
            if item.kind in {
                "analysis.plan",
                "analysis.plan.validation",
                "planning.context",
            }:
                continue
            grouped.setdefault(step_id, []).append(item)
        return {key: tuple(value) for key, value in grouped.items()}

    def _analysis_budget_failure(
        self,
        plan: DbAnalysisPlan,
        evidence: tuple[Evidence, ...],
        *,
        started_at: float,
    ) -> dict[str, Any] | None:
        usage = _analysis_budget_usage(evidence, started_at=started_at)
        failures = []
        budgets = plan.budgets
        if usage["total_rows"] > budgets.max_total_rows:
            failures.append("budget_max_total_rows_exceeded")
        if usage["llm_calls"] > budgets.max_llm_calls:
            failures.append("budget_max_llm_calls_exceeded")
        if usage["context_chars"] > budgets.max_context_chars:
            failures.append("budget_max_context_chars_exceeded")
        if usage["duration_seconds"] > budgets.max_duration_seconds:
            failures.append("budget_max_duration_seconds_exceeded")
        if not failures:
            return None
        return {
            "budget_exceeded": True,
            "failures": failures,
            "budget_usage": usage,
            "budgets": budgets.to_dict(),
        }

    @staticmethod
    def _analysis_step_budget_failure(
        step: Any,
        evidence: tuple[Evidence, ...],
    ) -> dict[str, Any] | None:
        rows = sum(
            _analysis_query_result_rows(item)
            for item in evidence
            if item.kind == "query.result"
        )
        if rows <= step.budgets.max_rows:
            return None
        return {
            "budget_exceeded": True,
            "failures": ["step_max_rows_exceeded"],
            "budget_usage": {"step_rows": rows},
            "budgets": step.budgets.to_dict(),
        }

    @staticmethod
    def _analysis_steps_in_order(plan: DbAnalysisPlan) -> tuple[Any, ...]:
        remaining = {step.id: step for step in plan.steps}
        ordered = []
        while remaining:
            ready = [
                step
                for step in remaining.values()
                if all(dependency not in remaining for dependency in step.depends_on)
            ]
            if not ready:
                return tuple((*ordered, *remaining.values()))
            for step in ready:
                ordered.append(step)
                remaining.pop(step.id, None)
        return tuple(ordered)

    def _select_ready_analysis_steps(
        self,
        plan: DbAnalysisPlan,
        completed_by_step: dict[str, tuple[Evidence, ...]],
        *,
        serial: bool = True,
    ) -> tuple[Any, ...]:
        completed_step_ids = set(completed_by_step)
        ready = tuple(
            step
            for step in self._analysis_steps_in_order(plan)
            if step.id not in completed_step_ids
            and self._analysis_step_dependencies_satisfied(step, completed_by_step)
        )
        if serial:
            return ready[:1]
        return ready

    @staticmethod
    def _analysis_step_dependencies_satisfied(
        step: Any,
        completed_by_step: dict[str, tuple[Evidence, ...]],
    ) -> bool:
        for dependency_id in step.depends_on:
            dependency_evidence = tuple(
                evidence
                for evidence in completed_by_step.get(dependency_id, ())
                if evidence.accepted
                and evidence.id
                and evidence.metadata.get("analysis_step_id") == dependency_id
            )
            if not dependency_evidence:
                return False
        return True

    async def _plan_kernel_task(self, task: Task) -> Task:
        """Persist a DB-planned task through the shared kernel planner."""
        return await self.kernel.plan_task(
            task_id=task.id,
            operation_id=task.operation_id,
            capability_id=task.capability_id,
            owner=str(task.metadata["owner"]) if task.metadata.get("owner") else None,
            input=task.input,
            metadata=task.metadata,
            dependencies=task.dependencies,
        )

    async def _persist_contract_tasks(
        self,
        operation: Operation,
        contract: DbOperationContract,
    ) -> None:
        existing = {
            (task.capability_id, task.executor_id, task.metadata.get("owner"))
            for task in await self.store.list_tasks(operation.id)
        }
        planned_by_capability: dict[tuple[str, str], Task] = {
            (task.capability_id, str(task.metadata.get("owner") or "")): task
            for task in await self.store.list_tasks(operation.id)
        }
        for sequence, selected in enumerate(
            contract.metadata.get("selected_capabilities", ()), start=1
        ):
            capability = self.registry.get_capability(
                str(selected["id"]),
                owner=str(selected["owner"]),
            )
            key = (capability.id, capability.executor, capability.owner)
            if key in existing:
                continue
            task = Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=_planned_task_input(operation, capability),
                required_evidence=capability.output_evidence,
                metadata={
                    "owner": capability.owner,
                    "reason": str(selected.get("reason") or "contract"),
                    "sequence": sequence,
                },
            )
            validation_task = planned_by_capability.get(
                ("db.sql.validate", capability.owner)
            )
            task = replace(
                task,
                input={
                    **task.input,
                    "input_hash": _stable_hash(task.input),
                },
                dependencies=_task_dependencies_for_capability(
                    operation,
                    capability,
                    validation_task=validation_task,
                ),
                metadata={
                    **task.metadata,
                    "input_hash": _stable_hash(task.input),
                    "idempotency_key": _stable_hash(
                        {
                            "operation_id": operation.id,
                            "task_id": task.id,
                            "capability_id": task.capability_id,
                            "input": task.input,
                        }
                    ),
                    "idempotent": capability.idempotent,
                    "replay_safe": capability.replay_safe,
                    "side_effecting": capability.side_effecting,
                },
            )
            await self._plan_kernel_task(task)
            existing.add(key)
            planned_by_capability[(capability.id, capability.owner)] = task

    def _resolve_skills(self, request: DbRequest) -> SkillResolution:
        return SkillResolver(self.registry).resolve(
            runtime_kind=self.runtime_kind,
            prompt=request.prompt,
            request=request,
            explicit_skills=_skill_names_from_request(request),
            runtime_context={
                "runtime_id": self.runtime_id,
                "available_capabilities": tuple(
                    capability.id for capability in self.registry.capabilities
                ),
                "config": self.config.metadata,
                "request_metadata": request.metadata,
                "mode": request.mode,
            },
        )

    async def _record_skill_resolution(
        self,
        operation_id: str,
        resolution: SkillResolution,
        contract: DbOperationContract,
    ) -> None:
        for activation in resolution.selected:
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation_id,
                message=f"Skill {activation.skill_name} selected.",
                plugin_id=activation.skill_id,
                payload={
                    "diagnostic": "skill.selected",
                    "activation": activation.to_dict(),
                },
            )
            if activation.context_loaded:
                await self.kernel.append_event(
                    RuntimeEventType.DIAGNOSTIC,
                    operation_id=operation_id,
                    message=f"Skill {activation.skill_name} context loaded.",
                    plugin_id=activation.skill_id,
                    payload={
                        "diagnostic": "skill.context_loaded",
                        "activation": activation.to_dict(),
                    },
                )
        for activation in resolution.skipped:
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation_id,
                message=f"Skill {activation.skill_name} activation skipped.",
                plugin_id=activation.skill_id,
                payload={
                    "diagnostic": "skill.activation_skipped",
                    "activation": activation.to_dict(),
                },
            )
        if _effects_modify_contract(resolution.effects):
            await self.kernel.append_event(
                RuntimeEventType.DIAGNOSTIC,
                operation_id=operation_id,
                message="Skills modified DB operation contract.",
                payload={
                    "diagnostic": "skill.contract_modified",
                    "selected_skill_ids": list(resolution.selected_ids()),
                    "required_capabilities": list(contract.required_capabilities),
                    "required_evidence": list(contract.required_evidence),
                    "policy_ids": list(contract.policy_ids),
                    "skill_contract_metadata": contract.metadata.get(
                        "skill_contract_metadata", {}
                    ),
                    "skill_verifier_metadata": contract.metadata.get(
                        "skill_verifier_metadata", {}
                    ),
                    "skill_synthesis_metadata": contract.metadata.get(
                        "skill_synthesis_metadata", {}
                    ),
                },
            )

    async def _planned_task_for_capability(
        self,
        operation_id: str,
        capability: Capability,
        *,
        metadata_match: dict[str, Any] | None = None,
    ) -> Task | None:
        for task in await self.store.list_tasks(operation_id):
            if task.status is not TaskStatus.PENDING:
                continue
            if task.capability_id != capability.id:
                continue
            if task.executor_id != capability.executor:
                continue
            if task.metadata.get("owner") != capability.owner:
                continue
            if metadata_match and any(
                task.metadata.get(key) != value for key, value in metadata_match.items()
            ):
                continue
            return task
        return None

    async def _persist_verification_result_evidence(
        self,
        operation: Operation,
        verification: Any,
        evidence: tuple[Evidence, ...],
    ) -> Evidence:
        existing = await self._latest_accepted_evidence(
            operation.id,
            "verification.result",
            payload={"passed": True},
        )
        if existing is not None:
            return existing
        accepted = tuple(item for item in evidence if item.accepted and item.id)
        evidence_details = [
            {
                "id": item.id,
                "kind": item.kind,
                "owner": item.owner,
                "task_id": item.task_id,
                "payload_fingerprint": item.metadata.get("payload_fingerprint")
                or _payload_fingerprint(item.payload),
            }
            for item in accepted
        ]
        payload = {
            "passed": bool(verification.passed),
            "evidence_refs": [item["id"] for item in evidence_details],
            "evidence_details": evidence_details,
            "warnings": list(verification.warnings),
            "missing_evidence": list(verification.missing_evidence),
            "diagnostics": verification.diagnostics,
            "input_fingerprint": _stable_hash(
                {
                    "operation_id": operation.id,
                    "evidence": evidence_details,
                    "warnings": list(verification.warnings),
                    "missing_evidence": list(verification.missing_evidence),
                }
            ),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        verification_evidence = Evidence(
            id=f"evidence-{uuid4()}",
            kind="verification.result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload=payload,
            metadata={
                "payload_fingerprint": _payload_fingerprint(payload),
                "input_fingerprint": payload["input_fingerprint"],
            },
        )
        await self.store.save_evidence(verification_evidence)
        return verification_evidence

    async def _execute_answer_synthesis(
        self,
        *,
        operation: Operation,
        intent: DbIntent,
        outcome_evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence, Task]:
        existing = await self._latest_accepted_evidence(
            operation.id,
            "answer.synthesis",
        )
        if existing is not None:
            task = next(
                (
                    item
                    for item in await self.store.list_tasks(operation.id)
                    if item.id == existing.task_id
                ),
                Task(
                    id=str(existing.task_id or f"db-task-{uuid4()}"),
                    operation_id=operation.id,
                    capability_id="db.answer.synthesize",
                    executor_id="db.answer.synthesize.runtime",
                    required_evidence=frozenset({"answer.synthesis"}),
                    metadata={"owner": "db_runtime", "reason": "answer_synthesis"},
                ),
            )
            return existing, task

        capability = self.registry.get_capability(
            "db.answer.synthesize", owner="db_runtime"
        )
        dependencies = _synthesis_dependencies(operation, intent, outcome_evidence)
        task_input = {
            "evidence_refs": [
                {
                    "id": dependency.evidence_id,
                    "kind": dependency.evidence_kind,
                    "payload_fingerprint": dependency.payload_fingerprint,
                }
                for dependency in dependencies
            ],
            "row_budget": _synthesis_context_option(
                self.config.metadata, "synthesis_row_budget", 25
            ),
            "char_budget": _synthesis_context_option(
                self.config.metadata, "synthesis_context_char_budget", 16000
            ),
        }
        input_hash = _stable_hash(task_input)
        task = Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**task_input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            dependencies=dependencies,
            metadata={
                "owner": capability.owner,
                "reason": "answer_synthesis",
                "sequence": 10_000,
                "input_hash": input_hash,
                "idempotency_key": _stable_hash(
                    {
                        "operation_id": operation.id,
                        "capability_id": capability.id,
                        "evidence_refs": task_input["evidence_refs"],
                    }
                ),
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        evidence = await self.execute_task(
            task,
            operation,
            context={"capability_owner": capability.owner},
        )
        synthesis = next(
            (
                item
                for item in evidence
                if item.kind == "answer.synthesis" and item.accepted
            ),
            None,
        )
        if synthesis is None:
            raise RuntimeError("answer.synthesis evidence was not produced")
        stored_task = await self.store.load_task(task.id)
        return synthesis, stored_task or task

    def _validation_capability_for_sql_execute(
        self,
        capability: Capability,
    ) -> Capability | None:
        if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
            return None
        try:
            return self.registry.get_capability(
                "db.sql.validate", owner=capability.owner
            )
        except KeyError:
            return None

    def _direct_capability_tasks(
        self,
        operation: Operation,
        capability: Capability,
        input: dict[str, Any],
        *,
        validation_capability: Capability | None,
    ) -> tuple[Task, ...]:
        if (
            capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}
            or validation_capability is None
        ):
            task = self._task_for_capability(
                operation,
                capability,
                input=input,
                reason="direct",
                sequence=1,
            )
            return (task,)
        validation_task = self._task_for_capability(
            operation,
            validation_capability,
            input={
                "sql": str(input.get("sql") or ""),
                "operation": (
                    "query"
                    if capability.id == "db.sql.execute_read"
                    else operation.operation_type
                ),
            },
            reason="direct_validation",
            sequence=1,
        )
        execute_input = (
            {
                "sql_ref": "sql.validation",
                "params": list(input.get("params") or []),
            }
            if capability.id == "db.sql.execute_read"
            else {
                "sql_ref": "sql.validation",
                "params": list(input.get("params") or []),
            }
        )
        execute_task = self._task_for_capability(
            operation,
            capability,
            input=execute_input,
            reason="direct",
            sequence=2,
            validation_task=validation_task,
        )
        return (validation_task, execute_task)

    def plan_validated_read_tasks(
        self,
        operation: Operation,
        *,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
        owner: str | None = None,
        reason: str = "validated_read",
        sequence: int = 1,
        focus: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Task, Task]:
        """Plan a DB SQL validation/read task pair under an existing operation."""

        read_capability = self.registry.get_capability(
            "db.sql.execute_read",
            owner=owner,
        )
        validation_capability = self._validation_capability_for_sql_execute(
            read_capability
        )
        if validation_capability is None:
            raise KeyError("db.sql.validate")
        validation_task = self._task_for_capability(
            operation,
            validation_capability,
            input={"sql": sql, "operation": "query"},
            reason=f"{reason}_validation",
            sequence=sequence,
            metadata=metadata,
        )
        execute_input: dict[str, Any] = {
            "sql_ref": "sql.validation",
            "params": list(params),
        }
        if focus is not None:
            execute_input["focus"] = focus
        read_task = self._task_for_capability(
            operation,
            read_capability,
            input=execute_input,
            reason=reason,
            sequence=sequence + 1,
            validation_task=validation_task,
            metadata=metadata,
        )
        return validation_task, read_task

    def _task_for_capability(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input: dict[str, Any],
        reason: str,
        sequence: int,
        validation_task: Task | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        input_hash = _stable_hash(input)
        task = Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            metadata={
                **dict(metadata or {}),
                "owner": capability.owner,
                "reason": reason,
                "sequence": sequence,
                "input_hash": input_hash,
                "idempotency_key": _stable_hash(
                    {
                        "operation_id": operation.id,
                        "capability_id": capability.id,
                        "executor_id": capability.executor,
                        "input": input,
                    }
                ),
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        return replace(
            task,
            dependencies=_task_dependencies_for_capability(
                operation,
                capability,
                validation_task=validation_task,
            ),
        )

    async def _task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        unsatisfied: list[dict[str, Any]] = []
        for dependency in task.dependencies:
            if dependency.kind.value == "evidence":
                if not await self._evidence_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
            elif dependency.kind.value == "approval":
                if not await self._approval_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
        return {
            "ready": not unsatisfied,
            "unsatisfied_dependencies": unsatisfied,
            "dependency_count": len(task.dependencies),
        }

    async def _evidence_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for evidence in await self.store.list_evidence(operation_id):
            if evidence.kind != dependency.evidence_kind:
                continue
            if (
                dependency.evidence_id is not None
                and evidence.id != dependency.evidence_id
            ):
                continue
            if (
                dependency.evidence_owner is not None
                and evidence.owner != dependency.evidence_owner
            ):
                continue
            if (
                dependency.producer_task_id is not None
                and evidence.task_id != dependency.producer_task_id
            ):
                continue
            if evidence.accepted is not dependency.evidence_accepted:
                continue
            if (
                dependency.input_hash is not None
                and evidence.metadata.get("task_input_hash") != dependency.input_hash
            ):
                continue
            if _payload_contains(evidence.payload, dependency.evidence_payload):
                if (
                    dependency.payload_fingerprint is not None
                    and dependency.payload_fingerprint
                    != _payload_fingerprint(evidence.payload)
                ):
                    continue
                return True
        return False

    async def _approval_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for approval in await self.store.list_approval_requests(operation_id):
            if (
                dependency.approval_id is not None
                and approval.approval_id != dependency.approval_id
            ):
                continue
            if (
                dependency.approval_policy_id is not None
                and approval.requested_by_policy_id != dependency.approval_policy_id
            ):
                continue
            if (
                dependency.approval_name is not None
                and approval.proposed_action.get("approval") != dependency.approval_name
            ):
                continue
            if (
                dependency.approval_version is not None
                and approval.metadata.get("version") != dependency.approval_version
            ):
                continue
            if approval.status is dependency.approval_status:
                return True
        return False

    async def _executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        if task.capability_id not in {
            "db.sql.execute_read",
            "db.sql.execute_write",
        }:
            return task.input
        validation_dependency = next(
            (
                dependency
                for dependency in task.dependencies
                if dependency.kind.value == "evidence"
                and dependency.evidence_kind == "sql.validation"
            ),
            None,
        )
        validation = (
            await self._accepted_evidence_for_dependency(
                operation.id,
                validation_dependency,
            )
            if validation_dependency is not None
            else await self._latest_accepted_evidence(
                operation.id,
                "sql.validation",
                payload={"valid": True},
            )
        )
        if validation is None:
            return task.input
        if task.metadata.get("monitor_action_role") == "write_execution":
            proposal_fingerprint = str(task.metadata.get("proposal_fingerprint") or "")
            proposal_evidence_id = str(task.metadata.get("proposal_evidence_id") or "")
            proposal_matches = [
                item
                for item in await self.store.list_evidence(operation.id)
                if item.kind == "monitor.write_proposal"
                and item.id == proposal_evidence_id
            ]
            proposal = proposal_matches[-1] if proposal_matches else None
            expected_validation_id = str(
                task.metadata.get("validation_evidence_id") or ""
            )
            expected_validation_fingerprint = str(
                task.metadata.get("validation_payload_fingerprint") or ""
            )
            actual_validation_fingerprint = validation.metadata.get(
                "payload_fingerprint"
            ) or _payload_fingerprint(validation.payload)
            if (
                proposal is None
                or proposal.payload.get("status")
                not in {"approval_required", "approved"}
                or proposal.payload.get("proposal_fingerprint") != proposal_fingerprint
                or validation.id != expected_validation_id
                or actual_validation_fingerprint != expected_validation_fingerprint
                or proposal.payload.get("validation_payload_fingerprint")
                != expected_validation_fingerprint
            ):
                raise RuntimeError("monitor_write_proposal_stale")
            facts = sql_validation_facts_from_evidence(validation)
            blocked_resources = blocked_scope_resources(
                facts.target_resources,
                tuple(task.metadata.get("source_scope") or ()),
            )
            if facts.valid is not True or blocked_resources:
                raise RuntimeError("monitor_write_validation_stale")
        sql = validation.payload.get("sql")
        if not sql:
            return task.input
        return {
            **task.input,
            "sql": sql,
            "validated_evidence_id": validation.id,
            "validated_task_id": validation.task_id,
        }

    async def _accepted_evidence_for_dependency(
        self,
        operation_id: str,
        dependency: TaskDependency,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == dependency.evidence_kind
            and evidence.accepted is dependency.evidence_accepted
            and (
                dependency.evidence_id is None or evidence.id == dependency.evidence_id
            )
            and (
                dependency.evidence_owner is None
                or evidence.owner == dependency.evidence_owner
            )
            and (
                dependency.producer_task_id is None
                or evidence.task_id == dependency.producer_task_id
            )
            and (
                dependency.input_hash is None
                or evidence.metadata.get("task_input_hash") == dependency.input_hash
            )
            and _payload_contains(evidence.payload, dependency.evidence_payload)
            and (
                dependency.payload_fingerprint is None
                or dependency.payload_fingerprint
                == _payload_fingerprint(evidence.payload)
            )
        ]
        return matches[-1] if matches else None

    async def _authoritative_validation_evidence(
        self,
        operation: Operation,
        task: Task | None,
    ) -> tuple[Evidence, ...]:
        if task is None or task.capability_id not in {
            "db.sql.execute_read",
            "db.sql.execute_write",
        }:
            return ()
        if task.metadata.get("monitor_action_role") == "write_execution":
            expected_validation_id = str(
                task.metadata.get("validation_evidence_id") or ""
            )
            if expected_validation_id:
                matches = [
                    evidence
                    for evidence in await self.store.list_evidence(operation.id)
                    if evidence.kind == "sql.validation"
                    and evidence.id == expected_validation_id
                    and evidence.accepted
                ]
                if matches:
                    return (matches[-1],)
        validation_dependency = next(
            (
                dependency
                for dependency in task.dependencies
                if dependency.kind.value == "evidence"
                and dependency.evidence_kind == "sql.validation"
            ),
            None,
        )
        if validation_dependency is None:
            return ()
        evidence = await self._accepted_evidence_for_dependency(
            operation.id,
            validation_dependency,
        )
        return (evidence,) if evidence is not None else ()

    async def _latest_accepted_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == kind
            and evidence.accepted
            and _payload_contains(evidence.payload, payload or {})
        ]
        return matches[-1] if matches else None

    async def _latest_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
        accepted: bool | None = None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == kind
            and (accepted is None or evidence.accepted is accepted)
            and _payload_contains(evidence.payload, payload or {})
        ]
        return matches[-1] if matches else None

    async def _latest_monitor_action_result(
        self,
        operation_id: str,
        *,
        action_plan_fingerprint: str,
    ) -> Evidence | None:
        return await self._latest_evidence(
            operation_id,
            "monitor.action_result",
            payload={"action_plan_fingerprint": action_plan_fingerprint},
        )

    async def _analysis_plan_state(
        self,
        operation_id: str,
        *,
        plan_evidence: Evidence,
        validation_evidence: Evidence,
    ) -> _AnalysisPlanState:
        parent_plan = DbAnalysisPlan.from_mapping(plan_evidence.payload)
        revision = await self._latest_accepted_plan_revision(
            operation_id,
            parent_plan_evidence_id=plan_evidence.id,
        )
        if revision is None:
            return _AnalysisPlanState(
                plan=parent_plan,
                plan_evidence=plan_evidence,
                validation_evidence=validation_evidence,
            )
        revised_plan = self._compose_analysis_revision_plan(
            parent_plan,
            revision,
        )
        validation = validate_analysis_plan_payload(
            revised_plan.to_dict(),
            plan_evidence=revision,
            registered_capabilities={
                (capability.owner, capability.id): capability
                for capability in self.registry.capabilities
            },
        )
        if not validation.valid:
            raise RuntimeError(
                "accepted analysis.plan.revision is not executable: "
                + ",".join(validation.errors)
            )
        return _AnalysisPlanState(
            plan=revised_plan,
            plan_evidence=plan_evidence,
            validation_evidence=validation_evidence,
            revision_evidence=revision,
        )

    async def _latest_accepted_plan_revision(
        self,
        operation_id: str,
        *,
        parent_plan_evidence_id: str | None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == "analysis.plan.revision"
            and evidence.accepted
            and evidence.payload.get("parent_plan_evidence_id")
            == parent_plan_evidence_id
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _compose_analysis_revision_plan(
        parent_plan: DbAnalysisPlan,
        revision: Evidence,
    ) -> DbAnalysisPlan:
        unchanged_step_ids = {
            str(item) for item in revision.payload.get("unchanged_step_ids") or ()
        }
        replacement_steps = [
            dict(item)
            for item in revision.payload.get("replacement_steps") or ()
            if isinstance(item, dict)
        ]
        parent_steps = [
            step.to_dict()
            for step in parent_plan.steps
            if step.id in unchanged_step_ids
        ]
        payload = {
            **parent_plan.to_dict(),
            "analysis_id": str(
                revision.payload.get("analysis_id") or parent_plan.analysis_id
            ),
            "steps": [*parent_steps, *replacement_steps],
            "budgets": dict(
                revision.payload.get("budgets") or parent_plan.budgets.to_dict()
            ),
            "diagnostics": {
                **parent_plan.diagnostics,
                "revision_evidence_id": revision.id,
                "revision_number": revision.payload.get("revision_number"),
            },
        }
        return DbAnalysisPlan.from_mapping(payload)

    async def _latest_final_analysis_synthesis(
        self,
        operation_id: str,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == "analysis.synthesis"
            and evidence.accepted
            and not _analysis_synthesis_is_partial(evidence)
        ]
        return matches[-1] if matches else None

    def _analysis_progress_payload(
        self,
        snapshot: OperationSnapshot | None,
        *,
        plan_evidence: Evidence | None = None,
    ) -> dict[str, Any]:
        if snapshot is None:
            return {}
        plan_evidence = plan_evidence or _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.plan",
        )
        validation_evidence = _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.plan.validation",
            payload={"valid": True},
        )
        plan_steps: tuple[Any, ...] = ()
        budgets: dict[str, Any] = {}
        analysis_id = None
        if plan_evidence is not None:
            try:
                plan = DbAnalysisPlan.from_mapping(plan_evidence.payload)
                plan_steps = plan.steps
                budgets = plan.budgets.to_dict()
                analysis_id = plan.analysis_id
            except Exception:
                analysis_id = str(
                    plan_evidence.payload.get("analysis_id")
                    or plan_evidence.metadata.get("analysis_id")
                    or ""
                )
                budgets = dict(plan_evidence.payload.get("budgets") or {})
        completed_steps = {
            str(item.metadata.get("analysis_step_id"))
            for item in snapshot.evidence
            if item.accepted
            and item.metadata.get("analysis_step_id")
            and item.kind
            not in {
                "analysis.plan",
                "analysis.plan.validation",
                "planning.context",
            }
            and (
                item.kind != "analysis.synthesis"
                or not _analysis_synthesis_is_partial(item)
            )
        }
        running_steps = sorted(
            {
                str(task.metadata.get("analysis_step_id"))
                for task in snapshot.tasks
                if task.status is TaskStatus.RUNNING
                and task.metadata.get("analysis_step_id")
            }
        )
        blocked_steps = sorted(
            {
                str(task.metadata.get("analysis_step_id"))
                for task in snapshot.tasks
                if task.status is TaskStatus.BLOCKED
                and task.metadata.get("analysis_step_id")
            }
        )
        remaining_step_ids = [
            step.id for step in plan_steps if step.id not in completed_steps
        ]
        latest_checkpoint = _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.checkpoint",
        )
        latest_synthesis = _latest_accepted_evidence_from_snapshot(
            snapshot,
            "analysis.synthesis",
        )
        final_synthesis = _latest_final_analysis_synthesis_from_snapshot(snapshot)
        return {
            "operation_id": snapshot.operation.id,
            "operation_status": snapshot.operation.status.value,
            "analysis_id": analysis_id,
            "plan_evidence_id": plan_evidence.id if plan_evidence else None,
            "validation_evidence_id": (
                validation_evidence.id if validation_evidence else None
            ),
            "completed_step_ids": sorted(completed_steps),
            "blocked_step_ids": blocked_steps,
            "running_step_ids": running_steps,
            "remaining_step_ids": remaining_step_ids,
            "budgets": budgets,
            "approvals": [
                {
                    "approval_id": approval.approval_id,
                    "status": approval.status.value,
                    "task_id": approval.task_id,
                    "policy_id": approval.requested_by_policy_id,
                }
                for approval in snapshot.approval_requests
            ],
            "next_resumable_task_ids": list(snapshot.resumable_task_ids),
            "latest_checkpoint_id": (
                latest_checkpoint.id if latest_checkpoint else None
            ),
            "latest_synthesis_id": latest_synthesis.id if latest_synthesis else None,
            "latest_synthesis_partial": (
                _analysis_synthesis_is_partial(latest_synthesis)
                if latest_synthesis is not None
                else None
            ),
            "final_synthesis_id": (
                final_synthesis.id if final_synthesis is not None else None
            ),
            "task_status_counts": _task_status_counts(snapshot.tasks),
            "evidence_counts": _evidence_kind_counts(snapshot.evidence),
        }

    @staticmethod
    def _has_pending_approvals(snapshot: OperationSnapshot) -> bool:
        return any(
            request.status is ApprovalStatus.PENDING
            for request in snapshot.approval_requests
        )

    async def _complete_resumed_run_operation(
        self,
        snapshot: OperationSnapshot,
    ) -> None:
        request = _db_request_from_context(snapshot.operation)
        intent = _db_intent_from_context(snapshot.operation)
        contract = _db_contract_from_context(snapshot.operation)
        evidence = tuple(await self.store.list_evidence(snapshot.operation.id))
        tasks = tuple(await self.store.list_tasks(snapshot.operation.id))
        verification = self.verifier.verify(contract, intent, evidence, tasks)
        if not verification.passed:
            await self._record_operation_result(
                DbOperationResult(
                    operation_id=snapshot.operation.id,
                    request=request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.FAILED,
                    answer="DB operation could not be verified against required evidence.",
                    evidence=evidence,
                    warnings=verification.warnings,
                    diagnostics={"verification": verification.to_dict()},
                ),
                operation=snapshot.operation,
            )
            return

        verification_evidence = await self._persist_verification_result_evidence(
            snapshot.operation,
            verification,
            evidence,
        )
        synthesis_evidence, synthesis_task = await self._execute_answer_synthesis(
            operation=snapshot.operation,
            intent=intent,
            outcome_evidence=(*evidence, verification_evidence),
        )
        final_evidence = (*evidence, verification_evidence, synthesis_evidence)
        final_tasks = (*tasks, synthesis_task) if synthesis_task not in tasks else tasks
        await self._record_operation_result(
            DbOperationResult(
                operation_id=snapshot.operation.id,
                request=request,
                intent=intent,
                contract=contract,
                status=OperationStatus.SUCCEEDED,
                answer=_answer_from_synthesis_evidence(synthesis_evidence),
                evidence=final_evidence,
                diagnostics={
                    "verification": verification.to_dict(),
                    "synthesis": synthesis_evidence.payload,
                    "execution": {
                        "task_count": len(final_tasks),
                        "tasks": [task.to_dict() for task in final_tasks],
                    },
                },
            ),
            operation=snapshot.operation,
        )

    async def evaluate_governance_persistence(
        self,
        operation: Operation,
        *,
        task: Task | None = None,
        capability: Capability | None = None,
        stage: str,
    ) -> _GovernancePersistence:
        """Build DB-owned governance facts for kernel task execution."""
        contract = (
            _db_contract_from_context(operation)
            if _operation_has_run_context(operation)
            else None
        )
        return await self._evaluate_governance(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=stage,
        )

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
    ) -> _MonitorEffectGovernanceDecision:
        """Evaluate monitor side-effect governance without owning approval state."""

        contract = (
            _db_contract_from_context(operation)
            if _operation_has_run_context(operation)
            else None
        )
        monitor_context = {}
        for key in ("monitor_action_context", "monitor_delivery_context"):
            value = operation.metadata.get(key)
            if isinstance(value, dict):
                monitor_context.update(value)
        extra_governance_facts: dict[str, Any] = {
            "monitor_effect": {
                "phase": phase,
                **monitor_context,
                "intent": dict(intent),
                "mutate_approvals": mutate_approvals,
            }
        }
        if operation_override:
            extra_governance_facts["operation_override"] = dict(operation_override)
        persistence = await self._evaluate_governance(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=f"monitor.{phase}",
            extra_governance_facts=extra_governance_facts,
            mutate_approvals=mutate_approvals,
        )
        await self.store.commit_governance_evaluation(
            decisions=persistence.result.decisions,
            audit_record=persistence.audit_record,
            approval_requests=(
                persistence.approvals_to_request if mutate_approvals else ()
            ),
            events=persistence.events,
        )
        if persistence.result.blocked:
            status = "blocked"
            reason = "governance_blocked"
        elif persistence.result.pending_approval:
            status = "blocked"
            reason = "governance_approval_required"
        else:
            status = "allowed"
            reason = None
        return _MonitorEffectGovernanceDecision(
            status=status,
            reason=reason,
            result=persistence.result,
            audit_record=persistence.audit_record,
        )

    async def task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        """Return DB-owned dependency readiness for kernel task execution."""
        return await self._task_readiness(task, operation)

    async def executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        """Hydrate DB task input from authoritative validation evidence."""
        return await self._executable_input_for_task(task, operation)

    async def _evaluate_governance(
        self,
        operation: Operation,
        contract: DbOperationContract | None = None,
        *,
        task: Task | None = None,
        capability: Capability | None = None,
        stage: str,
        extra_governance_facts: dict[str, Any] | None = None,
        mutate_approvals: bool = True,
    ) -> _GovernancePersistence:
        policies = self._active_governance_policies(
            contract,
            capability=capability,
        )
        current_evidence = tuple(await self.store.list_evidence(operation.id))
        current_approvals = tuple(await self.store.list_approval_requests(operation.id))
        authoritative_validation_evidence = (
            await self._authoritative_validation_evidence(operation, task)
        )
        governance_facts = _governance_fact_envelope(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=stage,
            source=self.source,
            evidence=current_evidence,
            authoritative_validation_evidence=authoritative_validation_evidence,
            approvals=current_approvals,
        )
        if extra_governance_facts:
            governance_facts = {
                **governance_facts,
                **extra_governance_facts,
                "fact_source": {
                    **dict(governance_facts.get("fact_source") or {}),
                    "sources": _ordered_unique(
                        (
                            *(
                                governance_facts.get("fact_source", {}).get("sources")
                                or ()
                            ),
                            "monitor",
                        )
                    ),
                    "monitor": True,
                },
            }
        governance_operation = _operation_for_governance(
            operation,
            task=task,
            capability=capability,
            stage=stage,
            governance_facts=governance_facts,
        )
        governance_contract = _governance_contract(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=stage,
            governance_facts=governance_facts,
        )
        result = PolicyEvaluator(policies).evaluate_operation(
            governance_operation,
            contract=governance_contract,
        )
        if mutate_approvals:
            result, approvals_to_request = await self._reconcile_approval_state(result)
        else:
            approvals_to_request = ()
        audit_record = await self._governance_audit_record(
            operation,
            result,
            policies=policies,
            contract=governance_contract,
            task=task,
            capability=capability,
            stage=stage,
            approvals_to_request=approvals_to_request,
            governance_facts=governance_facts,
        )
        events: list[RuntimeEvent] = []
        for decision in result.decisions:
            events.append(
                self._runtime_event(
                    type=RuntimeEventType.POLICY_DECISION,
                    operation_id=operation.id,
                    task_id=task.id if task is not None else None,
                    task=task,
                    capability=capability,
                    message=(
                        f"Policy {decision.owner}:{decision.policy_id} returned "
                        f"{decision.effect.value}."
                    ),
                    policy_id=decision.policy_id,
                    payload={"decision": decision.to_dict()},
                )
            )
        for approval in approvals_to_request:
            events.append(
                self._runtime_event(
                    type=RuntimeEventType.APPROVAL_REQUESTED,
                    operation_id=operation.id,
                    task_id=task.id if task is not None else None,
                    task=task,
                    capability=capability,
                    message=f"Approval {approval.approval_id} requested.",
                    policy_id=approval.requested_by_policy_id,
                    approval_id=approval.approval_id,
                    payload={"approval": approval.to_dict()},
                )
            )
        return _GovernancePersistence(
            result=result,
            audit_record=audit_record,
            approvals_to_request=approvals_to_request,
            events=tuple(events),
        )

    def _active_governance_policies(
        self,
        contract: DbOperationContract | None,
        *,
        capability: Capability | None = None,
    ) -> tuple[Any, ...]:
        active_policy_ids = set(contract.policy_ids if contract is not None else ())
        policies: list[Any] = [*default_db_policies(), *self.config.policies]
        if (
            capability is not None
            and capability.id == "db.answer.synthesize"
            and capability.access is AccessMode.NONE
            and not capability.side_effecting
        ):
            # Final prose consumes already-governed evidence and never touches
            # connector state, so extension policies stay scoped to DB work.
            return tuple(policies)

        for policy in self.registry.policies:
            identity = f"{policy.owner}:{policy.id}"
            owner_plugin = self.registry.get_plugin(policy.owner)
            owner_manifest = getattr(owner_plugin, "manifest", None)
            is_skill_policy = (
                owner_manifest is not None
                and getattr(owner_manifest, "kind", None) is PluginKind.SKILL
            )
            if not is_skill_policy or identity in active_policy_ids:
                policies.append(policy)

        return tuple(policies)

    async def _governance_audit_record(
        self,
        operation: Operation,
        result: GovernanceResult,
        *,
        policies: tuple[Any, ...],
        contract: dict[str, Any],
        task: Task | None,
        capability: Capability | None,
        stage: str,
        approvals_to_request: tuple[ApprovalRequest, ...],
        governance_facts: dict[str, Any],
    ) -> GovernanceAuditRecord:
        audit_id = f"governance-audit-{uuid4()}"
        actor = _actor_context(operation)
        tenant = _tenant_context(operation)
        source_scope = _source_scope_context(operation)
        resource = _resource_context(operation, self.source, capability)
        current_evidence = tuple(await self.store.list_evidence(operation.id))
        current_approvals = tuple(await self.store.list_approval_requests(operation.id))
        evidence_context = {
            "evidence": [_evidence_trace_summary(item) for item in current_evidence],
        }
        approval_context = {
            "approval_statuses": dict(result.metadata.get("approval_statuses") or {}),
            "pending_request_ids": [
                request.approval_id for request in result.approval_requests
            ],
            "new_request_ids": [
                request.approval_id for request in approvals_to_request
            ],
            "existing": [
                _approval_trace_summary(request) for request in current_approvals
            ],
            "requested": [
                _approval_trace_summary(request) for request in approvals_to_request
            ],
        }
        runtime_facts = {
            "runtime_id": self.runtime_id,
            "runtime_kind": self.runtime_kind,
            "stage": stage,
            "operation_type": operation.operation_type,
            "contract": contract,
            "governance_facts": governance_facts,
            "result": {
                "allowed": result.allowed,
                "blocked": result.blocked,
                "pending_approval": result.pending_approval,
            },
            "policies": [_policy_trace_summary(policy) for policy in policies],
            "decision_count": len(result.decisions),
        }
        evaluation_trace = _governance_evaluation_trace(
            result,
            policies=policies,
            runtime_facts=runtime_facts,
        )
        traces = tuple(
            PolicyDecisionTrace(
                trace_id=f"{audit_id}:decision:{index}",
                operation_id=operation.id,
                policy_id=decision.policy_id,
                owner=decision.owner,
                policy_version=decision.policy_version,
                policy_identity=str(decision.policy_identity),
                effect=decision.effect,
                reason=decision.reason,
                stage=stage,
                task_id=task.id if task is not None else None,
                capability_id=capability.id if capability is not None else None,
                approval_ids=_approval_ids_for_decision(
                    decision, result, approvals_to_request
                ),
                evidence_ids=tuple(
                    _ordered_unique(
                        (
                            *(
                                item.id
                                for item in decision.evidence
                                if item.id is not None
                            ),
                            *(decision.metadata.get("validation_evidence_ids") or ()),
                        )
                    )
                ),
                actor=actor,
                tenant=tenant,
                source_scope=source_scope,
                resource=resource,
                runtime_facts={
                    "contract": contract,
                    "governance_facts": governance_facts,
                    "approval_context": approval_context,
                    "evidence_context": evidence_context,
                    "decision_metadata": decision.metadata,
                },
            )
            for index, decision in enumerate(result.decisions, start=1)
        )
        return GovernanceAuditRecord(
            audit_id=audit_id,
            operation_id=operation.id,
            stage=stage,
            allowed=result.allowed,
            blocked=result.blocked,
            pending_approval=result.pending_approval,
            policy_decisions=result.decisions,
            traces=traces,
            task_id=task.id if task is not None else None,
            capability_id=capability.id if capability is not None else None,
            actor=actor,
            tenant=tenant,
            source_scope=source_scope,
            resource=resource,
            operation_context=_operation_trace_context(operation),
            task_context=_task_trace_context(task),
            capability_context=(
                _capability_governance_facts(capability)
                if capability is not None
                else {}
            ),
            approval_context=approval_context,
            evidence_context=evidence_context,
            runtime_facts=runtime_facts,
            evaluation_trace=evaluation_trace,
            metadata={
                "governance_metadata": result.metadata,
                "policy_identities": [
                    decision.policy_identity for decision in result.decisions
                ],
            },
        )

    async def _reconcile_approval_state(
        self, result: GovernanceResult
    ) -> tuple[GovernanceResult, tuple[ApprovalRequest, ...]]:
        if not result.approval_requests:
            return result, ()
        existing_by_id = {
            approval.approval_id: approval
            for approval in await self.store.list_approval_requests()
        }
        pending_requests: list[ApprovalRequest] = []
        approvals_to_request: list[ApprovalRequest] = []
        approval_statuses: dict[str, str] = {}
        terminal_blocking_statuses: dict[str, str] = {}

        for approval in result.approval_requests:
            existing = existing_by_id.get(approval.approval_id)
            if existing is None:
                pending_requests.append(approval)
                approvals_to_request.append(approval)
                approval_statuses[approval.approval_id] = ApprovalStatus.PENDING.value
                continue
            approval_statuses[approval.approval_id] = existing.status.value
            if existing.status is ApprovalStatus.APPROVED:
                continue
            if existing.status is ApprovalStatus.PENDING:
                pending_requests.append(existing)
                continue
            terminal_blocking_statuses[existing.approval_id] = existing.status.value

        blocked = result.blocked or bool(terminal_blocking_statuses)
        pending_approval = bool(pending_requests)
        metadata = {
            **result.metadata,
            "approval_statuses": approval_statuses,
        }
        if terminal_blocking_statuses:
            metadata["terminal_approval_statuses"] = terminal_blocking_statuses
        return (
            GovernanceResult(
                allowed=not blocked and not pending_approval,
                blocked=blocked,
                pending_approval=pending_approval,
                decisions=result.decisions,
                approval_requests=tuple(pending_requests),
                modified_contract=result.modified_contract,
                metadata=metadata,
            ),
            tuple(approvals_to_request),
        )

    async def _record_operation_result(
        self,
        result: DbOperationResult,
        *,
        operation: Operation | None = None,
    ) -> DbOperationResult:
        self._operation_results.append(result)
        self._audit_log.append(_audit_entry_from_result(result))
        if operation is not None:
            message = (
                f"Operation {result.operation_id} finished with "
                f"{result.status.value}."
            )
            payload = {"warnings": list(result.warnings)}
            if result.status in {
                OperationStatus.SUCCEEDED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
            }:
                await self.kernel.complete_operation(
                    result.operation_id,
                    status=result.status,
                    message=message,
                    payload=payload,
                )
            elif result.status is OperationStatus.BLOCKED:
                await self.kernel.block_operation(
                    result.operation_id,
                    message=message,
                    payload=payload,
                )
            else:
                await self.kernel.update_operation(
                    result.operation_id,
                    result.status,
                    message=message,
                    payload=payload,
                )
        return result

    async def _stored_operation_count(self) -> int:
        operations = await self.store.list_operations()
        return len(operations)

    async def _last_stored_operation_id(self) -> str | None:
        operations = await self.store.list_operations()
        return operations[-1].id if operations else None

    def _runtime_event(
        self,
        *,
        type: RuntimeEventType,
        operation_id: str,
        message: str,
        task_id: str | None = None,
        task: Task | None = None,
        capability: Capability | None = None,
        payload: dict[str, Any] | None = None,
        policy_id: str | None = None,
        approval_id: str | None = None,
        evidence_id: str | None = None,
    ) -> RuntimeEvent:
        if capability is None and task is not None:
            try:
                capability = self._capability_for_task(task)
            except Exception:
                capability = None
        trace_id, span_id = _current_trace_ids()
        return RuntimeEvent(
            type=type,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            operation_id=operation_id,
            task_id=task_id or (task.id if task is not None else None),
            capability_id=(
                capability.id
                if capability is not None
                else task.capability_id if task is not None else None
            ),
            executor_id=(
                capability.executor
                if capability is not None
                else task.executor_id if task is not None else None
            ),
            plugin_id=capability.owner if capability is not None else None,
            policy_id=policy_id,
            approval_id=approval_id,
            evidence_id=evidence_id,
            trace_id=trace_id,
            span_id=span_id,
            message=message,
            payload=dict(payload or {}),
        )

    def _capability_for_task(self, task: Task) -> Capability:
        owner = task.metadata.get("owner") if task.metadata else None
        if owner:
            return self.registry.get_capability(task.capability_id, owner=str(owner))
        try:
            return self.registry.get_capability(task.capability_id)
        except ValueError:
            for capability in self.registry.capabilities:
                if (
                    capability.id == task.capability_id
                    and capability.executor == task.executor_id
                ):
                    return capability
            raise


class DbRuntimeGovernanceBlocked(PermissionError):
    """Raised when persisted runtime governance blocks executor invocation."""

    def __init__(
        self,
        *,
        operation: Operation,
        task: Task | None,
        governance: GovernanceResult,
    ) -> None:
        self.operation = operation
        self.task = task
        self.governance = governance
        super().__init__(_governance_blocked_answer(governance))


class DbRuntimeTaskNotRunnable(RuntimeError):
    """Raised when a persisted task is not currently executable."""

    def __init__(
        self,
        task: Task,
        message: str,
        *,
        readiness: dict[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.readiness = readiness or {}
        super().__init__(message)


def _tasks_in_resume_order(tasks: tuple[Task, ...]) -> tuple[Task, ...]:
    return tuple(
        sorted(
            tasks,
            key=lambda task: (
                int(task.metadata.get("sequence") or 0),
                task.id,
            ),
        )
    )


def _operation_has_run_context(operation: Operation) -> bool:
    return isinstance(operation.metadata.get("resume_context"), dict)


def _snapshot_has_incomplete_analysis(snapshot: OperationSnapshot) -> bool:
    has_plan = any(
        evidence.kind == "analysis.plan" and evidence.accepted
        for evidence in snapshot.evidence
    )
    has_validation = any(
        evidence.kind == "analysis.plan.validation"
        and evidence.accepted
        and evidence.payload.get("valid") is True
        for evidence in snapshot.evidence
    )
    has_synthesis = any(
        evidence.kind == "analysis.synthesis"
        and evidence.accepted
        and not _analysis_synthesis_is_partial(evidence)
        for evidence in snapshot.evidence
    )
    return has_plan and has_validation and not has_synthesis


def _analysis_synthesis_is_partial(evidence: Evidence | None) -> bool:
    if evidence is None:
        return False
    return bool(evidence.payload.get("partial") or evidence.metadata.get("partial"))


def _latest_final_analysis_synthesis_from_snapshot(
    snapshot: OperationSnapshot,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in snapshot.evidence
        if evidence.kind == "analysis.synthesis"
        and evidence.accepted
        and not _analysis_synthesis_is_partial(evidence)
    ]
    return matches[-1] if matches else None


def _latest_accepted_evidence_from_snapshot(
    snapshot: OperationSnapshot,
    kind: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in snapshot.evidence
        if evidence.kind == kind
        and evidence.accepted
        and _payload_contains(evidence.payload, payload or {})
    ]
    return matches[-1] if matches else None


def _task_status_counts(tasks: tuple[Task, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.status.value] = counts.get(task.status.value, 0) + 1
    return counts


def _evidence_kind_counts(evidence: tuple[Evidence, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    return counts


def _checkpoint_reason(diagnostics: dict[str, Any]) -> str:
    if diagnostics.get("budget_exceeded"):
        return "budget_exhausted"
    if diagnostics.get("blocked_reason"):
        return "blocked"
    if diagnostics.get("cancelled") or diagnostics.get("cancelled_reason"):
        return "cancelled"
    if diagnostics.get("interrupted") or diagnostics.get("error"):
        return "interrupted"
    if diagnostics.get("pause_reason"):
        return "paused"
    return "checkpoint"


def _has_running_tasks(tasks: tuple[Task, ...]) -> bool:
    return any(task.status is TaskStatus.RUNNING for task in tasks)


def _analysis_budget_usage(
    evidence: tuple[Evidence, ...],
    *,
    started_at: float,
) -> dict[str, Any]:
    return {
        "total_rows": sum(
            _analysis_query_result_rows(item)
            for item in evidence
            if item.kind == "query.result" and item.accepted
        ),
        "llm_calls": sum(1 for item in evidence if _analysis_evidence_used_llm(item)),
        "context_chars": sum(
            len(str(item.payload.get("rendered_context") or ""))
            for item in evidence
            if item.kind == "planning.context" and item.accepted
        ),
        "duration_seconds": time.monotonic() - started_at,
    }


def _analysis_query_result_rows(evidence: Evidence) -> int:
    rows = evidence.payload.get("rows")
    if isinstance(rows, list):
        return len(rows)
    try:
        return int(evidence.payload.get("total_rows") or 0)
    except (TypeError, ValueError):
        return 0


def _analysis_evidence_used_llm(evidence: Evidence) -> bool:
    if not evidence.accepted:
        return False
    diagnostics = evidence.payload.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("mode") == "llm":
        return True
    planner_diagnostics = evidence.payload.get("planner_diagnostics")
    return isinstance(planner_diagnostics, dict) and bool(
        planner_diagnostics.get("model") or planner_diagnostics.get("provider")
    )


def _db_request_context(request: DbRequest) -> dict[str, Any]:
    return {
        "prompt": request.prompt,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "source_scope": list(request.source_scope),
        "mode": request.mode,
        "requested_capabilities": list(request.requested_capabilities),
        "constraints": request.constraints,
        "metadata": request.metadata,
    }


def _skill_names_from_request(request: DbRequest) -> tuple[str, ...]:
    value = request.metadata.get("skills")
    if value is None:
        value = request.metadata.get("selected_skills")
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _effects_modify_contract(effects: tuple[Any, ...]) -> bool:
    return any(
        effect.requested_capabilities
        or effect.required_evidence
        or effect.policy_ids
        or effect.contract_metadata
        or effect.verifier_metadata
        or effect.synthesis_metadata
        for effect in effects
    )


def _db_intent_context(intent: DbIntent) -> dict[str, Any]:
    return {
        "kind": intent.kind.value,
        "confidence": intent.confidence,
        "access": intent.access.value,
        "evidence_mode": intent.evidence_mode,
        "requested_outputs": list(intent.requested_outputs),
        "constraints": intent.constraints,
        "diagnostics": intent.diagnostics,
    }


def _db_contract_context(contract: DbOperationContract) -> dict[str, Any]:
    return {
        "operation_type": contract.operation_type,
        "required_capabilities": list(contract.required_capabilities),
        "required_evidence": list(contract.required_evidence),
        "access": contract.access.value,
        "limits": contract.limits.to_dict(),
        "policy_ids": list(contract.policy_ids),
        "metadata": contract.metadata,
    }


def _resume_context(operation: Operation) -> dict[str, Any]:
    context = operation.metadata.get("resume_context")
    return context if isinstance(context, dict) else {}


def _db_request_from_context(operation: Operation) -> DbRequest:
    context = _resume_context(operation).get("request") or operation.request
    return DbRequest(
        prompt=str(context.get("prompt") or ""),
        user_id=context.get("user_id"),
        session_id=context.get("session_id"),
        source_scope=tuple(context.get("source_scope") or ()),
        mode=context.get("mode"),
        requested_capabilities=tuple(context.get("requested_capabilities") or ()),
        constraints=dict(context.get("constraints") or {}),
        metadata=dict(context.get("metadata") or {}),
    )


def _db_intent_from_context(operation: Operation) -> DbIntent:
    context = _resume_context(operation).get("intent") or {}
    return DbIntent(
        kind=DbIntentKind(str(context.get("kind") or operation.operation_type)),
        confidence=float(context.get("confidence", 1.0)),
        access=AccessMode(str(context.get("access") or AccessMode.NONE.value)),
        evidence_mode=str(context.get("evidence_mode") or "none"),
        requested_outputs=tuple(context.get("requested_outputs") or ()),
        constraints=dict(context.get("constraints") or {}),
        diagnostics=dict(context.get("diagnostics") or {}),
    )


def _db_contract_from_context(operation: Operation) -> DbOperationContract:
    context = _resume_context(operation).get("contract") or {}
    limits = dict(context.get("limits") or {})
    return DbOperationContract(
        operation_type=str(context.get("operation_type") or operation.operation_type),
        required_capabilities=tuple(context.get("required_capabilities") or ()),
        required_evidence=tuple(
            context.get("required_evidence") or sorted(operation.required_evidence)
        ),
        access=AccessMode(str(context.get("access") or AccessMode.NONE.value)),
        limits=DbLimits(**limits) if limits else DbLimits(),
        policy_ids=tuple(context.get("policy_ids") or ()),
        metadata=dict(context.get("metadata") or {}),
    )


def _planned_task_input(operation: Operation, capability: Capability) -> dict[str, Any]:
    prompt = str(operation.request.get("prompt") or "")
    if capability.id in {"db.sql.execute_read", "db.sql.execute_write"}:
        return {"sql_ref": "sql.validation"}
    if capability.id == "db.sql.validate":
        return {"sql": prompt, "operation": operation.operation_type}
    return {"prompt": prompt}


def _synthesis_dependencies(
    operation: Operation,
    intent: DbIntent,
    evidence: tuple[Evidence, ...],
) -> tuple[TaskDependency, ...]:
    accepted = tuple(
        item
        for item in evidence
        if item.accepted and item.operation_id == operation.id and item.id
    )
    dependencies: list[TaskDependency] = []
    if intent.kind in {
        DbIntentKind.DATA_QUERY,
        DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
    }:
        _append_dependency_for_kind(dependencies, accepted, "planning.context")
        if not any(item.evidence_kind == "planning.context" for item in dependencies):
            _append_dependency_for_any(
                dependencies,
                accepted,
                ("schema.asset_profile", "catalog.source", "schema.search_result"),
            )
        for kind in (
            "query.result",
            "query.plan.proposal",
            "query.plan.validation",
            "sql.validation",
            "verification.result",
        ):
            _append_dependency_for_kind(dependencies, accepted, kind)
    elif intent.kind is DbIntentKind.SCHEMA_QUERY:
        _append_dependency_for_any(
            dependencies,
            accepted,
            (
                "planning.context",
                "schema.asset_profile",
                "catalog.source",
                "schema.search_result",
                "catalog.asset.profile",
            ),
        )
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    else:
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[TaskDependency] = []
    for dependency in dependencies:
        key = (dependency.evidence_kind, dependency.evidence_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(dependency)
    return tuple(unique)


def _append_dependency_for_kind(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
    kind: str,
) -> None:
    item = next(
        (candidate for candidate in reversed(evidence) if candidate.kind == kind),
        None,
    )
    if item is not None:
        dependencies.append(_dependency_for_evidence(item))


def _append_dependency_for_any(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
    kinds: tuple[str, ...],
) -> None:
    for kind in kinds:
        item = next(
            (candidate for candidate in reversed(evidence) if candidate.kind == kind),
            None,
        )
        if item is not None:
            dependencies.append(_dependency_for_evidence(item))
            return
    catalog = next(
        (
            candidate
            for candidate in reversed(evidence)
            if candidate.kind.startswith("catalog.")
        ),
        None,
    )
    if catalog is not None:
        dependencies.append(_dependency_for_evidence(catalog))


def _dependency_for_evidence(evidence: Evidence) -> TaskDependency:
    return TaskDependency(
        kind="evidence",
        evidence_kind=evidence.kind,
        evidence_id=evidence.id,
        evidence_owner=evidence.owner,
        producer_task_id=evidence.task_id,
        evidence_accepted=True,
        operation_id=evidence.operation_id,
        payload_fingerprint=evidence.metadata.get("payload_fingerprint")
        or _payload_fingerprint(evidence.payload),
    )


def _synthesis_context_option(
    metadata: dict[str, Any],
    key: str,
    default: int,
) -> int:
    options = metadata.get("from_db_options")
    if isinstance(options, dict) and options.get(key) is not None:
        try:
            return int(options[key])
        except (TypeError, ValueError):
            return default
    return default


def _answer_from_synthesis_evidence(evidence: Evidence) -> str:
    answer = evidence.payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("accepted answer.synthesis evidence is missing answer")
    return answer


def _answer_from_analysis_synthesis_evidence(evidence: Evidence) -> str:
    answer = evidence.payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("accepted analysis.synthesis evidence is missing answer")
    return answer


def _task_dependencies_for_capability(
    operation: Operation,
    capability: Capability,
    *,
    validation_task: Task | None = None,
) -> tuple[TaskDependency, ...]:
    if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
        return ()
    if capability.id == "db.sql.execute_read" and validation_task is None:
        return ()
    validation_dependency = TaskDependency(
        kind="evidence",
        evidence_kind="sql.validation",
        evidence_owner=(
            validation_task.metadata.get("owner") if validation_task else None
        ),
        producer_task_id=validation_task.id if validation_task else None,
        producer_capability_id=(
            validation_task.capability_id if validation_task else "db.sql.validate"
        ),
        producer_executor_id=(validation_task.executor_id if validation_task else None),
        evidence_payload={"valid": True},
        operation_id=operation.id,
        input_hash=(
            validation_task.metadata.get("input_hash") if validation_task else None
        ),
    )
    if capability.id == "db.sql.execute_read":
        return (validation_dependency,)
    return (
        validation_dependency,
        TaskDependency(
            kind="approval",
            approval_status=ApprovalStatus.APPROVED,
            approval_policy_id="approval_required_for_writes",
            approval_name="human",
            operation_id=operation.id,
        ),
    )


def _payload_contains(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True


def _accepted_task_evidence(
    evidence: Evidence,
    *,
    operation: Operation,
    task: Task,
) -> Evidence:
    evidence_identity = {
        "operation_id": operation.id,
        "task_id": task.id,
        "kind": evidence.kind,
        "payload": evidence.payload,
    }
    evidence_id = evidence.id or f"evidence-{_stable_hash(evidence_identity)}"
    return replace(
        evidence,
        id=evidence_id,
        operation_id=evidence.operation_id or operation.id,
        task_id=evidence.task_id or task.id,
        metadata={
            **evidence.metadata,
            "payload_fingerprint": _payload_fingerprint(evidence.payload),
            "task_input_hash": task.metadata.get("input_hash"),
        },
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return _stable_hash(payload)


def _normalize_monitor_action_plan(
    action_plan: dict[str, Any],
    *,
    operation_id: str,
) -> dict[str, Any]:
    raw = dict(action_plan or {})
    if raw.get("valid") is not None and isinstance(raw.get("analysis_plan"), dict):
        if raw.get("kind") == "scheduled_report" and "delivery_intent" not in raw:
            return {
                **raw,
                "delivery_intent": dict(raw.get("delivery") or {}),
                "delivery_status": raw.get("delivery_status") or "deferred",
            }
        return raw
    kind = str(raw.get("kind") or raw.get("type") or "").strip()
    steps = [dict(item) for item in raw.get("steps") or () if isinstance(item, dict)]
    if not kind:
        if any(str(step.get("kind") or "") == "report_generate" for step in steps):
            kind = "scheduled_report"
        elif steps:
            kind = "investigation"
    if kind in {"report", "scheduled-report", "scheduled_report"}:
        kind = "scheduled_report"
    if kind in {"investigate", "investigation"}:
        kind = "investigation"
    if kind in {
        "write",
        "write-proposal",
        "write_proposal",
        "remediation_sql",
        "propose_write",
    }:
        kind = "write_proposal"

    if kind == "investigation":
        analysis_steps = [
            _normalize_monitor_analysis_step(step)
            for step in steps
            if str(step.get("kind") or "") != "report_generate"
        ]
        if not analysis_steps:
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_executable_investigation_steps",
            )
        analysis_payload = {
            "analysis_id": str(
                raw.get("analysis_id") or f"monitor-action-{operation_id}"
            ),
            "goal": str(
                raw.get("goal") or raw.get("purpose") or "Investigate monitor trigger"
            ),
            "steps": analysis_steps,
            "budgets": dict(raw.get("budgets") or {}),
            "diagnostics": {
                **dict(raw.get("diagnostics") or {}),
                "source": "monitor.action_plan",
                "monitor_action_kind": "investigation",
            },
        }
        try:
            DbAnalysisPlan.from_mapping(analysis_payload)
        except Exception as exc:
            return _invalid_monitor_action(raw, kind=kind, reason=str(exc))
        return {
            "valid": True,
            "kind": "investigation",
            "goal": analysis_payload["goal"],
            "analysis_plan": analysis_payload,
            "original_action_plan": raw,
        }

    if kind == "scheduled_report":
        report_steps = _normalize_monitor_report_steps(steps)
        if not report_steps:
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_deterministic_report_steps",
            )
        analysis_steps = [
            _normalize_monitor_analysis_step(step)
            for step in steps
            if _is_monitor_report_analysis_step(step)
        ]
        if not any(step["kind"] == "synthesis" for step in analysis_steps):
            depends_on = [
                str(step.get("id"))
                for step in analysis_steps
                if step.get("id") and step.get("kind") != "checkpoint"
            ]
            analysis_steps.append(
                {
                    "id": "report_summary",
                    "kind": "synthesis",
                    "purpose": str(
                        raw.get("summary_purpose")
                        or "Generate the durable monitor report narrative"
                    ),
                    "depends_on": [],
                    "expected_evidence": ["analysis.synthesis"],
                    "input": {"report_step_ids": depends_on},
                }
            )
        analysis_payload = {
            "analysis_id": str(
                raw.get("analysis_id") or f"monitor-report-{operation_id}"
            ),
            "goal": str(
                raw.get("title")
                or raw.get("goal")
                or "Generate scheduled monitor report"
            ),
            "steps": analysis_steps,
            "budgets": dict(raw.get("budgets") or {}),
            "diagnostics": {
                **dict(raw.get("diagnostics") or {}),
                "source": "monitor.action_plan",
                "monitor_action_kind": "scheduled_report",
            },
        }
        try:
            DbAnalysisPlan.from_mapping(analysis_payload)
        except Exception as exc:
            return _invalid_monitor_action(raw, kind=kind, reason=str(exc))
        return {
            "valid": True,
            "kind": "scheduled_report",
            "title": raw.get("title") or raw.get("goal"),
            "steps": report_steps,
            "analysis_plan": analysis_payload,
            "output": dict(
                raw.get("output") or {"kind": "report", "format": "markdown"}
            ),
            "delivery_status": "deferred",
            "delivery_phase": 6,
            "delivery_intent": dict(
                raw.get("delivery_intent") or raw.get("delivery") or {}
            ),
            "original_action_plan": raw,
        }

    if kind == "write_proposal":
        proposal = raw.get("proposal")
        proposal = proposal if isinstance(proposal, dict) else {}
        sql = str(raw.get("sql") or proposal.get("sql") or "").strip()
        if not sql:
            return _invalid_monitor_action(
                raw,
                kind=kind,
                reason="missing_write_sql",
            )
        normalized = {
            "valid": True,
            "kind": "write_proposal",
            "sql": sql,
            "params": list(raw.get("params") or proposal.get("params") or ()),
            "source_scope": list(raw.get("source_scope") or ()),
            "purpose": str(
                raw.get("purpose")
                or proposal.get("purpose")
                or "Monitor write proposal"
            ),
            "original_action_plan": raw,
        }
        for key in ("capability_id", "capability_owner"):
            if raw.get(key) or proposal.get(key):
                normalized[key] = str(raw.get(key) or proposal.get(key))
        return normalized

    return _invalid_monitor_action(
        raw,
        kind=kind or "unknown",
        reason="unsupported_action_kind",
    )


def _normalize_monitor_analysis_step(step: dict[str, Any]) -> dict[str, Any]:
    kind = str(step.get("kind") or "").strip()
    normalized = {
        "id": str(step.get("id") or f"{kind}_step").strip(),
        "kind": kind,
        "purpose": str(step.get("purpose") or step.get("metric") or kind).strip(),
        "depends_on": [str(item) for item in step.get("depends_on") or ()],
        "input_refs": [
            dict(item)
            for item in step.get("input_refs") or ()
            if isinstance(item, dict)
        ],
        "expected_evidence": [
            str(item) for item in step.get("expected_evidence") or ()
        ],
        "input": dict(step.get("input") or {}),
        "context_evidence_refs": [
            dict(item)
            for item in step.get("context_evidence_refs") or ()
            if isinstance(item, dict)
        ],
        "budgets": dict(step.get("budgets") or {}),
    }
    for key in ("capability_id", "capability_owner"):
        if step.get(key):
            normalized[key] = str(step[key])
    return normalized


def _normalize_monitor_report_steps(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        kind = str(step.get("kind") or "").strip()
        if kind == "report_generate":
            continue
        if kind == "query" and step.get("sql"):
            kind = "metric_sql"
        if kind not in {"metric_sql", "freshness_sql", "planned_read"}:
            continue
        sql = str(step.get("sql") or "").strip()
        if not sql:
            continue
        normalized.append(
            {
                "id": str(step.get("id") or f"report_step_{index}"),
                "kind": kind,
                "metric": step.get("metric"),
                "purpose": str(step.get("purpose") or step.get("metric") or kind),
                "sql": sql,
                "value_path": step.get("value_path"),
                "source_scope": list(step.get("source_scope") or ()),
                "parameters": list(step.get("parameters") or step.get("params") or ()),
                **(
                    {"capability_owner": str(step["capability_owner"])}
                    if step.get("capability_owner")
                    else {}
                ),
            }
        )
    return normalized


def _is_monitor_report_analysis_step(step: dict[str, Any]) -> bool:
    kind = str(step.get("kind") or "").strip()
    if kind in {"metric_sql", "freshness_sql", "planned_read", "report_generate"}:
        return False
    if kind == "query" and step.get("sql"):
        return False
    return kind in {"query", "checkpoint", "synthesis"} or (
        capability_contract_for_step_kind(kind) is not None
    )


def _invalid_monitor_action(
    raw: dict[str, Any],
    *,
    kind: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "valid": False,
        "kind": kind,
        "block_reason": reason,
        "original_action_plan": raw,
    }


def _monitor_action_budget_usage(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    total_rows = 0
    for item in evidence:
        if item.kind != "query.result":
            continue
        rows = item.payload.get("rows")
        if isinstance(rows, list):
            total_rows += len(rows)
    return {
        "evidence_count": len(evidence),
        "query_result_rows": total_rows,
    }


def _monitor_report_has_analysis_work(plan: DbAnalysisPlan) -> bool:
    return any(step.kind not in {"checkpoint", "synthesis"} for step in plan.steps)


def _monitor_action_context(operation: Operation) -> dict[str, Any]:
    context = operation.metadata.get("monitor_action_context")
    return context if isinstance(context, dict) else {}


def _monitor_delivery_context(operation: Operation) -> dict[str, Any]:
    context = operation.metadata.get("monitor_delivery_context")
    return context if isinstance(context, dict) else {}


def _terminal_monitor_approval_reason(
    approvals: tuple[ApprovalRequest, ...],
) -> str | None:
    statuses = {approval.status for approval in approvals}
    if ApprovalStatus.REJECTED in statuses:
        return "approval_rejected"
    if ApprovalStatus.CANCELLED in statuses:
        return "approval_cancelled"
    if ApprovalStatus.EXPIRED in statuses:
        return "approval_expired"
    return None


def _monitor_action_status_from_operation(operation: Operation) -> str:
    if operation.status is OperationStatus.BLOCKED:
        return "blocked"
    if operation.status is OperationStatus.FAILED:
        return "failed"
    if operation.status is OperationStatus.SUCCEEDED:
        return "succeeded"
    return operation.status.value


def _audit_entry_from_result(result: DbOperationResult) -> dict[str, Any]:
    """Build a redacted, JSON-safe summary for operation inspection."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation_id": result.operation_id,
        "prompt": result.request.prompt,
        "status": result.status.value,
        "intent_kind": result.intent.kind.value,
        "operation_type": result.contract.operation_type,
        "warnings": list(result.warnings),
        "evidence": [_evidence_audit_summary(item) for item in result.evidence],
        "evidence_refs": (
            result.diagnostics.get("execution", {}).get("evidence_refs", [])
            if isinstance(result.diagnostics.get("execution"), dict)
            else []
        ),
    }


def _evidence_audit_summary(evidence: Evidence) -> dict[str, Any]:
    payload = evidence.payload
    summary: dict[str, Any] = {
        "kind": evidence.kind,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
        "payload_keys": sorted(str(key) for key in payload),
    }
    if "sql" in payload:
        summary["sql"] = payload["sql"]
    if isinstance(payload.get("rows"), list):
        summary["row_count"] = len(payload["rows"])
    if "total_rows" in payload:
        summary["total_rows"] = payload["total_rows"]
    if "truncated" in payload:
        summary["truncated"] = payload["truncated"]
    if "success" in payload:
        summary["success"] = payload["success"]
    if "error" in payload:
        summary["error"] = payload["error"]
    return summary


def _evidence_trace_summary(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "operation_id": evidence.operation_id,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
        "schema_version": evidence.schema_version,
        "metadata": evidence.metadata,
        "payload_keys": sorted(str(key) for key in evidence.payload),
    }


def _approval_trace_summary(request: ApprovalRequest) -> dict[str, Any]:
    return {
        "approval_id": request.approval_id,
        "operation_id": request.operation_id,
        "status": request.status.value,
        "requested_by_policy_id": request.requested_by_policy_id,
        "owner": request.owner,
        "risk": request.risk.value,
        "evidence_ids": list(request.evidence_ids),
        "metadata": _metadata_summary(request.metadata),
        "proposed_action": _approval_action_summary(request.proposed_action),
    }


def _policy_trace_summary(policy: Any) -> dict[str, Any]:
    owner = str(getattr(policy, "owner", "runtime"))
    policy_id = str(getattr(policy, "id", "unknown"))
    version = str(
        getattr(policy, "policy_version", None)
        or getattr(policy, "version", None)
        or "1"
    )
    return {
        "owner": owner,
        "policy_id": policy_id,
        "policy_version": version,
        "policy_identity": f"{owner}:{policy_id}@{version}",
        "class": type(policy).__name__,
    }


def _governance_evaluation_trace(
    result: GovernanceResult,
    *,
    policies: tuple[Any, ...],
    runtime_facts: dict[str, Any],
) -> dict[str, Any]:
    if result.blocked:
        effect = "deny"
        reason = "At least one policy denied execution."
    elif result.pending_approval:
        effect = "require_approval"
        reason = "At least one policy required approval before execution."
    elif any(decision.effect.value == "modify" for decision in result.decisions):
        effect = "modify"
        reason = "Policy modifications were applied and execution was allowed."
    elif any(decision.effect.value == "warn" for decision in result.decisions):
        effect = "warn"
        reason = "Policy warnings were recorded and execution was allowed."
    else:
        effect = "allow"
        reason = "No applicable policy denied, modified, warned, or required approval."
    return {
        "effect": effect,
        "reason": reason,
        "policy_count": result.metadata.get("policy_count", len(policies)),
        "applicable_policy_count": result.metadata.get("applicable_policy_count"),
        "evaluated_policy_identities": [
            item["policy_identity"] for item in runtime_facts.get("policies", ())
        ],
        "decision_policy_identities": [
            decision.policy_identity for decision in result.decisions
        ],
        "runtime_facts": {
            "runtime_id": runtime_facts.get("runtime_id"),
            "runtime_kind": runtime_facts.get("runtime_kind"),
            "stage": runtime_facts.get("stage"),
            "operation_type": runtime_facts.get("operation_type"),
            "decision_count": runtime_facts.get("decision_count"),
            "result": runtime_facts.get("result"),
        },
    }


def _request_summary(request: dict[str, Any]) -> dict[str, Any]:
    prompt = request.get("prompt")
    input_payload = (
        request.get("input") if isinstance(request.get("input"), dict) else {}
    )
    metadata = (
        request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
    )
    return {
        "has_prompt": bool(prompt),
        "prompt_hash": _stable_hash(prompt) if prompt else None,
        "user_id": request.get("user_id"),
        "session_id": request.get("session_id"),
        "source_scope": list(_source_scope_from_value(request.get("source_scope"))),
        "mode": request.get("mode"),
        "requested_capabilities": list(request.get("requested_capabilities") or ()),
        "constraint_keys": sorted(
            str(key) for key in dict(request.get("constraints") or {})
        ),
        "metadata_keys": sorted(str(key) for key in metadata),
        "input_keys": sorted(str(key) for key in input_payload),
        "capability_id": request.get("capability_id"),
        "capability_owner": request.get("capability_owner"),
        "governance_stage": request.get("governance_stage"),
    }


def _task_input_summary(input: dict[str, Any]) -> dict[str, Any]:
    sql = input.get("sql")
    prompt = input.get("prompt")
    query = input.get("query")
    return {
        "keys": sorted(str(key) for key in input),
        "input_hash": input.get("input_hash") or _stable_hash(input),
        "sql_hash": _stable_hash(sql) if sql else None,
        "prompt_hash": _stable_hash(prompt) if prompt else None,
        "query_hash": _stable_hash(query) if query else None,
        "sql_ref": input.get("sql_ref"),
        "validated_evidence_id": input.get("validated_evidence_id"),
        "operation": input.get("operation"),
    }


def _approval_action_summary(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_type": action.get("operation_type"),
        "approval": action.get("approval"),
        "keys": sorted(str(key) for key in action),
        "request": (
            _request_summary(action["request"])
            if isinstance(action.get("request"), dict)
            else None
        ),
        "contract_keys": sorted(str(key) for key in dict(action.get("contract") or {})),
    }


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    safe_values = {}
    for key in ("runtime_id", "intent_kind", "access", "governance_stage"):
        if key in metadata:
            safe_values[key] = metadata[key]
    return {
        **safe_values,
        "keys": sorted(str(key) for key in metadata),
    }


def _actor_context(operation: Operation) -> dict[str, Any]:
    request = operation.request
    metadata = (
        request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
    )
    input_payload = (
        request.get("input") if isinstance(request.get("input"), dict) else {}
    )
    actor = _dict_without_none(
        {
            "user_id": request.get("user_id")
            or input_payload.get("user_id")
            or metadata.get("user_id"),
            "session_id": request.get("session_id") or metadata.get("session_id"),
            "actor_id": request.get("actor_id")
            or input_payload.get("actor_id")
            or metadata.get("actor_id"),
            "actor_type": request.get("actor_type")
            or input_payload.get("actor_type")
            or metadata.get("actor_type"),
        }
    )
    nested_actor = metadata.get("actor") or input_payload.get("actor")
    if isinstance(nested_actor, dict):
        actor.update(nested_actor)
    return actor


def _tenant_context(operation: Operation) -> dict[str, Any]:
    request = operation.request
    metadata = (
        request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
    )
    input_payload = (
        request.get("input") if isinstance(request.get("input"), dict) else {}
    )
    tenant = _dict_without_none(
        {
            "tenant_id": request.get("tenant_id")
            or input_payload.get("tenant_id")
            or metadata.get("tenant_id")
            or operation.metadata.get("tenant_id"),
            "workspace_id": request.get("workspace_id")
            or input_payload.get("workspace_id")
            or metadata.get("workspace_id")
            or operation.metadata.get("workspace_id"),
        }
    )
    nested_tenant = metadata.get("tenant") or input_payload.get("tenant")
    if isinstance(nested_tenant, dict):
        tenant.update(nested_tenant)
    return tenant


def _source_scope_context(operation: Operation) -> tuple[str, ...]:
    value = operation.request.get("source_scope")
    if value is None and isinstance(operation.request.get("metadata"), dict):
        value = operation.request["metadata"].get("source_scope")
    if value is None:
        value = operation.metadata.get("source_scope")
    return _source_scope_from_value(value)


def _source_scope_from_value(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if item is not None)
    return ()


def _resource_context(
    operation: Operation,
    source: Any,
    capability: Capability | None,
) -> dict[str, Any]:
    resource = {
        "source_type": type(source).__name__ if source is not None else "none",
        "source_repr": _safe_source_repr(source),
        "operation_type": operation.operation_type,
    }
    if capability is not None:
        resource["capability"] = {
            "id": capability.id,
            "owner": capability.owner,
            "access": capability.access.value,
            "risk": capability.risk.value,
        }
    return resource


def _operation_trace_context(operation: Operation) -> dict[str, Any]:
    return {
        "id": operation.id,
        "operation_type": operation.operation_type,
        "status": operation.status.value,
        "required_evidence": sorted(operation.required_evidence),
        "request": _request_summary(operation.request),
        "metadata": _metadata_summary(operation.metadata),
    }


def _task_trace_context(task: Task | None) -> dict[str, Any]:
    if task is None:
        return {}
    return {
        "id": task.id,
        "operation_id": task.operation_id,
        "capability_id": task.capability_id,
        "executor_id": task.executor_id,
        "status": task.status.value,
        "input": _task_input_summary(task.input),
        "required_evidence": sorted(task.required_evidence),
        "dependencies": [dependency.to_dict() for dependency in task.dependencies],
        "metadata": _metadata_summary(task.metadata),
    }


def _approval_ids_for_decision(
    decision: Any,
    result: GovernanceResult,
    approvals_to_request: tuple[ApprovalRequest, ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    for request in (*result.approval_requests, *approvals_to_request):
        if (
            request.requested_by_policy_id == decision.policy_id
            and request.owner == decision.owner
            and request.approval_id not in ids
        ):
            ids.append(request.approval_id)
    for approval_id in result.metadata.get("approval_statuses") or {}:
        if (
            f":{decision.policy_id}:" in str(approval_id)
            and str(approval_id) not in ids
        ):
            ids.append(str(approval_id))
    return tuple(ids)


def _dict_without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _governance_fact_envelope(
    operation: Operation,
    *,
    contract: DbOperationContract | None,
    task: Task | None,
    capability: Capability | None,
    stage: str,
    source: Any,
    evidence: tuple[Evidence, ...],
    authoritative_validation_evidence: tuple[Evidence, ...],
    approvals: tuple[ApprovalRequest, ...],
) -> dict[str, Any]:
    planned = _planned_governance_facts(operation, contract, capability)
    validation_context = _sql_validation_governance_facts(evidence)
    authoritative_validation = _sql_validation_governance_facts(
        authoritative_validation_evidence
    )
    task_facts = _task_governance_facts(task)
    approval_facts = _approval_governance_facts(approvals)
    sources = ["runtime"]
    if planned.get("has_planned_facts"):
        sources.append("planning")
    if validation_context.get("evidence_ids"):
        sources.append("sql_validation")
    if authoritative_validation.get("evidence_ids"):
        sources.append("task_dependency")
    if capability is not None:
        sources.append("capability")
    if task is not None:
        sources.append("task")
    if approvals:
        sources.append("approval_store")
    return {
        "version": "15.10",
        "fact_source": {
            "sources": sources,
            "planned": planned.get("has_planned_facts", False),
            "validated": bool(validation_context.get("evidence_ids")),
            "authoritative_validation": bool(
                authoritative_validation.get("evidence_ids")
            ),
            "task": task is not None,
            "capability": capability is not None,
            "approvals": bool(approvals),
        },
        "stage": stage,
        "authoritative": {
            "source": _authoritative_fact_source(
                stage=stage,
                task=task,
                authoritative_validation=authoritative_validation,
                planned=planned,
            ),
            "operation": planned,
            "capability": (
                _capability_governance_facts(capability)
                if capability is not None
                else {}
            ),
            "task": task_facts,
            "validation": authoritative_validation,
        },
        "context": {
            "validation": {
                "operation_wide": validation_context,
            },
        },
        "operation": planned,
        "capability": (
            _capability_governance_facts(capability) if capability is not None else {}
        ),
        "task": task_facts,
        "validation": authoritative_validation,
        "approvals": approval_facts,
        "actor": _actor_context(operation),
        "tenant": _tenant_context(operation),
        "source_scope": list(_source_scope_context(operation)),
        "resource": _resource_context(operation, source, capability),
    }


def _planned_governance_facts(
    operation: Operation,
    contract: DbOperationContract | None,
    capability: Capability | None,
) -> dict[str, Any]:
    contract_metadata = contract.metadata if contract is not None else {}
    planned = dict(contract_metadata.get("planned_operation") or {})
    access = (
        contract.access.value
        if contract is not None
        else operation.metadata.get("access")
    )
    if access is None and capability is not None:
        access = capability.access.value
    selected = contract_metadata.get("selected_capabilities") or ()
    side_effecting = (
        bool(capability.side_effecting) if capability is not None else False
    )
    side_effecting = side_effecting or any(
        bool(item.get("side_effecting")) for item in selected if isinstance(item, dict)
    )
    operation_type = (
        contract.operation_type if contract is not None else operation.operation_type
    )
    admin = bool(planned.get("admin")) or access == AccessMode.ADMIN.value
    destructive = bool(planned.get("destructive"))
    write_or_admin = (
        operation_type in {"write.propose", "write.execute", "admin"}
        or access in {AccessMode.WRITE.value, AccessMode.ADMIN.value}
        or side_effecting
    )
    return {
        **planned,
        "has_planned_facts": bool(planned or contract is not None or capability),
        "operation_id": operation.id,
        "operation_type": operation_type,
        "access": access,
        "intent_kind": operation.metadata.get("intent_kind")
        or planned.get("intent_kind"),
        "required_evidence": sorted(operation.required_evidence),
        "capability_ids": (
            list(contract.required_capabilities)
            if contract is not None
            else ([capability.id] if capability is not None else [])
        ),
        "admin": admin,
        "destructive": destructive,
        "write_or_admin_context": write_or_admin,
        "side_effecting": side_effecting,
    }


def _authoritative_fact_source(
    *,
    stage: str,
    task: Task | None,
    authoritative_validation: dict[str, Any],
    planned: dict[str, Any],
) -> str:
    if task is not None and authoritative_validation.get("evidence_ids"):
        return "task_dependency"
    if planned.get("has_planned_facts"):
        return "planning"
    return stage


def _task_governance_facts(task: Task | None) -> dict[str, Any]:
    if task is None:
        return {}
    return {
        "id": task.id,
        "operation_id": task.operation_id,
        "capability_id": task.capability_id,
        "executor_id": task.executor_id,
        "status": task.status.value,
        "required_evidence": sorted(task.required_evidence),
        "input": _task_input_summary(task.input),
        "dependencies": [
            {
                "kind": dependency.kind.value,
                "evidence_kind": dependency.evidence_kind,
                "producer_task_id": dependency.producer_task_id,
                "producer_capability_id": dependency.producer_capability_id,
                "producer_executor_id": dependency.producer_executor_id,
                "approval_policy_id": dependency.approval_policy_id,
                "approval_name": dependency.approval_name,
                "approval_status": (
                    dependency.approval_status.value
                    if dependency.approval_status is not None
                    else None
                ),
                "payload_fingerprint": dependency.payload_fingerprint,
                "evidence_payload_keys": sorted(
                    str(key) for key in dependency.evidence_payload
                ),
            }
            for dependency in task.dependencies
        ],
        "metadata": _metadata_summary(task.metadata),
    }


def _approval_governance_facts(
    approvals: tuple[ApprovalRequest, ...],
) -> dict[str, Any]:
    return {
        "ids": [approval.approval_id for approval in approvals],
        "pending_ids": [
            approval.approval_id
            for approval in approvals
            if approval.status is ApprovalStatus.PENDING
        ],
        "approved_ids": [
            approval.approval_id
            for approval in approvals
            if approval.status is ApprovalStatus.APPROVED
        ],
        "statuses": {
            approval.approval_id: approval.status.value for approval in approvals
        },
        "policy_ids": sorted(
            {
                approval.requested_by_policy_id
                for approval in approvals
                if approval.requested_by_policy_id
            }
        ),
    }


def _sql_validation_governance_facts(
    evidence: tuple[Evidence, ...],
) -> dict[str, Any]:
    statements: list[dict[str, Any]] = []
    for item in evidence:
        if item.kind != "sql.validation" or not item.accepted:
            continue
        payload = item.payload
        raw_facts = payload.get("statement_facts")
        facts = raw_facts if isinstance(raw_facts, dict) else {}
        statement_type = str(
            facts.get("statement_type") or payload.get("statement_type") or ""
        ).upper()
        mutating = _statement_classes(
            facts.get("mutating_statement_classes")
            or facts.get("mutating_statement_types")
            or payload.get("mutating_statement_classes")
            or payload.get("mutating_statement_types")
            or ()
        )
        destructive = _statement_classes(
            facts.get("destructive_statement_classes")
            or payload.get("destructive_statement_classes")
            or ()
        )
        admin = _statement_classes(
            facts.get("admin_statement_classes")
            or payload.get("admin_statement_classes")
            or ()
        )
        if statement_type:
            if statement_type in {
                "DELETE",
                "DROP",
                "ALTER",
                "TRUNCATETABLE",
                "TRUNCATE",
            }:
                destructive = _ordered_unique((*destructive, statement_type))
            if statement_type in {
                "CREATE",
                "DROP",
                "ALTER",
                "TRUNCATETABLE",
                "TRUNCATE",
            }:
                admin = _ordered_unique((*admin, statement_type))
        target_resources = _safe_string_list(
            facts.get("target_resources")
            or payload.get("target_resources")
            or payload.get("tables")
            or payload.get("referenced_tables")
            or ()
        )
        statements.append(
            {
                "evidence_id": item.id,
                "task_id": item.task_id,
                "owner": item.owner,
                "valid": bool(payload.get("valid") or payload.get("ok")),
                "statement_type": statement_type.lower() if statement_type else None,
                "statement_count": facts.get("statement_count")
                or payload.get("statement_count"),
                "is_read": facts.get("is_read", payload.get("is_read")),
                "mutating_statement_classes": list(mutating),
                "destructive_statement_classes": list(destructive),
                "admin_statement_classes": list(admin),
                "target_resources": list(target_resources),
                "guardrail_result": facts.get("guardrail_result")
                or payload.get("guardrail_result")
                or ("passed" if payload.get("valid") or payload.get("ok") else None),
                "sql_fingerprint": facts.get("sql_fingerprint")
                or payload.get("sql_fingerprint"),
            }
        )
    return {
        "evidence_ids": [
            item["evidence_id"] for item in statements if item["evidence_id"]
        ],
        "task_ids": [item["task_id"] for item in statements if item["task_id"]],
        "statement_types": _ordered_unique(
            item["statement_type"] for item in statements if item["statement_type"]
        ),
        "mutating_statement_classes": _ordered_unique(
            cls for item in statements for cls in item["mutating_statement_classes"]
        ),
        "destructive_statement_classes": _ordered_unique(
            cls for item in statements for cls in item["destructive_statement_classes"]
        ),
        "admin_statement_classes": _ordered_unique(
            cls for item in statements for cls in item["admin_statement_classes"]
        ),
        "target_resources": _ordered_unique(
            resource for item in statements for resource in item["target_resources"]
        ),
        "guardrail_results": _ordered_unique(
            item["guardrail_result"] for item in statements if item["guardrail_result"]
        ),
        "sql_fingerprints": _ordered_unique(
            item["sql_fingerprint"] for item in statements if item["sql_fingerprint"]
        ),
        "statements": statements,
    }


def _statement_classes(values: Any) -> tuple[str, ...]:
    return tuple(
        str(value).upper() for value in _safe_string_list(values) if str(value).strip()
    )


def _ordered_unique(values: Any) -> tuple[Any, ...]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _safe_string_list(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, (list, tuple, set, frozenset)):
        return tuple(str(value) for value in values if value is not None)
    return (str(values),)


def _operation_for_governance(
    operation: Operation,
    *,
    task: Task | None,
    capability: Capability | None,
    stage: str,
    governance_facts: dict[str, Any],
) -> Operation:
    request = dict(operation.request)
    request["governance_stage"] = stage
    request["governance_facts"] = governance_facts
    operation_override = _governance_operation_override(governance_facts)
    override_access = operation_override.get("access")
    if override_access:
        request["access"] = str(override_access)
        request_metadata = request.get("metadata")
        if isinstance(request_metadata, dict):
            request["metadata"] = {
                **request_metadata,
                "access": str(override_access),
            }
    if task is not None:
        request["task"] = {
            "id": task.id,
            "capability_id": task.capability_id,
            "executor_id": task.executor_id,
            "input": _task_input_summary(task.input),
            "metadata": task.metadata,
        }
    if capability is not None:
        request["capability"] = _capability_governance_facts(capability)
    metadata = {
        **operation.metadata,
        "governance_stage": stage,
    }
    if override_access:
        metadata["access"] = str(override_access)
    if task is not None:
        metadata["task_id"] = task.id
    if capability is not None:
        metadata["capability_id"] = capability.id
        metadata["capability_owner"] = capability.owner
    operation_type = str(
        operation_override.get("operation_type") or operation.operation_type
    )
    return replace(
        operation,
        operation_type=operation_type,
        request=request,
        metadata=metadata,
    )


def _governance_contract(
    operation: Operation,
    *,
    contract: DbOperationContract | None,
    task: Task | None,
    capability: Capability | None,
    stage: str,
    governance_facts: dict[str, Any],
) -> dict[str, Any]:
    if contract is not None:
        shaped: dict[str, Any] = {
            "operation_type": contract.operation_type,
            "required_capabilities": list(contract.required_capabilities),
            "required_evidence": list(contract.required_evidence),
            "access": contract.access.value,
            "policy_ids": list(contract.policy_ids),
            "metadata": contract.metadata,
        }
    else:
        shaped = {
            "operation_type": operation.operation_type,
            "required_capabilities": [],
            "required_evidence": sorted(operation.required_evidence),
            "metadata": operation.metadata,
        }
    shaped["governance_stage"] = stage
    shaped["governance_facts"] = governance_facts
    operation_override = _governance_operation_override(governance_facts)
    if operation_override.get("operation_type"):
        shaped["operation_type"] = str(operation_override["operation_type"])
    if operation_override.get("access"):
        shaped["access"] = str(operation_override["access"])
    if task is not None:
        shaped["task"] = {
            "id": task.id,
            "capability_id": task.capability_id,
            "executor_id": task.executor_id,
            "required_evidence": sorted(task.required_evidence),
            "metadata": task.metadata,
        }
    if capability is not None:
        shaped["capability"] = _capability_governance_facts(capability)
    return shaped


def _governance_operation_override(
    governance_facts: dict[str, Any],
) -> dict[str, Any]:
    override = governance_facts.get("operation_override")
    return dict(override) if isinstance(override, dict) else {}


def _capability_governance_facts(capability: Capability) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "domains": sorted(capability.domains),
        "operation_types": sorted(capability.operation_types),
        "access": capability.access.value,
        "risk": capability.risk.value,
        "runtime_only": capability.runtime_only,
        "side_effecting": capability.side_effecting,
        "executor": capability.executor,
        "output_evidence": sorted(capability.output_evidence),
    }


def _prompt_from_direct_input(input: dict[str, Any]) -> str:
    for key in ("prompt", "sql", "query", "content"):
        value = input.get(key)
        if value:
            return str(value)
    return ""


def _current_trace_ids() -> tuple[str | None, str | None]:
    try:
        from daita.core.tracing import get_trace_manager

        context = get_trace_manager().trace_context
        return context.current_trace_id, context.current_span_id
    except Exception:
        return None, None


def _governance_blocked_answer(governance: GovernanceResult) -> str:
    if governance.blocked:
        return "This operation was denied by governance policy."
    if governance.pending_approval:
        return "This operation requires approval before execution."
    return "This operation was blocked by governance policy."


def _governance_policy_block_reason(governance: GovernanceResult) -> str | None:
    for decision in governance.decisions:
        if decision.effect is PolicyEffect.DENY:
            return decision.policy_id
    if governance.blocked:
        return "governance_blocked"
    if governance.pending_approval:
        return "governance_approval_required"
    return None


def _governance_blocked_warning(governance: GovernanceResult) -> str:
    if governance.blocked:
        return "db_runtime_governance_denied"
    if governance.pending_approval:
        return "db_runtime_approval_required"
    return "db_runtime_governance_blocked"


def _safe_source_repr(source: Any) -> str:
    if source is None:
        return "none"
    if not isinstance(source, str):
        return type(source).__name__
    try:
        parts = urlsplit(source)
    except ValueError:
        return "<source>"
    if not parts.scheme:
        return source
    netloc = parts.netloc
    if "@" in netloc and ":" in netloc.split("@", 1)[0]:
        credentials, host = netloc.rsplit("@", 1)
        user = credentials.split(":", 1)[0]
        netloc = f"{user}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _schema_cache_ttl(metadata: dict[str, Any]) -> float | None:
    options = _from_db_options(metadata)
    if "cache_ttl" in options:
        value = options.get("cache_ttl")
        return None if value is None else float(value)
    return None


def _source_schema_fingerprint(schema: dict[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _analysis_replan_enabled(metadata: dict[str, Any]) -> bool:
    return bool(_from_db_options(metadata).get("analysis_replan_enabled"))


def _analysis_parallel_enabled(metadata: dict[str, Any]) -> bool:
    return bool(_from_db_options(metadata).get("analysis_parallel_enabled"))


def _analysis_max_concurrency(metadata: dict[str, Any]) -> int:
    value = _from_db_options(metadata).get("analysis_max_concurrency", 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}


def _default_monitor_store(store: RuntimeStore) -> DbMonitorStore:
    if isinstance(store, SQLiteRuntimeStore):
        return SQLiteDbMonitorStore(store.path)
    return InMemoryDbMonitorStore(store)
