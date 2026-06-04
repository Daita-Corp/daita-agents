"""
Skeleton database runtime built on the extension registry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import time
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from daita.plugins import ExtensionRegistry, PluginContext, ServiceRegistry
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
    Task,
    TaskDependency,
    TaskStatus,
)

from .execution import DbOperationExecutor
from .governance import default_db_policies
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
from .planning import DbContractBuilder, DbIntentClassifier
from .synthesis import DbSynthesizer
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
        approval_channel: InMemoryApprovalChannel | None = None,
        runtime_id: str | None = None,
    ) -> None:
        self.source = source
        self.config = config or DbRuntimeConfig()
        self.registry = registry or ExtensionRegistry()
        self.store = store or InMemoryRuntimeStore()
        self.approval_channel = approval_channel or InMemoryApprovalChannel(self.store)
        self.runtime_id = runtime_id or f"db-runtime-{uuid4()}"
        self.intent_classifier = DbIntentClassifier()
        self.verifier = DbVerifier()
        self.synthesizer = DbSynthesizer()
        self._setup_context: PluginContext | None = None
        self._is_setup = False
        self._schema_profile_cache: dict[str, Any] | None = None
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
            plugin_ids=self.registry.plugin_ids,
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
            await self._persist_planned_task(task)
        elif (
            stored_task.status is TaskStatus.PENDING and stored_task.input != task.input
        ):
            task = replace(stored_task, input=task.input)
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
        validation_capability = self._validation_capability_for_write(capability)
        if (
            capability.id == "db.sql.execute_write"
            and validation_capability is not None
        ):
            output_evidence = frozenset(
                (
                    *sorted(validation_capability.output_evidence),
                    *sorted(output_evidence),
                )
            )
        operation = Operation(
            id=operation_id or f"db-op-{uuid4()}",
            operation_type=operation_type,
            status=OperationStatus.RUNNING,
            request={
                "prompt": _prompt_from_direct_input(input or {}),
                "input": input or {},
                "capability_id": capability.id,
                "capability_owner": capability.owner,
            },
            required_evidence=output_evidence,
            metadata={
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                "direct_capability_id": capability.id,
                "direct_capability_owner": capability.owner,
            },
        )
        await self.store.save_operation(operation)
        await self.store.append_event(
            self._runtime_event(
                type=RuntimeEventType.OPERATION_CREATED,
                operation_id=operation.id,
                message=f"Operation {operation.id} created.",
            )
        )
        tasks = self._direct_capability_tasks(
            operation,
            capability,
            input or {},
            validation_capability=validation_capability,
        )
        for task in tasks:
            await self._persist_planned_task(task)
        primary_task = tasks[-1]
        if (
            capability.id == "db.sql.execute_write"
            and validation_capability is None
            and not (input or {}).get("validated_evidence_id")
        ):
            blocked_task = replace(primary_task, status=TaskStatus.BLOCKED)
            await self._persist_pre_execution_task_block(
                operation=replace(operation, status=OperationStatus.BLOCKED),
                task=blocked_task,
                events=(
                    self._runtime_event(
                        type=RuntimeEventType.OPERATION_UPDATED,
                        operation_id=operation.id,
                        message=(
                            "Direct write execution requires db.sql.validate "
                            "or a validated_evidence_id."
                        ),
                    ),
                ),
            )
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                "Direct write execution requires db.sql.validate or validated_evidence_id.",
            )
        governance_persistence = await self._evaluate_governance(
            operation,
            task=None,
            capability=capability,
            stage="operation",
        )
        governance = governance_persistence.result
        if governance.blocked or governance.pending_approval:
            blocked_task = replace(primary_task, status=TaskStatus.BLOCKED)
            await self.store.commit_governance_blocked(
                operation=replace(operation, status=OperationStatus.BLOCKED),
                task=blocked_task,
                decisions=governance.decisions,
                audit_record=governance_persistence.audit_record,
                approval_requests=governance_persistence.approvals_to_request,
                events=(
                    *governance_persistence.events,
                    self._runtime_event(
                        type=RuntimeEventType.OPERATION_UPDATED,
                        operation_id=operation.id,
                        message=(
                            f"Operation {operation.id} blocked by governance policy."
                        ),
                        payload={"governance": governance.to_dict()},
                    ),
                ),
            )
            raise DbRuntimeGovernanceBlocked(
                operation=operation,
                task=None,
                governance=governance,
            )
        await self.store.commit_governance_evaluation(
            decisions=governance.decisions,
            audit_record=governance_persistence.audit_record,
            approval_requests=governance_persistence.approvals_to_request,
            events=governance_persistence.events,
        )
        try:
            collected: list[Evidence] = []
            for task in tasks:
                collected.extend(await self.execute_task(task, operation))
            evidence = tuple(collected)
        except DbRuntimeGovernanceBlocked:
            await self.store.save_operation(
                replace(operation, status=OperationStatus.BLOCKED)
            )
            raise
        except Exception:
            await self.store.save_operation(
                replace(operation, status=OperationStatus.FAILED)
            )
            raise
        await self.store.save_operation(
            replace(operation, status=OperationStatus.SUCCEEDED)
        )
        await self.store.append_event(
            self._runtime_event(
                type=RuntimeEventType.OPERATION_UPDATED,
                operation_id=operation.id,
                message=f"Operation {operation.id} succeeded.",
            )
        )
        return evidence

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

    async def resume_operation(self, operation_id: str) -> OperationSnapshot:
        """Resume persisted operation state without re-running completed tasks."""
        snapshot = await self.inspect_operation(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        await self.store.append_event(
            self._runtime_event(
                type=RuntimeEventType.OPERATION_RESUMED,
                operation_id=operation_id,
                message=f"Operation {operation_id} resume requested.",
                payload={
                    "completed_task_ids": list(snapshot.completed_task_ids),
                    "resumable_task_ids": list(snapshot.resumable_task_ids),
                },
            )
        )
        for task in snapshot.tasks:
            if task.id in snapshot.completed_task_ids:
                await self.store.append_event(
                    self._runtime_event(
                        type=RuntimeEventType.TASK_SKIPPED,
                        operation_id=operation_id,
                        task_id=task.id,
                        task=task,
                        message=(f"Task {task.id} already completed; not re-running."),
                    )
                )

        terminal_snapshot = await self._apply_terminal_approval_state(snapshot)
        if terminal_snapshot is not None:
            return terminal_snapshot

        if self._has_pending_approvals(snapshot):
            await self.store.save_operation(
                replace(snapshot.operation, status=OperationStatus.BLOCKED)
            )
            resumed = await self.inspect_operation(operation_id)
            if resumed is None:
                raise KeyError(operation_id)
            return resumed

        snapshot = await self._recover_expired_task_claims(snapshot)

        resumable_tasks = tuple(
            task
            for task in _tasks_in_resume_order(snapshot.tasks)
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
            and not task.metadata.get("manual_recovery_required")
        )
        operation = snapshot.operation
        if resumable_tasks:
            operation = replace(snapshot.operation, status=OperationStatus.RUNNING)
            await self.store.save_operation(operation)
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
            except Exception:
                await self.store.save_operation(
                    replace(operation, status=OperationStatus.FAILED)
                )
                raise

        completed = await self.inspect_operation(operation_id)
        if completed is None:
            raise KeyError(operation_id)
        if _has_running_tasks(completed.tasks):
            return completed
        if completed.resumable_task_ids:
            return completed

        if _operation_has_run_context(completed.operation):
            await self._complete_resumed_run_operation(completed)
        elif (
            completed.tasks
            and completed.operation.status is not OperationStatus.SUCCEEDED
        ):
            await self.store.save_operation(
                replace(completed.operation, status=OperationStatus.SUCCEEDED)
            )
            await self.store.append_event(
                self._runtime_event(
                    type=RuntimeEventType.OPERATION_UPDATED,
                    operation_id=operation_id,
                    message=f"Operation {operation_id} succeeded after resume.",
                )
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

    def classify_request(self, request: DbRequest | str) -> DbIntent:
        """Classify a prompt into a DB intent."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        return self.intent_classifier.classify(db_request)

    def build_contract(
        self,
        request: DbRequest | str,
        intent: DbIntent | None = None,
    ) -> DbOperationContract:
        """Build a structured operation contract from registry capabilities."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        resolved_intent = intent or self.classify_request(db_request)
        return DbContractBuilder(self.registry, self.config).build(
            db_request, resolved_intent
        )

    async def run(self, request: DbRequest | str) -> DbOperationResult:
        """Plan and execute a DB operation through typed runtime capabilities."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        if not self._is_setup:
            await self.setup()
        intent = self.classify_request(db_request)
        contract = self.build_contract(db_request, intent)
        operation_id = f"db-op-{uuid4()}"
        operation = Operation(
            id=operation_id,
            operation_type=contract.operation_type,
            status=OperationStatus.RUNNING,
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
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                "intent_kind": intent.kind.value,
                "access": contract.access.value,
                "resume_context": {
                    "request": _db_request_context(db_request),
                    "intent": _db_intent_context(intent),
                    "contract": _db_contract_context(contract),
                },
            },
        )
        await self.store.save_operation(operation)
        await self.store.append_event(
            self._runtime_event(
                type=RuntimeEventType.OPERATION_CREATED,
                operation_id=operation.id,
                message=f"Operation {operation.id} created.",
                payload={"operation_type": operation.operation_type},
            )
        )

        base_diagnostics = {
            "runtime_id": self.runtime_id,
            "registered_plugins": list(self.registry.plugin_ids),
            "contract": contract.metadata,
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

        await self._persist_contract_tasks(operation, contract)
        governance_persistence = await self._evaluate_governance(
            operation,
            contract=contract,
            stage="operation",
        )
        governance = governance_persistence.result
        base_diagnostics = {
            **base_diagnostics,
            "governance": governance.to_dict(),
        }
        if governance.blocked:
            await self._commit_operation_governance_blocked(
                operation,
                governance_persistence,
                governance=governance,
            )
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
            )
        if governance.pending_approval:
            await self._commit_operation_governance_blocked(
                operation,
                governance_persistence,
                governance=governance,
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
            )
        await self.store.commit_governance_evaluation(
            decisions=governance.decisions,
            audit_record=governance_persistence.audit_record,
            approval_requests=governance_persistence.approvals_to_request,
            events=governance_persistence.events,
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

        synthesis = self.synthesizer.synthesize(
            db_request, intent, contract, outcome.evidence, verification
        )
        return await self._record_operation_result(
            DbOperationResult(
                operation_id=operation_id,
                request=db_request,
                intent=intent,
                contract=contract,
                status=OperationStatus.SUCCEEDED,
                answer=synthesis.answer,
                evidence=outcome.evidence,
                warnings=tuple((*outcome.warnings, *synthesis.warnings)),
                diagnostics={
                    **base_diagnostics,
                    "execution": {
                        **outcome.diagnostics,
                        "task_count": len(outcome.tasks),
                        "tasks": [task.to_dict() for task in outcome.tasks],
                    },
                    "verification": verification.to_dict(),
                    "synthesis": synthesis.to_dict(),
                },
            ),
            operation=operation,
        )

    async def _persist_planned_task(self, task: Task) -> None:
        await self.store.save_task(replace(task, status=TaskStatus.PENDING))
        await self.store.append_event(
            self._runtime_event(
                type=RuntimeEventType.TASK_CREATED,
                operation_id=task.operation_id,
                task_id=task.id,
                task=task,
                message=f"Task {task.id} planned.",
                payload={
                    "capability_id": task.capability_id,
                    "executor_id": task.executor_id,
                    "input": task.input,
                    "required_evidence": sorted(task.required_evidence),
                    "metadata": task.metadata,
                },
            )
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
            await self._persist_planned_task(task)
            existing.add(key)
            planned_by_capability[(capability.id, capability.owner)] = task

    async def _planned_task_for_capability(
        self,
        operation_id: str,
        capability: Capability,
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
            return task
        return None

    def _validation_capability_for_write(
        self,
        capability: Capability,
    ) -> Capability | None:
        if capability.id != "db.sql.execute_write":
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
        if capability.id != "db.sql.execute_write" or validation_capability is None:
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
                "operation": operation.operation_type,
            },
            reason="direct_validation",
            sequence=1,
        )
        write_task = self._task_for_capability(
            operation,
            capability,
            input={"sql_ref": "sql.validation"},
            reason="direct",
            sequence=2,
            validation_task=validation_task,
        )
        return (validation_task, write_task)

    def _task_for_capability(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input: dict[str, Any],
        reason: str,
        sequence: int,
        validation_task: Task | None = None,
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
        if task.capability_id != "db.sql.execute_write":
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
        if task is None or task.capability_id != "db.sql.execute_write":
            return ()
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

    async def _persist_pre_execution_task_block(
        self,
        *,
        operation: Operation | None,
        task: Task,
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        commit = getattr(self.store, "commit_task_blocked", None)
        if commit is not None:
            await commit(operation=operation, task=task, events=events)
            return
        if operation is not None:
            await self.store.save_operation(operation)
        await self.store.save_task(task)
        for event in events:
            await self.store.append_event(event)

    async def _recover_expired_task_claims(
        self,
        snapshot: OperationSnapshot,
    ) -> OperationSnapshot:
        now = time.time()
        changed = False
        for task in snapshot.tasks:
            if task.status is not TaskStatus.RUNNING:
                continue
            lease_expires_at = task.metadata.get("lease_expires_at")
            if lease_expires_at is None or float(lease_expires_at) > now:
                continue
            capability = self._capability_for_task(task)
            if (
                capability.idempotent
                or capability.replay_safe
                or not capability.side_effecting
            ):
                await self.store.save_task(
                    replace(
                        task,
                        status=TaskStatus.PENDING,
                        metadata={
                            **task.metadata,
                            "lease_owner": None,
                            "lease_expires_at": None,
                            "expired_lease_recovered": True,
                        },
                    )
                )
            else:
                await self.store.save_operation(
                    replace(snapshot.operation, status=OperationStatus.BLOCKED)
                )
                await self.store.save_task(
                    replace(
                        task,
                        status=TaskStatus.BLOCKED,
                        metadata={
                            **task.metadata,
                            "manual_recovery_required": True,
                            "manual_recovery_reason": "expired_side_effecting_lease",
                        },
                    )
                )
                await self.store.append_event(
                    self._runtime_event(
                        type=RuntimeEventType.TASK_UPDATED,
                        operation_id=snapshot.operation.id,
                        task_id=task.id,
                        task=task,
                        message=(
                            f"Task {task.id} lease expired; manual recovery required "
                            "before replaying side-effecting work."
                        ),
                    )
                )
            changed = True
        if not changed:
            return snapshot
        recovered = await self.inspect_operation(snapshot.operation.id)
        if recovered is None:
            raise KeyError(snapshot.operation.id)
        return recovered

    async def _apply_terminal_approval_state(
        self,
        snapshot: OperationSnapshot,
    ) -> OperationSnapshot | None:
        statuses = {request.status for request in snapshot.approval_requests}
        if ApprovalStatus.REJECTED in statuses:
            await self.store.save_operation(
                replace(snapshot.operation, status=OperationStatus.FAILED)
            )
            message = (
                f"Operation {snapshot.operation.id} failed because approval was "
                "rejected."
            )
        elif ApprovalStatus.CANCELLED in statuses:
            await self.store.save_operation(
                replace(snapshot.operation, status=OperationStatus.CANCELLED)
            )
            message = f"Operation {snapshot.operation.id} cancelled by approval state."
        elif ApprovalStatus.EXPIRED in statuses:
            await self.store.save_operation(
                replace(snapshot.operation, status=OperationStatus.BLOCKED)
            )
            message = (
                f"Operation {snapshot.operation.id} remains blocked because "
                "approval expired."
            )
        else:
            return None

        await self.store.append_event(
            self._runtime_event(
                type=RuntimeEventType.OPERATION_UPDATED,
                operation_id=snapshot.operation.id,
                message=message,
            )
        )
        updated = await self.inspect_operation(snapshot.operation.id)
        if updated is None:
            raise KeyError(snapshot.operation.id)
        return updated

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

        synthesis = self.synthesizer.synthesize(
            request, intent, contract, evidence, verification
        )
        await self._record_operation_result(
            DbOperationResult(
                operation_id=snapshot.operation.id,
                request=request,
                intent=intent,
                contract=contract,
                status=OperationStatus.SUCCEEDED,
                answer=synthesis.answer,
                evidence=evidence,
                diagnostics={
                    "verification": verification.to_dict(),
                    "synthesis": synthesis.to_dict(),
                    "execution": {
                        "task_count": len(tasks),
                        "tasks": [task.to_dict() for task in tasks],
                    },
                },
            ),
            operation=snapshot.operation,
        )

    async def _enforce_governance(
        self,
        operation: Operation,
        contract: DbOperationContract | None = None,
        *,
        task: Task | None = None,
        capability: Capability | None = None,
        stage: str,
    ) -> GovernanceResult:
        governance = await self._evaluate_governance(
            operation,
            contract=contract,
            task=task,
            capability=capability,
            stage=stage,
        )
        if not governance.result.blocked and not governance.result.pending_approval:
            await self.store.commit_governance_evaluation(
                decisions=governance.result.decisions,
                audit_record=governance.audit_record,
                approval_requests=governance.approvals_to_request,
                events=governance.events,
            )
        return governance.result

    async def evaluate_governance_persistence(
        self,
        operation: Operation,
        *,
        task: Task | None = None,
        capability: Capability | None = None,
        stage: str,
    ) -> _GovernancePersistence:
        """Build DB-owned governance facts for kernel task execution."""
        return await self._evaluate_governance(
            operation,
            task=task,
            capability=capability,
            stage=stage,
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
    ) -> _GovernancePersistence:
        policies = (
            *default_db_policies(),
            *self.config.policies,
            *self.registry.policies,
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
        result, approvals_to_request = await self._reconcile_approval_state(result)
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

    async def _commit_operation_governance_blocked(
        self,
        operation: Operation,
        governance_persistence: _GovernancePersistence,
        *,
        governance: GovernanceResult,
    ) -> None:
        await self.store.commit_governance_blocked(
            operation=replace(operation, status=OperationStatus.BLOCKED),
            task=None,
            decisions=governance.decisions,
            audit_record=governance_persistence.audit_record,
            approval_requests=governance_persistence.approvals_to_request,
            events=(
                *governance_persistence.events,
                self._runtime_event(
                    type=RuntimeEventType.OPERATION_UPDATED,
                    operation_id=operation.id,
                    message=f"Operation {operation.id} blocked by governance policy.",
                    payload={"governance": governance.to_dict()},
                ),
            ),
        )

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
            await self.store.save_operation(replace(operation, status=result.status))
            await self.store.append_event(
                self._runtime_event(
                    type=RuntimeEventType.OPERATION_UPDATED,
                    operation_id=result.operation_id,
                    message=(
                        f"Operation {result.operation_id} finished with "
                        f"{result.status.value}."
                    ),
                    payload={"warnings": list(result.warnings)},
                )
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


def _has_running_tasks(tasks: tuple[Task, ...]) -> bool:
    return any(task.status is TaskStatus.RUNNING for task in tasks)


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
    if capability.id == "db.sql.execute_write":
        return {"sql_ref": "sql.validation"}
    if capability.id == "db.sql.validate":
        return {"sql": prompt, "operation": operation.operation_type}
    return {"prompt": prompt}


def _task_dependencies_for_capability(
    operation: Operation,
    capability: Capability,
    *,
    validation_task: Task | None = None,
) -> tuple[TaskDependency, ...]:
    if capability.id != "db.sql.execute_write":
        return ()
    return (
        TaskDependency(
            kind="evidence",
            evidence_kind="sql.validation",
            evidence_owner=(
                validation_task.metadata.get("owner") if validation_task else None
            ),
            producer_task_id=validation_task.id if validation_task else None,
            producer_capability_id=(
                validation_task.capability_id if validation_task else "db.sql.validate"
            ),
            producer_executor_id=(
                validation_task.executor_id if validation_task else None
            ),
            evidence_payload={"valid": True},
            operation_id=operation.id,
            input_hash=(
                validation_task.metadata.get("input_hash") if validation_task else None
            ),
        ),
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
    if task is not None:
        metadata["task_id"] = task.id
    if capability is not None:
        metadata["capability_id"] = capability.id
        metadata["capability_owner"] = capability.owner
    return replace(operation, request=request, metadata=metadata)


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


def _capability_governance_facts(capability: Capability) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "domains": sorted(capability.domains),
        "operation_types": sorted(capability.operation_types),
        "access": capability.access.value,
        "risk": capability.risk.value,
        "runtime_only": capability.runtime_only,
        "specialist_only": capability.specialist_only,
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


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}
