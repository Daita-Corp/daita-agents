"""
Skeleton database runtime built on the extension registry.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any
from uuid import uuid4

from daita.plugins import ExtensionRegistry, PluginContext, ServiceRegistry
from daita.runtime import (
    Capability,
    ContextAudience,
    ContextBlock,
    ApprovalStatus,
    GovernanceResult,
    InMemoryApprovalChannel,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    OperationSnapshot,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeKernel,
    RuntimeKernelGovernanceBlocked,
    RuntimeStore,
    Task,
)
from daita.skills import SkillResolution, SkillResolver

from ..execution import DbOperationExecutor
from ..llm_service import DbLLMService, db_llm_service_from_metadata
from ..models import (
    DbIntent,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
    DbRuntimeConfig,
    DbRuntimeInspection,
)
from ..monitors import DbMonitorStore
from ..planning import DbContractBuilder, DbIntentClassifier
from ..synthesis import DbSynthesizer
from ..verification import DbVerifier
from .analysis import DbRuntimeAnalysisMixin
from .cache import DbRuntimeCacheMixin
from .extensions import DbRuntimePlanningPlugin
from .governance import (
    DbRuntimeGovernanceMixin,
    _safe_source_repr,
)
from .monitors import DbRuntimeMonitorsMixin, _default_monitor_store
from .resume import (
    DbRuntimeResumeMixin,
    _answer_from_synthesis_evidence,
    _db_contract_context,
    _db_contract_from_context,
    _db_intent_context,
    _db_intent_from_context,
    _db_request_context,
    _db_request_from_context,
)
from .results import DbRuntimeResultsMixin
from .tasks import DbRuntimeTasksMixin
from .types import (
    _SourcePreparationSnapshot,
    _governance_blocked_answer,
    _governance_blocked_warning,
    DbRuntimeGovernanceBlocked,
)


class DbRuntime(
    DbRuntimeAnalysisMixin,
    DbRuntimeCacheMixin,
    DbRuntimeGovernanceMixin,
    DbRuntimeMonitorsMixin,
    DbRuntimeResultsMixin,
    DbRuntimeResumeMixin,
    DbRuntimeTasksMixin,
):
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

    @staticmethod
    def _has_pending_approvals(snapshot: OperationSnapshot) -> bool:
        return any(
            request.status is ApprovalStatus.PENDING
            for request in snapshot.approval_requests
        )

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


def _current_trace_ids() -> tuple[str | None, str | None]:
    try:
        from daita.core.tracing import get_trace_manager

        context = get_trace_manager().trace_context
        return context.current_trace_id, context.current_span_id
    except Exception:
        return None, None
