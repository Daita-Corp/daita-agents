"""
Skeleton database runtime built on the extension registry.
"""

from __future__ import annotations

import secrets
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.plugins import ExtensionRegistry, PluginContext, ServiceRegistry
from daita.runtime import (
    AccessMode,
    Capability,
    ContextAudience,
    ContextBlock,
    Evidence,
    ApprovalStatus,
    InMemoryApprovalChannel,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    OperationSnapshot,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeKernel,
    RuntimeStore,
    Task,
)
from daita.skills import SkillResolution, SkillResolver

from ..evidence import evidence_in_task_plan_order
from ..loop import DbAgentLoop, DbLoopResult
from ..llm_agent_planner import DbLLMAgentPlanner
from ..llm_service import DbLLMService
from ..models import (
    DbIntent,
    DbIntentKind,
    DbLimits,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
    DbRuntimeConfig,
    DbRuntimeInspection,
)
from ..monitors import DbMonitorMutation, DbMonitorStore
from ..planning import DbContractBuilder, build_safety_frame, classify_db_request
from ..planner_protocol import DbAgentPlanner
from ..session_context import (
    db_session_context_from_request,
    persist_session_query_scopes,
)
from ..synthesis import DbSynthesizer
from ..verification import DbVerifier, db_run_finalization_check
from .analysis import DbRuntimeAnalysisMixin
from .cache import DbRuntimeCacheMixin
from .extensions import DbRuntimePlanningPlugin
from .governance import (
    DbRuntimeGovernanceMixin,
    _safe_source_repr,
)
from .memory_learning import DbRuntimeMemoryLearningMixin
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
from .tasks.context import DbTaskContext
from .tasks.models import DbTaskPlan, DbTaskSpec
from .tasks.runtime import DbTaskRuntime
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
    DbRuntimeMemoryLearningMixin,
    DbRuntimeResultsMixin,
    DbRuntimeResumeMixin,
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
        host_services: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.config = config or DbRuntimeConfig()
        self.registry = registry or ExtensionRegistry()
        self.store = store or InMemoryRuntimeStore()
        self.monitor_store = monitor_store or _default_monitor_store(self.store)
        self.approval_channel = approval_channel or InMemoryApprovalChannel(self.store)
        self.runtime_id = runtime_id or f"db-runtime-{uuid4()}"
        self.verifier = DbVerifier()
        self.synthesizer = DbSynthesizer()
        self.db_llm_service = db_llm_service or DbLLMService(None)
        self.host_services = dict(host_services or {})
        if "audit_fingerprint_key" in self.host_services:
            audit_fingerprint_key = self.host_services.pop("audit_fingerprint_key")
            if (
                not isinstance(audit_fingerprint_key, bytes)
                or len(audit_fingerprint_key) < 32
            ):
                raise ValueError(
                    "host_services['audit_fingerprint_key'] must be bytes with "
                    "length at least 32"
                )
        else:
            audit_fingerprint_key = secrets.token_bytes(32)
        self._audit_fingerprint_key = audit_fingerprint_key
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
        task_context = DbTaskContext(
            registry=self.registry,
            store=self.store,
            kernel=self.kernel,
            config=self.config,
            runtime_id=self.runtime_id,
        )
        self.tasks = DbTaskRuntime(task_context)

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
                **self.host_services,
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

    async def execute_task(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        return await self.tasks.execute_task(task, operation, context)

    async def commit_monitor_mutation(self, mutation: DbMonitorMutation) -> None:
        """Persist one monitor mutation and publish its committed events."""
        await self.kernel.commit_events(
            lambda: self.monitor_store.commit_monitor_mutation(mutation),
            mutation.events,
        )

    async def execute_capability(
        self,
        capability_id: str,
        *,
        owner: str | None = None,
        operation_type: str,
        input: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> tuple[Evidence, ...]:
        if not self._is_setup:
            await self.setup()
        return await self.tasks.execute_capability(
            capability_id,
            owner=owner,
            operation_type=operation_type,
            input=input,
            operation_id=operation_id,
        )

    async def plan_task_specs(
        self,
        operation: Operation,
        specs: Iterable[DbTaskSpec],
        *,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
    ) -> DbTaskPlan:
        return await self.tasks.plan_task_specs(
            operation,
            specs,
            contract=contract,
        )

    async def task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        return await self.tasks.task_readiness(task, operation)

    async def executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        return await self.tasks.executable_input_for_task(task, operation)

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
        return classify_db_request(db_request)

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
        fallback = _db_intent_from_context(operation)
        contract = self._db_contract_from_context(operation)
        return _intent_from_loop_contract(contract, fallback)

    def _db_contract_from_context(self, operation: Operation) -> DbOperationContract:
        fallback = _db_contract_from_context(operation)
        return _contract_from_latest_loop_snapshot(operation, fallback)

    async def run(
        self,
        request: DbRequest | str,
        *,
        operation_id: str | None = None,
    ) -> DbOperationResult:
        """Plan and execute a DB operation through typed runtime capabilities."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        if not self._is_setup:
            await self.setup()
        skill_resolution = self._resolve_skills(db_request)
        safety_frame = build_safety_frame(self.registry, self.config, db_request)
        intent = _neutral_run_intent()
        contract = _neutral_run_contract(
            self.config,
            request=db_request,
            registry=self.registry,
            safety_frame=safety_frame.to_dict(),
            skill_resolution=skill_resolution,
        )
        operation = await self.kernel.create_operation(
            operation_id=operation_id,
            operation_type="db.run",
            request={
                "prompt": db_request.prompt,
                "user_id": db_request.user_id,
                "session_id": db_request.session_id,
                "source_scope": list(db_request.source_scope),
                "mode": db_request.mode,
                "requested_capabilities": list(db_request.requested_capabilities),
                "constraints": db_request.constraints,
                "metadata": db_request.metadata,
                "session_context": db_request.session_context,
            },
            required_evidence=frozenset(),
            metadata={
                "intent_kind": intent.kind.value,
                "access": contract.access.value,
                "normalized_request": {
                    "prompt": db_request.prompt,
                    "source_scope": list(db_request.source_scope),
                    "mode": db_request.mode,
                    "requested_capabilities": list(db_request.requested_capabilities),
                    "constraints": db_request.constraints,
                },
                "source_scope": list(db_request.source_scope),
                "mode": db_request.mode,
                "requested_capabilities": list(db_request.requested_capabilities),
                "constraints": db_request.constraints,
                "skills": skill_resolution.to_metadata(),
                "runtime_limits": self.config.limits.to_dict(),
                "safety_frame": safety_frame.to_dict(),
                "loop_state": {"status": "bootstrap"},
                "resume_context": {
                    "request": _db_request_context(db_request),
                    "intent": _db_intent_context(intent),
                    "contract": _db_contract_context(contract),
                    "skills": skill_resolution.to_metadata(),
                    "safety_frame": safety_frame.to_dict(),
                },
            },
            evaluate_governance=False,
        )
        operation_id = operation.id
        operation = await self._persist_trace_correlation(
            operation,
            intent_kind=intent.kind.value,
        )
        await self._record_skill_resolution(operation_id, skill_resolution, contract)

        base_diagnostics = {
            "runtime_id": self.runtime_id,
            "registered_plugins": list(self.registry.plugin_ids),
            "contract": contract.metadata,
            "safety_frame": safety_frame.to_dict(),
            "skills": skill_resolution.to_metadata(),
        }
        session_context = db_session_context_from_request(db_request)
        if session_context is not None:
            base_diagnostics["session_context"] = session_context.to_diagnostic_dict()

        planner = self._select_db_agent_planner()
        if planner is None:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer=(
                        "DB LLM service is required for semantic DB planning. "
                        "Configure a from_db model and provider to run this request."
                    ),
                    warnings=("db_runtime_llm_configuration_required",),
                    diagnostics={
                        **base_diagnostics,
                        "configuration_required": True,
                    },
                ),
                operation=operation,
            )

        try:
            loop_result = await DbAgentLoop(self, planner).run(
                operation,
                safety_frame=safety_frame.to_dict(),
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
                    answer="DB operation failed before final synthesis.",
                    warnings=("db_runtime_execution_failed",),
                    diagnostics={
                        **base_diagnostics,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                ),
                operation=operation,
            )

        if loop_result.status == "finished":
            return await self._finalize_run_operation(
                operation_id=operation_id,
                request=db_request,
                fallback_intent=intent,
                fallback_contract=contract,
                loop_result=loop_result,
                base_diagnostics=base_diagnostics,
            )

        current_operation = await self.store.load_operation(operation_id) or operation
        tasks = tuple(await self.store.list_tasks(operation_id))
        evidence = evidence_in_task_plan_order(
            await self.store.list_evidence(operation_id),
            tasks,
        )
        return await self._record_operation_result(
            DbOperationResult(
                operation_id=operation_id,
                request=db_request,
                intent=intent,
                contract=contract,
                status=_operation_status_from_loop_status(loop_result.status),
                answer=_answer_from_loop_result(loop_result),
                evidence=evidence,
                warnings=tuple(loop_result.warnings),
                diagnostics={
                    **base_diagnostics,
                    "planner": _planner_diagnostics(loop_result),
                    "execution": _execution_diagnostics(
                        operation=current_operation,
                        tasks=tasks,
                        evidence=evidence,
                    ),
                },
            ),
            operation=current_operation,
        )

    def _select_db_agent_planner(self) -> DbAgentPlanner | None:
        return (
            self.host_services.get("db_agent_planner")
            or self.host_services.get("db_planner")
            or (
                DbLLMAgentPlanner(self.db_llm_service)
                if self.db_llm_service.available
                else None
            )
        )

    async def _finalize_run_operation(
        self,
        *,
        operation_id: str,
        request: DbRequest,
        fallback_intent: DbIntent,
        fallback_contract: DbOperationContract,
        loop_result: DbLoopResult | None = None,
        base_diagnostics: dict[str, Any] | None = None,
    ) -> DbOperationResult:
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        tasks = tuple(await self.store.list_tasks(operation_id))
        evidence = evidence_in_task_plan_order(
            await self.store.list_evidence(operation_id),
            tasks,
        )
        await persist_session_query_scopes(
            self.store,
            operation,
            tasks,
            evidence,
        )
        evidence = evidence_in_task_plan_order(
            await self.store.list_evidence(operation_id),
            tasks,
        )
        contract = _contract_from_latest_loop_snapshot(operation, fallback_contract)
        intent = _intent_from_loop_contract(contract, fallback_intent)
        diagnostics = dict(base_diagnostics or {})
        if loop_result is not None:
            diagnostics["planner"] = _planner_diagnostics(loop_result)
        loop_warnings = loop_result.warnings if loop_result is not None else ()
        finalization = db_run_finalization_check(
            operation=operation,
            verifier=self.verifier,
            contract=contract,
            intent=intent,
            evidence=evidence,
            tasks=tasks,
        )
        verification = finalization.verification
        if not finalization.finalizable:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.FAILED,
                    answer="DB operation could not be verified against required evidence.",
                    evidence=evidence,
                    warnings=tuple((*loop_warnings, *verification.warnings)),
                    diagnostics={
                        **diagnostics,
                        "execution": _execution_diagnostics(
                            operation=operation,
                            tasks=tasks,
                            evidence=evidence,
                        ),
                        "verification": verification.to_dict(),
                        "finalization": {
                            **finalization.to_dict(),
                            "intent": _intent_diagnostics(intent),
                            "contract": _contract_diagnostics(contract),
                        },
                    },
                ),
                operation=operation,
            )

        verification_evidence = await self.tasks.persist_verification_result_evidence(
            operation,
            verification,
            evidence,
        )
        synthesis_evidence, synthesis_task = await self.tasks.execute_answer_synthesis(
            operation=operation,
            intent=intent,
            outcome_evidence=(*evidence, verification_evidence),
        )
        refreshed_tasks = tuple(await self.store.list_tasks(operation_id))
        final_tasks = refreshed_tasks
        final_evidence = evidence_in_task_plan_order(
            await self.store.list_evidence(operation_id),
            final_tasks,
        )
        if synthesis_task not in final_tasks:
            final_tasks = (*final_tasks, synthesis_task)
        if synthesis_evidence not in final_evidence:
            final_evidence = (*final_evidence, synthesis_evidence)
        raw_synthesis_warnings = synthesis_evidence.payload.get("warnings")
        synthesis_warnings = (
            tuple(str(item) for item in raw_synthesis_warnings)
            if isinstance(raw_synthesis_warnings, list)
            else ()
        )
        return await self._record_operation_result(
            DbOperationResult(
                operation_id=operation_id,
                request=request,
                intent=intent,
                contract=contract,
                status=OperationStatus.SUCCEEDED,
                answer=_answer_from_synthesis_evidence(synthesis_evidence),
                evidence=final_evidence,
                warnings=(*loop_warnings, *synthesis_warnings),
                diagnostics={
                    **diagnostics,
                    "execution": _execution_diagnostics(
                        operation=operation,
                        tasks=final_tasks,
                        evidence=final_evidence,
                    ),
                    "verification": verification.to_dict(),
                    "synthesis": synthesis_evidence.payload,
                },
            ),
            operation=operation,
        )

    async def _run_operation_finalization_state(
        self,
        operation_id: str,
        *,
        fallback_intent: DbIntent | None = None,
        fallback_contract: DbOperationContract | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        tasks = tuple(await self.store.list_tasks(operation_id))
        evidence = evidence_in_task_plan_order(
            await self.store.list_evidence(operation_id),
            tasks,
        )
        fallback_contract = fallback_contract or self._db_contract_from_context(
            operation
        )
        fallback_intent = fallback_intent or self._db_intent_from_operation(operation)
        contract = _contract_from_latest_loop_snapshot(operation, fallback_contract)
        intent = _intent_from_loop_contract(contract, fallback_intent)
        check = db_run_finalization_check(
            operation=operation,
            verifier=self.verifier,
            contract=contract,
            intent=intent,
            evidence=evidence,
            tasks=tasks,
        )
        return check.finalizable, {
            **check.to_dict(),
            "intent": _intent_diagnostics(intent),
            "contract": _contract_diagnostics(contract),
        }

    async def _try_finalize_run_operation_from_snapshot(
        self,
        snapshot: OperationSnapshot,
        *,
        request: DbRequest,
        fallback_intent: DbIntent,
        fallback_contract: DbOperationContract,
        base_diagnostics: dict[str, Any] | None = None,
    ) -> OperationSnapshot | None:
        finalizable, finalization = await self._run_operation_finalization_state(
            snapshot.operation.id,
            fallback_intent=fallback_intent,
            fallback_contract=fallback_contract,
        )
        if not finalizable:
            return None
        await self._finalize_run_operation(
            operation_id=snapshot.operation.id,
            request=request,
            fallback_intent=fallback_intent,
            fallback_contract=fallback_contract,
            base_diagnostics={
                **dict(base_diagnostics or {}),
                "pre_planner_finalization": True,
                "finalization": finalization,
            },
        )
        finalized = await self.inspect_operation(snapshot.operation.id)
        if finalized is None:
            raise KeyError(snapshot.operation.id)
        return finalized

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
        return len(
            [operation for operation in operations if _inspection_counts(operation)]
        )

    async def _last_stored_operation_id(self) -> str | None:
        operations = await self.store.list_operations()
        counted = [
            operation for operation in operations if _inspection_counts(operation)
        ]
        return counted[-1].id if counted else None

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
                capability = self.tasks.capability_for_task(task)
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


def _inspection_counts(operation: Operation) -> bool:
    return operation.operation_type != "db.memory.learning"


def _neutral_run_intent() -> DbIntent:
    return DbIntent(
        kind=DbIntentKind.CONVERSATIONAL,
        confidence=1.0,
        access=AccessMode.NONE,
        evidence_mode="planner_loop",
        diagnostics={"source": "db_run_bootstrap"},
    )


def _neutral_run_contract(
    config: DbRuntimeConfig,
    *,
    request: DbRequest,
    registry: ExtensionRegistry,
    safety_frame: dict[str, Any],
    skill_resolution: SkillResolution,
) -> DbOperationContract:
    skill_contract_metadata = _merge_skill_metadata(
        effect.contract_metadata for effect in skill_resolution.effects
    )
    skill_verifier_metadata = _merge_skill_metadata(
        effect.verifier_metadata for effect in skill_resolution.effects
    )
    skill_synthesis_metadata = _merge_skill_metadata(
        effect.synthesis_metadata for effect in skill_resolution.effects
    )
    requested_capabilities = _ordered_unique(
        (
            *request.requested_capabilities,
            *[
                capability_id
                for effect in skill_resolution.effects
                for capability_id in effect.requested_capabilities
            ],
        )
    )
    available_capabilities = {capability.id for capability in registry.capabilities}
    missing_capabilities = [
        capability_id
        for capability_id in requested_capabilities
        if capability_id not in available_capabilities
    ]
    required_evidence = _ordered_unique(
        evidence
        for effect in skill_resolution.effects
        for evidence in effect.required_evidence
    )
    policy_ids = _ordered_unique(
        policy_id
        for effect in skill_resolution.effects
        for policy_id in effect.policy_ids
    )
    metadata = {
        "planned_operation": {
            "operation_type": "db.run",
            "access": AccessMode.NONE.value,
            "source": "db_run_bootstrap",
        },
        "requested_capabilities": list(requested_capabilities),
        "required_evidence": list(required_evidence),
        "missing_capabilities": missing_capabilities,
        "safety_frame": safety_frame,
        "skills": skill_resolution.to_metadata(),
        "skill_contract_metadata": skill_contract_metadata,
        "skill_verifier_metadata": skill_verifier_metadata,
        "skill_synthesis_metadata": skill_synthesis_metadata,
    }
    metadata.update(skill_contract_metadata)
    return DbOperationContract(
        operation_type="db.run",
        required_capabilities=tuple(requested_capabilities),
        required_evidence=tuple(required_evidence),
        access=AccessMode.NONE,
        limits=config.limits,
        policy_ids=tuple(policy_ids),
        metadata=metadata,
    )


def _contract_from_latest_loop_snapshot(
    operation: Operation,
    fallback: DbOperationContract,
) -> DbOperationContract:
    context = operation.metadata.get("resume_context")
    context = context if isinstance(context, dict) else {}
    snapshot = (
        operation.metadata.get("latest_compiled_contract_snapshot")
        or context.get("latest_compiled_contract_snapshot")
        or context.get("contract")
    )
    if not isinstance(snapshot, dict):
        return fallback
    limits = snapshot.get("limits")
    limits = limits if isinstance(limits, dict) else {}
    metadata = snapshot.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return DbOperationContract(
        operation_type=str(snapshot.get("operation_type") or fallback.operation_type),
        required_capabilities=tuple(snapshot.get("required_capabilities") or ()),
        required_evidence=tuple(snapshot.get("required_evidence") or ()),
        access=AccessMode(str(snapshot.get("access") or fallback.access.value)),
        limits=DbLimits(**limits) if limits else fallback.limits,
        policy_ids=tuple(snapshot.get("policy_ids") or ()),
        metadata=metadata,
    )


def _intent_from_loop_contract(
    contract: DbOperationContract,
    fallback: DbIntent,
) -> DbIntent:
    intent_metadata = contract.metadata.get("planner_intent")
    intent_metadata = intent_metadata if isinstance(intent_metadata, dict) else {}
    operation_type = str(
        intent_metadata.get("operation_type") or contract.operation_type
    )
    try:
        kind = DbIntentKind(operation_type)
    except ValueError:
        kind = fallback.kind
    return DbIntent(
        kind=kind,
        confidence=1.0,
        access=contract.access,
        evidence_mode="planner_loop",
        diagnostics={
            "source": "planner_compiled_contract",
            "operation_type": operation_type,
            "planner_intent": intent_metadata,
        },
    )


def _operation_status_from_loop_status(status: str) -> OperationStatus:
    if status in {"blocked", "configuration_required", "clarification_required"}:
        return OperationStatus.BLOCKED
    if status == "budget_exhausted":
        return OperationStatus.BLOCKED
    if status == "failed":
        return OperationStatus.FAILED
    return OperationStatus.FAILED


def _answer_from_loop_result(loop_result: DbLoopResult) -> str:
    if loop_result.status == "configuration_required":
        return (
            "DB LLM service is required for semantic DB planning. Configure a "
            "from_db model and provider to run this request."
        )
    if loop_result.status == "clarification_required":
        question = loop_result.diagnostics.get("clarification_question")
        if question:
            return str(question)
        return "The DB planner needs clarification before it can continue."
    if loop_result.status == "blocked":
        return "This operation was blocked before execution completed."
    if loop_result.status == "budget_exhausted":
        return "The DB planner exhausted its turn budget before finishing."
    return "DB operation failed before final synthesis."


def _intent_diagnostics(intent: DbIntent) -> dict[str, Any]:
    return {
        "kind": intent.kind.value,
        "access": intent.access.value,
        "evidence_mode": intent.evidence_mode,
        "diagnostics": intent.diagnostics,
    }


def _contract_diagnostics(contract: DbOperationContract) -> dict[str, Any]:
    return {
        "operation_type": contract.operation_type,
        "required_capabilities": list(contract.required_capabilities),
        "required_evidence": list(contract.required_evidence),
        "access": contract.access.value,
        "limits": contract.limits.to_dict(),
        "policy_ids": list(contract.policy_ids),
        "metadata": contract.metadata,
    }


def _planner_diagnostics(loop_result: DbLoopResult) -> dict[str, Any]:
    return {
        "status": loop_result.status,
        "warnings": list(loop_result.warnings),
        "diagnostics": dict(loop_result.diagnostics),
    }


def _execution_diagnostics(
    *,
    operation: Operation,
    tasks: tuple[Task, ...],
    evidence: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        "operation_id": operation.id,
        "task_count": len(tasks),
        "tasks": [task.to_dict() for task in tasks],
        "evidence_refs": [_evidence_ref(item) for item in evidence],
        "planned_sql": _planned_sql_from_persisted_state(tasks, evidence),
    }


def _evidence_ref(evidence: Any) -> dict[str, Any]:
    return {
        "id": getattr(evidence, "id", None),
        "kind": getattr(evidence, "kind", None),
        "task_id": getattr(evidence, "task_id", None),
        "accepted": bool(getattr(evidence, "accepted", False)),
    }


def _planned_sql_from_persisted_state(
    tasks: tuple[Task, ...],
    evidence: tuple[Any, ...],
) -> str | None:
    for item in reversed(evidence):
        sql = _sql_from_payload(getattr(item, "payload", {}) or {})
        if sql:
            return sql
    for task in reversed(tasks):
        sql = task.input.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql
    return None


def _sql_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("sql", "selected_sql"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    structured = payload.get("structured_plan")
    if isinstance(structured, dict):
        value = structured.get("selected_sql")
        if isinstance(value, str) and value.strip():
            return value
    return None


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


def _merge_skill_metadata(values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        for key, item in dict(value).items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(item, dict)
            ):
                merged[key] = {**merged[key], **item}
            else:
                merged[key] = item
    return merged


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(str(value))
    return out


def _current_trace_ids() -> tuple[str | None, str | None]:
    try:
        from daita.core.tracing import get_trace_manager

        context = get_trace_manager().trace_context
        return context.current_trace_id, context.current_span_id
    except Exception:
        return None, None
