"""
Skeleton database runtime built on the extension registry.
"""

from __future__ import annotations

from dataclasses import replace
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

from ..agent_loop import DbAgentLoop, DbAgentLoopBlocked
from ..authorization import normalize_authorization
from ..contracts import DbContractBuilder
from ..fallback_planner import ContractFallbackDbAgentPlanner
from ..llm_service import DbLLMService, db_llm_service_from_metadata
from ..monitor_commands.loop_adapter import DbMonitorCommandLoopPlanner
from ..models import (
    DbOperationContract,
    DbOperationResult,
    DbRequest,
    DbRuntimeConfig,
    DbRuntimeInspection,
)
from ..monitors import DbMonitorStore
from ..safety import DbCapabilityLane, DbSafetyFrame, DbSafetyVerifier
from ..session_context import db_session_context_from_request
from ..synthesis import DbSynthesizer
from ..verification import DbVerifier
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
    _db_contract_context,
    _db_contract_from_context,
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
    DbRuntimeMemoryLearningMixin,
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
        host_services: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.config = config or DbRuntimeConfig()
        self.registry = registry or ExtensionRegistry()
        self.store = store or InMemoryRuntimeStore()
        self.monitor_store = monitor_store or _default_monitor_store(self.store)
        self.approval_channel = approval_channel or InMemoryApprovalChannel(self.store)
        self.runtime_id = runtime_id or f"db-runtime-{uuid4()}"
        self.safety_verifier = DbSafetyVerifier()
        self.verifier = DbVerifier()
        self.synthesizer = DbSynthesizer()
        self.db_llm_service = db_llm_service or db_llm_service_from_metadata(
            self.config.metadata
        )
        self.host_services = dict(host_services or {})
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

    def build_safety_frame(self, request: DbRequest | str) -> DbSafetyFrame:
        """Build deterministic DB safety facts for a request."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        return self.safety_verifier.verify(db_request)

    def build_contract(
        self,
        request: DbRequest | str,
        *,
        safety_frame: DbSafetyFrame | None = None,
        skill_resolution: SkillResolution | None = None,
        include_skills: bool = True,
    ) -> DbOperationContract:
        """Build a lane-based operation contract from registry capabilities."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        resolved_safety_frame = safety_frame or self.build_safety_frame(db_request)
        resolved_skills = skill_resolution
        if resolved_skills is None and include_skills:
            resolved_skills = self._resolve_skills(db_request)
        return DbContractBuilder(self.registry, self.config).build(
            db_request,
            resolved_safety_frame,
            skill_effects=resolved_skills.effects if resolved_skills else (),
        )

    def _db_request_from_operation(self, operation: Operation) -> DbRequest:
        return _db_request_from_context(operation)

    def _db_contract_from_context(self, operation: Operation) -> DbOperationContract:
        return _db_contract_from_context(operation)

    async def run(self, request: DbRequest | str) -> DbOperationResult:
        """Plan and execute a DB operation through typed runtime capabilities."""
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        if not self._is_setup:
            await self.setup()
        safety_frame = self.build_safety_frame(db_request)
        skill_resolution = self._resolve_skills(db_request)
        contract = self.build_contract(
            db_request,
            safety_frame=safety_frame,
            skill_resolution=skill_resolution,
        )
        resume_context = {
            "request": _db_request_context(db_request),
            "safety_frame": safety_frame.to_dict(),
            "contract": _db_contract_context(contract),
            "skills": skill_resolution.to_metadata(),
        }
        authorization = normalize_authorization(db_request.metadata)
        resume_context["authorization"] = authorization
        operation_metadata = {
            "access": contract.access.value,
            "authorization": authorization,
            "skills": skill_resolution.to_metadata(),
            "safety_frame": safety_frame.to_dict(),
            "granted_lanes": [lane.value for lane in safety_frame.granted_lanes],
            "forbidden_capabilities": list(safety_frame.forbidden_capabilities),
            "contract": contract.metadata,
            "contract_metadata": contract.metadata,
            "resume_context": resume_context,
        }
        monitor_command = db_request.metadata.get("db_monitor_command")
        if isinstance(monitor_command, dict):
            operation_metadata.update(
                {
                    "control_plane": "db.monitor",
                    "command_kind": monitor_command.get("kind"),
                    "monitor_id": monitor_command.get("monitor_id"),
                    "user_id": db_request.user_id,
                    "session_id": db_request.session_id,
                    "source_scope": list(db_request.source_scope),
                    "request_metadata": dict(db_request.metadata),
                }
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
                "session_context": db_request.session_context,
                "metadata": {
                    **db_request.metadata,
                    "authorization": authorization,
                },
            },
            required_evidence=frozenset(contract.required_evidence),
            metadata=operation_metadata,
            evaluate_governance=False,
        )
        operation_id = operation.id
        operation = await self._persist_trace_correlation(operation)
        await self._record_skill_resolution(operation_id, skill_resolution, contract)

        base_diagnostics = {
            "runtime_id": self.runtime_id,
            "registered_plugins": list(self.registry.plugin_ids),
            "safety_frame": safety_frame.to_dict(),
            "granted_lanes": [lane.value for lane in safety_frame.granted_lanes],
            "forbidden_capabilities": list(safety_frame.forbidden_capabilities),
            "contract": contract.metadata,
            "skills": skill_resolution.to_metadata(),
        }
        session_context = db_session_context_from_request(db_request)
        if session_context is not None:
            base_diagnostics["session_context"] = session_context.to_diagnostic_dict()
        if contract.metadata.get("missing_capabilities"):
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer="Required DB capabilities are not registered.",
                    warnings=("db_runtime_missing_capabilities",),
                    diagnostics=base_diagnostics,
                ),
                operation=operation,
            )

        try:
            if _defer_monitor_management_governance(contract):
                governance = GovernanceResult(
                    True,
                    False,
                    False,
                    metadata={"deferred_to_task_governance": True},
                )
            else:
                governance = await self.kernel.evaluate_operation_governance(
                    operation.id
                )
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

        if self._should_route_multi_step_analysis(db_request, contract):
            return await self._run_multi_step_analysis(
                db_request,
                contract,
                operation,
                base_diagnostics=base_diagnostics,
            )

        outcome_diagnostics: dict[str, Any] = {}
        outcome_warnings: tuple[str, ...] = ()
        planner = self._db_agent_planner(db_request)
        planner_source = self._db_agent_planner_source(planner)
        try:
            loop_result = await DbAgentLoop(
                self,
                planner,
            ).run(
                request=db_request,
                operation=operation,
                contract=contract,
            )
        except DbRuntimeGovernanceBlocked as exc:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
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
        except (DbAgentLoopBlocked, ValueError) as exc:
            operation = await self._persist_planner_loop_metadata(
                operation,
                planner_status="blocked",
                planner_error=exc,
            )
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
                    contract=contract,
                    status=OperationStatus.BLOCKED,
                    answer="DB planner action was blocked by the operation contract.",
                    warnings=("db_planner_action_blocked",),
                    diagnostics={
                        **base_diagnostics,
                        "planner": {
                            "status": "blocked",
                            "source": planner_source,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                    },
                ),
                operation=operation,
            )
        except Exception as exc:
            return await self._record_operation_result(
                DbOperationResult(
                    operation_id=operation_id,
                    request=db_request,
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

        operation = await self._persist_planner_loop_metadata(
            operation,
            planner_status=loop_result.status,
            planner_observations=[
                observation.to_dict() for observation in loop_result.observations
            ],
            planner_decision=(
                loop_result.decision.to_dict() if loop_result.decision else None
            ),
            planner_task_ids=[task.id for task in loop_result.tasks],
        )
        final_evidence = tuple(await self.store.list_evidence(operation_id))
        final_tasks = tuple(await self.store.list_tasks(operation_id))
        verification = None
        synthesis = None
        if loop_result.status not in {"blocked", "clarify"} and any(
            item.accepted for item in final_evidence
        ):
            verification = self.verifier.verify(contract, final_evidence, final_tasks)
            verification_evidence = await self._persist_verification_result_evidence(
                operation,
                verification,
                final_evidence,
            )
            final_evidence = tuple(await self.store.list_evidence(operation_id))
            if verification.passed:
                synthesis, _synthesis_task = await self._execute_answer_synthesis(
                    operation=operation,
                    outcome_evidence=(*final_evidence, verification_evidence),
                )
                final_evidence = tuple(await self.store.list_evidence(operation_id))
                final_tasks = tuple(await self.store.list_tasks(operation_id))
            else:
                final_tasks = tuple(await self.store.list_tasks(operation_id))
        result_status = (
            OperationStatus.FAILED
            if _has_rejected_memory_proposal(final_evidence)
            or bool(outcome_diagnostics.get("repair_exhausted"))
            else (
                OperationStatus.BLOCKED
                if loop_result.status in {"blocked", "clarify"}
                or (verification is not None and not verification.passed)
                else OperationStatus.SUCCEEDED
            )
        )
        warnings = (
            ("db_planner_clarification_required",)
            if loop_result.status == "clarify"
            else (
                ("db_planner_loop_blocked",)
                if loop_result.status == "blocked"
                else (
                    tuple(
                        [
                            *(
                                f"missing_evidence:{kind}"
                                for kind in verification.missing_evidence
                            ),
                            *verification.warnings,
                        ]
                    )
                    if verification is not None and not verification.passed
                    else outcome_warnings
                )
            )
        )
        if _has_rejected_memory_proposal(final_evidence):
            warnings = tuple(
                dict.fromkeys(
                    (
                        *warnings,
                        *_memory_proposal_rejection_reasons(final_evidence),
                    )
                )
            )
        synthesized_answer = (
            str(synthesis.payload.get("answer"))
            if synthesis is not None and synthesis.payload.get("answer")
            else None
        )
        return await self._record_operation_result(
            DbOperationResult(
                operation_id=operation_id,
                request=db_request,
                contract=contract,
                status=result_status,
                answer=synthesized_answer
                or loop_result.message
                or _answer_from_planner_status(loop_result.status),
                evidence=final_evidence,
                warnings=warnings,
                diagnostics={
                    **base_diagnostics,
                    "planner": {
                        "status": loop_result.status,
                        "source": planner_source,
                        "decision": (
                            loop_result.decision.to_dict()
                            if loop_result.decision
                            else None
                        ),
                        "observations": [
                            observation.to_dict()
                            for observation in loop_result.observations
                        ],
                    },
                    "execution": {
                        **outcome_diagnostics,
                        "task_count": len(final_tasks),
                        "tasks": [task.to_dict() for task in final_tasks],
                        "evidence_refs": [
                            {
                                "id": item.id,
                                "kind": item.kind,
                                "owner": item.owner,
                                "task_id": item.task_id,
                            }
                            for item in final_evidence
                            if item.id
                        ],
                        "planned_sql": outcome_diagnostics.get("planned_sql")
                        or _latest_sql_from_evidence(final_evidence),
                    },
                    **(
                        {"verification": verification.to_dict()}
                        if verification is not None
                        else {}
                    ),
                    **(
                        {
                            "synthesis": {
                                "sufficiency": synthesis.payload.get("sufficiency"),
                                "cited_evidence_refs": synthesis.payload.get(
                                    "cited_evidence_refs", []
                                ),
                                "diagnostics": synthesis.payload.get("diagnostics", {}),
                            }
                        }
                        if synthesis is not None
                        else {}
                    ),
                },
            ),
            operation=operation,
        )

    def _db_agent_planner(self, request: DbRequest | None = None) -> Any:
        monitor_command = (
            request.metadata.get("db_monitor_command") if request is not None else None
        )
        if isinstance(monitor_command, dict):
            return DbMonitorCommandLoopPlanner(monitor_command)
        return self._injected_db_agent_planner() or ContractFallbackDbAgentPlanner(
            self.config.metadata
        )

    def _db_agent_planner_source(self, planner: Any) -> str:
        if isinstance(planner, DbMonitorCommandLoopPlanner):
            return "monitor_command_adapter"
        injected = self._injected_db_agent_planner()
        if injected is not None and planner is injected:
            return "injected"
        return "fallback"

    def _injected_db_agent_planner(self) -> Any:
        return self.host_services.get("db_agent_planner") or self.host_services.get(
            "db_planner"
        )

    async def _persist_planner_loop_metadata(
        self,
        operation: Operation,
        *,
        planner_status: str,
        planner_observations: list[dict[str, Any]] | None = None,
        planner_decision: dict[str, Any] | None = None,
        planner_task_ids: list[str] | None = None,
        planner_error: Exception | None = None,
    ) -> Operation:
        planner_metadata: dict[str, Any] = {
            "status": planner_status,
            "observations": list(planner_observations or ()),
            "task_ids": list(planner_task_ids or ()),
        }
        if planner_decision is not None:
            planner_metadata["decision"] = planner_decision
        if planner_error is not None:
            planner_metadata["error"] = {
                "type": type(planner_error).__name__,
                "message": str(planner_error),
            }
        updated = replace(
            operation,
            metadata={
                **operation.metadata,
                "planner": planner_metadata,
                "planner_diagnostics": planner_metadata,
                "planner_observations": planner_metadata["observations"],
            },
        )
        await self.store.save_operation(updated)
        return updated

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


def _inspection_counts(operation: Operation) -> bool:
    return operation.operation_type != "db.memory.learning"


def _defer_monitor_management_governance(contract: DbOperationContract) -> bool:
    lanes = set(contract.metadata.get("granted_lanes") or ())
    return (
        contract.operation_type.startswith("monitor.")
        and "monitor_write" in lanes
        and "monitor_execute" not in lanes
    )


def _latest_sql_from_evidence(evidence: tuple[Evidence, ...]) -> str | None:
    for kind in ("query.result", "sql.validation", "query.plan.validation"):
        for item in reversed(evidence):
            if item.kind == kind and item.accepted and item.payload.get("sql"):
                return str(item.payload["sql"])
    return None


def _has_rejected_memory_proposal(evidence: tuple[Evidence, ...]) -> bool:
    return any(
        item.kind == "db.memory.proposal" and item.accepted is False
        for item in evidence
    )


def _memory_proposal_rejection_reasons(
    evidence: tuple[Evidence, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for item in evidence:
        if item.kind != "db.memory.proposal" or item.accepted is not False:
            continue
        reasons.append("memory_proposal_not_accepted")
        validation = item.payload.get("validation")
        if isinstance(validation, dict):
            reasons.extend(str(reason) for reason in validation.get("reasons") or ())
    return tuple(dict.fromkeys(reasons))


def _answer_from_planner_status(status: str) -> str:
    if status == "clarify":
        return "The DB planner needs clarification before it can continue."
    if status == "blocked":
        return "The DB planner could not complete within the operation contract."
    return "DB planner completed."


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
