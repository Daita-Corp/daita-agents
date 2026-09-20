"""Typed foreground admission for the unreleased Phase 4 graph slice."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityGrant,
    CapabilityInputError,
    CapabilityRegistry,
    ExecutionAdmissionPolicy,
    ExecutionContractBindings,
    ExecutionContractReader,
    ExecutionPreference,
    Executor,
    OperationalEffect,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from ...capability_runtime import CapabilityFailure, SideEffectPlan
from ...distribution.owner import DistributionOwner
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.models import RunInput, RunOrigin
from ...loop.session import RunSession
from ...scope import EffectiveSourceScope
from ..owner import JobError, JobOwner
from .capabilities import (
    GRAPH_RESULT_FINALIZE_CAPABILITY_ID,
    PLANNER_CAPABILITY_IDS,
    REVIEW_CAPABILITY_IDS,
    TASK_BLOCK_CAPABILITY_ID,
    TASK_CHECKPOINT_CAPABILITY_ID,
    TASK_COMMENT_CAPABILITY_ID,
    TASK_COMPLETE_CAPABILITY_ID,
    TASK_REQUEST_REVIEW_CAPABILITY_ID,
)
from .execution import PLANNER_CATALOG_CAPABILITY_IDS
from .models import (
    BudgetAmount,
    BudgetLimit,
    EdgeKind,
    GraphAdmission,
    GraphAuthority,
    GraphDesiredState,
    GraphJob,
    GraphJobSpecification,
    GraphLimits,
    GraphState,
    GraphTask,
    GraphTaskSpecification,
    JobGraph,
    TaskDependency,
    TaskExecutionKind,
    TaskRole,
    TaskState,
    canonical_digest,
    topology_digest,
)

GRAPH_ADMISSION_DOMAIN_OWNER_ID = "jobs.graph_admission"
START_GRAPH_JOB_CAPABILITY_ID = "jobs.graph.start"
START_GRAPH_JOB_EXECUTOR_ID = "jobs.graph.start.executor"
START_GRAPH_JOB_OUTPUT_KIND = "graph.job_started"
START_GRAPH_JOB_TOOL_NAME = "start_graph_job"

_MIN_DEADLINE_SECONDS = 60
_MAX_DEADLINE_SECONDS = 24 * 60 * 60
_GRAPH_SUPPORT_CAPABILITY_IDS = (
    TASK_BLOCK_CAPABILITY_ID,
    TASK_CHECKPOINT_CAPABILITY_ID,
    TASK_COMMENT_CAPABILITY_ID,
    TASK_COMPLETE_CAPABILITY_ID,
    TASK_REQUEST_REVIEW_CAPABILITY_ID,
)
_NATIVE_GRAPH_EFFECT_CAPABILITY_IDS = frozenset(
    {"data.update_rows", "data.upsert_rows"}
)


def _native_preview_capability(capability_id: str) -> str:
    if capability_id == "data.update_rows":
        return "data.preview_update_rows"
    if capability_id == "data.upsert_rows":
        return "data.preview_upsert_rows"
    raise ValueError("capability is not a released native graph write")


class GraphGrantPreparer(Protocol):
    async def __call__(
        self,
        capability_id: str,
        requested_constraints: Mapping[str, object],
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> CapabilityGrant: ...


@dataclass(frozen=True, slots=True)
class InitialTaskProposal:
    """Code-resolved initial work; model JSON cannot supply its authority."""

    capability_id: str
    arguments: Mapping[str, object]
    expected_result_contract: Mapping[str, object]
    source_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    connector_binding_ids: tuple[str, ...]
    access_mode: AccessMode
    model_route_id: str
    sensitivity: ModelSensitivity
    contract_bindings: ExecutionContractBindings
    execution_policy: ExecutionAdmissionPolicy
    max_steps: int
    max_wall_time_seconds: int
    per_run_max_tokens: int
    per_run_max_cost_usd: Decimal
    preview_capability_id: str | None = None
    requested_grant_constraints: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.capability_id, "initial capability_id"),
            (self.model_route_id, "initial model route"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        object.__setattr__(
            self, "arguments", FrozenJsonObject.from_mapping(self.arguments)
        )
        object.__setattr__(
            self,
            "expected_result_contract",
            FrozenJsonObject.from_mapping(self.expected_result_contract),
        )
        if self.preview_capability_id is not None and (
            not isinstance(self.preview_capability_id, str)
            or not self.preview_capability_id
        ):
            raise ValueError("initial preview capability must be non-empty text")
        if self.requested_grant_constraints is not None:
            object.__setattr__(
                self,
                "requested_grant_constraints",
                FrozenJsonObject.from_mapping(self.requested_grant_constraints),
            )
        for name in ("source_ids", "resource_ids", "connector_binding_ids"):
            values = tuple(getattr(self, name))
            if values != tuple(sorted(set(values))) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain sorted exact identities")
            object.__setattr__(self, name, values)
        if not isinstance(self.access_mode, AccessMode):
            raise TypeError("initial task access mode is invalid")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("initial task sensitivity is invalid")
        if not isinstance(self.contract_bindings, ExecutionContractBindings):
            raise TypeError("initial task contract bindings are invalid")
        if not isinstance(self.execution_policy, ExecutionAdmissionPolicy):
            raise TypeError("initial task execution policy is invalid")
        if (
            self.execution_policy.admission_error(
                self.arguments, ExecutionPreference.DURABLE
            )
            != "durable_execution_required"
        ):
            raise ValueError("initial task shape is not graph-V1 eligible")
        for numeric_value, name in (
            (self.max_steps, "initial task step limit"),
            (self.max_wall_time_seconds, "initial task wall-time limit"),
            (self.per_run_max_tokens, "initial task token limit"),
        ):
            if (
                not isinstance(numeric_value, int)
                or isinstance(numeric_value, bool)
                or numeric_value < 1
            ):
                raise ValueError(f"{name} must be positive")
        if self.max_wall_time_seconds > self.execution_policy.graph_max_wall_seconds:
            raise ValueError("initial task wall time exceeds its graph policy")
        if (
            not isinstance(self.per_run_max_cost_usd, Decimal)
            or not self.per_run_max_cost_usd.is_finite()
            or self.per_run_max_cost_usd < 0
        ):
            raise ValueError("initial task cost limit is invalid")

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    self.capability_id,
                    *(
                        ()
                        if self.preview_capability_id is None
                        else (self.preview_capability_id,)
                    ),
                    *_GRAPH_SUPPORT_CAPABILITY_IDS,
                )
            )
        )

    @property
    def effectful(self) -> bool:
        return self.requested_grant_constraints is not None

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "capability_id": self.capability_id,
                "arguments": self.arguments,
                "expected_result_contract": self.expected_result_contract,
                "source_ids": self.source_ids,
                "resource_ids": self.resource_ids,
                "connector_binding_ids": self.connector_binding_ids,
                "access_mode": self.access_mode.value,
                "model_route_id": self.model_route_id,
                "sensitivity": self.sensitivity.value,
                "contract_bindings": self.contract_bindings.material(),
                "preview_capability_id": self.preview_capability_id,
                "requested_grant_constraints": self.requested_grant_constraints,
                "max_steps": self.max_steps,
                "max_wall_time_seconds": self.max_wall_time_seconds,
                "per_run_max_tokens": self.per_run_max_tokens,
                "per_run_max_cost_usd": str(self.per_run_max_cost_usd),
            }
        )


class InitialTaskProposalResolver(Protocol):
    async def __call__(
        self,
        *,
        proposal: Mapping[str, object],
        source_scope: EffectiveSourceScope | None,
        sensitivity: ModelSensitivity,
    ) -> InitialTaskProposal: ...


class RegistryInitialTaskProposalResolver:
    """Re-derive a proposal from current registry and binding truth."""

    def __init__(
        self,
        *,
        agent_id: str,
        registry: CapabilityRegistry,
        contract_reader: ExecutionContractReader,
        model_route_id: str,
        max_steps: int,
        per_run_max_tokens: int,
        per_run_max_cost_usd: Decimal,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("proposal resolver agent_id must be non-empty")
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("proposal resolver requires CapabilityRegistry")
        if not callable(contract_reader):
            raise TypeError("proposal resolver requires a contract reader")
        self._agent_id = agent_id
        self._registry = registry
        self._contract_reader = contract_reader
        self._model_route_id = model_route_id
        self._max_steps = max_steps
        self._per_run_max_tokens = per_run_max_tokens
        self._per_run_max_cost_usd = per_run_max_cost_usd

    async def __call__(
        self,
        *,
        proposal: Mapping[str, object],
        source_scope: EffectiveSourceScope | None,
        sensitivity: ModelSensitivity,
    ) -> InitialTaskProposal:
        capability_id = proposal.get("capability_id")
        arguments = proposal.get("arguments")
        result_contract = proposal.get("expected_result_contract")
        references = proposal.get("retained_references", {})
        requested_grant = proposal.get("effect_grant")
        if (
            not isinstance(capability_id, str)
            or not isinstance(arguments, Mapping)
            or not isinstance(result_contract, Mapping)
            or not isinstance(references, Mapping)
        ):
            raise CapabilityInputError(
                "graph_initial_task_invalid",
                "The typed initial-task proposal is malformed.",
            )
        try:
            capability = self._registry.capability(capability_id)
        except KeyError as error:
            raise CapabilityInputError(
                "execution_not_supported_for_requested_shape",
                "The proposed capability is not registered for this host.",
            ) from error
        policy = capability.execution_admission_policy
        effectful = capability.operational_effect is not OperationalEffect.NONE
        preview_capability_id: str | None = None
        requested_constraints: Mapping[str, object] | None = None
        if effectful:
            owner_id = self._registry.resolve_domain_owner(capability_id)
            native = capability_id in _NATIVE_GRAPH_EFFECT_CAPABILITY_IDS
            mcp_action = (
                owner_id == "mcp"
                and capability.operational_effect is OperationalEffect.EXTERNAL_ACTION
            )
            if not (native or mcp_action):
                raise CapabilityInputError(
                    "execution_not_supported_for_requested_shape",
                    "Only the specifically released native writes and qualifying MCP actions are graph eligible.",
                )
            if (
                not isinstance(requested_grant, Mapping)
                or not isinstance(requested_grant.get("constraints"), Mapping)
                or set(requested_grant) != {"constraints"}
            ):
                raise CapabilityInputError(
                    "graph_effect_grant_required",
                    "The effectful task requires one exact code-normalized graph grant proposal.",
                )
            requested_constraints = requested_grant["constraints"]
            if native:
                preview_capability_id = _native_preview_capability(capability_id)
                validated = self._registry.validate_arguments(
                    preview_capability_id, arguments
                )
            else:
                validated = self._registry.validate_arguments(capability_id, arguments)
        else:
            if requested_grant is not None:
                raise CapabilityInputError(
                    "graph_effect_grant_invalid",
                    "Effect-free graph work cannot retain an effect grant.",
                )
            validated = self._registry.validate_arguments(capability_id, arguments)
        if (
            policy is None
            or policy.admission_error(validated, ExecutionPreference.DURABLE)
            != "durable_execution_required"
            or (
                effectful
                and (
                    capability.automation_eligibility
                    is not AutomationEligibility.AUTOMATION_DIRECT
                    or capability.effect_receipt_policy is None
                    or capability.automation_grant_policy is None
                )
            )
        ):
            raise CapabilityInputError(
                "execution_not_supported_for_requested_shape",
                "The proposed task is not eligible for the released graph contract.",
            )
        source_ids = _reference_ids(references, "source_ids")
        resource_ids = _reference_ids(references, "resource_ids")
        connector_ids = _reference_ids(references, "connector_binding_ids")
        if source_scope is not None and (
            not set(source_ids) <= set(source_scope.source_ids)
            or not set(resource_ids) <= set(source_scope.resource_ids)
        ):
            raise CapabilityInputError(
                "source_scope_violation",
                "The proposed task references resources outside this run's exact scope.",
            )
        capability_ids = tuple(
            sorted(
                (
                    capability_id,
                    *((preview_capability_id,) if preview_capability_id else ()),
                    *_GRAPH_SUPPORT_CAPABILITY_IDS,
                )
            )
        )
        bindings = await self._contract_reader(
            agent_id=self._agent_id,
            source_ids=source_ids,
            resource_ids=resource_ids,
            capability_ids=capability_ids,
            connector_binding_ids=connector_ids,
            model_route_ids=(self._model_route_id,),
        )
        bindings.validate_coverage(
            capability_ids=capability_ids,
            resource_ids=resource_ids,
            route_ids=(self._model_route_id,),
            mcp_capability_ids=bindings.tool_origins,
        )
        if any(
            bindings.capability_contracts[item] != self._registry.contract_digest(item)
            for item in capability_ids
        ):
            raise CapabilityInputError(
                "execution_contract_changed",
                "A proposed graph capability contract is no longer current.",
            )
        result_kind = result_contract.get("result_kind")
        if not isinstance(result_kind, str) or not result_kind:
            raise CapabilityInputError(
                "graph_initial_task_invalid",
                "The initial task requires one bounded result_kind.",
            )
        return InitialTaskProposal(
            capability_id=capability_id,
            arguments=validated,
            expected_result_contract=result_contract,
            source_ids=source_ids,
            resource_ids=resource_ids,
            connector_binding_ids=connector_ids,
            access_mode=capability.access_mode,
            model_route_id=self._model_route_id,
            sensitivity=sensitivity,
            contract_bindings=bindings,
            execution_policy=policy,
            max_steps=self._max_steps,
            max_wall_time_seconds=policy.graph_max_wall_seconds,
            per_run_max_tokens=self._per_run_max_tokens,
            per_run_max_cost_usd=self._per_run_max_cost_usd,
            preview_capability_id=preview_capability_id,
            requested_grant_constraints=requested_constraints,
        )


class GraphAdmissionBuilder:
    """Create only the Phase 4 minimal work-plus-finalizer topology."""

    def __init__(
        self,
        *,
        agent_id: str,
        registry: CapabilityRegistry,
        distribution: DistributionOwner,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self._agent_id = agent_id
        self._registry = registry
        self._distribution = distribution
        self._clock = clock
        self._id_factory = id_factory
        self._grant_preparer: GraphGrantPreparer | None = None

    def bind_grant_preparer(self, preparer: GraphGrantPreparer) -> None:
        if self._grant_preparer is not None:
            raise RuntimeError("graph grant preparer is already bound")
        if not callable(preparer):
            raise TypeError("graph grant preparer must be callable")
        self._grant_preparer = preparer

    async def build_admission(
        self,
        *,
        run_id: str,
        call_id: str,
        conversation_id: str,
        objective: str,
        outcome_contract: Mapping[str, object],
        deadline_seconds: int,
        proposal: InitialTaskProposal,
    ) -> GraphAdmission:
        grant: CapabilityGrant | None = None
        if proposal.effectful:
            if self._grant_preparer is None:
                raise CapabilityInputError(
                    "graph_effect_grant_unavailable",
                    "Graph effect grant preparation is unavailable.",
                )
            assert proposal.requested_grant_constraints is not None
            target = self._distribution.resolve_conversation_inbox(
                conversation_id, sensitivity_ceiling=proposal.sensitivity
            )
            plan = self._distribution.resolve_plan(
                conversation_id,
                destination_id=target.destination_id,
                sensitivity_ceiling=proposal.sensitivity,
            )
            effect = self._registry.capability(
                proposal.capability_id
            ).operational_effect
            grant = await self._grant_preparer(
                proposal.capability_id,
                proposal.requested_grant_constraints,
                1,
                AutomationScopeProposal(
                    agent_id=self._agent_id,
                    principal_id=self._agent_id,
                    allowed_source_ids=proposal.source_ids,
                    allowed_resource_ids=proposal.resource_ids,
                    allowed_connector_binding_ids=proposal.connector_binding_ids,
                    allowed_capability_ids=proposal.capability_ids,
                    allowed_access_modes=frozenset(
                        {
                            AccessMode.NONE,
                            *(
                                self._registry.capability(capability_id).access_mode
                                for capability_id in proposal.capability_ids
                            ),
                        }
                    ),
                    allowed_operational_effects=frozenset(
                        {OperationalEffect.NONE, effect}
                    ),
                    sensitivity_ceiling=proposal.sensitivity,
                    eligible_model_routes=(proposal.model_route_id,),
                    per_run_max_cost_usd=proposal.per_run_max_cost_usd,
                    per_run_max_tokens=proposal.per_run_max_tokens,
                    expires_at=self._clock() + timedelta(seconds=deadline_seconds),
                    distribution_plan_digest=plan.plan_digest,
                ),
            )
        return self.build(
            run_id=run_id,
            call_id=call_id,
            conversation_id=conversation_id,
            objective=objective,
            outcome_contract=outcome_contract,
            deadline_seconds=deadline_seconds,
            proposal=proposal,
            prepared_grant=grant,
        )

    def build(
        self,
        *,
        run_id: str,
        call_id: str,
        conversation_id: str,
        objective: str,
        outcome_contract: Mapping[str, object],
        deadline_seconds: int,
        proposal: InitialTaskProposal,
        prepared_grant: CapabilityGrant | None = None,
    ) -> GraphAdmission:
        if not 60 <= deadline_seconds <= _MAX_DEADLINE_SECONDS:
            raise CapabilityInputError(
                "graph_deadline_invalid",
                "A graph deadline must be between sixty seconds and twenty-four hours.",
            )
        finalizer = self._registry.capability(GRAPH_RESULT_FINALIZE_CAPABILITY_ID)
        finalizer_digest = self._registry.contract_digest(finalizer.id)
        if finalizer.operational_effect is not OperationalEffect.NONE:
            raise ValueError("graph finalizer must be effect-free")
        planner_catalog_ids: list[str] = []
        for capability_id in sorted(PLANNER_CATALOG_CAPABILITY_IDS):
            try:
                self._registry.capability(capability_id)
            except KeyError:
                continue
            planner_catalog_ids.append(capability_id)
        planner_capability_ids = tuple(
            sorted((*PLANNER_CAPABILITY_IDS, *planner_catalog_ids))
        )
        reviewer_capability_ids = tuple(sorted(REVIEW_CAPABILITY_IDS))
        planner_capabilities = tuple(
            self._registry.capability(capability_id)
            for capability_id in planner_capability_ids
        )
        if any(
            capability.operational_effect is not OperationalEffect.NONE
            or capability.effect_receipt_policy is not None
            or capability.automation_grant_policy is not None
            for capability in planner_capabilities
        ):
            raise ValueError("graph planner capability must be effect-free")
        root_access_modes = tuple(
            sorted(
                {
                    AccessMode.NONE.value,
                    *(
                        self._registry.capability(capability_id).access_mode.value
                        for capability_id in proposal.capability_ids
                    ),
                    *(item.access_mode.value for item in planner_capabilities),
                }
            )
        )
        planner_contracts = {
            capability_id: self._registry.contract_digest(capability_id)
            for capability_id in planner_capability_ids
        }
        reviewer_contracts = {
            capability_id: self._registry.contract_digest(capability_id)
            for capability_id in reviewer_capability_ids
        }
        reviewer_capabilities = tuple(
            self._registry.capability(capability_id)
            for capability_id in reviewer_capability_ids
        )
        if any(
            capability.operational_effect is not OperationalEffect.NONE
            or capability.access_mode is not AccessMode.NONE
            or capability.effect_receipt_policy is not None
            or capability.automation_grant_policy is not None
            for capability in reviewer_capabilities
        ):
            raise ValueError("graph reviewer capability must be effect-free and local")
        callable_tool_names = tuple(
            sorted(
                name
                for name in self._registry.tool_names
                if self._registry.tool_capability(name).id == proposal.capability_id
            )
        )
        if not callable_tool_names:
            raise CapabilityInputError(
                "execution_not_supported_for_requested_shape",
                "The proposed capability has no model-visible tool in this host.",
            )
        now = self._clock()
        deadline = now + timedelta(seconds=deadline_seconds)
        job_id = self._id_factory("job")
        work_id = self._id_factory("task")
        finalizer_id = self._id_factory("task")
        target = self._distribution.resolve_conversation_inbox(
            conversation_id, sensitivity_ceiling=proposal.sensitivity
        )
        plan = self._distribution.resolve_plan(
            conversation_id,
            destination_id=target.destination_id,
            sensitivity_ceiling=proposal.sensitivity,
        )
        grant = prepared_grant
        if proposal.effectful != (grant is not None):
            raise CapabilityInputError(
                "graph_effect_grant_unavailable",
                "Effectful graph admission requires one prepared exact grant.",
            )
        grant_bindings: dict[str, object] = {}
        if grant is not None:
            grant_bindings["capability_grants"] = {
                grant.capability_id: {
                    "grant": grant.material(),
                    "grant_digest": grant.grant_digest,
                }
            }
        worker_effects = tuple(
            sorted(
                {
                    OperationalEffect.NONE.value,
                    self._registry.capability(
                        proposal.capability_id
                    ).operational_effect.value,
                }
            )
        )
        worker_authority = GraphAuthority(
            source_ids=proposal.source_ids,
            resource_ids=proposal.resource_ids,
            connector_ids=proposal.connector_binding_ids,
            capability_ids=proposal.capability_ids,
            access_modes=tuple(
                sorted(
                    {
                        AccessMode.NONE.value,
                        *(
                            self._registry.capability(capability_id).access_mode.value
                            for capability_id in proposal.capability_ids
                        ),
                    }
                )
            ),
            operational_effects=worker_effects,
            model_route_ids=(proposal.model_route_id,),
            sensitivity=proposal.sensitivity,
            contract_bindings={
                **proposal.contract_bindings.material(),
                **grant_bindings,
            },
        )
        finalizer_bindings = ExecutionContractBindings(
            capability_contracts={finalizer.id: finalizer_digest}
        )
        finalizer_authority = GraphAuthority(
            capability_ids=(finalizer.id,),
            access_modes=(AccessMode.NONE.value,),
            operational_effects=(OperationalEffect.NONE.value,),
            sensitivity=proposal.sensitivity,
            contract_bindings=finalizer_bindings.material(),
        )
        root_bindings = ExecutionContractBindings(
            capability_contracts={
                **dict(proposal.contract_bindings.capability_contracts),
                finalizer.id: finalizer_digest,
                **planner_contracts,
                **reviewer_contracts,
            },
            tool_origins=proposal.contract_bindings.tool_origins,
            resource_revisions=proposal.contract_bindings.resource_revisions,
            model_routes=proposal.contract_bindings.model_routes,
        )
        root_authority = GraphAuthority(
            source_ids=proposal.source_ids,
            resource_ids=proposal.resource_ids,
            connector_ids=proposal.connector_binding_ids,
            capability_ids=tuple(
                sorted(
                    (
                        *proposal.capability_ids,
                        finalizer.id,
                        *planner_capability_ids,
                        *reviewer_capability_ids,
                    )
                )
            ),
            access_modes=root_access_modes,
            operational_effects=worker_effects,
            model_route_ids=(proposal.model_route_id,),
            sensitivity=proposal.sensitivity,
            contract_bindings={**root_bindings.material(), **grant_bindings},
        )
        root = GraphJobSpecification(
            principal_id=self._agent_id,
            objective=objective,
            outcome_contract=outcome_contract,
            authority=root_authority,
            distribution_plan_digest=plan.plan_digest,
            budgets=(BudgetLimit("work_units", 50, control_reserved=27),),
            deadline_at=deadline,
            limits=GraphLimits(max_tasks=64, max_edges=192, max_parallelism=4),
            retry_policy={"max_attempts": 3, "protocol_violation_attempts": 2},
            cancellation_policy={"preserve_evidence": True},
            effect_mode="exact_grants" if grant is not None else "disabled",
            planner_task_template={
                "role": TaskRole.PLANNER.value,
                "execution_kind": TaskExecutionKind.MODEL.value,
                "priority": 900,
                "specification": {
                    "title": "Replan blocked graph work",
                    "description": (
                        "Inspect durable graph evidence and make only bounded typed "
                        "topology mutations inside the immutable root authority."
                    ),
                    "expected_result_contract": {
                        "kind": "model_task",
                        "result_kind": "graph.plan",
                        "model_route_id": proposal.model_route_id,
                        "per_run_max_tokens": proposal.per_run_max_tokens,
                        "per_run_max_cost_usd": str(proposal.per_run_max_cost_usd),
                        "attempt_budgets": {"work_units": 1},
                    },
                    "authority": GraphAuthority(
                        source_ids=root_authority.source_ids,
                        resource_ids=root_authority.resource_ids,
                        connector_ids=root_authority.connector_ids,
                        capability_ids=tuple(
                            sorted(
                                (
                                    *(
                                        ()
                                        if grant is not None
                                        else proposal.capability_ids
                                    ),
                                    *planner_capability_ids,
                                )
                            )
                        ),
                        access_modes=root_authority.access_modes,
                        operational_effects=(OperationalEffect.NONE.value,),
                        model_route_ids=root_authority.model_route_ids,
                        sensitivity=root_authority.sensitivity,
                        contract_bindings=ExecutionContractBindings(
                            capability_contracts={
                                capability_id: root_bindings.capability_contracts[
                                    capability_id
                                ]
                                for capability_id in sorted(
                                    (
                                        *(
                                            ()
                                            if grant is not None
                                            else proposal.capability_ids
                                        ),
                                        *planner_capability_ids,
                                    )
                                )
                            },
                            tool_origins={
                                capability_id: origin
                                for capability_id, origin in root_bindings.tool_origins.items()
                                if grant is None
                            },
                            resource_revisions=root_bindings.resource_revisions,
                            model_routes=root_bindings.model_routes,
                        ).material(),
                    ).digest_material(),
                    "budgets": ({"dimension": "work_units", "amount": 3},),
                    "max_steps": proposal.max_steps,
                    "max_wall_time_seconds": min(
                        proposal.max_wall_time_seconds, deadline_seconds
                    ),
                    "created_by": "supervisor_replan",
                },
            },
            finalizer_task_template={
                "kind": "graph_result_finalizer",
                "capability_id": finalizer.id,
                "contract_digest": finalizer_digest,
            },
        )
        if grant is None:
            initial_call: Mapping[str, object] = {
                "capability_id": proposal.capability_id,
                "callable_tool_names": callable_tool_names,
                "arguments": proposal.arguments,
            }
        else:
            preview_names = tuple(
                sorted(
                    name
                    for name in self._registry.tool_names
                    if proposal.preview_capability_id is not None
                    and self._registry.tool_capability(name).id
                    == proposal.preview_capability_id
                )
            )
            if proposal.preview_capability_id is not None and not preview_names:
                raise CapabilityInputError(
                    "execution_not_supported_for_requested_shape",
                    "The required native preview has no model-visible tool.",
                )
            initial_call = {
                "effect_call": {
                    "capability_id": proposal.capability_id,
                    "callable_tool_names": callable_tool_names,
                    "preview_capability_id": proposal.preview_capability_id,
                    "preview_callable_tool_names": preview_names,
                    "intent_arguments": proposal.arguments,
                    "grant_digest": grant.grant_digest,
                }
            }
        work_spec = GraphTaskSpecification(
            title="Execute admitted graph work",
            description=(
                "Execute the exact code-resolved initial call and finish through one "
                "exclusive lifecycle terminator."
            ),
            expected_result_contract={
                "kind": "model_task",
                "result_kind": proposal.expected_result_contract["result_kind"],
                "expected_result_contract": proposal.expected_result_contract,
                "initial_call": initial_call,
                "model_route_id": proposal.model_route_id,
                "per_run_max_tokens": proposal.per_run_max_tokens,
                "per_run_max_cost_usd": str(proposal.per_run_max_cost_usd),
                "attempt_budgets": {"work_units": 1},
            },
            authority=worker_authority,
            budgets=(BudgetAmount("work_units", 3),),
            max_steps=proposal.max_steps,
            max_wall_time_seconds=min(proposal.max_wall_time_seconds, deadline_seconds),
            created_by="job_owner",
        )
        finalizer_spec = GraphTaskSpecification(
            title="Finalize graph result",
            description="Authenticate the accepted work result and publish it once.",
            expected_result_contract={
                "kind": "graph_result_finalizer",
                "output_kind": START_GRAPH_JOB_OUTPUT_KIND.replace(
                    "job_started", "result_finalized"
                ),
                "capability_id": finalizer.id,
                "contract_digest": finalizer_digest,
                "attempt_budgets": {"work_units": 1},
            },
            authority=finalizer_authority,
            budgets=(BudgetAmount("work_units", 3),),
            max_steps=1,
            max_wall_time_seconds=min(300, deadline_seconds),
            created_by="job_owner",
        )
        work = GraphTask(
            agent_id=self._agent_id,
            job_id=job_id,
            task_id=work_id,
            state=TaskState.READY,
            role=TaskRole.WORKER,
            execution_kind=TaskExecutionKind.MODEL,
            priority=100,
            not_before=None,
            current_attempt_id=None,
            task_revision=1,
            specification=work_spec,
            task_spec_digest=work_spec.digest,
            task_scope_digest=worker_authority.digest,
            attempt_count=0,
            failure_streak=0,
            fencing_epoch=0,
            created_at=now,
            updated_at=now,
        )
        finish = GraphTask(
            agent_id=self._agent_id,
            job_id=job_id,
            task_id=finalizer_id,
            state=TaskState.PENDING,
            role=TaskRole.FINALIZER,
            execution_kind=TaskExecutionKind.INTERNAL_CAPABILITY,
            priority=1_000,
            not_before=None,
            current_attempt_id=None,
            task_revision=1,
            specification=finalizer_spec,
            task_spec_digest=finalizer_spec.digest,
            task_scope_digest=finalizer_authority.digest,
            attempt_count=0,
            failure_streak=0,
            fencing_epoch=0,
            created_at=now,
            updated_at=now,
        )
        dependency = TaskDependency(
            agent_id=self._agent_id,
            job_id=job_id,
            upstream_task_id=work_id,
            downstream_task_id=finalizer_id,
            edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
            created_at=now,
            creator_key="job_owner",
        )
        tasks = (work, finish)
        dependencies = (dependency,)
        job = GraphJob(
            agent_id=self._agent_id,
            job_id=job_id,
            conversation_id=conversation_id,
            origin_run_id=run_id,
            origin_call_id=call_id,
            state=GraphState.QUEUED,
            desired_state=GraphDesiredState.RUN,
            created_at=now,
            updated_at=now,
            deadline_at=deadline,
            specification=root,
            specification_digest=root.digest,
            finalizer_task_id=finalizer_id,
        )
        graph = JobGraph(
            agent_id=self._agent_id,
            job_id=job_id,
            revision=0,
            task_count=2,
            edge_count=1,
            mutation_count=0,
            active_attempt_count=0,
            next_ready_at=now,
            finalization_attempt_id=None,
            finalization_started_revision=None,
            created_at=now,
            updated_at=now,
            topology_digest=topology_digest(tasks, dependencies),
        )
        return GraphAdmission(job, graph, tasks, dependencies)


class StartGraphJobExecutor:
    executor_id = START_GRAPH_JOB_EXECUTOR_ID

    def __init__(
        self,
        *,
        owner: JobOwner,
        resolver: InitialTaskProposalResolver,
        builder: GraphAdmissionBuilder,
    ) -> None:
        self._owner = owner
        self._resolver = resolver
        self._builder = builder

    async def _proposal(self, request: ToolExecution) -> InitialTaskProposal:
        initial = request.arguments["initial_task"]
        assert isinstance(initial, Mapping)
        return await self._resolver(
            proposal=initial,
            source_scope=request.source_scope,
            sensitivity=request.request_sensitivity,
        )

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        proposal = await self._proposal(request)
        return FrozenJsonObject.from_mapping(
            {
                "proposal_digest": proposal.digest,
                "capability_id": proposal.capability_id,
                "resource_ids": proposal.resource_ids,
            }
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        if request.conversation_id is None:
            raise CapabilityInputError(
                "graph_conversation_required",
                "A durable graph requires one exact originating conversation.",
            )
        proposal = await self._proposal(request)
        objective = request.arguments["objective"]
        outcome_contract = request.arguments["outcome_contract"]
        deadline_seconds = request.arguments.get("deadline_seconds", 3_600)
        assert isinstance(objective, str)
        assert isinstance(outcome_contract, Mapping)
        assert isinstance(deadline_seconds, int)
        admission = await self._builder.build_admission(
            run_id=request.run_id,
            call_id=request.call_id,
            conversation_id=request.conversation_id,
            objective=objective,
            outcome_contract=outcome_contract,
            deadline_seconds=deadline_seconds,
            proposal=proposal,
        )
        job = await self._owner.admit(admission)
        return ToolOutput(
            kind=START_GRAPH_JOB_OUTPUT_KIND,
            data={
                "job_id": job.job_id,
                "state": job.state.value,
                "specification_digest": job.specification_digest,
                "initial_task_proposal_digest": proposal.digest,
                "task_count": 2,
            },
            sensitivity=proposal.sensitivity,
            sensitivity_provenance={
                "authority": "code_resolved_initial_task_proposal",
                "job_id": job.job_id,
                "resource_ids": proposal.resource_ids,
            },
        )


class GraphAdmissionCapabilityDomain:
    domain_owner_id = GRAPH_ADMISSION_DOMAIN_OWNER_ID

    def __init__(self, declarations: CapabilityDeclarations) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("graph admission declarations have the wrong owner")
        self._declarations = declarations

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    async def project(
        self, run: RunInput, session: RunSession | None = None
    ) -> tuple[str, ...]:
        del session
        return (START_GRAPH_JOB_TOOL_NAME,) if run.origin is RunOrigin.USER else ()

    project_session = project

    def normalize_arguments(
        self, capability: Capability, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        del capability
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
        session: RunSession | None = None,
    ) -> FrozenJsonObject:
        del call, capability, request_sensitivity, session
        if run.origin is not RunOrigin.USER:
            raise CapabilityInputError(
                "graph_admission_foreground_only",
                "Graph admission is available only in a foreground run.",
            )
        return arguments

    prepare_session_call = prepare_call

    async def prepare_automation_grant(
        self,
        capability: Capability,
        constraints: FrozenJsonObject,
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> FrozenJsonObject:
        del capability, constraints, max_calls_per_occurrence, proposal
        raise CapabilityInputError(
            "automation_grant_unsupported",
            "Graph admission cannot be delegated to unattended execution.",
        )

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        del run, call, capability, fingerprint
        initial = execution.arguments.get("initial_task")
        effectful = isinstance(initial, Mapping) and "effect_grant" in initial
        return SideEffectPlan(
            approval_required=effectful,
            recheck_after_approval=True,
            approval_arguments=(
                FrozenJsonObject.from_mapping(execution.arguments)
                if effectful
                else None
            ),
            approval_reason=(
                "Approve this exact durable graph effect grant and initial task?"
                if effectful
                else "Start this bounded effect-free durable graph?"
            ),
        )

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
        session: RunSession | None = None,
    ) -> ToolOutput:
        del run, call, capability, arguments, request_sensitivity, session
        return output

    finalize_session_output = finalize_output

    def normalize_error(
        self, call: ToolCall, error: BaseException
    ) -> CapabilityFailure | None:
        del call
        if isinstance(error, JobError):
            return CapabilityFailure(error.code, str(error))
        return None


def graph_admission_declarations(
    *,
    owner: JobOwner,
    resolver: InitialTaskProposalResolver,
    builder: GraphAdmissionBuilder,
) -> tuple[CapabilityDeclarations, tuple[Executor, ...]]:
    capability = Capability(
        id=START_GRAPH_JOB_CAPABILITY_ID,
        description=(
            "Start one bounded durable graph from an exact typed initial-task "
            "proposal. Released effects require a separately approved exact grant."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "objective": {"type": "string", "minLength": 1, "maxLength": 16384},
                "outcome_contract": {"type": "object"},
                "deadline_seconds": {
                    "type": "integer",
                    "minimum": _MIN_DEADLINE_SECONDS,
                    "maximum": _MAX_DEADLINE_SECONDS,
                },
                "initial_task": {
                    "type": "object",
                    "properties": {
                        "capability_id": {"type": "string", "minLength": 1},
                        "arguments": {"type": "object"},
                        "expected_result_contract": {"type": "object"},
                        "retained_references": {
                            "type": "object",
                            "properties": {
                                "source_ids": {
                                    "type": "array",
                                    "items": {"type": "string", "minLength": 1},
                                    "uniqueItems": True,
                                    "maxItems": 16,
                                },
                                "resource_ids": {
                                    "type": "array",
                                    "items": {"type": "string", "minLength": 1},
                                    "uniqueItems": True,
                                    "maxItems": 16,
                                },
                                "connector_binding_ids": {
                                    "type": "array",
                                    "items": {"type": "string", "minLength": 1},
                                    "uniqueItems": True,
                                    "maxItems": 1,
                                },
                            },
                            "additionalProperties": False,
                        },
                        "effect_grant": {
                            "type": "object",
                            "properties": {"constraints": {"type": "object"}},
                            "required": ["constraints"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "capability_id",
                        "arguments",
                        "expected_result_contract",
                        "retained_references",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": [
                "objective",
                "outcome_contract",
                "deadline_seconds",
                "initial_task",
            ],
            "additionalProperties": False,
        },
        output_kind=START_GRAPH_JOB_OUTPUT_KIND,
        output_schema={"type": "object", "additionalProperties": True},
        executor_id=START_GRAPH_JOB_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.SUBMIT_EXECUTION_GRAPH,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        durable_foreground_entrypoint=True,
    )
    view = ToolView(
        name=START_GRAPH_JOB_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        presentation=ToolPresentation(
            toolbox_id=ToolboxId.JOBS,
            load_mode=ToolLoadMode.ON_DEMAND,
            text_trust=ToolTextTrust.CODE,
            summary=capability.description,
            when_to_use=(
                "Use for explicit durable intent or a graph-eligible call rejected "
                "with durable_execution_required."
            ),
            keywords=("background", "durable", "graph", "job"),
        ),
    )
    declarations = CapabilityDeclarations(
        domain_owner_id=GRAPH_ADMISSION_DOMAIN_OWNER_ID,
        capabilities=(capability,),
        executor_ids=(START_GRAPH_JOB_EXECUTOR_ID,),
        tool_views=(view,),
    )
    return declarations, (
        StartGraphJobExecutor(owner=owner, resolver=resolver, builder=builder),
    )


def _reference_ids(references: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw = references.get(key, ())
    if not isinstance(raw, tuple) or any(not isinstance(item, str) for item in raw):
        raise CapabilityInputError(
            "graph_initial_task_invalid",
            "The initial task retained references are malformed.",
        )
    values = tuple(sorted(raw))
    if len(values) != len(set(values)):
        raise CapabilityInputError(
            "graph_initial_task_invalid",
            "The initial task retained references must be distinct.",
        )
    return values


__all__ = [
    "GRAPH_ADMISSION_DOMAIN_OWNER_ID",
    "START_GRAPH_JOB_CAPABILITY_ID",
    "START_GRAPH_JOB_OUTPUT_KIND",
    "START_GRAPH_JOB_TOOL_NAME",
    "GraphAdmissionBuilder",
    "GraphAdmissionCapabilityDomain",
    "InitialTaskProposal",
    "InitialTaskProposalResolver",
    "RegistryInitialTaskProposalResolver",
    "graph_admission_declarations",
]
