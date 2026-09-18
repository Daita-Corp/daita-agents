"""Explicit integration-only composition for the Phase 4 model-task slice."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    Capability,
    CapabilityDeclarations,
    CapabilityRegistry,
    ExecutionAdmissionPolicy,
    ExecutionContractBindings,
    ExecutionContractReader,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime
from daita.distribution.owner import DistributionOwner
from daita.hosting.execution_governor import RunAdmissionCoordinator
from daita.jobs.graph.admission import GraphAdmissionBuilder, InitialTaskProposal
from daita.jobs.graph.capabilities import (
    GRAPH_TASK_DOMAIN_OWNER_ID,
    TASK_BLOCK_CAPABILITY_ID,
    TASK_CHECKPOINT_CAPABILITY_ID,
    TASK_COMMENT_CAPABILITY_ID,
    TASK_COMPLETE_CAPABILITY_ID,
    TASK_REQUEST_REVIEW_CAPABILITY_ID,
    GraphTaskCapabilityDomain,
    graph_task_capability_declarations,
)
from daita.jobs.graph.models import GraphAdmission, GraphInspection, GraphState
from daita.jobs.owner import JobOwner
from daita.jobs.supervisor import JobSupervisor
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import LoopLimits
from daita.storage.sqlite import SQLiteStateStore
from tests.support.capability_runtime import StaticTestDomain
from tests.support.loop import TranscriptContext
from tests.support.static_graph_integration import DeterministicIds

AGENT_ID = "agent-model-graph"
CONVERSATION_ID = "conversation-model-graph"
READ_CAPABILITY_ID = "test.graph.read"
READ_TOOL_NAME = "graph_read"
MODEL_ROUTE_ID = "mock:model-graph"
_MODEL_CAPABILITY_IDS = tuple(
    sorted(
        (
            READ_CAPABILITY_ID,
            TASK_BLOCK_CAPABILITY_ID,
            TASK_CHECKPOINT_CAPABILITY_ID,
            TASK_COMMENT_CAPABILITY_ID,
            TASK_COMPLETE_CAPABILITY_ID,
            TASK_REQUEST_REVIEW_CAPABILITY_ID,
        )
    )
)


def _digest(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


class GraphReadExecutor:
    executor_id = "test.graph.read.executor"

    def __init__(self) -> None:
        self.calls: list[ToolExecution] = []

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.calls.append(request)
        return ToolOutput(
            kind="test.graph.read",
            data={"resource_ids": request.arguments["resource_ids"], "value": 42},
            sensitivity=request.request_sensitivity,
            sensitivity_provenance={"authority": "deterministic_test_read"},
        )


def graph_read_declarations() -> tuple[CapabilityDeclarations, GraphReadExecutor]:
    capability = Capability(
        id=READ_CAPABILITY_ID,
        description="Return one deterministic bounded read result.",
        input_schema={
            "type": "object",
            "properties": {
                "resource_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 16,
                    "uniqueItems": True,
                }
            },
            "required": ["resource_ids"],
            "additionalProperties": False,
        },
        output_kind="test.graph.read",
        output_schema={"type": "object", "additionalProperties": True},
        executor_id=GraphReadExecutor.executor_id,
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        execution_admission_policy=ExecutionAdmissionPolicy(
            shape="relational_read",
            inline_eligible=True,
            graph_v1_eligible=True,
            target_count_argument="resource_ids",
            inline_max_targets=4,
            graph_max_targets=16,
        ),
    )
    view = ToolView(
        name=READ_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
        presentation=ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.PINNED,
            text_trust=ToolTextTrust.CODE,
            summary=capability.description,
            when_to_use="Use only for the exact admitted graph read.",
            keywords=("graph", "read", "test"),
        ),
    )
    declarations = CapabilityDeclarations(
        domain_owner_id="test.graph",
        capabilities=(capability,),
        executor_ids=(capability.executor_id,),
        tool_views=(view,),
    )
    return declarations, GraphReadExecutor()


@dataclass(slots=True)
class ModelGraphIntegration:
    store: SQLiteStateStore
    artifacts: AgentHomeArtifactStore
    coordinator: RunAdmissionCoordinator
    distribution: DistributionOwner
    owner: JobOwner
    registry: CapabilityRegistry
    contracts: ExecutionContractReader
    contract_state: dict[str, bool]
    runtime: CapabilityRuntime
    supervisor: JobSupervisor
    builder: GraphAdmissionBuilder
    reader: GraphReadExecutor
    provider: MockModelProvider
    clock: Callable[[], datetime]
    ids: DeterministicIds

    @classmethod
    async def open(
        cls,
        root: Path,
        *,
        script: tuple[ModelResponse | Exception, ...] | None = None,
        namespace: str = "model-graph",
        clock: Callable[[], datetime] | None = None,
        loop_max_total_tokens: int = 10_000,
    ) -> ModelGraphIntegration:
        resolved_clock = clock or (lambda: datetime.now(UTC))
        ids = DeterministicIds(namespace)
        store = await SQLiteStateStore.open_draft_graph(
            root / "state.sqlite", initialize=True, clock=resolved_clock
        )
        artifacts = await AgentHomeArtifactStore.open(
            agent_id=AGENT_ID,
            agent_home=root,
            references=store,
            clock=resolved_clock,
            id_factory=ids,
        )
        owner = JobOwner(
            agent_id=AGENT_ID,
            store=store,
            clock=resolved_clock,
            id_factory=ids,
        )
        distribution = DistributionOwner(agent_id=AGENT_ID, store=store)
        business, reader = graph_read_declarations()
        lifecycle = graph_task_capability_declarations(
            owner,
            store,
            clock=resolved_clock,
            id_factory=ids,
        )
        lifecycle_bundle = CapabilityDeclarations(
            domain_owner_id=GRAPH_TASK_DOMAIN_OWNER_ID,
            capabilities=lifecycle.capabilities,
            executor_ids=tuple(item.executor_id for item in lifecycle.capabilities),
            tool_views=lifecycle.tool_views,
        )
        registry = CapabilityRegistry(
            declarations=(business, lifecycle_bundle),
            executors=(reader, *lifecycle.executors),
        )
        contract_state = {"revoked": False}

        async def contracts(
            *,
            agent_id: str,
            source_ids: tuple[str, ...],
            resource_ids: tuple[str, ...],
            capability_ids: tuple[str, ...],
            connector_binding_ids: tuple[str, ...],
            model_route_ids: tuple[str, ...],
        ) -> ExecutionContractBindings:
            assert agent_id == AGENT_ID
            assert source_ids == connector_binding_ids == ()
            return ExecutionContractBindings(
                capability_contracts={
                    item: (
                        _digest(f"revoked:{item}")
                        if contract_state["revoked"]
                        else registry.contract_digest(item)
                    )
                    for item in capability_ids
                },
                resource_revisions={item: _digest(item) for item in resource_ids},
                model_routes={item: _digest(item) for item in model_route_ids},
            )

        business_domain = StaticTestDomain(
            business.capabilities,
            business.tool_views,
            domain_owner_id=business.domain_owner_id,
        )
        lifecycle_domain = GraphTaskCapabilityDomain(lifecycle_bundle, owner)
        coordinator = RunAdmissionCoordinator(
            execution_capacity=5,
            foreground_execution_reserve=1,
            provider_capacity=2,
            foreground_provider_reserve=1,
            sqlite_pressure_capacity=4,
        )
        runtime = CapabilityRuntime(
            registry,
            (business_domain, lifecycle_domain),
            artifacts=artifacts,
            clock=resolved_clock,
            admission_coordinator=coordinator,
            effect_coordinator=coordinator.effect_coordinator,
            execution_contract_reader=contracts,
        )
        responses = script or _successful_script()
        provider = MockModelProvider(
            responses,
            provider_id=MODEL_ROUTE_ID,
            complete_pricing=True,
        )
        loop = AgentLoop(
            model=provider,
            context_builder=TranscriptContext(),
            tools=runtime,
            transcripts=store,
            limits=LoopLimits(
                max_steps=6,
                max_total_tokens=loop_max_total_tokens,
                max_estimated_cost_usd=Decimal(1),
            ),
            clock=resolved_clock,
        )
        supervisor = JobSupervisor(
            agent_id=AGENT_ID,
            store=store,
            owner=owner,
            runtime=runtime,
            revalidate_external=_unused_external_revalidation,
            artifacts=artifacts,
            clock=resolved_clock,
            id_factory=ids,
            admission_coordinator=coordinator,
            distribution=distribution,
            graph_parallelism=4,
            graph_model_loop=loop,
            poll_seconds=0.005,
        )
        owner.bind_wake(supervisor.wake)
        builder = GraphAdmissionBuilder(
            agent_id=AGENT_ID,
            registry=registry,
            distribution=distribution,
            clock=resolved_clock,
            id_factory=ids,
        )
        return cls(
            store,
            artifacts,
            coordinator,
            distribution,
            owner,
            registry,
            contracts,
            contract_state,
            runtime,
            supervisor,
            builder,
            reader,
            provider,
            resolved_clock,
            ids,
        )

    def proposal(
        self,
        *,
        resource_count: int = 5,
        model_route_id: str = MODEL_ROUTE_ID,
        per_run_max_tokens: int = 10_000,
        per_run_max_cost_usd: Decimal = Decimal(1),
    ) -> InitialTaskProposal:
        resource_ids = tuple(f"resource-{index}" for index in range(resource_count))
        capability_ids = _MODEL_CAPABILITY_IDS
        bindings = ExecutionContractBindings(
            capability_contracts={
                item: self.registry.contract_digest(item) for item in capability_ids
            },
            resource_revisions={item: _digest(item) for item in resource_ids},
            model_routes={model_route_id: _digest(model_route_id)},
        )
        capability = self.registry.capability(READ_CAPABILITY_ID)
        policy = capability.execution_admission_policy
        assert policy is not None
        return InitialTaskProposal(
            capability_id=READ_CAPABILITY_ID,
            arguments={"resource_ids": resource_ids},
            expected_result_contract={"result_kind": "test.graph.read"},
            source_ids=(),
            resource_ids=resource_ids,
            connector_binding_ids=(),
            access_mode=AccessMode.READ,
            model_route_id=model_route_id,
            sensitivity=ModelSensitivity.INTERNAL,
            contract_bindings=bindings,
            execution_policy=policy,
            max_steps=6,
            max_wall_time_seconds=300,
            per_run_max_tokens=per_run_max_tokens,
            per_run_max_cost_usd=per_run_max_cost_usd,
        )

    def build(
        self,
        *,
        resource_count: int = 5,
        model_route_id: str = MODEL_ROUTE_ID,
        per_run_max_tokens: int = 10_000,
        per_run_max_cost_usd: Decimal = Decimal(1),
    ) -> GraphAdmission:
        return self.builder.build(
            run_id="run-" + "a" * 32,
            call_id="start-model-graph",
            conversation_id=CONVERSATION_ID,
            objective="Read the exact admitted resources.",
            outcome_contract={"required_result_kind": "test.graph.read"},
            deadline_seconds=300,
            proposal=self.proposal(
                resource_count=resource_count,
                model_route_id=model_route_id,
                per_run_max_tokens=per_run_max_tokens,
                per_run_max_cost_usd=per_run_max_cost_usd,
            ),
        )

    async def admit_and_start(self, admission: GraphAdmission) -> None:
        await self.owner.admit_static_graph(admission)
        await self.supervisor.start_graph_integration()

    async def wait_terminal(
        self, job_id: str, *, timeout: float = 5.0
    ) -> GraphInspection:
        deadline = asyncio.get_running_loop().time() + timeout
        inspection = None
        while asyncio.get_running_loop().time() < deadline:
            inspection = await self.owner.inspect_graph(job_id)
            assert inspection is not None
            if inspection.job.state in {
                GraphState.SUCCEEDED,
                GraphState.FAILED,
                GraphState.BLOCKED,
                GraphState.NEEDS_ATTENTION,
            }:
                return inspection
            await asyncio.sleep(0.005)
        driver = self.supervisor._graph_driver
        error = None if driver is None or not driver.done() else driver.exception()
        raise AssertionError(
            f"graph did not terminate: {inspection!r}; driver={error!r}"
        )

    async def close(self) -> None:
        await self.supervisor.close()
        await self.artifacts.close()
        await self.store.close()


def _successful_script() -> tuple[ModelResponse, ...]:
    usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
    return (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="read-call",
                    name=READ_TOOL_NAME,
                    arguments={
                        "resource_ids": tuple(f"resource-{index}" for index in range(5))
                    },
                ),
            ),
            usage=usage,
        ),
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="complete-call",
                    name="task_complete",
                    arguments={
                        "result_kind": "test.graph.read",
                        "summary": "The deterministic read completed.",
                        "payload": {"value": 42},
                        "evidence_call_ids": ("read-call",),
                        "artifact_ids": (),
                        "residual_risk": None,
                        "downstream_constraints": {},
                    },
                ),
            ),
            usage=usage,
        ),
    )


async def _unused_external_revalidation(_job: object) -> None:
    raise AssertionError("model graph integration cannot call the legacy job engine")


__all__ = [
    "AGENT_ID",
    "CONVERSATION_ID",
    "MODEL_ROUTE_ID",
    "READ_CAPABILITY_ID",
    "READ_TOOL_NAME",
    "GraphReadExecutor",
    "ModelGraphIntegration",
    "graph_read_declarations",
]
