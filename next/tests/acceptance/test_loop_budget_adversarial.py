from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import pytest

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import LoopBudgets, LoopExitKind, LoopPhase, Readiness, Turn
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import (
    ModelCallStatus,
    OperationRuntime,
    OperationSnapshot,
    TaskExecutionTimeout,
)

NOW = datetime(2026, 7, 17, 14, 0, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self) -> None:
        self.current = NOW

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class DeadlineCrossingIdFactory:
    """Advance the runtime clock while the task-created event is assembled."""

    def __init__(self, clock: MutableClock, *, deadline_seconds: float) -> None:
        self._clock = clock
        self._deadline_seconds = deadline_seconds
        self._counts: dict[str, int] = {}
        self._event_count = 0

    def __call__(self, prefix: str) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        if prefix == "event":
            self._event_count += 1
            # The first six events end at model_response.recorded. Event seven
            # is task.created, after the driver calculated remaining wall time
            # but before the runtime is allowed to invoke the executor.
            if self._event_count == 7:
                self._clock.advance(self._deadline_seconds)
        return f"{prefix}-{self._counts[prefix]}"


def _trigger(name: str) -> AgentTrigger:
    return AgentTrigger(
        id=f"trigger-{name}",
        agent_id="agent-adversarial-budget",
        kind=TriggerKind.USER,
        source_id="user-adversarial-budget",
        payload={"message": name},
        created_at=NOW,
    )


def _capability() -> Capability:
    return Capability(
        id="adversarial.read",
        owner="loop-lab",
        description="Read through an adversarial timeout boundary.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="adversarial.read.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        executor_id="adversarial.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


class CancellationSuppressingExecutor:
    executor_id = "adversarial.read.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A late result must never become accepted evidence merely because
            # an adapter incorrectly suppresses deadline cancellation.
            return EvidenceCandidate(
                kind="adversarial.read.result",
                schema_version=1,
                payload={"value": "late"},
            )
        raise AssertionError("the executor should only return after cancellation")


class ImmediateExecutor:
    executor_id = "adversarial.read.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        return EvidenceCandidate(
            kind="adversarial.read.result",
            schema_version=1,
            payload={"value": "on-time"},
        )


AdversarialExecutor = CancellationSuppressingExecutor | ImmediateExecutor


def _registry(executor: AdversarialExecutor) -> CapabilityRegistry:
    capability = _capability()
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_adversarial_value",
                capability_id=capability.id,
                description="Read one adversarial timeout-test value.",
            ),
        ),
    )


class StaticContextBuilder:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Exercise the adversarial budget boundary."),),
                ),
            ),
            tools=tools,
        )


class AdversarialDomain:
    def __init__(
        self,
        *,
        registry: CapabilityRegistry | None = None,
        clock: MutableClock | None = None,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self.readiness_calls = 0

    async def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        if self._registry is None:
            return ()
        return self._registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        assert self._registry is not None
        view, capability = self._registry.resolve_tool(call.name)
        arguments = self._registry.validate_arguments(capability.id, call.arguments)
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=arguments,
            proposed_at=(self._clock() if self._clock is not None else NOW),
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        payload: Mapping[str, object] = evidence.payload
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            call_id="call-read",
            code="adversarial.read.succeeded",
            message="Adversarial read completed.",
            payload=payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=(self._clock() if self._clock is not None else NOW),
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        self.readiness_calls += 1
        return Readiness(
            allowed=True,
            code="readiness.ready",
            message="The answer is ready.",
            evaluated_at=(self._clock() if self._clock is not None else NOW),
        )


class CancellationSuppressingProvider:
    provider_id = "mock:cancellation-suppressing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The timeout owns this cancellation. Returning a late response
            # must not let the operation complete past its wall-time budget.
            return ModelResponse(
                text="A response returned after the wall deadline.",
                finish_reason=FinishReason.STOP,
            )
        raise AssertionError("the provider should only return after cancellation")


def _tool_response() -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-read",
                name="read_adversarial_value",
                arguments={"key": "alpha"},
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
    )


async def _commit_proposal(
    runtime: OperationRuntime,
    registry: CapabilityRegistry,
    budgets: LoopBudgets,
) -> tuple[str, ActionProposal]:
    started = await runtime.begin(_trigger("runtime-timeout"), budgets=budgets)
    operation_id = started.operation.id
    turn = await runtime.begin_turn(operation_id)
    request = await StaticContextBuilder().build(
        started,
        turn,
        registry.tool_definitions(),
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:runtime-timeout",
        request,
    )
    response = _tool_response()
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        response,
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    call = response.tool_calls[0]
    return operation_id, ActionProposal(
        operation_id=operation_id,
        turn_id=turn.id,
        call_id=call.id,
        capability_id="adversarial.read",
        arguments=call.arguments,
        proposed_at=NOW,
    )


async def test_provider_cannot_suppress_wall_timeout_and_complete_late() -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = CancellationSuppressingProvider()
    domain = AdversarialDomain()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=LoopBudgets(max_wall_time_seconds=0.01),
    )

    result = await asyncio.wait_for(
        loop.run(_trigger("provider-suppression")),
        timeout=1.0,
    )
    snapshot = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "wall_time_budget_exhausted"
    assert snapshot.operation.status is OperationStatus.FAILED
    assert snapshot.operation.final_text is None
    assert snapshot.model_calls[-1].status is ModelCallStatus.FAILED
    assert snapshot.model_calls[-1].response is None
    assert domain.readiness_calls == 0
    assert any(event.type == "budget.exhausted" for event in snapshot.events)


async def test_executor_cannot_suppress_task_timeout_and_publish_evidence() -> None:
    executor = CancellationSuppressingExecutor()
    registry = _registry(executor)
    budgets = LoopBudgets(task_timeout_seconds=0.01)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    operation_id, proposal = await _commit_proposal(runtime, registry, budgets)

    with pytest.raises(TaskExecutionTimeout):
        await asyncio.wait_for(runtime.submit(proposal), timeout=1.0)

    snapshot = await runtime.inspect(operation_id)
    assert len(executor.requests) == 1
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "task_timeout"
    assert snapshot.tasks[0].evidence_ids == ()
    assert snapshot.evidence == ()
    assert not any(event.type == "evidence.accepted" for event in snapshot.events)


async def test_wall_deadline_crossed_during_task_persistence_blocks_executor() -> None:
    clock = MutableClock()
    deadline_seconds = 5.0
    executor = ImmediateExecutor()
    registry = _registry(executor)
    runtime = OperationRuntime(
        capabilities=registry,
        clock=clock,
        id_factory=DeadlineCrossingIdFactory(
            clock,
            deadline_seconds=deadline_seconds,
        ),
    )
    loop = AgentLoop(
        runtime=runtime,
        model=MockModelProvider((_tool_response(),)),
        context_builder=StaticContextBuilder(),
        domain=AdversarialDomain(registry=registry, clock=clock),
        budgets=LoopBudgets(
            max_wall_time_seconds=deadline_seconds,
            task_timeout_seconds=1.0,
        ),
    )

    result = await loop.run(_trigger("wall-crossed-during-task-persistence"))
    snapshot = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "wall_time_budget_exhausted"
    assert executor.requests == []
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "task_timeout"
    assert snapshot.tasks[0].evidence_ids == ()
    assert snapshot.evidence == ()
    assert snapshot.observations == ()
    assert any(event.type == "budget.exhausted" for event in snapshot.events)
