from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal

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
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import LoopBudgets, LoopExitKind, Readiness, Turn
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
)

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self, current: datetime = NOW) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class StaticContextBuilder:
    def __init__(self, *, on_build: Callable[[], None] | None = None) -> None:
        self._on_build = on_build

    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        if self._on_build is not None:
            self._on_build()
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Exercise the configured loop budget."),),
                ),
            ),
            tools=tools,
        )


def _capability() -> Capability:
    return Capability(
        id="budget.read",
        owner="loop-lab",
        description="Read one deterministic value for budget tests.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="budget.read.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        executor_id="budget.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


class BudgetExecutor:
    executor_id = "budget.read.executor"

    def __init__(self, *, hang: bool = False) -> None:
        self.hang = hang
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        if self.hang:
            await asyncio.Event().wait()
        key = request.arguments["key"]
        assert isinstance(key, str)
        return EvidenceCandidate(
            kind="budget.read.result",
            schema_version=1,
            payload={"key": key, "value": key.upper()},
        )


def _registry(executor: BudgetExecutor) -> CapabilityRegistry:
    capability = _capability()
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_budget_value",
                capability_id=capability.id,
                description="Read one deterministic budget-test value.",
            ),
        ),
    )


class BudgetDomain:
    def __init__(
        self,
        *,
        registry: CapabilityRegistry | None = None,
        readiness_allowed: bool = True,
        observation_message: str = "Budget read completed.",
        observation_payload: Mapping[str, object] | None = None,
        clock: Callable[[], datetime] = lambda: NOW,
    ) -> None:
        self.registry = registry
        self.readiness_allowed = readiness_allowed
        self.observation_message = observation_message
        self.observation_payload = observation_payload
        self.clock = clock
        self.validation_calls: list[str] = []
        self.readiness_calls = 0

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        if self.registry is None:
            return ()
        return self.registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        assert self.registry is not None
        self.validation_calls.append(call.id)
        view, capability = self.registry.resolve_tool(call.name)
        arguments = self.registry.validate_arguments(capability.id, call.arguments)
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=arguments,
            proposed_at=self.clock(),
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        payload = (
            evidence.payload
            if self.observation_payload is None
            else self.observation_payload
        )
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            code="budget.read.succeeded",
            message=self.observation_message,
            payload=payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=self.clock(),
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        self.readiness_calls += 1
        return Readiness(
            allowed=self.readiness_allowed,
            code=("readiness.ready" if self.readiness_allowed else "readiness.retry"),
            message=(
                "The answer is ready."
                if self.readiness_allowed
                else "Another model turn is required."
            ),
            missing_facts=(() if self.readiness_allowed else ("another_turn",)),
            evaluated_at=self.clock(),
        )


class HangingModelProvider:
    provider_id = "mock:hanging"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("the wall-time guard must cancel the hanging model")


def _trigger(name: str) -> AgentTrigger:
    return AgentTrigger(
        id=f"trigger-{name}",
        agent_id="agent-budget",
        kind=TriggerKind.USER,
        source_id="user-budget",
        payload={"message": name},
        created_at=NOW,
    )


def _text_response(
    text: str,
    *,
    usage: ModelUsage = ModelUsage(),
) -> ModelResponse:
    return ModelResponse(
        text=text,
        finish_reason=FinishReason.STOP,
        usage=usage,
    )


def _tool_response(
    *calls: ToolCall,
    usage: ModelUsage = ModelUsage(),
) -> ModelResponse:
    return ModelResponse(
        tool_calls=calls,
        finish_reason=FinishReason.TOOL_CALLS,
        usage=usage,
    )


def _budget_event(snapshot: OperationSnapshot, budget: str):
    events = [
        event
        for event in snapshot.events
        if event.type == "budget.exhausted" and event.payload.get("budget") == budget
    ]
    assert len(events) == 1
    return events[0]


async def test_bound_turn_budget_allows_n_turns_and_never_starts_n_plus_one() -> None:
    budgets = LoopBudgets(max_turns=2, max_repairs=8)
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _text_response("Not ready on turn one."),
            _text_response("Not ready on turn two."),
            _text_response("This third response must remain unconsumed."),
        )
    )
    domain = BudgetDomain(readiness_allowed=False)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=budgets,
    )

    result = await loop.run(_trigger("turn-limit"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "turn_budget_exhausted"
    assert final.budgets == budgets
    assert final.loop_state.turn_count == 2
    assert len(final.turns) == 2
    assert len(final.model_calls) == 2
    assert len(provider.requests) == 2
    assert domain.readiness_calls == 2
    event = _budget_event(final, "turns")
    assert event.payload["limit"] == 2
    assert event.payload["used"] == 2
    assert [item.type for item in final.events[-2:]] == [
        "budget.exhausted",
        "operation.failed",
    ]


async def test_action_budget_executes_n_calls_but_never_materializes_n_plus_one() -> (
    None
):
    executor = BudgetExecutor()
    registry = _registry(executor)
    budgets = LoopBudgets(max_actions=1)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _tool_response(
                ToolCall(
                    id="call-allowed",
                    name="read_budget_value",
                    arguments={"key": "alpha"},
                ),
                ToolCall(
                    id="call-blocked",
                    name="read_budget_value",
                    arguments={"key": "beta"},
                ),
            ),
        )
    )
    domain = BudgetDomain(registry=registry)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=budgets,
    )

    result = await loop.run(_trigger("action-limit"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "action_budget_exhausted"
    assert final.loop_state.action_count == 1
    assert len(final.tasks) == 1
    assert final.tasks[0].call_id == "call-allowed"
    assert final.tasks[0].status is TaskStatus.SUCCEEDED
    assert len(final.evidence) == 1
    assert len(final.observations) == 1
    assert [request.arguments["key"] for request in executor.requests] == ["alpha"]
    assert domain.validation_calls == ["call-allowed"]
    assert not any(task.call_id == "call-blocked" for task in final.tasks)
    event = _budget_event(final, "actions")
    assert event.call_id == "call-blocked"
    assert event.payload["limit"] == 1
    assert event.payload["used"] == 1


async def test_token_overrun_commits_usage_but_executes_no_tool() -> None:
    executor = BudgetExecutor()
    registry = _registry(executor)
    budgets = LoopBudgets(max_total_tokens=10)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _tool_response(
                ToolCall(
                    id="call-over-token-budget",
                    name="read_budget_value",
                    arguments={"key": "alpha"},
                ),
                usage=ModelUsage(input_tokens=7, output_tokens=4),
            ),
        )
    )
    domain = BudgetDomain(registry=registry)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=budgets,
    )

    result = await loop.run(_trigger("token-overrun"))
    final = await runtime.inspect(result.operation_id)

    assert result.reason == "token_budget_exhausted"
    assert final.loop_state.input_tokens == 7
    assert final.loop_state.output_tokens == 4
    assert final.loop_state.input_tokens + final.loop_state.output_tokens == 11
    assert final.model_calls[-1].status is ModelCallStatus.COMPLETED
    assert final.tasks == ()
    assert final.evidence == ()
    assert executor.requests == []
    assert domain.validation_calls == []
    assert domain.readiness_calls == 0
    event = _budget_event(final, "total_tokens")
    assert event.payload["limit"] == 10
    assert event.payload["used"] == 11


async def test_cost_overrun_commits_usage_but_never_evaluates_readiness() -> None:
    budgets = LoopBudgets(max_estimated_cost_usd=Decimal("0.01"))
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _text_response(
                "This answer costs too much.",
                usage=ModelUsage(
                    input_tokens=2,
                    output_tokens=2,
                    estimated_cost_usd=Decimal("0.02"),
                ),
            ),
        )
    )
    domain = BudgetDomain()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=budgets,
    )

    result = await loop.run(_trigger("cost-overrun"))
    final = await runtime.inspect(result.operation_id)

    assert result.reason == "estimated_cost_budget_exhausted"
    assert final.loop_state.estimated_cost_usd == Decimal("0.02")
    assert final.readiness == ()
    assert domain.readiness_calls == 0
    event = _budget_event(final, "estimated_cost_usd")
    assert event.payload["limit"] == "0.01"
    assert event.payload["used"] == "0.02"


@pytest.mark.parametrize(
    ("budgets", "usage"),
    [
        pytest.param(
            LoopBudgets(max_total_tokens=10),
            ModelUsage(input_tokens=6, output_tokens=4),
            id="tokens",
        ),
        pytest.param(
            LoopBudgets(max_estimated_cost_usd=Decimal("0.01")),
            ModelUsage(estimated_cost_usd=Decimal("0.01")),
            id="estimated-cost",
        ),
    ],
)
async def test_exact_usage_limit_can_complete_the_current_response(
    budgets: LoopBudgets,
    usage: ModelUsage,
) -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = MockModelProvider((_text_response("Exactly on budget.", usage=usage),))
    domain = BudgetDomain()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=budgets,
    )

    result = await loop.run(_trigger(f"exact-{usage.total_tokens}"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert final.operation.status is OperationStatus.SUCCEEDED
    assert domain.readiness_calls == 1
    assert not any(event.type == "budget.exhausted" for event in final.events)


async def test_exact_token_limit_blocks_a_required_followup_before_model_io() -> None:
    budgets = LoopBudgets(max_total_tokens=10, max_repairs=3)
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _text_response(
                "A correction turn is still required.",
                usage=ModelUsage(input_tokens=6, output_tokens=4),
            ),
            _text_response("This followup must remain unconsumed."),
        )
    )
    domain = BudgetDomain(readiness_allowed=False)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=budgets,
    )

    result = await loop.run(_trigger("exact-token-followup"))
    final = await runtime.inspect(result.operation_id)

    assert result.reason == "token_budget_exhausted"
    assert final.loop_state.turn_count == 1
    assert len(provider.requests) == 1
    assert domain.readiness_calls == 1
    assert len(final.observations) == 1
    event = _budget_event(final, "total_tokens")
    assert event.payload["limit"] == 10
    assert event.payload["used"] == 10


async def test_observation_overrun_keeps_evidence_linkage_then_stops() -> None:
    executor = BudgetExecutor()
    registry = _registry(executor)
    budgets = LoopBudgets(max_observation_characters=16)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _tool_response(
                ToolCall(
                    id="call-large-observation",
                    name="read_budget_value",
                    arguments={"key": "alpha"},
                )
            ),
            _text_response("This synthesis must remain unconsumed."),
        )
    )
    domain = BudgetDomain(
        registry=registry,
        observation_message="The committed observation exceeds the character limit.",
        observation_payload={"summary": "x" * 64},
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=domain,
        budgets=budgets,
    )

    result = await loop.run(_trigger("observation-overrun"))
    final = await runtime.inspect(result.operation_id)

    assert result.reason == "observation_budget_exhausted"
    assert len(provider.requests) == 1
    assert len(final.tasks) == 1
    assert final.tasks[0].status is TaskStatus.SUCCEEDED
    assert len(final.evidence) == 1
    assert len(final.observations) == 1
    assert final.observations[0].task_id == final.tasks[0].id
    assert final.observations[0].evidence_id == final.evidence[0].id
    assert final.loop_state.observation_characters > 16
    event = _budget_event(final, "observation_characters")
    assert event.payload["limit"] == 16
    assert event.payload["used"] == final.loop_state.observation_characters


async def test_wall_time_at_exact_limit_after_context_stops_before_model_io() -> None:
    clock = MutableClock()
    budgets = LoopBudgets(max_wall_time_seconds=5.0)
    runtime = OperationRuntime(clock=clock)
    provider = MockModelProvider((_text_response("Must not be requested."),))
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(on_build=lambda: clock.advance(5.0)),
        domain=BudgetDomain(clock=clock),
        budgets=budgets,
    )

    result = await loop.run(_trigger("wall-context"))
    final = await runtime.inspect(result.operation_id)

    assert result.reason == "wall_time_budget_exhausted"
    assert final.loop_state.turn_count == 1
    assert len(final.turns) == 1
    assert final.model_calls == ()
    assert provider.requests == ()
    event = _budget_event(final, "wall_time_seconds")
    assert event.payload["limit"] == 5.0
    assert event.payload["used"] == 5.0


async def test_hanging_model_is_bounded_by_remaining_wall_time() -> None:
    budgets = LoopBudgets(max_wall_time_seconds=0.01)
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = HangingModelProvider()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=BudgetDomain(),
        budgets=budgets,
    )

    result = await asyncio.wait_for(
        loop.run(_trigger("wall-hanging-model")),
        timeout=1.0,
    )
    final = await runtime.inspect(result.operation_id)

    assert result.reason == "wall_time_budget_exhausted"
    assert len(provider.requests) == 1
    assert final.model_calls[-1].status is ModelCallStatus.FAILED
    assert final.model_calls[-1].error_code == "wall_time_budget_exhausted"
    assert final.model_calls[-1].response is None
    _budget_event(final, "wall_time_seconds")


async def test_executor_timeout_is_terminal_and_never_accepts_evidence() -> None:
    executor = BudgetExecutor(hang=True)
    registry = _registry(executor)
    budgets = LoopBudgets(
        max_wall_time_seconds=1.0,
        task_timeout_seconds=0.01,
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _tool_response(
                ToolCall(
                    id="call-timeout",
                    name="read_budget_value",
                    arguments={"key": "alpha"},
                )
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=BudgetDomain(registry=registry),
        budgets=budgets,
    )

    result = await asyncio.wait_for(
        loop.run(_trigger("task-timeout")),
        timeout=1.0,
    )
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "task_timeout"
    assert len(executor.requests) == 1
    assert len(final.tasks) == 1
    assert final.tasks[0].status is TaskStatus.FAILED
    assert final.tasks[0].error_code == "task_timeout"
    assert final.evidence == ()
    assert final.observations == ()
    event = _budget_event(final, "task_timeout_seconds")
    assert event.task_id == final.tasks[0].id
    assert event.payload["limit"] == 0.01


async def test_remaining_wall_time_bounds_executor_and_wins_terminal_reason() -> None:
    executor = BudgetExecutor(hang=True)
    registry = _registry(executor)
    budgets = LoopBudgets(
        max_wall_time_seconds=0.01,
        task_timeout_seconds=1.0,
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            _tool_response(
                ToolCall(
                    id="call-wall-timeout",
                    name="read_budget_value",
                    arguments={"key": "alpha"},
                )
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=BudgetDomain(registry=registry),
        budgets=budgets,
    )

    result = await asyncio.wait_for(
        loop.run(_trigger("executor-wall-timeout")),
        timeout=1.0,
    )
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "wall_time_budget_exhausted"
    assert len(executor.requests) == 1
    assert len(final.tasks) == 1
    assert final.tasks[0].status is TaskStatus.FAILED
    assert final.tasks[0].error_code == "task_timeout"
    assert final.evidence == ()
    assert final.observations == ()
    event = _budget_event(final, "wall_time_seconds")
    assert event.task_id == final.tasks[0].id
    assert event.payload["limit"] == 0.01
    assert not any(
        event.type == "budget.exhausted"
        and event.payload.get("budget") == "task_timeout_seconds"
        for event in final.events
    )
