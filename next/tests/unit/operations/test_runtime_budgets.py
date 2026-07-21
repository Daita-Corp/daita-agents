from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityExecutionError,
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
)
from daita.loop.models import LoopBudgets, LoopExitKind, LoopPhase
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    OperationStatus,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime

NOW = datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc)


class HangingExecutor:
    executor_id = "budget.runtime.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("the runtime timeout must cancel this executor")


class ExecutorOwnedTimeout:
    executor_id = "budget.runtime.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        raise TimeoutError("executor raised its own timeout")


BudgetExecutor = HangingExecutor | ExecutorOwnedTimeout


def _registry(executor: BudgetExecutor) -> CapabilityRegistry:
    capability = Capability(
        id="budget.runtime.read",
        owner="loop-lab",
        description="Exercise the operation-runtime timeout boundary.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="budget.runtime.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        executor_id=executor.executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_runtime_budget",
                capability_id=capability.id,
                description="Read through the runtime timeout boundary.",
            ),
        ),
    )


def _trigger(name: str) -> AgentTrigger:
    return AgentTrigger(
        id=f"trigger-{name}",
        agent_id="agent-runtime-budget",
        kind=TriggerKind.USER,
        source_id="user-runtime-budget",
        payload={"message": name},
        created_at=NOW,
    )


async def _commit_tool_call(
    runtime: OperationRuntime,
    registry: CapabilityRegistry,
    budgets: LoopBudgets,
    *,
    name: str,
) -> tuple[str, ActionProposal]:
    started = await runtime.begin(_trigger(name), budgets=budgets)
    operation_id = started.operation.id
    turn = await runtime.begin_turn(operation_id)
    tools = registry.tool_definitions()
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id=started.operation.agent_id,
                operation_id=operation_id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Run the timeout test action."),),
            ),
        ),
        tools=tools,
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:runtime-budget",
        request,
    )
    call = ToolCall(
        id=f"call-{name}",
        name="read_runtime_budget",
        arguments={"key": "alpha"},
    )
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        ModelResponse(
            tool_calls=(call,),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    return operation_id, ActionProposal(
        operation_id=operation_id,
        turn_id=turn.id,
        call_id=call.id,
        capability_id="budget.runtime.read",
        arguments=call.arguments,
        proposed_at=NOW,
    )


async def test_operation_bound_timeout_fails_task_without_accepting_evidence() -> None:
    executor = HangingExecutor()
    budgets = LoopBudgets(task_timeout_seconds=0.01)
    registry = _registry(executor)
    runtime = OperationRuntime(
        capabilities=registry,
        clock=lambda: NOW,
    )
    operation_id, proposal = await _commit_tool_call(
        runtime,
        registry,
        budgets,
        name="bound-timeout",
    )

    with pytest.raises(CapabilityExecutionError, match="exceeded.*seconds"):
        await asyncio.wait_for(runtime.submit(proposal), timeout=1.0)

    snapshot = await runtime.inspect(operation_id)
    assert snapshot.budgets == budgets
    assert snapshot.operation.status is OperationStatus.RUNNING
    assert len(executor.requests) == 1
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "task_timeout"
    assert snapshot.tasks[0].evidence_ids == ()
    assert len(snapshot.task_leases) == 1
    assert snapshot.task_leases[0].started_at is not None
    assert snapshot.task_leases[0].released_at is not None
    assert snapshot.task_leases[0].release_reason == "task_timeout"
    assert snapshot.evidence == ()
    task_failure_events = [
        event
        for event in snapshot.events
        if event.type in {"task.failed", "executor.failed"}
    ]
    assert [event.type for event in task_failure_events] == [
        "task.failed",
        "executor.failed",
    ]
    assert all(
        event.payload["error_code"] == "task_timeout" for event in task_failure_events
    )


async def test_executor_raised_timeout_remains_an_executor_failure() -> None:
    executor = ExecutorOwnedTimeout()
    budgets = LoopBudgets(task_timeout_seconds=1.0)
    registry = _registry(executor)
    runtime = OperationRuntime(
        capabilities=registry,
        clock=lambda: NOW,
    )
    operation_id, proposal = await _commit_tool_call(
        runtime,
        registry,
        budgets,
        name="executor-timeout",
    )

    with pytest.raises(CapabilityExecutionError, match="executor failed"):
        await runtime.submit(proposal)

    snapshot = await runtime.inspect(operation_id)
    assert len(executor.requests) == 1
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "executor_failed"
    assert len(snapshot.task_leases) == 1
    assert snapshot.task_leases[0].started_at is not None
    assert snapshot.task_leases[0].released_at is not None
    assert snapshot.task_leases[0].release_reason == "executor_failed"
    assert snapshot.evidence == ()
    assert not any(
        event.type == "budget.exhausted"
        or event.payload.get("error_code") == "task_timeout"
        for event in snapshot.events
    )


async def test_fail_budget_commits_inspectable_facts_and_terminal_state() -> None:
    budgets = LoopBudgets(max_turns=2)
    runtime = OperationRuntime(clock=lambda: NOW)
    started = await runtime.begin(_trigger("budget-facts"), budgets=budgets)
    turn = await runtime.begin_turn(started.operation.id)

    result = await runtime.fail_budget(
        started.operation.id,
        "turn_budget_exhausted",
        budget="turns",
        limit=2,
        used=2,
        turn_id=turn.id,
    )

    snapshot = await runtime.inspect(started.operation.id)
    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "turn_budget_exhausted"
    assert snapshot.operation.status is OperationStatus.FAILED
    assert snapshot.operation.terminal_reason == "turn_budget_exhausted"
    assert snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert snapshot.budgets == budgets
    assert [event.type for event in snapshot.events[-2:]] == [
        "budget.exhausted",
        "operation.failed",
    ]
    budget_event = snapshot.events[-2]
    assert budget_event.turn_id == turn.id
    assert dict(budget_event.payload) == {
        "budget": "turns",
        "limit": 2,
        "reason": "turn_budget_exhausted",
        "used": 2,
    }
    assert snapshot.events[-1].payload["reason"] == "turn_budget_exhausted"


async def test_fail_budget_state_and_events_commit_atomically() -> None:
    sequence = 0
    terminal_event_count = 0
    inject_failure = False

    def fail_operation_event(prefix: str) -> str:
        nonlocal sequence, terminal_event_count
        sequence += 1
        if inject_failure and prefix == "event":
            terminal_event_count += 1
            if terminal_event_count == 2:
                raise RuntimeError("injected budget terminal event failure")
        return f"{prefix}-{sequence}"

    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=fail_operation_event,
    )
    started = await runtime.begin(
        _trigger("budget-atomicity"),
        budgets=LoopBudgets(max_actions=1),
    )
    before = await runtime.inspect(started.operation.id)
    inject_failure = True

    with pytest.raises(RuntimeError, match="injected budget terminal event failure"):
        await runtime.fail_budget(
            started.operation.id,
            "action_budget_exhausted",
            budget="actions",
            limit=1,
            used=1,
        )

    assert await runtime.inspect(started.operation.id) == before


async def test_fail_required_context_commits_exact_safe_facts_and_terminal_state() -> (
    None
):
    runtime = OperationRuntime(clock=lambda: NOW)
    started = await runtime.begin(_trigger("context-overflow"))
    turn = await runtime.begin_turn(started.operation.id)

    result = await runtime.fail_required_context(
        started.operation.id,
        profile_id="mock:context-overflow",
        input_limit_tokens=100,
        output_reserve_tokens=20,
        tool_tokens=10,
        required_system_tokens=10,
        required_routing_tokens=5,
        required_intent_tokens=5,
        current_operation_envelope_tokens=20,
        current_operation_body_tokens=10,
        minimum_session_tokens=20,
        projected_session_tokens=30,
        required_tokens=70,
        available_tokens=60,
        total_required_tokens=80,
        optional_omitted_tokens=40,
    )

    snapshot = await runtime.inspect(started.operation.id)
    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "context.required_overflow"
    assert snapshot.operation.status is OperationStatus.FAILED
    assert snapshot.operation.terminal_reason == "context.required_overflow"
    assert [event.type for event in snapshot.events[-2:]] == [
        "context.required_overflow",
        "operation.failed",
    ]
    overflow = snapshot.events[-2]
    assert overflow.turn_id == turn.id
    assert dict(overflow.payload) == {
        "available_tokens": 60,
        "code": "context.required_overflow",
        "current_operation_body_tokens": 10,
        "current_operation_envelope_tokens": 20,
        "input_limit_tokens": 100,
        "minimum_session_tokens": 20,
        "optional_omitted_tokens": 40,
        "output_reserve_tokens": 20,
        "profile_id": "mock:context-overflow",
        "projected_session_tokens": 30,
        "required_intent_tokens": 5,
        "required_routing_tokens": 5,
        "required_system_tokens": 10,
        "required_tokens": 70,
        "schema_version": 1,
        "tool_tokens": 10,
        "total_required_tokens": 80,
    }
    assert dict(snapshot.events[-1].payload) == {"reason": "context.required_overflow"}


async def test_fail_required_context_state_and_events_commit_atomically() -> None:
    sequence = 0
    terminal_event_count = 0
    inject_failure = False

    def fail_operation_event(prefix: str) -> str:
        nonlocal sequence, terminal_event_count
        sequence += 1
        if inject_failure and prefix == "event":
            terminal_event_count += 1
            if terminal_event_count == 2:
                raise RuntimeError("injected context terminal event failure")
        return f"{prefix}-{sequence}"

    runtime = OperationRuntime(clock=lambda: NOW, id_factory=fail_operation_event)
    started = await runtime.begin(_trigger("context-overflow-atomicity"))
    await runtime.begin_turn(started.operation.id)
    before = await runtime.inspect(started.operation.id)
    inject_failure = True

    with pytest.raises(RuntimeError, match="injected context terminal event failure"):
        await runtime.fail_required_context(
            started.operation.id,
            profile_id="mock:context-overflow",
            input_limit_tokens=100,
            output_reserve_tokens=20,
            tool_tokens=10,
            required_system_tokens=10,
            required_routing_tokens=5,
            required_intent_tokens=5,
            current_operation_envelope_tokens=20,
            current_operation_body_tokens=10,
            minimum_session_tokens=20,
            projected_session_tokens=30,
            required_tokens=70,
            available_tokens=60,
            total_required_tokens=80,
            optional_omitted_tokens=None,
        )

    assert await runtime.inspect(started.operation.id) == before
