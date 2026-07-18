from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    Executor,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.events.models import RuntimeEvent
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
from daita.loop.models import LoopBudgets, LoopExit, LoopPhase, Readiness, Turn
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
    OperationStateError,
)
from daita.operations.store import CommitResult, InMemoryOperationStore

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)


def _trigger(name: str) -> AgentTrigger:
    return AgentTrigger(
        id=f"trigger-{name}",
        agent_id="agent-cancellation",
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"message": "Exercise the cancellation boundary."},
        created_at=NOW,
    )


def _capability() -> Capability:
    return Capability(
        id="fake.read",
        owner="loop-lab",
        description="Read one deterministic fake value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="fake.read.result",
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
        executor_id="fake.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


def _registry(executor: Executor) -> CapabilityRegistry:
    capability = _capability()
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_fake_value",
                capability_id=capability.id,
                description="Read one fake value by key.",
            ),
        ),
    )


class SingleTurnContextBuilder:
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
                    content=(TextBlock("Exercise the cancellation boundary."),),
                ),
            ),
            tools=tools,
        )


class TextOnlyDomain:
    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("text-only cancellation must not validate an action")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only cancellation must not project evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        raise AssertionError("a cancelled model call cannot reach readiness")


class BlockingProvider:
    provider_id = "mock:blocking-cancellation"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.operation_id: str | None = None
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.operation_id = request.operation_id
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.finished.set()
        raise AssertionError("the blocking provider must be cancelled")


class CancellationSuppressingProvider(BlockingProvider):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.operation_id = request.operation_id
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.finished.set()
            return ModelResponse(
                text="A late response after caller cancellation.",
                finish_reason=FinishReason.STOP,
            )
        raise AssertionError("the provider should only return after cancellation")


class FakeReadDomain:
    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        block_projection: bool = False,
    ) -> None:
        self._registry = registry
        self._block_projection = block_projection
        self.projection_started = asyncio.Event()
        self.projection_finished = asyncio.Event()

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return self._registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        view, capability = self._registry.resolve_tool(call.name)
        arguments = self._registry.validate_arguments(capability.id, call.arguments)
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=arguments,
            proposed_at=NOW,
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        if self._block_projection:
            self.projection_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.projection_finished.set()
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            call_id="call-read",
            code="fake.read.succeeded",
            message="Fake read completed.",
            payload=evidence.payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=NOW,
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        raise AssertionError("cancelled fake-read loops cannot reach readiness")


class BlockingReadExecutor:
    executor_id = "fake.read.executor"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.finished.set()
        raise AssertionError("the blocking executor must be cancelled")


class CancellationSuppressingReadExecutor(BlockingReadExecutor):
    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.finished.set()
            return EvidenceCandidate(
                kind="fake.read.result",
                schema_version=1,
                payload={"key": "alpha", "value": "LATE"},
            )
        raise AssertionError("the executor should only return after cancellation")


class ImmediateReadExecutor:
    executor_id = "fake.read.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        key = request.arguments["key"]
        assert isinstance(key, str)
        return EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": key, "value": key.upper()},
        )


def _tool_response() -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-read",
                name="read_fake_value",
                arguments={"key": "alpha"},
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=ModelUsage(input_tokens=5, output_tokens=3),
    )


async def test_cancellation_during_model_io_persists_intent_and_reraises() -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = BlockingProvider()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SingleTurnContextBuilder(),
        domain=TextOnlyDomain(),
    )
    running = asyncio.create_task(loop.run(_trigger("model-io")))

    await provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert provider.finished.is_set()
    assert provider.operation_id is not None
    snapshot = await runtime.inspect(provider.operation_id)
    model_call = snapshot.model_calls[-1]

    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.operation.terminal_reason == "run_cancelled"
    assert snapshot.operation.final_text is None
    assert snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert snapshot.loop_state.interruption_reason == "run_cancelled"
    assert model_call.status is ModelCallStatus.STARTED
    assert model_call.cancellation_requested is True
    assert model_call.response is None
    assert model_call.error_code is None
    assert snapshot.tasks == ()
    assert snapshot.evidence == ()
    assert snapshot.observations == ()
    assert len(provider.requests) == 1
    assert [event.type for event in snapshot.events] == [
        "trigger.received",
        "operation.created",
        "turn.created",
        "context.built",
        "model_call.started",
        "model_call.cancellation_requested",
        "operation.interrupted",
    ]
    child_event, operation_event = snapshot.events[-2:]
    assert child_event.turn_id == model_call.turn_id
    assert child_event.payload["model_call_id"] == model_call.id
    assert child_event.payload["reason"] == "run_cancelled"
    assert operation_event.payload["reason"] == "run_cancelled"


async def test_provider_cannot_suppress_caller_cancellation() -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = CancellationSuppressingProvider()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SingleTurnContextBuilder(),
        domain=TextOnlyDomain(),
    )
    running = asyncio.create_task(loop.run(_trigger("suppressed-model-cancel")))

    await provider.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert provider.operation_id is not None
    snapshot = await runtime.inspect(provider.operation_id)
    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.model_calls[-1].status is ModelCallStatus.STARTED
    assert snapshot.model_calls[-1].cancellation_requested is True
    assert snapshot.model_calls[-1].response is None
    assert not any(event.type == "model_response.recorded" for event in snapshot.events)


async def test_cancellation_during_executor_io_preserves_running_task_intent() -> None:
    executor = BlockingReadExecutor()
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider((_tool_response(),))
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SingleTurnContextBuilder(),
        domain=FakeReadDomain(registry),
    )
    running = asyncio.create_task(loop.run(_trigger("executor-io")))

    await executor.started.wait()
    operation_id = executor.requests[0].operation_id
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert executor.finished.is_set()
    snapshot = await runtime.inspect(operation_id)
    task = snapshot.tasks[-1]

    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.operation.terminal_reason == "run_cancelled"
    assert snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert snapshot.loop_state.interruption_reason == "run_cancelled"
    assert snapshot.model_calls[-1].status is ModelCallStatus.COMPLETED
    assert snapshot.model_calls[-1].cancellation_requested is False
    assert task.status is TaskStatus.RUNNING
    assert task.cancellation_requested is True
    assert task.evidence_ids == ()
    assert task.error_code is None
    assert snapshot.evidence == ()
    assert snapshot.observations == ()
    assert len(snapshot.task_leases) == 1
    assert snapshot.task_leases[0].started_at is not None
    assert snapshot.task_leases[0].released_at is None
    assert [event.type for event in snapshot.events][-7:] == [
        "task.created",
        "governance.allowed",
        "task.ready",
        "task.claimed",
        "executor.started",
        "task.cancellation_requested",
        "operation.interrupted",
    ]
    child_event = snapshot.events[-2]
    assert child_event.turn_id == task.turn_id
    assert child_event.call_id == task.call_id
    assert child_event.task_id == task.id
    assert child_event.capability_id == task.capability_id
    assert child_event.executor_id == task.executor_id
    assert child_event.payload["reason"] == "run_cancelled"
    assert not any(
        event.type in {"executor.failed", "task.failed", "evidence.accepted"}
        for event in snapshot.events
    )
    provider.assert_consumed()


async def test_executor_cannot_suppress_caller_cancellation() -> None:
    executor = CancellationSuppressingReadExecutor()
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider((_tool_response(),))
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SingleTurnContextBuilder(),
        domain=FakeReadDomain(registry),
    )
    running = asyncio.create_task(loop.run(_trigger("suppressed-executor-cancel")))

    await executor.started.wait()
    operation_id = executor.requests[0].operation_id
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    snapshot = await runtime.inspect(operation_id)
    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.tasks[-1].status is TaskStatus.RUNNING
    assert snapshot.tasks[-1].cancellation_requested is True
    assert len(snapshot.task_leases) == 1
    assert snapshot.task_leases[0].started_at is not None
    assert snapshot.task_leases[0].released_at is None
    assert snapshot.evidence == ()
    assert snapshot.observations == ()
    assert not any(event.type == "evidence.accepted" for event in snapshot.events)


async def test_cancellation_after_evidence_preserves_committed_task_success() -> None:
    executor = ImmediateReadExecutor()
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider((_tool_response(),))
    domain = FakeReadDomain(registry, block_projection=True)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SingleTurnContextBuilder(),
        domain=domain,
    )
    running = asyncio.create_task(loop.run(_trigger("after-evidence")))

    await domain.projection_started.wait()
    operation_id = executor.requests[0].operation_id
    before_cancel = await runtime.inspect(operation_id)
    assert before_cancel.tasks[-1].status is TaskStatus.SUCCEEDED
    assert before_cancel.evidence[-1].accepted is True
    assert before_cancel.observations == ()

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert domain.projection_finished.is_set()
    snapshot = await runtime.inspect(operation_id)
    task = snapshot.tasks[-1]

    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.operation.terminal_reason == "run_cancelled"
    assert snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert snapshot.loop_state.interruption_reason == "run_cancelled"
    assert task.status is TaskStatus.SUCCEEDED
    assert task.cancellation_requested is False
    assert len(task.evidence_ids) == 1
    assert len(snapshot.evidence) == 1
    assert snapshot.evidence[0].id == task.evidence_ids[0]
    assert snapshot.evidence[0].accepted is True
    assert snapshot.observations == ()
    assert [event.type for event in snapshot.events][-4:] == [
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
        "operation.interrupted",
    ]
    assert not any(
        event.type.endswith(".cancellation_requested") for event in snapshot.events
    )
    provider.assert_consumed()


class DelayedInterruptRuntime(OperationRuntime):
    def __init__(self) -> None:
        super().__init__(clock=lambda: NOW)
        self.interrupt_started = asyncio.Event()
        self.allow_interrupt = asyncio.Event()

    async def interrupt(
        self,
        operation_id: str,
        reason: str = "run_cancelled",
    ) -> LoopExit:
        self.interrupt_started.set()
        await self.allow_interrupt.wait()
        return await super().interrupt(operation_id, reason)


async def test_repeated_cancel_cannot_abort_the_interruption_commit() -> None:
    runtime = DelayedInterruptRuntime()
    provider = BlockingProvider()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SingleTurnContextBuilder(),
        domain=TextOnlyDomain(),
    )
    running = asyncio.create_task(loop.run(_trigger("resistant-commit")))

    await provider.started.wait()
    running.cancel()
    try:
        await asyncio.wait_for(runtime.interrupt_started.wait(), timeout=0.5)
        running.cancel()
        runtime.allow_interrupt.set()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        runtime.allow_interrupt.set()
        if not running.done():
            running.cancel()
            with suppress(asyncio.CancelledError):
                await running

    assert provider.operation_id is not None
    snapshot = await runtime.inspect(provider.operation_id)
    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.model_calls[-1].cancellation_requested is True
    assert [event.type for event in snapshot.events][-2:] == [
        "model_call.cancellation_requested",
        "operation.interrupted",
    ]


class FailingInterruptRuntime(OperationRuntime):
    async def interrupt(
        self,
        operation_id: str,
        reason: str = "run_cancelled",
    ) -> LoopExit:
        raise OperationStateError(f"forced interrupt conflict: {operation_id}")


def _cleanup_loop(runtime: OperationRuntime) -> AgentLoop:
    return AgentLoop(
        runtime=runtime,
        model=BlockingProvider(),
        context_builder=SingleTurnContextBuilder(),
        domain=TextOnlyDomain(),
    )


async def test_interruption_cleanup_accepts_only_an_authoritative_terminal_race() -> (
    None
):
    runtime = FailingInterruptRuntime(clock=lambda: NOW)
    started = await runtime.begin(_trigger("terminal-cleanup-race"))
    await runtime.fail(started.operation.id, "external_terminal_winner")

    await _cleanup_loop(runtime)._persist_interruption(started.operation.id)

    snapshot = await runtime.inspect(started.operation.id)
    assert snapshot.operation.status is OperationStatus.FAILED
    assert snapshot.operation.terminal_reason == "external_terminal_winner"


async def test_interruption_cleanup_surfaces_a_nonterminal_persistence_failure() -> (
    None
):
    runtime = FailingInterruptRuntime(clock=lambda: NOW)
    started = await runtime.begin(_trigger("nonterminal-cleanup-failure"))

    with pytest.raises(OperationStateError, match="forced interrupt conflict"):
        await _cleanup_loop(runtime)._persist_interruption(started.operation.id)

    snapshot = await runtime.inspect(started.operation.id)
    assert snapshot.operation.status is OperationStatus.RUNNING


class CancelThenConflictStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__()
        self.transition_blocked = asyncio.Event()
        self.release_transition = asyncio.Event()
        self.block_next_commit = True

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        if self.block_next_commit:
            self.block_next_commit = False
            self.transition_blocked.set()
            await self.release_transition.wait()
        return await super().commit(
            snapshot,
            expected_revision=expected_revision,
        )

    async def commit_external(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        return await InMemoryOperationStore.commit(
            self,
            snapshot,
            expected_revision=expected_revision,
        )


async def test_cancellation_wins_when_a_blocked_transition_later_loses_cas() -> None:
    store = CancelThenConflictStore()
    runtime = OperationRuntime(store=store, clock=lambda: NOW)
    provider = BlockingProvider()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SingleTurnContextBuilder(),
        domain=TextOnlyDomain(),
    )
    trigger = _trigger("cancel-before-transition-conflict")
    running = asyncio.create_task(loop.run(trigger))

    await store.transition_blocked.wait()
    current = await store.load_by_trigger(trigger.id)
    assert current is not None
    external_event = RuntimeEvent(
        id="event-external-transition-race",
        type="checkpoint.external",
        agent_id=current.snapshot.operation.agent_id,
        operation_id=current.snapshot.operation.id,
        session_id=current.snapshot.operation.session_id,
        payload={},
        created_at=NOW,
    )
    external = replace(
        current.snapshot,
        events=(*current.snapshot.events, external_event),
    )
    external_result = await store.commit_external(
        external,
        expected_revision=current.revision,
    )

    running.cancel()
    await asyncio.sleep(0)
    store.release_transition.set()
    with pytest.raises(asyncio.CancelledError):
        await running

    committed = await store.load(current.snapshot.operation.id)
    assert committed.revision == external_result.operation.revision + 1
    assert committed.snapshot.events[: len(external.events)] == external.events
    assert committed.snapshot.operation.status is OperationStatus.INTERRUPTED
    assert committed.snapshot.events[-1].type == "operation.interrupted"
    assert provider.requests == []


class MutableClock:
    def __init__(self) -> None:
        self.current = NOW

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class WallExpiringContextBuilder(SingleTurnContextBuilder):
    def __init__(self, clock: MutableClock, seconds: float) -> None:
        self._clock = clock
        self._seconds = seconds
        self.operation_id: str | None = None

    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        request = await super().build(operation, turn, tools)
        self.operation_id = operation.operation.id
        self._clock.advance(self._seconds)
        return request


class BlockingBudgetCommitRuntime(OperationRuntime):
    def __init__(self, clock: MutableClock) -> None:
        super().__init__(clock=clock)
        self.budget_commit_started = asyncio.Event()
        self.budget_commit_finished = asyncio.Event()
        self.allow_budget_commit = asyncio.Event()

    async def fail_budget(
        self,
        operation_id: str,
        reason: str,
        *,
        budget: str,
        limit: int | float | str,
        used: int | float | str,
        turn_id: str | None = None,
        call_id: str | None = None,
        task_id: str | None = None,
    ) -> LoopExit:
        self.budget_commit_started.set()
        try:
            await self.allow_budget_commit.wait()
        finally:
            self.budget_commit_finished.set()
        return await super().fail_budget(
            operation_id,
            reason,
            budget=budget,
            limit=limit,
            used=used,
            turn_id=turn_id,
            call_id=call_id,
            task_id=task_id,
        )


async def test_cancellation_during_wall_budget_commit_persists_interruption() -> None:
    clock = MutableClock()
    runtime = BlockingBudgetCommitRuntime(clock)
    context_builder = WallExpiringContextBuilder(clock, seconds=5.0)
    provider = MockModelProvider(
        (
            ModelResponse(
                text="Must not be requested.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=context_builder,
        domain=TextOnlyDomain(),
        budgets=LoopBudgets(max_wall_time_seconds=5.0),
    )
    running = asyncio.create_task(loop.run(_trigger("wall-budget-commit")))

    await runtime.budget_commit_started.wait()
    assert context_builder.operation_id is not None
    before_cancel = await runtime.inspect(context_builder.operation_id)
    assert before_cancel.operation.status is OperationStatus.RUNNING
    assert not any(event.type == "budget.exhausted" for event in before_cancel.events)

    running.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        runtime.allow_budget_commit.set()

    assert runtime.budget_commit_finished.is_set()
    snapshot = await runtime.inspect(context_builder.operation_id)
    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.operation.terminal_reason == "run_cancelled"
    assert snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert snapshot.loop_state.interruption_reason == "run_cancelled"
    assert snapshot.model_calls == ()
    assert snapshot.tasks == ()
    assert provider.requests == ()
    assert [event.type for event in snapshot.events][-1] == "operation.interrupted"
    assert not any(
        event.type in {"budget.exhausted", "operation.failed"}
        for event in snapshot.events
    )
