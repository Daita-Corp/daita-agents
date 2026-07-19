from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from daita import Agent
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
from daita.loop.models import LoopExitKind, Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.governance import ApprovalRequest, ApprovalStatus
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TriggerKind,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
HASH = "sha256:" + ("0" * 64)


class TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class TextDomain:
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
        raise AssertionError("text-only control test cannot validate an action")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only control test cannot project evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        return Readiness(
            allowed=True,
            code="ready.text",
            message="Text response is ready.",
            evaluated_at=NOW,
        )


class BlockingProvider:
    provider_id = "mock:embedded-control"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.operation_id: str | None = None

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.operation_id = request.operation_id
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.finished.set()
        raise AssertionError("blocking provider must be cancelled by the test")


async def test_create_and_open_start_no_hidden_background_tasks(tmp_path) -> None:
    current = asyncio.current_task()
    before = {task for task in asyncio.all_tasks() if task is not current}
    agent = await Agent.create("atlas", root=tmp_path, clock=lambda: NOW)
    await asyncio.sleep(0)
    try:
        after = {task for task in asyncio.all_tasks() if task is not current}
        assert after == before
    finally:
        await agent.close()

    before_reopen = {task for task in asyncio.all_tasks() if task is not current}
    reopened = await Agent.open("atlas", root=tmp_path, clock=lambda: NOW)
    await asyncio.sleep(0)
    try:
        after_reopen = {task for task in asyncio.all_tasks() if task is not current}
        assert after_reopen == before_reopen
    finally:
        await reopened.close()


async def test_exact_trigger_replay_reuses_session_operation_and_committed_events(
    tmp_path,
) -> None:
    provider = MockModelProvider(
        (ModelResponse(text="done", finish_reason=FinishReason.STOP),),
        provider_id="mock:exact-trigger",
    )
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
    )
    trigger = AgentTrigger(
        id="host-trigger-a",
        agent_id=agent.id,
        kind=TriggerKind.USER,
        source_id="host:user-a",
        session_id="session-a",
        payload={"message": "Run the exact durable trigger."},
        created_at=NOW,
    )
    try:
        first = await agent._embedded.run_trigger(trigger)
        replay = await agent._embedded.run_trigger(trigger)
        events = await agent.events(limit=100)
        stream = agent.subscribe_events()
        try:
            first_streamed = await anext(stream)
        finally:
            await stream.aclose()  # type: ignore[attr-defined]

        assert replay == first
        assert len(provider.requests) == 1
        assert first_streamed == events[0]
        assert events[-1].event.type == "operation.succeeded"
    finally:
        await agent.close()


async def test_approval_decision_bypasses_operation_lock_and_does_not_resume(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockModelProvider((), provider_id="mock:approval-control")
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
    )
    embedded = agent._embedded
    expected = ApprovalRequest(
        id="approval-a",
        operation_id="operation-a",
        task_id="task-a",
        task_fingerprint=HASH,
        policy_fingerprint=HASH,
        requested_at=NOW,
        status=ApprovalStatus.APPROVED,
        decided_at=NOW,
        decided_by="reviewer-a",
        decision_reason="Approved.",
    )
    resume_calls = 0

    async def decide_approval(*args, **kwargs) -> ApprovalRequest:
        return expected

    async def forbidden_resume(operation_id: str):
        nonlocal resume_calls
        resume_calls += 1
        raise AssertionError("approval channel must not resume execution")

    monkeypatch.setattr(embedded._runtime, "decide_approval", decide_approval)
    assert embedded._loop is not None
    monkeypatch.setattr(embedded._loop, "resume", forbidden_resume)
    await embedded._mutation_lock.acquire()
    try:
        decided = await asyncio.wait_for(
            agent.approve(
                "approval-a",
                decided_by="reviewer-a",
                reason="Approved.",
            ),
            timeout=0.5,
        )
    finally:
        embedded._mutation_lock.release()
        await agent.close()

    assert decided == expected
    assert resume_calls == 0


async def test_interrupt_and_nonterminal_inspection_bypass_blocked_run_lock(
    tmp_path,
) -> None:
    provider = BlockingProvider()
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
    )
    run_task = asyncio.create_task(agent.run("Block in the model provider."))
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        assert provider.operation_id is not None
        nonterminal = await asyncio.wait_for(
            agent._embedded.inspect_nonterminal(),
            timeout=0.5,
        )
        interrupted = await asyncio.wait_for(
            agent.cancel(provider.operation_id, reason="host_cancelled"),
            timeout=0.5,
        )
        snapshot = await agent.inspect(provider.operation_id)

        assert len(nonterminal) == 1
        assert nonterminal[0].operation.id == provider.operation_id
        assert interrupted.kind is LoopExitKind.INTERRUPTED
        assert snapshot.operation.status is OperationStatus.INTERRUPTED
        assert snapshot.model_calls[-1].cancellation_requested
    finally:
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        await asyncio.wait_for(provider.finished.wait(), timeout=1)
        await agent.close()


async def test_startup_recovery_is_serialized_without_spawning_work(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockModelProvider((), provider_id="mock:startup-control")
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        context_builder=TextContext(),
        domain=TextDomain(),
        clock=lambda: NOW,
    )
    embedded = agent._embedded
    assert embedded._loop is not None
    calls: list[str] = []

    async def recover(agent_id: str):
        assert embedded._mutation_lock.locked()
        calls.append(agent_id)
        return ()

    monkeypatch.setattr(embedded._loop, "recover_startup", recover)
    before = asyncio.all_tasks()
    try:
        assert await embedded.recover_startup() == ()
        assert asyncio.all_tasks() == before
        assert calls == [agent.id]
    finally:
        await agent.close()
