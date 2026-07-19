from __future__ import annotations

from datetime import datetime, timezone

import pytest

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    TextBlock,
)
from daita.loop.models import LoopExitKind, LoopPhase
from daita.operations.models import (
    AgentTrigger,
    OperationStatus,
    TriggerKind,
)
from daita.operations.runtime import (
    ModelCall,
    ModelCallStatus,
    OperationRuntime,
    OperationStateError,
)

NOW = datetime(2026, 7, 17, 12, 30, tzinfo=timezone.utc)


class SequentialIdFactory:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._armed_event_failure: int | None = None
        self._armed_event_count = 0

    def __call__(self, prefix: str) -> str:
        if prefix == "event" and self._armed_event_failure is not None:
            self._armed_event_count += 1
            if self._armed_event_count == self._armed_event_failure:
                raise RuntimeError("injected interruption event failure")
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}-{self._counts[prefix]}"

    def fail_on_interruption_event(self, number: int) -> None:
        self._armed_event_failure = number
        self._armed_event_count = 0


async def _runtime_with_active_model_call(
    *,
    id_factory: SequentialIdFactory | None = None,
) -> tuple[OperationRuntime, str, ModelCall]:
    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=id_factory or SequentialIdFactory(),
    )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-runtime-interrupt",
            agent_id="agent-runtime-interrupt",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "wait"},
            created_at=NOW,
        )
    )
    operation_id = started.operation.id
    turn = await runtime.begin_turn(operation_id)
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id=started.operation.agent_id,
                operation_id=operation_id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Wait for cancellation."),),
            ),
        ),
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:interrupt",
        request,
    )
    return runtime, operation_id, model_call


async def test_interrupt_returns_typed_exit_and_marks_active_model_intent() -> None:
    runtime, operation_id, model_call = await _runtime_with_active_model_call()

    result = await runtime.interrupt(operation_id)
    snapshot = await runtime.inspect(operation_id)

    assert result.operation_id == operation_id
    assert result.kind is LoopExitKind.INTERRUPTED
    assert result.reason == "run_cancelled"
    assert result.final_text is None
    assert result.created_at == NOW
    assert snapshot.operation.status is OperationStatus.INTERRUPTED
    assert snapshot.operation.terminal_reason == result.reason
    assert snapshot.operation.final_text is None
    assert snapshot.loop_state.phase is LoopPhase.TERMINAL
    assert snapshot.loop_state.interruption_reason == result.reason
    assert snapshot.model_calls[-1].id == model_call.id
    assert snapshot.model_calls[-1].status is ModelCallStatus.STARTED
    assert snapshot.model_calls[-1].cancellation_requested is True
    assert [event.type for event in snapshot.events][-2:] == [
        "model_call.cancellation_requested",
        "operation.interrupted",
    ]


async def test_exact_interruption_replay_is_a_durable_noop() -> None:
    runtime, operation_id, _ = await _runtime_with_active_model_call()

    first = await runtime.interrupt(operation_id, "operator_cancelled")
    after_first = await runtime.inspect(operation_id)
    replay = await runtime.interrupt(operation_id, "operator_cancelled")
    after_replay = await runtime.inspect(operation_id)

    assert replay == first
    assert after_replay == after_first

    with pytest.raises(
        OperationStateError,
        match="already interrupted for another reason",
    ):
        await runtime.interrupt(operation_id, "different_reason")


async def test_interrupt_event_failure_commits_no_partial_state() -> None:
    id_factory = SequentialIdFactory()
    runtime, operation_id, _ = await _runtime_with_active_model_call(
        id_factory=id_factory
    )
    before = await runtime.inspect(operation_id)
    id_factory.fail_on_interruption_event(2)

    with pytest.raises(RuntimeError, match="interruption event failure"):
        await runtime.interrupt(operation_id)

    after = await runtime.inspect(operation_id)
    assert after == before
    assert after.operation.status is OperationStatus.RUNNING
    assert after.loop_state.phase is LoopPhase.AWAITING_MODEL
    assert after.loop_state.interruption_reason is None
    assert after.model_calls[-1].status is ModelCallStatus.STARTED
    assert after.model_calls[-1].cancellation_requested is False
    assert not any(
        event.type.endswith(".cancellation_requested")
        or event.type == "operation.interrupted"
        for event in after.events
    )
