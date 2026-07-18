from __future__ import annotations

from datetime import datetime, timezone

import pytest

from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
)
from daita.loop.models import LoopPhase
from daita.operations.models import AgentTrigger, OperationStatus, TriggerKind
from daita.operations.runtime import ModelCallStatus, OperationRuntime

NOW = datetime(2026, 7, 16, 13, 0, tzinfo=timezone.utc)


async def test_rejected_event_trigger_leaves_no_reservation_or_partial_state() -> None:
    issued_ids: list[str] = []

    def recording_id_factory(prefix: str) -> str:
        identifier = f"{prefix}-{len(issued_ids) + 1}"
        issued_ids.append(identifier)
        return identifier

    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=recording_id_factory,
    )
    rejected = AgentTrigger(
        id="trigger-reusable",
        agent_id="agent-1",
        kind=TriggerKind.EVENT,
        source_id="event-source-1",
        payload={"message": "reserved for a later phase"},
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="event triggers.*later phase"):
        await runtime.begin(rejected)

    assert issued_ids == []
    accepted = AgentTrigger(
        id=rejected.id,
        agent_id=rejected.agent_id,
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"message": "the trigger id remains available"},
        created_at=NOW,
    )
    started = await runtime.begin(accepted)
    snapshot = await runtime.inspect(started.operation.id)

    assert snapshot.trigger == accepted
    assert snapshot.operation.status is OperationStatus.RUNNING
    assert snapshot.turns == ()
    assert snapshot.model_calls == ()
    assert snapshot.tasks == ()
    assert snapshot.evidence == ()
    assert snapshot.observations == ()
    assert [event.type for event in snapshot.events] == [
        "trigger.received",
        "operation.created",
    ]


async def test_exact_duplicate_trigger_returns_the_existing_operation() -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    trigger = AgentTrigger(
        id="trigger-1",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"message": "hello"},
        created_at=NOW,
    )
    started = await runtime.begin(trigger)

    redelivered = await runtime.begin(trigger)

    unchanged = await runtime.inspect(started.operation.id)
    assert redelivered == started
    assert unchanged == started
    assert [event.type for event in unchanged.events] == [
        "trigger.received",
        "operation.created",
    ]


async def test_failed_in_memory_commit_does_not_publish_partial_state() -> None:
    counter = 0

    def fail_during_response_event(prefix: str) -> str:
        nonlocal counter
        counter += 1
        if counter == 9:
            raise RuntimeError("injected event commit failure")
        return f"{prefix}-{counter}"

    runtime = OperationRuntime(
        clock=lambda: NOW,
        id_factory=fail_during_response_event,
    )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-1",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "hello"},
            created_at=NOW,
        )
    )
    turn = await runtime.begin_turn(started.operation.id)
    request = ModelRequest(
        operation_id=started.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=started.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        ),
    )
    model_call = await runtime.begin_model_call(
        started.operation.id,
        turn.id,
        "mock:scripted",
        request,
    )
    before = await runtime.inspect(started.operation.id)

    with pytest.raises(RuntimeError, match="injected event commit failure"):
        await runtime.record_model_response(
            started.operation.id,
            model_call.id,
            ModelResponse(text="hello back", finish_reason=FinishReason.STOP),
            next_phase=LoopPhase.SYNTHESIZING,
        )

    after = await runtime.inspect(started.operation.id)
    assert after == before
    assert after.model_calls[-1].status is ModelCallStatus.STARTED
    assert after.turns[-1].model_response_id is None
    assert after.loop_state.phase is LoopPhase.AWAITING_MODEL
    assert after.events[-1].type == "model_call.started"
