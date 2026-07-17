from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from daita.events.models import RuntimeEvent
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
from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Readiness, Turn
from daita.operations import runtime as operation_runtime
from daita.operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from daita.operations.models import (
    AgentTrigger,
    Evidence,
    Observation,
    Operation,
    OperationStatus,
    Task,
    TaskStatus,
    TriggerKind,
)

NOW = datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc)


def _request(
    *,
    operation_id: str = "operation-1",
    turn_id: str = "turn-1",
) -> ModelRequest:
    return ModelRequest(
        operation_id=operation_id,
        turn_id=turn_id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=operation_id,
                turn_id=turn_id,
                role=MessageRole.USER,
                content=(TextBlock("Read alpha."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="fake.read",
                description="Read a deterministic test value.",
                input_schema={"type": "object"},
            ),
        ),
    )


def _tool_response() -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(id="call-1", name="fake.read", arguments={"key": "alpha"}),
        ),
    )


def _model_call(
    *,
    status: ModelCallStatus = ModelCallStatus.STARTED,
    request: ModelRequest | None = None,
    response: ModelResponse | None = None,
    error_code: str | None = None,
    created_at: datetime = NOW,
    updated_at: datetime = NOW,
    cancellation_requested: bool = False,
) -> ModelCall:
    return ModelCall(
        id="model-call-1",
        operation_id="operation-1",
        turn_id="turn-1",
        provider_id="mock:scripted",
        request=request or _request(),
        status=status,
        response=response,
        error_code=error_code,
        cancellation_requested=cancellation_requested,
        created_at=created_at,
        updated_at=updated_at,
    )


def _empty_snapshot() -> OperationSnapshot:
    trigger = AgentTrigger(
        id="trigger-1",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        session_id="session-1",
        payload={"message": "Read alpha."},
        created_at=NOW,
    )
    operation = Operation(
        id="operation-1",
        agent_id="agent-1",
        trigger_id=trigger.id,
        session_id=trigger.session_id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
    )
    events = (
        RuntimeEvent(
            id="event-1",
            type="trigger.received",
            agent_id=operation.agent_id,
            operation_id=operation.id,
            session_id=operation.session_id,
            payload={},
            created_at=NOW,
        ),
        RuntimeEvent(
            id="event-2",
            type="operation.created",
            agent_id=operation.agent_id,
            operation_id=operation.id,
            session_id=operation.session_id,
            payload={},
            created_at=NOW,
        ),
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(phase=LoopPhase.PREPARING_CONTEXT),
        budgets=LoopBudgets(),
        turns=(),
        model_calls=(),
        readiness=(),
        tasks=(),
        evidence=(),
        observations=(),
        events=events,
    )


def _full_snapshot() -> OperationSnapshot:
    base = _empty_snapshot()
    turn = Turn(
        id="turn-1",
        operation_id=base.operation.id,
        number=1,
        model_request_id="model-call-1",
        model_response_id="model-call-1",
        created_at=NOW,
    )
    model_call = _model_call(
        status=ModelCallStatus.COMPLETED,
        response=_tool_response(),
    )
    evidence = Evidence(
        id="evidence-1",
        operation_id=base.operation.id,
        task_id="task-1",
        turn_id=turn.id,
        capability_id="fake.read",
        executor_id="fake.executor",
        kind="fake.value",
        schema_version=1,
        attempt=1,
        accepted=True,
        payload={"value": 42},
        content_hash="sha256:test",
        created_at=NOW,
    )
    task = Task(
        id="task-1",
        operation_id=base.operation.id,
        turn_id=turn.id,
        call_id="call-1",
        capability_id=evidence.capability_id,
        executor_id=evidence.executor_id,
        status=TaskStatus.SUCCEEDED,
        attempt=1,
        arguments={"key": "alpha"},
        evidence_ids=(evidence.id,),
        created_at=NOW,
        updated_at=NOW,
    )
    observation = Observation(
        operation_id=base.operation.id,
        turn_id=turn.id,
        call_id=task.call_id,
        task_id=task.id,
        evidence_id=evidence.id,
        code="fake.read.succeeded",
        message="Read completed.",
        payload={"value": 42},
        success=True,
        created_at=NOW,
    )
    correlated_event = RuntimeEvent(
        id="event-3",
        type="evidence.accepted",
        agent_id=base.operation.agent_id,
        operation_id=base.operation.id,
        session_id=base.operation.session_id,
        turn_id=turn.id,
        model_call_id=model_call.id,
        call_id=task.call_id,
        task_id=task.id,
        evidence_id=evidence.id,
        capability_id=task.capability_id,
        executor_id=task.executor_id,
        payload={"task_id": task.id, "evidence_id": evidence.id},
        created_at=NOW,
    )
    return replace(
        base,
        loop_state=replace(
            base.loop_state,
            phase=LoopPhase.OBSERVING,
            turn_count=1,
            action_count=1,
        ),
        turns=(turn,),
        model_calls=(model_call,),
        tasks=(task,),
        evidence=(evidence,),
        observations=(observation,),
        events=(*base.events, correlated_event),
    )


def test_model_call_records_exact_provider_neutral_lifecycle_shapes() -> None:
    started = _model_call()
    cancelled = _model_call(cancellation_requested=True)
    completed = _model_call(
        status=ModelCallStatus.COMPLETED,
        response=_tool_response(),
    )
    failed = _model_call(
        status=ModelCallStatus.FAILED,
        error_code="provider_unavailable",
    )

    assert started.status is ModelCallStatus.STARTED
    assert started.response is None
    assert started.error_code is None
    assert cancelled.cancellation_requested is True
    assert completed.response == _tool_response()
    assert completed.error_code is None
    assert failed.response is None
    assert failed.error_code == "provider_unavailable"


def test_model_call_requires_identity_request_linkage_and_ordered_aware_times() -> None:
    call = _model_call()

    with pytest.raises(ValueError, match="non-empty"):
        replace(call, id=" ")
    with pytest.raises(ValueError, match="non-empty"):
        replace(call, operation_id="")
    with pytest.raises(ValueError, match="non-empty"):
        replace(call, turn_id=" ")
    with pytest.raises(ValueError, match="non-empty"):
        replace(call, provider_id="")
    with pytest.raises(ValueError, match="request.*operation|operation.*request"):
        replace(call, request=_request(operation_id="operation-other"))
    with pytest.raises(ValueError, match="request.*turn|turn.*request"):
        replace(call, request=_request(turn_id="turn-other"))
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(call, created_at=datetime(2026, 7, 17, 15, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(call, updated_at=datetime(2026, 7, 17, 15, 0))
    with pytest.raises(ValueError, match="updated_at"):
        replace(call, updated_at=NOW - timedelta(microseconds=1))
    with pytest.raises(TypeError, match="boolean"):
        replace(call, cancellation_requested=1)  # type: ignore[arg-type]


def test_model_call_rejects_incoherent_terminal_payloads() -> None:
    with pytest.raises(ValueError, match="started.*response"):
        _model_call(response=_tool_response())
    with pytest.raises(ValueError, match="started.*error"):
        _model_call(error_code="unexpected")
    with pytest.raises(ValueError, match="completed.*response"):
        _model_call(status=ModelCallStatus.COMPLETED)
    with pytest.raises(ValueError, match="completed.*error"):
        _model_call(
            status=ModelCallStatus.COMPLETED,
            response=_tool_response(),
            error_code="unexpected",
        )
    with pytest.raises(ValueError, match="failed.*error"):
        _model_call(status=ModelCallStatus.FAILED)
    with pytest.raises(ValueError, match="failed.*response"):
        _model_call(
            status=ModelCallStatus.FAILED,
            response=_tool_response(),
            error_code="provider_unavailable",
        )


def test_operation_snapshot_normalizes_collection_inputs_to_immutable_tuples() -> None:
    base = _full_snapshot()
    turns = list(base.turns)
    model_calls = list(base.model_calls)
    readiness: list[Readiness] = []
    tasks = list(base.tasks)
    evidence = list(base.evidence)
    observations = list(base.observations)
    events = list(base.events)

    snapshot = OperationSnapshot(
        trigger=base.trigger,
        operation=base.operation,
        loop_state=base.loop_state,
        budgets=base.budgets,
        turns=cast(tuple[Turn, ...], turns),
        model_calls=cast(tuple[ModelCall, ...], model_calls),
        readiness=cast(tuple[Readiness, ...], readiness),
        tasks=cast(tuple[Task, ...], tasks),
        evidence=cast(tuple[Evidence, ...], evidence),
        observations=cast(tuple[Observation, ...], observations),
        events=cast(tuple[RuntimeEvent, ...], events),
    )
    turns.clear()
    model_calls.clear()
    tasks.clear()
    evidence.clear()
    observations.clear()
    events.clear()

    assert snapshot.turns == base.turns
    assert snapshot.model_calls == base.model_calls
    assert snapshot.readiness == ()
    assert snapshot.tasks == base.tasks
    assert snapshot.evidence == base.evidence
    assert snapshot.observations == base.observations
    assert snapshot.events == base.events
    assert all(
        isinstance(items, tuple)
        for items in (
            snapshot.turns,
            snapshot.model_calls,
            snapshot.readiness,
            snapshot.tasks,
            snapshot.evidence,
            snapshot.observations,
            snapshot.events,
        )
    )


def test_operation_snapshot_requires_exact_trigger_operation_linkage() -> None:
    snapshot = _empty_snapshot()

    with pytest.raises(ValueError, match="agent"):
        replace(snapshot, operation=replace(snapshot.operation, agent_id="agent-2"))
    with pytest.raises(ValueError, match="trigger"):
        replace(
            snapshot,
            operation=replace(snapshot.operation, trigger_id="trigger-other"),
        )
    with pytest.raises(ValueError, match="session"):
        replace(
            snapshot,
            operation=replace(snapshot.operation, session_id="session-other"),
        )
    with pytest.raises(TypeError, match="budgets"):
        replace(snapshot, budgets=cast(LoopBudgets, object()))


def test_operation_snapshot_rejects_orphaned_child_records_and_events() -> None:
    snapshot = _full_snapshot()
    turn = snapshot.turns[0]
    model_call = snapshot.model_calls[0]
    task = snapshot.tasks[0]
    evidence = snapshot.evidence[0]
    observation = snapshot.observations[0]
    event = snapshot.events[-1]

    with pytest.raises(ValueError, match="turn.*operation"):
        replace(snapshot, turns=(replace(turn, operation_id="operation-other"),))

    orphan_request = _request(turn_id="turn-other")
    orphan_model_call = replace(
        model_call,
        turn_id="turn-other",
        request=orphan_request,
    )
    with pytest.raises(ValueError, match="model call.*turn|turn.*model call"):
        replace(snapshot, model_calls=(orphan_model_call,))

    with pytest.raises(ValueError, match="task.*operation"):
        replace(snapshot, tasks=(replace(task, operation_id="operation-other"),))
    with pytest.raises(ValueError, match="evidence.*task|task.*evidence"):
        replace(snapshot, evidence=(replace(evidence, task_id="task-other"),))
    with pytest.raises(ValueError, match="observation.*evidence|evidence.*observation"):
        replace(
            snapshot,
            observations=(replace(observation, evidence_id="evidence-other"),),
        )
    with pytest.raises(ValueError, match="event.*agent|agent.*event"):
        replace(snapshot, events=(*snapshot.events[:-1], replace(event, agent_id="x")))
    with pytest.raises(ValueError, match="event.*model call|model call.*event"):
        replace(
            snapshot,
            events=(
                *snapshot.events[:-1],
                replace(event, model_call_id="model-call-other"),
            ),
        )
    with pytest.raises(ValueError, match="event.*model call|model call.*event"):
        replace(
            snapshot,
            events=(*snapshot.events[:-1], replace(event, model_call_id=None)),
        )


def test_operation_snapshot_rejects_duplicate_lifecycle_record_ids() -> None:
    snapshot = _full_snapshot()

    with pytest.raises(ValueError, match="duplicate.*turn"):
        replace(snapshot, turns=(snapshot.turns[0], snapshot.turns[0]))
    with pytest.raises(ValueError, match="duplicate.*model call"):
        replace(
            snapshot,
            model_calls=(snapshot.model_calls[0], snapshot.model_calls[0]),
        )
    with pytest.raises(ValueError, match="duplicate.*task"):
        replace(snapshot, tasks=(snapshot.tasks[0], snapshot.tasks[0]))
    with pytest.raises(ValueError, match="duplicate.*evidence"):
        replace(snapshot, evidence=(snapshot.evidence[0], snapshot.evidence[0]))
    with pytest.raises(ValueError, match="duplicate.*event"):
        replace(snapshot, events=(*snapshot.events, snapshot.events[0]))


def test_turn_model_pointers_must_resolve_to_a_call_owned_by_that_turn() -> None:
    snapshot = _full_snapshot()
    first_turn = snapshot.turns[0]
    second_request = _request(turn_id="turn-2")
    second_model_call = replace(
        snapshot.model_calls[0],
        id="model-call-2",
        turn_id="turn-2",
        request=second_request,
    )
    second_turn = Turn(
        id="turn-2",
        operation_id=snapshot.operation.id,
        number=2,
        model_request_id=second_model_call.id,
        model_response_id=second_model_call.id,
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="turn.*model call|model call.*turn"):
        replace(
            snapshot,
            turns=(
                replace(first_turn, model_request_id=second_model_call.id),
                second_turn,
            ),
            model_calls=(*snapshot.model_calls, second_model_call),
        )
    with pytest.raises(ValueError, match="turn.*model call|model call.*turn"):
        replace(
            snapshot,
            turns=(
                replace(first_turn, model_response_id=second_model_call.id),
                second_turn,
            ),
            model_calls=(*snapshot.model_calls, second_model_call),
        )


def test_every_model_call_must_be_owned_by_its_turn_pointers() -> None:
    snapshot = _full_snapshot()
    unlinked_call = replace(snapshot.model_calls[0], id="model-call-unlinked")

    with pytest.raises(ValueError, match="model call.*turn|turn.*model call"):
        replace(snapshot, model_calls=(*snapshot.model_calls, unlinked_call))

    base = _empty_snapshot()
    started_call = _model_call()
    premature_response_pointer = Turn(
        id="turn-1",
        operation_id=base.operation.id,
        number=1,
        model_request_id=started_call.id,
        model_response_id=started_call.id,
        created_at=NOW,
    )
    with pytest.raises(ValueError, match="response.*completed|completed.*response"):
        replace(
            base,
            turns=(premature_response_pointer,),
            model_calls=(started_call,),
        )


def test_event_child_correlations_must_resolve_to_one_explicit_turn() -> None:
    snapshot = _full_snapshot()
    second_model_call = replace(
        snapshot.model_calls[0],
        id="model-call-2",
        turn_id="turn-2",
        request=_request(turn_id="turn-2"),
    )
    second_turn = Turn(
        id="turn-2",
        operation_id=snapshot.operation.id,
        number=2,
        model_request_id=second_model_call.id,
        model_response_id=second_model_call.id,
        created_at=NOW,
    )
    cross_turn_event = replace(
        snapshot.events[-1],
        turn_id=None,
        model_call_id=second_model_call.id,
    )
    with pytest.raises(ValueError, match="event.*turn|turn.*event"):
        replace(
            snapshot,
            turns=(*snapshot.turns, second_turn),
            model_calls=(*snapshot.model_calls, second_model_call),
            events=(*snapshot.events[:-1], cross_turn_event),
        )

    base = _empty_snapshot()
    orphan_call_event = RuntimeEvent(
        id="event-orphan-call",
        type="action.proposed",
        agent_id=base.operation.agent_id,
        operation_id=base.operation.id,
        session_id=base.operation.session_id,
        call_id="call-not-in-any-turn",
        payload={},
        created_at=NOW,
    )
    with pytest.raises(ValueError, match="event.*turn|turn.*event"):
        replace(base, events=(*base.events, orphan_call_event))

    model_call = snapshot.model_calls[0]
    assert model_call.response is not None
    second_tool_call = ToolCall(
        id="call-2",
        name="fake.read",
        arguments={"key": "beta"},
    )
    expanded_model_call = replace(
        model_call,
        response=replace(
            model_call.response,
            tool_calls=(*model_call.response.tool_calls, second_tool_call),
        ),
    )
    mismatched_task_call_event = replace(
        snapshot.events[-1],
        call_id=second_tool_call.id,
    )
    with pytest.raises(ValueError, match="event.*call|call.*event"):
        replace(
            snapshot,
            model_calls=(expanded_model_call,),
            events=(*snapshot.events[:-1], mismatched_task_call_event),
        )


def test_task_call_must_exist_in_its_turn_committed_model_response() -> None:
    snapshot = _full_snapshot()
    task = replace(snapshot.tasks[0], call_id="forged-call")
    event = replace(snapshot.events[-1], call_id=task.call_id)

    with pytest.raises(ValueError, match="task.*call|tool call.*task"):
        replace(
            snapshot,
            tasks=(task,),
            observations=(),
            events=(*snapshot.events[:-1], event),
        )


def test_taskless_observation_call_must_exist_in_its_turn_response() -> None:
    snapshot = _full_snapshot()
    forged_observation = Observation(
        operation_id=snapshot.operation.id,
        turn_id=snapshot.turns[0].id,
        call_id="forged-call",
        code="action.rejected",
        message="The action was rejected.",
        payload={},
        success=False,
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="observation.*call|tool call.*observation"):
        replace(
            snapshot,
            observations=(*snapshot.observations, forged_observation),
        )


def test_event_call_must_exist_in_its_correlated_turn_response() -> None:
    snapshot = _full_snapshot()
    forged_event = RuntimeEvent(
        id="event-forged-call",
        type="action.rejected",
        agent_id=snapshot.operation.agent_id,
        operation_id=snapshot.operation.id,
        session_id=snapshot.operation.session_id,
        turn_id=snapshot.turns[0].id,
        call_id="forged-call",
        payload={"code": "invalid_arguments"},
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="event.*call|tool call.*event"):
        replace(snapshot, events=(*snapshot.events, forged_event))


def test_task_evidence_and_observation_ownership_is_symmetric() -> None:
    snapshot = _full_snapshot()
    first_evidence = snapshot.evidence[0]
    second_evidence = replace(first_evidence, id="evidence-2")
    task = replace(snapshot.tasks[0], evidence_ids=(second_evidence.id,))

    with pytest.raises(ValueError, match="task.*evidence|evidence.*task"):
        replace(
            snapshot,
            tasks=(task,),
            evidence=(first_evidence, second_evidence),
        )


def test_operation_runtime_reexports_canonical_checkpoints_for_compatibility() -> None:
    assert operation_runtime.ModelCallStatus is ModelCallStatus
    assert operation_runtime.ModelCall is ModelCall
    assert operation_runtime.OperationSnapshot is OperationSnapshot
