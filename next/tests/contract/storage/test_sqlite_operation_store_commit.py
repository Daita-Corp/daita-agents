from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

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
from daita.operations.store import (
    CommitResult,
    InvalidOperationCheckpointError,
    OperationAlreadyExistsError,
    OperationNotFoundError,
    OperationRevisionConflict,
    OperationStoreError,
    TriggerAlreadyClaimedError,
)
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
MODEL_COMPLETED_AT = NOW + timedelta(seconds=1)
TASK_COMPLETED_AT = NOW + timedelta(seconds=2)
AGENT_ID = "agent-commit-contract"
SESSION_ID = "session-commit-contract"

NORMALIZED_TABLES = (
    "triggers",
    "operations",
    "loop_state",
    "turns",
    "model_calls",
    "readiness",
    "tasks",
    "task_dependencies",
    "task_leases",
    "evidence",
    "task_evidence",
    "observations",
    "runtime_events",
)


def _event(
    operation_id: str,
    event_id: str,
    event_type: str,
    created_at: datetime,
    *,
    turn_id: str | None = None,
    model_call_id: str | None = None,
    call_id: str | None = None,
    task_id: str | None = None,
    evidence_id: str | None = None,
    capability_id: str | None = None,
    executor_id: str | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        id=event_id,
        type=event_type,
        agent_id=AGENT_ID,
        operation_id=operation_id,
        session_id=SESSION_ID,
        turn_id=turn_id,
        model_call_id=model_call_id,
        call_id=call_id,
        task_id=task_id,
        evidence_id=evidence_id,
        capability_id=capability_id,
        executor_id=executor_id,
        payload={"event_id": event_id},
        created_at=created_at,
    )


def _request(operation_id: str, turn_id: str, *, with_tool: bool) -> ModelRequest:
    return ModelRequest(
        operation_id=operation_id,
        turn_id=turn_id,
        messages=(
            CanonicalMessage(
                agent_id=AGENT_ID,
                operation_id=operation_id,
                session_id=SESSION_ID,
                turn_id=turn_id,
                role=MessageRole.USER,
                content=(TextBlock("Advance the durable checkpoint."),),
            ),
        ),
        tools=(
            (
                ToolDefinition(
                    name="fake.read",
                    description="Read deterministic evidence.",
                    input_schema={"type": "object"},
                ),
            )
            if with_tool
            else ()
        ),
    )


def _initial_snapshot(
    *,
    operation_id: str = "operation-commit",
    trigger_id: str = "trigger-commit",
    mutable_task_status: TaskStatus = TaskStatus.PENDING,
) -> OperationSnapshot:
    model_turn_id = f"{operation_id}:turn:model"
    task_turn_id = f"{operation_id}:turn:task"
    started_call_id = f"{operation_id}:model:started"
    tools_call_id = f"{operation_id}:model:tools"
    stable_tool_call_id = f"{operation_id}:call:stable"
    mutable_tool_call_id = f"{operation_id}:call:mutable"
    stable_task_id = f"{operation_id}:task:stable"
    mutable_task_id = f"{operation_id}:task:mutable"
    stable_evidence_id = f"{operation_id}:evidence:stable"

    trigger = AgentTrigger(
        id=trigger_id,
        agent_id=AGENT_ID,
        kind=TriggerKind.USER,
        source_id="user-commit-contract",
        session_id=SESSION_ID,
        payload={"message": "commit this operation"},
        created_at=NOW,
    )
    operation = Operation(
        id=operation_id,
        agent_id=AGENT_ID,
        trigger_id=trigger.id,
        session_id=SESSION_ID,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
    )
    model_turn = Turn(
        id=model_turn_id,
        operation_id=operation_id,
        number=1,
        model_request_id=started_call_id,
        created_at=NOW,
    )
    task_turn = Turn(
        id=task_turn_id,
        operation_id=operation_id,
        number=2,
        model_request_id=tools_call_id,
        model_response_id=tools_call_id,
        created_at=NOW,
    )
    started_call = ModelCall(
        id=started_call_id,
        operation_id=operation_id,
        turn_id=model_turn.id,
        provider_id="mock:started",
        request=_request(operation_id, model_turn.id, with_tool=False),
        status=ModelCallStatus.STARTED,
        created_at=NOW,
        updated_at=NOW,
    )
    tools_call = ModelCall(
        id=tools_call_id,
        operation_id=operation_id,
        turn_id=task_turn.id,
        provider_id="mock:tools",
        request=_request(operation_id, task_turn.id, with_tool=True),
        response=ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id=stable_tool_call_id,
                    name="fake.read",
                    arguments={"key": "stable"},
                ),
                ToolCall(
                    id=mutable_tool_call_id,
                    name="fake.read",
                    arguments={"key": "mutable"},
                ),
            ),
        ),
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    stable_evidence = Evidence(
        id=stable_evidence_id,
        operation_id=operation_id,
        task_id=stable_task_id,
        turn_id=task_turn.id,
        capability_id="fake.read",
        executor_id="fake.executor",
        kind="fake.record",
        schema_version=1,
        attempt=1,
        accepted=True,
        payload={"value": "stable"},
        content_hash="sha256:" + "1" * 64,
        created_at=NOW,
    )
    stable_task = Task(
        id=stable_task_id,
        operation_id=operation_id,
        turn_id=task_turn.id,
        call_id=stable_tool_call_id,
        capability_id="fake.read",
        executor_id="fake.executor",
        status=TaskStatus.SUCCEEDED,
        attempt=1,
        arguments={"key": "stable"},
        evidence_ids=(stable_evidence.id,),
        created_at=NOW,
        updated_at=NOW,
    )
    mutable_task = Task(
        id=mutable_task_id,
        operation_id=operation_id,
        turn_id=task_turn.id,
        call_id=mutable_tool_call_id,
        capability_id="fake.read",
        executor_id="fake.executor",
        status=mutable_task_status,
        attempt=1,
        arguments={"key": "mutable"},
        created_at=NOW,
        updated_at=NOW,
    )
    stable_observation = Observation(
        operation_id=operation_id,
        turn_id=task_turn.id,
        call_id=stable_task.call_id,
        task_id=stable_task.id,
        evidence_id=stable_evidence.id,
        code="fake.read.succeeded",
        message="Stable evidence was accepted.",
        payload={"value": "stable"},
        success=True,
        created_at=NOW,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.AWAITING_EXECUTION,
            turn_count=2,
            action_count=2,
        ),
        budgets=LoopBudgets(),
        turns=(model_turn, task_turn),
        model_calls=(started_call, tools_call),
        readiness=(
            Readiness(
                allowed=False,
                code="pending_task",
                message="One task is still pending.",
                missing_facts=("mutable value",),
                evaluated_at=NOW,
            ),
        ),
        tasks=(stable_task, mutable_task),
        evidence=(stable_evidence,),
        observations=(stable_observation,),
        events=(
            _event(
                operation_id,
                f"{operation_id}:event:created",
                "operation.created",
                NOW,
            ),
        ),
    )


def _with_completed_model(snapshot: OperationSnapshot) -> OperationSnapshot:
    model_turn = snapshot.turns[0]
    started_call = snapshot.model_calls[0]
    completed_call = replace(
        started_call,
        status=ModelCallStatus.COMPLETED,
        response=ModelResponse(
            finish_reason=FinishReason.STOP,
            text="The model call completed.",
        ),
        updated_at=MODEL_COMPLETED_AT,
    )
    event = _event(
        snapshot.operation.id,
        f"{snapshot.operation.id}:event:model-completed",
        "model.completed",
        MODEL_COMPLETED_AT,
        turn_id=model_turn.id,
        model_call_id=completed_call.id,
    )
    return replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            updated_at=MODEL_COMPLETED_AT,
        ),
        turns=(
            replace(model_turn, model_response_id=completed_call.id),
            *snapshot.turns[1:],
        ),
        model_calls=(completed_call, *snapshot.model_calls[1:]),
        events=(*snapshot.events, event),
    )


def _with_task_cancellation(
    snapshot: OperationSnapshot,
    *,
    event_type: str = "task.cancellation_requested",
) -> OperationSnapshot:
    task_turn = snapshot.turns[1]
    tools_call = snapshot.model_calls[1]
    mutable_task = snapshot.tasks[1]
    cancelled_task = replace(
        mutable_task,
        cancellation_requested=True,
        updated_at=TASK_COMPLETED_AT,
    )
    event = _event(
        snapshot.operation.id,
        f"{snapshot.operation.id}:event:task-cancellation",
        event_type,
        TASK_COMPLETED_AT,
        turn_id=task_turn.id,
        model_call_id=tools_call.id,
        call_id=cancelled_task.call_id,
        task_id=cancelled_task.id,
        capability_id=cancelled_task.capability_id,
        executor_id=cancelled_task.executor_id,
    )
    return replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            updated_at=TASK_COMPLETED_AT,
        ),
        tasks=(snapshot.tasks[0], cancelled_task),
        events=(*snapshot.events, event),
    )


def _with_plain_event(
    snapshot: OperationSnapshot,
    *,
    event_id: str,
    event_type: str,
) -> OperationSnapshot:
    return replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            updated_at=MODEL_COMPLETED_AT,
        ),
        events=(
            *snapshot.events,
            _event(
                snapshot.operation.id,
                event_id,
                event_type,
                MODEL_COMPLETED_AT,
            ),
        ),
    )


def _normalized_rows(path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    connection = sqlite3.connect(path)
    try:
        return {
            table: tuple(
                connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            )
            for table in NORMALIZED_TABLES
        }
    finally:
        connection.close()


def _revision(path: Path, operation_id: str) -> int | None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT revision FROM operations WHERE id = ?",
            (operation_id,),
        ).fetchone()
        return None if row is None else int(row[0])
    finally:
        connection.close()


def _install_runtime_event_abort(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("""
            CREATE TRIGGER contract_abort_runtime_event
            BEFORE INSERT ON runtime_events
            WHEN NEW.type = 'test.force_abort'
            BEGIN
                SELECT RAISE(ABORT, 'forced runtime event failure');
            END
            """)
        connection.commit()
    finally:
        connection.close()


def _drop_runtime_event_abort(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER contract_abort_runtime_event")
        connection.commit()
    finally:
        connection.close()


async def test_commit_persists_legal_mutations_and_exact_event_suffixes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    initial = _initial_snapshot(mutable_task_status=TaskStatus.RUNNING)
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(initial)
        model_completed = _with_completed_model(initial)

        second = await store.commit(model_completed, expected_revision=1)

        assert second.operation.revision == 2
        assert second.operation.snapshot == model_completed
        assert second.committed_events == (model_completed.events[-1],)
        assert model_completed.turns[0].model_response_id == (
            model_completed.model_calls[0].id
        )
        assert model_completed.model_calls[0].status is ModelCallStatus.COMPLETED

        cancellation_committed = _with_task_cancellation(model_completed)
        third = await store.commit(cancellation_committed, expected_revision=2)

        assert third.operation.revision == 3
        assert third.operation.snapshot == cancellation_committed
        assert third.committed_events == (cancellation_committed.events[-1],)
        assert cancellation_committed.tasks[1].status is TaskStatus.RUNNING
        assert cancellation_committed.tasks[1].cancellation_requested is True
        assert await store.load(initial.operation.id) == third.operation
        assert await store.load_by_trigger(initial.trigger.id) == third.operation
        assert created.operation.revision == 1
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert await reopened.load(initial.operation.id) == third.operation
        assert await reopened.load_by_trigger(initial.trigger.id) == third.operation
    finally:
        await reopened.close()


async def test_commit_not_found_and_stale_conflict_are_typed_and_atomic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    unknown = _initial_snapshot(
        operation_id="operation-unknown",
        trigger_id="trigger-unknown",
    )
    initial = _initial_snapshot()
    store = await SQLiteOperationStore.open(path)
    try:
        with pytest.raises(OperationNotFoundError) as missing:
            await store.commit(unknown, expected_revision=1)
        assert missing.value.operation_id == unknown.operation.id
        assert issubclass(type(missing.value), OperationStoreError)
        assert await store.load_by_trigger(unknown.trigger.id) is None

        created = await store.create(initial)
        baseline = _normalized_rows(path)
        candidate = _with_plain_event(
            initial,
            event_id="operation-commit:event:stale",
            event_type="checkpoint.stale",
        )

        with pytest.raises(OperationRevisionConflict) as conflict:
            await store.commit(candidate, expected_revision=0)

        assert conflict.value.operation_id == initial.operation.id
        assert conflict.value.expected_revision == 0
        assert conflict.value.actual_revision == 1
        assert issubclass(type(conflict.value), OperationStoreError)
        assert await store.load(initial.operation.id) == created.operation
        assert _normalized_rows(path) == baseline
        assert _revision(path, initial.operation.id) == 1
    finally:
        await store.close()


async def test_immutable_and_append_only_history_rejections_change_no_raw_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    initial = _initial_snapshot()
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(initial)
        baseline = _normalized_rows(path)
        rewritten_event = replace(
            initial.events[0],
            payload={"event_id": "silently-rewritten"},
        )
        mutations = (
            replace(initial, trigger=replace(initial.trigger, payload={"x": 1})),
            replace(
                initial,
                budgets=replace(
                    initial.budgets, max_turns=initial.budgets.max_turns + 1
                ),
            ),
            replace(
                initial,
                turns=(replace(initial.turns[0], number=99), *initial.turns[1:]),
            ),
            replace(
                initial,
                model_calls=(
                    initial.model_calls[0],
                    replace(initial.model_calls[1], provider_id="mock:forged"),
                ),
            ),
            replace(
                initial,
                tasks=(
                    replace(initial.tasks[0], arguments={"key": "forged"}),
                    initial.tasks[1],
                ),
            ),
            replace(
                initial,
                evidence=(replace(initial.evidence[0], payload={"x": 1}),),
            ),
            replace(
                initial,
                readiness=(replace(initial.readiness[0], message="Forged."),),
            ),
            replace(
                initial,
                observations=(replace(initial.observations[0], message="Forged."),),
            ),
            replace(initial, events=(rewritten_event,)),
        )

        for index, mutation in enumerate(mutations):
            candidate = replace(
                mutation,
                operation=replace(
                    mutation.operation,
                    updated_at=TASK_COMPLETED_AT,
                ),
                events=(
                    *mutation.events,
                    _event(
                        mutation.operation.id,
                        f"{mutation.operation.id}:event:invalid:{index}",
                        "checkpoint.invalid",
                        TASK_COMPLETED_AT,
                    ),
                ),
            )
            with pytest.raises(InvalidOperationCheckpointError) as invalid:
                await store.commit(candidate, expected_revision=1)
            assert invalid.value.operation_id == initial.operation.id
            assert issubclass(type(invalid.value), OperationStoreError)
            assert await store.load(initial.operation.id) == created.operation
            assert _normalized_rows(path) == baseline
            assert _revision(path, initial.operation.id) == 1
    finally:
        await store.close()


async def test_two_open_stores_have_exactly_one_typed_cas_winner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    initial = _initial_snapshot()
    first = await SQLiteOperationStore.open(path)
    second = await SQLiteOperationStore.open(path)
    try:
        await first.create(initial)
        candidates = (
            _with_plain_event(
                initial,
                event_id="operation-commit:event:writer-a",
                event_type="writer.a",
            ),
            _with_plain_event(
                initial,
                event_id="operation-commit:event:writer-b",
                event_type="writer.b",
            ),
        )

        outcomes = await asyncio.gather(
            first.commit(candidates[0], expected_revision=1),
            second.commit(candidates[1], expected_revision=1),
            return_exceptions=True,
        )

        winners = [item for item in outcomes if isinstance(item, CommitResult)]
        losers = [
            item for item in outcomes if isinstance(item, OperationRevisionConflict)
        ]
        assert len(winners) == 1
        assert len(losers) == 1
        assert losers[0].expected_revision == 1
        assert losers[0].actual_revision == 2
        assert await first.load(initial.operation.id) == winners[0].operation
        assert await second.load(initial.operation.id) == winners[0].operation
        assert winners[0].operation.snapshot in candidates
        assert _revision(path, initial.operation.id) == 2
    finally:
        await asyncio.gather(first.close(), second.close())


async def test_two_open_stores_return_typed_duplicate_create_race_losers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    first = await SQLiteOperationStore.open(path)
    second = await SQLiteOperationStore.open(path)
    try:
        same_operation = (
            _initial_snapshot(
                operation_id="operation-raced",
                trigger_id="trigger-raced-a",
            ),
            _initial_snapshot(
                operation_id="operation-raced",
                trigger_id="trigger-raced-b",
            ),
        )
        operation_outcomes = await asyncio.gather(
            first.create(same_operation[0]),
            second.create(same_operation[1]),
            return_exceptions=True,
        )
        operation_winners = [
            item for item in operation_outcomes if isinstance(item, CommitResult)
        ]
        operation_losers = [
            item
            for item in operation_outcomes
            if isinstance(item, OperationAlreadyExistsError)
        ]
        assert len(operation_winners) == 1
        assert len(operation_losers) == 1
        assert operation_losers[0].operation_id == "operation-raced"

        same_trigger = (
            _initial_snapshot(
                operation_id="operation-trigger-a",
                trigger_id="trigger-shared",
            ),
            _initial_snapshot(
                operation_id="operation-trigger-b",
                trigger_id="trigger-shared",
            ),
        )
        trigger_outcomes = await asyncio.gather(
            first.create(same_trigger[0]),
            second.create(same_trigger[1]),
            return_exceptions=True,
        )
        trigger_winners = [
            item for item in trigger_outcomes if isinstance(item, CommitResult)
        ]
        trigger_losers = [
            item
            for item in trigger_outcomes
            if isinstance(item, TriggerAlreadyClaimedError)
        ]
        assert len(trigger_winners) == 1
        assert len(trigger_losers) == 1
        assert trigger_losers[0].trigger_id == "trigger-shared"
        assert trigger_losers[0].operation_id == (
            trigger_winners[0].operation.snapshot.operation.id
        )
        assert await first.load_by_trigger("trigger-shared") == (
            trigger_winners[0].operation
        )
    finally:
        await asyncio.gather(first.close(), second.close())


async def test_runtime_event_abort_rolls_back_create_across_every_table(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    candidate = _initial_snapshot(
        operation_id="operation-create-abort",
        trigger_id="trigger-create-abort",
    )
    candidate = replace(
        candidate,
        events=(
            _event(
                candidate.operation.id,
                f"{candidate.operation.id}:event:abort",
                "test.force_abort",
                NOW,
            ),
        ),
    )
    store = await SQLiteOperationStore.open(path)
    try:
        baseline = _normalized_rows(path)
        _install_runtime_event_abort(path)
        try:
            with pytest.raises(sqlite3.IntegrityError, match="forced runtime event"):
                await store.create(candidate)

            assert _normalized_rows(path) == baseline
            assert _revision(path, candidate.operation.id) is None
            with pytest.raises(OperationNotFoundError):
                await store.load(candidate.operation.id)
            assert await store.load_by_trigger(candidate.trigger.id) is None
        finally:
            _drop_runtime_event_abort(path)
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert _normalized_rows(path) == baseline
        with pytest.raises(OperationNotFoundError):
            await reopened.load(candidate.operation.id)
        assert await reopened.load_by_trigger(candidate.trigger.id) is None
    finally:
        await reopened.close()


async def test_runtime_event_abort_rolls_back_commit_to_exact_prior_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    initial = _initial_snapshot(mutable_task_status=TaskStatus.RUNNING)
    candidate = _with_task_cancellation(
        _with_completed_model(initial),
        event_type="test.force_abort",
    )
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(initial)
        baseline = _normalized_rows(path)
        _install_runtime_event_abort(path)
        try:
            with pytest.raises(sqlite3.IntegrityError, match="forced runtime event"):
                await store.commit(candidate, expected_revision=1)

            assert _normalized_rows(path) == baseline
            assert _revision(path, initial.operation.id) == 1
            assert await store.load(initial.operation.id) == created.operation
            assert await store.load_by_trigger(initial.trigger.id) == created.operation
        finally:
            _drop_runtime_event_abort(path)
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert _normalized_rows(path) == baseline
        assert await reopened.load(initial.operation.id) == created.operation
        assert await reopened.load_by_trigger(initial.trigger.id) == created.operation
    finally:
        await reopened.close()
